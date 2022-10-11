import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from hw_asr.base import BaseTrainer
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.logger.utils import plot_spectrogram_to_buf
from hw_asr.metric.utils import calc_cer, calc_wer
from hw_asr.utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics,
            optimizer,
            config,
            device,
            dataloaders,
            text_encoder,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
            sortagrad=False,
            beam_size=100,
            use_lm=False
    ):
        super().__init__(model, criterion, metrics, optimizer, config, device)
        self.skip_oom = skip_oom
        self.text_encoder = text_encoder
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if sortagrad:
            self.train_sortagrad_dataloader = dataloaders["train_sortagrad"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            if sortagrad:
                self.train_sortagrad_dataloader = inf_loop(self.train_sortagrad_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train" and k != "train_sortagrad"}
        self.lr_scheduler = lr_scheduler
        self.log_step = 50

        self.train_metrics_names = [m.name for m in self.metrics if m.name != "WER (BeamSearch + LM)" and m.name != "CER (BeamSearch + LM)" and m.name != "WER (oracle)" and m.name != "CER (oracle)"]
        self.train_metrics = MetricTracker(
            "loss", "grad norm", *self.train_metrics_names,
            writer=self.writer
        )
        self.evaluation_metrics_names = [m.name for m in self.metrics if m.name != "WER (BeamSearch + LM)" and m.name != "CER (BeamSearch + LM)" and m.name != "WER (oracle)" and m.name != "CER (oracle)"]
        self.evaluation_metrics = MetricTracker(
            "loss", *self.evaluation_metrics_names,
            writer=self.writer
        )
        self.metrics_names = [m.name for m in self.metrics]
        self.sortagrad = sortagrad
        self.beam_size = beam_size
        self.use_lm = use_lm

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["spectrogram", "text_encoded"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)

        if epoch == self.start_epoch:
            # SortaGrad
            dataloader_curr = self.train_sortagrad_dataloader
        else:
            dataloader_curr = self.train_dataloader

        for batch_idx, batch in enumerate(
                tqdm(dataloader_curr, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                    metrics_names=self.train_metrics_names
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                self._log_predictions(**batch)
                self._log_spectrogram(batch["spectrogram"])
                self._log_audio(batch["wav"])
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker, metrics_names: list):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()
        outputs = self.model(**batch)
        if type(outputs) is dict:
            batch.update(outputs)
        else:
            batch["logits"] = outputs

        batch["log_probs"] = F.log_softmax(batch["logits"], dim=-1)
        batch["log_probs_length"] = self.model.transform_input_lengths(
            batch["spectrogram_length"]
        )
        batch["loss"] = self.criterion(**batch)
        if is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        metrics.update("loss", batch["loss"].item())
        for met in self.metrics:
            if met.name in metrics_names:
                metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                    metrics_names=self.evaluation_metrics_names
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_predictions(**batch)
            self._log_spectrogram(batch["spectrogram"])

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(
            self,
            text,
            log_probs,
            log_probs_length,
            audio_path,
            examples_to_log=10,
            *args,
            **kwargs,
    ):
        if self.writer is None:
            return
        
        argmax_inds = log_probs.cpu().argmax(-1).numpy()
        argmax_inds = [
            inds[: int(ind_len)]
            for inds, ind_len in zip(argmax_inds, log_probs_length.numpy())
        ]
        argmax_texts_raw = [self.text_encoder.decode(inds) for inds in argmax_inds]
        argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]
        lengths = log_probs_length.detach().cpu().numpy()
        tuples = list(zip(text, argmax_texts_raw, argmax_texts, log_probs, lengths, audio_path))
        shuffle(tuples)
        rows = {}
        cers, wers = [], []
        for target, raw_pred, argmax_pred, log_prob, length, audio_path in tuples[:examples_to_log]:
            target = BaseTextEncoder.normalize_text(target)
            argmax_wer = calc_wer(target, argmax_pred) * 100
            argmax_cer = calc_cer(target, argmax_pred) * 100

            if self.use_lm:
                hypos = self.text_encoder.fast_beam_search_with_shallow_fusion(
                    log_prob.exp().detach().cpu().numpy(), length, beam_size=self.beam_size
                )
            else:
                hypos = self.text_encoder.ctc_beam_search(
                    log_prob.exp().detach().cpu().numpy(), length, beam_size=self.beam_size
                )
            beam_search_pred = hypos[0].text
            beam_search_wer = calc_wer(target, beam_search_pred) * 100
            beam_search_cer = calc_cer(target, beam_search_pred) * 100

            cers += [beam_search_cer / 100]
            wers += [beam_search_wer / 100]

            rows[Path(audio_path).name] = {
                "target": target,
                "raw argmax prediction": raw_pred,
                "argmax prediction": argmax_pred,
                "argmax wer": argmax_wer,
                "argmax cer": argmax_cer,
                "beam search prediction": beam_search_pred,
                "beam search wer": beam_search_wer,
                "beam search cer": beam_search_cer,
            }
        self.writer.add_table("predictions", pd.DataFrame.from_dict(rows, orient="index"))

        if "WER (BeamSearch + LM)" in self.metrics_names:
            self.writer.add_scalar("WER (BeamSearch + LM)", sum(wers) / len(wers))
        if "CER (BeamSearch + LM)" in self.metrics_names:
            self.writer.add_scalar("CER (BeamSearch + LM)", sum(cers) / len(cers))
        if "WER (oracle)" in self.metrics_names:
            self.writer.add_scalar("WER (oracle)", min(wers))
        if "CER (oracle)" in self.metrics_names:
            self.writer.add_scalar("CER (oracle)", min(cers))

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))
    
    def _log_audio(self, audio_batch):
        audio = random.choice(audio_batch)
        self.writer.add_audio("audio", audio, sample_rate=self.config["preprocessing"]["sr"])

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
 