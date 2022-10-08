from typing import List

import torch
from torch import Tensor

from hw_asr.base.base_metric import BaseMetric
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
from hw_asr.metric.utils import calc_cer


class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        cers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            else:
                pred_text = self.text_encoder.decode(log_prob_vec[:length])
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)


class BeamSearchCERMetric(BaseMetric):
    """
    Here we predict the most probable hypothesis from beam search
    """
    def __init__(self, text_encoder: CTCCharTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], beam_size: int, **kwargs):
        cers = []
        lengths = log_probs_length.detach().numpy()
        for log_prob, length, target_text in zip(log_probs, lengths, text):
            prob = log_prob.exp().detach().cpu().numpy()
            hypos = self.text_encoder.ctc_beam_search(prob, length, beam_size)
            pred = hypos[0].text
            target_text = BaseTextEncoder.normalize_text(target_text)
            cers.append(calc_cer(target_text, pred))
        return sum(cers) / len(cers)


class BeamSearchWithLMCERMetric(BaseMetric):
    """
    Here we predict the most probable hypothesis from beam search with shallow fusion
    """
    def __init__(self, text_encoder: CTCCharTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], beam_size: int, **kwargs):
        cers = []
        lengths = log_probs_length.detach().numpy()
        for log_prob, length, target_text in zip(log_probs, lengths, text):
            prob = log_prob.exp().detach().cpu().numpy()
            hypos = self.text_encoder.fast_beam_search_with_shallow_fusion(prob, length, beam_size)
            pred = hypos[0].text
            target_text = BaseTextEncoder.normalize_text(target_text)
            cers.append(calc_cer(target_text, pred))
        return sum(cers) / len(cers)


class OracleCERMetric(BaseMetric):
    """
    The lowest CER among all hypothesis from beam search
    """
    def __init__(self, text_encoder: CTCCharTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], beam_size: int, **kwargs):
        cers = []
        lengths = log_probs_length.detach().numpy()
        for log_prob, length, target_text in zip(log_probs, lengths, text):
            prob = log_prob.exp().detach().cpu().numpy()
            hypos = self.text_encoder.fast_beam_search_with_shallow_fusion(prob, length, beam_size)
            cer = float("+inf")
            for hypo in hypos:
                target_text = BaseTextEncoder.normalize_text(target_text)
                cer_cur = calc_cer(target_text, hypo.text)
                if cer_cur < cer:
                    cer = cer_cur
            cers.append(cer)
        return sum(cers) / len(cers)