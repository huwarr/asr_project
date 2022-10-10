import logging
from typing import List
import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    if len(dataset_items) == 0:
        return {}

    # first, let's pad everything we need to
    spec_lengths = [item['spectrogram'].shape[-1] for item in dataset_items]
    text_enc_lengths = [item['text_encoded'].shape[-1] for item in dataset_items]
    batch_size = len(dataset_items)

    batch_specs = torch.zeros(batch_size, dataset_items[0]['spectrogram'].shape[1], max(spec_lengths), dtype=torch.float)
    batch_text_enc = torch.zeros(batch_size, max(text_enc_lengths), dtype=torch.float)
    for i, item in enumerate(dataset_items):
        # batch_wavs[i, :wav_lengths[i]] = item['audio']
        batch_specs[i, :, :spec_lengths[i]] = item['spectrogram']
        batch_text_enc[i, :text_enc_lengths[i]] = item['text_encoded']
    
    # now we can save the rest, which doesn't require padding
    batch_durations = torch.tensor([item['duration'] for item in dataset_items])
    batch_texts = [item['text'] for item in dataset_items]
    batch_paths = [item['audio_path'] for item in dataset_items]
    batch_wavs = [item['audio'] for item in dataset_items]

    return {
        'wav': batch_wavs,
        'spectrogram': batch_specs,
        'spectrogram_length': torch.tensor(spec_lengths).long(),
        'text_encoded': batch_text_enc,
        'text_encoded_length': torch.tensor(text_enc_lengths).long(),
        'text': batch_texts,
        'audio_path': batch_paths
    }