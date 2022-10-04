import torch_audiomentations
from  torch import Tensor

from hw_asr.augmentations.base import AugmentationBase

class PitchShift(AugmentationBase):
    """
    Inpired by seminar 3 + https://pytorch.org/audio/0.10.0/tutorials/audio_data_augmentation_tutorial.html#audio-data-augmentation 
    """
    def __init__(self, sample_rate, **kwargs):
        self._aug = torch_audiomentations.PitchShift(sample_rate=sample_rate)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
