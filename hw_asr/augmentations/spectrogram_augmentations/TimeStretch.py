import torchaudio
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class TimeStretch(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torchaudio.transforms.TimeStretch(*args, **kwargs)
        self.rate = 1.2

    def __call__(self, data: Tensor):
        return self._aug(data, self.rate)
