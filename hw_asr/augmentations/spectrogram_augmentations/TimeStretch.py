import torchaudio
from torch import Tensor
import random

from hw_asr.augmentations.base import AugmentationBase


class TimeStretch(AugmentationBase):
    def __init__(self, n_freq, **kwargs):
        self._aug = torchaudio.transforms.TimeStretch(n_freq=n_freq, **kwargs)
        self.rates = [1.2, 0.8, 1.5, 1.]

    def __call__(self, data: Tensor):
        return self._aug(data, random.choice(self.rates))
