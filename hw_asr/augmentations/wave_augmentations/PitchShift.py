import torchaudio
from  torch import Tensor
import random

from hw_asr.augmentations.base import AugmentationBase

class PitchShift(AugmentationBase):
    """
    Inpired by seminar 3
    """
    def __init__(self, sample_rate, **kwargs):
        self.steps = [-3, -2, -1, 1, 2, 3]
        self.sr = sample_rate
        
    def __call__(self, data: Tensor):
        n_steps = random.choice(self.steps)
        return torchaudio.transforms.PitchShift(sample_rate=self.sr, n_steps=n_steps)(data)
