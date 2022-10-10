from torch import Tensor
from torch import distributions

from hw_asr.augmentations.base import AugmentationBase


class GaussianNoise(AugmentationBase):
    """
    Inspired by seminar 3
    """
    def __init__(self, std, **kwargs):
        self._aug = distributions.Normal(0, std)

    def __call__(self, data: Tensor):
        return data + self._aug.sample(data.size())
