import torchaudio
import torch.nn.functional as F
import torch

from hw_asr.augmentations.base import AugmentationBase


class RoomImpulseResponse(AugmentationBase):
    """
    Inpired by seminar 3
    """
    def __init__(self, *args, **kwargs):
        rir, _ = torchaudio.load('h001_Bedroom_65.wav')
        self.left_pad = self.right_pad = rir.shape[-1] - 1
        self.flipped_rir = rir.squeeze().flip(0)

    def __call__(self, data: torch.Tensor):
        x = F.pad(data, [self.left_pad, self.right_pad]).view(1, 1, -1)
        x = torch.conv1d(x, self.flipped_rir.view(1, 1, -1)).squeeze()
        if x.abs().max() > 1:
            x /= x.abs().max()
        return x
