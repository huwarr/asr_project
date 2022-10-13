import torchaudio
import torch.nn.functional as F
import torch
import requests

from hw_asr.augmentations.base import AugmentationBase

class RoomImpulseResponse(AugmentationBase):
    """
    Inpired by seminar 3 + https://pytorch.org/audio/0.10.0/tutorials/audio_data_augmentation_tutorial.html#audio-data-augmentation 
    """
    def __init__(self, strength, **kwargs):
        # SAMPLE_DIR = 'data/assets'
        SAMPLE_RIR_URL = 'https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/distant-16k/room-response/rm1/impulse/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo.wav'
        SAMPLE_RIR_PATH = 'rir.wav'

        with open(SAMPLE_RIR_PATH, 'wb+') as file:
            file.write(requests.get(SAMPLE_RIR_URL).content)
        
        rir, _ = torchaudio.load(SAMPLE_RIR_PATH)
        self.left_pad = self.right_pad = rir.shape[-1] - 1
        self.flipped_rir = rir.squeeze().flip(0)

        self.strength = strength

    def __call__(self, data: torch.Tensor):
        batch_size = data.shape[0]
        x = F.pad(data, [self.left_pad, self.right_pad]).view(batch_size, 1, -1)
        x = F.conv1d(x, self.strength * self.flipped_rir.view(1, 1, -1)).squeeze(dim=1)
        if x.abs().max() > 1:
            x /= x.abs().max()
        return x
