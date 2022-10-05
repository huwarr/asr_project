import logging

from hw_asr.base.base_dataset import BaseDataset
from datasets import load_dataset

logger = logging.getLogger(__name__)

class CommonVoiceDataset(BaseDataset):
    def __init__(self, part, *args, **kwargs):
        # part = [train / validation / test]
        dataset = load_dataset('common_voice', subset='en', split=part)
        self._index = dataset._index
        super().__init__([], preloaded=True, *args, **kwargs)
