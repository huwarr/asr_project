import logging

from hw_asr.base.base_dataset import BaseDataset
from sklearn.model_selection import train_test_split
from datasets import load_dataset

logger = logging.getLogger(__name__)

class LJDataset(BaseDataset):
    def __init__(self, part, *args, **kwargs):
        data = load_dataset('lj_speech')
        train, val_test = train_test_split(data._index, test_size=0.4, random_state=42)
        val, test = train_test_split(val_test, test_size=0.5, random_state=42)

        if part == "train":
            self._index = train
        elif part == "val":
            self._index = val
        else:
            self._index = test

        super().__init__([], preloaded=True, *args, **kwargs)
