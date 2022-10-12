import json
from pathlib import Path
from string import ascii_lowercase
from typing import List, Union

import numpy as np
from torch import Tensor
import sentencepiece

from hw_asr.base.base_text_encoder import BaseTextEncoder


class BPETextEncoder(BaseTextEncoder):
    def __init__(self, empty_tok):
        self.text_path = 'lm/librispeech-vocab.txt'
        model_prefix = 'bpe_tokenizer'

        sentencepiece.SentencePieceTrainer.train(
            '--input={} --model_prefix={} --user_defined_symbols={}'.format(
                self.text_path, model_prefix, empty_tok
            )
        )

        self.tokenizer = sentencepiece.SentencePieceProcessor()
        self.tokenizer.load('{}.model'.format(self.model_prefix))
        self.vocab_size = self.tokenizer.get_piece_size()

    def __len__(self):
        return self.vocab_size

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.tokenizer.decode_ids(item)

    def encode(self, text) -> Tensor:
        text = self.normalize_text(text)
        return Tensor(self.tokenizer.encode_as_ids(text)).unsqueeze(0)

    def decode(self, vector: Union[Tensor, np.ndarray, List[int]]):
        return self.tokenizer.decode_ids(vector).strip()

    def dump(self, file):
        with Path(file).open('w') as f:
            json.dump(self.ind2char, f)
