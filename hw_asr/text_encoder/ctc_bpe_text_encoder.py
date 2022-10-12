from typing import List, NamedTuple
from collections import defaultdict
import os
import gzip
import shutil

import torch
import kenlm
from pyctcdecode import build_ctcdecoder

from .bpe_text_encoder import BPETextEncoder

class Hypothesis(NamedTuple):
    text: str
    prob: float

class CTCBPETextEncoder(BPETextEncoder):
    EMPTY_TOK = "^"

    def __init__(self):
        super().__init__(self.EMPTY_TOK)
        # we don't want to waste time on downloading models if we don't need them
        self.lm = None
        self.alpha = 0.5
        self.beta = 0.05
        self.lm_gz_path = 'lm/3-gram.pruned.3e-7.arpa.gz'
        self.decompressed_lm_path = 'lm/3-gram.pruned.3e-7.arpa'
        self.lowered_lm_path = 'lm/lowercase_3-gram.pruned.3e-7.arpa'

        # for fast beam search
        self.unigrams_path = 'lm/librispeech-vocab.txt'
        self.fast_decoder = None
        self.labels = [''] + list(self.alphabet)


    def ctc_decode(self, inds: List[int]) -> str:
        EMPTY_IND = self.tokenizer.piece_to_id(self.EMPTY_TOK)
        last_char_ind = EMPTY_IND
        decoded_chars = []
        for ind in inds:
            if ind == last_char_ind or ind == EMPTY_IND:
                continue
            last_char_ind = ind
            decoded_chars.append(self.tokenizer.decode_ids(ind))
        return ''.join(decoded_chars)

    def fast_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        pass

    def fast_beam_search_with_shallow_fusion(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        if not self.fast_decoder:
            if not os.path.exists(self.decompressed_lm_path):
                with gzip.open(self.lm_gz_path, 'rb') as f_in:
                    with open(self.decompressed_lm_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            if not os.path.exists(self.lowered_lm_path):
                with open(self.decompressed_lm_path, 'r') as f_upper:
                    with open(self.lowered_lm_path, 'w') as f_lower:
                        for line in f_upper:
                            f_lower.write(line.lower())
            with open(self.unigrams_path) as f:
                unigram_list = [t.lower() for t in f.read().strip().split("\n")]
            
            self.fast_decoder = build_ctcdecoder(
                self.labels,
                kenlm_model_path=self.lowered_lm_path,
                unigrams=unigram_list,
                alpha=self.alpha,
                beta=self.beta
            )
        
        beam_search_results = self.fast_decoder.decode_beams(probs[:probs_length], beam_width=beam_size)

        return [Hypothesis(hypo[0], hypo[3]) for hypo in beam_search_results]
