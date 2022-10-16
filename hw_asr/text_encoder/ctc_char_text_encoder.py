from typing import List, NamedTuple
from collections import defaultdict
import os
import gzip
import shutil

import torch
import kenlm
from pyctcdecode import build_ctcdecoder

from .char_text_encoder import CharTextEncoder

class Hypothesis(NamedTuple):
    text: str
    prob: float

class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

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
        """
        Inspired by seminar 3
        """
        last_char_ind = self.char2ind[self.EMPTY_TOK]
        decoded_chars = []
        for ind in inds:
            if ind == last_char_ind or ind == self.char2ind[self.EMPTY_TOK]:
                continue
            last_char_ind = ind
            decoded_chars.append(self.ind2char[ind])
        return ''.join(decoded_chars)

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).

        Inspired by seminar 3
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        # Start dynamic programming
        dp = {
            ('', self.EMPTY_TOK): 1.0
        }
        for j, prob in enumerate(probs[:probs_length]):
            new_dp = defaultdict(float)
            for (res, last_char), v in dp.items():
                for i in range(len(prob)):
                    if self.ind2char[i] == last_char:
                        new_dp[(res, last_char)] += v * prob[i]
                    else:
                        new_dp[((res + self.ind2char[i]).replace(self.EMPTY_TOK, ''), self.ind2char[i])] += v * prob[i]
            if j < len(probs - 1):
                dp = dict(list(sorted(new_dp.items(), key=lambda x: x[1], reverse=True))[:beam_size])
            else:
                dp = new_dp
        
        # at the end we might have 2 hypothesis, that are decoded into 
        # the same text, but one of them ends with empty token, and 
        # another one - with some other character. let's fix this
        hypos_dict = defaultdict(float)
        for (res, last_char), v in dp.items():
            hypos_dict[res] += v
        
        hypos = sorted([Hypothesis(text, prob) for text, prob in hypos_dict.items()], key=lambda x: x.prob, reverse=True)[:beam_size]
        return hypos

    def ctc_beam_search_with_shallow_fusion(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        # download LM for shallow fusion if haven't done it yet
        if not self.lm:
            if not os.path.exists(self.decompressed_lm_path):
                with gzip.open(self.lm_gz_path, 'rb') as f_in:
                    with open(self.decompressed_lm_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            if not os.path.exists(self.lowered_lm_path):
                with open(self.decompressed_lm_path, 'r') as f_upper:
                    with open(self.lowered_lm_path, 'w') as f_lower:
                        for line in f_upper:
                            f_lower.write(line.lower())
            
            self.lm = kenlm.Model(self.lowered_lm_path)

        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        # Start dynamic programming
        dp = {
            ('', self.EMPTY_TOK): 1.0
        }
        for j, prob in enumerate(probs[:probs_length]):
            new_dp = defaultdict(float)
            for (res, last_char), v in dp.items():
                for i in range(len(prob)):
                    if self.ind2char[i] == last_char:
                        new_dp[(res, last_char)] += v * prob[i]
                    else:
                        new_dp[((res + self.ind2char[i]).replace(self.EMPTY_TOK, ''), self.ind2char[i])] += v * prob[i]
            # Rescore hypothesis with LM before cutting the beam
            texts = [key[0] for key in new_dp.keys()]
            scores = list(new_dp.values())
            lengths = [len(text) for text in texts]
            # https://github.com/kpu/kenlm/issues/150 - LN returns log_10 (logits)
            lm_scores = torch.tensor([10**self.lm.score(text) for text in texts])
            new_scores = torch.tensor(scores) + self.alpha * lm_scores - self.beta * torch.tensor(lengths)
            
            dp = dict([((res, last_char), new_scores[i]) for i, (res, last_char) in enumerate(new_dp.keys())])

            if j < len(probs - 1):
                dp = dict(list(sorted(new_dp.items(), key=lambda x: x[1], reverse=True))[:beam_size])
            else:
                dp = new_dp
        
        hypos_dict = defaultdict(float)
        for (res, last_char), v in dp.items():
            hypos_dict[res] += v
        
        hypos = sorted([Hypothesis(text, prob) for text, prob in hypos_dict.items()], key=lambda x: x.prob, reverse=True)[:beam_size]
        return hypos

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
