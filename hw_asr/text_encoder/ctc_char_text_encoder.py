from typing import List, NamedTuple
from collections import defaultdict

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    """
    Add last character for conducting dynamic programming in Beam Search
    """
    text: str
    last_char: str
    prob: float

class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

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
        hypos: List[Hypothesis] = []
        # Start dynamic programming
        hypos.append(Hypothesis('', self.EMPTY_TOK, 1.0))
        for prob in probs:
            updated_hypos: List[Hypothesis] = []
            for text, last_char, prob in hypos:
                for i in range(voc_size):
                    if self.ind2char[i] == last_char:
                        updated_hypos.append(
                            Hypothesis(text, last_char, prob * probs[i])
                        )
                    else:
                        updated_hypos.append(
                            Hypothesis(
                                (text + last_char).replace(self.EMPTY_TOK, ''), 
                                self.ind2char[i], 
                                prob * probs[i]
                            )
                        )
            hypos = sorted(updated_hypos, key=lambda x: x.prob, reverse=True)[:beam_size]
        return hypos
