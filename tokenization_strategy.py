from abc import ABC
from typing import Dict, List

import re


class TokenizationStrategy(ABC):
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError(
            "Tokenization strategy must implement the encode method"
        )

    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError("Tokenization strategy must implement decode method")


class WhitespaceTokenizationStrategy(TokenizationStrategy):
    def __init__(self, vocab: Dict[str, int]):
        self._str_to_int = vocab
        self._int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> List[int]:
        return [
            self._str_to_int.get(token, self._str_to_int.get("<|unk|>"))
            for token in text.split()
        ]

    def decode(self, ids: List[int]) -> str:
        return " ".join(self._int_to_str.get(i, "<|unk|>") for i in ids)


class RegexTokenizationStrategy(TokenizationStrategy):
    def __init__(self, vocab: dict):
        self._str_to_int = vocab
        self._int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> List[int]:
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [
            self._str_to_int.get(item, self._str_to_int.get("<|unk|>"))
            for item in preprocessed
        ]
        return ids

    def decode(self, ids: List) -> str:
        text = " ".join(self._int_to_str.get(i, "<|unk|>") for i in ids)
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text


class BPETokenizationStrategy(TokenizationStrategy):
    def __init__(self):
        self._vocab = {}
        self._inverse_vocab = {}
        self._bpe_merges = {}
        self._bpe_ranks = {}

    def train(
        self,
        text: str,
        allowed_special={"<|endoftext|>", "<|unk|>"},
    ):
        unique_chars = [chr(i) for i in range(256)]
        unique_chars.extend(sorted(list(set(text))))
        print(unique_chars)
