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
        return list(map(lambda x: self._str_to_int[x], text.split()))

    def decode(self, ids: List[int]) -> str:
        return " ".join(map(lambda x: self._int_to_str[x], ids))


class SimpleTokenizer(TokenizationStrategy):
    def __init__(self, vocab: dict):
        self._str_to_int = vocab
        self._int_to_str = {i: s for s, i in vocab.items()}

    def tokenize(self, text: str) -> List[str]:
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self._str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids: List) -> str:
        text = " ".join(map(lambda x: self._int_to_str[x], ids))
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text
