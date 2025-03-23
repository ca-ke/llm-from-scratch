from typing import List
from tokenization_strategy import TokenizationStrategy


class Tokenizer:
    def __init__(self, strategy: TokenizationStrategy):
        self.strategy = strategy

    def tokenize(self, text: str) -> List[int]:
        return self.strategy.encode(text)

    def text(self, ids: List[int]) -> str:
        return self.strategy.decode(ids)

