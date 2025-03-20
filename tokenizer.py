from typing import List
from tokenization_strategy import TokenizationStrategy


class Tokenizer():
    def __init__(self, strategy: TokenizationStrategy):
        self.strategy = strategy
        
    def tokenize(self, text: str) -> List[str]:
        return self.strategy.tokenize(text)