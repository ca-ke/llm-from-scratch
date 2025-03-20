from abc import ABC
from typing import List

import re

class TokenizationStrategy(ABC):
    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError("Tokenization strategy must implement the tokenize method")
    
class WhitespaceTokenizationStrategy(TokenizationStrategy):
    def tokenize(self, text: str) -> List[str]:
        return text.split()
    
class RegularExpressionTokenizationStrategy(TokenizationStrategy):
    def __init__(self, pattern: str):
        self._pattern = pattern

    def tokenize(self, text: str) -> List[str]:
        return  re.findall(self._pattern, text)
        