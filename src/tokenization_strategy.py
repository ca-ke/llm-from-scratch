from abc import ABC
from typing import Dict, List

import re

from utils import get_freq_pair, update_pair


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
        vocab_size: int,
        allowed_special={"<|endoftext|>", "<|unk|>"},
    ):  
        assert text
        assert vocab_size > len(allowed_special)
        
        unique_chars = [chr(i) for i in range(256)]
        unique_chars.extend(sorted(set(text)))

        self._vocab = {i: char for i, char in enumerate(unique_chars)}
        self._inverse_vocab = {char: i for i, char in enumerate(unique_chars)}

        for token in allowed_special:
            if token not in self._inverse_vocab:
                new_id = len(self._vocab)
                self._vocab[new_id] = token
                self._inverse_vocab[token] = new_id
        token_ids = [self._inverse_vocab[char] for char in text if char in self._inverse_vocab]
        
        for new_id in range(len(self._vocab), vocab_size):
            try:
                pair_id = get_freq_pair(token_ids, mode="most")
                if not pair_id:
                    break

                token_ids = update_pair(token_ids, pair_id, new_id)
                self._bpe_merges[pair_id] = new_id
            except Exception as e:
                print(f"Error during BPE training: {e}")
                break
            
        for (p0, p1), new_id in self._bpe_merges.items():
            merged_token = self._vocab[p0] + self._vocab[p1]
            self._vocab[new_id] = merged_token
            self._inverse_vocab[merged_token] = new_id

    def get_vocab(self) -> Dict:
        return self._vocab
    
    def encode(self, text: str) -> List[int]:
        tokens = text.split()
        token_ids = []
        
        for token in tokens:
            print(f"Oi - {token} - {self._inverse_vocab.get(token)}")
            token_id = self._inverse_vocab.get(token, self._inverse_vocab.get("<|unk|>"))
            token_ids.append(token_id)
        
        return token_ids
        
