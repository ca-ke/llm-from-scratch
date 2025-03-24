from abc import ABC
from typing import Dict, List, Optional, Tuple

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
        token_ids = [
            self._inverse_vocab[char] for char in text if char in self._inverse_vocab
        ]

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
            if token in self._inverse_vocab:
                token_ids.append(self._inverse_vocab[token])
            else:
                sub_token_ids = self._tokenize_with_bpe(token)
                token_ids.extend(sub_token_ids)

        return token_ids

    def decode(self, ids: List[int]) -> str:
        return "".join([self._vocab[id] for id in ids])

    def _tokenize_with_bpe(self, token: str) -> List[int]:
        token_ids = self._get_token_ids(token)
        if not self._bpe_ranks:
            return self._merge_tokens(token_ids)

        return self._merge_symbols(token_ids)

    def _get_token_ids(self, token: str) -> List[int]:
        return [self._inverse_vocab.get(char, "<|unk|>") for char in token]

    def _merge_symbols(self, token_ids: List[int]) -> List[int]:
        symbols = [self._vocab[id_num] for id_num in token_ids]

        while True:
            pairs = set(zip(symbols, symbols[1:]))
            if not pairs:
                break

            bigram = self._get_most_frequent_bigram(pairs)
            if bigram is None:
                break

            symbols = self._merge_bigram(symbols, bigram)
            if len(symbols) == 1:
                break
        return [self._inverse_vocab[sym] for sym in symbols]

    def _merge_tokens(self, token_ids: List[int]) -> List[int]:
        can_merge = True
        while can_merge and len(token_ids) > 1:
            can_merge = False
            new_tokens = []
            i = 0
            while i < len(token_ids) - 1:
                pair = (token_ids[i], token_ids[i + 1])
                if pair in self._bpe_merges:
                    merged_token_id = self._bpe_merges[pair]
                    new_tokens.append(merged_token_id)
                    i += 2
                    can_merge = True
                else:
                    new_tokens.append(token_ids[i])
                    i += 1
            if i < len(token_ids):
                new_tokens.append(token_ids[i])
            token_ids = new_tokens
        return token_ids

    def _get_most_frequent_bigram(
        self,
        pairs: set,
    ) -> Optional[Tuple[str, str]]:
        min_rank = float("inf")
        bigram = None
        for p in pairs:
            current_rank = self._bpe_ranks.get(p, float("inf"))
            if current_rank < min_rank:
                min_rank = current_rank
                bigram = p
        return bigram

    def _merge_bigram(
        self,
        symbols: List[str],
        bigram: Tuple[str, str],
    ) -> List[str]:
        first, second = bigram
        new_symbols = []
        i = 0

        while i < len(symbols):
            if (
                i < len(symbols) - 1
                and symbols[i] == first
                and symbols[i + 1] == second
            ):
                new_symbols.append(first + second)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        return new_symbols
