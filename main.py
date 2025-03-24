from tokenization_strategy import (
    BPETokenizationStrategy,
    RegexTokenizationStrategy,
    WhitespaceTokenizationStrategy,
)
from tokenizer import Tokenizer

import re


def main():
    text = "This is a sample text"
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    special_chars = ["<|endoftext|>", "<|unk|>"]

    tokens = sorted(set(preprocessed))
    tokens.extend(special_chars)

    vocab = {token: integer for integer, token in enumerate(tokens)}

    strategy = BPETokenizationStrategy()
    strategy.train(text, allowed_special=special_chars)


if __name__ == "__main__":
    main()
