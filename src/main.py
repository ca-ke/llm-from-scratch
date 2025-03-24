from tokenization_strategy import (
    BPETokenizationStrategy,
)

import re


def main():
    text = "This is a sample text"
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    special_chars = ["<|endoftext|>", "<|unk|>"]

    tokens = sorted(set(preprocessed))
    tokens.extend(special_chars)
    
    strategy = BPETokenizationStrategy()
    strategy.train(text, vocab_size=300, allowed_special=special_chars)
    strategy.encode(text)


if __name__ == "__main__":
    main()
