from tokenization_strategy import (
    WhitespaceTokenizationStrategy,
)
from tokenizer import Tokenizer

import re


def main():
    text = "This is a sample text"
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]

    vocab = {token: integer for integer, token in enumerate(sorted(set(preprocessed)))}

    tokenizer = Tokenizer(strategy=WhitespaceTokenizationStrategy(vocab))
    print(tokenizer.tokenize("This is a sample text"))


if __name__ == "__main__":
    main()
