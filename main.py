from tokenization_strategy import (
    RegexTokenizationStrategy,
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
    print(tokenizer.tokenize(text))

    tokenizer = Tokenizer(strategy=RegexTokenizationStrategy(vocab))
    print(tokenizer.tokenize(text))


if __name__ == "__main__":
    main()
