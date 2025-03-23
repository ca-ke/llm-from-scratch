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

    tokens = sorted(set(preprocessed))
    tokens.extend(["<|endoftext|>", "<|unk|>"])

    vocab = {token: integer for integer, token in enumerate(tokens)}

    tokenizer = Tokenizer(strategy=WhitespaceTokenizationStrategy(vocab))
    print(tokenizer.tokenize(text))

    tokenizer = Tokenizer(strategy=RegexTokenizationStrategy(vocab))
    print("Known words")
    ids = tokenizer.tokenize(text)
    print(ids)
    print(tokenizer.text(ids))
    print("One word unknow")
    ids = tokenizer.tokenize(text + " xablau")
    print(ids)
    print(tokenizer.text(ids))


if __name__ == "__main__":
    main()
