import os

import urllib3
from tokenization_strategy import (
    BPETokenizationStrategy,
)

import re

def main():
    text = "This is a sample text"
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]

    with open("assets/the-verdict.txt", "r", encoding="utf-8") as f:  
        text = f.read()
        
    strategy = BPETokenizationStrategy()
    strategy.train(text, vocab_size=1000, allowed_special={"<|endoftext|>"})
    input_text = "Jack embraced beauty through art and life."
    print(strategy.encode(input_text))
    
if __name__ == "__main__":
    main()
