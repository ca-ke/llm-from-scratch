
from tokenization_strategy import RegularExpressionTokenizationStrategy, WhitespaceTokenizationStrategy
from tokenizer import Tokenizer


def main():
    tokenizer = Tokenizer(strategy=WhitespaceTokenizationStrategy())
    print(tokenizer.tokenize("This is a sample text"))
    
    regex_tokenizer = Tokenizer(RegularExpressionTokenizationStrategy(r'\w+'))
    print(regex_tokenizer.tokenize("This is a sample text, with punctuation!"))
    

if __name__ == "__main__":
    main()
