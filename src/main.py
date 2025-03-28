from gpt_dataset import GPTDataset
from create_dataloader import CreateDataLoader
from create_embedding import CreateEmbedding
from tokenizer import Tokenizer
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
    dataset = GPTDataset(tokenizer=Tokenizer(strategy=strategy))
    dataset.create_chunks(text, max_length=4, stride=1)
    create_data_loader = CreateDataLoader(dataset=dataset)
    dataloader = create_data_loader.execute(batch_size=1, shuffle=False)
    
    input, targets = next(iter(dataloader))
    
    embedding = CreateEmbedding(vocab_size=1000, output_dim=256, seed=123)
    token_embeddings = embedding.embbed(input)    
    print(token_embeddings)
    
if __name__ == "__main__":
    main()
