import torch
from torch.utils.data import Dataset

from tokenizer import Tokenizer

class GPTDataset(Dataset):
    def __init__(self,tokenizer: Tokenizer,):
        self._input_ids=[]
        self._target_ids=[]
        self._tokenizer = tokenizer
        
    def create_chunks(self, text: str, max_length: int, stride: int,):
        token_ids = self._tokenizer.encode(text)
        for i in range(0, len(token_ids)-max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1: i+max_length+1]
            self._input_ids.append(torch.tensor(input_chunk))
            self._target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self._input_ids)
    
    def __getitem__(self, index):
        return self._input_ids[index], self._target_ids[index]
        