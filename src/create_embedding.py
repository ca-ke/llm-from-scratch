from typing import Optional
import torch


class CreateEmbedding:
    def __init__(self, vocab_size: int, output_dim: int, seed: Optional[int]):
        if seed:
            torch.manual_seed(seed)

        self._embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    def get_weight(self) -> torch.Tensor:
        return self._embedding_layer.weight
    
    def embbed(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self._embedding_layer(input_tensor)