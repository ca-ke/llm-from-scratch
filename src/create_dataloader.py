from gpt_dataset import GPTDataset
from torch.utils.data import DataLoader

class CreateDataLoader:
    def __init__(
        self,
        dataset: GPTDataset,
    ):
        self._dataset = dataset

    def execute(
        self,
        batch_size: int = 2,
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 0,
    ):
        return DataLoader(
            self._dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )
