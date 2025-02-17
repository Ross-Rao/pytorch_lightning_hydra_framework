import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

__all__ = ['RandomDataset', 'RandomDataModule']


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index], torch.randint(0, 2, (1,)).float()

    def __len__(self):
        return self.len


class RandomDataModule(LightningDataModule):
    """
    used to create a simple dataset for trainer testing.
    """
    def __init__(self, batch_size=32):
        super().__init__()
        self.val_dataset = None
        self.train_dataset = None
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = RandomDataset(64, 1000)
        self.val_dataset = RandomDataset(64, 200)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
