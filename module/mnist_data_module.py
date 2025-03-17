# python import
import os
# package import
import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
# local import


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./preprocess/mnist", batch_size: int = 512):
        super(MNISTDataModule, self).__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir

        self.train = None
        self.val = None
        self.test = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        download = not os.path.exists(self.data_dir)
        if stage == 'fit' or stage is None:
            self.train = MNIST(self.data_dir, train=True, download=download, transform=ToTensor())
            self.val = MNIST(self.data_dir, train=False, download=download, transform=ToTensor())

        if stage == 'test' or stage is None:
            self.test = MNIST(self.data_dir, train=False, download=download, transform=ToTensor())

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=4)
