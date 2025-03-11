# python import
import logging
# package import
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# local import
from module.metadata import load_metadata, split_dataset_folds, load_data_to_monai_dataset

logger = logging.getLogger(__name__)
__all__ = ['SimpleDataModule']


class SimpleDataModule(pl.LightningDataModule):
    def __init__(self,
                 metadata: dict,
                 split: dict,
                 load: dict,
                 loader: dict):
        super().__init__()
        self.metadata = metadata
        self.split = split
        self.load = load
        self.loader = loader

        metadata = load_metadata(**self.metadata)
        split_dataset_folds(metadata, **self.split)
        self.train_dataset, self.val_dataset, self.test_dataset = load_data_to_monai_dataset(**self.load)
        logger.info('dataset loaded')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.loader['train_loader'])

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.loader['val_loader'])

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.loader['test_loader'])
