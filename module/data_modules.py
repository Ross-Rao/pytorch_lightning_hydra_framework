# python import
import os
import logging
from typing import Any, Callable
# package import
import pytorch_lightning as pl
from torch.utils.data import DataLoader
# local import
from module.datasets import RawDataset
from module.dataset_folds import LoadedDatasetFolds, SplitDatasetFolds

logger = logging.getLogger(__name__)
__all__ = ['LoadedDataModule']


class LoadedDataModule(pl.LightningDataModule):
    def __init__(self, fold: int,
                 data_dir: str,
                 train_loader: dict,
                 val_loader: dict,
                 test_loader: dict,
                 use_preprocessed: bool = True,
                 grouped_attribute: Callable[[str], str] = None,
                 is_valid_file: Callable[[str], bool] = None,
                 is_valid_label: Callable[[str], Any] = None,
                 n_folds: int = 5,
                 test_split_radio: float = 0.2,
                 seed: int = 42,
                 transform: dict = None,
                 target_transform: dict = None,
                 processed_data_save_dir: str = './preprocessed',
                 ):
        super().__init__()
        # data
        assert 0 <= fold < n_folds, "fold must be in [0, n_folds)"
        self.fold = fold
        self.n_folds = n_folds
        self.data_dir = data_dir
        self.test_split_radio = test_split_radio
        self.seed = seed
        self.transform = transform
        self.target_transform = target_transform
        self.is_valid_file = is_valid_file
        self.is_valid_label = is_valid_label
        self.grouped_attribute = grouped_attribute
        self.use_preprocessed = use_preprocessed
        self.preprocessed_data_save_dir = processed_data_save_dir

        # loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # dataset folds
        raw_dataset = RawDataset(data_dir=self.data_dir,
                                 is_valid_file=self.is_valid_file,
                                 is_valid_label=self.is_valid_label,
                                 grouped_attribute=self.grouped_attribute)
        split_dataset = SplitDatasetFolds(raw_dataset=raw_dataset,
                                          n_folds=self.n_folds,
                                          test_split_radio=self.test_split_radio,
                                          seed=self.seed)
        self.dataset_folds = LoadedDatasetFolds(split_folds=split_dataset,
                                                transform=self.transform,
                                                target_transform=self.target_transform,
                                                processed_data_save_dir=self.preprocessed_data_save_dir)

        # dataset
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        if not self.dataset_folds.check_integrity():
            logger.info("Start to prepare data.")
            self.dataset_folds.save(save_pt=self.use_preprocessed)
        else:
            logger.info("All data files are already existed.")

    def setup(self, stage=None):
        self.train_dataset, self.val_dataset, self.test_dataset = (
            self.dataset_folds.load(self.fold, use_pt=self.use_preprocessed))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.train_loader)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.val_loader)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.test_loader)
