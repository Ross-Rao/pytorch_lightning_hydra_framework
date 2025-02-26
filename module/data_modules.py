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
                 is_valid_file: Callable[[str], bool] = None,
                 is_valid_label: Callable[[str], Any] = None,
                 n_folds: int = 5,
                 test_split_radio: float = 0.2,
                 seed: int = 42,
                 transform: dict = None,
                 target_transform: dict = None,
                 processed_data_save_dir_name: str = 'preprocessed',
                 processed_data_save_name_dict: dict = None,
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

        # save name dict
        self.processed_data_save_dir = os.path.join(self.data_dir, processed_data_save_dir_name)
        if processed_data_save_name_dict is None:
            self.save_name_dict = {
                'train': 'train_{0}.pt',  # 0 is placeholder
                'val': 'val_{0}.pt',
                'test': 'test.pt',
            }
        else:
            assert {'train', 'val', 'test'} == set(processed_data_save_name_dict.keys()), \
                "save_name_dict must have keys: train, val, test"
            self.save_name_dict = processed_data_save_name_dict

        # loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # dataset folds
        self.raw_dataset = RawDataset(data_dir=self.data_dir,
                                      is_valid_file=self.is_valid_file,
                                      is_valid_label=self.is_valid_label)
        self.split_dataset = SplitDatasetFolds(raw_dataset=self.raw_dataset,
                                               n_folds=self.n_folds,
                                               test_split_radio=self.test_split_radio,
                                               seed=self.seed)
        self.dataset_folds = LoadedDatasetFolds(data_dir=self.data_dir,
                                                split_folds=self.split_dataset,
                                                transform=self.transform,
                                                target_transform=self.target_transform,
                                                processed_data_save_dir_name=self.processed_data_save_dir,
                                                processed_data_save_name_dict=self.save_name_dict)

        # dataset
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def check_integrity(self):
        paths = [os.path.join(self.processed_data_save_dir, self.save_name_dict['test'])]
        paths += [os.path.join(self.processed_data_save_dir, self.save_name_dict['train'].format(fold))
                  for fold in range(self.n_folds)]
        paths += [os.path.join(self.processed_data_save_dir, self.save_name_dict['val'].format(fold))
                  for fold in range(self.n_folds)]
        return all([os.path.exists(path) for path in paths])

    def prepare_data(self):
        if not self.check_integrity():
            logging.info("Start to prepare data.")
            self.dataset_folds.save2pts()
        else:
            logging.info("All data files are already existed.")

    def setup(self, stage=None):
        self.train_dataset, self.val_dataset, self.test_dataset = self.dataset_folds.get_fold(self.fold)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.train_loader)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.val_loader)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.test_loader)
