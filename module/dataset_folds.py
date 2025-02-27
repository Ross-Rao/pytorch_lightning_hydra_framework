# python import
import os
import logging
from functools import partial
# package import
import torch
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import random_split, Subset, TensorDataset
# local import
from module.datasets import RawDataset, LoadedDataset


logger = logging.getLogger(__name__)
__all__ = ['SplitDatasetFolds', 'LoadedDatasetFolds']


class SplitDatasetFolds:
    """
    Example:
        >>> hcc = "/home/user2/data/HCC-WCH/old"
        >>> is_valid_file = "lambda path: path.endswith('.nii.gz')"
        >>> is_valid_label = "lambda path: float(os.path.basename(path).split('.')[0].rsplit('-', 1)[-1])"
        >>> raw_dataset = RawDataset(hcc, is_valid_file=eval(is_valid_file), is_valid_label=eval(is_valid_label))
        >>> split_dataset = SplitDatasetFolds(raw_dataset, 5, 0.2)
        >>> split_dataset.save2dfs(".")
    """
    def __init__(self, raw_dataset: RawDataset,
                 n_folds: int,
                 test_split_radio: float,
                 shuffle: bool = True,
                 seed: int = 42,
                 save_name_dict: dict = None):
        # load raw dataset
        self.raw_dataset = raw_dataset
        self.n_folds = n_folds

        # split test dataset
        test_len = int(len(self.raw_dataset) * test_split_radio)
        train_val_len = len(self.raw_dataset) - test_len
        train_val, self.test_dataset = random_split(self.raw_dataset, [train_val_len, test_len],
                                                    generator=torch.Generator().manual_seed(seed))

        # split train and validation dataset
        self.train_datasets = []
        self.val_datasets = []
        k_fold = KFold(n_splits=n_folds, shuffle=shuffle, random_state=seed)
        for train_idx, var_idx in k_fold.split(train_val):
            self.train_datasets.append(Subset(train_val, train_idx))
            self.val_datasets.append(Subset(train_val, var_idx))

        # save name dict
        if save_name_dict is None:
            self.save_name_dict = {
                'train': 'train_{0}.csv',  # 0 is placeholder
                'val': 'val_{0}.csv',
                'test': 'test.csv',
                'metadata': 'metadata.csv',
            }
        else:
            assert {'train', 'val', 'tset'} == set(save_name_dict.keys()), \
                "save_name_dict must have keys: train, val, test"
            self.save_name_dict = save_name_dict

    def get_train_val_datasets(self, fold: int):
        return self.train_datasets[fold], self.val_datasets[fold]

    def get_test_dataset(self):
        return self.test_dataset

    def __len__(self):
        return self.n_folds

    def generator(self):
        for fold in range(self.n_folds):
            yield self.get_train_val_datasets(fold), self.get_test_dataset()

    def save2dfs(self, save_dir: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        original_dataset = self.raw_dataset
        original_path = os.path.join(save_dir, self.save_name_dict['metadata'])
        original_dataset.save2df(original_path)

        for fold, (train, val) in enumerate(zip(self.train_datasets, self.val_datasets)):
            train_path = os.path.join(save_dir, self.save_name_dict['train'].format(fold))
            val_path = os.path.join(save_dir, self.save_name_dict['val'].format(fold))

            train_samples = [original_dataset[i] for i in train.indices]
            val_samples = [original_dataset[i] for i in val.indices]

            pd.DataFrame(train_samples, columns=['path', 'label']).to_csv(train_path, index=False)
            pd.DataFrame(val_samples, columns=['path', 'label']).to_csv(val_path, index=False)

            logger.info(f"{train_path} saved")
            logger.info(f"{val_path} saved")

        test_path = os.path.join(save_dir, self.save_name_dict['test'])
        test_samples = [original_dataset[i] for i in self.test_dataset.indices]
        pd.DataFrame(test_samples, columns=['path', 'label']).to_csv(test_path, index=False)
        logger.info(f"{test_path} saved")


class LoadedDatasetFolds:
    """
    Example:
        >>> hcc = "/home/user2/data/HCC-WCH"
        >>> is_valid_file = "lambda path: path.endswith('.nii.gz')"
        >>> is_valid_label = "lambda path: float(os.path.basename(path).split('.')[0].rsplit('-', 1)[-1])"
        >>> raw_dataset = RawDataset(hcc, is_valid_file=eval(is_valid_file), is_valid_label=eval(is_valid_label))
        >>> split_dataset = SplitDatasetFolds(raw_dataset, 5, 0.2)
        >>> trans = dict(NiftiToTensor=None, PadChannels=88, Resize=256, RandomCrop=224)
        >>> save_dir_name = "processed_data"
        >>> save_name_dict = {'train': 'train_{0}.pt', 'val': 'val_{0}.pt', 'test': 'test.pt'}
        >>> loaded_dataset = LoadedDatasetFolds(hcc, save_dir_name, save_name_dict, split_dataset, transform=trans)
        >>> loaded_dataset.save2pts()
        >>> train_dataset, val_dataset, test_dataset = loaded_dataset.get_fold(0)
    """
    def __init__(self, data_dir: str,
                 processed_data_save_dir_name: str,
                 processed_data_save_name_dict: dict,
                 split_folds: SplitDatasetFolds,
                 transform: dict = None,
                 target_transform: dict = None,
                 ):
        # get split datasets
        self.split_datasets = split_folds

        self.transform = transform
        self.target_transform = target_transform
        self.processed_data_save_dir = os.path.join(data_dir, processed_data_save_dir_name)
        assert {'train', 'val', 'test'} == set(processed_data_save_name_dict.keys()), \
            "save_name_dict must have keys: train, val, test"
        self.save_name_dict = processed_data_save_name_dict

    def save2pts(self):
        if not os.path.exists(self.processed_data_save_dir):
            os.makedirs(self.processed_data_save_dir)

        # save loaded split data
        self.split_datasets.save2dfs(self.processed_data_save_dir)
        partial_load_dataset = partial(LoadedDataset, transform=self.transform,
                                       target_transform=self.target_transform)

        test_dataset = partial_load_dataset(dataset=self.split_datasets.get_test_dataset())
        test_save_path = os.path.join(self.processed_data_save_dir, self.save_name_dict['test'])
        test_dataset.save2pt(test_save_path)
        logger.info(f"{self.save_name_dict['test']} saved")

        for fold in range(self.split_datasets.n_folds):
            # save loaded fold data
            train_dataset, val_dataset = self.split_datasets.get_train_val_datasets(fold)

            train_dataset = partial_load_dataset(dataset=train_dataset)
            train_save_path = os.path.join(self.processed_data_save_dir, self.save_name_dict['train'].format(fold))
            train_dataset.save2pt(train_save_path)

            val_dataset = partial_load_dataset(dataset=val_dataset)
            val_save_path = os.path.join(self.processed_data_save_dir, self.save_name_dict['val'].format(fold))
            val_dataset.save2pt(val_save_path)
            logger.info(f"{self.save_name_dict['train'].format(fold)} and "
                        f"{self.save_name_dict['val'].format(fold)} saved")

    def get_fold(self, fold: int):
        train_save_path = os.path.join(self.processed_data_save_dir, self.save_name_dict['train'].format(fold))
        val_save_path = os.path.join(self.processed_data_save_dir, self.save_name_dict['val'].format(fold))
        test_save_path = os.path.join(self.processed_data_save_dir, self.save_name_dict['test'])

        test_lt = torch.load(test_save_path)
        test_dataset = TensorDataset(*test_lt)

        train_lt = torch.load(train_save_path)
        train_dataset = TensorDataset(*train_lt)

        val_lt = torch.load(val_save_path)
        val_dataset = TensorDataset(*val_lt)

        return train_dataset, val_dataset, test_dataset

    def __len__(self):
        return len(self.split_datasets)

    def generator(self):
        for fold in range(len(self.split_datasets)):
            yield self.get_fold(fold)
