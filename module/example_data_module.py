import os
import logging
from typing import Callable, Union
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
# local import
from utils.file_io import loader
from utils import transforms as custom_transforms
from utils.util import get_multi_attr

logger = logging.getLogger(__name__)


class ExampleDataModule(pl.LightningDataModule):
    """
    ExampleDataModule is a PyTorch Lightning DataModule for handling data loading and preprocessing.
    It supports custom transformations, data splitting, and saving/loading preprocessed data.
    """
    def __init__(self, data_dir: str,
                 train_loader: dict,
                 val_loader: dict,
                 test_loader: dict,
                 is_valid_label: Callable[[str], Union[str, int, float]],
                 is_valid_file: Union[str, Callable[[str], bool]] = None,
                 processed_data_save_dir: str = 'preprocessed',
                 processed_data_save_name_dict: dict = None,
                 val_split: float = 0.1,
                 test_split: float = 0.2,
                 seed: int = 42,
                 transform: dict = None,
                 target_transform: dict = None, ):
        super().__init__()
        # paths
        assert os.path.exists(data_dir) and os.path.isdir(data_dir), \
            f"Path {data_dir} does not exist or is not a directory."
        self.data_dir = data_dir
        self.processed_data_save_dir = os.path.join(self.data_dir, processed_data_save_dir)
        self.processed_data_save_name_dict = processed_data_save_name_dict if processed_data_save_name_dict \
            else self._default_processed_data_save_dict()
        assert {'metadata', 'train_input', 'train_label'}.issubset(self.processed_data_save_name_dict.keys()), \
            "name_dict must contain keys 'metadata', 'train_input', and 'train_label'."

        # data selection
        self.is_valid_file = is_valid_file
        self.is_valid_label = is_valid_label
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

        # Transform settings
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            # if you want to use your own transform, you can add them to utils/transforms.py
            # they will be imported by get_multi_attr
            transforms_lt = get_multi_attr([custom_transforms, transforms], transform)
            self.transform = transforms.Compose(transforms_lt)

        if target_transform is None:
            self.target_transform = transforms.Compose([])
        else:
            target_transforms_lt = get_multi_attr([custom_transforms, transforms], target_transform)
            self.target_transform = transforms.Compose(target_transforms_lt)

        # DataLoader settings
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # processed data
        self.train_data = None
        self.train_label = None
        self.val_data = None
        self.val_label = None
        self.test_data = None
        self.test_label = None

    @staticmethod
    def _default_processed_data_save_dict():
        return {
            'metadata': 'metadata.csv',
            'train_input': 'train.pt',
            'train_label': 'train_targets.pt',
            'val_input': 'val.pt',
            'val_label': 'val_targets.pt',
            'test_input': 'test.pt',
            'test_label': 'test_targets.pt'
        }

    def prepare_data(self):
        """
        Prepares the data for training, validation, and testing.
        If preprocessed data already exists, it uses that data.
        Otherwise, it preprocesses the raw data, splits it into train/val/test sets, and saves the processed data.

        Steps:
        1. Check if preprocessed data exists.
        2. If not, preprocess the raw data.
        3. Save metadata.
        4. Split data into train/val/test sets.
        5. Save the processed data for each split.
        """
        if all([os.path.exists(os.path.join(self.processed_data_save_dir, file))
                for file in self.processed_data_save_name_dict.values()]):
            logger.info("using processed data.")
            return

        # else prepare data
        logger.info("Missing processed data, Preprocessing...")

        # first prepare input data
        input_paths = [os.path.join(root, file) for root, _, files in os.walk(self.data_dir) for file in files]
        if callable(self.is_valid_file):
            input_paths = [file for file in input_paths if self.is_valid_file(file)]
        elif isinstance(self.is_valid_file, str):
            input_paths = [file for file in input_paths if file.endswith(self.is_valid_file)]
        else:
            raise ValueError("is_valid_file must be either a string or a callable.")

        # then prepare labels
        labels = [self.is_valid_label(file) for file in input_paths]

        # finally, create metadata
        metadata = pd.DataFrame({"input": input_paths, "label": labels})

        # save metadata
        metadata_name = self.processed_data_save_name_dict['metadata']
        if not os.path.exists(self.processed_data_save_dir):
            os.makedirs(self.processed_data_save_dir)
        metadata.to_csv(os.path.join(self.processed_data_save_dir, metadata_name), index=False)
        logger.info("metadata saved at %s", os.path.join(self.processed_data_save_dir, metadata_name))

        # Split data
        train_val, test = train_test_split(metadata, test_size=self.test_split, random_state=self.seed)
        train, val = train_test_split(train_val, test_size=self.val_split, random_state=self.seed)

        for split, df in zip(['train', 'val', 'test'], [train, val, test]):
            try:
                # Load input data
                input_data = [self.transform(loader(path)) for path in df['input']]
                input_data = torch.stack(input_data)
                torch.save(input_data, os.path.join(self.processed_data_save_dir,
                                                    self.processed_data_save_name_dict[f'{split}_input']))
                # Load labels
                labels = [self.target_transform(label) for label in df['label']]
                labels = torch.tensor(labels)
                torch.save(labels, os.path.join(self.processed_data_save_dir,
                                                self.processed_data_save_name_dict[f'{split}_label']))
            except Exception as e:
                logger.error(f"Error saving {split} data to {os.path.join(self.processed_data_save_dir)}:")
                logger.error(e)
                logger.error("check the size of images whether use CenterCrop or RandomCrop in transform, "
                             "or check the channel of images")
                raise e
            logger.info(f"{split} data saved to {os.path.join(self.processed_data_save_dir)}.")

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_data = torch.load(os.path.join(self.processed_data_save_dir,
                                                      self.processed_data_save_name_dict['train_input']),
                                         weights_only=True)
            self.train_label = torch.load(os.path.join(self.processed_data_save_dir,
                                                       self.processed_data_save_name_dict['train_label']),
                                          weights_only=True)
            self.val_data = torch.load(os.path.join(self.processed_data_save_dir,
                                                    self.processed_data_save_name_dict['val_input']),
                                       weights_only=True)
            self.val_label = torch.load(os.path.join(self.processed_data_save_dir,
                                                     self.processed_data_save_name_dict['val_label']),
                                        weights_only=True)
        else:
            self.test_data = torch.load(os.path.join(self.processed_data_save_dir,
                                                     self.processed_data_save_name_dict['test_input']),
                                        weights_only=True)
            self.test_label = torch.load(os.path.join(self.processed_data_save_dir,
                                                      self.processed_data_save_name_dict['test_label']),
                                         weights_only=True)

    def train_dataloader(self):
        train_dataset = TensorDataset(self.train_data, self.train_label)
        return DataLoader(train_dataset, **self.train_loader)

    def val_dataloader(self):
        val_dataset = TensorDataset(self.val_data, self.val_label)
        return DataLoader(val_dataset, **self.val_loader)

    def test_dataloader(self):
        test_dataset = TensorDataset(self.test_data, self.test_label)
        return DataLoader(test_dataset, **self.test_loader)
