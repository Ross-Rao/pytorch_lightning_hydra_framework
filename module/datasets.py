# python import
import os
import logging
from typing import Callable, Union, Any
# package import
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset, Subset
# local import
from utils.file_io import loader
from utils import custom_transforms
from utils.util import get_multi_attr

logger = logging.getLogger(__name__)
__all__ = ['RawDataset', 'LoadedDataset']


class RawDataset(Dataset):
    """
    A custom dataset class for loading raw data files and their corresponding labels.

    Example:
        >>> hcc = "/home/user2/data/HCC-WCH/old"
        >>> is_valid_file = "lambda path: path.endswith('.nii.gz')"
        >>> is_valid_label = "lambda path: float(os.path.basename(path).split('.')[0].rsplit('-', 1)[-1])"
        >>> raw_dataset = RawDataset(hcc, is_valid_file=eval(is_valid_file), is_valid_label=eval(is_valid_label))
        >>> print(len(raw_dataset))
        99
    """

    def __init__(self, data_dir: str,
                 is_valid_file: Callable[[str], bool] = None,
                 is_valid_label: Callable[[str], Any] = None):
        """
        Initializes the RawDataset.

        Args:
            data_dir (str): Directory containing the raw data files.
            is_valid_file (Callable[[str], bool], optional): Function to validate file paths. Defaults to None.
            is_valid_label (Callable[[str], Any], optional): Function to extract labels from file paths.
                Defaults to None.
        """
        self.input_paths = []
        self.labels = []

        for root, _, files in os.walk(data_dir):
            for file in files:
                input_path = os.path.join(root, file)
                if is_valid_file is not None and not is_valid_file(input_path):
                    continue
                self.input_paths.append(input_path)
                if is_valid_label is not None:
                    self.labels.append(is_valid_label(input_path))

        assert len(self.input_paths) > 0, f"No valid files found in {data_dir}"

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx: int) -> Union[str, tuple]:
        if self.labels:
            return self.input_paths[idx], self.labels[idx]
        else:
            return self.input_paths[idx]

    def save2df(self, save_path: str):
        if self.labels:
            df = pd.DataFrame({'path': self.input_paths, 'label': self.labels})
        else:
            df = pd.DataFrame({'path': self.input_paths})
        df.to_csv(save_path, index=False)
        logger.info(f"Save raw dataset to {save_path}")


class LoadedDataset(Dataset):
    """

    Example:
        >>> hcc = "/home/user2/data/HCC-WCH/old"
        >>> is_valid_file = "lambda path: path.endswith('.nii.gz')"
        >>> is_valid_label = "lambda path: float(os.path.basename(path).split('.')[0].rsplit('-', 1)[-1])"
        >>> raw_dataset = RawDataset(hcc, is_valid_file=eval(is_valid_file), is_valid_label=eval(is_valid_label))
        >>> trans = dict(NiftiToTensor=None, PadChannels=88, Resize=256, RandomCrop=224)
        >>> loaded_dataset = LoadedDataset(raw_dataset, transform=trans)
        >>> # save_path = "./old.pt"
        >>> # loaded_dataset.save2pt(save_path)
        >>> # tensor = torch.load(save_path)
        >>> # print(tensor['inputs'].shape)
        >>> # print(tensor['targets'].shape)
    """
    def __init__(self, dataset: Union[RawDataset, Subset],
                 transform: dict = None,
                 target_transform: dict = None, ):
        # dataset loading
        self.dataset = dataset

        # Transform settings
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            # if you want to use your own transform, you can add them to utils/custom_transforms.py
            # they will be imported by get_multi_attr
            transforms_lt = get_multi_attr([custom_transforms, transforms], transform)
            self.transform = transforms.Compose(transforms_lt)

        if target_transform is None:
            self.target_transform = transforms.Compose([custom_transforms.ToTensorWithoutNormalization()])
        else:
            target_transforms_lt = get_multi_attr([custom_transforms, transforms], target_transform)
            self.target_transform = transforms.Compose(target_transforms_lt)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if isinstance(sample, tuple):
            inputs, targets = sample
            inputs = self.transform(loader(inputs))
            targets = self.target_transform(targets)
            return inputs, targets
        else:
            inputs = sample
            return self.transform(loader(inputs))

    def save2pt(self, path: str):
        data = [self[i] for i in range(len(self.dataset))]
        assert len(data) > 0, 'No data to save'
        try:
            if isinstance(data[0], tuple):
                inputs, targets = zip(*data)
                torch.save([torch.stack(inputs), torch.stack(targets)], path)
            else:
                inputs = data
                torch.save([torch.stack(inputs)], path)
        except Exception as e:
            logger.error(f"Error saving data to {path}:")
            logger.error(e)
            logger.error("check the size of images whether use CenterCrop or RandomCrop in transform or not, "
                         "or check the channel of images")
            raise e

        logger.info(f'Save dataset to {path}, you can use to load it')
