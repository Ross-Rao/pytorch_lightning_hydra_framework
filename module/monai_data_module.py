# python import
import os
import logging
from functools import partial
# package import
import monai
import pandas as pd
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold
from monai import transforms
from monai.data import PatchIterd
# local import
from utils import transforms as custom_transforms
from utils.util import get_multi_attr

logger = logging.getLogger(__name__)
__all__ = ['MonaiDataModule']


def load_metadata(data_dir: str, parser: dict, group_by: list = None):
    """
    Example parser:
    data_dir: '${oc.env:DATASET_LOCATION, ./data}/MVI数据/ROI_224'
    parser:
        image: "lambda path: path.endswith('.jpg')"
        label: "lambda path: int(os.path.basename(path).split('.')[0].split('_')[0][-1])"
        patient_id: "lambda path: os.path.basename(path).split('.')[0].split('_')[0][:2]"
        model: "lambda path: os.path.basename(path).split('.')[0].split('_')[1]"
        number: "lambda path: os.path.basename(path).split('.')[0].split('_')[2]"
    """
    assert 'image' in parser.keys(), 'image parser is required'
    eval_parser = {k: eval(v) for k, v in parser.items()}
    metadata = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            file_metadata = {}
            input_path = os.path.join(root, file)

            # parse image files, invalid file skipped
            if eval_parser['image'] is not None and not eval_parser['image'](input_path):
                continue
            else:
                file_metadata['image'] = input_path

            # parse metadata
            for key, parser in eval_parser.items():
                if key == 'image':
                    continue
                file_metadata[key] = parser(input_path)
            metadata.append(file_metadata)

    assert len(metadata) > 0, f"No valid files found in {data_dir}, please check the path or the function"
    logger.info(f"Found {len(metadata)} valid files in {data_dir}")
    meta_df = pd.DataFrame(metadata)

    if group_by:
        assert all([col in meta_df.columns for col in group_by]), f"Columns {group_by} not found in metadata"
        assert set(group_by) != set(meta_df.columns), "group_by columns should not be all columns"
        assert 'image' not in group_by, "image column should not be in group_by columns"

        meta_df = meta_df.groupby(group_by)['image'].apply(list).reset_index()
        logger.info(f"Grouped metadata by {group_by}")
    return meta_df


def split_dataset_folds(
        df: pd.DataFrame,
        n_folds: int,
        test_split_ratio: float,
        split_cols: list = None,
        shuffle: bool = True,
        seed: int = 42,
        save_dir: str = "./",
        save_name_dict: dict = None,
        reset_split_index: bool = False
):
    """
    Split a DataFrame into training, validation, and test sets, and save them as CSV files.

    Args:
        df (pd.DataFrame): The input DataFrame to be split.
        n_folds (int): Number of folds for cross-validation.
        test_split_ratio (float): Ratio of the dataset to be used as the test set.
        split_cols (list, optional): Columns to consider when splitting the dataset. Defaults to None.
        shuffle (bool, optional): Whether to shuffle the data before splitting. Defaults to True.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        save_dir (str, optional): Directory to save the split datasets. Defaults to "./".
        save_name_dict (dict, optional): Dictionary specifying the save names for train, val, and test sets.
                                         Defaults to None, which uses a predefined naming scheme.
        reset_split_index (bool, optional): Whether to reset the index of the split datasets. Defaults to False.
    """
    paths = [os.path.join(save_dir, save_name_dict['test'])]
    paths += [os.path.join(save_dir, save_name_dict['train'].format(fold)) for fold in range(n_folds)]
    paths += [os.path.join(save_dir, save_name_dict['val'].format(fold)) for fold in range(n_folds)]
    if all([os.path.exists(path) for path in paths]):
        logger.info(f"Dataset split already exists in {save_dir}. Skipping split.")
        return

    # Check and create save directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define default save names if not provided
    if save_name_dict is None:
        save_name_dict = {
            'train': 'train_{0}.csv',  # {0} is a placeholder for fold number
            'val': 'val_{0}.csv',
            'test': 'test.csv',
        }
    else:
        assert {'train', 'val', 'test'} == set(save_name_dict.keys()), \
            "save_name_dict must have keys: train, val, test"

    # Split test set
    if split_cols is not None and len(split_cols) > 0:
        group_keys = df.groupby(split_cols).groups.keys()
        group_keys = [key if isinstance(key, tuple) else (key,) for key in group_keys]
        train_keys, test_keys = train_test_split(group_keys, test_size=test_split_ratio,
                                                 shuffle=shuffle, random_state=seed)
        train_val_df = df[df[split_cols].apply(tuple, axis=1).isin(train_keys)]
        test_df = df[df[split_cols].apply(tuple, axis=1).isin(test_keys)]
    else:
        train_val_df, test_df = train_test_split(df, test_size=test_split_ratio,
                                                 shuffle=shuffle, random_state=seed)

    # Save test set
    if reset_split_index:
        test_df.reset_index(drop=True, inplace=True)
        test_df.index += len(train_val_df)
    test_df.to_csv(os.path.join(save_dir, save_name_dict['test']), index=True)

    # Split train and validation sets using KFold
    kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=seed)
    if split_cols is not None and len(split_cols) > 0:
        train_val = train_val_df.groupby(split_cols).groups.keys()
        train_val = [key if isinstance(key, tuple) else (key,) for key in train_val]
    else:
        train_val = train_val_df
    for fold, (train_index, val_index) in enumerate(kf.split(train_val)):
        if split_cols is not None and len(split_cols) > 0:
            train_df = df[df[split_cols].apply(tuple, axis=1).isin([train_val[i] for i in train_index])]
            val_df = df[df[split_cols].apply(tuple, axis=1).isin([train_val[i] for i in val_index])]
        else:
            train_df = train_val.iloc[train_index]
            val_df = train_val.iloc[val_index]

        if reset_split_index:
            train_df.reset_index(drop=True, inplace=True)
            val_df.reset_index(drop=True, inplace=True)
            val_df.index += len(train_df)

        # Save train and validation sets
        train_df.to_csv(os.path.join(save_dir, save_name_dict['train'].format(fold)), index=True)
        val_df.to_csv(os.path.join(save_dir, save_name_dict['val'].format(fold)), index=True)

    logger.info(f"Dataset split completed. Files saved to {save_dir}")


def load_data_to_monai_dataset(
        fold: int,
        pre_transform: dict = None,
        transform: dict = None,
        dataset: str = 'Dataset',
        dataset_params: dict = None,
        load_dir: str = "./",
        load_name_dict: dict = None,
):
    """
    Load data from CSV files into MONAI Dataset.

    Args:
        fold (int): The fold number to load (used for train and val datasets).
        pre_transform (dict): The transformation pipeline for pre-processing data.
        transform (dict): The transformation pipeline for training data.
        dataset (str): The MONAI dataset class to use. Defaults to 'Dataset'.
        dataset_params (dict): Additional parameters for the MONAI dataset class.
        load_dir (str): The directory where the CSV files are stored.
        load_name_dict (dict): Dictionary specifying the file names for train, val, and test datasets.

    Returns:
        train_ds (Dataset): Training dataset.
        val_ds (Dataset): Validation dataset.
        test_ds (Dataset): Test dataset.
    """
    # Ensure required keys are in load_name_dict
    if load_name_dict is None:
        load_name_dict = {
            'train': 'train_{0}.csv',  # {0} is a placeholder for fold number
            'val': 'val_{0}.csv',
            'test': 'test.csv',
        }
    else:
        required_keys = {"train", "val", "test"}
        assert required_keys.issubset(set(load_name_dict.keys())), \
            f"load_name_dict must contain keys: {required_keys}"

    # Load train and validation datasets
    train_file = os.path.join(load_dir, load_name_dict["train"].format(fold))
    val_file = os.path.join(load_dir, load_name_dict["val"].format(fold))

    train_df = pd.read_csv(train_file, index_col=0,
                           converters={'image': lambda x: eval(x) if x.startswith('[') and x.endswith(']') else x})
    val_df = pd.read_csv(val_file, index_col=0,
                         converters={'image': lambda x: eval(x) if x.startswith('[') and x.endswith(']') else x})

    # Convert DataFrame to list of dictionaries
    train_data = train_df.reset_index().to_dict(orient="records")
    val_data = val_df.reset_index().to_dict(orient="records")

    # Load test dataset
    test_file = os.path.join(load_dir, load_name_dict["test"])
    test_df = pd.read_csv(test_file, index_col=0,
                          converters={'image': lambda x: eval(x) if x.startswith('[') and x.endswith(']') else x})
    test_data = test_df.reset_index().to_dict(orient="records")

    # Transform settings
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        # if you want to use your own transform, you can add them to utils/custom_transforms.py
        # they will be imported by get_multi_attr
        transforms_lt = get_multi_attr([custom_transforms, transforms, monai.data], transform)
        transform = transforms.Compose(transforms_lt)

    # Pre-transform settings
    if pre_transform:
        transforms_lt = get_multi_attr([custom_transforms, transforms, monai.data], pre_transform)
        pre_transform = transforms.Compose(transforms_lt)
        train_data, val_data, test_data = pre_transform(train_data), pre_transform(val_data), pre_transform(test_data)

    # Create MONAI Datasets
    if dataset_params is None:
        dataset_params = {}
    else:
        if 'patch_iter' in dataset_params.keys():  # used for GridPatchDataset
            dataset_params['patch_iter'] = PatchIterd(**dataset_params['patch_iter'])

    dataset_class = partial(getattr(monai.data, dataset), **dataset_params)
    train_ds = dataset_class(data=train_data, transform=transform)
    val_ds = dataset_class(data=val_data, transform=transform)
    test_ds = dataset_class(data=test_data, transform=transform)

    return train_ds, val_ds, test_ds


class MonaiDataModule(pl.LightningDataModule):
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
