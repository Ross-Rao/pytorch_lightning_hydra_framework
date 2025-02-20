import re
import os
import pandas as pd
import logging
import nibabel
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from sklearn.model_selection import train_test_split
# local import
from utils import transforms as custom_transforms
from utils.util import get_multi_attr

logger = logging.getLogger(__name__)


def loader(path: str):
    if '.nii.gz' in path:
        return nibabel.load(path)
    else:
        return default_loader(path)


def dir_to_df(root: str, ext: str, name_template: str = None) -> pd.DataFrame:
    """
    Converts files in a directory to a pandas DataFrame.

    Args:
        root (str): The root directory containing the files.
        ext (str): The file extension to filter files.
        name_template (str, optional): A template for extracting information from file names.

    Returns:
        pd.DataFrame: A DataFrame containing file paths and extracted information, all type is str.

    Example:
        >>> # Assuming the following directory structure:
        >>> # /home/user2/data/HCC-WCH/old
        >>> # ├── 1-1.nii.gz
        >>> # ├── 1-2.nii.gz
        >>> # ├── ...

        >>> # data_root = os.getenv("DATASET_LOCATION", '/home/user2/data')
        >>> # wch = os.path.join(data_root, 'HCC-WCH', 'old')
        >>> # df = dir_to_df(root=wch, ext=".nii.gz", name_template='{patient_number}-{label}')
    """
    assert os.path.exists(root) and os.path.isdir(root), f"Path {root} does not exist or is not a directory."

    file_paths = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(ext)]
    file_names = [os.path.basename(f)[: -len(ext)] for f in file_paths]
    if ext == '.nii.gz':
        image_shapes = [nibabel.load(path).get_fdata().shape for path in file_paths]
    else:
        image_shapes = [Image.open(path).size for path in file_paths]

    data = {'path': file_paths, 'name': file_names, 'shape': image_shapes}

    if name_template:
        template_pattern = re.sub(r'{(\w+)}', r'(?P<\1>.+)', name_template)
        regex = re.compile(template_pattern)

        for key in regex.groupindex.keys():
            data[key] = []

        for name in file_names:
            match = regex.match(name)
            if match:
                for key, value in match.groupdict().items():
                    data[key].append(value)
            else:
                for key in regex.groupindex.keys():
                    data[key].append("")
                logger.warning(f"Name {name} does not match the template {name_template}.")

    file_df = pd.DataFrame(data)
    return file_df


def preprocess_data(metadata: pd.DataFrame, test_split: float, val_split: float, save_path: str,
                    transform: dict = None, seed: int = 42, nii_pad_channels: int = 220) -> None:
    """
    Preprocesses the data by splitting it into training, validation, and test sets, applying transformations,
    and saving the processed data to .pt files.

    Args:
        metadata (pd.DataFrame): DataFrame containing metadata with file paths.
        test_split (float): Proportion of the data to be used as the test set.
        val_split (float): Proportion of the training data to be used as the validation set.
        save_path (str): Directory where the processed data will be saved.
        transform (dict, optional): Dictionary specifying the transformations to be applied. Defaults to None.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        nii_pad_channels (int, optional): Number of channels to pad the .nii.gz files to. Defaults to 220.

    Returns:
        None

    Example:
        >>> data_root = os.getenv("DATASET_LOCATION", '/home/user2/data')
        >>> wch = os.path.join(data_root, 'HCC-WCH', 'old')
        >>> save = os.path.join(data_root, 'HCC-WCH', 'preprocessed')

        >>> df = dir_to_df(root=wch, ext=".nii.gz", name_template='{patient_number}-{label}')
        >>> preprocess_data(metadata=df, test_split=0.2, val_split=0.1, save_path=save,
        ...                 transform={"ResampleNifti": None,
        ...                            "PermuteDimensions": {"dim_order": (2, 0, 1)},
        ...                            "PadChannels": {"target_channels": 220},
        ...                            "Resize": {"size": 256},
        ...                            "RandomCrop": {"size": 224},})

        >>> # another example
        >>> data_root = os.getenv("DATASET_LOCATION", '/home/user2/data')
        >>> wch = os.path.join(data_root, 'ISIC2018', 'ISIC2018_Task1_Test_GroundTruth')
        >>> save = os.path.join(data_root, 'ISIC2018', 'preprocessed')

        >>> df = dir_to_df(root=wch, ext=".png", name_template='ISIC_{patient_number}_segmentation')
        >>> preprocess_data(metadata=df, test_split=0.2, val_split=0.1, save_path=save,
        ...                 transform={"Resize": {"size": 256}, "RandomCrop": {"size": 224}, 'ToTensor': None, })
    """
    if not os.path.exists(save_path) or not os.path.isdir(save_path):
        os.makedirs(save_path)

    # Split data
    train_val, test = train_test_split(metadata, test_size=test_split, random_state=seed)
    train, val = train_test_split(train_val, test_size=val_split, random_state=seed)

    # Transform settings
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        # if you want to use your own transform, you can add them to utils/transforms.py
        transforms_lt = get_multi_attr([custom_transforms, transforms], transform)
        transform = transforms.Compose(transforms_lt)

    # save data to pt file
    for path_df, name in zip([val, test, train], ["val", "test", "train"]):
        # read images and apply transform (custom or torchvision)
        images = [transform(loader(path)) for path in tqdm(path_df['path'].tolist())]

        try:
            # must keep the same size of images
            # use CenterCrop or RandomCrop in transform to resize the images
            torch.save(torch.stack(images), os.path.join(save_path, f"{name}.pt"))
        except Exception as e:
            logger.error(f"Error saving data to {os.path.join(save_path, f'{name}.pt')}:")
            logger.error(e)
            logger.error("check the size of images whether use CenterCrop or RandomCrop in transform, "
                         "or check the channel of images")
            raise e
        logger.info(f"Data saved to {os.path.join(save_path, f'{name}.pt')}")
