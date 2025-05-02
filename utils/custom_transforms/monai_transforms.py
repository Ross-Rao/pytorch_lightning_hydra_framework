# python import
import logging
# package import
import torch
import numpy as np
from monai.transforms import MapTransform, Transform
from monai.config import KeysCollection
# local import


logger = logging.getLogger(__name__)
__all__ = ["StackImaged",
           "StackTensorTransformd",
           "IndexTransformd",
           "DropSliced",
           "UpdatePatchIndexd",
           "CreateCopyd",
           "CustomMixUpD"]


class StackImaged(Transform):
    def __init__(self, keys=None, dim=0):
        self.keys = keys if keys is not None else ["image"]
        self.dim = dim

    def __call__(self, data):
        for key in self.keys:
            if key not in data:
                raise KeyError(f"Key '{key}' not found in data dictionary.")
            images = data[key]
            stacked_images = np.stack(images, axis=self.dim)
            data[key] = stacked_images
        return data


class StackTensorTransformd(Transform):
    def __init__(self, keys: list):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            tensor = data[key]
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"数据键 '{key}' 的值必须是张量，但当前类型为 {type(tensor)}")
            if tensor.dim() >= 3:
                data[key] = tensor.reshape(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])
            else:
                data[key] = tensor.reshape(tensor.shape[0] * tensor.shape[1])
        return data


class IndexTransformd(Transform):
    def __init__(self, key: str):
        self.key = key

    def __call__(self, data):
        assert 'image' in data.keys(), "数据字典中必须包含键 'image'"
        original_index = data[self.key]
        num_elements = 1 if len(data['image'].shape) <= 3 else len(data['image'])  # 获取张量列表的长度
        new_indices = original_index * num_elements + torch.arange(num_elements)
        data[self.key] = new_indices
        return data


class DropSliced(Transform):
    def __init__(self, key: str, slice_idx: int):
        self.key = key
        self.slice = slice_idx

    def __call__(self, data):
        image = data[self.key]
        if isinstance(image, torch.Tensor):
            sliced_image = image[self.slice, :, :].unsqueeze(0)
            other_slices = torch.cat([image[:self.slice, :, :], image[self.slice+1:, :, :]], dim=0)
        elif isinstance(image, np.ndarray):
            sliced_image = image[self.slice, :, :].reshape(1, *image.shape[1:])
            other_slices = np.concatenate([image[:self.slice, :, :], image[self.slice+1:, :, :]], axis=0)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        data[self.key] = other_slices
        data[f"{self.key}_slice"] = sliced_image
        return data


class UpdatePatchIndexd(Transform):
    def __init__(self, key: str, overlap: float):
        self.key = key
        self.overlap = overlap

    def __call__(self, data):
        assert 'image' in data.keys(), "数据字典中必须包含键 'image'"
        # 获取原始图像的索引
        original_index = data[self.key]
        # 获取当前 patch 的偏移量
        origin_space = torch.tensor(data['original_spatial_shape'])
        window_space = torch.tensor(data['image'].shape[-len(origin_space):])
        coordinate = torch.tensor(data['patch_coords'][-len(origin_space):])[:, 0]
        stride = (window_space * (1 - self.overlap)).long()
        coordinate = (coordinate / stride).long()
        count = (origin_space / stride).long()
        offset_array = torch.arange(int(count.prod())).reshape(count.tolist())
        offset = offset_array[tuple(coordinate.tolist())]

        # find neighbor coordinates
        d_lt = [line for line in torch.eye(count.size(0), dtype=torch.long)]
        prob_coords = [coordinate + d for d in d_lt] + [coordinate - d for d in d_lt]
        zero = torch.zeros_like(count, dtype=torch.long)
        bool_valid = [(zero <= coord).all() and (coord < count).all() for coord in prob_coords]
        first_true_index = next((i for i, tensor in enumerate(bool_valid) if tensor.item()), None)
        neighbor_coords = prob_coords[first_true_index]
        neighbor_offset = offset_array[tuple(neighbor_coords.tolist())]

        # 计算 patch 的索引
        patch_index = original_index * count.prod() + offset
        neighbor_index = original_index * count.prod() + neighbor_offset
        # 将更新后的索引添加到数据中
        data[self.key] = patch_index
        data[f"original_{self.key}"] = original_index
        data[f'neighbor_{self.key}'] = neighbor_index
        return data


class CreateCopyd(Transform):
    def __init__(self, keys: list):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if isinstance(data[key], torch.Tensor):
                data[key + '_copy'] = data[key].clone()
            elif isinstance(data[key], np.ndarray):
                data[key + '_copy'] = data[key].copy()
            else:
                raise TypeError(f"Unsupported data type for key '{key}': {type(data[key])}")
        return data


class CustomMixUpD(MapTransform):
    """
    Custom Mixup data augmentation Transform.
    Performs Mixup operations on specified keys in the data dictionary, generating a specified number of linearly
        combined samples.
    used for stacked images and labels, not suitable for batched images and labels.

    Args:
        keys (list[str]): The keys in the data dictionary that need to be Mixup processed.
        n_lam (int): The number of linearly combined samples to generate.

    Example:
        >>> from monai.data import Dataset
        >>> data = [{"img": np.random.rand(10, 224, 224), "label": np.random.randint(0, 10, size=10)}]
        >>> dataset = Dataset(data)
        >>> mixup = MixUpD(keys=["img", 'label'], n_lam=4, num_classes=10)
        >>> for d in dataset:  # Perform Mix-up operation on each sample in the dataset
        ...     d = mixup(d)
        ...     print(d['mix_img'].shape, d['mix_label'].shape, d['lam'].shape)
        (4, 224, 224) (4, 10) torch.Size([4])
    """
    def __init__(self, keys: KeysCollection, n_lam: int, num_classes: int = None):
        super().__init__(keys)
        self.n_lam = n_lam
        self.num_classes = num_classes

    def __call__(self, data):
        d = dict(data)
        lam_list = torch.tensor(np.random.beta(1, 1, size=self.n_lam))

        # Initialize mixed data dictionary
        mixed_dt = {key: [] for key in self.keys}

        for lam in lam_list:
            for key in self.keys:
                if key not in d:
                    raise KeyError(f"Key {key} not found in data.")
                value = d[key]
                assert isinstance(value, (np.ndarray, torch.Tensor)), f"Unsupported type for key {key}: {type(value)}"
                idx1, idx2 = np.random.choice(value.shape[0], 2, replace=False)
                img1, img2 = value[idx1], value[idx2]

                # Perform different Mix-up operations based on data type and dimension
                if value.ndim >= 2:  # If it is a multidimensional array or tensor
                    mixed = lam * img1 + (1 - lam) * img2
                elif value.ndim == 1:  # If it is a 1D array or tensor
                    if isinstance(value, np.ndarray):
                        if np.issubdtype(value.dtype, np.floating):  # Check if dtype is float
                            mixed = lam * img1 + (1 - lam) * img2
                        elif np.issubdtype(value.dtype, np.integer):  # Check if dtype is integer
                            if self.num_classes is None:
                                assert img1 == img2, "If num_classes is not provided, img1 and img2 must be the same."
                                mixed = img1
                            else:
                                onehot1 = np.eye(self.num_classes)[img1]
                                onehot2 = np.eye(self.num_classes)[img2]
                                mixed = lam * onehot1 + (1 - lam) * onehot2
                        else:
                            raise ValueError(f"Unsupported data type for key {key}: {value.dtype}")
                    elif isinstance(value, torch.Tensor):
                        if value.dtype in [torch.float32, torch.float64]:  # Check if dtype is float
                            mixed = lam * img1 + (1 - lam) * img2
                        elif value.dtype in [torch.int32, torch.int64, torch.long]:  # Check if dtype is integer
                            if self.num_classes is None:
                                assert img1 == img2, "If num_classes is not provided, img1 and img2 must be the same."
                                mixed = img1
                            else:
                                onehot1 = torch.eye(self.num_classes)[img1]
                                onehot2 = torch.eye(self.num_classes)[img2]
                                mixed = lam * onehot1 + (1 - lam) * onehot2
                        else:
                            raise ValueError(f"Unsupported data type for key {key}: {value.dtype}")
                    else:
                        raise ValueError(f"Unsupported data type for key {key}: {type(value)}")
                else:
                    raise ValueError(f"Unsupported data dimension for key {key}: {value.ndim}")
                mixed_dt[key].append(mixed)

        # Update the original data dictionary with mixed data
        for key in self.keys:
            d[f"mix_{key}"] = np.array(mixed_dt[key]) if isinstance(d[key], np.ndarray) else torch.stack(mixed_dt[key])
        d["lam"] = lam_list  # Add lam values to the data dictionary
        return d
