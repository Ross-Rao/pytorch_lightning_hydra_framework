# python import
import logging
# package import
import torch
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
from torchvision import transforms
from monai.transforms import Transform
# local import


logger = logging.getLogger(__name__)
__all__ = ["ResampleNifti",
           "PermuteDimensions",
           "PadChannels",
           "NiftiToTensor",
           "ToTensorWithoutNormalization",
           "StackImaged",
           "StackTensorTransformd",
           "IndexTransformd"]


class ResampleNifti:
    """
    Resample the input Nifti1Image to the same resolution as the original image.

    Args:
        img (sitk.Image): Input sitk.Image instance.

    Example:
        >>> transform_pipeline = transforms.Compose([ResampleNifti(),])
        >>> nifti_path = "/home/user2/data/HCC-WCH/old/3-1.nii.gz"
        >>> nifti_img = sitk.ReadImage(nifti_path, sitk.sitkFloat32)
        >>> resampled_tensor = transform_pipeline(nifti_img)
        >>> print("Shape of Tensor After resample:", resampled_tensor.shape)
        Shape of Tensor After resample: torch.Size([415, 259, 202])
    """
    def __call__(self, img):

        original_spacing = img.GetSpacing()
        original_size = img.GetSize()
        img_data = sitk.GetArrayFromImage(img)

        new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in
                    zip(original_size, original_spacing, (1.0, 1.0, 1.0))]

        # Convert to Tensor and perform resampling
        img_tensor = torch.tensor(img_data).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        resampled_tensor = F.interpolate(img_tensor, size=new_size,
                                         mode='trilinear', align_corners=False).squeeze()
        return resampled_tensor


class PermuteDimensions:
    """

    Permute the dimensions of the input Tensor.

    Example:
        >>> import torch
        >>> example_tensor = torch.randn(10, 20, 30)
        >>> permute_transform = PermuteDimensions(dim_order=(2, 0, 1))
        >>> transformed_tensor = permute_transform(example_tensor)
        >>> print("Original Tensor shape:", example_tensor.shape)
        Original Tensor shape: torch.Size([10, 20, 30])
        >>> print("Transformed Tensor shape:", transformed_tensor.shape)
        Transformed Tensor shape: torch.Size([30, 10, 20])
    """
    def __init__(self, dim_order):
        """
        Initialize the PermuteDimensions transform.

        :param dim_order: A tuple specifying the new order of dimensions. For example, (2, 0, 1).
        """
        self.dim_order = dim_order if isinstance(dim_order, tuple) else tuple(dim_order)

    def __call__(self, img_tensor):
        if not isinstance(img_tensor, torch.Tensor):
            raise TypeError("Input must be a Tensor.")

        # Use the permute method to reorder dimensions
        return img_tensor.permute(self.dim_order)

    def __repr__(self):
        return f"{self.__class__.__name__}(dim_order={self.dim_order})"


class PadChannels:
    """
    A transform to pad a Tensor along a specified dimension to match a target number of channels.

    This transform calculates the padding needed to reach the target number of channels and applies
    padding to the front and back of the specified dimension. The padding is done in 'constant' mode
    with a user-specified value.

    Attributes:
        target_channels (int): The target number of channels to pad the Tensor to.
        dim (int): The dimension along which to apply padding (default is 0).
        value (float): The value used for padding (default is 0).

    Example:
        >>> import torch
        >>> example_tensor = torch.randn(3, 20, 30)
        >>> pad_transform = PadChannels(target_channels=8, dim=0, value=0.5)
        >>> padded_tensor = pad_transform(example_tensor)
        >>> print("Original Tensor shape:", example_tensor.shape)
        Original Tensor shape: torch.Size([3, 20, 30])
        >>> print("Padded Tensor shape:", padded_tensor.shape)
        Padded Tensor shape: torch.Size([8, 20, 30])
    """

    def __init__(self, target_channels, dim=0, value=0):
        """
        Initialize the PadChannels transform.

        Args:
            target_channels (int): The target number of channels to pad the Tensor to.
            dim (int, optional): The dimension along which to apply padding (default is 0).
            value (float, optional): The value used for padding (default is 0).
        """
        self.target_channels = target_channels
        self.dim = dim
        self.value = value

    def __call__(self, img_tensor):
        if not isinstance(img_tensor, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")

        current_channels = img_tensor.shape[self.dim]

        if current_channels == self.target_channels:
            return img_tensor
        elif current_channels > self.target_channels:
            raise ValueError(f"Input Tensor already has {current_channels} channels, "
                             f"which is not less than the target {self.target_channels} channels.")
        else:
            # Calculate padding sizes
            pad_front = (self.target_channels - current_channels) // 2
            pad_back = self.target_channels - current_channels - pad_front

            # Prepare padding tuple (only pad along the specified dimension)
            pad_tuple = [0, 0] * len(img_tensor.shape)
            pad_tuple[2 * self.dim] = pad_front
            pad_tuple[2 * self.dim + 1] = pad_back

            # Apply padding using F.pad
            padded_tensor = F.pad(img_tensor, tuple(reversed(pad_tuple)), mode='constant', value=self.value)
            return padded_tensor

    def __repr__(self):
        return f"{self.__class__.__name__}(target_channels={self.target_channels}, dim={self.dim}, value={self.value})"


class NiftiToTensor:
    """
    A transform to convert a NIfTI image (sitk.Image) to a PyTorch Tensor.

    This transform assumes the input is a sitk.Image object. It extracts the image data
    and converts it to a PyTorch Tensor with a specified data type. Optionally, it can normalize
    the data to the range [0, 1] by shifting and scaling the data.

    Attributes:
        dtype (str): The target data type for the output Tensor (default: torch.float32).
        normalize (bool): Whether to normalize the data to the range [0, 1] (default: False).

    Example:
        >>> nifti_path = "/home/user2/data/HCC-WCH/old/3-1.nii.gz"
        >>> nifti_image = sitk.ReadImage(nifti_path)
        >>> to_tensor = NiftiToTensor(dtype='float64')
        >>> tensor_data = to_tensor(nifti_image)
        >>> print(tensor_data.shape)
        torch.Size([72, 220, 352])
        >>> print(tensor_data.dtype)
        torch.float64
    """

    def __init__(self, dtype='float32', normalize=False):
        self.dtype = getattr(torch, dtype, torch.float32)
        self.normalize = normalize

    def __call__(self, nifti_image):
        if not isinstance(nifti_image, sitk.Image):
            raise TypeError("Input must be a SimpleITK Image object.")

        tensor_data = torch.tensor(sitk.GetArrayFromImage(nifti_image), dtype=self.dtype)

        # Normalize data to [0, 1] if required
        if self.normalize:
            min_val = tensor_data.min()
            max_val = tensor_data.max()
            if min_val < 0:
                logger.warning("Data contains negative values. Normalizing by shifting and scaling.")

            # Shift and scale the data to [0, 1]
            tensor_data = (tensor_data - min_val) / (max_val - min_val)

        return tensor_data

    def __repr__(self):
        return f"{self.__class__.__name__}(dtype={self.dtype})"


class ToTensorWithoutNormalization:
    def __init__(self, dtype='float32'):
        self.dtype = getattr(torch, dtype, torch.float32)

    def __call__(self, img):
        return torch.tensor(img, dtype=self.dtype)

    def __repr__(self):
        return f"{self.__class__.__name__}(dtype={self.dtype})"


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
        num_elements = len(data['image'])  # 获取张量列表的长度
        new_indices = original_index * num_elements + torch.arange(num_elements)
        data[self.key] = new_indices
        return data


class DropSliced(Transform):
    def __init__(self, key: str, slice_idx: int):
        self.key = key
        self.slice = slice_idx

    def __call__(self, data):
        image = data[self.key]
        sliced_image = image[self.slice, :, :].unsqueeze(0)
        other_slices = torch.cat([image[:self.slice, :, :], image[self.slice+1:, :, :]], dim=0)
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
        data['neighbor_index'] = neighbor_index
        return data
