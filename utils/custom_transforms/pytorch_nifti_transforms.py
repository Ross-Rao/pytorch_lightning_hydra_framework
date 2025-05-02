# python import
import logging
# package import
import torch
import SimpleITK as sitk
import torch.nn.functional as F
from torchvision import transforms
# local import


logger = logging.getLogger(__name__)


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
