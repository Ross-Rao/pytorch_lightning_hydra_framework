import nibabel as nib
import torch
import torch.nn.functional as F
from torchvision import transforms


class ResampleNifti:
    """
    Resample the input Nifti1Image to the same resolution as the original image.

    Args:
        img (Nifti1Image): Input Nifti1Image instance.

    Example:
        >>> transform_pipeline = transforms.Compose([ResampleNifti(),])
        >>> nifti_path = "/home/user2/data/HCC-WCH/old/3-1.nii.gz"
        >>> nifti_img = nib.load(nifti_path)
        >>> resampled_tensor = transform_pipeline(nifti_img)
        >>> print("Shape of Tensor After resample:", resampled_tensor.shape)
        Shape of Tensor After resample: torch.Size([415, 259, 202])
    """
    def __call__(self, img):
        """
        Accepts Nifti1Image instance input and returns the resampled Tensor.
        """
        # If the input is a string, assume it is a file path
        if isinstance(img, nib.Nifti1Image):
            img_data = img.get_fdata()
            header = img.header
        else:
            raise TypeError("Input must be a Nifti1Image object.")

        pixdim = header.get('pixdim')
        new_width = round(img_data.shape[0] * pixdim[1])
        new_height = round(img_data.shape[1] * pixdim[2])
        new_depth = round(img_data.shape[2] * pixdim[3])

        # Convert to Tensor and perform resampling
        img_tensor = torch.tensor(img_data).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        resampled_tensor = F.interpolate(img_tensor, size=(new_width, new_height, new_depth),
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
        self.dim_order = dim_order

    def __call__(self, img_tensor):
        if not isinstance(img_tensor, torch.Tensor):
            raise TypeError("Input must be a Tensor.")

        # Use the permute method to reorder dimensions
        return img_tensor.permute(self.dim_order)


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

        if current_channels >= self.target_channels:
            raise ValueError(f"Input Tensor already has {current_channels} channels, "
                             f"which is not less than the target {self.target_channels} channels.")

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
