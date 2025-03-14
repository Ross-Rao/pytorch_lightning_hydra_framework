# python import
# package import
import torch
# local import

__all__ = ["get_multi_attr", "AverageMeter", "patches2images", "patches2image"]


def get_multi_attr(modules: list, attr: dict):
    """Get multiple attributes from multiple modules.

    Args:
        modules : List of modules.
        attr (dict): Dictionary of attributes to get from the modules.

    Returns:
        list: List of results from the attributes.

    Example:
        >>> from utils import custom_transforms as custom_transforms
        >>> from torchvision import transforms
        >>> transform = {"ResampleNifti": None, "PermuteDimensions": (2, 0, 1), "PadChannels": 220,
        >>>              "Resize": {"size": 256}, "RandomCrop": {"size": 224}, }
        >>> transform_lt = get_multi_attr([transforms, custom_transforms], transform)
        >>> transform = transforms.Compose(transform_lt)
    """
    results = []
    for func_name, params in attr.items():
        funcs = [getattr(module, func_name, None) for module in modules]
        valid_funcs = [func for func in funcs if func is not None]
        if not valid_funcs:
            raise AttributeError(f"Attribute '{func_name}' not found in any of the modules: {str(modules)}")
        elif len(valid_funcs) > 1:
            raise AttributeError(f"Attribute '{func_name}' found in multiple modules: {str(modules)}")
        else:
            func = valid_funcs[0]
            # if failed, check the value of params
            if params is None:
                result = func()
            elif isinstance(params, dict):
                result = func(**params)
            else:
                result = func(params)
            results.append(result)

    return results


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def patches2images(image_indices, patches, coords):
    """
    Reconstructs images from their patches and coordinates.

    Args:
        image_indices (torch.Tensor): Tensor containing the indices of the original images.
        patches (torch.Tensor): Tensor of shape (b, c, h, w) containing image patches.
        coords (torch.Tensor): Tensor of shape (b, 3, 2) containing the coordinates of the patches.

    Returns:
        tuple: A tuple containing:
            - images (list): List of reconstructed images as byte tensors.
            - original_pic_indices (torch.Tensor): Tensor containing the unique indices of the original images.
    """
    original_pic_indices = torch.unique(image_indices)
    images = []
    for original_pic_index in original_pic_indices:
        patch_indices = (image_indices == original_pic_index)
        image = patches2image(patches[patch_indices] * 255, coords[patch_indices])
        images.append(image)
    return images, original_pic_indices


def patches2image(patches, coords):
    """
        Reconstructs an image from its patches and coordinates.

        Args:
            patches (torch.Tensor): Tensor of shape (b, c, h, w) containing image patches.
            coords (torch.Tensor): Tensor of shape (b, 3, 2) containing the coordinates of the patches.

        Returns:
            torch.Tensor: Reconstructed image as a byte tensor.
        """
    b, c, h, w = patches.shape
    max_y = int(coords[:, 1, :].max().item())
    max_x = int(coords[:, 2, :].max().item())
    image = torch.zeros((b, c, max_y, max_x), dtype=patches.dtype, device=patches.device)
    for i in range(b):
        image[i, :, coords[i, -2, 0]:coords[i, -2, 1], coords[i, -1, 0]:coords[i, -1, 1]] = patches[i]
    non_zero_counts = torch.count_nonzero(image, dim=0)  # (c, max_y, max_x)
    image = torch.sum(image, dim=0)  # stack patches to (c, max_y, max_x)
    image = torch.where(non_zero_counts > 0, image / non_zero_counts, 0)
    return torch.round(image).byte()
