from typing import Literal
import SimpleITK as sitk
import numpy as np
from torchvision.datasets.folder import default_loader


__all__ = ['loader']


def loader(path: str, return_type: Literal['obj', 'nparray'] = 'obj'):
    if '.nii.gz' in path:
        if return_type == 'nparray':
            return sitk.GetArrayFromImage(sitk.ReadImage(path, sitk.sitkFloat32))
        else:
            return sitk.ReadImage(path, sitk.sitkFloat32)
    else:
        if return_type == 'nparray':
            return np.array(default_loader(path))
        else:
            return default_loader(path)
