import SimpleITK as sitk
from torchvision.datasets.folder import default_loader


def loader(path: str):
    if '.nii.gz' in path:
        return sitk.ReadImage(path, sitk.sitkFloat32)
    else:
        return default_loader(path)
