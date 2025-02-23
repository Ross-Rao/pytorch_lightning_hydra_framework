import nibabel
from torchvision.datasets.folder import default_loader


def loader(path: str):
    if '.nii.gz' in path:
        return nibabel.load(path)
    else:
        return default_loader(path)
