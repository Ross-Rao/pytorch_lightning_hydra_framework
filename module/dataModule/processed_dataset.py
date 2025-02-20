import os
import logging
import torch
from torchvision import datasets


logger = logging.getLogger(__name__)


class ProcessedDataset(datasets.VisionDataset):
    def __init__(self, root: str, env: str):
        super(ProcessedDataset, self).__init__(root)
        self.root = root
        self.env = env
        assert self.env in ['train', 'test', 'val'], 'env must be in [train, test, val]'

        self.data = torch.load(os.path.join(root, f'{self.env}.pt'), weights_only=True)
        self.targets = torch.load(os.path.join(root, f'{self.env}_targets.pt'), weights_only=True)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target

    def __len__(self):
        return len(self.data)
