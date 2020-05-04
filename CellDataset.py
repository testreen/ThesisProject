import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from parseData import parseData


class CellDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x.cpu())
            if torch.cuda.is_available():
                x = x.to(device=torch.device('cuda') )
        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def imshow(img, title=''):
    """Plot the image batch.
    """
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)), cmap='gray')
    plt.show()
