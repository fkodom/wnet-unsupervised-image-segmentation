from typing import Tuple
from torch import Tensor

import h5py
import numpy as np
import torch.utils.data


def load_data(path: str) -> Tuple[Tensor, Tensor]:
    r"""Loads data from HDF5 file, converts to Tensor objects, and normalizes the input values.

    :param path: Complete path to the data file
    :return: Tuple of (training, validation) inputs
    """
    with h5py.File(path, 'r') as f:
        print('Loading training data...')
        x_train = f['Training']['Inputs'].__array__()

        print('Loading validation data...')
        x_val = f['Validation']['Inputs'].__array__()

        x_train = torch.from_numpy(x_train.astype(np.float32))
        x_val = torch.from_numpy(x_val.astype(np.float32))
        x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())
        x_val = (x_val - x_val.min()) / (x_val.max() - x_val.min())

    return x_train, x_val


def get_data_loader(x: Tensor, y: Tensor, batch_size=5) -> torch.utils.data.DataLoader:
    """Fetches a DataLoader, which is built into PyTorch, and provides a
    convenient (and efficient) method for sampling.

    :param x: (torch.Tensor) inputs
    :param y: (torch.Tensor) labels
    :param batch_size: (int)
    """
    dataset = torch.utils.data.TensorDataset(x, y)
    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=4, shuffle=True, batch_size=batch_size)

    return data_loader
