r"""
filter.py
-----------
Functions for smoothing/filtering 2D images.
"""

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm


def gaussian_kernel(radius: int = 3, sigma: float = 4, device='cpu'):
    x_2 = np.linspace(-radius, radius, 2*radius+1) ** 2
    dist = np.sqrt(x_2.reshape(-1, 1) + x_2.reshape(1, -1)) / sigma
    kernel = norm.pdf(dist) / norm.pdf(0)
    kernel = torch.from_numpy(kernel.astype(np.float32))
    kernel = kernel.view((1, 1, kernel.shape[0], kernel.shape[1]))

    if device == 'cuda':
        kernel = kernel.cuda()

    return kernel


class GaussianBlur2D(nn.Module):

    def __init__(self, radius: int = 2, sigma: float = 1):
        super(GaussianBlur2D, self).__init__()
        self.radius = radius
        self.sigma = sigma

    def forward(self, x):
        batch, nchan, nrow, ncol = x.shape
        kernel = gaussian_kernel(radius=self.radius, sigma=self.sigma, device=x.device.type)

        for c in range(nchan):
            x[:, c:c+1] = F.conv2d(x[:, c:c+1], kernel, padding=self.radius)

        return x


class CRFSmooth2D(nn.Module):

    def __init__(self, radius: int = 1, sigma_1: float = 0.5, sigma_2: float = 0.5):
        super(CRFSmooth2D, self).__init__()
        self.radius = radius
        self.sigma_1 = sigma_1  # Spatial standard deviation
        self.sigma_2 = sigma_2  # Pixel value standard deviation

    def forward(self, labels: Tensor, inputs: Tensor, *args):
        num_classes = labels.shape[1]
        kernel = gaussian_kernel(radius=self.radius, sigma=self.sigma_1, device=labels.device.type)
        result = torch.zeros_like(labels)

        for k in range(num_classes):
            class_probs = labels[:, k].unsqueeze(1)
            class_mean = torch.mean(inputs * class_probs, dim=(2, 3), keepdim=True) / \
                torch.add(torch.mean(class_probs, dim=(2, 3), keepdim=True), 1e-5)
            diff = (inputs - class_mean).pow(2).sum(dim=1).unsqueeze(1)
            weights = torch.exp(diff.pow(2).mul(-1 / self.sigma_2 ** 2))

            numerator = F.conv2d(class_probs * weights, kernel, padding=self.radius)
            denominator = F.conv2d(weights, kernel, padding=self.radius) + 1e-6
            result[:, k:k+1] = numerator / denominator

        return result
