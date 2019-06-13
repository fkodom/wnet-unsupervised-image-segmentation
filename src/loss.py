r"""
loss.py
--------
Implementations of loss functions used for training W-Net CNN models.
"""

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.ndimage import grey_opening

from utils.filter import gaussian_kernel


class NCutLoss2D(nn.Module):
    r"""Implementation of the continuous N-Cut loss, as in:
    'W-Net: A Deep Model for Fully Unsupervised Image Segmentation', by Xia, Kulis (2017)"""

    def __init__(self, radius: int = 4, sigma_1: float = 5, sigma_2: float = 1):
        r"""
        :param radius: Radius of the spatial interaction term
        :param sigma_1: Standard deviation of the spatial Gaussian interaction
        :param sigma_2: Standard deviation of the pixel value Gaussian interaction
        """
        super(NCutLoss2D, self).__init__()
        self.radius = radius
        self.sigma_1 = sigma_1  # Spatial standard deviation
        self.sigma_2 = sigma_2  # Pixel value standard deviation

    def forward(self, labels: Tensor, inputs: Tensor) -> Tensor:
        r"""Computes the continuous N-Cut loss, given a set of class probabilities (labels) and raw images (inputs).
        Small modifications have been made here for efficiency -- specifically, we compute the pixel-wise weights
        relative to the class-wide average, rather than for every individual pixel.

        :param labels: Predicted class probabilities
        :param inputs: Raw images
        :return: Continuous N-Cut loss
        """
        num_classes = labels.shape[1]
        kernel = gaussian_kernel(radius=self.radius, sigma=self.sigma_1, device=labels.device.type)
        loss = 0

        for k in range(num_classes):
            # Compute the average pixel value for this class, and the difference from each pixel
            class_probs = labels[:, k].unsqueeze(1)
            class_mean = torch.mean(inputs * class_probs, dim=(2, 3), keepdim=True) / \
                torch.add(torch.mean(class_probs, dim=(2, 3), keepdim=True), 1e-5)
            diff = (inputs - class_mean).pow(2).sum(dim=1).unsqueeze(1)

            # Weight the loss by the difference from the class average.
            weights = torch.exp(diff.pow(2).mul(-1 / self.sigma_2 ** 2))

            # Compute N-cut loss, using the computed weights matrix, and a Gaussian spatial filter
            numerator = torch.sum(class_probs * F.conv2d(class_probs * weights, kernel, padding=self.radius))
            denominator = torch.sum(class_probs * F.conv2d(weights, kernel, padding=self.radius))
            loss += nn.L1Loss()(numerator / torch.add(denominator, 1e-6), torch.zeros_like(numerator))

        return num_classes - loss


class OpeningLoss2D(nn.Module):
    r"""Computes the Mean Squared Error between computed class probabilities their grey opening.  Grey opening is a
    morphology operation, which performs an erosion followed by dilation.  Conceptually, this encourages the network
    to return sharper boundaries to objects in the class probabilities.

    NOTE:  Original loss term -- not derived from the paper for NCutLoss2D."""

    def __init__(self, radius: int = 2):
        r"""
        :param radius: Radius for the channel-wise grey opening operation
        """
        super(OpeningLoss2D, self).__init__()
        self.radius = radius

    def forward(self, labels: Tensor, *args) -> Tensor:
        r"""Computes the Opening loss -- i.e. the MSE due to performing a greyscale opening operation.

        :param labels: Predicted class probabilities
        :param args: Extra inputs, in case user also provides input/output image values.
        :return: Opening loss
        """
        smooth_labels = labels.clone().detach().cpu().numpy()
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                smooth_labels[i, j] = grey_opening(smooth_labels[i, j], self.radius)

        smooth_labels = torch.from_numpy(smooth_labels.astype(np.float32))
        if labels.device.type == 'cuda':
            smooth_labels = smooth_labels.cuda()

        return nn.MSELoss()(labels, smooth_labels.detach())
