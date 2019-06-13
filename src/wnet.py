r"""
wnet.py
---------
Implementation of a W-Net CNN for unsupervised learning of image segmentations.
"""

from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn

from src.network import Network
from src.loss import NCutLoss2D, OpeningLoss2D


class ConvPoolBlock(nn.Module):
    r"""Performs multiple 2D convolutions, followed by a 2D max-pool operation.  Many of these are contained within
    each UNet module, for down sampling image data."""

    def __init__(self, in_features: int, out_features: int):
        r"""
        :param in_features: Number of feature channels in the incoming data
        :param out_features: Number of feature channels in the outgoing data
        """
        super(ConvPoolBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReplicationPad2d(2),
            nn.Conv2d(in_features, out_features, 5),
            nn.ReplicationPad2d(1),
            nn.Conv2d(out_features, out_features, 3),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Pushes a set of inputs (x) through the network.

        :param x: Input values
        :return: Module outputs
        """
        return self.layers(x)


class DeconvBlock(nn.Module):
    r"""Performs multiple 2D transposed convolutions, with a stride of 2 on the last layer.  Many of these are contained
    within each UNet module, for up sampling image data."""

    def __init__(self, in_features: int, out_features: int):
        r"""
        :param in_features: Number of feature channels in the incoming data
        :param out_features: Number of feature channels in the outgoing data
        """
        super(DeconvBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ConvTranspose2d(in_features, out_features, 5, padding=2),
            nn.ConvTranspose2d(out_features, out_features, 2, stride=2),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Pushes a set of inputs (x) through the network.

        :param x: Input values
        :return: Module outputs
        """
        return self.layers(x)


class OutputBlock(nn.Module):
    r"""Performs multiple 2D convolutions, without any pooling or strided operations."""

    def __init__(self, in_features: int, out_features: int):
        r"""
        :param in_features: Number of feature channels in the incoming data
        :param out_features: Number of feature channels in the outgoing data
        """
        super(OutputBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_features, out_features, 3),
            nn.ReplicationPad2d(1),
            nn.Conv2d(out_features, out_features, 3),
            nn.Conv2d(out_features, out_features, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Pushes a set of inputs (x) through the network.

        :param x: Input values
        :return: Module outputs
        """
        return self.layers(x)


class UNetEncoder(nn.Module):
    r"""The first half (encoder) of the W-Net architecture.  Returns class probabilities for each pixel in the image."""

    def __init__(self, num_channels: int = 3, num_classes: int = 10):
        r"""
        :param num_channels: Number of channels in the raw image data
        :param num_classes: Number of classes in the output class probabilities
        """
        super(UNetEncoder, self).__init__()
        self.conv1 = ConvPoolBlock(num_channels, 32)
        self.conv2 = ConvPoolBlock(32, 32)
        self.conv3 = ConvPoolBlock(32, 64)
        self.deconv1 = DeconvBlock(64, 32)
        self.deconv2 = DeconvBlock(64, 32)
        self.deconv3 = DeconvBlock(64, 32)
        self.output = OutputBlock(32, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Pushes a set of inputs (x) through the network.

        :param x: Input values
        :return: Network outputs
        """
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        x = self.conv3(c2)
        x = self.deconv1(x)
        x = self.deconv2(torch.cat((c2, x), dim=1))
        x = self.deconv3(torch.cat((c1, x), dim=1))
        x = self.output(x)

        return x

    @property
    def device(self) -> str:
        r"""Gets the name of the device where network weights/biases are stored. ('cpu' or 'cuda').
        """
        return self.conv1.layers[0].weight.device.type


class UNetDecoder(nn.Module):
    r"""The second half (decoder) of the W-Net architecture.  Returns a reconstruction of the original image."""

    def __init__(self, num_channels: int = 3, num_classes: int = 20):
        r"""
        :param num_channels: Number of channels in the raw image data
        :param num_classes: Number of classes in the output class probabilities
        """
        super(UNetDecoder, self).__init__()
        self.conv1 = ConvPoolBlock(num_classes, 32)
        self.conv2 = ConvPoolBlock(32, 32)
        self.conv3 = ConvPoolBlock(32, 64)
        self.deconv1 = DeconvBlock(64, 32)
        self.deconv2 = DeconvBlock(64, 32)
        self.deconv3 = DeconvBlock(64, 32)
        self.output = OutputBlock(32, num_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Pushes a set of inputs (x) through the network.

        :param x: Input values
        :return: Network outputs
        """
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        x = self.conv3(c2)
        x = self.deconv1(x)
        x = self.deconv2(torch.cat((c2, x), dim=1))
        x = self.deconv3(torch.cat((c1, x), dim=1))
        x = self.output(x)

        return x

    @property
    def device(self) -> str:
        r"""Gets the name of the device where network weights/biases are stored. ('cpu' or 'cuda').
        """
        return self.conv1.layers[0].weight.device.type


class WNet(Network):
    r"""Implements a W-Net CNN model for learning unsupervised image segmentations.  First encodes image data into
    class probabilities using UNet, and then decodes the labels into a reconstruction of the original image using a
    second UNet."""

    def __init__(self, num_channels: int = 3, num_classes: int = 20):
        r"""
        :param num_channels: Number of channels in the raw image data
        :param num_classes: Number of classes in the output class probabilities
        """
        super(WNet, self).__init__()
        self.encoder = UNetEncoder(num_channels=num_channels, num_classes=num_classes)
        self.decoder = UNetDecoder(num_channels=num_channels, num_classes=num_classes)

    def get_device_type(self) -> str:
        r"""Gets the name of the device where network weights/biases are stored. ('cpu' or 'cuda').
        """
        return self.encoder.device

    def forward_encode_(self, x: Tensor) -> Tensor:
        """Pushes a set of inputs (x) through only the encoder network.

        :param x: Input values
        :return: Class probabilities
        """
        device_type = self.get_device_type()
        if device_type == 'cuda':
            x = x.cuda()

        return self.encoder(x)

    def forward_reconstruct_(self, mask: Tensor) -> Tensor:
        """Pushes a set of class probabilities (mask) through only the decoder network.

        :param mask: Class probabilities
        :return: Reconstructed image
        """
        device_type = self.get_device_type()
        if device_type == 'cuda':
            mask = mask.cuda()

        outputs = self.decoder(mask)
        outputs = nn.ReLU()(outputs)

        return outputs

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Pushes a set of inputs (x) through the network.

        :param x: Input values
        :return: Network outputs
        """
        device_type = self.get_device_type()
        if device_type == 'cuda':
            x = x.cuda()

        encoded = self.forward_encode_(x).transpose(1, -1)
        mask = nn.Softmax(-1)(encoded).transpose(-1, 1)
        reconstructed = self.forward_reconstruct_(mask)

        return mask, reconstructed

    def get_loss(self, labels: Tensor, inputs: Tensor) -> Tensor:
        """Computes the training/validation loss of the bpr_model, given a set of inputs and truth labels.

        :param labels: Ground truth labels
        :param inputs: Training or validation inputs
        :return: Loss tensor
        """
        device_type = self.get_device_type()
        if device_type == 'cuda':
            labels, inputs = labels.cuda(), inputs.cuda()

        masks, outputs = self.forward(inputs)
        inputs, labels, outputs = inputs.contiguous(), labels.contiguous(), outputs.contiguous()

        # Weights for NCutLoss2D, MSELoss, and OpeningLoss2D, respectively
        alpha, beta, gamma = 1e-3, 1, 1e-1
        ncut_loss = alpha * NCutLoss2D()(masks, inputs)
        mse_loss = beta * nn.MSELoss()(outputs, inputs.detach())
        smooth_loss = gamma * OpeningLoss2D()(masks)
        loss = ncut_loss + mse_loss + smooth_loss

        return loss
