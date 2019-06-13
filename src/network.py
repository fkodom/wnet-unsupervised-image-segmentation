import time
from typing import Tuple, Iterable, Callable

from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim

from utils.data import get_data_loader


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

    def get_device_type(self) -> str or Exception:
        """Gets the name of the device where this bpr_model is currently stored ('cpu' or 'cuda')
        """
        msg = f'get_device_type has not been implemented for {self.__class__.__name__}'
        raise NotImplementedError(msg)

    def forward(self, x: Tensor) -> Tensor or Exception:
        """Pushes a set of inputs (x) through the network.

        :param x: Input values
        :return: Network outputs
        """
        msg = f'forward has not been implemented for {self.__class__.__name__}'
        raise NotImplementedError(msg)

    def batch_forward(self, x: Tensor, batch: int = 25) -> Tensor:
        """Pushes a large number of inputs through the bpr_model using batch processing.  Intended
        to be more memory efficient, allowing CUDA usage for large input arrays.

        :param x: Input tensors
        :param batch: batch size for sequential processing of the inputs
        :return: Network outputs
        """
        return torch.cat(
            tuple(self.forward(x[i:i+batch]).cpu() for i in range(0, x.shape[0], batch)), 0)

    def get_loss(self, labels: Tensor, inputs: Tensor) -> Tensor or Exception:
        """Computes the training/validation loss of the bpr_model, given a set of inputs and truth labels.

        :param labels: Ground truth labels
        :param inputs: Training or validation inputs
        :return: Loss tensor
        """
        msg = f'get_loss has not been implemented for {self.__class__.__name__}'
        raise NotImplementedError(msg)

    def step(self, x_train: Tensor, y_train: Tensor, x_val: Tensor or None, y_val: Tensor or None,
             optimizer, batch_size: int) -> Tuple[Tensor, Tensor or None]:
        """Performs a full epoch of training and validation.

        :param x_train: Training inputs
        :param y_train: Training truth labels
        :param x_val: Validation inputs
        :param y_val: Validation truth labels
        :param optimizer:
        :param batch_size: Batch size for training
        """
        device_type = self.get_device_type()

        # optimizer = optim.SGD(self.parameters(), lr=learn_rate)

        tr_loss = 0
        train_loader = get_data_loader(x_train, y_train, batch_size)
        train_batches = len(train_loader)
        progress_bar = tqdm(desc='TRAIN', total=train_batches)

        for i, (inputs, labels) in enumerate(train_loader):
            progress_bar.update()
            if device_type == 'cuda':
                inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            loss = self.get_loss(labels, inputs)
            loss.backward()
            optimizer.step()
            tr_loss += loss.data.item() / train_batches

        progress_bar.close()
        if x_val is None or y_val is None:
            return tr_loss, None

        va_loss = 0
        val_loader = get_data_loader(x_val, y_val, batch_size)
        val_batches = len(val_loader)
        progress_bar = tqdm(desc='  VAL', total=val_batches)

        for i, (inputs, labels) in enumerate(val_loader):
            progress_bar.update()
            if device_type == 'cuda':
                inputs, labels = inputs.cuda(), labels.cuda()

            loss = self.get_loss(labels, inputs)
            va_loss += loss.data.item() / val_batches

        progress_bar.close()

        return tr_loss, va_loss

    def fit(self, x_train: Tensor, y_train: Tensor, x_val: Tensor or None = None, y_val: Tensor or None = None,
            learn_rate: float = 1e-3, weight_decay: float = 1e-3, epochs: int = 10, batch_size: int = 25,
            callbacks: Iterable[Callable] = (), plot: bool = True) -> None:
        """Initiates a complete training sequence for the network.

        :param x_train: Training inputs
        :param y_train: Training truth labels
        :param x_val: (optional) Validation inputs
        :param y_val: (optional) Validation truth labels
        :param learn_rate: Learning rate for optimization algorithm
        :param weight_decay: L2 loss for weights/biases during training
        :param epochs: Number of training epochs
        :param batch_size: Batch size for training
        :param callbacks: Callable functions to execute at the end of each epoch
        :param plot: If True, the training and validation loss are plotted when training is complete (default: True)
        """
        num_epochs = 0
        tr_losses, va_losses = [], []
        optimizer = optim.Adam(self.parameters(), lr=learn_rate, weight_decay=weight_decay)

        for epoch in range(epochs):
            num_epochs = epoch + 1
            tr_loss, va_loss = self.step(x_train, y_train, x_val, y_val, optimizer, batch_size)
            tr_losses.append(tr_loss)
            va_losses.append(va_loss)

            time.sleep(1e-3)    # ensures progress_bar has completely closed
            print(f'EPOCH:      {num_epochs}')
            print(f'Train loss: {tr_loss:.4E}')
            print(f'Valid loss: {va_loss:.4E}')

            for callback in callbacks:
                callback(self)

            if len(va_losses) >= 3 and va_losses[-1] > va_losses[-3]:
                print('Validation loss stopped decreasing.  Ending training.')
                break

        if plot:
            t = range(1, num_epochs)
            plt.plot(t, tr_losses[1:], 'b-', t, va_losses[1:], 'r-')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(['Training Loss', 'Validation Loss'])
            plt.title('Training Statistics')
            plt.show()
