r"""
callbacks.py
---------------
"""

from typing import Callable

import torch


def model_checkpoint(save_path: str) -> Callable:
    """Defines a callback for saving PyTorch models during training.  The callback function will always accept `self`
    as the first argument.  We include `*args` positional arguments for added flexibility, so that a network could
    pass multiple arguments (e.g. training/validation loss, epoch number, etc.) without breaking it.

    :param save_path:  Absolute path to the bpr_model's save file
    :return:  Callback function for bpr_model saving
    """
    def callback(model: torch.nn.Module, *args):
        device = model.get_device_type()
        torch.save(model.cpu(), save_path)
        if device == 'cuda':
            model.cuda()

    return callback
