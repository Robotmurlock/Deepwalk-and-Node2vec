"""
PyTorch's extension with simple functions
"""
from torch.optim import Optimizer


def get_optim_lr(optimizer: Optimizer) -> float:
    """
    Gets current optimizer learning rate

    Args:
        optimizer: Torch optimizer

    Returns:
        Learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_optim_lr(optimizer: Optimizer, lr: float) -> None:
    """
    Manually sets optimizer learning rate.

    Args:
        optimizer: Torch optimizer
        lr: New learning rate value
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
