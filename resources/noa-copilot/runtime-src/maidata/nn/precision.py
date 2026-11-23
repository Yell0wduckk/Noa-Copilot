"""Shared CUDA runtime precision policy; no model parameters or saved state."""
import torch


def enable_tf32(device):
    """Allow Tensor Cores for FP32 matrix/conv branches on CUDA.

    Autocast and explicit FP32 outputs retain their dtypes. TF32 changes CUDA
    multiply precision, so it is not a bitwise-equivalence guarantee.
    """
    if torch.device(device).type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
