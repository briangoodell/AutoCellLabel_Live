import torch
import numpy as np

def batched_calculate_gncc(fixed_images, transformed_moving_images, memory_dict=None, batch_size=None):
    """
    Calculate GNCC between a fixed image and a batch of moving images.

    Args:
    - fixed_images: PyTorch tensor of shape (N x H x W)
    - transformed_moving_images: PyTorch tensor of shape (N x H x W)
    - memory_dict: dictionary of preallocated tensors
    - batch_size: number of images to process at once

    Returns:
    - List of GNCC values for each moving image.
    """
    if batch_size is None:
        batch_size = transformed_moving_images.shape[0]

    if memory_dict is None:
        memory_dict['mu_f'] = torch.zeros(fixed_images.shape[0], device=fixed_images.device, dtype=torch.float32)
        memory_dict['mu_m'] = torch.zeros(fixed_images.shape[0], device=fixed_images.device, dtype=torch.float32)

        memory_dict['a'] = torch.zeros(fixed_images.shape[0], device=fixed_images.device, dtype=torch.float32)
        memory_dict['b'] = torch.zeros(fixed_images.shape[0], device=fixed_images.device, dtype=torch.float32)

    mu_f = memory_dict['mu_f'][:batch_size]
    mu_m = memory_dict['mu_m'][:batch_size]
    a = memory_dict['a'][:batch_size]
    b = memory_dict['b'][:batch_size]

    fixed_images = fixed_images[:batch_size]
    transformed_moving_images = transformed_moving_images[:batch_size]

    mu_f[:] = torch.mean(fixed_images, dim=[1,2,3], keepdim=True)
    mu_m[:] = torch.mean(transformed_moving_images, dim=[1,2,3], keepdim=True)

    a[:] = torch.sum(((fixed_images - mu_f) * (transformed_moving_images - mu_m)), dim=[1,2,3]) / (fixed_images.shape[1] * fixed_images.shape[2] * fixed_images.shape[3])
    b[:] = torch.var(fixed_images, dim=[1,2,3]) * torch.var(transformed_moving_images, dim=[1,2,3])# + 1E-10

    return a * a / b

def calculate_gncc(fixed, moving):

    mu_f = np.mean(fixed)
    mu_m = np.mean(moving)
    a = np.sum((fixed - mu_f) * (moving - mu_m))
    b = np.sqrt(np.sum((fixed - mu_f) ** 2) * np.sum((moving - mu_m) ** 2))

    return a / b
