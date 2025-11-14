import torch, itertools
from math import pi
import numpy as np

def initialize(fixed_image, moving_image, dx, dy, angles, batch_size, device):
    """
    Initialize the memory dictionary for the GNCC calculation.

    Arguments:
    - fixed_image: Fixed image, as an array of shape (H x W)
    - moving_image: Moving image, as an array of shape (H x W)
    - dx: list of translations in the x direction
    - dy: list of translations in the y direction
    - angles: list of angles to rotate the image (in degrees)
    - batch_size: number of images to process at once
    - device: PyTorch device to use (e.g. torch.device("cuda:0"))
    """
    memory_dict = dict()

    memory_dict["transformations"] = list(itertools.product(dx, dy, angles))
    transformations = memory_dict["transformations"]

    # Move data to GPU
    if not isinstance(fixed_image, torch.Tensor):
        fixed_image = torch.tensor(fixed_image, device=device, dtype=torch.float32)
        moving_image = torch.tensor(moving_image, device=device, dtype=torch.float32)

    # Repeat the image tensor to batch process
    memory_dict["moving_images_repeated"] = moving_image.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    memory_dict["fixed_images_repeated"] = fixed_image.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    # Preallocate memory
    memory_dict["output_tensor"] = torch.zeros_like(memory_dict["moving_images_repeated"], device=device, dtype=torch.float32)
    memory_dict["grid"] = torch.zeros((batch_size, fixed_image.shape[0], fixed_image.shape[1], 2), device=device, dtype=torch.float32)
    memory_dict["gncc_results"] = torch.zeros(len(transformations), device=device, dtype=torch.float32)
    memory_dict["mu_f"] = torch.zeros((batch_size, 1, 1, 1), device=device, dtype=torch.float32)
    memory_dict["mu_m"] = torch.zeros((batch_size, 1, 1, 1), device=device, dtype=torch.float32)
    memory_dict["a"] = torch.zeros(batch_size, device=device, dtype=torch.float32)
    memory_dict["b"] = torch.zeros(batch_size, device=device, dtype=torch.float32)
    memory_dict["angles_rad"] = torch.zeros(len(transformations), device=device, dtype=torch.float32)
    memory_dict["dx_gpu"] = torch.zeros(len(transformations), device=device, dtype=torch.float32)
    memory_dict["dy_gpu"] = torch.zeros(len(transformations), device=device, dtype=torch.float32)
    memory_dict["cos_vals"] = torch.zeros(batch_size, device=device, dtype=torch.float32)
    memory_dict["sin_vals"] = torch.zeros(batch_size, device=device, dtype=torch.float32)
    memory_dict["rotation_matrices"] = torch.zeros((batch_size, 2, 3), device=device, dtype=torch.float32)

    # Precompute lists of transformations to try
    for i in range(0, len(transformations), batch_size):
        max_idx = min(i+batch_size, len(transformations))
        batched_transformations = transformations[i:max_idx]

        # # Unzip the transformations
        batched_dx, batched_dy, batched_angles = zip(*batched_transformations)
        memory_dict["dx_gpu"][i:max_idx] = torch.tensor(batched_dx, device=device, dtype=torch.float32)
        memory_dict["dy_gpu"][i:max_idx] = torch.tensor(batched_dy, device=device, dtype=torch.float32)
        memory_dict["angles_rad"][i:max_idx] = (torch.tensor(batched_angles) * pi / 180).to(device)

    return memory_dict


def max_intensity_projection_and_downsample(image,
            downsample_factor,
            projection_axis):

    """
    Create a maximum-intensity projection of a 3D image along the z dimension and then downsample it.

    Parameters:
    # - image (numpy array): 3D image of shape (width, height, depth), i.e., (x, y, z)
    - image (numpy array): 3D image of shape (depth, height, width), i.e., (z, y, x)
    - downsample_factor (int): factor by which to downsample the 2D projection

    Returns:
    - downsampled_image (numpy array): 2D downsampled image of shape (width // downsample_factor, height // downsample_factor)
    """
    # Maximum intensity projection along x, y, or z dimension
    mip = np.max(image, axis=projection_axis)

    # Downsampling
    downsampled_shape = (mip.shape[0] // downsample_factor, mip.shape[1] // downsample_factor)
    downsampled_image = np.zeros(downsampled_shape)

    for i in range(0, mip.shape[0], downsample_factor):
        for j in range(0, mip.shape[1], downsample_factor):
            downsampled_image[i // downsample_factor, j // downsample_factor] = np.mean(
                mip[i:i+downsample_factor, j:j+downsample_factor]
            )

    return downsampled_image


def max_intensity_projection_and_downsample_torch(image,
            downsample_factor,
            projection_axis):

    """
    Create a maximum-intensity projection of a 3D image along the z dimension and then downsample it.

    Parameters:
    - image (numpy array): 3D image of shape (width, height, depth), i.e., (x, y, z)
    - downsample_factor (int): factor by which to downsample the 2D projection

    Returns:
    - downsampled_image (numpy array): 2D downsampled image of shape (width // downsample_factor, height // downsample_factor)
    """
    # Maximum intensity projection along x, y, or z dimension
    mip = torch.amax(image, projection_axis)

    # Downsampling
    downsampled_shape = (mip.shape[0] // downsample_factor, mip.shape[1] // downsample_factor)
    downsampled_image = torch.zeros(downsampled_shape, device=image.device)

    for i in range(0, mip.shape[0], downsample_factor):
        for j in range(0, mip.shape[1], downsample_factor):
            downsampled_image[i // downsample_factor, j // downsample_factor] = torch.mean(
                mip[i:i+downsample_factor, j:j+downsample_factor]
            )

    return downsampled_image