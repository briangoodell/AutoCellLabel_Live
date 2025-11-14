from .preprocess import initialize
from .gncc import calculate_gncc
import numpy as np
import torch
import torch.nn.functional as F


def transform_image(images_repeated,
                    dx_gpu,
                    dy_gpu,
                    angles_rad,
                    memory_dict,
                    interpolation="bilinear"):
    """
    Rotate the image by a list of angles.

    Arguments:

    - images_repeated: PyTorch tensor (N x C x H x W)
        * N := number of batches
        * C := number of channel (should be 1)
        * H := height of the image
        * W := width of the image
    - dx_gpu: list of translations in the x direction, as a PyTorch tensor
    - dy_gpu: list of translations in the y direction, as a PyTorch tensor
    - angles_rad: list of angles (in radians) to rotate the image, as a PyTorch tensor
    - memory_dict: dictionary of preallocated tensors
    - interpolation: interpolation method to use (default: bilinear)

    Returns:
    - a tensor of rotated images with size (N x 1 x H x W)
    """
    batch_size = dx_gpu.shape[0]
    H, W = images_repeated.shape[2], images_repeated.shape[3]

    # Initialize variables
    images_repeated = images_repeated[:batch_size]
    cos_vals = memory_dict['cos_vals'][:batch_size]
    sin_vals = memory_dict['sin_vals'][:batch_size]
    rotation_matrices = memory_dict['rotation_matrices'][:batch_size]
    output_tensor = memory_dict['output_tensor'][:batch_size]
    grid = memory_dict['grid'][:batch_size]

    # Rotation and translation
    cos_vals[:] = torch.cos(angles_rad)
    sin_vals[:] = torch.sin(angles_rad)

    # Directly assign values to the preallocated tensor
    rotation_matrices[:, 0, 0] = cos_vals
    rotation_matrices[:, 0, 1] = -sin_vals * H / W
    rotation_matrices[:, 0, 2] = dy_gpu
    rotation_matrices[:, 1, 0] = sin_vals * W / H
    rotation_matrices[:, 1, 1] = cos_vals
    rotation_matrices[:, 1, 2] = dx_gpu

    # Grid sample expects input in (N x C x H x W) format

    grid[:] = F.affine_grid(rotation_matrices, images_repeated.size(),
            align_corners=False)
    output_tensor[:] = F.grid_sample(images_repeated, grid,
            align_corners=False, mode=interpolation)


    return output_tensor


def transform_image_3d(resized_moving_image_xyz,
                      memory_dict,
                      best_transformation,
                      device,
                      dimension,
                      interpolation="bilinear"):
    """
    Transform a 3D image along a specified dimension.

    Arguments:
    - resized_moving_image_xyz: NumPy array
        The 3D image to be transformed.
    - memory_dict: dict
        Dictionary containing preallocated tensors for efficiency.
    - best_transformation: list or tuple
        Contains the best transformation parameters [dx, dy, angle].
    - device: torch.device
        The device (CPU or GPU) to perform computations on.
    - dimension: int
        The dimension along which to perform the transformation (0, 1, or 2).
    - interpolation: str, optional
        Interpolation method to use (default: "bilinear").

    Returns:
    - NumPy array
        The transformed 3D image.

    Raises:
    - ValueError: If dimension is not 0, 1, or 2.
    """
    axis_dimension = resized_moving_image_xyz.shape[dimension]
    if dimension == 0:
        moving_image_xyz_tensor = torch.tensor(
                resized_moving_image_xyz.astype(np.float32),
                device=device,
                dtype=torch.float32).unsqueeze(1).repeat(1, 1, 1, 1)
    elif dimension == 1:
        moving_image_xyz_tensor = torch.tensor(
                resized_moving_image_xyz.astype(np.float32).transpose(1, 0, 2),
                device=device,
                dtype=torch.float32).unsqueeze(1).repeat(1, 1, 1, 1)
    elif dimension == 2:
        moving_image_xyz_tensor = torch.tensor(
                resized_moving_image_xyz.astype(np.float32).transpose(2, 0, 1),
                device=device,
                dtype=torch.float32).unsqueeze(1).repeat(1, 1, 1, 1)
    else:
        raise ValueError("dimension must be 0, 1, or 2")

    transformed_moving_image_xyz = transform_image(
                moving_image_xyz_tensor,
                best_transformation[0].repeat(axis_dimension),
                best_transformation[1].repeat(axis_dimension),
                best_transformation[2].repeat(axis_dimension),
                memory_dict,
                interpolation=interpolation)
                
    if dimension == 0:
        return np.squeeze(transformed_moving_image_xyz.cpu().numpy(),
                          axis=1)
    elif dimension == 1:
        return np.transpose(np.squeeze(
                 transformed_moving_image_xyz.cpu().numpy(),
                 axis=1),
                 (1, 0, 2))
    elif dimension == 2:
        return np.transpose(np.squeeze(
                 transformed_moving_image_xyz.cpu().numpy(),
                 axis=1),
                 (1, 2, 0))


def transform_image_3d_torch(resized_moving_image_xyz,
                      memory_dict,
                      best_transformation,
                      device,
                      dimension,
                      interpolation="bilinear"):
    """
    Transform a 3D image along a specified dimension.

    Arguments:
    - resized_moving_image_xyz: NumPy array
        The 3D image to be transformed.
    - memory_dict: dict
        Dictionary containing preallocated tensors for efficiency.
    - best_transformation: list or tuple
        Contains the best transformation parameters [dx, dy, angle].
    - device: torch.device
        The device (CPU or GPU) to perform computations on.
    - dimension: int
        The dimension along which to perform the transformation (0, 1, or 2).
    - interpolation: str, optional
        Interpolation method to use (default: "bilinear").

    Returns:
    - NumPy array
        The transformed 3D image.

    Raises:
    - ValueError: If dimension is not 0, 1, or 2.
    """
    axis_dimension = resized_moving_image_xyz.shape[dimension]
    if dimension == 0:
        moving_image_xyz_tensor = resized_moving_image_xyz.unsqueeze(1).repeat(1, 1, 1, 1)
    elif dimension == 1:
        moving_image_xyz_tensor = resized_moving_image_xyz.permute(1, 0, 2).unsqueeze(1).repeat(1, 1, 1, 1)
    elif dimension == 2:
        moving_image_xyz_tensor = resized_moving_image_xyz.permute(2, 0, 1).unsqueeze(1).repeat(1, 1, 1, 1)
    else:
        raise ValueError("dimension must be 0, 1, or 2")

    # print(moving_image_xyz_tensor.shape)

    transformed_moving_image_xyz = transform_image(
                moving_image_xyz_tensor,
                best_transformation[0].repeat(axis_dimension),
                best_transformation[1].repeat(axis_dimension),
                best_transformation[2].repeat(axis_dimension),
                memory_dict,
                interpolation=interpolation)
                
    if dimension == 0:
        return transformed_moving_image_xyz.squeeze(1)
    elif dimension == 1:
        return transformed_moving_image_xyznp.squeeze(1).permute((1, 0, 2))
    elif dimension == 2:
        return transformed_moving_image_xyz.squeeze(1).permute((1, 2, 0))


def translate_along_z(shift_range,
                      resized_fixed_image_xyz,
                      transformed_moving_image_xyz,
                      moving_image_median):
    """
    Translate the moving image along the z-axis to find the best alignment with the fixed image.

    Arguments:
    - shift_range: range or list
        The range of z-axis shifts to try.
    - resized_fixed_image_xyz: NumPy array
        The fixed 3D image.
    - transformed_moving_image_xyz: NumPy array
        The moving 3D image after initial Euler transformation, but in need of z-translation.
    - moving_image_median: float
        The median value of the moving image, used for padding.

    Returns:
    - dz: int
        The optimal z-axis shift.
    - gncc: float
        The global normalized cross-correlation value for the best alignment.
    - final_moving_image_xyz: NumPy array
        The final transformed and z-shifted moving image.
    """

    final_moving_image_xyz = np.full(
            transformed_moving_image_xyz.shape,
            moving_image_median)

    dz, gncc = search_for_z(shift_range,
                      resized_fixed_image_xyz,
                      transformed_moving_image_xyz,
                      moving_image_median)
    if dz < 0:
        final_moving_image_xyz[:, :, :dz] = \
            transformed_moving_image_xyz[:, :, -dz:]
    elif dz > 0:
        final_moving_image_xyz[:, :, dz:] = \
            transformed_moving_image_xyz[:, :, :-dz]
    elif dz == 0:
        final_moving_image_xyz = transformed_moving_image_xyz

    return dz, gncc, final_moving_image_xyz


def search_for_z(shift_range, resized_fixed_image_xyz,
        transformed_moving_image_xyz, moving_image_median):
    """
    Search for the optimal z-axis shift to align the moving image with the fixed image.

    Arguments:
    - shift_range: range or list
        The range of z-axis shifts to try.
    - resized_fixed_image_xyz: NumPy array
        The fixed 3D image.
    - transformed_moving_image_xyz: NumPy array
        The moving 3D image after initial Euler transformation, but in need of z-translation.
    - moving_image_median: float
        The median value of the moving image, used for padding.

    Returns:
    - dz: int
        The optimal z-axis shift.
    - gncc: float
        The global normalized cross-correlation value for the best alignment.
    """

    new_moving_image_xyz = np.full(
            transformed_moving_image_xyz.shape,
            moving_image_median)

    gncc = calculate_gncc(resized_fixed_image_xyz,
            transformed_moving_image_xyz)
    dz = 0
    for shift in shift_range:
        if shift < 0:
            new_moving_image_xyz[:, :, :shift] = \
                transformed_moving_image_xyz[:, :, -shift:]
        elif shift > 0:
            new_moving_image_xyz[:, :, shift:] = \
                transformed_moving_image_xyz[:, :, :-shift]

        new_gncc = calculate_gncc(resized_fixed_image_xyz, new_moving_image_xyz)
        if new_gncc > gncc:
            gncc = new_gncc
            dz = shift

    return dz, gncc
