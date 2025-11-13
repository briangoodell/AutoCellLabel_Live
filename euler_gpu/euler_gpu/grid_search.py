from euler_gpu.gncc import batched_calculate_gncc
from euler_gpu.transform import transform_image
import torch


def grid_search(memory_dict):
    """
    Compute GNCC values for all possible combinations of dx, dy, and theta in batches.

    Args:
    - memory_dict: dictionary of preallocated tensors containing input images and other data.

    Returns:
    - List of GNCC values for each transformation combination.
    """
    transformations = memory_dict["transformations"]
    batch_size = memory_dict["moving_images_repeated"].shape[0]

    for i in range(0, len(transformations), batch_size):
        max_idx = min(i+batch_size, len(transformations))

        dx_gpu = memory_dict["dx_gpu"][i:max_idx]
        dy_gpu = memory_dict["dy_gpu"][i:max_idx]
        angles_rad = memory_dict["angles_rad"][i:max_idx]

        # Rotate and translate images for the current batch
        transform_image(memory_dict['moving_images_repeated'], dx_gpu, dy_gpu, angles_rad, memory_dict)

        # Compute GNCC for the current batch of transformed images
        batched_gncc_values = batched_calculate_gncc(
                        memory_dict['fixed_images_repeated'],
                        memory_dict['output_tensor'],
                        memory_dict=memory_dict,
                        batch_size=dx_gpu.shape[0])
        memory_dict['gncc_results'][i:min(i+batch_size, len(transformations))] = batched_gncc_values[:]

    best_score, best_transformation = get_best_score(memory_dict['gncc_results'], memory_dict)

    return best_score, best_transformation


def get_best_score(gncc_results, memory_dict):
    """
    Get the best score and corresponding transformation from a list of GNCC results.

    Args:
    - gncc_results: PyTorch tensor of GNCC values for each transformation combination.
    - memory_dict: Dictionary containing tensors of the transformations used.

    Returns:
    - The best GNCC score.
    - The transformation corresponding to the best score.
    """
    gncc_results[torch.isnan(gncc_results)] = float('-inf')
    best_score, best_index = torch.max(gncc_results, 0)

    best_dx = memory_dict['dx_gpu'][best_index:best_index+1]
    best_dy = memory_dict['dy_gpu'][best_index:best_index+1]
    best_angle = memory_dict['angles_rad'][best_index:best_index+1]

    return torch.sqrt(best_score), (best_dx, best_dy, best_angle)
