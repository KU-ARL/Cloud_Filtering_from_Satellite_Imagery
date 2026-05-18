import numpy as np
from netCDF4 import Dataset


def calculate_cloud_iou(nc_path: str, my_mask: np.ndarray) -> float:
    """
    Calculate IoU between KMA cloud ma(skCL D) and user's cloud mask.

    Args:
        nc_path:
            Path to KMA .nc file

        my_mask:
            User-generated cloud mask
            Shape must match CLD shape
            cloud = 1
            clear = 0

            Allowed formats:
            - 0 / 1
            - 0 / 255

    Returns:
        IoU score (float)
    """

    # =====================================================
    # 1. Read CLD data
    # =====================================================

    nc = Dataset(nc_path)

    cld = nc.variables['CLD'][:]

    # =====================================================
    # 2. Convert ground truth to binary mask
    # =====================================================
    # 0,1 -> cloud
    # 2   -> clear

    ground_truth = np.where(cld <= 1, 1, 0).astype(np.uint8)

    # =====================================================
    # 3. Convert user's mask to binary
    # =====================================================

    my_mask = (my_mask > 0).astype(np.uint8)

    # =====================================================
    # 4. Shape check
    # =====================================================

    if ground_truth.shape != my_mask.shape:
        raise ValueError(
            f"Shape mismatch: "
            f"ground_truth={ground_truth.shape}, "
            f"my_mask={my_mask.shape}"
        )

    # =====================================================
    # 5. Calculate IoU
    # =====================================================

    intersection = np.logical_and(ground_truth, my_mask)
    union = np.logical_or(ground_truth, my_mask)

    union_sum = union.sum()

    if union_sum == 0:
        return 0.0

    iou = intersection.sum() / union_sum

    return float(iou)