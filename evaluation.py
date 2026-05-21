import cv2
import numpy as np
from netCDF4 import Dataset


# =========================================================
# 평가 모듈
#
# KMA NetCDF (CTT) Ground Truth 로드 + IoU 계산
# RGB PNG와 NetCDF 해상도 불일치 → align_mask로 자동 해결
# =========================================================


def load_ground_truth(nc_path: str) -> np.ndarray:
    # =====================================================
    # Ground Truth 로드
    #
    # GK2A CLD 변수: 0,1 → cloud / 2 → clear
    # CTT 저온 영역(-40°C 이하) = 높은 구름 = 정답 마스크
    # =====================================================
    nc = Dataset(nc_path)
    cld = nc.variables['CLD'][:]

    ground_truth = np.where(cld <= 1, 1, 0).astype(np.uint8)
    return ground_truth


def align_mask(
    my_mask: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    # =====================================================
    # Mask 해상도 정렬
    #
    # RGB PNG와 NetCDF CLD 그리드의 해상도 불일치 해결
    # INTER_NEAREST: binary mask 픽셀값(0/1) 보존
    # =====================================================
    h, w = target_shape
    resized = cv2.resize(
        (my_mask > 0).astype(np.uint8) * 255,
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )
    return (resized > 0).astype(np.uint8)


def calculate_iou(
    ground_truth: np.ndarray,
    my_mask: np.ndarray,
) -> float:
    # =====================================================
    # IoU 계산
    #
    # 형태 불일치 시 align_mask로 자동 보정
    # =====================================================
    my_mask_bin = (my_mask > 0).astype(np.uint8)

    if ground_truth.shape != my_mask_bin.shape:
        my_mask_bin = align_mask(my_mask_bin, ground_truth.shape)

    intersection = np.logical_and(ground_truth, my_mask_bin).sum()
    union        = np.logical_or(ground_truth,  my_mask_bin).sum()

    return float(intersection / union) if union > 0 else 0.0
