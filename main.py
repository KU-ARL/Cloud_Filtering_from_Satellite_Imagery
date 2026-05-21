import os
import cv2

from pipeline   import run_pipeline, DEFAULT_CONFIG
from evaluation import load_ground_truth, calculate_iou

# =========================================================
# 입력 파일 경로 설정
# 입력 파일 2개를 src/ 디렉토리에 위치시켜야 함
# =========================================================
SRC_DIR   = os.path.join(os.path.dirname(__file__), "src")
rgb_image = "gk2a_ami_le1b_rgb-s-true_ko020lc_202605210410.png"
nc_file   = "gk2a_ami_le2_cld_ko020lc_202605210410.nc"

rgb_path = os.path.join(SRC_DIR, rgb_image)
nc_path  = os.path.join(SRC_DIR, nc_file)

# =========================================================
# 파이프라인 실행
# =========================================================
mask, result = run_pipeline(rgb_path, DEFAULT_CONFIG)

# =========================================================
# IoU 평가
# =========================================================
ground_truth = load_ground_truth(nc_path)
iou = calculate_iou(ground_truth, mask)

print(f"IoU = {iou:.4f}")

# =========================================================
# 결과 시각화
# =========================================================
cv2.imshow("Cloud Extraction Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
