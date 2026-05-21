import os
from pipeline   import run_pipeline, DEFAULT_CONFIG
from evaluation import load_ground_truth, calculate_iou, save_ground_truth_image


# =========================================================
# 실험 설정 목록
#
# Prior 유무 / threshold 파라미터 변경 조합
# 새로운 조합을 추가하려면 EXPERIMENT_CONFIGS에 항목 추가
# =========================================================
EXPERIMENT_CONFIGS: list[dict] = [

    # =====================================================
    # Baseline: prior 없음 + region_grow 없음 (순수 threshold)
    # =====================================================
    {
        "name": "baseline (no prior, no rg)",
        "config": {
            **DEFAULT_CONFIG,
            "rgb_prior":  {**DEFAULT_CONFIG["rgb_prior"],  "enabled": False},
            "hsv_prior":  {**DEFAULT_CONFIG["hsv_prior"],  "enabled": False},
            "region_grow": {**DEFAULT_CONFIG["region_grow"], "enabled": False},
        },
    },

    # =====================================================
    # Prior 적용 여부 비교
    # =====================================================
    {
        "name": "no_prior (rg active)",
        "config": {
            **DEFAULT_CONFIG,
            "rgb_prior": {**DEFAULT_CONFIG["rgb_prior"], "enabled": False},
            "hsv_prior": {**DEFAULT_CONFIG["hsv_prior"], "enabled": False},
        },
    },
    {
        "name": "both_prior (no region_grow)",
        "config": {
            **DEFAULT_CONFIG,
            "region_grow": {**DEFAULT_CONFIG["region_grow"], "enabled": False},
        },
    },
    {
        "name": "both_prior + region_grow",
        "config": DEFAULT_CONFIG,
    },

    # =====================================================
    # v_min 완화: 좌측 회색빛 나선 구름 검출 향상
    # 180→150 구간에서 가장 큰 IoU 향상폭 확인됨
    # =====================================================
    {
        "name": "low_vmin (v>90,  relax v>60)",
        "config": {
            **DEFAULT_CONFIG,
            "hsv_prior":   {**DEFAULT_CONFIG["hsv_prior"],   "v_min": 90},
            "hsv_relaxed": {**DEFAULT_CONFIG["hsv_relaxed"], "v_min": 60},
        },
    },

    # =====================================================
    # Adaptive Threshold: 국소 영역 기준 임계값
    # Otsu 단독보다 both(otsu+adaptive) OR 조합이 유효함 확인
    # =====================================================
    {
        "name": "both_thresh (otsu+adaptive)",
        "config": {
            **DEFAULT_CONFIG,
            "threshold": {**DEFAULT_CONFIG["threshold"], "method": "both", "adaptive_block": 51, "adaptive_c": -10},
        },
    },

    # =====================================================
    # 최종 조합: both_thresh + 가장 완화된 v_min
    # =====================================================
    {
        "name": "best_combo (both_thresh + v>90)",
        "config": {
            **DEFAULT_CONFIG,
            "threshold":   {**DEFAULT_CONFIG["threshold"],   "method": "both", "adaptive_block": 51, "adaptive_c": -10},
            "hsv_prior":   {**DEFAULT_CONFIG["hsv_prior"],   "v_min": 90},
            "hsv_relaxed": {**DEFAULT_CONFIG["hsv_relaxed"], "v_min": 60},
        },
    },

    # =====================================================
    # 낮은 threshold + prior 필터링
    # 애초에 넓게 잡고 prior로 깎아내는 원래 의도 구조
    # Otsu(자동)보다 낮은 고정값으로 더 많은 후보 확보
    # =====================================================
    {
        "name": "low_thresh (t=100) + both_prior",
        "config": {
            **DEFAULT_CONFIG,
            "threshold": {**DEFAULT_CONFIG["threshold"], "value": 100},
        },
    },
    {
        "name": "low_thresh (t=80)  + both_prior",
        "config": {
            **DEFAULT_CONFIG,
            "threshold": {**DEFAULT_CONFIG["threshold"], "value": 80},
        },
    },
    {
        "name": "low_thresh (t=60)  + both_prior",
        "config": {
            **DEFAULT_CONFIG,
            "threshold": {**DEFAULT_CONFIG["threshold"], "value": 60},
        },
    },
    {
        "name": "low_thresh (t=60)  + no_prior",
        "config": {
            **DEFAULT_CONFIG,
            "threshold":  {**DEFAULT_CONFIG["threshold"],  "value": 60},
            "rgb_prior":  {**DEFAULT_CONFIG["rgb_prior"],  "enabled": False},
            "hsv_prior":  {**DEFAULT_CONFIG["hsv_prior"],  "enabled": False},
            "region_grow": {**DEFAULT_CONFIG["region_grow"], "enabled": False},
        },
    },
]


# =====================================================
# 탐색 완료 / 주석 처리된 실험들
# =====================================================
# ARCHIVED_CONFIGS = [
#     {"name": "rgb_prior only", ...},
#     {"name": "hsv_prior only", ...},
#     {"name": "both_prior (tight s<40)", ...},
#     {"name": "both_prior (loose s<80)", ...},
#     {"name": "low_vmin (v>150)", ...},
#     {"name": "low_vmin (v>130)", ...},
#     {"name": "low_vmin (v>110)", ...},
#     {"name": "tight_s+low_v (s<40, v>130)", ...},
#     {"name": "tight_s+low_v (s<30, v>130)", ...},
#     {"name": "tight_s+low_v (s<40, v>110)", ...},
#     {"name": "adaptive_thresh (block=51)", ...},
#     {"name": "adaptive_thresh (block=31)", ...},
# ]


def compare_configs(
    rgb_path: str,
    nc_path: str,
    configs: list[dict] = EXPERIMENT_CONFIGS,
    result_dir: str | None = None,
) -> list[dict]:
    # =====================================================
    # 여러 설정 조합을 순서대로 실행하고 IoU 비교
    # result_dir 지정 시 각 설정의 overlay 이미지를 저장
    # =====================================================
    ground_truth = load_ground_truth(nc_path)
    results = []

    if result_dir:
        os.makedirs(result_dir, exist_ok=True)

    print(f"\n{'설정 이름':<35} IoU")
    print("-" * 45)

    for cfg in configs:
        mask, overlay = run_pipeline(rgb_path, cfg["config"])
        iou = calculate_iou(ground_truth, mask)

        results.append({"name": cfg["name"], "iou": iou})
        print(f"{cfg['name']:<35} {iou:.4f}")

        if result_dir:
            filename = cfg["name"].replace(" ", "_").replace("<", "").replace(">", "") + ".png"
            cv2.imwrite(os.path.join(result_dir, filename), overlay)

    return results


if __name__ == "__main__":
    import cv2

    SRC_DIR    = os.path.join(os.path.dirname(__file__), "src")
    RESULT_DIR = os.path.join(SRC_DIR, "result")
    rgb_path   = os.path.join(SRC_DIR, "gk2a_ami_le1b_rgb-s-true_ko020lc_202605210410.png")
    nc_path    = os.path.join(SRC_DIR, "gk2a_ami_le2_cld_ko020lc_202605210410.nc")

    save_ground_truth_image(nc_path, os.path.join(RESULT_DIR, "ground_truth.png"))
    compare_configs(rgb_path, nc_path, result_dir=RESULT_DIR)
