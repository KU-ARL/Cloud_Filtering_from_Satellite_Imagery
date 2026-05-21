import os
from pipeline   import run_pipeline, DEFAULT_CONFIG
from evaluation import load_ground_truth, calculate_metrics, save_ground_truth_image


# =========================================================
# 실험 설정 목록
#
# Prior 유무 / threshold 파라미터 변경 조합
# 새로운 조합을 추가하려면 EXPERIMENT_CONFIGS에 항목 추가
# =========================================================
EXPERIMENT_CONFIGS: list[dict] = [

    # =====================================================
    # [1] Baseline
    #     순수 Otsu+Canny, prior/region_grow 없음
    #     → 아무것도 안 했을 때의 기준선
    # =====================================================
    {
        "name": "baseline (no prior, no rg)",
        "config": {
            **DEFAULT_CONFIG,
            "rgb_prior":   {**DEFAULT_CONFIG["rgb_prior"],   "enabled": False},
            "hsv_prior":   {**DEFAULT_CONFIG["hsv_prior"],   "enabled": False},
            "region_grow": {**DEFAULT_CONFIG["region_grow"], "enabled": False},
        },
    },

    # =====================================================
    # [2] Prior only (region_grow 없음)
    #     → Prior가 Precision을 극대화함을 증명
    #       "구름이라고 판단하면 거의 틀리지 않는다"
    # =====================================================
    {
        "name": "both_prior (no region_grow)",
        "config": {
            **DEFAULT_CONFIG,
            "region_grow": {**DEFAULT_CONFIG["region_grow"], "enabled": False},
        },
    },

    # =====================================================
    # [3] Prior + Region Growing
    #     → Region Growing이 Recall을 얼마나 회복하는지 확인
    # =====================================================
    {
        "name": "both_prior + region_grow",
        "config": DEFAULT_CONFIG,
    },

    # =====================================================
    # [4] Best combo (Prior 방법 중 F1 최고)
    #     Adaptive Threshold + v_min 완화 + region_grow
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
    # [5] Low threshold + prior
    #     → threshold를 극단적으로 낮춰도 prior 결과가 동일함
    #       "Prior가 있으면 threshold는 무의미하다" 증명
    # =====================================================
    {
        "name": "low_thresh (t=30) + both_prior",
        "config": {
            **DEFAULT_CONFIG,
            "threshold": {**DEFAULT_CONFIG["threshold"], "value": 30},
        },
    },

    # =====================================================
    # [6] Low threshold + no prior (Recall 최대)
    #     → Precision-Recall 트레이드오프 극단 비교군
    #       "Prior 없이 무조건 많이 잡으면 Recall은 높지만 Precision이 낮다"
    # =====================================================
    {
        "name": "low_thresh (t=30) + no_prior",
        "config": {
            **DEFAULT_CONFIG,
            "threshold":   {**DEFAULT_CONFIG["threshold"],   "value": 30},
            "rgb_prior":   {**DEFAULT_CONFIG["rgb_prior"],   "enabled": False},
            "hsv_prior":   {**DEFAULT_CONFIG["hsv_prior"],   "enabled": False},
            "region_grow": {**DEFAULT_CONFIG["region_grow"], "enabled": False},
        },
    },
]


# =====================================================
# 아카이브 (탐색 완료, 위 실험들로 대표됨)
# =====================================================
# - no_prior (rg active)           → [6]으로 대체
# - low_vmin sweep (v>150~90)      → [4]에 통합
# - both_thresh alone              → [4]에 통합
# - low_thresh (t=100/80/60)+prior → [5]와 동일 결과 확인됨
# - rgb/hsv prior only             → [2]로 대표
# - s_max 변형 실험들              → 효과 없음 확인됨
# - adaptive_thresh block 변형     → both_thresh가 단독보다 유효 확인됨


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

    print(f"\n{'설정 이름':<35} {'IoU':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    print("-" * 65)

    for cfg in configs:
        mask, overlay = run_pipeline(rgb_path, cfg["config"])
        m = calculate_metrics(ground_truth, mask)

        results.append({"name": cfg["name"], **m})
        print(
            f"{cfg['name']:<35} "
            f"{m['iou']:>6.4f}  "
            f"{m['precision']:>6.4f}  "
            f"{m['recall']:>6.4f}  "
            f"{m['f1']:>6.4f}"
        )

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
