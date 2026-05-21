import os
from pipeline   import run_pipeline, DEFAULT_CONFIG
from evaluation import load_ground_truth, calculate_iou


# =========================================================
# 실험 설정 목록
#
# Prior 유무 / threshold 파라미터 변경 조합
# 새로운 조합을 추가하려면 EXPERIMENT_CONFIGS에 항목 추가
# =========================================================
EXPERIMENT_CONFIGS: list[dict] = [
    {
        "name": "no_prior",
        "config": {
            **DEFAULT_CONFIG,
            "rgb_prior": {**DEFAULT_CONFIG["rgb_prior"], "enabled": False},
            "hsv_prior": {**DEFAULT_CONFIG["hsv_prior"], "enabled": False},
        },
    },
    {
        "name": "rgb_prior only",
        "config": {
            **DEFAULT_CONFIG,
            "hsv_prior": {**DEFAULT_CONFIG["hsv_prior"], "enabled": False},
        },
    },
    {
        "name": "hsv_prior only",
        "config": {
            **DEFAULT_CONFIG,
            "rgb_prior": {**DEFAULT_CONFIG["rgb_prior"], "enabled": False},
        },
    },
    {
        "name": "both_prior (default s<60)",
        "config": DEFAULT_CONFIG,
    },
    {
        "name": "both_prior (tight   s<40)",
        "config": {
            **DEFAULT_CONFIG,
            "hsv_prior": {**DEFAULT_CONFIG["hsv_prior"], "s_max": 40},
        },
    },
    {
        "name": "both_prior (loose   s<80)",
        "config": {
            **DEFAULT_CONFIG,
            "hsv_prior": {**DEFAULT_CONFIG["hsv_prior"], "s_max": 80},
        },
    },
    {
        "name": "both_prior (no region_grow)",
        "config": {
            **DEFAULT_CONFIG,
            "region_grow": {**DEFAULT_CONFIG["region_grow"], "enabled": False},
        },
    },
]


def compare_configs(
    rgb_path: str,
    nc_path: str,
    configs: list[dict] = EXPERIMENT_CONFIGS,
) -> list[dict]:
    # =====================================================
    # 여러 설정 조합을 순서대로 실행하고 IoU 비교
    # =====================================================
    ground_truth = load_ground_truth(nc_path)
    results = []

    print(f"\n{'설정 이름':<35} IoU")
    print("-" * 45)

    for cfg in configs:
        mask, _ = run_pipeline(rgb_path, cfg["config"])
        iou = calculate_iou(ground_truth, mask)

        results.append({"name": cfg["name"], "iou": iou})
        print(f"{cfg['name']:<35} {iou:.4f}")

    return results


if __name__ == "__main__":
    SRC_DIR   = os.path.join(os.path.dirname(__file__), "src")
    rgb_path  = os.path.join(SRC_DIR, "gk2a_ami_le1b_rgb-s-true_ko020lc_202605210410.png")
    nc_path   = os.path.join(SRC_DIR, "gk2a_ami_le2_cld_ko020lc_202605210410.nc")

    compare_configs(rgb_path, nc_path)
