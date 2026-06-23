from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from skimage.morphology import skeletonize

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.scanner import ScanParams, enhance_and_binarize  # noqa: E402


METHOD_KEYS = (
    "binary_fixed",
    "binary_otsu",
    "binary_sauvola",
    "binary_niblack",
    "binary_wolf",
    "binary_wolf_fused",
    "binary_nick",
    "binary_bradley",
    "binary_gatos_like",
    "binary_majority",
    "binary_readable",
    "binary_readable_refined",
)
DISPLAY_KEYS = (
    "input",
    "gt",
    "binary_fixed",
    "binary_otsu",
    "binary_sauvola",
    "binary_niblack",
    "binary_wolf",
    "binary_nick",
    "binary_bradley",
    "binary_gatos_like",
    "binary_majority",
    "binary_readable",
    "binary_readable_refined",
    "binary_wolf_fused",
)
DISPLAY_LABELS = {
    "binary_wolf_fused": "Ours (binary_wolf_fuse)",
}
ENSEMBLE_SOURCE_KEYS = (
    "binary_fixed",
    "binary_otsu",
    "binary_sauvola",
    "binary_niblack",
    "binary_wolf",
    "binary_nick",
    "binary_bradley",
    "binary_gatos_like",
)


@dataclass(frozen=True)
class Case:
    dataset: str
    track: str
    case_id: str
    input_path: Path
    gt_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the document binarization artifacts on DIBCO 2019 and H-DIBCO 2018."
    )
    parser.add_argument("--dibco2019", type=Path, default=Path("data/raw/dibco2019"))
    parser.add_argument("--hdibco2018", type=Path, default=Path("data/raw/hdibco2018"))
    parser.add_argument("--out", type=Path, default=Path("runtime/experiments/dibco_eval_v1"))
    parser.add_argument("--fixed-threshold", type=int, default=180)
    parser.add_argument("--sauvola-window", type=int, default=35)
    parser.add_argument("--sauvola-k", type=float, default=0.2)
    parser.add_argument("--cleanup-kernel", type=int, default=3)
    parser.add_argument("--classic-source", choices=("raw", "enhanced"), default="raw")
    parser.add_argument("--local-window", type=int, default=35)
    parser.add_argument("--niblack-k", type=float, default=-0.2)
    parser.add_argument("--wolf-k", type=float, default=0.5)
    parser.add_argument("--nick-k", type=float, default=-0.2)
    parser.add_argument("--bradley-t", type=float, default=0.15)
    parser.add_argument("--wolf-fused-weak-scale", type=float, default=1.05)
    parser.add_argument("--wolf-fused-strong-scale", type=float, default=1.25)
    parser.add_argument("--wolf-fused-weak-percentile", type=float, default=75.0)
    parser.add_argument("--wolf-fused-strong-percentile", type=float, default=90.0)
    parser.add_argument("--wolf-fused-min-area", type=int, default=2)
    parser.add_argument("--gatos-window", type=int, default=35)
    parser.add_argument("--gatos-sauvola-k", type=float, default=0.2)
    parser.add_argument("--gatos-background-window", type=int, default=75)
    parser.add_argument("--majority-min-votes", type=int, default=0)
    parser.add_argument("--readable-refine-threshold-scale", type=float, default=1.14)
    parser.add_argument("--readable-refine-percentile", type=float, default=45.0)
    parser.add_argument("--readable-refine-min-area-ratio", type=float, default=0.42)
    parser.add_argument("--readable-refine-hole-min-area-ratio", type=float, default=0.01)
    parser.add_argument("--readable-refine-hole-max-area-ratio", type=float, default=0.55)
    parser.add_argument("--limit", type=int, default=0, help="Optional total case limit for smoke tests.")
    parser.add_argument("--skip-contact-sheets", action="store_true")
    parser.add_argument(
        "--contact-cell-width",
        type=int,
        default=0,
        help="Contact sheet panel width. Use 0 to keep each artifact at original resolution.",
    )
    parser.add_argument(
        "--contact-columns",
        type=int,
        default=0,
        help="Contact sheet columns. Use 0 to automatically arrange artifacts in two rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = ScanParams(
        fixed_threshold=args.fixed_threshold,
        sauvola_window=args.sauvola_window,
        sauvola_k=args.sauvola_k,
        cleanup_kernel=args.cleanup_kernel,
    ).normalized()

    cases = discover_cases(args.dibco2019, args.hdibco2018)
    if args.limit > 0:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("No DIBCO/H-DIBCO cases found. Check --dibco2019 and --hdibco2018 paths.")

    artifacts_root = args.out / "artifacts"
    sheets_root = args.out / "contact_sheets"
    tables_root = args.out / "tables"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    sheets_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)
    write_run_config(args.out / "run_config.json", args, params, len(cases))

    per_method_rows: list[dict[str, str | float | int]] = []
    case_summary_rows: list[dict[str, str | float | int]] = []

    for index, case in enumerate(cases, start=1):
        print(f"[{index:02d}/{len(cases):02d}] {case.dataset} {case.track} {case.case_id}", flush=True)
        image = read_color(case.input_path)
        gt = read_gray(case.gt_path)
        artifacts, pipeline_metrics = enhance_and_binarize(image, params)
        artifacts.update(compute_classic_binarization_artifacts(image, artifacts["text_enhanced"], args))
        artifacts["binary_wolf_fused"] = compute_wolf_fused_artifact(
            image,
            artifacts["background"],
            artifacts["binary_wolf"],
            artifacts["binary_nick"],
            artifacts["binary_readable"],
            pipeline_metrics,
            args,
        )
        artifacts["binary_gatos_like"] = compute_gatos_like_artifact(image, args, params.cleanup_kernel)
        artifacts["binary_majority"] = majority_ensemble(artifacts, args.majority_min_votes)
        artifacts["binary_readable_refined"] = refine_readable_artifact(
            image,
            artifacts["background"],
            artifacts["binary_readable"],
            pipeline_metrics,
            args,
        )

        case_key = f"{case.dataset}_{case.track}_{case.case_id}"
        case_dir = artifacts_root / case_key
        case_dir.mkdir(parents=True, exist_ok=True)
        saved_images = save_case_artifacts(case_dir, image, gt, artifacts)

        best_method = ""
        best_f1 = -1.0
        for method in METHOD_KEYS:
            pred = artifacts[method]
            metrics = compute_binary_metrics(pred, gt)
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_method = method
            per_method_rows.append(
                {
                    "dataset": case.dataset,
                    "track": case.track,
                    "case_id": case.case_id,
                    "method": method,
                    **metrics,
                }
            )

        case_summary_rows.append(
            {
                "dataset": case.dataset,
                "track": case.track,
                "case_id": case.case_id,
                "best_method": best_method,
                "best_f1": round(best_f1, 6),
                "foreground_ratio_gt": round(float((to_foreground_mask(gt)).mean()), 6),
                "text_background_contrast": pipeline_metrics.get("text_background_contrast", ""),
                "gray_std_before": pipeline_metrics.get("gray_std_before", ""),
                "gray_std_after_correction": pipeline_metrics.get("gray_std_after_correction", ""),
                "readable_text_ratio": pipeline_metrics.get("readable_text_ratio", ""),
                "readable_text_components": pipeline_metrics.get("readable_text_components", ""),
            }
        )

        if not args.skip_contact_sheets:
            sheet = build_contact_sheet(saved_images, args.contact_cell_width, args.contact_columns)
            cv2.imwrite(str(sheets_root / f"{case_key}_sheet.png"), sheet)

    write_csv(tables_root / "per_image_metrics.csv", per_method_rows)
    write_csv(tables_root / "case_summary.csv", case_summary_rows)
    write_csv(tables_root / "summary_by_method.csv", summarize_by_method(per_method_rows))
    write_csv(tables_root / "summary_by_dataset.csv", summarize_by_dataset(per_method_rows))

    print("", flush=True)
    print(f"Cases: {len(cases)}", flush=True)
    print(f"Per-image metrics: {tables_root / 'per_image_metrics.csv'}", flush=True)
    print(f"Method summary: {tables_root / 'summary_by_method.csv'}", flush=True)
    print(f"Dataset summary: {tables_root / 'summary_by_dataset.csv'}", flush=True)
    if not args.skip_contact_sheets:
        print(f"Contact sheets: {sheets_root}", flush=True)


def write_run_config(path: Path, args: argparse.Namespace, params: ScanParams, case_count: int) -> None:
    config = {
        "case_count": case_count,
        "datasets": {
            "dibco2019": str(args.dibco2019),
            "hdibco2018": str(args.hdibco2018),
        },
        "params": {
            "fixed_threshold": params.fixed_threshold,
            "sauvola_window": params.sauvola_window,
            "sauvola_k": params.sauvola_k,
            "cleanup_kernel": params.cleanup_kernel,
            "contact_cell_width": args.contact_cell_width,
            "contact_columns": normalized_contact_columns(args.contact_columns),
            "classic_source": args.classic_source,
            "local_window": normalized_odd(args.local_window, 3),
            "niblack_k": args.niblack_k,
            "wolf_k": args.wolf_k,
            "wolf_fused_weak_scale": args.wolf_fused_weak_scale,
            "wolf_fused_strong_scale": args.wolf_fused_strong_scale,
            "wolf_fused_weak_percentile": args.wolf_fused_weak_percentile,
            "wolf_fused_strong_percentile": args.wolf_fused_strong_percentile,
            "wolf_fused_min_area": args.wolf_fused_min_area,
            "nick_k": args.nick_k,
            "bradley_t": args.bradley_t,
            "gatos_window": normalized_odd(args.gatos_window, 3),
            "gatos_sauvola_k": args.gatos_sauvola_k,
            "gatos_background_window": normalized_odd(args.gatos_background_window, 3),
            "majority_min_votes": normalized_majority_min_votes(args.majority_min_votes, len(ENSEMBLE_SOURCE_KEYS)),
            "majority_sources": list(ENSEMBLE_SOURCE_KEYS),
            "readable_refine_threshold_scale": args.readable_refine_threshold_scale,
            "readable_refine_percentile": args.readable_refine_percentile,
            "readable_refine_min_area_ratio": args.readable_refine_min_area_ratio,
            "readable_refine_hole_min_area_ratio": args.readable_refine_hole_min_area_ratio,
            "readable_refine_hole_max_area_ratio": args.readable_refine_hole_max_area_ratio,
        },
        "methods": list(METHOD_KEYS),
        "metrics": [
            "precision",
            "recall",
            "f1",
            "pseudo_f1",
            "iou",
            "accuracy",
            "psnr",
            "nrm",
            "drd",
            "foreground_ratio_error",
        ],
    }
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def discover_cases(dibco2019_root: Path, hdibco2018_root: Path) -> list[Case]:
    cases: list[Case] = []
    dibco_dataset = dibco2019_root / "Dataset"
    dibco_gt = dibco2019_root / "GT"
    for case_num in range(1, 21):
        input_path = dibco_dataset / f"{case_num}.bmp"
        gt_path = dibco_gt / f"{case_num}.bmp"
        if input_path.exists() and gt_path.exists():
            track = "trackA" if case_num <= 10 else "trackB"
            cases.append(Case("dibco2019", track, f"{case_num:02d}", input_path, gt_path))

    hdibco_dataset = hdibco2018_root / "dataset"
    hdibco_gt = hdibco2018_root / "gt"
    for case_num in range(1, 11):
        input_path = hdibco_dataset / f"{case_num}.bmp"
        gt_path = hdibco_gt / f"{case_num}_gt.bmp"
        if input_path.exists() and gt_path.exists():
            cases.append(Case("hdibco2018", "handwritten", f"{case_num:02d}", input_path, gt_path))
    return cases


def read_color(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return normalize_binary_image(image)


def normalize_binary_image(image: np.ndarray) -> np.ndarray:
    return np.where(image < 128, 0, 255).astype(np.uint8)


def save_case_artifacts(case_dir: Path, image: np.ndarray, gt: np.ndarray, artifacts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    output_images: dict[str, np.ndarray] = {
        "input": image,
        "gt": gt,
        "text_enhanced": artifacts["text_enhanced"],
        "binary_readable": artifacts["binary_readable"],
        "binary_fixed": artifacts["binary_fixed"],
        "binary_otsu": artifacts["binary_otsu"],
        "binary_sauvola": artifacts["binary_sauvola"],
        "binary_niblack": artifacts["binary_niblack"],
        "binary_wolf": artifacts["binary_wolf"],
        "binary_wolf_fused": artifacts["binary_wolf_fused"],
        "binary_nick": artifacts["binary_nick"],
        "binary_bradley": artifacts["binary_bradley"],
        "binary_gatos_like": artifacts["binary_gatos_like"],
        "binary_majority": artifacts["binary_majority"],
        "binary_readable_refined": artifacts["binary_readable_refined"],
    }
    for name, img in output_images.items():
        cv2.imwrite(str(case_dir / f"{name}.png"), img)
    return output_images


def compute_classic_binarization_artifacts(
    image: np.ndarray, text_enhanced: np.ndarray, args: argparse.Namespace
) -> dict[str, np.ndarray]:
    source = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if args.classic_source == "raw" else text_enhanced
    window = normalized_odd(args.local_window, 3)
    mean, std, mean_square = local_statistics(source, window)

    return {
        "binary_niblack": threshold_dark_foreground(source, niblack_threshold(mean, std, args.niblack_k)),
        "binary_wolf": threshold_dark_foreground(source, wolf_threshold(source, mean, std, args.wolf_k)),
        "binary_nick": threshold_dark_foreground(source, nick_threshold(mean, mean_square, args.nick_k)),
        "binary_bradley": threshold_dark_foreground(source, bradley_threshold(mean, args.bradley_t)),
    }


def local_statistics(image: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_float = image.astype(np.float32)
    mean = cv2.boxFilter(image_float, ddepth=-1, ksize=(window, window), normalize=True, borderType=cv2.BORDER_REPLICATE)
    mean_square = cv2.boxFilter(
        image_float * image_float,
        ddepth=-1,
        ksize=(window, window),
        normalize=True,
        borderType=cv2.BORDER_REPLICATE,
    )
    variance = np.maximum(mean_square - mean * mean, 0.0)
    return mean, np.sqrt(variance), mean_square


def niblack_threshold(mean: np.ndarray, std: np.ndarray, k: float) -> np.ndarray:
    return mean + k * std


def wolf_threshold(image: np.ndarray, mean: np.ndarray, std: np.ndarray, k: float) -> np.ndarray:
    min_gray = float(np.min(image))
    max_std = max(float(np.max(std)), 1.0)
    return mean + k * ((std / max_std) - 1.0) * (mean - min_gray)


def nick_threshold(mean: np.ndarray, mean_square: np.ndarray, k: float) -> np.ndarray:
    return mean + k * np.sqrt(np.maximum(mean_square, 0.0))


def bradley_threshold(mean: np.ndarray, t: float) -> np.ndarray:
    return mean * (1.0 - t)


def threshold_dark_foreground(image: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    return np.where(image.astype(np.float32) > threshold, 255, 0).astype(np.uint8)


def compute_wolf_fused_artifact(
    image: np.ndarray,
    background: np.ndarray,
    binary_wolf: np.ndarray,
    binary_nick: np.ndarray,
    binary_readable: np.ndarray,
    pipeline_metrics: dict[str, float | int | str],
    args: argparse.Namespace,
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if background.shape != gray.shape:
        background = cv2.resize(background, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA)

    relative_dark = relative_dark_detail(gray, background)
    wolf_mask = to_foreground_mask(binary_wolf)
    nick_mask = to_foreground_mask(binary_nick)
    readable_mask = to_foreground_mask(binary_readable)
    candidate_source = np.logical_or(nick_mask, readable_mask)

    source_values = relative_dark[candidate_source]
    weak_percentile = float(np.clip(args.wolf_fused_weak_percentile, 0.0, 100.0))
    strong_percentile = float(np.clip(args.wolf_fused_strong_percentile, weak_percentile, 100.0))
    weak_floor = float(np.percentile(source_values, weak_percentile)) if source_values.size else 6.0
    strong_floor = float(np.percentile(source_values, strong_percentile)) if source_values.size else weak_floor
    base_threshold = safe_float(pipeline_metrics.get("readable_text_threshold"), weak_floor)
    weak_threshold = max(base_threshold * max(float(args.wolf_fused_weak_scale), 0.1), weak_floor)
    strong_threshold = max(base_threshold * max(float(args.wolf_fused_strong_scale), 0.1), strong_floor, weak_threshold)

    strong_seed = np.logical_or(wolf_mask, np.logical_and(candidate_source, relative_dark >= strong_threshold))
    weak_candidate = np.logical_and(candidate_source, relative_dark >= weak_threshold)
    candidate = np.logical_or(wolf_mask, weak_candidate)
    fused = keep_components_touching_seed(candidate, strong_seed, max(int(args.wolf_fused_min_area), 1))
    return np.where(fused, 0, 255).astype(np.uint8)


def relative_dark_detail(gray: np.ndarray, background: np.ndarray) -> np.ndarray:
    dark_detail = cv2.subtract(background, gray)
    relative_dark = dark_detail.astype(np.float32) * 255.0 / np.maximum(background.astype(np.float32), 1.0)
    relative_dark = cv2.GaussianBlur(np.clip(relative_dark, 0, 255).astype(np.uint8), (3, 3), 0)
    return relative_dark.astype(np.float32)


def keep_components_touching_seed(candidate: np.ndarray, seed: np.ndarray, min_area: int) -> np.ndarray:
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
    kept = np.zeros_like(candidate, dtype=bool)
    for label in range(1, labels_count):
        component = labels == label
        if int(stats[label, cv2.CC_STAT_AREA]) < min_area:
            continue
        if np.any(np.logical_and(component, seed)):
            kept[component] = True
    return kept


def compute_gatos_like_artifact(image: np.ndarray, args: argparse.Namespace, cleanup_kernel: int) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    window = normalized_odd(args.gatos_window, 3)
    mean, std, _ = local_statistics(gray, window)
    initial_threshold = sauvola_threshold(mean, std, args.gatos_sauvola_k, r=128.0)
    initial_foreground = gray.astype(np.float32) <= initial_threshold
    initial_foreground = clean_foreground_mask(initial_foreground, 3)

    background_window = normalized_odd(args.gatos_background_window, window)
    background = estimate_background_from_mask(gray, ~initial_foreground, background_window)
    corrected = cv2.divide(gray, background, scale=255)
    corrected = percentile_stretch_uint8(corrected, 1.0, 99.0)
    _, binary = cv2.threshold(corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cleanup_dark_foreground_binary(binary, cleanup_kernel)


def sauvola_threshold(mean: np.ndarray, std: np.ndarray, k: float, r: float) -> np.ndarray:
    return mean * (1.0 + k * ((std / max(r, 1.0)) - 1.0))


def clean_foreground_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cleaned.astype(bool)


def estimate_background_from_mask(gray: np.ndarray, background_mask: np.ndarray, window: int) -> np.ndarray:
    gray_float = gray.astype(np.float32)
    if int(background_mask.sum()) < max(32, int(gray.size * 0.05)):
        background_mask = np.ones_like(background_mask, dtype=bool)

    weights = background_mask.astype(np.float32)
    values = gray_float * weights
    blurred_values = cv2.GaussianBlur(values, (window, window), 0, borderType=cv2.BORDER_REPLICATE)
    blurred_weights = cv2.GaussianBlur(weights, (window, window), 0, borderType=cv2.BORDER_REPLICATE)
    smooth_gray = cv2.GaussianBlur(gray_float, (window, window), 0, borderType=cv2.BORDER_REPLICATE)

    background = np.where(blurred_weights > 1e-3, blurred_values / np.maximum(blurred_weights, 1e-3), smooth_gray)
    background = cv2.GaussianBlur(background, (window, window), 0, borderType=cv2.BORDER_REPLICATE)
    return np.clip(background, 1, 255).astype(np.uint8)


def percentile_stretch_uint8(image: np.ndarray, low_percentile: float, high_percentile: float) -> np.ndarray:
    low, high = np.percentile(image, (low_percentile, high_percentile))
    if float(high - low) < 4.0:
        return np.clip(image, 0, 255).astype(np.uint8)
    stretched = (image.astype(np.float32) - float(low)) * (255.0 / float(high - low))
    return np.clip(stretched, 0, 255).astype(np.uint8)


def cleanup_dark_foreground_binary(binary: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    foreground = (binary < 128).astype(np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel, iterations=1)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=1)
    return np.where(foreground > 0, 0, 255).astype(np.uint8)


def majority_ensemble(artifacts: dict[str, np.ndarray], requested_min_votes: int) -> np.ndarray:
    source_masks = [to_foreground_mask(artifacts[key]) for key in ENSEMBLE_SOURCE_KEYS if key in artifacts]
    if not source_masks:
        raise ValueError("No source masks available for majority ensemble.")
    min_votes = normalized_majority_min_votes(requested_min_votes, len(source_masks))
    votes = np.sum(np.stack(source_masks, axis=0), axis=0)
    return np.where(votes >= min_votes, 0, 255).astype(np.uint8)


def normalized_majority_min_votes(requested_min_votes: int, source_count: int) -> int:
    if requested_min_votes > 0:
        return min(max(int(requested_min_votes), 1), source_count)
    return source_count // 2 + 1


def refine_readable_artifact(
    image: np.ndarray,
    background: np.ndarray,
    binary_readable: np.ndarray,
    pipeline_metrics: dict[str, float | int | str],
    args: argparse.Namespace,
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if background.shape != gray.shape:
        background = cv2.resize(background, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA)

    mask = to_foreground_mask(binary_readable)
    if not np.any(mask):
        return binary_readable.copy()

    dark_detail = cv2.subtract(background, gray)
    relative_dark = dark_detail.astype(np.float32) * 255.0 / np.maximum(background.astype(np.float32), 1.0)
    relative_dark = np.clip(relative_dark, 0, 255)

    foreground_values = relative_dark[mask]
    trim_percentile = float(np.clip(args.readable_refine_percentile, 0.0, 100.0))
    fallback_threshold = float(np.percentile(foreground_values, trim_percentile)) if foreground_values.size else 6.0
    base_threshold = safe_float(pipeline_metrics.get("readable_text_threshold"), fallback_threshold)
    threshold_scale = max(float(args.readable_refine_threshold_scale), 0.1)
    min_area_ratio = float(np.clip(args.readable_refine_min_area_ratio, 0.05, 0.95))
    hole_min_area_ratio = float(np.clip(args.readable_refine_hole_min_area_ratio, 0.0, 0.5))
    hole_max_area_ratio = float(np.clip(args.readable_refine_hole_max_area_ratio, hole_min_area_ratio, 0.95))
    boundary_threshold = max(base_threshold * threshold_scale, fallback_threshold)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    refined = np.zeros_like(mask, dtype=np.uint8)

    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        component = labels == label
        if area <= 6:
            refined[component] = 1
            continue

        eroded = cv2.erode(component.astype(np.uint8), kernel, iterations=1).astype(bool)
        boundary = np.logical_and(component, ~eroded)
        weak_pixels = np.logical_and(component, relative_dark < boundary_threshold)
        weak_boundary = np.logical_and(boundary, weak_pixels)
        weak_holes = internal_weak_regions(weak_pixels, boundary, area, hole_min_area_ratio, hole_max_area_ratio)
        weak_to_remove = np.logical_or(weak_boundary, weak_holes)
        candidate = np.logical_and(component, ~weak_to_remove)

        min_kept_area = max(2, int(round(area * min_area_ratio)))
        if int(candidate.sum()) < min_kept_area:
            refined[component] = 1
        else:
            refined[candidate] = 1

    return np.where(refined > 0, 0, 255).astype(np.uint8)


def internal_weak_regions(
    weak_pixels: np.ndarray,
    component_boundary: np.ndarray,
    component_area: int,
    min_area_ratio: float,
    max_area_ratio: float,
) -> np.ndarray:
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(weak_pixels.astype(np.uint8), 8)
    holes = np.zeros_like(weak_pixels, dtype=bool)
    min_area = max(2, int(round(component_area * min_area_ratio)))
    max_area = max(min_area, int(round(component_area * max_area_ratio)))

    for label in range(1, labels_count):
        weak_region = labels == label
        if np.any(np.logical_and(weak_region, component_boundary)):
            continue
        area = int(stats[label, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            holes[weak_region] = True
    return holes


def safe_float(value: object, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def normalized_odd(value: int, minimum: int) -> int:
    value = max(int(value), minimum)
    return value if value % 2 == 1 else value + 1


def compute_binary_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    pred = normalize_binary_image(to_gray(prediction))
    gt = normalize_binary_image(to_gray(ground_truth))
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

    pred_fg = to_foreground_mask(pred)
    gt_fg = to_foreground_mask(gt)

    tp = int(np.logical_and(pred_fg, gt_fg).sum())
    fp = int(np.logical_and(pred_fg, ~gt_fg).sum())
    fn = int(np.logical_and(~pred_fg, gt_fg).sum())
    tn = int(np.logical_and(~pred_fg, ~gt_fg).sum())
    total = max(tp + fp + fn + tn, 1)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    pseudo_recall = pseudo_foreground_recall(pred_fg, gt_fg)
    pseudo_f1 = safe_div(2 * precision * pseudo_recall, precision + pseudo_recall)
    iou = safe_div(tp, tp + fp + fn)
    accuracy = (tp + tn) / total
    nrm = 0.5 * (safe_div(fn, fn + tp) + safe_div(fp, fp + tn))
    drd = distortion_reciprocal_distance(pred_fg, gt_fg)
    mse = float(np.mean((pred.astype(np.float32) - gt.astype(np.float32)) ** 2))
    psnr = 99.0 if mse == 0 else float(20.0 * math.log10(255.0 / math.sqrt(mse)))
    pred_ratio = float(pred_fg.mean())
    gt_ratio = float(gt_fg.mean())

    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "pseudo_f1": round(pseudo_f1, 6),
        "iou": round(iou, 6),
        "accuracy": round(accuracy, 6),
        "psnr": round(psnr, 6),
        "nrm": round(nrm, 6),
        "drd": round(drd, 6),
        "foreground_ratio_pred": round(pred_ratio, 6),
        "foreground_ratio_gt": round(gt_ratio, 6),
        "foreground_ratio_error": round(abs(pred_ratio - gt_ratio), 6),
    }


def pseudo_foreground_recall(pred_fg: np.ndarray, gt_fg: np.ndarray) -> float:
    gt_skeleton = skeletonize(gt_fg).astype(bool)
    skeleton_pixels = int(gt_skeleton.sum())
    if skeleton_pixels == 0:
        return 1.0 if int(pred_fg.sum()) == 0 else 0.0
    return safe_div(int(np.logical_and(pred_fg, gt_skeleton).sum()), skeleton_pixels)


def distortion_reciprocal_distance(pred_fg: np.ndarray, gt_fg: np.ndarray) -> float:
    misclassified = pred_fg != gt_fg
    if not np.any(misclassified):
        return 0.0

    weights = drd_weights()
    gt_float = gt_fg.astype(np.float32)
    weighted_neighbors = cv2.filter2D(gt_float, -1, weights, borderType=cv2.BORDER_REPLICATE)
    local_cost = gt_float * (1.0 - weighted_neighbors) + (1.0 - gt_float) * weighted_neighbors
    non_uniform_blocks = count_non_uniform_blocks(gt_fg)
    return float(local_cost[misclassified].sum() / max(non_uniform_blocks, 1))


def drd_weights() -> np.ndarray:
    weights = np.zeros((5, 5), dtype=np.float32)
    for y in range(-2, 3):
        for x in range(-2, 3):
            if x == 0 and y == 0:
                continue
            weights[y + 2, x + 2] = 1.0 / math.sqrt(float(x * x + y * y))
    total = float(weights.sum())
    if total > 0:
        weights /= total
    return weights


def count_non_uniform_blocks(mask: np.ndarray, block_size: int = 8) -> int:
    count = 0
    height, width = mask.shape[:2]
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = mask[y : y + block_size, x : x + block_size]
            if block.size and np.any(block) and not np.all(block):
                count += 1
    return count


def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def to_foreground_mask(image: np.ndarray) -> np.ndarray:
    return to_gray(image) < 128


def safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def summarize_by_method(rows: list[dict[str, str | float | int]]) -> list[dict[str, str | float | int]]:
    best_counts = best_method_counts(rows, ("dataset", "track", "case_id"))
    best_drd_counts = best_drd_counts_by_group(rows, ("dataset", "track", "case_id"))
    summaries: list[dict[str, str | float | int]] = []
    for method in METHOD_KEYS:
        method_rows = [row for row in rows if row["method"] == method]
        summaries.append(
            {
                "method": method,
                "mean_precision": mean_metric(method_rows, "precision"),
                "mean_recall": mean_metric(method_rows, "recall"),
                "mean_f1": mean_metric(method_rows, "f1"),
                "mean_pseudo_f1": mean_metric(method_rows, "pseudo_f1"),
                "mean_iou": mean_metric(method_rows, "iou"),
                "mean_accuracy": mean_metric(method_rows, "accuracy"),
                "mean_psnr": mean_metric(method_rows, "psnr"),
                "mean_nrm": mean_metric(method_rows, "nrm"),
                "mean_drd": mean_metric(method_rows, "drd"),
                "mean_foreground_ratio_error": mean_metric(method_rows, "foreground_ratio_error"),
                "best_f1_count": best_counts.get(method, 0),
                "best_drd_count": best_drd_counts.get(method, 0),
            }
        )
    return summaries


def summarize_by_dataset(rows: list[dict[str, str | float | int]]) -> list[dict[str, str | float | int]]:
    summaries: list[dict[str, str | float | int]] = []
    groups = sorted({(str(row["dataset"]), str(row["track"])) for row in rows})
    for dataset, track in groups:
        group_rows = [row for row in rows if row["dataset"] == dataset and row["track"] == track]
        best_counts = best_method_counts(group_rows, ("case_id",))
        best_drd_counts = best_drd_counts_by_group(group_rows, ("case_id",))
        for method in METHOD_KEYS:
            method_rows = [row for row in group_rows if row["method"] == method]
            summaries.append(
                {
                    "dataset": dataset,
                    "track": track,
                    "method": method,
                    "mean_f1": mean_metric(method_rows, "f1"),
                    "mean_pseudo_f1": mean_metric(method_rows, "pseudo_f1"),
                    "mean_iou": mean_metric(method_rows, "iou"),
                    "mean_psnr": mean_metric(method_rows, "psnr"),
                    "mean_nrm": mean_metric(method_rows, "nrm"),
                    "mean_drd": mean_metric(method_rows, "drd"),
                    "mean_foreground_ratio_error": mean_metric(method_rows, "foreground_ratio_error"),
                    "best_f1_count": best_counts.get(method, 0),
                    "best_drd_count": best_drd_counts.get(method, 0),
                }
            )
    return summaries


def best_method_counts(rows: list[dict[str, str | float | int]], group_keys: tuple[str, ...]) -> dict[str, int]:
    grouped: dict[tuple[str, ...], list[dict[str, str | float | int]]] = {}
    for row in rows:
        group = tuple(str(row[key]) for key in group_keys)
        grouped.setdefault(group, []).append(row)

    counts: dict[str, int] = {}
    for group_rows in grouped.values():
        best_row = max(group_rows, key=lambda row: float(row["f1"]))
        method = str(best_row["method"])
        counts[method] = counts.get(method, 0) + 1
    return counts


def best_drd_counts_by_group(rows: list[dict[str, str | float | int]], group_keys: tuple[str, ...]) -> dict[str, int]:
    grouped: dict[tuple[str, ...], list[dict[str, str | float | int]]] = {}
    for row in rows:
        group = tuple(str(row[key]) for key in group_keys)
        grouped.setdefault(group, []).append(row)

    counts: dict[str, int] = {}
    for group_rows in grouped.values():
        best_row = min(group_rows, key=lambda row: float(row["drd"]))
        method = str(best_row["method"])
        counts[method] = counts.get(method, 0) + 1
    return counts


def mean_metric(rows: list[dict[str, str | float | int]], metric: str) -> float:
    if not rows:
        return 0.0
    return round(float(np.mean([float(row[metric]) for row in rows])), 6)


def write_csv(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_contact_sheet(images: dict[str, np.ndarray], cell_width: int = 0, columns: int = 0) -> np.ndarray:
    cells = [make_labeled_cell(label, images[label], cell_width) for label in DISPLAY_KEYS]
    columns = normalized_contact_columns(columns)
    rows = [hstack_with_padding(cells[index : index + columns]) for index in range(0, len(cells), columns)]
    return vstack_with_padding(rows)


def make_labeled_cell(label: str, image: np.ndarray, width: int) -> np.ndarray:
    preview = to_preview_bgr(image, width)
    display_label = DISPLAY_LABELS.get(label, label)
    is_original_size = width <= 0
    label_height = 58 if is_original_size else 34
    font_scale = 1.0 if is_original_size else 0.58
    baseline = 39 if is_original_size else 23
    thickness = 2
    label_bar = np.full((label_height, preview.shape[1], 3), 255, dtype=np.uint8)
    font_scale = fit_text_scale(display_label, preview.shape[1] - 20, font_scale, thickness)
    cv2.putText(
        label_bar,
        display_label,
        (12, baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (20, 34, 40),
        thickness,
        cv2.LINE_AA,
    )
    return np.vstack([label_bar, preview])


def fit_text_scale(text: str, max_width: int, preferred_scale: float, thickness: int) -> float:
    scale = preferred_scale
    while scale > 0.34:
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
        if text_width <= max_width:
            return scale
        scale -= 0.04
    return scale


def to_preview_bgr(image: np.ndarray, width: int) -> np.ndarray:
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image
    if width <= 0:
        return bgr.copy()
    scale = width / bgr.shape[1]
    height = max(1, int(round(bgr.shape[0] * scale)))
    return cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)


def hstack_with_padding(cells: list[np.ndarray]) -> np.ndarray:
    max_height = max(cell.shape[0] for cell in cells)
    padded_cells: list[np.ndarray] = []
    for cell in cells:
        if cell.shape[0] == max_height:
            padded_cells.append(cell)
            continue
        pad_height = max_height - cell.shape[0]
        padding = np.full((pad_height, cell.shape[1], 3), 255, dtype=np.uint8)
        padded_cells.append(np.vstack([cell, padding]))
    return np.hstack(padded_cells)


def vstack_with_padding(rows: list[np.ndarray], gap: int = 18) -> np.ndarray:
    max_width = max(row.shape[1] for row in rows)
    padded_rows: list[np.ndarray] = []
    for index, row in enumerate(rows):
        if row.shape[1] < max_width:
            pad_width = max_width - row.shape[1]
            padding = np.full((row.shape[0], pad_width, 3), 255, dtype=np.uint8)
            row = np.hstack([row, padding])
        padded_rows.append(row)
        if index < len(rows) - 1 and gap > 0:
            padded_rows.append(np.full((gap, max_width, 3), 245, dtype=np.uint8))
    return np.vstack(padded_rows)


def normalized_contact_columns(requested_columns: int) -> int:
    if requested_columns > 0:
        return max(1, min(int(requested_columns), len(DISPLAY_KEYS)))
    return math.ceil(len(DISPLAY_KEYS) / 2)


if __name__ == "__main__":
    main()
