from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.scanner import (  # noqa: E402
    BINARIZATION_METHOD_KEYS,
    PROCESSING_PIPELINE_VERSION,
    ScanParams,
    process_rectified_document,
    resize_for_output,
)


METHOD_KEYS = BINARIZATION_METHOD_KEYS
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
    "binary_readable",
    "binary_wolf_fused",
)
DISPLAY_LABELS = {
    "binary_wolf_fused": "Ours (binary_wolf_fuse)",
}


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
        processed_image, rectified_scale = resize_for_output(image)
        artifacts, pipeline_metrics = process_rectified_document(processed_image, params)

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
                "rectified_scale": round(rectified_scale, 6),
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
        "pipeline_version": PROCESSING_PIPELINE_VERSION,
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
    }
    for name, img in output_images.items():
        cv2.imwrite(str(case_dir / f"{name}.png"), img)
    return output_images


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
