from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.scanner import BINARIZATION_METHOD_KEYS, PROCESSING_PIPELINE_VERSION, ScanParams, order_points, process_image  # noqa: E402


CONDITION_LABELS = {
    "T": "table",
    "K": "keyboard",
    "H": "hand",
    "P": "partial",
    "C": "clutter",
    "D": "distorted",
    "L": "low-light",
}

ARTIFACT_KEYS = (
    "original",
    "gt_pred_overlay",
    "rectified",
    "text_enhanced",
    "binary_readable",
    "binary_fixed",
    "binary_otsu",
    "binary_sauvola",
    "binary_wolf",
    "binary_wolf_fused",
)


@dataclass(frozen=True)
class MidvCase:
    dataset: str
    document_id: str
    source_path: Path
    source_kind: str
    image_ref: str
    gt_ref: str
    condition_code: str
    condition: str
    condition_label: str
    device: str
    frame_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the current document detection and enhancement pipeline on MIDV-style datasets. "
            "MIDV provides frame-level document quadrangles, so the primary metrics are geometric."
        )
    )
    parser.add_argument("--midv500", type=Path, default=Path("data/raw/midv500"))
    parser.add_argument("--midv2019", type=Path, default=Path("data/raw/midv2019"))
    parser.add_argument("--out", type=Path, default=Path("runtime/experiments/midv_eval_v1"))
    parser.add_argument(
        "--datasets",
        default="midv500,midv2019",
        help="Comma-separated dataset labels to include: midv500, midv2019.",
    )
    parser.add_argument(
        "--documents",
        default="",
        help="Optional comma-separated document ids, e.g. 01_alb_id,09_chn_id.",
    )
    parser.add_argument(
        "--conditions",
        default="",
        help="Optional comma-separated condition letters or codes, e.g. T,K,H or TA,PS.",
    )
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=3,
        help="Evenly sample this many frames from each condition/device video. Use 0 for all frames.",
    )
    parser.add_argument("--doc-limit", type=int, default=0, help="Optional document archive/directory limit.")
    parser.add_argument("--limit", type=int, default=0, help="Optional total frame limit.")
    parser.add_argument("--artifact-limit", type=int, default=24, help="Save image artifacts for the first N cases.")
    parser.add_argument("--skip-artifacts", action="store_true")
    parser.add_argument("--contact-cell-width", type=int, default=360)
    parser.add_argument("--fixed-threshold", type=int, default=180)
    parser.add_argument("--sauvola-window", type=int, default=35)
    parser.add_argument("--sauvola-k", type=float, default=0.2)
    parser.add_argument("--cleanup-kernel", type=int, default=3)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = ScanParams(
        fixed_threshold=args.fixed_threshold,
        sauvola_window=args.sauvola_window,
        sauvola_k=args.sauvola_k,
        cleanup_kernel=args.cleanup_kernel,
    ).normalized()

    cases = discover_cases(args)
    cases = sample_cases(cases, args.frames_per_video, args.limit)
    if not cases:
        raise SystemExit("No MIDV cases found. Check --midv500/--midv2019 and filters.")

    artifacts_root = args.out / "artifacts"
    tables_root = args.out / "tables"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)
    write_run_config(args.out / "run_config.json", args, params, cases)

    per_frame_rows: list[dict[str, str | float | int]] = []
    method_rows: list[dict[str, str | float | int]] = []

    for index, case in enumerate(cases, start=1):
        print(
            f"[{index:04d}/{len(cases):04d}] {case.dataset} {case.document_id} "
            f"{case.condition_code} {case.frame_id}",
            flush=True,
        )
        image = read_case_image(case)
        gt_quad = read_case_quad(case)

        result = process_image(image, params)
        pred_quad = np.asarray(result.corners, dtype=np.float32)
        geometry_metrics = compute_geometry_metrics(pred_quad, gt_quad, image.shape)

        row = {
            **case_metadata(case),
            **geometry_metrics,
            "candidate_score": result.candidate_score,
            "candidate_count": int(result.metrics.get("candidate_count", 0)),
            "candidate_source": str(result.metrics.get("candidate_source", "")),
            "warning_count": len(result.warnings),
            "output_width": int(result.metrics.get("output_width", 0)),
            "output_height": int(result.metrics.get("output_height", 0)),
            "text_background_contrast": result.metrics.get("text_background_contrast", ""),
            "readable_text_ratio": result.metrics.get("readable_text_ratio", ""),
            "readable_text_components": result.metrics.get("readable_text_components", ""),
            "folded_corner_refinements": result.metrics.get("folded_corner_refinements", ""),
            "candidate_paper_boundary_score": result.metrics.get("candidate_paper_boundary_score", ""),
            "pipeline_version": result.metrics.get("pipeline_version", ""),
        }
        per_frame_rows.append(row)

        for method in BINARIZATION_METHOD_KEYS:
            method_rows.append(
                {
                    **case_metadata(case),
                    "method": method,
                    **binary_diagnostics(result.artifacts[method]),
                }
            )

        if not args.skip_artifacts and index <= args.artifact_limit:
            save_case_artifacts(
                artifacts_root / case_key(case),
                image,
                pred_quad,
                gt_quad,
                result.artifacts["rectified"],
                result.artifacts,
                geometry_metrics,
                args.contact_cell_width,
            )

    write_csv(tables_root / "per_frame_metrics.csv", per_frame_rows)
    write_csv(tables_root / "summary_overall.csv", summarize_geometry(per_frame_rows, ()))
    write_csv(tables_root / "summary_by_dataset.csv", summarize_geometry(per_frame_rows, ("dataset",)))
    write_csv(tables_root / "summary_by_document.csv", summarize_geometry(per_frame_rows, ("dataset", "document_id")))
    write_csv(tables_root / "summary_by_condition.csv", summarize_geometry(per_frame_rows, ("dataset", "condition_code")))
    write_csv(tables_root / "method_diagnostics.csv", method_rows)
    write_csv(tables_root / "summary_by_method.csv", summarize_methods(method_rows, ("method",)))
    write_csv(
        tables_root / "summary_by_condition_method.csv",
        summarize_methods(method_rows, ("dataset", "condition_code", "method")),
    )

    print("", flush=True)
    print(f"Cases: {len(cases)}", flush=True)
    print(f"Per-frame metrics: {tables_root / 'per_frame_metrics.csv'}", flush=True)
    print(f"Geometry summary: {tables_root / 'summary_by_condition.csv'}", flush=True)
    print(f"Method diagnostics: {tables_root / 'summary_by_method.csv'}", flush=True)
    if not args.skip_artifacts:
        print(f"Artifacts: {artifacts_root}", flush=True)


def discover_cases(args: argparse.Namespace) -> list[MidvCase]:
    selected_datasets = parse_csv_filter(args.datasets)
    selected_documents = parse_csv_filter(args.documents)
    selected_conditions = parse_csv_filter(args.conditions)
    roots = {
        "midv500": args.midv500,
        "midv2019": args.midv2019,
    }

    cases: list[MidvCase] = []
    for dataset, root in roots.items():
        if selected_datasets and dataset not in selected_datasets:
            continue
        if not root.exists():
            continue
        dataset_sources = discover_dataset_sources(root, args.doc_limit)
        for source_kind, source_path in dataset_sources:
            document_id = document_id_from_source(source_path)
            if selected_documents and document_id not in selected_documents:
                continue
            if source_kind == "zip":
                source_cases = discover_zip_cases(dataset, source_path, selected_conditions)
            else:
                source_cases = discover_directory_cases(dataset, source_path, selected_conditions)
            cases.extend(source_cases)
    return sorted(cases, key=lambda item: (item.dataset, item.document_id, item.condition_code, item.frame_id))


def discover_dataset_sources(root: Path, doc_limit: int) -> list[tuple[str, Path]]:
    zip_sources = sorted(root.glob("*.zip"))
    directory_sources = sorted(path for path in root.iterdir() if path.is_dir() and (path / "images").is_dir())
    sources = [("zip", path) for path in zip_sources] + [("directory", path) for path in directory_sources]
    if doc_limit > 0:
        return sources[:doc_limit]
    return sources


def discover_zip_cases(dataset: str, archive_path: Path, selected_conditions: set[str]) -> list[MidvCase]:
    cases: list[MidvCase] = []
    document_id = document_id_from_source(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        for gt_ref in sorted(name for name in names if is_frame_gt_member(name)):
            parts = Path(gt_ref).parts
            if len(parts) < 4:
                continue
            condition_code = parts[-2]
            if not condition_matches(condition_code, selected_conditions):
                continue
            image_ref = str(Path(*parts[:-3], "images", condition_code, f"{Path(gt_ref).stem}.tif"))
            if image_ref not in names:
                continue
            cases.append(make_case(dataset, document_id, archive_path, "zip", image_ref, gt_ref, condition_code))
    return cases


def discover_directory_cases(dataset: str, source_path: Path, selected_conditions: set[str]) -> list[MidvCase]:
    cases: list[MidvCase] = []
    document_id = document_id_from_source(source_path)
    ground_truth_root = source_path / "ground_truth"
    images_root = source_path / "images"
    for gt_path in sorted(ground_truth_root.glob("*/*.json")):
        condition_code = gt_path.parent.name
        if not condition_matches(condition_code, selected_conditions):
            continue
        image_path = images_root / condition_code / f"{gt_path.stem}.tif"
        if not image_path.exists():
            continue
        cases.append(
            make_case(
                dataset,
                document_id,
                source_path,
                "directory",
                str(image_path.relative_to(source_path)),
                str(gt_path.relative_to(source_path)),
                condition_code,
            )
        )
    return cases


def is_frame_gt_member(name: str) -> bool:
    path = Path(name)
    parts = path.parts
    return (
        len(parts) >= 4
        and parts[-3] == "ground_truth"
        and path.suffix.lower() == ".json"
        and len(parts[-2]) >= 2
    )


def make_case(
    dataset: str,
    document_id: str,
    source_path: Path,
    source_kind: str,
    image_ref: str,
    gt_ref: str,
    condition_code: str,
) -> MidvCase:
    condition = condition_code[:1]
    device = condition_code[1:2] if len(condition_code) >= 2 else ""
    return MidvCase(
        dataset=dataset,
        document_id=document_id,
        source_path=source_path,
        source_kind=source_kind,
        image_ref=image_ref,
        gt_ref=gt_ref,
        condition_code=condition_code,
        condition=condition,
        condition_label=CONDITION_LABELS.get(condition, condition),
        device=device,
        frame_id=Path(image_ref).stem,
    )


def sample_cases(cases: list[MidvCase], frames_per_video: int, limit: int) -> list[MidvCase]:
    if frames_per_video > 0:
        grouped: dict[tuple[str, str, str, Path], list[MidvCase]] = {}
        for case in cases:
            key = (case.dataset, case.document_id, case.condition_code, case.source_path)
            grouped.setdefault(key, []).append(case)
        sampled: list[MidvCase] = []
        for group_cases in grouped.values():
            ordered = sorted(group_cases, key=lambda item: item.frame_id)
            if len(ordered) <= frames_per_video:
                sampled.extend(ordered)
                continue
            indices = np.linspace(0, len(ordered) - 1, frames_per_video)
            sampled.extend(ordered[int(round(index))] for index in indices)
        cases = sorted(sampled, key=lambda item: (item.dataset, item.document_id, item.condition_code, item.frame_id))
    if limit > 0:
        return cases[:limit]
    return cases


def read_case_image(case: MidvCase) -> np.ndarray:
    data = read_case_bytes(case, case.image_ref)
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {case.source_path}:{case.image_ref}")
    return image


def read_case_quad(case: MidvCase) -> np.ndarray:
    data = read_case_bytes(case, case.gt_ref)
    payload = json.loads(data.decode("utf-8"))
    if "quad" not in payload:
        raise ValueError(f"Frame ground truth has no quad: {case.source_path}:{case.gt_ref}")
    quad = np.asarray(payload["quad"], dtype=np.float32)
    if quad.shape != (4, 2):
        raise ValueError(f"Invalid quad shape in {case.source_path}:{case.gt_ref}")
    return order_points(quad)


def read_case_bytes(case: MidvCase, ref: str) -> bytes:
    if case.source_kind == "zip":
        with zipfile.ZipFile(case.source_path) as archive:
            return archive.read(ref)
    return (case.source_path / ref).read_bytes()


def compute_geometry_metrics(pred_quad: np.ndarray, gt_quad: np.ndarray, image_shape: tuple[int, ...]) -> dict[str, float | int]:
    pred = order_points(np.asarray(pred_quad, dtype=np.float32))
    gt = order_points(np.asarray(gt_quad, dtype=np.float32))
    height, width = image_shape[:2]
    image_diag = math.hypot(float(width), float(height))
    corner_errors = np.linalg.norm(pred - gt, axis=1)
    mean_error = float(np.mean(corner_errors))
    max_error = float(np.max(corner_errors))
    iou = polygon_iou(pred, gt, (height, width))
    gt_area = polygon_area(gt)
    gt_visible_area = float(np.count_nonzero(polygon_mask(gt, (height, width))))
    pred_visible_area = float(np.count_nonzero(polygon_mask(pred, (height, width))))
    image_area = float(max(height * width, 1))
    visible_ratio = min(1.0, safe_div(gt_visible_area, gt_area))
    norm_error = safe_div(mean_error, image_diag)

    return {
        "corner_mean_error_px": round(mean_error, 4),
        "corner_max_error_px": round(max_error, 4),
        "corner_mean_error_norm": round(norm_error, 6),
        "polygon_iou": round(iou, 6),
        "gt_visible_ratio": round(visible_ratio, 6),
        "gt_area_ratio": round(gt_visible_area / image_area, 6),
        "pred_area_ratio": round(pred_visible_area / image_area, 6),
        "success_iou_80": int(iou >= 0.80),
        "success_iou_90": int(iou >= 0.90),
        "success_corner_2pct": int(norm_error <= 0.02),
        "success_corner_5pct": int(norm_error <= 0.05),
    }


def polygon_iou(first: np.ndarray, second: np.ndarray, shape: tuple[int, int]) -> float:
    first_mask = polygon_mask(first, shape)
    second_mask = polygon_mask(second, shape)
    intersection = int(np.logical_and(first_mask, second_mask).sum())
    union = int(np.logical_or(first_mask, second_mask).sum())
    return safe_div(intersection, union)


def polygon_mask(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
    return mask.astype(bool)


def polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def binary_diagnostics(binary: np.ndarray) -> dict[str, float | int]:
    mask = to_foreground_mask(binary)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    component_areas = [int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, labels_count)]
    small_components = sum(1 for area in component_areas if area <= 8)
    largest_component_ratio = safe_div(max(component_areas, default=0), mask.size)
    return {
        "foreground_ratio": round(float(mask.mean()), 6),
        "component_count": len(component_areas),
        "small_component_count": small_components,
        "largest_component_ratio": round(largest_component_ratio, 6),
    }


def to_foreground_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return gray < 128


def save_case_artifacts(
    case_dir: Path,
    image: np.ndarray,
    pred_quad: np.ndarray,
    gt_quad: np.ndarray,
    rectified: np.ndarray,
    artifacts: dict[str, np.ndarray],
    geometry_metrics: dict[str, float | int],
    contact_cell_width: int,
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    overlay = draw_gt_pred_overlay(image, pred_quad, gt_quad, geometry_metrics)
    images = {
        "original": image,
        "gt_pred_overlay": overlay,
        "rectified": rectified,
        "text_enhanced": artifacts["text_enhanced"],
        "binary_readable": artifacts["binary_readable"],
        "binary_fixed": artifacts["binary_fixed"],
        "binary_otsu": artifacts["binary_otsu"],
        "binary_sauvola": artifacts["binary_sauvola"],
        "binary_wolf": artifacts["binary_wolf"],
        "binary_wolf_fused": artifacts["binary_wolf_fused"],
    }
    for name, artifact in images.items():
        cv2.imwrite(str(case_dir / f"{name}.png"), artifact)
    cv2.imwrite(str(case_dir / "contact_sheet.png"), build_contact_sheet(images, contact_cell_width))


def draw_gt_pred_overlay(
    image: np.ndarray, pred_quad: np.ndarray, gt_quad: np.ndarray, metrics: dict[str, float | int]
) -> np.ndarray:
    overlay = image.copy()
    gt = np.round(gt_quad).astype(np.int32)
    pred = np.round(pred_quad).astype(np.int32)
    cv2.polylines(overlay, [gt], True, (255, 0, 180), 6, cv2.LINE_AA)
    cv2.polylines(overlay, [pred], True, (20, 184, 166), 6, cv2.LINE_AA)
    for index, point in enumerate(gt):
        cv2.circle(overlay, tuple(point), 10, (255, 0, 180), -1)
        cv2.putText(overlay, f"G{index + 1}", tuple(point + np.array([12, -12])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 180), 2)
    for index, point in enumerate(pred):
        cv2.circle(overlay, tuple(point), 10, (20, 184, 166), -1)
        cv2.putText(overlay, f"P{index + 1}", tuple(point + np.array([12, 24])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 184, 166), 2)

    label = f"IoU={float(metrics['polygon_iou']):.3f} err={float(metrics['corner_mean_error_px']):.1f}px"
    cv2.rectangle(overlay, (20, 20), (520, 82), (255, 255, 255), -1)
    cv2.putText(overlay, label, (34, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 40, 52), 2, cv2.LINE_AA)
    return overlay


def build_contact_sheet(images: dict[str, np.ndarray], cell_width: int) -> np.ndarray:
    cells = [make_labeled_cell(key, images[key], cell_width) for key in ARTIFACT_KEYS]
    columns = 4
    rows = [hstack_with_padding(cells[index : index + columns]) for index in range(0, len(cells), columns)]
    return vstack_with_padding(rows)


def make_labeled_cell(label: str, image: np.ndarray, width: int) -> np.ndarray:
    preview = to_preview_bgr(image, width)
    label_height = 34
    label_bar = np.full((label_height, preview.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(label_bar, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 34, 40), 2, cv2.LINE_AA)
    return np.vstack([label_bar, preview])


def to_preview_bgr(image: np.ndarray, width: int) -> np.ndarray:
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
    if width <= 0:
        return bgr.copy()
    scale = width / bgr.shape[1]
    height = max(1, int(round(bgr.shape[0] * scale)))
    return cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)


def hstack_with_padding(cells: list[np.ndarray]) -> np.ndarray:
    max_height = max(cell.shape[0] for cell in cells)
    padded = []
    for cell in cells:
        if cell.shape[0] < max_height:
            pad = np.full((max_height - cell.shape[0], cell.shape[1], 3), 255, dtype=np.uint8)
            cell = np.vstack([cell, pad])
        padded.append(cell)
    return np.hstack(padded)


def vstack_with_padding(rows: list[np.ndarray], gap: int = 18) -> np.ndarray:
    max_width = max(row.shape[1] for row in rows)
    padded_rows: list[np.ndarray] = []
    for index, row in enumerate(rows):
        if row.shape[1] < max_width:
            pad = np.full((row.shape[0], max_width - row.shape[1], 3), 255, dtype=np.uint8)
            row = np.hstack([row, pad])
        padded_rows.append(row)
        if index < len(rows) - 1:
            padded_rows.append(np.full((gap, max_width, 3), 245, dtype=np.uint8))
    return np.vstack(padded_rows)


def summarize_geometry(
    rows: list[dict[str, str | float | int]], group_keys: tuple[str, ...]
) -> list[dict[str, str | float | int]]:
    grouped = group_rows(rows, group_keys)
    summaries: list[dict[str, str | float | int]] = []
    for group, group_rows_value in grouped.items():
        summary: dict[str, str | float | int] = group_to_summary(group_keys, group)
        summary.update(
            {
                "case_count": len(group_rows_value),
                "mean_polygon_iou": mean_metric(group_rows_value, "polygon_iou"),
                "median_polygon_iou": median_metric(group_rows_value, "polygon_iou"),
                "mean_corner_error_px": mean_metric(group_rows_value, "corner_mean_error_px"),
                "mean_corner_error_norm": mean_metric(group_rows_value, "corner_mean_error_norm"),
                "success_iou_80_rate": mean_metric(group_rows_value, "success_iou_80"),
                "success_iou_90_rate": mean_metric(group_rows_value, "success_iou_90"),
                "success_corner_2pct_rate": mean_metric(group_rows_value, "success_corner_2pct"),
                "success_corner_5pct_rate": mean_metric(group_rows_value, "success_corner_5pct"),
                "fallback_rate": mean_if(group_rows_value, lambda row: str(row.get("candidate_source")) == "fallback"),
                "mean_candidate_score": mean_metric(group_rows_value, "candidate_score"),
                "mean_gt_visible_ratio": mean_metric(group_rows_value, "gt_visible_ratio"),
                "mean_readable_text_ratio": mean_metric(group_rows_value, "readable_text_ratio"),
                "mean_readable_text_components": mean_metric(group_rows_value, "readable_text_components"),
            }
        )
        summaries.append(summary)
    return summaries


def summarize_methods(
    rows: list[dict[str, str | float | int]], group_keys: tuple[str, ...]
) -> list[dict[str, str | float | int]]:
    grouped = group_rows(rows, group_keys)
    summaries: list[dict[str, str | float | int]] = []
    for group, group_rows_value in grouped.items():
        summary: dict[str, str | float | int] = group_to_summary(group_keys, group)
        summary.update(
            {
                "case_count": len(group_rows_value),
                "mean_foreground_ratio": mean_metric(group_rows_value, "foreground_ratio"),
                "mean_component_count": mean_metric(group_rows_value, "component_count"),
                "mean_small_component_count": mean_metric(group_rows_value, "small_component_count"),
                "mean_largest_component_ratio": mean_metric(group_rows_value, "largest_component_ratio"),
            }
        )
        summaries.append(summary)
    return summaries


def group_rows(
    rows: list[dict[str, str | float | int]], group_keys: tuple[str, ...]
) -> dict[tuple[str, ...], list[dict[str, str | float | int]]]:
    grouped: dict[tuple[str, ...], list[dict[str, str | float | int]]] = {}
    if not group_keys:
        grouped[("overall",)] = rows
        return grouped
    for row in rows:
        key = tuple(str(row[key]) for key in group_keys)
        grouped.setdefault(key, []).append(row)
    return dict(sorted(grouped.items()))


def group_to_summary(group_keys: tuple[str, ...], group: tuple[str, ...]) -> dict[str, str]:
    if not group_keys:
        return {"group": "overall"}
    return {key: value for key, value in zip(group_keys, group)}


def mean_metric(rows: list[dict[str, str | float | int]], metric: str) -> float:
    values = [safe_numeric(row.get(metric)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return 0.0
    return round(float(np.mean(values)), 6)


def median_metric(rows: list[dict[str, str | float | int]], metric: str) -> float:
    values = [safe_numeric(row.get(metric)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return 0.0
    return round(float(np.median(values)), 6)


def mean_if(rows: list[dict[str, str | float | int]], predicate: Callable[[dict[str, str | float | int]], bool]) -> float:
    return round(float(np.mean([1.0 if predicate(row) else 0.0 for row in rows])), 6) if rows else 0.0


def safe_numeric(value: object) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def write_run_config(path: Path, args: argparse.Namespace, params: ScanParams, cases: list[MidvCase]) -> None:
    config = {
        "case_count": len(cases),
        "pipeline_version": PROCESSING_PIPELINE_VERSION,
        "datasets": {
            "midv500": str(args.midv500),
            "midv2019": str(args.midv2019),
        },
        "available_cases_by_dataset": count_by(cases, "dataset"),
        "available_cases_by_condition": count_by(cases, "condition_code"),
        "params": {
            "fixed_threshold": params.fixed_threshold,
            "sauvola_window": params.sauvola_window,
            "sauvola_k": params.sauvola_k,
            "cleanup_kernel": params.cleanup_kernel,
            "frames_per_video": args.frames_per_video,
        },
        "primary_metrics": [
            "polygon_iou",
            "corner_mean_error_px",
            "corner_mean_error_norm",
            "success_iou_80",
            "success_iou_90",
            "success_corner_2pct",
            "success_corner_5pct",
        ],
        "method_artifact_keys": list(BINARIZATION_METHOD_KEYS),
    }
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def count_by(cases: list[MidvCase], field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        key = str(getattr(case, field_name))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def write_csv(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def case_metadata(case: MidvCase) -> dict[str, str]:
    return {
        "dataset": case.dataset,
        "document_id": case.document_id,
        "condition_code": case.condition_code,
        "condition": case.condition,
        "condition_label": case.condition_label,
        "device": case.device,
        "frame_id": case.frame_id,
        "source": str(case.source_path),
        "image_ref": case.image_ref,
        "gt_ref": case.gt_ref,
    }


def case_key(case: MidvCase) -> str:
    return f"{case.dataset}_{case.document_id}_{case.frame_id}"


def parse_csv_filter(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def condition_matches(condition_code: str, selected_conditions: set[str]) -> bool:
    if not selected_conditions:
        return True
    return condition_code in selected_conditions or condition_code[:1] in selected_conditions


def document_id_from_source(path: Path) -> str:
    return path.stem if path.suffix.lower() == ".zip" else path.name


if __name__ == "__main__":
    main()
