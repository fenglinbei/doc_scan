from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import cv2
import numpy as np
from skimage.filters import threshold_sauvola


DETECTION_MAX_DIM = 1000
RECTIFIED_OUTPUT_MAX_DIM = 1800
PROCESSING_PIPELINE_VERSION = "2.7"
LOCAL_BINARIZATION_WINDOW = 35
NIBLACK_K = -0.2
WOLF_K = 0.5
NICK_K = -0.2
BRADLEY_T = 0.15
WOLF_FUSED_WEAK_SCALE = 1.05
WOLF_FUSED_STRONG_SCALE = 1.25
WOLF_FUSED_WEAK_PERCENTILE = 75.0
WOLF_FUSED_STRONG_PERCENTILE = 90.0
WOLF_FUSED_MIN_AREA = 2
BINARIZATION_METHOD_KEYS = (
    "binary_fixed",
    "binary_otsu",
    "binary_sauvola",
    "binary_niblack",
    "binary_wolf",
    "binary_wolf_fused",
    "binary_nick",
    "binary_bradley",
    "binary_readable",
)


@dataclass(frozen=True)
class ScanParams:
    canny_low: int = 50
    canny_high: int = 150
    fixed_threshold: int = 180
    illumination_kernel: int = 45
    sauvola_window: int = 35
    sauvola_k: float = 0.2
    cleanup_kernel: int = 3

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "ScanParams":
        if not data:
            return cls()
        return cls(
            canny_low=int(data.get("canny_low", cls.canny_low)),
            canny_high=int(data.get("canny_high", cls.canny_high)),
            fixed_threshold=int(data.get("fixed_threshold", cls.fixed_threshold)),
            illumination_kernel=int(data.get("illumination_kernel", cls.illumination_kernel)),
            sauvola_window=int(data.get("sauvola_window", cls.sauvola_window)),
            sauvola_k=float(data.get("sauvola_k", cls.sauvola_k)),
            cleanup_kernel=int(data.get("cleanup_kernel", cls.cleanup_kernel)),
        ).normalized()

    def normalized(self) -> "ScanParams":
        low = int(np.clip(self.canny_low, 1, 255))
        high = int(np.clip(self.canny_high, 1, 255))
        if low >= high:
            high = min(255, low + 1)
        return ScanParams(
            canny_low=low,
            canny_high=high,
            fixed_threshold=int(np.clip(self.fixed_threshold, 1, 254)),
            illumination_kernel=ensure_odd(self.illumination_kernel, 3, 151),
            sauvola_window=ensure_odd(self.sauvola_window, 3, 151),
            sauvola_k=float(np.clip(self.sauvola_k, 0.0, 0.8)),
            cleanup_kernel=ensure_odd(self.cleanup_kernel, 1, 21),
        )


@dataclass
class Candidate:
    corners: np.ndarray
    score: float
    area_ratio: float
    angle_score: float
    edge_score: float
    contrast_score: float
    source: str
    side_score: float
    surface_score: float
    margin_score: float
    aspect_score: float
    paper_boundary_score: float


@dataclass
class LineGroup:
    kind: str
    points: np.ndarray
    length: float
    position: float
    direction: np.ndarray
    point: np.ndarray


@dataclass
class ScanOutput:
    corners: list[list[float]]
    candidate_score: float
    warnings: list[str]
    metrics: dict[str, float | int | str]
    artifacts: dict[str, np.ndarray]


def ensure_odd(value: int, minimum: int, maximum: int) -> int:
    value = int(np.clip(value, minimum, maximum))
    if value % 2 == 0:
        value += 1
    return min(value, maximum if maximum % 2 == 1 else maximum - 1)


def decode_image(image_bytes: bytes) -> np.ndarray:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Uploaded file is not a readable image.")
    return image


def process_image_bytes(image_bytes: bytes, params: ScanParams | None = None) -> ScanOutput:
    return process_image(decode_image(image_bytes), params or ScanParams())


def process_image(image: np.ndarray, params: ScanParams | None = None) -> ScanOutput:
    params = (params or ScanParams()).normalized()
    warnings: list[str] = []
    timings: dict[str, float] = {}
    stage_start = time.perf_counter()
    original = image.copy()
    timings["copy_input"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    detect_image, resize_ratio = resize_for_detection(original)
    gray = cv2.cvtColor(detect_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, params.canny_low, params.canny_high)
    timings["prepare_edges"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    connected_edges = connect_document_edges(edges)
    timings["connect_edges"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    candidates = find_quad_candidates(connected_edges, gray, detect_image)
    timings["find_candidates"] = time.perf_counter() - stage_start
    best_candidate: Candidate | None = None
    if candidates:
        best_candidate = max(candidates, key=lambda candidate: candidate.score)
        corners_detect = order_points(best_candidate.corners)
        corners_detect, folded_corner_refinements = refine_folded_corners(corners_detect, detect_image)
        candidate_score = float(best_candidate.score)
    else:
        warnings.append("未找到可靠四边形候选，已回退到整张图像边界。")
        h, w = detect_image.shape[:2]
        corners_detect = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        folded_corner_refinements = 0
        candidate_score = 0.0

    original_corners = corners_detect / resize_ratio
    original_corners = clamp_corners(original_corners, original.shape[1], original.shape[0])
    stage_start = time.perf_counter()
    rectified = four_point_warp(original, original_corners)
    rectified, rectified_scale = resize_for_output(rectified)
    timings["warp_and_resize"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    artifacts, threshold_metrics = process_rectified_document(rectified, params)
    timings["enhance_and_binarize"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    corner_overlay = draw_corner_overlay(original, original_corners, candidate_score)
    artifacts = {
        "original": original,
        "edges": connected_edges,
        "corner_detection": corner_overlay,
        "rectified": rectified,
        **artifacts,
    }
    timings["assemble_artifacts"] = time.perf_counter() - stage_start

    metrics = {
        "pipeline_version": PROCESSING_PIPELINE_VERSION,
        "candidate_count": len(candidates),
        "candidate_score": round(candidate_score, 4),
        "candidate_source": best_candidate.source if best_candidate else "fallback",
        "candidate_paper_boundary_score": round(best_candidate.paper_boundary_score, 4) if best_candidate else 0.0,
        "folded_corner_refinements": folded_corner_refinements,
        "detection_width": int(detect_image.shape[1]),
        "detection_height": int(detect_image.shape[0]),
        "rectified_scale": round(rectified_scale, 4),
        "output_width": int(rectified.shape[1]),
        "output_height": int(rectified.shape[0]),
        **threshold_metrics,
        **{f"time_{name}_ms": int(round(value * 1000)) for name, value in timings.items()},
        "time_total_ms": int(round(sum(timings.values()) * 1000)),
    }

    if candidate_score and candidate_score < 0.45:
        warnings.append("文档边界候选分数偏低，建议检查复杂背景、阴影或裁切结果。")

    return ScanOutput(
        corners=np.round(original_corners, 2).tolist(),
        candidate_score=round(candidate_score, 4),
        warnings=warnings,
        metrics=metrics,
        artifacts=artifacts,
    )


def resize_for_detection(image: np.ndarray, max_dim: int = DETECTION_MAX_DIM) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    largest = max(h, w)
    if largest <= max_dim:
        return image.copy(), 1.0
    ratio = max_dim / float(largest)
    resized = cv2.resize(image, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_AREA)
    return resized, ratio


def resize_for_output(image: np.ndarray, max_dim: int = RECTIFIED_OUTPUT_MAX_DIM) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    largest = max(h, w)
    if largest <= max_dim:
        return image, 1.0
    ratio = max_dim / float(largest)
    resized = cv2.resize(image, (int(round(w * ratio)), int(round(h * ratio))), interpolation=cv2.INTER_AREA)
    return resized, ratio


def connect_document_edges(edges: np.ndarray) -> np.ndarray:
    short_side = min(edges.shape[:2])
    directional_size = ensure_odd(int(short_side * 0.04), 15, 61)
    square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (directional_size, 3))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, directional_size))

    square_connected = cv2.dilate(edges, square_kernel, iterations=1)
    square_connected = cv2.morphologyEx(square_connected, cv2.MORPH_CLOSE, square_kernel, iterations=2)
    horizontal_connected = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel, iterations=1)
    vertical_connected = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vertical_kernel, iterations=1)
    return cv2.bitwise_or(square_connected, cv2.bitwise_or(horizontal_connected, vertical_connected))


def find_quad_candidates(edges: np.ndarray, gray: np.ndarray, image: np.ndarray | None = None) -> list[Candidate]:
    image_area = float(edges.shape[0] * edges.shape[1])
    paper_mask = estimate_paper_mask(image)
    candidates: list[Candidate] = []

    for corners in deduplicate_quad_proposals(contour_quad_proposals(edges)):
        append_candidate(candidates, corners, edges, gray, image_area, "contour", paper_mask)

    for corners in deduplicate_quad_proposals(bright_region_quad_proposals(gray, image)):
        append_candidate(candidates, corners, edges, gray, image_area, "bright-region", paper_mask)

    for corners in deduplicate_quad_proposals(hough_quad_proposals(edges)):
        append_candidate(candidates, corners, edges, gray, image_area, "hough-lines", paper_mask)

    return deduplicate_candidates(candidates)


def append_candidate(
    candidates: list[Candidate],
    corners: np.ndarray,
    edges: np.ndarray,
    gray: np.ndarray,
    image_area: float,
    source: str,
    paper_mask: np.ndarray | None,
) -> None:
    candidate = build_candidate(corners, edges, gray, image_area, source, paper_mask)
    if candidate is not None:
        candidates.append(candidate)


def contour_quad_proposals(edges: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(edges.shape[0] * edges.shape[1])
    proposals: list[np.ndarray] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:48]:
        if cv2.contourArea(contour) < image_area * 0.015:
            continue
        proposals.extend(quad_proposals_from_contour(contour))
    return proposals


def bright_region_quad_proposals(gray: np.ndarray, image: np.ndarray | None) -> list[np.ndarray]:
    if image is None:
        return []

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    lightness = lab[:, :, 0]

    value_threshold = max(115, int(np.percentile(value, 62)))
    lightness_threshold = max(125, int(np.percentile(lightness, 58)))
    saturation_threshold = int(np.clip(np.percentile(saturation, 62), 45, 105))
    low_saturation = saturation <= saturation_threshold
    light_surface = np.logical_and(value >= value_threshold, lightness >= lightness_threshold)
    mask = np.logical_and(light_surface, low_saturation).astype(np.uint8) * 255

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_or(mask, cv2.bitwise_and(otsu, (low_saturation.astype(np.uint8) * 255)))

    proposals: list[np.ndarray] = []
    image_area = float(gray.shape[0] * gray.shape[1])
    for candidate_mask in bright_region_masks(mask):
        contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            if cv2.contourArea(contour) < image_area * 0.04:
                continue
            proposals.extend(quad_proposals_from_contour(contour))
    return proposals


def estimate_paper_mask(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    lightness = lab[:, :, 0]

    saturation_threshold = int(np.clip(np.percentile(saturation, 55), 38, 70))
    value_threshold = int(max(120, min(165, np.percentile(value, 58))))
    lightness_threshold = int(max(135, min(175, np.percentile(lightness, 58))))
    paper = np.logical_and.reduce(
        (
            saturation <= saturation_threshold,
            value >= value_threshold,
            lightness >= lightness_threshold,
        )
    ).astype(np.uint8)
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return paper


def refine_folded_corners(corners: np.ndarray, image: np.ndarray) -> tuple[np.ndarray, int]:
    ordered = order_points(corners)
    paper_mask = estimate_paper_mask(image)
    if paper_mask is None:
        return ordered, 0

    contour = largest_paper_component_near_quad(paper_mask, ordered)
    if contour is None:
        return ordered, 0

    contour_points = contour.reshape(-1, 2).astype(np.float32)
    if len(contour_points) < 80:
        return ordered, 0

    width_estimate = max(float(np.linalg.norm(ordered[1] - ordered[0])), float(np.linalg.norm(ordered[2] - ordered[3])))
    height_estimate = max(float(np.linalg.norm(ordered[3] - ordered[0])), float(np.linalg.norm(ordered[2] - ordered[1])))
    if width_estimate < 1 or height_estimate < 1:
        return ordered, 0

    destination = np.array(
        [[0, 0], [width_estimate, 0], [width_estimate, height_estimate], [0, height_estimate]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered.astype(np.float32), destination)
    normalized = normalize_quad_points(contour_points, matrix, width_estimate, height_estimate)
    if normalized is None:
        return ordered, 0
    u, v = normalized[:, 0], normalized[:, 1]

    lines = {
        "top": fit_line_from_points(contour_points[(v > -0.12) & (v < 0.08) & (u > 0.20) & (u < 0.96)]),
        "right": fit_line_from_points(contour_points[(u > 0.92) & (u < 1.10) & (v > 0.10) & (v < 0.92)]),
        "bottom": fit_line_from_points(contour_points[(v > 0.92) & (v < 1.10) & (u > 0.08) & (u < 0.92)]),
        "left": fit_line_from_points(contour_points[(u > -0.12) & (u < 0.08) & (v > 0.20) & (v < 0.96)]),
    }
    intersections = [
        intersect_fitted_lines(lines["top"], lines["left"]),
        intersect_fitted_lines(lines["top"], lines["right"]),
        intersect_fitted_lines(lines["bottom"], lines["right"]),
        intersect_fitted_lines(lines["bottom"], lines["left"]),
    ]

    refined = ordered.copy()
    center = np.mean(ordered, axis=0)
    short_side = min(width_estimate, height_estimate)
    min_shift = max(12.0, short_side * 0.05)
    max_shift = max(min_shift + 1.0, short_side * 0.22)
    refinements = 0

    for index, point in enumerate(intersections):
        if point is None or not np.all(np.isfinite(point)):
            continue
        shift = float(np.linalg.norm(point - ordered[index]))
        if shift < min_shift or shift > max_shift:
            continue
        old_radius = float(np.linalg.norm(ordered[index] - center))
        new_radius = float(np.linalg.norm(point - center))
        if new_radius <= old_radius + min_shift * 0.25:
            continue
        refined[index] = point.astype(np.float32)
        refinements += 1

    if refinements == 0:
        return ordered, 0
    refined = clamp_corners(refined, image.shape[1], image.shape[0])
    if not cv2.isContourConvex(refined.astype(np.float32)):
        return ordered, 0
    if abs(cv2.contourArea(refined)) < abs(cv2.contourArea(ordered)) * 0.92:
        return ordered, 0
    return order_points(refined), refinements


def largest_paper_component_near_quad(paper_mask: np.ndarray, corners: np.ndarray) -> np.ndarray | None:
    height, width = paper_mask.shape[:2]
    center = np.mean(corners, axis=0)
    expanded = center + (corners - center) * 1.14
    roi = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(roi, [clamp_corners(expanded, width, height).astype(np.int32)], 255)
    masked = cv2.bitwise_and((paper_mask * 255).astype(np.uint8), roi)
    masked = cv2.morphologyEx(masked, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    min_area = max(300.0, abs(cv2.contourArea(corners.astype(np.float32))) * 0.18)
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return None
    return contour


def normalize_quad_points(
    points: np.ndarray, matrix: np.ndarray, width: float, height: float
) -> np.ndarray | None:
    homogeneous = np.hstack([points.astype(np.float32), np.ones((len(points), 1), dtype=np.float32)])
    projected = (matrix @ homogeneous.T).T
    denominator = projected[:, 2:3]
    valid = np.abs(denominator[:, 0]) > 1e-6
    if np.count_nonzero(valid) < 32:
        return None
    safe_denominator = denominator.copy()
    safe_denominator[~valid] = np.nan
    uv = projected[:, :2] / safe_denominator
    normalized = np.column_stack((uv[:, 0] / max(width, 1.0), uv[:, 1] / max(height, 1.0))).astype(np.float32)
    return normalized


def fit_line_from_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(points) < 24:
        return None
    vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).reshape(4)
    direction = np.array([float(vx), float(vy)], dtype=np.float32)
    point = np.array([float(x0), float(y0)], dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return None
    return direction / norm, point


def intersect_fitted_lines(
    first: tuple[np.ndarray, np.ndarray] | None,
    second: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray | None:
    if first is None or second is None:
        return None
    first_direction, first_point = first
    second_direction, second_point = second
    cross = float(first_direction[0] * second_direction[1] - first_direction[1] * second_direction[0])
    if abs(cross) < 1e-5:
        return None
    delta = second_point - first_point
    t = float(delta[0] * second_direction[1] - delta[1] * second_direction[0]) / cross
    return first_point + t * first_direction


def bright_region_masks(mask: np.ndarray) -> list[np.ndarray]:
    short_side = min(mask.shape[:2])
    open_size = ensure_odd(int(short_side * 0.006), 3, 13)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_size, open_size))
    close_sizes = (
        ensure_odd(int(short_side * 0.010), 5, 19),
        ensure_odd(int(short_side * 0.018), 9, 31),
        ensure_odd(int(short_side * 0.030), 15, 55),
    )
    masks: list[np.ndarray] = []
    for close_size in close_sizes:
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
        candidate = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, open_kernel, iterations=1)
        masks.append(candidate)
    return masks


def quad_proposals_from_contour(contour: np.ndarray) -> list[np.ndarray]:
    proposals: list[np.ndarray] = []
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return proposals

    shapes = [contour, cv2.convexHull(contour)]
    for shape in shapes:
        shape_perimeter = cv2.arcLength(shape, True)
        if shape_perimeter <= 0:
            continue
        for epsilon_factor in (0.008, 0.012, 0.016, 0.02, 0.03, 0.045, 0.06, 0.08):
            approx = cv2.approxPolyDP(shape, epsilon_factor * shape_perimeter, True)
            points = approx.reshape(-1, 2).astype(np.float32)
            if len(points) == 4:
                proposals.append(points)
            elif 5 <= len(points) <= 16:
                extreme_quad = quad_from_extreme_points(points)
                if extreme_quad is not None:
                    proposals.append(extreme_quad)

        hull_points = cv2.convexHull(shape).reshape(-1, 2).astype(np.float32)
        if len(hull_points) >= 4:
            extreme_quad = quad_from_extreme_points(hull_points)
            if extreme_quad is not None:
                proposals.append(extreme_quad)

    rect = cv2.minAreaRect(contour)
    proposals.append(cv2.boxPoints(rect).astype(np.float32))
    return proposals


def quad_from_extreme_points(points: np.ndarray) -> np.ndarray | None:
    if len(points) < 4:
        return None
    points = points.astype(np.float32)
    point_sum = points.sum(axis=1)
    point_diff = np.diff(points, axis=1).reshape(-1)
    quad = np.array(
        [
            points[np.argmin(point_sum)],
            points[np.argmin(point_diff)],
            points[np.argmax(point_sum)],
            points[np.argmax(point_diff)],
        ],
        dtype=np.float32,
    )
    rounded = np.round(quad, 1)
    if len(np.unique(rounded, axis=0)) < 4:
        return None
    return quad


def hough_quad_proposals(edges: np.ndarray) -> list[np.ndarray]:
    height, width = edges.shape[:2]
    short_side = min(height, width)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(70, int(short_side * 0.09)),
        minLineLength=max(70, int(short_side * 0.22)),
        maxLineGap=max(20, int(short_side * 0.05)),
    )
    if lines is None:
        return []

    horizontal_groups, vertical_groups = group_hough_segments(lines.reshape(-1, 4), edges.shape)
    if len(horizontal_groups) < 2 or len(vertical_groups) < 2:
        return []

    top_groups = select_line_groups(horizontal_groups, 0.03 * height, 0.78 * height)
    bottom_groups = select_line_groups(horizontal_groups, 0.22 * height, 0.98 * height)
    left_groups = select_line_groups(vertical_groups, 0.02 * width, 0.76 * width)
    right_groups = select_line_groups(vertical_groups, 0.24 * width, 0.99 * width)

    proposals: list[np.ndarray] = []
    for top in top_groups:
        for bottom in bottom_groups:
            if bottom.position <= top.position + 0.18 * height:
                continue
            for left in left_groups:
                for right in right_groups:
                    if right.position <= left.position + 0.18 * width:
                        continue
                    corners = intersections_for_lines(top, right, bottom, left)
                    if corners is not None:
                        proposals.append(corners)
    return proposals


def group_hough_segments(segments: np.ndarray, shape: tuple[int, ...]) -> tuple[list[LineGroup], list[LineGroup]]:
    height, width = shape[:2]
    short_side = min(height, width)
    min_length = max(70.0, short_side * 0.18)
    raw_groups: list[LineGroup] = []

    for x1, y1, x2, y2 in segments.astype(np.float32):
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < min_length:
            continue
        angle = float(np.degrees(np.arctan2(dy, dx)))
        if abs(angle) <= 24:
            kind = "horizontal"
        elif abs(abs(angle) - 90) <= 24:
            kind = "vertical"
        else:
            continue
        group = make_line_group(kind, np.array([[x1, y1], [x2, y2]], dtype=np.float32), length, shape)
        if np.isfinite(group.position):
            raw_groups.append(group)

    horizontal = merge_line_groups([group for group in raw_groups if group.kind == "horizontal"], shape)
    vertical = merge_line_groups([group for group in raw_groups if group.kind == "vertical"], shape)
    return horizontal, vertical


def merge_line_groups(groups: list[LineGroup], shape: tuple[int, ...]) -> list[LineGroup]:
    if not groups:
        return []
    short_side = min(shape[:2])
    merge_distance = max(14.0, short_side * 0.025)
    merged: list[LineGroup] = []

    for group in sorted(groups, key=lambda item: item.position):
        if not merged or abs(group.position - merged[-1].position) > merge_distance:
            merged.append(group)
            continue
        previous = merged[-1]
        points = np.vstack([previous.points, group.points])
        merged[-1] = make_line_group(previous.kind, points, previous.length + group.length, shape)

    return sorted(merged, key=lambda item: item.length, reverse=True)


def make_line_group(kind: str, points: np.ndarray, length: float, shape: tuple[int, ...]) -> LineGroup:
    height, width = shape[:2]
    points = points.reshape(-1, 2).astype(np.float32)
    if len(points) >= 2:
        vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).reshape(4)
        direction = np.array([float(vx), float(vy)], dtype=np.float32)
        point = np.array([float(x0), float(y0)], dtype=np.float32)
    else:
        direction = np.array([1.0, 0.0], dtype=np.float32)
        point = points[0]
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm == 0:
        direction = np.array([1.0, 0.0], dtype=np.float32)
    else:
        direction = direction / direction_norm
    position = line_position_at_center(kind, direction, point, width, height)
    return LineGroup(kind=kind, points=points, length=float(length), position=position, direction=direction, point=point)


def line_position_at_center(kind: str, direction: np.ndarray, point: np.ndarray, width: int, height: int) -> float:
    if kind == "horizontal":
        if abs(float(direction[0])) < 1e-4:
            return float(point[1])
        t = (width * 0.5 - float(point[0])) / float(direction[0])
        return float(point[1] + direction[1] * t)
    if abs(float(direction[1])) < 1e-4:
        return float(point[0])
    t = (height * 0.5 - float(point[1])) / float(direction[1])
    return float(point[0] + direction[0] * t)


def select_line_groups(groups: list[LineGroup], minimum: float, maximum: float, limit: int = 4) -> list[LineGroup]:
    selected = [group for group in groups if minimum <= group.position <= maximum]
    return selected[:limit]


def intersections_for_lines(
    top: LineGroup,
    right: LineGroup,
    bottom: LineGroup,
    left: LineGroup,
) -> np.ndarray | None:
    intersections = [
        intersect_line_groups(top, left),
        intersect_line_groups(top, right),
        intersect_line_groups(bottom, right),
        intersect_line_groups(bottom, left),
    ]
    if any(point is None for point in intersections):
        return None
    return np.array(intersections, dtype=np.float32)


def intersect_line_groups(first: LineGroup, second: LineGroup) -> np.ndarray | None:
    p = first.point.astype(np.float32)
    r = first.direction.astype(np.float32)
    q = second.point.astype(np.float32)
    s = second.direction.astype(np.float32)
    cross = float(r[0] * s[1] - r[1] * s[0])
    if abs(cross) < 1e-4:
        return None
    q_minus_p = q - p
    t = float(q_minus_p[0] * s[1] - q_minus_p[1] * s[0]) / cross
    return p + t * r


def build_candidate(
    corners: np.ndarray,
    edges: np.ndarray,
    gray: np.ndarray,
    image_area: float,
    source: str,
    paper_mask: np.ndarray | None = None,
) -> Candidate | None:
    height, width = edges.shape[:2]
    corners = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if len(corners) != 4 or not np.all(np.isfinite(corners)):
        return None
    if is_far_outside_image(corners, width, height):
        return None

    ordered = order_points(corners)
    ordered = clamp_corners(ordered, width, height)
    if not cv2.isContourConvex(ordered.astype(np.float32)):
        return None

    area = abs(cv2.contourArea(ordered))
    area_ratio = area / image_area
    if area_ratio < 0.055 or area_ratio > 0.98:
        return None
    if min_side_length(ordered) < min(width, height) * 0.08:
        return None

    angle_score = rectangle_angle_score(ordered)
    edge_score = edge_coverage_score(edges, ordered)
    side_score = side_coverage_score(edges, ordered)
    contrast_score = contrast_score_for_quad(gray, ordered)
    surface_score = surface_score_for_quad(gray, ordered)
    margin_score = margin_score_for_quad(ordered, width, height)
    aspect_score = aspect_score_for_quad(ordered)
    paper_boundary_score = paper_boundary_score_for_quad(paper_mask, ordered, gray.shape)
    area_score = min(area_ratio / 0.42, 1.0)
    border_touches = image_border_touch_count(ordered, width, height)

    score = (
        0.18 * area_score
        + 0.16 * angle_score
        + 0.08 * edge_score * paper_boundary_score
        + 0.16 * side_score * paper_boundary_score
        + 0.08 * contrast_score
        + 0.06 * surface_score
        + 0.06 * margin_score
        + 0.04 * aspect_score
        + 0.18 * paper_boundary_score
    )
    if source == "bright-region" and area_ratio >= 0.35 and surface_score >= 0.72 and paper_boundary_score >= 0.84:
        score += 0.04
    if source == "bright-region" and paper_boundary_score < 0.72:
        score -= 0.07
    if source == "hough-lines" and area_ratio < 0.42:
        if side_score >= 0.55 and paper_boundary_score >= 0.65:
            score -= 0.03
        else:
            score -= 0.10
    if border_touches >= 2 and area_ratio >= 0.45 and side_score < 0.45:
        score -= 0.12 + 0.05 * float(border_touches - 2)
    if border_touches >= 3 and area_ratio >= 0.58 and edge_score < 0.45:
        score -= 0.10
    if border_touches >= 2 and side_score < 0.20 and edge_score < 0.25:
        score -= 0.14
    elif border_touches >= 3 and side_score < 0.25:
        score -= 0.10
    return Candidate(
        corners=ordered,
        score=float(score),
        area_ratio=float(area_ratio),
        angle_score=float(angle_score),
        edge_score=float(edge_score),
        contrast_score=float(contrast_score),
        source=source,
        side_score=float(side_score),
        surface_score=float(surface_score),
        margin_score=float(margin_score),
        aspect_score=float(aspect_score),
        paper_boundary_score=float(paper_boundary_score),
    )


def is_far_outside_image(corners: np.ndarray, width: int, height: int) -> bool:
    margin_x = width * 0.08
    margin_y = height * 0.08
    return bool(
        np.any(corners[:, 0] < -margin_x)
        or np.any(corners[:, 0] > width - 1 + margin_x)
        or np.any(corners[:, 1] < -margin_y)
        or np.any(corners[:, 1] > height - 1 + margin_y)
    )


def min_side_length(corners: np.ndarray) -> float:
    return float(
        min(
            np.linalg.norm(corners[(index + 1) % 4] - corners[index])
            for index in range(4)
        )
    )


def deduplicate_candidates(candidates: list[Candidate]) -> list[Candidate]:
    unique: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if any(mean_corner_distance(candidate.corners, existing.corners) < 10.0 for existing in unique):
            continue
        unique.append(candidate)
    return unique


def deduplicate_quad_proposals(proposals: list[np.ndarray], distance_threshold: float = 5.0) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for corners in proposals:
        ordered = order_points(np.asarray(corners, dtype=np.float32))
        if any(mean_corner_distance(ordered, existing) < distance_threshold for existing in unique):
            continue
        unique.append(ordered)
    return unique


def mean_corner_distance(first: np.ndarray, second: np.ndarray) -> float:
    first_ordered = order_points(first)
    second_ordered = order_points(second)
    return float(np.mean(np.linalg.norm(first_ordered - second_ordered, axis=1)))


def order_points(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    point_sum = points.sum(axis=1)
    point_diff = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(point_sum)]
    ordered[2] = points[np.argmax(point_sum)]
    ordered[1] = points[np.argmin(point_diff)]
    ordered[3] = points[np.argmax(point_diff)]
    return ordered


def clamp_corners(corners: np.ndarray, width: int, height: int) -> np.ndarray:
    clamped = corners.copy()
    clamped[:, 0] = np.clip(clamped[:, 0], 0, width - 1)
    clamped[:, 1] = np.clip(clamped[:, 1], 0, height - 1)
    return clamped.astype(np.float32)


def rectangle_angle_score(corners: np.ndarray) -> float:
    scores = []
    for index in range(4):
        prev_point = corners[(index - 1) % 4]
        point = corners[index]
        next_point = corners[(index + 1) % 4]
        v1 = prev_point - point
        v2 = next_point - point
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom == 0:
            return 0.0
        cos_angle = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
        angle = np.degrees(np.arccos(cos_angle))
        scores.append(max(0.0, 1.0 - abs(angle - 90.0) / 90.0))
    return float(np.mean(scores))


def edge_coverage_score(edges: np.ndarray, corners: np.ndarray) -> float:
    outline = np.zeros_like(edges)
    cv2.polylines(outline, [corners.astype(np.int32)], True, 255, 3)
    outline_pixels = outline > 0
    if not np.any(outline_pixels):
        return 0.0
    dilated_edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    overlap = np.logical_and(outline_pixels, dilated_edges > 0)
    return float(np.count_nonzero(overlap) / np.count_nonzero(outline_pixels))


def side_coverage_score(edges: np.ndarray, corners: np.ndarray) -> float:
    dilated_edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    side_scores: list[float] = []
    for index in range(4):
        start = corners[index]
        end = corners[(index + 1) % 4]
        length = float(np.linalg.norm(end - start))
        if length <= 1:
            side_scores.append(0.0)
            continue
        samples = int(np.clip(length / 4.0, 32, 260))
        weights = np.linspace(0.06, 0.94, samples, dtype=np.float32)
        xs = np.round(start[0] + (end[0] - start[0]) * weights).astype(np.int32)
        ys = np.round(start[1] + (end[1] - start[1]) * weights).astype(np.int32)
        valid = (0 <= xs) & (xs < edges.shape[1]) & (0 <= ys) & (ys < edges.shape[0])
        if not np.any(valid):
            side_scores.append(0.0)
            continue
        hits = dilated_edges[ys[valid], xs[valid]] > 0
        side_scores.append(float(np.count_nonzero(hits) / np.count_nonzero(valid)))

    if not side_scores:
        return 0.0
    return float(0.7 * np.mean(side_scores) + 0.3 * min(side_scores))


def contrast_score_for_quad(gray: np.ndarray, corners: np.ndarray) -> float:
    inner_values = sample_quad_grid_values(gray, corners, grid_size=18, shrink=0.78)
    outer_values = sample_outer_band_values(gray, corners)
    if inner_values.size < 80 or outer_values.size < 40:
        return 0.0
    delta = abs(float(np.mean(inner_values)) - float(np.mean(outer_values)))
    return float(np.clip(delta / 80.0, 0.0, 1.0))


def surface_score_for_quad(gray: np.ndarray, corners: np.ndarray) -> float:
    pixels = sample_quad_grid_values(gray, corners, grid_size=24, shrink=0.72)
    if pixels.size < 100:
        return 0.0
    median = float(np.median(pixels))
    mad = float(np.median(np.abs(pixels.astype(np.float32) - median)))
    brightness_score = float(np.clip((median - 85.0) / 135.0, 0.0, 1.0))
    smoothness_score = float(np.clip(1.0 - mad / 45.0, 0.0, 1.0))
    return float(0.7 * brightness_score + 0.3 * smoothness_score)


def margin_score_for_quad(corners: np.ndarray, width: int, height: int) -> float:
    min_margin = min(
        float(np.min(corners[:, 0])),
        float(width - 1 - np.max(corners[:, 0])),
        float(np.min(corners[:, 1])),
        float(height - 1 - np.max(corners[:, 1])),
    )
    return float(np.clip(min_margin / max(1.0, min(width, height) * 0.035), 0.0, 1.0))


def image_border_touch_count(corners: np.ndarray, width: int, height: int) -> int:
    tolerance = max(2.0, min(width, height) * 0.006)
    return (
        int(float(np.min(corners[:, 0])) <= tolerance)
        + int(float(np.max(corners[:, 0])) >= width - 1 - tolerance)
        + int(float(np.min(corners[:, 1])) <= tolerance)
        + int(float(np.max(corners[:, 1])) >= height - 1 - tolerance)
    )


def aspect_score_for_quad(corners: np.ndarray) -> float:
    tl, tr, br, bl = corners
    width_estimate = max(float(np.linalg.norm(tr - tl)), float(np.linalg.norm(br - bl)))
    height_estimate = max(float(np.linalg.norm(bl - tl)), float(np.linalg.norm(br - tr)))
    shorter = max(1.0, min(width_estimate, height_estimate))
    longer = max(width_estimate, height_estimate)
    ratio = longer / shorter
    if ratio <= 3.0:
        return 1.0
    return float(np.clip(1.0 - (ratio - 3.0) / 3.0, 0.35, 1.0))


def paper_boundary_score_for_quad(
    paper_mask: np.ndarray | None, corners: np.ndarray, shape: tuple[int, ...]
) -> float:
    if paper_mask is None:
        return 0.75
    height, width = shape[:2]
    pts = clamp_corners(order_points(corners), width, height).astype(np.float32)
    if abs(cv2.contourArea(pts)) < 100:
        return 0.0

    center = np.mean(pts, axis=0)
    band_width = float(np.clip(min(height, width) * 0.018, 8.0, 24.0))
    side_ratios: list[float] = []
    border_samples: list[np.ndarray] = []
    for index in range(4):
        start = pts[index]
        end = pts[(index + 1) % 4]
        side = end - start
        length = float(np.linalg.norm(side))
        if length <= 1:
            continue

        sample_count = int(np.clip(length / 3.0, 32, 240))
        weights = np.linspace(0.04, 0.96, sample_count, dtype=np.float32)
        base_points = start[None, :] + side[None, :] * weights[:, None]

        normal = np.array([-side[1], side[0]], dtype=np.float32) / length
        midpoint = (start + end) * 0.5
        if float(np.dot(normal, center - midpoint)) < 0:
            normal = -normal

        side_samples: list[np.ndarray] = []
        for offset in (0.0, band_width * 0.45, band_width * 0.9):
            values = sample_mask_at_points(paper_mask, base_points + normal[None, :] * offset)
            if values.size:
                side_samples.append(values)
                border_samples.append(values)
        if side_samples:
            side_ratios.append(float(np.mean(np.concatenate(side_samples))))

    if not side_ratios or not border_samples:
        return 0.0
    border_ratio = float(np.mean(np.concatenate(border_samples)))
    side_mean = float(np.mean(side_ratios)) if side_ratios else 0.0
    side_floor = float(min(side_ratios)) if side_ratios else 0.0
    return float(np.clip(0.55 * border_ratio + 0.25 * side_mean + 0.20 * side_floor, 0.0, 1.0))


def sample_mask_at_points(mask: np.ndarray, points: np.ndarray) -> np.ndarray:
    return sample_array_at_points(mask, points)


def sample_array_at_points(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    xs = np.round(points[:, 0]).astype(np.int32)
    ys = np.round(points[:, 1]).astype(np.int32)
    valid = (0 <= xs) & (xs < image.shape[1]) & (0 <= ys) & (ys < image.shape[0])
    if not np.any(valid):
        return np.array([], dtype=np.float32)
    return image[ys[valid], xs[valid]].astype(np.float32)


def sample_quad_grid_values(
    image: np.ndarray, corners: np.ndarray, grid_size: int = 20, shrink: float = 1.0
) -> np.ndarray:
    ordered = order_points(corners.astype(np.float32))
    if shrink < 1.0:
        center = np.mean(ordered, axis=0)
        ordered = center + (ordered - center) * float(shrink)
    tl, tr, br, bl = ordered
    weights = np.linspace(0.08, 0.92, grid_size, dtype=np.float32)
    uu, vv = np.meshgrid(weights, weights)
    top = tl[None, None, :] + (tr - tl)[None, None, :] * uu[:, :, None]
    bottom = bl[None, None, :] + (br - bl)[None, None, :] * uu[:, :, None]
    points = top + (bottom - top) * vv[:, :, None]
    return sample_array_at_points(image, points.reshape(-1, 2))


def sample_outer_band_values(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    height, width = gray.shape[:2]
    pts = clamp_corners(order_points(corners), width, height).astype(np.float32)
    center = np.mean(pts, axis=0)
    band_width = float(np.clip(min(height, width) * 0.02, 8.0, 24.0))
    samples: list[np.ndarray] = []
    for index in range(4):
        start = pts[index]
        end = pts[(index + 1) % 4]
        side = end - start
        length = float(np.linalg.norm(side))
        if length <= 1:
            continue
        sample_count = int(np.clip(length / 4.0, 24, 180))
        weights = np.linspace(0.08, 0.92, sample_count, dtype=np.float32)
        base_points = start[None, :] + side[None, :] * weights[:, None]
        inward = np.array([-side[1], side[0]], dtype=np.float32) / length
        midpoint = (start + end) * 0.5
        if float(np.dot(inward, center - midpoint)) < 0:
            inward = -inward
        outward = -inward
        for offset in (band_width * 0.9, band_width * 1.8):
            values = sample_array_at_points(gray, base_points + outward[None, :] * offset)
            if values.size:
                samples.append(values)
    if not samples:
        return np.array([], dtype=np.float32)
    return np.concatenate(samples)


def four_point_warp(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    rect = order_points(corners)
    tl, tr, br, bl = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_width = max(1, int(round(max(width_a, width_b))))
    max_height = max(1, int(round(max(height_a, height_b))))
    destination = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect.astype(np.float32), destination)
    return cv2.warpPerspective(image, matrix, (max_width, max_height), flags=cv2.INTER_LINEAR)


def process_rectified_document(
    rectified: np.ndarray, params: ScanParams | None = None
) -> tuple[dict[str, np.ndarray], dict[str, float | int | str]]:
    params = (params or ScanParams()).normalized()
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    text_enhanced, binary_readable, background, illumination_corrected, text_metrics = enhance_low_contrast_text(gray)
    morphology_enhanced = text_enhanced.copy()

    local_binaries = compute_local_binarization_artifacts(gray)
    binary_wolf_fused = compute_wolf_fused_binary(
        gray,
        background,
        local_binaries["binary_wolf"],
        local_binaries["binary_nick"],
        binary_readable,
        text_metrics,
    )

    _, binary_fixed = cv2.threshold(text_enhanced, params.fixed_threshold, 255, cv2.THRESH_BINARY)
    otsu_threshold, binary_otsu = cv2.threshold(text_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sauvola_threshold = threshold_sauvola(text_enhanced, window_size=params.sauvola_window, k=params.sauvola_k)
    binary_sauvola = (text_enhanced > sauvola_threshold).astype(np.uint8) * 255

    binary_fixed = cleanup_binary(binary_fixed, params.cleanup_kernel)
    binary_otsu = cleanup_binary(binary_otsu, params.cleanup_kernel)
    binary_sauvola = cleanup_binary(binary_sauvola, params.cleanup_kernel)

    metrics = compute_threshold_metrics(gray, illumination_corrected, text_enhanced, binary_readable)
    metrics.update(text_metrics)
    metrics["otsu_threshold"] = round(float(otsu_threshold), 2)
    metrics["sauvola_window"] = params.sauvola_window
    metrics["sauvola_k"] = round(params.sauvola_k, 3)
    metrics["final_output"] = "text_enhanced"

    return (
        {
            "background": background,
            "illumination_corrected": illumination_corrected,
            "detail_enhanced": text_enhanced,
            "text_enhanced": text_enhanced,
            "binary_readable": binary_readable,
            "morphology_enhanced": morphology_enhanced,
            "binary_fixed": binary_fixed,
            "binary_otsu": binary_otsu,
            "binary_sauvola": binary_sauvola,
            **local_binaries,
            "binary_wolf_fused": binary_wolf_fused,
            "final": text_enhanced,
        },
        metrics,
    )


def enhance_and_binarize(rectified: np.ndarray, params: ScanParams) -> tuple[dict[str, np.ndarray], dict[str, float | int | str]]:
    return process_rectified_document(rectified, params)


def enhance_low_contrast_text(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    short_side = min(gray.shape[:2])
    background_kernel = ensure_odd(int(short_side * 0.075), 41, 151)
    background = cv2.GaussianBlur(gray, (background_kernel, background_kernel), 0)
    background = np.maximum(background, 1).astype(np.uint8)

    illumination_corrected = cv2.divide(gray, background, scale=235)
    dark_detail = cv2.subtract(background, gray)
    dark_detail = cv2.GaussianBlur(dark_detail, (3, 3), 0)

    detail_high = float(np.percentile(dark_detail, 99.3))
    if detail_high > 1.0:
        detail_gain = float(np.clip(72.0 / detail_high, 1.35, 5.0))
        text_boost = np.clip(dark_detail.astype(np.float32) * detail_gain, 0, 92)
    else:
        detail_gain = 0.0
        text_boost = np.zeros_like(dark_detail, dtype=np.float32)

    enhanced_float = illumination_corrected.astype(np.float32) - text_boost
    text_enhanced = percentile_stretch(enhanced_float, 0.7, 99.8)
    blur = cv2.GaussianBlur(text_enhanced, (0, 0), sigmaX=0.75)
    text_enhanced = cv2.addWeighted(text_enhanced, 1.32, blur, -0.32, 0)
    text_enhanced = cv2.bilateralFilter(text_enhanced, d=3, sigmaColor=10, sigmaSpace=8)

    relative_dark = dark_detail.astype(np.float32) * 255.0 / np.maximum(background.astype(np.float32), 1.0)
    relative_dark = cv2.GaussianBlur(np.clip(relative_dark, 0, 255).astype(np.uint8), (3, 3), 0)
    otsu_threshold, _ = cv2.threshold(relative_dark, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    target_threshold = foreground_limited_threshold(relative_dark, max_foreground_ratio=0.12)
    text_threshold = float(np.clip(max(6.0, otsu_threshold * 0.82, target_threshold), 6.0, 42.0))
    text_mask = relative_dark > text_threshold
    text_mask = filter_readable_text_mask(text_mask)
    binary_readable = np.where(text_mask, 0, 255).astype(np.uint8)

    text_ratio = float(np.count_nonzero(text_mask) / max(text_mask.size, 1))
    component_count = count_readable_text_components(text_mask)
    return (
        text_enhanced,
        binary_readable,
        background,
        illumination_corrected,
        {
            "text_detail_strength": round(detail_high, 2),
            "text_detail_gain": round(detail_gain, 3),
            "readable_text_threshold": round(text_threshold, 2),
            "readable_text_ratio": round(text_ratio, 4),
            "readable_text_components": int(component_count),
        },
    )


def percentile_stretch(image: np.ndarray, low_percentile: float, high_percentile: float) -> np.ndarray:
    low, high = np.percentile(image, (low_percentile, high_percentile))
    if high - low < 8:
        return np.clip(image, 0, 255).astype(np.uint8)
    stretched = (image.astype(np.float32) - float(low)) * (255.0 / float(high - low))
    return np.clip(stretched, 0, 255).astype(np.uint8)


def foreground_limited_threshold(relative_dark: np.ndarray, max_foreground_ratio: float) -> float:
    positive = relative_dark[relative_dark > 0]
    if positive.size < 32:
        return 6.0
    percentile = float(np.clip(100.0 * (1.0 - max_foreground_ratio), 70.0, 96.5))
    return float(np.percentile(positive, percentile))


def filter_readable_text_mask(mask: np.ndarray) -> np.ndarray:
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    filtered = np.zeros_like(mask, dtype=np.uint8)
    max_area = max(80, int(mask.size * 0.018))
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if 2 <= area <= max_area:
            filtered[labels == label] = 1
    return filtered.astype(bool)


def count_readable_text_components(mask: np.ndarray) -> int:
    labels_count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    max_area = max(80, int(mask.size * 0.006))
    count = 0
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if 2 <= area <= max_area:
            count += 1
    return count


def compute_local_binarization_artifacts(gray: np.ndarray) -> dict[str, np.ndarray]:
    window = ensure_odd(LOCAL_BINARIZATION_WINDOW, 3, 151)
    mean, std, mean_square = local_statistics(gray, window)
    return {
        "binary_niblack": threshold_dark_foreground(gray, niblack_threshold(mean, std, NIBLACK_K)),
        "binary_wolf": threshold_dark_foreground(gray, wolf_threshold(gray, mean, std, WOLF_K)),
        "binary_nick": threshold_dark_foreground(gray, nick_threshold(mean, mean_square, NICK_K)),
        "binary_bradley": threshold_dark_foreground(gray, bradley_threshold(mean, BRADLEY_T)),
    }


def local_statistics(image: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_float = image.astype(np.float32)
    mean = cv2.boxFilter(
        image_float,
        ddepth=-1,
        ksize=(window, window),
        normalize=True,
        borderType=cv2.BORDER_REPLICATE,
    )
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


def compute_wolf_fused_binary(
    gray: np.ndarray,
    background: np.ndarray,
    binary_wolf: np.ndarray,
    binary_nick: np.ndarray,
    binary_readable: np.ndarray,
    text_metrics: dict[str, float | int],
) -> np.ndarray:
    if background.shape != gray.shape:
        background = cv2.resize(background, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA)

    relative_dark = relative_dark_detail(gray, background)
    wolf_mask = to_foreground_mask(binary_wolf)
    nick_mask = to_foreground_mask(binary_nick)
    readable_mask = to_foreground_mask(binary_readable)
    candidate_source = np.logical_or(nick_mask, readable_mask)

    source_values = relative_dark[candidate_source]
    weak_percentile = float(np.clip(WOLF_FUSED_WEAK_PERCENTILE, 0.0, 100.0))
    strong_percentile = float(np.clip(WOLF_FUSED_STRONG_PERCENTILE, weak_percentile, 100.0))
    weak_floor = float(np.percentile(source_values, weak_percentile)) if source_values.size else 6.0
    strong_floor = float(np.percentile(source_values, strong_percentile)) if source_values.size else weak_floor
    base_threshold = safe_float(text_metrics.get("readable_text_threshold"), weak_floor)
    weak_threshold = max(base_threshold * WOLF_FUSED_WEAK_SCALE, weak_floor)
    strong_threshold = max(base_threshold * WOLF_FUSED_STRONG_SCALE, strong_floor, weak_threshold)

    strong_seed = np.logical_or(wolf_mask, np.logical_and(candidate_source, relative_dark >= strong_threshold))
    weak_candidate = np.logical_and(candidate_source, relative_dark >= weak_threshold)
    candidate = np.logical_or(wolf_mask, weak_candidate)
    fused = keep_components_touching_seed(candidate, strong_seed, WOLF_FUSED_MIN_AREA)
    return np.where(fused, 0, 255).astype(np.uint8)


def relative_dark_detail(gray: np.ndarray, background: np.ndarray) -> np.ndarray:
    dark_detail = cv2.subtract(background, gray)
    relative_dark = dark_detail.astype(np.float32) * 255.0 / np.maximum(background.astype(np.float32), 1.0)
    relative_dark = cv2.GaussianBlur(np.clip(relative_dark, 0, 255).astype(np.uint8), (3, 3), 0)
    return relative_dark.astype(np.float32)


def to_foreground_mask(binary: np.ndarray) -> np.ndarray:
    return binary < 128


def keep_components_touching_seed(candidate: np.ndarray, seed: np.ndarray, min_area: int) -> np.ndarray:
    constrained = candidate.astype(bool)
    grown = np.logical_and(seed, constrained).astype(np.uint8)
    if not np.any(grown):
        return np.zeros_like(constrained, dtype=bool)

    candidate_u8 = constrained.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    max_iterations = int(np.clip(min(candidate.shape[:2]) * 0.02, 3, 24))
    for _ in range(max_iterations):
        previous_count = int(grown.sum())
        grown = cv2.bitwise_and(cv2.dilate(grown, kernel, iterations=1), candidate_u8)
        if int(grown.sum()) == previous_count:
            break

    if min_area <= 1:
        return grown.astype(bool)
    if min_area <= 2:
        neighbor_count = cv2.filter2D(
            grown,
            cv2.CV_16U,
            np.ones((3, 3), dtype=np.uint8),
            borderType=cv2.BORDER_CONSTANT,
        )
        return np.logical_and(grown > 0, neighbor_count >= min_area)

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(grown, 8)
    kept = np.zeros_like(constrained, dtype=bool)
    for label in range(1, labels_count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            kept[labels == label] = True
    return kept


def safe_float(value: object, fallback: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    if not np.isfinite(result):
        return fallback
    return result


def cleanup_binary(binary: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    foreground = 255 - binary
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel, iterations=1)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=1)
    return 255 - foreground


def compute_threshold_metrics(
    gray: np.ndarray, illumination_corrected: np.ndarray, enhanced: np.ndarray, binary: np.ndarray
) -> dict[str, float | int | str]:
    foreground = binary < 128
    background = ~foreground
    foreground_count = int(np.count_nonzero(foreground))
    total = int(binary.size)
    if foreground_count == 0 or foreground_count == total:
        contrast = 0.0
    else:
        contrast = float(np.mean(enhanced[background]) - np.mean(enhanced[foreground]))

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats((foreground.astype(np.uint8) * 255), 8)
    small_components = 0
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if 1 <= area <= 8:
            small_components += 1

    before_std = float(np.std(gray))
    after_std = float(np.std(illumination_corrected))
    return {
        "foreground_ratio": round(foreground_count / max(total, 1), 4),
        "text_background_contrast": round(contrast, 2),
        "small_noise_components": small_components,
        "gray_std_before": round(before_std, 2),
        "gray_std_after_correction": round(after_std, 2),
    }


def draw_corner_overlay(image: np.ndarray, corners: np.ndarray, score: float) -> np.ndarray:
    overlay = image.copy()
    pts = corners.astype(np.int32)
    cv2.polylines(overlay, [pts], True, (20, 184, 166), 6)
    for index, point in enumerate(pts):
        cv2.circle(overlay, tuple(point), 12, (0, 117, 255), -1)
        cv2.putText(
            overlay,
            str(index + 1),
            (int(point[0]) + 14, int(point[1]) - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (10, 40, 52),
            3,
            cv2.LINE_AA,
        )
    label = f"score={score:.2f}"
    cv2.rectangle(overlay, (20, 20), (260, 78), (255, 255, 255), -1)
    cv2.putText(overlay, label, (34, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 40, 52), 2, cv2.LINE_AA)
    return overlay
