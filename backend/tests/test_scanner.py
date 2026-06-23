import cv2
import numpy as np

from app.scanner import BINARIZATION_METHOD_KEYS, PROCESSING_PIPELINE_VERSION, ScanParams, ensure_odd, order_points, process_image


def test_ensure_odd_bounds() -> None:
    assert ensure_odd(44, 3, 151) == 45
    assert ensure_odd(2, 3, 151) == 3
    assert ensure_odd(200, 3, 151) == 151


def test_order_points_returns_tl_tr_br_bl() -> None:
    points = np.array([[100, 300], [20, 20], [320, 40], [310, 330]], dtype=np.float32)
    ordered = order_points(points)
    assert ordered.tolist() == [[20.0, 20.0], [320.0, 40.0], [310.0, 330.0], [100.0, 300.0]]


def test_process_image_generates_all_artifacts() -> None:
    image = synthetic_document()
    result = process_image(image, ScanParams())
    expected = {
        "original",
        "edges",
        "corner_detection",
        "rectified",
        "background",
        "illumination_corrected",
        "detail_enhanced",
        "text_enhanced",
        "binary_readable",
        "morphology_enhanced",
        "binary_fixed",
        "binary_otsu",
        "binary_sauvola",
        "binary_niblack",
        "binary_wolf",
        "binary_nick",
        "binary_bradley",
        "binary_wolf_fused",
        "final",
    }
    assert set(result.artifacts.keys()) == expected
    assert set(BINARIZATION_METHOD_KEYS).issubset(result.artifacts)
    assert np.array_equal(result.artifacts["final"], result.artifacts["text_enhanced"])
    assert np.array_equal(result.artifacts["detail_enhanced"], result.artifacts["text_enhanced"])
    assert set(np.unique(result.artifacts["binary_wolf_fused"])).issubset({0, 255})
    assert result.metrics["pipeline_version"] == PROCESSING_PIPELINE_VERSION
    assert result.metrics["final_output"] == "text_enhanced"
    assert result.metrics["readable_text_ratio"] > 0
    assert result.candidate_score > 0.35
    assert result.metrics["output_width"] > 100
    assert result.metrics["foreground_ratio"] > 0


def test_process_image_detects_complex_whiteboard_scene() -> None:
    image, expected_corners = synthetic_whiteboard_scene()
    result = process_image(image, ScanParams())
    actual = order_points(np.array(result.corners, dtype=np.float32))
    expected = order_points(expected_corners.astype(np.float32))
    mean_error = float(np.mean(np.linalg.norm(actual - expected, axis=1)))

    assert result.candidate_score > 0.65
    assert result.metrics["candidate_source"] != "fallback"
    assert mean_error < 90


def test_process_image_detects_projected_slide_content() -> None:
    image, expected_corners = synthetic_projected_slide_scene()
    result = process_image(image, ScanParams())
    actual = order_points(np.array(result.corners, dtype=np.float32))
    expected = order_points(expected_corners.astype(np.float32))
    mean_error = float(np.mean(np.linalg.norm(actual - expected, axis=1)))

    assert result.candidate_score > 0.7
    assert result.metrics["candidate_source"] != "fallback"
    assert mean_error < 100
    assert 0.02 < result.metrics["readable_text_ratio"] < 0.16


def test_process_image_falls_back_without_quad() -> None:
    image = np.full((320, 420, 3), 180, dtype=np.uint8)
    result = process_image(image, ScanParams())
    assert result.candidate_score == 0.0
    assert result.warnings
    assert result.artifacts["final"].shape[:2] == (319, 419)


def synthetic_document() -> np.ndarray:
    canvas = np.zeros((720, 960, 3), dtype=np.uint8)
    canvas[:] = (54, 65, 72)
    document = np.array([[180, 90], [790, 130], [835, 640], [120, 610]], dtype=np.int32)
    cv2.fillConvexPoly(canvas, document, (232, 235, 226))
    for idx in range(9):
        y = 180 + idx * 42
        cv2.line(canvas, (220, y), (730, y + 20), (45, 55, 62), 5, cv2.LINE_AA)
    shadow = np.linspace(0.65, 1.0, canvas.shape[1], dtype=np.float32)
    canvas = np.clip(canvas.astype(np.float32) * shadow[None, :, None], 0, 255).astype(np.uint8)
    return canvas


def synthetic_whiteboard_scene() -> tuple[np.ndarray, np.ndarray]:
    canvas = np.zeros((1350, 1800, 3), dtype=np.uint8)
    canvas[:] = (62, 58, 52)

    for x in (70, 520, 1460, 1630):
        cv2.line(canvas, (x, 0), (x + 55, 1349), (28, 32, 36), 12, cv2.LINE_AA)
    for y in (170, 1185):
        cv2.line(canvas, (0, y), (1799, y + 35), (88, 88, 84), 6, cv2.LINE_AA)
    cv2.rectangle(canvas, (1540, 0), (1799, 1349), (116, 118, 120), -1)

    board = np.array([[150, 210], [1548, 236], [1572, 1110], [130, 1140]], dtype=np.int32)
    cv2.fillConvexPoly(canvas, board, (220, 224, 216))

    border_color = (150, 154, 150)
    cv2.polylines(canvas, [board], True, border_color, 20, cv2.LINE_AA)
    inner = np.array([[177, 245], [1524, 268], [1542, 1078], [160, 1104]], dtype=np.int32)
    cv2.fillConvexPoly(canvas, inner, (224, 228, 220))

    for corner in board:
        cx, cy = int(corner[0]), int(corner[1])
        cv2.rectangle(canvas, (cx - 28, cy - 28), (cx + 92, cy + 92), (148, 151, 148), -1)
        cv2.rectangle(canvas, (cx - 8, cy - 8), (cx + 72, cy + 72), (224, 228, 220), -1)
    cv2.rectangle(canvas, (270, 155), (350, 235), (150, 154, 154), -1)
    cv2.rectangle(canvas, (1320, 165), (1410, 235), (95, 98, 98), -1)

    for row in range(4):
        y = 330 + row * 190
        for col in range(5):
            x = 250 + col * 235
            cv2.line(canvas, (x, y), (x + 105, y + 12), (42, 44, 42), 5, cv2.LINE_AA)
            cv2.line(canvas, (x + 10, y + 28), (x + 85, y + 95), (42, 44, 42), 4, cv2.LINE_AA)
    cv2.line(canvas, (420, 760), (840, 790), (42, 44, 42), 6, cv2.LINE_AA)
    cv2.line(canvas, (1190, 930), (1480, 950), (42, 44, 42), 6, cv2.LINE_AA)

    shadow = np.linspace(0.78, 1.0, canvas.shape[1], dtype=np.float32)
    canvas = np.clip(canvas.astype(np.float32) * shadow[None, :, None], 0, 255).astype(np.uint8)
    return canvas, board.astype(np.float32)


def synthetic_projected_slide_scene() -> tuple[np.ndarray, np.ndarray]:
    canvas = np.zeros((900, 1280, 3), dtype=np.uint8)
    canvas[:] = (54, 56, 55)

    cv2.rectangle(canvas, (40, 125), (1240, 185), (202, 204, 202), -1)
    cv2.rectangle(canvas, (60, 860), (1230, 895), (32, 32, 32), -1)

    slide = np.array([[78, 220], [1200, 238], [1216, 815], [72, 852]], dtype=np.int32)
    cv2.fillConvexPoly(canvas, slide, (217, 229, 220))
    cv2.polylines(canvas, [slide], True, (42, 45, 45), 7, cv2.LINE_AA)

    cv2.putText(canvas, "Subjective Question   10 pts", (115, 285), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (105, 112, 110), 4, cv2.LINE_AA)
    cv2.putText(canvas, "Read the paragraph and find mistakes.", (115, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (105, 112, 110), 3, cv2.LINE_AA)
    y = 410
    lines = [
        "In our study, we aimed to explore artificial intelligence in smart education.",
        "We collected data from three different online learning platforms.",
        "Each platform has unique features and a user interface.",
        "To analyze the data, we used a deep learning algorithm.",
        "The training process took approximately two weeks.",
        "Our study aims to fill this gap by providing detailed analysis.",
    ]
    for line in lines:
        cv2.putText(canvas, line, (120, y), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (92, 98, 96), 2, cv2.LINE_AA)
        y += 58

    gradient = np.linspace(0.9, 1.05, canvas.shape[1], dtype=np.float32)
    canvas = np.clip(canvas.astype(np.float32) * gradient[None, :, None], 0, 255).astype(np.uint8)
    return canvas, slide.astype(np.float32)
