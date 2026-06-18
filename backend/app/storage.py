from __future__ import annotations

import os
import tempfile
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np


ARTIFACT_ORDER = (
    "original",
    "edges",
    "corner_detection",
    "rectified",
    "background",
    "illumination_corrected",
    "morphology_enhanced",
    "binary_fixed",
    "binary_otsu",
    "binary_sauvola",
    "final",
)


def result_root() -> Path:
    configured = os.environ.get("DOC_SCAN_RESULT_DIR")
    root = Path(configured) if configured else Path(tempfile.gettempdir()) / "doc_scan_results"
    root.mkdir(parents=True, exist_ok=True)
    return root


def create_job_dir() -> tuple[str, Path]:
    job_id = uuid4().hex
    job_dir = result_root() / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    return job_id, job_dir


def save_artifacts(job_dir: Path, artifacts: dict[str, np.ndarray]) -> dict[str, str]:
    saved: dict[str, str] = {}
    for name in ARTIFACT_ORDER:
        image = artifacts[name]
        path = job_dir / f"{name}.png"
        ok = cv2.imwrite(str(path), image)
        if not ok:
            raise RuntimeError(f"Failed to save artifact: {name}")
        saved[name] = path.name
    return saved


def artifact_path(job_id: str, artifact: str) -> Path:
    safe_job_id = "".join(ch for ch in job_id if ch.isalnum())
    safe_artifact = Path(artifact).name
    if not safe_artifact.endswith(".png"):
        safe_artifact = f"{safe_artifact}.png"
    return result_root() / safe_job_id / safe_artifact
