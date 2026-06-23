import json

import cv2

from app.main import get_result, health, parse_params
from app.scanner import ScanParams, process_image_bytes
from app.storage import create_job_dir, save_artifacts
from tests.test_scanner import synthetic_document


def test_scan_pipeline_storage_and_result_response(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DOC_SCAN_RESULT_DIR", str(tmp_path))
    assert health() == {"status": "ok"}
    parsed = parse_params(json.dumps({"sauvola_window": 37}))
    assert parsed.sauvola_window == 37

    ok, encoded = cv2.imencode(".jpg", synthetic_document())
    assert ok
    output = process_image_bytes(encoded.tobytes(), ScanParams.from_mapping(parsed.model_dump()))
    job_id, job_dir = create_job_dir()
    saved = save_artifacts(job_dir, output.artifacts)
    assert saved["final"] == "final.png"
    assert saved["binary_wolf_fused"] == "binary_wolf_fused.png"
    assert output.candidate_score > 0.35

    final_response = get_result(job_id, "final.png")
    assert final_response.media_type == "image/png"
