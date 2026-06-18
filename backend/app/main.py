from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .models import ArtifactUrls, ScanParamsModel, ScanResultModel
from .scanner import ScanParams, process_image_bytes
from .storage import artifact_path, create_job_dir, save_artifacts


MAX_UPLOAD_BYTES = 12 * 1024 * 1024

app = FastAPI(title="Document Scan Lab API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/scan", response_model=ScanResultModel)
async def scan_document(file: UploadFile = File(...), params: str | None = Form(default=None)) -> ScanResultModel:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image uploads are supported.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 12MB.")

    parsed_params = parse_params(params)
    try:
        output = process_image_bytes(image_bytes, ScanParams.from_mapping(parsed_params.model_dump()))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - keeps user-facing API stable on unexpected CV errors.
        raise HTTPException(status_code=500, detail=f"Image processing failed: {exc}") from exc

    job_id, job_dir = create_job_dir()
    saved = save_artifacts(job_dir, output.artifacts)
    urls = {name: f"/api/results/{job_id}/{filename}" for name, filename in saved.items()}

    return ScanResultModel(
        job_id=job_id,
        corners=output.corners,
        candidate_score=output.candidate_score,
        warnings=output.warnings,
        metrics=output.metrics,
        artifacts=ArtifactUrls(**urls),
    )


@app.get("/api/results/{job_id}/{artifact}")
def get_result(job_id: str, artifact: str) -> FileResponse:
    path = artifact_path(job_id, artifact)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Result artifact not found.")
    return FileResponse(path, media_type="image/png", filename=Path(path).name)


def parse_params(params: str | None) -> ScanParamsModel:
    if not params:
        return ScanParamsModel()
    try:
        data = json.loads(params)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="params must be valid JSON.") from exc
    return ScanParamsModel(**data)
