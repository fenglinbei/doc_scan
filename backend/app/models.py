from pydantic import BaseModel, Field


class ScanParamsModel(BaseModel):
    canny_low: int = Field(default=50, ge=1, le=255)
    canny_high: int = Field(default=150, ge=1, le=255)
    fixed_threshold: int = Field(default=180, ge=1, le=254)
    illumination_kernel: int = Field(default=45, ge=3, le=151)
    sauvola_window: int = Field(default=35, ge=3, le=151)
    sauvola_k: float = Field(default=0.2, ge=0.0, le=0.8)
    cleanup_kernel: int = Field(default=3, ge=1, le=21)


class ArtifactUrls(BaseModel):
    original: str
    edges: str
    corner_detection: str
    rectified: str
    background: str
    illumination_corrected: str
    morphology_enhanced: str
    binary_fixed: str
    binary_otsu: str
    binary_sauvola: str
    final: str


class ScanResultModel(BaseModel):
    job_id: str
    corners: list[list[float]]
    candidate_score: float
    warnings: list[str]
    metrics: dict[str, float | int | str]
    artifacts: ArtifactUrls
