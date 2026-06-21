export type ScanParams = {
  canny_low: number;
  canny_high: number;
  fixed_threshold: number;
  illumination_kernel: number;
  sauvola_window: number;
  sauvola_k: number;
  cleanup_kernel: number;
};

export type ArtifactKey =
  | "original"
  | "edges"
  | "corner_detection"
  | "rectified"
  | "background"
  | "illumination_corrected"
  | "detail_enhanced"
  | "text_enhanced"
  | "binary_readable"
  | "morphology_enhanced"
  | "binary_fixed"
  | "binary_otsu"
  | "binary_sauvola"
  | "final";

export type ScanResult = {
  job_id: string;
  corners: number[][];
  candidate_score: number;
  warnings: string[];
  metrics: Record<string, number | string>;
  artifacts: Record<ArtifactKey, string>;
};

export type ResultItem = {
  key: ArtifactKey;
  label: string;
  description: string;
};

export type TaskPhase =
  | "idle"
  | "ready"
  | "preparing"
  | "uploading"
  | "processing"
  | "receiving"
  | "done"
  | "error";

export type ScanTask = {
  phase: TaskPhase;
  progress: number;
  message: string;
  artifactDone: number;
  artifactTotal: number;
};

export type ArtifactStatus = {
  sourceUrl: string;
  objectUrl: string | null;
  progress: number;
  status: "idle" | "loading" | "done" | "error";
  error: string | null;
};

export type PreparedUpload = {
  file: File;
  originalBytes: number;
  uploadBytes: number;
  wasCompressed: boolean;
  width: number;
  height: number;
};
