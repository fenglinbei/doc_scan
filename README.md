# Document Scan Lab

基于传统计算机视觉的移动端文档扫描验证原型。项目实现了 OpenCV 文档角点定位、透视矫正、形态学光照校正，以及固定阈值、Otsu、Sauvola 三种二值化对比；同时提供一个移动端优先的 Web 界面，用于手机拍照上传并同步查看中间结果。

## Project Layout

```text
backend/   FastAPI API, OpenCV/scikit-image scanning pipeline, tests
frontend/  React + Vite + TypeScript mobile-first verification UI
docs/      Project design document and UI concept image
experiments/  Reproducible dataset evaluation scripts
```

## Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API:

- `GET /api/health`
- `POST /api/scan`
- `GET /api/results/{job_id}/{artifact}`

## Frontend

```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0
```

The Vite dev server proxies `/api` to `http://127.0.0.1:8000`. On a phone in the same LAN, open the Vite URL shown by the dev server and use the camera upload button.

## Tests

```bash
cd backend
pytest

cd frontend
npm run build
```

## DIBCO/H-DIBCO Evaluation

Downloaded datasets are expected under `data/raw/`, which is intentionally ignored by Git. Run the objective binarization benchmark with:

```bash
python3 experiments/evaluate_dibco.py --out runtime/experiments/dibco_eval_v1
```

The script writes per-image artifacts, contact sheets, and CSV summaries under `runtime/experiments/`. See `docs/dibco_hdibco_evaluation.md` for the dataset layout, metrics, and current baseline results.

## MIDV Evaluation

MIDV uses mobile-captured frames with document-boundary quadrangle ground truth, so it evaluates corner detection and perspective robustness rather than DIBCO-style pixel binarization accuracy.

```bash
python3 experiments/evaluate_midv.py --datasets midv500 --frames-per-video 1 --out runtime/experiments/midv_eval_available_fpv1
```

See `docs/midv_evaluation.md` for the MIDV data status, metrics, commands, and current lightweight baseline.

## Constraints

- No deep learning framework.
- No pretrained model.
- No OCR dependency in the core pipeline.
- Results are temporary files under the system temp directory by default.
