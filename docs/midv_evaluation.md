# MIDV document-boundary and robustness evaluation

This document records how to evaluate the current mobile document scan pipeline on MIDV-style datasets.

MIDV is not a DIBCO replacement. DIBCO/H-DIBCO provides pixel-level binarization ground truth, so it can report F1, pseudo-F1, PSNR, NRM, and DRD for binary document images. MIDV provides mobile-captured identity-document frames with frame-level document quadrangles. Therefore, the primary MIDV metrics should evaluate document boundary detection and perspective robustness.

## 1. Current local data status

The local checkout currently has:

```text
data/raw/midv500/
  01_alb_id.zip
  09_chn_id.zip
  48_usa_passportcard.zip
  readme.txt
  license.txt
  md5.txt

data/raw/midv2019/
  empty
```

The three MIDV-500 zip files above match the MD5 hashes listed in `md5.txt`. This is enough for a small closed-loop evaluation, but it is not the full MIDV-500 benchmark. Full MIDV-500 has 50 document types, 10 videos per document type, and 30 extracted frames per video.

## 2. Evaluation entry

Script:

```bash
python3 experiments/evaluate_midv.py
```

The script reads MIDV zip archives directly; it does not require unpacking the large files first.

Smoke test:

```bash
python3 experiments/evaluate_midv.py \
  --datasets midv500 \
  --documents 01_alb_id \
  --conditions TA \
  --frames-per-video 1 \
  --limit 1 \
  --out runtime/experiments/midv_eval_smoke
```

Current lightweight local run, using the available three MIDV-500 document types and one frame from each condition/device video:

```bash
python3 experiments/evaluate_midv.py \
  --datasets midv500 \
  --frames-per-video 1 \
  --skip-artifacts \
  --out runtime/experiments/midv_eval_available_fpv1
```

Default sampled run, using three frames per condition/device video:

```bash
python3 experiments/evaluate_midv.py \
  --datasets midv500 \
  --frames-per-video 3 \
  --artifact-limit 24 \
  --out runtime/experiments/midv_eval_available_fpv3
```

All local frames from the currently downloaded three documents:

```bash
python3 experiments/evaluate_midv.py \
  --datasets midv500 \
  --frames-per-video 0 \
  --skip-artifacts \
  --out runtime/experiments/midv_eval_available_all_frames
```

After all 50 MIDV-500 zips are present, the same command becomes the full MIDV-500 geometry benchmark. Expect it to be much slower: full MIDV-500 is 15,000 frames.

## 3. Dataset structure expected by the script

For each document type archive:

```text
XX_DOC_TYPE.zip
  XX_DOC_TYPE/
    images/
      XX_DOC_TYPE.tif
      TA/
        TAXX_01.tif ... TAXX_30.tif
      ...
    ground_truth/
      XX_DOC_TYPE.json
      TA/
        TAXX_01.json ... TAXX_30.json
      ...
```

The frame-level JSON has:

```json
{
  "quad": [[97, 672], [904, 643], [931, 1142], [122, 1185]]
}
```

The quadrangle points are the document boundary in the captured frame. The script compares this GT quadrangle with the current `process_image` detected corners.

## 4. Primary metrics

| Metric | Direction | Meaning |
| --- | ---: | --- |
| `polygon_iou` | higher | IoU between predicted document polygon and MIDV GT polygon after rasterization into the frame. |
| `corner_mean_error_px` | lower | Mean Euclidean distance between corresponding predicted and GT corners. |
| `corner_mean_error_norm` | lower | Mean corner error divided by the image diagonal, useful across resolutions. |
| `success_iou_80` | higher | 1 if `polygon_iou >= 0.80`, else 0. |
| `success_iou_90` | higher | 1 if `polygon_iou >= 0.90`, else 0. |
| `success_corner_2pct` | higher | 1 if normalized mean corner error is at most 2% of the image diagonal. |
| `success_corner_5pct` | higher | 1 if normalized mean corner error is at most 5% of the image diagonal. |
| `fallback_rate` | lower | Share of frames where no candidate was found and the whole image boundary was used. |
| `gt_visible_ratio` | context | Fraction of the GT document polygon visible inside the frame. Partial shots can legitimately have low values. |

Recommended headline metrics for MIDV:

1. `median_polygon_iou`
2. `success_iou_90_rate`
3. `success_corner_2pct_rate`
4. `fallback_rate`

`mean_polygon_iou` is still useful, but partial or failed frames can pull it down sharply.

## 5. Secondary method diagnostics

The script uses the same backend `process_image` output as the API. This means MIDV geometry metrics, rectified images, and binarization diagnostics all come from the v2.7 backend processing line, including folded-corner refinement.

- `binary_fixed`
- `binary_otsu`
- `binary_sauvola`
- `binary_niblack`
- `binary_wolf`
- `binary_wolf_fused`
- `binary_nick`
- `binary_bradley`
- `binary_readable`

Because MIDV does not provide pixel-level binary GT, these rows are diagnostics rather than accuracy scores. Use them to compare stability across real mobile captures:

| Metric | Meaning |
| --- | --- |
| `foreground_ratio` | Share of black foreground pixels. Large shifts by condition indicate threshold instability. |
| `component_count` | Connected foreground component count. |
| `small_component_count` | Tiny foreground components, a rough noise proxy. |
| `largest_component_ratio` | Largest foreground component size divided by the image size. Large values can signal merged borders or blobs. |

For true binary accuracy, keep using `experiments/evaluate_dibco.py`.

## 6. Output files

Each run writes:

```text
runtime/experiments/midv_eval_.../
  run_config.json
  tables/
    per_frame_metrics.csv
    summary_overall.csv
    summary_by_dataset.csv
    summary_by_document.csv
    summary_by_condition.csv
    method_diagnostics.csv
    summary_by_method.csv
    summary_by_condition_method.csv
  artifacts/
    midv500_01_alb_id_TA01_01/
      original.png
      gt_pred_overlay.png
      rectified.png
      text_enhanced.png
      binary_readable.png
      binary_fixed.png
      binary_otsu.png
      binary_sauvola.png
      binary_wolf.png
      binary_wolf_fused.png
      contact_sheet.png
```

In `gt_pred_overlay.png`, the MIDV GT quadrangle is magenta and the current algorithm prediction is teal.

## 7. Current lightweight baseline

Command:

```bash
python3 experiments/evaluate_midv.py \
  --datasets midv500 \
  --frames-per-video 1 \
  --skip-artifacts \
  --out runtime/experiments/midv_eval_v27_available_fpv1
```

Scope: the currently downloaded 3 document types, 10 condition/device videos each, first frame per video, 30 frames total. Current numbers are from the v2.7 detector/scoring path.

Overall:

| Cases | Mean IoU | Median IoU | Mean corner error | IoU >= 0.90 | Corner <= 2% diag | Fallback rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 0.6773 | 0.9328 | 244.92 px | 53.33% | 53.33% | 0.00% |

Condition-level read:

- Strong: `KS`, `TS`, `TA`, `HA`, and `KA` are stable in this small run.
- Mixed: `CA`, `CS`, `HS`, and `PA` still have document-type-specific failures, but v2.7 reduces several large-background false positives in clutter/partial scenes.
- Weak: `PS` remains the hardest because MIDV partial frames can have little visible target area and corners outside the image; the current candidate generators often do not propose the true document boundary there.

Treat these numbers as a smoke baseline, not a publishable result. Increase `--frames-per-video`, add the remaining MIDV-500 document zips, and run without `--skip-artifacts` for selected failure cases when tuning the detector.
