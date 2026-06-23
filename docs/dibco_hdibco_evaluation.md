# DIBCO/H-DIBCO 二值化客观评测

本文档记录当前项目中 DIBCO2019 Track A、DIBCO2019 Track B 和 H-DIBCO2018 的可复现实验方法。该实验只评估光照校正、文字增强和二值化输出，不评估文档角点定位与透视矫正。

原因是 DIBCO/H-DIBCO 图像本身已经是文档内容图，并提供像素级二值化 GT；它们适合做二值化客观指标，不适合验证手机拍摄场景下的外轮廓检测。角点定位与透视矫正后续应放到 MIDV-500/MIDV-2019 或带四角标注的数据上评估。

## 1. 数据目录

当前脚本默认读取以下目录：

```text
data/raw/dibco2019/
  Dataset/
    1.bmp ... 20.bmp
  GT/
    1.bmp ... 20.bmp

data/raw/hdibco2018/
  dataset/
    1.bmp ... 10.bmp
  gt/
    1_gt.bmp ... 10_gt.bmp
```

DIBCO2019 中 `1-10` 归为 `trackA`，`11-20` 归为 `trackB`。H-DIBCO2018 统一归为 `handwritten`。

原始数据与实验输出都不进入 Git：`data/` 和 `runtime/` 已在 `.gitignore` 中忽略。

## 2. 评测入口

脚本位置：

```bash
python3 experiments/evaluate_dibco.py
```

快速 smoke test：

```bash
python3 experiments/evaluate_dibco.py \
  --limit 2 \
  --out runtime/experiments/dibco_eval_smoke
```

全量基线：

```bash
python3 experiments/evaluate_dibco.py \
  --out runtime/experiments/dibco_eval_v1
```

扩展经典局部阈值方法后的 v2 对比：

```bash
python3 experiments/evaluate_dibco.py \
  --out runtime/experiments/dibco_eval_v2_methods
```

加入 Gatos-like 背景估计和 majority ensemble 后的完整 v3 对比：

```bash
python3 experiments/evaluate_dibco.py \
  --out runtime/experiments/dibco_eval_v3_methods
```

如果只想先跑最终指标，不生成较大的 contact sheet：

```bash
python3 experiments/evaluate_dibco.py \
  --out runtime/experiments/dibco_eval_v3_methods \
  --skip-contact-sheets
```

默认生成高清 contact sheet，每个面板保持 artifact 原始分辨率，并自动排成两行。若只想生成轻量预览，可显式指定缩略宽度：

```bash
python3 experiments/evaluate_dibco.py \
  --contact-cell-width 260 \
  --out runtime/experiments/dibco_eval_preview
```

当前共有 12 个展示面板，默认布局为 `6 x 2`。如需自定义每行列数，可使用：

```bash
python3 experiments/evaluate_dibco.py \
  --contact-columns 4 \
  --out runtime/experiments/dibco_eval_grid4
```

生成少量示例效果图，可使用：

```bash
python3 experiments/evaluate_dibco.py \
  --limit 3 \
  --out runtime/experiments/dibco_eval_v3_examples
```

常用调参示例：

```bash
python3 experiments/evaluate_dibco.py \
  --sauvola-window 51 \
  --sauvola-k 0.18 \
  --fixed-threshold 170 \
  --out runtime/experiments/dibco_eval_sauvola51_k018
```

## 3. 方法设置

脚本复用后端当前增强函数 `enhance_and_binarize`，对每张图生成并评估以下方法：

| 方法 | 含义 |
| --- | --- |
| `binary_fixed` | 对 `text_enhanced` 使用固定阈值，默认 `180`。 |
| `binary_otsu` | 对 `text_enhanced` 使用 Otsu 全局阈值。 |
| `binary_sauvola` | 对 `text_enhanced` 使用 Sauvola 局部阈值，默认窗口 `35`、`k=0.2`。 |
| `binary_niblack` | Niblack 局部阈值，默认作用于原始灰度图，窗口 `35`、`k=-0.2`。 |
| `binary_wolf` | Wolf-Jolion 局部阈值，默认作用于原始灰度图，窗口 `35`、`k=0.5`。 |
| `binary_wolf_fused` | Ours (`binary_wolf_fuse`)：以 Wolf/NICK 的细笔画作为强种子，结合相对暗细节图中的弱候选，只保留与强种子连通的低对比笔画。Contact sheet 中显示为 `Ours (binary_wolf_fuse)`。 |
| `binary_nick` | NICK 局部阈值，默认作用于原始灰度图，窗口 `35`、`k=-0.2`。 |
| `binary_bradley` | Bradley-Roth 积分图局部均值阈值，默认作用于原始灰度图，窗口 `35`、`t=0.15`。 |
| `binary_gatos_like` | Gatos-style 近似实现：Sauvola 初始前景、背景估计、光照归一化、Otsu 二值化和小核清理。 |
| `binary_majority` | 无监督 majority ensemble，默认对 fixed、Otsu、Sauvola、Niblack、Wolf、NICK、Bradley、Gatos-like 这 8 个专家做严格多数投票。 |
| `binary_readable` | 当前项目的可读二值图，基于相对暗文字细节和连通域过滤。 |
| `binary_readable_refined` | `binary_readable` 的笔画细化变体：保留高置信度笔画核心，剥离低置信度边界灰边，并尝试恢复连通域内部低置信白洞，用于缓解笔画偏粗和 `o/a/e` 等字符内孔被填满的问题。 |

`final` 在当前应用中仍指向增强灰度图 `text_enhanced`，因此实验中不把 `final` 当二值化候选。

新增经典局部阈值方法默认使用 `--classic-source raw`，即从原始灰度图直接二值化，作为文献常见传统基线。如果要比较“同一增强结果上的不同阈值公式”，可以改用：

```bash
python3 experiments/evaluate_dibco.py \
  --classic-source enhanced \
  --out runtime/experiments/dibco_eval_v2_enhanced_source
```

## 4. 输出内容

全量运行后会生成：

```text
runtime/experiments/dibco_eval_v1/
  run_config.json
  artifacts/
    dibco2019_trackA_01/
      input.png
      gt.png
      text_enhanced.png
      binary_readable.png
      binary_fixed.png
      binary_otsu.png
      binary_sauvola.png
      binary_niblack.png
      binary_wolf.png
      binary_wolf_fused.png
      binary_nick.png
      binary_bradley.png
      binary_gatos_like.png
      binary_majority.png
      binary_readable_refined.png
  contact_sheets/
    dibco2019_trackA_01_sheet.png
  tables/
    per_image_metrics.csv
    case_summary.csv
    summary_by_method.csv
    summary_by_dataset.csv
```

其中：

- `per_image_metrics.csv`：每张图、每种方法的完整指标。
- `case_summary.csv`：每张图的 F1 最优方法，以及后端增强阶段的可解释指标。
- `summary_by_method.csv`：按方法聚合的总表。
- `summary_by_dataset.csv`：按数据集和 track 聚合的分表。
- `contact_sheets/`：横向拼接的定性检查图，默认按原始分辨率输出，方便放大检查细节。

## 5. 指标口径

GT 统一按黑色前景、白色背景处理，所有预测图也会归一化为 `0/255`。

| 指标 | 方向 | 说明 |
| --- | --- | --- |
| `precision` | 越高越好 | 预测前景中有多少是真前景。 |
| `recall` | 越高越好 | GT 前景中有多少被找回。 |
| `f1` | 越高越好 | Precision 和 Recall 的调和平均。 |
| `pseudo_f1` | 越高越好 | 基于 GT 前景骨架的 pseudo-F，强调笔画主体召回。 |
| `iou` | 越高越好 | 前景区域交并比。 |
| `accuracy` | 越高越好 | 像素级整体准确率，但受大面积背景影响较大。 |
| `psnr` | 越高越好 | 预测二值图与 GT 的像素差异。 |
| `nrm` | 越低越好 | 标准化识别错误率，综合 FNR 和 FPR。 |
| `drd` | 越低越好 | 距离倒数失真，DIBCO 常用失真指标。 |
| `foreground_ratio_error` | 越低越好 | 预测前景比例与 GT 前景比例的绝对差。 |

报告中建议以 `f1`、`pseudo_f1`、`drd` 为主，`psnr`、`nrm`、前景比例误差作为辅助。

## 6. 当前基线结果

运行命令：

```bash
python3 experiments/evaluate_dibco.py \
  --out runtime/experiments/dibco_eval_v1
```

参数：

```text
fixed_threshold=180
sauvola_window=35
sauvola_k=0.2
cleanup_kernel=3
```

总体结果：

| 方法 | F1 ↑ | pseudo-F1 ↑ | IoU ↑ | PSNR ↑ | NRM ↓ | DRD ↓ | F1 最优次数 | DRD 最优次数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `binary_fixed` | 0.6027 | 0.6014 | 0.4501 | 12.6797 | 0.1667 | 3.2992 | 1 | 3 |
| `binary_otsu` | 0.5825 | 0.5866 | 0.4328 | 13.1573 | 0.2106 | 3.7115 | 6 | 2 |
| `binary_sauvola` | 0.6147 | 0.6151 | 0.4605 | 12.6527 | 0.1547 | 3.1922 | 5 | 7 |
| `binary_readable` | 0.6714 | 0.6927 | 0.5234 | 13.3874 | 0.1138 | 2.9263 | 18 | 18 |

分数据集结论：

- DIBCO2019 Track A：`binary_readable` 明显领先，F1 最优 8/10，DRD 最优 10/10。
- DIBCO2019 Track B：`binary_otsu` 的平均 F1 最高，`binary_sauvola` 的平均 DRD 最低，说明全局阈值在部分打印类样例上更锐利，但 Sauvola 的失真更稳。
- H-DIBCO2018：`binary_readable` 平均 F1、pseudo-F1、PSNR 最好；三种传统阈值中 Sauvola 与固定阈值接近，Sauvola 的 DRD 更低。

## 7. 下一步实验

建议按以下顺序扩展，不需要人工打分：

1. 以当前 `dibco_eval_v1` 作为 baseline，保留 `run_config.json` 和四张 CSV。
2. 做 Sauvola 参数扫描：`window in {25, 35, 51, 75}`，`k in {0.15, 0.2, 0.25, 0.3}`。
3. 做固定阈值扫描：`threshold in {150, 165, 180, 195, 210}`。
4. 汇总每组参数的 `mean_f1`、`mean_pseudo_f1`、`mean_drd`，选择一个全局默认参数。
5. 从 `case_summary.csv` 中抽取 F1 最低和方法分歧最大的样例，再看 `contact_sheets/` 做定性解释。
6. 再进入 MIDV-500/MIDV-2019，评估角点定位、透视矫正和移动端拍摄鲁棒性。
