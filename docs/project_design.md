# 基于形态学增强的文档图像智能扫描与矫正项目设计

## 1. 项目目标

本项目实现一个不依赖深度学习框架或预训练模型的文档扫描验证系统，面向手机拍摄的实验报告、试卷、书籍页面和公开文档数据集图像。系统需要完成文档边界自动定位、透视矫正、阴影与光照不均校正、文字增强、三种二值化方法对比，并通过移动端 Web 界面展示每一步中间结果。

核心交付物：

- OpenCV/scikit-image 传统图像处理流水线。
- FastAPI 后端接口。
- React + Vite + TypeScript 移动端验证界面。
- 公开集为主、自建集补充的实验设计。

不纳入当前版本：

- 深度学习检测、分割、去阴影或去弯曲模型。
- OCR 作为核心算法。
- 连续视频流实时扫描。
- 多用户任务管理或数据库历史记录。

## 2. 总体架构

```text
手机浏览器
  -> 拍照/相册上传
  -> React 参数面板与结果浏览
  -> POST /api/scan
  -> FastAPI
  -> OpenCV/scikit-image 全流程处理
  -> 临时结果目录
  -> GET /api/results/{job_id}/{artifact}
```

前端只负责采集、参数提交、状态展示和中间图浏览；后端负责所有图像处理与结果落盘。结果采用临时文件，不引入数据库，便于课程项目部署和复现实验。

## 3. 算法流程

### 3.1 输入预处理

1. 读取上传图片并解码为 BGR 图像。
2. 长边超过 1400 像素时缩放副本用于边界检测，保留缩放比例。
3. 转灰度，使用 `GaussianBlur(5x5)` 抑制噪声。

### 3.2 文档角点定位

1. 使用 Canny 边缘检测，默认阈值为 `50/150`。
2. 使用 `5x5`、水平长核、垂直长核分别连接边缘，再合并为检测边缘图，提升白板/黑板长边连续性。
3. 从三类来源生成候选四边形：
   - 轮廓候选：通过 `findContours(RETR_LIST)` 提取候选，允许多组 `approxPolyDP` epsilon，并对 5-16 点凸轮廓提取四个极值角点。
   - 亮色平面候选：结合 HSV 饱和度、亮度和 Lab 亮度提取低饱和高亮区域，经过形态学闭/开运算后拟合四边形。
   - Hough 长边候选：从长水平/垂直线段聚类出上、下、左、右四组边线，求交点作为轮廓法失败时的兜底候选。
4. 对候选做基础过滤：
   - 凸四边形。
   - 面积占比在 `0.055-0.98` 之间。
   - 角点不能明显越出图像边界。
   - 边长不能过短。

### 3.3 候选四边形综合评分

候选四边形不是简单取最大面积，而是综合评分：

```text
score =
  0.22 * area_score +
  0.16 * angle_score +
  0.12 * edge_score +
  0.22 * side_score +
  0.10 * contrast_score +
  0.08 * surface_score +
  0.06 * margin_score +
  0.04 * aspect_score
```

各项含义：

- `area_score`：文档区域面积占比，避免小噪声轮廓。
- `angle_score`：四个夹角接近 90 度的程度。
- `edge_score`：候选边框与 Canny 边缘的重合程度。
- `side_score`：四条边分别采样后的边缘覆盖率，避免只有局部边框命中的候选得高分。
- `contrast_score`：候选内部与外部扩张带的平均灰度差异。
- `surface_score`：候选中心区域的亮度和平滑度，提升白板/纸张类大平面得分。
- `margin_score`：候选是否过度贴近整图边界，用于压低整图回退式误检。
- `aspect_score`：弱宽高比先验，降低极端狭长候选。

当无可靠候选时，系统回退到整张图片边界，并返回 warning，保证接口仍可输出完整中间结果。

### 3.4 透视矫正

1. 使用四点排序得到左上、右上、右下、左下。
2. 根据上下边最大宽度、左右边最大高度确定输出尺寸。
3. 使用 `getPerspectiveTransform` 和 `warpPerspective` 输出正视图。

该方法假设文档近似平面。对于弯曲书页，单应性透视变换只能校正整体视角，无法消除页面局部弯曲。

### 3.5 保细节增强与形态学增强

1. 对矫正图转灰度。
2. 保细节分支直接基于矫正灰度图做轻量百分位拉伸、轻锐化和双边滤波，输出 `detail_enhanced`。该分支不做强二值化，也不做会吞掉细笔画的形态学开运算，面向白板、手写和 OCR 阅读。
3. 形态学分支使用大核闭运算估计背景光照场，默认核大小 `45x45`。
4. 用 `gray / background * 255` 做除法归一化，降低阴影影响。
5. 对归一化图像执行：
   - 顶帽变换，增强局部亮细节。
   - 底帽变换，提取暗文字和阴影结构。
6. 使用 `illumination_corrected - blackhat + tophat` 得到 `morphology_enhanced`。

默认 `final` 指向 `detail_enhanced`，强二值化结果作为对照 artifact 保留。

### 3.6 三种二值化对比

同一张形态学增强图上输出三种结果：

- 固定阈值：默认 `180`，用于展示简单全局阈值的局限。
- Otsu：自动全局阈值，适合直方图接近双峰的图像。
- Sauvola：局部自适应阈值，默认窗口 `35`、`k=0.2`。

后处理统一采用小核开运算去除小噪声、小核闭运算连接断笔。由于输出是白底黑字，后处理先反相为黑字前景，再恢复为白底黑字。

## 4. 后端接口设计

### 4.1 `GET /api/health`

返回：

```json
{"status": "ok"}
```

### 4.2 `POST /api/scan`

请求类型：`multipart/form-data`

字段：

- `file`：图片文件。
- `params`：可选 JSON 字符串。

默认参数：

```json
{
  "canny_low": 50,
  "canny_high": 150,
  "fixed_threshold": 180,
  "illumination_kernel": 45,
  "sauvola_window": 35,
  "sauvola_k": 0.2,
  "cleanup_kernel": 3
}
```

响应：

```json
{
  "job_id": "string",
  "corners": [[0, 0], [100, 0], [100, 200], [0, 200]],
  "candidate_score": 0.82,
  "warnings": [],
  "metrics": {
    "candidate_count": 4,
    "foreground_ratio": 0.14,
    "text_background_contrast": 96.2
  },
  "artifacts": {
    "original": "/api/results/{job_id}/original.png",
    "edges": "/api/results/{job_id}/edges.png",
    "corner_detection": "/api/results/{job_id}/corner_detection.png",
    "rectified": "/api/results/{job_id}/rectified.png",
    "background": "/api/results/{job_id}/background.png",
    "illumination_corrected": "/api/results/{job_id}/illumination_corrected.png",
    "detail_enhanced": "/api/results/{job_id}/detail_enhanced.png",
    "morphology_enhanced": "/api/results/{job_id}/morphology_enhanced.png",
    "binary_fixed": "/api/results/{job_id}/binary_fixed.png",
    "binary_otsu": "/api/results/{job_id}/binary_otsu.png",
    "binary_sauvola": "/api/results/{job_id}/binary_sauvola.png",
    "final": "/api/results/{job_id}/final.png"
  }
}
```

### 4.3 `GET /api/results/{job_id}/{artifact}`

读取临时 PNG 结果。`artifact` 支持上方 artifacts 中的所有名称。不存在时返回 404。

## 5. 移动端 Web 界面设计

界面采用“移动端实验验证台”方向，强调可解释过程，不做营销页。概念图保存在：

```text
docs/assets/mobile-ui-concept.png
```

核心区域：

- 顶部标题：`文档扫描验证`。
- 图片输入区：同时提供“拍照”和“从相册选择”两个入口，避免移动端只能调用相机。
- 基础参数面板：Canny、形态学核大小、Sauvola 窗口/k 值、固定阈值。
- 状态条：待选择、待处理、上传中、处理中、结果回传中、完成、失败，并显示当前阶段进度。
- 中间结果网格：角点检测、透视矫正、保细节增强、形态学增强、固定阈值、Otsu、Sauvola、最终结果。
- 大图预览：点击任意结果缩略图进入检查模式。

桌面端使用双列布局，左侧上传和参数，右侧结果；移动端按单列纵向布局，底部操作区保持易触达。

### 5.1 处理进度与结果回传进度

点击“处理”后，前端创建一次本地 `scanTask` 状态，包含 `phase`、`progress`、`message`、`error` 和 `artifactProgress`。按钮进入禁用或取消状态，避免用户在同一张图片处理中重复提交。

进度条按阶段合成：

| 阶段 | 进度范围 | 前端行为 |
| --- | --- | --- |
| 图片准备 | `0%-5%` | 校验文件类型和大小，读取预览图，修正 EXIF 方向，必要时进行客户端压缩。 |
| 上传图片 | `5%-35%` | 使用 `XMLHttpRequest.upload.onprogress` 读取真实上传进度；上传慢时展示已上传比例。 |
| 后端处理 | `35%-75%` | 请求已发出但未返回时展示处理阶段提示，如边界定位、透视矫正、光照校正、二值化、整理结果。当前同步接口无法拿到真实算法子步骤，先用阶段式进度；如果后续改成异步任务，可通过轮询或 SSE 读取后端真实进度。 |
| 结果回传 | `75%-100%` | 收到 `artifacts` 后，前端按缩略图优先级预加载结果图，显示“已回传 n/m”；支持 `Content-Length` 时叠加字节级进度。 |

结果回传期间，中间结果网格逐项显示 loading 占位；已加载的缩略图立即可点开，未加载项继续显示独立进度。网络失败时保留已加载结果，并提供“重试加载结果”按钮，只重新请求失败的 artifact。

### 5.2 移动端图片来源

移动端图片输入拆成两个明确按钮：

- 拍照：`<input type="file" accept="image/*" capture="environment">`，优先调用后置相机。
- 从相册选择：`<input type="file" accept="image/*">`，不设置 `capture`，允许 iOS/Android 浏览器打开相册或文件选择器。

两个入口共用同一套文件校验、预览、压缩和提交逻辑。选择新图片时重置上一次的结果、warning 和进度；如果用户只修改参数并重新处理，则复用当前已选图片。

### 5.3 网络传输优化

由于手机局域网、校园网或隧道环境可能导致上传和结果回传较慢，前端优先做轻量传输优化：

- 上传前按最长边限制压缩到约 `1600-2000px`，保留原始长宽比例；课程验证需要的是文档边界和文字可读性，不需要上传手机原始超大图。
- 对拍照得到的 JPEG 使用 `canvas.toBlob("image/jpeg", 0.86)` 生成提交图，避免 8-12MB 原图直接上传。
- 提交参数中记录压缩后的尺寸，后端 metrics 也返回输出尺寸，便于实验说明压缩对效果的影响。
- 结果图先加载小尺寸预览或关键结果：`corner_detection`、`rectified`、`detail_enhanced`、`final` 优先，其余 artifact 延迟加载。
- 大图只在用户点击预览时再拉取，避免一次性下载全部高分辨率 PNG。
- 后端可补充缩略图 artifact，例如 `thumbnail_final.webp`，前端网格优先使用缩略图，检查模式再加载原 PNG。
- 对重复点击处理的同一图片和同一参数，前端可用 `fileHash + paramsHash` 做本地结果缓存，避免重复上传和重复下载。

## 6. 数据集与实验设计

### 6.1 公开数据集为主

- MIDV-500/MIDV-2019：用于移动端拍摄文档定位、透视矫正和复杂背景稳健性验证。
- B-MOD：用于手机拍摄低质量文档、文字可读性和 OCR 前处理相关评估。
- DIBCO/H-DIBCO：用于二值化方法对照，尤其是 Otsu 与 Sauvola 的差异分析。

### 6.2 自建数据集补充

保留 10-20 张手机拍摄图片，满足题目要求并覆盖真实课程场景：

- 实验报告。
- 试卷。
- 书籍页面。
- 均匀光照、单侧阴影、强透视、复杂桌面背景、低对比。

建议文件组织：

```text
data/
  public/
  self_captured/
    raw/
    annotations/
```

自建集至少标注四角点，用于计算角点误差或四边形 IoU。

## 7. 评价指标

- 角点定位成功率：是否找到可接受文档四边形。
- 四边形 IoU：预测四边形与标注四边形重叠程度。
- 角点平均误差：四个角点到标注点的平均像素距离。
- 前景比例：二值图中文字前景占比是否异常。
- 小噪声连通域数量：反映二值化噪点。
- 文字/背景对比度：增强图中背景与文字区域的平均灰度差。
- 中间结果可视化：展示角点检测、光照校正、形态学增强和最终二值图。

## 8. 测试计划

单元测试：

- 四点排序稳定性。
- 透视变换尺寸计算。
- 参数奇数化和边界裁剪。
- 无四边形候选时的 fallback 和 warning。

集成测试：

- 上传样例图后生成全部中间结果。
- 阴影样例输出三种二值化对照。
- 参数变化能影响返回结果并写入 metrics。

移动端验证：

- 同一局域网手机访问 Vite 服务。
- 拍照上传成功。
- 从相册选择图片成功，且不会强制打开相机。
- 点击处理后能看到图片准备、上传、处理中、结果回传四段进度。
- 慢速网络下能显示上传比例和 artifact 回传数量，失败的结果图可以单独重试。
- 上传前压缩后的图片仍能完成边界定位、矫正和二值化展示。
- 参数面板可修改并重新处理。
- 点击缩略图可查看大图。
- 非图片、超大图片、模糊图、边界缺失图能得到可读错误或 warning。

## 9. 运行与部署

后端：

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

前端：

```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0
```

手机测试时，电脑和手机需在同一局域网。浏览器打开 Vite 输出的局域网地址即可拍照上传。

## 10. 参考资料

- OpenCV Canny: https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
- OpenCV morphology: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
- OpenCV thresholding: https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
- OpenCV perspective transform: https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html
- scikit-image Sauvola: https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_sauvola
- MIDV-500: https://arxiv.org/abs/1807.05786
- B-MOD: https://arxiv.org/abs/1907.01307
- UVDoc: https://github.com/tanguymagne/UVDoc
