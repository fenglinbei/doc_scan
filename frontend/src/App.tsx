import { useEffect, useMemo, useRef, useState } from "react";
import type { Dispatch, MutableRefObject, SetStateAction } from "react";
import { fetchArtifactObjectUrl, scanDocument } from "./api";
import { ImageLightbox } from "./components/ImageLightbox";
import { MetricsPanel } from "./components/MetricsPanel";
import { ParameterPanel } from "./components/ParameterPanel";
import { ResultGallery } from "./components/ResultGallery";
import { StatusStrip } from "./components/StatusStrip";
import { UploadPanel } from "./components/UploadPanel";
import { prepareUploadImage } from "./imageProcessing";
import type { ArtifactKey, ArtifactStatus, ResultItem, ScanParams, ScanResult, ScanTask, TaskPhase } from "./types";

const DEFAULT_PARAMS: ScanParams = {
  canny_low: 50,
  canny_high: 150,
  fixed_threshold: 180,
  illumination_kernel: 45,
  sauvola_window: 35,
  sauvola_k: 0.2,
  cleanup_kernel: 3
};

const RESULT_ITEMS: ResultItem[] = [
  { key: "corner_detection", label: "角点检测", description: "候选四边形与角点编号" },
  { key: "rectified", label: "透视矫正", description: "四点单应性变换后的正视图" },
  { key: "background", label: "背景估计", description: "大核闭运算得到的光照场" },
  { key: "text_enhanced", label: "低对比增强", description: "统一增强文档与 PPT 的灰度文字细节" },
  { key: "binary_readable", label: "可读二值化", description: "基于暗文字细节的黑白对照结果" },
  { key: "morphology_enhanced", label: "增强灰度", description: "兼容保留的文字增强灰度图" },
  { key: "binary_fixed", label: "固定阈值", description: "全局固定阈值二值化" },
  { key: "binary_otsu", label: "Otsu", description: "自动全局阈值二值化" },
  { key: "binary_sauvola", label: "Sauvola", description: "局部自适应阈值二值化" },
  { key: "final", label: "最终结果", description: "默认采用低对比文字增强结果" }
];

const PRIMARY_ARTIFACTS: ArtifactKey[] = ["corner_detection", "rectified", "text_enhanced", "final"];
const ALL_ARTIFACTS = RESULT_ITEMS.map((item) => item.key);
const SECONDARY_ARTIFACTS = ALL_ARTIFACTS.filter((key) => !PRIMARY_ARTIFACTS.includes(key));
const BUSY_PHASES: TaskPhase[] = ["preparing", "uploading", "processing", "receiving"];
const INITIAL_TASK: ScanTask = {
  phase: "idle",
  progress: 0,
  message: "等待手机拍照或从相册选择文档图片",
  artifactDone: 0,
  artifactTotal: RESULT_ITEMS.length
};

const PROCESSING_MESSAGES = ["正在定位文档边界", "正在做透视矫正", "正在增强低对比文字", "正在生成可读二值化对照"];

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [params, setParams] = useState<ScanParams>(DEFAULT_PARAMS);
  const [task, setTask] = useState<ScanTask>(INITIAL_TASK);
  const [result, setResult] = useState<ScanResult | null>(null);
  const [artifactStatuses, setArtifactStatuses] = useState<Partial<Record<ArtifactKey, ArtifactStatus>>>({});
  const [selectedItem, setSelectedItem] = useState<ResultItem | null>(null);
  const artifactUrlsRef = useRef<string[]>([]);
  const processingTimerRef = useRef<number | null>(null);
  const runIdRef = useRef(0);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
      clearProcessingTimer(processingTimerRef);
      revokeArtifactObjectUrls(artifactUrlsRef);
    };
  }, [previewUrl]);

  const isBusy = BUSY_PHASES.includes(task.phase);
  const canRun = useMemo(() => Boolean(file) && !isBusy, [file, isBusy]);

  function handleFileChange(nextFile: File) {
    runIdRef.current += 1;
    clearProcessingTimer(processingTimerRef);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    revokeArtifactObjectUrls(artifactUrlsRef);
    setFile(nextFile);
    setPreviewUrl(URL.createObjectURL(nextFile));
    setResult(null);
    setArtifactStatuses({});
    setSelectedItem(null);
    setTask({
      phase: "ready",
      progress: 0,
      message: "图片已就绪，可以开始处理",
      artifactDone: 0,
      artifactTotal: RESULT_ITEMS.length
    });
  }

  async function handleRun() {
    if (!file) {
      return;
    }

    const runId = runIdRef.current + 1;
    runIdRef.current = runId;
    clearProcessingTimer(processingTimerRef);
    revokeArtifactObjectUrls(artifactUrlsRef);
    setResult(null);
    setArtifactStatuses({});
    setSelectedItem(null);
    setTask({
      phase: "preparing",
      progress: 2,
      message: "正在准备图片",
      artifactDone: 0,
      artifactTotal: RESULT_ITEMS.length
    });

    try {
      const prepared = await prepareUploadImage(file);
      if (runIdRef.current !== runId) {
        return;
      }

      const prepareMessage = prepared.wasCompressed
        ? `已压缩 ${formatBytes(prepared.originalBytes)} -> ${formatBytes(prepared.uploadBytes)}`
        : `图片无需压缩，上传 ${formatBytes(prepared.uploadBytes)}`;

      setTask({
        phase: "uploading",
        progress: 5,
        message: prepareMessage,
        artifactDone: 0,
        artifactTotal: RESULT_ITEMS.length
      });

      const nextResult = await scanDocument(prepared.file, params, {
        onUploadProgress: (uploadProgress) => {
          if (runIdRef.current !== runId) {
            return;
          }
          if (uploadProgress >= 100) {
            setTask((current) => ({
              ...current,
              phase: "processing",
              progress: Math.max(current.progress, 36),
              message: "图片已上传，等待后端处理"
            }));
            startProcessingTimer(processingTimerRef, setTask);
            return;
          }
          setTask((current) => ({
            ...current,
            phase: "uploading",
            progress: 5 + Math.round(uploadProgress * 0.3),
            message: `正在上传图片 ${uploadProgress}%`
          }));
        }
      });
      clearProcessingTimer(processingTimerRef);
      if (runIdRef.current !== runId) {
        return;
      }

      setResult(nextResult);
      await preloadArtifacts(nextResult, runId);
    } catch (error) {
      clearProcessingTimer(processingTimerRef);
      if (runIdRef.current !== runId) {
        return;
      }
      setTask((current) => ({
        ...current,
        phase: "error",
        progress: 0,
        message: error instanceof Error ? error.message : "处理失败"
      }));
    }
  }

  async function preloadArtifacts(nextResult: ScanResult, runId: number) {
    const initialStatuses = ALL_ARTIFACTS.reduce<Partial<Record<ArtifactKey, ArtifactStatus>>>((accumulator, key) => {
      accumulator[key] = {
        sourceUrl: nextResult.artifacts[key],
        objectUrl: null,
        progress: 0,
        status: "idle",
        error: null
      };
      return accumulator;
    }, {});

    setArtifactStatuses(initialStatuses);
    setTask({
      phase: "receiving",
      progress: 75,
      message: `正在回传结果 0/${RESULT_ITEMS.length}`,
      artifactDone: 0,
      artifactTotal: RESULT_ITEMS.length
    });

    const loadSettledArtifact = async (key: ArtifactKey) => {
      const ok = await loadArtifact(key, nextResult.artifacts[key], runId);
      if (runIdRef.current !== runId) {
        return ok;
      }
      setTask((current) => {
        const nextDone = Math.min(RESULT_ITEMS.length, current.artifactDone + 1);
        return {
          ...current,
          phase: "receiving",
          artifactDone: nextDone,
          progress: 75 + Math.round((nextDone / RESULT_ITEMS.length) * 25),
          message: `正在回传结果 ${nextDone}/${RESULT_ITEMS.length}`
        };
      });
      return ok;
    };

    const primaryResults = await Promise.allSettled(PRIMARY_ARTIFACTS.map(loadSettledArtifact));
    if (runIdRef.current === runId) {
      setTask((current) => ({
        ...current,
        message: "关键结果已可查看，继续加载对照图"
      }));
    }
    const secondaryResults = await Promise.allSettled(SECONDARY_ARTIFACTS.map(loadSettledArtifact));

    if (runIdRef.current !== runId) {
      return;
    }

    const settledResults = [...primaryResults, ...secondaryResults];
    const failedCount = settledResults.filter((item) => item.status === "rejected" || !item.value).length;
    setTask((current) => ({
      ...current,
      phase: "done",
      progress: 100,
      message: failedCount > 0 ? `处理完成，${failedCount} 张结果图加载失败，可单独重试` : "处理完成，已生成中间结果和最终二值图"
    }));
  }

  async function loadArtifact(key: ArtifactKey, sourceUrl: string, runId: number): Promise<boolean> {
    setArtifactStatuses((current) => ({
      ...current,
      [key]: {
        sourceUrl,
        objectUrl: current[key]?.objectUrl ?? null,
        progress: 0,
        status: "loading",
        error: null
      }
    }));

    try {
      const objectUrl = await fetchArtifactObjectUrl(sourceUrl, (progress) => {
        if (runIdRef.current !== runId) {
          return;
        }
        setArtifactStatuses((current) => ({
          ...current,
          [key]: {
            sourceUrl,
            objectUrl: current[key]?.objectUrl ?? null,
            progress,
            status: "loading",
            error: null
          }
        }));
      });

      if (runIdRef.current !== runId) {
        URL.revokeObjectURL(objectUrl);
        return false;
      }

      artifactUrlsRef.current.push(objectUrl);
      setArtifactStatuses((current) => {
        const previousUrl = current[key]?.objectUrl;
        if (previousUrl && previousUrl !== objectUrl) {
          URL.revokeObjectURL(previousUrl);
          artifactUrlsRef.current = artifactUrlsRef.current.filter((url) => url !== previousUrl);
        }
        return {
          ...current,
          [key]: {
            sourceUrl,
            objectUrl,
            progress: 100,
            status: "done",
            error: null
          }
        };
      });
      return true;
    } catch (error) {
      if (runIdRef.current !== runId) {
        return false;
      }
      setArtifactStatuses((current) => ({
        ...current,
        [key]: {
          sourceUrl,
          objectUrl: current[key]?.objectUrl ?? null,
          progress: 0,
          status: "error",
          error: error instanceof Error ? error.message : "结果图加载失败"
        }
      }));
      return false;
    }
  }

  async function handleRetryArtifact(key: ArtifactKey) {
    if (!result || isBusy) {
      return;
    }
    const runId = runIdRef.current;
    await loadArtifact(key, result.artifacts[key], runId);
  }

  return (
    <main className="app-shell">
      <header className="top-bar">
        <div>
          <h1>文档扫描验证</h1>
          <p>OpenCV 角点定位、透视矫正、形态学光照校正与二值化对比。</p>
        </div>
      </header>

      <StatusStrip task={task} />

      <div className="workspace">
        <div className="control-column">
          <UploadPanel
            fileName={file?.name ?? null}
            previewUrl={previewUrl}
            disabled={!canRun}
            busy={isBusy}
            onFileChange={handleFileChange}
            onRun={handleRun}
          />
          <ParameterPanel params={params} disabled={isBusy} onChange={setParams} />
          <MetricsPanel result={result} />
        </div>

        <ResultGallery
          result={result}
          artifactStatuses={artifactStatuses}
          items={RESULT_ITEMS}
          onRetry={handleRetryArtifact}
          onSelect={setSelectedItem}
        />
      </div>

      <ImageLightbox item={selectedItem} artifactStatuses={artifactStatuses} result={result} onClose={() => setSelectedItem(null)} />
    </main>
  );
}

function startProcessingTimer(timerRef: MutableRefObject<number | null>, setTask: Dispatch<SetStateAction<ScanTask>>) {
  clearProcessingTimer(timerRef);
  timerRef.current = window.setInterval(() => {
    setTask((current) => {
      if (current.phase !== "processing") {
        return current;
      }
      const messageIndex = Math.min(PROCESSING_MESSAGES.length - 1, Math.floor((current.progress - 36) / 10));
      return {
        ...current,
        progress: Math.min(74, current.progress + (current.progress < 58 ? 3 : 1)),
        message: PROCESSING_MESSAGES[messageIndex]
      };
    });
  }, 700);
}

function clearProcessingTimer(timerRef: MutableRefObject<number | null>) {
  if (timerRef.current !== null) {
    window.clearInterval(timerRef.current);
    timerRef.current = null;
  }
}

function revokeArtifactObjectUrls(artifactUrlsRef: MutableRefObject<string[]>) {
  artifactUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
  artifactUrlsRef.current = [];
}

function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024) {
    return `${(bytes / 1024 / 1024).toFixed(1)}MB`;
  }
  if (bytes >= 1024) {
    return `${Math.round(bytes / 1024)}KB`;
  }
  return `${bytes}B`;
}
