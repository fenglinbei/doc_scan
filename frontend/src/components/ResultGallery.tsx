import { AlertCircle, Download, Loader2, Maximize2, RefreshCw } from "lucide-react";
import type { ArtifactKey, ArtifactStatus, ResultItem, ScanResult } from "../types";

type Props = {
  result: ScanResult | null;
  artifactStatuses: Partial<Record<ArtifactKey, ArtifactStatus>>;
  items: ResultItem[];
  onRetry: (key: ArtifactKey) => void;
  onSelect: (item: ResultItem) => void;
};

export function ResultGallery({ result, artifactStatuses, items, onRetry, onSelect }: Props) {
  return (
    <section className="results-shell" aria-labelledby="results-title">
      <div className="results-heading">
        <div>
          <h2 id="results-title">同步可视化结果</h2>
          <p>关键结果优先回传，其余对照图随后加载。</p>
        </div>
        {result ? <span className="score-chip">score {result.candidate_score.toFixed(2)}</span> : null}
      </div>

      {result ? (
        <div className="result-grid">
          {items.map((item) => {
            const artifact = artifactStatuses[item.key];
            const imageUrl = artifact?.objectUrl ?? null;
            const isDone = artifact?.status === "done" && Boolean(imageUrl);
            const isLoading = artifact?.status === "loading";
            const isError = artifact?.status === "error";

            return (
              <article
                className={`result-card ${item.key === "final" || item.key === "binary_wolf_fused" ? "featured-result" : ""}`}
                key={item.key}
              >
                <button type="button" className="image-button" disabled={!isDone} onClick={() => onSelect(item)}>
                  {imageUrl ? (
                    <img src={imageUrl} alt={item.label} />
                  ) : (
                    <span className="artifact-placeholder">
                      {isError ? <AlertCircle size={22} /> : <Loader2 className={isLoading ? "spin" : ""} size={22} />}
                      <span>{isError ? "加载失败" : isLoading ? `回传 ${artifact?.progress ?? 0}%` : "等待回传"}</span>
                    </span>
                  )}
                  {isLoading ? (
                    <span className="artifact-progress" aria-label={`${item.label}回传进度 ${artifact?.progress ?? 0}%`}>
                      <span style={{ width: `${artifact?.progress ?? 0}%` }} />
                    </span>
                  ) : null}
                  {isDone ? (
                    <span className="inspect-overlay">
                      <Maximize2 size={16} />
                      查看
                    </span>
                  ) : null}
                </button>
                <div className="result-meta">
                  <div>
                    <h3>{item.label}</h3>
                    <p>{isError ? artifact?.error || "结果图加载失败" : item.description}</p>
                  </div>
                  {isError ? (
                    <button type="button" className="icon-link" onClick={() => onRetry(item.key)} aria-label={`重试加载${item.label}`}>
                      <RefreshCw size={17} />
                    </button>
                  ) : (
                    <a className="icon-link" href={result.artifacts[item.key]} download aria-label={`下载${item.label}`}>
                      <Download size={17} />
                    </a>
                  )}
                </div>
              </article>
            );
          })}
        </div>
      ) : (
        <div className="empty-results">
          <div className="empty-line" />
          <div className="empty-line short" />
          <p>上传图片后，这里会展示角点检测、低对比增强、v2.5 融合二值化和对照结果。</p>
        </div>
      )}
    </section>
  );
}
