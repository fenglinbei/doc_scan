import type { ScanResult } from "../types";

type Props = {
  result: ScanResult | null;
};

const METRIC_LABELS: Record<string, string> = {
  candidate_count: "候选数",
  candidate_score: "候选分",
  candidate_source: "候选来源",
  final_output: "最终分支",
  text_detail_strength: "文字细节",
  readable_text_ratio: "可读前景",
  foreground_ratio: "前景比例",
  text_background_contrast: "文字对比",
  small_noise_components: "小噪点",
  otsu_threshold: "Otsu 阈值",
  output_width: "输出宽",
  output_height: "输出高",
  time_total_ms: "CV 耗时",
  time_process_call_ms: "处理调用",
  time_save_artifacts_ms: "保存耗时",
  time_api_total_ms: "接口总耗时"
};

export function MetricsPanel({ result }: Props) {
  if (!result) {
    return null;
  }

  const entries = Object.entries(result.metrics).filter(([key]) => key in METRIC_LABELS);
  return (
    <section className="panel metrics-panel" aria-labelledby="metrics-title">
      <div className="section-heading compact-heading">
        <div>
          <h2 id="metrics-title">处理指标</h2>
          <p>用于快速判断边界与二值化质量。</p>
        </div>
      </div>
      <div className="metric-grid">
        {entries.map(([key, value]) => (
          <div className="metric-tile" key={key}>
            <span>{METRIC_LABELS[key]}</span>
            <strong>{String(value)}</strong>
          </div>
        ))}
      </div>
      {result.warnings.length ? (
        <div className="warning-list">
          {result.warnings.map((warning) => (
            <p key={warning}>{warning}</p>
          ))}
        </div>
      ) : null}
    </section>
  );
}
