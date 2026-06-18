import { AlertCircle, CheckCircle2, Loader2, Sparkles } from "lucide-react";
import type { ScanTask } from "../types";

type Props = {
  task: ScanTask;
};

const PHASE_LABELS: Record<ScanTask["phase"], string> = {
  idle: "待选择",
  ready: "待处理",
  preparing: "图片准备",
  uploading: "上传中",
  processing: "处理中",
  receiving: "结果回传",
  done: "完成",
  error: "失败"
};

export function StatusStrip({ task }: Props) {
  const busy = task.phase === "preparing" || task.phase === "uploading" || task.phase === "processing" || task.phase === "receiving";
  const Icon = busy ? Loader2 : task.phase === "done" ? CheckCircle2 : task.phase === "error" ? AlertCircle : Sparkles;

  return (
    <div className={`status-strip status-${task.phase}`}>
      <div className="status-main">
        <Icon className={busy ? "spin" : ""} size={18} strokeWidth={2.2} />
        <span>{task.message}</span>
      </div>
      <div className="status-progress" aria-label={`当前进度 ${task.progress}%`}>
        <span className="phase-chip">{PHASE_LABELS[task.phase]}</span>
        <span className="progress-value">{task.progress}%</span>
        <span className="progress-track">
          <span style={{ width: `${task.progress}%` }} />
        </span>
      </div>
    </div>
  );
}
