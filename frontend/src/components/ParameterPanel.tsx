import type { ScanParams } from "../types";

type Props = {
  params: ScanParams;
  disabled: boolean;
  onChange: (next: ScanParams) => void;
};

const CONTROLS: Array<{
  key: keyof ScanParams;
  label: string;
  min: number;
  max: number;
  step: number;
}> = [
  { key: "canny_low", label: "Canny 低阈值", min: 1, max: 180, step: 1 },
  { key: "canny_high", label: "Canny 高阈值", min: 20, max: 255, step: 1 },
  { key: "illumination_kernel", label: "形态学核大小", min: 3, max: 151, step: 2 },
  { key: "fixed_threshold", label: "固定阈值", min: 1, max: 254, step: 1 },
  { key: "sauvola_window", label: "Sauvola 窗口", min: 3, max: 151, step: 2 },
  { key: "sauvola_k", label: "Sauvola k", min: 0, max: 0.8, step: 0.01 },
  { key: "cleanup_kernel", label: "清理核大小", min: 1, max: 21, step: 2 }
];

export function ParameterPanel({ params, disabled, onChange }: Props) {
  return (
    <section className="panel" aria-labelledby="params-title">
      <div className="section-heading compact-heading">
        <div>
          <h2 id="params-title">基础参数</h2>
          <p>用于边界检测、光照校正和二值化对比。</p>
        </div>
      </div>

      <div className="param-list">
        {CONTROLS.map((control) => (
          <label className="param-row" key={control.key}>
            <span>
              <span className="param-label">{control.label}</span>
              <span className="param-value">{formatValue(params[control.key])}</span>
            </span>
            <input
              type="range"
              min={control.min}
              max={control.max}
              step={control.step}
              value={params[control.key]}
              disabled={disabled}
              onChange={(event) => {
                const value = control.step < 1 ? Number(event.target.value) : Math.round(Number(event.target.value));
                onChange({ ...params, [control.key]: value });
              }}
            />
          </label>
        ))}
      </div>
    </section>
  );
}

function formatValue(value: number) {
  return Number.isInteger(value) ? String(value) : value.toFixed(2);
}
