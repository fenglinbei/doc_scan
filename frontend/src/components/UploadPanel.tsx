import { Camera, FileImage, ImagePlus, Play } from "lucide-react";
import type { ChangeEvent } from "react";

type Props = {
  fileName: string | null;
  previewUrl: string | null;
  disabled: boolean;
  busy: boolean;
  onFileChange: (file: File) => void;
  onRun: () => void;
};

export function UploadPanel({ fileName, previewUrl, disabled, busy, onFileChange, onRun }: Props) {
  function handleInputChange(event: ChangeEvent<HTMLInputElement>) {
    const nextFile = event.target.files?.[0];
    if (nextFile) {
      onFileChange(nextFile);
    }
    event.target.value = "";
  }

  return (
    <section className="panel upload-panel" aria-labelledby="upload-title">
      <div className="section-heading">
        <div>
          <h2 id="upload-title">图片上传</h2>
          <p>拍照或从相册选择文档，后端返回全流程中间图。</p>
        </div>
        <FileImage size={22} />
      </div>

      <div className="drop-zone" aria-label="待处理图片预览">
        {previewUrl ? (
          <img src={previewUrl} alt="待处理文档预览" />
        ) : (
          <span className="empty-preview">
            <Camera size={30} />
            <span>拍照或选择相册图片</span>
          </span>
        )}
      </div>

      <div className="source-actions">
        <label className={`source-button ${busy ? "source-button-disabled" : ""}`}>
          <Camera size={18} />
          拍照
          <input type="file" accept="image/*" capture="environment" disabled={busy} onChange={handleInputChange} />
        </label>
        <label className={`source-button ${busy ? "source-button-disabled" : ""}`}>
          <ImagePlus size={18} />
          相册
          <input type="file" accept="image/*" disabled={busy} onChange={handleInputChange} />
        </label>
      </div>

      <div className="upload-actions">
        <span className="file-name">{fileName || "尚未选择图片"}</span>
        <button type="button" className="primary-button" disabled={disabled} onClick={onRun}>
          <Play size={18} fill="currentColor" />
          开始处理
        </button>
      </div>
    </section>
  );
}
