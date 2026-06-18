import { X } from "lucide-react";
import type { ArtifactKey, ArtifactStatus, ResultItem, ScanResult } from "../types";

type Props = {
  item: ResultItem | null;
  artifactStatuses: Partial<Record<ArtifactKey, ArtifactStatus>>;
  result: ScanResult | null;
  onClose: () => void;
};

export function ImageLightbox({ item, artifactStatuses, result, onClose }: Props) {
  if (!item || !result) {
    return null;
  }

  const imageUrl = artifactStatuses[item.key]?.objectUrl ?? result.artifacts[item.key];

  return (
    <div className="lightbox" role="dialog" aria-modal="true" aria-label={`${item.label}大图`}>
      <div className="lightbox-header">
        <div>
          <h2>{item.label}</h2>
          <p>{item.description}</p>
        </div>
        <button type="button" className="icon-button" onClick={onClose} aria-label="关闭预览">
          <X size={20} />
        </button>
      </div>
      <img src={imageUrl} alt={item.label} />
    </div>
  );
}
