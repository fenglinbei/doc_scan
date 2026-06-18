import type { ScanParams, ScanResult } from "./types";

type ScanDocumentOptions = {
  onUploadProgress?: (progress: number) => void;
};

export function scanDocument(file: File, params: ScanParams, options: ScanDocumentOptions = {}): Promise<ScanResult> {
  const body = new FormData();
  body.append("file", file);
  body.append("params", JSON.stringify(params));

  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", "/api/scan");
    request.responseType = "text";

    request.upload.onprogress = (event) => {
      if (event.lengthComputable && event.total > 0) {
        options.onUploadProgress?.(Math.round((event.loaded / event.total) * 100));
      }
    };
    request.upload.onload = () => options.onUploadProgress?.(100);

    request.onload = () => {
      if (request.status < 200 || request.status >= 300) {
        reject(new Error(readXhrError(request)));
        return;
      }

      try {
        resolve(JSON.parse(request.responseText) as ScanResult);
      } catch {
        reject(new Error("接口返回内容不是有效 JSON"));
      }
    };

    request.onerror = () => reject(new Error("网络连接失败，请检查后端服务或局域网连接"));
    request.ontimeout = () => reject(new Error("请求超时，请稍后重试"));
    request.send(body);
  });
}

export async function fetchArtifactObjectUrl(url: string, onProgress?: (progress: number) => void): Promise<string> {
  onProgress?.(0);
  const response = await fetch(url);

  if (!response.ok) {
    const message = await readError(response);
    throw new Error(message);
  }

  const contentType = response.headers.get("content-type") || "image/png";
  const total = Number(response.headers.get("content-length") || 0);

  if (!response.body) {
    const blob = await response.blob();
    onProgress?.(100);
    return URL.createObjectURL(blob);
  }

  const reader = response.body.getReader();
  const chunks: ArrayBuffer[] = [];
  let loaded = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    if (value) {
      const chunk = new Uint8Array(value.byteLength);
      chunk.set(value);
      chunks.push(chunk.buffer);
      loaded += value.byteLength;
      if (total > 0) {
        onProgress?.(Math.min(99, Math.round((loaded / total) * 100)));
      } else {
        onProgress?.(60);
      }
    }
  }

  onProgress?.(100);
  return URL.createObjectURL(new Blob(chunks, { type: contentType }));
}

async function readError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: string };
    return payload.detail || `请求失败：${response.status}`;
  } catch {
    return `请求失败：${response.status}`;
  }
}

function readXhrError(request: XMLHttpRequest): string {
  try {
    const payload = JSON.parse(request.responseText) as { detail?: string | unknown[] };
    if (typeof payload.detail === "string") {
      return payload.detail;
    }
    if (Array.isArray(payload.detail)) {
      return payload.detail.map(String).join("；");
    }
  } catch {
    return `请求失败：${request.status}`;
  }
  return `请求失败：${request.status}`;
}
