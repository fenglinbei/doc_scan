import type { PreparedUpload } from "./types";

const MAX_UPLOAD_EDGE = 1800;
const JPEG_QUALITY = 0.86;
const COMPRESSION_SIZE_THRESHOLD = 2.5 * 1024 * 1024;

export async function prepareUploadImage(file: File): Promise<PreparedUpload> {
  if (!file.type.startsWith("image/")) {
    throw new Error("请选择图片文件");
  }

  const { image, url } = await loadImage(file);
  try {
    const width = image.naturalWidth || image.width;
    const height = image.naturalHeight || image.height;
    const longestEdge = Math.max(width, height);
    const shouldResize = longestEdge > MAX_UPLOAD_EDGE;
    const shouldCompress = shouldResize || file.size > COMPRESSION_SIZE_THRESHOLD;

    if (!shouldCompress || file.type === "image/svg+xml") {
      return {
        file,
        originalBytes: file.size,
        uploadBytes: file.size,
        wasCompressed: false,
        width,
        height
      };
    }

    const scale = Math.min(1, MAX_UPLOAD_EDGE / longestEdge);
    const targetWidth = Math.max(1, Math.round(width * scale));
    const targetHeight = Math.max(1, Math.round(height * scale));
    const canvas = document.createElement("canvas");
    canvas.width = targetWidth;
    canvas.height = targetHeight;

    const context = canvas.getContext("2d");
    if (!context) {
      throw new Error("当前浏览器不支持图片压缩");
    }
    context.drawImage(image, 0, 0, targetWidth, targetHeight);

    const blob = await canvasToBlob(canvas);
    if (!shouldResize && blob.size >= file.size) {
      return {
        file,
        originalBytes: file.size,
        uploadBytes: file.size,
        wasCompressed: false,
        width,
        height
      };
    }

    const uploadFile = new File([blob], compressedName(file.name), {
      type: "image/jpeg",
      lastModified: file.lastModified
    });

    return {
      file: uploadFile,
      originalBytes: file.size,
      uploadBytes: uploadFile.size,
      wasCompressed: true,
      width: targetWidth,
      height: targetHeight
    };
  } finally {
    URL.revokeObjectURL(url);
  }
}

function loadImage(file: File): Promise<{ image: HTMLImageElement; url: string }> {
  const url = URL.createObjectURL(file);
  const image = new Image();
  image.decoding = "async";

  return new Promise((resolve, reject) => {
    image.onload = () => resolve({ image, url });
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("图片读取失败，请更换图片后重试"));
    };
    image.src = url;
  });
}

function canvasToBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error("图片压缩失败，请更换图片后重试"));
        }
      },
      "image/jpeg",
      JPEG_QUALITY
    );
  });
}

function compressedName(name: string): string {
  const baseName = name.replace(/\.[^.]+$/, "");
  return `${baseName || "document"}-upload.jpg`;
}
