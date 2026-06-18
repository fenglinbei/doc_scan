from __future__ import annotations

import argparse
import mimetypes
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


class DocScanGateway(BaseHTTPRequestHandler):
    static_root: Path
    backend_base: str

    server_version = "DocScanGateway/0.1"

    def do_GET(self) -> None:
        if self.path.startswith("/api/"):
            self.proxy_to_backend()
            return
        self.serve_static()

    def do_HEAD(self) -> None:
        if self.path.startswith("/api/"):
            self.proxy_to_backend()
            return
        self.serve_static(send_body=False)

    def do_POST(self) -> None:
        self.proxy_to_backend()

    def do_OPTIONS(self) -> None:
        self.proxy_to_backend()

    def proxy_to_backend(self) -> None:
        body = self._read_body()
        target = f"{self.backend_base}{self.path}"
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "host"
        }
        headers["Host"] = urlsplit(self.backend_base).netloc
        headers["X-Forwarded-Proto"] = "http"
        headers["X-Forwarded-Host"] = self.headers.get("Host", "")

        request = Request(target, data=body, headers=headers, method=self.command)
        try:
            with urlopen(request, timeout=180) as response:
                self.send_response(response.status)
                self._copy_response_headers(response.headers)
                self.end_headers()
                if self.command != "HEAD":
                    shutil.copyfileobj(response, self.wfile)
        except HTTPError as exc:
            self.send_response(exc.code)
            self._copy_response_headers(exc.headers)
            self.end_headers()
            if self.command != "HEAD":
                shutil.copyfileobj(exc, self.wfile)
        except URLError as exc:
            self.send_error(502, f"Backend unavailable: {exc.reason}")

    def serve_static(self, send_body: bool = True) -> None:
        request_path = urlsplit(self.path).path
        relative = request_path.lstrip("/")
        candidate = (self.static_root / relative).resolve()
        root = self.static_root.resolve()

        if not str(candidate).startswith(str(root)) or not candidate.is_file():
            candidate = root / "index.html"

        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(candidate.stat().st_size))
        self.end_headers()
        if send_body:
            with candidate.open("rb") as file:
                shutil.copyfileobj(file, self.wfile)

    def _read_body(self) -> bytes | None:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return None
        return self.rfile.read(length)

    def _copy_response_headers(self, headers) -> None:
        for key, value in headers.items():
            if key.lower() not in HOP_BY_HOP_HEADERS:
                self.send_header(key, value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Doc Scan frontend and proxy /api to FastAPI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--static-root", required=True)
    parser.add_argument("--backend", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    DocScanGateway.static_root = Path(args.static_root)
    DocScanGateway.backend_base = args.backend.rstrip("/")
    server = ThreadingHTTPServer((args.host, args.port), DocScanGateway)
    print(f"DocScan gateway listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
