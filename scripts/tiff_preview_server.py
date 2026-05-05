#!/usr/bin/env python3
"""Small Flask UI to preview a BigTIFF locally (headless-friendly)."""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

from flask import Flask, Response, render_template_string, request
import pyvips

HTML_PAGE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>TIFF Preview</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; background: #f7f7f5; color: #1b1b1b; }
    h1 { margin: 0 0 8px; font-size: 22px; }
    .meta { font-size: 13px; color: #4a4a4a; margin-bottom: 16px; }
    .panel { background: #fff; border: 1px solid #ddd; padding: 16px; margin: 12px 0; border-radius: 6px; }
    .controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-bottom: 10px; }
    label { font-size: 13px; }
    input { width: 110px; padding: 4px; }
    button { padding: 6px 10px; cursor: pointer; }
    img { max-width: 100%; height: auto; border: 1px solid #ccc; background: #fafafa; }
    .row { display: grid; grid-template-columns: 1fr; gap: 12px; }
  </style>
</head>
<body>
  <h1>TIFF Preview</h1>
  <div class="meta">
    File: {{ path }}<br>
    Size: {{ width }} x {{ height }}, bands={{ bands }}, format={{ fmt }}, interpretation={{ interp }}, subIFDs={{ subifds }}
  </div>

  <div class="panel">
    <div class="controls">
      <label>Thumbnail size (px)
        <input id="thumbSize" type="number" min="256" max="8000" step="64" value="{{ default_size }}">
      </label>
      <button onclick="loadThumb()">Load thumbnail</button>
    </div>
    <img id="thumb" src="/thumb?size={{ default_size }}" alt="thumbnail">
  </div>

  <div class="panel">
    <div class="controls">
      <label>X <input id="rx" type="number" min="0" value="0"></label>
      <label>Y <input id="ry" type="number" min="0" value="0"></label>
      <label>W <input id="rw" type="number" min="64" step="64" value="1024"></label>
      <label>H <input id="rh" type="number" min="64" step="64" value="1024"></label>
      <button onclick="loadRegion()">Load region</button>
    </div>
    <img id="region" src="/region?x=0&y=0&w=1024&h=1024" alt="region">
  </div>

  <script>
    function loadThumb() {
      const size = document.getElementById('thumbSize').value || 2048;
      document.getElementById('thumb').src = `/thumb?size=${size}&t=${Date.now()}`;
    }
    function loadRegion() {
      const x = document.getElementById('rx').value || 0;
      const y = document.getElementById('ry').value || 0;
      const w = document.getElementById('rw').value || 1024;
      const h = document.getElementById('rh').value || 1024;
      document.getElementById('region').src = `/region?x=${x}&y=${y}&w=${w}&h=${h}&t=${Date.now()}`;
    }
  </script>
</body>
</html>
"""


def _get_meta(path: Path) -> dict:
    img = pyvips.Image.new_from_file(str(path), access="sequential")
    meta = {
        "width": img.width,
        "height": img.height,
        "bands": img.bands,
        "fmt": img.format,
        "interp": img.interpretation,
    }
    try:
        meta["subifds"] = img.get("n-subifds")
    except Exception:
        meta["subifds"] = "n/a"
    return meta


def _to_png_bytes(img: pyvips.Image) -> bytes:
    if img.format != "uchar":
        img = img.cast("uchar")
    return img.write_to_buffer(".png")


def _clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(val, hi))


def create_app(path: Path, default_size: int) -> Flask:
    app = Flask(__name__)
    meta = _get_meta(path)
    thumb_cache: dict[int, bytes] = {}

    @app.route("/")
    def index() -> str:
        return render_template_string(
            HTML_PAGE,
            path=str(path),
            default_size=default_size,
            **meta,
        )

    @app.route("/thumb")
    def thumb() -> Response:
        try:
            size = int(request.args.get("size", default_size))
        except ValueError:
            size = default_size
        size = _clamp(size, 256, 12000)
        if size not in thumb_cache:
            img = pyvips.Image.thumbnail(str(path), size)
            thumb_cache[size] = _to_png_bytes(img)
        return Response(thumb_cache[size], mimetype="image/png")

    @app.route("/region")
    def region() -> Response:
        try:
            x = int(request.args.get("x", 0))
            y = int(request.args.get("y", 0))
            w = int(request.args.get("w", 1024))
            h = int(request.args.get("h", 1024))
        except ValueError:
            x, y, w, h = 0, 0, 1024, 1024
        x = _clamp(x, 0, meta["width"] - 1)
        y = _clamp(y, 0, meta["height"] - 1)
        w = _clamp(w, 64, meta["width"] - x)
        h = _clamp(h, 64, meta["height"] - y)
        img = pyvips.Image.new_from_file(str(path), access="random")
        img = img.crop(x, y, w, h)
        payload = _to_png_bytes(img)
        return Response(payload, mimetype="image/png")

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview a TIFF in a small web UI")
    parser.add_argument(
        "--path",
        default="/home/sec312/project/tsgh/full_wsi_run/output/wsi_run/wsi_run_overlay.tiff",
        help="Path to TIFF file",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=5000, help="Bind port")
    parser.add_argument("--size", type=int, default=2048, help="Default thumbnail size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(path)
    app = create_app(path, args.size)
    print(
        "Starting preview server on http://{host}:{port}".format(
            host=args.host,
            port=args.port,
        )
    )
    time.sleep(0.1)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
