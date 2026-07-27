"""Environment gate for doc 24 §3: are the GPU codec libraries usable on THIS box?

doc 24 §2.1 proposes replacing Phase D's CPU LZW TIFF encode with nvTIFF / nvCOMP /
nvImageCodec / cuCIM, and §3 makes that conditional on two things nobody had checked:
whether those libraries exist for CUDA 13 + Blackwell (sm_120) at all, and what they
cost in VRAM if they ever ran inside the `workers>1` pool. Per §3's own warning (round
3's bundled `uv sync` regression, still unexplained), nothing here may be installed into
the project venv -- run this in a throwaway venv built by `uv venv`.

Reports, per library: import status, version, whether it can be initialised on this GPU,
and -- when an encoder exists -- the throughput of one encode of a representative
overlay-sized buffer. The pyvips LZW baseline is measured in whichever venv has pyvips,
so the two numbers can be compared directly.

Usage:
  <spike-venv>/bin/python scripts/gpu_codec_spike.py --out spike.json
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time

import numpy as np

W = H = 4096                       # a representative slab of the overlay slide


def _sample(w: int, h: int) -> np.ndarray:
    """Overlay-like content: mostly flat background with sparse structure, so LZW
    compresses roughly the way it does on real annotated tiles."""
    rng = np.random.default_rng(0)
    a = np.full((h, w, 3), 255, dtype=np.uint8)
    ys = rng.integers(0, h, size=h * w // 400)
    xs = rng.integers(0, w, size=h * w // 400)
    a[ys, xs] = rng.integers(0, 255, size=(len(ys), 3), dtype=np.uint8)
    a[::64, :, :] = 0
    a[:, ::64, :] = 0
    return a


def gpu_used_mb() -> int:
    """Device memory in use, from outside the process -- what the `workers>1` VRAM
    budget (doc 24 §0.5) is actually spent against."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
        return int(out)
    except Exception:                             # noqa: BLE001
        return -1


def gpu_info() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as exc:                      # noqa: BLE001
        out = f"unavailable: {exc!r}"
    return {"nvidia_smi": out, "python": platform.python_version()}


def probe_pyvips(arr: np.ndarray, res: dict) -> None:
    try:
        import pyvips
    except Exception as exc:                      # noqa: BLE001
        res["pyvips"] = {"import": f"FAIL: {exc!r}"}
        return
    import tempfile
    from pathlib import Path
    img = pyvips.Image.new_from_memory(
        arr.tobytes(), arr.shape[1], arr.shape[0], 3, "uchar")
    with tempfile.TemporaryDirectory() as td:
        dst = str(Path(td) / "cpu.tiff")
        best = None
        for _ in range(3):
            t0 = time.perf_counter()
            img.tiffsave(dst, tile=True, pyramid=True, compression="lzw", bigtiff=True)
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        size = Path(dst).stat().st_size
    mp = arr.shape[0] * arr.shape[1] / 1e6
    res["pyvips"] = {
        "import": "ok", "version": pyvips.version(0),
        "encode_s": round(best, 4),
        "megapixels_per_s": round(mp / best, 1),
        "output_bytes": size,
        "note": "tile+pyramid+lzw, exactly what _stitch_overlay_slide calls",
    }


def probe_cupy(res: dict) -> None:
    try:
        import cupy as cp
    except Exception as exc:                      # noqa: BLE001
        res["cupy"] = {"import": f"FAIL: {exc!r}"}
        return
    entry = {"import": "ok", "version": cp.__version__}
    try:
        dev = cp.cuda.Device(0)
        cc = dev.compute_capability
        a = cp.arange(1 << 20, dtype=cp.float32)
        entry["compute_capability"] = cc
        entry["kernel_ok"] = bool(float((a * 2).sum()) > 0)
        free, total = cp.cuda.runtime.memGetInfo()
        entry["device_free_mb"] = round(free / 1e6, 1)
    except Exception as exc:                      # noqa: BLE001
        entry["runtime"] = f"FAIL: {exc!r}"
    res["cupy"] = entry


def probe_nvimgcodec(arr: np.ndarray, res: dict) -> None:
    try:
        from nvidia import nvimgcodec
    except Exception as exc:                      # noqa: BLE001
        res["nvimgcodec"] = {"import": f"FAIL: {exc!r}"}
        return
    entry = {"import": "ok", "version": getattr(nvimgcodec, "__version__", "?")}
    try:
        enc = nvimgcodec.Encoder()
        entry["encoder"] = "constructed"
        try:
            import cupy as cp
            gpu = nvimgcodec.as_image(cp.asarray(arr))
            t0 = time.perf_counter()
            blob = enc.encode(gpu, "tiff")
            dt = time.perf_counter() - t0
            entry["tiff_encode_s"] = round(dt, 4)
            entry["tiff_bytes"] = len(bytes(blob)) if blob is not None else 0
        except Exception as exc:                  # noqa: BLE001
            entry["tiff_encode"] = f"FAIL: {exc!r}"
    except Exception as exc:                      # noqa: BLE001
        entry["encoder"] = f"FAIL: {exc!r}"
    res["nvimgcodec"] = entry


def probe_nvtiff(arr: np.ndarray, res: dict) -> None:
    entry = {}
    try:
        import nvtiff                             # python bindings, if any exist
        entry["import"] = "ok"
        entry["version"] = getattr(nvtiff, "__version__", "?")
        entry["attrs"] = sorted(a for a in dir(nvtiff) if not a.startswith("_"))[:40]
    except Exception as exc:                      # noqa: BLE001
        entry["import"] = f"FAIL: {exc!r}"
    res["nvtiff"] = entry


def probe_cucim(res: dict) -> None:
    entry = {}
    try:
        import cucim
        entry["import"] = "ok"
        entry["version"] = getattr(cucim, "__version__", "?")
        try:
            from cucim import CuImage
            entry["CuImage"] = "ok"
            entry["has_write"] = hasattr(CuImage, "write")
        except Exception as exc:                  # noqa: BLE001
            entry["CuImage"] = f"FAIL: {exc!r}"
    except Exception as exc:                      # noqa: BLE001
        entry["import"] = f"FAIL: {exc!r}"
    res["cucim"] = entry


def probe_nvcomp(res: dict) -> None:
    entry = {}
    try:
        from nvidia import nvcomp
        entry["import"] = "ok"
        entry["version"] = getattr(nvcomp, "__version__", "?")
        entry["algorithms"] = sorted(
            a for a in dir(nvcomp) if a.isupper() or a[:1].isupper())[:40]
    except Exception as exc:                      # noqa: BLE001
        entry["import"] = f"FAIL: {exc!r}"
    res["nvcomp"] = entry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--size", type=int, default=W)
    args = ap.parse_args()

    arr = _sample(args.size, args.size)
    res = {"env": gpu_info(), "buffer": [args.size, args.size, 3],
           "vram_used_mb": {"baseline": gpu_used_mb()}}
    probe_pyvips(arr, res)
    res["vram_used_mb"]["after_pyvips"] = gpu_used_mb()
    probe_cupy(res)
    res["vram_used_mb"]["after_cupy_context"] = gpu_used_mb()
    probe_nvimgcodec(arr, res)
    res["vram_used_mb"]["after_nvimgcodec"] = gpu_used_mb()
    probe_nvtiff(arr, res)
    probe_nvcomp(res)
    probe_cucim(res)
    res["vram_used_mb"]["after_all"] = gpu_used_mb()

    print(json.dumps(res, indent=2))
    if args.out:
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
