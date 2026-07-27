"""Non-invasive performance measurement harness for the hybrid pipeline.

Executes the plan in docs/hybrid-pipeline/09-measurement-analysis-plan.md.
Does NOT modify pipeline code: it wraps pipeline functions with timing/counter
shims at runtime (monkeypatch on the hybrid_pipeline module namespace) and runs
resource monitors (RAM/VRAM sampler thread + optional nvidia-smi dmon) around the
run. Emits a JSON of per-function/per-phase wall time + call counts + I/O bytes.

Phases (see plan sec 3):
  A  precut_paired_tiles            (I/O bound tiler)
  B1 GPU forward   : generate_ihc_core_mask, segment_masked_dish, segment_windowed
  B2 file I/O out  : _save_tile_array (split PNG vs TIFF), export_per_cell_images,
                     render_overlay_image
  B2r tile read    : _read_rgb
  B3 M3 analysis   : build_all_positive_results, enlarge_cell_instances,
                     detect_all_dots, merge_dot_results_to_cell_analysis
  B4 tile-boundary : gc.collect + torch.cuda.empty_cache
  B-M1 overlay     : apply_mask_to_ihc_image, overlay_ihc_mask_on_dish,
                     fuse_masked_ihc_with_dish
  B-stitch/tile    : filter_and_absolutize, clear_slide_edge_cells
  C  global merge  : export_tile_csv, export_summary_statistics
  D  overlay stitch: _stitch_overlay_slide
  init             : the 3 model _init_* calls

Usage:
  python scripts/perf_measure.py --ihc <path> --dish <path> --output <dir> \
      --label <name> [--workers 8] [--cprofile] [--gpu-dmon]
"""
from __future__ import annotations

import argparse
import functools
import gc
import json
import logging
import os

logging.getLogger("pyvips").setLevel(logging.WARNING)
import subprocess
import sys
import threading
import time
from pathlib import Path

# --- import pipeline ---
HYBRID = Path(__file__).resolve().parent.parent / "backend" / "algorithms" / "hybrid"
sys.path.insert(0, str(HYBRID))
sys.path.insert(0, str(HYBRID.parent.parent.parent))

import hybrid_pipeline as HP  # noqa: E402
import torch  # noqa: E402
import psutil  # noqa: E402

# ------------------------------------------------------------------
# timing registry
# ------------------------------------------------------------------
TIMINGS: dict[str, dict] = {}


def _rec(bucket: str, dt: float, extra: dict | None = None):
    b = TIMINGS.setdefault(bucket, {"n": 0, "t": 0.0})
    b["n"] += 1
    b["t"] += dt
    if extra:
        for k, v in extra.items():
            b[k] = b.get(k, 0) + v


def wrap(module, name: str, bucket: str, bytes_of=None):
    """Wrap module.name with a perf_counter timer accumulating into bucket."""
    orig = getattr(module, name, None)
    if orig is None:
        # Namespace moved (pipeline refactor): skip rather than crash the run.
        print(f"[perf_measure] skip wrap: {module.__name__}.{name} not found")
        return None

    @functools.wraps(orig)
    def shim(*a, **k):
        t0 = time.perf_counter()
        out = orig(*a, **k)
        dt = time.perf_counter() - t0
        extra = None
        if bytes_of is not None:
            try:
                extra = {"bytes": bytes_of(a, k, out)}
            except Exception:
                extra = None
        _rec(bucket, dt, extra)
        return out

    setattr(module, name, shim)
    return orig


# ------------------------------------------------------------------
# GPU-timeline (cuda Event) instrumentation -- doc 17 §3/§4-3
# ------------------------------------------------------------------
# A Python perf_counter around a forward also counts time the GPU was still draining
# queued work, so it cannot answer "how long was the device actually idle while this
# CPU-only segment ran". Events recorded on the default stream can: between two
# consecutive forwards no kernels are enqueued, so the device time from forward k's
# trailing event to forward k+1's leading event *is* the idle window.
#
# Per tile the marker sequence is
#   unet.b unet.a  [M1 overlay glue]  m2.b m2.a  [clear_edge+build+enlarge]  m3b.b m3b.a
# (background tiles stop after unet), and the gap from tile N-1's last marker to tile
# N's first covers _read_rgb + gc.collect + empty_cache + the background-thread join.
CUDA_EVENTS = False
_ev_pending: list = []      # [(label, "b"|"a", event)] for the tile in progress
_ev_prev_tail = None        # ("<label>", event) -- previous tile's trailing marker
_ev_seg_calls = 0           # segment_windowed calls this tile: 1st = M2, 2nd = M3b


def _ev_mark(label: str, phase: str) -> None:
    if not CUDA_EVENTS:
        return
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    _ev_pending.append((label, phase, ev))


def _ev_flush() -> None:
    """Synchronize, reduce this tile's markers into buckets, then drop them."""
    global _ev_prev_tail
    if not CUDA_EVENTS or not _ev_pending:
        return
    torch.cuda.synchronize()

    if _ev_prev_tail is not None:
        _pn, pev = _ev_prev_tail
        fname, fphase, fev = _ev_pending[0]
        if fphase == "b":
            _rec("E_gap_tile_boundary", pev.elapsed_time(fev) / 1000.0)

    for i in range(len(_ev_pending) - 1):
        n0, p0, e0 = _ev_pending[i]
        n1, p1, e1 = _ev_pending[i + 1]
        dt = e0.elapsed_time(e1) / 1000.0
        if p0 == "b" and p1 == "a":
            _rec(f"E_busy_{n0}", dt)
        elif p0 == "a" and p1 == "b":
            _rec(f"E_gap_{n0}__{n1}", dt)

    tail = _ev_pending[-1]
    _ev_prev_tail = (tail[0], tail[2]) if tail[1] == "a" else None
    _ev_pending.clear()


def wrap_gpu(module, name: str, bucket: str, label_of=None):
    """Like wrap(), plus cuda Event markers bracketing the call."""
    orig = getattr(module, name, None)
    if orig is None:
        print(f"[perf_measure] skip wrap_gpu: {module.__name__}.{name} not found")
        return None

    @functools.wraps(orig)
    def shim(*a, **k):
        label = label_of() if label_of else name
        _ev_mark(label, "b")
        t0 = time.perf_counter()
        out = orig(*a, **k)
        dt = time.perf_counter() - t0
        _ev_mark(label, "a")
        _rec(bucket, dt)
        return out

    setattr(module, name, shim)
    return orig


def wrap_tile_boundary():
    """Reset the per-tile segment counter on entry, flush markers on exit."""
    orig = HP._process_precut_tile_gpu

    @functools.wraps(orig)
    def shim(*a, **k):
        global _ev_seg_calls
        _ev_seg_calls = 0
        try:
            return orig(*a, **k)
        finally:
            _ev_flush()

    HP._process_precut_tile_gpu = shim


def _seg_label() -> str:
    """segment_windowed is called twice per tile; name them by order of call."""
    global _ev_seg_calls
    _ev_seg_calls += 1
    return "m2_cellpose" if _ev_seg_calls == 1 else "m3b_cellpose"


_blank_ctx = threading.local()


def wrap_save_tile_array():
    """_save_tile_array: split by suffix into PNG-encode vs TIFF-encode buckets,
    and count output bytes after write.

    Writes issued from _write_blank_tile get their own `_blank` buckets (doc 24
    §0.4 footnote assumed, but never measured, that background-tile writes are a
    small fast-compressing share of B2_png_encode -- this splits them out so the
    assumption is checked instead of inherited)."""
    orig = HP._save_tile_array

    @functools.wraps(orig)
    def shim(path, array):
        suffix = str(path).lower().rsplit(".", 1)[-1]
        bucket = "B2_png_encode" if suffix == "png" else "B2_tiff_encode"
        if getattr(_blank_ctx, "active", False):
            bucket += "_blank"
        t0 = time.perf_counter()
        out = orig(path, array)
        dt = time.perf_counter() - t0
        try:
            nbytes = Path(path).stat().st_size
        except Exception:
            nbytes = 0
        _rec(bucket, dt, {"bytes": nbytes})
        return out

    HP._save_tile_array = shim


def wrap_write_blank_tile():
    """_write_blank_tile: Candidate F (doc 24 §2.5) -- the six placeholder writes a
    background tile issues, never separately measured before this round."""
    orig = HP._write_blank_tile

    @functools.wraps(orig)
    def shim(*a, **k):
        _blank_ctx.active = True
        t0 = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            _rec("F_write_blank_tile", time.perf_counter() - t0)
            _blank_ctx.active = False

    HP._write_blank_tile = shim


# Candidate G (doc 24 §2.6): _save_tile_array runs `path.parent.mkdir(parents=True,
# exist_ok=True)` before every write, i.e. 6 mkdir syscalls per tile against six
# fixed directories that exist after the first tile. Timing Path.mkdir process-wide
# and splitting by target tells us the redundant share without touching the pipeline.
_FIXED_OUT_DIRS = {
    "core_mask", "masked_ihc", "dish_mask_overlay", "instance_mask",
    "dish_nucleus_mask", "overlay_annotated", "merge_overlay",
}


def wrap_mkdir():
    import pathlib
    orig = pathlib.Path.mkdir

    @functools.wraps(orig)
    def shim(self, *a, **k):
        t0 = time.perf_counter()
        try:
            return orig(self, *a, **k)
        finally:
            dt = time.perf_counter() - t0
            bucket = ("G_mkdir_fixed_outdir" if self.name in _FIXED_OUT_DIRS
                      else "G_mkdir_other")
            _rec(bucket, dt)

    pathlib.Path.mkdir = shim


# ------------------------------------------------------------------
# resource sampler
# ------------------------------------------------------------------
class ResourceSampler(threading.Thread):
    def __init__(self, out_path: Path, interval: float = 0.5):
        super().__init__(daemon=True)
        self.out_path = out_path
        self.interval = interval
        self._stopev = threading.Event()
        self.proc = psutil.Process(os.getpid())
        self.rows = []
        self.t0 = time.perf_counter()

    def run(self):
        while not self._stopev.is_set():
            t = time.perf_counter() - self.t0
            try:
                rss = self.proc.memory_info().rss / 1e9
                # include children (thread pool doesn't fork, but be safe)
                for ch in self.proc.children(recursive=True):
                    try:
                        rss += ch.memory_info().rss / 1e9
                    except Exception:
                        pass
            except Exception:
                rss = 0.0
            try:
                alloc = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
            except Exception:
                alloc = reserved = 0.0
            self.rows.append((round(t, 2), round(rss, 3), round(alloc, 3), round(reserved, 3)))
            self._stopev.wait(self.interval)

    def stop(self):
        self._stopev.set()
        self.join(timeout=5)
        with open(self.out_path, "w") as f:
            f.write("t_s,rss_gb,cuda_alloc_gb,cuda_reserved_gb\n")
            for r in self.rows:
                f.write(",".join(str(x) for x in r) + "\n")


def start_dmon(out_path: Path):
    try:
        f = open(out_path, "w")
        p = subprocess.Popen(
            ["nvidia-smi", "dmon", "-s", "um", "-d", "1", "-o", "DT"],
            stdout=f, stderr=subprocess.STDOUT,
        )
        return p, f
    except Exception:
        return None, None


# ------------------------------------------------------------------
# install all wrappers on the hybrid_pipeline namespace
# ------------------------------------------------------------------
def install_wrappers():
    # B1 GPU forward
    if CUDA_EVENTS:
        wrap_gpu(HP, "generate_ihc_core_mask", "B1_unet_coremask",
                 label_of=lambda: "unet_coremask")
        wrap_gpu(HP, "segment_windowed", "B1_m3b_cellpose", label_of=_seg_label)
        wrap_tile_boundary()
    else:
        wrap(HP, "generate_ihc_core_mask", "B1_unet_coremask")
        wrap(HP, "segment_windowed", "B1_m3b_cellpose")
    wrap(HP, "segment_masked_dish", "B1_m2_cellpose")
    # B2 file I/O out
    wrap_save_tile_array()
    wrap_write_blank_tile()
    wrap_mkdir()
    wrap(HP, "export_per_cell_images", "B2_percell_crops")
    wrap(HP, "render_overlay_image", "B2_render_overlay")
    # B2r read
    wrap(HP, "_read_rgb", "B2r_tile_read")
    # B3 M3
    wrap(HP, "build_all_positive_results", "B3_build_results")
    wrap(HP, "enlarge_cell_instances", "B3_enlarge_cells")
    wrap(HP, "detect_all_dots", "B3_detect_dots")
    wrap(HP, "merge_dot_results_to_cell_analysis", "B3_merge_dots")
    # B-M1 overlay
    wrap(HP, "apply_mask_to_ihc_image", "BM1_apply_mask")
    wrap(HP, "overlay_ihc_mask_on_dish", "BM1_overlay_dish")
    wrap(HP, "fuse_masked_ihc_with_dish", "BM1_fuse")
    # B-stitch per tile
    wrap(HP, "filter_and_absolutize", "Bs_filter_absolutize")
    wrap(HP, "clear_slide_edge_cells", "Bs_clear_edge")
    # C global merge
    wrap(HP, "export_tile_csv", "C_export_csv")
    wrap(HP, "export_summary_statistics", "C_export_summary")
    # D stitch
    wrap(HP, "_stitch_overlay_slide", "D_stitch_overlay")
    # init
    wrap(HP, "_init_unet_inferencer", "init_unet")
    wrap(HP, "_init_cellpose_segmenter", "init_cellpose_m2")
    wrap(HP, "_init_dish_cellpose_segmenter", "init_cellpose_m3b")
    # B4 tile-boundary cleanup: patch gc.collect + torch.cuda.empty_cache
    _orig_gc = gc.collect

    def gc_shim(*a, **k):
        t0 = time.perf_counter()
        r = _orig_gc(*a, **k)
        _rec("B4_gc_collect", time.perf_counter() - t0)
        return r

    gc.collect = gc_shim
    _orig_ec = torch.cuda.empty_cache

    def ec_shim(*a, **k):
        t0 = time.perf_counter()
        r = _orig_ec(*a, **k)
        _rec("B4_empty_cache", time.perf_counter() - t0)
        return r

    torch.cuda.empty_cache = ec_shim
    # per-tile total
    wrap(HP, "process_precut_tile", "B_process_precut_tile_TOTAL")


# ------------------------------------------------------------------
def dir_bytes(p: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(p):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return total


def _sum_worker_timings(per_worker: list) -> dict:
    """Fold every worker's TIMINGS into one bucket table.

    The sum is *aggregate CPU time across workers*, not wall — at `workers=4` a
    bucket totalling 200 s occupied roughly 50 s of the run. It answers "where did
    the work go", which is what the parent-only instrumentation could not; it does
    not answer "how long did the run take". Use `wall.end_to_end_total_s` for that.
    """
    out: dict = {}
    for entry in per_worker:
        for bucket, vals in entry.get("timings", {}).items():
            acc = out.setdefault(bucket, {})
            for k, v in vals.items():
                if isinstance(v, (int, float)):
                    acc[k] = acc.get(k, 0) + v
    for vals in out.values():
        if "t" in vals:
            vals["t"] = round(vals["t"], 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ihc", required=True)
    ap.add_argument("--dish", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cprofile", action="store_true")
    ap.add_argument("--gpu-dmon", action="store_true")
    ap.add_argument("--cuda-events", action="store_true",
                    help="measure device-side idle gaps between GPU forwards "
                         "(doc 17 §3); adds a per-tile torch.cuda.synchronize, so use "
                         "these runs for bubble sizing, not for wall-clock comparison")
    ap.add_argument("--cellpose-batch-size", type=int, default=None,
                    help="override config.cellpose_batch_size for a sweep (doc 13 P4); "
                         "omit to use the config default")
    ap.add_argument("--mp-workers", type=int, default=1,
                    help="cross-tile multiprocessing worker count (doc 20 Candidate D); "
                         "1 = today's single-process path. The monkeypatched TIMINGS only "
                         "see the parent, so with >1 pass --worker-timings to have each "
                         "worker report its own buckets; end-to-end wall remains the only "
                         "thing to *decide* on, per the playbook's step 4")
    ap.add_argument("--worker-timings", action="store_true",
                    help="with --mp-workers>1, install the same timing shims inside each "
                         "spawned worker and collect their TIMINGS back over the result "
                         "queue (DISCOVERED #40). Adds the shim overhead to the worker "
                         "hot path, so use these runs for stage breakdown, not for "
                         "wall-clock comparison")
    ap.add_argument("--resume", action="store_true",
                    help="enable run_batch's per-tile checkpoint so an interrupted run "
                         "restarts where it stopped (doc 19 #1c); intended for full-slide "
                         "runs, not for repeat-measurement rounds -- a resumed run does "
                         "less work and its wall-clock is not comparable")
    ap.add_argument("--cuda-alloc-conf", default=None,
                    help="override config.cuda_alloc_conf for the spawned workers, e.g. "
                         "'expandable_segments:True' (doc 19 #7b). Only meaningful with "
                         "--mp-workers>1; the parent never allocates on the device there")
    ap.add_argument("--stream-precut", action="store_true",
                    help="overlap phase A precut with the analysis loop (doc 17 §4-4); "
                         "phaseA_precut_s then measures only the grid computation")
    ap.add_argument("--metrics-dir", default=None)
    args = ap.parse_args()

    global CUDA_EVENTS
    CUDA_EVENTS = args.cuda_events

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(args.metrics_dir) if args.metrics_dir else out.parent / "_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    install_wrappers()

    from config import config as CFG, compute_config_hash  # noqa

    # Mutating the shared singleton before the models are built is what makes the sweep
    # take effect: hybrid_pipeline's _init_*_segmenter reads config.cellpose_batch_size
    # at init time, and both modules hold the same Config object.
    if args.cellpose_batch_size is not None:
        CFG.cellpose_batch_size = args.cellpose_batch_size
    if args.cuda_alloc_conf is not None:
        CFG.cuda_alloc_conf = args.cuda_alloc_conf

    # The worker probe is passed by environment because `spawn` children inherit it
    # and re-import everything else from scratch; a function reference would have to
    # be picklable out of `__main__`, which re-executes this script in the child.
    if args.worker_timings:
        if args.mp_workers <= 1:
            print("[perf_measure] --worker-timings needs --mp-workers>1; ignoring")
        else:
            os.environ["HYBRID_MP_WORKER_PROBE"] = "mp_worker_probe:install"
            os.environ["PYTHONPATH"] = os.pathsep.join(
                filter(None, [str(Path(__file__).resolve().parent),
                              os.environ.get("PYTHONPATH", "")])
            )

    sampler = ResourceSampler(metrics_dir / f"{args.label}_resource.csv", interval=0.5)
    dmon_p = dmon_f = None
    if args.gpu_dmon:
        dmon_p, dmon_f = start_dmon(metrics_dir / f"{args.label}_gpu_dmon.txt")
    sampler.start()

    scratch = out / "_precut_scratch"
    ihc_out = scratch / "ihc"
    dish_out = scratch / "dish"

    prof = None
    if args.cprofile:
        import cProfile
        prof = cProfile.Profile()

    t_start = time.perf_counter()

    # Phase A: precut. With --stream-precut the cutting is overlapped with the analysis
    # loop instead of preceding it, so phaseA here only covers the header read + grid
    # computation and the cutting cost lands inside runbatch. End-to-end wall is then
    # the only honest comparison between the two modes.
    stream = None
    tA0 = time.perf_counter()
    if args.stream_precut:
        stream = HP.PrecutStream(
            Path(args.ihc), Path(args.dish), ihc_out, dish_out,
            tile_size=CFG.default_tile_size, overlap=CFG.window_overlap_px,
            workers=args.workers,
        )
        positions = stream.positions
    else:
        positions = HP.precut_paired_tiles(
            Path(args.ihc), Path(args.dish), ihc_out, dish_out,
            tile_size=CFG.default_tile_size, overlap=CFG.window_overlap_px,
            workers=args.workers,
        )
    tA = time.perf_counter() - tA0
    n_tiles = len(positions)
    precut_bytes = dir_bytes(scratch)

    # Phase B+C+D: run_batch
    tB0 = time.perf_counter()
    if prof:
        prof.enable()
    stats = HP.run_batch(ihc_out, dish_out, out, tile_stream=stream,
                         workers=args.mp_workers, checkpoint=args.resume)
    if prof:
        prof.disable()
    tBCD = time.perf_counter() - tB0

    total = time.perf_counter() - t_start
    sampler.stop()
    if dmon_p:
        dmon_p.terminate()
        if dmon_f:
            dmon_f.close()

    out_bytes = dir_bytes(out) - precut_bytes

    result = {
        "label": args.label,
        "n_tiles": n_tiles,
        "mp_workers": args.mp_workers,
        "grid": {"cols": len(set(x for x, _ in positions)),
                 "rows": len(set(y for _, y in positions))},
        "wall": {
            "end_to_end_total_s": round(total, 3),
            "phaseA_precut_s": round(tA, 3),
            "runbatch_BCD_s": round(tBCD, 3),
        },
        "disk_bytes": {
            "precut_scratch": precut_bytes,
            "analysis_output": out_bytes,
        },
        "stats": stats,
        "timings": TIMINGS,
        "worker_timings": HP.LAST_MP_WORKER_TIMINGS,
        "worker_timings_total": _sum_worker_timings(HP.LAST_MP_WORKER_TIMINGS),
        "cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "config_hash": compute_config_hash(CFG),
        "peak_cuda_reserved_gb": round(max((r[3] for r in sampler.rows), default=0), 3),
        "peak_rss_gb": round(max((r[1] for r in sampler.rows), default=0), 3),
    }
    with open(metrics_dir / f"{args.label}_timings.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    if prof:
        import pstats
        st = pstats.Stats(prof)
        st.dump_stats(str(metrics_dir / f"{args.label}_cprofile.prof"))
        with open(metrics_dir / f"{args.label}_cprofile_top.txt", "w") as f:
            st.stream = f
            st.sort_stats("cumulative").print_stats(40)
            st.sort_stats("tottime").print_stats(40)

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
