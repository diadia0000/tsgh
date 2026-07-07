"""Aggregate perf_measure timing JSONs into the plan's required outputs:
 - a % ranking table (plan sec 5.1): every phase/sub-item's share of the
   end-to-end anchor total, sorted desc.
 - Amdahl ceiling 1/(1-p) per candidate (plan sec 5.2).
Emits a machine-readable summary JSON per label; the HTML report is built
separately from these.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

# bucket -> (phase, human label)
BUCKET_MAP = {
    "B1_unet_coremask":   ("B1", "M1 UNet++ core mask forward"),
    "B1_m2_cellpose":     ("B1", "M2 Cellpose forward (IHC-DISH cells)"),
    "B1_m3b_cellpose":    ("B1", "M3b Cellpose forward (DISH nuclei)"),
    "B2_png_encode":      ("B2", "PNG encode+write (core/masked/overlay)"),
    "B2_tiff_encode":     ("B2", "int32 TIFF encode+write (instance/nucleus/overlay)"),
    "B2_percell_crops":   ("B2", "per-cell crop export loop"),
    "B2_render_overlay":  ("B2", "render_overlay_image (annotate)"),
    "B2r_tile_read":      ("B2r", "tile read (_read_rgb)"),
    "B3_build_results":   ("B3", "build_all_positive_results"),
    "B3_enlarge_cells":   ("B3", "enlarge_cell_instances"),
    "B3_detect_dots":     ("B3", "detect_all_dots (HER2/CEP17)"),
    "B3_merge_dots":      ("B3", "merge_dot_results_to_cell_analysis"),
    "B4_gc_collect":      ("B4", "gc.collect (per tile)"),
    "B4_empty_cache":     ("B4", "torch.cuda.empty_cache (per tile)"),
    "BM1_apply_mask":     ("B-M1", "apply_mask_to_ihc_image"),
    "BM1_overlay_dish":   ("B-M1", "overlay_ihc_mask_on_dish"),
    "BM1_fuse":           ("B-M1", "fuse_masked_ihc_with_dish"),
    "Bs_filter_absolutize": ("B-stitch", "filter_and_absolutize (per tile)"),
    "Bs_clear_edge":      ("B-stitch", "clear_slide_edge_cells"),
    "C_export_csv":       ("C", "export_tile_csv"),
    "C_export_summary":   ("C", "export_summary_statistics"),
    "D_stitch_overlay":   ("D", "_stitch_overlay_slide"),
    "init_unet":          ("init", "UNet++ model init"),
    "init_cellpose_m2":   ("init", "Cellpose M2 model init"),
    "init_cellpose_m3b":  ("init", "Cellpose M3b model init"),
}


def amdahl(p: float) -> float:
    return 1.0 / (1.0 - p) if p < 0.999999 else float("inf")


def aggregate(path: Path) -> dict:
    d = json.loads(path.read_text())
    total = d["wall"]["end_to_end_total_s"]
    precut = d["wall"]["phaseA_precut_s"]
    T = d["timings"]

    rows = []
    # Phase A from wall directly (precut is called outside run_batch)
    rows.append({
        "bucket": "A_precut", "phase": "A", "label": "precut_paired_tiles (Phase A)",
        "t": precut, "n": d["n_tiles"], "pct": 100 * precut / total,
    })
    for bucket, meta in BUCKET_MAP.items():
        if bucket in T:
            t = T[bucket]["t"]
            rows.append({
                "bucket": bucket, "phase": meta[0], "label": meta[1],
                "t": t, "n": T[bucket]["n"], "pct": 100 * t / total,
                "bytes": T[bucket].get("bytes"),
            })
    rows.sort(key=lambda r: r["t"], reverse=True)
    for r in rows:
        r["amdahl_ceiling"] = round(amdahl(r["pct"] / 100.0), 2)

    # phase rollup
    phase_tot: dict[str, float] = {}
    for r in rows:
        phase_tot[r["phase"]] = phase_tot.get(r["phase"], 0.0) + r["t"]
    phase_rows = sorted(
        ({"phase": k, "t": v, "pct": 100 * v / total,
          "amdahl_ceiling": round(amdahl(v / total), 2)} for k, v in phase_tot.items()),
        key=lambda r: r["t"], reverse=True,
    )
    accounted = sum(phase_tot.values())
    residual = total - accounted
    return {
        "label": d["label"],
        "n_tiles": d["n_tiles"],
        "grid": d.get("grid"),
        "end_to_end_total_s": total,
        "config_hash": d.get("config_hash"),
        "peak_rss_gb": d.get("peak_rss_gb"),
        "peak_cuda_reserved_gb": d.get("peak_cuda_reserved_gb"),
        "disk_bytes": d.get("disk_bytes"),
        "stats": d.get("stats"),
        "wall": d["wall"],
        "sub_ranking": rows,
        "phase_ranking": phase_rows,
        "residual_unattributed_s": round(residual, 3),
        "residual_pct": round(100 * residual / total, 2),
    }


def fmt_pct(x):
    return f"{x:.2f}%"


def print_table(agg: dict):
    print(f"\n{'='*80}\nLABEL: {agg['label']}  | tiles={agg['n_tiles']} grid={agg['grid']}")
    print(f"END-TO-END ANCHOR TOTAL: {agg['end_to_end_total_s']:.2f} s  (config {agg['config_hash']})")
    print(f"peak RSS={agg['peak_rss_gb']} GB  peak VRAM reserved={agg['peak_cuda_reserved_gb']} GB")
    print(f"{'-'*80}\nPHASE ROLLUP (% of anchor):")
    print(f"{'phase':<10}{'sec':>12}{'%':>10}{'Amdahl':>10}")
    for r in agg["phase_ranking"]:
        print(f"{r['phase']:<10}{r['t']:>12.3f}{r['pct']:>9.2f}%{r['amdahl_ceiling']:>10}")
    print(f"{'residual':<10}{agg['residual_unattributed_s']:>12.3f}{agg['residual_pct']:>9.2f}%")
    print(f"{'-'*80}\nSUB-ITEM RANKING (% of anchor, desc):")
    print(f"{'phase':<8}{'sec':>11}{'%':>9}{'Amdahl':>9}  bucket / label")
    for r in agg["sub_ranking"]:
        print(f"{r['phase']:<8}{r['t']:>11.3f}{r['pct']:>8.2f}%{r['amdahl_ceiling']:>9}  {r['label']}")


if __name__ == "__main__":
    metrics_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out = {}
    for jf in sorted(metrics_dir.glob("*_timings.json")):
        agg = aggregate(jf)
        out[agg["label"]] = agg
        print_table(agg)
        (metrics_dir / f"{agg['label']}_agg.json").write_text(json.dumps(agg, indent=2, default=str))
    (metrics_dir / "all_agg.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {len(out)} aggregate summaries to {metrics_dir}")
