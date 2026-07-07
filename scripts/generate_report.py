"""Assemble the measurement report HTML from perf_measure/aggregate artifacts.

Reads _metrics/{all_agg.json, all_resource_summary.json, env_stamp.txt,
*_CPROFILE_cprofile_top.txt} and writes measurement/perf_report.html.
Numbers come from the JSONs; narrative/classification is embedded here.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

MDIR = Path(__file__).resolve().parent.parent / "docs" / "hybrid-pipeline" / "measurement"
MET = MDIR / "_metrics"

FULL_WSI_TILES = 35700  # 204 x 175 at 1024px tile / 768 stride on 156222x134028


def load():
    agg = json.loads((MET / "all_agg.json").read_text())
    res = json.loads((MET / "all_resource_summary.json").read_text())
    env = {}
    for line in (MET / "env_stamp.txt").read_text().splitlines():
        if ":" in line and not line.startswith("#"):
            k, v = line.split(":", 1)
            env[k.strip()] = v.strip()
    return agg, res, env


def cprofile_rows(n=22):
    p = MET / "small_25tile_CPROFILE_cprofile_top.txt"
    if not p.exists():
        return []
    rows = []
    started = False
    for line in p.read_text().splitlines():
        if "Ordered by: cumulative" in line:
            started = True
            continue
        if started and "Ordered by: internal" in line:
            break
        m = re.match(r"\s*(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(.*)", line)
        if m and started:
            ncalls, tot, _pc, cum, _pc2, fn = m.groups()
            fn = fn.replace("/data/taro_Projects/tsgh/", "").replace(
                ".venv/lib/python3.11/site-packages/", "site:")
            rows.append((ncalls, tot, cum, fn))
        if len(rows) >= n:
            break
    return rows


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def extrapolate(agg):
    """Linear per-tile fit from the scale points -> full WSI projection."""
    pts = []
    for lb, a in agg.items():
        if "CPROFILE" in lb:
            continue
        pts.append((a["n_tiles"], a["end_to_end_total_s"]))
    pts.sort()
    if len(pts) < 2:
        return None
    # fit total = intercept + slope*tiles (least squares)
    n = len(pts)
    sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
    sxx = sum(p[0]**2 for p in pts); sxy = sum(p[0]*p[1] for p in pts)
    slope = (n*sxy - sx*sy) / (n*sxx - sx*sx)
    intercept = (sy - slope*sx) / n
    proj = intercept + slope*FULL_WSI_TILES
    return {"slope_s_per_tile": slope, "intercept_s": intercept,
            "points": pts, "full_tiles": FULL_WSI_TILES,
            "projected_s": proj, "projected_h": proj/3600.0}


CSS = """
:root{--bg:#0d1117;--card:#161b22;--fg:#e6edf3;--mut:#9198a1;--acc:#58a6ff;--warn:#e3b341;--bad:#f85149;--good:#3fb950;--bord:#30363d}
@media(prefers-color-scheme:light){:root{--bg:#ffffff;--card:#f6f8fa;--fg:#1f2328;--mut:#59636e;--acc:#0969da;--warn:#9a6700;--bad:#cf222e;--good:#1a7f37;--bord:#d1d9e0}}
:root[data-theme=light]{--bg:#fff;--card:#f6f8fa;--fg:#1f2328;--mut:#59636e;--acc:#0969da;--warn:#9a6700;--bad:#cf222e;--good:#1a7f37;--bord:#d1d9e0}
:root[data-theme=dark]{--bg:#0d1117;--card:#161b22;--fg:#e6edf3;--mut:#9198a1;--acc:#58a6ff;--warn:#e3b341;--bad:#f85149;--good:#3fb950;--bord:#30363d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:20px;margin:38px 0 10px;border-bottom:1px solid var(--bord);padding-bottom:6px}
h3{font-size:16px;margin:24px 0 8px;color:var(--acc)}
.sub{color:var(--mut);font-size:13px;margin-bottom:18px}
.card{background:var(--card);border:1px solid var(--bord);border-radius:10px;padding:16px 18px;margin:14px 0}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:8px 0}
.scroll{overflow-x:auto}
th,td{text-align:right;padding:6px 10px;border-bottom:1px solid var(--bord)}
th:first-child,td:first-child{text-align:left}
th{color:var(--mut);font-weight:600;white-space:nowrap}
tbody tr:hover{background:rgba(88,166,255,.06)}
.bar{height:8px;border-radius:4px;background:var(--acc);display:inline-block;vertical-align:middle}
.tag{display:inline-block;font-size:11px;padding:2px 8px;border-radius:20px;border:1px solid var(--bord);color:var(--mut);margin-right:4px}
.hi{color:var(--bad);font-weight:700}.mid{color:var(--warn);font-weight:600}.lo{color:var(--mut)}
.good{color:var(--good)}
.kv{display:grid;grid-template-columns:200px 1fr;gap:2px 14px;font-size:13.5px}
.kv div:nth-child(odd){color:var(--mut)}
code{background:rgba(128,128,128,.15);padding:1px 5px;border-radius:4px;font-size:12.5px}
.note{border-left:3px solid var(--warn);padding:6px 12px;margin:10px 0;background:rgba(227,179,65,.08)}
.crit{border-left:3px solid var(--bad);padding:6px 12px;margin:10px 0;background:rgba(248,81,73,.08)}
ul{margin:6px 0 6px 0;padding-left:22px}li{margin:3px 0}
pre{background:var(--card);border:1px solid var(--bord);border-radius:8px;padding:12px;overflow-x:auto;font-size:12px}
"""


def phase_color(pct):
    if pct >= 10:
        return "hi"
    if pct >= 5:
        return "mid"
    return "lo"


def build():
    agg, res, env = load()
    scales = [lb for lb in ["small_25tile", "medium_121tile", "large_441tile"] if lb in agg]
    primary = "large_441tile" if "large_441tile" in agg else scales[-1]
    ex = extrapolate(agg)

    H = []
    H.append(f"<div class=wrap>")
    H.append("<h1>Hybrid Pipeline — Deep Bottleneck Measurement Report</h1>")
    H.append(f"<div class=sub>Executes docs/hybrid-pipeline/09-measurement-analysis-plan.md &nbsp;·&nbsp; "
             f"measured {esc(env.get('date_local'))} &nbsp;·&nbsp; git <code>{esc(env.get('git_commit','')[:10])}</code> "
             f"&nbsp;·&nbsp; config_hash <code>{esc(agg[primary].get('config_hash'))}</code></div>")

    # Environment
    H.append("<h2>1 · Environment &amp; provenance</h2><div class=card><div class=kv>")
    for k in ["git_commit", "git_branch", "python", "gpu", "nvidia_driver", "cuda_version",
              "cpu_model", "cpu_cores_logical", "ram_total_gb", "venv_path"]:
        if k in env:
            H.append(f"<div>{esc(k)}</div><div><code>{esc(env[k])}</code></div>")
    H.append("<div>torch</div><div><code>2.10.0+cu130</code> · CUDA available, device cap <code>(12,0)</code> (sm_120), real GPU matmul + Cellpose forward verified</div>")
    H.append("<div>key pkgs</div><div>numpy 2.2.6 · cellpose 4.0.8 (SAM backbone) · pyvips 3.1.1 · scikit-image 0.25.2 · scipy 1.16.3 · opencv-headless 4.12 · full freeze in <code>_metrics/pip_freeze_actual.txt</code></div>")
    H.append("</div></div>")
    H.append("<div class=note><b>Env caveats found (do not affect perf numbers):</b> "
             "(1) requirements.txt pins <code>torch==2.10.0+cu130</code> but PyTorch's R2 CDN "
             "(<code>download-r2.pytorch.org</code>) fails TLS handshake on this host; wheels were "
             "installed by fetching them directly from <code>download.pytorch.org</code> and resolving "
             "the CUDA-13 deps from PyPI. (2) requirements.txt is <b>missing "
             "<code>segmentation_models_pytorch</code></b> (imported by <code>unet_inference.py</code>) "
             "and <code>fastapi/uvicorn</code> (imported by the API layer) — installed manually. "
             "(3) <code>pip check</code> crashes on the local-file torch wheel's <code>+cu130</code> "
             "local tag (pip bug, not a real conflict).</div>")

    # Method / execution order
    H.append("<h2>2 · Method &amp; execution order (plan §1.4)</h2><div class=card>")
    H.append("<p>Non-invasive harness (<code>scripts/perf_measure.py</code>): the pipeline is run "
             "unchanged; timing shims are monkeypatched onto the <code>hybrid_pipeline</code> module "
             "namespace, plus a 0.5 s RAM/VRAM sampler thread and 1 s <code>nvidia-smi dmon</code>. "
             "Order follows the plan: <b>(1) measure one clean end-to-end wall-clock anchor first</b> "
             "(no cProfile — cProfile inflates the Python-heavy items), <b>(2) that serial run is "
             "itself the \"dumb-version\" control</b> (<code>run_batch</code> is a hard-coded sequential "
             "loop), <b>(3) every sub-item is divided by the anchor to get % before looking at absolute "
             "seconds.</b> A separate cProfile pass supplies the function-level Top table only.</p>")
    H.append(f"<p><b>Input scales</b> (all real, from the 156222×134028 warped WSI pair): "
             f"25-tile (4096² test ROI), 121-tile (8192² crop), 441-tile (16384² crop). "
             f"A full-WSI run = <b>{FULL_WSI_TILES:,} tiles</b> (~20 h) is projected, not run.</p></div>")

    # Anchor table
    H.append("<h2>3 · End-to-end anchors (the control numbers)</h2>")
    H.append("<div class=note>These are the preserved \"dumb-version\" baselines. Any future optimization "
             "must beat the matching anchor on total wall-clock or be treated as a negative optimization "
             "(plan §1.4/§2.7). Raw logs/traces kept under <code>_metrics/</code>.</div>")
    H.append("<div class=scroll><table><thead><tr><th>scale</th><th>tiles</th><th>grid</th>"
             "<th>end-to-end (s)</th><th>s / tile</th><th>precut A (s)</th><th>run_batch (s)</th>"
             "<th>cells found</th><th>bg tiles</th><th>peak RSS</th><th>peak VRAM</th></tr></thead><tbody>")
    for lb in scales:
        a = agg[lb]
        g = a["grid"]
        spt = a["end_to_end_total_s"] / a["n_tiles"]
        H.append(f"<tr><td>{esc(lb.split('_')[0])}</td><td>{a['n_tiles']}</td>"
                 f"<td>{g['cols']}×{g['rows']}</td><td><b>{a['end_to_end_total_s']:.1f}</b></td>"
                 f"<td>{spt:.2f}</td><td>{a['wall']['phaseA_precut_s']:.2f}</td>"
                 f"<td>{a['wall']['runbatch_BCD_s']:.1f}</td><td>{a['stats']['success']}</td>"
                 f"<td>{a['stats']['skipped']}</td><td>{a['peak_rss_gb']} GB</td>"
                 f"<td>{a['peak_cuda_reserved_gb']} GB</td></tr>")
    H.append("</tbody></table></div>")
    if ex:
        H.append(f"<div class=crit><b>Full-WSI projection.</b> Least-squares fit over the three scales: "
                 f"<code>total ≈ {ex['intercept_s']:.1f} s + {ex['slope_s_per_tile']:.3f} s/tile × tiles</code>. "
                 f"At {ex['full_tiles']:,} tiles ⇒ <b>~{ex['projected_s']/3600:.1f} hours</b> "
                 f"(confidence: extrapolated, per-tile cost is regime-stable across the 3 measured points; "
                 f"precut &amp; stitch are ≤2.4% and ≤0.75% respectively at every scale so they do not change "
                 f"the order of magnitude). This is an <b>upper bound</b>: the three crops are "
                 f"tissue-dense (~14–15% background tiles), whereas a real WSI is mostly white background "
                 f"whose empty-core tiles short-circuit cheaply — so the true full-slide run is likely "
                 f"below 18.9 h. Note the plan's \"1287-tile / 39×33\" figure assumed 4096px "
                 f"tiles; the pipeline's actual <code>default_tile_size=1024</code> yields "
                 f"<b>{ex['full_tiles']:,} tiles</b> — a documentation gap, see §8.</div>")

    # % ranking per scale (plan 5.1) — phase rollup + sub-item for primary
    H.append("<h2>4 · % ranking table (plan §5.1) — share of the anchor</h2>")
    H.append("<p class=sub>Candidate bottlenecks are ranked by <b>%, not absolute seconds</b>. "
             "Amdahl ceiling = 1/(1−p). Phase codes: A precut · B1 GPU forward · B2 output I/O · "
             "B2r tile read · B3 M3 analysis · B4 tile-boundary GC/cache · B-M1 overlay · "
             "B-stitch per-tile dedup · C global merge · D overlay stitch · init model load.</p>")

    # regime comparison of phase %
    phases = ["B1", "B3", "B2", "B4", "A", "init", "B-M1", "B2r", "D", "B-stitch", "C"]
    H.append("<h3>4.1 Phase rollup across scales (regime stability)</h3><div class=scroll><table><thead><tr>"
             "<th>phase</th>" + "".join(f"<th>{esc(lb.split('_')[0])} %</th>" for lb in scales)
             + "<th>Amdahl (primary)</th></tr></thead><tbody>")
    pmap = {lb: {r["phase"]: r for r in agg[lb]["phase_ranking"]} for lb in scales}
    for ph in phases:
        cells = []
        for lb in scales:
            r = pmap[lb].get(ph)
            if r:
                cells.append(f'<td class={phase_color(r["pct"])}>{r["pct"]:.1f}%</td>')
            else:
                cells.append("<td>–</td>")
        amd = pmap[primary].get(ph, {}).get("amdahl_ceiling", "–")
        H.append(f"<tr><td>{ph}</td>{''.join(cells)}<td>{amd}</td></tr>")
    # residual
    cells = "".join(f'<td class=lo>{agg[lb]["residual_pct"]:.1f}%</td>' for lb in scales)
    H.append(f"<tr><td>residual/unattributed</td>{cells}<td>–</td></tr>")
    H.append("</tbody></table></div>")

    # sub-item ranking for primary
    H.append(f"<h3>4.2 Sub-item ranking — {esc(primary)} (most representative: init amortized)</h3>")
    H.append("<div class=scroll><table><thead><tr><th>rank</th><th>sub-item</th><th>phase</th>"
             "<th>sec</th><th>%</th><th>Amdahl</th><th>share</th></tr></thead><tbody>")
    for i, r in enumerate(agg[primary]["sub_ranking"], 1):
        w = max(1, round(r["pct"] * 4))
        H.append(f"<tr><td>{i}</td><td>{esc(r['label'])}</td><td>{r['phase']}</td>"
                 f"<td>{r['t']:.2f}</td><td class={phase_color(r['pct'])}>{r['pct']:.2f}%</td>"
                 f"<td>{r['amdahl_ceiling']}</td><td><span class=bar style='width:{w}px'></span></td></tr>")
    H.append("</tbody></table></div>")

    # GPU / RAM / VRAM
    H.append("<h2>5 · Cross-cutting: GPU utilization timeline, RAM/VRAM (plan §4)</h2>")
    H.append("<div class=crit><b>The headline finding.</b> Across every scale the GPU is <b>idle "
             "~48–59% of wall-clock</b> with mean SM utilization only <b>~24–29%</b>, even though this "
             "is the sole GPU workload. The serial <code>run_batch</code> loop interleaves GPU forward "
             "passes (B1) with CPU-bound <code>detect_all_dots</code> (B3), PNG encode (B2) and per-tile "
             "<code>gc.collect</code> (B4); while any of those run, the GPU stalls. B1's ~48% wall share "
             "is therefore <i>elapsed</i> time that includes inter-forward gaps, not 48% GPU saturation.</div>")
    H.append("<div class=scroll><table><thead><tr><th>scale</th><th>GPU sm mean</th><th>sm median</th>"
             "<th>sm p90</th><th>idle frac (sm=0)</th><th>busy≥50%</th><th>VRAM peak</th>"
             "<th>RSS start→peak→end</th></tr></thead><tbody>")
    for lb in scales:
        r = res.get(lb, {})
        g = r.get("gpu", {}); m = r.get("mem", {})
        H.append(f"<tr><td>{esc(lb.split('_')[0])}</td><td class=hi>{g.get('sm_mean_pct')}%</td>"
                 f"<td>{g.get('sm_median_pct')}%</td><td>{g.get('sm_p90_pct')}%</td>"
                 f"<td class=hi>{g.get('sm_idle_frac')}</td><td>{g.get('sm_busy_ge50_frac')}</td>"
                 f"<td class=good>{g.get('vram_peak_mb')} MB</td>"
                 f"<td>{m.get('rss_start_gb')}→{m.get('rss_peak_gb')}→{m.get('rss_end_gb')} GB</td></tr>")
    H.append("</tbody></table></div>")
    H.append("<div class=note><b>Memory-growth claim verified.</b> VRAM peak is flat at ~5.16 GB / 32 GB "
             "at every scale (does <i>not</i> grow with tile count) and RSS grows sub-linearly "
             "(25→441 tiles = 17.6× tiles, RSS ~2.8→~3.3 GB ≈ +18%). The new architecture's "
             "\"memory bounded, not linear in tiles\" claim holds under real large input. VRAM has huge "
             "headroom — the fixed batch_size=16 leaves the 5090 heavily under-filled.</div>")

    # supply/consume
    H.append("<h3>5.1 Adjacent-phase supply vs consume (plan §4.7)</h3><div class=card>"
             "<ul><li><b>A → B:</b> strictly serial by construction — <code>precut_paired_tiles()</code> "
             "runs to completion, then <code>run_batch()</code> starts. Zero overlap; precut wall adds "
             "directly on top of analysis. Precut writes all tiles with an 8-thread pool (pyvips releases "
             "the GIL) and stays ~2.3% of wall at every scale, so it is real parallel I/O but a small slice.</li>"
             "<li><b>within B, GPU vs CPU:</b> the true \"looks-parallel-but-serial\" trap — three models "
             "share one CUDA context and run one tile at a time; per tile, GPU forward and CPU dot-detection "
             "execute in sequence, so neither the GPU nor the 20 CPU cores are ever saturated at the same "
             "time. This is the ~48% GPU idle.</li>"
             "<li><b>B → D:</b> serial. <code>_stitch_overlay_slide()</code> runs only after all tiles are "
             "analyzed; it is a single non-parallel pyvips join+lzw+tiffsave. Measured ≤0.75% of wall at "
             "every scale (read+join+compress are fused inside one <code>tiffsave</code> C call and cannot "
             "be separated by Python-level timing).</li></ul></div>")

    # cProfile
    rows = cprofile_rows()
    if rows:
        H.append("<h2>6 · Function-level Top table (cProfile, 25-tile)</h2>")
        H.append("<p class=sub>Main-thread cumulative time. Confirms the old 03-doc ranking still holds on "
                 "current HEAD: the Cellpose 4.x <b>SAM backbone</b> <code>get_rel_pos</code> / "
                 "<code>add_decomposed_rel_pos</code> dominate the forward passes; "
                 "<code>detect_all_dots</code> runs via joblib parallel workers (the ~11.5 s "
                 "<code>time.sleep</code> is the main thread waiting on those subprocess workers, i.e. its "
                 "real CPU cost is spread across cores).</p>")
        H.append("<div class=scroll><table><thead><tr><th>ncalls</th><th>tottime</th><th>cumtime</th>"
                 "<th>function</th></tr></thead><tbody>")
        for ncalls, tot, cum, fn in rows:
            H.append(f"<tr><td>{esc(ncalls)}</td><td>{esc(tot)}</td><td>{esc(cum)}</td>"
                     f"<td style='text-align:left'><code>{esc(fn)}</code></td></tr>")
        H.append("</tbody></table></div>")

    # Bottleneck list
    H.append("<h2>7 · Bottleneck list (plan §5.2 / §5.3)</h2>")
    H.append("<p class=sub>Amdahl stop-loss (plan §5.2): items below ~10% of total are recorded but "
             "<b>not</b> deep-analyzed this round (ceiling &lt;1.11). Classification uses plan §6 "
             "(1 algo/model complexity · 2 hardware · 3 parallel/concurrency · 4 memory lifecycle · "
             "5 I/O &amp; storage · 6 framework overhead · 7 config/dead-code). Only classes named, no fixes.</p>")

    B = agg[primary]
    def pct_of(bucket_label):
        for r in B["sub_ranking"]:
            if bucket_label in r["label"]:
                return r["pct"], r["amdahl_ceiling"], r["t"]
        return 0, 0, 0
    b1 = pmap[primary]["B1"]
    dd = pct_of("detect_all_dots")
    png = pct_of("PNG encode")
    gc_ = pct_of("gc.collect")

    cards = [
        ("① GPU under-utilisation from a fully serial pipeline", "crit",
         f"GPU idle ~{res[primary]['gpu']['sm_idle_frac']*100:.0f}% of wall, mean SM "
         f"~{res[primary]['gpu']['sm_mean_pct']:.0f}%. B1 forward = {b1['pct']:.0f}% of wall "
         f"(M2 Cellpose + M3b Cellpose + M1 UNet, three sequential forwards/tile sharing one CUDA "
         f"context). VRAM peak only 5.16/32 GB.",
         f"{b1['pct']:.1f}% (B1 phase)", b1["amdahl_ceiling"],
         "Primary. Regime-stable across 25/121/441 tiles.",
         "Classes 3 (parallel/concurrency) + 6 (architecture): serial run_batch loop; no cross-tile "
         "GPU batching; three models never overlapped with CPU work.", "measured directly"),
        ("② detect_all_dots (M3 HER2/CEP17 dot detection)", "crit",
         f"{dd[2]:.1f} s = {dd[0]:.0f}% of wall — the single largest sub-item, above any one GPU forward. "
         f"Pure-CPU LAB/H-morphology dot detection over cells, dispatched with joblib n_jobs=-1; the main "
         f"thread blocks on subprocess workers (see §6). Old 03-doc had it at 17.1%; now higher.",
         f"{dd[0]:.1f}%", dd[1],
         "Regime-stable (26.4% at both 121 and 441 tiles).",
         "Class 1 (algorithm complexity) + 3 (parallel — already joblib, but on CPU while GPU idles).",
         "measured directly"),
        ("③ PNG encode + write of per-tile artifacts", "note",
         f"{png[2]:.1f} s = {png[0]:.1f}% of wall. Every tile unconditionally writes core_mask / masked_ihc "
         f"/ dish_mask_overlay as PNG via skimage.io.imsave. (int32 TIFF label writes are separately &lt;0.5%.)",
         f"{png[0]:.1f}%", png[1],
         "New vs old doc (old lumped 13.4% file I/O differently). Borderline — just above/below the 10% "
         "floor depending on scale.",
         "Class 5 (I/O &amp; storage layout): lossless PNG encode of debug arrays on the critical path.",
         "measured directly"),
        ("④ Per-tile gc.collect() (tile-boundary cleanup)", "note",
         f"{gc_[2]:.1f} s = {gc_[0]:.1f}% of wall, entirely from the explicit gc.collect() called once per "
         f"tile in run_batch (torch.cuda.empty_cache is a separate &lt;0.3%). A full Python GC sweep every "
         f"tile is fixed overhead unrelated to real work; grows linearly with tile count.",
         f"{gc_[0]:.1f}%", gc_[1],
         "Below Amdahl floor (ceiling ~1.04) — recorded, not deep-analyzed (plan §5.2). Never measured in "
         "old docs.",
         "Class 4 (memory lifecycle) + 6 (framework overhead).", "measured directly"),
        ("⑤ Phase A precut &amp; Phase D overlay stitch (new stages)", "note",
         "Precut A ~2.3% and stitch D ~0.6–0.75% of wall at every scale — both brand-new stages the old "
         "perf_report never covered. Both scale with tile count but stay small; D is a single serial "
         "pyvips join+lzw+tiffsave after all analysis.",
         "A ~2.3% · D ~0.7%", "~1.02 / ~1.01",
         "Below floor at all measured scales. At full WSI (35.7k tiles) D becomes a single serial ~minutes "
         "block but still &lt;1% of a ~20 h run.",
         "Class 5 (I/O) for both; D also class 3 (serial, could overlap analysis).", "measured + extrapolated"),
        ("⑥ Model init (one-time)", "note",
         "UNet + 2 Cellpose loads = 7.3% at 25 tiles but 1.3% at 121 and &lt;1% at 441 — pure one-time cost "
         "that amortizes to negligible at real scale.",
         "1.3% (121) → ~0.5% (441)", "~1.01",
         "Amortizes away; not a bottleneck at WSI scale.", "Class 6 (framework) — informational only.",
         "measured directly"),
        ("⑦ API / job layer (Phase E)", "note",
         "submit_job enqueue = 2.3 µs mean; BackgroundTask dispatch ~3.8 ms. ~10⁻⁷ of a multi-hour run.",
         "&lt;0.001%", "1.00",
         "Negligible fixed overhead. Only caveat: concurrent analysis requests each take a threadpool "
         "worker but serialize on the single GPU/CUDA context (future multi-request risk, not a current "
         "hotspot).", "Class 6 (framework/architecture).", "measured directly"),
    ]
    for title, cls, phenom, share, amd, regime, klass, conf in cards:
        H.append(f"<div class={cls}><b>{title}</b><div class=kv style='margin-top:8px'>"
                 f"<div>現象 / phenomenon</div><div>{phenom}</div>"
                 f"<div>佔總時間 / share</div><div><b>{share}</b></div>"
                 f"<div>Amdahl ceiling</div><div>{amd}</div>"
                 f"<div>量測規模 / regime</div><div>{regime}</div>"
                 f"<div>分類方向 / class (§6)</div><div>{klass}</div>"
                 f"<div>信心 / confidence</div><div>{conf}</div></div></div>")

    # Gaps
    H.append("<h2>8 · Doc/code/config gaps found (kept out of the perf conclusions)</h2>")
    H.append("<div class=card><ul>"
             "<li><b>G-A · WSI tile count off by ~28×.</b> Plan/03-doc say the full WSI is \"1287 tiles "
             "(39×33)\" — that assumes 4096px tiles. The pipeline's actual <code>default_tile_size=1024</code> "
             "(stride 768) on 156222×134028 gives <b>204×175 = 35,700 tiles</b>. All full-WSI extrapolations "
             "must use 35,700, not 1287.</li>"
             "<li><b>G-B · cellpose_batch_size still dead.</b> Confirmed on HEAD: <code>config_example.py</code> "
             "has no <code>cellpose_batch_size</code> field; both segmenters use "
             "<code>getattr(config,\"cellpose_batch_size\",16)</code> → hard-wired 16. Any batch-size "
             "sweep needs the field added first (plan §2.5). VRAM headroom (5/32 GB) shows 16 under-fills "
             "the 5090.</li>"
             "<li><b>G-C · requirements.txt missing deps.</b> <code>segmentation_models_pytorch</code> "
             "(M1 UNet), <code>fastapi</code>, <code>uvicorn</code> are imported but absent from "
             "requirements.txt.</li>"
             "<li><b>G-D · torch cu130 wheel install path.</b> R2 CDN TLS failure on this host; documented "
             "workaround in §1.</li>"
             "<li><b>G-E · config_example G2 gotcha resolved.</b> Old docs flagged a missing "
             "<code>compute_config_hash()</code>/<code>config=Config()</code> tail — present and working on "
             "this HEAD.</li></ul></div>")

    H.append("<div class=sub style='margin-top:30px'>Raw artifacts (JSON timings, cProfile .prof, "
             "nvidia-smi dmon traces, resource CSVs, pip freeze, env stamp) are preserved under "
             "<code>docs/hybrid-pipeline/measurement/_metrics/</code>. Harness: "
             "<code>scripts/perf_measure.py</code>, <code>aggregate_report.py</code>, "
             "<code>resource_analyze.py</code>, <code>generate_report.py</code>. "
             "No pipeline code was modified.</div>")
    H.append("</div>")

    html = f"<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>Hybrid Pipeline Bottleneck Report</title><style>{CSS}</style></head><body>{''.join(H)}</body></html>"
    (MDIR / "perf_report.html").write_text(html)
    print("wrote", MDIR / "perf_report.html", len(html), "bytes")


if __name__ == "__main__":
    build()
