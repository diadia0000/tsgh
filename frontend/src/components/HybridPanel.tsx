import { useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '../api/client'
import { useJob } from '../api/jobs'
import type { components } from '../api/schema'
import type { ImageRect } from './SlideViewer'

type HybridResult = components['schemas']['HybridResult']

// M0-M4 as the pipeline actually runs them (backend/algorithms/hybrid/CLAUDE.md).
// The backend reports one job status for the whole chain -- there is no per-stage
// or per-tile signal -- so this list documents what is happening, and only the
// overall status is live. Faking a stage cursor here would be inventing progress.
const STAGES = [
  { key: 'M0', label: '前處理', detail: '切成 1024px 重疊 tile（precut）' },
  { key: 'M1', label: '疊合', detail: 'UNet++ core mask → IHC/DISH 融合' },
  { key: 'M2', label: '細胞分割', detail: 'Cellpose 產生細胞 instance mask' },
  { key: 'M3', label: '判讀', detail: 'HER2/CEP17 訊號點偵測 + 擴增判定' },
  { key: 'M4', label: '匯出', detail: 'overlay_slide.tiff + summary.txt + report.csv' },
]

// Mirrors config.default_tile_size / window_overlap_px
// (backend/algorithms/hybrid/config.py). Only used for the "how big is this
// job" estimate shown before submitting -- the backend computes the real grid.
const TILE_PX = 1024
const OVERLAP_PX = 256

/** Tiles the backend's grid would produce for a w×h region (one axis at a time:
 *  stride = tile - overlap, with the last window snapped back to the edge). */
function estimateTiles(w: number, h: number) {
  const along = (extent: number) =>
    extent <= TILE_PX ? 1 : Math.ceil((extent - TILE_PX) / (TILE_PX - OVERLAP_PX)) + 1
  return along(w) * along(h)
}

const STATUS_STYLE: Record<string, string> = {
  pending: 'bg-amber-500/15 text-amber-300',
  running: 'bg-sky-500/15 text-sky-300',
  done: 'bg-emerald-500/15 text-emerald-300',
  error: 'bg-red-500/15 text-red-300',
}

/**
 * Hybrid IHC-DISH cell analysis: pick the two aligned slides, run the M0-M4
 * chain as one background job, then read back the run's report.
 *
 * Deliberately whole-slide: POST /api/hybrid/tile takes two slide_ids and
 * nothing else, so the ROI selection in docs/UI/hybrid_flow_mockup.html cannot
 * be wired up yet -- PrecutStream has no region parameter (m0_reader.py:100).
 */
export function HybridPanel({
  onViewSlide,
  viewRect,
}: {
  /** Show a slide in the main viewer -- used to preview the IHC slide while the
   *  user frames a ROI, and to open the annotated overlay when a run finishes. */
  onViewSlide?: (slideId: string) => void
  /** The viewer's current viewport in image pixels, or null when it is showing
   *  something other than a single slide. Source for "用目前檢視範圍". */
  viewRect?: ImageRect | null
}) {
  const queryClient = useQueryClient()
  const [ihc, setIhc] = useState('')
  const [dish, setDish] = useState('')
  const [roi, setRoi] = useState<ImageRect | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [startedAt, setStartedAt] = useState<number | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const [submitError, setSubmitError] = useState<string | null>(null)
  const handledJobs = useRef(new Set<string>())
  const job = useJob(jobId)

  const slides = useQuery({
    queryKey: ['slides'],
    queryFn: async () => {
      const { data, error } = await api.GET('/api/tiles')
      if (error) throw new Error('slide list failed')
      return data
    },
  })

  const result = useQuery<HybridResult>({
    queryKey: ['hybrid-result'],
    queryFn: async () => {
      const { data, error } = await api.GET('/api/hybrid/result')
      if (error) throw new Error('result lookup failed')
      return data
    },
  })

  const status = job.data?.status
  const running = status === 'running' || status === 'pending'
  const busy = running
  // A ROI smaller than one tile is rejected by PrecutStream; block it here so
  // the user is not told by a job that fails seconds after submission.
  const roiValid = !roi || Math.min(roi.w, roi.h) >= TILE_PX
  const canRun = Boolean(ihc && dish && ihc !== dish) && roiValid && !busy

  const analyze = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST('/api/hybrid/tile', {
        body: {
          ihc_slide_id: ihc,
          dish_slide_id: dish,
          // Omitted entirely for a whole-slide run; the backend treats all-null
          // as "no ROI" and four-of-four as the region.
          ...(roi ? { roi_x: roi.x, roi_y: roi.y, roi_w: roi.w, roi_h: roi.h } : {}),
        },
      })
      if (error || !data) throw new Error('送出分析失敗（切片 id 可能不存在，或 ROI 超出範圍）')
      return data.job_id
    },
    onSuccess: (id) => {
      setSubmitError(null)
      setJobId(id)
      setStartedAt(Date.now())
      setElapsed(0)
    },
    onError: (e) => setSubmitError(e instanceof Error ? e.message : '送出分析失敗'),
  })

  useEffect(() => {
    if (!startedAt || !running) return
    const t = setInterval(() => setElapsed((Date.now() - startedAt) / 1000), 250)
    return () => clearInterval(t)
  }, [startedAt, running])

  // A finished run wrote three new files; pull the report and hand the overlay
  // to the viewer. Guarded by handledJobs so a re-render cannot re-fire it.
  useEffect(() => {
    if (status !== 'done' || !jobId || handledJobs.current.has(jobId)) return
    handledJobs.current.add(jobId)
    queryClient.invalidateQueries({ queryKey: ['slides'] })
    result.refetch().then(({ data }) => {
      if (data?.overlay_slide_id) onViewSlide?.(data.overlay_slide_id)
    })
  }, [status, jobId, queryClient, result, onViewSlide])

  const options = slides.data ?? []

  return (
    <div className="flex flex-col gap-4 text-sm text-neutral-200">
      <section>
        <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-neutral-400">
          Hybrid 細胞分析
        </h2>

        <label className="mb-2 flex flex-col gap-1 text-xs text-neutral-400">
          IHC (Her2) 切片
          <select
            value={ihc}
            disabled={busy}
            onChange={(e) => {
              setIhc(e.target.value)
              // Preview it so the user can frame a ROI on the slide they are
              // about to analyse; the ROI would otherwise be blind numbers.
              if (e.target.value) onViewSlide?.(e.target.value)
            }}
            className="rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 font-mono text-xs text-neutral-200 disabled:opacity-40"
          >
            <option value="">— 選擇切片 —</option>
            {options.map((id) => (
              <option key={id} value={id}>{id}</option>
            ))}
          </select>
        </label>

        <label className="mb-2 flex flex-col gap-1 text-xs text-neutral-400">
          DISH 切片
          <select
            value={dish}
            disabled={busy}
            onChange={(e) => setDish(e.target.value)}
            className="rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 font-mono text-xs text-neutral-200 disabled:opacity-40"
          >
            <option value="">— 選擇切片 —</option>
            {options.map((id) => (
              <option key={id} value={id}>{id}</option>
            ))}
          </select>
        </label>

        {ihc && dish && ihc === dish && (
          <p className="mb-2 text-xs text-amber-300">IHC 與 DISH 不能是同一張切片</p>
        )}
        {/* The pipeline requires the pair to be pixel-aligned and identical in
            size -- PrecutStream raises on a mismatch (m0_reader.py:113). Say so
            before a multi-hour job dies on it. */}
        <p className="mb-3 text-xs text-neutral-500">兩張切片須為同尺寸、已像素對齊的對齊輸出。</p>

        <div className="mb-3 rounded-md border border-neutral-800 p-2">
          <div className="mb-1 flex items-center justify-between">
            <span className="text-xs font-medium text-neutral-400">分析範圍</span>
            <span className="text-xs text-neutral-500">
              {roi ? `約 ${estimateTiles(roi.w, roi.h)} 個 tile` : '整張切片'}
            </span>
          </div>

          {!roi && (
            <p className="mb-2 text-xs text-amber-300/80">
              整片分析可達數萬個 tile、耗時以小時計；建議縮放到病灶後再取範圍。
            </p>
          )}

          <div className="mb-2 flex gap-2">
            <button
              type="button"
              disabled={busy || !viewRect}
              onClick={() => viewRect && setRoi(viewRect)}
              className="flex-1 rounded-md border border-neutral-700 bg-neutral-800 px-2 py-1 text-xs hover:bg-neutral-700 disabled:opacity-40"
            >
              用目前檢視範圍
            </button>
            <button
              type="button"
              disabled={busy || !roi}
              onClick={() => setRoi(null)}
              className="rounded-md border border-neutral-700 px-2 py-1 text-xs text-neutral-400 hover:bg-neutral-800 disabled:opacity-40"
            >
              清除
            </button>
          </div>
          {!viewRect && !roi && (
            <p className="mb-2 text-xs text-neutral-600">
              先在上方選 IHC 切片，於右側縮放平移到目標區域後再取範圍。
            </p>
          )}

          {roi && (
            <div className="grid grid-cols-4 gap-1">
              {(['x', 'y', 'w', 'h'] as const).map((k) => (
                <label key={k} className="flex flex-col gap-0.5 text-[10px] text-neutral-500">
                  {k}
                  <input
                    type="number"
                    value={roi[k]}
                    disabled={busy}
                    onChange={(e) =>
                      setRoi({ ...roi, [k]: Math.max(0, Number(e.target.value) || 0) })
                    }
                    className="w-full rounded border border-neutral-700 bg-neutral-900 px-1 py-0.5 font-mono text-[11px] text-neutral-200 disabled:opacity-40"
                  />
                </label>
              ))}
            </div>
          )}
          {roi && Math.min(roi.w, roi.h) < TILE_PX && (
            <p className="mt-1 text-xs text-red-400">範圍的寬與高都必須 ≥ {TILE_PX}px</p>
          )}
        </div>

        <button
          type="button"
          disabled={!canRun}
          onClick={() => analyze.mutate()}
          className="w-full rounded-md bg-violet-600 px-3 py-2 font-medium hover:bg-violet-500 disabled:opacity-40"
        >
          {busy ? '分析中…' : '開始分析'}
        </button>
        {submitError && <p className="mt-2 text-xs text-red-400">{submitError}</p>}
      </section>

      {jobId && (
        <section className="rounded-lg border border-neutral-800 bg-neutral-900/60 p-3">
          <div className="mb-2 flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wide text-neutral-400">
              分析狀態
            </span>
            {status && (
              <span
                className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                  STATUS_STYLE[status] ?? 'bg-neutral-700 text-neutral-300'
                }`}
              >
                {status}
              </span>
            )}
          </div>
          <div className="flex items-center justify-between">
            <p className="break-all font-mono text-xs text-neutral-500">{jobId}</p>
            {startedAt !== null && (
              <span className="shrink-0 pl-2 font-mono text-xs text-neutral-500">
                {elapsed.toFixed(0)}s
              </span>
            )}
          </div>

          <ul className="mt-3 flex flex-col gap-1">
            {STAGES.map(({ key, label, detail }) => (
              <li key={key} className="text-xs">
                <span className="font-mono text-neutral-600">{key}</span>{' '}
                <span className="text-neutral-400">{label}</span>
                <span className="ml-1 text-neutral-600">· {detail}</span>
              </li>
            ))}
          </ul>
          {running && (
            <div className="mt-2 h-1 overflow-hidden rounded-full bg-neutral-800">
              <div className="animate-indeterminate h-full w-1/3 rounded-full bg-violet-500" />
            </div>
          )}
          {job.data?.error && (
            <p className="mt-2 break-all text-xs text-red-400">{job.data.error}</p>
          )}
          {job.data?.metadata && (
            <p className="mt-2 text-xs text-emerald-400">
              tile 成功 {String((job.data.metadata as Record<string, unknown>).success ?? '?')} ·
              略過 {String((job.data.metadata as Record<string, unknown>).skipped ?? '?')}
            </p>
          )}
        </section>
      )}

      {(result.data?.summary || result.data?.has_report) && (
        <section className="rounded-lg border border-neutral-800 bg-neutral-900/60 p-3">
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-neutral-400">
            判讀結果
          </h3>
          {result.data?.summary && (
            <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-all text-[11px] leading-relaxed text-neutral-300">
              {result.data.summary}
            </pre>
          )}
          <div className="mt-2 flex flex-col gap-1 text-xs">
            {result.data?.has_report && (
              <a href="/api/hybrid/report.csv" download className="text-sky-400 hover:underline">
                下載 report.csv
              </a>
            )}
            {result.data?.overlay_slide_id && (
              <button
                type="button"
                onClick={() => onViewSlide?.(result.data!.overlay_slide_id!)}
                className="self-start text-sky-400 hover:underline"
              >
                在檢視器開啟標註結果
              </button>
            )}
          </div>
        </section>
      )}
    </div>
  )
}
