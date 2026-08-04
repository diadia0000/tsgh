import { useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '../api/client'
import { useJob } from '../api/jobs'
import type { components } from '../api/schema'
import { CollapsibleSection } from './CollapsibleSection'
import { parseSummary } from '../lib/summaryReport'
import type { ImageRect } from './SlideViewer'

type HybridResult = components['schemas']['HybridResult']

// What the analysis does, in the order it does it. The backend reports one
// status for the whole chain -- there is no per-stage signal -- so this is a
// description of the work, not a live cursor, and it stays behind a switch:
// during a run the only thing the reader needs is that it is still running.
const STAGES = [
  '將選取範圍切成小塊處理',
  '疊合兩張切片的影像',
  '找出每個細胞的範圍',
  '偵測訊號點並判讀是否擴增',
  '匯出報表與標註影像',
]

// The slide_ids the alignment run registers its two warped layers under
// (backend/api/alignment.py `_publish_aligned_layers`). Fixed names, so the
// panel can default to them the moment they appear in the slide list.
const ALIGNED_IHC = 'aligned_her2'
const ALIGNED_DISH = 'aligned_dish'

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

/** Where a fresh selection box starts: centred on what the user is looking at,
 *  inset to 80% of it. Inset rather than the full viewport so the frame and its
 *  resize handle are both on screen and clear of OSD's bottom-right navigator.
 *  Derived from the (already image-clamped) view rect, so it cannot start off
 *  the slide. */
function boxFromView(view: ImageRect): ImageRect {
  const w = Math.max(1, Math.round(view.w * 0.8))
  const h = Math.max(1, Math.round(view.h * 0.8))
  return {
    x: Math.round(view.x + (view.w - w) / 2),
    y: Math.round(view.y + (view.h - h) / 2),
    w,
    h,
  }
}

const STATUS_STYLE: Record<string, string> = {
  pending: 'bg-amber-500/15 text-amber-300',
  running: 'bg-sky-500/15 text-sky-300',
  done: 'bg-emerald-500/15 text-emerald-300',
  error: 'bg-red-500/15 text-red-300',
}

const STATUS_TEXT: Record<string, string> = {
  pending: '等待中',
  running: '執行中',
  done: '已完成',
  error: '發生錯誤',
}

/**
 * What each colour on the annotated overlay means. Taken from the renderer's own
 * constants (backend/algorithms/hybrid/m4_module/overlay.py), which are declared
 * **BGR** for OpenCV -- the swatches below are the RGB equivalents. Read off the
 * source rather than the mockup, whose palette is illustrative.
 */
const OVERLAY_LEGEND = [
  { color: '#00ff00', label: '細胞邊界' },
  { color: '#ffa500', label: 'DISH 細胞核' },
  { color: '#ff1493', label: 'DISH 細胞核（已配對）' },
  { color: '#000000', label: 'HER2 訊號點', ring: true },
  { color: '#dc0000', label: 'CEP17 訊號點' },
  { color: '#ffff00', label: '判定為擴增的細胞' },
  { color: '#00c8ff', label: '判定為未擴增的細胞' },
  { color: '#00008b', label: '位移箭頭' },
]

/** Seconds as something a person says out loud. */
function formatElapsed(seconds: number) {
  if (seconds < 60) return `${Math.round(seconds)} 秒`
  const minutes = Math.floor(seconds / 60)
  return minutes < 60 ? `${minutes} 分 ${Math.round(seconds % 60)} 秒` : `${Math.floor(minutes / 60)} 小時 ${minutes % 60} 分`
}

/** Where the running analysis's start time is kept, so a reload can still show
 *  a truthful elapsed clock. The job id itself comes from the server, which is
 *  the authority on whether it is still running. */
const RUN_CLOCK_KEY = 'tsgh.hybridRunClock'

function readRunClock(): { jobId: string; startedAt: number } | null {
  try {
    const parsed = JSON.parse(localStorage.getItem(RUN_CLOCK_KEY) ?? 'null')
    return parsed && typeof parsed.jobId === 'string' && typeof parsed.startedAt === 'number'
      ? parsed
      : null
  } catch {
    return null
  }
}

function writeRunClock(jobId: string, startedAt: number) {
  try {
    localStorage.setItem(RUN_CLOCK_KEY, JSON.stringify({ jobId, startedAt }))
  } catch {
    // Private mode: the clock is cosmetic, recovery itself does not depend on it.
  }
}

/**
 * Hybrid IHC-DISH cell analysis: pick the two aligned slides, frame the region
 * to analyse, run the whole chain as one background job, then read back the
 * run's report.
 *
 * The region is a rectangle the user drags on the slide, per
 * docs/UI/hybrid_flow_mockup.html 畫面 1; SlideViewer draws and edits it, this
 * panel submits it as roi_x/y/w/h (image pixels) to POST /api/hybrid/tile.
 * Leaving it unset runs the whole slide, which is hours of GPU -- hence the
 * warning next to the control.
 */
export function HybridPanel({
  onViewSlide,
  viewRect,
  roi,
  onRoiChange,
  externalBusy,
  onBusyChange,
}: {
  /** Show a slide in the main viewer -- used to preview the IHC slide while the
   *  user frames a ROI, and to open the annotated overlay when a run finishes. */
  onViewSlide?: (slideId: string) => void
  /** The viewer's current viewport in image pixels, or null when it is showing
   *  something other than a single slide. Seeds a new selection box. */
  viewRect?: ImageRect | null
  /** The selection box, in image pixels. Owned by App because the viewer draws
   *  and drags the same rectangle this panel submits. Null = whole slide. */
  roi?: ImageRect | null
  onRoiChange?: (rect: ImageRect | null) => void
  /** The alignment pipeline is occupying the machine; both contend for the same
   *  GPU, so this panel's controls stay locked until it finishes. */
  externalBusy?: boolean
  /** Raised while an analysis of this panel's is submitting or running. */
  onBusyChange?: (busy: boolean) => void
}) {
  const queryClient = useQueryClient()
  const [ihc, setIhc] = useState('')
  const [dish, setDish] = useState('')
  const [analysisOpen, setAnalysisOpen] = useState(true)
  const [showStages, setShowStages] = useState(false)
  const [showLegend, setShowLegend] = useState(false)
  const autoFilled = useRef(false)
  const setRoi = onRoiChange ?? (() => {})
  const [jobId, setJobId] = useState<string | null>(null)
  const [startedAt, setStartedAt] = useState<number | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const [submitError, setSubmitError] = useState<string | null>(null)
  const handledJobs = useRef(new Set<string>())
  // One-shot: after the panel has re-attached once, a finished job must not be
  // adopted again on the next refetch.
  const adoptedJob = useRef(false)
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
      if (error || !data) throw new Error('送出分析失敗，請確認選擇的切片與分析範圍')
      return data.job_id
    },
    onSuccess: (id) => {
      setSubmitError(null)
      setJobId(id)
      const now = Date.now()
      setStartedAt(now)
      setElapsed(0)
      writeRunClock(id, now)
    },
    onError: (e) => setSubmitError(e instanceof Error ? e.message : '送出分析失敗'),
  })

  const status = job.data?.status
  const running = status === 'running' || status === 'pending'
  const ownBusy = running || analyze.isPending
  const busy = ownBusy || Boolean(externalBusy)

  // Own work only; feeding externalBusy back would deadlock both panels.
  useEffect(() => {
    onBusyChange?.(ownBusy)
  }, [ownBusy, onBusyChange])

  // A ROI smaller than one tile is rejected by PrecutStream; block it here so
  // the user is not told by a job that fails seconds after submission.
  const roiValid = !roi || Math.min(roi.w, roi.h) >= TILE_PX
  const canRun = Boolean(ihc && dish && ihc !== dish) && roiValid && !busy

  // What the server says is still executing, regardless of what this tab knows.
  // Fetched once on mount: a job that starts later was started from here.
  const activeJob = useQuery({
    queryKey: ['hybrid-active'],
    queryFn: async () => {
      const { data, error } = await api.GET('/api/hybrid/active')
      if (error) throw new Error('active lookup failed')
      return data
    },
    staleTime: Infinity,
  })

  // Re-attach to a run this tab lost: a reload, a crash, or simply a doctor who
  // closed the laptop. Nothing is re-submitted -- the job kept running on the
  // server, and this only points the poller back at it.
  useEffect(() => {
    const serverJob = activeJob.data?.job_id
    if (!serverJob || jobId || adoptedJob.current) return
    adoptedJob.current = true
    setJobId(serverJob)
    // The elapsed clock is only truthful if this browser remembers starting
    // *this* job; otherwise show the run without inventing a start time.
    const clock = readRunClock()
    setStartedAt(clock?.jobId === serverJob ? clock.startedAt : null)
    setElapsed(0)
  }, [activeJob.data, jobId])

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

  const report = parseSummary(result.data?.summary)
  const options = slides.data ?? []
  const hasAlignedPair = options.includes(ALIGNED_IHC) && options.includes(ALIGNED_DISH)
  const usingAligned = ihc === ALIGNED_IHC && dish === ALIGNED_DISH

  // Once the alignment run publishes, there is exactly one correct pair here --
  // the doctor should not have to find it in a list that also carries smoke
  // files, previous overlays and unaligned originals. Picking the wrong pair
  // yields a wrong report rather than an error, so a default is a safety
  // feature, not just a convenience. Functional updates keep a choice the
  // doctor already made; autoFilled makes this a one-shot, so clearing a select
  // afterwards is not undone on the next slide-list refetch.
  //
  // Showing the slide is part of the default, not a flourish: framing a ROI
  // needs viewRect, which only exists while a single slide is on screen. Filling
  // the select alone would leave 框選範圍 disabled under a hint telling the
  // doctor to pick the IHC slide they can see is already picked -- and
  // re-picking the same option fires no change event, so that is a dead end.
  useEffect(() => {
    if (autoFilled.current || !hasAlignedPair) return
    autoFilled.current = true
    setIhc((prev) => prev || ALIGNED_IHC)
    setDish((prev) => prev || ALIGNED_DISH)
    onViewSlide?.(ALIGNED_IHC)
  }, [hasAlignedPair, onViewSlide])

  return (
    <div className="flex flex-col gap-4 text-sm text-neutral-200">
      {/* Never folded automatically: this is where the doctor works, and the two
          stages above fold precisely so that it is the section left open. */}
      <CollapsibleSection
        title="Hybrid 細胞分析"
        summary={ihc && dish ? `${ihc} + ${dish}` : '尚未選擇切片'}
        open={analysisOpen}
        onToggle={() => setAnalysisOpen((v) => !v)}
      >

        <label className="mb-2 flex flex-col gap-1 text-xs text-neutral-400">
          HER2 染色切片
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

        {usingAligned && (
          <p className="mb-2 text-xs text-emerald-400">已自動選用剛完成的對齊結果，需要時可改選</p>
        )}
        {ihc && dish && ihc === dish && (
          <p className="mb-2 text-xs text-amber-300">HER2 與 DISH 不能選同一張切片</p>
        )}
        {/* The pipeline requires the pair to be pixel-aligned and identical in
            size -- PrecutStream raises on a mismatch (m0_reader.py:113). Say so
            before a multi-hour job dies on it. */}
        <p className="mb-3 text-xs text-neutral-500">兩張切片必須來自同一次對齊的結果。</p>

        <div className="mb-3 rounded-md border border-neutral-800 p-2">
          <div className="mb-1 flex items-center justify-between">
            <span className="text-xs font-medium text-neutral-400">分析範圍</span>
            <span className="text-xs text-neutral-500">
              {roi ? `約 ${estimateTiles(roi.w, roi.h)} 個影像區塊` : '整張切片'}
            </span>
          </div>

          {!roi && (
            <p className="mb-2 text-xs text-amber-300/80">
              分析整張切片需要好幾個小時；建議先放大到病灶位置，再框選要分析的範圍。
            </p>
          )}

          <div className="mb-2 flex gap-2">
            <button
              type="button"
              disabled={busy || !viewRect}
              onClick={() => viewRect && setRoi(boxFromView(viewRect))}
              className="flex-1 rounded-md border border-neutral-700 bg-neutral-800 px-2 py-1 text-xs hover:bg-neutral-700 disabled:opacity-40"
            >
              {roi ? '重設選取框' : '框選範圍'}
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
              請先在上方選擇 HER2 染色切片，在右側畫面移動、縮放到目標位置後再框選。
            </p>
          )}
          {roi && (
            <p className="mb-2 text-xs text-neutral-500">
              藍框可直接拖曳移動，右下角的把手可調整大小。
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
            <p className="mt-1 text-xs text-red-400">選取範圍的寬和高都要至少 {TILE_PX} 像素</p>
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
      </CollapsibleSection>

      {jobId && (
        <section className="rounded-lg border border-neutral-800 bg-neutral-900/60 p-3">
          <div className="mb-2 flex items-center justify-between">
            <span className="text-xs font-semibold text-neutral-400">執行狀態</span>
            {status && (
              <span
                className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                  STATUS_STYLE[status] ?? 'bg-neutral-700 text-neutral-300'
                }`}
              >
                {STATUS_TEXT[status] ?? status}
              </span>
            )}
          </div>
          {startedAt !== null && (
            <p className="text-xs text-neutral-500">已經過 {formatElapsed(elapsed)}</p>
          )}

          {running && (
            <>
              <p className="mt-2 text-xs text-violet-300">分析進行中，可以離開或關閉頁面，回來後會自動接續</p>
              <div className="mt-2 h-1 overflow-hidden rounded-full bg-neutral-800">
                <div className="animate-indeterminate h-full w-1/3 rounded-full bg-violet-500" />
              </div>
            </>
          )}
          {job.data?.error && (
            <p className="mt-2 break-all text-xs text-red-400">{job.data.error}</p>
          )}
          {job.data?.metadata && (
            <p className="mt-2 text-xs text-emerald-400">
              已完成 {String((job.data.metadata as Record<string, unknown>).success ?? '?')} 個影像區塊
              {Number((job.data.metadata as Record<string, unknown>).skipped ?? 0) > 0 &&
                `（略過 ${String((job.data.metadata as Record<string, unknown>).skipped)} 個）`}
            </p>
          )}

          <button
            type="button"
            onClick={() => setShowStages((v) => !v)}
            className="mt-2 text-xs text-neutral-500 underline-offset-2 hover:underline"
          >
            {showStages ? '隱藏細節' : '顯示細節'}
          </button>
          {showStages && (
            <>
              <ol className="mt-2 flex list-decimal flex-col gap-1 pl-4 text-xs text-neutral-500">
                {STAGES.map((stage) => (
                  <li key={stage}>{stage}</li>
                ))}
              </ol>
              <p className="mt-2 break-all text-[10px] text-neutral-600">任務編號 {jobId}</p>
            </>
          )}
        </section>
      )}

      {(result.data?.summary || result.data?.has_report) && (
        <section className="rounded-lg border border-neutral-800 bg-neutral-900/60 p-3">
          <h3 className="mb-2 text-xs font-semibold text-neutral-400">判讀結果</h3>

          {report ? (
            <>
              {/* The verdict is the one line a doctor reads first, so it is the
                  only thing here that carries colour. */}
              <div
                className={`rounded-md px-3 py-2 text-sm font-medium ${
                  report.amplified === true
                    ? 'bg-red-500/15 text-red-300'
                    : report.amplified === false
                      ? 'bg-emerald-500/15 text-emerald-300'
                      : 'bg-neutral-700/40 text-neutral-200'
                }`}
              >
                {report.verdict}
              </div>
              <p className="mt-1 text-[10px] text-neutral-600">ASCO/CAP 2013 判讀標準</p>

              <dl className="mt-2 grid grid-cols-2 gap-1">
                {report.metrics.map((m) => (
                  <div key={m.label} className="rounded-md bg-neutral-800/50 px-2 py-1.5">
                    <dt className="text-[10px] leading-tight text-neutral-500">{m.label}</dt>
                    <dd className="font-mono text-sm text-neutral-100">{m.value}</dd>
                  </div>
                ))}
              </dl>

              {report.distribution.length > 0 && (
                <>
                  <p className="mt-3 text-[10px] font-medium text-neutral-500">
                    細胞分佈（佔有效細胞比例）
                  </p>
                  <ul className="mt-1 flex flex-col gap-0.5">
                    {report.distribution.map((d) => (
                      <li key={d.label} className="flex items-center gap-2 text-[11px]">
                        <span className="w-24 shrink-0 text-neutral-400">{d.label}</span>
                        <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-neutral-800">
                          <span
                            className="block h-full rounded-full bg-neutral-500"
                            style={{ width: d.share }}
                          />
                        </span>
                        <span className="w-20 shrink-0 text-right font-mono text-neutral-400">
                          {d.count}（{d.share}）
                        </span>
                      </li>
                    ))}
                  </ul>
                </>
              )}
            </>
          ) : (
            // A summary this build cannot parse is still the result the doctor
            // needs; show it verbatim rather than nothing.
            result.data?.summary && (
              <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-all text-[11px] leading-relaxed text-neutral-300">
                {result.data.summary}
              </pre>
            )
          )}

          <div className="mt-3 flex flex-col gap-1 text-xs">
            {result.data?.overlay_slide_id && (
              <button
                type="button"
                onClick={() => onViewSlide?.(result.data!.overlay_slide_id!)}
                className="self-start text-sky-400 hover:underline"
              >
                在檢視器開啟標註影像
              </button>
            )}
            {result.data?.summary && (
              <a href="/api/hybrid/summary.txt" download className="text-sky-400 hover:underline">
                下載判讀摘要（summary.txt）
              </a>
            )}
            {result.data?.has_report && (
              <a href="/api/hybrid/report.csv" download className="text-sky-400 hover:underline">
                下載逐細胞資料（report.csv）
              </a>
            )}
          </div>

          {result.data?.overlay_slide_id && (
            <>
              <button
                type="button"
                onClick={() => setShowLegend((v) => !v)}
                className="mt-3 text-xs text-neutral-500 underline-offset-2 hover:underline"
              >
                {showLegend ? '隱藏標註說明' : '標註說明'}
              </button>
              {showLegend && (
                <ul className="mt-2 grid grid-cols-2 gap-x-2 gap-y-1">
                  {OVERLAY_LEGEND.map((l) => (
                    <li key={l.label} className="flex items-center gap-1.5 text-[10px] text-neutral-400">
                      <span
                        className={`h-2.5 w-2.5 shrink-0 rounded-sm ${l.ring ? 'ring-1 ring-neutral-600' : ''}`}
                        style={{ backgroundColor: l.color }}
                      />
                      {l.label}
                    </li>
                  ))}
                </ul>
              )}
            </>
          )}
        </section>
      )}
    </div>
  )
}
