import { useCallback, useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '../api/client'
import { useJob } from '../api/jobs'
import type { components } from '../api/schema'
import { ImageUploader } from './ImageUploader'

type AlignStep = 'preprocess' | 'align' | 'roi-eval' | 'thumbnail'
const ALIGN_STEPS: { step: AlignStep; label: string }[] = [
  { step: 'preprocess', label: '前處理' },
  { step: 'align', label: '對齊' },
  { step: 'roi-eval', label: 'ROI 評估' },
  { step: 'thumbnail', label: '疊合縮圖' },
]
type AlignmentConfig = components['schemas']['AlignmentConfigIn']
type RunSummary = components['schemas']['RunSummary']

// The backend derives every server-side path (czi_input / output) from run_id,
// so the client only carries the run_id the upload returned -- no filesystem
// paths cross the wire (guardrail 2). Modality/reference defaults live in the
// backend's RegistrationConfig.

const STATUS_STYLE: Record<string, string> = {
  pending: 'bg-amber-500/15 text-amber-300',
  running: 'bg-sky-500/15 text-sky-300',
  done: 'bg-emerald-500/15 text-emerald-300',
  error: 'bg-red-500/15 text-red-300',
}

/**
 * Triggers the alignment and hybrid pipelines and tracks the resulting job.
 * Every endpoint returns { job_id } immediately (long tasks run in the
 * backend's BackgroundTasks); we then poll via useJob. One active job at a time
 * keeps Phase 3 minimal.
 *
 * On load the newest run on the server is adopted automatically, so a reload --
 * or a backend restart, which wipes the in-memory job registry -- resumes from
 * the first step whose artifacts are missing instead of re-running everything.
 */
export function PipelinePanel({ onResultSlide }: { onResultSlide?: (slideId: string) => void }) {
  const queryClient = useQueryClient()
  // The run the panel is working on, plus the steps already finished for it.
  // Null until an upload or the auto-adopt below supplies one; the pipeline
  // buttons stay disabled while it is, since every step needs a run_id.
  const [run, setRun] = useState<{ runId: string; done: string[] } | null>(null)
  const [restart, setRestart] = useState(false)
  const [pipelineStepIndex, setPipelineStepIndex] = useState<number | null>(null)
  const [pipelineError, setPipelineError] = useState<string | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [startedAt, setStartedAt] = useState<number | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const handledPipelineJobs = useRef(new Set<string>())
  const adopted = useRef(false)
  const job = useJob(jobId)

  // Start a fresh job: reset the elapsed clock so it counts from submission.
  const startJob = (id: string) => {
    setJobId(id)
    setStartedAt(Date.now())
    setElapsed(0)
  }

  // Every job folder on the server, newest first, with the steps it already has
  // artifacts for. The run directory is the only durable progress record
  // (jobs.py's registry is in-memory), so this is what "找回進度" reads.
  const runs = useQuery<RunSummary[]>({
    queryKey: ['runs'],
    queryFn: async () => {
      const { data, error } = await api.GET('/api/alignment/runs')
      if (error) throw new Error('run lookup failed')
      return data
    },
  })

  // Re-point the viewer at a finished run's TIFFs. `aligned_result` is one
  // global slide_id, so a resumed run has to reclaim it -- no recomputation.
  // Also what re-establishes the server-side pointer after a backend restart.
  const publish = useMutation({
    mutationFn: async (runId: string) => {
      const { error } = await api.POST('/api/alignment/publish', { body: { run_id: runId } })
      if (error) throw new Error('publish failed')
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['slides'] })
      onResultSlide?.('aligned_result')
    },
  })
  const publishRun = publish.mutate

  const align = useMutation({
    mutationFn: async ({ step, config }: { step: AlignStep; config: AlignmentConfig }) => {
      let data: { job_id: string } | undefined
      let error: unknown
      switch (step) {
        case 'preprocess':
          ;({ data, error } = await api.POST('/api/alignment/preprocess', { body: config }))
          break
        case 'align':
          ;({ data, error } = await api.POST('/api/alignment/align', { body: config }))
          break
        case 'roi-eval':
          ;({ data, error } = await api.POST('/api/alignment/roi-eval', { body: config }))
          break
        case 'thumbnail':
          ;({ data, error } = await api.POST('/api/alignment/thumbnail', { body: config }))
          break
      }
      if (error) throw new Error('submit failed')
      if (!data) throw new Error('submit failed')
      return data.job_id
    },
    onSuccess: startJob,
  })

  const status = job.data?.status
  const jobLookupError = job.error
  const running = status === 'running' || status === 'pending'
  // Combined with `running` (not just the mutations' own isPending) so a
  // second job can't be started while a prior one is still processing in the
  // background -- these all race on the same fixed storage paths.
  const busy = align.isPending || running || pipelineStepIndex !== null

  // First step without artifacts on disk; -1 once the run is complete. Ticking
  // `restart` re-runs from the top, the only way to redo a step whose output
  // turned out bad.
  const nextIndex = run ? ALIGN_STEPS.findIndex((s) => !run.done.includes(s.step)) : 0
  const complete = nextIndex === -1
  const startIndex = restart || complete ? 0 : nextIndex

  // A just-created folder isn't in the server list until it refetches; keep it
  // in the picker so the selection is never blank.
  const serverRuns = runs.data ?? []
  const runOptions =
    run && !serverRuns.some((r) => r.run_id === run.runId)
      ? [{ run_id: run.runId, done: run.done, job_id: null }, ...serverRuns]
      : serverRuns

  const submitAlignmentStep = useCallback((index: number, config: AlignmentConfig) => {
    // Discard the completed job before submitting the next request. Otherwise
    // its cached `done` state could be mistaken for the newly selected step.
    setJobId(null)
    setPipelineStepIndex(index)
    align.mutate(
      { step: ALIGN_STEPS[index].step, config },
      {
        onError: (error) => {
          setPipelineStepIndex(null)
          setPipelineError(error instanceof Error ? error.message : '送出任務失敗')
        },
      },
    )
  }, [align])

  const runPipeline = () => {
    if (!run) return
    handledPipelineJobs.current.clear()
    setPipelineError(null)
    submitAlignmentStep(startIndex, { run_id: run.runId })
  }

  // Switch the panel to a job folder, picking up wherever that run left off.
  const adopt = useCallback((summary: RunSummary) => {
    setRun({ runId: summary.run_id, done: summary.done })
    setRestart(false)
    setPipelineError(null)
    handledPipelineJobs.current.clear()
    if (summary.job_id) {
      // A step is still executing server-side; only the browser's copy of its
      // id was lost. Re-attach so polling and the step chain carry on.
      startJob(summary.job_id)
      setPipelineStepIndex(Math.min(summary.done.length, ALIGN_STEPS.length - 1))
      return
    }
    setJobId(null)
    setPipelineStepIndex(null)
    // Finished run: `aligned_result` is one global slide_id, so reclaim it.
    if (summary.done.length === ALIGN_STEPS.length) publishRun(summary.run_id)
  }, [publishRun])

  // Pick up the newest run on load, once. No button for the common case: the
  // user's work is whatever they last ran, and the server knows how far it got.
  useEffect(() => {
    const newest = runs.data?.[0]
    if (adopted.current || !newest) return
    adopted.current = true
    adopt(newest)
  }, [runs.data, adopt])

  // Tick the elapsed clock while the job runs; it freezes at the final value
  // once the job reaches a terminal state (the interval stops).
  useEffect(() => {
    if (!startedAt || !running) return
    const t = setInterval(() => setElapsed((Date.now() - startedAt) / 1000), 250)
    return () => clearInterval(t)
  }, [startedAt, running])

  useEffect(() => {
    if (status === 'done') queryClient.invalidateQueries({ queryKey: ['slides'] })
  }, [queryClient, status])

  // Advance only after the current job reaches a terminal state. Each request
  // is submitted from this effect, so the backend never receives parallel
  // alignment steps.
  useEffect(() => {
    if (pipelineStepIndex === null || !jobId || handledPipelineJobs.current.has(jobId)) {
      return
    }
    if (job.isError) {
      handledPipelineJobs.current.add(jobId)
      setPipelineStepIndex(null)
      setPipelineError(jobLookupError instanceof Error ? jobLookupError.message : '無法取得任務狀態')
      return
    }
    if (!status) return
    if (status === 'error') {
      handledPipelineJobs.current.add(jobId)
      setPipelineStepIndex(null)
      setPipelineError(job.data?.error ?? '任務執行失敗')
      return
    }
    if (status !== 'done') return

    handledPipelineJobs.current.add(jobId)
    const finished = ALIGN_STEPS[pipelineStepIndex].step
    setRun((prev) =>
      prev && !prev.done.includes(finished) ? { ...prev, done: [...prev.done, finished] } : prev,
    )
    if (pipelineStepIndex < ALIGN_STEPS.length - 1) {
      if (run) submitAlignmentStep(pipelineStepIndex + 1, { run_id: run.runId })
      return
    }
    setPipelineStepIndex(null)
    setRestart(false)
    const slideId = job.data?.metadata?.slide_id
    if (typeof slideId === 'string' && job.data?.result_path) onResultSlide?.(slideId)
  }, [job.data, job.isError, jobId, jobLookupError, onResultSlide, pipelineStepIndex, run, status, submitAlignmentStep])

  return (
    <div className="flex flex-col gap-6 text-sm text-neutral-200">
      <ImageUploader
        disabled={busy}
        existingRunIds={serverRuns.map((r) => r.run_id)}
        onUploaded={(runId) => {
          adopted.current = true // a fresh upload wins over the auto-adopted run
          // done:[] even when overwriting an existing folder -- new CZIs make
          // every downstream artifact stale, so the whole pipeline must re-run.
          setRun({ runId, done: [] })
          setRestart(false)
          queryClient.invalidateQueries({ queryKey: ['runs'] })
        }}
      />

      <section>
        <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-neutral-400">
          影像對齊 Pipeline
        </h2>
        {!run && (
          <p className="mb-2 text-xs text-neutral-500">
            {runs.isPending ? '檢查既有工作…' : '請先上傳影像後再執行'}
          </p>
        )}
        {run && (
          <>
            <label className="mb-2 flex flex-col gap-1 text-xs text-neutral-400">
              工作資料夾
              <select
                value={run.runId}
                disabled={busy}
                onChange={(e) => {
                  const picked = runOptions.find((r) => r.run_id === e.target.value)
                  if (picked) adopt(picked)
                }}
                className="rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 font-mono text-xs text-neutral-200 disabled:opacity-40"
              >
                {runOptions.map((r) => (
                  <option key={r.run_id} value={r.run_id}>
                    {r.run_id} · {(r.run_id === run.runId ? run.done : r.done).length}/
                    {ALIGN_STEPS.length}
                    {r.job_id ? ' · 執行中' : ''}
                  </option>
                ))}
              </select>
            </label>
            <ul className="mb-2 flex flex-col gap-1 text-xs">
              {ALIGN_STEPS.map(({ step, label }, i) => {
                const isDone = run.done.includes(step)
                const isActive = i === pipelineStepIndex
                return (
                  <li key={step} className="flex items-center gap-2">
                    <span
                      className={
                        isDone ? 'text-emerald-400' : isActive ? 'text-sky-300' : 'text-neutral-600'
                      }
                    >
                      {isDone ? '✓' : isActive ? '⟳' : '○'}
                    </span>
                    <span className={isDone ? 'text-neutral-400' : 'text-neutral-500'}>{label}</span>
                  </li>
                )
              })}
            </ul>
            {nextIndex > 0 && (
              <label className="mb-2 flex items-center gap-2 text-xs text-neutral-500">
                <input
                  type="checkbox"
                  checked={restart}
                  disabled={busy}
                  onChange={(e) => setRestart(e.target.checked)}
                />
                忽略既有結果，從頭重跑
              </label>
            )}
          </>
        )}
        <button
          type="button"
          disabled={busy || !run}
          onClick={runPipeline}
          className="w-full rounded-md bg-sky-600 px-3 py-2 font-medium hover:bg-sky-500 disabled:opacity-40"
        >
          {startIndex === 0
            ? '執行完整流程'
            : `繼續執行（從${ALIGN_STEPS[startIndex].label}開始）`}
        </button>
        {publish.isError && (
          <p className="mt-2 text-xs text-red-400">無法載入既有結果到檢視器</p>
        )}
      </section>

      {jobId && (
        <section className="rounded-lg border border-neutral-800 bg-neutral-900/60 p-3">
          <div className="mb-2 flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wide text-neutral-400">
              任務狀態
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

          {running && (
            <>
              {pipelineStepIndex !== null && (
                <p className="mt-2 text-xs text-sky-300">
                  {pipelineStepIndex + 1}/{ALIGN_STEPS.length} 正在執行{ALIGN_STEPS[pipelineStepIndex].label}…
                </p>
              )}
              {/* Indeterminate: the backend can't report a real %, so we don't
                  fake one — the sliding bar just signals "still working". */}
              <div className="mt-2 h-1 overflow-hidden rounded-full bg-neutral-800">
                <div className="animate-indeterminate h-full w-1/3 rounded-full bg-sky-500" />
              </div>
            </>
          )}
          {job.data?.error && (
            <p className="mt-2 text-xs text-red-400">{job.data.error}</p>
          )}
          {job.data?.result_path && (
            <p className="mt-2 break-all text-xs text-emerald-400">
              結果：{job.data.result_path}
            </p>
          )}
        </section>
      )}
      {pipelineError && <p className="text-xs text-red-400">流程已停止：{pipelineError}</p>}
    </div>
  )
}
