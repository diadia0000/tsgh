import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '../api/client'
import { useJob } from '../api/jobs'

type AlignStep = 'preprocess' | 'align' | 'roi-eval' | 'thumbnail'
const ALIGN_STEPS: { step: AlignStep; label: string; path: string }[] = [
  { step: 'preprocess', label: '前處理', path: '/api/alignment/preprocess' },
  { step: 'align', label: '對齊', path: '/api/alignment/align' },
  { step: 'roi-eval', label: 'ROI 評估', path: '/api/alignment/roi-eval' },
  { step: 'thumbnail', label: '疊合縮圖', path: '/api/alignment/thumbnail' },
]

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
 */
export function PipelinePanel() {
  const queryClient = useQueryClient()
  const [jobId, setJobId] = useState<string | null>(null)
  const [startedAt, setStartedAt] = useState<number | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const job = useJob(jobId)

  // Start a fresh job: reset the elapsed clock so it counts from submission.
  const startJob = (id: string) => {
    setJobId(id)
    setStartedAt(Date.now())
    setElapsed(0)
  }

  const align = useMutation({
    mutationFn: async (path: string) => {
      const { data, error } = await api.POST(path as '/api/alignment/align', {
        body: {}, // all fields optional → backend uses RegistrationConfig defaults
      })
      if (error) throw new Error('submit failed')
      return data.job_id
    },
    onSuccess: startJob,
  })

  // Hybrid inputs are chosen from the slide list, never typed as filesystem
  // paths (guardrail 2); the backend resolves the ids under SLIDES_DIR.
  const slides = useQuery({
    queryKey: ['slides'],
    queryFn: async () => {
      const { data, error } = await api.GET('/api/tiles')
      if (error) throw new Error('slide list failed')
      return data
    },
  })
  const [ihcId, setIhcId] = useState('')
  const [dishId, setDishId] = useState('')
  const hybrid = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST('/api/hybrid/tile', {
        body: { ihc_slide_id: ihcId, dish_slide_id: dishId },
      })
      if (error) throw new Error('submit failed')
      return data.job_id
    },
    onSuccess: startJob,
  })

  const busy = align.isPending || hybrid.isPending
  const status = job.data?.status
  const running = status === 'running' || status === 'pending'

  // Tick the elapsed clock while the job runs; it freezes at the final value
  // once the job reaches a terminal state (the interval stops).
  useEffect(() => {
    if (!startedAt || !running) return
    const t = setInterval(() => setElapsed((Date.now() - startedAt) / 1000), 250)
    return () => clearInterval(t)
  }, [startedAt, running])

  // A finished job may have written a new viewable slide; refresh the picker so
  // it (and the Hybrid dropdowns) pick it up without a manual reload.
  useEffect(() => {
    if (status === 'done') queryClient.invalidateQueries({ queryKey: ['slides'] })
  }, [status, queryClient])

  return (
    <div className="flex flex-col gap-6 text-sm text-neutral-200">
      <section>
        <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-neutral-400">
          影像對齊 Pipeline
        </h2>
        <div className="grid grid-cols-2 gap-2">
          {ALIGN_STEPS.map(({ step, label, path }) => (
            <button
              key={step}
              type="button"
              disabled={busy}
              onClick={() => align.mutate(path)}
              className="rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 font-medium hover:bg-neutral-700 disabled:opacity-40"
            >
              {label}
            </button>
          ))}
        </div>
      </section>

      <section>
        <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-neutral-400">
          Hybrid 細胞分割 (IHC · DISH)
        </h2>
        <div className="flex flex-col gap-2">
          <label className="flex flex-col gap-1 text-xs text-neutral-400">
            IHC 切片
            <select
              value={ihcId}
              onChange={(e) => setIhcId(e.target.value)}
              className="rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-sm text-neutral-200"
            >
              <option value="">選擇切片…</option>
              {slides.data?.map((id) => (
                <option key={id} value={id}>
                  {id}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-xs text-neutral-400">
            DISH 切片
            <select
              value={dishId}
              onChange={(e) => setDishId(e.target.value)}
              className="rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-sm text-neutral-200"
            >
              <option value="">選擇切片…</option>
              {slides.data?.map((id) => (
                <option key={id} value={id}>
                  {id}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            disabled={busy || !ihcId || !dishId}
            onClick={() => hybrid.mutate()}
            className="rounded-md bg-violet-600 px-3 py-2 font-medium hover:bg-violet-500 disabled:opacity-40"
          >
            執行 Hybrid Pipeline
          </button>
        </div>
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
              {job.data?.message && (
                <p className="mt-2 truncate text-xs text-sky-300" title={job.data.message}>
                  {job.data.message}
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
    </div>
  )
}
