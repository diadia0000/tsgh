import { useCallback, useEffect, useRef, useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
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
 */
export function PipelinePanel({ onResultSlide }: { onResultSlide?: (slideId: string) => void }) {
  const queryClient = useQueryClient()
  // Gate the pipeline steps: they run against the fixed CZI_INPUT_DIR the
  // uploader writes to, so require a successful upload first.
  const [uploaded, setUploaded] = useState(false)
  const [alignmentConfig, setAlignmentConfig] = useState<AlignmentConfig | null>(null)
  const [pipelineStepIndex, setPipelineStepIndex] = useState<number | null>(null)
  const [pipelineError, setPipelineError] = useState<string | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [startedAt, setStartedAt] = useState<number | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const handledPipelineJobs = useRef(new Set<string>())
  const job = useJob(jobId)

  // Start a fresh job: reset the elapsed clock so it counts from submission.
  const startJob = (id: string) => {
    setJobId(id)
    setStartedAt(Date.now())
    setElapsed(0)
  }

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
    if (!alignmentConfig) return
    handledPipelineJobs.current.clear()
    setPipelineError(null)
    submitAlignmentStep(0, alignmentConfig)
  }

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
    if (pipelineStepIndex < ALIGN_STEPS.length - 1) {
      if (alignmentConfig) submitAlignmentStep(pipelineStepIndex + 1, alignmentConfig)
      return
    }
    setPipelineStepIndex(null)
    const slideId = job.data?.metadata?.slide_id
    if (typeof slideId === 'string' && job.data?.result_path) onResultSlide?.(slideId)
  }, [alignmentConfig, job.data, job.isError, jobId, jobLookupError, onResultSlide, pipelineStepIndex, status, submitAlignmentStep])

  return (
    <div className="flex flex-col gap-6 text-sm text-neutral-200">
      <ImageUploader
        disabled={busy}
        onUploaded={(runId) => {
          setUploaded(true)
          setAlignmentConfig({ run_id: runId })
        }}
      />

      <section>
        <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-neutral-400">
          影像對齊 Pipeline
        </h2>
        {!uploaded && (
          <p className="mb-2 text-xs text-neutral-500">請先上傳影像後再執行</p>
        )}
        <button
          type="button"
          disabled={busy || !uploaded}
          onClick={runPipeline}
          className="mb-2 w-full rounded-md bg-sky-600 px-3 py-2 font-medium hover:bg-sky-500 disabled:opacity-40"
        >
          執行完整流程
        </button>
        <div className="grid grid-cols-2 gap-2">
          {ALIGN_STEPS.map(({ step, label }) => (
            <button
              key={step}
              type="button"
              disabled={busy || !uploaded}
              onClick={() => alignmentConfig && align.mutate({ step, config: alignmentConfig })}
              className="rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 font-medium hover:bg-neutral-700 disabled:opacity-40"
            >
              {label}
            </button>
          ))}
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
              {pipelineStepIndex !== null && (
                <p className="mt-2 text-xs text-sky-300">
                  {pipelineStepIndex + 1}/{ALIGN_STEPS.length} 正在執行{ALIGN_STEPS[pipelineStepIndex].label}…
                </p>
              )}
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
      {pipelineError && <p className="text-xs text-red-400">流程已停止：{pipelineError}</p>}
    </div>
  )
}
