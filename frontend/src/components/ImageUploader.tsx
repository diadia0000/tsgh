import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import * as tus from 'tus-js-client'

/**
 * Uploads the three CZI modalities via the tus resumable-upload protocol
 * (tus-js-client -> the backend's tuspyserver router at /api/alignment/tus).
 * tus handles chunking, retry and mid-transfer resume, so an interrupted chunk
 * continues from the server's offset instead of failing the whole upload.
 */
const MODALITIES = [
  { key: 'her2', label: 'HER2 (CZI)' },
  { key: 'dish', label: 'DISH (CZI)' },
  { key: 'he', label: 'HE (CZI)' },
] as const
type ModalityKey = (typeof MODALITIES)[number]['key']

const CHUNK_SIZE = 500 * 1024 * 1024
const VERY_LARGE_FILE_BYTES = 10 * 1024 * 1024 * 1024
const INITIAL_PROGRESS: Record<ModalityKey, number> = { her2: 0, dish: 0, he: 0 }
const INITIAL_DONE: Record<ModalityKey, boolean> = { her2: false, dish: false, he: false }

// The run_id is the name of the job folder the backend creates under its
// STORAGE_DIR, so it must match the server's rule in schemas/alignment.py --
// checking here just turns a 422 into an inline hint.
const RUN_ID_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/

function formatBytes(bytes: number) {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`
}

function uploadOne(file: File, runId: string, modality: ModalityKey, onPercent: (p: number) => void) {
  return new Promise<void>((resolve, reject) => {
    const upload = new tus.Upload(file, {
      endpoint: '/api/alignment/tus',
      chunkSize: CHUNK_SIZE,
      retryDelays: [0, 1000, 3000, 5000, 10000],
      removeFingerprintOnSuccess: true,
      // tus's default fingerprint is file+endpoint only, so an interrupted
      // upload would resume under the job it was *started* in even after the
      // user picks a different one. Scope it to the job folder it targets.
      fingerprint: async (f) => `tsgh-${runId}-${modality}-${f.name}-${f.size}-${f.lastModified}`,
      metadata: {
        filename: file.name,
        filetype: file.type || 'application/octet-stream',
        run_id: runId,
        modality,
      },
      onError: reject,
      onProgress: (sent, total) => onPercent(total ? (sent / total) * 100 : 0),
      onSuccess: () => resolve(),
    })
    // Resume across reloads / prior attempts when tus remembers this file.
    // .catch(reject): a rejected findPreviousUploads/resume (stale resume URL,
    // storage disabled) must fail the upload, not leave the promise pending
    // forever — an unsettled promise here freezes the mutation on "上傳中…".
    upload.findPreviousUploads().then((prev) => {
      if (prev.length) upload.resumeFromPreviousUpload(prev[0])
      upload.start()
    }).catch(reject)
    activeUploads.current.push(upload)
  })
}

// Module-scoped so the cancel button can abort in-flight uploads.
const activeUploads: { current: tus.Upload[] } = { current: [] }

export function ImageUploader({
  disabled,
  onUploaded,
  existingRunIds,
}: {
  disabled?: boolean
  onUploaded: (runId: string) => void
  existingRunIds: string[]
}) {
  const [files, setFiles] = useState<Record<ModalityKey, File | null>>({ her2: null, dish: null, he: null })
  // Prefilled so the common case is one keystroke-free upload; the collision
  // warning below covers a second job on the same day.
  const [runId, setRunId] = useState(() => `run-${new Date().toISOString().slice(0, 10)}`)
  const [progress, setProgress] = useState(INITIAL_PROGRESS)
  const [done, setDone] = useState(INITIAL_DONE)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const allChosen = MODALITIES.every(({ key }) => files[key])
  const validRunId = RUN_ID_RE.test(runId)
  const overwriting = validRunId && existingRunIds.includes(runId)
  const largeFiles = MODALITIES.filter(({ key }) => (files[key]?.size ?? 0) >= VERY_LARGE_FILE_BYTES)

  const pick = (key: ModalityKey) => (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] ?? null
    if (file && !file.name.toLowerCase().endsWith('.czi')) {
      e.target.value = ''
      return
    }
    setFiles((prev) => ({ ...prev, [key]: file }))
    setProgress((prev) => ({ ...prev, [key]: 0 }))
    setDone((prev) => ({ ...prev, [key]: false }))
    setUploadError(null)
  }

  const upload = useMutation({
    mutationFn: async () => {
      setUploadError(null)
      setDone(INITIAL_DONE)
      activeUploads.current = []

      // Sequential: each file finishes before the next starts, so the link
      // only ever carries one multi-GB transfer at a time.
      for (const { key } of MODALITIES) {
        // await resolves on tus onSuccess = server acknowledged, not just sent.
        await uploadOne(files[key]!, runId, key, (p) => setProgress((prev) => ({ ...prev, [key]: p })))
        setDone((prev) => ({ ...prev, [key]: true }))
      }
      return runId
    },
    onError: (error) => {
      setUploadError(error instanceof Error ? error.message : '上傳失敗，請重試')
    },
    onSuccess: onUploaded,
  })

  const cancel = () => {
    activeUploads.current.forEach((u) => u.abort())
    upload.reset()
    setUploadError('上傳已取消；重試時會從伺服器已收到的位移繼續。')
  }

  return (
    <section>
      <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-neutral-400">
        上傳影像 (CZI)
      </h2>
      <div className="flex flex-col gap-2">
        <label className="flex flex-col gap-1 text-xs text-neutral-400">
          工作名稱（伺服器上的資料夾）
          <input
            type="text"
            value={runId}
            onChange={(e) => setRunId(e.target.value)}
            disabled={disabled || upload.isPending}
            className="rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 font-mono text-xs text-neutral-200"
          />
        </label>
        {!validRunId && (
          <p className="text-xs text-red-400">名稱只能用英數字、- 或 _，需以英數字開頭，最多 64 字</p>
        )}
        {overwriting && (
          <p className="text-xs text-amber-300">「{runId}」已存在，上傳會覆寫該工作的影像</p>
        )}
        {MODALITIES.map(({ key, label }) => (
          <label key={key} className="flex flex-col gap-1 text-xs text-neutral-400">
            {label}
            <input
              type="file"
              accept=".czi"
              onChange={pick(key)}
              disabled={disabled || upload.isPending}
              className="text-xs text-neutral-300"
            />
            {files[key] && (
              <div className="flex items-center gap-2 text-[16px] text-neutral-500">
                <progress className="h-1.5 flex-1" max={100} value={progress[key]} />
                <span className={done[key] ? 'text-emerald-400' : undefined}>
                  {done[key] ? '✓ 完成' : progress[key] >= 100 ? '處理中…' : `${progress[key].toFixed(1)}%`}
                </span>
                <span>{formatBytes(files[key]!.size)}</span>
              </div>
            )}
          </label>
        ))}
        {largeFiles.length > 0 && (
          <p className="text-xs text-amber-300">
            大型檔案提醒：{largeFiles.map(({ label }) => label).join('、')} 將需要較長的上傳時間；系統不會因檔案大小阻擋上傳。
          </p>
        )}
        <button
          type="button"
          disabled={disabled || upload.isPending || !allChosen || !validRunId}
          onClick={() => upload.mutate()}
          className="rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 font-medium hover:bg-neutral-700 disabled:opacity-40"
        >
          {upload.isPending ? '上傳中…' : upload.isError ? '重試上傳' : '上傳影像'}
        </button>
        {upload.isPending && (
          <button
            type="button"
            onClick={cancel}
            className="rounded-md border border-red-800 px-3 py-2 text-red-300 hover:bg-red-950"
          >
            取消上傳
          </button>
        )}
        {upload.isSuccess && <p className="text-xs text-emerald-400">上傳完成，可執行流程</p>}
        {uploadError && <p className="text-xs text-red-400">{uploadError}</p>}
      </div>
    </section>
  )
}
