import { useEffect, useRef, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import * as tus from 'tus-js-client'

import { readCziLabel } from '../lib/cziLabel'

/**
 * Uploads the three CZI modalities via the tus resumable-upload protocol
 * (tus-js-client -> the backend's tuspyserver router at /api/alignment/tus).
 * tus handles chunking, retry and mid-transfer resume, so an interrupted chunk
 * continues from the server's offset instead of failing the whole upload.
 *
 * Each modality has its own file picker, so the three slides of a case may live
 * in three different folders and nothing is ever inferred from a filename: in
 * production the files are not named HER2/DISH/HE, and the scanner writes no
 * stain information into the CZI either (all three carry one identical
 * `TL Brightfield` channel), so a guess would be a coin flip on which pair of
 * slides gets registered. The doctor assigns every file by hand.
 *
 * What each pick shows instead is the slide's own label photo, read straight out
 * of the chosen file (see lib/cziLabel.ts) -- the stain is printed or written on
 * the glass, so the doctor can see they picked the slide they meant before a
 * multi-GB upload starts. What goes on the wire is unchanged: still three uploads
 * tagged with `modality`, which is all the backend ever reads.
 */
const MODALITIES = [
  { key: 'her2', label: 'HER2 染色切片' },
  { key: 'dish', label: 'DISH 染色切片' },
  { key: 'he', label: 'HE 染色切片' },
] as const
type ModalityKey = (typeof MODALITIES)[number]['key']

/** The label photo of a picked file: read lazily, and absent for a file whose
 *  label the scanner never stored. Never blocks the upload -- it is a
 *  confirmation aid, not a validation step. */
type LabelPreview = { state: 'reading' } | { state: 'none' } | { state: 'ok'; url: string }

const CHUNK_SIZE = 500 * 1024 * 1024
const VERY_LARGE_FILE_BYTES = 10 * 1024 * 1024 * 1024
const INITIAL_PROGRESS: Record<ModalityKey, number> = { her2: 0, dish: 0, he: 0 }
const INITIAL_DONE: Record<ModalityKey, boolean> = { her2: false, dish: false, he: false }
const NO_FILES: Record<ModalityKey, File | null> = { her2: null, dish: null, he: null }

// The run_id is the name of the job folder the backend creates under its
// STORAGE_DIR, so it must match the server's rule in schemas/alignment.py --
// checking here just turns a 422 into an inline hint.
const RUN_ID_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/

function formatBytes(bytes: number) {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`
}

/** The picked slide's label photo. Kept small and non-blocking: a file whose
 *  label cannot be read is still perfectly uploadable, so this degrades to a
 *  note rather than an error. */
function LabelThumb({ preview, alt }: { preview: LabelPreview | undefined; alt: string }) {
  // Wide enough that the stain printed on the glass is actually readable -- a
  // preview too small to read is not a confirmation of anything.
  if (preview?.state === 'ok') {
    return (
      <img
        src={preview.url}
        alt={alt}
        title={alt}
        className="w-36 shrink-0 rounded border border-neutral-700 bg-neutral-900"
      />
    )
  }
  return (
    <span className="flex h-24 w-36 shrink-0 items-center justify-center rounded border border-dashed border-neutral-700 text-center text-[10px] leading-tight text-neutral-600">
      {preview?.state === 'reading' ? '讀取標籤…' : '無標籤'}
    </span>
  )
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
  onBusyChange,
}: {
  disabled?: boolean
  onUploaded: (runId: string) => void
  existingRunIds: string[]
  /** Raised while a transfer is in flight. The panel above forwards it so no
   *  pipeline step can start on CZIs that are still arriving. */
  onBusyChange?: (busy: boolean) => void
}) {
  const [files, setFiles] = useState<Record<ModalityKey, File | null>>(NO_FILES)
  const [labels, setLabels] = useState<Partial<Record<ModalityKey, LabelPreview>>>({})
  // The file each picker most recently produced. A label read is async, so a
  // second pick while the first is still decoding must not paint the old slide's
  // label under the new file -- showing the wrong label is the exact mistake this
  // preview exists to catch.
  const latestPick = useRef<Record<ModalityKey, File | null>>(NO_FILES)
  // Prefilled so the common case is one keystroke-free upload; the collision
  // warning below covers a second job on the same day.
  const [runId, setRunId] = useState(() => `run-${new Date().toISOString().slice(0, 10)}`)
  const [progress, setProgress] = useState(INITIAL_PROGRESS)
  const [done, setDone] = useState(INITIAL_DONE)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const chosen = MODALITIES.map(({ key }) => files[key]).filter((f): f is File => f !== null)
  const allChosen = chosen.length === MODALITIES.length
  // Three separate pickers make it possible to point two modalities at one file,
  // which would register a slide against itself; block rather than warn. Each
  // picker yields its own File object even for the same path, so compare on what
  // identifies the file rather than on object identity.
  const duplicated =
    new Set(chosen.map((f) => `${f.name}:${f.size}:${f.lastModified}`)).size !== chosen.length
  const validRunId = RUN_ID_RE.test(runId)
  const overwriting = validRunId && existingRunIds.includes(runId)
  const largeFiles = MODALITIES.filter(({ key }) => (files[key]?.size ?? 0) >= VERY_LARGE_FILE_BYTES)

  const pick = (key: ModalityKey) => (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] ?? null
    latestPick.current = { ...latestPick.current, [key]: file }
    setFiles((prev) => ({ ...prev, [key]: file }))
    setProgress((prev) => ({ ...prev, [key]: 0 }))
    setDone((prev) => ({ ...prev, [key]: false }))
    setLabels((prev) => ({ ...prev, [key]: file ? { state: 'reading' } : undefined }))
    setUploadError(null)
    if (!file) return
    readCziLabel(file).then((url) => {
      if (latestPick.current[key] !== file) return
      setLabels((prev) => ({ ...prev, [key]: url ? { state: 'ok', url } : { state: 'none' } }))
    })
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
    setUploadError('上傳已取消。重新上傳時會從中斷的地方接續，不會從頭開始。')
  }

  const uploading = upload.isPending
  useEffect(() => {
    onBusyChange?.(uploading)
  }, [uploading, onBusyChange])

  // The heading lives on the CollapsibleSection wrapper in PipelinePanel, which
  // folds this away once a run exists to work on.
  return (
    <div className="flex flex-col gap-2">
      <label className="flex flex-col gap-1 text-xs text-neutral-400">
        工作名稱
        <input
          type="text"
          value={runId}
          onChange={(e) => setRunId(e.target.value)}
          disabled={disabled || upload.isPending}
          className="rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 font-mono text-xs text-neutral-200"
        />
      </label>
      {!validRunId && (
        <p className="text-xs text-red-400">名稱只能使用英文、數字、- 或 _，且需以英文或數字開頭</p>
      )}
      {overwriting && (
        <p className="text-xs text-amber-300">「{runId}」已經存在，上傳會覆蓋原本的影像</p>
      )}
      <p className="text-xs text-neutral-500">
        三張切片分別選取，可以來自不同資料夾。選好後會顯示玻片標籤，請核對是否為正確的切片。
      </p>
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
            <>
              <div className="flex items-center gap-2 text-[16px] text-neutral-500">
                <progress className="h-1.5 flex-1" max={100} value={progress[key]} />
                <span className={done[key] ? 'text-emerald-400' : undefined}>
                  {done[key] ? '✓ 完成' : progress[key] >= 100 ? '處理中…' : `${progress[key].toFixed(1)}%`}
                </span>
                <span>{formatBytes(files[key]!.size)}</span>
              </div>
              <div className="flex items-start gap-2">
                <LabelThumb preview={labels[key]} alt={`${label} 玻片標籤`} />
                <span className="truncate font-mono text-xs text-neutral-300" title={files[key]!.name}>
                  {files[key]!.name}
                </span>
              </div>
            </>
          )}
        </label>
      ))}
      {duplicated && (
        <p className="text-xs text-red-400">同一個檔案被選了兩次，請改成三個不同的檔案</p>
      )}
      {largeFiles.length > 0 && (
        <p className="text-xs text-amber-300">
          {largeFiles.map(({ label }) => label).join('、')} 檔案較大，上傳需要較長時間。
        </p>
      )}
      <button
        type="button"
        disabled={disabled || upload.isPending || !allChosen || duplicated || !validRunId}
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
      {upload.isSuccess && <p className="text-xs text-emerald-400">上傳完成，可以開始對齊</p>}
      {uploadError && <p className="text-xs text-red-400">{uploadError}</p>}
    </div>
  )
}
