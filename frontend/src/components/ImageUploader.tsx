import { useRef, useState } from 'react'
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

const CHUNK_SIZE = 64 * 1024 * 1024
const VERY_LARGE_FILE_BYTES = 10 * 1024 * 1024 * 1024
const INITIAL_PROGRESS: Record<ModalityKey, number> = { her2: 0, dish: 0, he: 0 }

function formatBytes(bytes: number) {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`
}

// RFC-4122 v4 from getRandomValues (available in insecure contexts, unlike
// crypto.randomUUID). Backend requires a parseable UUID (schemas/alignment.py).
function uuidv4() {
  const b = crypto.getRandomValues(new Uint8Array(16))
  b[6] = (b[6] & 0x0f) | 0x40
  b[8] = (b[8] & 0x3f) | 0x80
  const h = [...b].map((x) => x.toString(16).padStart(2, '0'))
  return `${h.slice(0, 4).join('')}-${h.slice(4, 6).join('')}-${h.slice(6, 8).join('')}-${h.slice(8, 10).join('')}-${h.slice(10, 16).join('')}`
}

// A stable run_id per file-set: kept in localStorage so a resumed upload lands
// under the SAME run_id the pipeline is later told to run against.
function runIdFor(files: Record<ModalityKey, File>) {
  const fp = MODALITIES.map(({ key }) => {
    const f = files[key]
    return `${key}:${f.name}:${f.size}:${f.lastModified}`
  }).join('|')
  const storageKey = `czi-run:${fp}`
  let runId: string | null = null
  try {
    runId = localStorage.getItem(storageKey)
  } catch {
    // Storage disabled: fall through to a fresh id; resume across reloads is lost but upload still works.
  }
  // Discard a stale non-UUID id left by an older build; the backend validates
  // run_id with uuid.UUID() and would 500 on the tus on_upload_complete hook.
  if (runId && !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(runId)) {
    runId = null
  }
  // ponytail: crypto.randomUUID is secure-context only (HTTPS/localhost); fall back
  // to getRandomValues (works over plain-HTTP LAN) — must stay a valid UUID v4, the
  // backend validates it with uuid.UUID() (schemas/alignment.py).
  runId ??= crypto.randomUUID?.() ?? uuidv4()
  try {
    localStorage.setItem(storageKey, runId)
  } catch { /* best-effort */ }
  return { runId, clear: () => { try { localStorage.removeItem(storageKey) } catch { /* ignore */ } } }
}

function uploadOne(file: File, runId: string, modality: ModalityKey, onPercent: (p: number) => void) {
  return new Promise<void>((resolve, reject) => {
    const upload = new tus.Upload(file, {
      endpoint: '/api/alignment/tus',
      chunkSize: CHUNK_SIZE,
      retryDelays: [0, 1000, 3000, 5000, 10000],
      removeFingerprintOnSuccess: true,
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
    upload.findPreviousUploads().then((prev) => {
      if (prev.length) upload.resumeFromPreviousUpload(prev[0])
      upload.start()
    })
    activeUploads.current.push(upload)
  })
}

// Module-scoped so the cancel button can abort in-flight uploads.
const activeUploads: { current: tus.Upload[] } = { current: [] }

export function ImageUploader({
  disabled,
  onUploaded,
}: {
  disabled?: boolean
  onUploaded: (runId: string) => void
}) {
  const [files, setFiles] = useState<Record<ModalityKey, File | null>>({ her2: null, dish: null, he: null })
  const [progress, setProgress] = useState(INITIAL_PROGRESS)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const runIdRef = useRef<string | null>(null)
  const allChosen = MODALITIES.every(({ key }) => files[key])
  const largeFiles = MODALITIES.filter(({ key }) => (files[key]?.size ?? 0) >= VERY_LARGE_FILE_BYTES)

  const pick = (key: ModalityKey) => (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] ?? null
    if (file && !file.name.toLowerCase().endsWith('.czi')) {
      e.target.value = ''
      return
    }
    setFiles((prev) => ({ ...prev, [key]: file }))
    setProgress((prev) => ({ ...prev, [key]: 0 }))
    setUploadError(null)
  }

  const upload = useMutation({
    mutationFn: async () => {
      setUploadError(null)
      activeUploads.current = []
      const chosen = Object.fromEntries(
        MODALITIES.map(({ key }) => [key, files[key]!]),
      ) as Record<ModalityKey, File>
      const { runId, clear } = runIdFor(chosen)
      runIdRef.current = runId

      // ponytail: all three concurrently. Triples the transport burst on multi-GB
      // files; go back to a sequential for-await loop if the link saturates.
      await Promise.all(
        MODALITIES.map(({ key }) =>
          uploadOne(chosen[key], runId, key, (p) => setProgress((prev) => ({ ...prev, [key]: p }))),
        ),
      )
      clear()
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
              <div className="flex items-center gap-2 text-[11px] text-neutral-500">
                <progress className="h-1.5 flex-1" max={100} value={progress[key]} />
                <span>{progress[key].toFixed(1)}%</span>
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
          disabled={disabled || upload.isPending || !allChosen}
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
        {upload.isError && uploadError && <p className="text-xs text-red-400">{uploadError}</p>}
      </div>
    </section>
  )
}
