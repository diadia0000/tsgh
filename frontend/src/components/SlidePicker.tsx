import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'

/**
 * Lists the slides available under the backend's SLIDES_DIR and lets the user
 * pick one — no typing an id by hand. The frontend only ever handles slide_ids
 * (guardrail 2); the backend is the sole thing that touches the filesystem.
 */
export function SlidePicker({
  activeSlideId,
  onPick,
}: {
  activeSlideId: string | null
  onPick: (slideId: string) => void
}) {
  const slides = useQuery({
    queryKey: ['slides'],
    queryFn: async () => {
      const { data, error } = await api.GET('/api/tiles')
      if (error) throw new Error('slide list failed')
      return data
    },
  })

  return (
    <div className="flex min-h-0 flex-col gap-2">
      <div className="flex items-center justify-between">
        <label className="text-xs font-semibold uppercase tracking-wide text-neutral-400">
          切片
        </label>
        <button
          type="button"
          onClick={() => slides.refetch()}
          disabled={slides.isFetching}
          className="text-xs text-neutral-400 hover:text-neutral-200 disabled:opacity-40"
        >
          {slides.isFetching ? '整理中…' : '重新整理'}
        </button>
      </div>

      {slides.isError ? (
        <p className="text-xs text-red-400">無法取得切片清單(後端未啟動?)</p>
      ) : slides.data && slides.data.length === 0 ? (
        <p className="text-xs text-neutral-500">
          slides 資料夾是空的。把 .tiff / .svs 切片放進去再按重新整理。
        </p>
      ) : (
        <ul className="flex flex-col gap-1 overflow-y-auto">
          {slides.data?.map((id) => {
            const active = id === activeSlideId
            return (
              <li key={id}>
                <button
                  type="button"
                  onClick={() => onPick(id)}
                  title={id}
                  className={`flex w-full items-center gap-2 truncate rounded-md px-3 py-2 text-left text-sm ${
                    active
                      ? 'bg-neutral-100 font-medium text-neutral-900'
                      : 'text-neutral-200 hover:bg-neutral-800'
                  }`}
                >
                  <span
                    className={active ? 'text-neutral-500' : 'text-neutral-600'}
                  >
                    ▸
                  </span>
                  <span className="truncate">{id}</span>
                </button>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
