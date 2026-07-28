import { useEffect, useRef, useState } from 'react'
import OpenSeadragon from 'openseadragon'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'

/**
 * Blends the aligned modalities over a white base with a live per-layer alpha
 * slider each. HER2 + DISH are published by the alignment pipeline as
 * per-modality warped slides (aligned_her2 / aligned_dish); HE is intentionally
 * empty (slideId:null — doctors don't need it yet), so its slider is a no-op.
 * Layers only appear once alignment has produced them; until then a hint shows.
 */
const LAYERS = [
  { key: 'her2', label: 'HER2', slideId: 'aligned_her2', initial: 0.5 },
  { key: 'dish', label: 'DISH', slideId: 'aligned_dish', initial: 0.5 },
  { key: 'he', label: 'HE', slideId: null, initial: 0 },
] as const
type LayerKey = (typeof LAYERS)[number]['key']

const INITIAL_ALPHA = Object.fromEntries(
  LAYERS.map((l) => [l.key, l.initial]),
) as Record<LayerKey, number>

export function OverlayViewer() {
  const containerRef = useRef<HTMLDivElement>(null)
  const viewerRef = useRef<OpenSeadragon.Viewer | null>(null)
  const [alpha, setAlpha] = useState<Record<LayerKey, number>>(INITIAL_ALPHA)
  const [error, setError] = useState<string | null>(null)

  const slides = useQuery({
    queryKey: ['slides'],
    queryFn: async () => {
      const { data, error } = await api.GET('/api/tiles')
      if (error) throw new Error('slide list failed')
      return data
    },
  })

  // Which run these layers actually came from. `aligned_*` are global slide_ids
  // owned by whichever run published last, so selecting another job in the panel
  // leaves the previous run's images on screen -- naming the source is what stops
  // them being read as the selected job's result.
  const published = useQuery({
    queryKey: ['published'],
    queryFn: async () => {
      const { data, error } = await api.GET('/api/alignment/published')
      if (error) throw new Error('published lookup failed')
      return data
    },
  })

  // Layers with an aligned image actually on the backend, in LAYERS order. Their
  // position here IS the OpenSeadragon world index the sliders address.
  const present = LAYERS.filter((l) => l.slideId && slides.data?.includes(l.slideId))
  const presentKey = present.map((l) => l.slideId).join(',')
  const ready = present.length > 0

  useEffect(() => {
    if (!containerRef.current || !ready) return
    setError(null)
    setAlpha(INITIAL_ALPHA)

    const viewer = OpenSeadragon({
      element: containerRef.current,
      showNavigationControl: false,
      showNavigator: true,
      navigatorPosition: 'BOTTOM_RIGHT',
      gestureSettingsMouse: { clickToZoom: false },
    })
    viewerRef.current = viewer
    viewer.addHandler('open-failed', (e) => {
      setError(`無法載入圖層：${e.message ?? 'not found'}`)
    })

    present.forEach((l, i) => {
      viewer.addTiledImage({
        tileSource: `/api/tiles/${encodeURIComponent(l.slideId!)}.dzi`,
        opacity: INITIAL_ALPHA[l.key],
        index: i,
      })
    })

    return () => {
      viewer.destroy()
      viewerRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, presentKey])

  function setLayerAlpha(key: LayerKey, value: number) {
    setAlpha((prev) => ({ ...prev, [key]: value }))
    const i = present.findIndex((l) => l.key === key)
    if (i >= 0) viewerRef.current?.world.getItemAt(i)?.setOpacity(value)
  }

  return (
    // White base: the OSD canvas is transparent, so this shows through wherever a
    // layer's alpha < 1 — i.e. the "白色底圖" under the blended layers.
    <div className="relative h-full w-full bg-white">
      <div ref={containerRef} className="h-full w-full" />

      {ready && (
        <div className="absolute left-3 top-3 flex w-64 flex-col gap-3 rounded-lg bg-neutral-950/80 p-3 text-xs text-neutral-200 backdrop-blur">
          <span className="font-semibold uppercase tracking-wide text-neutral-400">
            圖層透明度
          </span>
          <span className="-mt-2 break-all text-[11px] text-neutral-500">
            來源工作：
            <span className="font-mono text-neutral-300">
              {published.data?.run_id ?? '未知（後端重啟前的結果）'}
            </span>
          </span>
          {LAYERS.map((l) => {
            const empty = !l.slideId || !slides.data?.includes(l.slideId)
            return (
              <label key={l.key} className="flex flex-col gap-1">
                <span className="flex items-center justify-between gap-2">
                  <span>
                    {l.label}
                    {empty && <span className="ml-1 text-neutral-500">(空)</span>}
                  </span>
                  <span className="shrink-0 text-neutral-500">
                    {Math.round(alpha[l.key] * 100)}%
                  </span>
                </span>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={alpha[l.key]}
                  onChange={(e) => setLayerAlpha(l.key, Number(e.target.value))}
                  className="accent-neutral-800"
                />
              </label>
            )
          })}
        </div>
      )}

      {!ready && !slides.isError && (
        <div className="absolute inset-0 flex items-center justify-center p-6">
          <p className="max-w-md rounded-lg bg-neutral-950/80 px-4 py-3 text-center text-sm text-neutral-200">
            尚無疊合圖，請先完成對齊流程後再檢視。
          </p>
        </div>
      )}
      {slides.isError && (
        <div className="absolute inset-0 flex items-center justify-center p-6">
          <p className="max-w-md rounded-lg bg-red-950/80 px-4 py-3 text-center text-sm text-red-200">
            無法取得切片清單(後端未啟動?)
          </p>
        </div>
      )}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center p-6">
          <p className="max-w-md rounded-lg bg-red-950/80 px-4 py-3 text-center text-sm text-red-200">
            {error}
          </p>
        </div>
      )}
    </div>
  )
}
