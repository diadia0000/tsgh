import { useEffect, useRef, useState } from 'react'
import OpenSeadragon from 'openseadragon'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'

/**
 * Stacks every slide under SLIDES_DIR as OpenSeadragon layers in one viewer,
 * each with its own opacity slider. The slides there are the pixel-aligned
 * modality outputs (HER2 / DISH / HE, all cropped to the same overlap region),
 * so they register on top of each other (same dimensions, same origin) and can
 * be blended. Layers are addressed only by slide_id (guardrail 2); the .dzi /
 * tile URLs mirror SlideViewer.tsx.
 */
export function OverlayViewer() {
  const containerRef = useRef<HTMLDivElement>(null)
  const viewerRef = useRef<OpenSeadragon.Viewer | null>(null)
  const [opacities, setOpacities] = useState<Record<string, number>>({})
  const [error, setError] = useState<string | null>(null)

  const slides = useQuery({
    queryKey: ['slides'],
    queryFn: async () => {
      const { data, error } = await api.GET('/api/tiles')
      if (error) throw new Error('slide list failed')
      return data
    },
  })
  const slideIds = slides.data

  useEffect(() => {
    if (!containerRef.current || !slideIds || slideIds.length === 0) return
    setError(null)
    setOpacities(Object.fromEntries(slideIds.map((id) => [id, 1])))

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

    // index: i keeps world order == slideIds order, so the sliders below can
    // address each layer by its list position regardless of load-finish order.
    slideIds.forEach((id, i) => {
      viewer.addTiledImage({
        tileSource: `/api/tiles/${encodeURIComponent(id)}.dzi`,
        opacity: 1,
        index: i,
      })
    })

    return () => {
      viewer.destroy()
      viewerRef.current = null
    }
  }, [slideIds])

  function setLayerOpacity(index: number, id: string, value: number) {
    setOpacities((prev) => ({ ...prev, [id]: value }))
    viewerRef.current?.world.getItemAt(index)?.setOpacity(value)
  }

  return (
    <div className="relative h-full w-full bg-neutral-900">
      <div ref={containerRef} className="h-full w-full" />

      {slideIds && slideIds.length > 0 && (
        <div className="absolute left-3 top-3 flex w-64 flex-col gap-3 rounded-lg bg-neutral-950/80 p-3 text-xs text-neutral-200 backdrop-blur">
          <span className="font-semibold uppercase tracking-wide text-neutral-400">
            圖層透明度
          </span>
          {slideIds.map((id, i) => (
            <label key={id} className="flex flex-col gap-1">
              <span className="flex items-center justify-between gap-2">
                <span className="truncate" title={id}>
                  {id}
                </span>
                <span className="shrink-0 text-neutral-500">
                  {Math.round((opacities[id] ?? 1) * 100)}%
                </span>
              </span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={opacities[id] ?? 1}
                onChange={(e) => setLayerOpacity(i, id, Number(e.target.value))}
                className="accent-neutral-200"
              />
            </label>
          ))}
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
