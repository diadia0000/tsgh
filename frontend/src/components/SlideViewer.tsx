import { useEffect, useRef, useState } from 'react'
import OpenSeadragon from 'openseadragon'

/**
 * Deep-zoom viewer for one slide. Points OpenSeadragon at the backend's DeepZoom
 * descriptor `/api/tiles/{slideId}.dzi`; OSD then fetches tiles on demand from
 * `/api/tiles/{slideId}_files/{level}/{col}_{row}.jpeg` (see backend/api/tiles.py).
 * The slide is addressed only by `slideId` — never a filesystem path (guardrail 2).
 */
export type ImageRect = { x: number; y: number; w: number; h: number }

export function SlideViewer({
  slideId,
  onViewportChange,
}: {
  slideId: string
  /** Current viewport in *image* pixels, reported after every pan/zoom. This is
   *  what lets the analysis panel turn "what I am looking at" into a ROI. */
  onViewportChange?: (rect: ImageRect) => void
}) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [error, setError] = useState<string | null>(null)
  // Held in a ref so a new callback identity never tears down and rebuilds the
  // viewer (which would reset the user's zoom mid-selection).
  const reportRef = useRef(onViewportChange)
  reportRef.current = onViewportChange

  useEffect(() => {
    if (!containerRef.current || !slideId) return
    setError(null)

    const viewer = OpenSeadragon({
      element: containerRef.current,
      // No prefixUrl / nav-control buttons: those need CDN-hosted images, and
      // the packaged app is offline (docs/UI/01). Scroll to zoom, drag to pan,
      // plus the navigator thumbnail — none of which need image assets.
      showNavigationControl: false,
      // slide_ids can contain (, ), + etc.; encode so the .dzi URL — and the
      // {id}_files/... tile URLs OSD derives from it — stay valid.
      tileSources: `/api/tiles/${encodeURIComponent(slideId)}.dzi`,
      showNavigator: true,
      navigatorPosition: 'BOTTOM_RIGHT',
      gestureSettingsMouse: { clickToZoom: false },
    })
    viewer.addHandler('open-failed', (e) => {
      setError(`無法載入切片「${slideId}」：${e.message ?? 'not found'}`)
    })

    // The viewport can extend past the slide (OSD lets you pan into blank
    // space), so clamp before reporting: a ROI must be a real sub-rectangle of
    // the image or the backend rejects it.
    const report = () => {
      const item = viewer.world.getItemAt(0)
      if (!item) return
      const size = item.getContentSize()
      const r = viewer.viewport.viewportToImageRectangle(viewer.viewport.getBounds(true))
      const x = Math.max(0, Math.round(r.x))
      const y = Math.max(0, Math.round(r.y))
      reportRef.current?.({
        x,
        y,
        w: Math.min(Math.round(r.width), size.x - x),
        h: Math.min(Math.round(r.height), size.y - y),
      })
    }
    viewer.addHandler('open', report)
    viewer.addHandler('animation-finish', report)

    return () => viewer.destroy()
  }, [slideId])

  return (
    <div className="relative h-full w-full bg-neutral-900">
      <div ref={containerRef} className="h-full w-full" />
      {slideId === 'aligned_result' && (
        <div className="absolute left-3 top-3 rounded-md bg-neutral-950/80 px-3 py-2 text-xs text-neutral-200 shadow">
          <p className="font-medium">aligned_result.tiff</p>
          <p className="mt-0.5 text-neutral-400">對齊流程輸出 · TIFF</p>
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
