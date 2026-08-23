import { useEffect, useRef, useState } from 'react'
import OpenSeadragon from 'openseadragon'

/**
 * Deep-zoom viewer for one slide. Points OpenSeadragon at the backend's DeepZoom
 * descriptor `/api/tiles/{slideId}.dzi`; OSD then fetches tiles on demand from
 * `/api/tiles/{slideId}_files/{level}/{col}_{row}.jpeg` (see backend/api/tiles.py).
 * The slide is addressed only by `slideId` — never a filesystem path (guardrail 2).
 */
export type ImageRect = { x: number; y: number; w: number; h: number }

/**
 * Readable names for the slide_ids the pipelines publish under. Anything not
 * listed -- an uploaded original, a smoke-test file -- is shown by its own id,
 * which is what the picker calls it, so the label always matches what the user
 * selected rather than going blank on an unfamiliar slide.
 */
const SLIDE_NAMES: Record<string, { name: string; detail?: string }> = {
  aligned_result: { name: '對齊結果', detail: '三張切片對齊後的疊合影像' },
  aligned_her2: { name: 'HER2 切片', detail: '對齊後' },
  aligned_dish: { name: 'DISH 切片', detail: '對齊後' },
  hybrid_overlay: { name: '細胞分析結果', detail: '含判讀標註' },
}

/** Fallback for `minRoiPx`: just enough to stop a drag past the anchor from
 *  inverting the rectangle. The real minimum is the pipeline's, and HybridPanel
 *  owns that rule (it knows the tile size) -- it passes it in. */
const MIN_DRAG_PX = 1

export function SlideViewer({
  slideId,
  onViewportChange,
  roi,
  onRoiChange,
  minRoiPx = MIN_DRAG_PX,
}: {
  slideId: string
  /** Current viewport in *image* pixels, reported after every pan/zoom. This is
   *  what lets the analysis panel turn "what I am looking at" into a ROI. */
  onViewportChange?: (rect: ImageRect) => void
  /** The analysis region to draw, in image pixels. Null = no box. */
  roi?: ImageRect | null
  /** Called while the user drags the box or its resize handle. Omit to render
   *  the box read-only. */
  onRoiChange?: (rect: ImageRect) => void
  /** Smallest side the resize handle will produce, in image pixels. */
  minRoiPx?: number
}) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [error, setError] = useState<string | null>(null)
  // Held in a ref so a new callback identity never tears down and rebuilds the
  // viewer (which would reset the user's zoom mid-selection).
  const reportRef = useRef(onViewportChange)
  reportRef.current = onViewportChange
  // Same reasoning for the ROI: the box is an OSD overlay driven imperatively,
  // so the drag handlers read the current values through refs rather than being
  // re-bound (and the viewer re-created) on every state update.
  const viewerRef = useRef<OpenSeadragon.Viewer | null>(null)
  const boxRef = useRef<HTMLDivElement | null>(null)
  const roiRef = useRef(roi)
  roiRef.current = roi
  const onRoiChangeRef = useRef(onRoiChange)
  onRoiChangeRef.current = onRoiChange
  const minRoiRef = useRef(minRoiPx)
  minRoiRef.current = minRoiPx

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

    viewerRef.current = viewer
    return () => {
      viewerRef.current = null
      boxRef.current = null
      viewer.destroy()
    }
  }, [slideId])

  // ---- ROI box (docs/UI/hybrid_flow_mockup.html 畫面 1) ----------------------
  // Drawn as an OpenSeadragon overlay, not a fixed div on top of the canvas, so
  // it is anchored to image coordinates: pan and zoom keep it over the same
  // tissue instead of sliding off it.
  useEffect(() => {
    const viewer = viewerRef.current
    if (!viewer) return

    if (!roi) {
      if (boxRef.current) {
        viewer.removeOverlay(boxRef.current)
        boxRef.current = null
      }
      return
    }

    const place = () =>
      viewer.viewport.imageToViewportRectangle(
        new OpenSeadragon.Rect(roi.x, roi.y, roi.w, roi.h),
      )

    if (!boxRef.current) {
      boxRef.current = buildBox(viewer, roiRef, onRoiChangeRef, minRoiRef)
      viewer.addOverlay({ element: boxRef.current, location: place() })
    } else {
      viewer.updateOverlay(boxRef.current, place())
    }
    // The size label is the only part that is not positional, so it is the only
    // thing that has to be rewritten on every change.
    const tag = boxRef.current.firstElementChild as HTMLElement | null
    if (tag) tag.textContent = `分析範圍 ${roi.w}×${roi.h}`
  }, [roi, slideId])

  return (
    <div className="relative h-full w-full bg-neutral-900">
      <div ref={containerRef} className="h-full w-full" />
      {/* Stacked in one column: the name and the ROI hint were both pinned to
          left-3 top-3 and drew on top of each other whenever a box was being
          edited on the aligned result. pointer-events-none so neither swallows a
          pan that starts in the corner. */}
      <div className="pointer-events-none absolute left-3 top-3 flex flex-col gap-2">
        <div className="rounded-md bg-neutral-950/80 px-3 py-2 text-xs shadow">
          <p className="font-medium text-neutral-200">{SLIDE_NAMES[slideId]?.name ?? slideId}</p>
          {SLIDE_NAMES[slideId]?.detail && (
            <p className="mt-0.5 text-neutral-400">{SLIDE_NAMES[slideId].detail}</p>
          )}
          {/* The id stays visible under a friendly name: it is what the picker
              and the analysis panel call the same slide. */}
          {SLIDE_NAMES[slideId] && (
            <p className="mt-0.5 break-all font-mono text-[10px] text-neutral-500">{slideId}</p>
          )}
        </div>
        {roi && onRoiChange && (
          <div className="self-start rounded-md bg-neutral-950/80 px-3 py-1.5 text-xs text-neutral-300 shadow">
            拖曳藍框移動 · 右下角把手調整大小
          </div>
        )}
      </div>
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

/**
 * The draggable selection rectangle from docs/UI/hybrid_flow_mockup.html: drag
 * the body to move it, drag the bottom-right handle to resize, everything
 * outside it is dimmed. Plain DOM rather than JSX because OSD owns the element
 * (it lives in the overlay container and OSD repositions it every frame), and
 * inline styles rather than classes because it is created outside the tree
 * Tailwind scans.
 */
function buildBox(
  viewer: OpenSeadragon.Viewer,
  roiRef: { current: ImageRect | null | undefined },
  onChangeRef: { current: ((rect: ImageRect) => void) | undefined },
  minRef: { current: number },
): HTMLDivElement {
  const el = document.createElement('div')
  Object.assign(el.style, {
    border: '2px solid #3b82f6',
    background: 'rgba(59,130,246,.15)',
    boxShadow: '0 0 0 9999px rgba(2,6,23,.45)',   // dim everything outside
    cursor: 'move',
    touchAction: 'none',
  })

  const tag = document.createElement('div')
  Object.assign(tag.style, {
    position: 'absolute', top: '-22px', left: '0',
    padding: '1px 6px', borderRadius: '4px',
    background: '#2563eb', color: '#fff',
    font: '11px ui-monospace, SFMono-Regular, monospace',
    whiteSpace: 'nowrap',
  })

  const handle = document.createElement('div')
  Object.assign(handle.style, {
    position: 'absolute', right: '-7px', bottom: '-7px',
    width: '14px', height: '14px',
    background: '#3b82f6', border: '2px solid #fff', borderRadius: '2px',
    cursor: 'nwse-resize', touchAction: 'none',
  })

  el.append(tag, handle)   // tag first: the effect above rewrites firstElementChild

  // Screen pixels -> image pixels. Going through the viewport makes a drag
  // scale-correct at any zoom, which a raw pixel delta would not be.
  const toImage = (e: PointerEvent) => {
    const bounds = viewer.element.getBoundingClientRect()
    const point = viewer.viewport.pointFromPixel(
      new OpenSeadragon.Point(e.clientX - bounds.left, e.clientY - bounds.top),
      true,
    )
    return viewer.viewport.viewportToImageCoordinates(point)
  }

  let mode: 'move' | 'resize' | null = null
  let anchor = { x: 0, y: 0 }
  let from: ImageRect = { x: 0, y: 0, w: 0, h: 0 }

  el.addEventListener('pointerdown', (e) => {
    if (!roiRef.current || !onChangeRef.current) return
    mode = e.target === handle ? 'resize' : 'move'
    anchor = toImage(e)
    from = { ...roiRef.current }
    // Without this OSD pans the slide out from under the box mid-drag.
    viewer.setMouseNavEnabled(false)
    // Tracked on window, not on the box: a resize drag routinely outruns the
    // edge it is dragging, and the pointer must keep being followed once it is
    // outside the element. It also means the drag always ends -- an element
    // that never sees pointerup would otherwise stay in drag mode and keep
    // editing the ROI on a plain mouse move.
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onEnd)
    window.addEventListener('pointercancel', onEnd)
    e.preventDefault()
    e.stopPropagation()
  })

  function onMove(e: PointerEvent) {
    if (!mode) return
    // No button held = the pointerup was missed (drag ended off-window, say).
    if (e.buttons === 0) return onEnd(e)
    const size = viewer.world.getItemAt(0)?.getContentSize()
    if (!size) return
    const now = toImage(e)
    const dx = now.x - anchor.x
    const dy = now.y - anchor.y
    // Clamped to the image: the backend rejects a ROI that runs off the slide.
    const next: ImageRect =
      mode === 'move'
        ? {
            ...from,
            x: Math.max(0, Math.min(Math.round(from.x + dx), size.x - from.w)),
            y: Math.max(0, Math.min(Math.round(from.y + dy), size.y - from.h)),
          }
        : {
            ...from,
            w: Math.max(minRef.current, Math.min(Math.round(from.w + dx), size.x - from.x)),
            h: Math.max(minRef.current, Math.min(Math.round(from.h + dy), size.y - from.y)),
          }
    onChangeRef.current?.(next)
    e.preventDefault()
  }

  function onEnd(_e: PointerEvent) {
    if (!mode) return
    mode = null
    viewer.setMouseNavEnabled(true)
    window.removeEventListener('pointermove', onMove)
    window.removeEventListener('pointerup', onEnd)
    window.removeEventListener('pointercancel', onEnd)
  }

  return el
}
