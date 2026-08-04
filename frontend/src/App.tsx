import { useState } from 'react'
import { OverlayViewer } from './components/OverlayViewer'
import { PipelinePanel } from './components/PipelinePanel'
import { HybridPanel } from './components/HybridPanel'
import { SlideViewer, type ImageRect } from './components/SlideViewer'

function App() {
  // The main area shows either the blended alignment layers (default) or one
  // single slide: the IHC slide being framed for a ROI, or the hybrid
  // pipeline's annotated overlay once a run finishes.
  const [singleSlide, setSingleSlide] = useState<string | null>(null)
  // Lifted out of SlideViewer so the analysis panel can seed a ROI from the
  // current view. Null whenever no single slide is on screen.
  const [viewRect, setViewRect] = useState<ImageRect | null>(null)
  // The analysis region, in image pixels. Owned here because two children share
  // it: the viewer draws and drags the box, the panel reads it and submits it.
  const [roi, setRoi] = useState<ImageRect | null>(null)
  // Which pipeline currently owns the machine. Both panels drive the same GPU
  // and the same output directories, so exactly one may be working at a time;
  // each panel reports its own state and locks itself on the other's.
  const [alignBusy, setAlignBusy] = useState(false)
  const [hybridBusy, setHybridBusy] = useState(false)

  const showSlide = (slideId: string) => {
    if (slideId !== singleSlide) {
      setViewRect(null)   // stale rect belongs to the old slide
      setRoi(null)        // ...and so does the selection drawn on it
    }
    setSingleSlide(slideId)
  }

  return (
    <div className="flex h-full bg-neutral-950 text-neutral-100">
      <aside className="flex w-80 shrink-0 flex-col gap-6 overflow-y-auto border-r border-neutral-800 p-4">
        <div>
          <h1 className="text-lg font-semibold">tsgh 切片工作台</h1>
          <p className="mt-1 text-xs text-neutral-500">切片檢視 · 影像對齊 · 細胞分析</p>
        </div>

        <div className="border-t border-neutral-800 pt-4">
          <PipelinePanel externalBusy={hybridBusy} onBusyChange={setAlignBusy} />
        </div>

        <div className="border-t border-neutral-800 pt-4">
          {alignBusy && (
            <p className="mb-2 rounded-md border border-amber-900/60 bg-amber-950/30 px-2 py-1 text-xs text-amber-300">
              對齊流程執行中，完成後才能開始分析
            </p>
          )}
          <HybridPanel
            onViewSlide={showSlide}
            viewRect={viewRect}
            roi={roi}
            onRoiChange={setRoi}
            externalBusy={alignBusy}
            onBusyChange={setHybridBusy}
          />
        </div>
      </aside>

      <main className="relative min-w-0 flex-1">
        {singleSlide ? (
          <>
            <SlideViewer
              slideId={singleSlide}
              onViewportChange={setViewRect}
              roi={roi}
              onRoiChange={setRoi}
            />
            <button
              type="button"
              onClick={() => {
                setSingleSlide(null)
                setViewRect(null)
                setRoi(null)
              }}
              className="absolute right-3 top-3 rounded-md border border-neutral-700 bg-neutral-950/80 px-3 py-1.5 text-xs text-neutral-200 backdrop-blur hover:bg-neutral-800"
            >
              返回疊合檢視
            </button>
          </>
        ) : (
          <OverlayViewer />
        )}
      </main>
    </div>
  )
}

export default App
