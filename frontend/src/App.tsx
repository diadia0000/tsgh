import { useState } from 'react'
import { SlideViewer } from './components/SlideViewer'
import { OverlayViewer } from './components/OverlayViewer'
import { SlidePicker } from './components/SlidePicker'
import { PipelinePanel } from './components/PipelinePanel'

type ViewMode = 'single' | 'overlay'

function App() {
  const [mode, setMode] = useState<ViewMode>('single')
  const [slideId, setSlideId] = useState<string | null>(null)

  return (
    <div className="flex h-full bg-neutral-950 text-neutral-100">
      <aside className="flex w-80 shrink-0 flex-col gap-6 overflow-y-auto border-r border-neutral-800 p-4">
        <div>
          <h1 className="text-lg font-semibold">tsgh 切片工作台</h1>
          <p className="mt-1 text-xs text-neutral-500">
            WSI 檢視 · 對齊 / 細胞分割 pipeline
          </p>
        </div>

        <div className="flex rounded-md bg-neutral-900 p-1 text-sm">
          {(['single', 'overlay'] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              className={`flex-1 rounded px-3 py-1.5 ${
                mode === m
                  ? 'bg-neutral-100 font-medium text-neutral-900'
                  : 'text-neutral-300 hover:bg-neutral-800'
              }`}
            >
              {m === 'single' ? '單張' : '疊合'}
            </button>
          ))}
        </div>

        {mode === 'single' && (
          <SlidePicker activeSlideId={slideId} onPick={setSlideId} />
        )}

        <div className="border-t border-neutral-800 pt-4">
          <PipelinePanel />
        </div>
      </aside>

      <main className="min-w-0 flex-1">
        {mode === 'overlay' ? (
          <OverlayViewer />
        ) : slideId ? (
          <SlideViewer slideId={slideId} />
        ) : (
          <div className="flex h-full items-center justify-center text-sm text-neutral-500">
            從左側清單選一張切片
          </div>
        )}
      </main>
    </div>
  )
}

export default App
