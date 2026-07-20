import { OverlayViewer } from './components/OverlayViewer'
import { PipelinePanel } from './components/PipelinePanel'

function App() {
  return (
    <div className="flex h-full bg-neutral-950 text-neutral-100">
      <aside className="flex w-80 shrink-0 flex-col gap-6 overflow-y-auto border-r border-neutral-800 p-4">
        <div>
          <h1 className="text-lg font-semibold">tsgh 切片工作台</h1>
          <p className="mt-1 text-xs text-neutral-500">WSI 檢視 · 對齊 pipeline</p>
        </div>

        <div className="border-t border-neutral-800 pt-4">
          <PipelinePanel />
        </div>
      </aside>

      <main className="min-w-0 flex-1">
        <OverlayViewer />
      </main>
    </div>
  )
}

export default App
