import type { ReactNode } from 'react'

/**
 * A sidebar section that folds away once its stage is finished.
 *
 * The panel holds three stages a doctor walks through once each; leaving all of
 * them expanded means the controls that still matter are always below controls
 * that no longer do. Folding a finished stage is only safe if the header keeps
 * saying what it holds, so `summary` is not decoration -- a collapsed section
 * with a bare title hides state instead of tidying it.
 *
 * The header is a filled bar with an explicit 展開/收合 word rather than a bare
 * chevron: a 12px glyph is not enough to tell a reader that a section exists,
 * that it is clickable, or which way it is currently folded. The caret rotates
 * as a second, redundant signal, and the summary sits on its own full-width line
 * so a long run id is readable instead of truncated.
 *
 * Controlled: the parent decides when a stage is done, and a doctor who opens
 * one back up must not have it fold under them on the next state change.
 */
export function CollapsibleSection({
  title,
  summary,
  open,
  onToggle,
  children,
}: {
  title: string
  /** What the section holds, shown while collapsed (e.g. the run it is on). */
  summary?: string
  open: boolean
  onToggle: () => void
  children: ReactNode
}) {
  return (
    <section>
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={open}
        className="flex w-full items-center gap-2 rounded-md border border-neutral-800 bg-neutral-900/60 px-2 py-1.5 text-left hover:bg-neutral-800"
      >
        {/* U+25B8 (not U+25B6): the larger triangle gets emoji presentation in
            some fonts and renders as a blue play-button box. */}
        <span
          className={`shrink-0 text-xs text-neutral-400 transition-transform ${open ? 'rotate-90' : ''}`}
        >
          ▸
        </span>
        <span className="min-w-0 flex-1 truncate text-xs font-semibold uppercase tracking-wide text-neutral-300">
          {title}
        </span>
        <span className="shrink-0 text-xs text-neutral-400">{open ? '收合' : '展開'}</span>
      </button>
      {!open && summary && <p className="mt-1 px-2 text-xs text-neutral-500">{summary}</p>}
      {open && <div className="mt-2">{children}</div>}
    </section>
  )
}
