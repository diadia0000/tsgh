/**
 * Turns an analysis job's state into a percentage, a phase caption and an ETA.
 *
 * A run has two stretches and only the first one can be counted. The pipeline
 * announces every tile it starts, so `/api/jobs/{id}` reports `done/total`
 * tiles and that part of the bar is a measurement. What follows -- the global
 * cell merge and the overlay stitch -- reports nothing until it is finished,
 * so no honest number exists for it.
 *
 * Rather than let the bar hit 100% and sit there (which reads as "hung", the
 * worst of the options), the counted stretch is given `ANALYZE_SHARE` of the
 * bar and the uncounted tail creeps through the rest, never reaching 100 until
 * the job actually reports done. The tail's speed comes from how long stitching
 * took on this machine before, measured **per tile** rather than as a flat
 * duration -- a ROI of 40 tiles and a whole slide of 27,000 do not stitch in
 * remotely the same time, so a single median seconds-figure would be worthless
 * across both.
 *
 * `estimated` marks a number that came from the creep rather than the counter,
 * so the caller can say so instead of presenting a guess as a measurement.
 */

import { formatEta } from './stepTiming'

const STORAGE_KEY = 'tsgh.hybridStitchRate'
const KEEP_RUNS = 5           // enough to outvote one anomalous run, few enough to track a machine change
const ANALYZE_SHARE = 85      // the counted stretch's share of the bar
const CEILING = 99            // the tail approaches this and never arrives
// Without history the tail creeps on this time constant: after one of these it
// has covered ~63% of the remaining gap, and it never closes it.
const BLIND_CREEP_TAU_S = 180

export type JobProgressCounts = { phase: string; done: number; total: number; unit_label: string }

export type HybridProgress = {
  percent: number
  /** What is happening, in the words shown to the user. */
  caption: string
  /** Formatted remaining time, or null when nothing backs an estimate. */
  eta: string | null
  /** True while the bar is creeping rather than counting. */
  estimated: boolean
}

/** Median seconds spent stitching per tile, over the last few runs. */
export function readStitchRate(): number | null {
  const samples = safeRead()
  return samples.length ? median(samples) : null
}

/** Record a finished run's stitch phase, normalised per tile. */
export function recordStitchRate(stitchSeconds: number, totalTiles: number): void {
  if (!Number.isFinite(stitchSeconds) || stitchSeconds <= 0 || totalTiles <= 0) return
  const samples = [...safeRead(), stitchSeconds / totalTiles].slice(-KEEP_RUNS)
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(samples))
  } catch {
    // Private mode / quota: the tail loses its ETA, nothing else.
  }
}

export function computeHybridProgress({
  progress,
  elapsed,
  tailElapsed,
  done,
}: {
  /** Counts from the job, or null when the backend is publishing none. */
  progress: JobProgressCounts | null
  /** Seconds since the run was submitted. */
  elapsed: number
  /** Seconds since the tile counter was exhausted, or null before that. */
  tailElapsed: number | null
  /** The job reached status "done". */
  done: boolean
}): HybridProgress {
  if (done) return { percent: 100, caption: '完成', eta: null, estimated: false }

  // No counter at all: an older backend, or the pipeline's log line was
  // reworded (tests/test_hybrid_progress.py exists to catch the latter). Say so
  // rather than invent a number -- the elapsed clock beside this is still true.
  if (!progress || progress.total <= 0) {
    return { percent: 0, caption: '分析中…', eta: null, estimated: true }
  }

  // `done` is the tile the pipeline has *started*, 1-based -- it logs the line
  // before processing, not after -- so the number finished is one less. The
  // caption says "第 N/M" because that is what a person watching wants ("which
  // one is it on"), while the bar uses the finished count, which is the only
  // one that has actually been earned.
  const { done: current, total, unit_label: unit } = progress
  const finished = Math.max(current - 1, 0)
  const counting = tailElapsed === null

  if (counting) {
    // Rate measured on this run, so it needs no history and adapts to a slide
    // that happens to be slower. Meaningless until a tile has actually finished.
    const perTile = finished > 0 ? elapsed / finished : null
    const stitchRate = readStitchRate()
    const eta =
      perTile !== null
        ? formatEta(perTile * (total - finished) + (stitchRate ?? 0) * total)
        : null
    return {
      percent: (finished / total) * ANALYZE_SHARE,
      caption: `分析中… 第 ${current}/${total} ${unit}`,
      eta,
      estimated: false,
    }
  }

  // Counter exhausted: the last tile is still being processed, and behind it
  // come the merge and the stitch. Nothing in that stretch reports anything.
  const gap = CEILING - ANALYZE_SHARE
  const stitchRate = readStitchRate()
  if (stitchRate !== null) {
    const expected = stitchRate * total
    const fraction = Math.min(tailElapsed / expected, 1)
    return {
      percent: ANALYZE_SHARE + gap * fraction,
      caption: '收尾中…（最後一塊與影像縫合）',
      eta: formatEta(Math.max(expected - tailElapsed, 1)),
      estimated: true,
    }
  }
  // First run on this machine: approach the ceiling without ever reaching it.
  return {
    percent: ANALYZE_SHARE + gap * (1 - Math.exp(-tailElapsed / BLIND_CREEP_TAU_S)),
    caption: '合併與縫合影像中…',
    eta: null,
    estimated: true,
  }
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

function safeRead(): number[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '[]')
    return Array.isArray(parsed) ? parsed.filter((n) => typeof n === 'number' && n > 0) : []
  } catch {
    return []
  }
}
