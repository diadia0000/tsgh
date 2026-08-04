/**
 * Turns "which alignment steps are finished" into a percentage and an ETA.
 *
 * The backend reports only pending/running/done -- no progress number exists to
 * read -- so the only honest percentage is the one the step chain itself gives:
 * four steps, each either finished or not. That alone would leave the bar frozen
 * at 25% for the length of a step, so a step already timed on this machine also
 * contributes its elapsed fraction, and the steps are weighted by how long they
 * actually took rather than counted equally.
 *
 * Everything here degrades to plain step counting on a machine with no history
 * yet, and `fromHistory` says which of the two the caller is showing -- an
 * estimate presented as a measurement is worse than no estimate.
 */

const STORAGE_KEY = 'tsgh.alignStepDurations'
const KEEP_RUNS = 5          // enough to outvote one anomalous run, few enough to track a machine change
const CURRENT_STEP_CAP = 0.99 // a running step never reads as finished, however long it overruns

export type StepDurations = Record<string, number>

/** Median of the last few runs of each step, in seconds. */
export function readStepDurations(): StepDurations {
  const raw = safeRead()
  const out: StepDurations = {}
  for (const [step, samples] of Object.entries(raw)) {
    if (samples.length) out[step] = median(samples)
  }
  return out
}

export function recordStepDuration(step: string, seconds: number): void {
  if (!Number.isFinite(seconds) || seconds <= 0) return
  const raw = safeRead()
  raw[step] = [...(raw[step] ?? []), seconds].slice(-KEEP_RUNS)
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(raw))
  } catch {
    // Private mode / quota: timing history is a nicety, never worth an error.
  }
}

export function estimateProgress({
  steps,
  doneSteps,
  currentStep,
  elapsedInStep,
  durations,
}: {
  steps: readonly string[]
  doneSteps: readonly string[]
  currentStep: string | null
  elapsedInStep: number
  durations: StepDurations
}): { percent: number; etaSeconds: number | null; fromHistory: boolean } {
  const fromHistory = steps.every((s) => durations[s] > 0)
  const weight = (step: string) => (fromHistory ? durations[step] : 1)
  const total = steps.reduce((sum, s) => sum + weight(s), 0)
  if (total <= 0) return { percent: 0, etaSeconds: null, fromHistory: false }

  const isDone = (step: string) => doneSteps.includes(step)
  let elapsedWeight = steps.filter(isDone).reduce((sum, s) => sum + weight(s), 0)

  // The running step counts for the fraction of its usual duration spent so far.
  let remainingCurrent = 0
  if (currentStep && !isDone(currentStep) && steps.includes(currentStep)) {
    const expected = durations[currentStep]
    if (expected > 0) {
      const fraction = Math.min(elapsedInStep / expected, CURRENT_STEP_CAP)
      elapsedWeight += weight(currentStep) * fraction
      remainingCurrent = Math.max(expected - elapsedInStep, 0)
    }
  }

  const percent = Math.min((elapsedWeight / total) * 100, 100)

  let etaSeconds: number | null = null
  if (fromHistory) {
    const pending = steps
      .filter((s) => !isDone(s) && s !== currentStep)
      .reduce((sum, s) => sum + durations[s], 0)
    etaSeconds = Math.round(pending + remainingCurrent)
  }
  return { percent, etaSeconds, fromHistory }
}

/** "1 小時 5 分" / "3 分 20 秒" / "45 秒" -- coarse on purpose, since the estimate is. */
export function formatEta(seconds: number): string {
  if (seconds < 60) return `${Math.max(1, Math.round(seconds))} 秒`
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes} 分`
  return `${Math.floor(minutes / 60)} 小時 ${minutes % 60} 分`
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

function safeRead(): Record<string, number[]> {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}')
    if (!parsed || typeof parsed !== 'object') return {}
    // Written by an older build, or hand-edited: keep only what is usable.
    return Object.fromEntries(
      Object.entries(parsed as Record<string, unknown>)
        .map(([k, v]) => [k, Array.isArray(v) ? v.filter((n) => typeof n === 'number' && n > 0) : []])
        .filter(([, v]) => (v as number[]).length),
    ) as Record<string, number[]>
  } catch {
    return {}
  }
}
