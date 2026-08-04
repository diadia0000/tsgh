/**
 * Turns the pipeline's `summary.txt` into the report card of
 * docs/UI/hybrid_flow_mockup.html 畫面 3 -- headline verdict, the six figures it
 * rests on, and the cell distribution -- instead of dumping the file as text.
 *
 * Parsing rather than a new API: the file's layout is a protected contract
 * (.claude/rules/no-regression.md §1, "summary.txt 的 ASCO/CAP 版面"), so it is
 * stable to read, and every number the mockup asks for is already in it. The
 * alternative -- having the backend emit the same numbers as JSON -- would mean
 * two producers of one report that could disagree.
 *
 * The file is written with the dot vocabulary of the detector (黑點 = HER2 signal,
 * 紅點 = CEP17); the labels here are the clinical names the mockup uses. Nothing
 * is computed or rounded: every value is shown exactly as the file states it.
 *
 * Unparseable input yields null, and the caller falls back to showing the raw
 * text -- a summary that cannot be turned into a card must still be readable.
 */

export type ReportMetric = { label: string; value: string; hint?: string }
export type ReportRow = { label: string; count: string; share: string }
export type SummaryReport = {
  verdict: string
  amplified: boolean | null   // null = the verdict line said neither
  metrics: ReportMetric[]
  distribution: ReportRow[]
}

/** `  黑點/紅點 比值  1.50` -> "1.50". Two or more spaces separate label from value. */
function field(text: string, label: string): string | null {
  const line = text.split('\n').find((l) => l.trimStart().startsWith(label))
  if (!line) return null
  const value = line.trimStart().slice(label.length).trim()
  return value || null
}

/** `  比值 < 2         1  (100.0%)` -> count "1", share "100.0%". */
function row(line: string): ReportRow | null {
  const m = line.trim().match(/^(.*?)\s{2,}(\d+)\s*\(([\d.]+%)\)$/)
  return m ? { label: m[1].trim(), count: m[2], share: m[3] } : null
}

export function parseSummary(text: string | null | undefined): SummaryReport | null {
  if (!text) return null
  const verdict = field(text, '判讀結論')
  if (!verdict) return null   // no verdict line = not a summary we understand

  const metrics: ReportMetric[] = []
  const add = (label: string, key: string, hint?: string) => {
    const value = field(text, key)
    if (value) metrics.push({ label, value, hint })
  }
  add('HER2/CEP17 比值', '黑點/紅點 比值')
  add('平均 HER2 拷貝數', '平均黑點數')
  add('HER2 訊號總數', '黑點總數', '黑點')
  add('CEP17 訊號總數', '紅點總數', '紅點')
  add('有效細胞數', '有效細胞數')

  const distribution = text
    .split('\n')
    .map(row)
    .filter((r): r is ReportRow => r !== null)

  // The verdict line carries both the English term and a Chinese gloss; the
  // English one is what decides the colour, since it is the pipeline's own word.
  const amplified = /not amplified|negative/i.test(verdict)
    ? false
    : /amplified|positive/i.test(verdict)
      ? true
      : null

  return { verdict, amplified, metrics, distribution }
}
