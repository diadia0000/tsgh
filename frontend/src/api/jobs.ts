import { useQuery } from '@tanstack/react-query'
import { api } from './client'
import type { components } from './schema'

export type JobStatus = components['schemas']['JobStatus']

const TERMINAL = new Set(['done', 'error'])

/**
 * Poll `/api/jobs/{jobId}` until it reaches a terminal state. Uses TanStack
 * Query's refetchInterval rather than a hand-rolled WebSocket — the dataflow
 * contract (docs/UI/05) is BackgroundTasks + polling.
 */
export function useJob(jobId: string | null) {
  return useQuery<JobStatus>({
    queryKey: ['job', jobId],
    enabled: jobId !== null,
    refetchInterval: (query) => {
      const status = query.state.data?.status
      return status && TERMINAL.has(status) ? false : 6000
    },
    // Keep polling while the tab is in the background. These jobs run for
    // minutes to hours and the user will switch windows; with the default
    // (false) the interval pauses when the tab is hidden, which does not just
    // freeze the status display -- PipelinePanel submits each alignment step
    // only after the previous job's poll reports "done", so a backgrounded tab
    // stalls the whole chain until the user comes back.
    refetchIntervalInBackground: true,
    queryFn: async () => {
      const { data, error } = await api.GET('/api/jobs/{job_id}', {
        params: { path: { job_id: jobId! } },
      })
      // A dead backend yields neither: the fetch fails and openapi-fetch has no
      // response to report. Returning undefined here makes react-query raise
      // "data is undefined" with its query key attached, which is what the user
      // ends up reading -- so name the actual situation instead.
      if (error || !data) throw new Error('無法連線到系統，請確認服務是否仍在執行')
      return data
    },
  })
}
