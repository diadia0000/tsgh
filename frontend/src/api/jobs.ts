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
    queryFn: async () => {
      const { data, error } = await api.GET('/api/jobs/{job_id}', {
        params: { path: { job_id: jobId! } },
      })
      if (error) throw new Error('job lookup failed')
      return data
    },
  })
}
