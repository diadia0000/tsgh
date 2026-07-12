import createClient from 'openapi-fetch'
import type { paths } from './schema'

// One typed client for the whole app. Same-origin: requests hit /api/* which
// Vite proxies to the 127.0.0.1:8000 backend in dev (see vite.config.ts), and
// are served by the embedding backend in the packaged app. The frontend never
// knows a host or a filesystem path (guardrail 2, docs/UI/04).
export const api = createClient<paths>()
