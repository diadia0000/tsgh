import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // Backend binds 127.0.0.1:8000 (docs/UI/05 dataflow contract); the frontend
    // only ever talks to /api/* and never sees a filesystem path (guardrail 2).
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
})
