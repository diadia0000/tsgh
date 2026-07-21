import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // Bind all interfaces so the browser reaches vite over Tailscale
    // (http://<server-tailscale-ip>:5173). WireGuard tunnels straight in, past
    // the campus firewall, on a connection separate from VSCode's SSH -- so a
    // multi-GB upload never shares the editor's channel and can't freeze it.
    host: true,
    // Backend binds 127.0.0.1:8000 (docs/UI/05 dataflow contract); the frontend
    // only ever talks to /api/* and never sees a filesystem path (guardrail 2).
    // changeOrigin:false forwards the browser's Host so tuspyserver's absolute
    // Location (POST -> Location -> PATCH) points back through this proxy, not
    // at the browser-unreachable 127.0.0.1:8000.
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: false },
    },
  },
})
