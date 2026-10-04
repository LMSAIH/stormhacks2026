import { resolve } from "node:path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// COOP/COEP enable SharedArrayBuffer → onnxruntime-web multithreaded WASM.
// Safe here: wasm assets are same-origin from /public, and the lip-reading model comes from
// Hugging Face, which answers with CORS headers (so require-corp lets the fetch through).
const crossOriginIsolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": resolve(import.meta.dirname, "./src"),
    },
  },
  server: { headers: crossOriginIsolation },
  preview: { headers: crossOriginIsolation },
  // onnxruntime-web ships prebuilt wasm; don't let Vite try to pre-bundle it.
  optimizeDeps: { exclude: ["onnxruntime-web"] },
})
