import { resolve } from "node:path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// COOP/COEP enable SharedArrayBuffer → onnxruntime-web multithreaded WASM.
// Safe here because every wasm/model asset is served same-origin from /public.
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
