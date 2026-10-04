import { readdirSync, rmSync, statSync } from "node:fs"
import { join, resolve } from "node:path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig, loadEnv, type Plugin } from "vite"

// COOP/COEP enable SharedArrayBuffer → onnxruntime-web multithreaded WASM.
// Safe here: wasm assets are same-origin from /public, and the lip-reading model comes from
// Hugging Face, which answers with CORS headers (so require-corp lets the fetch through).
// Production (Cloudflare Pages) sends the same headers from public/_headers.
const crossOriginIsolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
}

// Cloudflare Pages rejects a deploy with any file over 25 MiB. The WebGPU ORT build's .wasm files
// (asyncify 27 MB, jsep 28 MB) are only fetched when WebGPU is asked for (VITE_ORT_WEBGPU=1, see
// onnxRecognizer.ts), so other builds leave them out; the WASM runtime's .wasm is 14 MB.
const PAGES_MAX_FILE_BYTES = 25 * 1024 * 1024

function dropWebGpuOnlyWasm(webgpu: boolean): Plugin {
  let outDir = "dist"
  return {
    name: "drop-webgpu-only-wasm",
    apply: "build",
    configResolved(config) {
      outDir = resolve(config.root, config.build.outDir)
    },
    closeBundle() {
      if (webgpu) return
      const ortDir = join(outDir, "ort")
      for (const name of readdirSync(ortDir)) {
        const path = join(ortDir, name)
        if (name.endsWith(".wasm") && statSync(path).size > PAGES_MAX_FILE_BYTES)
          rmSync(path)
      }
    },
  }
}

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, import.meta.dirname, "VITE_")
  return {
    plugins: [
      react(),
      tailwindcss(),
      dropWebGpuOnlyWasm(env.VITE_ORT_WEBGPU === "1"),
    ],
    resolve: {
      alias: {
        "@": resolve(import.meta.dirname, "./src"),
      },
    },
    server: { headers: crossOriginIsolation },
    preview: { headers: crossOriginIsolation },
    // onnxruntime-web ships prebuilt wasm; don't let Vite try to pre-bundle it.
    optimizeDeps: { exclude: ["onnxruntime-web"] },
  }
})
