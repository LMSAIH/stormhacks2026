# ORT-web WASM thread timing

How long one read takes in the browser (int8 speed model, random 2.8 s input = 70 frames) at different
`ort.env.wasm.numThreads`, with and without ORT's proxy worker (the app uses the proxy and leaves
threads at ORT's default = half the logical cores, max 4).

```bash
cd frontend && pnpm install                     # copies public/ort/
cp -r public/ort bench/ort-threads/
curl -L -o bench/ort-threads/model.onnx \
  https://huggingface.co/eschmechel/auto-avsr-lrs3-vsr-int8-onnx/resolve/9359b251b8d9b8e2d63bada99013bb92eee3b087/lipread_ctc.int8.onnx
cd bench/ort-threads && python3 serve.py        # http://127.0.0.1:8765, sends COOP/COEP
```

Open `http://127.0.0.1:8765/bench.html?proxy=1&threads=4` (then 6, 8, and no `threads` = default),
wait a few seconds, and read `window.result` in the console (`median_ms` is what matters).

4-core cloud container, headless Chromium (2026-10-04): 1 thread 3.47 s · 2 (default there) 2.1 s ·
4 threads 1.35 s; the proxy worker costs nothing measurable.

Demo laptop, i9-13900H (20 logical cores), headless Chromium, proxy on (2026-10-04): default (4)
1.09 s · 4 threads 1.03 s · 6 → 0.97 s · 8 → 1.15 s · 12 → 0.93 s median. Within noise (≤15%), and
more threads would compete with the face tracker and capture loop: the app keeps ORT's default.
