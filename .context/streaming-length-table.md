# How long can one lip-reading chunk be? (2026-10-03, D51)

int8 speed model, greedy CTC. Accuracy: 12 stitched LRS3 test clips per length (whole clips joined,
never cut; they jump between speakers, so this tests length, not natural pauses).
Browser delay: Chromium, cross-origin isolated page, the app's own `ort/ort.webgpu.bundle.min.mjs`
on WASM, median of 3 (`ml/scripts/bench_length.py` for the CPU columns).

| chunk | word errors (WER) | browser delay (WASM) | laptop CPU delay (native, 4 threads) |
|---|---|---|---|
| 2 s | 33.3% | 0.7 s | 0.20 s |
| 4 s | 49.3% | 1.4 s | 0.36 s |
| 6 s | 48.5% | 2.1 s | 0.54 s |
| 8 s | 40.4% | 2.8 s | 0.74 s |
| 10 s | 34.7% | 3.5 s | 0.94 s |
| 15 s | 34.3% | 5.3 s | 1.46 s |
| 20 s | 31.4% | 7.2 s | 1.80 s |

What it says:
- Errors don't rise with length up to 20 s. The swings (31–49%) come from which clips landed in each
  group (12 clips, 84–730 words), not from length. Longer chunks give the model more context.
- Delay is the only limit, and it is linear: browser ≈ 0.355 s per second of video, native CPU ≈
  0.09 s per second (4× faster). Browser thread count made no difference (default = 1 thread).
- The teammate's 2.5 s buffer is about the longest chunk the browser reads in under 1 s (~0.9 s).
- Beam search + LM on the pod's RTX 4090 (§11): ~0.9 s for a ~2.5 s clip plus ~0.3 s network.

Open for the user (D51): pick the cap. Re-reading the whole sentence at every short pause (D50)
costs the full row above each time, so a 6 s sentence costs 2.1 s per re-read in the browser.
