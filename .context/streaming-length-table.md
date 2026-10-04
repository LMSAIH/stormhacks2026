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

## Quality vs Normal on the GPU pod (2026-10-03, Q56-B)

Normal = int8 greedy on this laptop's CPU (what the browser runs, minus WASM). Quality = beam + LM on
a secure-cloud RTX 4090 pod, gzipped crops to `/lipread/crops` from this laptop (round trip incl. the
RunPod proxy). Same clips, same crops. `ml/scripts/bench.py` and `bench_length.py --url`.

Single natural sentences:

| clips | Normal WER | Quality WER | Quality round trip p50 / p95 |
|---|---|---|---|
| LRS3 test, 100 | 28.5% | **22.6%** | 1.1 s / 2.2 s |
| raw_eval real faces, 20 | 25.4% | 29.5% | 1.4 s / 2.5 s |

- LRS3: beam better on 35 clips, worse on 12, same on 53.
- raw_eval: every loss is a GRID clip ("bin blue at f two now"), a made-up command grammar the LM
  pulls toward real English. On the 14 natural sentences (CREMA-D, RAVDESS) beam is better on 4 and
  worse on none. Users say natural sentences, so Quality helps where it matters.

Stitched long clips (speaker changes mid-clip, so the attention decoder and LM get mixed context):

| target | Normal WER | Quality WER (beam 40) | round trip p50 / p95 | beam 10 WER | beam 10 p50 / p95 |
|---|---|---|---|---|---|
| 2 s | 33.3% | 29.8% | 2.4 s / 5.0 s | 39.3% | 1.6 s / 6.0 s |
| 4 s | 49.3% | 52.7% | 2.1 s / 4.9 s | 51.3% | 1.5 s / 4.3 s |
| 6 s | 48.5% | 49.8% | 3.2 s / 10.7 s | 48.1% | 2.2 s / 8.0 s |
| 8 s | 40.4% | 38.2% | 3.5 s / 4.1 s | 38.2% | 3.2 s / 14.0 s |
| 10 s | 34.7% | 40.4% | 6.0 s / 16.2 s | 41.9% | 4.2 s / 12.5 s |
| 15 s | 34.3% | 33.5% | 7.7 s / 9.0 s | 33.3% | 6.5 s / 7.5 s |
| 20 s | 31.4% | 33.0% | 11.3 s / 21.8 s | 35.1% | 8.4 s / 10.4 s |

What it says:
- On long stitched clips beam is no better than greedy, and its delay grows to ~11 s at 20 s (the
  decoder runs once per output word). Beam 10 saves ~25% and reads a little worse: not worth it.
- p95 swings (5–22 s) are mostly the RunPod proxy; the GPU part is ~0.9 s for a short clip.
- The client timeout was a flat 15 s, below the 20 s p95. It is now 10 s + 1 s per second of video
  (30 s at the 20 s cap); a timeout still falls back to the on-device read.
- Open (Q57): keep Quality's cap at 20 s, or lower it to ~8 s where its delay stays near 3 s.
  Real one-speaker long sentences may favour beam more than stitched clips do; needs team takes.
