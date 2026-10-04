# Vendored third-party code

| Path | Upstream | Commit | Licence |
|---|---|---|---|
| `third_party/auto_avsr/` | https://github.com/mpc001/auto_avsr | `182b62837773ab01052d4ac21ef1d2203ea7d267` | Apache-2.0 (`auto_avsr/LICENSE`) |
| `third_party/espnet/` | Chaplin's espnet copy, https://github.com/amanvirparhar/chaplin | `7aee1f8fca776ce4f63690063310b53573b7d804` | Apache-2.0 (Imperial College / ESPnet headers; `LICENSE-auto_avsr-apache2`) |
| `src/lipread/vendor/` | Chaplin `pipelines/` (model.py, detectors/mediapipe, tokens) | `7aee1f8fca776ce4f63690063310b53573b7d804` | MIT (`vendor/LICENSE-chaplin`) + Apache-2.0 file headers |

Pretrained weights (`LRS3_V_WER19.1`, `lm_en_subword`, HF mirrors `Amanvir/*`) are research /
non-commercial (LRS3 / BBC terms) and are **not** in git.

## Local modifications (`[stormhacks patch]` comments)

- `auto_avsr/`: removed `datamodule/babble_noise.wav`, `preparation/vox-en.id`, `spm/input.txt`
  (audio-noise augmentation, VoxCeleb2 id list, SentencePiece training corpus — 55 MB, unused for
  video-only fine-tuning with the stock unigram5000 model) and `preparation/detectors/retinaface/`.
  `datamodule/data_module.py` only builds `AudioTransform` for audio modalities (it loads the
  removed noise file).
- `vendor/mediapipe/detector.py`: pick the largest face across detections (upstream reset the
  running max inside the loop).
- `vendor/mediapipe/video_process.py`: undefined name in an assert message.
- Everything else is byte-identical to upstream at the commits above.
