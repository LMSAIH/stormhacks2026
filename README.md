# HearD: silent speech to voice (StormHacks 2026)

Mouth a sentence at a webcam and the app reads your lips and speaks the sentence aloud with
ElevenLabs. It is built for people who can move their lips but can't make sound. Hackathon project:
the lip-reading model's weights are licensed for non-commercial research use only.

## Where things are

| Folder | What | Start here |
|---|---|---|
| `frontend/` | React app: camera, face tracking, mouth crop, on-device reader, phrase memory | [frontend/README.md](frontend/README.md) |
| `ml/` | Lip-reading model: FastAPI server, fine-tuning, ONNX export and quantization, benchmarks | [ml/README.md](ml/README.md) |
| `backend/` | The backend team's FastAPI: Google sign-in, ElevenLabs voice and captions, notes | [backend/.env.example](backend/.env.example) |
| `.context/` | Decision log, measured numbers, plans | [.context/project-brief.md](.context/project-brief.md) |
| `docs/` | Architecture diagrams, Devpost text, demo script and checklist | below |

`./smoke.sh` checks that everything still installs, lints, tests and builds.

## Architecture

Diagrams in [docs/architecture/](docs/architecture/), drawn in mermaid so GitHub renders them:

- [System](docs/architecture/system.md): the browser app, our ML server, the backend team's server
  and Hugging Face, and what crosses the network
- [Lip-reading pipeline](docs/architecture/lipread-pipeline.md): from the camera to the spoken
  sentence
- [Modes](docs/architecture/modes.md): Instant, Normal and Quality, one sequence diagram each
- [Training loop](docs/architecture/ml-training.md): from the training-clip opt-in to a new model in
  the app
- [Deployment](docs/architecture/deployment.md): where each piece runs, its settings and ports
- [Evaluation](docs/architecture/eval.md): what each check measures and where its numbers live

Each diagram ends with the list of files it depicts. `scripts/check_diagrams.sh` prints the
diagrams a change has probably made stale.

Submission material (Devpost text, demo script, pre-demo checklist):
[docs/submission/](docs/submission/).
