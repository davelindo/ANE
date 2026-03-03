# Repository Layout

This repository is organized by product surface instead of by historical phase.

## Packages

- `packages/ane-core`
  - Shared ANE runtime headers and low-level ANE tests.
  - Also owns research probe build wiring (`research/probes`).
- `packages/ane-train`
  - Training binaries and training-focused ANE tests.
  - Tooling: token data extraction and training dashboard.
- `packages/ane-infer`
  - Inference conversion/evaluation pipelines.
  - Nomic path is productionized; ModernBERT path is scaffolded.

## Data and Artifacts

All non-source outputs should live under `.local/`:

- `.local/cache`: downloaded/model cache + token data
- `.local/models`: converted ONNX/CoreML artifacts
- `.local/checkpoints`: training checkpoints
- `.local/runs`: reports and benchmark outputs
- `.local/build`: compiled binaries
- `.local/venvs`: python virtual environments

## Build Orchestration

- Root `Makefile` delegates to package `Makefile`s.
- Package `Makefile`s can still run independently.
- `make e2e-test` is intentionally opinionated: lint/fmt/lint/build/data/fresh-train/resume.
