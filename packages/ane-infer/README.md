# ane-infer

Inference conversion/evaluation pipelines for ANE targets.

## Pipelines

- `src/ane/infer/nomic`: implemented pipeline for `nomic-ai/nomic-embed-text-v1.5`
- `src/ane/infer/modernbert`: scaffold only (command surface, implementation pending)

## Commands

From repo root:

```bash
make nomic-embed-setup
make nomic-embed-convert
make nomic-embed-bench
make nomic-embed-gate
```

Direct package usage:

```bash
make -C packages/ane-infer nomic-embed-run
```

Artifacts default to `.local/models` and reports to `.local/runs`.
