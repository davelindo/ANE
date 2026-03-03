# ANE

Research code for running custom training and inference pipelines on Apple Neural Engine (ANE) using private APIs.

## Repository Layout

```text
packages/
  ane-core/      # ANE runtime headers + low-level tests/probes build wiring
  ane-train/     # Training binaries, ANE training tests, dashboard/token tooling
  ane-infer/     # Inference conversion/eval pipelines (Nomic + ModernBERT scaffold)
research/
  probes/        # Standalone ANE exploration binaries
  notes/         # Experiment notes
.local/          # Local-only artifacts (build/cache/models/runs/checkpoints/venvs)
```

Detailed docs:

- `docs/architecture/repo-layout.md`
- `docs/usage/quickstart.md`
- `docs/migration/layout-v2.md`

## Common Commands

```bash
make lint
make fmt
make build
make train-large
make e2e-test
```

## E2E Test

`make e2e-test` runs:

1. lint
2. fmt
3. re-lint
4. clean
5. build
6. token data prep
7. fresh `train_large`
8. resume `train_large`
9. summary

Artifacts are written to `.local/*`.

## Inference (Nomic Embed)

```bash
make nomic-embed-setup
make nomic-embed-convert
make nomic-embed-smoke-test
make nomic-embed-bench
make nomic-embed-gate
```

## Notes

- This repository uses undocumented Apple ANE interfaces for research purposes.
- The code is expected to run on Apple Silicon macOS hosts with ANE-capable hardware.

## License

MIT (`LICENSE`)
