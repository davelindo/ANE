# Migration to Package Layout (v2)

## What changed

- Source moved from mixed root/`training/` layout to:
  - `packages/ane-core`
  - `packages/ane-train`
  - `packages/ane-infer`
  - `research/probes`
- Runtime artifacts moved to `.local/*`.

## One-time migration

```bash
scripts/migrate_legacy_artifacts.sh
```

This migrates legacy files when present, including:

- `training/tinystories_data00.bin` -> `.local/cache/tinystories_data00.bin`
- `training/ane_stories110M_ckpt.bin` -> `.local/checkpoints/ane_stories110M_ckpt.bin`
- `.venv-nomic-ane` -> `.local/venvs/nomic-ane`

## Compatibility notes

- Root `make e2e-test` and `make nomic-embed-*` remain the primary interfaces.
- Default paths can be overridden with env vars:
  - `ANE_CKPT_PATH`
  - `ANE_DATA_PATH`
  - `ANE_MODEL_PATH`
