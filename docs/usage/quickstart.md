# Quickstart

## 1) Prepare local directories

```bash
scripts/bootstrap_local_dirs.sh
```

## 2) Lint and format

```bash
make lint
make fmt
```

## 3) Build train binaries

```bash
make train-large
```

Built binary path:

```text
.local/build/ane-train/train_large
```

## 4) Run end-to-end smoke test

```bash
make e2e-test
```

Useful knobs:

```bash
make e2e-test E2E_STEPS_FIRST=5 E2E_STEPS_RESUME=8
make e2e-test E2E_ALLOW_SYNTH_DATA=0
```

Note: `E2E_STEPS_FIRST` must be at least `10` to guarantee checkpoint creation
for the resume stage.

## 5) Nomic embedding pipeline

```bash
make nomic-embed-setup
make nomic-embed-convert
make nomic-embed-bench
```

All reports/artifacts land under `.local/runs` and `.local/models`.
