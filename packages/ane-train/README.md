# ane-train

Training binaries and tooling for ANE experiments.

## Contents

- `src/stories/train_large.m`: main Stories110M training path
- `src/tiny/*.m`: tiny/legacy training experiments
- `tests/ane/*.m`: training-oriented ANE kernel tests
- `tools/tokenize.py`: extracts TinyStories pretokenized data
- `tools/dashboard.py`: terminal dashboard for `train_large`

## Build

From repo root:

```bash
make train-large
```

Direct package build:

```bash
make -C packages/ane-train build
```

Binary outputs:

```text
.local/build/ane-train/train
.local/build/ane-train/train_large
```

## Token Data

```bash
make -C packages/ane-train tokenize \
  TOKEN_ZIP=$HOME/tiny_stories_data_pretokenized.zip \
  TOKEN_DATA=.local/cache/tinystories_data00.bin
```

## Runtime Path Overrides

`train_large` supports:

- `ANE_CKPT_PATH`
- `ANE_DATA_PATH`
- `ANE_MODEL_PATH`

Defaults resolve under `.local/*`.
