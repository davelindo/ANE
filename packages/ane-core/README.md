# ane-core

Shared ANE runtime headers and low-level validation binaries.

## Includes

- `include/ane/core/ane_runtime.h`
- `include/ane/core/ane_mil_gen.h`

## Tests and Probes

- `tests/ane/*.m`: ANE runtime behavior tests
- `research/probes/*.m`: standalone exploration probes (built via this package)

## Commands

```bash
make -C packages/ane-core lint
make -C packages/ane-core build
make -C packages/ane-core tests
make -C packages/ane-core probes
```
