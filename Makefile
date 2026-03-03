ROOT_DIR := $(abspath .)
CORE_DIR := packages/ane-core
TRAIN_DIR := packages/ane-train
INFER_DIR := packages/ane-infer
LOCAL_DIR := $(ROOT_DIR)/.local

E2E_STEPS_FIRST ?= 12
E2E_STEPS_RESUME ?= 14
E2E_ALLOW_SYNTH_DATA ?= 1
E2E_TOKEN_ZIP ?= $(HOME)/tiny_stories_data_pretokenized.zip
E2E_TOKEN_DATA ?= $(LOCAL_DIR)/cache/tinystories_data00.bin
E2E_CKPT_PATH ?= $(LOCAL_DIR)/checkpoints/ane_stories110M_ckpt.bin
E2E_MODEL_PATH ?= $(LOCAL_DIR)/models/stories110M.bin
TRAIN_LARGE_BIN := $(LOCAL_DIR)/build/ane-train/train_large

.PHONY: help lint fmt clean build \
	lint-objc lint-py fmt-objc fmt-py \
	train train-large train_large ane-tests core-tests core-probes \
	nomic-embed-setup nomic-embed-convert nomic-embed-smoke-test nomic-embed-run \
	nomic-embed-export-static nomic-embed-analyze nomic-embed-validate-strict \
	nomic-embed-bench nomic-embed-eval-quality nomic-embed-eval-stsb nomic-embed-gate \
	nomic-embed-export-static-embed nomic-embed-cleanup-models \
	e2e-test

help:
	@printf "ANE monorepo commands\n\n"
	@printf "  make lint|fmt|build|clean\n"
	@printf "  make train-large\n"
	@printf "  make nomic-embed-run (and other nomic-embed-* targets)\n"
	@printf "  make e2e-test\n"

lint:
	$(MAKE) -C $(CORE_DIR) lint
	$(MAKE) -C $(TRAIN_DIR) lint
	$(MAKE) -C $(INFER_DIR) lint

fmt:
	$(MAKE) -C $(CORE_DIR) fmt
	$(MAKE) -C $(TRAIN_DIR) fmt
	$(MAKE) -C $(INFER_DIR) fmt

lint-objc:
	$(MAKE) -C $(CORE_DIR) lint-objc
	$(MAKE) -C $(TRAIN_DIR) lint-objc

lint-py:
	$(MAKE) -C $(TRAIN_DIR) lint-py
	$(MAKE) -C $(INFER_DIR) lint-py

fmt-objc:
	$(MAKE) -C $(CORE_DIR) fmt-objc
	$(MAKE) -C $(TRAIN_DIR) fmt-objc

fmt-py:
	$(MAKE) -C $(TRAIN_DIR) fmt-py
	$(MAKE) -C $(INFER_DIR) fmt-py

build:
	$(MAKE) -C $(CORE_DIR) build
	$(MAKE) -C $(TRAIN_DIR) build

clean:
	$(MAKE) -C $(CORE_DIR) clean
	$(MAKE) -C $(TRAIN_DIR) clean
	rm -f "$(E2E_CKPT_PATH)"

train:
	$(MAKE) -C $(TRAIN_DIR) train

train_large: train-large

train-large:
	$(MAKE) -C $(TRAIN_DIR) train-large

ane-tests:
	$(MAKE) -C $(TRAIN_DIR) ane-tests

core-tests:
	$(MAKE) -C $(CORE_DIR) tests

core-probes:
	$(MAKE) -C $(CORE_DIR) probes

nomic-embed-%:
	$(MAKE) -C $(INFER_DIR) $@

e2e-test:
	@set -e; \
	mkdir -p "$(LOCAL_DIR)/cache" "$(LOCAL_DIR)/checkpoints" "$(LOCAL_DIR)/models" "$(LOCAL_DIR)/runs"; \
	printf "\n\n============================================================\n"; \
	printf "                 END-TO-END TEST PIPELINE\n"; \
	printf "============================================================\n"; \
	printf "Running full lint/format/build/data/train/resume sequence.\n\n"; \
	printf "\n[STAGE 1/9] Lint...\n"; \
	$(MAKE) lint; \
	printf "\n[STAGE 2/9] Format...\n"; \
	$(MAKE) fmt; \
	printf "\n[STAGE 3/9] Re-lint...\n"; \
	$(MAKE) lint; \
	printf "\n[STAGE 4/9] Clean training binaries...\n"; \
	$(MAKE) -C $(TRAIN_DIR) clean; \
	rm -f "$(E2E_CKPT_PATH)"; \
	printf "\n[STAGE 5/9] Build train + train_large + probes...\n"; \
	$(MAKE) -C $(TRAIN_DIR) train train-large; \
	$(MAKE) -C $(CORE_DIR) tests probes; \
	printf "\n[STAGE 6/9] Prepare token data...\n"; \
	if [ -f "$(E2E_TOKEN_DATA)" ]; then \
		printf "  Existing token data detected.\n"; \
	elif [ -f "$(E2E_TOKEN_ZIP)" ]; then \
		printf "  Found token ZIP at $(E2E_TOKEN_ZIP).\n"; \
		$(MAKE) -C $(TRAIN_DIR) tokenize TOKEN_ZIP="$(E2E_TOKEN_ZIP)" TOKEN_DATA="$(E2E_TOKEN_DATA)"; \
	elif [ "$(E2E_ALLOW_SYNTH_DATA)" = "1" ]; then \
		printf "  ZIP missing. Generating synthetic token data.\n"; \
		python3 scripts/generate_synth_tokens.py --output "$(E2E_TOKEN_DATA)" --count 8192 --vocab-size 32000; \
	else \
		printf "  ERROR: token ZIP missing and synthetic data disabled (E2E_ALLOW_SYNTH_DATA=0).\n"; \
		printf "  Put tiny_stories_data_pretokenized.zip in $$HOME or enable synthetic data.\n"; \
		exit 1; \
	fi; \
	printf "\n[STAGE 7/9] Fresh train_large run (--steps=$(E2E_STEPS_FIRST))...\n"; \
	ANE_CKPT_PATH="$(E2E_CKPT_PATH)" ANE_DATA_PATH="$(E2E_TOKEN_DATA)" ANE_MODEL_PATH="$(E2E_MODEL_PATH)" "$(TRAIN_LARGE_BIN)" --steps $(E2E_STEPS_FIRST) </dev/null; \
	if [ ! -f "$(E2E_CKPT_PATH)" ]; then \
		printf "  ERROR: checkpoint was not produced in stage 7; resume path cannot run.\n"; \
		exit 1; \
	fi; \
	printf "\n[STAGE 8/9] Resume train_large run (--resume --steps=$(E2E_STEPS_RESUME))...\n"; \
	ANE_CKPT_PATH="$(E2E_CKPT_PATH)" ANE_DATA_PATH="$(E2E_TOKEN_DATA)" ANE_MODEL_PATH="$(E2E_MODEL_PATH)" "$(TRAIN_LARGE_BIN)" --resume --steps $(E2E_STEPS_RESUME) </dev/null; \
	printf "\n[STAGE 9/9] Summary...\n"; \
	printf "E2E complete. Review output above for failures/warnings.\n\n"
