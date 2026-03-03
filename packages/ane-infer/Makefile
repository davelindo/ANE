ROOT_DIR := $(abspath $(CURDIR)/../..)
LOCAL_DIR := $(ROOT_DIR)/.local

NOMIC_VENV ?= $(LOCAL_DIR)/venvs/nomic-ane
NOMIC_DEPS_STAMP ?= $(LOCAL_DIR)/cache/nomic_embed_ane_deps.stamp
NOMIC_PIPELINE ?= src/ane/infer/nomic/pipeline.py
NOMIC_MODEL_ID ?= nomic-ai/nomic-embed-text-v1.5
NOMIC_REVISION ?=
NOMIC_SEQ_LEN ?= 512
NOMIC_MATRYOSHKA_DIM ?= 512
NOMIC_BACKEND ?= onnx-coreml
NOMIC_ONNX_FILENAME ?= onnx/model_fp16.onnx
NOMIC_MODELS_ROOT ?= $(LOCAL_DIR)/models/nomic_embed_text_v1_5
NOMIC_ARTIFACT ?= $(NOMIC_MODELS_ROOT)/base/fp16.onnx
NOMIC_STATIC_OUTPUT_KIND ?= hidden
NOMIC_STATIC_OUTPUT_MATRYOSHKA_DIM ?= 0
NOMIC_STATIC_VARIANT ?= $(if $(filter embedding,$(NOMIC_STATIC_OUTPUT_KIND)),embedding,hidden)
NOMIC_STATIC_ARTIFACT ?= $(NOMIC_MODELS_ROOT)/static/$(NOMIC_STATIC_VARIANT)/b1_s$(NOMIC_SEQ_LEN).onnx
NOMIC_STATIC_RAW_ARTIFACT ?= $(NOMIC_MODELS_ROOT)/static/$(NOMIC_STATIC_VARIANT)/b1_s$(NOMIC_SEQ_LEN).raw.onnx
NOMIC_COMPUTE_UNIT ?= cpu_and_ne
NOMIC_REPORT_DIR ?= $(LOCAL_DIR)/runs/nomic_ane
NOMIC_BENCH_BATCH_SIZE ?= 8
NOMIC_BENCH_WARMUP ?= 5
NOMIC_BENCH_ITERS ?= 30
NOMIC_QUALITY_TOPK ?= 5
NOMIC_QUALITY_MIN_SPEARMAN ?= 0.99
NOMIC_QUALITY_MAX_AUC_DROP ?= 0.02
NOMIC_QUALITY_REQUIRE_FULL_COREML ?= 1
NOMIC_STSB_SPLIT ?= validation
NOMIC_STSB_MAX_SAMPLES ?= 0
NOMIC_STSB_MIN_CPU_SPEARMAN ?= 0.75
NOMIC_STSB_MAX_SPEARMAN_DROP ?= 0.01
NOMIC_STSB_MAX_MEAN_ABS_DELTA ?= 0.005
NOMIC_STSB_REQUIRE_FULL_COREML ?= 1
NOMIC_GATE_MIN_COVERAGE_PCT ?= 90
NOMIC_GATE_MIN_SPEEDUP ?= 1.7

NOMIC_REVISION_FLAG := $(if $(NOMIC_REVISION),--revision $(NOMIC_REVISION),)
NOMIC_QUALITY_REQUIRE_FULL_COREML_FLAG := $(if $(filter 1 true yes,$(NOMIC_QUALITY_REQUIRE_FULL_COREML)),--quality-require-full-coreml,)
NOMIC_STSB_REQUIRE_FULL_COREML_FLAG := $(if $(filter 1 true yes,$(NOMIC_STSB_REQUIRE_FULL_COREML)),--stsb-require-full-coreml,)
NOMIC_ENV := HOME=$(ROOT_DIR) HF_HOME=$(LOCAL_DIR)/cache/hf HUGGINGFACE_HUB_CACHE=$(LOCAL_DIR)/cache/hf TRANSFORMERS_CACHE=$(LOCAL_DIR)/cache/hf XDG_CACHE_HOME=$(LOCAL_DIR)/cache TMPDIR=$(LOCAL_DIR)/cache/tmp

PY_FILES := $(shell find src -type f -name '*.py' | sort)

.PHONY: lint fmt lint-py fmt-py \
	nomic-embed-setup nomic-embed-convert nomic-embed-smoke-test nomic-embed-run \
	nomic-embed-export-static nomic-embed-analyze nomic-embed-validate-strict \
	nomic-embed-bench nomic-embed-eval-quality nomic-embed-eval-stsb nomic-embed-gate \
	nomic-embed-export-static-embed nomic-embed-cleanup-models

lint: lint-py

fmt: fmt-py

lint-py:
	@set -e; \
	if command -v ruff >/dev/null 2>&1; then \
		echo "lint py (ruff)"; \
		ruff check --select F $(PY_FILES); \
	elif command -v pyflakes >/dev/null 2>&1; then \
		echo "lint py (pyflakes)"; \
		pyflakes $(PY_FILES); \
	else \
		echo "No Python linter found (expected ruff or pyflakes)."; \
		exit 1; \
	fi

fmt-py:
	@set -e; \
	if command -v black >/dev/null 2>&1; then \
		echo "fmt py (black)"; \
		black $(PY_FILES); \
	else \
		echo "black not found."; \
		exit 1; \
	fi

$(NOMIC_VENV)/bin/python:
	python3 -m venv $(NOMIC_VENV)
	$(NOMIC_VENV)/bin/python -m pip install --upgrade pip

$(NOMIC_DEPS_STAMP): requirements.txt | $(NOMIC_VENV)/bin/python
	mkdir -p $(LOCAL_DIR)/cache/tmp $(LOCAL_DIR)/cache/hf
	$(NOMIC_VENV)/bin/python -m pip install -r requirements.txt
	touch $(NOMIC_DEPS_STAMP)

nomic-embed-setup: $(NOMIC_DEPS_STAMP)

nomic-embed-convert: nomic-embed-setup
	$(NOMIC_ENV) $(NOMIC_VENV)/bin/python $(NOMIC_PIPELINE) \
		--backend "$(NOMIC_BACKEND)" \
		--model-id "$(NOMIC_MODEL_ID)" $(NOMIC_REVISION_FLAG) \
		--seq-len $(NOMIC_SEQ_LEN) \
		--artifact "$(NOMIC_ARTIFACT)" \
		--onnx-filename "$(NOMIC_ONNX_FILENAME)" \
		convert

nomic-embed-smoke-test: nomic-embed-setup
	$(NOMIC_ENV) $(NOMIC_VENV)/bin/python $(NOMIC_PIPELINE) \
		--backend "$(NOMIC_BACKEND)" \
		--model-id "$(NOMIC_MODEL_ID)" $(NOMIC_REVISION_FLAG) \
		--seq-len $(NOMIC_SEQ_LEN) \
		--matryoshka-dim $(NOMIC_MATRYOSHKA_DIM) \
		--compute-unit "$(NOMIC_COMPUTE_UNIT)" \
		--artifact "$(NOMIC_ARTIFACT)" \
		--onnx-filename "$(NOMIC_ONNX_FILENAME)" \
		smoke-test

nomic-embed-run: nomic-embed-setup
	$(NOMIC_ENV) $(NOMIC_VENV)/bin/python $(NOMIC_PIPELINE) \
		--backend "$(NOMIC_BACKEND)" \
		--model-id "$(NOMIC_MODEL_ID)" $(NOMIC_REVISION_FLAG) \
		--seq-len $(NOMIC_SEQ_LEN) \
		--matryoshka-dim $(NOMIC_MATRYOSHKA_DIM) \
		--compute-unit "$(NOMIC_COMPUTE_UNIT)" \
		--artifact "$(NOMIC_ARTIFACT)" \
		--onnx-filename "$(NOMIC_ONNX_FILENAME)" \
		run

nomic-embed-export-static: nomic-embed-setup
	$(NOMIC_ENV) $(NOMIC_VENV)/bin/python $(NOMIC_PIPELINE) \
		--backend onnx-coreml \
		--model-id "$(NOMIC_MODEL_ID)" $(NOMIC_REVISION_FLAG) \
		--seq-len $(NOMIC_SEQ_LEN) \
		--static-artifact "$(NOMIC_STATIC_ARTIFACT)" \
		--static-raw-artifact "$(NOMIC_STATIC_RAW_ARTIFACT)" \
		--static-output-kind "$(NOMIC_STATIC_OUTPUT_KIND)" \
		--static-output-matryoshka-dim "$(NOMIC_STATIC_OUTPUT_MATRYOSHKA_DIM)" \
		--report-dir "$(NOMIC_REPORT_DIR)" \
		export-static-onnx

nomic-embed-export-static-embed: nomic-embed-setup
	$(MAKE) nomic-embed-export-static \
		NOMIC_STATIC_OUTPUT_KIND=embedding \
		NOMIC_STATIC_OUTPUT_MATRYOSHKA_DIM=$(NOMIC_MATRYOSHKA_DIM)

nomic-embed-analyze: nomic-embed-setup
	$(NOMIC_ENV) $(NOMIC_VENV)/bin/python $(NOMIC_PIPELINE) \
		--backend onnx-coreml \
		--model-id "$(NOMIC_MODEL_ID)" $(NOMIC_REVISION_FLAG) \
		--artifact "$(NOMIC_ARTIFACT)" \
		--compute-unit "$(NOMIC_COMPUTE_UNIT)" \
		--report-dir "$(NOMIC_REPORT_DIR)" \
		analyze

nomic-embed-validate-strict: nomic-embed-setup
	$(NOMIC_ENV) $(NOMIC_VENV)/bin/python $(NOMIC_PIPELINE) \
		--backend onnx-coreml \
		--model-id "$(NOMIC_MODEL_ID)" $(NOMIC_REVISION_FLAG) \
		--artifact "$(NOMIC_ARTIFACT)" \
		--seq-len $(NOMIC_SEQ_LEN) \
		--matryoshka-dim $(NOMIC_MATRYOSHKA_DIM) \
		--compute-unit "$(NOMIC_COMPUTE_UNIT)" \
		--report-dir "$(NOMIC_REPORT_DIR)" \
		validate-strict

nomic-embed-bench: nomic-embed-setup
	$(NOMIC_ENV) $(NOMIC_VENV)/bin/python $(NOMIC_PIPELINE) \
		--backend onnx-coreml \
		--model-id "$(NOMIC_MODEL_ID)" $(NOMIC_REVISION_FLAG) \
		--artifact "$(NOMIC_ARTIFACT)" \
		--seq-len $(NOMIC_SEQ_LEN) \
		--matryoshka-dim $(NOMIC_MATRYOSHKA_DIM) \
		--compute-unit "$(NOMIC_COMPUTE_UNIT)" \
		--bench-batch-size $(NOMIC_BENCH_BATCH_SIZE) \
		--bench-warmup $(NOMIC_BENCH_WARMUP) \
		--bench-iters $(NOMIC_BENCH_ITERS) \
		--report-dir "$(NOMIC_REPORT_DIR)" \
		bench

nomic-embed-eval-quality: nomic-embed-setup
	$(NOMIC_ENV) $(NOMIC_VENV)/bin/python $(NOMIC_PIPELINE) \
		--backend onnx-coreml \
		--model-id "$(NOMIC_MODEL_ID)" $(NOMIC_REVISION_FLAG) \
		--artifact "$(NOMIC_ARTIFACT)" \
		--seq-len $(NOMIC_SEQ_LEN) \
		--matryoshka-dim $(NOMIC_MATRYOSHKA_DIM) \
		--compute-unit "$(NOMIC_COMPUTE_UNIT)" \
		--quality-topk $(NOMIC_QUALITY_TOPK) \
		--quality-min-spearman $(NOMIC_QUALITY_MIN_SPEARMAN) \
		--quality-max-auc-drop $(NOMIC_QUALITY_MAX_AUC_DROP) \
		$(NOMIC_QUALITY_REQUIRE_FULL_COREML_FLAG) \
		--report-dir "$(NOMIC_REPORT_DIR)" \
		eval-quality

nomic-embed-eval-stsb: nomic-embed-setup
	$(NOMIC_ENV) $(NOMIC_VENV)/bin/python $(NOMIC_PIPELINE) \
		--backend onnx-coreml \
		--model-id "$(NOMIC_MODEL_ID)" $(NOMIC_REVISION_FLAG) \
		--artifact "$(NOMIC_ARTIFACT)" \
		--seq-len $(NOMIC_SEQ_LEN) \
		--matryoshka-dim $(NOMIC_MATRYOSHKA_DIM) \
		--compute-unit "$(NOMIC_COMPUTE_UNIT)" \
		--stsb-split "$(NOMIC_STSB_SPLIT)" \
		--stsb-max-samples $(NOMIC_STSB_MAX_SAMPLES) \
		--stsb-min-cpu-spearman $(NOMIC_STSB_MIN_CPU_SPEARMAN) \
		--stsb-max-spearman-drop $(NOMIC_STSB_MAX_SPEARMAN_DROP) \
		--stsb-max-mean-abs-delta $(NOMIC_STSB_MAX_MEAN_ABS_DELTA) \
		$(NOMIC_STSB_REQUIRE_FULL_COREML_FLAG) \
		--report-dir "$(NOMIC_REPORT_DIR)" \
		eval-stsb

nomic-embed-gate: nomic-embed-setup
	$(NOMIC_ENV) $(NOMIC_VENV)/bin/python $(NOMIC_PIPELINE) \
		--backend onnx-coreml \
		--model-id "$(NOMIC_MODEL_ID)" $(NOMIC_REVISION_FLAG) \
		--artifact "$(NOMIC_ARTIFACT)" \
		--seq-len $(NOMIC_SEQ_LEN) \
		--matryoshka-dim $(NOMIC_MATRYOSHKA_DIM) \
		--compute-unit "$(NOMIC_COMPUTE_UNIT)" \
		--bench-batch-size $(NOMIC_BENCH_BATCH_SIZE) \
		--bench-warmup $(NOMIC_BENCH_WARMUP) \
		--bench-iters $(NOMIC_BENCH_ITERS) \
		--min-coverage-pct $(NOMIC_GATE_MIN_COVERAGE_PCT) \
		--min-speedup $(NOMIC_GATE_MIN_SPEEDUP) \
		--report-dir "$(NOMIC_REPORT_DIR)" \
		gate

nomic-embed-cleanup-models:
	@set -e; \
	mkdir -p "$(NOMIC_MODELS_ROOT)/base" "$(NOMIC_MODELS_ROOT)/static/hidden" "$(NOMIC_MODELS_ROOT)/static/embedding"; \
	move_if_needed() { \
		src="$$1"; dst="$$2"; \
		if [ ! -f "$$src" ]; then return 0; fi; \
		if [ -f "$$dst" ]; then \
			if cmp -s "$$src" "$$dst"; then \
				rm -f "$$src"; \
				printf "Removed duplicate: %s\n" "$$src"; \
			else \
				printf "Keeping source (destination differs): %s -> %s\n" "$$src" "$$dst"; \
			fi; \
			return 0; \
		fi; \
		mkdir -p "$$(dirname "$$dst")"; \
		mv "$$src" "$$dst"; \
		printf "Moved: %s -> %s\n" "$$src" "$$dst"; \
	}; \
	move_if_needed "$(ROOT_DIR)/models/nomic_embed_text_v1_5_fp16.onnx" "$(NOMIC_MODELS_ROOT)/base/fp16.onnx"; \
	move_if_needed "$(ROOT_DIR)/models/nomic_embed_text_v1_5_static_b1_s512.onnx" "$(NOMIC_MODELS_ROOT)/static/hidden/b1_s512.onnx"; \
	move_if_needed "$(ROOT_DIR)/models/nomic_embed_text_v1_5_static_b1_s512.raw.onnx" "$(NOMIC_MODELS_ROOT)/static/hidden/b1_s512.raw.onnx"; \
	move_if_needed "$(ROOT_DIR)/models/nomic_embed_text_v1_5_static_b1_s128.onnx" "$(NOMIC_MODELS_ROOT)/static/hidden/b1_s128.onnx"; \
	move_if_needed "$(ROOT_DIR)/models/nomic_embed_text_v1_5_static_b1_s128.raw.onnx" "$(NOMIC_MODELS_ROOT)/static/hidden/b1_s128.raw.onnx"; \
	move_if_needed "$(ROOT_DIR)/models/nomic_embed_text_v1_5_static_b1_s512_embed.onnx" "$(NOMIC_MODELS_ROOT)/static/embedding/b1_s512.onnx"; \
	move_if_needed "$(ROOT_DIR)/models/nomic_embed_text_v1_5_static_b1_s512_embed.raw.onnx" "$(NOMIC_MODELS_ROOT)/static/embedding/b1_s512.raw.onnx"; \
	move_if_needed "$(ROOT_DIR)/models/nomic_embed_text_v1_5_static_b1_s128_embed.onnx" "$(NOMIC_MODELS_ROOT)/static/embedding/b1_s128.onnx"; \
	move_if_needed "$(ROOT_DIR)/models/nomic_embed_text_v1_5_static_b1_s128_embed.raw.onnx" "$(NOMIC_MODELS_ROOT)/static/embedding/b1_s128.raw.onnx"; \
	printf "Model cleanup complete under %s\n" "$(NOMIC_MODELS_ROOT)"
