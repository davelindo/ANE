CC := xcrun clang
CFLAGS := -O2 -Wall -Wno-deprecated-declarations -fobjc-arc
FRAMEWORKS := -framework Foundation -framework CoreML -framework IOSurface -framework Accelerate
LDFLAGS := $(FRAMEWORKS) -ldl

ROOT_DIR := $(abspath $(CURDIR)/../..)
LOCAL_DIR := $(ROOT_DIR)/.local
BUILD_DIR := $(LOCAL_DIR)/build/ane-train
SRC_TINY := src/tiny
SRC_STORIES := src/stories
TEST_ANE := tests/ane
TOOLS_DIR := tools
ANE_CORE_INCLUDE := $(ROOT_DIR)/packages/ane-core/include

TINY_OBJC_FILES := $(SRC_TINY)/train.m $(SRC_TINY)/tiny_train.m $(SRC_TINY)/tiny_train_old.m
STORIES_OBJC_FILES := $(SRC_STORIES)/train_large.m
TRAIN_TEST_OBJC_FILES := \
	$(TEST_ANE)/test_ane_causal_attn.m \
	$(TEST_ANE)/test_ane_sdpa5.m \
	$(TEST_ANE)/test_conv_attn3.m \
	$(TEST_ANE)/test_full_fused.m \
	$(TEST_ANE)/test_fused_bwd.m \
	$(TEST_ANE)/test_fused_qkv.m
ALL_OBJC_FILES := $(TINY_OBJC_FILES) $(STORIES_OBJC_FILES) $(TRAIN_TEST_OBJC_FILES)
HEADER_FILES := $(shell find src -type f -name '*.h' | sort)
PY_FILES := $(TOOLS_DIR)/dashboard.py $(TOOLS_DIR)/tokenize.py

TRAIN_TEST_BINS := $(patsubst $(TEST_ANE)/%.m,$(BUILD_DIR)/tests/%,$(TRAIN_TEST_OBJC_FILES))

TOKEN_ZIP ?= $(HOME)/tiny_stories_data_pretokenized.zip
TOKEN_DATA ?= $(LOCAL_DIR)/cache/tinystories_data00.bin

INCLUDES := -I$(SRC_TINY) -I$(SRC_STORIES) -I$(ANE_CORE_INCLUDE)

.PHONY: build clean lint lint-objc lint-py fmt fmt-objc fmt-py \
	train train-large train_large ane-tests tokenize

build: train train-large ane-tests

train: $(BUILD_DIR)/train

train_large: train-large

train-large: $(BUILD_DIR)/train_large

ane-tests: $(TRAIN_TEST_BINS)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/tests:
	mkdir -p $@

$(BUILD_DIR)/train: $(SRC_TINY)/train.m $(SRC_TINY)/model.h $(SRC_TINY)/forward.h $(SRC_TINY)/backward.h $(ANE_CORE_INCLUDE)/ane/core/ane_runtime.h $(ANE_CORE_INCLUDE)/ane/core/ane_mil_gen.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(SRC_TINY)/train.m $(LDFLAGS)

$(BUILD_DIR)/train_large: $(SRC_STORIES)/train_large.m $(SRC_STORIES)/stories_config.h $(SRC_STORIES)/stories_io.h $(SRC_STORIES)/stories_mil.h $(SRC_STORIES)/stories_cpu_ops.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(SRC_STORIES)/train_large.m $(LDFLAGS)

$(BUILD_DIR)/tests/%: $(TEST_ANE)/%.m | $(BUILD_DIR)/tests
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

tokenize:
	python3 $(TOOLS_DIR)/tokenize.py --zip "$(TOKEN_ZIP)" --output "$(TOKEN_DATA)"

lint: lint-objc lint-py

lint-objc:
	@set -e; \
	for f in $(ALL_OBJC_FILES); do \
		echo "lint objc $$f"; \
		$(CC) $(CFLAGS) $(INCLUDES) -fsyntax-only $$f; \
	done

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

fmt: fmt-objc fmt-py

fmt-objc:
	@set -e; \
	if command -v clang-format >/dev/null 2>&1; then \
		echo "fmt objc (clang-format)"; \
		clang-format -i $(ALL_OBJC_FILES) $(HEADER_FILES); \
	else \
		echo "clang-format not found."; \
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

clean:
	rm -f $(BUILD_DIR)/train $(BUILD_DIR)/train_large $(TRAIN_TEST_BINS)
