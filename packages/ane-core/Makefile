CC := xcrun clang
CFLAGS := -O2 -Wall -Wno-deprecated-declarations -fobjc-arc
FRAMEWORKS := -framework Foundation -framework CoreML -framework IOSurface
LDFLAGS := $(FRAMEWORKS) -ldl

ROOT_DIR := $(abspath $(CURDIR)/../..)
LOCAL_DIR := $(ROOT_DIR)/.local
BUILD_DIR := $(LOCAL_DIR)/build/ane-core
TEST_ANE := tests/ane
PROBES_DIR := $(ROOT_DIR)/research/probes

CORE_TEST_OBJC_FILES := \
	$(TEST_ANE)/test_weight_reload.m \
	$(TEST_ANE)/test_perf_stats.m \
	$(TEST_ANE)/test_qos_sweep.m \
	$(TEST_ANE)/test_ane_advanced.m
PROBE_OBJC_FILES := \
	$(PROBES_DIR)/api_exploration.m \
	$(PROBES_DIR)/inmem_basic.m \
	$(PROBES_DIR)/inmem_bench.m \
	$(PROBES_DIR)/inmem_peak.m \
	$(PROBES_DIR)/sram_bench.m \
	$(PROBES_DIR)/sram_probe.m
HEADER_FILES := $(shell find include -type f -name '*.h' | sort)
ALL_OBJC_FILES := $(CORE_TEST_OBJC_FILES) $(PROBE_OBJC_FILES)

CORE_TEST_BINS := $(patsubst $(TEST_ANE)/%.m,$(BUILD_DIR)/tests/%,$(CORE_TEST_OBJC_FILES))
PROBE_BINS := $(patsubst $(PROBES_DIR)/%.m,$(BUILD_DIR)/probes/%,$(PROBE_OBJC_FILES))

.PHONY: build clean lint lint-objc fmt fmt-objc tests probes

build: tests probes

tests: $(CORE_TEST_BINS)

probes: $(PROBE_BINS)

$(BUILD_DIR)/tests:
	mkdir -p $@

$(BUILD_DIR)/probes:
	mkdir -p $@

$(BUILD_DIR)/tests/%: $(TEST_ANE)/%.m | $(BUILD_DIR)/tests
	$(CC) $(CFLAGS) -Iinclude -o $@ $< $(LDFLAGS)

$(BUILD_DIR)/probes/%: $(PROBES_DIR)/%.m | $(BUILD_DIR)/probes
	$(CC) $(CFLAGS) -Iinclude -o $@ $< $(LDFLAGS)

lint: lint-objc

lint-objc:
	@set -e; \
	for f in $(ALL_OBJC_FILES); do \
		echo "lint objc $$f"; \
		$(CC) $(CFLAGS) -Iinclude -fsyntax-only $$f; \
	done

fmt: fmt-objc

fmt-objc:
	@set -e; \
	if command -v clang-format >/dev/null 2>&1; then \
		echo "fmt objc (clang-format)"; \
		clang-format -i $(ALL_OBJC_FILES) $(HEADER_FILES); \
	else \
		echo "clang-format not found."; \
		exit 1; \
	fi

clean:
	rm -f $(CORE_TEST_BINS) $(PROBE_BINS)
