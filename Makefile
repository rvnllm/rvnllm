# Makefile for rvnllm

CARGO := cargo
FEATURES :=

# Default: debug build
.PHONY: all
all: build

.PHONY: build
build:
	$(CARGO) build --features "$(FEATURES)"

.PHONY: release
release:
	$(CARGO) build --release --features "$(FEATURES)"

.PHONY: cuda
cuda:
	$(MAKE) FEATURES="cuda" build

.PHONY: cuda-release
cuda-release:
	$(MAKE) FEATURES="cuda" release

.PHONY: test
test:
	$(CARGO) test --features "$(FEATURES)"

