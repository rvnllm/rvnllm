[![rvn-info CI](https://github.com/rvnllm/rvnllm/actions/workflows/ci.yml/badge.svg)](https://github.com/rvnllm/rvnllm/actions/workflows/ci.yml)
[![Python Wheels](https://github.com/rvnllm/rvnllm/actions/workflows/build-python.yml/badge.svg)](https://github.com/rvnllm/rvnllm/actions/workflows/build-python.yml)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
[![Built with Rust](https://img.shields.io/badge/built%20with-rust-orange?logo=rust)](https://www.rust-lang.org/)
[![PyPI version](https://img.shields.io/pypi/v/rvnllm?logo=pypi&label=PyPI)](https://pypi.org/project/rvnllm/)


# rvnllm

**Lightning-fast LLM model introspection and diffing for GGUF and safetensors.**

`rvnllm` is a high-performance Python package (with a Rust core) for analyzing, diffing, and inspecting large LLM model files in milliseconds. 
Note: At the moment only Gguf V3 and V2 supported


## Features

- Zero-copy model parsing (mmap-powered)
- Metadata & tensor-level inspection (`shape`, `dtype`, `quant`)
- Structural diffs between models
- Fast, even on 70B+ parameter models
- Cross-platform wheels (Linux, macOS, Windows)
- Full polars dataframe support


## Installation

```bash
pip install rvnllm
```

```
# Inspect a model
iimport rvnllm
]
df = rvnllm.info("Llama-3-70B.Q4_0.gguf")
print(df.head())

# Diff two models
diff = rvnllm.diff("DeepSeek-70B.gguf", "Llama-3.3-70B.gguf")
print(diff.head())
```

## What You Can Do with It  
- Inspect metadata
- Inspect tensor number, shapes and types
- Compare metadata, shape, and dtype between models
- Analyze real-world structural differences in seconds or miliseconds


## Requirements
- Python 3.8+
- No compile-time dependencies (prebuilt wheels)

## About
- Built with Rust for speed, wrapped with Python for usability.
- Ideal for researchers, LLMops, and devs working with model internals.
