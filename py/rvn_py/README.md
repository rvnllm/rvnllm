# rvnllm

**rvnllm** is a blazing-fast Python package for inspecting, diffing, and analyzing GGUF-based LLM model files.

> Think `ExifTool`, but for large language models.

## Features

- Zero-copy GGUF model parsing
- Tensor-level metadata inspection
- Structural diffs between model files
- Cross-platform wheels (Linux, macOS, Windows)

## Installation

### pip
```bash
pip install rvnllm
```

### Build from source
```
git clone https://github.com/rvnllm/rvnllm.git
cd py/rvn_py
maturin build --release
pip install dist/*.whl
```

### Usage example
```
import rvnllm

df = rvnllm.info("Llama-3-70B-Q4_0.gguf")
print(df.head())
```
