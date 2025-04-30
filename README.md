# rvnllm

**rvnllm** is a lightweight, modular LLM (Large Language Model) engine focused on performance, clarity, and adaptability. Built in Rust, it aims to provide a clean, traceable implementation of inference for transformer-based models.

## Features

- Efficient memory mapping for large GGUF models
- Modular kernel design (CPU-first, CUDA planned)
- Minimal runtime dependencies
- Clear separation of concerns across crates
- Test-driven development from the ground up

## Status

Work in progress. Currently focused on:
- Basic transformer block inference
- End-to-end token-to-logit pass
- Core ops: matmul, softmax, norm, activation

## License

MIT

