![CI](https://github.com/rvnllm/rvnllm/actions/workflows/ci.yaml/badge.svg)

      
# rvnllm

## Why the name "Raven"?

In folklore, especially in Pacific Northwest and Native American cultures, the raven is a symbol of **transformation**, **trickery**, and **communication between worlds**.

- **Transformation** — just like transformer models
- **Trickster** — like quantization and optimization algorithms bending precision
- **Messenger** — passing token-to-token, architecture-to-architecture, bridging model and machine

**RavenLLM** or **rvnllm** for short embraces that identity — a fast, adaptive, and introspective engine for modern LLMs.


**rvnllm** 
- a lightweight, modular LLM (Large Language Model) engine focused on performance, clarity, and adaptability. Built in Rust, it aims to provide a clean, traceable implementation of inference for transformer-based models.
- a high-performance, security-enabled GGUF loader and LLM engine built in Rust.

It’s not just about running models — it’s about inspecting, validating, and profiling them with surgical precision. From `tok_embeddings` to attention internals, **rvnllm** gives you visibility, performance, and control.


## Features (WIP - under heavy development, not ready, part of the evolving roadmap)

- Efficient memory mapping for large GGUF models
- Modular kernel design (CPU-first, CUDA planned)
- Minimal runtime dependencies
- Clear separation of concerns across crates
- Test-driven development from the ground up
- GGUF model loader with deep introspection
- Memory-mapped tensor access (fast + low overhead)
- Tensor dissection, shape/dtype validation, offset bounds checking
- Forward pass execution
- Experimental model integrity scanning (trigger words, logits entropy, tokenizer sanity)
- Modular op system (CPU-first, CUDA WIP)

## Status

Work in progress. Currently focused on:
- Basic transformer block inference
- End-to-end token-to-logit pass
- Core ops: matmul, softmax, norm, activation
- Basic GGUF loader

## Example usage (amongst other things, currently constantly changing)

```bash
# Show GGUF model header and metadata
rvnllm info header --file llama2.gguf

# Dump a specific tensor
rvnllm dump tensor --file llama2.gguf --name blk.0.attn_q.weight --format f32

# Run a paranoid integrity check
rvnllm validate --file llama2.gguf --profile paranoid

# Run inference with a model personality mod
rvnllm forward --file llama2.gguf --input "The raven is" --mod wife_0
```

## Build
```
make                  # Debug build (no CUDA)
make cuda             # Debug build with CUDA
make release          # Release build (no CUDA)
make cuda-release     # Release build with CUDA
make test             # Run tests (default features)
```

## License

This project is licensed under either of:

- [MIT License](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)
