![Build](https://github.com/rvnllm/rvnllm/actions/workflows/build.yaml/badge.svg?branch=main)
![Test](https://github.com/rvnllm/rvnllm/actions/workflows/test.yaml/badge.svg?branch=main)

    
> "In the stories, Raven was a trickster. A transformer. A messenger. So is this."

# Raven (`rvn`)
A blazing-fast, introspective LLM engine in Rust.

Why the name? In Indigenous mythology, the raven symbolizes:

- **Transformation** — like transformers reshaping context.
- **Trickery** — like quantization bending precision.
- **Communication** — bridging token to token, model to machine.

**`rvn`**, and **`rvnllm`**,  embraces that spirit:
A fast, adaptive, introspective LLM engine and tools written in Rust.

---
Note: Work in Progress. Project is under active development and evolving daily. 

## What is `rvn`?

A **zero-BS**, high-performance GGUF + LLM engine written in Rust. Built for:

- Deep model introspection
- Hardware-aligned inference
- Blazing-fast memory-mapped access
- Debug-first workflows
- Quantization sanity
- Structural & anomaly detection

---

## Core Features (Please note: These features are already available, work in progress or on the roadmap)

- **Memory-mapped GGUF loading**  
  Load 10GB+ models in milliseconds (depending on the hardware)

- **Tensor inspection and decode**  
  Shapes, types, offsets, anomalies (NaN/Inf/Zero), previews

- **Forward pass execution engine**  
  CPU-first (CUDA planned), matmul, softmax, norms, activations, more later

- **Debug + profile tools**  
  Model-wide dumps, per-layer inspection, output logits, cache behavior, quantization checks

- **Validation & analysis**  
  Trigger phrase detection, entropy analysis, tokenizer sanity checks

- **Modular op kernels**  
  Device routing, CPU now, CUDA later

- **Security-conscious design**  
  Bounds checks, offset integrity, format validation, structural checks, anomaly detection, range checks

---

## Philosophy

This isn’t just about speed.  
It’s about respect between you and the machine:
- Align with memory
- Honor structure
- Avoid waste
- Avoid repetition
- Be traceable

---


## CLI Usage Examples

```bash
# List all tensors
rvn list --file model.gguf

# Inspect & decode a specific tensor
rvn decode-test --file model.gguf --name blk.0.attn_q.weight

# Smoke test: single-head attention
rvn forward-simple --file model.gguf --q q.weight --k k.weight --v v.weight

# Dump everything to file
rvn debug --file model.gguf > model_dump.txt
```

---

## Roadmap

- [x] GGUF mmap loader
- [x] Tensor inspection and decode
- [x] Quantized tensor support (Q2_K, Q4_K, etc.)
- [x] CLI utilities (list, inspect, debug)
- [x] Forward pass smoke tests
- [ ] Full transformer block pipeline
- [ ] CUDA backend (WIP)

---
## License

This project is licensed under either of:

- [MIT License](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)
