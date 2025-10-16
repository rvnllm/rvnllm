# rvnllm — A Modern ML Engine

A modern ML engine designed from the ground up for performance, clarity, and control. CPU-first. GPU-optional. No legacy baggage

## What is rvnllm?

`rvnllm` is a next-generation machine learning engine with built-in LLM support — crafted in modern C++ and designed for seamless Rust and Python integration.

The name comes from **RVN** — a nod to *raven* (the bird known for intelligence and adaptability) — and **LLM**, because language models are just the beginning.

At its core, `rvnllm` is an **agentic execution engine**:  
Every ML model is a **node** in a dynamic computation graph.  
Nodes can think, act, and communicate — enabling flexible, composable AI systems beyond monolithic inference.

---

## Why rvnllm?

- **Modern C++ only** — built with GCC 15.2 and C++23 (with an eye on C++26 and C23).
- **Rust-inspired design** — ownership, safety, and clarity over legacy cruft.
- **LLM-native** — but not LLM-only. Any ML model can be a graph node.
- **CPU-first** — no forced GPU dependencies; CUDA is a bonus, not a requirement.
- **Bare-metal where it matters** — Assembly for low-level CPU interactions.
- **Rust up top** — for orchestration, agents, and higher-level runtime.
- **Language bindings** — Python and more, for developer flexibility.

---

## Tech Stack

- **Languages:** Modern C++23 (C++26 features planned), Rust, Assembly  
- **Compiler:** GCC 15.2+  
- **Primary Platform:** CPU-first (CUDA optional)  
- **Bindings:** Python (initial), more to come

---

## Roadmap (Day Zero)

- [ ] Project scaffolding and core build system  
- [ ] Minimal runtime with graph execution skeleton  
- [ ] First node interface design  
- [ ] CPU-level optimization hooks  
- [ ] Rust orchestration layer prototype  
- [ ] Python bindings  
- [ ] First agentic node execution

*Note: This is Day Zero. No promises, only ambition and a good compiler.*

---

## Philosophy

Ravens are intelligent, adaptive, and thrive in the wild.  
`rvnllm` aims to do the same: **lightweight**, **smart**, and **capable of working outside the cloud-comfort zone**.

- Less “framework,” more “engine.”  
- Less “legacy,” more “clarity.”  
- Less “magic,” more “intentional design.”

---

## Contributing (Eventually)

This project is still at its inception.  
In the near future, contributions will be welcomed — but the design philosophy will stay *lean and ruthless*.

If you believe ML infrastructure should be:

- Fast and understandable,  
- Composable, not monolithic,  
- Owned by engineers, not frameworks…

then you’ll fit right in.

---

## License

To be decided.  
(Likely permissive — ravens don’t like cages.)

---

## Status

`rvnllm` is currently in **pre-alpha / day zero**.  
Nothing works. Everything is possible.
