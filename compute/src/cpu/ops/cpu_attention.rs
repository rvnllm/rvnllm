use ctensor::tensor_view::{TensorDType, TensorView};
use anyhow::Result;
use crate::cpu::ops::{
    cpu_matmul::matmul, 
    cpu_softmax::softmax};

//[input]  
//   │
//   ├──> (1) Linear projection → Q (query)  
//   ├──> (2) Linear projection → K (key)  
//   ├──> (3) Linear projection → V (value)  
//   │
//   └──> (Attention part)
//            │
//            ├──> Compute attention scores (Q × Kᵀ)  
//            ├──> Scale + Softmax (normalize scores)
//            ├──> Weighted sum of V's
//            ▼
//       [attention output]
//            │
//            ├── Residual connection: add [input]
//            │
//            └── RMSNorm (normalize the sum)
//            ▼
//[residual normalized output]   (This is the output of Attention block.)
//
//   │
//   └──> (Feedforward MLP part)
//            │
//            ├──> Linear layer (expand dimension, think “more neurons”)
//            ├──> GELU activation
//            ├──> Linear layer (reduce back)
//            │
//            ├── Residual connection: add previous output
//            │
//            └── RMSNorm again
//            ▼
//[block final output]
//
//      ┌────────────────────────────┐
//      │         Input X           │  ← Token embedding or previous block
//      └────────────────────────────┘
//                  │
//        ┌─────────┴─────────┐
//        │        Linear     │  Q_proj: Wq • X
//        │     (Q = Wq · X)  │
//        └─────────┬─────────┘
//                  │
//        ┌─────────┴─────────┐
//        │        Linear     │  K_proj: Wk • X
//        │     (K = Wk · X)  │
//        └─────────┬─────────┘
//                  │
//        ┌─────────┴─────────┐
//        │        Linear     │  V_proj: Wv • X
//        │     (V = Wv · X)  │
//        └─────────┬─────────┘
//                  │
//          ┌───────▼───────┐
//          │ Q × Kᵀ        │  → attention scores
//          └───────┬───────┘
//                  │
//          ┌───────▼─── ────┐
//          │ Scale + Softmax│
//          └───────┬────────┘
//                  │
//          ┌───────▼────────┐
//          │ Weights × V    │  ← weighted value mix
//          └───────┬────────┘
//                  │
//       ┌──────────▼──────────┐
//       │ Residual Add ( + X )│
//       └──────────┬──────────┘
//                  │
//            ┌─────▼─────┐
//            │  RMSNorm  │
//            └─────┬─────┘
//                  ▼
//       ┌─────────────────────┐
//       │     Feedforward     │
//       │   (Linear → GeLU →  │
//       │    Linear)          │
//       └───────┬─────────────┘
//               │
//       ┌───────▼────────────┐
//       │ Residual Add ( + ) │ ← to normed attention output
//       └────────┬───────────┘
//                │
//           ┌────▼────┐
//           │ RMSNorm │
//           └────┬────┘
//                ▼
//           [Block Output]
//

//| Step | Type | Notes |
//|:----|:-----|:------|
//| Input               | TensorView        | (tokens embeddings, or previous block output) |
//| Linear layers       | matmul + bias     | (for Q, K, V separately) |
//| Attention           | matmul + softmax  | compute focus |
//| Residual + Norm     | add + RMSNorm     | |
//| FeedForward MLP     | matmul + GELU + matmul            | |
//| Residual + Norm     | add + RMSNorm                     | |
//| Output              | TensorView | Ready for next block |

// Simple rule -> Each Block → takes an input → breathes attention → breathes feedforward → gives an output → feeds to next Block.
// Repeat that 10, 20, 30, 70 times depending on model size.

//Symbol	Meaning	Think of it as...
//Q (Query)	What I’m asking about	A question vector: “What do I want to know?”
//K (Key)	What others represent	Each token's identity tag
//V (Value)	What others carry	The actual content/info you want to borrow
// CPU Attention Forward (single Head)
pub fn attention_forward(
    q: &TensorView, // [1, d_k]
    k: &TensorView, // [n_tokens, d_k]
    v: &TensorView, // [n_tokens, d_v]
    out: &mut [f32], //[1, d, v]
) -> Result<()> {
    // 1. q @ k.T => [1, n_tokens] attention_scores
    let mut attn_scores = vec![0.0f32; k.shape[0]]; // n_tokens
    let k_t = TensorView {
        data: k.data,
        shape: vec![k.shape[1], k.shape[0]], // transpose: 
        dtype: k.dtype
    };
    matmul(q, &k_t, &mut attn_scores)?;

    // 2. Scale scores
    let scale = 1.0 / (q.shape[1] as f32).sqrt();
    for s in attn_scores.iter_mut() {
        *s *= scale;
    }

    // 3. Softmax
    softmax(&TensorView {
        data: unsafe {
            std::slice::from_raw_parts(attn_scores.as_ptr() as *const u8, attn_scores.len() * 4)
        },
        shape: vec![attn_scores.len()],
        dtype: ctensor::tensor_view::TensorDType::F32,
    }, &mut attn_scores)?;

    // 4. attn_weights @ v → output
    // attn_weights [1, n_tokens], v [n_tokens, d_v] => out [1, d_v]
    let attn_tensor = TensorView {
        data: unsafe {
            std::slice::from_raw_parts(attn_scores.as_ptr() as *const u8, attn_scores.len() * 4)
        },
        shape: vec![1, attn_scores.len()],
        dtype: ctensor::tensor_view::TensorDType::F32,
    };
    matmul(&attn_tensor, v, out)?;

    Ok(()) 
}
