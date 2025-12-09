# Attention Layer Implementation Guide

This guide establishes the architectural standards, coding conventions, and best practices for implementing new attention layers within the `dl_techniques` framework.

**Goal:** Ensure all attention mechanisms—whether standard, efficient, or experimental—share a consistent API, utilize shared infrastructure (RoPE, Norms, Probability), and integrate seamlessly with the framework's modular ecosystem.

---

## 1. Core Design Philosophy

### The "Factory First" Approach
The framework relies on centralized factories to ensure consistency, serialization safety, and global configuration management. Your attention layer should not reinvent wheels; it should assemble existing, high-quality components.

*   **Do not** implement custom positional encodings if the **Embedding Module** already provides them.
*   **Do not** write custom probability functions if the **Probability Module** covers them.
*   **Do not** hardcode MLP blocks if the **FFN Module** offers a configurable equivalent.

### Keras 3.0 Compliance
*   **Separation of Concerns:** Create sub-layers in `__init__`, build them in `build()`, execute in `call()`.
*   **Serialization:** Every layer must be decorated with `@keras.saving.register_keras_serializable()` and implement a complete `get_config()`.
*   **Backend Agnostic:** Use `keras.ops` for all tensor manipulations. Avoid `tf.*` or `torch.*` specific calls.

---

## 2. API Standards & Naming Conventions

To resolve inconsistencies found in legacy layers, strictly adhere to these parameter names:

| Parameter Concept | Standard Name | Forbidden Names |
| :--- | :--- | :--- |
| **Model Dimension** | `dim` | `channels`, `attention_channels`, `units` |
| **Head Count** | `num_heads` | `heads`, `n_heads` |
| **Head Dimension** | `head_dim` | `key_dim`, `value_dim` (unless distinct) |
| **Dropout** | `dropout_rate` | `dropout`, `attn_dropout` |
| **Query Projection** | `query_proj` | `w_q`, `q_dense`, `q_linear` |
| **Output Projection** | `output_proj` | `w_o`, `o_dense`, `proj_dense` |

### Standard Constructor Signature
```python
def __init__(
    self,
    dim: int,
    num_heads: int,
    head_dim: Optional[int] = None,
    dropout_rate: float = 0.0,
    use_bias: bool = False,
    # ... layer specific args ...
    **kwargs
)
```

---

## 3. Shared Components & Available Infrastructure

**CRITICAL:** Do not re-implement foundational logic. Check the specific modules below before writing custom code.

### 3.1. Probability Outputs (Attention Scores)
When calculating attention scores ($QK^T$), **never** hardcode `Softmax`. Use the unified `ProbabilityOutput` wrapper. This allows the user to easily switch between standard attention (`softmax`), sparse attention (`sparsemax`, `threshmax`), or entropy-aware attention (`adaptive`) via configuration.

**Usage:**
```python
from dl_techniques.layers.activations import ProbabilityOutput

# In __init__
self.score_activation = ProbabilityOutput(
    probability_type=score_activation_type, # e.g. "softmax"
    type_config=score_activation_config,    # e.g. {"axis": -1}
    name='score_activation'
)

# In call()
attn_weights = self.score_activation(scores)
```

**Supported Probability Types & Configurations:**

| Type | Description | Relevant Config (`type_config`) | Use Case |
| :--- | :--- | :--- | :--- |
| **`softmax`** | Standard exponential normalization. | `{'axis': -1}` | Standard Attention (Transformer). |
| **`sparsemax`** | Euclidean projection to simplex. Outputs exact zeros. | `{'axis': -1}` | Sparse Attention (Interpretability). |
| **`threshmax`** | Differentiable confidence thresholding. | `{'slope': 10.0, 'trainable_slope': False}` | Hard Gating / Noise Filtration. |
| **`adaptive`** | Temperature scaling based on entropy. | `{'min_temp': 0.1, 'max_temp': 1.0}` | Stabilizing training dynamics. |

*(Note: `routing` and `hierarchical` types exist but are generally used for Mixture-of-Experts gating or classification heads, not internal attention scores).*

### 3.2. Embeddings & RoPE (`dl_techniques.layers.embedding`)
For relative positional encoding, use the embedding factory. Do not implement complex rotation math manually.

**Available RoPE Types:** 
*   `rope`: Standard Rotary Position Embedding.
*   `dual_rope`: For architectures splitting global/local contexts (e.g., Gemma).
*   `continuous_rope`: For 3D/spatial data.

```python
# ✅ CORRECT
from dl_techniques.layers.embedding import create_embedding_layer

# In __init__
self.rope = create_embedding_layer(
    'rope', 
    head_dim=self.head_dim, 
    max_seq_len=self.max_seq_len
)

# In call()
q_rotated = self.rope(q)
```

### 3.3. Internal Feed-Forward Logic (`dl_techniques.layers.ffn`)
If your attention mechanism requires complex internal gating, mixing, or compatibility functions (beyond simple linear projections), use the FFN factory.

**Available FFN Types:** `mlp`, `glu`, `geglu`, `swiglu`, `differential`, `gated_mlp`, `logic`, etc.

```python
# ✅ CORRECT: For a complex gating mechanism inside attention
from dl_techniques.layers.ffn import create_ffn_layer

self.gate_generator = create_ffn_layer(
    'glu', 
    hidden_dim=dim, 
    output_dim=dim
)
```

### 3.4. Normalization (`dl_techniques.layers.norms`)
Never hardcode `LayerNormalization`.

```python
# ✅ CORRECT
from dl_techniques.layers.norms import create_normalization_layer
self.norm = create_normalization_layer('layer_norm', axis=-1)
```

---

## 4. Implementation Patterns

### Pattern A: The Wrapper (Preferred)
If your layer is a variation of standard attention (e.g., specific masking, cross-modal setup), wrap `MultiHeadCrossAttention` or standard `keras.layers.MultiHeadAttention`.

### Pattern B: The Monolith (For Novel Architectures)
If the mathematical operation is fundamentally different (e.g., `RingAttention`, `LinearAttention`), implement from scratch using `keras.ops`.

**Example:**
```python
import keras 
from dl_techniques.layers.activations import ProbabilityOutput

@keras.saving.register_keras_serializable()
class MyNewAttention(keras.layers.Layer):
    def __init__(self, dim, num_heads, prob_type="softmax", prob_config=None, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.prob_type = prob_type
        self.prob_config = prob_config or {}
        
        # 1. Define Sub-layers (Unbuilt)
        self.query_proj = keras.layers.Dense(dim, name="query_proj")
        
        # Use ProbabilityOutput for score normalization
        self.score_activation = ProbabilityOutput(
            probability_type=self.prob_type,
            type_config=self.prob_config
        ) 
        self.output_proj = keras.layers.Dense(dim, name="output_proj")
        
    def build(self, input_shape):
        # 2. Build Sub-layers Explicitly
        B, L, D = input_shape
        self.query_proj.build(input_shape)
        # Build probability layer (assuming scores shape B, H, L, L)
        self.score_activation.build((B, self.num_heads, L, L)) 
        self.output_proj.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, mask=None):
        # 3. Use keras.ops for logic
        q = self.query_proj(inputs)
        # ... calculation of scores ...
        attn_weights = self.score_activation(scores) # Apply Softmax/Sparsemax/etc
        # ... application to values ...
        return self.output_proj(out)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "prob_type": self.prob_type,
            "prob_config": self.prob_config
        })
        return config
```

---

## 5. Factory Integration Checklist

When adding a new layer, you must update `dl_techniques/layers/attention/factory.py`.

1.  **Define Type Literal:** Add your layer string to `AttentionType`.
2.  **Update Registry:** Add mapping in `_ATTENTION_LAYER_REGISTRY`.
3.  **Define Params:** Add parameter definitions to `_ATTENTION_PARAMS`.
    *   Define `required` parameters (validation will fail if missing).
    *   Define `optional` parameters with defaults.

**Example `_ATTENTION_PARAMS` entry:**
```python
'my_new_attention': {
    'class': MyNewAttention,
    'required': ['dim', 'num_heads'],
    'optional': {
        'dropout_rate': 0.0,
        'prob_type': 'softmax',     # Expose probability type
        'prob_config': None,        # Expose probability config
        'use_bias': True
    }
}
```

---

## 6. Common Pitfalls to Avoid

1.  **Implicit Reshaping:** Do not assume 3D inputs (`B, S, D`). If your layer supports Vision (4D), handle it explicitly or validate/raise error in `build`.
2.  **Hardcoded Integers:** Avoid magic numbers for head dimensions. Calculate them dynamically: `self.head_dim = dim // num_heads`.
3.  **Missing `get_config`:** If you add a parameter to `__init__`, you **must** add it to `get_config`.
4.  **Mixed Backends:** Never import `torch` or `tensorflow` directly inside the layer unless absolutely necessary for a specific backend-only optimization.
5.  **Ignoring Factories:** Hardcoding `keras.activations.softmax` instead of `ProbabilityOutput`, or hardcoding manual RoPE math instead of `create_embedding_layer('rope')`.

---

## 7. Migration of Legacy Layers

If refactoring an existing layer found in the codebase:
1.  Rename `channels` -> `dim`.
2.  Replace manual `BatchNormalization` with `create_normalization_layer`.
3.  **Replace manual `Softmax` calls with `ProbabilityOutput` (defaulting to 'softmax').**
4.  **Replace manual RoPE math with `create_embedding_layer('rope')`**.
5.  Extract repeated masking logic to `utils.py`.
6.  Ensure `build()` builds *all* sub-layers.