# Attention Layer Implementation Guide

This guide establishes the architectural standards, coding conventions, and best practices for implementing new attention layers within the `dl_techniques` framework.

**Goal:** Ensure all attention mechanisms—whether standard, efficient, or experimental—share a consistent API, utilize shared infrastructure (RoPE, Norms), and integrate seamlessly with the factory system.

---

## 1. Core Design Philosophy

### The "Factory First" Approach
Like the `ffn` and `embedding` modules, the Attention module relies on a centralized factory. Every new layer must be designed to be instantiated via `create_attention_layer`.
*   **Do not** rely on complex inheritance chains.
*   **Do** favor composition (injecting Norms/RoPE via config) over hardcoding specific implementations.

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

## 3. Shared Components & Infrastructure

**CRITICAL:** Do not re-implement foundational logic. Use the framework's factories to ensure global configuration changes propagate to your layer.

### 3.1. Normalization
Never hardcode `LayerNormalization` or `BatchNormalization`.
```python
# ✅ CORRECT
from dl_techniques.layers.norms import create_normalization_layer

self.norm = create_normalization_layer(
    self.normalization_type,  # e.g., 'rms_norm'
    axis=-1,
    name='norm'
)

# ❌ INCORRECT
self.norm = keras.layers.LayerNormalization(epsilon=1e-6)
```

### 3.2. Rotary Embeddings (RoPE)
Never manually implement rotation logic or instantiate `RotaryPositionEmbedding` directly.
```python
# ✅ CORRECT
from dl_techniques.layers.embedding import create_embedding_layer

self.rope = create_embedding_layer(
    'rope', 
    head_dim=self.head_dim,
    max_seq_len=self.max_seq_len
)

# ❌ INCORRECT
self.rope = RotaryPositionEmbedding(...) 
# ❌ INCORRECT
q_rotated = q * cos + rotate_half(q) * sin
```

### 3.3. Masking
Do not write custom broadcasting logic for masks inside `call()`. Use the utility:
```python
from dl_techniques.layers.attention.utils import apply_attention_mask

# Inside call():
scores = apply_attention_mask(scores, attention_mask)
```

---

## 4. Implementation Patterns

### Pattern A: The Wrapper (Preferred)
If your layer is a variation of standard attention (e.g., specific masking, cross-modal setup), wrap `MultiHeadCrossAttention` or standard `keras.layers.MultiHeadAttention`.

**Example:** `PerceiverAttention` wraps `MultiHeadCrossAttention`.

### Pattern B: The Monolith (For Novel Architectures)
If the mathematical operation is fundamentally different (e.g., `RingAttention`, `LinearAttention`), implement from scratch but follow this structure:

```python
@keras.saving.register_keras_serializable()
class MyNewAttention(keras.layers.Layer):
    def __init__(self, dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        
        # 1. Define Sub-layers (Unbuilt)
        self.query_proj = keras.layers.Dense(dim, name="query_proj")
        self.key_proj = keras.layers.Dense(dim, name="key_proj")
        self.value_proj = keras.layers.Dense(dim, name="value_proj")
        self.output_proj = keras.layers.Dense(dim, name="output_proj")
        
    def build(self, input_shape):
        # 2. Build Sub-layers Explicitly
        B, L, D = input_shape
        self.query_proj.build(input_shape)
        self.key_proj.build(input_shape)
        self.value_proj.build(input_shape)
        self.output_proj.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, mask=None):
        # 3. Use keras.ops for logic
        q = self.query_proj(inputs)
        # ... logic ...
        return self.output_proj(out)
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
        'use_bias': True,
        'special_param': 1.0
    }
}
```

## 6. Common Pitfalls to Avoid

1.  **Implicit Reshaping:** Do not assume 3D inputs (`B, S, D`). If your layer supports Vision (4D), handle it explicitly or validate/raise error in `build`.
2.  **Hardcoded Integers:** Avoid magic numbers for head dimensions. Calculate them dynamically: `self.head_dim = dim // num_heads`.
3.  **Missing `get_config`:** If you add a parameter to `__init__`, you **must** add it to `get_config`.
4.  **Mixed Backends:** Never import `torch` or `tensorflow` directly inside the layer unless absolutely necessary for a specific backend-only optimization (and gate it properly).

---

## 7. Migration of Legacy Layers

If refactoring an existing layer found in the codebase:
1.  Rename `channels` -> `dim`.
2.  Replace manual `BatchNormalization` with `create_normalization_layer`.
3.  Replace manual RoPE math with `create_embedding_layer`.
4.  Extract repeated masking logic to `utils.py`.
5.  Ensure `build()` builds *all* sub-layers.