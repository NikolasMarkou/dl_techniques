# Attention Module

The `dl_techniques.layers.attention` module provides a comprehensive collection of attention mechanisms for deep learning, with a unified factory interface for consistent layer creation, configuration management, and parameter validation.

## Overview

This module includes thirty-one different attention layer types, ranging from standard multi-head attention to specialized variants for vision, efficiency, and advanced modeling. All layers are built using Keras 3 for backend-agnostic compatibility and support full serialization. The factory system ensures a standardized, safe, and introspectable way to integrate any of these attention mechanisms into your models.

## Available Attention Types

The following layers are supported by the factory system with automated parameter validation and defaults:

| Type | Class | Description | Use Case | Input Shape |
|------|-------|-------------|----------|-------------|
| `anchor` | `AnchorAttention` | Hierarchical attention with anchor tokens. | Long-sequence models where full self-attention is too costly. | `(batch, seq_len, dim)` |
| `capsule_routing` | `CapsuleRoutingSelfAttention` | Self-attention with capsule network dynamic routing. | Experimental models aiming for better contextualization. | `(batch, seq_len, dim)` |
| `cbam` | `CBAM` | Convolutional Block Attention Module (Channel + Spatial). | Plug-and-play attention module for any CNN to refine features. | `(batch, H, W, channels)` |
| `channel` | `ChannelAttention` | Channel attention module from CBAM. | CNNs to recalibrate channel-wise feature responses. | `(batch, H, W, channels)` |
| `differential` | `DifferentialMultiHeadAttention` | Dual MHA to amplify signal and cancel noise. | Transformers requiring improved focus and reduced hallucination. | `(batch, seq_len, dim)` |
| `energy` | `EnergyAttention` | Energy-based attention (Energy Transformer). No value matrix: `call()` returns the exact closed-form **negative gradient** of a scalar token-mixing energy. | Energy Transformer blocks whose residual stream is a Lyapunov descent rather than an opaque `attn -> FFN` stack. | `(batch, seq_len, dim)` |
| `fnet` | `FNetFourierTransform` | Parameter-free token mixing with Fourier Transforms. | Efficient replacement for self-attention in sequence models. | `(batch, seq_len, dim)` |
| `gated` | `GatedAttention` | Attention with normalization, partial RoPE, and output gating. | High-performance transformers requiring stability and expressiveness. | `(batch, seq_len, dim)` |
| `group_query` | `GroupedQueryAttention` | GQA with shared K/V heads for efficiency. | Large language models where K/V cache size is a bottleneck. | `(batch, seq_len, dim)` |
| `hopfield` | `HopfieldAttention` | Modern Hopfield Network for pattern retrieval. | Associative memory tasks; mimics standard attention with `update_steps_max=0`. | `(batch, seq_len, dim)` or `[query, key, value]` |
| `lighthouse` | `LighthouseAttention` | Coarse-to-fine pyramid + top-K causal SDPA with scatter-back via segment_sum. | Long-context causal language modeling needing exact attention with sub-quadratic cost. | `(batch, seq_len, dim)` |
| `linear` | `LinearAttention` | Bias-free, degree-1-homogeneous linear (O(N)) attention via a positively-homogeneous feature map + associativity (Miyasawa-compliant, non-causal). | Bias-free denoiser stacks needing O(N), homogeneity-preserving attention; long-sequence self-attention. | `(batch, seq_len, dim)` |
| `mobile_mqa` | `MobileMQA` | Mobile-optimized Multi-Query Attention for vision. | Efficient attention in vision models for mobile and edge devices. | `(batch, H, W, dim)` |
| `multi_head` | `MultiHeadAttention` | Standard Multi-Head Self-Attention (wrapper for cross-attention). | General-purpose self-attention in vision and sequence models. | `(batch, seq_len, dim)` |
| `multi_head_cross` | `MultiHeadCrossAttention` | Unified layer for self- and cross-attention with adaptive softmax. | Core component for encoder-decoders or advanced custom attention. | `query: (batch, q_len, dim)`, `kv: (batch, kv_len, dim)` |
| `multi_head_latent` | `MultiHeadLatentAttention` | DeepSeek-V2 MLA with KV compression. | Efficient LLM inference with small KV cache. | `(batch, seq_len, dim)` |
| `non_local` | `NonLocalAttention` | Non-local attention for capturing long-range dependencies in CNNs. | Augmenting CNNs with global context reasoning. | `(batch, H, W, channels)` |
| `perceiver` | `PerceiverAttention` | Cross-attention from the Perceiver architecture (wrapper for cross-attention). | Cross-modal attention (e.g., text to image) and latent bottleneck models. | `query: (batch, q_len, dim)`, `kv: (batch, kv_len, dim)` |
| `performer` | `PerformerAttention` | Approximates softmax attention with linear complexity via random features. | Models processing very long sequences (e.g., 65K+ tokens). | `(batch, seq_len, dim)` |
| `ring` | `RingAttention` | Exact attention for long sequences via blockwise processing. | Models requiring near-infinite context length with exact attention. | `(batch, seq_len, dim)` |
| `rpc` | `RPCAttention` | Robust attention via Principal Component Pursuit decomposition. | Models needing robustness to noise and adversarial attacks. | `(batch, seq_len, dim)` |
| `shared_weights_cross` | `SharedWeightsCrossAttention`| Cross-attention between modalities with shared weights. | Efficient multi-modal learning where different data types exchange information. | `(batch, total_seq_len, dim)` |
| `spatial` | `SpatialAttention` | Spatial attention module from CBAM. | CNNs to highlight spatially significant feature regions. | `(batch, H, W, channels)` |
| `tripse1` | `TripSE1` | Triplet Attention with Post-Fusion Squeeze-and-Excitation. | Vision tasks needing comprehensive 3D attention (Spatial + Channel). | `(batch, H, W, channels)` |
| `tripse2` | `TripSE2` | Triplet Attention with Pre-Process Squeeze-and-Excitation. | Vision tasks where channel recalibration should precede spatial rotation. | `(batch, H, W, channels)` |
| `tripse3` | `TripSE3` | Triplet Attention with Parallel Squeeze-and-Excitation. | Vision tasks requiring independent spatial and channel modeling. | `(batch, H, W, channels)` |
| `tripse4` | `TripSE4` | Hybrid 3D Attention with Affine Fusion of logits. | Advanced vision tasks requiring deep integration of spatial/channel contexts. | `(batch, H, W, channels)` |
| `single_window` | `SingleWindowAttention` | Single-window Multi-Head Attention over the full sequence as one window (optional relative-position bias). | Vision/sequence models needing windowed attention without grid partitioning. | `(batch, seq_len, dim)` |
| `wave_field` | `WaveFieldAttention` | FFT-based token mixing with a learned wave-field coupling kernel. | Long-sequence models seeking efficient frequency-domain mixing. | `(batch, seq_len, dim)` |
| `window` | `WindowAttention` [^win] | Windowed Multi-Head Attention from Swin Transformer, using grid-based partitioning for efficient local attention. | Vision transformers (e.g., Swin) for efficient local attention. | `(batch, seq_len, dim)` |
| `window_zigzag` | `WindowAttention` [^win] | Windowed attention with zigzag partitioning to group frequency-proximate tokens. Induces a frequency-based locality bias. | Vision models where frequency-domain relationships are important. | `(batch, seq_len, dim)` |

[^win]: The `window` and `window_zigzag` registry entries dispatch through the factory functions `create_grid_window_attention` / `create_zigzag_window_attention` (which set the partitioning mode); the instance they return is a `WindowAttention` layer.

## Call-signature caveats

The factory (`create_attention_layer`) is **construction-only** — it standardizes how layers are *built*, not how they are *called*. Most layers follow the standard self-attention call signature `call(inputs, attention_mask=None, training=None)`, but seven layers deviate for intentional, architectural reasons. These are documented (not "fixed"): renaming them would break serialized configs and existing call sites. When invoking these layers directly, use their native signatures:

| Layer | Non-standard call signature | Reason |
|-------|-----------------------------|--------|
| `rpc_attention` | mask passed as `mask=` (not `attention_mask=`) | Distinct parameter name predates the standard convention. |
| `shared_weights_cross_attention` | requires a positional `split_sizes` argument | Needs explicit per-modality segment boundaries to split the concatenated sequence. |
| `anchor_attention` | accepts **no** mask argument | Anchor/local windowing defines the attention pattern internally. |
| `performer_attention` | accepts **no** mask argument | FAVOR+ linear-attention kernel does not support a dense additive mask. |
| `lighthouse_attention` | accepts **no** mask argument; requires static seq-len | Causality is enforced by the pyramid scatter-back shift; raises `RuntimeError` on dynamic/`None` seq-len. |
| `group_query_attention` | `call(inputs, training=None, attention_mask=None)` (order swapped) | `training` precedes `attention_mask` positionally. |
| `ring_attention` | `call(inputs, training=None, attention_mask=None)` (order swapped) | `training` precedes `attention_mask` positionally. |
| `mobile_mqa` | `call(inputs, training=None, attention_mask=None, return_attention_weights=False)` (order swapped + extra flag) | `training` precedes `attention_mask` positionally; an extra `return_attention_weights` flag follows. |
| `differential_attention` | `call(inputs, attention_mask=None, layer_idx=0, training=None)` (extra positional `layer_idx`) | An extra `layer_idx` positional argument sits between `attention_mask` and `training`. |
| `spatial` | 4D input `(batch, H, W, channels)`; **no** mask argument | Spatial CBAM attention operates over the full feature map; there is no token mask to apply. |
| `non_local` | 4D input `(batch, H, W, channels)`; mask argument **ignored** | Non-local attention computes dense global affinities over spatial positions; a token/sequence mask does not apply. |

For `group_query`/`ring`/`mobile_mqa`, pass `attention_mask` as a keyword argument to avoid the positional-order pitfall. For `differential_attention`, pass `training` as a keyword argument: because `layer_idx` is the 3rd positional parameter (`call(inputs, attention_mask=None, layer_idx=0, training=None)`), a positionally-passed `training` would otherwise bind to `layer_idx`.

## Customization Hooks

Most softmax-based attention layers expose two unified customization hooks (defaults preserve standard behavior):

- `probability_type` / `probability_config` — selects the attention-score normalization via `ProbabilityOutput` (from `dl_techniques.layers.activations.probability_output`). Supported types include `softmax` (default), `sparsemax`, `threshmax`, `adaptive` (entropy-adaptive softmax). Routing/hierarchical modes are rejected (they consume features, not logits).
- `qk_norm_type` / `qk_norm_kwargs` — optional Q/K normalization routed through `create_normalization_layer` (from `dl_techniques.layers.norms.factory`). Set to one of `rms_norm`, `layer_norm`, `zero_centered_rms_norm`, etc., or `None` (default for most layers) for no normalization.

```python
mha = create_attention_layer(
    'multi_head', dim=256, num_heads=8,
    probability_type='sparsemax',          # sparse attention
    qk_norm_type='rms_norm',               # QK-norm for training stability
)
```

Layer-specific notes:
- `gated_attention`: `qk_norm_type` defaults to `'zero_centered_rms_norm'` (cannot be `None`).
- `multi_head_latent_attention` and `lighthouse_attention`: `qk_norm_type` defaults to `'rms_norm'`.
- `hopfield_attention`: `qk_norm_type` defaults to `'layer_norm'` (use `None` for no pattern normalization).
- `ring_attention`: only `qk_norm_type` is exposed; the online-softmax algorithm is mathematically tied to exponential normalization and does not support custom `probability_type`.
- `non_local_attention`: also exposes `output_norm_type` / `output_norm_kwargs` for the spatial output normalization (default `'batch_norm'`).
- Out-of-scope (no hooks): `channel`, `spatial`, `cbam`, `tripse*`, `fnet`, `performer`, `wave_field` — these layers do not use softmax over Q@K^T scores.

## Factory Interface

### Basic Usage

```python
from dl_techniques.layers.attention import create_attention_layer

# Create a standard multi-head attention layer
mha = create_attention_layer('multi_head', dim=256, num_heads=8)

# Create a CBAM block for a CNN
cbam = create_attention_layer('cbam', channels=128, ratio=16)
```

```python
from dl_techniques.layers.attention import create_attention_layer

# Create a standard multi-head attention layer
mha = create_attention_layer('multi_head', dim=256, num_heads=8)

# Create a TripSE block for 3D attention in a CNN
tripse = create_attention_layer('tripse1', kernel_size=7, reduction_ratio=0.0625)
```

### Configuration-Based Creation

```python
from dl_techniques.layers.attention import create_attention_from_config

config = {
    'type': 'group_query',
    'dim': 1024,
    'num_heads': 16,
    'num_kv_heads': 4,
    'name': 'gqa_block_1'
}

gqa_layer = create_attention_from_config(config)
```

### Parameter Discovery

```python
from dl_techniques.layers.attention import get_attention_info

# Get information about all attention types
info = get_attention_info()

# Print requirements for a specific type
gqa_info = info['group_query']
print(f"Required: {gqa_info['required_params']}")
print(f"Optional: {list(gqa_info['optional_params'].keys())}")
```

### Validation

```python
from dl_techniques.layers.attention import validate_attention_config

# Validate configuration before creation
try:
    validate_attention_config('window', dim=96, window_size=7, num_heads=4)
    print("Configuration is valid")
except ValueError as e:
    print(f"Validation error: {e}")
```

## Layer-Specific Parameters

### `anchor`
**Required:** `dim`  
**Optional:** `num_heads` (default: 8), `dropout_rate` (default: 0.0)
```python
attn = create_attention_layer(
    'anchor',
    dim=512,
    num_heads=8,
    dropout_rate=0.1
)
```

### `capsule_routing`
**Required:** `num_heads`  
**Optional:** `key_dim` (default: None), `routing_iterations` (default: 3)
```python
attn = create_attention_layer(
    'capsule_routing',
    num_heads=8,
    key_dim=64,
    routing_iterations=5
)
```

### `cbam`
**Required:** `channels`  
**Optional:** `ratio` (default: 8), `kernel_size` (default: 7)
```python
attn = create_attention_layer(
    'cbam',
    channels=256,
    ratio=16,
    kernel_size=5
)
```

### `channel`
**Required:** `channels`  
**Optional:** `ratio` (default: 8), `use_bias` (default: False)
```python
attn = create_attention_layer(
    'channel',
    channels=256,
    ratio=16
)
```

### `differential`
**Required:** `dim`, `num_heads`, `head_dim`  
**Optional:** `dropout_rate` (default: 0.0), `attention_dropout_rate` (default: 0.0), `lambda_init` (default: 0.8)
```python
attn = create_attention_layer(
    'differential',
    dim=512,
    num_heads=8,
    head_dim=64,
    attention_dropout_rate=0.1
)
```

### `energy`
**Required:** `dim`  
**Optional:** `num_heads` (default: 8), `head_dim` (default: None), `beta` (default: None), `attn_self` (default: False), `kernel_initializer` (default: None)
```python
attn = create_attention_layer(
    'energy',
    dim=768,
    num_heads=12,
    head_dim=64,
    attn_self=False
)
```
**Notes:**
- **Not a weighted sum of values — there is no value matrix.** The layer defines a scalar energy
  `E_ATT(g) = -(1/beta) * sum_h sum_m logsumexp_n(beta * A_hnm)` over bias-free `(head_dim, num_heads, dim)`
  key/query projections, and `call()` returns the exact closed-form `-dE_ATT/dg` (a **descent direction**,
  not a contextualized value). It is therefore **not** a drop-in replacement for `multi_head`: the output is
  an *update* to be added to the residual stream, not a new token representation.
- **Two gradient terms.** The update carries a second, ET-specific term (the token in its *key* role) that is
  absent from vanilla attention and is what makes the recurrent dynamics provably energy-descending. Cost is
  ~2x standard attention's flops at the same `O(N^2)` scaling.
- Also exposes `energy(g, attention_mask=None) -> (B,)` and `update(g, attention_mask=None) -> (B, N, D)`.
- `attn_self=False` (the paper's ET-Full config) masks the diagonal; on a single-token input both the energy
  and the update are exactly zero.
- **Mask semantics deviate from the siblings for rank-2 masks:** a `(B, N)` mask is a per-token *validity*
  mask applied symmetrically to the key **and** query axes (a key-only mask cannot guarantee zero influence
  here, because the second gradient term sums over query columns). `(B, N, N)` / `(B, H, N, N)` masks keep the
  house `(key, query)` semantics.
- Paper: Energy Transformer, [arXiv:2302.07253](https://arxiv.org/abs/2302.07253). Composed by
  `dl_techniques.layers.transformers.energy_transformer.EnergyTransformer`.

### `fnet`
**Required:** None  
**Optional:** `implementation` (default: 'matrix'), `normalize_dft` (default: True)
```python
attn = create_attention_layer(
    'fnet',
    implementation='matrix'
)
```
**Known limitation:** `implementation='fft'` is **not** implemented — a true O(N log N) FFT path does not exist. Requesting `'fft'` emits a one-time warning and transparently falls back to the matrix DFT (`'matrix'`), whose cost is `O(S^2 * D + S * D^2)` (two complex matmuls), not O(N log N). Output is byte-identical to `'matrix'`.

### `gated`
**Required:** `dim`, `num_heads`  
**Optional:** `head_dim` (default: None), `max_seq_len` (default: 4096), `rope_percentage` (default: 0.5), `probability_type` (default: 'softmax'), `probability_config` (default: None), `qk_norm_type` (default: 'zero_centered_rms_norm'), `gate_activation_type` (default: 'sigmoid')
```python
attn = create_attention_layer(
    'gated',
    dim=768,
    num_heads=12,
    max_seq_len=2048
)
```

### `group_query`
**Required:** `dim`, `num_heads`, `num_kv_heads`  
**Optional:** `max_seq_len` (default: 2048), `dropout_rate` (default: 0.0)
```python
attn = create_attention_layer(
    'group_query',
    dim=1024,
    num_heads=16,
    num_kv_heads=4
)
```

### `hopfield`
**Required:** `num_heads`, `key_dim`  
**Optional:** `update_steps_max` (default: 0), `update_steps_eps` (default: 1e-4)
```python
attn = create_attention_layer(
    'hopfield',
    num_heads=8,
    key_dim=64,
    update_steps_max=3
)
```

### `lighthouse`
**Required:** `dim`, `num_heads`
**Optional:** `head_dim` (default: None), `num_levels` (default: 3), `pooling_factor` (default: 4), `top_k` (default: 1536), `full_attention` (default: False), `qk_norm_type` (default: 'rms_norm')
```python
# Pyramid path: 3 levels, branch factor 4, top-1536 entries
attn = create_attention_layer(
    'lighthouse',
    dim=768,
    num_heads=12,
    num_levels=3,
    pooling_factor=4,
    top_k=1536
)

# Stage-2 SDPA-resume: bypasses pyramid for plain causal MHA
attn_full = create_attention_layer(
    'lighthouse',
    dim=768,
    num_heads=12,
    full_attention=True
)
```

### `linear`
**Required:** `dim`
**Optional:** `num_heads` (default: 8), `head_dim` (default: None), `dropout_rate` (default: 0.0), `use_bias` (default: False), `feature_map` (default: 'relu'), `epsilon` (default: 1e-6)
```python
attn = create_attention_layer(
    'linear',
    dim=256,
    num_heads=8,
    feature_map='relu'
)
```
**Notes:**
- **Bias-free + degree-1 homogeneous (Miyasawa-compliant):** with `use_bias=False` (the default) every Q/K/V/output projection is bias-free and `f(alpha*x) = alpha*f(x)` for `alpha > 0`, so it drops into bias-free / additive-Gaussian denoiser stacks without breaking the `residual = sigma^2 * score` identity. The `feature_map` must stay positively homogeneous (`relu`, `relu_squared`, `abs`); `elu_plus_one`/`exp`/`softmax` are rejected because they break degree-1.
- **Non-causal (v1):** `linear.call` has **no** `attention_mask` parameter (it accepts an ignored `mask=` kwarg only for API uniformity). The `N x N` attention matrix is never materialized (`O(N)` associativity path).

### `mobile_mqa`
**Required:** `dim`  
**Optional:** `num_heads` (default: 8), `use_downsampling` (default: False)

> **Mask caveat:** `mobile_mqa.call` accepts an `attention_mask` argument for signature compatibility but **ignores it** — the mask is never applied to the attention scores. Optional spatial downsampling of K/V changes the key/value length, so a general token mask cannot be applied unambiguously. Do not rely on masking with this layer (a documented limitation, like `spatial`).

```python
attn = create_attention_layer(
    'mobile_mqa',
    dim=256,
    num_heads=8,
    use_downsampling=True
)
```

### `multi_head`
**Required:** `dim`  
**Optional:** `num_heads` (default: 8), `dropout_rate` (default: 0.0)
```python
attn = create_attention_layer(
    'multi_head',
    dim=512,
    num_heads=8,
    dropout_rate=0.1
)
```

### `multi_head_cross`
**Required:** `dim`  
**Optional:** `num_heads` (default: 8), `dropout_rate` (default: 0.0), `shared_qk_projections` (default: False), `probability_type` (default: 'softmax'), `probability_config` (default: None), `qk_norm_type` (default: None)
```python
# Create a cross-attention layer with an adaptive (entropy-adaptive softmax)
# attention-probability function
attn = create_attention_layer(
    'multi_head_cross',
    dim=512,
    num_heads=8,
    probability_type='adaptive',
    probability_config={'min_temp': 0.2, 'max_temp': 1.5}
)
```

### `multi_head_latent`
**Required:** `dim`, `num_heads`, `kv_latent_dim`
**Optional:** `qk_nope_head_dim` (default: 128), `qk_rope_head_dim` (default: 64), `v_head_dim` (default: 128), `q_latent_dim` (default: None), `qk_norm_type` (default: 'rms_norm')
```python
attn = create_attention_layer(
    'multi_head_latent',
    dim=2048,
    num_heads=16,
    kv_latent_dim=512,
    q_latent_dim=1536  # Optional query compression
)
```

### `non_local`
**Required:** `attention_channels`  
**Optional:** `output_norm_type` (default: 'batch_norm'), `attention_mode` (default: 'gaussian')
```python
attn = create_attention_layer(
    'non_local',
    attention_channels=128,
    output_norm_type='layer_norm',
    attention_mode='dot_product'
)
```

### `perceiver`
**Required:** `dim`  
**Optional:** `num_heads` (default: 8), `dropout_rate` (default: 0.0)
```python
attn = create_attention_layer(
    'perceiver',
    dim=256,
    num_heads=8,
    dropout_rate=0.1
)
```

### `performer`
**Required:** `dim`
**Optional:** `num_heads` (default: 8), `nb_features` (default: 256), `ortho_scaling` (default: 0.0), `causal` (default: False)
```python
attn = create_attention_layer(
    'performer',
    dim=512,
    num_heads=8,
    nb_features=256
)
```
**Known limitations:**
- `ortho_scaling` applies a plain scalar multiply to the random features; it does **not** perform FAVOR+ orthogonalization (that path is not implemented). See the layer docstring.
- `performer.call` has **no** `attention_mask` parameter (its call signature is `call(inputs, training=None, return_attention_scores=False)`). Factory registration is construction-only; the mask-less call is an intentional, documented quirk and is not renamed. Do not pass `attention_mask` to a factory-created performer layer.

### `ring`
**Required:** `dim`
**Optional:** `num_heads` (default: 8), `block_size` (default: 512), `qk_norm_type` (default: None)
```python
attn = create_attention_layer(
    'ring',
    dim=768,
    num_heads=12,
    block_size=1024
)
```

### `rpc`
**Required:** `dim`
**Optional:** `num_heads` (default: 8), `lambda_sparse` (default: 0.1), `max_pcp_iter` (default: 10), `svd_threshold` (default: 1.0), `probability_type` (default: 'softmax'), `qk_norm_type` (default: None)
```python
attn = create_attention_layer(
    'rpc',
    dim=512,
    num_heads=8,
    lambda_sparse=0.15
)
```
**Known limitations:**
- `lambda_sparse` is a sparsity-regularization weight (`>0`), **not** a 0-1 dropout-style rate.
- `rpc.call` uses a `mask` parameter, **not** `attention_mask` (call signature `call(inputs, mask=None, training=None, return_attention_scores=False)`). Factory registration is construction-only; the parameter name is an intentional, documented quirk and is not renamed. Pass `mask=` (not `attention_mask=`) to a factory-created rpc layer.

### `shared_weights_cross`
**Required:** `dim`  
**Optional:** `num_heads` (default: 8), `dropout_rate` (default: 0.0)
```python
attn = create_attention_layer(
    'shared_weights_cross',
    dim=256,
    num_heads=4,
    dropout_rate=0.1
)
```

### `spatial`
**Required:** None  
**Optional:** `kernel_size` (default: 7), `use_bias` (default: True)
```python
attn = create_attention_layer(
    'spatial',
    kernel_size=5
)
```

### `tripse1` / `tripse2` / `tripse3` / `tripse4`
**Required:** None  
**Optional:** `reduction_ratio` (default: 0.0625), `kernel_size` (default: 7), `use_bias` (default: False)
```python
# Create TripSE1 (Post-Fusion SE)
attn = create_attention_layer(
    'tripse1',
    reduction_ratio=0.125,
    kernel_size=5
)
```

### `window`
**Required:** `dim`, `window_size`, `num_heads`  
**Optional:** `dropout_rate` (default: 0.0), `qkv_bias` (default: True)
```python
attn = create_attention_layer(
    'window',
    dim=96,
    window_size=7,
    num_heads=4,
    dropout_rate=0.05
)
```

### `window_zigzag`
**Required:** `dim`, `window_size`, `num_heads`  
**Optional:** `dropout_rate` (default: 0.0), `qkv_bias` (default: True), `probability_type` (default: 'softmax'), `probability_config` (default: None), `use_relative_position_bias` (default: False)
```python
# Create a zigzag window attention with an adaptive (entropy-adaptive softmax)
# attention-probability function
attn = create_attention_layer(
    'window_zigzag',
    dim=96,
    window_size=7,
    num_heads=4,
    probability_type='adaptive',
    probability_config={'min_temp': 0.1, 'max_temp': 2.0}
)
```

## Direct Layer Instantiation

While the factory is recommended, direct instantiation is always available.

```python
from dl_techniques.layers.attention import MultiHeadAttention, CBAM, TripSE1, WindowAttention

# Direct instantiation (bypasses factory validation and defaults)
mha = MultiHeadAttention(dim=512, num_heads=8)
cbam = CBAM(channels=256, ratio=16)
tripse = TripSE1(reduction_ratio=0.0625, kernel_size=7)
window_attn = WindowAttention(dim=96, window_size=7, num_heads=4)
```

## Integration Patterns

### In a Custom Transformer Block

```python
import keras
from dl_techniques.layers.attention import create_attention_layer

@keras.saving.register_keras_serializable()
class TransformerBlock(keras.layers.Layer):
    def __init__(self, dim, num_heads, attention_type='multi_head', **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.attention_type = attention_type
        
        # Create attention using the factory
        attn_params = {'dim': dim, 'num_heads': num_heads}

        self.attn = create_attention_layer(attention_type, name='attention', **attn_params)
        # ... other layers like FFN, LayerNorm
    
    def call(self, inputs):
        x = self.attn(inputs)
        # ... rest of the block
        return x
```

### In Model Builders with Configuration Files

```python
import json
from dl_techniques.layers.attention import create_attention_from_config

# Load configuration from file
with open('model_config.json', 'r') as f:
    config = json.load(f)

# Create attention layer from the 'attention' section of the config
attention_layer = create_attention_from_config(config['attention'])
```

## Parameter Validation

The factory performs comprehensive validation on layer creation.

**Missing Required Parameters:**
```python
# Raises ValueError: "Required parameters for 'group_query' are missing: ['num_kv_heads']"
create_attention_layer('group_query', dim=512, num_heads=8)
```

**Invalid Value Ranges:**
```python
# Raises ValueError: "Parameter 'num_heads' must be positive"
create_attention_layer('multi_head', dim=256, num_heads=-8)

# Raises ValueError: "Parameter 'dropout_rate' must be between 0.0 and 1.0"
create_attention_layer('multi_head', dim=256, dropout_rate=1.5)
```

**Unknown Attention Type:**
```python
# Raises ValueError: "Unknown attention type 'vanilla_attention'"
create_attention_layer('vanilla_attention', dim=512)
```

## Logging and Debugging

The factory provides detailed logging to aid development.

**INFO Level:** Shows parameters used for layer creation.
```
INFO Creating 'group_query' layer (GroupedQueryAttention) with parameters: {'dim': 1024, 'num_heads': 16, 'num_kv_heads': 4, 'name': 'gqa_block_1', ...}
```

**ERROR Level:** Provides context for failed layer creation.
```
ERROR Failed to create 'group_query' layer (GroupedQueryAttention). Required parameters: ['dim', 'num_heads', 'num_kv_heads']. Provided parameters: ['dim', 'num_heads']. Please verify parameter compatibility. Original error: ...
```

## API Reference

### Functions

-   **`create_attention_layer(attention_type, name=None, **kwargs)`**: Factory for creating attention layers with validation.
-   **`create_attention_from_config(config)`**: Creates a layer from a configuration dictionary.
-   **`validate_attention_config(attention_type, **kwargs)`**: Validates parameters before creation.
-   **`get_attention_info()`**: Returns a dictionary with details about all available attention types.