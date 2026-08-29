# Embedding Layers Module

The `dl_techniques.layers.embedding` module provides a comprehensive collection of embedding layer implementations for deep learning architectures, particularly for Transformers and models dealing with spatial data. It features a unified factory interface for consistent layer creation, configuration management, and robust parameter validation.

## Overview

This module holds eighteen layer classes. Thirteen are factory-registered, covering patch tokenization, learned absolute positional encodings, fixed 2D sinusoidal encodings, various forms of Rotary Position Embeddings (RoPE), scalar/timestep sinusoidal embeddings, and BERT/ModernBERT/ALBERT-style token embeddings. The remaining five have no factory key and are imported directly. All eighteen are designed for modern Keras 3.x compatibility with full serialization support.

## Available Embedding Types

The thirteen types below are supported by the factory system, which provides automated parameter validation, default value handling, and a consistent creation interface. The five direct-import-only classes are listed in the next section.

| Type | Class | Description | Use Case |
| :--- | :--- | :--- | :--- |
| `patch_1d` | `PatchEmbedding1D` | 1D patch embedding for time series data. | Tokenizing time series or other 1D sequential data for transformers. |
| `patch_2d` | `PatchEmbedding2D` | 2D image to patch embedding for Vision Transformers. | Converting images into a sequence of patch embeddings in ViT models. |
| `positional_learned` | `PositionalEmbedding`| Adds learned, trainable positional embeddings to a sequence. | Standard absolute positional encoding for transformer models. |
| `rope` | `RotaryPositionEmbedding` | Standard RoPE for relative position encoding. | Injecting relative positional information into Q/K vectors in attention. |
| `dual_rope` | `DualRotaryPositionEmbedding` | Dual RoPE with separate global/local configurations. | Gemma3-style models using both global and local attention patterns. |
| `continuous_rope`| `ContinuousRoPE` | RoPE extended for continuous multi-dimensional coordinates. | Positional encoding for data with spatial coordinates (e.g., 3D point clouds). |
| `continuous_sincos`| `ContinuousSinCosEmbed`| Embeds continuous coordinates using sine/cosine functions. | Creating fixed, smooth positional representations for continuous data. |
| `bert_embeddings` | `BertEmbeddings` | BERT embeddings combining word, position, and token type embeddings. | BERT-style language models with configurable normalization. |
| `modern_bert_embeddings` | `ModernBertEmbeddings` | Word + token-type embeddings, normalized; no learned positional embedding (RoPE applied in attention). | ModernBERT-style encoders with rotary attention. |
| `albert_factorized` | `AlbertFactorizedEmbedding` | ALBERT-style factorized embedding (vocab -> bottleneck -> hidden). | Parameter-efficient token embeddings. |
| `positional_sine_2d` | `PositionEmbeddingSine2D` | Fixed 2D sinusoidal positional encoding. Emits **channels-first** `(B, 2*num_pos_feats, H, W)` — caller transposes as needed. | DETR / ViT detectors needing a non-learnable 2D positional grid. |
| `scalar_sinusoidal` | `ScalarSinusoidalEmbedding` | Sinusoidal embedding of a scalar (e.g. diffusion time in `[0,1]`) followed by a SiLU MLP. | Timestep/scalar conditioning in diffusion / flow-matching models. |
| `mrope_ideogram4` | `Ideogram4MRoPE` | 3D multi-axis (t/h/w) rotary position embedding with per-axis frequency-band interleave. | Image/text multimodal positional encoding (Ideogram4-style DiT). |

## Direct-Import-Only Classes

These five classes have no factory key. Import them from their own module and construct them directly; the factory's validation and defaults do not apply.

| Class | Module | Description | Use Case |
| :--- | :--- | :--- | :--- |
| `AxialRoPE2D` | `axial_rope_2d` | Rotates post-head-split `q`/`k` by a token's 2D grid position. Reads the token axis as a row-major flattening of an `(H, W)` grid, so flat index `t` sits at row `t // W`, column `t % W`. | Vision backbones whose tokens carry a 2D position, e.g. the SAM 2 memory attention and the SAM 3 ViTDet trunk. |
| `ClassTokenPrepend` | `class_token` | Prepends one learnable `[CLS]` token to a `(B, L, D)` sequence. Holds a single `(1, 1, D)` weight; the sequence grows by exactly one position. | ViT-style trunks that need a classification token, e.g. DINO v1/v3 and BEiT. |
| `MaskTokenApply` | `mask_token` | Replaces masked positions of `(B, L, D)` with a learnable mask token. Called on the pair `layer((patch_embeddings, mask))` with a boolean `(B, L)` mask, where `True` marks a position to replace (the iBOT convention). Sequence length is unchanged. | Masked-image-modelling pre-training, e.g. DINO v2, BEiT and the vision Energy Transformer. |
| `RegisterTokens` | `register_tokens` | Produces `num_tokens` independent learnable tokens for a batch from a single `(1, R, D)` weight. The input sequence is a batch-size reference only: its values are discarded and the output length is `R`. | Register / storage tokens concatenated onto a ViT sequence, e.g. DINO v2. |
| `HierarchicalCodebookEmbedding` | `hierarchical_codebook_embedding` | Splits each token ID into `num_chunks` fixed-width chunks, looks each chunk up in its own codebook and sums the results. Stores `num_chunks * 2**chunk_bits * output_dim` parameters instead of `vocab_size * output_dim`. | Parameter-efficient vocabulary embedding where the restricted embedding manifold is acceptable. |

```python
from dl_techniques.layers.embedding.class_token import ClassTokenPrepend
from dl_techniques.layers.embedding.hierarchical_codebook_embedding import (
    HierarchicalCodebookEmbedding,
)

cls_token = ClassTokenPrepend(name='cls_token')
codebook = HierarchicalCodebookEmbedding(
    vocab_size=50261, output_dim=128, num_chunks=2,
    epsilon=1e-6,   # see below: the DEFAULT is Keras' 1e-3, not this
)
```

`HierarchicalCodebookEmbedding`'s `epsilon` sets the variance floor of its
internal `LayerNormalization`. Its default is **`1e-3`** — Keras' own default,
which is what the layer used before the parameter existed, so adding the knob
moved no trained model. Every other normalization site in this repo uses
`1e-6` (`BertEmbeddings`, `ModernBertEmbeddings`, `create_normalization_layer`).
**Pass `epsilon=1e-6` explicitly for a new model.**

`AxialRoPE2D` is the one exception to the direct-import rule: it is also re-exported from the package, so `from dl_techniques.layers.embedding import AxialRoPE2D` works.

## Factory Interface

### Basic Usage

```python
from dl_techniques.layers.embedding import create_embedding_layer

# Create a patch embedding layer for a Vision Transformer
patch_embed = create_embedding_layer(
    'patch_2d',
    patch_size=16,
    embed_dim=768,
    name='vit_patch_embed'
)

# Create a standard Rotary Position Embedding layer
rope = create_embedding_layer(
    'rope',
    head_dim=64,
    max_seq_len=4096,
    rope_percentage=0.5
)

# Create BERT embeddings for language modeling
bert_embed = create_embedding_layer(
    'bert_embeddings',
    vocab_size=30522,
    hidden_size=768,
    max_position_embeddings=512,
    type_vocab_size=2,
    normalization_type='rms_norm'
)
```

### Configuration-Based Creation

```python
from dl_techniques.layers.embedding import create_embedding_from_config

config = {
    'type': 'bert_embeddings',
    'vocab_size': 30522,
    'hidden_size': 768,
    'max_position_embeddings': 512,
    'type_vocab_size': 2,
    'dropout_rate': 0.1,
    'normalization_type': 'layer_norm',
    'name': 'bert_embeddings'
}

bert_embed = create_embedding_from_config(config)
```

### Parameter Discovery

```python
# get_embedding_info is not re-exported from the package __init__.
from dl_techniques.layers.embedding.factory import get_embedding_info

# Get information about all embedding types
info = get_embedding_info()

# Print requirements for a specific type
bert_info = info['bert_embeddings']
print(f"Required: {bert_info['required_params']}")
print(f"Optional: {list(bert_info['optional_params'].keys())}")
# Required: ['vocab_size', 'hidden_size', 'max_position_embeddings']
# Optional: ['type_vocab_size', 'initializer_range', 'layer_norm_eps',
#            'dropout_rate', 'normalization_type', 'use_token_type_embeddings',
#            'position_embedding_type', 'mask_zero']

# The result is a DEEP COPY. Editing the returned `required_params` list or
# `optional_params` dict cannot reach `EMBEDDING_REGISTRY`. Only each entry's
# `class` value is shared, because it is the layer type itself.
```

### Validation

```python
from dl_techniques.layers.embedding import validate_embedding_config

# Validate configuration before creation
try:
    validate_embedding_config(
        'bert_embeddings', 
        vocab_size=30522, 
        hidden_size=768,
        max_position_embeddings=512,
        type_vocab_size=2
    )
    print("Configuration is valid")
except ValueError as e:
    print(f"Validation error: {e}")
```

## Layer-Specific Parameters

Every `**Optional**` list below is COMPLETE for its type: it names every key in that type's `optional_params` in the registry, with the registry's own default. `get_embedding_info()` returns the same mapping at runtime. Eight of the thirteen registry types have a subsection here; `albert_factorized`, `modern_bert_embeddings`, `mrope_ideogram4`, `positional_sine_2d` and `scalar_sinusoidal` do not, so read their parameters from `get_embedding_info()`.

### PatchEmbedding1D (`patch_1d`)
- **Required**: `patch_size`, `embed_dim`
- **Optional**: `stride` (default: `None`), `padding` (default: `'causal'`), `use_bias` (default: `True`), `kernel_initializer` (default: `'glorot_uniform'`), `bias_initializer` (default: `'zeros'`)

```python
ts_embed = create_embedding_layer(
    'patch_1d',
    patch_size=16,
    embed_dim=128,
    stride=8  # Create overlapping patches
)
```

### PatchEmbedding2D (`patch_2d`)
- **Required**: `patch_size`, `embed_dim`
- **Optional**: `flatten` (default: `True`), `activation` (default: `'linear'`), `use_bias` (default: `True`), `kernel_initializer` (default: `'glorot_normal'`), `kernel_regularizer` (default: `None`), `bias_initializer` (default: `'zeros'`), `bias_regularizer` (default: `None`)

Set `flatten=False` to get the raw 4D grid `(batch, H/P_h, W/P_w, embed_dim)` instead of the 3D sequence. Window-attention backbones need that layout; the SAM 1 image encoder is the one caller in `src/` that uses it.

```python
img_embed = create_embedding_layer(
    'patch_2d',
    patch_size=(16, 16),
    embed_dim=768
)
```

### PositionalEmbedding (`positional_learned`)
- **Required**: `max_seq_len`, `dim`
- **Optional**: `dropout_rate` (default: `0.0`), `scale` (default: `0.02`), `pos_initializer` (default: `'truncated_normal'`)
- `scale` sets the standard deviation of the DEFAULT initializer only. Pass an
  initializer instance and it is used exactly as given, `scale` ignored.

```python
pos_embed = create_embedding_layer(
    'positional_learned',
    max_seq_len=512,
    dim=768,
    dropout_rate=0.1
)
```

### RotaryPositionEmbedding (`rope`)
- **Required**: `head_dim`, `max_seq_len`
- **Optional**: `rope_theta` (default: `10000.0`), `rope_percentage` (default: `0.5`)

```python
rope_embed = create_embedding_layer(
    'rope',
    head_dim=64,
    max_seq_len=2048,
    rope_percentage=0.25 # Apply RoPE to first 25% of dimensions
)
```

### DualRotaryPositionEmbedding (`dual_rope`)
- **Required**: `head_dim`, `max_seq_len`
- **Optional**: `global_theta_base` (default: `1,000,000.0`), `local_theta_base` (default: `10,000.0`)

```python
dual_rope_embed = create_embedding_layer(
    'dual_rope',
    head_dim=256,
    max_seq_len=8192,
    global_theta_base=500000.0 # Custom base for long context
)
```

### ContinuousRoPE (`continuous_rope`)
- **Required**: `dim`, `ndim`
- **Optional**: `max_wavelength` (default: `10000.0`)

```python
# For 3D point cloud data
continuous_rope_3d = create_embedding_layer(
    'continuous_rope',
    dim=128, # Embedding dimension
    ndim=3   # 3D coordinates (x, y, z)
)
```

### ContinuousSinCosEmbed (`continuous_sincos`)
- **Required**: `dim`, `ndim`
- **Optional**: `max_wavelength` (default: `10000.0`)

```python
# For 2D spatial data
sincos_embed_2d = create_embedding_layer(
    'continuous_sincos',
    dim=256, # Embedding dimension
    ndim=2   # 2D coordinates (x, y)
)
```

### BertEmbeddings (`bert_embeddings`)
- **Required**: `vocab_size`, `hidden_size`, `max_position_embeddings`
- **Optional**: `type_vocab_size` (default: `None`), `initializer_range` (default: `0.02`), `layer_norm_eps` (default: `1e-8`), `dropout_rate` (default: `0.0`), `normalization_type` (default: `'layer_norm'`), `use_token_type_embeddings` (default: `True`), `position_embedding_type` (default: `'learned'`), `mask_zero` (default: `True`)

`type_vocab_size` is optional in the registry but conditionally required at validation time: omit it while `use_token_type_embeddings` is `True`, which is the default, and the factory raises. Pass `use_token_type_embeddings=False` for a model without segment embeddings.

```python
# Standard BERT embeddings
bert_embed = create_embedding_layer(
    'bert_embeddings',
    vocab_size=30522,
    hidden_size=768,
    max_position_embeddings=512,
    type_vocab_size=2
)

# BERT embeddings with RMS normalization for efficiency
bert_rms_embed = create_embedding_layer(
    'bert_embeddings',
    vocab_size=50257,  # GPT-2 vocab size
    hidden_size=1024,
    max_position_embeddings=1024,
    type_vocab_size=1,  # Only one segment type
    normalization_type='rms_norm',
    dropout_rate=0.1
)

# BERT embeddings with Band RMS for improved stability
bert_band_embed = create_embedding_layer(
    'bert_embeddings',
    vocab_size=30522,
    hidden_size=768,
    max_position_embeddings=512,
    type_vocab_size=2,
    normalization_type='band_rms',
    layer_norm_eps=1e-12
)
```

## Direct Layer Instantiation
While the factory is recommended for consistency and validation, direct instantiation is always available.

```python
# The package __init__ re-exports only AxialRoPE2D, the three factory
# functions and the STRICT_DROPPED_KEY_MARKER string, so a layer class must
# be imported from its own module.
from dl_techniques.layers.embedding.patch_embedding import PatchEmbedding2D
from dl_techniques.layers.embedding.rotary_position_embedding import (
    RotaryPositionEmbedding,
)
from dl_techniques.layers.embedding.bert_embeddings import BertEmbeddings

# Direct instantiation (bypasses factory validation and defaults)
patch_embed = PatchEmbedding2D(patch_size=16, embed_dim=768)
rope_embed = RotaryPositionEmbedding(head_dim=64, max_seq_len=1024)
bert_embed = BertEmbeddings(
    vocab_size=30522,
    hidden_size=768,
    max_position_embeddings=512,
    type_vocab_size=2
)
```

## Integration Patterns

### In a BERT-Style Language Model

```python
import keras
from dl_techniques.utils.keras_registration import register_dl_technique

# The package string is the defining module's dotted path -- `my_project.<module>` for your
# own code, `dl_techniques.<module.path>` for code inside this repo. Never a bare
# `@keras.saving.register_keras_serializable()`: its `Custom>ClassName` key carries no module
# path, so two same-named classes claim one slot and the last import wins (`MIGRATIONS.md`).
@register_dl_technique("my_project.bert_input_layer")
class BertInputLayer(keras.layers.Layer):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, 
                 type_vocab_size, normalization_type='layer_norm', **kwargs):
        super().__init__(**kwargs)
        
        # Create BERT embeddings using the factory
        self.embeddings = create_embedding_layer(
            'bert_embeddings',
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            normalization_type=normalization_type,
            name='bert_embeddings'
        )

    def call(self, input_ids, token_type_ids=None, position_ids=None, training=None):
        return self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            training=training
        )
```

### In a Vision Transformer (ViT)

```python
@register_dl_technique("my_project.vit_input_block")
class ViTInputBlock(keras.layers.Layer):
    def __init__(self, patch_size, embed_dim, max_seq_len, **kwargs):
        super().__init__(**kwargs)
        # Create patch and positional embeddings using the factory
        self.patch_embed = create_embedding_layer(
            'patch_2d', patch_size=patch_size, embed_dim=embed_dim
        )
        # Assuming num_patches + 1 (for CLS token) fits in max_seq_len
        self.pos_embed = create_embedding_layer(
            'positional_learned', max_seq_len=max_seq_len, dim=embed_dim
        )

    def call(self, images):
        patches = self.patch_embed(images)
        # Here you would typically add a CLS token
        return self.pos_embed(patches)
```

### In an Attention Block with RoPE

`RotaryPositionEmbedding` takes a POST-head-split 4D tensor
`(batch, heads, seq_len, head_dim)`. A `Dense` projection produces 3D
`(batch, seq_len, heads * head_dim)`, so the reshape and transpose below are
not optional: calling the layer on the raw 3D projection raises
`ValueError: Expected 4D input (batch, heads, seq_len, head_dim), got shape
with 3 dimensions`.

```python
@register_dl_technique("my_project.attention_with_rope")
class AttentionWithRoPE(keras.layers.Layer):
    def __init__(self, num_heads, head_dim, max_seq_len, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv_proj = keras.layers.Dense(num_heads * head_dim * 3)
        self.out_proj = keras.layers.Dense(num_heads * head_dim)
        self.rope = create_embedding_layer(
            'rope', head_dim=head_dim, max_seq_len=max_seq_len
        )

    def _split_heads(self, x):
        # (B, L, H*Dh) -> (B, H, L, Dh). RoPE needs this 4D form.
        b = keras.ops.shape(x)[0]
        length = keras.ops.shape(x)[1]
        x = keras.ops.reshape(x, [b, length, self.num_heads, self.head_dim])
        return keras.ops.transpose(x, [0, 2, 1, 3])

    def call(self, inputs):
        q, k, v = keras.ops.split(self.qkv_proj(inputs), 3, axis=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Apply RoPE to queries and keys BEFORE attention.
        q = self.rope(q)
        k = self.rope(k)

        scale = 1.0 / keras.ops.sqrt(float(self.head_dim))
        scores = keras.ops.matmul(
            q, keras.ops.transpose(k, [0, 1, 3, 2])) * scale
        out = keras.ops.matmul(keras.ops.softmax(scores, axis=-1), v)

        # Merge the heads back: (B, H, L, Dh) -> (B, L, H*Dh).
        b = keras.ops.shape(out)[0]
        length = keras.ops.shape(out)[2]
        out = keras.ops.transpose(out, [0, 2, 1, 3])
        out = keras.ops.reshape(
            out, [b, length, self.num_heads * self.head_dim])
        return self.out_proj(out)


layer = AttentionWithRoPE(num_heads=4, head_dim=64, max_seq_len=512)
layer(keras.random.normal((2, 16, 128))).shape
```

Executed on 2026-08-28, the last line prints `(2, 16, 256)`.

## RoPE pairing conventions — read this before loading foreign weights

This package ships **four** rotary layers and they do **not** all use the same
pairing. A RoPE rotation mixes exactly two channels, and there are two
conventions for choosing the partner:

- **INTERLEAVED** — channel `0` rotates with channel `1`, `2` with `3`, … This
  is Su et al.'s original RoFormer formulation and GPT-J's
  `rotate_every_two`; Meta's official LLaMA release uses it too.
- **SPLIT-HALF** — channel `j` rotates with channel `j + head_dim/2`. This is
  GPT-NeoX, HF's `LlamaModel`, HF Gemma and HF Qwen, i.e. anything built on
  HF's `rotate_half`.

**Both are valid rotations and both give the relative-position property, so
the wrong one trains fine.** The convention is invisible in a config, invisible
in a shape and silent at load time; a checkpoint from the other convention
produces plausible, wrong numbers. It is a real-world trap, not a theoretical
one: huggingface/transformers issue #25199, "[LLaMA] Rotary positional
embedding differs with official implementation", exists because HF's LLaMA is
split-half while Meta's official code is interleaved. The same weights serve
both only because the HF conversion script **permutes the `q_proj`/`k_proj`
rows** (`convert_llama_weights_to_hf.py::permute`). That permutation is what
you need if you want to move a checkpoint between the two columns below.

Measured by impulse (a one-hot at channel 0 at a non-zero position; the
partner index IS the convention). Harness:
`plans/.../evidence/rope_convention_impulse.py`.

| Layer | Convention | Can consume `q_proj`/`k_proj` from | Reference |
|---|---|---|---|
| `RotaryPositionEmbedding` (`rotary_position_embedding.py`) | **INTERLEAVED** (0 ↔ 1) | RoFormer, GPT-J (`rotate_every_two`), Meta's official LLaMA (`apply_rotary_emb`) | Su et al., "RoFormer", arXiv:2104.09864 |
| `AxialRoPE2D` (`axial_rope_2d.py`) | **INTERLEAVED** (0 ↔ 1) | SAM 2's memory-attention RoPE (adjacent-pair packing = upstream's `view_as_complex` on a `(..., -1, 2)` reshape); same set as above | Su et al., arXiv:2104.09864. No axial-specific paper is cited because the design follows SAM 2, and it is a deliberate minority choice against the vision axial-RoPE literature's more common split-half-per-axis form |
| `Ideogram4MRoPE` (`multi_axis_rope.py`) | **SPLIT-HALF** (0 ↔ `head_dim/2`) | Qwen2-VL / Qwen2.5-VL M-RoPE, GPT-NeoX, HF `LlamaModel`, HF Gemma — anything using HF `rotate_half` | Su et al., arXiv:2104.09864; Wang et al., "Qwen2-VL", arXiv:2409.12191. Port of Ideogram4's own layer, which publishes no arXiv id |
| `DualRotaryPositionEmbedding` (`dual_rotary_position_embedding.py`) | **SPLIT-HALF** (0 ↔ `head_dim/2`) | same HF `rotate_half` lineage as the row above | Su et al., arXiv:2104.09864; Google, "Gemma 3 Technical Report" (the dual global/local theta design) |

Two things that are **not** the pairing convention and are easy to confuse
with it: `AxialRoPE2D`'s split of the head dimension into an x-half and a
y-half, and `Ideogram4MRoPE`'s assignment of frequency *slots* to `(t, h, w)`
via `mrope_section`. Both are axis-assignment choices. Only the intra-pair
rotation form has to match a checkpoint.

## Behaviour Changes

Changes here that reject something the package previously accepted, and what
each does to a stored `.keras` archive. Every new raise is paired with a
migration path: the constructor stays strict so fresh code has to be correct,
the deserialization path substitutes and warns so an old checkpoint still
loads.

### 2026-08-28 — `PositionalEmbedding` checks the `max_seq_len` ceiling in `build()`

`build()` now raises `ValueError` when the input's STATICALLY known sequence
length exceeds `max_seq_len`. Previously nothing checked it and the failure was
`InvalidArgumentError: Expected size[1] in [0, 8], but got 20 [Op:Slice]`, from
inside `ops.slice` at call time. A dynamic sequence axis (`None`) carries no
length and is unaffected.

*Migration*: `build_from_config()` — the path Keras uses to replay a recorded
build shape when loading — logs a warning and builds anyway instead of raising,
so a model saved with an over-long recorded shape still loads. Such a layer
still fails at call time on an input that long; the ceiling itself has not
moved. Only `build_from_config()` is relaxed, and the relaxation is cleared in
a `finally`, so the next ordinary `build()` is strict again.

### 2026-08-28 — `PositionEmbeddingSine2D` rejects an odd `num_pos_feats`

`__init__` now raises `ValueError` for an odd `num_pos_feats`. The class
docstring always required an even value; nothing enforced it, and the failure
was `InvalidArgumentError: Shapes of all inputs must match ... [Op:Pack]` from
`ops.stack` at call time. This layer owns no weights and has no `build()`, so
`__init__` is the only construction-time site available.

*Migration*: `from_config()` rounds an odd stored value UP to the next even
number and logs a warning, rather than raising, so an archive written before
the check still deserializes. **The output width changes** in that case, from
`2 * n` to `2 * (n + 1)` channels — a model whose downstream layers were built
against the odd width will fail on the shape. It could never have run at all
before, since the stack raised, so nothing that previously worked is broken.
`from_config()` copies the config; the caller's dict is not modified.

### 2026-08-28 — `ContinuousRoPE` and `ContinuousSinCosEmbed` drop `assert_positive`

The `assert_positive` constructor parameter is **gone** from both classes and
from their `EMBEDDING_REGISTRY` `optional_params`. It was stored, serialized and
echoed by a test, and it triggered no check at any point in either layer's life:
the original `ops.convert_to_numpy(ops.min(coords))` check was removed for graph
safety and never replaced. Passing it now raises `ValueError` from the
constructor (Keras' own unrecognized-keyword error), and `create_embedding_layer('continuous_rope', ..., assert_positive=True)`
raises `ValueError` naming the accepted key set.

It was **not** reinstated as a graph-safe check, and that was decided by
measurement, not preference. `keras.ops` (keras 3.8) exposes no assert op at
all — the only conditional primitive is `ops.cond`, which must return a tensor
from both branches and therefore cannot raise. The one candidate,
`tf.debugging.assert_non_negative`, is TF-only, which breaks this package's
`keras.ops` backend-agnosticism, and was measured **silently dropped under
XLA**: with `tf.function(jit_compile=True)` a negative input produced no error
and TensorFlow logged `Ignoring Assert operator ...`. A check that is inert in
the compiled regime `fit()` uses is the same dead knob wearing a different hat.

*Migration*: `from_config()` on both classes pops an `assert_positive` key left
in a stored config and logs a warning, so an archive written before the removal
still deserializes. Numerics are unaffected — the parameter never influenced a
single output value. The registry's 13 keys and its uniform 5-key entry shape
are unchanged; only the contents of two `optional_params` dicts moved.

## Parameter Validation
The factory performs comprehensive validation, catching common errors before layer creation.

#### Required Parameter Checking
```python
# Raises ValueError: "Required parameters missing for patch_2d: ['embed_dim']"
create_embedding_layer('patch_2d', patch_size=16)

# type_vocab_size is NOT in required_params, so this is not a missing-required
# error. It fails the conditional rule instead:
# Raises ValueError: "type_vocab_size is required and must be positive when
#                     use_token_type_embeddings is True (the default), got None."
create_embedding_layer('bert_embeddings', vocab_size=30522, hidden_size=768, max_position_embeddings=512)
```

#### Value Range Validation
```python
# Raises ValueError: "rope_percentage must be in (0, 1], got 1.5"
create_embedding_layer('rope', head_dim=64, max_seq_len=512, rope_percentage=1.5)

# Raises ValueError: "vocab_size must be positive, got -30522"
create_embedding_layer('bert_embeddings', vocab_size=-30522, hidden_size=768, 
                      max_position_embeddings=512, type_vocab_size=2)

# Raises ValueError: "dropout_rate must be in [0, 1], got 1.5"
create_embedding_layer('bert_embeddings', vocab_size=30522, hidden_size=768,
                      max_position_embeddings=512, type_vocab_size=2, dropout_rate=1.5)
```

#### Invalid String Value Validation
```python
# Raises ValueError: "padding must be 'same', 'valid', or 'causal', got invalid_padding"
create_embedding_layer('patch_1d', patch_size=16, embed_dim=128, padding='invalid_padding')

# Raises ValueError: "normalization_type must be one of ['layer_norm', 'rms_norm', 'band_rms', 'batch_norm'], got invalid_norm"
create_embedding_layer('bert_embeddings', vocab_size=30522, hidden_size=768,
                      max_position_embeddings=512, type_vocab_size=2, normalization_type='invalid_norm')
```

## Logging and Debugging
The factory provides detailed logging to aid in debugging model configurations.

#### Info Level Logging
Shows all parameters passed to each layer, including resolved defaults.
```
INFO Creating 'bert_embeddings' embedding layer with parameters:
INFO   dropout_rate: 0.1
INFO   hidden_size: 768
INFO   initializer_range: 0.02
INFO   layer_norm_eps: 1e-08
INFO   mask_zero: True
INFO   max_position_embeddings: 512
INFO   name: 'bert_embeddings'
INFO   normalization_type: 'rms_norm'
INFO   position_embedding_type: 'learned'
INFO   type_vocab_size: 2
INFO   use_token_type_embeddings: True
INFO   vocab_size: 30522
```

#### Debug Level Logging
Confirms successful layer creation.
```
DEBUG Successfully created 'bert_embeddings' layer: bert_embeddings
```

#### Error Logging
Provides detailed context when layer creation fails.
```
ERROR Failed to create 'bert_embeddings' embedding layer (BertEmbeddings).
  Required params: ['vocab_size', 'hidden_size', 'max_position_embeddings']
  Provided params: ['vocab_size', 'hidden_size', 'max_position_embeddings']
  Check parameter compatibility and types. Use get_embedding_info() for details.
  Original error: type_vocab_size is required and must be positive when use_token_type_embeddings is True (the default), got None. Pass a positive int, or pass use_token_type_embeddings=False for a model without segment embeddings.
```

## BERT Embeddings Advanced Usage

### Custom Normalization Configuration

```python
# Using different normalization types for different use cases
configs = {
    'standard_bert': {
        'type': 'bert_embeddings',
        'vocab_size': 30522,
        'hidden_size': 768,
        'max_position_embeddings': 512,
        'type_vocab_size': 2,
        'normalization_type': 'layer_norm',  # Standard BERT
        'layer_norm_eps': 1e-12
    },
    'efficient_bert': {
        'type': 'bert_embeddings',
        'vocab_size': 30522,
        'hidden_size': 768,
        'max_position_embeddings': 512,
        'type_vocab_size': 2,
        'normalization_type': 'rms_norm',  # Faster training
        'dropout_rate': 0.1
    },
    'stable_bert': {
        'type': 'bert_embeddings',
        'vocab_size': 30522,
        'hidden_size': 768,
        'max_position_embeddings': 512,
        'type_vocab_size': 2,
        'normalization_type': 'band_rms',  # Better stability
        'layer_norm_eps': 1e-8
    }
}

# Create different variants
embeddings = {name: create_embedding_from_config(config) 
              for name, config in configs.items()}
```

### Single Segment Models

```python
# For models that don't use segment embeddings (like GPT-style)
gpt_embed = create_embedding_layer(
    'bert_embeddings',
    vocab_size=50257,  # GPT-2 vocab
    hidden_size=1024,
    max_position_embeddings=1024,
    type_vocab_size=1,  # Only one segment type
    normalization_type='layer_norm',
    dropout_rate=0.1
)
```

### Large Model Configuration

```python
# For large language models with extended context
large_bert_embed = create_embedding_layer(
    'bert_embeddings',
    vocab_size=100000,  # Large vocabulary
    hidden_size=1536,   # Large hidden size
    max_position_embeddings=4096,  # Long sequences
    type_vocab_size=8,  # Multiple segment types
    normalization_type='rms_norm',  # Efficient normalization
    initializer_range=0.01,  # Smaller init for stability
    layer_norm_eps=1e-6,
    dropout_rate=0.05  # Lower dropout for large models
)
```

## API Reference

#### Functions
- `create_embedding_layer(embedding_type, name=None, **kwargs)`: Factory function for creating embedding layers with validation.
- `create_embedding_from_config(config)`: Create an embedding layer from a configuration dictionary.
- `validate_embedding_config(embedding_type, **kwargs)`: Validate configuration parameters before creation.
- `get_embedding_info()`: Get comprehensive information about all available embedding types.

#### Types
- `EmbeddingType`: A `Literal` type defining valid embedding type strings.

#### BERT Embeddings Input/Output
- **Input shapes**:
  - `input_ids`: `(batch_size, sequence_length)` - Token IDs
  - `token_type_ids`: `(batch_size, sequence_length)` - Optional segment IDs
  - `position_ids`: `(batch_size, sequence_length)` - Optional position IDs
- **Output shape**: `(batch_size, sequence_length, hidden_size)` - Combined embeddings

#### Available Normalization Types for BERT Embeddings
- `'layer_norm'`: Standard LayerNormalization (BERT default)
- `'rms_norm'`: RMSNorm for efficiency
- `'band_rms'`: Band-constrained RMS for stability
- `'batch_norm'`: BatchNormalization (less common for transformers)

#### Available Position Embedding Types for BERT Embeddings
- `'learned'`: A trainable position embedding table (default)
- `'sinusoidal'`: A fixed sinusoidal table

Both lists are the tuples `VALID_NORMALIZATION_TYPES` and `VALID_POSITION_EMBEDDING_TYPES` in `bert_embeddings.py`, which `factory.py` imports. They have exactly one home; do not restate them as literals anywhere else.

#### Strict Unknown-Keyword Rule
`create_embedding_layer` raises `ValueError` on any keyword the chosen type does not declare, rather than dropping it. The message carries the substring exported as `STRICT_DROPPED_KEY_MARKER`. Match guards on that constant.

```python
# Raises ValueError: "... 1 unsupported parameter(s) ['dtype']. 'patch_2d'
#                     (PatchEmbedding2D) accepts only [...]"
create_embedding_layer('patch_2d', patch_size=4, embed_dim=32, dtype='float64')
```