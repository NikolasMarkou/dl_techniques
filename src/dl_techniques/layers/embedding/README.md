# `dl_techniques.layers.embedding`

Twenty embedding layer classes: patch tokenizers, learned and fixed positional encodings, four
rotary (RoPE) variants, scalar/timestep sinusoidal embeddings, BERT/ModernBERT/ALBERT token
embeddings and a class-label table with a classifier-free-guidance dropout row. **Fifteen** are
reachable through the factory; the other five are direct-import only.

`create_embedding_layer(embedding_type, name=None, **kwargs)` looks the type up in `EMBEDDING_REGISTRY`, rejects any
keyword the target class does not declare, fills in the registry defaults and constructs. The
factory contract and the registry sizes are owned by `src/dl_techniques/layers/CLAUDE.md`.

The package `__init__` exports ten names — `AxialRoPE2D`, `ClassLabelEmbedding`,
`TimestepEmbedding`, the three factory functions, `STRICT_DROPPED_KEY_MARKER` and the three
pure-NumPy sin-cos table builders of `sincos_pos_embed_2d`. Every other class is imported from its
own module.

## Factory catalogue (15 keys)

| Key | Class | What it is | Pick it when | Required params |
|---|---|---|---|---|
| `patch_1d` | `PatchEmbedding1D` | 1D patch tokenizer. | Turning a time series into transformer tokens. | `patch_size`, `embed_dim` |
| `patch_2d` | `PatchEmbedding2D` | Image -> patch tokens. | ViT-style trunks. | `patch_size`, `embed_dim` |
| `positional_learned` | `PositionalEmbedding` | Trainable absolute position table added to the sequence. | Standard learned absolute positions. | `max_seq_len`, `dim` |
| `positional_sine_2d` | `PositionEmbeddingSine2D` | Fixed 2D sinusoidal grid. Emits **channels-first** `(B, 2*num_pos_feats, H, W)`; transpose it yourself. | DETR-style detectors needing a non-learnable 2D grid. | (none) |
| `scalar_sinusoidal` | `ScalarSinusoidalEmbedding` | Sinusoidal embedding of a scalar followed by a SiLU MLP. | Diffusion / flow-matching timestep conditioning. | `dim` |
| `rope` | `RotaryPositionEmbedding` | Standard RoPE applied to Q/K. | Relative positions inside attention. | `head_dim`, `max_seq_len` |
| `dual_rope` | `DualRotaryPositionEmbedding` | RoPE with separate global and local theta. | Gemma3-style mixed global/local attention. | `head_dim`, `max_seq_len` |
| `continuous_rope` | `ContinuousRoPE` | RoPE over continuous N-D coordinates. | Point clouds and other coordinate-carrying data. | `dim`, `ndim` |
| `continuous_sincos` | `ContinuousSinCosEmbed` | Fixed sin/cos embedding of continuous coordinates. | Smooth non-learned encodings of coordinates. | `dim`, `ndim` |
| `mrope_ideogram4` | `Ideogram4MRoPE` | 3D multi-axis (t/h/w) RoPE with per-axis frequency bands. | Ideogram4-style multimodal DiT. | `head_dim`, `rope_theta`, `mrope_section` |
| `bert_embeddings` | `BertEmbeddings` | Word + position + token-type embeddings, normalized. | BERT-style encoders. | `vocab_size`, `hidden_size`, `max_position_embeddings` |
| `modern_bert_embeddings` | `ModernBertEmbeddings` | Word + token-type only; positions come from RoPE in attention. | ModernBERT-style rotary encoders. | `vocab_size`, `hidden_size`, `type_vocab_size`, `initializer_range`, `layer_norm_eps`, `dropout_rate`, `use_bias` |
| `albert_factorized` | `AlbertFactorizedEmbedding` | Vocab -> bottleneck -> hidden factorization. | Parameter-efficient token embeddings. | `vocab_size`, `bottleneck_dim`, `output_dim` |
| `class_label` | `ClassLabelEmbedding` | Categorical label table with one extra "unconditional" row. The table has `num_classes + 1` rows when `dropout_rate > 0` and `num_classes` rows when it is `0`, so an *optional* parameter sizes a weight. | Conditional diffusion transformers (DiT and descendants) needing a learned null token for classifier-free guidance. | `num_classes`, `hidden_size` |
| `timestep` | `TimestepEmbedding` | Cos-first sinusoidal timestep basis of width `frequency_embedding_size`, refined by `Dense -> SiLU -> Dense`. The DiT/GLIDE numerics: the ladder divides by `half`, **not** `half - 1`, and the basis is `concat([cos, sin])`. **Not** interchangeable with `scalar_sinusoidal`, which differs on both of those plus an input rescale onto `[0, 1e4]`. | Diffusion transformers (DiT and descendants) conditioning on a scalar timestep, where the basis width is independent of the model width. | `hidden_size` |

`modern_bert_embeddings` is the one entry with no optional parameters: everything it takes is
required. Read all defaults from the registry rather than from prose:

```python
from dl_techniques.layers.embedding.factory import get_embedding_info  # not re-exported

info = get_embedding_info()['bert_embeddings']
print(info['required_params'], sorted(info['optional_params']))
```

`get_embedding_info()` returns a deep copy; editing it cannot reach `EMBEDDING_REGISTRY`. Only each
entry's `class` value is shared, because it is the layer type itself.

## Direct-import-only classes (5)

No factory key; the factory's validation and defaults do not apply.

| Class | Module | What it is | Pick it when |
|---|---|---|---|
| `AxialRoPE2D` | `axial_rope_2d` | Rotates post-head-split `q`/`k` by a token's 2D grid position, reading the token axis as a row-major `(H, W)` flattening (flat index `t` -> row `t // W`, column `t % W`). | Vision backbones whose tokens carry a 2D position (SAM 2 memory attention, SAM 3 ViTDet trunk). |
| `ClassTokenPrepend` | `class_token` | Prepends one learnable `[CLS]` token to `(B, L, D)`; one `(1, 1, D)` weight, sequence grows by one. | ViT trunks needing a classification token (DINO v1/v3, BEiT). |
| `MaskTokenApply` | `mask_token` | Replaces masked positions of `(B, L, D)` with a learnable token. Called as `layer((patch_embeddings, mask))` with a boolean `(B, L)` mask where `True` marks a position to replace (the iBOT convention). Length unchanged. | Masked-image-modelling pre-training (DINO v2, BEiT, vision Energy Transformer). |
| `RegisterTokens` | `register_tokens` | `num_tokens` independent learnable tokens from one `(1, R, D)` weight. The input sequence is a batch-size reference only — its values are discarded and the output length is `R`. | Register/storage tokens concatenated onto a ViT sequence (DINO v2). |
| `HierarchicalCodebookEmbedding` | `hierarchical_codebook_embedding` | Splits each token ID into `num_chunks` fixed-width chunks, looks each up in its own codebook and sums. Stores `num_chunks * 2**chunk_bits * output_dim` parameters instead of `vocab_size * output_dim`. | Parameter-efficient vocabulary embedding, when a restricted embedding manifold is acceptable. |

`AxialRoPE2D` is the one of the five also re-exported from the package.

```python
from dl_techniques.layers.embedding.class_token import ClassTokenPrepend
from dl_techniques.layers.embedding.hierarchical_codebook_embedding import (
    HierarchicalCodebookEmbedding,
)

cls_token = ClassTokenPrepend(name='cls_token')
codebook = HierarchicalCodebookEmbedding(
    vocab_size=50261, output_dim=128, num_chunks=2,
    epsilon=1e-6,   # the DEFAULT is Keras' 1e-3 — see Gotchas
)
```

## Module-level helpers (not layers)

`sincos_pos_embed_2d.py` holds three **pure-NumPy** functions — no Keras dependency, no factory key,
no registration — that build the fixed 2-D sin-cos positional table every DiT descendant freezes
into its patch stream:

| Function | Returns |
|---|---|
| `get_1d_sincos_pos_embed_from_grid(embed_dim, pos)` | `(M, embed_dim)` float64, `[sin \| cos]` |
| `get_2d_sincos_pos_embed_from_grid(embed_dim, grid)` | `(H*W, embed_dim)`; `grid[0]` (the **column** index) fills the first half, `grid[1]` (the row index) the second |
| `get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0)` | `(grid_size**2, embed_dim)`, optionally with `extra_tokens` leading zero rows |

Install the result as a **non-trainable weight**:

```python
self.pos_embed = self.add_weight(
    name='pos_embed', shape=table.shape, trainable=False,
    initializer=keras.initializers.Constant(table),
)
```

Never as a plain tensor attribute (it does not survive a `.keras` round trip), and never
`add_weight(...)` + `.assign(...)` inside `build()` — `StatelessScope` discards the assign and the
table stays all zeros in every real model, with no shape symptom.

The grid is built `w`-first (`np.meshgrid(grid_w, grid_h)`), which is upstream's own annotation.
Swapping the two meshgrid arguments, or the two output halves, is a pure permutation on a square
grid: shape, dtype and every norm are identical, so only an elementwise check against an
independently computed destination index sees it.

## Construction

```python
from dl_techniques.layers.embedding import (
    create_embedding_layer, create_embedding_from_config, validate_embedding_config,
)

patch_embed = create_embedding_layer('patch_2d', patch_size=16, embed_dim=768, name='vit_patch')
rope = create_embedding_layer('rope', head_dim=64, max_seq_len=4096, rope_percentage=0.5)

bert_embed = create_embedding_from_config({
    'type': 'bert_embeddings',
    'vocab_size': 30522,
    'hidden_size': 768,
    'max_position_embeddings': 512,
    'type_vocab_size': 2,
    'dropout_rate': 0.1,
    'normalization_type': 'layer_norm',
    'name': 'bert_embeddings',
})

validate_embedding_config(          # pre-flight check; builds nothing
    'bert_embeddings', vocab_size=30522, hidden_size=768,
    max_position_embeddings=512, type_vocab_size=2,
)
```

### RoPE in an attention block

`RotaryPositionEmbedding` takes a **post-head-split 4D** tensor `(batch, heads, seq_len, head_dim)`.
A `Dense` projection gives 3D `(batch, seq_len, heads * head_dim)`, so the reshape and transpose are
not optional — calling the layer on the raw 3D projection raises
`ValueError: Expected 4D input (batch, heads, seq_len, head_dim), got shape with 3 dimensions`.

```python
import keras
from dl_techniques.layers.embedding import create_embedding_layer

num_heads, head_dim = 4, 64
qkv = keras.layers.Dense(num_heads * head_dim * 3)
rope = create_embedding_layer('rope', head_dim=head_dim, max_seq_len=512)

def split_heads(x):
    b, length = keras.ops.shape(x)[0], keras.ops.shape(x)[1]
    x = keras.ops.reshape(x, [b, length, num_heads, head_dim])
    return keras.ops.transpose(x, [0, 2, 1, 3])   # (B, H, L, Dh)

x = keras.random.normal((2, 16, 128))
q, k, v = keras.ops.split(qkv(x), 3, axis=-1)
q, k = rope(split_heads(q)), rope(split_heads(k))   # RoPE BEFORE attention
```

## RoPE pairing conventions — read this before loading foreign weights

Four rotary layers ship here and they do **not** all pair channels the same way. A RoPE rotation
mixes exactly two channels, and there are two conventions for choosing the partner:

- **Interleaved** — channel `0` rotates with `1`, `2` with `3`, ... Su et al.'s original RoFormer
  form, GPT-J's `rotate_every_two`, and Meta's official LLaMA release.
- **Split-half** — channel `j` rotates with `j + head_dim/2`. GPT-NeoX, HF `LlamaModel`, HF Gemma,
  HF Qwen — anything built on HF's `rotate_half`.

**Both are valid rotations and both give the relative-position property, so the wrong one trains
fine.** The convention is invisible in a config, invisible in a shape and silent at load time; a
checkpoint from the other convention produces plausible, wrong numbers. HF `transformers` issue
#25199 exists for exactly this: HF's LLaMA is split-half while Meta's is interleaved, and the same
weights serve both only because the conversion script **permutes the `q_proj`/`k_proj` rows**
(`convert_llama_weights_to_hf.py::permute`). That permutation is what you need to move a checkpoint
across the table below.

| Layer | Convention | Can consume `q_proj`/`k_proj` from | Reference |
|---|---|---|---|
| `RotaryPositionEmbedding` | **Interleaved** (0 <-> 1) | RoFormer, GPT-J, Meta's official LLaMA | Su et al., RoFormer, arXiv:2104.09864 |
| `AxialRoPE2D` | **Interleaved** (0 <-> 1) | SAM 2 memory-attention RoPE (adjacent-pair packing); same set as above | arXiv:2104.09864; follows SAM 2, deliberately against the more common split-half-per-axis vision form |
| `Ideogram4MRoPE` | **Split-half** (0 <-> `head_dim/2`) | Qwen2-VL / Qwen2.5-VL M-RoPE, GPT-NeoX, HF LLaMA/Gemma | arXiv:2104.09864; Wang et al., Qwen2-VL, arXiv:2409.12191 |
| `DualRotaryPositionEmbedding` | **Split-half** (0 <-> `head_dim/2`) | the same HF `rotate_half` lineage | arXiv:2104.09864; Gemma 3 Technical Report (dual global/local theta) |

Two things that are **not** the pairing convention: `AxialRoPE2D`'s split of the head dimension into
an x-half and a y-half, and `Ideogram4MRoPE`'s assignment of frequency slots to `(t, h, w)` via
`mrope_section`. Both are axis assignments. Only the intra-pair rotation form must match a checkpoint.

## Gotchas

- **An undeclared keyword raises.** The message carries `STRICT_DROPPED_KEY_MARKER`; match guards on
  that constant, not on prose. `create_embedding_layer('patch_2d', patch_size=4, embed_dim=32,
  dtype='float64')` raises — `dtype` is not a declared parameter of that entry.
- **`positional_learned` checks its ceiling in `build()`.** A statically known sequence length above
  `max_seq_len` raises `ValueError` there. A dynamic (`None`) sequence axis carries no length and is
  unaffected. `build_from_config()` — the path Keras replays on load — warns and builds anyway, so
  an old archive still loads; the call-time ceiling has not moved.
- **`positional_sine_2d` requires an even `num_pos_feats`.** `__init__` raises on an odd value.
  `from_config()` rounds an odd stored value up and warns, which **changes the output width** from
  `2*n` to `2*(n+1)` channels — such a model could never have run before, since `ops.stack` raised.
- **`continuous_rope` / `continuous_sincos` have no `assert_positive`.** The parameter was removed
  from both classes and their `optional_params`; passing it now raises. It never triggered a check.
  It was not reinstated because `keras.ops` (Keras 3.8) has no assert op, and the TF-only
  `tf.debugging.assert_non_negative` is silently dropped under XLA. `from_config()` pops a stale
  key and warns, so old archives still load; no numerics change.
- **`HierarchicalCodebookEmbedding`'s `epsilon` defaults to `1e-3`** (Keras' own LayerNorm default,
  which is what the layer used before the knob existed). Every other normalization site in this repo
  uses `1e-6`. Pass `epsilon=1e-6` explicitly for a new model.
- **`positional_sine_2d` output is channels-first** `(B, 2*num_pos_feats, H, W)`.
- **`timestep` and `scalar_sinusoidal` are NOT interchangeable.** `TimestepEmbedding` (DiT/GLIDE)
  divides the frequency ladder by `half` and emits `concat([cos, sin])`;
  `ScalarSinusoidalEmbedding` (Ideogram4) divides by `half - 1`, emits `concat([sin, cos])`, and
  rescales its input onto `[0, 1e4]` first. Each divergence is a silent numeric change with no
  shape symptom. `TimestepEmbedding` also decouples `frequency_embedding_size` from `hidden_size`,
  which `ScalarSinusoidalEmbedding`'s single `dim` does not.
- **`TimestepEmbedding` is cos-first, `get_1d_sincos_pos_embed_from_grid` is sin-first.** That is
  not an inconsistency to fix: GLIDE and MAE specify the two bases independently.
- **BERT embeddings normalization and position types** come from `VALID_NORMALIZATION_TYPES`
  (`layer_norm`, `rms_norm`, `band_rms`, `batch_norm`) and `VALID_POSITION_EMBEDDING_TYPES`
  (`learned`, `sinusoidal`) in `bert_embeddings.py`, which `factory.py` imports. That is their one
  home; do not restate them as literals.
- **`BertEmbeddings` never propagates a Keras mask.** It declares `supports_masking = False` and the
  inner `Embedding`'s mask dies at the `word + position` sum, so `mask_zero` is numerically inert at
  this layer's output and only controls whether the inner `Embedding` computes a mask at all.
