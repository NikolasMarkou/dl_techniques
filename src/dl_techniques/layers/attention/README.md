# `dl_techniques.layers.attention`

Thirty-four attention and token-mixing layers behind one factory. `create_attention_layer(attention_type, name=None, **kwargs)`
looks the type up in `ATTENTION_REGISTRY`, rejects any keyword the target class does not declare,
fills in the registry defaults and constructs. Nothing on that path filters and drops: an undeclared
keyword is a `ValueError`, never a discarded argument.

- Factory contract and registry sizes: `src/dl_techniques/layers/CLAUDE.md`.
- Deeper design notes, including the shared-primitive contracts: `GUIDE.md` in this directory.

The factory is **construction-only**. It standardizes how layers are built, not how they are called;
see [Call-signature caveats](#call-signature-caveats).

## Catalogue

Thirty-four keys. `list_attention_types()` returns them sorted; `get_attention_requirements(key)`
returns one entry's metadata (a deep copy, except the `class` value).

| Key | Class | What it is | Pick it when | Required params |
|---|---|---|---|---|
| `anchor` | `AnchorAttention` | Hierarchical attention through anchor tokens; queries cross-attend only to anchors. | Long sequences where full self-attention is too costly. | `dim`, `num_heads` |
| `area` | `AreaAttention` | Splits a flattened `H*W` map into `area` contiguous groups and attends inside each; `area=1` is global. Depthwise 5x5 positional branch, own 1x1 output projection. Falls back to global when `H*W % area != 0`. | YOLOv12-style detection backbones needing feature-map self-attention below quadratic pixel cost. | `dim` |
| `beit` | `BeitAttention` | Learnable relative-position bias over a `(Wh, Ww)` patch grid plus three cls slots, added pre-softmax; asymmetric QKV bias (K has **no** bias). | BEiT-style ViTs; any ViT wanting a 2D relative bias instead of absolute positions. | `dim`, `num_heads`, `window_size` |
| `capsule_routing` | `CapsuleRoutingSelfAttention` | Self-attention with capsule dynamic routing. | Experimental contextualization work. | `num_heads` |
| `cbam` | `CBAM` | Convolutional Block Attention Module (channel + spatial). | Drop-in refinement for any CNN. | `channels` |
| `channel` | `ChannelAttention` | The channel half of CBAM. | Channel-wise recalibration in CNNs. | `channels` |
| `differential` | `DifferentialMultiHeadAttention` | Two attention maps subtracted to cancel common-mode noise. | Transformers wanting sharper focus / less attention noise. | `dim`, `num_heads`, `head_dim` |
| `energy` | `EnergyAttention` | No value matrix; `call()` returns the closed-form negative gradient of a scalar token-mixing energy. | Energy Transformer blocks whose residual stream is a Lyapunov descent. | `dim` |
| `fnet` | `FNetFourierTransform` | Parameter-free Fourier token mixing. | Cheap replacement for self-attention. | (none) |
| `gated` | `GatedAttention` | QK-norm, partial RoPE and an output gate. | Stability-sensitive high-performance transformers. | `dim`, `num_heads` |
| `group_query` | `GroupedQueryAttention` | GQA: fewer K/V heads than Q heads. | LLMs where the KV cache is the bottleneck. | `dim`, `num_heads`, `num_kv_heads` |
| `hopfield` | `HopfieldAttention` | Modern Hopfield network retrieval; `update_steps_max=0` mimics standard attention. | Associative-memory tasks. | `key_dim`, `num_heads` |
| `lighthouse` | `LighthouseAttention` | Coarse-to-fine pyramid plus top-K causal SDPA, scattered back with `segment_sum`. | Long-context causal LM wanting exact attention sub-quadratically. | `dim`, `num_heads` |
| `linear` | `LinearAttention` | Bias-free, degree-1-homogeneous O(N) attention via a positive feature map and associativity. | Bias-free denoiser stacks; long non-causal sequences. | `dim` |
| `mobile_mqa` | `MobileMQA` | Multi-query attention for vision on mobile/edge. | Edge vision models. | `dim` |
| `multi_head` | `MultiHeadAttention` | Standard MHSA. | The default. | `dim` |
| `multi_head_cross` | `MultiHeadCrossAttention` | Self- or cross-attention with pluggable attention probabilities. | Encoder-decoders and custom attention. | `dim` |
| `multi_head_latent` | `MultiHeadLatentAttention` | DeepSeek-V2 MLA with KV compression into a latent. | LLM inference with a small KV cache. | `dim`, `num_heads`, `kv_latent_dim` |
| `non_local` | `NonLocalAttention` | Non-local block over spatial positions. | Global context inside a CNN. | `attention_channels` |
| `perceiver` | `PerceiverAttention` | Perceiver cross-attention into a latent array. | Cross-modal / latent-bottleneck models. | `dim` |
| `performer` | `PerformerAttention` | Random-feature approximation of softmax attention, linear cost. | Very long sequences. | `dim` |
| `ring` | `RingAttention` | Exact attention via blockwise online softmax. | Near-unbounded context with exact attention. | `dim` |
| `rpc` | `RPCAttention` | Principal Component Pursuit decomposition of the attention matrix. | Robustness to noise / adversarial input. | `dim` |
| `shared_weights_cross` | `SharedWeightsCrossAttention` | Cross-attention between modalities with tied weights. | Multi-modal exchange on one concatenated sequence. | `dim` |
| `single_window` | `SingleWindowAttention` | MHA over the whole sequence as one window, optional relative bias. | Windowed attention without grid partitioning. | `dim`, `num_heads`, `window_size` |
| `spatial` | `SpatialAttention` | The spatial half of CBAM. | Highlighting spatial regions in a CNN. | (none) |
| `tripse1` | `TripSE1` | Triplet attention, post-fusion squeeze-and-excitation. | 3D (spatial + channel) vision attention. | (none) |
| `tripse2` | `TripSE2` | Triplet attention, pre-process SE. | Channel recalibration before spatial rotation. | (none) |
| `tripse3` | `TripSE3` | Triplet attention, parallel SE. | Independent spatial and channel modelling. | (none) |
| `tripse4` | `TripSE4` | Hybrid 3D attention with affine fusion of logits. | Deep spatial/channel integration. | (none) |
| `wave_field` | `WaveFieldAttention` | FFT token mixing with a learned wave-field coupling kernel. | Frequency-domain mixing on long sequences. | `dim` |
| `window` | `create_grid_window_attention` -> `WindowAttention` | Swin grid windows. Folds a 1-D sequence into a `ceil(sqrt(N))` grid. `O(N*M)` for `N > M = window_size**2`, `O(N^2)` otherwise. | Vision transformers with local windows. | `dim`, `num_heads`, `window_size` |
| `window_zigzag` | `create_zigzag_window_attention` -> `WindowAttention` | Zigzag partitioning, grouping frequency-proximate tokens. | Vision models where frequency locality matters. | `dim`, `num_heads`, `window_size` |
| `window_band` | `create_band_window_attention` -> `WindowAttention` | Symmetric 1-D band: query `i` sees key `j` iff `abs(i-j) <= window_size`, a **half-width in tokens**. Cost is `O(N^2)`, same order as `multi_head`; what it buys is the right adjacency for a sequence, not a lower asymptotic. | Text encoders with local layers (ModernBERT / Longformer): pass `local_attention // 2`. | `dim`, `num_heads`, `window_size` |

Everything not listed as required is optional and has a registry default. Read them with
`get_attention_requirements(key)['optional_params']` rather than trusting a copy in prose.

### Window-family facts

- The three `window*` keys map to the **wrapper functions** `create_grid_window_attention`,
  `create_zigzag_window_attention` and `create_band_window_attention` (in `window_attention.py`),
  each of which fixes a partitioning mode and returns a `WindowAttention` instance. The
  `WindowAttention` class has no key of its own; import it directly for arbitrary configurations.
- `window_band` rejects `use_relative_position_bias=True`: the bias indexes a 2-D tile this
  layout does not have.
- `create_kan_key_window_attention` and `create_adaptive_softmax_window_attention` are test-only.
  They are deliberately not registered and not exported.

## Construction

```python
from dl_techniques.layers.attention import create_attention_layer

mha = create_attention_layer('multi_head', dim=256, num_heads=8)
cbam = create_attention_layer('cbam', channels=128, ratio=16)
gqa = create_attention_layer('group_query', dim=1024, num_heads=16, num_kv_heads=4, name='gqa_1')
```

From a config dict (the `type` key selects the entry, the rest are kwargs):

```python
from dl_techniques.layers.attention import create_attention_from_config

layer = create_attention_from_config(
    {'type': 'group_query', 'dim': 1024, 'num_heads': 16, 'num_kv_heads': 4, 'name': 'gqa_1'}
)
```

Discovery and pre-flight validation:

```python
from dl_techniques.layers.attention import (
    get_attention_info, validate_attention_config,
)

info = get_attention_info()['group_query']
print(info['required_params'], sorted(info['optional_params']))

validate_attention_config('window', dim=96, window_size=7, num_heads=4)  # raises on a bad config
```

`validate_attention_config` refuses exactly what `create_attention_layer` refuses, including an
undeclared keyword (a typo like `num_head=4`), so it is a real pre-flight check for callers that go
on to build the class directly.

Direct instantiation is always available and bypasses factory validation and defaults:

```python
from dl_techniques.layers.attention import MultiHeadAttention, CBAM, WindowAttention

mha = MultiHeadAttention(dim=512, num_heads=8)
cbam = CBAM(channels=256, ratio=16)
win = WindowAttention(dim=96, window_size=7, num_heads=4)
```

## Customization hooks

Most softmax-based layers expose two hooks; the defaults preserve standard behaviour.

- `probability_type` / `probability_config` — the score normalization, via `ProbabilityOutput`
  (`dl_techniques.layers.activations.probability_output`): `softmax` (default), `sparsemax`,
  `threshmax`, `adaptive`. Routing/hierarchical modes are rejected — they consume features, not logits.
- `qk_norm_type` / `qk_norm_kwargs` — optional Q/K normalization through
  `create_normalization_layer` (`dl_techniques.layers.norms.factory`): `rms_norm`, `layer_norm`,
  `zero_centered_rms_norm`, ... or `None`.

```python
mha = create_attention_layer(
    'multi_head', dim=256, num_heads=8,
    probability_type='sparsemax',
    qk_norm_type='rms_norm',
)
```

Non-default hook behaviour:

| Layer | Note |
|---|---|
| `gated` | `qk_norm_type` defaults to `'zero_centered_rms_norm'` and cannot be `None`. |
| `multi_head_latent` | `qk_norm_type` defaults to `'rms_norm'`. |
| `hopfield` | `qk_norm_type` defaults to `'layer_norm'`; pass `None` for no pattern normalization. |
| `ring` | Only `qk_norm_type`; the online softmax is tied to exponential normalization. |
| `non_local` | Also has `output_norm_type` / `output_norm_kwargs` (default `'batch_norm'`). |
| `channel`, `spatial`, `cbam`, `tripse*`, `fnet`, `performer`, `wave_field` | No hooks — they do not softmax over `Q@K^T`. |

## Call-signature caveats

Most layers are `call(inputs, attention_mask=None, training=None)`. **Twelve deviate**, for
architectural reasons; they are documented, not "fixed", because renaming would break serialized
configs and call sites.

| Layer | Deviation | Why |
|---|---|---|
| `rpc` | mask is `mask=`, not `attention_mask=` | Parameter name predates the convention. |
| `shared_weights_cross` | positional `split_sizes` required | Needs per-modality segment boundaries. |
| `anchor` | no mask argument | The anchor/local pattern is defined internally. |
| `performer` | no mask argument | The FAVOR+ kernel takes no dense additive mask. |
| `lighthouse` | no mask argument; static seq-len required | Causality comes from the pyramid shift; dynamic/`None` seq-len raises `RuntimeError`. |
| `group_query` | `call(inputs, training=None, attention_mask=None)` | `training` precedes `attention_mask`. |
| `ring` | `call(inputs, training=None, attention_mask=None)` | Same order swap. |
| `mobile_mqa` | order swapped, plus `return_attention_weights` | Same order swap, extra flag. |
| `differential` | `call(inputs, attention_mask=None, layer_idx=0, training=None)` | Extra positional `layer_idx`. |
| `spatial` | 4D input; `attention_mask` accepted but **ignored** | CBAM spatial attention runs over the whole feature map; there is no token mask to apply. |
| `area` | 4D input; `attention_mask` is a **spatial keep mask** `(B, H, W)` or `(B, H*W)` | Attends over flattened spatial positions; `1 = keep`, forwarded verbatim. |
| `non_local` | 4D input; mask argument **ignored** | Dense global spatial affinities; a token mask does not apply. |

Pass `attention_mask` as a keyword to `group_query` / `ring` / `mobile_mqa`, and pass `training`
as a keyword to `differential` — positionally it would bind to `layer_idx`.

## Gotchas

- `fnet`: `implementation='fft'` is **not implemented**. It warns once and falls back to the matrix
  DFT, whose cost is `O(S^2*D + S*D^2)`, not `O(N log N)`. Output is identical to `'matrix'`.
- `performer`: `ortho_scaling` is a scalar multiply on the random features, not FAVOR+
  orthogonalization.
- `rpc`: `lambda_sparse` is a regularization weight (`> 0`), not a 0-1 rate.
- `mobile_mqa`: accepts `attention_mask` for signature compatibility and **ignores it** — optional
  K/V downsampling changes the key length, so a token mask cannot be applied unambiguously.
- `linear`: non-causal. `call` takes only an ignored `mask=` kwarg. Keep `feature_map` positively
  homogeneous (`relu`, `relu_squared`, `abs`); `elu_plus_one` / `exp` / `softmax` are rejected
  because they break degree-1 homogeneity and therefore the bias-free denoiser identity.
- `energy`: the output is an **update to add to the residual stream**, not a contextualized value,
  so it is not a drop-in for `multi_head`. Cost is about 2x standard attention. A rank-2 `(B, N)`
  mask is a per-token validity mask applied to key **and** query axes; `(B, N, N)` and
  `(B, H, N, N)` keep the house `(key, query)` semantics. With `attn_self=False` (the paper's
  ET-Full config) a single-token input gives exactly zero energy and zero update.
  ([arXiv:2302.07253](https://arxiv.org/abs/2302.07253))
- `window` folds a 1-D sequence into a square grid, so for text you want `window_band`.

## Shared primitives (`common.py`)

An implementation detail, not re-exported from `__init__.py`; layers do `from .common import ...`.
Five module-level names, no classes: `MASK_BIAS_VALUE`, `mask_dtype(compute_dtype)`,
`apply_attention_mask(logits, keep, *, out_dtype=None, rescue_axis=-1)`,
`validate_head_divisibility(dim, num_heads, ...)` and `compute_attention_scale(head_dim)`.

`apply_attention_mask` is the prescribed way to apply a mask: it applies the bias with `ops.where`
inside `mask_dtype(...)`, so the `0 * -inf = NaN` product is never formed. Two things it will not
decide for you:

- **Polarity is per call site.** `keep` is the keep predicate and no polarity is inferred. Pass your
  site's own spelling verbatim; a uniform `mask > 0` rewrite silently inverts masking at some sites.
- **`rescue_axis` is the axis YOUR softmax reduces over.** Default `-1` enables the degenerate-row
  rescue; `None` opts out. `ring_attention` must opt out per tile — a per-tile rescue would un-mask
  the future under a causal mask while every finiteness test still passed.

`GUIDE.md` section 3.5 documents the contract and the limits of each entry, including why adopting
`compute_attention_scale` at an existing site needs a bit-identity probe (`x ** -0.5` is not
bit-identical to it). Read it before using these at a new call site.
