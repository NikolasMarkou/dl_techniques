# `dl_techniques.layers.transformers`

Assembled transformer blocks and complete modality stacks. Each block composes the `attention/`,
`ffn/` and `norms/` factories, so an architecture is mostly a choice of type strings.

**This package deliberately has no factory of its own.** There is no `create_transformer_block(type)`
and no registry: blocks are imported by class from `dl_techniques.layers.transformers` (or from their
own module) and constructed directly. The type strings you pass *into* them are the attention / FFN /
normalization registry keys, which are documented in the sibling packages and in
`src/dl_techniques/layers/CLAUDE.md`.

## What is in here

| Class | What it is | Pick it when |
|---|---|---|
| `TransformerLayer` | The workhorse: self-attention + FFN, each with a residual and factory-configurable normalization, pre- or post-norm. Optional stochastic depth, LayerScale and MoE. | Any encoder-style block. Start here. |
| `TransformerDecoderLayer` | Causal self-attention + cross-attention to encoder memory + FFN. Same lifecycle and serialization contract as `TransformerLayer`. | Encoder-decoder stacks. |
| `VisionEncoder` | Complete ViT encoder: patch embedding, optional `[CLS]`, a stack of `TransformerLayer`s, pooled output. | ViT-style image backbones. |
| `TextEncoder` | Complete BERT-style bidirectional encoder: token/positional embeddings plus the stack. | NLU encoders. |
| `TextDecoder` | Complete GPT-style causal decoder. | Autoregressive LM stacks. |
| `SwinTransformerBlock` | Windowed (W-MSA) and shifted-window (SW-MSA) attention on a 4D `(B, H, W, C)` map. | Swin backbones, dense prediction. |
| `SwinConvBlock` | Parallel Swin path + conv path, split-transform-merge. **Input channels must equal `conv_dim + trans_dim`.** | Hybrid local/global vision blocks. |
| `PerceiverTransformerLayer` | Asymmetric cross-attention: a small latent array queries a large byte array, `O(M*N)`. | Very large inputs behind a latent bottleneck. |
| `EomtTransformer` | Encoder-only mask transformer: joint patch + object-query sequence with optional ground-truth masked self-attention. | Instance segmentation. |
| `AdaLNZeroConditionalBlock` | DiT-style adaptive LayerNorm-zero conditional block. | Diffusion transformers and other conditioned stacks. |
| `AreaAttentionBlock` | The YOLOv12 attention stage over a 4D feature map: `AreaAttention` + a 1x1-conv MLP, each in a plain residual. | YOLOv12-style detection backbones. |
| `GatedLinearAttentionBlock` | Recurrent sequence mixing with one `(head_dim, head_dim)` state per head; linear, not quadratic, in sequence length. | Long-sequence causal mixing (Qwen3-Next uses it 3x per block). |
| `PFTBlock` | Progressive Focused Transformer block (PFT-SR, CVPR 2025). | Single-image super-resolution. |
| `FreeTransformerLayer` / `BinaryMapper` | FREE latent-variable transformer: a causal block with an optional encoder path inferring a discrete latent, and the straight-through binary latent mapper. | Research on latent-variable transformers only — see the gotcha below. |
| `EnergyTransformer` / `HopfieldNetwork` | Energy Transformer block (Hoover et al., NeurIPS 2023): `T` steps of gradient descent on one scalar energy, and its associative-memory sub-layer. | Energy-based / Lyapunov-descent residual streams. |

Six more block classes live here but are **not** re-exported from the package `__init__`; import them
from their module: `Gemma3TransformerBlock` (`gemma3_transformer.py`), `Ideogram4TransformerBlock`
and `Ideogram4FinalLayer` (`ideogram4_block.py`), and `AdaLayerNormZero`, `AdaLayerNormZeroX`,
`AdaLayerNormContinuous` (`sd3_adaln.py`, the SD3 modulation layers).

Convenience constructors, all thin presets over the two encoder classes: `create_vision_encoder`,
`create_vit_encoder`, `create_siglip_encoder`, `create_text_encoder`, `create_bert_encoder`
(vocab 30522), `create_roberta_encoder` (50265), `create_modern_encoder` (1024-wide, depth 24),
`create_efficient_encoder` (512-wide, depth 8).

## TransformerLayer

```python
import keras
from dl_techniques.layers.transformers import TransformerLayer

inputs = keras.Input(shape=(128, 512))
block = TransformerLayer(
    hidden_size=512,
    num_heads=8,
    intermediate_size=2048,
    normalization_position='pre',
    ffn_type='swiglu',
    use_stochastic_depth=True,
    stochastic_depth_rate=0.1,
)
outputs = block(inputs)
```

All 28 constructor parameters (plus `**kwargs`, forwarded to `keras.layers.Layer`):

| Argument | Type | Description | Default |
|---|---|---|---|
| `hidden_size` | `int` | **Required.** Input/output width. Positive and divisible by `num_heads`. | |
| `num_heads` | `int` | **Required.** Attention heads. | |
| `intermediate_size` | `int` | **Required.** FFN hidden width. Positive unless `moe_config` is set — and still read under `moe_config`, see below. | |
| `attention_type` | `str` | Any `ATTENTION_REGISTRY` key. | `'multi_head'` |
| `attention_args` | `dict?` | Merged last into the attention factory call. Never pre-filtered, so an unknown key is reported by the factory. | `None` |
| `normalization_type` | `str` | Any `create_normalization_layer` key. | `'layer_norm'` |
| `normalization_position` | `str` | `'pre'` or `'post'`. Validated: anything else raises. | `'post'` |
| `attention_norm_args` | `dict?` | Extra kwargs for the attention-side norm only. | `None` |
| `ffn_norm_args` | `dict?` | Extra kwargs for the FFN-side norm only. | `None` |
| `ffn_type` | `str` | Any `FFN_REGISTRY` key. Ignored under `moe_config`. | `'mlp'` |
| `ffn_args` | `dict?` | Merged last into the FFN factory call. Ignored under `moe_config`. | `None` |
| `moe_config` | `MoEConfig \| dict?` | Replaces the FFN sub-layer with a `MixtureOfExperts`. | `None` |
| `dropout_rate` | `float` | FFN-output dropout. The **only** dropout this layer applies itself, and never after attention. | `0.1` |
| `attention_dropout_rate` | `float` | Attention-**internal** (weight) dropout, forwarded as the attention sub-layer's own `dropout_rate`. | `0.1` |
| `use_stochastic_depth` | `bool` | Drop-path on both residual branches. | `False` |
| `stochastic_depth_rate` | `float` | Drop probability. | `0.1` |
| `activation` | `str \| Callable` | FFN activation. | `'gelu'` |
| `use_bias` | `bool` | Bias on the linear sub-layers. | `True` |
| `kernel_initializer` | `str \| Initializer` | Forwarded to the sub-layer factories. | `'glorot_uniform'` |
| `residual_output_kernel_initializer` | `str \| Initializer?` | Optional separate initializer for the residual output projections. | `None` |
| `bias_initializer` | `str \| Initializer` | Forwarded to the sub-layer factories. | `'zeros'` |
| `kernel_regularizer` | `Regularizer?` | Forwarded to the sub-layer factories. | `None` |
| `bias_regularizer` | `Regularizer?` | Forwarded to the sub-layer factories. | `None` |
| `window_size` | `int` | Window size for `attention_type='window'`. | `8` |
| `n_kv_head` | `int?` | KV head count for `attention_type='group_query'`; `None` means `num_heads`. | `None` |
| `lambda_init` | `float` | Initial lambda for `attention_type='differential'`. | `0.8` |
| `use_layer_scale` | `bool` | LayerScale on both residual branches. | `False` |
| `layer_scale_init_value` | `float` | LayerScale initial value. | `1e-5` |

There is **no `norm_args` parameter** on `TransformerLayer` — the two normalization sites are
configured separately. (`TextEncoder` and `VisionEncoder` do have a single `norm_args`.)

### With Mixture of Experts

```python
from dl_techniques.layers.moe import MoEConfig, ExpertConfig, GatingConfig

moe_block = TransformerLayer(
    hidden_size=512,
    num_heads=8,
    intermediate_size=2048,          # NOT ignored — see below
    moe_config=MoEConfig(
        num_experts=8,
        expert_config=ExpertConfig(
            ffn_config={'type': 'swiglu', 'output_dim': 512, 'ffn_expansion_factor': 4}),
        gating_config=GatingConfig(top_k=2),
    ),
)
```

`ffn_type` and `ffn_args` are ignored under `moe_config`, but **`intermediate_size` is not**: it is
the fallback for the expert FFN's `hidden_dim` whenever `moe_config.expert_config.ffn_config` omits
that key **and** the expert type is one of `_MOE_EXPERT_TYPES_USING_INTERMEDIATE_SIZE` in
`transformer.py` (`mlp`, `differential`, `glu`, `geglu`, `residual`, `swin_mlp`). With the `swiglu`
expert above it genuinely goes unused; switch the expert to `mlp` without a `hidden_dim` and the
`2048` becomes load-bearing.

## The complete stacks

```python
import keras
from dl_techniques.layers.transformers import (
    VisionEncoder, TextEncoder, TextDecoder, TransformerDecoderLayer,
)

vit = VisionEncoder(                       # ViT-B/16
    img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
    patch_embed_type='linear', use_cls_token=True, output_mode='cls',
)

bert = TextEncoder(                        # BERT-base shape
    vocab_size=30522, embed_dim=768, depth=12, num_heads=12, max_seq_len=512,
    positional_type='learned', output_mode='none',
)

modern = TextEncoder(                      # SwiGLU + RMSNorm pre-norm encoder
    vocab_size=50000, embed_dim=768, depth=12, num_heads=12,
    positional_type='sincos', ffn_type='swiglu',
    normalization_type='rms_norm', normalization_position='pre',
)

gpt = TextDecoder(                         # GPT-style causal decoder
    vocab_size=50257, embed_dim=768, depth=12, num_heads=12, max_seq_len=1024,
)

dec_block = TransformerDecoderLayer(
    hidden_size=512, num_heads=8, intermediate_size=2048,
    normalization_position='pre', ffn_type='swiglu',
)
# y = dec_block(target_embeddings, encoder_output=memory)
```

`VisionEncoder` takes 27 constructor parameters, `TextEncoder` 34 and `TextDecoder` 18 (plus
`**kwargs`); every
`attention_type`, `ffn_type` and `normalization_type` is a `Literal` of the live registry keys, so
your editor and `inspect.signature` are the authoritative list.

## Specialized and Hybrid Blocks

```python
import keras
from dl_techniques.layers.transformers import (
    SwinTransformerBlock, SwinConvBlock, PerceiverTransformerLayer, EomtTransformer,
    AreaAttentionBlock, GatedLinearAttentionBlock, PFTBlock, EnergyTransformer,
)

swin = SwinTransformerBlock(dim=96, num_heads=3, window_size=7, shift_size=7 // 2)
hybrid = SwinConvBlock(conv_dim=64, trans_dim=64, head_dim=32, window_size=7,
                       block_type='SW', drop_path_rate=0.1)      # needs 128 input channels

perceiver = PerceiverTransformerLayer(dim=512, num_heads=8)
latents = perceiver(query_input=keras.random.normal((2, 256, 512)),
                    kv_input=keras.random.normal((2, 4096, 512)))   # (2, 256, 512)

eomt = EomtTransformer(hidden_size=768, num_heads=12, use_masked_attention=True,
                       mask_probability=0.8, mask_annealing_steps=10000)

area = AreaAttentionBlock(dim=256, num_heads=8, area=4)
gla = GatedLinearAttentionBlock(dim=512, num_heads=8, max_seq_len=2048)
pft = PFTBlock(dim=180, num_heads=6, window_size=16, mlp_ratio=2.0)

et = EnergyTransformer(embed_dim=768, num_heads=12, head_dim=64, hopfield_dim=3072,
                       num_steps=12, step_size=0.1, attn_self=False)
```

### AdaLNZeroConditionalBlock

DiT-style adaptive layer-norm "zero" block (Peebles & Xie 2023, adapted in LeWM). Two inputs per
call: content `x` of shape `(B, T, D)` and conditioning `c` broadcastable to `x`. The conditioning
drives six modulation streams (shift / scale / gate for the attention and FFN sub-blocks) through one
SiLU-Linear projection whose final `Dense` is zero-initialized, so **at initialization the block is
the identity map in `x`**. Norms, attention, FFN and the AdaLN activation are all factory-configurable;
leaving every factory kwarg at its default reproduces the original DiT/LeWM construction bit-exactly.
A `normalization_type` swap must disable affine (`use_scale=False`) — the modulation supplies the
scale.

### EnergyTransformer

Replaces the `attn -> FFN` residual stream with `T` steps of gradient descent on one scalar energy
`E = E_ATT + E_HN` ([arXiv:2302.07253](https://arxiv.org/abs/2302.07253)):

```
for t in 1..T:
    g = EnergyLayerNorm(x)
    x = x + step_size * (attn.update(g) + hopfield.update(g))     # update == -dE/dg
```

`EnergyAttention` (attention key `'energy'`) supplies the token mixing with no value matrix.
`HopfieldNetwork` is a per-token associative memory with a **single tied** `(hopfield_dim, dim)`
matrix used in both directions and no bias — not FFN-shaped, which is why it is deliberately not in
the FFN factory. Because every `update()` is the analytic negative gradient, the energy is provably
non-increasing across steps (for `noise_std=0`, `gamma > 0` and a small enough `step_size`); both
closed forms are checked against an autodiff oracle in the tests. `return_energy=True` makes the
block return `(x, energies)` with `energies` of shape `(batch, num_steps + 1)`. This is the block
only: no patchify, no `MASK` token, no decoder.

## Gotchas

- **`TextEncoder` accepts only `positional_type='learned'` or `'sincos'`.** `'rope'` and
  `'dual_rope'` are in the `Literal` but raise `NotImplementedError`: RoPE is not wired into the
  factory attention layers, so choosing it would silently build a model with no positional
  information at all.
- **`SwinConvBlock` input channels must equal `conv_dim + trans_dim`.** The block splits the input
  between its two paths.
- **`EomtTransformer` with `use_masked_attention=True` needs a maskable `attention_type`.**
  `'fnet'`, `'anchor'` and `'lighthouse'` take no attention mask and that combination raises.
  Call it as `eomt({'inputs': tokens, 'mask': masks}, training=True)`.
- **`PerceiverTransformerLayer` takes `dropout_rate`, and is called with keywords**
  (`query_input=`, `kv_input=`).
- **`AreaAttentionBlock`: `normalization_kwargs=None` gives the norm factory's `epsilon=1e-6`, not
  YOLOv12's.** YOLOv12's `epsilon=1e-3, momentum=0.97` pair lives in exactly one place,
  `dl_techniques.layers.yolo12_blocks.YOLO12_NORM_KWARGS`, and must be threaded in explicitly —
  `layers/transformers/` sits below `yolo12_blocks.py` in the dependency order and cannot import it.
  Omitting it moved the block output by `3.1e-02 .. 8.3e-02` on the relocation grid, 255x-285x the
  float32 reassociation tolerance. The block also deliberately has **no** Pre/Post-Norm, LayerScale
  or StochasticDepth, and its MLP is a `ConvBlock` pair rather than `create_ffn_layer('gated_mlp')`
  (which would drop the intermediate BatchNorm). Adding either is a numerics change, not a tidy-up.
  `area > 1` engages only when `H * W` is divisible by `area`; otherwise the attention falls back to
  global, which YOLOv12 exercises routinely.
- **`GatedLinearAttentionBlock`: `max_seq_len` is advisory.** It shapes no weight and bounds no loop,
  so a longer sequence is still computed exactly; `build()` only warns when a static length exceeds
  it. An earlier revision did cap the scan and returned **52 of 60 timesteps all-zero, silently**.
  Its `ffn_args` / `q_norm_args` / `k_norm_args` / `v_norm_args` **raise** on an unrecognized key —
  the layer pre-filters only its own generic defaults, and the factories are strict. `head_dim=None`
  requires `dim % num_heads == 0`; `v_proj` emits `2 * qk_dim` channels, split per head into a write
  half that enters the state and a read-out half that bypasses it (not a skip connection over the
  block input). `chunk_size` is a pure performance knob; the chunked and sequential paths agree only
  to floating-point reassociation, never bitwise.
- **`FreeTransformerLayer`'s latent width is `num_latent_bits`, not `num_bits`.** Also documented,
  not redesigned: the encoder path's cross-attention does not receive the sequence as separate K/V,
  so the posterior `Q(Z|S)` is unconditional on `S` (in-code note D-002). No production model depends
  on this layer.
- **`PFTBlock` lives in `progressive_focused_transformer.py`** — the module name, not
  `progressive_focused_transformer_block`.
- **`TransformerLayer`'s `dropout_rate` never applies after attention.** If you want dropout on the
  attention weights, that is `attention_dropout_rate`.
