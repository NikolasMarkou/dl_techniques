# Ideogram4 — maintainer notes

Keras 3 port of the Ideogram4 neural core. Plan:
`plans/plan_2026-06-12_59a18a10/` (iter-1). User-facing docs: `README.md`.

This file is the **maintainer-facing** record. Read the
"WHAT DOESN'T FIT / WHAT WAS SKIPPED / WHAT CHANGED" section before touching
this package or comparing it against the upstream PyTorch reference.

---

## WHAT DOESN'T FIT / WHAT WAS SKIPPED / WHAT CHANGED

This is a **faithful-in-math, NOT weight-compatible, NOT end-to-end** port. The
deltas below are deliberate and locked by the plan's scope decisions (D1/D2/D3).

### Conditioning abstracted as an input (D1)

The original conditions the DiT on **13 stacked Qwen3-VL-8B hidden-state taps**
(`QWEN3_VL_ACTIVATION_LAYERS = (0,3,6,9,12,15,18,21,24,27,30,33,35)`, each 4096
wide → `llm_features_dim = 53248`). That 8B PyTorch LLM has **no Keras
equivalent** and cannot run on local hardware. This port takes `llm_features` as
a **precomputed call input** to `Ideogram4Transformer`. A real end-to-end system
would need a **torch-side Qwen3-VL feature extractor bridged to Keras** — this is
documented as an out-of-scope stub and is NOT implemented here.

### Published weights are NOT loadable

`ideogram-ai/ideogram-4-nf4` ships as **nf4/fp8-quantized PyTorch safetensors in
diffusers layout**. This Keras architecture is **faithful in MATH but not
weight-compatible**: there is no torch→Keras nf4-dequant converter, no parameter
name map, and no quantized-load path. The deliverable is a **trainable
architecture** (see the tiny preset + real smoke-train), not a drop-in for the
released checkpoint.

### SKIPPED upstream files (and why)

| upstream file | why skipped |
|---------------|-------------|
| `magic_prompt.py` | OpenRouter / Ideogram **HTTP API** prompt expansion — external service, not a neural net. |
| `safety.py` | Hive moderation **HTTP API** — external service. |
| `quantized_loading.py` | bitsandbytes 4-bit / fp8 **PyTorch Linear** swapping — torch/CUDA-specific; Keras has its own quantization story. |
| `caption_verifier.py` | pure-Python JSON validator — left **UN-ported as an optional util** (no NN content; could be ported trivially if ever needed). |

### Structural deltas from the repo's `AdaLNZeroConditionalBlock`

The Ideogram block is **NOT** the repo's standard DiT AdaLN-zero, so a dedicated
`Ideogram4TransformerBlock` was built (D-002); `AdaLNZeroConditionalBlock` was
**reference only**.

| | repo `AdaLNZeroConditionalBlock` | `Ideogram4TransformerBlock` |
|---|---|---|
| modulation streams | 6 (shift + scale + gate ×2) | **4 (scale + gate ×2, NO shift)** |
| gate | (implicit) | **tanh-gated** |
| scale | shift+scale | **`1 + scale` only** |
| norm placement | pre-norm only | **4-RMSNorm sandwich** (pre-norm + post-norm INSIDE the residual on the sublayer output) |
| activation | SiLU | tanh gates; SiLU only on the AdaLN input projection |

```text
x = x + tanh(gate_msa) * attn_norm2(attn(attn_norm1(x) * (1 + scale_msa)))
x = x + tanh(gate_mlp) * ffn_norm2(ffn(ffn_norm1(x) * (1 + scale_mlp)))
```

The post-norm-inside-the-residual is unusual but replicated exactly.

### Other deltas / decisions

- **GroupNorm**: uses the built-in `keras.layers.GroupNormalization(groups=32,
  eps=1e-6)` directly — the repo norms factory has no GroupNorm. The
  channel/32 invariant is enforced both in `config.py` (`_validate_vae_groupnorm`)
  and defensively in `vae.py` ctors.
- **Time embedding serialization bug avoided**: instead of the existing
  `TimestepEmbedding` (stores sinusoidal `freqs` as a plain tensor → `.keras`
  round-trip breaks), a fresh `ScalarSinusoidalEmbedding` keeps `freqs` in a
  **non-trainable weight** (`add_weight(trainable=False)`).
- **mRoPE static one-hot select (D-003)**: the PyTorch dynamic in-place scatter
  for the t/h/w band interleave became a **static per-slot one-hot selector** +
  `einsum`, computed at `build()`. XLA-safe; verified element-wise at atol 1e-6
  against a NumPy reproduction of the PyTorch forward.
- **Attention additive mask (D-004)**: the PyTorch boolean keep-mask handed to
  `F.scaled_dot_product_attention` became an **additive finite mask**
  (`where(same_segment, 0.0, -1e9)`) added to the scaled scores before softmax.
  Avoids all-`-inf` (NaN) rows and is XLA-safe.
- **VAE `bn` omitted**: the PyTorch `AutoEncoder` carries a `BatchNorm2d` on
  patchified latents (training/latent-norm path); it is **unused at decode** and
  is OMITTED to avoid dead code. The pipeline applies the explicit shift/scale
  latent normalization (`latent_norm.py`) instead.
- **Pipeline single shared transformer**: defaults to ONE transformer for both
  CFG branches; an optional `unconditional_transformer=` arg recovers the
  two-model form.
- **Latent denorm guarded on `in_channels == 128`**: pipeline latent denorm uses
  the 128-element `LATENT_SHIFT`/`LATENT_SCALE`; it is **skipped for the tiny
  preset** (`in_channels == 32`).
- **Trainer image-token slice**: the trainer model returns `velocity[:, T:]`
  (image-token velocities only) so the velocity MSE is computed on image tokens.

### Known cosmetic issue

`setup_gpu` logs `"Physical devices cannot be modified after being
initialized"` because TF initializes the device list **before** the call when the
trainer is launched via `-m`. **Training still runs on the GPU** (confirmed by
the smoke-train). Minor follow-up only — does not affect correctness.

---

## In-code DECISION anchors

All anchors carry the plan-id prefix `plan_2026-06-12_59a18a10`:

| anchor | file:line | summary |
|--------|-----------|---------|
| D-003 | `layers/embedding/multi_axis_rope.py:183` | static one-hot band-interleave (replaces dynamic scatter) |
| D-004 | `layers/attention/ideogram4_attention.py:226` | additive finite block-diagonal segment mask (replaces boolean keep-mask) |
| D-005 | `models/ideogram4/vae.py:317` | thin `Upsample` wrapper (replaces functional `upsample()` builder) |

D-001 (overall scope) and D-002 (net-new `ScalarSinusoidalEmbedding` +
`Ideogram4TransformerBlock`) are design-level decisions; D-002's anchor lives in
`scalar_sinusoidal_embedding.py`. See `decisions.md`.

---

## Maintenance notes

- **House rules**: every custom layer/model uses
  `@keras.saving.register_keras_serializable()`, `keras.ops` (backend-agnostic,
  no raw TF), full `get_config()`/`from_config()` round-trip, centralized
  `dl_techniques.utils.logger` (no `print`). The transformer velocity output is
  **cast to float32** at the head regardless of mixed precision.
- **Config invariant guards** (`config.py` `__post_init__` +
  `get_ideogram4_config`): keep these intact — they prevent silent build-time
  failures downstream:
  - `emb_dim % num_heads == 0` and `head_dim` even (mRoPE needs `head_dim/2`
    freqs);
  - mRoPE band bound (h/w bands fit inside `head_dim/2`, the EXACT check mirrored
    from `Ideogram4MRoPE.__init__`);
  - VAE `ch` and every `ch * m` divisible by 32 (GroupNorm groups=32);
  - `in_channels == z_channels * patch_size**2`;
  - `config.z_channels == ae.z_channels`.
- The mRoPE band-bound check is intentionally **duplicated** between `config.py`
  and `Ideogram4MRoPE.__init__` (config validates ahead of layer build); they
  must stay in lockstep — if you change one, change the other.
- **Testing**: scope pytest to this package's test dirs (listed in `README.md`
  §6). **Never run the full suite** (`make test` / `pytest tests/`) as a routine
  check — it is the ~1.5h pre-push hook.
- **Weights / end-to-end**: do not claim checkpoint-loading or end-to-end
  prompt→image until the two out-of-scope stubs above (Qwen3-VL extractor,
  nf4-dequant converter) are actually built.
