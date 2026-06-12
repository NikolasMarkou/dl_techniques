# SD3 MMDiT Port Notes — "What Does NOT Fit / What Must Change"

*Port: MiniDiffusion (PyTorch SD3-style MMDiT text-to-image stack) → `dl_techniques` (Keras 3).*
*Plan: `plan_2026-06-12_dfce0712`. Decisions referenced as D-NNN live in that plan's `decisions.md`.*

---

## 1. Overview

This package (`src/dl_techniques/models/sd3_mmdit/`) is a Keras 3 / TensorFlow 2.18
port of the PyTorch MiniDiffusion SD3-style **MMDiT** (Multimodal Diffusion
Transformer) text-to-image stack: a dual-stream rectified-flow diffusion
transformer over a 16-channel spatial VAE latent, conditioned on CLIP +
OpenCLIP pooled vectors and a T5 token sequence. The port is
**architecture-faithful and train-from-scratch**: it reproduces the SD3 module
graph, forward shapes, and serialization contract, but it carries **NO
pretrained weights**. The PyTorch repo is organized around weight loaders
(`load_dit` / `load_clip` / `load_t5` / `load_vae` reading `.pth`
checkpoints); `dl_techniques` has **no weight-loading convention**, so there is
no equivalent here and every `from_variant(..., pretrained=True)`-style path is
absent by design. The package passes **177 unit tests** (scoped to the new
modules) and the tiny preset **smoke-trains cleanly** (synthetic data, loss
`2.255 → 1.568`, ~-30.5%, zero NaN over the run). Every new class carries
`@keras.saving.register_keras_serializable()`, builds sub-layers explicitly in
`build()`, implements `compute_output_shape()`, and round-trips through `.keras`
at `atol=1e-6`. Logging is via `dl_techniques.utils.logger` (no `print`).

---

## 2. What Was REUSED (drop-in)

Existing, tested `dl_techniques` components reused without modification (the only
shared-file edit in the whole port is the additive FFN-factory `gelu_tanh`
registration in §3):

| PyTorch source component | dl_techniques reuse | Import path | Note |
|---|---|---|---|
| SD3 VAE (ResNet enc/dec, GroupNorm32, attn mid-block, KL reparam) | `AutoEncoder` with `z_channels=16` | `dl_techniques.models.ideogram4.vae.AutoEncoder` | Architecture is structurally equivalent; only the latent-norm convention differs — **D-002** / **D-008** |
| `DiagonalGaussianDistribution` (reparameterization) | `Sampling` | `dl_techniques.layers.sampling.Sampling` | Used internally by the reused `AutoEncoder` |
| rectified-flow velocity MSE objective | `FlowMatchingVelocityLoss` | `dl_techniques.losses.flow_matching_velocity_loss.FlowMatchingVelocityLoss` | Plain MSE; logit-normal weighting kept in the trainer (HARD constraint) |
| timestep sinusoidal embedding | `ScalarSinusoidalEmbedding(dim, input_range=(0, 1000))` | `dl_techniques.layers.embedding.scalar_sinusoidal_embedding.ScalarSinusoidalEmbedding` | SD3 timestep range is `[0, 1000]`, not `[0, 1]` |
| patchify Conv2D projection | `PatchEmbedding2D` | `dl_techniques.layers.embedding.patch_embedding.PatchEmbedding2D` | `kernel=stride=patch_size` (see §4 source-bug note) |
| QK-norm / T5 RMSNorm | `RMSNorm` | `dl_techniques.layers.norms.rms_norm.RMSNorm` | Built on per-head `(..., head_dim)` so scale is `(head_dim,)` |
| logit-normal time-warp formula | reused formula/pattern | `dl_techniques.models.ideogram4.scheduler` (`LogitNormalSchedule`, scipy `ndtri`/`expit`) | Re-expressed in `FlowMatchEulerScheduler`; see §3 |
| GPU setup / seeding / callbacks / config-IO | `setup_gpu`, `set_seeds`, `create_callbacks`, `save_config_json` | `train.common` | Trainer wiring, identical to the ideogram4 precedent |
| (optional) CLIP-guidance loss | `CLIPContrastiveLoss` | `dl_techniques.losses.clip_contrastive_loss.CLIPContrastiveLoss` | Available if CLIP-guided training is added; not used by the smoke path |

---

## 3. What Was ADAPTED / BUILT NEW

Components with no faithful in-repo equivalent. Each maps to a concrete SD3 need
(no speculative abstraction — **D-003** sanctions the greenfield size while still
enforcing earned-abstraction per class):

| PyTorch component | New dl_techniques artifact | Gap that forced it |
|---|---|---|
| `PagedJointAttention` | **NEW** `MMDiTJointAttention` — `dl_techniques.layers.attention.mmdit_joint_attention` | No dual-stream concat→attend→split joint attention existed; the source's PAGING / KV-cache deque was **dropped** — **D-004** |
| `AdaLayerNormZero` / `AdaLayerNormZeroX` / `AdaLayerNormContinuous` | **NEW** trio `AdaLayerNormZero` (6-way), `AdaLayerNormZeroX` (9-way), `AdaLayerNormContinuous` (2-way) — `dl_techniques.layers.transformers.sd3_adaln` | Repo only had the 6-way `AdaLNZeroConditionalBlock`; the 9-way (dual-attention gates) and 2-way (scale+shift, no gate) variants were absent |
| GELU-tanh `FeedForward` | **NEW** `GELUMLPFFN`, registered as `gelu_tanh` in the FFN factory — `dl_techniques.layers.ffn.gelu_mlp_ffn.GELUMLPFFN` (`factory.py` key `gelu_tanh`) | Repo had exact-erf `GeGLUFFN` only; SD3 uses approximate (tanh) GELU. Only this layer is factory-registered (clean single-tensor call) — **D-001** |
| `DiTBlock` | **NEW** `MMDiTBlock` + `MMDiTFinalLayer` — `dl_techniques.models.sd3_mmdit.blocks` | Dual-stream block container; the dual-attention `attn2` is a plain `keras.layers.MultiHeadAttention`, which **omits attn2 per-head QK-norm** — **D-005** |
| `DiT` (MMDiT) | **NEW** `SD3MMDiT` + `create_sd3_mmdit(variant, **overrides)` — `dl_techniques.models.sd3_mmdit.transformer` | 2D sin-cos positional embedding implemented as a **static non-trainable weight, crop-centered** at call time, avoiding the error-prone `PositionEmbeddingSine2D` NCHW→NHWC reshape — **D-006** |
| `NoiseScheduler` | **NEW** `FlowMatchEulerScheduler` — `dl_techniques.models.sd3_mmdit.scheduler` | `add_noise` / `euler_step` / logit-normal time sampling + weight; `t=0` data, `t=1` noise, **signed** `dt` — **D-007** |
| `CLIP` / `OpenCLIP` / `T5` text encoders | **NEW** `CLIPTextEncoder`, `OpenCLIPTextEncoder`, `T5Encoder` — `dl_techniques.models.sd3_mmdit.text_encoders` | Repo `models/clip` is SwiGLU/GQA (not OpenAI-CLIP-faithful); no OpenCLIP; no T5. Built from scratch (token-id input, no weights) — **D-009** |
| `HandlePrompt` + inference loop | **NEW** `SD3Pipeline`, `assemble_prompt_features`, `create_sd3_pipeline` — `dl_techniques.models.sd3_mmdit.pipeline` | End-to-end text→latent→Euler-denoise→VAE-decode surface |
| `train.py` | **NEW** `train_sd3_mmdit.py` (`SD3FlowTrainer`, `TrainingConfig`, `make_synthetic_dataset`, `train`) — `src/train/sd3_mmdit/` | Custom logit-normal-weighted `train_step`; named `train_sd3_mmdit.py` (never `train.py`, which would shadow the `train` package) |

---

## 4. What Does NOT Fit / What Had to CHANGE

This is the substantive report.

### 4.1 No pretrained weights (the single biggest mismatch)
The entire PyTorch MiniDiffusion repo is built around **loading** SD3.5 / CLIP /
OpenCLIP / T5 / VAE `.pth` checkpoints — its `load_*` functions are the spine of
the codebase. `dl_techniques` has **no weight-loading convention** and no
`.pth → keras` converter. Consequence: the text encoders and the VAE here are
**architecture-faithful but UNTRAINED** — their embeddings are *meaningless*
until trained. A real text-to-image run requires either (a) training the whole
stack from scratch on paired data, or (b) building a `.pth`→Keras weight-port
(not in scope). This is the dominant "does not fit": everything else is a
mechanical convention swap; this one removes the repo's reason-for-being
(inference with released weights) and replaces it with from-scratch training.

### 4.2 VAE latent-normalization convention (scalar vs per-channel) — **D-008**
SD3 normalizes the VAE latent with **scalars** (`scaling_factor = 0.13025`,
`shift_factor = 0.0`; diffusers' `StableDiffusion3Pipeline` convention). The
reused ideogram4 `AutoEncoder` ships a `latent_norm.py` with **128-element
per-channel** `LATENT_SHIFT` / `LATENT_SCALE` vectors derived for a
`z_channels=32` *patchified* latent (`128 = 32 × patch_size²`). Those vectors are
both **dimensionally wrong** (128 vs SD3's 16 spatial channels) and
**semantically wrong** (different VAE, different latent layout). The port
therefore does **not** import `latent_norm.py`; `vae.py` exposes scalar
constants `SD3_SCALING_FACTOR` / `SD3_SHIFT_FACTOR` and
`normalize_latent` / `denormalize_latent`. The two conventions are genuinely
different normalization schemes for two different VAEs and cannot be centralized
into one shared constant (recorded in D-008 — this is a legitimate per-port
constant, not a duplication smell).

### 4.3 Paging / KV-cache dropped — **D-004**
The source `PagedJointAttention` carried a `# TODO: REVISIT THE PAGING LOGIC`
deque / KV-cache mechanism. It is an inference micro-optimization irrelevant to
the (correct) training-time joint-attention math, and it is stateful and not
graph-safe. The Keras `MMDiTJointAttention` implements plain stateless joint
scaled-dot-product attention (per-stream Q/K/V → concat along seq → manual SDPA
with `keras.ops` → split back) and **drops the cache entirely**. If paged
inference is ever needed it belongs in the pipeline, not in this layer.

### 4.4 `attn2` QK-norm omitted — **D-005**
The SD3.5-medium dual-attention path's second self-attention (`attn2`) uses
`keras.layers.MultiHeadAttention` rather than a 4th bespoke QK-RMSNorm attention
class. This **omits per-head QK-RMSNorm on the dual path only** (a minor
SD3.5-medium detail). The *primary* joint attention keeps full per-head
QK-RMSNorm via `MMDiTJointAttention`. Exact parity on the dual path is a focused
follow-up, not a correctness blocker for training.

### 4.5 T5 attention scaling deliberately omitted — **D-009**
The `T5Encoder` self-attention is `scores = q @ kᵀ + position_bias` with **no
`1/√head_dim` scaling** — T5 folds the scale into its initializer, and adding it
is the single most common T5-port bug (it silently changes the logit scale and
breaks reference parity). The omission is intentional and anchored. Related D-009
choices: the relative-position-bucket bias is built for the **dynamic** sequence
length `L` (not a fixed `max_seq_len` slice), and the CLIP causal mask uses an
`arange` index comparison rather than `ops.tril` (which routes through a `tf.cond`
that breaks the symbolic `.keras` save/load trace).

### 4.6 Tokenizers NOT ported (out of scope)
SD3's tokenizers — **byte-level BPE** (CLIP/OpenCLIP) and **SentencePiece
Unigram** (T5) — are absent from the repo (the existing `BPETokenizer` is
character-level, not byte-level; there is no SentencePiece anywhere). The three
encoders here therefore consume **integer token-id tensors directly**; the smoke
trainer feeds random ids. A real run needs the tokenizers — documented as a
follow-up (§6).

### 4.7 FID metric NOT ported (out of scope)
There is **no InceptionV3 feature extractor and no Frechet-distance / FID
infrastructure** anywhere in `src/` (the `metrics/` dir has retrieval/CLIP
metrics, not generation-quality metrics). FID must be built separately for
evaluation.

### 4.8 Source bugs in the PyTorch original that the port silently corrects
Substantiated from the findings / reused-layer behavior (not invented):
- **PatchEmbed kernel/stride.** The original `PatchEmbed` used a hardcoded
  `kernel_size=2 / stride=2` while exposing a `patch_size` parameter (the patch
  size and the conv geometry could silently disagree). The reused
  `PatchEmbedding2D` sets `kernel = stride = patch_size` **correctly and by
  construction** (its docstring documents this as the explicit invariant), so the
  port cannot reproduce the original's mismatch.
- **Timestep range.** The timestep embedding is configured with the correct SD3
  range `input_range=(0, 1000)` rather than a `[0, 1]` default; the pipeline
  applies the matching `t × 1000` scaling at the call boundary so the embedding
  and the scheduler agree on time units.

*(Only bugs substantiable from the findings are listed; no additional source-bug
claims are made beyond these.)*

---

## 5. Reuse vs Build Summary

| Category | Count |
|---|---|
| Drop-in reuses (no edit) | **9** (ideogram4 `AutoEncoder`, `Sampling`, `FlowMatchingVelocityLoss`, `ScalarSinusoidalEmbedding`, `PatchEmbedding2D`, `RMSNorm`, logit-normal formula, `train.common`, optional `CLIPContrastiveLoss`) |
| New shared layers (under `layers/`) | **5** (`MMDiTJointAttention`, `AdaLayerNormZero`, `AdaLayerNormZeroX`, `AdaLayerNormContinuous`, `GELUMLPFFN`) + 1 additive FFN-factory registration (`gelu_tanh`) |
| New package files (`models/sd3_mmdit/`) | **8** source (`__init__`, `config`, `blocks`, `transformer`, `scheduler`, `vae`, `text_encoders`, `pipeline`) + this `PORT_NOTES.md` |
| New text encoders | **3** (`CLIPTextEncoder`, `OpenCLIPTextEncoder`, `T5Encoder`) |
| New trainer | **1** (`SD3FlowTrainer` in `train/sd3_mmdit/train_sd3_mmdit.py`) |
| Approx. source lines added | **~4,700** across the new layer/model/train files (greenfield, sanctioned by D-003) |
| Unit tests | **177 pass** (scoped to the new modules; not the 1.5 h full suite) |
| Smoke result | tiny preset, synthetic data: loss **2.255 → 1.568** (~-30.5%), **0 NaN**; `SD3FlowTrainer` `.keras` round-trips |

---

## 6. Follow-ups / Not-Yet-Done

1. **Tokenizers.** Byte-level BPE (CLIP/OpenCLIP) and SentencePiece Unigram (T5)
   — required for any real text input; encoders currently take token-id tensors.
2. **FID / generation metrics.** Build an InceptionV3 feature extractor +
   Frechet-distance computation for eval (none exists in the repo).
3. **Pretrained-weight conversion.** A `.pth`→Keras weight port (SD3.5 / CLIP /
   OpenCLIP / T5 / VAE) if released-weight inference is ever desired.
4. **Full-preset training.** Only the **tiny** preset was smoke-trained (pinned
   to GPU1, 12 GB). The **full** preset (`dim=1536, heads=24, depth=24`) is
   *defined but untrained* and will need multi-GPU.
5. **Real-data pipeline.** A non-synthetic dataset path (e.g. Fashion-MNIST or a
   paired image-text set) to replace the random-feature smoke loop.
