# Audit: Inline Layers in models/ vs Reusable layers/
*Date: 2026-06-13. Scope: classify every inline `keras.layers.Layer` subclass in `src/dl_techniques/models/` against the reusable vocabulary in `src/dl_techniques/layers/`. Report-only — no code changed.*

## Scope & Method
This audit reads every inline `keras.layers.Layer` subclass defined inside `src/dl_techniques/models/` — approximately **94 classes across 53 files / ~40 packages**, per `findings/audit-surface.md`. The `model-import-scan.md` Cluster 6 probe additionally surfaces a `PatchEmbedding` in `sam/image_encoder.py` that is **not** in the 94-line inventory, bringing the working total to **~95**; the exact denominator is reconciled in the Summary (Synthesis step).

This is a **report-only** deliverable: nothing under `src/dl_techniques/models/` or `src/dl_techniques/layers/` is modified. The user reviews the report and decides what (if anything) to act on later.

The reference vocabulary is `src/dl_techniques/layers/`, catalogued in `findings/layers-inventory.md`. It exposes 8 factory registries:
- **attention** — 23 keys
- **ffn** — 15 keys
- **norms** — 16 keys
- **embedding** — 10 keys
- **activations** — 22 keys
- **mixtures** — 3 keys
- **seq-pooling** — 3 keys
- **heads** — 21 keys

plus canonical standalone primitives (`RMSNorm`, `TransformerLayer`, `SwiGLUFFN`, `SqueezeExcitation`, `PatchEmbedding2D`, `StochasticDepth`, `LearnableMultiplier`, `Sampling`, `VectorQuantizer`, geometric `CliffordNetBlock` / `GatedGeometricResidual`, etc.).

## Rubric
Each inline class receives exactly one of three verdicts:

- **REPLACE** — an existing `layers/` class or factory key is functionally equivalent. The row names the exact target (class + import path, or factory key) and flags **drop-in** vs **needs-adapter**.
- **RELOCATE** — reusable, but no `layers/` equivalent exists; it belongs there. The row names the proposed `layers/` home and states why it is reusable.
- **KEEP** — model-specific; stays inline. The row justifies why.

### Source-read rules (anti-false-positive)
Distilled from `findings/sample-classification.md` § Rubric Gotchas. Every REPLACE/RELOCATE claim must satisfy these:

1. **Read BOTH sides' `call` / `build` / `get_config` before any REPLACE or RELOCATE — never name-match.** Check bias, normalization `center`, output format, and loss contract on both the inline class and the candidate.
2. **One-param gaps still count as REPLACE-with-adapter, not KEEP.** A single discriminating parameter (e.g. `use_bias` on a gate `Dense`) is an adapter, not a reason to keep a duplicate class.
3. **Output-contract differences are silent-semantics traps and block a clean REPLACE.** Bare logits vs a dict-with-softmax (e.g. `from_logits=True` vs probabilities) changes loss semantics even when the underlying math is identical.
4. **Same-name-different-math is NOT a replacement.** `_LearnedQueryPool1D` (scaled dot-product single-query) is not `SequencePooling('attention')` (tanh-projected context vector); identical names, fundamentally different operations.
5. **Flag latent bugs found during the audit, but do NOT recommend preserving them.** A relocation/replacement is an opportunity to fix a bug, never to carry it forward.

## Summary
*(filled in Synthesis step)*

### Counts by verdict
*(filled in Synthesis step)*

### REPLACE list
*(filled in Synthesis step)*

### RELOCATE list
*(filled in Synthesis step)*

### KEEP rationale rollup
*(filled in Synthesis step)*

## Appendix: Flagged bugs & anti-patterns
*(filled in Synthesis step)*

## Recommendations (prioritized)
*(filled in Synthesis step)*

## Findings by family

### Batch 0 — bias_free_denoisers + cliffordnet (14 classes)

| Class | file:line | Verdict | Target / proposed home | Drop-in? | Evidence (both sides) | Notes |
|---|---|---|---|---|---|---|
| `ConvUNextStem` | `bias_free_denoisers/bfconvunext.py:59-131` | KEEP | — | — | inline: `Conv2D(use_bias=False)` → `GlobalResponseNormalization` → `gelu` (3-op stem). Candidates `layers/convnext_v2_block.py:60` & `layers/convnext_v1_block.py:54` are DepthwiseConv + MLP-bottleneck residual blocks — different topology. | Reusable across bias-free U-Nets; promote to `BiasFreeConvStem` only if a second caller appears. |
| `DenseConditioningInjection` | `bias_free_denoisers/bfunet_conditional_unified.py:112-243` | RELOCATE | `layers/` (or `layers/ffn/`) as **BiasFreeFeatureModulation** | — | inline: two-input bias-free FiLM (film/multiplication/concatenation; `addition` rejected). Absence: `grep -rn "film\|FiLM\|conditioning.*inject"` in `layers/` → no FiLM/conditioning-injection layer exists. | Bias-free multiplicative-only modulation; cross-cutting for any bias-free U-Net / VAE / diffusion model. |
| `DiscreteConditioningInjection` | `bias_free_denoisers/bfunet_conditional_unified.py:247-375` | RELOCATE | `layers/` as **BiasFreeEmbeddingBroadcast** | — | inline: embedding → spatial broadcast (`spatial_broadcast` / `channel_concat`). Absence: `layers/sequence_pooling/` pools sequences, `layers/heads/` output predictions — neither does mid-network spatial broadcast. | Standard in conditional diffusion / generative U-Nets. **BUG (for appendix):** constructs `RepeatVector`/`Reshape` inline in `call` (new layer objects every forward pass) — anti-pattern, fix on relocate. |
| `_ClassificationHeadBlock` | `cliffordnet/unet.py:83-173` | REPLACE | `layers/heads/vision/factory.py:633` (`ClassificationHead`) | needs-adapter | inline: GAP → `LayerNorm(1e-6)` → opt Dropout → opt `Dense(gelu)` → `Dense(num_classes)`, returns **bare logits**. Candidate applies `softmax` and returns `{'logits','probabilities'}` dict. Same math, different output contract. | Output-contract trap (rule 3): needs softmax stripped + dict unwrapped, or a `use_logits=True` mode on `ClassificationHead`. Not drop-in. |
| `_SpatialHeadBlock` | `cliffordnet/unet.py:177-259` | KEEP | — | — | inline: `LayerNorm(1e-6)` → opt `Conv2D(3, gelu)` → `Conv2D(1)`, raw values. Candidates `DepthEstimationHead`/`SegmentationHead` (`layers/heads/vision/factory.py`) add task scaling / sigmoid / softmax + dict output. | Lightweight 3-op head; "raw values" contract intentionally differs from head factories. |
| `_DetectionHeadBlock` | `cliffordnet/unet.py:275-339` | KEEP | — | — | inline: thin wrapper over `layers/yolo12_heads.py` `YOLOv12DetectionHead` (lazy import + list/tuple validation + unused `use_bias`). | Glue/adapter for the `CliffordNetUNet` head protocol; adds no math. Note: inline-able — callers can use `YOLOv12DetectionHead` directly. |
| `BiasFreeClifordNetBlock` | `cliffordnet/denoiser.py:61-237` | REPLACE | `layers/geometric/clifford_block.py:482` (`CliffordNetBlock`) **+ needs `norm_center` param** | needs-adapter | inline: identical dual-stream arch but `LayerNorm(center=False)`, `BN(center=False)`, `use_bias=False` throughout. Candidate already accepts `use_bias` (`clifford_block.py:575`) default True, `center=True`, configurable norm + bias init/reg. | `CliffordNetBlock` minus `center` parameterization. Add `norm_center: bool` to `CliffordNetBlock` (or promote this class alongside it). |
| `BiasFreeGatedGeometricResidual` | `cliffordnet/denoiser.py:246-333` | REPLACE | `layers/geometric/clifford_block.py:287/346` (`GatedGeometricResidual`) **+ needs `use_bias`** | needs-adapter | inline gate: `Dense(C, use_bias=False)` (`denoiser.py:282`). Candidate gate: `Dense(C, use_bias=True)` (`clifford_block.py:375`); gamma init / drop_path / silu+gate formula bit-for-bit identical. | One-param gap (rule 2). Add `use_bias: bool = True` to `GatedGeometricResidual` → this class becomes redundant. |
| `_LayerScale1D` | `cliffordnet/clip.py:153-194` | REPLACE | `layers/layer_scale.py:97` (`LearnableMultiplier`) | drop-in | inline: `build` adds `gamma (channels,)` init `init_value`; `call: x * gamma`. Candidate `LearnableMultiplier(multiplier_type='CHANNEL', initializer=Constant(1e-5), constraint=None)` does `ops.multiply(inputs, gamma)`, gamma shape `(input_shape[-1],)`. | Drop-in with explicit args (defaults differ: `constraint='non_neg'`→None, `initializer='ones'`→`Constant(1e-5)`). Checkpoint shim needed for existing `Custom>_LayerScale1D` saves. |
| `_LearnedQueryPool1D` | `cliffordnet/clip.py:198-264` | RELOCATE | `layers/sequence_pooling/` as **LearnedQueryPool** | — | inline: learnable `query (channels,)`, scaled dot-product `einsum("bnd,d->bn")/sqrt(C)` + opt mask + softmax → `(B,D)`. Candidate `layers/sequence_pooling/attention_pooling.py` uses `tanh(Wx+b)·u` (Lin et al. 2017) — **different math**. | Same-name-different-math trap (rule 4): NOT `SequencePooling('attention')`. Reusable class-token-free attention pool (CLIP/EVA). Preserve the `mask` arg on relocate. |
| `BiasFreeConditionedGGR` | `cliffordnet/conditional_denoiser.py:78-178` | REPLACE | `layers/geometric/clifford_block.py` (`GatedGeometricResidual`) **+ needs `use_bias`** | needs-adapter | inline: byte-for-byte equivalent to `BiasFreeGatedGeometricResidual` (`denoiser.py:246-333`) — `Dense(C, use_bias=False)` gate, LayerScale gamma, opt drop_path; adds `get_build_config`/`build_from_config`. "conditioned" in name but no conditioning in this class. | Duplicate (same param gap). Once `GatedGeometricResidual` gains `use_bias`, both this and `BiasFreeGatedGeometricResidual` are dead code. |
| `BiasFreeConditionedCliffordBlock` | `cliffordnet/conditional_denoiser.py:187-451` | RELOCATE | `layers/geometric/clifford_block.py` as **ConditionedCliffordNetBlock** | — | inline: `CliffordNetBlock` arch + `use_bias=False`/`center=False` + two optional conditioning arms (`cond_gamma_proj` multiplicative FiLM, `class_proj` additive class-embedding broadcast). Candidate `CliffordNetBlock` (`clifford_block.py:482`) lacks `enable_dense_conditioning`/`enable_discrete_conditioning`. | RELOCATE not REPLACE: conditioning flags absent from `CliffordNetBlock`. **BUG (for appendix):** discrete conditioning is injected **additively** (`x_norm = x_norm + cls_feat`, `conditional_denoiser.py:391`) despite docstring claiming all modulations multiplicative — violates bias-free property; fix on relocate, do not preserve. |
| `BiasFreeGeometricDownsample` | `cliffordnet/conditional_denoiser.py:460-619` | RELOCATE | `layers/geometric/clifford_block.py` | — | inline: ctx stream `DWConv(s1)→DWConv(s2)→BN(center=False)→SiLU`; detail `Dense(bias=False)→AvgPool(2)`; fusion `SparseRollingGeometricProduct→Conv2D(1,bias=False)→BN(center=False)`. Absence: `layers/downsample.py` is strided-Conv2D/MaxPool only — no Clifford geometric product. | Novel geometric strided-downsample primitive; general to any Clifford encoder. Belongs alongside `SparseRollingGeometricProduct`. |
| `BiasFreeGeometricUpsample` | `cliffordnet/conditional_denoiser.py:628-766` | RELOCATE | `layers/geometric/clifford_block.py` | — | inline: `UpSampling2D(2,'nearest')` → opt `Conv2D(1,bias=False)` channel transition → `BiasFreeConditionedCliffordBlock` (conditioning disabled) refinement. Absence: no geometric-aware upsample in `layers/`. | Symmetric to `BiasFreeGeometricDownsample`. Belongs in `layers/geometric/clifford_block.py`. |

**Bugs noted for Synthesis appendix:** (1) additive discrete-conditioning bias-free violation at `cliffordnet/conditional_denoiser.py:391` (`x_norm = x_norm + cls_feat`); (2) in-`call` sublayer construction anti-pattern in `DiscreteConditioningInjection` (`RepeatVector`/`Reshape` built per forward pass instead of in `__init__`/`build`).
