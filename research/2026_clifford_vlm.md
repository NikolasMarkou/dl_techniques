# CliffordVLM: A Geometric Algebra Vision-Language Model

**Author**: Nikolas Markou | **Date**: 2026-10-16 | **Status**: Research & experiment plan (byte-patch, English-only, 320px COCO)

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Components Inventory](#2-components-inventory)
3. [Background](#3-background)
4. [Architecture](#4-architecture)
5. [Vision Tower](#5-vision-tower)
6. [The Bridge](#6-the-bridge)
7. [Decoders](#7-decoders)
8. [Byte Text Representation](#8-byte-text-representation)
9. [Training](#9-training)
10. [Metrics & Monitoring](#10-metrics--monitoring)
11. [Variants & Budgets](#11-variants--budgets)
12. [Open Questions & Risks](#12-open-questions--risks)
13. [Roadmap](#13-roadmap)
14. [References](#14-references)
- [Appendix A: Pseudocode](#appendix-a-pseudocode)
- [Appendix B: The Honest Attention Comparison](#appendix-b-the-honest-attention-comparison)

---

## 1. Executive Summary

This document is the build plan for a **Clifford-operator ablation study at
10–20M params on a single GPU**, framed as a small Vision-Language Model:
one operator family — the CliffordNet block built on the sparse rolling
geometric product (SRGP) plus gated geometric residual (GGR) — for the vision
tower, the vision-to-language bridge, and (in two of three arms) the decoder.
Text is raw UTF-8 bytes in fixed patches of k=4; there is no tokenizer.
The skeleton is **NanoVLM** (`models/vision_language/nano_vlm/` +
`train/nano_vlm/`): V0/V1 are two component swaps on it, not a new VLM.

The thesis: the geometric product decomposes interaction into an **inner**
channel (scalar alignment) and a **wedge** channel (antisymmetric signed
second-stream term `B(u,v) − B(v,u)` at offset `s`, with `u ^ u = 0` exactly —
verified numerically; genuinely distinct from a wider inner shift set,
measured 71% relative divergence in §6). Standard
dot-product attention computes only the first. Whether the second channel
earns its keep is an empirical question, and this plan tests it
directly instead of asserting it. "Geometric algebra" here means exactly this
second signed channel — not associativity / invertibility / closure, which
SRGP does not have (Appendix B).

Three architectures go to experiment, plus one baseline:

| Arm | Vision | Bridge | Decoder |
|---|---|---|---|
| **V0** (baseline) | CliffordNet | Dense MLP | Gemma3 mixed attention |
| **V1** | CliffordNet | SRGP + GGR | Gemma3 mixed attention |
| **V2** | CliffordNet | SRGP + GGR | CliffordNetLM, uniform K=3 |
| **V3** | CliffordNet | SRGP + GGR | CliffordNetLM, variable-K |

V0 isolates the question "does Clifford earn anything, or is this just a
decent small VLM". V1 tests the projector thesis with a conventional decoder
(the bridge as built is a **unimodal vision projector** — see §6; fusion
happens in the decoder). V2
and V3 test full attention replacement; V3 exists because measured receptive
field says uniform-K stacks cannot use a vision prefix (see §7).

They run as a cascade (§13): a text-only bake-off gates the replacement
thesis, a visual needle probe sets the vision-token budget, a bridge ablation
gates §1's thesis, and only survivors reach the full bake-off. All arms train
from scratch (no pretrained text-LM checkpoint exists in-repo) at a matched
**~16M param target** (adjust Gemma3 depth / CliffordNetLM depth to hit it;
report FLOPs/token separately). Stage 0b byte-LM warmup is mandatory for all
four arms. P1 starts in week 1 in parallel with P0 — it is ~3–6h on a 4090,
not a gate on the spine.

---

## 2. Components Inventory

### 2.1 What exists and is used as-is

| Component | Location | Notes |
|---|---|---|
| `SparseRollingGeometricProduct` | `layers/geometric/clifford_block.py` | `channels`, `shifts` (required, no defaults); `cli_mode ∈ {inner, wedge, full}`; `dot_activation` (default SiLU) applies to the **inner term only**; no `ctx_mode` on this layer |
| `GatedGeometricResidual` | same | Returns **only** the gamma-scaled term `γ ⊙ (SiLU(h) + σ(gate([h;g])) ⊙ g)`; residual add + `StochasticDepth` are external and unconditional. `gamma` is per-channel, init 1e-5 |
| `CliffordNetBlock` / `CausalCliffordNetBlock` | same | Two stacked depthwise convs (K=3 → 5-wide window per block); `ctx_mode="diff"` default; sequence mode accepts rank-3 and rank-4 |
| `CliffordNet` (vision) | `models/vision/cliffordnet/model.py` | **Isotropic**: one stem, one width, no stages. Variants `nano/lite/lite_g` (all 128-wide, depth 12). **No positional signal of any kind** (no table, no RoPE, no coords) — needs `vision_positional_encoding`-style add before P2. Receptive field is 49px ≈ 3.1 tokens at patch-16/320px (spatial mixing is conv-only; SRGP rolls channels). No `image_size` param, no `return_stages` / `_apply_vision_body` (see `clifford_clip/model.py::_vision_tokens` precedent) |
| `CliffordNetLM` | `models/language/cliffordlm/model.py` | Causal LM, learned token + position tables, tied logits default. Param is `max_seq_length`. `call(ids)` only — no `embed_tokens`/`decode_embeddings`. No `generate()` on model (see `train/cliffordnet/train_cliffordnet_nlp.py` + `infer_cliffordnet_nlp.py` BPE skeleton). No length guard (bare `Embedding(max_seq_length)` + `arange`). `context_kernel_size` not plumbed (shared `_block_kw`, all K=3) |
| `Gemma3TransformerBlock` + `Gemma3` | `layers/transformers/gemma3_transformer.py`, `models/language/gemma/gemma3.py` | Per-block `attention_type ∈ {sliding_window, full_attention}`; causal+band masks built internally and OR-ed, rank-3; padding mask AND-ed (keys-only, right-pad packing-safe, no doc-boundary mask); no QK-norm; `rope_theta` not exposed. `call(input_ids)` only — **no embed/decode split** (needs one for prefix use, mirror of `TextDecoder.embed_tokens`/`decode_embeddings`). `max_seq_len` enforced via `SequenceLengthError`. No distributed weights (`pretrained=True` raises) |
| `NanoVLM` skeleton | `models/vision_language/nano_vlm/model.py`, `train/nano_vlm/train_nano_vlm.py` | **Reuse as-is for V0/V1**: `VisionEncoder(output_mode='none')` → `ModalityProjection` → concat `[vision; text]` → `TextDecoder.embed_tokens` / `decode_embeddings` over one causal mask → tied head → `generate()`. `test_the_text_logits_are_causal.py` is the RED causality precedent. V0/V1 = swap vision tower + projector only |
| `TextDecoder` split | `layers/transformers/text_decoder.py:448,468` | `embed_tokens(ids)` + `decode_embeddings(joint)` + static-length refusal — the interface `Gemma3` and `CliffordNetLM` must grow |
| `ModalityProjection` | `layers/modality_projection.py` | `PixelShuffle(scale_factor)` + `Dense` + GELU + LN. Rank-3 only, `(B, 1+H·W, C)`; **CLS at index 0 mandatory**; spatial count must be a perfect square divisible by `scale_factor`. Kwarg is `scale_factor`, not `s` |
| `NanoVLMLoss` + `text_position_logits` | `losses/nano_vlm_loss.py` | Slices the trailing `seq_len_true` positions (skipped when static lengths equal), then a 1-step left-drop shift in `NanoVLMLoss.call` (first text position never scored); `ignore_index=0` default; requires rank-2 targets — byte targets must be flattened to `(B, P·k)` first. Label smoothing is vocab-agnostic (256 OK). `make_lm_labels` negative convention unusable here |
| `byte_patch.py` / `byte_lm.py` | `datasets/byte_patch.py`, `datasets/byte_lm.py` | `bytes_to_patches(k=4)` pads to multiple of k with `PAD_BYTE_ID=0`, row-major. `patches_to_bytes` does **not** strip pad — round-trip is exact only when `len % k == 0`. `BOS_ID=254`, `EOS_ID=255` (ordinary vocab members, no mechanism). `pack_byte_patch_windows` lives in `byte_patch.py`, fixed-length, **no mask arg**, no doc separators. `byte_patch.py` has zero non-test consumers; `byte_lm.py` is live in `train/hnet/` — reuse both |
| `CLIPContrastiveLoss` | `losses/clip_contrastive_loss.py` | Standard two-way contrastive |
| `AffineInvariantLoss` | `losses/affine_invariant_loss.py` | Median+MAD normalized L1, per-sample `(batch,)` return; depth auxiliary |
| `SegmentationHead` | `layers/heads/vision/factory.py` | Power-of-two `upsampling_factor`; **every skip must match the base map's H×W** (checked, raises); returns a bare tensor |
| VLM task heads | `layers/heads/vlm/factory.py` | Captioning, VQA, grounding, matching |
| `DPTDecoder` | `models/vision/depth_anything/components.py` | Depth decoder for the downstream depth track |
| Synthetic-generator precedent | `datasets/document_rectification/synthetic_warp.py`, `datasets/document_restoration/dtsprompt.py` | numpy+scipy only, no cv2/skimage — guard-enforced; new generators follow the same rule |

### 2.2 Exact SRGP math (what the code does, not the paper's Eq. 11)

For each shift `s` (`keras.ops.roll` indexes `(c−s) mod D` — a documented,
deliberate deviation from the paper's `(c+s)`):

```
wedge_c^(s) = z_det,c · z_ctx,(c−s) − z_ctx,c · z_det,(c−s)   [no activation, raw quadratic]
inner_c^(s) = SiLU(z_det,c · z_ctx,(c−s))                    [activation here only]
full: interleave per shift [wedge_s1, inner_s1, wedge_s2, inner_s2, …] → Dense(D)
```

Width before projection is `2·|S|·D`; the projection is what makes SRGP cost
`2·|S|·D²` parameters (measured §11). Shifts `≥ channels` are dropped with a
warning; all dropped raises. `get_config` stores the post-filter list (not
idempotent across a `channels` change).

Wedge status (measured, integer-exact): `W_s == I_s − roll(I_{−s}, s)` holds at
the **raw-product** level, but `full@[1,2,4]` is **not** reparameterizable as
`inner@[±1,±2,±4]` (71% relative divergence with re-derived weights) because
the inner column stores `SiLU(product)`. The wedge is a genuinely distinct
signed second channel. `cli_mode="wedge"` makes `ctx_mode` mathematically
inert (self-terms cancel) — C3 is `ctx_mode="abs"` by construction, so C2/C3
needs a `C6 = inner + ctx abs` control (§6).

### 2.3 What must be built (the complete new-code list)

1. **Reusable Clifford vision body** — stage-map (here: single-map) return.
   `CliffordNet.call()` inlines its loop over a `List[Dict]`; there is no
   `_apply_vision_body` and no `return_stages`. Copy the
   `clifford_clip/model.py::_vision_tokens` flatten pattern. Replace the
   pooled classifier head with a token map. Every variant needs this
   refactor first.
2. **Vision positional add** — opt-in learned 2-D add (copy
   `clifford_clip/model.py:vision_positional_encoding` precedent: `x + pos`).
   Tower is translation-equivariant without it; P2 is uninterpretable until
   this lands. Gate P2 on it.
3. **Byte patch-embed layer** (~50 lines) — 256-entry table → concat-k →
   `Dense` + LN (§8).
4. **Byte output head** — per byte-frame slot `concat(gather(h,−1-shifted),
   byte_emb[0..j])` + tiny local causal conv + 256-way `Dense`, with the §8
   masking rule (`h_{i-1}` ⊕ in-frame bytes; never repeat `h_i` ×k). Gated by
   the RED-first leak probe.
5. **Data-side pad/ignore producer** — right-pad packed windows to fixed patch
   count + emit pad mask consumed as `ignore_index=0`. `pack_byte_patch_windows`
   has no mask arg today. Specify doc separators + boundary handling for packed
   SFT (Gemma3 mask is keys-only, no boundary mask).
6. **`context_kernel_size` plumbing in `CliffordNetLM`** — `lm.py`
   uses one shared `_block_kw` (all K=3); V3 needs a per-block dict. Plus
   `global_context_period` already exists — use it for P1 A3 at zero plumbing.
7. **`CliffordNetLM` + `Gemma3` embed/decode split** — both `call()`s accept
   only token IDs (`token_embedding(ids) + position_embedding(arange)`
   internally); neither has an `embed_tokens`/`decode_embeddings` path for a
   pre-formed `[vision; text]` sequence. Mirror
   `TextDecoder.embed_tokens`/`decode_embeddings` (+ static-length refusal).
   V0/V1 need the Gemma3 side; V2/V3 need the CliffordNetLM side. Hard
   prerequisites, currently invisible. (If reusing full `NanoVLM`, only the
   tower/projector swap is needed for the `TextDecoder` path.)
8. **`CliffordNetLM` length guard** — bare `Embedding(max_seq_length)` +
   `arange(seq_len)` is a silent OOB gather when over-long. Add
   `check_sequence_length` refusal like Gemma3/`TextDecoder`.
9. **Inner/wedge norm taps in `SRGP`** — `call()` discards `components`;
   all geometric health metrics are unobservable until this lands.
10. **Scalar-metric sink** — `utils/logger.py` exports no functions; route
   Clifford scalars through `StepCheckpointCallback` (`src/train/common/`,
   dynamic `logs` → CSV, no schema), not a new `log_scalar`. Note: dynamic
   schema derives fields from first row; short probes may emit no CSV at
   default `log_every_steps=100`.
11. **Padding/masking pins** — sequence-mode blocks declare
   `supports_masking=False`, and the non-causal global branch pools over pad
   (measured per-position deltas 0.074–0.449, including position 0). Pin in
   P0: right-pad-only + causal decoder + `ignore_index=0` on the byte path,
   and `use_global_context=False` in the bridge while the token count is
   fixed.
12. **Multi-class Hungarian matcher** — `_Sam3HungarianMatcher`
   (`losses/sam3_detection_loss.py`) is private by decision, binary-only,
   and bound to the SAM-3 packed layout. A closed-80 DETR head needs its own
   `(Q,C)×(N,C)` cost written from scratch. Detection is a P5 item, not P0.
13. **Initializer spec** — `CliffordNetLM` defaults to `glorot_uniform` while
   variants use `TruncatedNormal(0.02)`; block output is quadratic in
   activations. Pin `TruncatedNormal(0.02)` for all LM runs. Revisit
   LayerScale init (1e-5 inherited from deep vision stacks; 12-layer LM may
   need larger) in P1.

No tokenizer, no `TextDecoder` changes (untouched by design — no
`attention_args`/`n_kv_head` passthrough exists there), no new fusion
layer (symmetric cross-attention leaks future text into causal positions;
the one consumer already moved off it).

---

## 3. Background

Three results shape this plan; everything else is context.

**LLaVA paradigm** [[Liu 2023]](#14-references): vision encoder → projector →
concatenated prefix → causal LM. All four arms follow it. The only question
is what the projector and the LM are made of.

**Penguin-VL** [[Zhang 2026]](#14-references): trains the vision encoder
during alignment (not projector-only, +3.3), uses reconstruction losses
(amplitude/direction/relation, relation alone +3.3) in a low-res warm-up,
mixes ~10% text-only data against forgetting, and runs a low→high resolution
curriculum. §9 adopts all four.

**JEPA** [[Assran 2023]](#14-references), [[Chen 2025]](#14-references):
predict in embedding space, not pixels/tokens; SRGP is structurally a JEPA
predictor (context + position → target embedding). VL-JEPA's selective
decoding is a deployment option, not a v1 item. Anti-collapse means VICReg
[[Tong 2024]](#14-references) if a JEPA pre-training track ever runs —
wedge antisymmetry and LayerScale are *not* anti-collapse mechanisms
(`u^u=0` holds exactly in the collapsed state).

---

## 4. Architecture

```
Image (B,320,320,3)                              raw UTF-8 bytes
     |                                                 |
     v                                                 v
[CliffordNet stem + L×Block]                  bytes_to_patches(k=4)
(bidirectional, isotropic)                      (B, P, 4)
     |                                                 |
     v                                                 v
(B,400,128) vision map                   patch-embed → (B, P, D)
     |
     v
[Bridge: Dense(128→384) + 1–3 × CliffordNetBlock (sequence) @400
         → CLS prepend + PixelShuffle(s=2) → (B,101,384)]
     |  (V0: Dense-GELU-Dense @400, then same compression)
     v
projected vision tokens (B, Sv, D) ──┐
                                      ├── concat (B, Sv+P, D)
text patch embeddings (B, P, D) ─────┘
                                      |
                                      v
                          decoder (V0/V1: Gemma3 mixed 5:1
                                   V2/V3: CliffordNetLM causal)
                                      |
                                      v
                          byte head → next-byte logits (B, Sv+P·k, 256)
```

Vision prefix positions never score loss (dropped by the trailing slice in
`text_position_logits`); text loss is next-byte CE with `ignore_index=0`.

---

## 5. Vision Tower

`CliffordNet` isotropic, one stem, constant width. Measured at 320px:

| Stem | Params (D=128, L=12) | Grid | Tokens |
|---|---|---|---|
| patch 8 | 1,444,865 | 40×40 | 1600 |
| patch 16 | 1,518,593 | 20×20 | 400 |

(Counts are whole-model with 1-output classifier; body-only is −12,771.
`image_size` is not a `CliffordNet` param — the constraint lives in CLIP only.
`CliffordNet` itself is resolution-agnostic.)

Default: **patch 16 → 400 tokens**. Two prefix options, both verified:

| Prefix | How | Tokens | Stride | Use |
|---|---|---|---|---|
| full | direct | 400 | 16px | P2 large-prefix arm |
| default | `Dense`(128→384) + blocks @400, then CLS + `ModalityProjection(384→384, s=2)` | **101** | 32px | ship default |

Receptive field (correct): SRGP rolls channels only; spatial mixing is the two
stacked same-padded depthwise K=3 convs → 5px/block → `5+11·4 = 49px` at
depth 12 → **≈3.1 tokens** at patch-16. There is no `4L+1`-token reach on the
tower. Bridge "reasoning at full resolution" therefore buys one ±2-token
neighborhood per block; a PixelShuffle(s=2) output already spans 64px.

Positional gap (load-bearing): the tower has **no positional signal** —
identical local patterns at different cells give identical features. Add the
opt-in learned 2-D `x + pos` add (copy `clifford_clip` precedent) before P2;
otherwise grid-cell needle / grounding numbers are uninterpretable.
Compression injects only cell-relative offset (parity within the 2×2 cell),
not absolute position.

The compression step was verified end-to-end (401 in → 101 out). Stride 32 ≈ one 9×9 sudoku
cell per token at 320px — the granularity §9's puzzle suite needs, and the
scale P2's needle probe will confirm or reject. The hierarchical
CliffordCLIP tower is *not* the VLM backbone (its 10×10 S4 map forces the
25-token prefix that cannot read a sudoku cell); its role is pre-training
initialization only (§9, Option 1).

---

## 6. The Bridge

The bridge as built is a **unimodal vision projector**: it sees vision tokens
only, no text. Fusion happens downstream — in Gemma3 attention (V0/V1) or
through the causal conv window crossing the prefix boundary at ±2 positions
(V2/V3). Reason-first-then-compress therefore sharpens vision features; it is
not cross-modal fusion. P3 ablates the projector; only V3 tests the §1 thesis
end-to-end.

A `Dense` dim-align (128→384 —
blocks are isotropic, channel mismatch raises) feeds 1–3 `CliffordNetBlock`s
in sequence mode over all 400 tokens (rank-3 verified); a learned-CLS
prepend and `PixelShuffle(scale_factor=2)` then compress to 101 tokens
(compression verified 401→101). Note the CLS arrives zero-padded from C to
C·s² before the Dense (constant contribution, not per-sample signal).
This ordering is deliberate:
compress-then-reason would hand the SRGP pre-aggregated 64px cells,
destroying the tower's 16px detail before any geometric filtering runs. Same
parameters either way; ~4× the bridge FLOPs, negligible beside the decoder.
One block at D=384 costs 1.33M params (measured; 885k of it the SRGP
projection).

Six arms, fixed tower and decoder, bridge swapped:

| Arm | Bridge |
|---|---|
| C1 | Dense-GELU-Dense (LLaVA-style; also V0's bridge) |
| C2 | SRGP + GGR, `cli_mode="inner"`, `ctx_mode="diff"` |
| C6 | SRGP + GGR, `cli_mode="inner"`, `ctx_mode="abs"` — control for C3 |
| C3 | SRGP + GGR, `cli_mode="wedge"` (= `abs` by construction) |
| C4 | SRGP + GGR, `cli_mode="full"` |
| C5 | dim-align + compress, no blocks — floor |

C2-vs-C3 without C6 is confounded (wedge-vs-inner AND diff-vs-abs). The
load-bearing comparison is C1-vs-C4 at matched params; C2/C6/C3 isolates the
wedge channel.
If `wedge` alone doesn't beat `inner`, the thesis is rewritten, not the
experiment. Ship gate (§13, P3): C4 beats C1 by ≥2% relative needle accuracy
and ≥0.01 bits/byte on captioning with no throughput regression; otherwise
the ship takes C1.

---

## 7. Decoders

**V0/V1: package-owned Gemma3 stack.** `TextDecoder` stays untouched (legacy
GPT-2 defaults, no KV-head passthrough) for the NanoVLM path. For the Gemma3
path, add `embed_tokens`/`decode_embeddings` + length refusal mirroring
`TextDecoder`. Per-block `attention_type`, sliding + full with a
**depth-aware** sliding fraction (5:1 at depth 6 is one global layer — scale
the ratio with depth, do not copy Gemma3-4B's 5:1 blindly), GQA with real
KV-head control, RoPE positions (no learned table to resize). Target **~16M
matched params**: 384/6L/6Q-2KV, w=32, GeGLU-1152, `max_seq_len=1024`
(measured 10.5M params — deepen to hit 16M; the 1024 cap covers every P2 prefix outcome —
a 512 cap hard-fails the joint sequence with `SequenceLengthError`). From
scratch — no distributed weights.

**V2/V3: `CliffordNetLM`.** Causal blocks, left-pad convs, causal cumulative
mean global branch, learned positions, tied byte logits. The deciding
measurement — look-back reach on random init (perturbation probe, relative
threshold, layers built once):

| Decoder | Nominal reach | **Measured (random init — smoke signal only)** |
|---|---|---|
| uniform K=3, depth 6–24 | 4L | **~22 (depth buys nothing)** |
| K = 3,5,9,17 ×3 | 180 | **~94** |
| K = 3,7,15,31 ×3 | 360 | **~173** |
| + cumulative-mean branch | global | global, but **rank-1** |

So V2's usable vision prefix is ~22 tokens and V3-at-`[3,5,9,17]` is ~94:
with Sv=101 neither reaches the first vision token on random-init numbers.
Do not size Sv off this table — re-check reach on trained weights in P1
(random-init RF with γ=1e-5 is largely an init artifact), then set Sv and the
ship K-schedule. The table as printed does not support Sv=101 for V3; the
`~170` reading belongs to the `[3,7,15,31]` row, not the ship schedule.
Position-specific retrieval ("which
patch?") is unanswerable in principle under a local window + rank-1 mean;
that is the accepted ceiling of the replacement track, and grounding-heavy
tasks are where V1 is expected to win.

Converted to responses (1 patch = 4 bytes ≈ 1.25 BPE tokens at the measured
3.20 bytes/token): V2's local reach ≈ **27 BPE tokens** (4% of the 512-patch
context), V3's ≈ **212** (33%). Past that horizon only the rank-1 mean
reaches the image — short answers (VQA, cell queries, captions ≈ 15 patches)
are fine on all arms; multi-turn SFT responses go blind on V2. P2's
delayed-query sweep measures exactly this.

Measured decoder sizes (vocab 256): D=384/L=12 → 16.3M; D=384/L=18 →
24.3M; D=256/L=18 → 10.9M. P1 matches all arms at **~16M** (adjust depth) and
reports FLOPs/token separately. Pin `TruncatedNormal(0.02)` for every LM run
(`glorot_uniform` default + quadratic block output is a first-order scale
error); revisit LayerScale 1e-5 in P1 (12-layer LM ≠ deep vision stack).

---

## 8. Byte Text Representation

No BPE, no tokenizer on the VLM path. Raw UTF-8 → fixed patches of k=4
(CANINE precedent [[Clark 2022]](#14-references)).

**Encoding** (`datasets/byte_patch.py`, reuse): pad byte-length to a multiple
of 4, row-major reshape. `0x00` is valid UTF-8 (U+0000) but absent from
cleaned text corpora, so it serves as pad; `0xFF` is never valid UTF-8 in any
position (neither lead nor continuation) and serves as EOS (in-tree convention:
`BOS_ID=254`, `EOS_ID=255` are conventions, not mechanisms). `patches_to_bytes`
does not strip pad — exact inverse only when `len % k == 0`.

**Length math** (measured 3.20 bytes/GPT-2-token on repo prose, not ~4):
512 BPE tokens ≈ 1,639 bytes ≈ **410 patches**. Text context is 512 patches
(≈640 BPE tokens); joint budget is Sv + 512 ≤ 1024.

**Patch embedding** (~50 lines, new): byte table 256×D (98,304 params at
D=384) → concat last two axes → `Dense(D)` + LN. The omitted patch-embed
`Dense` costs 590,592 params — it is in the budget (§11).

**Byte head** (new — the one subtle component). The global decoder runs at
patch resolution; only the head touches byte resolution. Naive
repeat-`h_i`-×k fails twice: no in-patch byte identity (bytes 1..3
unpredictable) and leak (`h_i` saw bytes through end of patch `i`). Build:

```
per byte-frame slot (i,j): concat(gather(h, i-1 shifted), byte_emb[0..j]) → tiny local causal conv → Dense(256)
```

**Causality rule (load-bearing):** byte *j* may see patch states of patches
**strictly before** `patch(j)`, plus earlier bytes inside its own patch.
Naive repeat-and-predict leaks: patch *i*'s state was computed from bytes
through the end of patch *i*. The mask is `patch_id_strictly_less OR
(same_patch AND byte_pos_less)` — both conditions required. There is a
sufficiency argument behind the design: under a causal decoder, `h_{i-1}` is
a sufficient statistic for every byte before patch *i*, so byte `4i+j` needs
only `h_{i-1}` plus ≤3 in-frame byte embeddings — no cross-frame attention
in the head, ever.

**Acceptance is gated by a RED-first leak probe** (repo's own causality-test
style: two forwards differing at one future byte leave all earlier
predictions unchanged): (a) shuffle future bytes *within* the target patch →
earlier-byte logits bit-identical; (b) shuffle a *future patch* →
current-patch logits bit-identical; (c) positive control — shuffle a *past*
byte → logits move. Plus: `ignore_index=0` pad-mask exactness **against the
P0 data-side producer** (no producer exists today — §2.3.5), per-position
CE vs a naive per-byte loop oracle on 8-byte inputs, byte-identity check
(bytes 1..3 of a patch predictable, not copies of byte 0), `.keras` value
round-trip, and `h_{-1}` boundary assert (`text_position_logits` left-drops
position 0, which covers frame 0 only by accident — pin it). No training run counts until (a)+(b) pass and (c) moves.

**Loss / generate / decode.** Flatten byte targets to `(B, P·k)` (rank-3
raises in `text_position_logits`); standard next-byte CE over trailing text
positions, `ignore_index=0` (= `PAD_BYTE_ID`; do not use
`make_lm_labels`' negative-ignore convention on this path). `generate()`
emits bytes autoregressively, stops at `0xFF`, UTF-8-decodes with replacement
at the end. Open items specified here, not later: partial final frames (1–3
bytes present), the frame-0 boundary `h_{-1}` (pinned by probe, not luck),
and packed-SFT doc separators / boundary masks.

---

## 9. Training

### 9.1 Data pipeline (English-only; bytes remove the tokenizer, not the language decision)

Packing via `datasets/byte_lm.py` (live in `train/hnet/` — reuse the pipeline,
don't re-derive); mandatory wherever caption batches would
leave the 512-patch tail untrained. COCO/CC3M stay CLIP-contrastive-only.
Specify doc separators + boundary masks for packed SFT (§2.3.5); Gemma3's
padding mask is keys-only with no boundary support.

| Stage | Dataset (HF id) | Size | Use |
|---|---|---|---|
| 0a (CLIP init) | CC3M / LAION-400M | 3–400M | Contrastive vision-tower init (one option among §9.2). Cited `cliffordclip_nano_20261015` run has zero artifacts — treat as not started |
| 0b (byte LM warmup) | FineWeb-Edu (bytes) | ~20B bytes (~6.2B tokens at 3.20 B/tok; ~19× Chinchilla-optimal for 16.5M, ~3–6h on 4090) | **Mandatory for all four arms** — every decoder is from scratch; no pretrained text LM exists in-repo. Largest data item, not largest compute item |
| 1 (align, 320px) | `allenai/pixmo-cap`, `nlphuji/flickr30k` | ~0.5M | Bridge warmup, short contexts; encoder at low LR (Penguin: train it, don't freeze it) |
| 2a (instruct, packed) | `liuhaotian/LLaVA-Instruct-150K` | 150k | Core instruction |
| 2b (instruct mix) | `HuggingFaceH4/llava-instruct-mix-vsft` | ~100k | Diversity |
| 2c (multi-task) | `HuggingFaceM4/the_cauldron` | 50–100k of 1.88M | VQAv2/GQA/ChartQA unified |
| 2d (synthetic, §9.3) | in-tree generators | 5–10% of mix | Puzzles; seeded, style-shift eval |
| 3a (text-in-image) | `facebook/textvqa` | 45k | GATED on P2: at 320px / stride-32 with a 1.5M tower, glyphs are sub-token. Stays out unless P2 needle shows readable cells; consistent with the 320px exclusion below |
| 3b (diagram math) | `hiyouga/geometry3k` | ~3k | MathVista proxy |
| 3c (docs, DEFERRED) | `HuggingFaceM4/Docmatix` | 2.4M imgs | Gated on the P2 resolution trigger |
| 3d (dense COCO) | COCO-2017 | 118k | Detection + seg heads (P5) |
| 4 (preference, later) | `HuggingFaceH4/rlaif-v_formatted` | 83k | Post-SFT hallucination reduction |

English filter at load; ~10% text-only bytes in Stage 2 (no forgetting).
Square resize/letterbox (generic ViT practice; no Clifford `image_size`
constraint exists — `CliffordNet` is resolution-agnostic).

### 9.2 Vision-tower initialization (options, cheapest first)

1. **CliffordCLIP contrastive** (in progress: `cliffordclip_nano_20261015`
   run in `results/`) — aligns vision with language but may suppress
   fine-grained cues (Penguin argument).
2. **Classification pre-training** — cheap, no alignment.
3. **Penguin-style reconstruction** (amplitude/direction/relation vs a
   frozen teacher; low-res warm-up, then drop) — dense supervision, most
   data-efficient; relation loss was +3.3 in Penguin.
4. **I-JEPA masking** with an SRGP predictor + EMA target — zero paired
   data; needs VICReg to not collapse.

### 9.3 Synthetic puzzle suite (5–10% of Stage 2 mix)

Seeded numpy+scipy at module scope, PIL lazy, no cv2/skimage (house rule,
guard-enforced). All Q/A byte-short (<128 bytes ⇒ <32 patches). Held-out
palette/texture/size bins per family; the plain-vs-styled gap is the
vision-robustness metric. Six families + A|B montage pairs: **Sudoku**
(valid → cell → row/col/box → full grid, mono → colors → textures →
rotations); **mazes** (rectangular / circular / 3D-projected with lens
distortion, answers in topology coords, next-move → path → unreachable);
**color nonograms** (clues rendered in-image); **ARC rule grids** (recolor,
symmetry, translate, count-and-point); **texture odd-one-out + tile swap**
(rotation/mirror/hue/grain/scale — probes wedge-vs-blob vision);
**spot-the-difference montages** (single-frame A|B, count → what → where).
Tiny-model note: validity/single-cell/next-move saturate first; full grids
and full paths are stretch goals gated on early-task saturation.

### 9.4 Stages, losses, optimizer

**Align** (bridge + projector warmup, encoder low LR, decoder frozen except
V0/V1's adapter) → **joint** (all trainable; backbone LR ×0.1; calibrated
per-task loss weights, per-task curves split) → **SFT** (instruction mix) →
**preference** (later). Reconstruction losses only in the low-res warm-up, if
used.

Primary loss is next-byte CE (§8). `MultiTaskLoss` is MSE-only and returns a
scalar — wrong tool; joint training uses stock `compile(loss={…})` dicts.
`exclude_from_weight_decay` (e.g. `logit_scale`) goes through
`optimization/optimizer.py` (`optimizer_builder()`; note it renames clipping
keys and the call **overwrites** rather than appends). `TerminateOnNaN`
(Keras built-in) from step one.

| Parameter | Align | Joint | SFT |
|---|---|---|---|
| LR (encoder) | 1e-3→5e-4 | 1e-4 | 1e-5 / frozen |
| LR (bridge) | 1e-3 | 1e-3 | 5e-4 |
| LR (decoder) | frozen (adapter only) | 2e-4 | 2e-5 |
| Batch / clip / precision | 128 / max_norm 1.0 / fp16-fwd-fp32-opt | 64 / same | 32 / same |

---

## 10. Metrics & Monitoring

Clifford inner/wedge norms and the scalar sink land in P0 (items 0.5–0.6) —
until then the per-block thresholds below are aspirations, not gates.

**Every step (all exist):** train loss (text positions only), perplexity,
per-component LR, grad norm before/after clipping, tokens/sec.

**Clifford health (needs P0 items 0.5–0.6):** `layerscale_gamma` mean/max per
block (access via `block.ggr.gamma.gamma` — `gamma` is a sub-layer, not a
weight); `inner_norm`, `wedge_norm`, `inner_wedge_ratio` per SRGP
(healthy 0.3–3.0; collapse of either side to 0 = dead channel);
`gate_activation` mean (0.3–0.7). If gamma sits at init past 30% of training,
check init scale first (1e-5 may be too small for a 12-layer LM) before
reading it as a dead channel — the model is a Dense VLM wearing a Clifford
badge only if gamma stays flat *after* init is fixed.

**Probes (the sharp instruments):** byte needle-in-haystack vs distance (P1);
grid-cell needle vs K (P2); plain-vs-styled puzzle gap (§9.3).

**Validation:** held-out caption loss/ppl; text-only ppl (forgetting tripwire
— rising while VLM loss falls means add text data); CIDEr/BLEU-4 on COCO-val;
VQA exact-match on a VQAv2 slice. **Expectation band** (single 4090, ~1–2M
pairs, decoder from scratch): COCO CIDEr ≈ 0.8–1.1, VQA near chance,
validity/single-cell puzzles learnable, full grids/paths not. These are
acceptable experiment outcomes and unacceptable model failures — P4
interprets against this band. **Scoped to 320px reality:** DocVQA,
OCRBench, MMMU, MathVista, RefCOCO, NYUv2/KITTI stay out of v1 gating —
Docmatix is already deferred on resolution grounds.

**Checkpointing:** every 2K steps align / 5K joint; best-3 by val loss; always
at phase transitions. Early stop: val flat 15K steps; grad norm >100 for 50
steps; gamma-flat warning above.

---

## 11. Variants & Budgets

All counts measured by execution (`count_params()` at build; assertions, not
estimates):

| Part | Config | Params |
|---|---|---|
| vision body | CliffordNet 128/12, patch 16 @320 | **1,518,593** |
| bridge | Dense(128→384) + 1 × Block-384 @400 + CLS/PixelShuffle(s=2) | **~2.0M** (block 1,332,096 incl. 885,120 SRGP; compress Dense 590,976 measured) |
| patch-embed | table 256×384 + concat-4 → Dense | 98,304 + 590,592 |
| byte table | 256×384, tied | 98,304 (assert `== 98,304`) |
| decoder V2/V3 | CliffordNetLM 384/12, ctx 1024 | ~16.5M |
| decoder V0/V1 | Gemma3 384/6, 6Q/2KV, 5:1 w=32 | **10,528,128** |
| detection head (P5) | Q=100, 2–3 cross layers | ~5M (new matcher required, §2.3.7) |
| seg head (P5) | `SegmentationHead` tap | ~1M |

Ship sketch (V2/V3, 1-block bridge, matched ~16M): 1.5 + 2.0 + 0.7 + 16.5 ≈ **~21M** —
deepen Gemma3 arms to the same ~16M decoder before any comparison (V1 sketch
at 10.5M is 1.5 + 0.9 + 10.5 ≈ **~13M** unmatched; do not compare 13M vs 21M).
Movers: decoder depth 12→18 (+8M), 3-block bridge (+2.7M), untied byte head (+98k — still don't).

Scale story, stated honestly: SRGP costs `2·|S|·D²` parameters — at every
recommended shift set it is parameter-*denser* than the cross-attention it
replaces (885k vs 590k at D=384). The surviving efficiency claim is
**per-token FLOPs at long sequences** (O(N) in length, no KV cache growth),
not parameter count. Comparison-table "attention-free / O(N)" rows may only
describe V2/V3, and only if P1 keeps them alive.

---

## 12. Open Questions & Risks

1. **Can CliffordNetLM hold English?** No perplexity number exists anywhere
   (doc or `results/`). P1 answers in ~2 days. Kill gate pre-registered.
2. **Does the wedge channel earn its keep cross-modally?** Asserted in §1,
   never tested. P3's C3 arm is the test.
3. **What is each arm's usable vision prefix?** Nominal `4L+1` overstates
   measured reach by 2–4× on random init. P2 replaces argument with a number;
   re-check RF on trained weights. P2's delayed-query sweep additionally
   measures reach *into the response* (vision access decays with distance);
   V2 is expected blind past ~22 patches, which excludes it from
   long-response SFT without a reach fix.
4. **Does variable-K survive training?** Dilated causal convs are the
   Clifford-native local/global answer, but large-K kernels on 384-d stacks
   are untested here. P1's A2/A3 arms decide.
5. **Grounding ceiling.** Local window + rank-1 global mean cannot do
   position-specific retrieval in principle. If RefCOCO-style grounding is a
   v1 requirement, V1 is the ship and the replacement track is research.
6. **Depth auxiliary** (Strategy: Clifford depth head + `AffineInvariantLoss`
   vs frozen-teacher pseudo-GT; bias-free blocks live in
   `models/vision/bias_free_denoisers/`, not in a `cliffordnet/denoiser.py`).
   P5 item; gate on spatial benchmarks without general regression.
7. **Byte-head leak** — gated by the §8 RED-first probe before any run counts.
8. **Tail-position training** — if tail ppl diverges while head ppl falls,
   add multi-turn mass, not more captions.
9. **Fixed k=4 vs morphology** — CANINE evidence says fine at this scale;
   ablation k=8 vs k=4 on TextVQA-val if time permits.

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Clifford LM uncompetitive at matched params | Medium | High | P1 kill gate; V1 ships |
| Wedge adds nothing cross-modally | Medium | High | P3 gate; C1 bridge, §1 rewritten |
| RF caps prefix below puzzle needs | Medium | Medium | P2 sets Sv; resolution trigger; V1 fallback |
| Vision access decays past reach mid-response | Medium | High | Delayed-query arm in P2; V2 out of long-response SFT without reach |
| Cross-modal shift subspaces don't interact | Medium | Medium | Bridge aligns first; LayerScale starts near-identity |
|Instability in joint training | Low | Medium | Frozen-decoder align first; `TerminateOnNaN`; grad clip |

---

## 13. Roadmap

```
P0  shared spine (built once, serves every arm) · ~3–5 days
 ├─> P1  text-only LM bake-off (starts parallel week 1) → keeps or kills replacement · ~2 days
 ├─> P2  visual needle probe (gated on vision pos add) → sets Sv and resolution trigger · ~3 days
 ├─> P3  bridge ablation (C1–C6) → keeps or kills the wedge thesis · ~2 days
 └─> P4  bake-off at scale       → which of V1/V2/V3 ships · ~1–2 weeks
         └─> P5  ship + scale + dense heads + depth/JEPA tracks
```

Deliverable framing: a controlled operator-family test at 10–20M params on
one GPU — not a SmolVLM/nanoVLM competitor (256M+). A kill is a successful
outcome.

**P0 — shared spine.** Nothing else starts until 0.1–0.5 and 0.8 land (build
prerequisites plus the padding pins), except P1 which starts in parallel
(~hours, text-only, needs only 0.6/0.8 + trainer skeleton). 0.1 reusable vision body (stage-map
return, drop classifier for token map); 0.2 byte head + RED-first leak probe
(concat `h_{i-1}` ⊕ in-frame bytes, not repeat); 0.3 `context_kernel_size`
per-block dict + length guard; 0.4 `embed/decode` split for **both**
`CliffordNetLM` and `Gemma3` (or reuse `NanoVLM`+`TextDecoder` for V0/V1);
0.5 SRGP norm taps + vision pos add (P2 gate); 0.6 metric sink via `StepCheckpointCallback`
(`src/train/common/`, watch dynamic-schema + short-run no-CSV); 0.7 pre-registered gates + frozen eval suite; 0.8 padding/masking pins +
data-side pad producer + SFT doc boundaries + `TruncatedNormal(0.02)` pin.
Side effects: fix the (small) spec blockers the audit found as you touch
them; delete nothing under `results/`.

**P1 — text-only bake-off.** Arms A0 (Gemma3 ref) / A1 (uniform K=3) / A2
(variable-K) / A3 (+global branch via existing `global_context_period`).
**Primary metric: byte-needle accuracy
vs distance** (reach) — perplexity is dominated by local statistics and can
pass a reach-blind arm; bits/byte **delta vs named baseline A0 at matched
tokens-seen** is secondary.
Gates on Δbits/byte (A-arm minus A0, same data/order): ≤−0.05 → replacement lives; −0.05–0.0 → proceed with V1
as ship candidate; >0.0 → kill replacement, V1 ships, Clifford survives as
the projector operator. (Absolute 0.25/0.50 bands in earlier drafts were
below the byte n-gram floor and unexecutable — deltas only.) Plus a reach floor no gate waives: a surviving arm must
retrieve above chance at 256 patches (≈ ship prefix + response hypothesis) —
otherwise it cannot serve the prefix regardless of bits/byte. Reuse
`train/cliffordnet/train_cliffordnet_nlp.py` skeleton + `GenerationProbeCallback`
(byte-path variant) — do not start from empty.

**P2 — visual needle probe (gated on vision pos add from P0).** Grid images, "value at (r,c)?", K ∈
{4,8,12,16,20}. Arms B0 (Gemma3) / B1 / B2 / B3 (B0+Dense bridge, isolates
decoder from bridge). Two sweeps: (a) grid size K at response start → max
operable prefix per arm → Sv, `scale_factor`, resolution trigger; (b)
**delayed query** — same question after N bytes of filler, sweep N →
**vision reach in patches per arm**, the number the whole design hinges on
(V2 is expected blind past ~22 patches; without this sweep P2 cannot detect
it and P4 would waste a week). Default hypothesis: Sv=101 @ stride 32.

**P3 — bridge ablation.** Arms C1–C6 (§6, C6 = inner-abs control). Metrics: P2 needle + captioning
bits/byte + wall-clock. Gate: C4 over C1 by the §6 margins or the ship
takes C1.

**P4 — bake-off.** Survivors only; identical data/schedule/token budget;
matched params + reported FLOPs. Data: COCO-320 captions + puzzle suite.
Metrics: caption loss, needle curve, sudoku cell accuracy, COCO CIDEr,
text bits/byte (forgetting check). Ship rule: best primary metric without
forgetting regression.

**P5 — ship and scale.** Freeze winner → 12-layer / higher-res variant →
640px trigger from P2's curve → dense heads (new matcher; seg skips at base
grid) → depth auxiliary → JEPA/Architecture-B research tracks.

**Cross-cutting:** pre-register every gate before running; record negative
results in `decisions.md` (a kill is a successful outcome); correct the
complexity story under every winner; each phase writes its own run directory
under `results/` and leaves it.

---

## 14. References

**CliffordNet family**

- **[Ji 2026]** Ji, S. (2026). CliffordNet: All You Need is Geometric Algebra. arXiv:2601.06793v2.
- **[Markou 2026]** Markou, N. (2026). Extending CliffordNet to Language Modeling and Image Denoising: A Keras 3 Implementation and Novel Applications.

**Clifford / geometric algebra**

- **[Hestenes & Sobczyk 1984]** Hestenes, D., Sobczyk, G. (1984). *Clifford Algebra to Geometric Calculus*. Springer.
- **[Brandstetter 2023]** Brandstetter, J., et al. (2023). Geometric Clifford Algebra Networks. ICML. arXiv:2302.06594.

**Vision-language models**

- **[Radford 2021]** Radford, A., et al. (2021). Learning Transferable Visual Representations from Natural Language Supervision (CLIP). ICML. arXiv:2103.00020.
- **[Liu 2023]** Liu, H., et al. (2023). Visual Instruction Tuning (LLaVA). NeurIPS. arXiv:2304.08485.
- **[Li 2023]** Li, J., et al. (2023). BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. ICML. arXiv:2301.12597.
- **[Zhang 2026]** Zhang, B., et al. (2026). Penguin-VL: Exploring the Efficiency Limits of VLM with LLM-based Vision Encoders. arXiv:2603.06569v2.
- **[Zhai 2023]** Zhai, X., et al. (2023). Sigmoid Loss for Language Image Pre-Training (SigLIP). arXiv:2303.15343.
- **[Allal 2025]** Allal, L.B., et al. (2025). SmolVLM: Redefining Small and Efficient Multimodal Models. arXiv:2504.05299.

**Byte-level models**

- **[Clark 2022]** Clark, J.H., et al. (2022). CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation. TACL.
- **[Pagnoni 2024]** Pagnoni, A., et al. (2024). Byte Latent Transformer: Patches Scale Better Than Tokens. arXiv:2412.09871. (Deferred alternative.)
- **[Hwang 2025]** Hwang, C., et al. (2025). H-Net: Dynamic Chunking for End-to-end Hierarchical Sequence Modeling. arXiv:2507.07955. (Deferred alternative.)

**JEPA family**

- **[LeCun 2022]** LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. Meta AI Technical Report.
- **[Assran 2023]** Assran, M., et al. (2023). Self-supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA). CVPR. arXiv:2301.08243.
- **[Chen 2025]** Chen, Y., et al. (2025). VL-JEPA: Joint Embedding Predictive Architecture for Vision-language. arXiv:2512.10942.
- **[Tong 2024]** Tong, Z., et al. (2024). Connecting Joint-Embedding Predictive Architecture with Contrastive Self-supervised Learning (C-JEPA). arXiv:2410.19560.

**Depth**

- **[Eigen 2014]** Eigen, D., Puhrsch, C., Fergus, R. (2014). Depth Map Prediction from a Single Image using a Multi-Scale Deep Network. NeurIPS. arXiv:1406.2283.
- **[Yang 2024b]** Yang, L., et al. (2024). Depth Anything V2. arXiv:2406.09414.

**Architecture patterns**

- **[Dosovitskiy 2021]** Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words (ViT). ICLR. arXiv:2010.11929.
- **[Yu 2022]** Yu, W., et al. (2022). MetaFormer is Actually What You Need for Vision. CVPR.

---

## Appendix A: Pseudocode

```python
class CliffordVLM(keras.Model):
    """One operator family on the NanoVLM skeleton (V0/V1 = swaps on NanoVLM)."""

    def __init__(self, vision_config, bridge_config, decoder_config):
        self.vision_body = CliffordVisionBody(**vision_config)  # (B,400,128) @320/p16 + pos add
        self.patch_embed = BytePatchEmbed(patch_size=4, dim=384)  # table 256×D
        self.bridge = CliffordBridge(**bridge_config)            # V0: DenseMLP (unimodal projector)
        self.decoder = CausalDecoder(**decoder_config)           # Gemma3 (needs split) or CliffordNetLM (needs split) or TextDecoder (has split)
        self.byte_head = ByteLMHead(patch_size=4, dim=384)       # concat(h_{i-1}, in-frame bytes) + local causal conv → 256-way (never repeat h_i)

    def call(self, inputs):
        image = inputs["image"]            # (B,320,320,3)
        patches = inputs["byte_patches"]   # (B,P,4) int32, PAD=0
        vision_map = self.vision_body(image)                 # (B,400,128)
        vision_tokens = self.bridge(vision_map)              # (B,Sv,384) — vision-only projector
        text_embeds = self.patch_embed(patches)              # (B,P,384)
        joint = concat([vision_tokens, text_embeds], axis=1)  # (B,Sv+P,384)
        states = self.decoder.decode_embeddings(joint)        # Gemma3/CliffordNetLM need this split; TextDecoder has it
        return self.byte_head(states, patches)                # (B,Sv+P·k,256) via h_{i-1} ⊕ bytes


class CliffordBridge(keras.layers.Layer):
    """Unimodal vision projector: filter at full resolution; compress last."""

    def __init__(self, vision_dim, lm_dim, num_blocks=1,
                 shifts=[1, 2, 4], cli_mode="full", ctx_mode="diff"):
        self.dim_align = keras.layers.Dense(lm_dim)      # 128 → 384
        self.blocks = [CliffordNetBlock(lm_dim, shifts, cli_mode, ctx_mode,
                                        input_mode="sequence")
                       for _ in range(num_blocks)]        # @400 tokens, ±2 neighborhood/block
        self.cls_token = keras.layers.Embedding(1, lm_dim)
        self.compress = ModalityProjection(lm_dim, lm_dim,
                                           scale_factor=2)  # 401 → 101
        # C1 arm: replace blocks with Dense-GELU-Dense.
        # C5: dim_align + compress only.
        # C6: cli_mode="inner", ctx_mode="abs" (wedge control).

    def call(self, vision_map):
        # vision_map (B,400,128) → align → (B,400,384) → Clifford blocks @400
        # → prepend CLS → (B,401,384) → PixelShuffle+Dense+GELU+LN → (B,101,384)
        b = keras.ops.shape(vision_map)[0]
        z = self.dim_align(vision_map)
        for block in self.blocks:
            z = z + block(z)     # external residual, unconditional
        cls = keras.ops.repeat(self.cls_token(keras.ops.zeros((b, 1), "int32")),
                               1, axis=0)
        return self.compress(concat([cls, z], axis=1))
```

## Appendix B: The Honest Attention Comparison

| Aspect | Cross-attention bridge+decoder | Clifford bridge+decoder |
|---|---|---|
| Cross-modal params @ D=384 | Q,K,V,O = 589,824 | SRGP(\|S\|=3) = 885,120 — **denser** |
| Sequence scaling | O(N²) time, KV cache grows | O(N) time, no KV cache |
| Interaction channels | scalar similarity | inner (alignment) + wedge (antisymmetric difference) |
| Position-specific retrieval | yes | no (local window + rank-1 global) — measured §7 |
| Algebraic status of SRGP | — | **not** associative / invertible / closed (measured max\|non-assoc\| = 3.74); only `u^u=0` transfers |

The Clifford bet is expressiveness-per-FLOP at long sequences plus the wedge
channel — not parameter efficiency, and not algebraic completeness. P1 and P3
price both claims.
