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

This document is the build plan for a **Clifford-native Vision-Language Model**:
one operator family — the CliffordNet block built on the sparse rolling
geometric product (SRGP) plus gated geometric residual (GGR) — for the vision
tower, the vision-to-language bridge, and (in two of three arms) the decoder.
Text is raw UTF-8 bytes in fixed patches of k=4; there is no tokenizer.

The thesis: the geometric product decomposes interaction into an **inner**
channel (scalar alignment) and a **wedge** channel (antisymmetric structural
difference, with `u ^ u = 0` exactly — verified numerically). Standard
dot-product attention computes only the first. Whether the second channel
earns its keep cross-modally is an empirical question, and this plan tests it
directly instead of asserting it.

Three architectures go to experiment, plus one baseline:

| Arm | Vision | Bridge | Decoder |
|---|---|---|---|
| **V0** (baseline) | CliffordNet | Dense MLP | Gemma3 mixed attention |
| **V1** | CliffordNet | SRGP + GGR | Gemma3 mixed attention |
| **V2** | CliffordNet | SRGP + GGR | CliffordNetLM, uniform K=3 |
| **V3** | CliffordNet | SRGP + GGR | CliffordNetLM, variable-K |

V0 isolates the question "does Clifford earn anything, or is this just a
decent small VLM". V1 tests the bridge thesis with a conventional decoder. V2
and V3 test full attention replacement; V3 exists because measured receptive
field says uniform-K stacks cannot use a vision prefix (see §7).

They run as a cascade (§13): a text-only bake-off gates the replacement
thesis, a visual needle probe sets the vision-token budget, a bridge ablation
gates §1's thesis, and only survivors reach the full bake-off.

---

## 2. Components Inventory

### 2.1 What exists and is used as-is

| Component | Location | Notes |
|---|---|---|
| `SparseRollingGeometricProduct` | `layers/geometric/clifford_block.py` | `channels`, `shifts` (required, no defaults); `cli_mode ∈ {inner, wedge, full}`; `dot_activation` (default SiLU) applies to the **inner term only**; no `ctx_mode` on this layer |
| `GatedGeometricResidual` | same | Returns **only** the gamma-scaled term `γ ⊙ (SiLU(h) + σ(gate([h;g])) ⊙ g)`; residual add + `StochasticDepth` are external and unconditional. `gamma` is per-channel, init 1e-5 |
| `CliffordNetBlock` / `CausalCliffordNetBlock` | same | Two stacked depthwise convs (K=3 → 5-wide window per block); `ctx_mode="diff"` default; sequence mode accepts rank-3 and rank-4 |
| `CliffordNet` (vision) | `models/vision/cliffordnet/model.py` | **Isotropic**: one stem, one width, no stages. Variants `nano/lite/lite_g` (all 128-wide, depth 12) |
| `CliffordNetLM` | `models/vision/cliffordnet/lm.py` | Causal LM, learned token + position tables, tied logits default. Param is `max_seq_length` |
| `Gemma3TransformerBlock` + `Gemma3` | `layers/transformers/gemma3_transformer.py`, `models/language/gemma/gemma3.py` | Per-block `attention_type ∈ {sliding_window, full_attention}`; causal+band masks built internally and OR-ed, rank-3; padding mask AND-ed (packing-safe); no QK-norm; `rope_theta` not exposed |
| `ModalityProjection` | `layers/modality_projection.py` | `PixelShuffle(scale_factor)` + `Dense` + GELU + LN. Rank-3 only, `(B, 1+H·W, C)`; **CLS at index 0 mandatory**; spatial count must be a perfect square divisible by `scale_factor`. Kwarg is `scale_factor`, not `s` |
| `NanoVLMLoss` + `text_position_logits` | `losses/nano_vlm_loss.py` | Slices the trailing `seq_len_true` positions, then a 1-step shift; `ignore_index=0` default; requires rank-2 targets — byte targets must be flattened to `(B, P·k)` first |
| `byte_patch.py` / `byte_lm.py` | `datasets/byte_patch.py`, `datasets/byte_lm.py` | `bytes_to_patches(k=4)`, exact round-trip, `PAD_BYTE_ID=0`, `pack_byte_patch_windows`, packing/estimate helpers. Zero consumers — reuse, don't re-derive |
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
wedge_c^(s) = z_det,c · z_ctx,(c−s) − z_ctx,c · z_det,(c−s)   [no activation]
inner_c^(s) = SiLU(z_det,c · z_ctx,(c−s))                    [activation here only]
full: concat [wedge_s1, inner_s1, wedge_s2, inner_s2, …] → Dense(D)
```

Width before projection is `2·|S|·D`; the projection is what makes SRGP cost
`2·|S|·D²` parameters (measured §11). Shifts `≥ channels` are dropped with a
warning; all dropped raises.

### 2.3 What must be built (the complete new-code list)

1. **Reusable Clifford vision body** — stage-map (here: single-map) return.
   `CliffordNet.call()` inlines its loop over a `List[Dict]`; there is no
   `_apply_vision_body` and no `return_stages`. Every variant needs this
   refactor first.
2. **Byte patch-embed layer** (~50 lines) — 256-entry table → concat-k →
   `Dense` + LN (§8).
3. **Byte output head** — scatter-repeat ×k + causal local head + 256-way
   `Dense`, with the §8 masking rule. Gated by the RED-first leak probe.
4. **`context_kernel_size` plumbing in `CliffordNetLM`** — `lm.py`
   hardcodes K=3 for all blocks; V3 needs per-block K. ~5 lines.
5. **Inner/wedge norm taps in `SRGP`** — `call()` discards `components`;
   all geometric health metrics are unobservable until this lands.
6. **Scalar-metric sink** — `utils/logger.py` exports no functions; route
   Clifford scalars through `StepCheckpointCallback`, not a new `log_scalar`.
7. **Multi-class Hungarian matcher** — `_Sam3HungarianMatcher`
   (`losses/sam3_detection_loss.py`) is private by decision, binary-only,
   and bound to the SAM-3 packed layout. A closed-80 DETR head needs its own
   `(Q,C)×(N,C)` cost written from scratch. Detection is a P5 item, not P0.

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
[Bridge: CLS prepend + PixelShuffle(s=2) + Dense + 1–3 × CliffordNetBlock (sequence)]
     |  (V0: same compression, Dense-GELU-Dense instead of blocks)
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

Default: **patch 16 → 400 tokens**. Two prefix options, both verified:

| Prefix | How | Tokens | Stride | Use |
|---|---|---|---|---|
| full | direct | 400 | 16px | P2 large-prefix arm |
| default | `ModalityProjection(128→384, scale_factor=2)` on (CLS + 400) | **101** | 32px | ship default |

101 was verified end-to-end (401 in → 101 out). Stride 32 ≈ one 9×9 sudoku
cell per token at 320px — the granularity §9's puzzle suite needs, and the
scale P2's needle probe will confirm or reject. The hierarchical
CliffordCLIP tower is *not* the VLM backbone (its 10×10 S4 map forces the
25-token prefix that cannot read a sudoku cell); its role is pre-training
initialization only (§9, Option 1).

Resolution changes need nothing but the stem grid: no hierarchy, no
positional table to interpolate on the vision side.

---

## 6. The Bridge

The bridge is a learned-CLS prepend, `PixelShuffle(scale_factor=2)` (400 →
101 tokens, verified), a `Dense` dim-align (blocks are isotropic — channel
mismatch raises), then 1–3 `CliffordNetBlock`s in sequence mode (rank-3
verified). The SRGP inside these blocks *is* the cross-modal geometric operator: inner
for vision-text alignment, wedge for antisymmetric structural difference.
One block at D=384 costs 1.33M params (measured; 885k of it the SRGP
projection).

Five arms, fixed tower and decoder, bridge swapped:

| Arm | Bridge |
|---|---|
| C1 | Dense-GELU-Dense (LLaVA-style; also V0's bridge) |
| C2 | SRGP + GGR, `cli_mode="inner"` |
| C3 | SRGP + GGR, `cli_mode="wedge"` — **the paper's novelty** |
| C4 | SRGP + GGR, `cli_mode="full"` |
| C5 | dim-align only — floor |

If `wedge` alone doesn't beat `inner`, the thesis is rewritten, not the
experiment. Ship gate (§13, P3): C4 beats C1 by ≥2% relative needle accuracy
and ≥0.01 bits/byte on captioning with no throughput regression; otherwise
the ship takes C1.

---

## 7. Decoders

**V0/V1: package-owned Gemma3 stack.** `TextDecoder` stays untouched (legacy
GPT-2 defaults, no KV-head passthrough). Per-block `attention_type`, 5×
sliding + 1× full, GQA with real KV-head control, RoPE positions (no learned
table to resize). Config: 384/6L/6Q-2KV, w=32, GeGLU-1152, `max_seq_len=1024`
(measured 10.5M params; the 1024 cap covers every P2 prefix outcome —
a 512 cap hard-fails the joint sequence with `SequenceLengthError`).

**V2/V3: `CliffordNetLM`.** Causal blocks, left-pad convs, causal cumulative
mean global branch, learned positions, tied byte logits. The deciding
measurement — look-back reach on random init (perturbation probe, relative
threshold, layers built once):

| Decoder | Nominal reach | **Measured** |
|---|---|---|
| uniform K=3, depth 6–24 | 4L | **~22 (depth buys nothing)** |
| K = 3,5,9,17 ×3 | 180 | **~94** |
| K = 3,7,15,31 ×3 | 360 | **~173** |
| + cumulative-mean branch | global | global, but **rank-1** |

So V2's usable vision prefix is ~22 tokens and V3's is ~170: V3 exists
because varying `context_kernel_size` (per-block reach `2(K−1)`) is a 4–8×
lever where depth is none. Re-check on trained weights in P1 — random-init RF
is a smoke signal, not a verdict. Position-specific retrieval ("which
patch?") is unanswerable in principle under a local window + rank-1 mean;
that is the accepted ceiling of the replacement track, and grounding-heavy
tasks are where V1 is expected to win.

Measured decoder sizes (vocab 256): D=384/L=12 → 16.3M; D=384/L=18 →
24.3M; D=256/L=18 → 10.9M. P1 matches params across arms (adjust depth) and
reports FLOPs/token separately.

---

## 8. Byte Text Representation

No BPE, no tokenizer on the VLM path. Raw UTF-8 → fixed patches of k=4
(CANINE precedent [[Clark 2022]](#14-references)).

**Encoding** (`datasets/byte_patch.py`, reuse): pad byte-length to a multiple
of 4, row-major reshape. `0x00` is valid UTF-8 (U+0000) but absent from
cleaned text corpora, so it serves as pad; `0xFF` is never valid UTF-8 alone
and serves as EOS (in-tree convention: `BOS_ID=254`, `EOS_ID=255` are
conventions, not mechanisms).

**Length math** (measured 3.20 bytes/GPT-2-token on repo prose, not ~4):
512 BPE tokens ≈ 1,639 bytes ≈ **410 patches**. Text context is 512 patches
(≈640 BPE tokens); joint budget is Sv + 512 ≤ 1024.

**Patch embedding** (~50 lines, new): byte table 256×D (98,304 params at
D=384) → concat last two axes → `Dense(D)` + LN. The omitted patch-embed
`Dense` costs 590,592 params — it is in the budget (§11).

**Byte head** (new — the one subtle component). The global decoder runs at
patch resolution; only the head touches byte resolution:

```
patch states (B, P, D) → scatter-repeat ×k → (B, P·k, D) byte-frame states
 → tiny local causal head → Dense(256) → next-byte logits
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
byte → logits move. Plus: `ignore_index=0` pad-mask exactness, per-position
CE vs a naive per-byte loop oracle on 8-byte inputs, `.keras` value
round-trip. No training run counts until (a)+(b) pass and (c) moves.

**Loss / generate / decode.** Flatten byte targets to `(B, P·k)` (rank-3
raises in `text_position_logits`); standard next-byte CE over trailing text
positions, `ignore_index=0` (= `PAD_BYTE_ID`; do not use
`make_lm_labels`' negative-ignore convention on this path). `generate()`
emits bytes autoregressively, stops at `0xFF`, UTF-8-decodes with replacement
at the end. Open items specified here, not later: partial final frames (1–3
bytes present) and the frame-0 boundary `h_{-1}`.

---

## 9. Training

### 9.1 Data pipeline (English-only; bytes remove the tokenizer, not the language decision)

Packing via `datasets/byte_lm.py`; mandatory wherever caption batches would
leave the 512-patch tail untrained. COCO/CC3M stay CLIP-contrastive-only.

| Stage | Dataset (HF id) | Size | Use |
|---|---|---|---|
| 0a (CLIP init) | CC3M / LAION-400M | 3–400M | Contrastive vision-tower init (one option among §9.2) |
| 0b (byte LM, opt) | FineWeb-Edu (bytes) | ~20B bytes | Decoder warmup before vision joins |
| 1 (align, 320px) | `allenai/pixmo-cap`, `nlphuji/flickr30k` | ~0.5M | Bridge warmup, short contexts; encoder at low LR (Penguin: train it, don't freeze it) |
| 2a (instruct, packed) | `liuhaotian/LLaVA-Instruct-150K` | 150k | Core instruction |
| 2b (instruct mix) | `HuggingFaceH4/llava-instruct-mix-vsft` | ~100k | Diversity |
| 2c (multi-task) | `HuggingFaceM4/the_cauldron` | 50–100k of 1.88M | VQAv2/GQA/ChartQA unified |
| 2d (synthetic, §9.3) | in-tree generators | 5–10% of mix | Puzzles; seeded, style-shift eval |
| 3a (text-in-image) | `facebook/textvqa` | 45k | Readable at 320px |
| 3b (diagram math) | `hiyouga/geometry3k` | ~3k | MathVista proxy |
| 3c (docs, DEFERRED) | `HuggingFaceM4/Docmatix` | 2.4M imgs | Gated on the P2 resolution trigger |
| 3d (dense COCO) | COCO-2017 | 118k | Detection + seg heads (P5) |
| 4 (preference, later) | `HuggingFaceH4/rlaif-v_formatted` | 83k | Post-SFT hallucination reduction |

English filter at load; ~10% text-only bytes in Stage 2 (no forgetting).
Square resize/letterbox (Clifford `image_size` constraint).

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

Only observable metrics gate anything. Clifford inner/wedge norms and the
scalar sink land in P0 — until then the per-block thresholds below are aspirations, not
gates.

**Every step (all exist):** train loss (text positions only), perplexity,
per-component LR, grad norm before/after clipping, tokens/sec.

**Clifford health (needs P0 items 0.4–0.5):** `layerscale_gamma` mean/max per
block (access via `block.ggr.gamma.gamma` — `gamma` is a sub-layer, not a
weight); `inner_norm`, `wedge_norm`, `inner_wedge_ratio` per SRGP
(healthy 0.3–3.0; collapse of either side to 0 = dead channel);
`gate_activation` mean (0.3–0.7). If gamma sits at 1e-5 past 30% of training,
the model is a Dense VLM wearing a Clifford badge.

**Probes (the sharp instruments):** byte needle-in-haystack vs distance (P1);
grid-cell needle vs K (P2); plain-vs-styled puzzle gap (§9.3).

**Validation:** held-out caption loss/ppl; text-only ppl (forgetting tripwire
— rising while VLM loss falls means add text data); CIDEr/BLEU-4 on COCO-val;
VQA exact-match on a VQAv2 slice. **Scoped to 320px reality:** DocVQA,
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
| bridge | CLS + PixelShuffle(s=2) + Dense + 1 × Block-384 | **~1.5M** (block 1,332,096 incl. 885,120 SRGP; compress Dense ~0.2M) |
| patch-embed | table 256×384 + concat-4 → Dense | 98,304 + 590,592 |
| byte table | 256×384, tied | 98,304 (assert `== 98,304`) |
| decoder V2/V3 | CliffordNetLM 384/12, ctx 1024 | ~16.5M |
| decoder V0/V1 | Gemma3 384/6, 6Q/2KV, 5:1 w=32 | **10,528,128** |
| detection head (P5) | Q=100, 2–3 cross layers | ~5M (new matcher required, §2.3.7) |
| seg head (P5) | `SegmentationHead` tap | ~1M |

Ship sketch (V2/V3, 1-block bridge): 1.5 + 1.5 + 0.7 + 16.5 ≈ **~20M**.
V1 sketch: 1.5 + 0.5 (Dense bridge) + 10.5 ≈ **~13M**. Movers: decoder depth
12→18 (+8M), 3-block bridge (+2.7M), untied byte head (+98k — still don't).

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
   re-check RF on trained weights.
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
| Cross-modal shift subspaces don't interact | Medium | Medium | Bridge aligns first; LayerScale starts near-identity |
|Instability in joint training | Low | Medium | Frozen-decoder align first; `TerminateOnNaN`; grad clip |

---

## 13. Roadmap

```
P0  shared spine (built once, serves every arm) · ~3–5 days
 ├─> P1  text-only LM bake-off   → keeps or kills replacement · ~2 days
 ├─> P2  visual needle probe     → sets Sv and resolution trigger · ~3 days
 ├─> P3  bridge ablation         → keeps or kills the wedge thesis · ~2 days
 └─> P4  bake-off at scale       → which of V1/V2/V3 ships · ~1–2 weeks
         └─> P5  ship + scale + dense heads + depth/JEPA tracks
```

**P0 — shared spine.** Nothing else starts until 0.1–0.3 land.
0.1 reusable vision body (stage-map return); 0.2 byte head + RED-first leak
probe; 0.3 `context_kernel_size` plumbing; 0.4 SRGP norm taps; 0.5 metric
sink via `StepCheckpointCallback`; 0.6 pre-registered gates + frozen eval
suite. Side effects: fix the (small) spec blockers the audit found as you
touch them; delete nothing under `results/`.

**P1 — text-only bake-off.** Arms A0 (Gemma3 ref) / A1 (uniform K=3) / A2
(variable-K) / A3 (+global branch). Metric: val bits/byte at matched
tokens-seen + byte needle-vs-distance. Gates: ≤0.25 → replacement lives;
0.25–0.50 → proceed with V1 as ship candidate; >0.50 → kill replacement,
V1 ships, Clifford survives as the bridge operator.

**P2 — visual needle probe.** Grid images, "value at (r,c)?", K ∈
{4,8,12,16,20}. Arms B0 (Gemma3) / B1 / B2 / B3 (B0+Dense bridge, isolates
decoder from bridge). Output: max operable prefix per arm → Sv,
`scale_factor`, resolution trigger. Default hypothesis: Sv=101 @ stride 32.

**P3 — bridge ablation.** Arms C1–C5 (§6). Metrics: P2 needle + captioning
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
    """One operator family: Clifford blocks for vision, bridge, decoder."""

    def __init__(self, vision_config, bridge_config, decoder_config):
        self.vision_body = CliffordVisionBody(**vision_config)  # (B,400,128) @320/p16
        self.patch_embed = BytePatchEmbed(patch_size=4, dim=384)  # table 256×D
        self.bridge = CliffordBridge(**bridge_config)            # V0: DenseMLP
        self.decoder = CausalDecoder(**decoder_config)           # Gemma3 or CliffordNetLM
        self.byte_head = ByteLMHead(patch_size=4, dim=384)       # 256-way

    def call(self, inputs):
        image = inputs["image"]            # (B,320,320,3)
        patches = inputs["byte_patches"]   # (B,P,4) int32, PAD=0
        vision_map = self.vision_body(image)                 # (B,400,128)
        vision_tokens = self.bridge(vision_map)              # (B,Sv,384)
        text_embeds = self.patch_embed(patches)              # (B,P,384)
        joint = concat([vision_tokens, text_embeds], axis=1)  # (B,Sv+P,384)
        states = self.decoder(joint)                          # (B,Sv+P,384)
        return self.byte_head(states, patches)                # (B,Sv+P·k,256)


class CliffordBridge(keras.layers.Layer):
    """Compress, align, then geometric fusion (the §1 thesis)."""

    def __init__(self, vision_dim, lm_dim, num_blocks=1,
                 shifts=[1, 2, 4], cli_mode="full"):
        self.cls_token = keras.layers.Embedding(1, vision_dim)
        self.compress = ModalityProjection(vision_dim, lm_dim,
                                           scale_factor=2)  # 401 → 101
        self.blocks = [CliffordNetBlock(lm_dim, shifts, cli_mode,
                                        input_mode="sequence")
                       for _ in range(num_blocks)]
        # C1 arm: replace blocks with Dense-GELU-Dense. C5: compress only.

    def call(self, vision_map):
        # vision_map (B,400,128) → prepend CLS → (B,401,128)
        # → PixelShuffle+Dense+GELU+LN → (B,101,384) → Clifford blocks
        b = keras.ops.shape(vision_map)[0]
        cls = keras.ops.repeat(self.cls_token(keras.ops.zeros((b, 1), "int32")),
                               1, axis=0)
        z = self.compress(concat([cls, vision_map], axis=1))
        for block in self.blocks:
            z = z + block(z)     # external residual, unconditional
        return z
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
