# CliffordVLM: A Geometric Algebra Vision-Language Model

**Author**: Nikolas Markou | **Date**: 2026-04-17 | **Status**: Research & Architecture Design

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Existing Components Inventory](#2-existing-components-inventory)
3. [VLM Architecture Landscape](#3-vlm-architecture-landscape)
4. [Proposed CliffordVLM Architecture](#4-proposed-cliffordvlm-architecture)
5. [Vision Tower: CliffordNet Vision Encoder](#5-vision-tower-cliffordnet-vision-encoder)
6. [Language Tower: CliffordNet Causal Decoder](#6-language-tower-cliffordnet-causal-decoder)
7. [The Bridge: Clifford Geometric Fusion](#7-the-bridge-clifford-geometric-fusion)
8. [Training Strategy](#8-training-strategy)
9. [Training Metrics & Monitoring](#9-training-metrics--monitoring)
10. [Model Variants & Scaling](#10-model-variants--scaling)
11. [Open Questions & Risks](#11-open-questions--risks)
12. [Implementation Roadmap](#12-implementation-roadmap)
13. [References](#13-references)
- [Appendix A: CliffordVLM Pseudocode](#appendix-a-cliffordvlm-pseudocode)
- [Appendix B: Why Not Just Use Cross-Attention?](#appendix-b-why-not-just-use-cross-attention)

---

## 1. Executive Summary

This document investigates how to build a **Clifford-native Vision-Language Model (VLM)** using the three existing CliffordNet components:

- **CliffordNet** (vision backbone) -- isotropic columnar architecture using bidirectional CliffordNetBlocks
- **CliffordNetLM** (causal language model) -- autoregressive decoder using CausalCliffordNetBlocks
- **CliffordCLIP** (contrastive vision-language) -- dual-encoder with Clifford-aware projection heads

The key thesis is that the geometric product -- decomposing into inner (scalar coherence) and wedge (bivector structural variation) -- provides a **richer cross-modal interaction** than standard dot-product attention. Where attention computes `softmax(QK^T/sqrt(d))V`, the Clifford geometric product simultaneously captures alignment (inner product) and structural difference (wedge product) between modalities. This enables the model to reason about both *what matches* and *how things differ* between vision and language.

### Why This Matters

Standard VLMs fuse vision and language through dot-product cross-attention, which captures only scalar similarity. The wedge product adds an antisymmetric "structural" channel that:

1. **Detects misalignment**: `u ^ u = 0` -- when vision and text representations are aligned, the wedge vanishes; non-zero wedge signals structural disagreement
2. **Preserves orientation**: The wedge product captures the *direction* of disagreement, not just its magnitude
3. **Is algebraically complete**: The geometric product `uv = u.v + u^v` is the only bilinear product on vectors that is both invertible and associative [[Hestenes & Sobczyk 1984]](#13-references)

### What We Already Have vs. What We Need

| Component | Status | Location |
|-----------|--------|----------|
| Vision backbone (CliffordNetBlock) | Done | `layers/geometric/clifford_block.py` |
| Causal LM backbone (CausalCliffordNetBlock) | Done | `layers/geometric/clifford_block.py` |
| Vision classifier (CliffordNet) | Done | `models/cliffordnet/model.py` |
| Causal LM (CliffordNetLM) | Done | `models/cliffordnet/lm.py` |
| Contrastive model (CliffordCLIP) | Done | `models/cliffordnet/clip.py` |
| Contrastive loss (CLIPContrastiveLoss) | Done | `losses/clip_contrastive_loss.py` |
| Multimodal fusion layer | Done | `layers/fusion/multimodal_fusion.py` |
| VLM task heads | Done | `layers/vlm_heads/factory.py` |
| **CliffordVLM (generative VLM)** | **Needed** | `models/cliffordnet/vlm.py` |
| **Clifford cross-modal fusion** | **Needed** | New layer or extension |
| **Vision-to-LM projector** | **Needed** | Lightweight adapter |
| **Clifford depth decoder** | **Needed** | CliffordNetBlock stack + 1x1 Conv head |
| **Training pipeline** | **Needed** | `train/cliffordnet/train_vlm.py` |

---

## 2. Existing Components Inventory

### 2.1 Clifford Algebra Primitives

All primitives live in `src/dl_techniques/layers/geometric/clifford_block.py`.

**SparseRollingGeometricProduct (SRGP)** -- the core mathematical operation:
```
For vectors u, v in R^D, the Clifford product in Cl(D, 0):

    uv = u . v   +   u ^ v
         -----       -----
         inner        wedge
       (grade 0)    (grade 2)

Sparse approximation via cyclic shifts s in S = {1, 2, 4, 8, 16}:

    D_c^(s) = SiLU(Z_det,c * Z_ctx,(c+s) mod D)           [inner/dot]
    W_c^(s) = Z_det,c * Z_ctx,(c+s) - Z_ctx,c * Z_det,(c+s)  [wedge]
```

The SRGP [[Ji 2026]](#13-references) avoids the O(D^2) full channel interaction (cf. [[Brandstetter 2023]](#13-references)) by sampling O(|S| * D) diagonal interactions. With S = {1,2,4,8,16}, this gives O(5D) operations -- comparable to a depthwise convolution.

**Modes**: `"inner"` (scalar only), `"wedge"` (bivector only), `"full"` (both). The `"full"` mode concatenates inner + wedge components and projects back to D dimensions.

**GatedGeometricResidual (GGR)** -- Euler-discretized ODE update:
```
alpha = sigmoid(W_gate[X_norm; G_feat])    [gating]
H_mix = SiLU(X_norm) + alpha * G_feat     [mix]
X_out = X_prev + DropPath(gamma * H_mix)  [residual with LayerScale]
```

LayerScale gamma initialized to 1e-5 ensures blocks start as near-identity, enabling stable training of deep stacks (20+ layers).

### 2.2 CliffordNet Vision Backbone

`src/dl_techniques/models/cliffordnet/model.py`

Isotropic columnar architecture -- constant spatial resolution throughout:
```
Image (B,H,W,3) -> Patch Stem -> BN -> L x CliffordNetBlock -> GAP -> LN -> Dense
```

Key properties:
- **No attention, no FFN**: The geometric product replaces both (the MetaFormer [[Yu 2022]](#13-references) two-stage decomposition)
- **O(N) complexity**: Depthwise convolutions + channel shifts, not O(N^2) attention
- **Isotropic**: Same resolution/channels at every layer (no hierarchical downsampling)
- **Dual-stream**: Detail (linear) + Context (DWConv) streams interact via SRGP
- **Global context branch**: Optional GAP-based whole-image summary with hardcoded shifts=[1,2]

Proven results: CliffordNet-Lite (2.6M params) beats ResNet-18 (11.2M) on CIFAR-100 [[Ji 2026]](#13-references).

### 2.3 CliffordNetLM Causal Decoder

`src/dl_techniques/models/cliffordnet/lm.py`

Autoregressive language model using causal Clifford blocks:
```
Token IDs (B,T) -> Embedding + PosEmbed -> LN -> Reshape(B,1,T,D) ->
L x CausalCliffordNetBlock -> Reshape(B,T,D) -> LN -> Dense(vocab) -> Logits
```

Three modifications for causality:
1. **Causal depthwise conv**: Left-only zero-padding before `padding="valid"` convolutions
2. **Causal cumulative mean**: `C_glo[i] = (1/(i+1)) * sum(X[0..i])` replaces global average pool
3. **4D reshape**: Sequences reshaped to `(B,1,T,D)` to reuse 2D convolution infrastructure

Causality verified via unit tests: changing token at position j does not affect logits at positions < j.

### 2.4 CliffordCLIP Contrastive Model

`src/dl_techniques/models/cliffordnet/clip.py`

Dual-encoder CLIP with Clifford-aware projection heads:
```
Vision: CliffordNetBlock x L_vis -> GAP(z_det) + LearnedQueryPool(z_ctx) -> SRGP -> LayerScale residual
Text:   CausalCliffordNetBlock x L_txt -> masked-mean(z_det) + LearnedQueryPool(z_ctx) -> SRGP -> LayerScale residual
```

The key innovation is the **learned_query_residual** head:
```
z_det = canonical anchor (GAP for vision, last-non-pad for text)
z_ctx = learned single-query attention pool over feature sequence
geo   = SparseRollingGeometricProduct(z_det, z_ctx)
mixed = z_det + gamma * geo    (gamma init 1e-5)
```

This head is the empirical winner on CC3M-smoke [[Markou 2026]](#13-references) -- it starts as plain CLIP (gamma ~ 0) and learns to inject Clifford geometric content only where it reduces contrastive loss.

### 2.5 Existing Depth Estimation Infrastructure

The codebase already has production-ready depth estimation components:

- **DepthAnything** (`models/depth_anything/model.py`): Complete monocular depth estimation model [[Yang 2024]](#13-references) with DINOv2 [[Oquab 2024]](#13-references) encoder + DPT decoder. Supports vit_s/vit_b/vit_l encoder sizes.
- **DPTDecoder** (`models/depth_anything/components.py`): Dense Prediction Transformer decoder for multi-scale depth map upsampling.
- **AffineInvariantLoss** (`losses/affine_invariant_loss.py`): Scale-and-shift-invariant loss using median normalization and MAD scaling. Critical for multi-dataset depth training where scale is ambiguous.
- **FeatureAlignmentLoss** (`losses/feature_alignment_loss.py`): Cosine similarity-based distillation loss for transferring semantic priors from a frozen teacher encoder.
- **Miyasawa conditional depth theory** (`research/miyasawas_theorem_conditional_depth.md`): Theoretical framework for conditional denoising applied to RGB-to-depth estimation.
- **CliffordNetDenoiser** (`models/cliffordnet/denoiser.py`): Bias-free Clifford blocks satisfying Miyasawa's theorem -- directly applicable to depth refinement (see Section 3.4).

### 2.6 Existing VLM Infrastructure

The codebase already has:
- **NanoVLM** (`models/nano_vlm/model.py`): VisionEncoder + TextDecoder + MultiModalFusion -- configurable fusion strategies
- **CLIP** (`models/clip/model.py`): Standard ViT [[Dosovitskiy 2021]](#13-references)-based dual encoder [[Radford 2021]](#13-references)
- **MultiModalFusion** (`layers/fusion/multimodal_fusion.py`): 8 fusion strategies (cross_attention, concatenation, gated, bilinear, tensor_fusion, etc.)
- **VLM task heads** (`layers/vlm_heads/factory.py`): Captioning, VQA, grounding, matching heads
- **TextDecoder** (`layers/transformers/text_decoder.py`): Transformer-based autoregressive decoder
- **VisionEncoder** (`layers/transformers/vision_encoder.py`): Configurable ViT-style encoder

---

## 3. VLM Architecture Landscape

### 3.1 Dominant Paradigms (2024-2026)

Modern VLMs have converged on a three-component architecture:

```
Image -> [Vision Encoder] -> vision tokens -> [Projector/Bridge] -> [Language Model] <- text tokens
```

| Paradigm | Examples | Vision Encoder | Bridge | LM |
|----------|----------|----------------|--------|----|
| Frozen encoder + adapter | LLaVA [[Liu 2023]](#13-references), BLIP-2 [[Li 2023]](#13-references) | Frozen CLIP ViT | MLP / Q-Former | Frozen LLM |
| End-to-end trained | Qwen-VL, InternVL | Trainable ViT | Cross-attention | Trainable LLM |
| Unified architecture | Fuyu, Chameleon | Native pixel input | None | Single decoder |
| LLM-initialized encoder | Penguin-VL [[Zhang 2026]](#13-references) | LLM-initialized (bidirectional) | 2-layer MLP | LLM decoder |

### 3.2 Lessons from Penguin-VL: Contrastive vs Generative Initialization

Penguin-VL [[Zhang 2026]](#13-references) makes a strong case **against** contrastive pre-training (CLIP [[Radford 2021]](#13-references)/SigLIP [[Zhai 2023]](#13-references)) for VLM vision encoders, arguing that contrastive objectives are fundamentally misaligned with the generative (next-token prediction) objective of VLMs:

**Penguin-VL's four arguments against contrastive initialization**:
1. **Objective mismatch**: Contrastive learning optimizes for *discrimination* (is this image-text pair matched?), which "enforces coarse and category-level invariances that suppress fine-grained visual cues needed for dense captioning and complex VLM reasoning"
2. **Representation collapse**: Contrastive encoders compress images to global summary tokens, losing spatial detail needed for grounding, counting, OCR
3. **Architecture lock-in**: CLIP/SigLIP enforce ViT-specific inductive biases that may not be optimal for VLM use
4. **Scalability ceiling**: SigLIP2 required >40B training samples to saturate, while Penguin's LLM-initialized encoder matched/exceeded it with ~240M samples -- a 150x data efficiency improvement

**Penguin-VL's alternative -- LLM-initialized vision encoder**:
- Initialize the vision encoder from an LLM (Qwen3-0.6B, ~400M params)
- Modify causal self-attention to bidirectional for non-causal visual processing
- Add 2D-RoPE positional embeddings for variable-resolution images
- Train with reconstruction losses (amplitude, direction, relation) from a frozen teacher (SigLIP)
- The encoder starts in a representation space inherently compatible with the downstream LLM

**Implications for CliffordVLM**:

CliffordVLM has a unique position in this debate. The CliffordNetBlock already avoids the "contrastive vs generative" dichotomy because:
- It uses **neither attention nor contrastive loss** for its core operation
- The geometric product captures both alignment (inner) and structural difference (wedge), which is richer than either contrastive or generative objectives alone
- CliffordCLIP pre-training does use contrastive loss, but the **Clifford-aware projection head** preserves fine-grained geometric structure that standard CLIP heads discard

This motivates two additional pre-training options (see Section 5.3, Options 4-5): initialize the CliffordNet vision encoder with reconstruction-based supervision (Penguin-style, Option 4) or I-JEPA self-supervised prediction (Option 5) rather than contrastive supervision.

### 3.3 Lessons from JEPA: Predict Meaning, Not Pixels

The Joint Embedding Predictive Architecture (JEPA) family -- spanning I-JEPA [[Assran 2023]](#13-references), V-JEPA [[Bardes 2024]](#13-references), V-JEPA 2 [[Assran 2025]](#13-references), and VL-JEPA [[Chen 2025]](#13-references) -- offers a third paradigm originating from [[LeCun 2022]](#13-references) beyond contrastive and generative that is deeply relevant to CliffordVLM.

**JEPA's core insight**: Predicting the input is the wrong objective. Pixel-level or token-level reconstruction wastes capacity on unpredictable high-entropy detail (flickering leaves, exact surface texture). Instead, predict in *embedding space* -- a lossy compression that retains only the structure needed to reduce uncertainty about what comes next.

```
Generative (MAE/GPT): Input -> Encoder -> Latent -> Decoder -> Reconstructed Pixels/Tokens
                                                                    |
                                                              Pixel-level loss (must predict noise)

JEPA:                  Input_x -> Encoder -> s_x -> Predictor -> s_hat_y
                       Input_y -> Encoder -> s_y <--- L2 loss ---> s_hat_y
                                                Embedding-level loss (only semantic content)
```

**Why JEPA matters for CliffordVLM -- three connections**:

**Connection 1: The Geometric Product as a JEPA Predictor**

The SRGP (SparseRollingGeometricProduct) is structurally analogous to a JEPA predictor. Both take a context representation and produce a prediction about a target representation:

```
JEPA predictor:      s_hat_y = g_phi(s_x, z)              [predict target from context + position]
CliffordNet SRGP:    geo     = SRGP(z_det, z_ctx)         [predict interaction from detail + context]
```

The JEPA predictor g_phi is typically a narrow transformer; the SRGP is a shift-based geometric product. Both operate entirely in embedding space and are bottlenecked relative to the encoder (the predictor is narrower, the SRGP uses sparse shifts). This is not a coincidence -- both are trying to capture the *relationship* between two representations without reconstructing raw input.

**Connection 2: VL-JEPA as Architecture D**

VL-JEPA [[Chen 2025]](#13-references) predicts **continuous text embeddings** from visual inputs -- no autoregressive token generation needed for perception tasks (classification, retrieval, VQA). The architecture uses:
- Frozen V-JEPA 2 [[Assran 2025]](#13-references) vision encoder
- Cross-attention predictor mapping visual embeddings to text embedding space
- Optional lightweight decoder only when text generation is needed ("selective decoding": 2.85x fewer decoder invocations)

A CliffordVLM variant could adopt this pattern:
```
Architecture D: Clifford-JEPA VLM

Image -> CliffordNet backbone -> vision embeddings s_v
                                       |
                                       v
                    CliffordCrossBlock predictor (SRGP-based)
                                       |
                                       v
                    Predicted text embedding s_hat_t
                                       |
                    +------------------+------------------+
                    |                  |                  |
                    v                  v                  v
               Retrieval:        Classification:     Generation:
               cosine(s_hat_t,   nearest class       CliffordNetLM
               candidate_embs)   embedding           decoder (only
                                                     when needed)
```

This is particularly attractive for CliffordVLM because:
- The SRGP-based predictor replaces cross-attention with O(|S|*D) geometric product (vs O(N*D) cross-attention)
- Selective decoding avoids running the full CliffordNetLM decoder for simple perception tasks
- The predictor operates in embedding space (predict meaning, not tokens) -- philosophically aligned with both JEPA and the geometric product's role of capturing inter-representational structure

**Connection 3: I-JEPA Masking for CliffordNet Vision Pre-training**

I-JEPA's [[Assran 2023]](#13-references) multi-block masking strategy (4 target blocks, each 15-20% of image, context sees remaining 80-85%) could be adapted for CliffordNet vision pre-training:

```
CliffordNet I-JEPA pre-training:

1. Patchify image into N patches
2. Sample 4 target blocks (contiguous rectangular regions)
3. Context patches -> CliffordNet backbone (bidirectional) -> context embeddings
4. Target patches  -> EMA CliffordNet (frozen) -> target embeddings
5. SRGP predictor: predict target embeddings from context + position tokens
6. Loss = L2(predicted, target) in embedding space
```

Advantages over standard CliffordNet classification pre-training:
- No labels required (self-supervised)
- Learns semantic representations, not class-discriminative features
- The CliffordNet's depthwise convolution context stream naturally handles spatial structure
- SRGP already captures the interaction between "what I see" (context) and "what I should predict" (target position)

I-JEPA's efficiency is also a good match: it uses only one view (no augmentations), the context encoder sees fewer patches, and trains in ~5x fewer GPU hours than MAE.

**Anti-collapse considerations**: JEPA variants must prevent representation collapse (all inputs map to same embedding). CliffordNet has natural anti-collapse properties:
- LayerScale (gamma init 1e-5) prevents blocks from collapsing to zero
- The wedge product is explicitly antisymmetric (u^u=0), which injects structural diversity
- The differential context mode (z_ctx -= z_det) ensures non-trivial context

However, for I-JEPA pre-training, additional anti-collapse may be needed. C-JEPA's [[Tong 2024]](#13-references) VICReg regularization (variance + covariance losses) is a proven option -- see Section 8.2 for the full formulation.

### 3.4 Depth Estimation as a Geometric Prior for VLMs

Modern VLMs struggle with 3D spatial reasoning -- questions like "which object is closer?", "how far is the table?", or "describe the depth layout of the scene" require understanding that standard 2D vision encoders do not explicitly capture. Monocular depth estimation, now a mature field led by foundation models like Depth Anything V2 [[Yang 2024b]](#13-references), Metric3D v2 [[Hu 2024]](#13-references), and Depth Pro [[Bochkovskii 2024]](#13-references), provides exactly this missing signal.

**Why depth matters for CliffordVLM specifically:**

The geometric product is inherently about spatial structure. The wedge product `u ^ v` computes the oriented area spanned by two vectors -- a fundamentally geometric quantity. Depth information enriches this with a third dimension:

```
Without depth:  CliffordNet operates on 2D patch features
                Wedge product captures planar structural differences
                "These two patches differ in color/texture"

With depth:     CliffordNet operates on 2D features + depth channel
                Wedge product captures 3D structural differences
                "These two patches are at different depths AND differ in texture"
                The geometric product now encodes occlusion, relative distance,
                and spatial layout -- not just appearance
```

**Three integration strategies for depth in CliffordVLM:**

**Strategy 1: Depth as auxiliary training signal (lightweight, recommended first)**

Train the CliffordNet vision encoder with an auxiliary depth prediction head alongside the primary VLM objective. The encoder learns depth-aware features without adding inference cost:

```
CliffordNet vision backbone -> shared features (B, N, D)
                                   |
                    +--------------+--------------+
                    |                             |
                    v                             v
            VLM projector                   Depth head (1x1 Conv)
            (to LM space)                  -> depth map (B, H', W', 1)
                                                  |
                                           AffineInvariantLoss
                                           vs pseudo-GT from
                                           Depth Anything V2
```

This uses the existing `AffineInvariantLoss` from the codebase and a frozen Depth Anything model as pseudo-ground-truth teacher (no real depth sensors needed). The depth head is discarded at inference.

**Strategy 2: CliffordNet depth decoder (end-to-end Clifford, recommended)**

Add a Clifford-native depth decoder head built entirely from CliffordNet blocks -- no external depth models at inference:

```
CliffordNet vision backbone -> shared features (B, H', W', D)
                                   |
                    +--------------+--------------+
                    |                             |
                    v                             v
            VLM projector                   CliffordNet Depth Decoder
            (to LM space)                  L_depth x CliffordNetBlock
                                           + 1x1 Conv -> depth (B, H', W', 1)
```

The depth decoder is a lightweight stack of 2-4 CliffordNetBlocks (bidirectional, same architecture as the backbone blocks) followed by a 1x1 Conv projection to a single depth channel. This is architecturally consistent -- the entire pipeline from pixels to depth uses only Clifford geometric products, no attention.

**Key design choices:**
- **Shared backbone**: The depth decoder shares the CliffordNet vision backbone with the VLM path. Only the depth-specific CliffordNetBlocks and final projection are separate. This forces the backbone to learn depth-aware features that also benefit the VLM.
- **Isotropic decoder**: Like the backbone, the depth decoder maintains constant spatial resolution. No U-Net-style skip connections or hierarchical upsampling -- consistent with CliffordNet's isotropic philosophy.
- **Bias-free option**: For depth tasks, the `BiasFreeCliffordNetBlock` from `CliffordNetDenoiser` can be used in the depth decoder, satisfying Miyasawa's theorem for optimal denoising of depth maps under additive noise.
- **Training loss**: `AffineInvariantLoss` against pseudo-GT from Depth Anything V2 [[Yang 2024b]](#13-references) during training. The external depth model is a teacher only -- no external models at inference.
- **Depth decoder is discarded or kept**: For VLM-only inference, discard the depth decoder (zero cost). For depth-aware VLM inference, keep it and feed depth features back into the projector (Strategy 3).

**Strategy 3: Depth-conditioned geometric fusion (full integration)**

Feed Clifford-predicted depth back into the VLM projector as an additional context signal:

```
CliffordNet backbone -> features (B, H', W', D)
         |                    |
         v                    v
  Depth decoder          VLM projector
  -> depth (B,H',W',1)       |
         |                    |
         v                    |
  Depth embedding             |
  Dense(depth) -> (B,N,D)    |
         |                    |
         +---------> z_ctx ---+  [depth as context for SRGP in projector]
                              |
                     SRGP(z_det=visual, z_ctx=depth_context)
                              |
                     Projected vision tokens -> LM decoder
```

The depth embedding becomes the **context stream** for the SRGP in the vision projector. This means the geometric product now computes:
- **Inner product** `v . d`: How well visual appearance aligns with depth structure (surfaces with consistent depth)
- **Wedge product** `v ^ d`: Where visual appearance and depth disagree (occlusion boundaries, transparent objects, reflections)

This is a uniquely Clifford-native depth integration -- no other VLM architecture computes the antisymmetric structural difference between appearance and geometry.

**SOTA context (April 2026)**: The field has bifurcated into relative/affine-invariant depth (for visual consistency) and metric depth (absolute scale in meters). For CliffordVLM:
- **Relative depth** (Depth Anything V2 [[Yang 2024b]](#13-references), Marigold [[Ke 2024]](#13-references)) is sufficient for spatial reasoning ("which is closer?") and works with `AffineInvariantLoss` (scale-invariant formulation following [[Eigen 2014]](#13-references))
- **Metric depth** (Metric3D v2 [[Hu 2024]](#13-references), Depth Pro [[Bochkovskii 2024]](#13-references)) is needed for quantitative answers ("how far is the chair?") and requires metric-aware training
- **Strategy 1 + relative depth** is the recommended starting point -- it requires no model changes and uses existing codebase infrastructure

### 3.5 What CliffordVLM Could Be

CliffordVLM occupies a unique position: **end-to-end trained, attention-free, geometrically-grounded**. Unlike standard VLMs that rely on dot-product attention for cross-modal interaction, CliffordVLM uses the geometric product -- meaning cross-modal fusion captures both scalar alignment *and* structural (bivector) differences.

Four candidate architectures for CliffordVLM:

| Architecture | Description | Pros | Cons |
|---|---|---|---|
| **A: LLaVA-style** | CliffordNet vision + projector + CliffordNetLM decoder | Simple, proven paradigm | Projector is a bottleneck |
| **B: Cross-fusion** | CliffordNet vision + CliffordCrossBlocks in decoder | Richer vision-language interaction | Needs new cross-attention layer |
| **C: Unified isotropic** | Single CliffordNet backbone, interleaved tokens | Maximally simple, single architecture | Needs careful token design |
| **D: Clifford-JEPA** | CliffordNet vision + SRGP predictor + selective decode | No decoder for perception; O(|S|D) fusion | Needs text embedding teacher; limited generative ability |

**Recommendation: Architecture A first (simplest), with Architecture B as the medium-term target and Architecture D as a research exploration for perception-heavy use cases (classification, retrieval, VQA without generation).**

---

## 4. Proposed CliffordVLM Architecture

### 4.1 Architecture A: LLaVA-Style CliffordVLM

```
                     CliffordVLM (Architecture A)
                     ===========================

Image (B,H,W,3)                              Text tokens (B,T)
     |                                             |
     v                                             |
[Patch Stem + BN]                                  |
     |                                             |
     v                                             |
L_vis x CliffordNetBlock                           |
(bidirectional, vision backbone)                   |
     |                                             |
     v                                             |
Reshape to (B, N_patches, D_vis)                   |
     |                                             |
     v                                             |
[Clifford Vision Projector]                        |
  z_det = Dense(vision_tokens)                     |
  z_ctx = LearnedQueryPool(vision_tokens)          |
  geo   = SRGP(z_det, z_ctx)                       |
  proj  = LN(z_det + gamma * geo)                  | 
  out   = Dense(D_lm)                              |
     |                                             |
     v                                             v
projected_vision_tokens (B, N_patches, D_lm)  Token Embed + PosEmbed
     |                                             |
     +-----> CONCATENATE <-------------------------+
                  |
                  v
         (B, N_patches + T, D_lm)
                  |
                  v
         Reshape to (B, 1, N_patches + T, D_lm)
                  |
                  v
         L_lm x CausalCliffordNetBlock
         (autoregressive decoder)
                  |
                  v
         Reshape to (B, N_patches + T, D_lm)
                  |
                  v
         LN -> Dense(vocab_size)
                  |
                  v
         Logits (B, T, vocab_size)  [only text positions]
```

### 4.2 The Clifford Vision Projector

The projector is where CliffordVLM diverges most from standard VLMs. Instead of a 2-layer MLP [[Liu 2023]](#13-references), CliffordVLM uses a **geometric projector** -- essentially a 1-layer CliffordNetBlock that maps vision features to LM space through the SRGP + GGR pipeline. LayerScale (gamma init 1e-5) ensures the projector starts as a simple linear map and gradually learns to inject geometric content -- the same pattern validated in CliffordCLIP [[Markou 2026]](#13-references). See Section 7.2 for the full projector design and design choices.

### 4.3 Architecture B: Cross-Fusion CliffordVLM (Future)

Architecture B adds **Clifford cross-attention** layers within the decoder, allowing the language model to attend to vision features at every layer rather than just through the input concatenation.

```
                     CliffordVLM (Architecture B)
                     ===========================

For each CausalCliffordNetBlock in the decoder:

    Text features X (B, 1, T, D)
         |
         v
    [Standard CausalCliffordNetBlock]  <-- self-interaction
         |
         v
    X_self (B, 1, T, D)
         |
         v
    [CliffordCrossBlock]  <-- NEW: cross-modal geometric interaction
         |
         |  Vision features V (B, N, D) used as context
         |  z_det = Dense(X_self)        [query from text]
         |  z_ctx = Dense(V_pooled)      [context from vision]
         |  geo   = SRGP(z_det, z_ctx)   [cross-modal geometric product]
         |  out   = GGR(X_self, geo)     [gated residual update]
         |
         v
    X_cross (B, 1, T, D)
```

The key innovation here is using the SRGP for **cross-modal** interaction:
- `z_det` comes from the text stream (what the model is "looking at")
- `z_ctx` comes from the vision stream (the visual context)
- The inner product captures vision-text alignment (semantic similarity)
- The wedge product captures vision-text structural differences (what's in the image but not in the text, and vice versa)

This requires a new layer: `CliffordCrossBlock`. It would be structurally similar to `CliffordNetBlock` but with the context stream sourced from a different modality rather than from depthwise convolutions of the same input.

### 4.4 Why Not Architecture C (Unified)?

A unified architecture that processes interleaved vision+text tokens through a single CliffordNet backbone is conceptually elegant but faces practical challenges:

1. **Token modality mismatch**: Vision patches and text tokens have very different statistical properties. The depthwise convolutions in CliffordNetBlock assume spatial locality, which doesn't hold across modality boundaries.
2. **Causality complexity**: Vision tokens are bidirectional (patch at position (3,4) should see patch at (5,6)), but text tokens are causal. Mixing both in a single block requires complex masking.
3. **Resolution**: The isotropic architecture processes all tokens at the same resolution, but vision may benefit from hierarchical features while language needs flat token sequences.

Architecture C is worth exploring after A and B are validated, but adds substantial complexity.

### 4.5 Architecture D: Clifford-JEPA (Perception-Optimized)

Architecture D, described in Section 3.3, replaces the autoregressive decoder with a JEPA-style embedding predictor for perception tasks. The SRGP-based predictor maps vision embeddings to predicted text embeddings; a lightweight CliffordNetLM decoder is invoked only when text generation is explicitly needed (VL-JEPA's "selective decoding" [[Chen 2025]](#13-references) reduces decoder invocations by 2.85x).

This is best suited for classification, retrieval, and closed-form VQA -- tasks where the answer can be determined from embedding similarity rather than token-by-token generation. It complements Architecture A rather than replacing it: train Architecture A for general-purpose VLM capability, then distill into Architecture D for efficient deployment on perception-heavy workloads.

---

## 5. Vision Tower: CliffordNet Vision Encoder

### 5.1 Adapting CliffordNet for VLM Vision

The existing CliffordNet vision backbone is designed for classification (GAP -> Dense -> logits). For VLM use, we need **dense spatial features** rather than a single pooled vector.

**Changes needed**:
1. **Remove the classification head**: No GAP, no Dense(num_classes)
2. **Reshape to token sequence**: From `(B, H', W', D)` to `(B, H'*W', D)` = `(B, N_patches, D)`
3. **Add positional information**: The isotropic architecture preserves spatial structure through depthwise convolutions, but the reshaping to 1D loses explicit 2D position. Add 2D sinusoidal or learned positional embeddings before the projector.

**What stays the same**:
- Patch stem (Conv2D-based tokenization)
- CliffordNetBlock stack (bidirectional dual-stream with SRGP + GGR)
- Optional global context branch
- BatchNorm after stem, LayerNorm within blocks

### 5.2 Resolution Handling

CliffordNet's isotropic architecture is naturally resolution-agnostic: since there's no hierarchical downsampling, the spatial dimensions are set by the patch stem and remain constant. This simplifies multi-resolution training:

| Patch Size | Image Size | Patches | Tokens |
|------------|------------|---------|--------|
| 2 | 64 | 32x32 | 1024 |
| 4 | 112 | 28x28 | 784 |
| 4 | 224 | 56x56 | 3136 |
| 8 | 224 | 28x28 | 784 |
| 8 | 336 | 42x42 | 1764 |
| 16 | 224 | 14x14 | 196 |
| 16 | 336 | 21x21 | 441 |

For VLM use, **patch_size=16 with image_size=224** (196 tokens) or **patch_size=8 with image_size=224** (784 tokens) are the practical options. 196 tokens is standard CLIP territory; 784 enables higher fidelity but increases the sequence length for the decoder.

### 5.3 Pre-training the Vision Tower

Five options, ordered by increasing sophistication:

**Option 1: CliffordCLIP contrastive pre-training**
- Train CliffordCLIP on image-text pairs (CC3M, LAION, etc.)
- Extract the vision tower weights
- The contrastive pre-training aligns vision features with language
- The Clifford-aware projection head preserves geometric structure during pre-training
- **Risk**: Penguin-VL [[Zhang 2026]](#13-references) shows contrastive objectives may suppress fine-grained visual cues needed for dense VLM tasks (OCR, counting, spatial reasoning)

**Option 2: CliffordNet classification pre-training**
- Train CliffordNet on ImageNet classification
- Extract backbone weights (discard classification head)
- Cheaper but provides no vision-language alignment
- May require longer VLM training to bridge the modality gap

**Option 3: Train from scratch (not recommended)**
- Only viable for very large datasets (100M+ image-text pairs)
- Wastes the opportunity to leverage existing pre-training

**Option 4: Reconstruction-based pre-training (Penguin-VL style) -- recommended**

Inspired by Penguin-VL's [[Zhang 2026]](#13-references) LLM-initialized encoder, train the CliffordNet vision encoder with generative reconstruction losses against a frozen teacher:

```
Image (B,H,W,3) -> CliffordNet backbone -> student features F_s (B, N, D)
Image (B,H,W,3) -> Frozen teacher (SigLIP/CliffordCLIP) -> teacher features F_t (B, N, D')

Three reconstruction losses:

Amplitude loss:  L_A = (1/N) * sum(|F_s - F_t|)
                 Direct feature magnitude supervision

Direction loss:  L_D = 1 - (1/N) * sum(cosine_sim(F_s_i, F_t_i))
                 Cosine alignment per patch -- aligns feature directions with teacher

Relation loss:   L_R = (1/N) * sum(|F_s*F_s^T/||F_s||^2 - F_t*F_t^T/||F_t||^2|)
                 Self-correlation supervision -- explicitly supervises inter-patch
                 relationships, focusing on attention-based token interactions
                 Penguin-VL ablation: this loss alone adds +3.3 absolute improvement

Combined: L_recon = L_A + L_D + L_R  (warm-up during initial phase)
```

**Two-phase encoder training** (following Penguin-VL):
- **Phase 1.1 (Low-res warm-up)**: ~600x600 pixels (~784 tokens with patch_size=8), large dataset (200M+ samples), LR=1e-3. Combined image-text LM loss + reconstruction losses.
- **Phase 1.2 (High-res fine-tune)**: Full resolution (up to 10K tokens), smaller curated dataset (~50M samples), LR=5e-4. Drop reconstruction losses, focus on fine-grained alignment.

**Why this is better for CliffordVLM**:
- Dense, token-by-token supervision (not just global contrastive)
- Preserves fine-grained spatial cues critical for VLM downstream tasks
- The CliffordNet backbone's dual-stream architecture (detail + context) naturally separates amplitude information (detail stream) from relational information (context stream), making it an excellent fit for Penguin's amplitude/direction/relation decomposition
- Dramatically more data-efficient than contrastive pre-training (see Section 3.2)

**Option 5: I-JEPA self-supervised pre-training (no labels, no pairs)**

Train the CliffordNet vision encoder using the I-JEPA masking objective -- predict masked patch embeddings from visible context, entirely in embedding space:

```
1. Patchify image, sample 4 target blocks (15-20% each), context = rest
2. Context -> CliffordNet backbone (trainable) -> s_context
3. Targets -> EMA CliffordNet (frozen, stop-grad) -> s_target
4. SRGP predictor: s_hat = SRGP_predict(s_context, target_positions)
5. L = ||s_hat - sg[s_target]||^2

Anti-collapse: EMA update (tau 0.996 -> 1.0) + predictor bottleneck
              + optional C-JEPA VICReg regularization
```

**Advantages**:
- No labels, no text, no image-text pairs required -- trains on any image collection
- Learns semantic (not pixel-level) representations, naturally discarding noise
- The CliffordNet dual-stream architecture is a natural fit: the detail stream (linear) encodes "what's here" while the context stream (DWConv) encodes "what's nearby" -- exactly the context-target relationship I-JEPA requires
- 5x cheaper than MAE pre-training [[Assran 2023]](#13-references)
- The SRGP predictor replaces I-JEPA's narrow transformer predictor -- testing whether the geometric product can serve as a general-purpose embedding predictor

**Risk**: I-JEPA pre-training produces features optimized for *predicting in embedding space*, not for *alignment with language*. A subsequent alignment stage (Stage 1) is still needed to bridge the modality gap.

**Recommended approach**: Start with **Option 1** (CliffordCLIP, since we're already training it), then investigate **Option 4** (Penguin-style reconstruction) as the medium-term upgrade. **Option 5** (I-JEPA) is worth exploring as a fully self-supervised alternative that requires zero paired data -- especially for scenarios where image-text pairs are scarce but unlabeled images are abundant.

### 5.4 Token Budget Management

Penguin-VL introduces **Temporal Redundancy-Aware (TRA) token compression** for video, with a three-stage strategy that progressively compresses tokens when the budget is exceeded. For CliffordVLM (image-only initially), a simpler version applies:

**Static token budget**: Set a maximum total sequence length T_max (e.g., 2048 or 4096). The vision token count N_vis determines how many text tokens remain: T_text = T_max - N_vis.

| Config | Image Size | Patch Size | Vision Tokens | Text Budget (T_max=2048) |
|--------|-----------|------------|---------------|-------------------------|
| Low-res | 112x112 | 4 | 784 | 1264 |
| Standard | 224x224 | 16 | 196 | 1852 |
| Standard | 224x224 | 8 | 784 | 1264 |
| High-res | 336x336 | 8 | 1764 | 284 (too few!) |
| High-res | 336x336 | 16 | 441 | 1607 |

**Observation**: High-resolution with small patches (patch_size=8, 336x336 = 1764 tokens) leaves almost no room for text. Two solutions:

1. **Increase T_max**: CliffordVLM's O(N) complexity makes longer sequences cheaper than attention-based models. T_max=4096 is practical.
2. **Spatial downsampling post-encoder**: Following Penguin-VL, apply 2x bilinear downsampling to vision features after the encoder. This halves the token count: 1764 -> 441 tokens, preserving spatial quality while fitting the budget.

### 5.5 Curriculum: Low-to-High Resolution

Following Penguin-VL's two-phase resolution curriculum:

1. **Phase 1.1 (Low-res, large data)**: 112x112, patch_size=4 -> 784 tokens. Train on ~3.5M pairs. Fast iteration, builds basic visual-semantic alignment.
2. **Phase 1.2 (High-res, curated data)**: 224x224, patch_size=8 -> 784 tokens (same token count, higher resolution). Train on ~1M curated pairs. Teaches fine-grained detail without changing sequence length.
3. **Optional high-res fine-tune** (see Stage 4 in Section 8.1): 336x336, patch_size=16 -> 441 tokens. Short fine-tuning at maximum resolution.

The isotropic architecture handles resolution changes gracefully -- only the positional embeddings need bilinear interpolation (2D grid interpolation for the learned embedding table).

---

## 6. Language Tower: CliffordNet Causal Decoder

### 6.1 Current CliffordNetLM Architecture

```
Token IDs (B,T) -> Embedding(vocab, D) + Embedding(T_max, D) -> LN -> Dropout
-> Reshape(B,1,T,D) -> L x CausalCliffordNetBlock -> Reshape(B,T,D)
-> LN -> Dropout -> Dense(vocab) -> Logits
```

The causal blocks use:
- Left-only padded depthwise convolutions (receptive field = 5 positions with 2 stacked 3x3)
- Causal cumulative mean for global context (position i sees average of 0..i)
- SRGP for geometric interaction between detail and context streams

### 6.2 Adapting for VLM

The CliffordNetLM needs minimal changes for VLM use:

1. **Accept pre-embedded inputs**: Instead of token IDs, accept a mixed sequence of projected vision tokens + text embeddings
2. **Modified positional embeddings**: Vision tokens get positions [0, N_patches-1], text tokens get positions [N_patches, N_patches+T-1]
3. **Causal mask handling**: Vision tokens can attend to each other bidirectionally (they're all "in the past" relative to text generation), text tokens are causal

**Critical design decision**: Should vision tokens be causal or bidirectional within the decoder?

| Approach | Pros | Cons |
|----------|------|------|
| **Fully causal** (vision + text) | Simpler, existing CausalCliffordNetBlock works as-is | Vision tokens can't see "future" patches (patch 5 can't see patch 6) |
| **Vision bidirectional, text causal** | Vision tokens attend to all patches | Requires block-diagonal masking, custom padding |
| **Prefix-causal** (vision all-to-all, text causal) | Best of both worlds, standard in LLaVA-style models | Causal convolution padding becomes tricky |

**Recommendation: Fully causal** for Architecture A. The causal ordering of vision patches (raster scan order) means each patch sees all patches above and to the left -- which is a reasonable inductive bias for images. More importantly, the CausalCliffordNetBlock's local receptive field (5 positions) means each vision token effectively sees its 4 preceding patches plus the global cumulative context, which provides sufficient vision-vision interaction for the VLM use case.

If the fully causal approach proves insufficient, the vision tokens can be pre-processed by a short bidirectional CliffordNet stack (2-3 layers) before concatenation with text tokens.

### 6.3 Receptive Field Analysis

The CausalCliffordNetBlock has a limited local receptive field due to depthwise convolutions:

```
Single block: 2 stacked 3x3 causal DWConv = effective RF of 5 positions
L blocks: effective RF grows linearly as 4L + 1 positions (not exponential)

With L=12 blocks: RF = 49 positions
With L=20 blocks: RF = 81 positions
```

For a VLM with 196 vision tokens + 256 text tokens = 452 total positions:
- L=12: Each text token can see ~49 previous tokens through local convolutions
- L=20: Each text token can see ~81 previous tokens through local convolutions

The **global context branch** (causal cumulative mean) compensates for limited local RF by giving each position access to the global average of all preceding positions. This is critical for VLM: even with L=12 local blocks, the cumulative mean ensures each text token has a summary of all vision tokens.

However, the cumulative mean is a **compressed summary** (single D-dimensional vector averaging over hundreds of positions). For fine-grained vision-language grounding (e.g., "what color is the small object in the upper-left corner?"), the local RF may be insufficient.

**Mitigation strategies**:
1. **Increase shifts**: Using shifts=[1,2,4,8,16,32,64] increases the effective mixing range
2. **More layers**: L=20+ gives RF > 81, covering more vision tokens
3. **Architecture B**: Cross-fusion gives direct per-layer access to all vision tokens (not through the sequential bottleneck)
4. **Vision token compression**: Reduce N_patches (e.g., use patch_size=16 for 196 tokens instead of 784)

### 6.4 Pre-training the Language Tower

Options:

**Option 1: Pre-train CliffordNetLM on text corpus (recommended)**
- Train on OpenWebText, FineWeb, or similar
- Establishes language modeling capability before VLM training
- Causal Clifford blocks learn text-appropriate geometric interactions

**Option 2: Initialize from CliffordCLIP text tower**
- The CliffordCLIP text tower uses the same CausalCliffordNetBlock
- Weights transfer directly (same architecture)
- Contrastive pre-training gives some language understanding
- May need additional LM pre-training to develop generative capability

**Option 3: Train from scratch with VLM objective**
- Only viable with very large image-text datasets
- Dual learning (language + vision-language) in one stage

---

## 7. The Bridge: Clifford Geometric Fusion

### 7.1 The Fundamental Question

How should vision and language representations interact through Clifford algebra?

In standard VLMs, the bridge is either:
- **Simple**: MLP projector (LLaVA [[Liu 2023]](#13-references)) -- linear/nonlinear mapping from vision space to LM space
- **Complex**: Q-Former (BLIP-2 [[Li 2023]](#13-references)) -- learned bottleneck with cross-attention queries
- **None**: Direct projection (Fuyu) -- linearly projected patches fed directly to the LM

CliffordVLM has a unique opportunity: use the **geometric product** as the bridge operation, capturing both alignment and structural difference between modalities.

### 7.2 Clifford Vision Projector (Architecture A)

The simplest Clifford-native bridge. Applied to vision tokens before concatenation with text:

```
Input: vision_features (B, N, D_vis) from CliffordNet backbone

1. Linear projection:     z = Dense(vision_features)       -> (B, N, D_lm)
2. Detail stream:         z_det = Dense(z)                 -> (B, N, D_lm)
3. Context stream:        z_ctx = DWConv(DWConv(BN(z)))    -> (B, N, D_lm)
   (or)                   z_ctx = LearnedQueryPool(z)       -> (B, D_lm) broadcast
4. Differential context:  z_ctx = z_ctx - z_det            -> high-pass filtered
5. Geometric product:     geo = SRGP(z_det, z_ctx)         -> (B, N, D_lm)
6. Gated residual:        out = z + gamma * geo            -> (B, N, D_lm)
7. Final projection:      projected = Dense(LN(out))       -> (B, N, D_lm)
```

This is essentially a 1-layer CliffordNetBlock applied to vision tokens, followed by dimension projection. It reuses all existing primitives.

**Design choices**:

| Choice | Option A | Option B | Recommendation |
|--------|----------|----------|----------------|
| Context source | DWConv (local patches) | LearnedQueryPool (global) | Both (like CliffordNetBlock with global branch) |
| Context mode | `"diff"` (high-pass) | `"abs"` (preserve intensity) | `"diff"` (proven winner in vision) |
| Shifts | [1,2] (minimal) | [1,2,4,8,16] (full) | [1,2,4] (balance cost/expressivity) |
| Depth | 1 block | 2-3 blocks | 1 block (projector should be lightweight) |

### 7.3 Clifford Cross-Fusion Block (Architecture B)

For deeper vision-language interaction, we need a **cross-modal** variant of the geometric product. This is a new component that doesn't exist yet.

**Concept**: In standard CliffordNetBlock, both z_det and z_ctx come from the same input (via different processing paths). In the cross-fusion block, z_det comes from one modality and z_ctx from another:

```
CliffordCrossBlock:

    Text features X_text (B, T, D)
    Vision features X_vis (B, N, D)
         |                |
         v                v
    z_det = Dense(X_text)     z_ctx = Pool(X_vis)  [pool N -> T or broadcast]
         |                |
         v                v
    geo = SRGP(z_det, z_ctx)   [cross-modal geometric product]
         |
         v
    out = GGR(X_text, geo)    [gated residual update to text stream]
```

The main challenge is **spatial alignment**: text has T positions, vision has N positions. Options:

1. **Global pool**: `z_ctx = mean(X_vis)` broadcast to all T text positions. Simple but loses spatial info.
2. **Learned query pool**: `z_ctx = LearnedQueryPool(X_vis)` per text position. Expensive but preserves spatial info.
3. **Grouped mapping**: Divide N vision tokens into T groups, mean-pool each group. Assumes rough spatial correspondence.
4. **Cross cumulative mean**: Prepend vision tokens to text, use causal cumulative mean. Each text position sees the average of all vision + preceding text tokens. **This is the most CliffordNet-native approach** -- it's exactly what the causal cumulative mean already does.

**Recommendation**: Option 4 (cross cumulative mean) for Architecture A/B alignment. This is already how the concatenation approach works -- the CausalCliffordNetBlock's global context branch automatically computes a running average that includes all vision tokens. Architecture B's explicit cross-fusion block should use Option 2 (learned query pool) for richer interaction.

### 7.4 Geometric Interpretation

Why is the geometric product a good cross-modal operator?

Consider a vision token `v` (representing a patch showing a red car) and a text token `t` (representing the word "car"):

```
v . t  (inner product)  -> scalar measuring alignment
                           High when vision and text representations are semantically similar
                           "This patch is about a car" -> high inner product

v ^ t  (wedge product)  -> bivector measuring structural difference
                           Non-zero when representations differ in structure
                           "This patch shows a RED car" -> the wedge captures
                           the 'redness' that is in the vision but not the text
```

The full geometric product `vt = v.t + v^t` gives the model access to both:
- **What matches** (inner): Essential for alignment and grounding
- **What differs** (wedge): Essential for detailed description, counting, attribute binding

Standard dot-product attention only computes `v.t` (the inner product part). The wedge product is the **novel channel** that CliffordVLM uniquely provides.

### 7.5 Comparison with Existing Fusion Strategies

| Fusion Type | Operation | Captures Alignment? | Captures Structure? | Params |
|---|---|---|---|---|
| Dot-product attention | `softmax(QK^T/sqrt(d))V` | Yes (via QK^T) | No | O(d^2) |
| Bilinear pooling | `x^T W y` | Yes | Partial (via W) | O(d^3) |
| Tensor fusion | `[1;x] x [1;y]` | Yes | Partial (outer product) | O(d^2) |
| Cross-attention + FFN | `Attn(Q,K,V) + FFN(.)` | Yes | Partial (via FFN) | O(d^2) |
| **Clifford geometric product** | `u.v + u^v` | Yes (inner) | Yes (wedge) | O(|S|*d) |

The geometric product is both more expressive (algebraically complete) and more parameter-efficient (O(|S|*d) vs O(d^2)) than alternatives.

---

## 8. Training Strategy

### 8.1 Multi-Stage Training

Following Penguin-VL's [[Zhang 2026]](#13-references) three-stage pipeline adapted for CliffordNet's geometric algebra backbone. Each stage has distinct frozen/trainable partitions and learning rate schedules per component.

**Stage 0: Component Pre-training** (done or in progress)
- CliffordCLIP contrastive training on CC3M/LAION -> vision tower + text tower weights
- CliffordNetLM pre-training on text corpus -> language model weights

**Stage 1: Vision Encoder + Projector Alignment**

Two sub-phases following Penguin-VL's resolution curriculum:

| Sub-phase | Resolution | Vision Tokens | Frozen | Trainable | LR (encoder) | LR (projector) | Data |
|-----------|-----------|---------------|--------|-----------|-------------|----------------|------|
| **1.1 Low-res** | 112x112, patch=4 | 784 | LM decoder | Vision encoder + projector | 1e-3 | 1e-3 | CC3M + LLaVA-Pretrain (~3.5M pairs) |
| **1.2 High-res** | 224x224, patch=8 | 784 | LM decoder | Vision encoder + projector | 5e-4 | 1e-3 | Curated subset (~1M pairs) |

- **Objective**: Next-token prediction on captions + optional reconstruction losses (Penguin-style, see Section 5.3 Option 4)
- **Key insight from Penguin-VL**: Training the encoder (not just the projector) during alignment is critical. Penguin's ablation shows LLM-initialized + encoder training beats frozen encoder + projector-only by +3.3 absolute points.
- **Reconstruction branch** (Phase 1.1 only): If using Penguin-style training, add amplitude/direction/relation losses against a frozen teacher. Remove reconstruction branch in Phase 1.2 and focus on LM loss only.
- **Duration**: Phase 1.1 ~1 day, Phase 1.2 ~half day (single RTX 4090)

**Stage 2: Full VLM Pre-training**
- **Trainable**: All parameters (vision encoder, projector, LM decoder)
- **LR**: Encoder 1e-4, projector 1e-3, LM decoder 2e-4
- **Data**: Mixed multi-task data following Penguin-VL's proportions:

| Category | Proportion | Examples |
|----------|-----------|----------|
| General captions | 60% | CC3M, COCO, Conceptual-12M |
| Document/OCR | 15% | DocVQA, ChartQA, TextVQA |
| Grounding | 7% | RefCOCO, Visual Genome |
| Math/Science | 5% | MathVista, ScienceQA |
| Code (multimodal) | 3% | Code screenshots, diagrams |
| Text-only (prevent forgetting) | 10% | OpenWebText, ShareGPT |

- **Objective**: Next-token prediction across all task types
- **Duration**: ~2-3 days (single RTX 4090 for base variant)
- **Critical**: Include text-only data (~10%) to prevent catastrophic forgetting of language capabilities (Penguin-VL principle)

**Stage 3: Supervised Fine-Tuning (SFT)**
- **Trainable**: LM decoder (full or LoRA), projector, vision encoder (low LR or frozen)
- **LR**: Encoder 1e-5 (or frozen), projector 5e-4, LM decoder 2e-5
- **Data**: High-quality instruction-following data

| Category | Proportion | Source |
|----------|-----------|--------|
| General & caption | 30% | LLaVA-Instruct, ShareGPT |
| Document, chart & table | 20% | DocVQA-instruct, ChartQA-instruct |
| OCR, text QA | 15% | TextVQA-instruct, InfoVQA |
| Grounding & counting | 10% | RefCOCO-instruct, counting datasets |
| Mathematics | 10% | MathV360K, geometry problems |
| Multi-image | 5% | NLVR2, interleaved conversations |
| Science/knowledge | 5% | ScienceQA, A-OKVQA |
| Conversation (text-only) | 5% | ShareGPT, Alpaca |

- **Objective**: Next-token prediction on instruction-response pairs
- **Duration**: ~1-2 days

**Optional Stage 4: High-Resolution Fine-tuning**
- Increase image resolution (224 -> 336 or 448, up to 1764 vision tokens)
- Interpolate positional embeddings via bilinear interpolation
- Short fine-tuning (~0.5 day) at high resolution
- Penguin-VL supports up to 10,240 visual tokens; CliffordVLM's O(N) complexity makes this more tractable than attention-based models

### 8.2 Loss Functions

**Primary loss**: Causal language modeling (next-token prediction)
```
L_lm = CrossEntropy(logits[:, N_patches:, :], target_tokens)
```

Only compute loss on text token positions, not vision token positions.

**Reconstruction losses** (Stage 1.1 only, if using Penguin-style pre-training -- see Section 5.3 Option 4 for formulas):
```
L = L_lm + lambda_A * L_A + lambda_D * L_D + lambda_R * L_R
(lambda_A = lambda_D = lambda_R = 1.0 initially; tune on validation)
```
Where L_A (amplitude), L_D (direction), and L_R (relation) supervise feature magnitude, cosine alignment, and inter-patch self-correlation against a frozen teacher, respectively.

**Depth auxiliary loss** (optional, all stages -- see Section 3.4 Strategy 1):
```
L_depth = AffineInvariantLoss(depth_head(features), depth_teacher(image))

where depth_teacher = frozen Depth Anything V2 producing pseudo-GT depth maps
      depth_head    = lightweight 1x1 Conv on shared CliffordNet features
      weight        = lambda_depth (0.1-0.5, tune on validation)

Combined: L = L_lm + lambda_depth * L_depth
```

The `AffineInvariantLoss` normalizes both prediction and target by median and MAD, making it robust to scale differences between depth teachers and datasets. The depth head is discarded at inference -- zero additional cost.

**Clifford-specific auxiliary losses** (optional, Stage 2+):

- **Geometric coherence loss**: Encourage the wedge component to be informative -- small when vision-text are semantically aligned, large when they differ structurally
  ```
  L_wedge = ||wedge(v_matched, t_matched)||^2 - ||wedge(v_unmatched, t_unmatched)||^2 + margin
  ```

- **Inner-wedge balance loss**: Prevent degenerate solutions where only inner or only wedge is used
  ```
  L_balance = |mean(||inner||) - mean(||wedge||)|
  ```
  Encourages both algebraic components to contribute roughly equally.

- **CLIP contrastive loss**: Maintain vision-language alignment during instruction tuning (from CliffordCLIP, optional auxiliary at low weight)

- **VICReg anti-collapse loss** (if using I-JEPA or JEPA-style pre-training, from C-JEPA):
  ```
  L_variance   = mean(relu(gamma - sqrt(Var(embeddings) + eps)))
  L_covariance = sum(off_diagonal(Cov(embeddings))^2) / d
  L_vicreg     = lambda_var * L_variance + lambda_cov * L_covariance
  (lambda_var = lambda_cov = 25.0, gamma = 1.0)
  ```
  Prevents representation collapse by ensuring embedding dimensions have sufficient variance and are decorrelated. Only needed during self-supervised pre-training stages (Options 4/5 in Section 5.3), not during standard LM-loss fine-tuning.

### 8.3 Optimizer & Schedule

| Parameter | Stage 1.1 | Stage 1.2 | Stage 2 | Stage 3 |
|-----------|-----------|-----------|---------|---------|
| Optimizer | AdamW | AdamW | AdamW | AdamW |
| Weight decay | 0.05 | 0.05 | 0.05 | 0.01 |
| beta1 / beta2 | 0.9 / 0.999 | 0.9 / 0.999 | 0.9 / 0.999 | 0.9 / 0.999 |
| LR schedule | Cosine w/ warmup | Cosine w/ warmup | Cosine w/ warmup | Cosine w/ warmup |
| Warmup ratio | 3% | 3% | 3% | 3% |
| LR (encoder) | 1e-3 | 5e-4 | 1e-4 | 1e-5 or frozen |
| LR (projector) | 1e-3 | 1e-3 | 1e-3 | 5e-4 |
| LR (LM decoder) | frozen | frozen | 2e-4 | 2e-5 |
| Batch size | 128 | 64 | 64 | 32 |
| Gradient clipping | max_norm=1.0 | max_norm=1.0 | max_norm=1.0 | max_norm=1.0 |
| Mixed precision | float16 fwd/bwd, float32 opt | same | same | same |
| Max seq length | 1024 | 1024 | 2048 | 2048 |

**Per-component LR scheduling** (following Penguin-VL): The vision encoder, projector, and LM decoder each get independent learning rates. This prevents the projector (randomly initialized) from destabilizing the pre-trained towers. Penguin-VL demonstrated that differential LR is critical -- the encoder needs a gentler schedule than the projector.

### 8.4 Data Pipeline

| Stage | Dataset | Size | Format | Reconstruction Teacher |
|-------|---------|------|--------|----------------------|
| 0a (CLIP) | CC3M, LAION-400M | 3-400M pairs | (image, caption) | -- |
| 0b (LM) | OpenWebText, FineWeb | 10-100B tokens | text | -- |
| 1.1 (low-res align) | CC3M + LLaVA-Pretrain | ~3.5M pairs | (image, caption) | Frozen CliffordCLIP/SigLIP |
| 1.2 (high-res align) | Curated high-quality | ~1M pairs | (image, long caption) | -- |
| 2 (pre-train) | Multi-task mix | ~5-10M samples | Mixed | -- |
| 3 (SFT) | Instruction data | ~500K conversations | (image, instruction, response) | -- |

### 8.5 Training Stability

Following lessons from both CliffordNet and Penguin-VL:

- **LayerScale (gamma init 1e-5)**: All CliffordNet blocks start as near-identity, preventing training collapse in deep stacks. This is especially important when combining pre-trained vision + LM towers.
- **Gradient clipping (max_norm=1.0)**: Prevents gradient explosions during cross-modal interaction learning.
- **Reconstruction loss warm-up**: When using Penguin-style reconstruction losses, ramp them up linearly over the first 5% of Stage 1.1. This ensures "smooth injection of visual knowledge" (Penguin-VL).
- **Text-only data mixing**: Include ~10% text-only samples in Stage 2 to prevent catastrophic forgetting of language capabilities.
- **Drop-path schedule**: Linear schedule from 0 to max_rate across depth, consistent with CliffordNet conventions.
- **Stochastic depth rates**: nano=0.05, mini=0.10, small=0.10, base=0.15, large=0.20.

---

## 9. Training Metrics & Monitoring

### 9.1 Core Training Metrics (Log Every Step)

| Metric | Formula / Description | What It Tells You | Alert Threshold |
|--------|----------------------|-------------------|-----------------|
| **train/loss** | `CrossEntropy(logits, targets)` on text tokens only | Primary training signal | Divergence: loss > 2x running mean |
| **train/perplexity** | `exp(loss)` | Interpretable LM quality | PPL > 1000 in early training = problem |
| **train/learning_rate** | Current LR per component (encoder, projector, decoder) | Schedule verification | Check warmup ramp, cosine shape |
| **train/grad_norm** | `||grad||_2` before clipping, per component | Training stability | Spikes > 10x mean = instability |
| **train/grad_norm_clipped** | `||grad||_2` after clipping | How often clipping activates | Sustained clipping = LR too high |
| **train/tokens_per_sec** | Throughput | Performance monitoring | Sudden drops = data pipeline bottleneck |
| **train/sequence_length** | Actual sequence length (vision + text) | Memory / compute tracking | -- |

### 9.2 Clifford-Specific Metrics (Log Every N Steps)

These metrics are unique to CliffordVLM and track the geometric algebra components:

| Metric | Formula / Description | What It Tells You | Healthy Range |
|--------|----------------------|-------------------|---------------|
| **clifford/layerscale_gamma_mean** | `mean(gamma)` across all blocks | How much Clifford content is injected | Should grow from ~1e-5 to ~0.01-0.1 during training |
| **clifford/layerscale_gamma_max** | `max(gamma)` across all blocks | Which blocks inject most Clifford content | > 0.5 may indicate over-reliance on single block |
| **clifford/inner_norm** | `mean(||inner_product||_2)` across SRGP layers | Strength of alignment signal | Should stabilize; collapse to 0 = dead channel |
| **clifford/wedge_norm** | `mean(||wedge_product||_2)` across SRGP layers | Strength of structural signal | Should stabilize; collapse to 0 = wedge is unused |
| **clifford/inner_wedge_ratio** | `mean(||inner||) / mean(||wedge||)` | Balance between alignment and structure | 0.3-3.0 is healthy; extremes suggest degenerate solution |
| **clifford/projector_inner_norm** | Inner product norm in the vision projector specifically | Cross-modal alignment strength | -- |
| **clifford/projector_wedge_norm** | Wedge product norm in the vision projector specifically | Cross-modal structural difference | -- |
| **clifford/gate_activation** | `mean(sigmoid(W_gate[X;G]))` in GGR blocks | How aggressively blocks mix geometric content | 0.3-0.7 is healthy; near 0 = blocks are identity |
| **clifford/cumulative_mean_contrib** | `mean(||global_context||) / mean(||local_context||)` | Balance of local vs global context | Very low = global branch underused |

**Interpretation guide**:
- If `inner_wedge_ratio` collapses to near 0: the model is ignoring alignment and only using structural differences. Consider initializing with CliffordCLIP weights to seed useful inner product representations.
- If `inner_wedge_ratio` grows unbounded: the wedge product is not learning useful features. This is expected early (LayerScale starts at 1e-5), but if it persists past 20% of training, the geometric product may not be helpful for this task.
- If `layerscale_gamma` stays near 1e-5: the Clifford content is not being used. The model is effectively a plain MLP-based VLM. Check that gradients flow through the SRGP and GGR layers.

### 9.3 JEPA Pre-training Metrics (If Using Option 5)

If pre-training the vision encoder with I-JEPA self-supervised learning:

| Metric | Formula | What It Tells You | Alert |
|--------|---------|-------------------|-------|
| **jepa/prediction_loss** | `L2(s_hat_target, sg[s_target])` | How well predictor reconstructs target embeddings | Should decrease steadily |
| **jepa/ema_tau** | Current EMA decay rate | Target encoder staleness | Should increase from 0.996 to ~1.0 |
| **jepa/embedding_std** | `mean(std(embeddings, dim=batch))` | Per-dimension embedding variance | Collapse if < 0.1 |
| **jepa/embedding_rank** | Effective rank of embedding covariance matrix | Dimensionality utilization | Should be > 0.5 * D |
| **jepa/vicreg_variance** | C-JEPA variance loss component | Anti-collapse regularization | Persistent high values = collapse risk |
| **jepa/vicreg_covariance** | C-JEPA covariance loss component | Dimension decorrelation | Should decrease during training |
| **jepa/context_target_cosine** | `mean(cosine_sim(s_predicted, s_target))` | Prediction quality in direction | Should approach ~0.8-0.9 |

**Collapse detection**: If `embedding_std` drops below 0.1 or `embedding_rank` drops below 0.3*D, the encoder is collapsing. Increase VICReg weight or decrease learning rate.

### 9.4 Reconstruction Loss Metrics (Stage 1.1 Only)

| Metric | Formula | What It Tells You |
|--------|---------|-------------------|
| **recon/amplitude_loss** | `L_A = (1/N) * sum(\|F_s - F_t\|)` | Feature magnitude alignment with teacher |
| **recon/direction_loss** | `L_D = cosine_distance(F_s, F_t)` | Feature direction alignment with teacher |
| **recon/relation_loss** | `L_R = self_correlation_distance(F_s, F_t)` | Inter-patch relationship alignment |
| **recon/total** | `L_A + L_D + L_R` | Combined reconstruction quality |
| **recon/student_teacher_cosine** | `mean(cosine_sim(F_s, F_t))` per patch | How well student matches teacher |

Monitor `recon/relation_loss` separately -- Penguin-VL [[Zhang 2026]](#13-references) showed it is the most impactful reconstruction component. If it plateaus early while other losses decrease, the CliffordNet encoder may need more capacity to capture inter-patch relationships.

### 9.5 Depth Auxiliary Metrics (If Using Depth Strategy 1)

| Metric | Formula | What It Tells You |
|--------|---------|-------------------|
| **depth/affine_invariant_loss** | `AffineInvariantLoss(pred, teacher)` | Structural depth accuracy (scale-free) |
| **depth/abs_rel** | `mean(\|d_pred - d_gt\| / d_gt)` | Relative depth error |
| **depth/delta_1** | `% of pixels where max(d_pred/d_gt, d_gt/d_pred) < 1.25` | Depth accuracy threshold |
| **depth/gradient_correlation** | `corr(grad(d_pred), grad(d_gt))` | Edge/surface alignment quality |
| **depth/head_grad_norm** | `\|\|grad(depth_head)\|\|_2` | Whether depth gradients help/hurt the backbone |

Monitor `depth/head_grad_norm` -- if it's orders of magnitude larger than the LM gradient norm, reduce `lambda_depth`. The depth auxiliary should help the backbone without dominating the training signal.

### 9.6 Validation Metrics (Log Every K Steps)

| Metric | Description | Benchmark | Frequency |
|--------|-------------|-----------|-----------|
| **val/loss** | Validation cross-entropy on held-out set | Internal split | Every 500 steps |
| **val/perplexity** | `exp(val/loss)` | Internal split | Every 500 steps |
| **val/cider** | CIDEr score on captioning subset | COCO-val (5K) | Every 2K steps |
| **val/bleu4** | BLEU-4 score on captioning subset | COCO-val (5K) | Every 2K steps |
| **val/vqa_accuracy** | Exact match accuracy on VQA subset | VQAv2-minival | Every 5K steps |
| **val/text_ppl** | Perplexity on text-only validation set | WikiText-103 val | Every 5K steps |

### 9.7 Evaluation Benchmarks (Run at End of Each Stage)

Following Penguin-VL's [[Zhang 2026]](#13-references) evaluation suite, categorized by capability:

**General Understanding**:
| Benchmark | Metric | Description |
|-----------|--------|-------------|
| MMBench | Accuracy | Multi-choice visual understanding |
| MMStar | Accuracy | Visual reasoning with minimal textual bias |
| MMMU | Accuracy | Multi-discipline multimodal understanding |
| SEED-Image | Accuracy | Spatial and visual understanding |

**Document & OCR**:
| Benchmark | Metric | Description |
|-----------|--------|-------------|
| DocVQA | ANLS | Document question answering |
| ChartQA | Accuracy | Chart comprehension |
| TextVQA | VQA Accuracy | Text reading in natural images |
| InfoVQA | ANLS | Infographic understanding |
| OCRBench | Accuracy | Comprehensive OCR evaluation |

**Math & Science**:
| Benchmark | Metric | Description |
|-----------|--------|-------------|
| MathVista | Accuracy | Mathematical reasoning with visuals |
| MathVerse | Accuracy | Math problem solving |
| AI2D | Accuracy | Science diagram understanding |

**Grounding & Spatial**:
| Benchmark | Metric | Description |
|-----------|--------|-------------|
| RefCOCO | IoU@0.5 | Referring expression grounding |
| POPE | F1 | Object hallucination detection |
| HallusionBench | Accuracy | Hallucination robustness |

**Depth & 3D Reasoning** (if using depth integration):
| Benchmark | Metric | Description |
|-----------|--------|-------------|
| NYUv2 | AbsRel, delta1 | Indoor depth estimation accuracy |
| KITTI | AbsRel, RMSE | Outdoor depth estimation accuracy |
| ScanNet-depth | AbsRel | 3D scene understanding |
| CV-Bench-3D | Accuracy | VLM 3D spatial reasoning questions |

**Efficiency Metrics** (run alongside benchmarks):
| Metric | Description | How to Measure |
|--------|-------------|----------------|
| **Throughput (tok/s)** | Tokens generated per second | `time(generate(100_prompts))` |
| **Prefill latency** | Time to process vision + prompt | `time(forward_pass(vision + prompt))` |
| **Memory (GB)** | Peak GPU memory during inference | TF memory profiler or `nvidia-smi` peak |
| **Params (M)** | Total trainable parameters | `model.count_params()` |
| **FLOPs (G)** | Floating-point operations per forward pass | Profiler or analytical estimate |

### 9.8 Ablation-Specific Metrics

Track these when running ablation studies to isolate the contribution of Clifford components:

| Ablation | What to Compare | Key Metric |
|----------|----------------|------------|
| SRGP vs MLP projector | CliffordVLM vs same model with Dense-Dense projector | val/vqa_accuracy, val/cider |
| Full vs inner-only mode | `cli_mode="full"` vs `cli_mode="inner"` | val/vqa_accuracy, clifford/wedge_norm |
| Global context on/off | `use_global_context=True/False` | val/vqa_accuracy, especially spatial questions |
| Shift sets | [1,2] vs [1,2,4] vs [1,2,4,8,16] | val/loss convergence speed |
| With/without reconstruction losses | Stage 1.1 with/without L_A + L_D + L_R | All Stage 2 benchmarks |
| CliffordCLIP vs Penguin-style init | Contrastive vs reconstruction pre-training | DocVQA (fine-grained), MMBench (general) |
| Vision token count | patch_size=8 (784 tokens) vs 16 (196 tokens) | val/loss, throughput, spatial benchmarks |
| I-JEPA vs CliffordCLIP init | Self-supervised JEPA pre-train vs contrastive pre-train | All benchmarks, especially fine-grained (DocVQA, TextVQA) |
| I-JEPA with/without VICReg | I-JEPA + C-JEPA regularization vs plain I-JEPA | jepa/embedding_std, jepa/embedding_rank |
| Depth auxiliary on/off | With/without depth prediction head + AffineInvariantLoss | Spatial reasoning benchmarks (RefCOCO, counting) |
| RGBD input vs RGB-only | 4-channel RGBD input vs standard 3-channel RGB | All benchmarks, especially depth-related questions |

### 9.9 Logging Implementation

Metrics should be logged via the existing `dl_techniques.utils.logger` framework and exported to TensorBoard/W&B:

```python
# Core metrics -- every step
logger.log_scalar("train/loss", loss, step)
logger.log_scalar("train/perplexity", math.exp(loss), step)
logger.log_scalar("train/grad_norm", grad_norm, step)

# Clifford metrics -- every 100 steps
for i, block in enumerate(model.lm_decoder.blocks):
    gamma = block.ggr.gamma.numpy()
    logger.log_scalar(f"clifford/gamma_block_{i}", float(np.mean(gamma)), step)

# SRGP component norms -- every 100 steps
inner_norms, wedge_norms = [], []
for block in model.lm_decoder.blocks:
    # Hook into SRGP forward pass to capture intermediate values
    inner_norms.append(block.srgp.last_inner_norm)
    wedge_norms.append(block.srgp.last_wedge_norm)
logger.log_scalar("clifford/inner_norm", float(np.mean(inner_norms)), step)
logger.log_scalar("clifford/wedge_norm", float(np.mean(wedge_norms)), step)

# Validation -- every 500 steps
if step % 500 == 0:
    val_loss = evaluate(model, val_dataset)
    logger.log_scalar("val/loss", val_loss, step)
    logger.log_scalar("val/perplexity", math.exp(val_loss), step)
```

### 9.10 Early Stopping & Checkpoint Strategy

| Criterion | Action | Rationale |
|-----------|--------|-----------|
| val/loss not decreasing for 5K steps | Reduce LR by 0.5x | Learning rate may be too high |
| val/loss not decreasing for 15K steps | Stop training, evaluate | Convergence reached |
| train/grad_norm > 100 for 50 consecutive steps | Stop, investigate | Training instability |
| clifford/layerscale_gamma_mean < 1e-4 after 30% of training | Log warning | Clifford components may not be learning |
| val/text_ppl increasing while val/loss decreasing | Add more text-only data | Catastrophic forgetting of language |

**Checkpointing**: Save every 2K steps during Stage 1, every 5K steps during Stage 2/3. Keep best 3 checkpoints by val/loss. Always save at stage transitions.

---

## 10. Model Variants & Scaling

### 10.1 Proposed Variants

Based on existing CliffordNet/CliffordNetLM/CliffordCLIP variant ladders:

| Variant | Vision (ch/depth/shifts) | LM (ch/depth/shifts) | Projector | Total Params | Target |
|---------|:-:|:-:|:-:|:-:|---|
| **nano** | 128/8/[1,2] | 128/12/[1,2] | 1-block SRGP | ~12M | Research/prototyping |
| **mini** | 192/12/[1,2,4] | 192/12/[1,2,4] | 1-block SRGP | ~25M | Edge/mobile |
| **small** | 192/15/[1,2,4] | 192/18/[1,2,4] | 1-block SRGP | ~35M | Efficient deployment |
| **base** | 256/16/[1,2,4,8] | 384/18/[1,2,4,8,16] | 2-block SRGP | ~65M | General purpose |
| **large** | 384/20/[1,2,4,8,16] | 512/20/[1,2,4,8,16] | 2-block SRGP | ~150M | High quality |

### 10.2 Dimension Mismatch Handling

When vision channels != LM channels (e.g., vision=256, LM=384), the projector handles the dimension change:

```
vision_features (B, N, 256)
  -> Dense(384)               [dimension alignment]
  -> SRGP + GGR               [geometric processing at D=384]
  -> Dense(384)               [final projection]
projected (B, N, 384)
```

### 10.3 Comparison with Existing VLMs

| Model | Vision | LM | Params | Architecture | Complexity |
|-------|--------|-----|--------|---|---|
| LLaVA-1.5-7B [[Liu 2023]](#13-references) | ViT-L/14 (304M) | Vicuna-7B | 7.3B | Attention-based | O(N^2) |
| Penguin-VL-2B [[Zhang 2026]](#13-references) | LLM-init (400M) | Qwen3-1.7B | 2.1B | Attention-based | O(N^2) |
| SmolVLM-256M [[Allal 2025]](#13-references) | SigLIP-so400m | SmolLM-135M | 256M | Attention-based | O(N^2) |
| **CliffordVLM-nano** | **CliffordNet-128/8 (~3M)** | **CliffordNetLM-128/12 (~5M)** | **~12M** | **Attention-free** | **O(N)** |
| **CliffordVLM-base** | **CliffordNet-256/16 (~15M)** | **CliffordNetLM-384/18 (~30M)** | **~65M** | **Attention-free** | **O(N)** |

CliffordVLM targets the **ultra-efficient** end of the spectrum. With O(N) complexity (no quadratic attention), it should be significantly faster at inference than attention-based VLMs of similar parameter count.

Note: Penguin-VL-2B achieves state-of-the-art results among 2B models (DocVQA 94.1, MathVista 67.3) [[Zhang 2026]](#13-references). CliffordVLM at ~65M params is 30x smaller and will not match these scores, but the goal is to establish whether Clifford geometric algebra is a viable alternative to attention for cross-modal fusion, not to compete on absolute benchmarks.

### 10.4 Compute Requirements

| Variant | Training Stage 1 | Training Stage 2 | Inference |
|---------|-----------------|-----------------|-----------|
| nano (12M) | 1x RTX 4070 (12GB), ~2h | 1x RTX 4090 (24GB), ~8h | CPU or any GPU |
| mini (25M) | 1x RTX 4090, ~4h | 1x RTX 4090, ~16h | Any GPU |
| base (65M) | 1x RTX 4090, ~12h | 1x RTX 4090, ~2 days | RTX 4070+ |
| large (150M) | 1x RTX 4090, ~1 day | 1x RTX 4090, ~4 days | RTX 4090 |

(Estimates based on CliffordNet's O(N) scaling; actual times depend on data pipeline speed.)

---

## 11. Open Questions & Risks

### 11.1 Critical Questions

**Q1: Is the causal convolution receptive field sufficient for vision-language grounding?**

The CausalCliffordNetBlock has a local RF of ~5 positions per block. With L=12 blocks, the effective RF is ~49 positions. For 196 vision tokens, this means each text token can only "see" about 25% of the vision tokens through local convolutions. The global cumulative mean compensates, but it's a compressed summary.

*Mitigation*: Test with global context enabled. If insufficient, move to Architecture B (cross-fusion) or increase shifts to [1,2,4,8,16,32,64] for longer-range mixing.

**Q2: Can the geometric product learn cross-modal interactions as effectively as cross-attention?**

The geometric product has been proven in same-modality interactions (vision-vision in CliffordNet, text-text in CliffordNetLM). Cross-modal use (vision-text) is untested. The key risk is that vision and text representations may occupy different subspaces where the cyclic shift approximation doesn't capture meaningful interactions.

*Mitigation*: The vision projector aligns representations before they enter the decoder. The LayerScale gating ensures the model can fall back to "ignore the geometric content" if it's not helpful. Start with Architecture A (simple concatenation) to validate the basic pipeline before adding cross-fusion.

**Q3: How does the lack of attention affect generation quality?**

Attention-based LMs can dynamically route information from any position to any other position. CliffordNetLM relies on local convolutions + cumulative mean, which provides less flexible routing. For VLM tasks requiring precise grounding ("what is the third item from the left?"), this may be a limitation.

*Mitigation*: The exponential shifts (1,2,4,8,16) provide O(log D) path length between any two positions, similar to how dilated convolutions work. The global context branch adds global information. If still insufficient, consider adding a small number of attention layers (hybrid approach) or using Architecture B's cross-fusion blocks.

**Q4: What is the optimal tokenizer for CliffordNetLM?**

The codebase has a BPE tokenizer implementation (`layers/tokenizers/bpe.py`). For VLM use, we need:
- Compatible with existing pre-trained LM weights (if using transfer from CliffordNetLM)
- Large enough vocabulary for instruction-following (32K-50K tokens)
- Support for special tokens: `<image>`, `<|im_start|>`, `<|im_end|>`, etc.

*Recommendation*: Use the GPT-2 [[Radford 2019]](#13-references) tokenizer (50257 tokens) for compatibility with existing CliffordCLIP training, or train a custom BPE tokenizer on the target corpus.

**Q5: Should the vision encoder use Penguin-style LLM initialization?**

Penguin-VL [[Zhang 2026]](#13-references) initializes its vision encoder from an LLM (Qwen3-0.6B), converting causal attention to bidirectional. For CliffordVLM, the analogous approach would be:
- Initialize the CliffordNet vision encoder from pre-trained CliffordNetLM weights
- Convert CausalCliffordNetBlock to CliffordNetBlock (remove causal padding, replace cumulative mean with GAP)

This is architecturally trivial since both blocks share 95% of their code -- the only differences are padding mode and global context computation. The LM weights would provide "rich linguistic knowledge from the outset" (Penguin-VL), potentially helping the vision encoder produce features already aligned with the LM's representation space.

*Assessment*: Worth testing as an ablation against CliffordCLIP initialization and random initialization. Penguin-VL's ablation shows +3.3 absolute improvement from LLM initialization vs random.

**Q6: Does the Clifford depth decoder improve VLM spatial reasoning?**

The hypothesis is that training a CliffordNet depth decoder alongside the VLM forces the shared backbone to learn depth-aware features, which should improve answers to spatial questions ("which is closer?", "how many objects in the foreground?"). The wedge product between visual features and depth embeddings (Strategy 3) provides a uniquely geometric channel for 3D reasoning.

*Key test*: Compare CliffordVLM with and without depth decoder on spatial reasoning benchmarks (RefCOCO, counting tasks, CV-Bench-3D). If the depth auxiliary improves spatial benchmarks without hurting general ones, it validates the geometric-depth integration thesis.

*Risk*: The depth auxiliary may compete for backbone capacity with the VLM objective, hurting general performance. Mitigate by keeping `lambda_depth` small (0.1) and monitoring both val/loss and depth/affine_invariant_loss.

### 11.2 Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Geometric product doesn't learn useful cross-modal interactions | Medium | High | LayerScale gating, fallback to simple projector |
| Limited RF causes poor grounding | Medium | Medium | Global context, more shifts, Architecture B |
| CliffordNetLM not competitive with transformer LMs at same scale | Medium | High | Benchmark CliffordNetLM separately first |
| Training instability with mixed vision-text sequences | Low | Medium | Stage 1 alignment freezes most parameters |
| Positional embedding interpolation fails for resolution changes | Low | Low | Use relative positions or RoPE variant |

### 11.3 Things We Won't Know Until We Try

1. Whether the wedge product provides measurable benefit over inner-product-only fusion
2. Whether the O(N) complexity advantage translates to real-world speedups at VLM scales
3. Whether the isotropic architecture's lack of hierarchical features hurts VLM performance
4. What the optimal number of vision tokens is for the concatenation approach
5. Whether CliffordCLIP pre-training is significantly better than ImageNet pre-training for the vision tower
6. Whether CliffordNetLM-to-CliffordNet weight transfer (Penguin-style LLM init) produces better vision features than training from scratch
7. Whether reconstruction losses (amplitude/direction/relation) interact beneficially with the CliffordNet dual-stream architecture
8. Whether the limited local RF (5 positions per block) causes systematic failures on spatial reasoning benchmarks (RefCOCO, counting tasks)
9. Whether the SRGP can serve as an effective JEPA predictor (predicting masked patch embeddings from context), and whether this produces better vision features than contrastive pre-training for downstream VLM use
10. Whether a Clifford-native depth decoder (CliffordNetBlocks + 1x1 Conv) can match Depth Anything V2's quality at a fraction of the parameters, and whether the depth auxiliary signal measurably improves spatial VLM reasoning
11. Whether the wedge product between visual features and depth embeddings (Strategy 3) encodes meaningful 3D structural information (occlusion boundaries, depth discontinuities) that dot-product-based fusion cannot capture

---

## 12. Implementation Roadmap

### Phase 1: Validate Components (1-2 weeks)

1. **Benchmark CliffordNetLM**: Train CliffordNetLM-mini on a small text corpus, compare perplexity with GPT-2-tiny [[Radford 2019]](#13-references) at similar parameter count
2. **Complete CliffordCLIP training**: Finish CC3M training, extract vision tower weights
3. **Test vision feature extraction**: Remove classification head from CliffordNet, verify dense spatial features are usable

### Phase 2: Build CliffordVLM Architecture A (2-3 weeks)

1. **Implement CliffordVLM model** (`models/cliffordnet/vlm.py`):
   - CliffordNet vision backbone (from pre-trained weights)
   - Clifford vision projector (SRGP-based, 1 block)
   - CliffordNetLM decoder (from pre-trained weights)
   - Token concatenation pipeline
   - `from_variant()` factory method
   - Full `get_config()`/`from_config()` round-trip
2. **Implement training pipeline** (`train/cliffordnet/train_vlm.py`):
   - Stage 1: Vision encoder + projector alignment (LM decoder frozen, per Section 8.1)
   - Stage 2: Full VLM pre-training (all parameters trainable)
   - Stage 3: SFT on instruction-following data
3. **Tests** (`tests/test_models/test_cliffordnet/test_cliffordnet_vlm.py`):
   - Forward pass, gradient flow, serialization, causality

### Phase 3: Train & Evaluate (2-4 weeks)

1. **Stage 1 training**: Feature alignment (Phase 1.1 low-res + Phase 1.2 high-res)
2. **Stage 2 training**: Full VLM pre-training on multi-task mix
3. **Stage 3 training**: SFT on instruction-following data
4. **Evaluation**: Run full benchmark suite (Section 9.7) after each stage
5. **Monitor Clifford metrics** (Section 9.2) throughout -- especially inner/wedge ratio and LayerScale gamma evolution
6. **Ablation studies** (see Section 9.8):
   - SRGP projector vs simple 2-layer MLP projector
   - `cli_mode="full"` vs `cli_mode="inner"` (isolate wedge contribution)
   - With vs without global context branch
   - Shift sets: [1,2] vs [1,2,4] vs [1,2,4,8,16]
   - Vision token count: patch_size 8 (784) vs 16 (196)
   - CliffordCLIP init vs CliffordNetLM-to-CliffordNet init (Penguin-style)
   - With vs without reconstruction losses in Stage 1.1

### Phase 4: Depth Integration (2-3 weeks)

1. **Implement Clifford depth decoder**: 2-4 CliffordNetBlocks (or BiasFreeCliffordNetBlocks) + 1x1 Conv -> depth map
2. **Add depth auxiliary loss**: `AffineInvariantLoss` against frozen Depth Anything V2 pseudo-GT
3. **Train CliffordVLM-nano with depth auxiliary** and compare spatial reasoning benchmarks
4. **Ablation**: Strategy 1 (auxiliary only) vs Strategy 2 (end-to-end) vs Strategy 3 (depth-conditioned fusion)
5. **Evaluate**: NYUv2, KITTI depth benchmarks + CV-Bench-3D spatial reasoning

### Phase 5: Architecture B - Cross-Fusion (4-6 weeks)

1. **Implement CliffordCrossBlock** layer
2. **Integrate into CliffordVLM decoder** (every K-th layer gets a cross-fusion block)
3. **Train and compare** with Architecture A
4. **Ablation**: Number and placement of cross-fusion blocks

### Phase 6: Optimization & Deployment (2-4 weeks)

1. **Quantization**: Test INT8/INT4 inference
2. **Speed benchmarks**: Compare with SmolVLM [[Allal 2025]](#13-references), NanoVLM at similar scales
3. **ONNX export**: For deployment
4. **Documentation**: Complete model documentation, training recipes

---

## 13. References

**CliffordNet Family**

- **[Ji 2026]** Ji, S. (2026). CliffordNet: All You Need is Geometric Algebra. arXiv:2601.06793v2.
- **[Markou 2026]** Markou, N. (2026). Extending CliffordNet to Language Modeling and Image Denoising: A Keras 3 Implementation and Novel Applications.

**Clifford / Geometric Algebra**

- **[Hestenes & Sobczyk 1984]** Hestenes, D., Sobczyk, G. (1984). *Clifford Algebra to Geometric Calculus*. Springer.
- **[Brandstetter 2023]** Brandstetter, J., et al. (2023). Geometric Clifford Algebra Networks. ICML. arXiv:2302.06594.

**Vision Encoders & Foundation Models**

- **[Dosovitskiy 2021]** Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT). ICLR. arXiv:2010.11929.
- **[Oquab 2024]** Oquab, M., et al. (2024). DINOv2: Learning Robust Visual Features without Supervision. TMLR. arXiv:2304.07193.

**Vision-Language Models**

- **[Radford 2021]** Radford, A., et al. (2021). Learning Transferable Visual Representations from Natural Language Supervision (CLIP). ICML. arXiv:2103.00020.
- **[Liu 2023]** Liu, H., et al. (2023). Visual Instruction Tuning (LLaVA). NeurIPS. arXiv:2304.08485.
- **[Li 2023]** Li, J., et al. (2023). BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. ICML. arXiv:2301.12597.
- **[Zhang 2026]** Zhang, B., et al. (2026). Penguin-VL: Exploring the Efficiency Limits of VLM with LLM-based Vision Encoders. arXiv:2603.06569v2.
- **[Zhai 2023]** Zhai, X., et al. (2023). Sigmoid Loss for Language Image Pre-Training (SigLIP). arXiv:2303.15343.

**Efficient VLMs & Language Models**

- **[Allal 2025]** Allal, L.B., et al. (2025). SmolVLM: Redefining Small and Efficient Multimodal Models. arXiv:2504.05299.
- **[Radford 2019]** Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners (GPT-2). OpenAI Technical Report.

**JEPA Family**

- **[LeCun 2022]** LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. Meta AI Technical Report.
- **[Assran 2023]** Assran, M., et al. (2023). Self-supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA). CVPR. arXiv:2301.08243.
- **[Bardes 2024]** Bardes, A., et al. (2024). Revisiting Feature Prediction for Learning Visual Representations from Video (V-JEPA). arXiv:2312.06742.
- **[Assran 2025]** Assran, M., et al. (2025). V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning. arXiv:2506.09985.
- **[Chen 2025]** Chen, Y., et al. (2025). VL-JEPA: Joint Embedding Predictive Architecture for Vision-language. arXiv:2512.10942.
- **[Tong 2024]** Tong, Z., et al. (2024). Connecting Joint-Embedding Predictive Architecture with Contrastive Self-supervised Learning (C-JEPA). arXiv:2410.19560.

**Monocular Depth Estimation**

- **[Yang 2024]** Yang, L., et al. (2024). Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data. CVPR. arXiv:2401.10891.
- **[Yang 2024b]** Yang, L., et al. (2024). Depth Anything V2. arXiv:2406.09414.
- **[Hu 2024]** Hu, M., et al. (2024). Metric3D v2: A Versatile Monocular Geometric Foundation Model. arXiv:2404.15506.
- **[Bochkovskii 2024]** Bochkovskii, A., et al. (2024). Depth Pro: Sharp Monocular Metric Depth in Less Than a Second. arXiv:2410.02073.
- **[Ke 2024]** Ke, B., et al. (2024). Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation (Marigold). CVPR. arXiv:2312.02145.
- **[Eigen 2014]** Eigen, D., Puhrsch, C., Fergus, R. (2014). Depth Map Prediction from a Single Image using a Multi-Scale Deep Network. NeurIPS. arXiv:1406.2283.

**Architecture Patterns**

- **[Yu 2022]** Yu, W., et al. (2022). MetaFormer is Actually What You Need for Vision. CVPR.

---

## Appendix A: CliffordVLM Pseudocode

```python
class CliffordVLM(keras.Model):
    """Clifford geometric algebra Vision-Language Model."""

    def __init__(self, vision_config, lm_config, projector_config, ...):
        # Vision tower: CliffordNet backbone (no classification head)
        self.vision_backbone = CliffordNetBackbone(**vision_config)

        # Clifford vision projector
        self.vision_projector = CliffordVisionProjector(**projector_config)

        # Language model: CliffordNetLM decoder (modified for pre-embedded input)
        self.lm_decoder = CliffordNetLMDecoder(**lm_config)

        # Token/positional embeddings for text
        self.token_embedding = Embedding(vocab_size, lm_channels)
        self.pos_embedding = Embedding(max_seq_len, lm_channels)

    def call(self, inputs):
        images = inputs["image"]       # (B, H, W, 3)
        tokens = inputs["text"]        # (B, T)

        # 1. Extract vision features
        vision_features = self.vision_backbone(images)  # (B, N, D_vis)

        # 2. Project vision features through Clifford projector
        vision_tokens = self.vision_projector(vision_features)  # (B, N, D_lm)

        # 3. Embed text tokens
        text_embeds = self.token_embedding(tokens)  # (B, T, D_lm)
        text_embeds += self.pos_embedding(positions[N:N+T])

        # 4. Concatenate vision + text tokens
        combined = concat([vision_tokens, text_embeds], axis=1)  # (B, N+T, D_lm)

        # 5. Run through causal CliffordNet decoder
        logits = self.lm_decoder(combined)  # (B, N+T, vocab_size)

        # 6. Return only text position logits
        return {"logits": logits[:, N:, :]}


class CliffordVisionProjector(keras.layers.Layer):
    """Geometric projector mapping vision features to LM space."""

    def __init__(self, vision_dim, lm_dim, shifts=[1,2,4], ...):
        self.dim_align = Dense(lm_dim)
        self.srgp = SparseRollingGeometricProduct(lm_dim, shifts, cli_mode="full")
        self.ggr = GatedGeometricResidual(lm_dim, layer_scale_init=1e-5)
        self.norm = LayerNormalization()
        self.proj = Dense(lm_dim)

    def call(self, vision_features):
        z = self.dim_align(vision_features)     # (B, N, D_lm)
        z_det = Dense(z)                         # detail stream
        z_ctx = LearnedQueryPool(z)              # context stream
        geo = self.srgp(z_det, z_ctx)            # geometric product
        mixed = self.ggr(z, geo)                 # gated residual
        return self.proj(self.norm(mixed))        # project
```

## Appendix B: Why Not Just Use Cross-Attention?

A natural question: why build a CliffordVLM when we could just add cross-attention layers to CliffordNetLM?

| Aspect | Cross-Attention | Clifford Geometric Product |
|--------|----------------|---------------------------|
| **Complexity** | O(T*N*D) per layer | O(T*|S|*D) per layer (|S|=5 typical) |
| **Information captured** | Scalar similarity (dot product) | Scalar alignment + structural difference |
| **Parameters** | Q,K,V projections: 3*D^2 | Shift-based: ~|S|*D (much fewer) |
| **Inductive bias** | Content-based routing | Local + global context + geometric algebra |
| **Architectural purity** | Breaks the attention-free promise | Maintains fully Clifford-native architecture |

The CliffordVLM approach is not necessarily *better* than cross-attention -- it's *different*. The goal is to explore whether geometric algebra provides a competitive alternative to attention for cross-modal fusion, with potential advantages in efficiency and expressiveness.

If the geometric approach proves insufficient for certain tasks, a hybrid approach (Clifford blocks + occasional attention layers) remains an option, but should be explored only after pure-Clifford baselines are established.
