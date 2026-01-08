# JEPA Family: Complete Technical Guide
## Joint Embedding Predictive Architectures (2022-2025)

**Author:** Technical Reference Guide  
**Last Updated:** January 2026  
**Status:** State-of-the-Art

---

## Table of Contents

1. [Introduction and Philosophy](#1-introduction-and-philosophy)
2. [Core JEPA Architecture](#2-core-jepa-architecture)
3. [I-JEPA: Image JEPA](#3-i-jepa-image-jepa)
4. [V-JEPA: Video JEPA](#4-v-jepa-video-jepa)
5. [V-JEPA 2 and V-JEPA 2-AC](#5-v-jepa-2-and-v-jepa-2-ac)
6. [VL-JEPA: Vision-Language JEPA](#6-vl-jepa-vision-language-jepa)
7. [C-JEPA: Contrastive JEPA](#7-c-jepa-contrastive-jepa)
8. [A-JEPA: Audio JEPA](#8-a-jepa-audio-jepa)
9. [Collapse Prevention Mechanisms](#9-collapse-prevention-mechanisms)
10. [Training Procedures](#10-training-procedures)
11. [Implementation Blueprints](#11-implementation-blueprints)
12. [Performance Benchmarks](#12-performance-benchmarks)
13. [Practical Recommendations](#13-practical-recommendations)

---

## 1. Introduction and Philosophy

### 1.1 The Problem with Generative Models

Traditional self-supervised learning approaches fall into two categories:

**Generative Models (MAE, BERT, GPT):**
- Reconstruct pixels or tokens directly
- Must model ALL variations including noise, lighting, texture
- Waste capacity on unpredictable low-level details
- Pixel-level loss averages over countless possible futures → blurry predictions

**Contrastive Models (SimCLR, CLIP, DINO):**
- Learn invariance to hand-crafted augmentations
- Augmentations encode human biases about what should be invariant
- May discard information useful for downstream tasks
- Require carefully designed negative sampling

### 1.2 The JEPA Solution

**Joint Embedding Predictive Architecture (JEPA)** was proposed by Yann LeCun in his 2022 position paper "A Path Towards Autonomous Machine Intelligence" as a foundation for building world models.

**Core Principle:** Predict representations in abstract embedding space, not pixel/token space.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     GENERATIVE vs JEPA                              │
│                                                                     │
│  GENERATIVE:                                                        │
│  Input ──► Encoder ──► Latent ──► Decoder ──► Reconstructed Pixels  │
│                                                    │                │
│                                              Pixel-level loss       │
│                                              (must predict noise,   │
│                                               texture, everything)  │
│                                                                     │
│  JEPA:                                                              │
│  Input x ──► Encoder ──► sₓ ──► Predictor ──► ŝᵧ                    │
│                                               │                     │
│  Input y ──► Encoder ──► sᵧ ◄─────────────────┘                     │
│                          │                                          │
│                    Embedding-level loss                             │
│                    (only semantic content)                          │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 Key Advantages

| Aspect | Generative | JEPA |
|--------|------------|------|
| Prediction target | Raw pixels/tokens | Abstract embeddings |
| Handles uncertainty | Poorly (blurry average) | Naturally (encoder ignores noise) |
| Compute efficiency | Must decode everything | No decoder needed |
| Representation quality | Often low-level | Semantic by design |
| Augmentation dependence | High (contrastive) | None required |

### 1.4 The Moravec Paradox Connection

The Moravec paradox states that tasks easy for humans (perception, motor control) are hard for AI, while tasks hard for humans (computation) are easy for AI. 

JEPA addresses this by:
- Learning from observation (like infants)
- Building internal world models
- Focusing on semantic understanding rather than surface reconstruction
- Enabling planning and reasoning in abstract space

---

## 2. Core JEPA Architecture

### 2.1 Fundamental Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                    JEPA CORE ARCHITECTURE                           │
│                                                                     │
│                                                                     │
│   ┌───────────────┐                                                 │
│   │   Input x     │  (e.g., visible portion of image/video)         │
│   │   (context)   │                                                 │
│   └───────┬───────┘                                                 │
│           │                                                         │
│           ▼                                                         │
│   ┌───────────────┐                                                 │
│   │   Context     │  Trainable encoder                              │
│   │   Encoder fₓ  │  (typically ViT)                                │
│   └───────┬───────┘                                                 │
│           │                                                         │
│           ▼                                                         │
│   ┌───────────────┐     ┌───────────────┐                           │
│   │   Context     │     │   Positional  │                           │
│   │   Embedding   │ + ──│   Mask Tokens │  (where to predict)       │
│   │      sₓ       │     │               │                           │
│   └───────┬───────┘     └───────────────┘                           │
│           │                                                         │
│           ▼                                                         │
│   ┌───────────────┐                                                 │
│   │   Predictor   │  Trainable (narrower than encoder)              │
│   │      g        │  Outputs predicted target embeddings            │
│   └───────┬───────┘                                                 │
│           │                                                         │
│           ▼                                                         │
│   ┌───────────────┐         ┌───────────────┐                       │
│   │   Predicted   │         │   Target      │                       │
│   │   Embedding   │◄─Loss──►│   Embedding   │                       │
│   │      ŝᵧ        │         │      sᵧ       │                       │
│   └───────────────┘         └───────┬───────┘                       │
│                                     │                               │
│                                     ▲                               │
│                             ┌───────┴───────┐                       │
│                             │    Target     │  EMA of context       │
│                             │   Encoder fᵧ  │  encoder (frozen)     │
│                             └───────┬───────┘                       │
│                                     │                               │
│                                     ▲                               │
│                             ┌───────┴───────┐                       │
│                             │   Input y     │  (masked region)      │
│                             │   (target)    │                       │
│                             └───────────────┘                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Details

**Context Encoder (fₓ):**
- Typically a Vision Transformer (ViT)
- Processes only visible/unmasked portions
- Fully trainable
- Outputs sequence of patch embeddings

**Target Encoder (fᵧ):**
- Same architecture as context encoder
- Weights are Exponential Moving Average (EMA) of context encoder
- Stop-gradient applied (no backprop through target encoder)
- Processes the full target region (unmasked)

**Predictor (g):**
- Narrower/shallower transformer than encoders
- Takes context embeddings + positional information for targets
- Outputs predicted embeddings for masked regions
- Key: Must be less powerful than encoder to prevent trivial solutions

### 2.3 Loss Function

The fundamental JEPA loss is simple regression in embedding space:

```python
# L2 Loss (most common)
L = mean((ŝᵧ - sᵧ)²)

# L1 Loss (alternative, used in some V-JEPA variants)
L = mean(|ŝᵧ - sᵧ|)

# Smooth L1 (Huber loss, robust to outliers)
L = smooth_l1(ŝᵧ, sᵧ)
```

**Why this works:**
- Target encoder provides stable targets (EMA smoothing)
- Stop-gradient prevents collapse to trivial solution
- Predictor must actually learn to model the world
- Embedding space captures semantics, not noise

### 2.4 Information Flow

```
Training Step:
1. Sample input (image/video)
2. Generate context mask (what model sees) and target masks (what to predict)
3. Context encoder: visible patches → context embeddings
4. Target encoder (EMA, no grad): target patches → target embeddings
5. Predictor: context embeddings + target positions → predicted embeddings
6. Loss: compare predicted vs actual target embeddings
7. Update context encoder and predictor via backprop
8. Update target encoder via EMA

Inference (downstream tasks):
- Only context encoder is used
- No predictor needed (unless doing generative tasks)
- Features extracted for classification, detection, etc.
```

---

## 3. I-JEPA: Image JEPA

**Paper:** "Self-supervised Learning from Images with a Joint-Embedding Predictive Architecture" (CVPR 2023)  
**Authors:** Assran et al. (Meta AI)  
**Code:** https://github.com/facebookresearch/ijepa

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         I-JEPA ARCHITECTURE                         │
│                                                                     │
│  Input Image (224×224)                                              │
│  ┌─────────────────────────────┐                                    │
│  │ ┌───┬───┬───┬───┬───┬───┐   │  Patch grid: 14×14 patches         │
│  │ │   │   │   │   │   │   │   │  (16×16 pixels each)               │
│  │ ├───┼───┼───┼───┼───┼───┤   │                                    │
│  │ │   │ T1│ T1│   │   │   │   │  T1-T4: Target blocks              │
│  │ ├───┼───┼───┼───┼───┼───┤   │  (15-20% of image each)            │
│  │ │   │ T1│ T1│ T2│ T2│   │   │                                    │
│  │ ├───┼───┼───┼───┼───┼───┤   │  Context: Everything NOT in        │
│  │ │   │   │ T3│ T3│ T2│   │   │  any target (85-100% coverage)     │
│  │ ├───┼───┼───┼───┼───┼───┤   │                                    │
│  │ │ T4│ T4│ T3│ T3│   │   │   │                                    │
│  │ ├───┼───┼───┼───┼───┼───┤   │                                    │
│  │ │ T4│ T4│   │   │   │   │   │                                    │
│  │ └───┴───┴───┴───┴───┴───┘   │                                    │
│  └─────────────────────────────┘                                    │
│                                                                     │
│  Masking Strategy:                                                  │
│  • 4 target blocks, each 15-20% of image                            │
│  • Blocks can overlap with each other                               │
│  • Context = one large block (85-100%) minus target regions         │
│  • High masking ratio (~75-90% masked)                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Multi-Block Masking Strategy

```python
def generate_ijepa_masks(image_size=224, patch_size=16, num_targets=4):
    """
    I-JEPA multi-block masking strategy.
    
    Returns:
        context_mask: Boolean mask for visible patches
        target_masks: List of boolean masks for each target region
    """
    num_patches = (image_size // patch_size) ** 2  # 196 for 224/16
    grid_size = image_size // patch_size  # 14
    
    target_masks = []
    all_target_patches = set()
    
    for _ in range(num_targets):
        # Target block size: 15-20% of image
        block_h = random.randint(int(0.15 * grid_size), int(0.2 * grid_size))
        block_w = random.randint(int(0.15 * grid_size), int(0.2 * grid_size))
        
        # Random position
        start_h = random.randint(0, grid_size - block_h)
        start_w = random.randint(0, grid_size - block_w)
        
        # Create mask
        mask = np.zeros((grid_size, grid_size), dtype=bool)
        mask[start_h:start_h+block_h, start_w:start_w+block_w] = True
        target_masks.append(mask.flatten())
        
        # Track all target patches
        for h in range(start_h, start_h + block_h):
            for w in range(start_w, start_w + block_w):
                all_target_patches.add(h * grid_size + w)
    
    # Context: large block covering most of image, minus targets
    context_mask = np.ones(num_patches, dtype=bool)
    for idx in all_target_patches:
        context_mask[idx] = False
    
    return context_mask, target_masks
```

### 3.3 Key Design Choices

**No Hand-Crafted Augmentations:**
- Unlike DINO, SimCLR, BYOL: no color jitter, blur, crop
- Only random resize crop for efficiency
- Masking alone drives representation learning
- Avoids encoding human biases about invariances

**Asymmetric Design:**
- Context encoder: processes ~25% of patches (unmasked)
- Target encoder: processes ~15-20% per target block
- Predictor: narrower than encoder (prevents shortcut)

**Efficiency:**
- Only one view needed (vs two views in contrastive methods)
- Context encoder sees fewer patches
- ~5× fewer GPU hours than MAE

### 3.4 Training Configuration

```yaml
# I-JEPA training configuration
model:
  encoder: vit_huge_patch14  # ViT-H/14
  predictor_depth: 12
  predictor_embed_dim: 384   # Narrower than encoder (1280)
  
masking:
  num_targets: 4
  target_aspect_ratio: [0.75, 1.5]
  target_scale: [0.15, 0.2]
  context_scale: [0.85, 1.0]

training:
  epochs: 300
  batch_size: 2048
  base_lr: 0.001
  warmup_epochs: 40
  
  # EMA schedule
  ema_decay_base: 0.996
  ema_decay_final: 1.0  # Increases during training
  
optimizer:
  type: adamw
  weight_decay: 0.05
  betas: [0.9, 0.95]
```

### 3.5 Predictor Visualization

The I-JEPA predictor learns semantic understanding, not pixel reconstruction:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PREDICTOR VISUALIZATION                          │
│                                                                     │
│  Original Image          Context (visible)      Predicted Content   │
│  ┌─────────────┐         ┌─────────────┐        ┌─────────────┐     │
│  │    🐕       │         │    🐕       │        │   (sketch   │     │
│  │   /  \      │         │   /  \      │   ──►  │    of dog   │     │
│  │  dog's      │         │  [MASKED]   │        │    head)    │     │
│  │   head      │         │             │        │             │     │
│  └─────────────┘         └─────────────┘        └─────────────┘     │
│                                                                     │
│  Key insight: Predictor outputs SEMANTIC content                    │
│  - Correct object parts (dog's head, not random texture)            │
│  - Correct pose/orientation                                         │
│  - Position uncertainty is modeled                                  │
│  - NOT pixel-perfect reconstruction                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. V-JEPA: Video JEPA

**Paper:** "Revisiting Feature Prediction for Learning Visual Representations from Video" (2024)  
**Authors:** Bardes et al. (Meta AI)  
**Released:** February 2024

### 4.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         V-JEPA ARCHITECTURE                         │
│                                                                     │
│  Input Video: T frames × H × W × 3                                  │
│                                                                     │
│  Patchification: 16×16 spatial × 2-frame temporal                   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Frame 1      Frame 2      Frame 3      Frame 4              │   │
│  │  ┌───────┐    ┌───────┐    ┌───────┐    ┌───────┐            │   │
│  │  │░░░░░░░│    │░░░░░░░│    │░░░░░░░│    │░░░░░░░│            │   │
│  │  │░░███░░│    │░░███░░│    │░░███░░│    │░░███░░│  ← SAME    │   │
│  │  │░░███░░│    │░░███░░│    │░░███░░│    │░░███░░│    MASK    │   │
│  │  │░░░░░░░│    │░░░░░░░│    │░░░░░░░│    │░░░░░░░│            │   │
│  │  └───────┘    └───────┘    └───────┘    └───────┘            │   │
│  │                                                              │   │
│  │  ░ = Context (visible)    █ = Target (masked)                │   │
│  │                                                              │   │
│  │  Critical: Same spatial mask applied across ALL frames       │   │
│  │  Prevents trivial frame-to-frame copying                     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Spatiotemporal Masking

```python
def generate_vjepa_masks(num_frames, grid_h, grid_w, num_targets=8):
    """
    V-JEPA spatiotemporal masking.
    
    Key: Same spatial mask across all frames to prevent
    trivial frame-to-frame copying.
    """
    total_patches_per_frame = grid_h * grid_w
    
    target_masks = []
    all_masked_spatial = set()
    
    for _ in range(num_targets):
        # Spatial block (same across all frames)
        block_h = random.randint(int(0.1 * grid_h), int(0.25 * grid_h))
        block_w = random.randint(int(0.1 * grid_w), int(0.25 * grid_w))
        start_h = random.randint(0, grid_h - block_h)
        start_w = random.randint(0, grid_w - block_w)
        
        # Temporal extent
        start_t = random.randint(0, num_frames - 2)
        end_t = random.randint(start_t + 1, num_frames)
        
        # Create spatiotemporal mask
        mask = np.zeros((num_frames, grid_h, grid_w), dtype=bool)
        mask[start_t:end_t, start_h:start_h+block_h, start_w:start_w+block_w] = True
        target_masks.append(mask)
        
        # Track masked spatial regions
        for h in range(start_h, start_h + block_h):
            for w in range(start_w, start_w + block_w):
                all_masked_spatial.add((h, w))
    
    # Context: all frames, excluding masked spatial regions
    context_mask = np.ones((num_frames, grid_h, grid_w), dtype=bool)
    for (h, w) in all_masked_spatial:
        context_mask[:, h, w] = False  # Mask across ALL frames
    
    return context_mask, target_masks
```

### 4.3 Why Same Mask Across Frames?

```
┌─────────────────────────────────────────────────────────────────────┐
│               WHY CONSISTENT SPATIAL MASKING?                       │
│                                                                     │
│  Problem with different masks per frame:                            │
│                                                                     │
│  Frame 1         Frame 2         Prediction                         │
│  ┌───────┐       ┌───────┐       ┌───────┐                          │
│  │ 🐕    │       │ ███   │       │ Just  │                          │
│  │    ███│  ──►  │ 🐕    │  ──►  │ copy  │  ← TRIVIAL!              │
│  │       │       │       │       │ 🐕    │                          │
│  └───────┘       └───────┘       └───────┘                          │
│                                                                     │
│  The model can just copy visible patches from adjacent frames       │
│  No actual understanding required!                                  │
│                                                                     │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  Solution: Same spatial mask across all frames                      │
│                                                                     │
│  Frame 1         Frame 2         Prediction                         │
│  ┌───────┐       ┌───────┐       ┌───────┐                          │
│  │ 🐕 ███│       │ 🐕 ███│       │ Must  │                          │
│  │    ███│  ──►  │    ███│  ──►  │ infer │  ← MEANINGFUL            │
│  │       │       │       │       │ motion│                          │
│  └───────┘       └───────┘       └───────┘                          │
│                                                                     │
│  Model must understand motion and object dynamics                   │
│  Cannot cheat by copying from other frames                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.4 Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    V-JEPA TRAINING PIPELINE                         │
│                                                                     │
│  1. Sample video clip (T frames)                                    │
│                    │                                                │
│                    ▼                                                │
│  2. Patchify: 16×16×2 spatiotemporal patches                        │
│                    │                                                │
│                    ▼                                                │
│  3. Generate masks (same spatial mask across frames)                │
│                    │                                                │
│         ┌─────────┴─────────┐                                       │
│         ▼                   ▼                                       │
│  ┌──────────────┐   ┌──────────────┐                                │
│  │ Context      │   │ Target       │                                │
│  │ Encoder      │   │ Encoder      │                                │
│  │ (trainable)  │   │ (EMA)        │                                │
│  │              │   │ stop_grad    │                                │
│  └──────┬───────┘   └──────┬───────┘                                │
│         │                  │                                        │
│         ▼                  │                                        │
│  ┌──────────────┐          │                                        │
│  │ Predictor    │          │                                        │
│  │ + pos tokens │          │                                        │
│  └──────┬───────┘          │                                        │
│         │                  │                                        │
│         ▼                  ▼                                        │
│  ┌─────────────────────────────────┐                                │
│  │     L1 Loss in Embedding Space  │                                │
│  │     L = mean(|ŝᵧ - sᵧ|)          │                                │
│  └─────────────────────────────────┘                                │
│                    │                                                │
│                    ▼                                                │
│  4. Update context encoder + predictor (backprop)                   │
│  5. Update target encoder (EMA)                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.5 Key Differences from I-JEPA

| Aspect | I-JEPA | V-JEPA |
|--------|--------|--------|
| Input | Single image | Video clip |
| Patches | 2D (16×16) | 3D (16×16×2) |
| Masking | Random blocks | Same spatial mask across time |
| Target blocks | 4 | 8 |
| Temporal modeling | N/A | Implicit via prediction |
| Loss | L2 | L1 |

---

## 5. V-JEPA 2 and V-JEPA 2-AC

**Paper:** "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning" (arXiv 2506.09985, June 2025)  
**Authors:** Assran et al. (Meta AI)

### 5.1 V-JEPA 2 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      V-JEPA 2 ARCHITECTURE                          │
│                                                                     │
│  Key Innovation: 3D Rotary Position Embeddings (3D-RoPE)            │
│                                                                     │
│  Standard RoPE (1D):                                                │
│    Rotates Q/K vectors based on sequence position                   │
│    q'ᵢ = Rᵢ · qᵢ  where Rᵢ is rotation matrix for position i        │
│                                                                     │
│  3D-RoPE (V-JEPA 2):                                                │
│    Rotates Q/K vectors in 2D subspaces across THREE dimensions:     │
│    - Height dimension (spatial)                                     │
│    - Width dimension (spatial)                                      │
│    - Time dimension (temporal)                                      │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  Token at (h, w, t):                                       │     │
│  │                                                            │     │
│  │  q'ₕ,ᵥ,ₜ = R_h(h) · R_w(w) · R_t(t) · qₕ,ᵥ,ₜ               │     │
│  │                                                            │     │
│  │  Each R is a block-diagonal rotation matrix                │     │
│  │  Operating on different subspaces of the embedding         │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                     │
│  Benefits:                                                          │
│  • Unified handling of space and time                               │
│  • No separate positional encoding schemes                          │
│  • Naturally extends to longer sequences                            │
│  • Better temporal reasoning                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Scale and Training

```yaml
# V-JEPA 2 training configuration
model:
  architecture: vit_giant  # ViT-g
  patch_size: 14
  hidden_dim: 1664
  num_heads: 16
  num_layers: 48
  position_embedding: 3d_rope
  
training:
  dataset: internet_video
  dataset_size: 1_000_000+ hours  # Massive scale
  frames_per_clip: 16
  frame_rate: 8 fps
  resolution: 224×224
  
  # Two-stage training
  stage1:
    objective: masked_prediction
    epochs: 100
    
  stage2:  # Optional for V-JEPA 2-AC
    objective: action_conditioned
    robot_data: droid_dataset (62 hours)
```

### 5.3 V-JEPA 2-AC: Action-Conditioned Extension

```
┌─────────────────────────────────────────────────────────────────────┐
│                    V-JEPA 2-AC ARCHITECTURE                         │
│              (Action-Conditioned World Model)                       │
│                                                                     │
│  Input: Visual history + Action sequence                            │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                                                             │    │
│  │  Visual States         Actions                              │    │
│  │  [s₀, s₁, ..., sₜ]    [a₀, a₁, ..., aₜ₋₁]                   │    │
│  │       │                     │                               │    │
│  │       ▼                     ▼                               │    │
│  │  ┌──────────┐         ┌──────────┐                          │    │
│  │  │ V-JEPA 2 │         │ Action   │                          │    │
│  │  │ Encoder  │         │ Encoder  │                          │    │
│  │  │ (frozen) │         │ (MLP)    │                          │    │
│  │  └────┬─────┘         └────┬─────┘                          │    │
│  │       │                    │                                │    │
│  │       └────────┬───────────┘                                │    │
│  │                ▼                                            │    │
│  │       ┌────────────────┐                                    │    │
│  │       │   Concatenate  │                                    │    │
│  │       │   [vᵢ; aᵢ]     │                                    │    │
│  │       └───────┬────────┘                                    │    │
│  │               ▼                                             │    │
│  │       ┌────────────────────────┐                            │    │
│  │       │  Action-Conditioned    │                            │    │
│  │       │  Predictor             │                            │    │
│  │       │  (block-causal attn)   │                            │    │
│  │       └───────┬────────────────┘                            │    │
│  │               ▼                                             │    │
│  │       Predicted Future States                               │    │
│  │       [ŝₜ₊₁, ŝₜ₊₂, ..., ŝₜ₊ₖ]                               │    │
│  │                                                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  Training Losses:                                                   │
│  1. Teacher-forcing: Predict sₜ₊₁ given true s₀:ₜ and a₀:ₜ          │
│  2. Rollout: Multi-step prediction using own predictions            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.4 Planning with V-JEPA 2-AC

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MPC + CEM PLANNING                               │
│        (Model Predictive Control + Cross-Entropy Method)            │
│                                                                     │
│  Goal: Find action sequence that reaches target state               │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                                                             │    │
│  │  1. Sample N action sequences: {a⁽¹⁾, a⁽²⁾, ..., a⁽ᴺ⁾}      │    │
│  │                                                             │    │
│  │  2. For each sequence, rollout V-JEPA 2-AC:                 │    │
│  │     s₀ ──[a⁽ⁱ⁾]──► ŝ₁ ──[a⁽ⁱ⁾]──► ŝ₂ ──► ... ──► ŝₜ         │    │
│  │                                                             │    │
│  │  3. Score each trajectory:                                  │    │
│  │     score(a⁽ⁱ⁾) = -||ŝₜ - s_goal||²                         │    │
│  │                                                             │    │
│  │  4. Select top-K sequences (elite set)                      │    │
│  │                                                             │    │
│  │  5. Fit Gaussian to elite set                               │    │
│  │     μ_new = mean(elite actions)                             │    │
│  │     σ_new = std(elite actions)                              │    │
│  │                                                             │    │
│  │  6. Resample from updated distribution                      │    │
│  │                                                             │    │
│  │  7. Repeat steps 2-6 for M iterations                       │    │
│  │                                                             │    │
│  │  8. Execute first action of best sequence                   │    │
│  │                                                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  Result: Zero-shot robotic manipulation without task rewards!       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.5 Performance Highlights

| Benchmark | V-JEPA 2 Score | Notes |
|-----------|----------------|-------|
| Something-Something v2 | 77.3% top-1 | Strong temporal reasoning |
| Kinetics-400 | 87.4% top-1 | Action recognition |
| Robot Pick & Place | Zero-shot success | No task-specific training |
| Training data (robot) | 62 hours | From Droid dataset |

---

## 6. VL-JEPA: Vision-Language JEPA

**Paper:** "VL-JEPA: Joint Embedding Predictive Architecture for Vision-language" (arXiv 2512.10942, December 2025)  
**Authors:** Chen et al. (Meta AI)

### 6.1 Core Innovation

VL-JEPA predicts **continuous text embeddings** instead of discrete tokens.

```
┌─────────────────────────────────────────────────────────────────────┐
│                 TRADITIONAL VLM vs VL-JEPA                          │
│                                                                     │
│  Traditional VLM (GPT-4V, LLaVA, etc.):                             │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Image ──► Vision Encoder ──► Project ──► [LLM tokens]      │    │
│  │                                              │              │    │
│  │  Text Query ──────────────────────────────► [LLM tokens]    │    │
│  │                                              │              │    │
│  │                                              ▼              │    │
│  │                                    ┌───────────────┐        │    │
│  │                                    │ Autoregressive│        │    │
│  │                                    │ LLM Decoder   │        │    │
│  │                                    │ (predict next │        │    │
│  │                                    │  token)       │        │    │
│  │                                    └──────┬────────┘        │    │
│  │                                           │                 │    │
│  │                                           ▼                 │    │
│  │                                    Token by token           │    │
│  │                                    generation               │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  VL-JEPA:                                                           │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Image/Video ──► V-JEPA 2 Encoder ───┐                      │    │
│  │                    (frozen)          │                      │    │
│  │                                      ▼                      │    │
│  │  Text Query ──► Text Encoder ──► ┌──────────┐               │    │
│  │                                  │ Predictor│               │    │
│  │                                  └────┬─────┘               │    │
│  │                                       │                     │    │
│  │                                       ▼                     │    │
│  │                            Predicted Text Embedding         │    │
│  │                                       │                     │    │
│  │                    ┌──────────────────┼───────────────┐     │    │
│  │                    ▼                  ▼               ▼     │    │
│  │               Retrieval        Classification    [Optional] │    │
│  │            (cosine sim)      (nearest class)    Text Decoder│    │
│  │                                                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Architecture Details

```
┌─────────────────────────────────────────────────────────────────────┐
│                    VL-JEPA ARCHITECTURE                             │
│                                                                     │
│  Components:                                                        │
│                                                                     │
│  1. Vision Encoder (frozen V-JEPA 2)                                │
│     - Processes image/video                                         │
│     - Outputs sequence of visual embeddings                         │
│     - NOT fine-tuned (preserves world model knowledge)              │
│                                                                     │
│  2. Text Encoder                                                    │
│     - Encodes text queries/prompts                                  │
│     - Can be SONAR, sentence transformer, etc.                      │
│                                                                     │
│  3. Vision-Language Predictor                                       │
│     - Cross-attention between visual and text embeddings            │
│     - Outputs predicted text embedding                              │
│     - Much smaller than full LLM decoder                            │
│                                                                     │
│  4. Optional Text Decoder (lightweight)                             │
│     - Only invoked when text generation needed                      │
│     - Translates embedding → tokens                                 │
│     - Can be skipped for classification/retrieval                   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                                                             │    │
│  │   Image/Video                    Text Query                 │    │
│  │       │                              │                      │    │
│  │       ▼                              ▼                      │    │
│  │  ┌──────────────┐            ┌──────────────┐               │    │
│  │  │  V-JEPA 2    │            │ Text Encoder │               │    │
│  │  │  (frozen)    │            │              │               │    │
│  │  └──────┬───────┘            └──────┬───────┘               │    │
│  │         │                           │                       │    │
│  │         │    ┌──────────────────────┘                       │    │
│  │         │    │                                              │    │
│  │         ▼    ▼                                              │    │
│  │    ┌─────────────────┐                                      │    │
│  │    │ Cross-Attention │                                      │    │
│  │    │ Predictor       │                                      │    │
│  │    └───────┬─────────┘                                      │    │
│  │            │                                                │    │
│  │            ▼                                                │    │
│  │   Predicted Text Embedding (continuous)                     │    │
│  │            │                                                │    │
│  │   ┌────────┴──────┐                                         │    │
│  │   │   Task Head   │                                         │    │
│  │   │               │                                         │    │
│  │   │ - Retrieval: cosine distance to candidates              │    │
│  │   │ - Classification: nearest class embedding               │    │
│  │   │ - VQA: lightweight decoder when needed                  │    │
│  │   │ - Selective: decode only uncertain predictions          │    │
│  │   └───────────────┘                                         │    │
│  │                                                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.3 Selective Decoding

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SELECTIVE DECODING                               │
│                                                                     │
│  Key insight: Not all predictions need text generation              │
│                                                                     │
│  For Classification:                                                │
│  - Predict embedding                                                │
│  - Compare to class embeddings via cosine similarity                │
│  - NO decoder needed                                                │
│                                                                     │
│  For Retrieval:                                                     │
│  - Predict embedding                                                │
│  - Find nearest neighbors in database                               │
│  - NO decoder needed                                                │
│                                                                     │
│  For VQA/Captioning:                                                │
│  - Predict embedding                                                │
│  - Check confidence (embedding norm, margin, etc.)                  │
│  - If confident: nearest neighbor answer                            │
│  - If uncertain: invoke decoder                                     │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                                                             │    │
│  │  Prediction ──► Confidence Check ──┬──► High: Skip decoder  │    │
│  │                                    │                        │    │
│  │                                    └──► Low: Use decoder    │    │
│  │                                                             │    │
│  │  Result: 2.85× fewer decoding operations                    │    │
│  │          with same performance                              │    │
│  │                                                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.4 Training Configuration

```yaml
# VL-JEPA training configuration
model:
  vision_encoder: vjepa2_giant  # Frozen
  text_encoder: sonar_large     # Or sentence-transformers
  predictor:
    type: cross_attention_transformer
    layers: 6
    heads: 12
    hidden_dim: 768
  decoder:
    type: small_llm  # Llama-3.2-1B or similar
    usage: selective  # Only when needed

training:
  # Align vision and text embeddings
  loss: l2_embedding_prediction
  
  data:
    - video_caption_pairs
    - vqa_datasets
    - retrieval_datasets
    
  freeze:
    - vision_encoder  # Always frozen
  trainable:
    - predictor
    - text_encoder  # Optional
```

### 6.5 Performance Comparison

| Model | Params (trainable) | GQA | TallyQA | POPE | Retrieval Avg |
|-------|-------------------|-----|---------|------|---------------|
| InstructBLIP | ~8B | 49.2 | 68.1 | 85.3 | - |
| QwenVL | ~10B | 57.5 | 70.2 | 87.1 | - |
| CLIP | 428M | - | - | - | 38.2 |
| SigLIP2 | 400M | - | - | - | 41.5 |
| **VL-JEPA** | **1.6B** | **51.8** | **69.5** | **86.2** | **46.4** |

Key: VL-JEPA achieves comparable VQA performance with 50% fewer trainable parameters and superior retrieval with 2.85× faster inference.

---

## 7. C-JEPA: Contrastive JEPA

**Paper:** "Connecting Joint-Embedding Predictive Architecture with Contrastive Self-supervised Learning" (arXiv 2410.19560)  
**Authors:** Tong et al.

### 7.1 Motivation

I-JEPA can suffer from representation collapse despite EMA + stop-gradient. C-JEPA adds explicit regularization from contrastive learning (VICReg) to ensure:
- High variance in embeddings (no collapse to point)
- Low covariance (decorrelated features)
- Richer, more diverse representations

### 7.2 VICReg Regularization

```
┌─────────────────────────────────────────────────────────────────────┐
│                    C-JEPA = I-JEPA + VICReg                         │
│                                                                     │
│  VICReg Loss Components:                                            │
│                                                                     │
│  1. VARIANCE (prevent collapse):                                    │
│     L_var = max(0, γ - √(Var(z) + ε))                               │
│                                                                     │
│     - Encourages each dimension to have variance ≥ γ                │
│     - Prevents all embeddings collapsing to same point              │
│                                                                     │
│  2. INVARIANCE (alignment):                                         │
│     L_inv = ||z₁ - z₂||²                                            │
│                                                                     │
│     - Brings embeddings of same content together                    │
│     - (In JEPA: predicted vs target embeddings)                     │
│                                                                     │
│  3. COVARIANCE (decorrelation):                                     │
│     L_cov = (1/d) Σᵢ≠ⱼ [Cov(zᵢ, zⱼ)]²                               │
│                                                                     │
│     - Decorrelates different embedding dimensions                   │
│     - Encourages diverse feature representations                    │
│                                                                     │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  C-JEPA Total Loss:                                                 │
│                                                                     │
│  L = L_pred + λ_var · L_var + λ_cov · L_cov                         │
│                                                                     │
│  where:                                                             │
│  - L_pred: Standard JEPA prediction loss (L2)                       │
│  - λ_var ≈ 25                                                       │
│  - λ_cov ≈ 25                                                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.3 Implementation

```python
def vicreg_loss(z1, z2, lambda_var=25.0, lambda_cov=25.0, gamma=1.0, eps=1e-4):
    """
    VICReg regularization loss.
    
    Args:
        z1: First embedding batch [B, D]
        z2: Second embedding batch [B, D]
        lambda_var: Weight for variance loss
        lambda_cov: Weight for covariance loss
        gamma: Target standard deviation
        eps: Numerical stability
    
    Returns:
        Total VICReg loss
    """
    # Invariance loss (MSE between embeddings)
    inv_loss = keras.ops.mean((z1 - z2) ** 2)
    
    # Variance loss (encourage std >= gamma)
    std_z1 = keras.ops.sqrt(keras.ops.var(z1, axis=0) + eps)
    std_z2 = keras.ops.sqrt(keras.ops.var(z2, axis=0) + eps)
    var_loss = keras.ops.mean(keras.ops.relu(gamma - std_z1))
    var_loss += keras.ops.mean(keras.ops.relu(gamma - std_z2))
    
    # Covariance loss (decorrelate dimensions)
    z1_centered = z1 - keras.ops.mean(z1, axis=0)
    z2_centered = z2 - keras.ops.mean(z2, axis=0)
    
    batch_size = keras.ops.shape(z1)[0]
    cov_z1 = keras.ops.matmul(keras.ops.transpose(z1_centered), z1_centered) / (batch_size - 1)
    cov_z2 = keras.ops.matmul(keras.ops.transpose(z2_centered), z2_centered) / (batch_size - 1)
    
    # Off-diagonal elements
    d = keras.ops.shape(z1)[1]
    cov_loss = off_diagonal(cov_z1).pow(2).sum() / d
    cov_loss += off_diagonal(cov_z2).pow(2).sum() / d
    
    return inv_loss + lambda_var * var_loss + lambda_cov * cov_loss


def cjepa_loss(predicted_emb, target_emb, lambda_var=25.0, lambda_cov=25.0):
    """
    C-JEPA combines prediction loss with VICReg regularization.
    """
    # Standard JEPA prediction loss
    pred_loss = keras.ops.mean((predicted_emb - target_emb) ** 2)
    
    # VICReg regularization on predicted embeddings
    vicreg = vicreg_loss(predicted_emb, target_emb, lambda_var, lambda_cov)
    
    return pred_loss + vicreg
```

### 7.4 Benefits of C-JEPA

| Aspect | I-JEPA | C-JEPA |
|--------|--------|--------|
| Collapse prevention | EMA + stop-grad (implicit) | Explicit variance regularization |
| Feature diversity | May have correlated dims | Decorrelated by design |
| Training stability | Good | Better |
| ImageNet linear probe | 81.5% | 82.8% |
| Current status | Superseded | Image benchmark leader (Apr 2025) |

---

## 8. A-JEPA: Audio JEPA

**Paper:** "A-JEPA: Joint-Embedding Predictive Architecture Can Listen" (arXiv, 2024)

### 8.1 Architecture

A-JEPA applies the JEPA framework to audio spectrograms.

```
┌─────────────────────────────────────────────────────────────────────┐
│                      A-JEPA ARCHITECTURE                            │
│                                                                     │
│  Input: Audio waveform                                              │
│              │                                                      │
│              ▼                                                      │
│  ┌───────────────────────┐                                          │
│  │ Mel Spectrogram       │  Convert to 2D representation            │
│  │ (freq × time)         │                                          │
│  └───────────┬───────────┘                                          │
│              │                                                      │
│              ▼                                                      │
│  ┌───────────────────────┐                                          │
│  │ Patchify              │  Similar to ViT for images               │
│  │ (16×16 spectrogram    │                                          │
│  │  patches)             │                                          │
│  └───────────┬───────────┘                                          │
│              │                                                      │
│              ▼                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                   Standard JEPA                             │    │
│  │                                                             │    │
│  │  Context patches ──► Context Encoder ──► Predictor          │    │
│  │                                              │              │    │
│  │  Target patches ──► Target Encoder ──────► Loss             │    │
│  │                     (EMA)                                   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  Masking: Block masking in time-frequency space                     │
│  - Masks continuous time segments                                   │
│  - Forces understanding of temporal structure                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 Key Adaptations for Audio

| Aspect | I-JEPA (Image) | A-JEPA (Audio) |
|--------|----------------|----------------|
| Input | RGB image | Mel spectrogram |
| Dimensions | Height × Width | Frequency × Time |
| Patch size | 16×16 pixels | 16×16 spectrogram bins |
| Masking | 2D spatial blocks | Time-frequency blocks |
| Temporal bias | None | Emphasize time dimension |

---

## 9. Collapse Prevention Mechanisms

### 9.1 The Collapse Problem

Without proper safeguards, JEPA can collapse to trivial solutions:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    COLLAPSE FAILURE MODES                           │
│                                                                     │
│  Mode 1: Constant Output                                            │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  All inputs ──► Same embedding                              │    │
│  │  Loss = 0 (trivially)                                       │    │
│  │  No useful information learned                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  Mode 2: Dimensional Collapse                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Embeddings vary, but use only few dimensions               │    │
│  │  Most dimensions = constant                                 │    │
│  │  Reduced representational capacity                          │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  Mode 3: Informational Shortcut                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Encoder + Predictor learn to share information directly    │    │
│  │  Bypasses meaningful representation learning                │    │
│  │  Poor generalization                                        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.2 Prevention Strategies

```
┌─────────────────────────────────────────────────────────────────────┐
│                 COLLAPSE PREVENTION MECHANISMS                      │
│                                                                     │
│  1. EXPONENTIAL MOVING AVERAGE (EMA) TARGET                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                                                             │    │
│  │  θ_target = τ · θ_target + (1 - τ) · θ_context              │    │
│  │                                                             │    │
│  │  - τ typically 0.996 → 0.9999 (increases during training)   │    │
│  │  - Target encoder moves slowly                              │    │
│  │  - Provides stable prediction targets                       │    │
│  │  - Prevents oscillation and collapse                        │    │
│  │                                                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  2. STOP GRADIENT                                                   │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                                                             │    │
│  │  target_emb = stop_gradient(target_encoder(x))              │    │
│  │                                                             │    │
│  │  - No gradients flow through target encoder                 │    │
│  │  - Target encoder only updated via EMA                      │    │
│  │  - Prevents encoder from learning to make prediction easy   │    │
│  │                                                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  3. ASYMMETRIC ARCHITECTURE                                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                                                             │    │
│  │  Predictor << Encoder                                       │    │
│  │                                                             │    │
│  │  - Predictor is narrower/shallower                          │    │
│  │  - Prevents predictor from memorizing/shortcutting          │    │
│  │  - Forces encoder to learn meaningful representations       │    │
│  │                                                             │    │
│  │  Example: ViT-H encoder (1280 dim) + Predictor (384 dim)    │    │
│  │                                                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  4. HIGH MASKING RATIO                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                                                             │    │
│  │  Mask 75-90% of input                                       │    │
│  │                                                             │    │
│  │  - Forces meaningful prediction (can't just copy)           │    │
│  │  - Requires understanding of global structure               │    │
│  │  - Creates challenging but learnable task                   │    │
│  │                                                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  5. EXPLICIT REGULARIZATION (C-JEPA, VJ-VCR)                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                                                             │    │
│  │  L_variance: Ensure embedding dimensions have variance      │    │
│  │  L_covariance: Decorrelate embedding dimensions             │    │
│  │                                                             │    │
│  │  Directly prevents both constant and dimensional collapse   │    │
│  │                                                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  6. MULTI-BLOCK TARGETS                                             │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                                                             │    │
│  │  Predict multiple diverse regions (4+ target blocks)        │    │
│  │                                                             │    │
│  │  - Different positions require different predictions        │    │
│  │  - Prevents position-independent collapse                   │    │
│  │  - Encourages spatially-aware representations               │    │
│  │                                                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.3 EMA Schedule

```python
def ema_schedule(step, total_steps, base_tau=0.996, final_tau=1.0):
    """
    EMA decay schedule: increases from base_tau to final_tau.
    
    Rationale:
    - Early training: lower τ allows faster learning
    - Late training: higher τ provides more stable targets
    """
    # Cosine schedule
    progress = step / total_steps
    tau = final_tau - (final_tau - base_tau) * (1 + math.cos(math.pi * progress)) / 2
    return tau


# Update target encoder
def update_ema(context_encoder, target_encoder, tau):
    for p_ctx, p_tgt in zip(context_encoder.variables, target_encoder.variables):
        p_tgt.assign(tau * p_tgt + (1 - tau) * p_ctx)
```

---

## 10. Training Procedures

### 10.1 General JEPA Training Loop

```python
class JEPATrainer:
    """
    General JEPA training framework.
    """
    
    def __init__(
        self,
        context_encoder,
        target_encoder,
        predictor,
        ema_decay_base=0.996,
        ema_decay_final=1.0,
    ):
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.ema_decay_base = ema_decay_base
        self.ema_decay_final = ema_decay_final
        
        # Initialize target encoder as copy of context encoder
        self._initialize_target_encoder()
    
    def _initialize_target_encoder(self):
        """Copy context encoder weights to target encoder."""
        for p_ctx, p_tgt in zip(
            self.context_encoder.trainable_variables,
            self.target_encoder.trainable_variables
        ):
            p_tgt.assign(p_ctx)
    
    def _get_ema_decay(self, step, total_steps):
        """Compute EMA decay for current step."""
        progress = step / total_steps
        tau = self.ema_decay_final - (
            self.ema_decay_final - self.ema_decay_base
        ) * (1 + math.cos(math.pi * progress)) / 2
        return tau
    
    def _update_target_encoder(self, tau):
        """Update target encoder via EMA."""
        for p_ctx, p_tgt in zip(
            self.context_encoder.trainable_variables,
            self.target_encoder.trainable_variables
        ):
            p_tgt.assign(tau * p_tgt + (1 - tau) * p_ctx)
    
    def train_step(self, batch, step, total_steps):
        """
        Single training step.
        
        Args:
            batch: Input data (images or videos)
            step: Current training step
            total_steps: Total training steps
            
        Returns:
            loss: Training loss value
        """
        with tf.GradientTape() as tape:
            # Generate masks
            context_mask, target_masks = self.generate_masks(batch)
            
            # Extract context and target patches
            context_patches = self.apply_mask(batch, context_mask)
            target_patches = [self.apply_mask(batch, m) for m in target_masks]
            
            # Context encoding
            context_emb = self.context_encoder(context_patches, training=True)
            
            # Target encoding (no gradient)
            target_embs = []
            for t_patches in target_patches:
                t_emb = self.target_encoder(t_patches, training=False)
                target_embs.append(keras.ops.stop_gradient(t_emb))
            
            # Prediction
            pred_embs = []
            for i, t_mask in enumerate(target_masks):
                pos_tokens = self.get_position_tokens(t_mask)
                pred = self.predictor(context_emb, pos_tokens, training=True)
                pred_embs.append(pred)
            
            # Loss
            loss = 0.0
            for pred, target in zip(pred_embs, target_embs):
                loss += keras.ops.mean((pred - target) ** 2)
            loss /= len(pred_embs)
        
        # Update context encoder and predictor
        trainable_vars = (
            self.context_encoder.trainable_variables +
            self.predictor.trainable_variables
        )
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update target encoder via EMA
        tau = self._get_ema_decay(step, total_steps)
        self._update_target_encoder(tau)
        
        return loss
```

### 10.2 Masking Strategies Comparison

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MASKING STRATEGIES                               │
│                                                                     │
│  I-JEPA (Images):                                                   │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  • 4 target blocks, 15-20% each                             │    │
│  │  • Context: 85-100% coverage minus targets                  │    │
│  │  • Blocks can overlap                                       │    │
│  │  • Random aspect ratios [0.75, 1.5]                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  V-JEPA (Video):                                                    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  • 8 target blocks, 10-25% spatial each                     │    │
│  │  • SAME spatial mask across ALL frames                      │    │
│  │  • Variable temporal extent per block                       │    │
│  │  • Prevents frame-to-frame copying                          │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  A-JEPA (Audio):                                                    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  • Block masking in time-frequency                          │    │
│  │  • Emphasis on contiguous time segments                     │    │
│  │  • Forces temporal structure learning                       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 10.3 Hyperparameter Guidelines

```yaml
# Recommended hyperparameters for JEPA family

# Common across all variants
common:
  optimizer: adamw
  weight_decay: 0.05
  betas: [0.9, 0.95]
  warmup_epochs: 40
  ema_decay_base: 0.996
  ema_decay_final: 1.0

# I-JEPA specific
ijepa:
  epochs: 300
  batch_size: 2048
  base_lr: 0.001
  encoder: vit_huge_patch14
  predictor_depth: 12
  predictor_embed_dim: 384
  num_targets: 4
  target_scale: [0.15, 0.2]
  context_scale: [0.85, 1.0]

# V-JEPA specific
vjepa:
  epochs: 100
  batch_size: 256
  base_lr: 0.0005
  encoder: vit_large_patch16
  predictor_depth: 6
  frames_per_clip: 16
  num_targets: 8
  loss: l1  # L1 instead of L2

# V-JEPA 2 specific
vjepa2:
  encoder: vit_giant_patch14
  position_encoding: 3d_rope
  training_hours: 1_000_000+
  frame_rate: 8

# C-JEPA specific
cjepa:
  # Same as I-JEPA plus:
  lambda_var: 25.0
  lambda_cov: 25.0
  gamma: 1.0  # Target std

# VL-JEPA specific
vljepa:
  vision_encoder: vjepa2 (frozen)
  text_encoder: sonar_large
  predictor_layers: 6
  selective_decoding: true
  decoding_threshold: 0.8
```

---

## 11. Implementation Blueprints

### 11.1 Core JEPA Layer (Keras 3)

```python
"""
Core JEPA components for Keras 3.
"""
import keras
from keras import ops


class JEPAEncoder(keras.layers.Layer):
    """
    JEPA Context/Target Encoder based on Vision Transformer.
    
    This encoder processes masked or unmasked patches and outputs
    embeddings suitable for the JEPA prediction task.
    
    :param embed_dim: Embedding dimension
    :type embed_dim: int
    :param num_heads: Number of attention heads
    :type num_heads: int
    :param num_layers: Number of transformer layers
    :type num_layers: int
    :param mlp_ratio: MLP expansion ratio
    :type mlp_ratio: float
    :param dropout: Dropout rate
    :type dropout: float
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout
    
    def build(self, input_shape):
        # Transformer blocks
        self.blocks = []
        for i in range(self.num_layers):
            block = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout_rate,
                name=f"block_{i}"
            )
            self.blocks.append(block)
        
        self.norm = keras.layers.LayerNormalization(epsilon=1e-6, name="norm")
        super().build(input_shape)
    
    def call(self, x, training=None):
        """
        Forward pass.
        
        :param x: Input patch embeddings [B, N, D]
        :param training: Training mode flag
        :return: Encoded embeddings [B, N, D]
        """
        for block in self.blocks:
            x = block(x, training=training)
        x = self.norm(x)
        return x


class JEPAPredictor(keras.layers.Layer):
    """
    JEPA Predictor module.
    
    Predicts target embeddings from context embeddings and 
    positional information about where targets are located.
    
    :param embed_dim: Input embedding dimension (from encoder)
    :type embed_dim: int
    :param predictor_dim: Predictor hidden dimension (typically smaller)
    :type predictor_dim: int
    :param num_heads: Number of attention heads
    :type num_heads: int
    :param num_layers: Number of transformer layers
    :type num_layers: int
    :param num_targets: Maximum number of target positions
    :type num_targets: int
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        predictor_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 6,
        num_targets: int = 196,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.predictor_dim = predictor_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_targets = num_targets
    
    def build(self, input_shape):
        # Project from encoder dim to predictor dim
        self.input_proj = keras.layers.Dense(
            self.predictor_dim, 
            name="input_proj"
        )
        
        # Learnable mask tokens for target positions
        self.mask_token = self.add_weight(
            name="mask_token",
            shape=(1, 1, self.predictor_dim),
            initializer="glorot_uniform",
            trainable=True
        )
        
        # Position embeddings for all possible positions
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(1, self.num_targets, self.predictor_dim),
            initializer="glorot_uniform",
            trainable=True
        )
        
        # Predictor transformer blocks
        self.blocks = []
        for i in range(self.num_layers):
            block = TransformerBlock(
                embed_dim=self.predictor_dim,
                num_heads=self.num_heads,
                mlp_ratio=4.0,
                name=f"pred_block_{i}"
            )
            self.blocks.append(block)
        
        self.norm = keras.layers.LayerNormalization(epsilon=1e-6, name="pred_norm")
        
        # Project back to encoder dim
        self.output_proj = keras.layers.Dense(
            self.embed_dim,
            name="output_proj"
        )
        
        super().build(input_shape)
    
    def call(self, context_emb, target_positions, training=None):
        """
        Predict target embeddings.
        
        :param context_emb: Context embeddings [B, N_ctx, D]
        :param target_positions: Target position indices [B, N_tgt]
        :param training: Training mode flag
        :return: Predicted target embeddings [B, N_tgt, D]
        """
        batch_size = ops.shape(context_emb)[0]
        num_targets = ops.shape(target_positions)[1]
        
        # Project context to predictor dim
        context = self.input_proj(context_emb)
        
        # Create mask tokens for targets
        mask_tokens = ops.tile(
            self.mask_token, 
            [batch_size, num_targets, 1]
        )
        
        # Add position embeddings to mask tokens
        target_pos_emb = ops.take(self.pos_embed[0], target_positions, axis=0)
        mask_tokens = mask_tokens + target_pos_emb
        
        # Concatenate context and mask tokens
        x = ops.concatenate([context, mask_tokens], axis=1)
        
        # Apply predictor transformer
        for block in self.blocks:
            x = block(x, training=training)
        x = self.norm(x)
        
        # Extract predictions for target positions
        predictions = x[:, -num_targets:, :]
        
        # Project to encoder dimension
        predictions = self.output_proj(predictions)
        
        return predictions


class TransformerBlock(keras.layers.Layer):
    """
    Standard transformer block with pre-norm.
    
    :param embed_dim: Embedding dimension
    :type embed_dim: int
    :param num_heads: Number of attention heads
    :type num_heads: int
    :param mlp_ratio: MLP expansion ratio
    :type mlp_ratio: float
    :param dropout: Dropout rate
    :type dropout: float
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout
    
    def build(self, input_shape):
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.attn = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            dropout=self.dropout_rate
        )
        
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        mlp_dim = int(self.embed_dim * self.mlp_ratio)
        self.mlp = keras.Sequential([
            keras.layers.Dense(mlp_dim, activation="gelu"),
            keras.layers.Dropout(self.dropout_rate),
            keras.layers.Dense(self.embed_dim),
            keras.layers.Dropout(self.dropout_rate)
        ])
        super().build(input_shape)
    
    def call(self, x, training=None):
        # Pre-norm attention
        x = x + self.attn(
            self.norm1(x), 
            self.norm1(x), 
            training=training
        )
        # Pre-norm MLP
        x = x + self.mlp(self.norm2(x), training=training)
        return x
```

### 11.2 JEPA Model Class

```python
class JEPA(keras.Model):
    """
    Joint Embedding Predictive Architecture.
    
    Complete JEPA model with context encoder, target encoder (EMA),
    and predictor.
    
    :param encoder_config: Configuration for encoders
    :type encoder_config: dict
    :param predictor_config: Configuration for predictor
    :type predictor_config: dict
    :param ema_decay: EMA decay rate for target encoder
    :type ema_decay: float
    """
    
    def __init__(
        self,
        encoder_config: dict,
        predictor_config: dict,
        ema_decay: float = 0.996,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.ema_decay = ema_decay
        
        # Context encoder (trainable)
        self.context_encoder = JEPAEncoder(**encoder_config, name="context_encoder")
        
        # Target encoder (EMA, non-trainable)
        self.target_encoder = JEPAEncoder(**encoder_config, name="target_encoder")
        self.target_encoder.trainable = False
        
        # Predictor
        self.predictor = JEPAPredictor(**predictor_config, name="predictor")
    
    def build(self, input_shape):
        # Build all components
        super().build(input_shape)
        
        # Initialize target encoder from context encoder
        self._copy_weights_to_target()
    
    def _copy_weights_to_target(self):
        """Initialize target encoder with context encoder weights."""
        for ctx_var, tgt_var in zip(
            self.context_encoder.trainable_variables,
            self.target_encoder.variables
        ):
            tgt_var.assign(ctx_var)
    
    def update_target_encoder(self):
        """Update target encoder via EMA."""
        for ctx_var, tgt_var in zip(
            self.context_encoder.trainable_variables,
            self.target_encoder.variables
        ):
            tgt_var.assign(
                self.ema_decay * tgt_var + (1 - self.ema_decay) * ctx_var
            )
    
    def call(self, inputs, training=None):
        """
        Forward pass for inference (feature extraction).
        
        :param inputs: Patch embeddings [B, N, D]
        :param training: Training mode flag
        :return: Encoded features [B, N, D]
        """
        return self.context_encoder(inputs, training=training)
    
    def compute_loss(
        self, 
        context_patches, 
        target_patches, 
        target_positions,
        training=True
    ):
        """
        Compute JEPA training loss.
        
        :param context_patches: Context patch embeddings [B, N_ctx, D]
        :param target_patches: Target patch embeddings [B, N_tgt, D]
        :param target_positions: Target position indices [B, N_tgt]
        :param training: Training mode flag
        :return: L2 loss value
        """
        # Context encoding
        context_emb = self.context_encoder(context_patches, training=training)
        
        # Target encoding (no gradient)
        target_emb = self.target_encoder(target_patches, training=False)
        target_emb = ops.stop_gradient(target_emb)
        
        # Prediction
        pred_emb = self.predictor(context_emb, target_positions, training=training)
        
        # L2 loss
        loss = ops.mean((pred_emb - target_emb) ** 2)
        
        return loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "ema_decay": self.ema_decay,
        })
        return config
```

### 11.3 Mask Generation Utilities

```python
def generate_ijepa_masks(
    batch_size: int,
    num_patches: int,
    num_targets: int = 4,
    target_scale: tuple = (0.15, 0.2),
    target_aspect_ratio: tuple = (0.75, 1.5),
    context_scale: tuple = (0.85, 1.0),
):
    """
    Generate I-JEPA style masks.
    
    :param batch_size: Number of samples in batch
    :param num_patches: Total number of patches (e.g., 196 for 14x14)
    :param num_targets: Number of target blocks
    :param target_scale: (min, max) fraction of image per target
    :param target_aspect_ratio: (min, max) aspect ratio
    :param context_scale: (min, max) context coverage
    :return: (context_masks, target_masks, target_positions)
    """
    grid_size = int(num_patches ** 0.5)
    
    context_masks = []
    target_masks_list = []
    target_positions_list = []
    
    for _ in range(batch_size):
        all_target_positions = set()
        sample_target_masks = []
        
        for _ in range(num_targets):
            # Sample target block size
            scale = np.random.uniform(*target_scale)
            aspect = np.random.uniform(*target_aspect_ratio)
            
            block_area = int(num_patches * scale)
            block_h = int(np.sqrt(block_area / aspect))
            block_w = int(block_area / block_h)
            
            block_h = min(block_h, grid_size)
            block_w = min(block_w, grid_size)
            
            # Random position
            start_h = np.random.randint(0, grid_size - block_h + 1)
            start_w = np.random.randint(0, grid_size - block_w + 1)
            
            # Create mask
            mask = np.zeros((grid_size, grid_size), dtype=bool)
            mask[start_h:start_h+block_h, start_w:start_w+block_w] = True
            sample_target_masks.append(mask.flatten())
            
            # Track positions
            for h in range(start_h, start_h + block_h):
                for w in range(start_w, start_w + block_w):
                    all_target_positions.add(h * grid_size + w)
        
        # Context mask (everything not in targets)
        context_mask = np.ones(num_patches, dtype=bool)
        for pos in all_target_positions:
            context_mask[pos] = False
        
        context_masks.append(context_mask)
        target_masks_list.append(sample_target_masks)
        target_positions_list.append(sorted(list(all_target_positions)))
    
    return context_masks, target_masks_list, target_positions_list


def generate_vjepa_masks(
    batch_size: int,
    num_frames: int,
    grid_h: int,
    grid_w: int,
    num_targets: int = 8,
    spatial_scale: tuple = (0.1, 0.25),
):
    """
    Generate V-JEPA style spatiotemporal masks.
    
    Key: Same spatial mask across all frames.
    
    :param batch_size: Number of samples
    :param num_frames: Number of video frames
    :param grid_h: Spatial height in patches
    :param grid_w: Spatial width in patches
    :param num_targets: Number of target blocks
    :param spatial_scale: (min, max) spatial extent per block
    :return: (context_masks, target_masks)
    """
    context_masks = []
    target_masks_list = []
    
    for _ in range(batch_size):
        # Track masked spatial positions (same across frames)
        masked_spatial = set()
        sample_targets = []
        
        for _ in range(num_targets):
            # Spatial block size
            scale = np.random.uniform(*spatial_scale)
            block_h = int(grid_h * np.sqrt(scale))
            block_w = int(grid_w * np.sqrt(scale))
            
            start_h = np.random.randint(0, max(1, grid_h - block_h))
            start_w = np.random.randint(0, max(1, grid_w - block_w))
            
            # Temporal extent
            start_t = np.random.randint(0, num_frames)
            end_t = np.random.randint(start_t + 1, num_frames + 1)
            
            # Create 3D mask
            mask = np.zeros((num_frames, grid_h, grid_w), dtype=bool)
            mask[start_t:end_t, start_h:start_h+block_h, start_w:start_w+block_w] = True
            sample_targets.append(mask)
            
            # Track spatial positions
            for h in range(start_h, start_h + block_h):
                for w in range(start_w, start_w + block_w):
                    masked_spatial.add((h, w))
        
        # Context: all frames, masked spatial positions removed
        context_mask = np.ones((num_frames, grid_h, grid_w), dtype=bool)
        for (h, w) in masked_spatial:
            context_mask[:, h, w] = False  # Same mask all frames!
        
        context_masks.append(context_mask)
        target_masks_list.append(sample_targets)
    
    return context_masks, target_masks_list
```

---

## 12. Performance Benchmarks

### 12.1 Image Classification (ImageNet-1K)

| Model | Params | Linear Probe | Fine-tuned | GPU Hours |
|-------|--------|--------------|------------|-----------|
| MAE (ViT-H) | 632M | 76.6% | 86.9% | 6400 |
| DINO (ViT-B) | 86M | 78.2% | 83.6% | 4800 |
| DINOv2 (ViT-g) | 1.1B | 86.5% | 87.0% | - |
| I-JEPA (ViT-H) | 632M | 81.5% | 86.5% | 1300 |
| C-JEPA (ViT-H) | 632M | 82.8% | 87.1% | 1400 |

**Key insight:** I-JEPA achieves competitive performance with ~5× less compute than MAE.

### 12.2 Video Understanding

| Model | SSv2 | Kinetics-400 | Notes |
|-------|------|--------------|-------|
| VideoMAE | 70.8% | 86.4% | Generative |
| VideoMAE v2 | 71.2% | 87.8% | Improved |
| V-JEPA | 72.4% | 85.1% | Non-generative |
| V-JEPA 2 | 77.3% | 87.4% | Current SOTA |

**Something-Something v2 (SSv2)** requires temporal reasoning, where JEPA excels.

### 12.3 Vision-Language (VL-JEPA)

| Model | Params | GQA | TallyQA | POPE | Decoding Speed |
|-------|--------|-----|---------|------|----------------|
| InstructBLIP | ~8B | 49.2% | 68.1% | 85.3% | 1x |
| QwenVL | ~10B | 57.5% | 70.2% | 87.1% | 0.8x |
| VL-JEPA | 1.6B | 51.8% | 69.5% | 86.2% | 2.85x |

**Key insight:** VL-JEPA matches larger models with far fewer parameters and faster inference.

### 12.4 Robotics (V-JEPA 2-AC)

| Task | Success Rate | Training Data | Notes |
|------|--------------|---------------|-------|
| Pick & Place | Zero-shot | 62 hours robot video | Franka arm |
| Object Manipulation | Zero-shot | No task-specific training | Goal-image conditioned |
| Lab Transfer | Zero-shot | Cross-laboratory | Different setups |

**Key insight:** V-JEPA 2-AC enables zero-shot robotic manipulation without task rewards.

---

## 13. Practical Recommendations

### 13.1 When to Use Which JEPA Variant

| Use Case | Recommended Model | Reason |
|----------|-------------------|--------|
| Image pretraining (no labels) | C-JEPA | Best image benchmarks |
| Video understanding | V-JEPA 2 | Strong temporal reasoning |
| Robotics/planning | V-JEPA 2-AC | Zero-shot capability |
| Vision-language (efficient) | VL-JEPA | 50% fewer params, 3x faster |
| Audio understanding | A-JEPA | Audio-specific adaptations |
| Research/experimentation | I-JEPA | Simpler, well-documented |

### 13.2 Implementation Checklist

```
□ Encoder architecture
  □ ViT backbone (B/L/H/g depending on compute)
  □ Pre-norm transformer blocks
  □ RoPE or 3D-RoPE for video
  
□ Predictor design
  □ Narrower than encoder (384 vs 768/1024)
  □ Fewer layers (6-12)
  □ Learnable mask tokens
  □ Position-aware prediction
  
□ Masking strategy
  □ Multi-block targets (4-8)
  □ High masking ratio (75-90%)
  □ Consistent spatial mask for video
  
□ Collapse prevention
  □ EMA target encoder (τ = 0.996 → 1.0)
  □ Stop gradient on targets
  □ Optional: VICReg regularization
  
□ Training
  □ AdamW optimizer
  □ Cosine LR schedule with warmup
  □ Mixed precision (AMP)
  □ Gradient clipping
```

### 13.3 Common Pitfalls

| Pitfall | Symptom | Solution |
|---------|---------|----------|
| Representation collapse | Loss drops to 0, features constant | Add VICReg, check EMA decay |
| Predictor too powerful | High training acc, poor transfer | Reduce predictor capacity |
| Low masking ratio | Easy task, weak features | Increase to 75-90% |
| Different masks per frame | V-JEPA cheats via copying | Enforce same spatial mask |
| No position tokens | Predictor can't locate targets | Add learnable position embeddings |
| EMA decay too low | Unstable training | Start at 0.996, increase to ~1.0 |

### 13.4 Compute Requirements

| Model | GPU Memory | Training Time | Hardware |
|-------|------------|---------------|----------|
| I-JEPA (ViT-H) | 32GB | 1300 GPU-hours | 8× A100 |
| V-JEPA (ViT-L) | 40GB | 2000 GPU-hours | 8× A100 |
| V-JEPA 2 (ViT-g) | 80GB | 10000+ GPU-hours | 64× A100 |
| VL-JEPA | 40GB | 500 GPU-hours | 8× A100 |

### 13.5 Future Directions

1. **Hierarchical JEPA:** Multi-scale predictions for long-horizon planning
2. **Multi-modal JEPA:** Unified architecture for vision, language, audio, actions
3. **Efficient JEPA:** Linear attention variants for longer sequences
4. **Online JEPA:** Continual learning with streaming data
5. **Sparse JEPA:** Mixture-of-experts for scaling

---

## References

1. LeCun, Y. (2022). "A Path Towards Autonomous Machine Intelligence"
2. Assran et al. (2023). "Self-supervised Learning from Images with a Joint-Embedding Predictive Architecture" (CVPR)
3. Bardes et al. (2024). "Revisiting Feature Prediction for Learning Visual Representations from Video"
4. Assran et al. (2025). "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning"
5. Chen et al. (2025). "VL-JEPA: Joint Embedding Predictive Architecture for Vision-language"
6. Tong et al. (2024). "Connecting Joint-Embedding Predictive Architecture with Contrastive Self-supervised Learning"
7. Bardes et al. (2022). "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning" (ICLR)

---

## Appendix: Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────────┐
│                    JEPA QUICK REFERENCE                             │
│                                                                     │
│  Core Idea: Predict embeddings, not pixels                          │
│                                                                     │
│  Components:                                                        │
│  • Context Encoder (trainable) - processes visible input            │
│  • Target Encoder (EMA) - provides stable targets                   │
│  • Predictor (narrow) - predicts target embeddings                  │
│                                                                     │
│  Loss: L = ||predicted_emb - target_emb||²                          │
│                                                                     │
│  Collapse Prevention:                                               │
│  • EMA (τ ≈ 0.996-0.999)                                            │
│  • Stop gradient on targets                                         │
│  • Asymmetric architecture                                          │
│  • High masking (75-90%)                                            │
│  • Optional: VICReg regularization                                  │
│                                                                     │
│  Variants:                                                          │
│  • I-JEPA: Images, multi-block masking                              │
│  • V-JEPA: Video, same spatial mask across frames                   │
│  • V-JEPA 2-AC: Action-conditioned for robotics                     │
│  • VL-JEPA: Vision-language, predicts text embeddings               │
│  • C-JEPA: With VICReg, best on images                              │
│  • A-JEPA: Audio spectrograms                                       │
│                                                                     │
│  Key Hyperparameters:                                               │
│  • num_targets: 4-8                                                 │
│  • target_scale: 0.15-0.25                                          │
│  • ema_decay: 0.996 → 1.0                                           │
│  • predictor_dim: 0.3-0.5× encoder_dim                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```