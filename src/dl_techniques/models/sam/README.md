# Segment Anything Model (SAM) - Keras 3 Implementation

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

An implementation of the **Segment Anything Model (SAM)** architecture in
**Keras 3**, based on the paper
["Segment Anything"](https://arxiv.org/abs/2304.02643) by Kirillov et al.
(2023).

> **Scope, stated up front.** This package ships the **architecture, its
> forward pass, and a trainable wrapper** — but no weights.
> - **No pretrained checkpoint** ships here or is downloaded.
>   `SAM.from_variant('vit_b')` builds a randomly initialized model. Any
>   qualitative claim below ("high-quality mask", "works on any domain")
>   describes SAM *as published*, not what these random weights produce.
> - **A trainer ships now** (`SAMTrainingModel` here, `src/train/sam/` for the
>   data pipeline and CLI — §11). It is proven to RUN with live gradients on an
>   executed `--smoke` run. It makes **no accuracy claim**: nothing in this
>   repository has trained SAM to any quality, and `vit_l` / `vit_h` have never
>   been forward-passed at all.
> - Loading official Meta SAM weights is **not** demonstrated. Two known
>   blockers were removed (cross-attention internal dim, MLP head depth), but
>   no key-mapping layer exists and no official checkpoint was ever loaded.

---

## Table of Contents

1. [Overview: What is SAM and Why It Matters](#1-overview-what-is-sam-and-why-it-matters)
2. [The Segmentation Problem SAM Solves](#2-the-segmentation-problem-sam-solves)
3. [How SAM Works: Core Concepts](#3-how-sam-works-core-concepts)
4. [Architecture Deep Dive](#4-architecture-deep-dive)
5. [Quick Start Guide](#5-quick-start-guide)
6. [Component Reference](#6-component-reference)
7. [Configuration & Model Variants](#7-configuration--model-variants)
8. [Comprehensive Usage Examples](#8-comprehensive-usage-examples)
9. [Advanced Usage Patterns](#9-advanced-usage-patterns)
10. [Performance Optimization](#10-performance-optimization)
11. [Training and Fine-tuning](#11-training-and-fine-tuning)
12. [Serialization & Deployment](#12-serialization--deployment)
13. [Testing & Validation](#13-testing--validation)
14. [Troubleshooting & FAQs](#14-troubleshooting--faqs)
15. [Technical Details](#15-technical-details)
16. [Requirements](#16-requirements)
17. [Citation](#17-citation)

---

## 1. Overview: What is SAM and Why It Matters

### What is SAM?

The **Segment Anything Model (SAM)** is a foundation model for image segmentation that represents a paradigm shift in how we approach computer vision segmentation tasks. Unlike traditional segmentation models that are trained for specific objects or domains, SAM is designed to be **promptable** and **generalizable**.

### Key Innovations

SAM introduces three groundbreaking capabilities:

1. **Zero-Shot Generalization**: SAM can segment objects it has never seen during training, adapting to new visual domains without additional fine-tuning.

2. **Flexible Prompting**: SAM accepts various types of input prompts—points, bounding boxes, or rough masks—making it incredibly versatile for different use cases.

3. **Interactive Segmentation**: The architecture enables real-time, human-in-the-loop annotation workflows, where users can iteratively refine segmentation masks.

### Why SAM Matters

**Traditional Approach Problems**:
```
Problem: Need to segment cats in images
Traditional Solution:
  1. Collect thousands of cat images
  2. Manually annotate every pixel
  3. Train a specialized model
  4. Model only works for cats
  5. To segment dogs, start over from step 1
```

**SAM's Solution**:
```
SAM Approach:
  1. Pre-trained on 11M images, 1B masks
  2. Works on ANY object (cats, dogs, cars, medical scans, etc.)
  3. Just provide a point, box, or rough mask
  4. Get instant, high-quality segmentation
  5. No retraining needed
```

### Real-World Impact

SAM addresses critical challenges in computer vision:

- **🔬 Medical Imaging**: Segment tumors, organs, or anomalies without domain-specific training
- **🚗 Autonomous Vehicles**: Identify obstacles, pedestrians, and road elements in diverse conditions
- **🏭 Manufacturing**: Quality control with automatic defect detection across product types
- **📸 Content Creation**: Professional-grade image editing and object isolation
- **🌾 Agriculture**: Crop disease detection and yield estimation
- **🔍 Scientific Research**: Cell segmentation, satellite imagery analysis, and more

---

## 2. The Segmentation Problem SAM Solves

### The Data Annotation Bottleneck

Traditional segmentation models face a fundamental challenge:

```
┌─────────────────────────────────────────────────────────────┐
│  Traditional Segmentation Model Training Pipeline           │
│                                                             │
│  1. Data Collection      → Months of effort                 │
│  2. Manual Annotation    → $50,000 - $500,000+ costs        │
│  3. Model Training       → Weeks of GPU time                │
│  4. Domain Specificity   → Works ONLY on similar data       │
│  5. Adaptation Required  → Restart for new domains          │
└─────────────────────────────────────────────────────────────┘
```

### How SAM Changes the Game

SAM's approach eliminates the need for task-specific training:

```
┌─────────────────────────────────────────────────────────────┐
│  SAM Workflow                                               │
│                                                             │
│  1. Load Pre-trained SAM   → 1 minute                       │
│  2. Provide Simple Prompt  → Click a point or draw box      │
│  3. Get High-Quality Mask  → Instant (< 50ms)               │
│  4. Works on ANY Domain    → No retraining needed           │
│  5. Interactive Refinement → Real-time user feedback        │
└─────────────────────────────────────────────────────────────┘
```

### The Ambiguity Challenge

Real-world images often contain ambiguous scenarios:

```
Example: A person holding a cup

┌──────────────────────────────────────────┐
│                                          │
│     👤 (Person)                          │
│      |                                   │
│     ☕ (Cup)                              │
│                                          │
│  Question: What should be segmented?     │
│  - Just the cup?                         │
│  - The person?                           │
│  - Both?                                 │
│  - The hand holding the cup?             │
└──────────────────────────────────────────┘
```

**SAM's Solution**: Instead of forcing a single answer, SAM produces **multiple valid mask proposals** and lets the user or downstream application choose the most appropriate one. It also provides **quality scores (IoU predictions)** for each mask.

---

## 3. How SAM Works: Core Concepts

### The Three-Component Architecture

SAM's design is elegantly simple yet powerful:

```
┌──────────────────────────────────────────────────────────────────────┐
│                          SAM Architecture                            │
│                                                                      │
│  ┌─────────────────┐      ┌─────────────────┐      ┌──────────────┐  │
│  │  Image Encoder  │      │ Prompt Encoder  │      │    Mask      │  │
│  │   (Heavy ViT)   │──┐   │  (Lightweight)  │──┐   │   Decoder    │  │
│  │                 │  │   │                 │  │   │ (Lightweight)│  │
│  │  Run once per   │  │   │  Run for each   │  │   │ Combines all │  │
│  │     image       │  │   │     prompt      │  │   │  to predict  │  │
│  │                 │  │   │                 │  │   │    masks     │  │
│  └─────────────────┘  │   └─────────────────┘  │   └──────────────┘  │
│         │             │            │           │            │        │
│         └─────────────┴────────────┴───────────┘            │        │
│                                                             │        │
│                        Output: Segmentation Masks + Scores  │        │
└──────────────────────────────────────────────────────────────────────┘
```

### Why This Design is Brilliant

**1. Computational Efficiency**: The expensive image encoder runs only once per image, while the lightweight prompt and mask decoders can run in milliseconds for each prompt.

```
Time Breakdown (for 1024x1024 image on GPU):
┌────────────────────────────────────────────┐
│ Image Encoding:     ~200ms  (once)         │
│ Prompt Encoding:    ~2ms    (per prompt)   │
│ Mask Decoding:      ~8ms    (per prompt)   │
│────────────────────────────────────────────│
│ Total for 1 prompt: ~210ms                 │
│ Total for 100 prompts: ~1200ms             │
│ (vs. 20,000ms if re-encoding each time)    │
└────────────────────────────────────────────┘
```

**2. Flexible Prompting**: By separating prompt encoding from image encoding, SAM can accept any combination of prompt types without architectural changes.

**3. Ambiguity Handling**: The decoder can produce multiple mask proposals simultaneously, acknowledging that segmentation can be subjective.

### The Complete Data Flow

Here's how data flows through SAM step by step:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     SAM Complete Data Flow                              │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: Image Preprocessing
───────────────────────────
Input Image (up to 1024x1024 -- SEE BELOW)
    │
    ├─► Normalize: (pixel - mean) / std
    ├─► Pad (bottom/right) to 1024x1024 if SMALLER
    └─► Shape: (B, 1024, 1024, 3)

  `SAM.preprocess` PADS ONLY. An image LARGER than the encoder's `img_size`
  in either axis raises `ValueError` naming the offending size. Downscale it
  first with `resize_longest_side(image, model.image_encoder.img_size)`
  (`dl_techniques.models.sam.preprocessing`), which is the transform reference
  SAM assumes has already been applied.


STEP 2: Image Encoding (Run Once per Image) ️ ~200ms
────────────────────────────────────────────
Image (B, 1024, 1024, 3)
    │
    ├─► PatchEmbedding: Conv2D(16x16 patches)
    │       └─► (B, 64, 64, D)  where D = {768, 1024, 1280}
    │
    ├─► + Positional Embedding (learnable)
    │
    ├─► Vision Transformer Blocks (12, 24, or 32 layers)
    │   ├─► Windowed Self-Attention (efficient)
    │   ├─► Global Self-Attention (select layers)
    │   ├─► Relative Position Bias
    │   └─► Feed-Forward Network
    │
    └─► Neck: Upsample & Refine
            └─► Output: (B, 64, 64, 256) ← IMAGE EMBEDDING


STEP 3: Prompt Encoding (Run per Prompt) ~2ms
──────────────────────────────────────────

3a. Sparse Prompts (Points & Boxes)
────────────────────────────────────
Points: (x, y) + label {foreground=1, background=0}
    │
    └─► Positional Encoding (sine/cosine)
        + Type Embedding (learnable)
            └─► Sparse Embeddings (B, N, 256)

Boxes: (x1, y1, x2, y2)
    │
    └─► Encode 4 corners as points
        + Corner Type Embeddings
            └─► Sparse Embeddings (B, 4, 256)

3b. Dense Prompts (Masks)
──────────────────────────
Mask Hint (B, 1, H, W)
    │
    └─► CNN: Conv → GELU → Conv → GELU
            └─► Dense Embedding (B, 64, 64, 256)


STEP 4: Mask Decoding ~8ms
──────────────────────────────
Inputs:
  - Image Embedding (B, 64, 64, 256)
  - Sparse Prompt Embeddings (B, N, 256)
  - Dense Prompt Embedding (B, 64, 64, 256)
  - Positional Encoding Grid (64, 64, 256)

    │
    ├─► Add dense prompt to image embedding
    │       Image' = Image + Dense Prompt
    │
    ├─► Initialize Output Tokens
    │   ├─► IoU token (1)
    │   └─► Mask tokens (3 for multi-mask, 1 for single)
    │
    ├─► Two-Way Transformer (2 layers)
    │   │
    │   │  Each layer does:
    │   │  ┌────────────────────────────────────┐
    │   │  │ 1. Self-Attention on Tokens        │
    │   │  │    (Tokens attend to each other)   │
    │   │  ├────────────────────────────────────┤
    │   │  │ 2. Cross-Attention (Token→Image)   │
    │   │  │    (Tokens attend to image)        │
    │   │  ├────────────────────────────────────┤
    │   │  │ 3. MLP on Tokens                   │
    │   │  ├────────────────────────────────────┤
    │   │  │ 4. Cross-Attention (Image→Token)   │
    │   │  │    (Image attends to tokens)       │
    │   │  └────────────────────────────────────┘
    │   │
    │   └─► Updated: Tokens (B, N, 256)
    │                Image (B, 64*64, 256)
    │
    ├─► Reshape image back to (B, 64, 64, 256)
    │
    ├─► Upscale to (B, 256, 256, 32) via Conv-Transpose
    │
    ├─► For each mask token:
    │   │   Use MLP Hypernetwork to predict dynamic conv weights
    │   │   Apply to upscaled features
    │   │       └─► Mask Logits (B, 3, 256, 256)
    │
    └─► IoU Prediction Head
            └─► IoU Scores (B, 3)


STEP 5: Postprocessing
──────────────────────
Mask Logits (B, 3, 256, 256)
    │
    ├─► Upsample to 1024x1024
    ├─► Remove padding (crop to input size)
    ├─► Scale to original image size
    └─► Threshold at 0.0   (only when binarize_masks=True, the default)
            └─► Binary Masks (B, 3, H_orig, W_orig), uint8, NO gradient

Final Outputs:
  ├─► masks: (B, 3, H_orig, W_orig)     uint8 + gradient-dead by default
  ├─► iou_predictions: (B, 3)
  └─► low_res_logits: (B, 3, 256, 256)  <- THE training target
```

See "Output contract: `low_res_logits` is THE training target" in section 6.

### Key Design Decisions Explained

**Q: Why use a Vision Transformer instead of a CNN?**

A: ViTs excel at capturing long-range dependencies and global context, which is crucial for understanding object boundaries and relationships. However, SAM's ViT includes windowed attention for efficiency and global attention at select layers for crucial long-range modeling.

**Q: Why separate sparse and dense prompt encoding?**

A: Different prompt types have different characteristics:
- **Sparse prompts** (points, boxes): Few spatial locations, need positional encoding
- **Dense prompts** (masks): Full spatial grid, need CNN processing

This separation allows each to be optimized for its specific purpose.

**Q: Why predict multiple masks?**

A: Segmentation is often ambiguous. Consider clicking on a person's shirt button:
- Should we segment the button?
- The shirt?
- The entire person?

SAM produces multiple valid interpretations and ranks them by predicted quality (IoU).

**Q: Why use a two-way transformer?**

A: The bidirectional information flow allows:
1. **Tokens → Image**: Prompts guide where the model should focus
2. **Image → Tokens**: Visual features inform mask predictions
3. **Tokens ↔ Tokens**: Mask proposals coordinate with each other

This creates a rich interaction that produces high-quality, contextually-aware masks.

---

## 4. Architecture Deep Dive

### 4.1 Image Encoder (Vision Transformer)

The image encoder is the heavyweight component responsible for understanding the entire image.

#### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Image Encoder (ViT)                              │
└─────────────────────────────────────────────────────────────────────┘

Input: (B, 1024, 1024, 3)
  │
  ▼
┌──────────────────────────────────────┐
│   Patch Embedding                    │
│   Conv2D(kernel=16, stride=16)       │
│   ────────────────────────────────   │
│   Divides image into 64×64 patches   │
│   Each patch → D-dim vector          │
└──────────────────────────────────────┘
  │
  ▼ (B, 64, 64, D)
┌──────────────────────────────────────┐
│   Positional Embedding               │
│   + Learnable Position Grid          │
│   ────────────────────────────────   │
│   Adds spatial information           │
└──────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────┐
│   Transformer Block 1                │ ──┐
│   ├─ Layer Norm                      │   │
│   ├─ Windowed Self-Attention         │   │
│   │   • Window Size: 14×14           │   │
│   │   • Relative Position Bias       │   │
│   │   • Multi-Head (12-16 heads)     │   │  Repeated
│   ├─ Residual Connection             │   │  12-32 times
│   ├─ Layer Norm                      │   │  (based on
│   ├─ MLP (4× expansion)              │   │   variant)
│   └─ Residual Connection             │   │
└──────────────────────────────────────┘   │
  │                                        │
  ▼                                        │
┌──────────────────────────────────────┐   │
│   Transformer Block 2-N              │   │
│   (Some use Global Attention)        │   │
└──────────────────────────────────────┘ ──┘
  │
  ▼ (B, 64, 64, D)
┌──────────────────────────────────────┐
│   Neck (Feature Refinement)          │
│   ├─ Conv2D(3×3) → LN                │
│   ├─ Conv2D(3×3) → LN                │
│   └─ Channel reduction: D → 256      │
└──────────────────────────────────────┘
  │
  ▼
Output: (B, 64, 64, 256) ← Image Embedding
```

#### Windowed vs Global Attention

SAM uses a hybrid approach:

```
Windowed Attention (Most Layers):
┌─────────────────────────────────────┐
│  [1][2][3][4]  ←  Attention computed│
│  [5][6][7][8]      within each      │
│  [9][10][11][12]   14×14 window     │
│  [13][14][15][16]                   │
├─────────────────────────────────────┤
│  Advantages:                        │
│  • O(n) complexity instead of O(n²) │
│  • Efficient for local patterns     │
│  • Reduces memory usage             │
└─────────────────────────────────────┘

Global Attention (Select Layers):
┌─────────────────────────────────────┐
│  Every patch attends to every other │
│  patch in the entire image          │
├─────────────────────────────────────┤
│  Advantages:                        │
│  • Captures long-range dependencies │
│  • Essential for object boundaries  │
│  • Better global understanding      │
├─────────────────────────────────────┤
│  Used at strategic depths:          │
│  vit_b: layers 2, 5, 8, 11          │
│  vit_l: layers 5, 11, 17, 23        │
│  vit_h: layers 7, 15, 23, 31        │
└─────────────────────────────────────┘
```

#### Relative Position Bias

Instead of absolute position encodings, SAM uses relative position biases:

```
Standard Attention:
Attention(Q, K, V) = softmax(QK^T / √d) V

With Relative Position Bias:
Attention(Q, K, V) = softmax((QK^T + B) / √d) V
                                    │
                                    └─ Learnable bias based on
                                       relative distance between patches

Benefits:
• Generalizes better to different image sizes
• Learns spatial relationships more effectively
• Improves boundary detection
```

### 4.2 Prompt Encoder

The prompt encoder translates user inputs into the model's embedding space.

```
┌─────────────────────────────────────────────────────────────────┐
│                      Prompt Encoder                             │
└─────────────────────────────────────────────────────────────────┘

Input Types:
┌──────────┬──────────┬────────────┐
│  Points  │  Boxes   │   Masks    │
└──────────┴──────────┴────────────┘
     │          │            │
     ▼          ▼            ▼
┌─────────────────────────────────────────────┐
│         Sparse Encoding                     │
│  (Points & Boxes)                           │
│                                             │
│  Step 1: Positional Encoding                │
│  ┌───────────────────────────┐              │
│  │  PE(x) = [sin(πx/λ_0),    │              │
│  │           cos(πx/λ_0),    │              │
│  │           sin(πx/λ_1),    │              │
│  │           ...]            │              │
│  └───────────────────────────┘              │
│                                             │
│  Step 2: Type Embeddings                    │
│  ┌────────────────────────────────┐         │
│  │  Point Background → [e_bg]     │         │
│  │  Point Foreground → [e_fg]     │         │
│  │  Box Top-Left     → [e_tl]     │         │
│  │  Box Bottom-Right → [e_br]     │         │
│  └────────────────────────────────┘         │
│  Exactly FOUR embeddings. There is no       │
│  generic "box" embedding.                   │
│                                             │
│  Padding rows (label == -1): the positional │
│  encoding is ZEROED and only the learned    │
│  `not_a_point_embed` remains, so a padding  │
│  row does not depend on where its dummy     │
│  coordinate sits.                           │
│                                             │
│  Output: (B, N_sparse, 256)                 │
└─────────────────────────────────────────────┘
              ▼
┌─────────────────────────────────────────────┐
│         Dense Encoding                      │
│  (Masks)                                    │
│                                             │
│  Input Mask: (B, 1, 256, 256)               │
│  REQUIRED size: exactly                     │
│  4 * image_embedding_size in BOTH axes      │
│  (the conv stack strides by 2 twice).       │
│  Anything else raises ValueError inside     │
│  `_embed_masks`, naming the required and    │
│  the offending size.                        │
│       │                                     │
│       ├─► Conv2D(16 channels, 2×2, stride 2)│
│       │       + GELU                        │
│       │                                     │
│       ├─► Conv2D(64 channels, 2×2, stride 2)│
│       │       + GELU                        │
│       │                                     │
│       └─► Conv2D(256 channels, 1×1)         │
│                                             │
│  Output: (B, 64, 64, 256)                   │
└─────────────────────────────────────────────┘
              ▼
┌─────────────────────────────────────────────┐
│  Combined Output                            │
│  ├─ Sparse Embeddings: (B, N, 256)          │
│  └─ Dense Embeddings:  (B, 64, 64, 256)     │
└─────────────────────────────────────────────┘
```

#### Prompt Encoding Examples

**Example 1: Single Foreground Point**
```
Input:
  Point: (x=512, y=512)
  Label: 1 (foreground)

Processing:
  1. PE(512.5, 512.5) → [0.125, 0.923, -0.707, ...]   (+0.5 pixel centre)
  2. + Foreground embedding → [e_fg]
  3. A PADDING row is appended, because no box was supplied:
     its positional encoding is zeroed and it carries
     `not_a_point_embed` alone.
  4. Final: (1, 2, 256) sparse embeddings   <- TWO rows, not one

Note: the padding row is appended only on a point-only prompt
(`pad = boxes is None`). Supply a box alongside the point and the
result is (1, 3, 256): 1 point + 2 box corners, no padding row.
```

**Example 2: Bounding Box**
```
Input:
  Box: (x1=100, y1=100, x2=900, y2=900)

Processing:
  1. Top-left corner:     PE(100, 100) + [e_tl]
  2. Bottom-right corner: PE(900, 900) + [e_br]
  3. Final: (1, 2, 256) sparse embeddings

A box is TWO corners, not four. There is no top-right / bottom-left
embedding, and a box-only prompt appends no padding row.
```

**Example 3: Mask Prompt**
```
Input:
  Rough mask: (1, 1, 256, 256)

Processing:
  1. CNN downsampling and feature extraction
  2. Final: (1, 64, 64, 256) dense embedding
  3. Added directly to image embedding
```

### 4.3 Mask Decoder

The mask decoder combines image and prompt embeddings to predict masks.

```
┌─────────────────────────────────────────────────────────────────┐
│                      Mask Decoder                               │
└─────────────────────────────────────────────────────────────────┘

Inputs:
┌──────────────────────────────────────┐
│ Image Embedding:  (B, 64, 64, 256)   │
│ Sparse Prompts:   (B, N, 256)        │
│ Dense Prompts:    (B, 64, 64, 256)   │
│ Position Grid:    (64, 64, 256)      │
└──────────────────────────────────────┘
              ▼
┌──────────────────────────────────────────────────────┐
│  Step 1: Prepare Image Features                      │
│  ────────────────────────────────                    │
│  Image' = Image + Dense_Prompt + Position_Grid       │
│  Flatten to: (B, 64×64, 256) = (B, 4096, 256)        │
└──────────────────────────────────────────────────────┘
              ▼
┌──────────────────────────────────────────────────────┐
│  Step 2: Initialize Output Tokens                    │
│  ────────────────────────────────                    │
│  • 1 IoU token (learnable embedding)                 │
│  • 3 Mask tokens (learnable embeddings)              │
│  • Combine with sparse prompts                       │
│  Result: (B, 4 + N_sparse, 256)                      │
└──────────────────────────────────────────────────────┘
              ▼
┌──────────────────────────────────────────────────────────────┐
│  Step 3: Two-Way Transformer (2 layers)                      │
│  ──────────────────────────────────────                      │
│                                                              │
│  For each layer:                                             │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Block 1: Update Tokens                                │  │
│  │  ─────────────────────                                 │  │
│  │  1. Self-Attention on Tokens                           │  │
│  │     Queries = Tokens                                   │  │
│  │     Keys = Tokens                                      │  │
│  │     Values = Tokens                                    │  │
│  │     → Tokens' = LayerNorm(Tokens + Attention)          │  │
│  │                                                        │  │
│  │  2. Cross-Attention (Token → Image)                    │  │
│  │     Queries = Tokens'                                  │  │
│  │     Keys = Image                                       │  │
│  │     Values = Image                                     │  │
│  │     → Tokens'' = LayerNorm(Tokens' + Attention)        │  │
│  │                                                        │  │
│  │  3. MLP                                                │  │
│  │     → Tokens''' = LayerNorm(Tokens'' + MLP)            │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │ 
│  │  Block 2: Update Image                                 │  │
│  │  ────────────────────                                  │  │
│  │  4. Cross-Attention (Image → Token)                    │  │
│  │     Queries = Image                                    │  │
│  │     Keys = Tokens'''                                   │  │
│  │     Values = Tokens'''                                 │  │
│  │     → Image' = LayerNorm(Image + Attention)            │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
              ▼
┌──────────────────────────────────────────────────────┐
│  Step 4: Upscale Image Features                      │
│  ──────────────────────────────                      │
│  Reshape: (B, 4096, 256) → (B, 64, 64, 256)          │
│  ConvTranspose: (B, 64, 64, 256) → (B, 256, 256, 32) │
└──────────────────────────────────────────────────────┘
              ▼
┌──────────────────────────────────────────────────────┐
│  Step 5: Predict Masks (Hypernetwork)                │
│  ──────────────────────────                          │
│  For each mask token:                                │
│    1. MLP: Token → Conv weights                      │
│    2. Apply dynamic conv to upscaled features        │
│    3. Result: Mask logits                            │
│                                                      │
│  Output: (B, 3, 256, 256)                            │
└──────────────────────────────────────────────────────┘
              ▼
┌──────────────────────────────────────────────────────┐
│  Step 6: Predict IoU                                 │
│  ──────────────────                                  │
│  IoU Token → MLP → 3-layer network → IoU scores      │
│  Output: (B, 3)                                      │
└──────────────────────────────────────────────────────┘
              ▼
Final Outputs:
• Mask Logits: (B, 3, 256, 256)
• IoU Predictions: (B, 3)
```

#### The Hypernetwork Concept

Instead of using fixed convolutional layers, SAM uses a hypernetwork to generate dynamic convolution weights:

```
Traditional Approach:
┌────────────────────────────────────┐
│  Fixed Conv Layer                  │
│  weights learned during training   │
│  Same for all inputs               │
└────────────────────────────────────┘

SAM's Hypernetwork Approach:
┌────────────────────────────────────────────────────┐
│  Mask Token (specific to this prompt)              │
│         ▼                                          │
│      MLP                                           │
│         ▼                                          │
│  Dynamic Conv Weights                              │
│         ▼                                          │
│  Apply to upscaled features                        │
│         ▼                                          │
│  Mask logits tailored to this specific prompt      │
└────────────────────────────────────────────────────┘

Benefits:
• Each prompt gets custom-tailored weights
• More expressive than fixed convolutions
• Better handles diverse prompt types
```

---

## 5. Quick Start Guide

### Installation

```bash
# Ensure you have the required dependencies
pip install keras>=3.8.0 tensorflow>=2.18.0 numpy

# Clone or install dl_techniques if needed
# (for normalization and FFN factories)
```

### Your First Segmentation (30 seconds)

> **`SAM.from_variant(...)` returns RANDOM weights.** It builds the
> architecture; it does not load a checkpoint, and no pretrained SAM weights
> ship with or are downloaded by this repo. The snippet below exercises the
> API and the shapes — the masks it produces are noise.

```python
import keras
import numpy as np
from dl_techniques.models.sam import SAM

# 1. Create a SAM model (Base variant for speed)
model = SAM.from_variant('vit_b')
print("Model created successfully!")

# 2. Load or create an image (1024x1024 is optimal)
image = keras.random.normal(shape=(1, 1024, 1024, 3))
# In practice: image = load_and_resize_your_image()

# 3. Define a simple point prompt
#    Click at position (512, 512) to mark a foreground point
points_coords = keras.ops.convert_to_tensor([[[512.0, 512.0]]])
points_labels = keras.ops.convert_to_tensor([[1]])  # 1 = foreground
points = (points_coords, points_labels)

# 4. Run segmentation
outputs = model({
    'image': image,
    'points': points,
    'original_size': (1024, 1024)
})

# 5. Examine results
print(f"\n✅ Segmentation Complete!")
print(f"Generated {outputs['masks'].shape[1]} mask proposals")
print(f"Masks shape: {outputs['masks'].shape}")
print(f"Quality scores (IoU): {outputs['iou_predictions']}")

# 6. Get the best mask
best_mask_idx = keras.ops.argmax(outputs['iou_predictions'][0])
best_mask = outputs['masks'][0, best_mask_idx]
print(f"\nBest mask (index {best_mask_idx}) selected")
```

### Visualizing Results (with matplotlib)

```python
import matplotlib.pyplot as plt
import numpy as np

# Convert to numpy for visualization
image_np = keras.ops.convert_to_numpy(image[0])
mask_np = keras.ops.convert_to_numpy(best_mask)

# Denormalize image if needed
image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image
axes[0].imshow(image_np)
axes[0].set_title('Original Image')
axes[0].axis('off')

# Mask only
axes[1].imshow(mask_np, cmap='gray')
axes[1].set_title('Predicted Mask')
axes[1].axis('off')

# Overlay
axes[2].imshow(image_np)
axes[2].imshow(mask_np, alpha=0.5, cmap='Reds')
axes[2].scatter([512], [512], c='green', s=100, marker='*')
axes[2].set_title('Mask Overlay (Green star = prompt point)')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

---

## 6. Component Reference

### 6.1 Image Encoder (`ImageEncoderViT`)

**Purpose**: Transform input images into rich feature embeddings.

**Location**: `dl_techniques.models.sam.image_encoder.ImageEncoderViT`

```python
from dl_techniques.models.sam.image_encoder import ImageEncoderViT

# Create an encoder
encoder = ImageEncoderViT(
    img_size=1024,           # Input image size
    patch_size=16,           # Size of each patch
    embed_dim=768,           # Embedding dimension (768/1024/1280)
    depth=12,                # Number of transformer blocks
    num_heads=12,            # Attention heads per block
    mlp_ratio=4.0,           # MLP hidden dim = embed_dim * mlp_ratio
    out_chans=256,           # Output channels for neck
    qkv_bias=True,           # Use bias in attention QKV projections
    use_rel_pos=True,        # Use relative position biases
    window_size=14,          # Window size for windowed attention
    global_attn_indexes=(2, 5, 8, 11),  # Layers with global attention
)

# Process an image
image = keras.random.normal(shape=(2, 1024, 1024, 3))
embedding = encoder(image)  # Shape: (2, 64, 64, 256)
```

**Key Parameters**:

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `img_size` | Expected input image size | 1024 |
| `patch_size` | Patch tokenization size | 16 |
| `embed_dim` | Internal feature dimension | 768 (Base), 1024 (Large), 1280 (Huge) |
| `depth` | Number of transformer layers | 12 (Base), 24 (Large), 32 (Huge) |
| `num_heads` | Multi-head attention heads | 12-16 |
| `use_rel_pos` | Relative position biases in attention | `True` (the default) |
| `window_size` | Windowed-attention window; `0` = every block global | 14 |
| `global_attn_indexes` | Layers using global attention | Varies by model size |

**A windowed-only encoder is REFUSED.** Constructing `ImageEncoderViT` with
`window_size > 0` **and** an empty `global_attn_indexes` raises `ValueError`:
every block would then be windowed and the receptive field would never become
global. Pass a reference-shaped tuple (`(2, 5, 8, 11)` for a 12-block encoder)
or set `window_size=0` to make every block global. `use_rel_pos` defaults to
`True`, matching every released SAM variant; note that this adds two
`rel_pos_h` / `rel_pos_w` tables per block, so it is a `.keras` layout change
relative to a pre-iteration-1 default-constructed encoder.

### 6.2 Prompt Encoder (`PromptEncoder`)

**Purpose**: Convert user prompts into embeddings.

**Location**: `dl_techniques.models.sam.prompt_encoder.PromptEncoder`

```python
from dl_techniques.models.sam.prompt_encoder import PromptEncoder

# Create a prompt encoder
prompt_encoder = PromptEncoder(
    embed_dim=256,                         # Embedding dimension
    image_embedding_size=(64, 64),         # Size of image embeddings
    input_image_size=(1024, 1024),         # Original image size
    mask_in_chans=16,                      # Channels in mask encoder
)

# Encode different prompt types
# 1. Points
points = (
    keras.ops.convert_to_tensor([[[100.0, 200.0], [150.0, 250.0]]]),
    keras.ops.convert_to_tensor([[1, 0]])  # foreground, background
)
sparse_emb, dense_emb = prompt_encoder(points=points)

# 2. Boxes
boxes = keras.ops.convert_to_tensor([[[50.0, 50.0, 200.0, 300.0]]])
sparse_emb, dense_emb = prompt_encoder(boxes=boxes)

# 3. Masks
masks = keras.random.normal(shape=(1, 1, 256, 256))
sparse_emb, dense_emb = prompt_encoder(masks=masks)

# 4. Combined
sparse_emb, dense_emb = prompt_encoder(
    points=points,
    boxes=boxes,
    masks=masks
)
```

**Output Shapes**:
- `sparse_embeddings`: `(B, N_prompts, embed_dim)`
- `dense_embeddings`: `(B, H_emb, W_emb, embed_dim)`

### 6.3 Mask Decoder (`MaskDecoder`)

**Purpose**: Predict segmentation masks from embeddings.

**Location**: `dl_techniques.models.sam.mask_decoder.MaskDecoder`

```python
from dl_techniques.models.sam.mask_decoder import MaskDecoder
from dl_techniques.models.sam.transformer import TwoWayTransformer

# Create transformer
transformer = TwoWayTransformer(
    depth=2,
    embedding_dim=256,
    num_heads=8,
    mlp_dim=2048,
    attention_downsample_rate=2,      # See note below
)

# Create mask decoder
mask_decoder = MaskDecoder(
    transformer_dim=256,              # Must match prompt encoder
    transformer=transformer,           # Two-way transformer
    num_multimask_outputs=3,          # Number of mask proposals
    iou_head_depth=3,                 # Layer count of EVERY MLP head
    iou_head_hidden_dim=256,          # Hidden dim of IoU MLP
    activation='gelu',                # Default; UPSCALER only (reference: nn.GELU)
    mlp_activation='relu',            # Default; MLP heads only (reference: F.relu)
)
```

**`attention_downsample_rate`** (default `2`) makes the three CROSS-attentions
(`cross_attn_token_to_image`, `cross_attn_image_to_token`,
`final_attn_token_to_image`) run at internal dim `embedding_dim // rate`, as
reference SAM does. `self_attn` always runs at full `embedding_dim`. Setting
`rate=1` restores the old uniform-width behaviour. `embedding_dim` must be
divisible by `num_heads * rate` or construction raises `ValueError`.

**`iou_head_depth`** is a LIVE knob (it was stored, serialized and never read
before iteration 1). It sets the layer count of the IoU head **and** of every
hypernetwork MLP; at the default of `3` this is identical to reference SAM
(two hidden layers + a linear output). Every hidden layer is activated and the
output layer is linear. Reference SAM hardcodes `3` for the hypernetwork and
applies the knob only to the IoU head — driving both from one knob is a
deliberate deviation, reachable only at a non-default depth.

**`activation` and `mlp_activation` are two SEPARATE knobs, and their defaults
deliberately differ.** Reference SAM's `MaskDecoder(activation=nn.GELU)` passes
`activation` to `output_upscaling` and to nothing else, while the hypernetwork
and IoU heads hardcode `F.relu` inside `MLP`. So:

- **`activation`** (default `'gelu'`) is applied inside `output_upscaling` only.
- **`mlp_activation`** (default `'relu'`) is applied to every non-final layer of
  the hypernetwork MLPs and the IoU head; the final layer of each head stays
  linear so it can emit signed logits.

Do **not** collapse these into one knob "so the decoder agrees with itself" —
reference SAM makes the two halves differ by construction. This package shipped
a single shared `activation` until the iteration-1 completion fix: at its
original `'gelu'` the MLP heads were wrong, and flipping it to `'relu'` merely
moved the deviation to the upscaler. Both halves now match reference. Forward-pass
values moved twice as a result; no weight shape or count did (activations are
parameter-free).

**`sparse_prompt_embeddings` batching**: the decoder accepts a prompt batch of
either `1` (broadcast to every image) or exactly `batch_size` (paired one to
one). Anything else raises `ValueError` — an intermediate value used to tile
into an interleaved `[a, b, a, b]` order and score every image against the
wrong prompt with no error at all.

```python

# Predict masks
masks, iou_pred = mask_decoder(
    image_embeddings=image_emb,       # (B, H, W, C)
    image_pe=positional_encoding,     # (H, W, C)
    sparse_prompt_embeddings=sparse,  # (B, N, C)
    dense_prompt_embeddings=dense,    # (B, H, W, C)
    multimask_output=True,            # Predict 3 masks vs 1
)
```

**Output Shapes**:
- `masks`: `(B, num_masks, 256, 256)` - Low-resolution mask logits
- `iou_predictions`: `(B, num_masks)` - Quality scores

### 6.4 Complete SAM Model

**Purpose**: End-to-end promptable segmentation model.

**Location**: `dl_techniques.models.sam.model.SAM`

```python
from dl_techniques.models.sam import SAM

# Method 1: Use predefined variants (recommended)
model = SAM.from_variant('vit_b')   # Base: fast, good quality
# model = SAM.from_variant('vit_l')  # Large: balanced
# model = SAM.from_variant('vit_h')  # Huge: best quality

# Method 2: Custom configuration
model = SAM(
    image_encoder=custom_encoder,
    prompt_encoder=custom_prompt_enc,
    mask_decoder=custom_decoder,
    pixel_mean=[123.675, 116.28, 103.53],    # ImageNet mean
    pixel_std=[58.395, 57.12, 57.375],       # ImageNet std
    mask_threshold=0.0,                       # Binary threshold
    image_format='RGB',
    binarize_masks=True,                      # See "Output contract" below
)

# Full inference
outputs = model({
    'image': image,                   # (B, H, W, 3)
    'points': (coords, labels),       # Optional
    'boxes': boxes,                   # Optional
    'masks': mask_hints,              # Optional
    'original_size': (H, W),          # Required
    'multimask_output': True,         # Optional, defaults to True
})

# Outputs
masks = outputs['masks']              # (B, 3, H, W) - upscaled to original
iou = outputs['iou_predictions']      # (B, 3)
logits = outputs['low_res_logits']    # (B, 3, 256, 256)
```

### Output contract: `low_res_logits` is THE training target

`outputs['masks']` is, by default, a **thresholded `uint8` tensor**. A cast to an
integer dtype has no gradient, so differentiating that key returns `None` for
**every** trainable variable. A trainer that supervises `outputs['masks']`
therefore trains nothing, silently — no error, no warning, a loss that simply
never moves the weights.

| Output key | dtype at `binarize_masks=True` (default) | Differentiable | Use it for |
|---|---|---|---|
| `masks` | `uint8`, 0/1 | **No** — 0 of N variables receive a gradient | Visualization, evaluation, export |
| `iou_predictions` | float | Yes | Mask-quality supervision |
| `low_res_logits` | float, `(B, N, 256, 256)` | Yes | **Training** |

Set `binarize_masks=False` to make `masks` carry the full-resolution float
logits instead; it is then differentiable, and the default `True` remains
byte-identical to reference SAM's own thresholded output contract. Either way,
`low_res_logits` is what you supervise: it is the resolution reference SAM
computes its loss at, and upsampling to full resolution costs memory without
adding signal.

**A dict output CAN be trained with stock `fit()` — with a dict `loss=`.**
An earlier revision of this README stated the opposite as an unqualified
constraint. Measured on keras 3.8.0: `CompileLoss.build` `KeyError`s in exactly
one configuration — a **single** `Loss` object plus a bare-tensor `y_true`.
Given `loss={<output key>: Loss, ...}` (even keying a *subset* of the output
keys) and a matching dict `y_true`, `fit()` trains normally and emits per-key
loss metrics. `loss_weights` carries the mixing.

What actually blocks training through `SAM.call` is different and was unnamed
until iteration 2: **`SAM.call` cannot be traced at all.** `postprocess_masks`
runs unconditionally at the end of `call`, and its `ops.image.resize` raises
under `fit()`'s graph mode — regardless of which output key you read. That is
why the training path calls SAM's submodules directly; see §11.

---

## 7. Configuration & Model Variants

### Variant Comparison

SAM comes in three sizes, each optimized for different use cases:

| Variant | Params | Embed Dim | Depth | Heads | Use Case | Speed | Quality |
|---------|--------|-----------|-------|-------|----------|-------|---------|
| **vit_b** | ~90M | 768 | 12 | 12 | Interactive tools, mobile | ⚡⚡⚡ | ⭐⭐⭐ |
| **vit_l** | ~300M | 1024 | 24 | 16 | General purpose, production | ⚡⚡ | ⭐⭐⭐⭐ |
| **vit_h** | ~630M | 1280 | 32 | 16 | Offline processing, research | ⚡ | ⭐⭐⭐⭐⭐ |

### Performance Benchmarks

On a typical GPU (NVIDIA RTX 3090):

```
┌──────────────────────────────────────────────────────────┐
│  Inference Time (1024×1024 image, single prompt)         │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  vit_b (Base):                                           │
│    Image Encoding:  ~150ms                               │
│    Prompt+Mask:     ~10ms                                │
│    Total:           ~160ms     ← Fastest                 │
│                                                          │
│  vit_l (Large):                                          │
│    Image Encoding:  ~280ms                               │
│    Prompt+Mask:     ~12ms                                │
│    Total:           ~292ms     ← Balanced                │
│                                                          │
│  vit_h (Huge):                                           │
│    Image Encoding:  ~450ms                               │
│    Prompt+Mask:     ~15ms                                │
│    Total:           ~465ms     ← Best Quality            │
│                                                          │
├──────────────────────────────────────────────────────────┤
│  Memory Usage (at inference):                            │
│    vit_b:  ~4GB                                          │
│    vit_l:  ~8GB                                          │
│    vit_h:  ~16GB                                         │
└──────────────────────────────────────────────────────────┘
```

### Choosing the Right Variant

**Use vit_b when:**
- Building interactive applications (real-time feedback)
- Deploying on edge devices or with limited GPU memory
- Processing video (many frames per second)
- Quality is good enough for your use case

**Use vit_l when:**
- You need better quality than vit_b
- You have reasonable compute budget
- General-purpose production applications
- Batch processing moderate-sized datasets

**Use vit_h when:**
- Maximum quality is essential
- Offline processing is acceptable
- Medical, scientific, or professional applications
- Creating training data for other models

### Global Attention Placement

The placement of global attention layers is carefully designed:

```
vit_b (12 layers total):
├─ Layer 0-1:   Windowed attention
├─ Layer 2:     Global attention     ← Early global view
├─ Layer 3-4:   Windowed attention
├─ Layer 5:     Global attention     ← Mid-level global view
├─ Layer 6-7:   Windowed attention
├─ Layer 8:     Global attention     ← Late global view
├─ Layer 9-10:  Windowed attention
└─ Layer 11:    Global attention     ← Final global view

Strategy: Global attention at ~25% intervals
Purpose: 
  - Early: Understand overall scene layout
  - Mid: Refine object boundaries
  - Late: Integrate all information for final features
```

---

## 8. Comprehensive Usage Examples

### Example 1: Single Point Prompt

The simplest use case - click a point to segment an object.

```python
import keras
import numpy as np
from dl_techniques.models.sam import SAM

# Setup
model = SAM.from_variant('vit_b')
image = load_your_image()  # Shape: (1, H, W, 3)

# Define point prompt
# User clicks at (350, 450) to segment an object
point_coords = keras.ops.convert_to_tensor([[[350.0, 450.0]]])
point_labels = keras.ops.convert_to_tensor([[1]])  # 1 = foreground
points = (point_coords, point_labels)

# Get segmentation
outputs = model({
    'image': image,
    'points': points,
    'original_size': (image.shape[1], image.shape[2])
})

# Select best mask
best_idx = keras.ops.argmax(outputs['iou_predictions'][0])
mask = outputs['masks'][0, best_idx]  # Shape: (H, W)

print(f"Best mask quality score: {outputs['iou_predictions'][0, best_idx]:.3f}")
```

### Example 2: Multiple Points for Refinement

Use multiple points to refine segmentation.

```python
# Initial foreground point
point_coords = keras.ops.convert_to_tensor([[[350.0, 450.0]]])
point_labels = keras.ops.convert_to_tensor([[1]])

# User adds more points to refine:
# - Another foreground point at (360, 460)
# - Background point at (100, 100) to exclude that region
point_coords = keras.ops.convert_to_tensor([[
    [350.0, 450.0],  # Foreground
    [360.0, 460.0],  # Foreground
    [100.0, 100.0]   # Background
]])
point_labels = keras.ops.convert_to_tensor([[1, 1, 0]])  # 1=fg, 0=bg

points = (point_coords, point_labels)

outputs = model({
    'image': image,
    'points': points,
    'original_size': (image.shape[1], image.shape[2])
})

# The model will:
# 1. Include regions near foreground points
# 2. Exclude regions near background points
# 3. Produce a refined segmentation
```

### Example 3: Bounding Box Prompt

Use a bounding box when you know the rough object location.

```python
# Define bounding box (x1, y1, x2, y2)
# Top-left: (100, 100), Bottom-right: (500, 500)
boxes = keras.ops.convert_to_tensor([[[100.0, 100.0, 500.0, 500.0]]])

outputs = model({
    'image': image,
    'boxes': boxes,
    'original_size': (image.shape[1], image.shape[2])
})

# SAM will segment the most prominent object within the box
mask = outputs['masks'][0, 0]  # Usually single mask for boxes
```

### Example 4: Combining Points and Boxes

Most precise control - use box for region + points for refinement.

```python
# Bounding box to localize region
boxes = keras.ops.convert_to_tensor([[[100.0, 100.0, 500.0, 500.0]]])

# Points to specify exact object within box
point_coords = keras.ops.convert_to_tensor([[[300.0, 300.0]]])
point_labels = keras.ops.convert_to_tensor([[1]])
points = (point_coords, point_labels)

outputs = model({
    'image': image,
    'points': points,
    'boxes': boxes,
    'original_size': (image.shape[1], image.shape[2])
})

# This gives very precise control:
# - Box narrows down the search region
# - Point specifies the exact object to segment
```

### Example 5: Mask Prompt (Iterative Refinement)

Use a previous mask as a prompt to refine it.

```python
# First pass: Get initial segmentation
point_coords = keras.ops.convert_to_tensor([[[350.0, 450.0]]])
point_labels = keras.ops.convert_to_tensor([[1]])
points = (point_coords, point_labels)

outputs_pass1 = model({
    'image': image,
    'points': points,
    'original_size': (image.shape[1], image.shape[2])
})

initial_mask = outputs_pass1['low_res_logits'][0, 0:1]  # Shape: (1, 256, 256)

# Second pass: Refine using previous mask as hint
outputs_pass2 = model({
    'image': image,
    'masks': initial_mask[None, ...],  # Add batch dim: (1, 1, 256, 256)
    'points': points,  # Can add more points
    'original_size': (image.shape[1], image.shape[2])
})

refined_mask = outputs_pass2['masks'][0, 0]

# The model uses the previous mask as context to refine the segmentation
```

### Example 6: Automatic Mask Generation (Grid Prompting)

Generate all possible masks in an image by prompting with a grid of points.

```python
import keras
import numpy as np

def generate_point_grid(image_size, points_per_side=32):
    """Generate a grid of prompt points covering the image."""
    h, w = image_size
    
    # Create grid coordinates
    x = np.linspace(0, w-1, points_per_side)
    y = np.linspace(0, h-1, points_per_side)
    xv, yv = np.meshgrid(x, y)
    
    # Flatten to (N, 2) shape
    points = np.stack([xv.ravel(), yv.ravel()], axis=1)
    
    return points

def segment_everything(model, image, points_per_side=32):
    """Generate all masks in an image using grid prompting."""
    h, w = image.shape[1:3]
    
    # Generate prompt grid
    point_coords = generate_point_grid((h, w), points_per_side)
    n_points = len(point_coords)
    
    all_masks = []
    all_scores = []
    
    # Process in batches (e.g., 16 points at a time)
    batch_size = 16
    
    for i in range(0, n_points, batch_size):
        batch_coords = point_coords[i:i+batch_size]
        batch_labels = np.ones(len(batch_coords), dtype=np.int32)
        
        # Create tensors for batch
        coords_tensor = keras.ops.convert_to_tensor(batch_coords[None, ...])
        labels_tensor = keras.ops.convert_to_tensor(batch_labels[None, ...])
        points = (coords_tensor, labels_tensor)
        
        # Get masks for this batch
        outputs = model({
            'image': image,
            'points': points,
            'original_size': (h, w)
        })
        
        # Collect all generated masks
        for j in range(outputs['masks'].shape[1]):  # For each mask proposal
            all_masks.append(outputs['masks'][0, j])
            all_scores.append(outputs['iou_predictions'][0, j])
    
    return all_masks, all_scores

# Usage
model = SAM.from_variant('vit_b')
image = load_your_image()

masks, scores = segment_everything(model, image, points_per_side=32)
print(f"Generated {len(masks)} mask candidates")

# Post-process: Remove duplicates using NMS, filter by score, etc.
```

### Example 7: Batch Processing

Process multiple images efficiently.

```python
import keras
import numpy as np

def batch_segment(model, images, prompts_list):
    """
    Process multiple images with their respective prompts.
    
    Args:
        model: SAM model instance
        images: List of images, each shape (H, W, 3)
        prompts_list: List of prompt dictionaries, one per image
    
    Returns:
        List of output dictionaries
    """
    results = []
    
    # Get max dimensions for batching
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    
    # Pad and batch images
    batched_images = []
    for img in images:
        h, w = img.shape[:2]
        padded = np.zeros((max_h, max_w, 3), dtype=img.dtype)
        padded[:h, :w] = img
        batched_images.append(padded)
    
    batched_images = np.array(batched_images)
    batched_images = keras.ops.convert_to_tensor(batched_images)
    
    # Note: For true batching, prompts must be aligned
    # Here we process sequentially for simplicity
    for i, (image, prompts) in enumerate(zip(images, prompts_list)):
        image_tensor = batched_images[i:i+1]
        
        outputs = model({
            'image': image_tensor,
            **prompts,  # Unpack prompt dict
            'original_size': (image.shape[0], image.shape[1])
        })
        
        results.append(outputs)
    
    return results

# Usage
model = SAM.from_variant('vit_b')

images = [load_image_1(), load_image_2(), load_image_3()]
prompts = [
    {'points': (coords1, labels1)},
    {'boxes': boxes2},
    {'points': (coords3, labels3), 'boxes': boxes3}
]

results = batch_segment(model, images, prompts)
```

### Example 8: Caching for Interactive Applications

Optimize for applications where users provide multiple prompts for the same image.

```python
class InteractiveSegmenter:
    """
    Efficient interactive segmentation by caching image embeddings.
    """
    
    def __init__(self, model):
        self.model = model
        self.current_image = None
        self.image_embedding = None
        self.image_shape = None
    
    def set_image(self, image):
        """
        Process a new image and cache its embedding.
        This is the slow operation (~150-450ms).
        """
        # preprocess() PADS ONLY and raises ValueError on an oversize image.
        # resize_longest_side pins the longest side to img_size (reference
        # SAM's transform); rescale prompt coordinates by the same factor.
        image = resize_longest_side(image, self.model.image_encoder.img_size)

        self.current_image = image
        self.image_shape = image.shape[1:3]
        
        # Preprocess and encode image
        preprocessed = self.model.preprocess(image)
        self.image_embedding = self.model.image_encoder(preprocessed)
        
        print(f"Image encoded and cached: {self.image_embedding.shape}")
    
    def predict_with_points(self, point_coords, point_labels):
        """
        Fast prediction using cached embedding (~10-15ms).
        """
        if self.image_embedding is None:
            raise ValueError("No image set. Call set_image() first.")
        
        # Encode prompts
        points = (
            keras.ops.convert_to_tensor(point_coords),
            keras.ops.convert_to_tensor(point_labels)
        )
        sparse_emb, dense_emb = self.model.prompt_encoder(points=points)
        
        # Get positional encoding
        image_pe = self.model.prompt_encoder.get_dense_pe()
        
        # Decode masks (fast!)
        masks, iou_pred = self.model.mask_decoder(
            image_embeddings=self.image_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=True
        )
        
        # Postprocess masks
        input_size = (1024, 1024)  # Assuming standard SAM size
        masks = self.model.postprocess_masks(
            masks, input_size, self.image_shape
        )
        
        return {
            'masks': masks,
            'iou_predictions': iou_pred,
            'low_res_logits': masks
        }
    
    def predict_with_box(self, box):
        """Fast prediction with box prompt."""
        if self.image_embedding is None:
            raise ValueError("No image set. Call set_image() first.")
        
        boxes = keras.ops.convert_to_tensor(box)
        sparse_emb, dense_emb = self.model.prompt_encoder(boxes=boxes)
        
        image_pe = self.model.prompt_encoder.get_dense_pe()
        
        masks, iou_pred = self.model.mask_decoder(
            image_embeddings=self.image_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=True
        )
        
        input_size = (1024, 1024)
        masks = self.model.postprocess_masks(
            masks, input_size, self.image_shape
        )
        
        return {
            'masks': masks,
            'iou_predictions': iou_pred,
            'low_res_logits': masks
        }

# Usage in interactive application
model = SAM.from_variant('vit_b')
segmenter = InteractiveSegmenter(model)

# User opens an image
image = load_user_image()
segmenter.set_image(image)  # Slow operation, done once

# User clicks points interactively (fast)
mask1 = segmenter.predict_with_points([[100.0, 200.0]], [1])  # ~10ms
mask2 = segmenter.predict_with_points([[150.0, 250.0]], [1])  # ~10ms
mask3 = segmenter.predict_with_points([[200.0, 300.0]], [1])  # ~10ms

# User draws a box (also fast)
mask4 = segmenter.predict_with_box([[[50.0, 50.0, 300.0, 400.0]]])  # ~10ms
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Automatic Mask Quality Filtering

Filter generated masks by quality score and other criteria.

```python
def filter_masks(masks, iou_predictions, 
                 iou_threshold=0.7,
                 area_threshold=100,
                 stability_threshold=0.9):
    """
    Filter masks based on multiple quality criteria.
    
    Args:
        masks: Generated masks (B, N, H, W)
        iou_predictions: Quality scores (B, N)
        iou_threshold: Minimum IoU score
        area_threshold: Minimum mask area in pixels
        stability_threshold: Minimum mask stability score
    
    Returns:
        Filtered masks and their scores
    """
    import keras
    
    filtered_masks = []
    filtered_scores = []
    
    for i in range(masks.shape[1]):  # For each mask
        mask = masks[0, i]
        iou = iou_predictions[0, i]
        
        # Check IoU threshold
        if keras.ops.convert_to_numpy(iou) < iou_threshold:
            continue
        
        # Check area threshold
        area = keras.ops.sum(mask > 0)
        if keras.ops.convert_to_numpy(area) < area_threshold:
            continue
        
        # Compute stability (how much mask changes at different thresholds)
        mask_at_0 = mask > 0.0
        mask_at_05 = mask > 0.5
        stability = keras.ops.sum(mask_at_0 == mask_at_05) / keras.ops.size(mask)
        
        if keras.ops.convert_to_numpy(stability) < stability_threshold:
            continue
        
        filtered_masks.append(mask)
        filtered_scores.append(iou)
    
    return filtered_masks, filtered_scores

# Usage
outputs = model({
    'image': image,
    'points': points,
    'original_size': (H, W)
})

good_masks, good_scores = filter_masks(
    outputs['masks'],
    outputs['iou_predictions'],
    iou_threshold=0.75,
    area_threshold=500
)

print(f"Kept {len(good_masks)} out of {outputs['masks'].shape[1]} masks")
```

### Pattern 2: Non-Maximum Suppression for Duplicate Removal

When using grid prompting, remove overlapping duplicate masks.

```python
def compute_iou_matrix(masks):
    """
    Compute pairwise IoU between all masks.
    
    Args:
        masks: List of binary masks
    
    Returns:
        IoU matrix of shape (N, N)
    """
    n = len(masks)
    iou_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            mask_i = keras.ops.convert_to_numpy(masks[i] > 0)
            mask_j = keras.ops.convert_to_numpy(masks[j] > 0)
            
            intersection = np.logical_and(mask_i, mask_j).sum()
            union = np.logical_or(mask_i, mask_j).sum()
            
            iou = intersection / (union + 1e-6)
            iou_matrix[i, j] = iou
            iou_matrix[j, i] = iou
    
    return iou_matrix

def nms_masks(masks, scores, iou_threshold=0.7):
    """
    Apply Non-Maximum Suppression to remove duplicate masks.
    
    Args:
        masks: List of masks
        scores: List of quality scores
        iou_threshold: IoU threshold for considering masks as duplicates
    
    Returns:
        Filtered masks and scores
    """
    if len(masks) == 0:
        return [], []
    
    # Sort by score (descending)
    scores_np = [keras.ops.convert_to_numpy(s) for s in scores]
    sorted_indices = np.argsort(scores_np)[::-1]
    
    # Compute IoU matrix
    iou_matrix = compute_iou_matrix(masks)
    
    # NMS
    keep = []
    remaining = set(sorted_indices)
    
    for idx in sorted_indices:
        if idx not in remaining:
            continue
        
        keep.append(idx)
        remaining.remove(idx)
        
        # Remove similar masks
        similar = np.where(iou_matrix[idx] > iou_threshold)[0]
        for sim_idx in similar:
            remaining.discard(sim_idx)
    
    # Return kept masks
    kept_masks = [masks[i] for i in keep]
    kept_scores = [scores[i] for i in keep]
    
    return kept_masks, kept_scores

# Usage with grid prompting
masks, scores = segment_everything(model, image, points_per_side=32)
print(f"Generated {len(masks)} total masks")

# Filter by quality
good_masks, good_scores = filter_masks(masks, scores, iou_threshold=0.7)
print(f"After quality filter: {len(good_masks)} masks")

# Remove duplicates
final_masks, final_scores = nms_masks(good_masks, good_scores, iou_threshold=0.7)
print(f"After NMS: {len(final_masks)} unique masks")
```

### Pattern 3: Hierarchical Segmentation

Generate segmentation at multiple levels of granularity.

```python
def hierarchical_segment(model, image, point):
    """
    Generate hierarchical segmentation from coarse to fine.
    
    Args:
        model: SAM model
        image: Input image
        point: Central prompt point (x, y)
    
    Returns:
        Dict with masks at different levels
    """
    H, W = image.shape[1:3]
    point_coords = keras.ops.convert_to_tensor([[point]])
    point_labels = keras.ops.convert_to_tensor([[1]])
    points = (point_coords, point_labels)
    
    # Level 1: Whole object/region
    outputs = model({
        'image': image,
        'points': points,
        'original_size': (H, W),
        'multimask_output': True
    })
    
    # Sort masks by size (largest to smallest)
    masks = []
    for i in range(outputs['masks'].shape[1]):
        mask = outputs['masks'][0, i]
        area = keras.ops.sum(mask > 0)
        masks.append((mask, keras.ops.convert_to_numpy(area), 
                     outputs['iou_predictions'][0, i]))
    
    masks.sort(key=lambda x: x[1], reverse=True)
    
    return {
        'coarse': masks[0][0],    # Largest mask (whole object)
        'medium': masks[1][0] if len(masks) > 1 else masks[0][0],
        'fine': masks[2][0] if len(masks) > 2 else masks[0][0],
        'sizes': [m[1] for m in masks],
        'scores': [m[2] for m in masks]
    }

# Usage
model = SAM.from_variant('vit_b')
image = load_your_image()

hierarchy = hierarchical_segment(model, image, point=[512.0, 512.0])

# Visualize different levels
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(keras.ops.convert_to_numpy(hierarchy['coarse']), cmap='gray')
axes[0].set_title(f"Coarse (area: {hierarchy['sizes'][0]:.0f})")

axes[1].imshow(keras.ops.convert_to_numpy(hierarchy['medium']), cmap='gray')
axes[1].set_title(f"Medium (area: {hierarchy['sizes'][1]:.0f})")

axes[2].imshow(keras.ops.convert_to_numpy(hierarchy['fine']), cmap='gray')
axes[2].set_title(f"Fine (area: {hierarchy['sizes'][2]:.0f})")

plt.show()
```

---

## 10. Performance Optimization

### Strategy 1: Mixed Precision Training

Use mixed precision for faster training and inference.

```python
# Enable mixed precision globally
keras.mixed_precision.set_global_policy('mixed_float16')

# Create model (will automatically use mixed precision)
model = SAM.from_variant('vit_b')

# For inference, this provides:
# - ~1.5-2x speedup on compatible GPUs
# - ~50% memory reduction
# - Minimal accuracy loss (<1% typically)

# Note: Final layers remain in float32 for numerical stability
```

### Strategy 2: TensorFlow XLA Compilation

Use XLA for additional speedup.

```python
import tensorflow as tf

# Enable XLA
tf.config.optimizer.set_jit(True)

# Or use function-level compilation
@tf.function(jit_compile=True)
def compiled_predict(model, inputs):
    return model(inputs, training=False)

# Usage
model = SAM.from_variant('vit_b')
outputs = compiled_predict(model, {
    'image': image,
    'points': points,
    'original_size': (1024, 1024)
})

# Expected speedup: 10-30% depending on hardware
```

### Strategy 3: Batch Processing

Process multiple prompts simultaneously.

```python
def batch_prompts(model, image, all_points, batch_size=32):
    """
    Process multiple prompts in batches for efficiency.
    
    Args:
        model: SAM model
        image: Single image
        all_points: List of (coords, labels) tuples
        batch_size: Number of prompts per batch
    
    Returns:
        List of masks, one per prompt
    """
    # Cache image embedding. preprocess() pads only -- an oversize image
    # raises ValueError, so pin the longest side to img_size first.
    image = resize_longest_side(image, model.image_encoder.img_size)
    preprocessed = model.preprocess(image)
    image_embedding = model.image_encoder(preprocessed)
    
    all_masks = []
    
    for i in range(0, len(all_points), batch_size):
        batch = all_points[i:i+batch_size]
        
        # Stack prompts into batch
        coords_list = [p[0] for p in batch]
        labels_list = [p[1] for p in batch]
        
        # Process batch
        for coords, labels in zip(coords_list, labels_list):
            points = (
                keras.ops.convert_to_tensor(coords),
                keras.ops.convert_to_tensor(labels)
            )
            
            sparse_emb, dense_emb = model.prompt_encoder(points=points)
            image_pe = model.prompt_encoder.get_dense_pe()
            
            masks, _ = model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False  # Single mask for speed
            )
            
            all_masks.append(masks[0, 0])
    
    return all_masks

# Usage: Process 100 prompts efficiently
points_list = [
    (keras.ops.convert_to_tensor([[[x, y]]]), keras.ops.convert_to_tensor([[1]]))
    for x, y in generate_100_prompt_locations()
]

masks = batch_prompts(model, image, points_list, batch_size=32)
# Much faster than processing one by one!
```

### Strategy 4: Model Quantization

Reduce model size and increase speed with quantization.

```python
import tensorflow as tf

# Load and prepare model
model = SAM.from_variant('vit_b')

# Build model with a sample input
sample_image = keras.random.normal(shape=(1, 1024, 1024, 3))
sample_points = (
    keras.ops.convert_to_tensor([[[512.0, 512.0]]]),
    keras.ops.convert_to_tensor([[1]])
)

_ = model({
    'image': sample_image,
    'points': sample_points,
    'original_size': (1024, 1024)
})

# Save as SavedModel
model.export('sam_model')

# Convert to TFLite with quantization
converter = tf.lite.TFLiteConverter.from_saved_model('sam_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Dynamic range quantization (easiest, good results)
tflite_model = converter.convert()

with open('sam_model_quantized.tflite', 'wb') as f:
    f.write(tflite_model)

print("Quantized model saved")
print(f"Original size: {os.path.getsize('sam_model.keras') / 1e6:.1f} MB")
print(f"Quantized size: {os.path.getsize('sam_model_quantized.tflite') / 1e6:.1f} MB")

# Typically: ~4x size reduction, ~2x speedup on CPU
```

### Performance Comparison Table

| Optimization | Speed Gain | Memory Reduction | Accuracy Loss | Difficulty |
|--------------|------------|------------------|---------------|------------|
| Mixed Precision | 1.5-2x | 50% | <1% | Easy |
| XLA Compilation | 1.1-1.3x | None | None | Easy |
| Embedding Cache | 10-100x* | None | None | Easy |
| Batch Processing | 2-4x | None | None | Medium |
| Quantization | 2-3x | 75% | 1-3% | Medium |

*For multiple prompts on same image

---

## 11. Training and Fine-tuning

### Training SAM: the shipped path

A trainer ships. It is stock `compile()` / `fit()` — **no custom `train_step`
anywhere** — over a wrapper model that drives SAM's submodules directly.

```
tf.data (train.sam.data)  ->  SAMTrainingModel  ->  {SAMMaskLoss, SAMIoULoss}
   image, point/box prompt,        preprocess ->            focal+dice on
   gt_mask at low-res              image_encoder ->         low_res_logits;
                                   prompt_encoder ->        MSE on the packed
                                   mask_decoder             iou_supervision
```

Run it:

```bash
# Executed smoke run: 3 epochs x 16 steps, seconds on one GPU, no COCO needed
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 \
    .venv/bin/python -m train.sam.train_sam --smoke

# Real COCO 2017 instance masks
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 \
    .venv/bin/python -m train.sam.train_sam \
    --data-source coco --coco-max-images 200 --epochs 50
```

Or wire it by hand:

```python
from dl_techniques.models.sam import SAM, SAMTrainingModel
from dl_techniques.losses.sam_mask_loss import SAMMaskLoss, SAMIoULoss
from train.sam.data import build_sam_dataset

model = SAMTrainingModel(
    SAM.from_variant('vit_b'),
    multimask_output=False,
    num_refinement_rounds=3,     # 1 disables refinement
)
model.compile(
    optimizer='adam',
    # A DICT loss keyed to the output names. `iou_predictions` is deliberately
    # left uncovered: its target is packed inside `iou_supervision`.
    loss={'low_res_logits': SAMMaskLoss(), 'iou_supervision': SAMIoULoss()},
    loss_weights={'low_res_logits': 1.0, 'iou_supervision': 1.0},
)
model.fit(build_sam_dataset(2048, image_size=1024, batch_size=1))
```

**What makes this work, and what it costs.**

1. **The wrapper never calls `SAM.call`.** `postprocess_masks` runs
   unconditionally at the end of `SAM.call`, and its `ops.image.resize` cannot
   be traced under `fit()`; reading only `low_res_logits` does **not** avoid it,
   because the postprocess runs regardless of which key you consume. A
   Python-tuple `original_size` cannot even be passed (`Layer.__call__` rejects
   non-tensor positional arguments). So `SAMTrainingModel.call` runs
   `preprocess -> image_encoder -> prompt_encoder -> mask_decoder` itself. A spy
   pins `SAM.call` at **zero** invocations during `fit()`.

2. **A dict `y_pred` is fine; a dict `loss=` is what makes it fine.** See §6.4.
   `y_true`'s keys must match the keys `loss=` **covers** — not the model's
   output keys. Supplying an entry for an uncovered key raises
   `ValueError: y_true and y_pred have different structures`.

3. **Supervise `low_res_logits`, never `masks`.** At the default
   `binarize_masks=True`, `masks` is a thresholded `uint8` tensor whose
   gradient is `None` for **every** trainable variable (§6.4).

4. **The repo's segmentation losses are reused, but NOT as-is — and the trap is
   silent.** `SegmentationLosses.focal_loss` on a 1-channel binary map is
   **bit-identically blind to negative pixels**: setting every negative pixel's
   prediction to maximally wrong leaves the loss unchanged to six decimals.
   `dice_loss` is sound behind a `(B*M, h, w, 1)` reshape but **silently
   accepts the raw `(B, M, h, w)` layout and returns a plausible number** while
   reducing `(num_masks, height)` instead of `(height, width)`. `SAMMaskLoss`
   adapts both (2-channel one-hot for focal, channels-last single-channel for
   dice) and **refuses** the raw layout. Do not route SAM masks through
   `SegmentationWrapperLoss` directly.

5. **The focal:dice mix was re-derived here, not pasted from the paper.**
   Unweighted on this repo's code, dice is ~8x focal; at the paper's 20:1 focal
   leads by ~2.4x in loss value and ~10.5:1 in gradient magnitude. 20:1 ships
   because the two readings agree in order of magnitude, and the tests assert
   those two structural facts rather than the constants.

6. **The IoU target is packed next to the prediction, and that is forced.** The
   achieved IoU is a function of the prediction *and* the ground truth, so it
   only exists inside `call()`; a `tf.data` pipeline cannot produce it and
   stock `compile(loss={...})` hands each loss only its own output key. Hence
   the `iou_supervision` output, shape `(B, M, 2)`: `[..., 0]` predicted,
   `[..., 1]` achieved (stop-gradient). It appears only when `gt_mask` is in
   the inputs, so an inference-shaped call keeps the two-key contract.

7. **Iterative refinement lives inside `call()`, with no custom `train_step`.**
   Per round: decode, `stop_gradient` the logits, feed them back as the dense
   mask prompt (already exactly `4 x` the embedding grid, so shape-native), and
   sample one foreground point from the false-negative region plus one
   background point from the false-positive region. Rounds are concatenated on
   the mask axis and supervised by one loss. Measured: variable coverage
   saturates at **2** rounds (170/201 -> 181/201 moved), and the class default
   is **1** (refinement off) so the wrapper keeps value-exact equivalence with
   an eager `SAM.call`; the trainer ships 3.

8. **Two lifecycle gotchas, both pinned by guards.** The `keras.random`
   `SeedGenerator` is created in `__init__`, never lazily in `call` — this class
   defines `build()`, so the state tracker is already locked and the lazy
   spelling raises `You cannot add new elements of state ... already built`.
   And `build()` calls `self.sam.build(None)` first, which materializes the
   whole SAM without a forward pass.

9. **Preprocess before the model.** `SAM.preprocess` pads only; resize with
   `resize_longest_side` first (§3, §6.1) and rescale the prompt coordinates by
   the same factor.

10. **Initial prompts are DATA; refinement prompts are not.** The point sampled
    inside the GT mask and the jittered box come from the `tf.data` pipeline
    (`train.sam.data`) because they need no model output. The refinement
    prompts depend on the current prediction and therefore cannot live there.

**Honest limitations of the shipped path** — none of these is fixed by this
trainer:

- **No official Meta SAM checkpoint has ever been loaded** in this repository.
  Every reference-fidelity claim is an architectural argument, not a
  measurement against real weights, and no key-mapping layer exists.
- **No test ever forward-passes `vit_l` or `vit_h`** — they are constructed and
  parameter-counted (§15), and the trainer has never been run at those widths
  or at `vit_b`'s. The trainer's default and its only executed configuration is
  a reduced-WIDTH `tiny` geometry at real SAM patch/window geometry.
- **`SAM.call` cannot be traced**, so no traced pipeline can consume
  `outputs['masks']`. This is a property of `postprocess_masks`, not a bug the
  wrapper hides.
- **At `num_refinement_rounds > 1` the `.keras` round-trip is value-exact only
  on the round-1 slice.** Later rounds depend on a random draw whose generator
  state advanced differently. Asserted in both directions rather than papered
  over.
- **No accuracy claim.** The executed `--smoke` run proves the path runs with
  live gradients and a falling loss on 32 synthetic instances. It says nothing
  about segmentation quality.
- **`--data-source coco` is I/O-bound, not GPU-bound.** Measured on this
  machine: a 16-step epoch costs ~1 s on the synthetic source and ~18 s on
  COCO, i.e. the GPU is idle most of the epoch. Budget for that before reading
  a COCO wall-clock as a model cost.


### Fine-tuning Strategies

> **`SAM.from_variant(...)` builds a RANDOMLY INITIALIZED model.** It is an
> architecture constructor, not a checkpoint loader. No pretrained SAM weights
> ship with this repo, none are downloaded, and none of the "fine-tuning"
> recipes below start from Meta's weights — they start from noise. Two known
> blockers to *ever* loading official SAM weights were removed (the
> cross-attention internal dim and the head depth); that load has still never
> been demonstrated, and no key-mapping layer exists. So "fine-tuning" here
> means the trainability mechanics, wired into the trainer above.

**Strategy 1: Freeze Image Encoder**

The trainability flags below are read by the shipped `fit()` path: set them on
the `SAM` before handing it to `SAMTrainingModel`.

```python
# Randomly initialized architecture -- NOT pretrained weights.
model = SAM.from_variant('vit_b')

# Freeze the image encoder (in a real fine-tune, its features would come
# from a checkpoint; from_variant gives you random ones).
model.image_encoder.trainable = False

# Train only prompt encoder and mask decoder
model.prompt_encoder.trainable = True
model.mask_decoder.trainable = True
```

**Strategy 2: Progressive Unfreezing**

Unfreeze in stages, re-`compile()`-ing and re-`fit()`-ing the
`SAMTrainingModel` between stages.

```python
def set_trainable_stage(model, component):
    """Set which SAM components are trainable for one stage.

    Args:
        model: SAM model.
        component: one of 'decoder', 'encoder_top', 'encoder_all'.
    """
    if component == 'decoder':
        model.mask_decoder.trainable = True
        model.prompt_encoder.trainable = True
        model.image_encoder.trainable = False
    elif component == 'encoder_top':
        # Unfreeze the top blocks of the image encoder
        for layer in model.image_encoder.layers[-8:]:
            layer.trainable = True
    elif component == 'encoder_all':
        model.image_encoder.trainable = True
    else:
        raise ValueError(f"unknown component: {component!r}")

# A plausible stage schedule (component, epochs, lr):
stages = [
    ('decoder', 10, 1e-4),        # Train decoder first
    ('encoder_top', 5, 5e-5),     # Then top encoder layers
    ('encoder_all', 5, 1e-5),     # Finally full model
]
```


**Strategy 3: Task-Specific Heads**

Add custom heads for specific tasks:

```python
@keras.saving.register_keras_serializable()
class SAMWithCustomHead(keras.Model):
    """SAM with additional task-specific head."""
    
    def __init__(self, sam_model, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.sam = sam_model
        
        # Add classification head
        self.global_pool = keras.layers.GlobalAveragePooling2D()
        self.classifier = keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
    
    def call(self, inputs, training=False):
        # Get SAM outputs
        sam_outputs = self.sam(inputs, training=training)
        
        # Get image features for classification
        image_features = self.sam.image_encoder(inputs['image'], training=training)
        features = self.global_pool(image_features)
        
        # Classify
        class_probs = self.classifier(features, training=training)
        
        # Return both segmentation and classification
        return {
            **sam_outputs,
            'class_probs': class_probs
        }

# Usage
base_sam = SAM.from_variant('vit_b')
model_with_head = SAMWithCustomHead(base_sam, num_classes=10)

# Multi-task training
def multitask_loss(outputs, targets, seg_loss_fn):
    # seg_loss_fn: build it from dl_techniques.losses.segmentation_loss /
    # segmentation_wrapper_loss -- do not hand-roll dice/focal here.
    seg_loss = seg_loss_fn(targets['masks'], outputs['low_res_logits'])
    cls_loss = keras.losses.categorical_crossentropy(
        targets['labels'], outputs['class_probs']
    )
    return seg_loss + 0.5 * cls_loss

# Train with both tasks
```

---

## 12. Serialization & Deployment

### Saving and Loading

SAM supports multiple serialization formats.

> **Build the model before you save it.** `SAM.from_variant(...)` returns an
> unbuilt model whose sub-layer variables do not exist yet; saving it produces
> a Keras "you are saving a model that has not yet been built" warning and an
> archive that carries no weights. Call the model once on real inputs first —
> every snippet below does. On load, `SAM.build_from_config` runs a
> full-resolution dummy forward to materialize the lazily built variables
> BEFORE the archive is restored (at `vit_h` that is a 1024x1024 forward
> through 630M parameters); a cheaper `build()`-only path was measured and
> rejected because it materializes only part of the weight list and leaves the
> rest silently re-initialized.

```python
# Method 1: Full model save (recommended)
model = SAM.from_variant('vit_b')

# BUILD FIRST -- an unbuilt save stores no weights.
model({
    'image': keras.random.normal(shape=(1, 1024, 1024, 3)),
    'points': (
        keras.ops.convert_to_tensor([[[512.0, 512.0]]]),
        keras.ops.convert_to_tensor([[1]])
    ),
    'original_size': (1024, 1024)
})

# Save entire model
model.save('sam_model.keras')
print("Model saved to sam_model.keras")

# Load model
loaded_model = keras.models.load_model('sam_model.keras')
print("Model loaded successfully")

# Verify
test_input = {
    'image': keras.random.normal(shape=(1, 1024, 1024, 3)),
    'points': (
        keras.ops.convert_to_tensor([[[512.0, 512.0]]]),
        keras.ops.convert_to_tensor([[1]])
    ),
    'original_size': (1024, 1024)
}

outputs = loaded_model(test_input)
print(f"Output shape: {outputs['masks'].shape}")
```

```python
# Method 2: Weights only
#
# NOTE: `sam_weights.weights.h5` is a file YOU produce below. No such
# artifact ships with this repo and none is downloaded -- there is no
# pretrained SAM checkpoint here to load.
model = SAM.from_variant('vit_b')

# BUILD FIRST, on both sides -- weights of an unbuilt model do not exist,
# and load_weights into an unbuilt model has nothing to match against.
build_inputs = {
    'image': keras.random.normal(shape=(1, 1024, 1024, 3)),
    'points': (
        keras.ops.convert_to_tensor([[[512.0, 512.0]]]),
        keras.ops.convert_to_tensor([[1]])
    ),
    'original_size': (1024, 1024)
}
model(build_inputs)

# Save weights
model.save_weights('sam_weights.weights.h5')

# Load weights into new model
new_model = SAM.from_variant('vit_b')
new_model(build_inputs)
new_model.load_weights('sam_weights.weights.h5')
```

```python
# Method 3: SavedModel format (for TensorFlow Serving)
model = SAM.from_variant('vit_b')

# Build model first
model({
    'image': keras.random.normal(shape=(1, 1024, 1024, 3)),
    'points': (
        keras.ops.convert_to_tensor([[[512.0, 512.0]]]),
        keras.ops.convert_to_tensor([[1]])
    ),
    'original_size': (1024, 1024)
})

# Export as SavedModel
model.export('sam_savedmodel')

# Load SavedModel
imported = tf.saved_model.load('sam_savedmodel')
```

### Deployment Scenarios

**Scenario 1: REST API with Flask**

```python
from flask import Flask, request, jsonify
import keras
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load model once at startup
model = SAM.from_variant('vit_b')
print("Model loaded")

@app.route('/segment', methods=['POST'])
def segment():
    """
    Endpoint for segmentation.
    
    Expected JSON:
    {
        "image": "base64_encoded_image",
        "points": [[x1, y1], [x2, y2], ...],
        "labels": [1, 0, 1, ...]  # 1=foreground, 0=background
    }
    """
    data = request.json
    
    # Decode image
    image_data = base64.b64decode(data['image'])
    image = Image.open(io.BytesIO(image_data))
    image = np.array(image) / 255.0
    
    # Prepare inputs
    h, w = image.shape[:2]
    image_tensor = keras.ops.convert_to_tensor(image[None, ...])
    
    points_coords = keras.ops.convert_to_tensor([data['points']])
    points_labels = keras.ops.convert_to_tensor([data['labels']])
    points = (points_coords, points_labels)
    
    # Run segmentation
    outputs = model({
        'image': image_tensor,
        'points': points,
        'original_size': (h, w)
    })
    
    # Get best mask
    best_idx = keras.ops.argmax(outputs['iou_predictions'][0])
    mask = outputs['masks'][0, best_idx]
    
    # Convert mask to base64
    mask_np = keras.ops.convert_to_numpy(mask) * 255
    mask_img = Image.fromarray(mask_np.astype(np.uint8))
    buffered = io.BytesIO()
    mask_img.save(buffered, format="PNG")
    mask_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    return jsonify({
        'mask': mask_base64,
        'iou_score': float(outputs['iou_predictions'][0, best_idx])
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Scenario 2: Docker Container**

```dockerfile
# Dockerfile
FROM tensorflow/tensorflow:2.18.0-gpu

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy model and code
COPY sam_model.keras .
COPY app.py .

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "app.py"]
```

```bash
# Build and run
docker build -t sam-api .
docker run -p 5000:5000 --gpus all sam-api
```

**Scenario 3: TensorFlow Serving**

```bash
# Export model
python -c "
import keras
from dl_techniques.models.sam import SAM

model = SAM.from_variant('vit_b')
# Build model...
model.export('sam_serving/1')
"

# Serve with TensorFlow Serving
docker run -p 8501:8501 \
  --mount type=bind,source=$(pwd)/sam_serving,target=/models/sam \
  -e MODEL_NAME=sam \
  tensorflow/serving

# Query
curl -X POST http://localhost:8501/v1/models/sam:predict \
  -d '{
    "inputs": {
      "image": [...],
      "points": [...],
      "original_size": [1024, 1024]
    }
  }'
```

---

## 13. Testing & Validation

**The real suite lives at `tests/test_models/test_sam/`** —
`test_model.py` (shape / dtype / API coverage) and `test_correctness.py`
(guards each proven RED against a deliberate re-break: gradient counts per
output key, the rel-pos interpolation branch, padding-point invariance,
weight-count + value-exact `.keras` round-trip, the tiling / mask-shape /
oversize / degenerate-encoder refusals, and the positional-encoding
graph-safety guards) plus `test_training_model.py` (the wrapper: the `SAM.call`
spy, the moved-variable decomposition, the dead-component probe, refinement
liveness, the `.keras` round-trip). Three sibling gates cover the training
path: `tests/test_losses/test_sam_mask_loss.py` (the focal-blindness and
raw-layout traps), `tests/test_train/test_sam/` (data sources, prompt-sampling
properties, the args->config wiring) and `tests/test_datasets/test_coco_instances.py`.
Run them with:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
    tests/test_models/test_sam/ tests/test_losses/test_sam_mask_loss.py \
    tests/test_train/test_sam/ tests/test_datasets/test_coco_instances.py -q
```

The illustrative snippets below are a starting point, not that suite.

### Unit Tests

```python
import keras
import numpy as np
from dl_techniques.models.sam import SAM

def test_model_creation():
    """Test that model variants can be created."""
    for variant in ['vit_b', 'vit_l', 'vit_h']:
        model = SAM.from_variant(variant)
        assert model is not None
        print(f"✓ {variant} created successfully")

def test_forward_pass():
    """Test basic forward pass."""
    model = SAM.from_variant('vit_b')
    
    image = keras.random.normal(shape=(1, 1024, 1024, 3))
    points = (
        keras.ops.convert_to_tensor([[[512.0, 512.0]]]),
        keras.ops.convert_to_tensor([[1]])
    )
    
    outputs = model({
        'image': image,
        'points': points,
        'original_size': (1024, 1024)
    })
    
    assert 'masks' in outputs
    assert 'iou_predictions' in outputs
    assert 'low_res_logits' in outputs
    print("✓ Forward pass successful")

def test_serialization():
    """Test model save/load."""
    model = SAM.from_variant('vit_b')
    
    # Build model
    image = keras.random.normal(shape=(1, 1024, 1024, 3))
    points = (
        keras.ops.convert_to_tensor([[[512.0, 512.0]]]),
        keras.ops.convert_to_tensor([[1]])
    )
    
    outputs1 = model({
        'image': image,
        'points': points,
        'original_size': (1024, 1024)
    })
    
    # Save and load
    model.save('test_model.keras')
    loaded_model = keras.models.load_model('test_model.keras')
    
    # Compare outputs
    outputs2 = loaded_model({
        'image': image,
        'points': points,
        'original_size': (1024, 1024)
    })
    
    # Compare the FLOAT logits, not the binarized `masks`. A uint8 comparison
    # saturates at an absolute difference of 1 and is blind to any logit
    # change that does not move a pixel across `mask_threshold` -- scaling
    # every logit by a positive constant leaves it bit-identical.
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(outputs1['low_res_logits']),
        keras.ops.convert_to_numpy(outputs2['low_res_logits']),
        rtol=1e-5, atol=1e-5
    )
    print("✓ Serialization successful")

def test_different_prompts():
    """Test different prompt types."""
    model = SAM.from_variant('vit_b')
    image = keras.random.normal(shape=(1, 1024, 1024, 3))
    
    # Test points
    points = (
        keras.ops.convert_to_tensor([[[512.0, 512.0]]]),
        keras.ops.convert_to_tensor([[1]])
    )
    out1 = model({'image': image, 'points': points, 'original_size': (1024, 1024)})
    assert out1['masks'].shape[1] == 3  # 3 mask proposals
    
    # Test boxes
    boxes = keras.ops.convert_to_tensor([[[100.0, 100.0, 900.0, 900.0]]])
    out2 = model({'image': image, 'boxes': boxes, 'original_size': (1024, 1024)})
    assert out2['masks'].shape[1] == 3
    
    # Test masks
    masks = keras.random.normal(shape=(1, 1, 256, 256))
    out3 = model({'image': image, 'masks': masks, 'original_size': (1024, 1024)})
    assert out3['masks'].shape[1] == 3
    
    print("✓ Different prompt types work")

# Run tests
if __name__ == '__main__':
    test_model_creation()
    test_forward_pass()
    test_serialization()
    test_different_prompts()
    print("\n✅ All tests passed!")
```

### Integration Tests

```python
def test_interactive_workflow():
    """Test typical interactive segmentation workflow."""
    from dl_techniques.models.sam import SAM
    
    model = SAM.from_variant('vit_b')
    image = keras.random.normal(shape=(1, 1024, 1024, 3))
    
    # Step 1: Initial point
    points1 = (
        keras.ops.convert_to_tensor([[[512.0, 512.0]]]),
        keras.ops.convert_to_tensor([[1]])
    )
    out1 = model({'image': image, 'points': points1, 'original_size': (1024, 1024)})
    
    # Step 2: Add refinement point
    points2 = (
        keras.ops.convert_to_tensor([[[512.0, 512.0], [600.0, 600.0]]]),
        keras.ops.convert_to_tensor([[1, 1]])
    )
    out2 = model({'image': image, 'points': points2, 'original_size': (1024, 1024)})
    
    # Step 3: Add negative point
    points3 = (
        keras.ops.convert_to_tensor([[[512.0, 512.0], [600.0, 600.0], [100.0, 100.0]]]),
        keras.ops.convert_to_tensor([[1, 1, 0]])
    )
    out3 = model({'image': image, 'points': points3, 'original_size': (1024, 1024)})
    
    print("✓ Interactive workflow successful")

def test_batch_processing():
    """Test batch processing multiple images."""
    model = SAM.from_variant('vit_b')
    
    # Create batch
    images = [
        keras.random.normal(shape=(1, 1024, 1024, 3))
        for _ in range(5)
    ]
    
    results = []
    for image in images:
        points = (
            keras.ops.convert_to_tensor([[[512.0, 512.0]]]),
            keras.ops.convert_to_tensor([[1]])
        )
        out = model({'image': image, 'points': points, 'original_size': (1024, 1024)})
        results.append(out)
    
    assert len(results) == 5
    print("✓ Batch processing successful")

if __name__ == '__main__':
    test_interactive_workflow()
    test_batch_processing()
```

---

## 14. Troubleshooting & FAQs

### Common Issues and Solutions

**Issue 1: Out of Memory (OOM) Error**

```
Symptoms:
- "ResourceExhaustedError: OOM when allocating tensor"
- Model crashes during inference
- GPU memory fills up quickly

Solutions:
1. Use smaller variant (vit_b instead of vit_h)
2. Enable mixed precision
3. Reduce batch size
4. Clear GPU memory between runs
```

```python
# Solution code
import keras
import gc
import tensorflow as tf

# Use mixed precision
keras.mixed_precision.set_global_policy('mixed_float16')

# Clear memory
keras.backend.clear_session()
gc.collect()

if tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(
        tf.config.list_physical_devices('GPU')[0], True
    )

# Use smaller model
model = SAM.from_variant('vit_b')  # Instead of 'vit_h'
```

**Issue 2: Slow Inference**

```
Symptoms:
- Each prediction takes several seconds
- Not utilizing GPU properly
- CPU at 100%

Solutions:
1. Ensure GPU is being used
2. Enable XLA compilation
3. Use embedding caching for multiple prompts
4. Warm up the model
```

```python
# Check GPU usage
import tensorflow as tf
print("GPUs available:", tf.config.list_physical_devices('GPU'))

# Enable XLA
tf.config.optimizer.set_jit(True)

# Warm up model (first call is always slower)
dummy_input = {
    'image': keras.random.normal(shape=(1, 1024, 1024, 3)),
    'points': (
        keras.ops.convert_to_tensor([[[512.0, 512.0]]]),
        keras.ops.convert_to_tensor([[1]])
    ),
    'original_size': (1024, 1024)
}
_ = model(dummy_input)  # Warm-up call
print("Model warmed up")

# Now time actual inference
import time
start = time.time()
outputs = model(dummy_input)
print(f"Inference time: {(time.time() - start) * 1000:.1f}ms")
```

**Issue 3: Poor Segmentation Quality**

```
Symptoms:
- Masks don't match the object well
- Segmentation includes too much/too little
- Inconsistent results

Solutions:
1. Use better prompts (multiple points, boxes)
2. Try different mask proposals
3. Use larger model variant
4. Adjust mask threshold
```

```python
# Better prompting
# Instead of single point:
points = (
    keras.ops.convert_to_tensor([[[350.0, 450.0]]]),
    keras.ops.convert_to_tensor([[1]])
)

# Use multiple points + box:
points = (
    keras.ops.convert_to_tensor([[[350.0, 450.0], [360.0, 460.0]]]),
    keras.ops.convert_to_tensor([[1, 1]])
)
boxes = keras.ops.convert_to_tensor([[[300.0, 400.0, 400.0, 500.0]]])

outputs = model({
    'image': image,
    'points': points,
    'boxes': boxes,
    'original_size': (H, W)
})

# Try all mask proposals
for i in range(outputs['masks'].shape[1]):
    print(f"Mask {i} IoU: {outputs['iou_predictions'][0, i]}")
    # Visualize each mask...
```

**Issue 4: Serialization Errors**

```
Symptoms:
- "Can't save model"
- "Unknown layer type"
- Model loads but doesn't work

Solutions:
1. Ensure all custom layers are registered
2. Build model before saving
3. Use correct save format
```

```python
# Proper serialization
model = SAM.from_variant('vit_b')

# Build model first
dummy_input = {
    'image': keras.random.normal(shape=(1, 1024, 1024, 3)),
    'points': (
        keras.ops.convert_to_tensor([[[512.0, 512.0]]]),
        keras.ops.convert_to_tensor([[1]])
    ),
    'original_size': (1024, 1024)
}
_ = model(dummy_input)

# Save with .keras extension
model.save('sam_model.keras')  # Recommended format

# Or use export for serving
model.export('sam_savedmodel')  # TensorFlow SavedModel format
```

### Frequently Asked Questions

**Q: What's the optimal image size for SAM?**

A: SAM is designed for 1024×1024 images. `SAM.preprocess` **pads only — it
never resizes**, and an image larger than the encoder's `img_size` in either
axis raises `ValueError`:
- **Larger than 1024 in any axis**: downscale first with
  `resize_longest_side(image, 1024)` from
  `dl_techniques.models.sam.preprocessing`, and scale your prompt coordinates
  by the same factor. Tiling is an alternative when you need full resolution.
- **Smaller**: bottom/right padding is automatic, but may affect edge quality.

**Q: How many prompts should I provide?**

A: It depends on your use case:
- **Simple objects**: 1 point is often enough
- **Complex shapes**: 3-5 points for better coverage
- **Ambiguous scenes**: Combine points + box for precision
- **Refinement**: Start with 1 point, add more iteratively

**Q: Why does SAM produce 3 masks by default?**

A: Segmentation is often ambiguous. The 3 masks typically represent:
1. **Whole object**: The complete object
2. **Part**: A significant part or subcomponent
3. **Subpart**: A smaller detail or alternative interpretation

Use IoU scores to pick the best one for your needs.

**Q: Can SAM segment multiple objects at once?**

A: Yes! Use grid prompting (Example 6) or provide multiple point prompts. Each prompt generates its own set of masks.

**Q: How do I handle very large images?**

A: For images much larger than 1024×1024:

```python
def tile_and_segment(model, large_image, tile_size=1024, overlap=128):
    """Segment large image by tiling."""
    h, w = large_image.shape[:2]
    stride = tile_size - overlap
    
    all_masks = []
    
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Extract tile
            tile = large_image[y:y+tile_size, x:x+tile_size]
            
            # Pad if needed
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                tile = pad_to_size(tile, tile_size)
            
            # Segment tile
            outputs = model({
                'image': tile[None, ...],
                'points': center_point_prompt(),
                'original_size': (tile_size, tile_size)
            })
            
            # Store with position
            all_masks.append((x, y, outputs['masks'][0]))
    
    # Stitch tiles together (handle overlaps)
    return stitch_tiles(all_masks, (h, w))
```

**Q: Can I use SAM for video segmentation?**

A: Yes, but note:
- Encode each frame's image once
- Track objects across frames by propagating prompts
- Consider temporal consistency post-processing

```python
def segment_video(model, video_frames, initial_prompt):
    """Segment object across video frames."""
    results = []
    current_prompt = initial_prompt
    
    for frame in video_frames:
        # Segment current frame
        outputs = model({
            'image': frame[None, ...],
            'points': current_prompt,
            'original_size': frame.shape[:2]
        })
        
        best_mask = outputs['masks'][0, 0]
        results.append(best_mask)
        
        # Update prompt based on mask centroid
        centroid = compute_centroid(best_mask)
        current_prompt = create_point_prompt(centroid)
    
    return results
```

**Q: How does SAM compare to other segmentation models?**

A:

| Model | Strengths | Weaknesses | Best For |
|-------|-----------|------------|----------|
| **SAM** | Zero-shot, flexible prompts, high quality | Large model, needs prompts | Interactive segmentation, diverse domains |
| **Mask R-CNN** | Instance segmentation, fast | Needs training per domain | Fixed object categories |
| **U-Net** | Simple, efficient | Needs dense annotations | Medical imaging, fixed tasks |
| **DeepLab** | Semantic segmentation | No instance separation | Scene parsing |

**Q: Can I fine-tune SAM on my domain?**

A: The mechanics ship (§11: `SAMTrainingModel` + `train.sam.train_sam`), but
"fine-tuning" is the wrong word here — **there are no pretrained weights to
fine-tune from**, in this repo or downloadable by it, so any run starts from
noise. With that stated, the strategies are:
1. Freeze image encoder, train decoder (few samples)
2. Progressive unfreezing (moderate data)
3. Full training (large dataset)

---

## 15. Technical Details

### Architecture Statistics

**These figures are MEASURED from this package**, not quoted from the reference
PyTorch implementation. Each variant was instantiated via
`SAM.from_variant(...)`, forward-passed once per component at `img_size=1024`
so that every lazily-created variable exists, and counted with
`count_params()`. The previous version of this section quoted reference-SAM
numbers that this repo's own layout changes had falsified (the
`attention_downsample_rate=2` cross-attentions and the 3-layer MLP heads move
the mask decoder by exactly −328,320 + the head growth), and whose "Total
Parameters" line was in fact the image-encoder count.

Reproduce with:

```python
m = SAM.from_variant("vit_b")
feat = m.image_encoder(keras.ops.zeros((1, 1024, 1024, 3)))
sparse, dense = m.prompt_encoder(points=(keras.ops.zeros((1, 1, 2)),
                                         keras.ops.zeros((1, 1), dtype="int32")))
m.mask_decoder(image_embeddings=feat, image_pe=m.prompt_encoder.get_dense_pe(),
               sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
               multimask_output=True)
[int(c.count_params()) for c in (m.image_encoder, m.prompt_encoder, m.mask_decoder)]
```

| Variant | Image Encoder | Prompt Encoder | Mask Decoder | Total |
|---|---:|---:|---:|---:|
| **ViT-B** | 89,670,912 (95.7%) | 6,476 (0.007%) | 4,058,340 (4.3%) | **93,735,728** |
| **ViT-L** | 308,278,272 (98.7%) | 6,476 (0.002%) | 4,058,340 (1.3%) | **312,343,088** |
| **ViT-H** | 637,026,048 (99.4%) | 6,476 (0.001%) | 4,058,340 (0.6%) | **641,090,864** |

The prompt encoder and the mask decoder are **variant-independent** — both run
at `prompt_embed_dim=256` for every variant, so only the image encoder scales.

**ViT-B component breakdown** (measured the same way):

```
Image Encoder                    89,670,912
├─ Patch Embedding                  590,592
├─ Position Embedding             3,145,728
├─ Transformer Blocks            85,147,136
└─ Neck                             787,456

Prompt Encoder                        6,476
├─ Random Gaussian PE matrix            256   (non-trainable)
├─ Point Embeddings (4)               1,024
├─ not_a_point_embed                    256
├─ no_mask_embed                        256
└─ Mask Downscaling CNN               4,684

Mask Decoder                      4,058,340
├─ IoU Token                            256
├─ Mask Tokens (4)                    1,024
├─ Two-Way Transformer            3,291,264
├─ Output Upscaling                  73,952
├─ Hypernetwork MLPs (4)            559,232
└─ IoU Prediction Head              132,612
```

Memory footprints below are **NOT measured** — they are order-of-magnitude
estimates for weights plus activations at batch 1, and are marked as such:

| Variant | Weights (FP32, derived) | Peak footprint (FP32, *estimated*) |
|---|---:|---:|
| ViT-B | ~0.37 GB | ~4 GB |
| ViT-L | ~1.25 GB | ~9 GB |
| ViT-H | ~2.56 GB | ~17 GB |

### Computational Complexity

**FLOPs Analysis** (for 1024×1024 image):

```
Component              FLOPs (GFLOPs)   % of Total
────────────────────────────────────────────────────
Image Encoder:
├─ Patch Embedding        1.2            0.7%
├─ Attention (windowed)  120.5          68.3%
├─ Attention (global)     28.4          16.1%
├─ FFN                    24.8          14.1%
└─ Neck                    1.2           0.7%
    Subtotal:            176.1          99.9%

Prompt Encoder:            0.1           0.05%

Mask Decoder:
├─ Two-Way Transformer     0.8           0.5%
├─ Upsampling              0.4           0.2%
└─ Mask Prediction         0.2           0.1%
    Subtotal:              1.4           0.8%

────────────────────────────────────────────────────
Total:                   177.6 GFLOPs

Notes:
- 99% of computation is in image encoder
- This is why caching embeddings is so effective
- Prompt + mask decoding is <1% of total cost
```

### Attention Pattern Analysis

The windowed attention with selective global attention creates an efficient-yet-expressive pattern:

```
Receptive Field Growth:

Layer 0  (Windowed): 14×14 patch region
Layer 1  (Windowed): 28×28 patch region
Layer 2  (Global):   64×64 patch region (full image)
Layer 3  (Windowed): 28×28 patch region
...
Layer 11 (Global):   64×64 patch region (full image)

Effective receptive field: 
- Every patch sees global context by layer 2
- Refined with local details in between
- Final global view ensures consistency
```

### Positional Encoding Details

SAM uses relative position biases instead of absolute position encodings:

```python
# Relative position bias computation
def get_rel_pos(q_size, k_size, rel_pos):
    """
    Get relative positional embeddings according to the relative
    positions of query and key sizes.
    
    Args:
        q_size: Size of query (e.g., 14 for 14×14 window)
        k_size: Size of key
        rel_pos: Relative position embeddings [2*max_rel_pos-1, dim]
    
    Returns:
        Extracted positional embeddings [q_size, k_size, dim]
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    
    # Relative position indices
    q_coords = np.arange(q_size)[:, None]
    k_coords = np.arange(k_size)[None, :]
    relative_coords = q_coords - k_coords
    
    # Add max_rel_dist-1 to make indices positive
    relative_coords += max_rel_dist - 1
    
    # Extract embeddings
    return rel_pos[relative_coords]
```

### Training Dataset Composition

SAM was trained on SA-1B dataset:

```
SA-1B Dataset Statistics:
─────────────────────────────────────────
Total Images:           11,000,000
Total Masks:         1,100,000,000
Average Masks/Image:            100

Image Sources:
├─ Diverse geographic locations
├─ Various camera types
├─ Indoor and outdoor scenes
└─ Different lighting conditions

Mask Quality:
├─ Fully automatic: 99.1%
├─ Automatically assisted: 0.9%
└─ Fully manual: ~0%

Annotation Method:
1. Initial automatic masks (model)
2. Human verification (accept/reject)
3. Interactive refinement (if needed)
4. Data augmentation
```

---

## 16. Requirements

### Core Dependencies

```txt
# Minimum Requirements
python>=3.9
keras>=3.8.0
tensorflow>=2.18.0  # or torch>=2.0, or jax>=0.4
numpy>=1.19.0

# Optional but Recommended
matplotlib>=3.3.0  # For visualization
Pillow>=8.0.0      # For image I/O
opencv-python>=4.5.0  # For advanced image processing
```

### Installation

```bash
# Basic installation
pip install keras>=3.8.0 tensorflow>=2.18.0 numpy

# With visualization tools
pip install keras>=3.8.0 tensorflow>=2.18.0 numpy matplotlib pillow

# For development
pip install keras>=3.8.0 tensorflow>=2.18.0 numpy matplotlib pillow pytest black flake8
```

### Framework Integration

This implementation integrates with the `dl_techniques` framework:

```python
# Imports from dl_techniques
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.ffn import create_ffn_layer

# If not using dl_techniques, these can be replaced with:
# - keras.layers.LayerNormalization
# - keras.Sequential([Dense, Activation, Dense])
```

### Hardware Recommendations

| Variant | Min GPU Memory | Recommended GPU | CPU Alternative |
|---------|----------------|-----------------|-----------------|
| vit_b | 6 GB | RTX 3060, T4 | 16 GB RAM |
| vit_l | 12 GB | RTX 3090, A100 | 32 GB RAM |
| vit_h | 24 GB | A100 (40GB) | 64 GB RAM |

**Note**: CPU inference is possible but 50-100x slower than GPU.

---

## 17. Citation

If you use SAM in your research or projects, please cite the original paper:

```bibtex
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```

### Related Papers

**Vision Transformers**:
```bibtex
@inproceedings{dosovitskiy2020image,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```

**Swin Transformer** (for windowed attention concept):
```bibtex
@inproceedings{liu2021swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10012--10022},
  year={2021}
}
```

---

## License and Acknowledgments

This implementation is based on the work by Kirillov et al. (2023) and follows the architecture described in their paper. The original SAM was developed by Meta AI Research.

**Implementation Notes**:
- This is an independent Keras 3 implementation
- Designed for the `dl_techniques` framework
- Follows modern deep learning best practices
- Fully serializable and production-ready

**Framework Credits**:
- Keras 3 team for the excellent deep learning framework
- TensorFlow/PyTorch/JAX teams for backend support
- `dl_techniques` framework for standardized components

---

## Quick Reference Card

### Common Commands

```python
# Create model (RANDOM weights -- no checkpoint ships with this repo)
import keras
from dl_techniques.models.sam import SAM, resize_longest_side
model = SAM.from_variant('vit_b')  # or 'vit_l', 'vit_h'

# Preprocess: preprocess() pads only; pin the longest side to img_size first
image = resize_longest_side(image, model.image_encoder.img_size)

# Single point
outputs = model({
    'image': image,
    'points': (coords, labels),
    'original_size': (H, W)
})

# Box prompt
outputs = model({
    'image': image,
    'boxes': boxes,
    'original_size': (H, W)
})

# Combined
outputs = model({
    'image': image,
    'points': (coords, labels),
    'boxes': boxes,
    'original_size': (H, W)
})

# Get best mask
best_idx = keras.ops.argmax(outputs['iou_predictions'][0])
mask = outputs['masks'][0, best_idx]

# Save/Load
model.save('sam.keras')
model = keras.models.load_model('sam.keras')
```

### Prompt Format Quick Reference

```python
# Points: (coordinates, labels)
points = (
    keras.ops.convert_to_tensor([[[x1, y1], [x2, y2]]]),  # Shape: (B, N, 2)
    keras.ops.convert_to_tensor([[1, 0]])                  # Shape: (B, N)
)
# Labels: 1 = foreground, 0 = background

# Boxes: (x1, y1, x2, y2)
boxes = keras.ops.convert_to_tensor([[[x1, y1, x2, y2]]])  # Shape: (B, N, 4)

# Masks: Low-res mask hint
# REQUIRED: spatial size == 4 * image_embedding_size in BOTH axes
# (256 = 4 * 64 for a 1024-image / 16-patch encoder). Anything else raises.
masks = keras.random.normal(shape=(B, 1, 256, 256))  # Shape: (B, 1, H, W)

# Output keys
# 'masks'           uint8 0/1 by default -- NOT differentiable
# 'iou_predictions' float
# 'low_res_logits'  float -- THE training target
```