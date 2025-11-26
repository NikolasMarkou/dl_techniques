# Extracting Neurosymbols with Sparse RBF Networks (v2.0)
## A Modernized Guide Integrating 2024-2025 SOTA

---

**Document Version**: 2.0 (SOTA Update)  
**Date**: January 2025  
**Based On**: Anthropic SAEs, FSQ, and Object-Centric Learning breakthroughs  
**Status**: Production-Ready Framework

---

## Table of Contents
1. [Executive Summary: The 2025 Shift](#executive-summary-the-2025-shift)
2. [RBFs in the Age of Sparse Autoencoders](#rbfs-in-the-age-of-sparse-autoencoders)
3. [Architecture: The Sparse-Quantized RBF](#architecture-the-sparse-quantized-rbf)
4. [Method 1: The SAE-RBF Hybrid (Monosemantic Features)](#method-1-the-sae-rbf-hybrid-monosemantic-features)
5. [Method 2: Collapse-Free Symbol Discovery (SimVQ/FSQ)](#method-2-collapse-free-symbol-discovery-simvqfsq)
6. [Method 3: Object-Centric Grounding (DINOSAUR Integration)](#method-3-object-centric-grounding-dinosaur-integration)
7. [From Activations to Logic: FOLD-SE-M](#from-activations-to-logic-fold-se-m)
8. [Updated Application Strategies](#updated-application-strategies)
9. [Comparison: RBF vs. Standard SAE](#comparison-rbf-vs-standard-sae)
10. [Common Pitfalls & SOTA Solutions](#common-pitfalls--sota-solutions)

---

## Executive Summary: The 2025 Shift

**The Core Update**: Traditional Radial Basis Function (RBF) networks were historically limited by "center collapse" and input scaling issues. However, 2024-2025 breakthroughs in **Sparse Autoencoders (SAEs)** and **Vector Quantization (VQ)** have revitalized this architecture.

By viewing RBF units not just as clusters, but as **distance-based monosemantic features**, we can now scale symbol extraction to millions of concepts (matching OpenAI/Anthropic scales) while retaining the geometric interpretability that RBFs provide.

**Key SOTA Integrations**:
- **Scaling Monosemanticity**: Adopting the *Top-K* sparsity constraints from OpenAI/Anthropic to force RBF centers to represent single, interpretable concepts.
- **Solving Collapse**: Replacing standard k-means initialization with **FSQ (Finite Scalar Quantization)** and **SimVQ** geometry to guarantee 100% codebook utilization.
- **Real-World Vision**: Moving from raw pixels to reconstructing **DINOv2/DINOv3** features, enabling object-centric symbol discovery on complex scenes (COCO, YouTube-VIS).
- **Logic Extraction**: Utilizing **FOLD-SE-M** to translate RBF activations directly into executable Prolog programs.

---

## RBFs in the Age of Sparse Autoencoders

### The Conceptual Bridge

In 2024, Anthropic and OpenAI demonstrated that high-dimensional sparse features (SAEs) can extract "monosemantic" concepts (e.g., a "Golden Gate Bridge" neuron).

**RBFs are Geometrically Bounded SAEs.**
While standard SAEs use dot products (unbounded activation), RBFs use Euclidean distance (bounded activation). This makes RBFs superior for **Out-of-Distribution (OOD) detection** and defining strict symbol boundaries, provided they are trained with modern sparsity constraints.

```
┌────────────────────────────────────────────────────────────────────────┐
│                  MODERN NEUROSYMBOLIC DEFINITION                       │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Old RBF View:      Clustering centroids in input space.               │
│                                                                        │
│  2025 SOTA View:    A Sparse Dictionary of Monosemantic Prototypes     │
│                     governed by Top-K activation dynamics.             │
│                                                                        │
│  φᵢ(x) = TopK_activation( exp(-γ || Enc(x) - cᵢ ||²) )                 │
│                                                                        │
│  Key Properties:                                                       │
│  1. High Expansion:  Hidden dim >> Input dim (e.g., 16x expansion)     │
│  2. k-Sparsity:      Only the k closest centers activate (k ≈ 10-50)   │
│  3. Feature Inputs:  x is never raw pixels; x is a DINO/CLIP embedding │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Architecture: The Sparse-Quantized RBF

This architecture replaces the traditional "Layer 1" with a pre-trained foundation model and adds SOTA quantization stability.

```
┌────────────────────────────────────────────────────────────────────┐
│               THE SOTA RBF-SYMBOL ARCHITECTURE                     │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  1. Foundation Encoder (Frozen)                                    │
│     [Image/Text] ──▶ [DINOv3 / BERT] ──▶ Embedding z (768-dim)     │
│                                                                    │
│  2. SimVQ / FSQ Projection (The Stability Fix)                     │
│     Projects z into a space where "collapse" is impossible.        │
│     z' = W_proj @ z                                                │
│                                                                    │
│  3. Sparse RBF Layer (The Symbol Extractor)                        │
│     • Expansion: 768 ──▶ 16,384 units (Centers cᵢ)                 │
│     • Distance: || z' - cᵢ ||²                                     │
│     • Activation: φ = Softmax( -Distance / τ )                     │
│     • Sparsity: Keep only Top-K values (e.g., k=5)                 │
│                                                                    │
│  4. Symbolic Output                                                │
│     Indices of Top-K units ──▶ {Symbol_402, Symbol_11}             │
│     Activations ──▶ Confidence / Fuzzy Truth Value                 │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Method 1: The SAE-RBF Hybrid (Monosemantic Features)

**Goal**: Extract millions of interpretable features (symbols) similar to Anthropic's Claude 3 Sonnet analysis, but using RBF mechanics for better stability.

### The Algorithm
1.  **Input**: Activations from a frozen layer of a large model (e.g., Layer 12 of Llama-3 or ViT).
2.  **Expansion**: Initialize $N$ RBF centers, where $N \approx 30 \times \text{Input\_Dim}$.
3.  **Top-K Training (OpenAI Method)**:
    *   Forward pass: Calculate distances to all centers.
    *   Keep only the $k$ smallest distances (highest activations). Set others to zero.
    *   Backward pass: Update only the centers corresponding to the Top-K.
    *   *Loss*: `Reconstruction_Loss + Auxiliary_L2_Loss` (to keep centers on the data manifold).
4.  **Dead Center Revival**: If a center hasn't activated in $T$ steps, reset it to a random input example with high reconstruction error (Re-initialization trick).

**Why this is SOTA**: This specific training regime (Top-K + Re-init) was proven in June 2024 to yield cleaner, more interpretable features than L1 regularization alone.

---

## Method 2: Collapse-Free Symbol Discovery (SimVQ/FSQ)

**The Problem**: Standard RBF training often leads to "center collapse" (many centers converging to the same spot) or "index collapse" (only a few symbols act as distinct codes).

**The 2024 Solution**: Integrate **Finite Scalar Quantization (FSQ)** or **SimVQ** mechanics.

### Approach A: FSQ-RBF (Implicit Centers)
Instead of learning moving centers $c_i$, we project the input into a lower-dimensional space where centers are fixed integer coordinates.
*   **Concept**: Symbols are not learned vectors; they are regions in a hypercube.
*   **Mechanism**:
    *   Project input $z$ to dimension $d$ (e.g., $d=5$).
    *   Apply `Round()` to get integer coordinates.
    *   The "Symbol ID" is the mapped integer index.
*   **Benefit**: **Zero** codebook collapse. 100% codebook utilization by design.

### Approach B: SimVQ-RBF (High Fidelity)
For applications requiring high precision (reconstruction):
*   Maintain a standard RBF layer.
*   **Update Rule**: Instead of gradient descent on centers, update centers using the **Rotation Trick** or linear reparameterization of the latent basis.
*   **Result**: Matches continuous VAE quality (rFID 0.49) while providing discrete symbols.

---

## Method 3: Object-Centric Grounding (DINOSAUR Integration)

**Refining the "Vision" Strategy**:
The 2023/2024 papers **DINOSAUR** and **VideoSAUR** proved that RBF-style "Slot Attention" fails on raw pixels but excels on self-supervised features.

### The Pipeline
1.  **Input Image** $\rightarrow$ **DINOv2 / DINOv3 ViT**.
2.  **Extract Patches**: Get the $16 \times 16$ feature grid.
3.  **RBF Slot Attention**:
    *   Initialize $K$ "Object Symbols" (slots).
    *   Iteratively refine symbols via Softmax attention over the feature grid (effectively a dynamic RBF).
4.  **Reconstruction**: The symbols must reconstruct the **DINO features**, not the RGB pixels.

**Result**: The RBF centers (slots) naturally bind to objects ("Car", "Road", "Person") even in complex real-world scenes (COCO), achieving ~40-68 FG-ARI (SOTA).

---

## From Activations to Logic: FOLD-SE-M

**The Gap**: How do we turn RBF activations (0.9, 0.1, 0.0) into Python/Prolog rules?
**The SOTA Solution**: Use **FOLD-SE-M** (Padalkar et al., 2025), specifically designed for this translation.

### Step-by-Step Extraction
1.  **Binarization**: Apply dynamic thresholding to RBF outputs to get binary attributes.
    *   $Symbol_{24}(x) = True$ if $\phi_{24}(x) > \tau$.
2.  **Algorithm**: Run FOLD-SE-M on the (Input, Symbol\_State) pairs.
    *   It effectively runs an optimized Inductive Logic Programming (ILP) routine.
3.  **Output**: Generates explanatory rules without "negation as failure" issues.

**Example Output Rule**:
```prolog
is_dangerous_object(X) :- 
    rbf_symbol_red(X), 
    rbf_symbol_cylindrical(X), 
    rbf_symbol_texture_metallic(X).
% Confidence: 98.2% derived from RBF activation strength
```

---

## Updated Application Strategies

### 1. Computer Vision (The "DINOSAUR" Approach)
*   **Old Way**: CNN $\rightarrow$ RBF.
*   **SOTA Way**: DINOv2 $\rightarrow$ **Slot Attention (Dynamic RBF)**.
*   **Benefit**: Works on real-world video (YouTube-VIS) without masks. Discovers objects like "basketball", "player", "hoop" automatically.

### 2. LLM Interpretability & Steering
*   **Architecture**: Freeze LLM $\rightarrow$ Train Sparse RBF (SAE) on residual stream.
*   **Action**: "Clamp" specific RBF centers to high values during inference.
*   **Effect**: Steer the model (e.g., force the "Safety" symbol to active) or detect anomalies (e.g., "Deception" symbol activates).

### 3. Scientific Discovery (PhysORD)
*   **Update**: Use RBFs as the "Symbolic Distillation" layer.
*   **Method**: Train a massive neural network, then distil it into a small RBF network. Apply symbolic regression (PySR) *only* to the centers of the RBF network.
*   **Result**: 96.9% parameter reduction while retaining physical law accuracy.

---

## Comparison: RBF vs. Standard SAE

Why choose RBFs (Distance-based) over the popular SAEs (Dot-product based) from the 2024 reports?

| Feature | Standard SAE (Dot Product) | Neurosymbolic RBF (Euclidean) |
| :--- | :--- | :--- |
| **Activation Geometry** | Half-plane (unbounded) | Hypersphere (bounded) |
| **OOD Behavior** | Activates strongly on huge noise | **Zero activation** on noise (Safety) |
| **Interpretability** | "Direction" in space | "Prototype" in space |
| **Logic Integration** | Harder (requires thresholds) | Easier (natural fuzzy membership) |
| **SOTA Status** | Industry Standard (Anthropic) | Research Frontier (Safety/NeSy) |

**Recommendation**: Use **RBF-SAEs** for safety-critical or logic-heavy applications where OOD detection is required. Use standard SAEs for pure compression or generative steering.

---

## Common Pitfalls & SOTA Solutions

### 1. The "Dead Center" Problem
*   **Old Solution**: Soft repulsion.
*   **2025 Solution**: **Re-initialization**. Every 1000 steps, identify centers with <0.1% activation. Move them instantly to the location of the data point with the highest current reconstruction error.

### 2. Codebook Collapse
*   **Old Solution**: Entropy loss.
*   **2025 Solution**: **FSQ (Finite Scalar Quantization)**. Don't learn the centers; learn the projection into a fixed quantized grid. This mathematically prevents collapse.

### 3. High-Dimensional Distance Failure
*   **Old Solution**: Mahalanobis distance.
*   **2025 Solution**: **Spherical Projection**. Normalize all inputs and centers to the unit hypersphere ($||x||=1, ||c||=1$). Euclidean distance on a hypersphere is equivalent to Cosine similarity, which behaves much better in high dimensions (768+).

---

## Conclusion: The "Quantized-Feature" Future

The RBF network is no longer just a "clustering layer." By integrating **Top-K sparsity** from SAE research and **FSQ/SimVQ** stability from VQ research, the RBF layer becomes a robust **Neurosymbolic Interface**. It translates the messy, continuous manifold of modern Foundation Models (DINO, Llama) into discrete, reliable, and logically manipulatable symbols.

**Implementation Priority**:
1.  **Start with Pre-trained Features** (DINOv2/CLIP).
2.  **Apply Top-K Sparsity** (k=16 to 32).
3.  **Use FSQ** if discrete logic is the priority; use **SimVQ** if reconstruction quality is priority.