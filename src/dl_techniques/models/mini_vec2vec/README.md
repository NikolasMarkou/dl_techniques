# Mini-Vec2Vec: Unsupervised Embedding Space Alignment

A production-ready Keras 3 implementation of the **mini-vec2vec** algorithm for aligning two embedding spaces without parallel data using a single linear transformation.

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Keras Version](https://img.shields.io/badge/Keras-3.0-red.svg)](https://keras.io/)
[![Paper](http://img.shields.io/badge/paper-arXiv-B31B1B.svg)](https://arxiv.org/abs/2510.02348)

## Table of Contents

1.  [Overview](#overview)
    *   [The Core Idea: Universal Geometry](#the-core-idea-universal-geometry)
    *   [Key Features](#key-features)
    *   [Use Cases](#use-cases)
2.  [The Model and Algorithm](#the-model-and-algorithm)
    *   [The Model: A Simple Linear Transformation](#the-model-a-simple-linear-transformation)
    *   [The "Training" Process: An Algorithmic Journey](#the-training-process-an-algorithmic-journey)
3.  [Quick Start](#quick-start)
4.  [Detailed Usage](#detailed-usage)
    *   [Data Preparation](#data-preparation)
    *   [Running the Alignment](#running-the-alignment)
    *   [Evaluation](#evaluation)
5.  [Hyperparameter Guide](#hyperparameter-guide)
6.  [API Reference](#api-reference)
7.  [Performance Considerations](#performance-considerations)
8.  [Model Serialization](#model-serialization)

## 1. Overview

This repository provides a robust implementation of the **mini-vec2vec** algorithm, which addresses a fundamental challenge: **unsupervised embedding space alignment**. Given two sets of embeddings that represent similar concepts (e.g., from different languages, modalities, or models), how can we make them directly comparable *without any paired examples*?

Previous methods often relied on complex and unstable GANs. `mini-vec2vec` introduces a significantly faster, more stable, and more interpretable algorithmic approach.

### The Core Idea: Universal Geometry

The method is grounded in the **Platonic Representation Hypothesis**, or the idea of **universal geometry**. It posits that well-trained models, regardless of their architecture, learn to organize concepts in geometrically similar ways. While their coordinate systems are arbitrary, the *relative distances and angles* between concepts are preserved.

`mini-vec2vec` exploits this shared structure. The goal is not to learn a complex, non-linear mapping, but simply to find the optimal **rotation and reflection** (an orthogonal linear transformation) that superimposes one space's geometric structure onto the other's.

### Key Features

-   **Unsupervised Learning**: No parallel data or paired examples are required.
-   **Stable & Efficient**: Replaces unstable adversarial training with a deterministic, multi-stage algorithm that runs in minutes on a CPU.
-   **Robust Alignment**: Uses an ensemble-based anchor matching technique to establish a strong initial correspondence.
-   **Interpretable**: Each step has a clear geometric purpose (clustering, QAP, Procrustes), making the process easy to understand.
-   **Keras 3 Native**: Implemented as a `keras.Model`, allowing the learned transformation to be saved, loaded, and seamlessly integrated into larger pipelines.

### Use Cases

-   **Cross-lingual Word Embeddings**: Align word embeddings from different languages to enable zero-shot translation.
-   **Multi-modal Alignment**: Map image and text embeddings into a shared space for cross-modal retrieval.
-   **Domain Adaptation**: Align embeddings from different domains (e.g., news vs. scientific papers) or time periods.
-   **Model Stitching**: Align the internal representations of two different neural networks.

## 2. The Model and Algorithm

### The Model: A Simple Linear Transformation

The "model" in `mini-vec2vec` is a single **orthogonal matrix, `W`**.

-   **Function:** It performs a linear transformation (a rotation and/or reflection) to map a vector from the source space (Space A) into the coordinate system of the target space (Space B).
    ```
    aligned_vector_A = original_vector_A @ W
    ```
-   **The Goal:** The entire "training" process is designed to find the optimal `W` matrix that best aligns the two spaces' universal geometries.

### The "Training" Process: An Algorithmic Journey

`mini-vec2vec` replaces gradient-based training with a multi-stage, algorithmic procedure. This is the heart of the paper's contribution.

#### Stage 1: Approximate Matching
The most critical stage, where reliable "pseudo-pairs" are created without supervision.
1.  **Anchor Discovery**: Both spaces are independently clustered using K-Means. The cluster centroids act as high-level "landmarks."
2.  **Anchor Matching (QAP)**: The structural similarity *between* centroids in each space is used to solve the **Quadratic Assignment Problem (QAP)**, finding the optimal matching of landmarks between the two spaces. This is ensembled over multiple runs for robustness.
3.  **Relative Representations**: Each embedding is re-represented by its vector of cosine similarities to the matched anchors. This creates a shared, coordinate-system-agnostic representation.
4.  **Pseudo-Pair Generation**: Embeddings from space A are matched to their nearest neighbors in this shared *relative space*. The average of these neighbors forms a stable pseudo-target.

#### Stage 2: Initial Mapping (Procrustes Analysis)
With the noisy-but-reliable pseudo-pairs, a strong initial transformation is computed.
-   **Procrustes Analysis** is used to find the optimal orthogonal matrix `W` that minimizes the distance between the source pairs and their pseudo-targets.

#### Stage 3: Iterative Refinement
The initial `W` is refined through two complementary bootstrapping strategies.
-   **Refine-1 (Matching-Based)**: `W` is used to transform samples from space A. New, higher-quality nearest-neighbor pairs are found directly in the target space. These pairs are used to compute an updated `W`, and the process is repeated with exponential smoothing.
-   **Refine-2 (Clustering-Based)**: `W` is used to guide the clustering of space B, creating a very stable, large-scale structural matching. This helps correct any errors from the local, matching-based refinement and produces the final, highly accurate `W`.

## 3. Quick Start

```python
import numpy as np
import keras
from dl_techniques.models.mini_vec2vec import MiniVec2VecAligner

# Assume you have your embedding matrices loaded
# Shape: (n_samples, embedding_dim)
source_embeddings = np.load('source_embeddings.npy')
target_embeddings = np.load('target_embeddings.npy')

# 1. Initialize the aligner
aligner = MiniVec2VecAligner(embedding_dim=128)
aligner.build(input_shape=(None, 128))

# 2. Run the alignment process
history = aligner.align(
    XA=source_embeddings,
    XB=target_embeddings,
)

# 3. Transform new source embeddings into the aligned space
new_source_data = np.random.rand(100, 128)
aligned_embeddings = aligner(new_source_data)

# 4. Save the fitted model for later use
aligner.save('mini_vec2vec_aligner.keras')

# 5. Load and reuse
loaded_aligner = keras.models.load_model('mini_vec2vec_aligner.keras')
realigned_embeddings = loaded_aligner(new_source_data)
```

## 4. Detailed Usage

### Data Preparation

The `aligner` expects NumPy arrays. The internal algorithm will handle centering and normalization, but providing pre-processed data is good practice.

-   **Format**: NumPy arrays or tensors convertible to NumPy.
-   **Shape**: `(n_samples, embedding_dim)`.
-   **Preprocessing**: The `align` method automatically centers the data and normalizes all vectors to the unit hypersphere.

### Running the Alignment

The `align` method exposes all key hyperparameters of the algorithm.

```python
# Align with custom hyperparameters
history = aligner.align(
    XA=embeddings_A,
    XB=embeddings_B,
    # Approximate Matching
    approx_clusters=20,      # Number of anchor clusters
    approx_runs=30,          # Ensemble runs for robustness
    approx_neighbors=10,     # Neighbors to average for pseudo-pairs
    # Refine-1 (Matching-based)
    refine1_iterations=75,   # Refinement iterations
    refine1_sample_size=5000,# Samples per iteration
    refine1_neighbors=10,    # Neighbors for matching
    # Refine-2 (Clustering-based)
    refine2_clusters=200,    # Fine-grained clusters
    # General
    smoothing_alpha=0.5      # Update smoothing factor
)

# The history object contains the state of W after each major stage
final_W = history['final_W']
```

### Evaluation

To verify the alignment quality, you need a held-out set of embeddings with known ground-truth pairings.

```python
from sklearn.neighbors import NearestNeighbors

def evaluate_alignment(aligner, test_A, test_B):
    aligned_A = aligner(test_A)
    nn = NearestNeighbors(n_neighbors=1, metric='cosine').fit(test_B)
    _, indices = nn.kneighbors(aligned_A)
    
    correct = np.sum(indices.flatten() == np.arange(len(test_A)))
    accuracy = correct / len(test_A)
    
    # Cosine similarity assumes unit vectors, which is true after alignment
    cosine_sim = np.mean(np.sum(aligned_A * test_B, axis=1))
    
    return {'top1_accuracy': accuracy, 'mean_cosine_similarity': cosine_sim}

# Assuming test_A and test_B are ground-truth pairs
metrics = evaluate_alignment(aligner, test_A, test_B)
print(f"Top-1 Accuracy: {metrics['top1_accuracy']:.4f}")
```

## 5. Hyperparameter Guide

#### `approx_clusters` (default: 20)
-   Number of anchor centroids.
-   **Recommendation**: 20 is a robust default. Increase to 30-50 for very complex or dissimilar spaces.

#### `approx_runs` (default: 30)
-   Number of ensemble runs for anchor matching.
-   **Recommendation**: 30 provides a great balance of robustness and speed. Increase to 50 for maximum stability.

#### `approx_neighbors` (default: 50)
-   Number of neighbors averaged to create pseudo-pairs.
-   **Recommendation**: Lower values (10-20) can be more precise but noisy. Higher values (50-100) are more stable. Start with 50.

#### `refine1_iterations` (default: 75)
-   Number of matching-based refinement loops.
-   **Recommendation**: 50-75 is sufficient for most cases. The process converges quickly.

#### `refine1_sample_size` (default: 10000)
-   Number of samples used in each Refine-1 iteration.
-   **Recommendation**: Use ~10-20% of your alignment dataset size for stable updates.

#### `refine2_clusters` (default: 500)
-   Number of clusters for the final, fine-grained refinement.
-   **Recommendation**: Use a significantly larger number than `approx_clusters`, typically 10x-25x larger.

## 6. API Reference

### `MiniVec2VecAligner`

```python
class MiniVec2VecAligner(keras.Model):
    """Unsupervised embedding space alignment model."""
    
    def __init__(self, embedding_dim: int, **kwargs):
        """Initializes the aligner with the embedding dimensionality."""
    
    def align(self, XA, XB, **kwargs) -> Dict[str, np.ndarray]:
        """Executes the full alignment pipeline to fit the transformation matrix W."""
    
    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Transforms input embeddings using the learned W matrix."""
```

## 7. Performance Considerations

-   **Complexity**: The runtime is dominated by K-Means and Nearest Neighbors searches. It scales linearly with the number of samples and is significantly faster than GAN-based approaches.
-   **Memory**: Peak memory usage occurs during the creation of relative representations. This is proportional to `(num_samples * approx_clusters * approx_runs)`.
-   **Optimization**: For very large datasets (> 1M samples), consider reducing `approx_runs` (e.g., to 15-20) and `refine1_sample_size`.

## 8. Model Serialization

The model is fully serializable using the standard Keras API.

```python
# Save the fitted model
aligner.save('aligner.keras')

# Load the model in a different session
import keras
loaded_aligner = keras.models.load_model('aligner.keras')

# The loaded model contains the fitted W matrix and is ready for inference
aligned_vectors = loaded_aligner(new_source_vectors)
```