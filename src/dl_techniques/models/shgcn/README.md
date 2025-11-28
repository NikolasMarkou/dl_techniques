# sHGCN: Simplified Hyperbolic Graph Convolutional Network

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Graph Learning](https://img.shields.io/badge/Task-Hyperbolic%20Graph%20Learning-purple)](https://arxiv.org/abs/2104.06942)

A production-ready, fully-featured implementation of **Simplified Hyperbolic Graph Convolutional Networks (sHGCN)** in **Keras 3**. This architecture leverages the expressive power of hyperbolic geometry to model hierarchical graph structures (like trees, taxonomies, and biological networks) while maintaining the computational efficiency of Euclidean operations.

sHGCN simplifies standard Hyperbolic GCNs by performing neighbor aggregation in the tangent (Euclidean) space, using hyperbolic geometry specifically for bias operations. This results in a faster, more stable model that still captures the exponential expansion of hierarchical data.

---

## Table of Contents

1. [Overview: What is sHGCN and Why It Matters](#1-overview-what-is-shgcn-and-why-it-matters)
2. [The Problem sHGCN Solves](#2-the-problem-shgcn-solves)
3. [How sHGCN Works: Core Concepts](#3-how-shgcn-works-core-concepts)
4. [Architecture Deep Dive](#4-architecture-deep-dive)
5. [Quick Start Guide](#5-quick-start-guide)
6. [Component Reference](#6-component-reference)
7. [Configuration & Model Variants](#7-configuration--model-variants)
8. [Comprehensive Usage Examples](#8-comprehensive-usage-examples)
9. [Advanced Usage Patterns](#9-advanced-usage-patterns)
10. [Performance Optimization](#10-performance-optimization)
11. [Training and Best Practices](#11-training-and-best-practices)
12. [Serialization & Deployment](#12-serialization--deployment)
13. [Testing & Validation](#13-testing--validation)
14. [Troubleshooting & FAQs](#14-troubleshooting--faqs)
15. [Technical Details](#15-technical-details)
16. [Citation](#16-citation)

---

## 1. Overview: What is sHGCN and Why It Matters

### What is sHGCN?

**sHGCN** is a Graph Neural Network (GNN) designed for **scale-free** and **hierarchical graphs**. While standard GCNs embed nodes in flat Euclidean space, sHGCN utilizes concepts from Hyperbolic geometry (specifically the Poincaré ball model).

However, full Hyperbolic GCNs are computationally expensive and numerically unstable. sHGCN serves as a hybrid:
1.  **Linear Transform**: Euclidean.
2.  **Bias Addition**: Hyperbolic (Möbius addition).
3.  **Aggregation**: Euclidean (Tangent space).

### Key Innovations of this Implementation

1.  **Task-Specific Wrappers**: Includes pre-built `SHGCNNodeClassifier` and `SHGCNLinkPredictor` classes for immediate deployment.
2.  **Learnable Curvature**: Each layer can learn its own curvature parameter $c$, adapting the geometry to the data automatically.
3.  **Fermi-Dirac Decoding**: Implements the specialized decoder for link prediction in hyperbolic-like latent spaces.
4.  **Keras 3 Native**: Built with modern Keras Functional API, supporting Sparse Tensors and backend-agnostic execution (JAX, TF, PyTorch).

### Why sHGCN Matters

**Standard Euclidean GCNs**:
```
Model: Flat Geometry
  1. Distortion: Cannot embed trees without high distortion.
  2. Crowding: Nodes deep in a hierarchy get crowded at the edge of the space.
  3. Result: Poor performance on citation networks, taxonomies, and disease maps.
```

**sHGCN Solution**:
```
Model: Hyperbolic-Euclidean Hybrid
  1. Capacity: Hyperbolic space expands exponentially, matching tree growth.
  2. Efficiency: Aggregates neighbors in tangent space (vector addition) 
     instead of using expensive Fréchet means.
  3. Result: High accuracy on hierarchical data with standard GCN speeds.
```

---

## 2. The Problem sHGCN Solves

### The Hierarchy Mismatch

Many real-world graphs (Internet, Citation Networks, biology) are hierarchical. In Euclidean space, the volume of a ball grows polynomially ($r^n$). In hierarchical graphs (trees), the number of nodes grows exponentially with depth.

To fit a tree into Euclidean space, you must squash nodes together, losing structural information. Hyperbolic space has exponential volume growth, naturally accommodating trees.

### The Complexity of Pure Hyperbolic GNNs

Pure HGCNs perform all operations in hyperbolic space.
*   **Problem**: Adding vectors in hyperbolic space (Möbius addition) and calculating centroids (Fréchet means) is slow and numerically unstable (exploding gradients near the boundary).
*   **Solution**: sHGCN proves that you only need the **Hyperbolic Bias** to capture the hierarchy. The heavy lifting (aggregation) can be done safely in Euclidean tangent space.

---

## 3. How sHGCN Works: Core Concepts

### The Processing Pipeline

sHGCN defines a layer that switches between spaces strategically.

```
Input Features (Euclidean)
    │
    ▼
[Linear Transform W] ──► (Feature extraction)
    │
    ▼
[Exp Map] ─────────────► (Project to Hyperbolic Space)
    │
    ▼
[Möbius Bias + b] ─────► (Apply Hierarchical Shift)
    │
    ▼
[Log Map] ─────────────► (Project back to Tangent Space)
    │
    ▼
[Aggregation Ã] ───────► (Sum Neighbors efficiently)
    │
    ▼
[Activation σ] ────────► Output
```

### The Fermi-Dirac Decoder

For link prediction, simple dot products don't work well with this geometry. We use a **Fermi-Dirac Decoder** which models the probability of an edge based on distance:
$$ P(edge) \propto \frac{1}{1 + e^{(dist^2 - r)/t}} $$
This effectively learns a "radius of interaction" ($r$) and a "softness" ($t$).

---

## 4. Architecture Deep Dive

### 4.1 `SHGCNLayer`
The core compute unit.
*   **Curvature ($c$)**: A learnable parameter $c = \text{softplus}(\theta)$. If $c \to 0$, the space becomes Euclidean. High $c$ implies high curvature (strict hierarchy).
*   **Möbius Bias**: The operation $z \oplus_c b$. This is the only hyperbolic operation in the layer.

### 4.2 `SHGCNNodeClassifier`
A wrapper for node classification.
*   **Backbone**: Stack of `SHGCNLayer`s.
*   **Head**: Standard Softmax Dense layer.
*   **Use Case**: Predicting paper subjects, protein functions.

### 4.3 `SHGCNLinkPredictor`
A wrapper for edge prediction.
*   **Backbone**: Stack of `SHGCNLayer`s.
*   **Head**: `FermiDiracDecoder`.
*   **Use Case**: Recommendation systems, knowledge graph completion.

---

## 5. Quick Start Guide

### Node Classification (30 seconds)

Classify nodes in a graph (e.g., Cora/Citeseer).

```python
import keras
from model import SHGCNNodeClassifier

# 1. Create Model
model = SHGCNNodeClassifier(
    num_classes=7,
    hidden_dims=[64, 32],
    embedding_dim=16,
    dropout_rate=0.5
)

# 2. Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 3. Dummy Data (100 nodes, 50 features)
# Adjacency must be a SparseTensor
import tensorflow as tf
import numpy as np

features = keras.random.normal((100, 50))
indices = [[0, 1], [1, 2], [2, 0]] # Simple triangle
values = [1.0, 1.0, 1.0]
adj = tf.sparse.SparseTensor(indices, values, dense_shape=(100, 100))
labels = keras.random.randint((100,), 0, 7)

# 4. Train
# Inputs are a list: [Features, Adjacency]
model.fit([features, adj], labels, epochs=10)
```

---

## 6. Component Reference

### 6.1 `SHGCNModel` (Base Class)

**Purpose**: Flexible backbone for custom tasks.

```python
from model import SHGCNModel

model = SHGCNModel(
    hidden_dims=[64, 64],
    output_dim=32,
    output_activation='linear',
    use_curvature=True
)
```

### 6.2 `FermiDiracDecoder`

**Purpose**: Calculates probability of edges based on node embeddings.

```python
from model import FermiDiracDecoder

decoder = FermiDiracDecoder()
probs = decoder([node_embeddings_u, node_embeddings_v])
```

---

## 7. Configuration & Model Variants

You can tune the model behavior via the constructor:

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `hidden_dims` | List[int] | Required | Width of hidden layers |
| `use_bias` | bool | `True` | Enable hyperbolic bias (key feature) |
| `use_curvature` | bool | `True` | Learn curvature $c$ per layer |
| `dropout_rate` | float | `0.0` | Regularization (usually 0.5-0.6 for GNNs) |

---

## 8. Comprehensive Usage Examples

### Example 1: Link Prediction

Training a model to predict missing edges.

```python
from model import SHGCNLinkPredictor

# 1. Setup
model = SHGCNLinkPredictor(hidden_dims=[32], embedding_dim=16)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["auc"])

# 2. Data
# Edge pairs: [Num_Edges, 2] containing node indices
pos_edges = np.array([[0, 1], [1, 2]]) 
neg_edges = np.array([[0, 5], [2, 9]]) # Sampled non-existent edges
edge_pairs = np.vstack([pos_edges, neg_edges])
labels = np.array([1, 1, 0, 0])

# 3. Train
# Input: [Features, Adjacency, EdgePairs]
model.fit([features, adj, edge_pairs], labels, epochs=50)
```

### Example 2: Analyzing Learned Curvature

Since curvature is learnable, we can inspect it to see how "hyperbolic" the data is.

```python
# Access the first layer
layer = model.backbone.hidden_layers[0]

# Get curvature value (c = softplus(theta))
c_val = layer.curvature.numpy()
print(f"Layer 1 Curvature: {c_val:.4f}")

# Interpretation:
# c near 0 -> Flat (Euclidean) data
# c >> 0   -> Highly hierarchical (Hyperbolic) data
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Sparse Adjacency Normalization
GCNs require normalized adjacency matrices $\tilde{A} = D^{-1/2}AD^{-1/2}$. sHGCN expects this normalization to be pre-calculated.

```python
# Pseudo-code for preprocessing
d = np.array(adj.sum(1))
d_inv_sqrt = np.power(d, -0.5).flatten()
d_mat = sp.diags(d_inv_sqrt)
norm_adj = d_mat.dot(adj).dot(d_mat)
# Convert norm_adj to SparseTensor for the model
```

### Pattern 2: Embedding Visualization
To visualize the learned hierarchy, project the output embeddings to the Poincaré disk.
*   The `SHGCNLinkPredictor` outputs Euclidean tangent space vectors.
*   Use `PoincareMath.exp_map_0` (from `shgcn_layer.py`) to map them to the disk for plotting (2D) to seeing the tree structure.

---

## 10. Performance Optimization

### Sparse Tensor Support
This implementation relies on `tf.sparse.sparse_dense_matmul` (via Keras ops). Ensure your adjacency matrix is passed as a **Sparse Tensor**. Passing a dense matrix for adjacency will cause OOM errors on large graphs ($N^2$ memory).

### Numerical Stability
Hyperbolic operations involve `exp` and `log`. The `SHGCNLayer` includes a `project` step to keep points away from the boundary of the Poincaré ball, preventing NaNs. If you encounter instability, try reducing the learning rate or increasing `eps` in `PoincareMath`.

---

## 11. Training and Best Practices

### Optimizer
`Adam` usually works well. For hyperbolic parameters, specialized optimizers (Riemannian Adam) are often used in literature, but sHGCN is designed to work well with standard **Euclidean Adam** because gradients are computed in the tangent space.

### Dropout
Graph datasets are often small and prone to overfitting. High dropout (`0.5` to `0.6`) is standard practice for the feature input and hidden layers.

---

## 12. Serialization & Deployment

sHGCN models are fully serializable.

```python
# Save
model.save("shgcn_cora.keras")

# Load
# Note: Custom objects are handled via @register_keras_serializable
loaded_model = keras.models.load_model("shgcn_cora.keras")
```

---

## 13. Testing & Validation

```python
def test_shgcn_forward():
    model = SHGCNNodeClassifier(num_classes=2, hidden_dims=[16])
    
    # B=1 graph, N=10 nodes, F=5 features
    x = keras.random.normal((10, 5))
    
    # Sparse Identity matrix as dummy adjacency
    indices = [[i, i] for i in range(10)]
    values = [1.0] * 10
    adj = tf.sparse.SparseTensor(indices, values, dense_shape=(10, 10))
    
    out = model([x, adj])
    assert out.shape == (10, 2)
    print("✓ Forward pass successful")

test_shgcn_forward()
```

---

## 14. Troubleshooting & FAQs

**Q: Can I use this on non-hierarchical graphs?**
*   **A:** Yes. If the graph is not hierarchical, the model should learn a curvature $c \approx 0$, effectively acting as a standard GCN.

**Q: Why "Simplified"?**
*   **A:** Standard HGCNs aggregate in hyperbolic space using the Fréchet mean, which is an iterative optimization problem inside every forward pass. sHGCN aggregates in Euclidean space (vector sum), which is a single matrix multiplication.

**Q: Input shapes?**
*   **A:** Features: `(N, F)`. Adjacency: `(N, N)` Sparse. Labels: `(N,)` for sparse categorical loss.

---

## 15. Technical Details

### The sHGCN Formula (Eq. 14)
$$ H^l = \sigma \left( \tilde{A} \log_0^c \left( \exp_0^c(W H^{l-1}) \oplus_c \exp_0^c(b) \right) \right) $$

1.  **$WH^{l-1}$**: Euclidean transform.
2.  **$\exp_0^c$**: Map to Manifold.
3.  **$\oplus_c$**: Add bias on Manifold (shifts the center of the hierarchy).
4.  **$\log_0^c$**: Map back to Tangent space.
5.  **$\tilde{A} \dots$**: Aggregate neighbors in Tangent space.

### Möbius Addition
$$ x \oplus_c y = \frac{(1 + 2c\langle x, y \rangle + c||y||^2)x + (1 - c||x||^2)y}{1 + 2c\langle x, y \rangle + c^2||x||^2 ||y||^2} $$
This operation is what allows the bias $b$ to act as a translation in hyperbolic space, effectively traversing the hierarchy.

---

## 16. Citation

If you use this model, please cite the original paper:

```bibtex
@inproceedings{arevalo2021simplified,
  title={Simplified Graph Convolutional Networks},
  author={Arevalo, John and Duc, Federico and Rizzo, Stefano and others},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2021}
}
```