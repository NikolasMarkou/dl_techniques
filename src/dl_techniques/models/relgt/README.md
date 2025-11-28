# RELGT: Modern Relational Graph Transformer

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Graph Learning](https://img.shields.io/badge/Task-Graph%20Learning-green)](https://github.com/topics/graph-neural-networks)

A production-ready, fully-featured implementation of the **Modern Relational Graph Transformer (RELGT)** in **Keras 3**. This architecture integrates multi-element tokenization with a hybrid local-global attention mechanism, designed specifically for complex predictive modeling on relational databases and heterogeneous graphs.

RELGT addresses the limitations of standard Graph Neural Networks (GNNs) by decoupling structural information from feature processing using learnable global centroids, allowing for the capture of long-range dependencies without the quadratic cost of standard transformers.

---

## Table of Contents

1. [Overview: What is RELGT and Why It Matters](#1-overview-what-is-relgt-and-why-it-matters)
2. [The Problem RELGT Solves](#2-the-problem-relgt-solves)
3. [How RELGT Works: Core Concepts](#3-how-relgt-works-core-concepts)
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

## 1. Overview: What is RELGT and Why It Matters

### What is RELGT?

**RELGT** is a specialized Transformer architecture for relational deep learning. It operates on subgraphs sampled around a specific target entity (the "seed node"). Unlike standard GNNs that aggregate messages recursively, RELGT tokenizes the entire subgraph—encoding features, node types, hop distances, and structural identifiers—and processes them through a Transformer.

Its defining feature is the use of **Global Centroids**: learnable "virtual tokens" that attend to the entire subgraph. These centroids act as information hubs, gathering context from local nodes and redistributing global context back to them.

### Key Innovations of this Implementation

1.  **Multi-Element Tokenization**: A unified `RELGTTokenEncoder` that fuses raw features, entity types, hop distances, and GNN-based positional encodings into a rich vector representation for every node.
2.  **Hybrid Local-Global Attention**: Uses a set of learnable global tokens (centroids) to capture graph-level context, avoiding the computational expense of full $O(N^2)$ attention while preventing over-smoothing.
3.  **Keras 3 Native**: Built with the modern Keras Functional API, ensuring backend agnosticism (JAX, TensorFlow, PyTorch) and full serialization support.
4.  **Factory Methods**: Includes `create_relgt_model` with presets (`small`, `base`, `large`) for rapid experimentation.

### Why RELGT Matters

**Standard GNNs (GCN/GAT)**:
```
Model: Recursive Neighborhood Aggregation
  1. Limited receptive field (depends on number of layers).
  2. Prone to "over-smoothing" (features become indistinguishable in deep nets).
  3. Struggles to model interactions between distant nodes.
```

**RELGT Solution**:
```
Model: Relational Graph Transformer
  1. Global Receptive Field: Global centroids see the entire subgraph immediately.
  2. Structural Awareness: Explicit encoding of hops and node types preserves 
     relational schema structure.
  3. Expressivity: Can model complex interactions between entity types 
     (e.g., User-Product-Review) better than simple message passing.
```

---

## 2. The Problem RELGT Solves

### The Challenge of Relational Data

Relational databases and heterogeneous graphs contain entities of different types (e.g., *Customers*, *Products*, *Orders*) connected by specific relationships.
1.  **Heterogeneity**: Handling different feature spaces for different node types is difficult for standard Transformers.
2.  **Long-Range Dependencies**: A decision about a *Customer* might depend on an *Order* placed months ago, which might be 4-5 hops away in the graph. GNNs often dilute this signal.

### The Global Centroid Solution

RELGT introduces a bottleneck mechanism. Instead of every node attending to every other node (expensive and noisy):
1.  Nodes attend to **Global Centroids**.
2.  Global Centroids aggregate information.
3.  Information is broadcast back to nodes (or used directly for prediction).

This creates a "global workspace" where the model can reason about the overall structure of the sampled subgraph.

---

## 3. How RELGT Works: Core Concepts

### The High-Level Architecture

RELGT processes a sampled subgraph (centered on a seed node) to predict a property of that seed node (or the whole graph).

```
┌──────────────────────────────────────────────────────────────────┐
│                      RELGT Architecture                          │
│                                                                  │
│  Input Dictionary (Subgraph)                                     │
│   ├─ Node Features                                               │
│   ├─ Node Types                                                  │
│   ├─ Hop Distances                                               │
│   └─ Adjacency Structure                                         │
│            │                                                     │
│            ▼                                                     │
│  ┌───────────────────┐                                           │
│  │ RELGTTokenEncoder │ (Fuses all inputs into Token Vectors)     │
│  └─────────┬─────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│  ┌───────────────────┐                                           │
│  │ Transformer Blocks│ (Local Tokens <-> Global Centroids)       │
│  └─────────┬─────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│  ┌───────────────────┐                                           │
│  │  Prediction Head  │ (FFN + Softmax/Linear)                    │
│  └───────────────────┘                                           │
└──────────────────────────────────────────────────────────────────┘
```

### The Data Flow

1.  **Tokenization**: Raw inputs are projected. Embeddings for discrete inputs (Types, Hops) are added. A mini-GNN adds structural positional encoding.
2.  **Seed Encoding**: The target node (index 0) gets special treatment to preserve its identity.
3.  **Interaction**: 
    *   Tokens update Centroids (Bottom-up).
    *   Centroids update Tokens (Top-down).
4.  **Prediction**: The final representation (usually the seed node's updated state or the centroids) is passed to an MLP.

---

## 4. Architecture Deep Dive

### 4.1 `RELGTTokenEncoder`

This component is responsible for turning a graph into a sequence of vectors.
*   **Feature Projection**: Linear layer maps variable-sized node features to `embedding_dim`.
*   **Type Embedding**: `Embedding(num_node_types, embedding_dim)` captures schema info.
*   **Hop Embedding**: `Embedding(max_hops, embedding_dim)` encodes distance from the seed.
*   **GNN PE**: A shallow GNN (e.g., GCN) runs on the structure to generate structural embeddings (Structure PE).

### 4.2 `RELGTTransformerBlock`

The core compute unit.
*   **Inputs**: Local Tokens (nodes) and Global Centroids (learnable parameters).
*   **Attention**: Performs Cross-Attention between Local Tokens and Global Centroids.
*   **FFN**: Standard Feed-Forward Network with residual connections and Normalization.

---

## 5. Quick Start Guide

### Your First RELGT Model (30 seconds)

Build a model for node classification with 5 distinct node types.

```python
import keras
from model import create_relgt_model

# 1. Create Model (Base configuration)
model = create_relgt_model(
    output_dim=2,               # Binary classification
    problem_type="classification",
    model_size="base",          # 128 dim, 4 heads
    num_node_types=5
)

# 2. Create Dummy Data (Batch of 2 subgraphs, 10 nodes each)
inputs = {
    "node_features": keras.random.normal((2, 10, 64)), # 64 raw features
    "node_types": keras.random.randint((2, 10), 0, 5), # Type IDs 0-4
    "node_hops": keras.random.randint((2, 10), 0, 3),  # 0, 1, or 2 hops
    "adjacency": keras.random.normal((2, 10, 10)),     # Adj matrix (or sparse)
    # Note: Real usage often requires specific tensor formats for GNN PE
}

# 3. Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 4. Forward Pass
output = model(inputs)
print(f"Output Shape: {output.shape}") # (2, 2)
```

---

## 6. Component Reference

### 6.1 `RELGT` (Model Class)

**Purpose**: The main Keras `Model` orchestrating the pipeline.

```python
from model import RELGT

model = RELGT(
    output_dim=10,
    embedding_dim=128,
    num_transformer_blocks=3,
    num_global_centroids=32,
    gnn_pe_dim=32
    # ... other args
)
```

### 6.2 Factory Function

#### `create_relgt_model(output_dim, problem_type, model_size)`
The recommended way to instantiate.
*   `model_size`: `'small'`, `'base'`, or `'large'`.

---

## 7. Configuration & Model Variants

Standard configurations provided by the factory function.

| Variant | Embed Dim | Heads | Global Centroids | Blocks | Params (Approx) | Use Case |
|:---:|:---:|:---:|:---:|:---:|:---|
| **`small`** | 64 | 2 | 16 | 1 | ~100k | Simple baselines, debugging |
| **`base`** | 128 | 4 | 32 | 2 | ~500k | Standard tabular/graph tasks |
| **`large`** | 256 | 8 | 64 | 4 | ~2M+ | Large-scale heterogeneous graphs |

---

## 8. Comprehensive Usage Examples

### Example 1: Regression on Molecular Data

Predicting a property (float) for a molecule (graph).

```python
import keras
from model import create_relgt_model

# 1. Setup
model = create_relgt_model(
    output_dim=1,
    problem_type="regression",
    model_size="small",
    num_node_types=10  # e.g., Atom types (C, O, N, etc.)
)

# 2. Compile (MSE for regression)
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3),
    loss="mean_squared_error",
    metrics=["mean_absolute_error"]
)

# 3. Training (Pseudo-code for data pipeline)
# train_ds needs to yield ({inputs_dict}, targets)
# history = model.fit(train_ds, epochs=50)
```

### Example 2: Preparing the Input Dictionary

RELGT relies on a specific input structure. Here is how to format it manually.

```python
import keras
import numpy as np

def format_batch(features, types, adjs):
    """
    features: (B, N, F)
    types: (B, N)
    adjs: (B, N, N)
    """
    batch_size, num_nodes, _ = features.shape
    
    # Calculate hops (simple BFS approximation or pre-calculated)
    # Here we just use dummy random hops for demonstration
    hops = np.random.randint(0, 3, size=(batch_size, num_nodes))
    
    return {
        "node_features": keras.ops.convert_to_tensor(features),
        "node_types": keras.ops.convert_to_tensor(types),
        "node_hops": keras.ops.convert_to_tensor(hops),
        "adjacency": keras.ops.convert_to_tensor(adjs)
    }

# Use this dict as input to model.fit or model.predict
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Customizing the GNN Positional Encoding

The `gnn_pe` component helps the transformer understand local topology before attention. You can tune its depth and dimension.

```python
model = RELGT(
    ...,
    gnn_pe_dim=64,    # Increase dimension for complex structures
    gnn_pe_layers=3   # More layers = larger local receptive field for PE
)
```

### Pattern 2: Seed-Oriented Tasks

RELGT assumes the node at index `0` in the input tensors is the "Seed Node" (the center of the subgraph).
*   The model explicitly encodes `inputs["node_features"][:, 0:1, :]` via `seed_encoder`.
*   Ensure your data pipeline sorts nodes such that the target node is first.

---

## 10. Performance Optimization

### Input Padding and Masking
Graph data is ragged (variable number of nodes). Keras models prefer fixed shapes or ragged tensors.
*   **Padding**: Pad all subgraphs to `max_nodes`.
*   **Masking**: RELGT handles masking internally if `mask` arguments are passed to the layers, but standard usage relies on the attention mechanism learning to ignore padded zeros (or explicit mask inputs if extended).

### Mixed Precision
Enable mixed precision for faster training on GPUs.

```python
keras.mixed_precision.set_global_policy('mixed_float16')
```

---

## 11. Training and Best Practices

### Hyperparameters
*   **Embedding Dim**: 128 is a sweet spot for relational data.
*   **Dropout**: `0.1` is standard, but increase to `0.3` for small, noisy graphs.
*   **Learning Rate**: Start with `1e-3` or `5e-4` with a Cosine Decay schedule.

### Data Normalization
*   Numerical node features should be standard scaled (mean 0, var 1).
*   Adjacency matrices should be normalized (e.g., symmetric normalization) if used by the internal GNN PE.

---

## 12. Serialization & Deployment

RELGT is fully serializable.

```python
# Save
model.save("relgt_model.keras")

# Load
loaded_model = keras.models.load_model("relgt_model.keras")

# Check config
print(loaded_model.get_config())
```

---

## 13. Testing & Validation

```python
import keras
from model import create_relgt_model

def test_relgt_forward():
    model = create_relgt_model(output_dim=1, model_size="small")
    
    # B=1, N=5, F=10
    inputs = {
        "node_features": keras.random.normal((1, 5, 10)),
        "node_types": keras.ops.zeros((1, 5), dtype="int32"),
        "node_hops": keras.ops.zeros((1, 5), dtype="int32"),
        "adjacency": keras.random.normal((1, 5, 5))
    }
    
    out = model(inputs)
    assert out.shape == (1, 1)
    print("✓ Forward pass successful")

test_relgt_forward()
```

---

## 14. Troubleshooting & FAQs

**Q: My loss is NaN.**
*   **A:** Check `node_features`. If they are unscaled large numbers, the `SeedEncoder` or `TokenEncoder` might explode. Apply `StandardScaler`. Also check for zeroes in `adjacency` if doing normalization manually.

**Q: How do I handle graphs larger than memory?**
*   **A:** RELGT is designed for *subgraphs*. You should not feed a whole million-node graph. Use a sampler (e.g., NeighborSampler) to extract small subgraphs (e.g., 20-50 nodes) around target entities.

**Q: Can I use this for Link Prediction?**
*   **A:** Yes. Construct a subgraph containing *both* source and target nodes of the link. Index 0 could be source, Index 1 target. Modify the `SeedEncoder` to extract both and concatenate them.

---

## 15. Technical Details

### Multi-Element Tokenization logic
The embedding for a node $u$ of type $\tau$ at hop $h$ with features $x_u$ is:
$$ \mathbf{z}_u = \text{Linear}(x_u) + \text{Emb}_{type}(\tau) + \text{Emb}_{hop}(h) + \text{GNN-PE}(u) $$

### Global Centroid Attention
Unlike standard self-attention ($\mathbf{Attention}(X, X, X)$), RELGT often utilizes:
$$ \mathbf{Z}' = \mathbf{Attention}(Q=\mathbf{C}, K=\mathbf{Z}, V=\mathbf{Z}) $$
Where $\mathbf{C}$ are the global centroids and $\mathbf{Z}$ are local node tokens. This compresses $N$ nodes into $K$ centroids.

---

## 16. Citation

This implementation is inspired by recent advancements in Relational Deep Learning and Graph Transformers. If you use this in your work, please cite the framework:

```bibtex
@misc{dl_techniques_relgt,
  author = {DL Techniques Contributors},
  title = {Modern Relational Graph Transformer (RELGT) Implementation},
  year = {2024},
  publisher = {GitHub},
  journal = {DL Techniques Repository}
}
```
