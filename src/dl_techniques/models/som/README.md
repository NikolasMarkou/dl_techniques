# Self-Organizing Map (SOM)

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18%2B-orange.svg)](https://www.tensorflow.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.23%2B-blueviolet.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-informational.svg)](https://matplotlib.org/)

A production-ready Keras 3 implementation of a **Self-Organizing Map (SOM)**, framed as a powerful associative memory system. This model provides a complete toolkit for unsupervised learning, clustering, classification, and, most importantly, for visualizing how high-dimensional data can be organized into a meaningful low-dimensional topological structure.

This implementation wraps a core `SOM2dLayer` into a full-featured `SOMModel`, providing an intuitive API for training, prediction, and an extensive suite of visualization methods. It is designed to be a practical tool for data scientists and an educational resource for understanding the principles of competitive learning and topological memory. The code adheres to modern Keras best practices, ensuring it is modular, well-documented, and fully serializable.

---

## Table of Contents

1. [Overview: What is a SOM and Why It Matters](#1-overview-what-is-a-som-and-why-it-matters)
2. [The Problem SOM Solves](#2-the-problem-som-solves)
3. [How SOM Works: Core Concepts](#3-how-som-works-core-concepts)
4. [Architecture Deep Dive](#4-architecture-deep-dive)
5. [Quick Start Guide](#5-quick-start-guide)
6. [Component Reference](#6-component-reference)
7. [Configuration & Parameters](#7-configuration--parameters)
8. [Comprehensive Usage Examples](#8-comprehensive-usage-examples)
9. [Advanced Usage: Classification](#9-advanced-usage-classification)
10. [Visualization Suite](#10-visualization-suite)
11. [Training and Best Practices](#11-training-and-best-practices)
12. [Serialization & Deployment](#12-serialization--deployment)
13. [Troubleshooting & FAQs](#13-troubleshooting--faqs)
14. [Technical Details](#14-technical-details)
15. [Citation](#15-citation)

---

## 1. Overview: What is a SOM and Why It Matters

### What is a SOM?

A **Self-Organizing Map (SOM)**, also known as a Kohonen map, is a type of neural network that uses unsupervised, competitive learning to produce a low-dimensional (typically 2D) representation of high-dimensional input data. This representation, called a "map," preserves the topological properties of the input space, meaning that similar data points are mapped to nearby locations on the grid.

### Key Features

1.  **Unsupervised Learning**: SOMs learn the structure of data without needing labels, making them ideal for exploratory data analysis.
2.  **Topological Preservation**: The key innovation of a SOM is that it organizes its neurons to reflect the relationships in the data. If two data points are similar, their corresponding "winning" neurons on the map will be close to each other.
3.  **Associative Memory**: The trained grid acts as a memory system. Presenting a new input activates a specific region (the "Best Matching Unit" and its neighbors), effectively recalling the learned prototype and associated data from that region.
4.  **Dimensionality Reduction & Visualization**: SOMs are unparalleled tools for projecting complex, high-dimensional data onto a 2D grid that humans can intuitively understand.

### Why a SOM Matters

**The "Black Box" Problem**:
```
Problem: Understand the hidden structure within a high-dimensional dataset.
Standard Approach (e.g., Autoencoders, PCA):
  1. Reduce data to a low-dimensional latent space.
  2. Visualize the latent space with scatter plots (e.g., t-SNE, UMAP).
  3. Limitation: These methods show clusters but often lose the continuous relationships
     or "topology" between data points. It's hard to see how one cluster transitions
     into another.
```

**The SOM Solution**:
```
SOM Approach:
  1. Define a 2D grid of "prototype" neurons.
  2. During training, neurons compete to represent input data.
  3. The winning neuron and its neighbors are updated to become more like the input.
  4. Benefit: This process creates a smooth, organized map where one can visually trace
     the relationships between different data clusters. It's not just about what's
     similar, but *how* things are similar.
```

### Real-World Impact

SOMs are exceptionally useful for gaining insights from complex datasets:

-   **Customer Segmentation**: Discovering natural groupings of customers and visualizing how segments relate to one another.
-   **Bioinformatics**: Clustering genes or proteins based on expression patterns to reveal biological pathways.
-   **Color Quantization**: Organizing a full-color palette into a smaller, representative set of colors.
-   **Robotics**: Creating maps of sensor data to help a robot understand its environment.
-   **Associative Memory Modeling**: Provides a simple yet powerful model for how biological brains might organize and recall information.

---

## 2. The Problem SOM Solves

### Bridging Unsupervised Learning and Intuitive Visualization

Many machine learning models can find clusters in data, but few can explain the relationships *between* those clusters. A standard clustering algorithm like K-Means might tell you there are three types of customers, but a SOM can show you that "Type A" is very similar to "Type B," while "Type C" is distinct.

```
┌───────────────────────────────────────────────────────────────┐
│  The Dilemma of Data Exploration                              │
│                                                               │
│  Clustering Algorithms (K-Means, DBSCAN):                     │
│    - Effective at partitioning data into discrete groups.     │
│    - Provide no information about inter-cluster similarity.   │
│                                                               │
│  Dimensionality Reduction (PCA, t-SNE):                       │
│    - Great for visualizing clusters in 2D or 3D.              │
│    - Distances in the projection can be misleading and do not │
│      always preserve the true "shape" of the data manifold.   │
└───────────────────────────────────────────────────────────────┘
```

The SOM provides a structured, grid-based visualization that overcomes these limitations. It doesn't just place similar points together; it organizes the entire data manifold onto a grid, making the topological structure explicit and interpretable.

---

## 3. How SOM Works: Core Concepts

The SOM learning process is an iterative algorithm that can be summarized in four steps:

1.  **Initialization**: Create a 2D grid of neurons. Each neuron is assigned a random weight vector with the same dimensionality as the input data.
2.  **Competition**: For each input data sample, find the neuron whose weight vector is most similar to it. This winning neuron is called the **Best Matching Unit (BMU)**.
3.  **Cooperation**: Identify the BMU's neighbors on the 2D grid. The size of this neighborhood shrinks over time.
4.  **Adaptation**: Update the weight vectors of the BMU and its neighbors, moving them closer to the input data sample. The amount of update (learning rate) also shrinks over time.

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                             SOM Training Cycle                                     │
│                                                                                    │
│  FOR each training epoch:                                                          │
│      // Learning rate and neighborhood radius decrease over time                   │
│      lr = update_learning_rate(epoch)                                              │
│      sigma = update_neighborhood_radius(epoch)                                     │
│                                                                                    │
│      FOR each input_sample in training_data:                                       │
│          1. COMPETITION:                                                           │
│             Find BMU = neuron with weights closest to input_sample                 │
│                                                                                    │
│          2. COOPERATION:                                                           │
│             Identify neighbors of BMU within radius `sigma`                        │
│                                                                                    │
│          3. ADAPTATION:                                                            │
│             FOR each neuron in neighbors:                                          │
│                 // Update is stronger for neurons closer to BMU                    │
│                 influence = calculate_influence(neuron, BMU, sigma)                │
│                 neuron.weights += lr * influence * (input_sample - neuron.weights) │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```
After many iterations, this process results in a smooth, topologically ordered map where adjacent neurons have similar weight vectors, forming a structured "memory" of the input data.

---

## 4. Architecture Deep Dive

### 4.1 `SOM2dLayer`

-   **Purpose**: The core Keras `Layer` that implements the SOM logic.
-   **Architecture**: This is not a typical deep learning layer. Its primary component is a single weight tensor, `weights_map`, of shape `(grid_height, grid_width, input_dim)`.
-   **Functionality**:
    -   **Forward Pass (`call`)**:
        1.  Calculates the Euclidean distance between the input batch and every neuron's weight vector in the `weights_map`.
        2.  Finds the coordinates of the BMU for each input sample.
        3.  If in `training` mode, it triggers the weight update mechanism for the BMUs and their neighbors.
        4.  Returns the BMU coordinates and the quantization error (distance to the BMU).
    -   **State Management**: It internally manages the current training iteration to decay the learning rate and neighborhood radius (`sigma`).

### 4.2 `SOMModel`

-   **Purpose**: The main `keras.Model` class that provides a high-level API for using the `SOM2dLayer`.
-   **Responsibilities**:
    1.  **Training Orchestration (`train_som`)**: Provides a custom training loop that iterates through the data for a specified number of epochs, feeding batches to the `SOM2dLayer`.
    2.  **Classification (`fit_class_prototypes`, `predict_classes`)**: Implements a post-training method to assign class labels to neurons, turning the unsupervised map into a classifier.
    3.  **Visualization Suite**: Contains a rich set of methods (`visualize_*`) to generate plots like the U-Matrix, class distribution maps, and memory recall demonstrations, making the learned map interpretable.

---

## 5. Quick Start Guide

### Installation

```bash
# Ensure you have the required dependencies
pip install keras>=3.0 tensorflow>=2.16 numpy matplotlib
```

### Your First SOM Model (30 seconds)

Let's build a small SOM and train it on some random data.

```python
import keras
import numpy as np

# Local imports from your project structure
from dl_techniques.models.memory.som.model import SOMModel

# 1. Create a SOM model
model = SOMModel(
    map_size=(10, 10),       # A 10x10 grid of neurons
    input_dim=3,             # For 3D data (e.g., RGB colors)
    initial_learning_rate=0.1,
    sigma=4.0                # Start with a large neighborhood
)
print("✅ SOM model created successfully!")

# 2. Create a dummy batch of data (e.g., 100 random colors)
dummy_data = np.random.rand(100, 3)

# 3. Train the SOM
# This is an unsupervised process! No labels needed.
history = model.train_som(dummy_data, epochs=5, verbose=1)
print("\n✅ SOM training complete!")
print(f"Final Quantization Error: {history['quantization_error'][-1]:.4f}")

# 4. Visualize the learned memory grid
# Since the input was colors, this will show a grid of organized color prototypes.
model.visualize_som_grid(figsize=(5, 5))
```

---

## 6. Component Reference

| Component | Location | Purpose |
| :--- | :--- | :--- |
| **`SOMModel`** | `...memory.som.model` | The main Keras `Model`. Provides the high-level API for training, prediction, and visualization. |
| **`SOM2dLayer`** | `...layers.memory.som_2d_layer` | The core Keras `Layer` containing the SOM weights and the competitive learning logic. |

---

## 7. Configuration & Parameters

### Key Architectural Parameters

-   **`map_size`**: `Tuple[int, int]`. The most important parameter. A larger map provides more detail but requires more training. A good starting point is `(10, 10)`.
-   **`input_dim`**: `int`. Must match the feature dimension of your input data.
-   **`initial_learning_rate`**: `float`. How strongly the weights are updated. A value like `0.1` is a common start. It will decay over time.
-   **`sigma`**: `float`. The initial radius of the neighborhood function, typically set to a fraction of the map size (e.g., `max(map_size) / 2`). It also decays over time.
-   **`neighborhood_function`**: `'gaussian'` or `'bubble'`. The Gaussian function provides a smoother update, while the bubble function is a simple step function. Gaussian is generally preferred.

---

## 8. Comprehensive Usage Examples

### Example: Training on MNIST Digits

Let's train a SOM on the MNIST dataset to see how it organizes handwritten digits topologically.

```python
import keras
from dl_techniques.models.memory.som.model import SOMModel

# 1. Load and prepare MNIST data
(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
# Flatten the 28x28 images into 784-dimensional vectors
x_train_flat = x_train.reshape((len(x_train), -1))

# 2. Create and train the SOM
som = SOMModel(
    map_size=(20, 20),
    input_dim=x_train_flat.shape[1], # 784
    sigma=8.0
)
som.train_som(x_train_flat, epochs=10, batch_size=256)

# 3. Visualize the result
# This will show a 20x20 grid where each cell is a learned "prototype" of a digit.
# You will see similar digits (like '1's and '7's) clustered in nearby regions.
som.visualize_som_grid()
```

---

## 9. Advanced Usage: Classification

After unsupervised training, you can "calibrate" the map by assigning a class label to each neuron. This turns the SOM into a classifier.

```python
# (Continuing from the MNIST example)

# 1. Fit class prototypes
# This finds the most activated neuron for each digit class (0-9).
print("Fitting class prototypes...")
som.fit_class_prototypes(x_train_flat, y_train)
print("Prototypes fitted!")

# 2. Predict on new data
(x_test, y_test), _ = keras.datasets.mnist.load_data()
x_test_flat = (x_test.astype('float32') / 255.).reshape((len(x_test), -1))
predictions = som.predict_classes(x_test_flat)

# 3. Evaluate accuracy
accuracy = np.mean(predictions == y_test)
print(f"Classification Accuracy: {accuracy * 100:.2f}%")
```

---

## 10. Visualization Suite

This `SOMModel` includes several powerful visualization methods to help you understand the learned memory structure.

-   **`visualize_som_grid()`**: Shows the prototype vectors of each neuron. For image data, it displays a grid of prototype images.
-   **`visualize_class_distribution()`**: Creates a scatter plot on the SOM grid, showing where data points from different classes are mapped. This reveals the class topology.
-   **`visualize_u_matrix()`**: Displays the "Unified Distance Matrix," which highlights the distances between adjacent neurons. High values indicate cluster boundaries, while low values indicate regions of similar data.
-   **`visualize_hit_histogram()`**: A heatmap showing how many data samples were mapped to each neuron. This reveals the density of the data distribution on the map.
-   **`visualize_memory_recall()`**: Takes a test sample, finds its BMU, and shows the learned prototype alongside the most similar training samples that map to the same region. This is a direct demonstration of associative recall.

---

## 11. Training and Best Practices

### The Two-Phase Training Approach

-   For best results, SOMs are often trained in two phases:
    1.  **Rough Organization Phase**: Use a large initial `sigma` and a higher learning rate for a few epochs. This allows the map to quickly form a coarse topological structure.
    2.  **Fine-Tuning Phase**: Use a smaller `sigma` and a lower learning rate for more epochs. This allows the neurons to fine-tune their positions and better represent the details of the data distribution.
-   This implementation handles the decay of `sigma` and learning rate automatically over the specified epochs.

### Choosing Map Size

-   The map size determines the model's capacity. A map that is too small may merge distinct clusters. A map that is too large may overfit and show too much detail.
-   A common heuristic is to set the number of neurons (`height * width`) to be around `5 * sqrt(N)`, where `N` is the number of training samples, but this is highly data-dependent. Experimentation is key.

---

## 12. Serialization & Deployment

The `SOMModel` is fully serializable using Keras 3's modern `.keras` format, as all custom components are registered with Keras.

### Saving and Loading

```python
# Assume `som` is a trained model from the examples
som.save('my_som_model.keras')

# Load the model in a new session
loaded_som = keras.models.load_model('my_som_model.keras')
print("✅ SOM model loaded successfully!")

# The loaded model retains its weights and configuration
assert loaded_som.map_size == som.map_size
```

---

## 13. Troubleshooting & FAQs

**Issue 1: My SOM grid looks random and disorganized after training.**

-   **Cause**: The training parameters might be off. This can happen if the initial `sigma` is too small, the learning rate is too low, or the number of epochs is insufficient.
-   **Solution**: Try increasing the initial `sigma` to be a significant fraction of the map size (e.g., `max(map_size)/2`). Increase the number of epochs. Ensure your data is properly normalized (e.g., to the [0, 1] range).

### Frequently Asked Questions

**Q: Is the SOM a deep learning model?**

A: Not in the traditional sense. It's a shallow network with a single layer of neurons. Its power comes from the competitive learning algorithm and topological organization, not from deep hierarchical feature extraction.

**Q: Can I use `model.fit()` to train the SOM?**

A: No. The SOM's training algorithm is fundamentally different from the backpropagation used by typical Keras models. It requires a custom training loop that updates weights based on neighborhood distance, not gradients. The provided `train_som()` method implements this specific algorithm.

**Q: How is the "Best Matching Unit" (BMU) found?**

A: For a given input vector, the model calculates the Euclidean distance between that vector and the weight vector of *every* neuron on the grid. The neuron with the smallest distance is declared the BMU.

---

## 14. Technical Details

### Neighborhood Functions

-   The `neighborhood_function` determines how much influence the BMU has on its neighbors.
    -   **`gaussian`**: The influence decays smoothly with distance from the BMU, following a Gaussian curve. This is generally preferred as it leads to smoother maps.
    -   **`bubble`**: All neurons within the `sigma` radius receive the same update, and all neurons outside receive none. It's a simpler but less smooth alternative.

### Parameter Decay

-   The learning rate (`alpha`) and neighborhood radius (`sigma`) must decrease over time for the map to stabilize. This implementation uses an exponential decay function:
    -   `alpha(t) = initial_learning_rate * exp(-t / T)`
    -   `sigma(t) = sigma * exp(-t / T)`
    -   Where `t` is the current iteration and `T` is a time constant related to the total number of iterations.

---

## 15. Citation

If you use this model in your research, please consider citing the original work by Teuvo Kohonen:

-   **Original Paper**:
    ```bibtex
    @article{kohonen1990self,
      title={The self-organizing map},
      author={Kohonen, Teuvo},
      journal={Proceedings of the IEEE},
      volume={78},
      number={9},
      pages={1464--1480},
      year={1990},
      publisher={IEEE}
    }
    ```