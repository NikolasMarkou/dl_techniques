# DL-Techniques Library Documentation

A comprehensive deep learning techniques library built on Keras 3.x with TensorFlow 2.18.0 backend.

## Library Overview

- **584** modules
- **806** public classes
- **383** Keras layers
- **3274** public functions
- **11** component categories

## Component Categories

The library is organized into the following categories:

### [Models](categories/models.md) (183 modules)
Complete model architectures ready for training

### [Layers](categories/layers.md) (252 modules)
Individual neural network layers and building blocks

### [Losses](categories/losses.md) (27 modules)
Loss functions for different tasks

### [Metrics](categories/metrics.md) (8 modules)
Evaluation metrics

### [Initializers](categories/initializers.md) (5 modules)
Weight initialization strategies

### [Regularizers](categories/regularizers.md) (7 modules)
Regularization techniques

### [Constraints](categories/constraints.md) (2 modules)
Weight constraints

### [Optimization](categories/optimization.md) (10 modules)
Training optimization utilities

### [Utils](categories/utils.md) (59 modules)
Utility functions and helpers

### [Analyzer](categories/analyzer.md) (24 modules)
Model analysis and evaluation tools

### [Visualization](categories/visualization.md) (7 modules)
Visualization utilities

## Documentation Navigation
- 🔍 [Component Reference](component_reference.md) - All components organized by type
- 📁 [Category Pages](categories/) - Components organized by purpose
- 📊 [JSON Index](module_index.json) - Machine-readable library index

## Getting Started

### Installation
```python
# Install dependencies
pip install keras==3.8.0 tensorflow==2.18.0
```

### Basic Usage
```python
import keras
from dl_techniques.models import ConvNeXtV2
from dl_techniques.layers import SwinTransformerBlock
from dl_techniques.losses import ClipContrastiveLoss

# Create a model
model = ConvNeXtV2(num_classes=10)

# Use custom layers
transformer_block = SwinTransformerBlock(dim=96, num_heads=3)
```