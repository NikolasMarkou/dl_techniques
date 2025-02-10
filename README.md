# DL Techniques

A versatile, modern library providing **advanced deep learning layers, initializers, constraints,** and **analysis tools** for Keras/TensorFlow. Whether you're researching new architectures, experimenting with custom constraints, or analyzing your network activations, **DL Techniques** brings you cutting-edge components to accelerate and enhance your workflows.

Copyright (C) 2025 

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

---

## Table of Contents
1. [Key Features](#key-features)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Project Structure](#project-structure)
5. [Feature Highlights](#feature-highlights)
6. [Usage Examples](#usage-examples)
7. [Testing](#testing)
8. [Contributing](#contributing)
9. [License](#license)
10. [References](#references)

---

## Key Features

- **Advanced Neural Network Layers**
  - **Attention Mechanisms**: Hopfield Attention, Non-Local Attention, Differential Transformers
  - **Normalization**: RMS Norm, Global Response Norm, Shell Clamp, Spherical Bound
  - **Vision**: CBAM (Convolutional Block Attention Module), ConvNeXt, CLAHE
  - **Complex Networks**: Complex Conv2D, Complex Dense
  - **Modern Architectures**: Capsule Networks, Gated MLPs, KAN (Kolmogorov-Arnold Networks)
  - **Special Layers**: RBF, Mish Activation, Canny Edge Detection, Shearlet Transform

- **Regularization & Optimization**
  - **Advanced Regularizers**: SRIP, Soft Orthogonal, Binary/Tri-State Preferences
  - **Training Utilities**: Deep Supervision, Warmup Scheduling
  - **Stochastic Techniques**: Stochastic Depth, Selective Gradients

- **Analysis & Visualization**
  - **Model Analysis**: Weight Analysis, Activation Distribution
  - **Visualization**: Advanced plotting utilities
  - **Logging**: Custom logging infrastructure

- **Complete Model Implementations**
  - Depth Anything
  - CoshNet
  - MobileNet v4
  - CapsNet

---

## Installation

1. **Clone** this repository:
   ```bash
   git clone https://github.com/your-username/dl_techniques.git
   ```

2. **Install** dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install** the package (development mode):
   ```bash
   pip install -e .
   ```

**Requirements**:
- Python ≥ 3.11
- Keras ≥ 3.8.0
- TensorFlow ≥ 2.18.0
- PyTest (for testing)
- Additional dependencies in `requirements.txt`

---

## Quick Start

```python
import tensorflow as tf
from dl_techniques.layers.adaptive_softmax import AdaptiveTemperatureSoftmax
from dl_techniques.layers.global_response_norm import GlobalResponseNorm

# Create a model with advanced layers
inputs = tf.keras.Input(shape=(64,))
x = tf.keras.layers.Dense(128)(inputs)
x = GlobalResponseNorm()(x)
x = AdaptiveTemperatureSoftmax()(x)
model = tf.keras.Model(inputs, x)

# Use provided training utilities
from dl_techniques.optimization.warmup_schedule import WarmupSchedule
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## Project Structure

```
dl_techniques/
├─ src/
│  └─ dl_techniques/
│     ├─ analysis/            # Analysis tools
│     ├─ constraints/         # Custom constraints
│     ├─ initializers/        # Weight initializers
│     ├─ layers/             # Neural network layers
│     ├─ losses/             # Custom loss functions
│     ├─ models/             # Complete model implementations
│     ├─ optimization/       # Training optimizations
│     ├─ regularizers/       # Custom regularizers
│     ├─ utils/              # Utility functions
│     └─ visualization/      # Visualization tools
├─ tests/                    # Comprehensive test suite
│  ├─ test_layers/
│  ├─ test_losses/
│  ├─ test_regularizers/
│  └─ test_utils/
├─ docs/                     # Documentation
├─ experiments/              # Example experiments
├─ pyproject.toml           # Project configuration
├─ requirements.txt         # Dependencies
├─ setup.py                # Package setup
└─ LICENSE                 # GNU GPL v3
```

### Key Components

- **layers/**: Advanced neural network layers including:
  - Attention mechanisms (Hopfield, Non-Local, Transformer)
  - Normalization layers (RMS, GRN, Shell Clamp)
  - Vision layers (CBAM, ConvNeXt, CLAHE)
  - Complex network layers
  - Modern architectures (Capsules, Gated MLP, KAN)

- **models/**: Complete model implementations:
  - Depth Anything
  - CoshNet
  - MobileNet v4
  - CapsNet

- **experiments/**: Real-world usage examples:
  - Band RMS experiments
  - KMeans clustering
  - RMS normalization studies
  - Activation function analyses

---

## Feature Highlights

1. **Advanced Normalization Techniques**
   - RMS Norm with Spherical Bounds
   - Global Response Normalization
   - Shell Clamping
   - Conditional Batch Normalization

2. **Attention Mechanisms**
   - Hopfield Attention
   - Non-Local Attention
   - Differential Transformers
   - Convolutional Block Attention

3. **Modern Architecture Components**
   - Capsule Networks
   - Kolmogorov-Arnold Networks
   - Gated MLPs
   - ConvNeXt Blocks

4. **Specialized Layers**
   - Shearlet Transform
   - Canny Edge Detection
   - Complex-valued Operations
   - RBF Networks

5. **Training Optimizations**
   - Deep Supervision
   - Warmup Scheduling
   - Stochastic Depth
   - Selective Gradients

---

## Usage Examples

1. **Using Advanced Normalization**

```python
from dl_techniques.layers.rms_norm_spherical_bound import RMSNormSphericalBound
from dl_techniques.layers.global_response_norm import GlobalResponseNorm

# Apply spherical bound RMS normalization
x = RMSNormSphericalBound()(inputs)

# Add global response normalization
x = GlobalResponseNorm()(x)
```

2. **Implementing Attention**

```python
from dl_techniques.layers.hopfield_attention import HopfieldAttention
from dl_techniques.layers.convolutional_block_attention_module import CBAM

# Add Hopfield attention
x = HopfieldAttention(heads=8)(inputs)

# Apply CBAM
x = CBAM(reduction_ratio=16)(x)
```

3. **Using Complex Layers**

```python
from dl_techniques.layers.complex_layers import ComplexConv2D, ComplexDense

# Create complex-valued network
x = ComplexConv2D(filters=32, kernel_size=3)(inputs)
x = ComplexDense(units=64)(x)
```

4. **Advanced Training**

```python
from dl_techniques.optimization.deep_supervision import DeepSupervision
from dl_techniques.optimization.warmup_schedule import WarmupSchedule

# Configure training
scheduler = WarmupSchedule(warmup_steps=1000)
supervisor = DeepSupervision(depth=3)
```

---

## Testing

Run the comprehensive test suite:

```bash
pytest tests/
```

Test coverage includes:
- Layer functionality
- Loss computations
- Regularizer behavior
- Utility functions
- End-to-end model tests

---

## Contributing

1. Fork & clone the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

**Requirements**:
- All tests must pass
- Type hints required
- Sphinx docstrings required
- Follow layer normalization order
- Proper kernel initialization
- Error handling
- Pre-commit hooks (see `.pre-commit-config.yaml`)

---

## License

GNU General Public License v3.0 - see [LICENSE](LICENSE) for details.

---

## References

- **Bishop, C.M.** (1994). _Mixture Density Networks._
- **Sabour, S. et al.** (2017). _Dynamic Routing Between Capsules._
- **Trabelsi, C. et al.** (2018). _Deep Complex Networks._
- **Woo, S. et al.** (2018). _CBAM: Convolutional Block Attention Module._
- **Liu, Z. et al.** (2022). _A ConvNet for the 2020s (ConvNeXt)._
- **Ramsauer, H. et al.** (2020). _Hopfield Networks is All You Need._

For extended documentation, see the docstrings and `docs/` directory.