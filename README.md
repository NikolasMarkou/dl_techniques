# Deep Learning Techniques

<p align="center">
  <img src="./imgs/minimalist-2d-logo-with-a-left-to-right-_sSVDZkeKR4C_eovommSCFQ_mJekanaZTB2Nbe5dBKOnPQ.png" alt="Deep Learning Techniques" width="512" height="512">
</p>
A versatile, modern library providing **advanced deep learning layers, initializers, constraints,** and **analysis tools** for Keras/TensorFlow. 

Whether you're researching new architectures, experimenting with custom constraints, or analyzing your network activations, **DL Techniques** brings you cutting-edge components to accelerate and enhance your workflows.

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

## Feature Highlights

### 1. Advanced Normalization Techniques

The library implements cutting-edge normalization techniques that improve training stability and model performance:

```python
from dl_techniques.layers.rms_norm_spherical_bound import RMSNormSphericalBound
from dl_techniques.layers.global_response_norm import GlobalResponseNorm

# Apply spherical bound RMS normalization
x = RMSNormSphericalBound()(inputs)

# Add global response normalization
x = GlobalResponseNorm()(x)
```

### 2. Attention Mechanisms

Implement state-of-the-art attention mechanisms for both vision and sequence models:

```python
from dl_techniques.layers.hopfield_attention import HopfieldAttention
from dl_techniques.layers.convolutional_block_attention_module import CBAM

# Add Hopfield attention
x = HopfieldAttention(heads=8)(inputs)

# Apply CBAM
x = CBAM(reduction_ratio=16)(x)
```

### 3. Capsule Networks

Create advanced capsule networks with dynamic routing:

```python
from dl_techniques.layers.capsules import PrimaryCapsule, RoutingCapsule

# Create primary capsules
primary_caps = PrimaryCapsule(
    num_capsules=32,
    dim_capsules=8,
    kernel_size=9,
    strides=2
)(conv_features)

# Add routing capsules
digit_caps = RoutingCapsule(
    num_capsules=10,
    dim_capsules=16,
    routing_iterations=3
)(primary_caps)
```

### 4. Complex-Valued Networks

Build networks that operate on complex numbers:

```python
from dl_techniques.layers.complex_layers import ComplexConv2D, ComplexDense

# Create complex-valued network
x = ComplexConv2D(filters=32, kernel_size=3)(inputs)
x = ComplexDense(units=64)(x)
```

## Usage Examples

### Example 1: Custom Vision Transformer with Hierarchical MLP Stem

```python
from dl_techniques.layers.hierarchical_vision_transformers import HierarchicalMlpStem
from dl_techniques.layers.vision_transformer import TransformerEncoder

# Create input
inputs = keras.layers.Input(shape=(224, 224, 3))

# Apply hierarchical MLP stem
x = HierarchicalMlpStem(
    embed_dim=768,
    patch_size=16
)(inputs)

# Add transformer encoder
x = TransformerEncoder(
    num_heads=12,
    mlp_dim=3072,
    num_layers=12
)(x)

# Output layer
outputs = keras.layers.Dense(1000, activation='softmax')(x)

# Create model
model = keras.Model(inputs, outputs)
```

### Example 2: Advanced Regularization

```python
from dl_techniques.regularizers.srip import SRIPRegularizer
from dl_techniques.regularizers.soft_orthogonal import SoftOrthogonalConstraintRegularizer

# Create layer with SRIP regularization
dense = keras.layers.Dense(
    units=512,
    kernel_regularizer=SRIPRegularizer(beta=1e-3)
)

# Create layer with soft orthogonal regularization
conv = keras.layers.Conv2D(
    filters=64,
    kernel_size=3,
    kernel_regularizer=SoftOrthogonalConstraintRegularizer(beta=1e-2)
)
```

### Example 3: Activation Analysis

```python
from dl_techniques.analysis.activation_activity import ActivationDistributionAnalyzer

# Create analyzer
analyzer = ActivationDistributionAnalyzer(model)

# Analyze activation distributions
stats = analyzer.compute_activation_stats(x_test)

# Visualize activations
analyzer.plot_activation_distributions(
    x_test,
    save_path='./activation_plots/'
)
```

### Example 4: Any-Loss Classification

```python
from dl_techniques.losses.any_loss import F1Loss, BalancedAccuracyLoss

# Train with F1 optimization
model.compile(
    optimizer='adam',
    loss=F1Loss(),
    metrics=['accuracy']
)

# Or train with balanced accuracy optimization
model.compile(
    optimizer='adam',
    loss=BalancedAccuracyLoss(),
    metrics=['accuracy']
)
```

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

## Documentation

For detailed documentation on each component, please refer to the `docs/` directory:

- [AnyLoss: Classification Metrics as Loss Functions](docs/anyloss_classification_metrics_loss_functions.md)
- [Custom RMS Normalization](docs/custom_rms_norm.md)
- [Hierarchical Vision Transformers](docs/hierarchical_vision_transformers.md)
- [Input Normalization Best Practices](docs/input_normalization.md)
- [Mixture Density Networks (MDNs)](docs/mdn.md)

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

## License

GNU General Public License v3.0 - see [LICENSE](LICENSE) for details.

## References

- **Bishop, C.M.** (1994). _Mixture Density Networks._
- **Sabour, S. et al.** (2017). _Dynamic Routing Between Capsules._
- **Trabelsi, C. et al.** (2018). _Deep Complex Networks._
- **Woo, S. et al.** (2018). _CBAM: Convolutional Block Attention Module._
- **Liu, Z. et al.** (2022). _A ConvNet for the 2020s (ConvNeXt)._
- **Ramsauer, H. et al.** (2020). _Hopfield Networks is All You Need._
- **Touvron et al.** "Three things everyone should know about Vision Transformers"
- **Doheon Han, Nuno Moniz, and Nitesh V Chawla** "AnyLoss: Transforming Classification Metrics into Loss Functions"
- **Arjovsky et al.** (2016). "Unitary Evolution Recurrent Neural Networks"
- **Hinton et al.** Work on capsule networks and part-whole relationships
- **Charles Martin** Original WeightWatcher implementation for neural network analysis
- **Graves, A.** (2013). _Generating Sequences with Recurrent Neural Networks._
- **Bishop, C.M.** (2006). _Pattern Recognition and Machine Learning._
- **Doheon Han, Nuno Moniz, and Nitesh V Chawla** "AnyLoss: Transforming Classification Metrics into Loss Functions"
- **Touvron et al.** "Three things everyone should know about Vision Transformers"
- **Glorot, X. and Bengio, Y.** (2010). _Understanding the difficulty of training deep feedforward neural networks._
- **He, K. et al.** (2015). _Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification._
- **Huang, G. et al.** (2016). _Deep Networks with Stochastic Depth._
- **Xie, S. et al.** (2017). _Aggregated Residual Transformations for Deep Neural Networks._
- **Ba, J.L., Kiros, J.R., and Hinton, G.E.** (2016). _Layer Normalization._
- **Wu, Y. and He, K.** (2018). _Group Normalization._
- **Misra, D.** (2019). _Mish: A Self Regularized Non-Monotonic Neural Activation Function._
- **Zhang, H. et al.** (2018). _SRIP: Matrix-Based Spectral Regularization for Fine-Tuning Deep Neural Networks._
- **Yu, F. and Koltun, V.** (2016). _Multi-Scale Context Aggregation by Dilated Convolutions._
- **Yang, G. et al.** (2020). _Soft Orthogonality Constraints in Neural Networks._
- **Dosovitskiy, A. et al.** (2021). _An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale._