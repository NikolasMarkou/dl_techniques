# Deep Learning Techniques

<p align="center">
  <img src="./imgs/minimalist-2d-logo-with-a-left-to-right-_sSVDZkeKR4C_eovommSCFQ_mJekanaZTB2Nbe5dBKOnPQ.png" alt="Deep Learning Techniques" width="512" height="512">
</p>

A versatile, modern library providing **advanced deep learning layers, initializers, constraints,** and **analysis tools** for Keras/TensorFlow.  
Use it to experiment with **cutting-edge network components**, build novel architectures, or perform in-depth **model analyses** – all while leveraging a robust test suite and real-world example experiments.

---

## Table of Contents

1. [Key Features](#key-features)  
2. [Installation](#installation)  
3. [Usage Examples](#usage-examples)  
4. [Project Structure](#project-structure)  
5. [Documentation](#documentation)  
6. [Contributing](#contributing)  
7. [License](#license)  

---

## Key Features

- **Wide Range of Layers**  
  - **Normalization**: RMS Norm, Global Response Norm, Shell Clamp, etc.  
  - **Attention Mechanisms**: Hopfield Attention, Non-Local Attention, Differential Transformer  
  - **Vision**: Convolutional Block Attention Module (CBAM), CLAHE, ConvNeXt blocks, hierarchical Vision Transformers  
  - **Miscellaneous**: Capsule Networks, RBF layers, advanced activation functions (Mish, GLU), Shearlet transforms  

- **Regularization & Optimization**  
  - **Regularizers**: SRIP, Soft Orthogonal, Tri-State Preferences  
  - **Optimizers**: Warmup schedules, selective gradients, deep supervision  
  - **Stochastic Techniques**: Stochastic Depth, Band-limited RMS Norm  

- **Advanced Analysis & Visualization**  
  - **Activation Distribution Analysis**: Inspect layer-by-layer activation stats  
  - **WeightWatcher Integration**: Assess weight matrix properties for generalization analysis  
  - **Logging Infrastructure**: In-built logger for consistent experiment tracking  

- **Full Model Implementations**  
  - **Depth Anything**, **CoshNet**, **MobileNet v4**, **Mixture Density Network (MDN) Model**, **CapsNet**, and more

---

## Installation

1. **Clone** this repository:

   ```bash
   git clone https://github.com/your-username/dl_techniques.git
   cd dl_techniques
   ```

2. **Install** requirements:

   ```bash
   pip install -r requirements.txt
   ```

3. **Install** the library in development mode:

   ```bash
   pip install -e .
   ```

**Dependencies** (listed in `requirements.txt`):
- Python ≥ 3.11
- TensorFlow ≥ 2.18.0  
- Keras ≥ 3.8.0  
- NumPy, SciPy, Matplotlib, scikit-learn, Pandas, PyTest, etc.

---

## Usage Examples

Here’s a quick example of how to leverage some of the library’s components in Keras/TensorFlow:

```python
import tensorflow as tf
from dl_techniques.layers.global_response_norm import GlobalResponseNorm
from dl_techniques.layers.adaptive_softmax import AdaptiveTemperatureSoftmax

# Simple model using advanced layers
inputs = tf.keras.Input(shape=(64,))
x = tf.keras.layers.Dense(128, activation='relu')(inputs)
x = GlobalResponseNorm()(x)  # apply GRN
x = AdaptiveTemperatureSoftmax()(x)
model = tf.keras.Model(inputs, x)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Example training
# model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32)
```

**Additional Capabilities**  
- **Custom Losses**: Try `AnyLoss` classes (e.g., F1Loss, BalancedAccuracyLoss) for direct metric optimization.  
- **MDN**: Mixture Density Networks for multi-modal output predictions.  
- **Analysis Tools**: Activation distribution analysis, WeightWatcher-based SVD smoothing, etc.

---

## Project Structure

```
dl_techniques/
├─ README.md                     # High-level overview and usage info
├─ LICENSE                      # GNU GPL v3
├─ pyproject.toml.py            # Project configuration (for tools like pytest)
├─ requirements.txt             # Dependencies
├─ setup.py                     # Package setup file
├─ docs/                        # Additional documentation & reference materials
│  ├─ anyloss_classification_metrics_loss_functions.md
│  ├─ custom_rms_norm.md
│  ├─ hierarchical_vision_transformers.md
│  ├─ input_normalization.md
│  └─ mdn.md
├─ imgs/                        # Images & graphics for docs
├─ src/
│  └─ dl_techniques/
│     ├─ analysis/              # Activation distribution, WeightWatcher, other analysis
│     ├─ constraints/           # Constraint layers
│     ├─ initializers/          # Specialized weight initializers
│     ├─ layers/                # Custom Keras layers (attention, capsules, transforms, etc.)
│     ├─ losses/                # Advanced loss functions (e.g., AnyLoss)
│     ├─ models/                # Complete model implementations
│     ├─ optimization/          # Warmup schedules, gradient manipulation
│     ├─ regularizers/          # Soft orthogonal, tri-state preference, SRIP
│     ├─ utils/                 # Helpers for logging, datasets, training
│     └─ visualization/         # Visualization tools
├─ experiments/                 # Example experiments & scripts
└─ tests/                       # Comprehensive test suite
   ├─ test_layers/
   ├─ test_losses/
   ├─ test_models/
   ├─ test_regularizers/
   └─ test_utils/
```

**Highlights**:
- **`src/dl_techniques/layers`**: Contains all custom neural network layers.  
- **`docs/`**: Reference materials and technique explanations.  
- **`experiments/`**: Real-world usage demos.  
- **`tests/`**: PyTest-based testing for reliability.

---

## Documentation

Detailed documentation resides in the `docs/` folder and includes:
1. **Advanced Loss Functions**: [AnyLoss Overview](docs/anyloss_classification_metrics_loss_functions.md)  
2. **Custom RMS Norm**: [RMS Normalization Variants](docs/custom_rms_norm.md)  
3. **Hierarchical Vision Transformers**: [Implementation Details](docs/hierarchical_vision_transformers.md)  
4. **Input Normalization**: [Best Practices](docs/input_normalization.md)  
5. **Mixture Density Networks**: [MDN Theory & Usage](docs/mdn.md)

---

## Contributing

1. **Fork** and **Clone** the repository  
2. **Create a branch** for your feature or bugfix  
3. **Write tests** covering your changes (see `tests/` folder)  
4. **Ensure** all pre-commit hooks pass (`pylint`, `pytest`, etc.)  
5. **Open a Pull Request** describing your changes

**Coding Standards**:
- Follow [PEP 8](https://peps.python.org/pep-0008/)  
- Use **type hints**  
- Include **Sphinx docstrings** for new layers, models, or utilities  
- Write comprehensive tests in `tests/`

---

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).  
See the [LICENSE file](LICENSE) for details.

---

**Happy experimenting with DL Techniques!** If you have questions, bug reports, or feature requests, feel free to open an issue or start a discussion.

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