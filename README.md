# DL Techniques

A versatile, modern library providing **advanced deep learning layers, initializers, constraints,** and **analysis tools** for Keras/TensorFlow. Whether you're researching new architectures, experimenting with custom constraints, or analyzing your network activations, **DL Techniques** brings you cutting-edge components to accelerate and enhance your workflows.

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

- **Advanced Layers**  
  Includes custom layers like:
  - **Gated MLPs**, **Capsule Networks**, **Complex Convolutions**  
  - **Adaptive Temperature Softmax**, **Depthwise Haar Wavelet**, **BatchConditionalOutput**  
  - **Convolutional Block Attention Modules (CBAM)**  
  - **Kolmogorov-Arnold Networks (KAN)**  
  - **Differential Transformers**, and more...

- **Custom Initializers**  
  - **HaarWaveletInitializer** for wavelet decomposition  
  - **OrthonormalInitializer** for stable centroid initialization  
  - **Rayleigh-based** magnitude-phase initializers for complex layers

- **Constraints & Regularizers**  
  - **ValueRangeConstraint** for bounding weights within a min-max range  
  - **SoftOrthogonalConstraintRegularizer** to gently encourage orthogonal transformations

- **Analysis Tools**  
  - **ActivationDistributionAnalyzer** for analyzing and visualizing activation distributions  
  - **Plotting** utilities for histograms, heatmaps, and more  
  - **Activation stats** (mean, std, sparsity, zero-ratio, percentiles)

- **Differentiable Clustering / K-Means**  
  - **KMeansLayer** for end-to-end trainable clustering

- **Out-of-Distribution & Other Utilities**  
  - **MaxLogitNorm**, **DecoupledMaxLogit** for OOD detection  
  - **Global Response Normalization**, **CLAHE**, **GaussianFilter** layers

---

## Installation

1. **Clone** this repository or include it as a submodule:
   ```bash
   git clone https://github.com/your-username/dl_techniques.git
   ```
2. **Install** using `pip` (in editable mode if you want to develop):
   ```bash
   pip install -e /path/to/dl_techniques
   ```
3. Make sure the dependencies in `setup.py` are satisfied:
   - `keras`
   - `tensorflow`
   - `numpy`
   - `pytest`
   - `jupyter`  
   *(For development, ensure `pylint` is also installed.)*

---

## Quick Start

```python
import tensorflow as tf
from dl_techniques.layers.adaptive_softmax import AdaptiveTemperatureSoftmax

# Sample usage in a Keras model
inputs = tf.keras.Input(shape=(64,))
x = tf.keras.layers.Dense(128, activation='relu')(inputs)
x = AdaptiveTemperatureSoftmax()(x)  # Adapts temperature based on entropy
model = tf.keras.Model(inputs, x)
model.compile(optimizer='adam', loss='mse')

print(model.summary())
```

That's it! You can now experiment with other advanced layers, constraints, or initializers from this library.

---

## Project Structure

```
dl_techniques/
├─ analysis/
│  └─ activation_activity.py      # ActivationDistributionAnalyzer & utilities
├─ constraints/
│  └─ value_range_constraint.py   # ValueRangeConstraint class
├─ initializers/
│  ├─ haar_wavelet_initializer.py # HaarWaveletInitializer & related
│  └─ orthonormal_initializer.py  # OrthonormalInitializer
├─ layers/
│  ├─ adaptive_softmax.py         # AdaptiveTemperatureSoftmax
│  ├─ batch_conditional_layer.py  # BatchConditionalOutputLayer
│  ├─ canny.py                    # Edge detection (Canny)
│  ├─ capsules.py                 # CapsNet layers, margin_loss, etc.
│  ├─ clahe.py                    # Contrast Limited Adaptive Histogram Eq
│  ├─ complex_layers.py           # ComplexConv2D, ComplexDense, etc.
│  ├─ conditional_output.py       # Another conditional output layer
│  ├─ conv2d_builder.py           # Helper wrapper for convolution ops
│  ├─ convnext_block.py           # ConvNext style block
│  ├─ ...
│  └─ gaussian_filter.py          # GaussianFilter layer
├─ ...
└─ setup.py
```

- **`analysis`**  
  Tools for analyzing neural network behavior, e.g., activation distributions.

- **`constraints`**  
  Custom constraints for bounding or clipping weights.

- **`initializers`**  
  Includes specialized initializers like Haar wavelets, orthonormal seeds, etc.

- **`layers`**  
  Advanced custom layers—Capsules, Gated MLPs, BatchConditional, attention modules, and more.

---

## Feature Highlights

1. **ActivationDistributionAnalyzer**
   - Quickly compute mean, std, sparsity, ratio of zeros
   - Create distribution plots & heatmaps
   - Save results for offline analysis

2. **ValueRangeConstraint**
   - Constrain layer weights to a `[min_value, max_value]` range
   - Prevent exploding or vanishing weights
   - Gradient clipping option for stability

3. **HaarWaveletInitializer**
   - Initialize convolution weights as Haar wavelet filters
   - Perfect for wavelet-based transforms or compressed representations

4. **Complex Layers**
   - **ComplexConv2D**, **ComplexDense**, **ComplexReLU**
   - Robust initialization (Rayleigh magnitude, uniform phase)
   - Perfect for signal processing, wave-based PDEs, or general complex domain tasks

5. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
   - Enhance local contrast in images
   - Trainable mapping kernel for further fine-tuning
   - Straightforward drop-in for image preprocessing layers

6. **Differential Transformers**
   - Weighted attention with entropic constraints
   - Scalability with large contexts
   - Out-of-distribution robust design

7. **KMeansLayer** (Differentiable K-means)
   - End-to-end clustering integrated into your network
   - Repulsion forces to avoid centroid collapse
   - Useful for image segmentation, attention grouping, or unsupervised tasks

---

## Usage Examples

1. **Analyze Model Activations**

```python
from dl_techniques.analysis.activation_activity import ActivationDistributionAnalyzer
from pathlib import Path

analyzer = ActivationDistributionAnalyzer(model)
stats = analyzer.compute_activation_stats(x_test)
analyzer.plot_activation_distributions(x_test, save_path=Path("./analysis_results"))
```
Generates distribution plots, heatmaps, and text-based stats.

2. **Use a Custom Weight Constraint**

```python
from dl_techniques.constraints.value_range_constraint import ValueRangeConstraint

# Constrain weights between [0.01, 1.0] with gradient clipping
constraint = ValueRangeConstraint(min_value=0.01, max_value=1.0, clip_gradients=True)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, kernel_constraint=constraint),
    ...
])
```

3. **Wavelet-Based Depthwise Convolution**

```python
from dl_techniques.initializers.haar_wavelet_initializer import create_haar_depthwise_conv2d

layer = create_haar_depthwise_conv2d(
    input_shape=(128, 128, 3),
    channel_multiplier=4,
    scale=1.0,
    trainable=False
)
```
Implements a Haar wavelet transform for input images.

4. **Complex-Valued Layers**

```python
from dl_techniques.layers.complex_layers import ComplexConv2D

inputs = tf.keras.Input(shape=(128, 128, 2))  # Real and Imag channels
x = ComplexConv2D(filters=16, kernel_size=3)(inputs)
model = tf.keras.Model(inputs, x)
```

---

## Testing

1. **Run Tests**  
   ```bash
   pytest
   ```
2. **Lint & Style Checks** (optional, requires `pylint`)  
   ```bash
   pylint src/dl_techniques
   ```

We use `pytest` for unit tests. The default test suite covers constraints, initializers, layers, and analysis modules.

---

## Contributing

Contributions are welcome! If you want to add a feature or fix a bug:

1. **Fork & Clone** this repository
2. **Create a Feature Branch** (`git checkout -b feature/new-stuff`)
3. **Add Tests** for your changes
4. **Open a Pull Request** on the main repo

Please ensure your code passes `pytest` and meets style guidelines.

---

## License

[MIT License](LICENSE) © 2025 Nikolas Markou

---

## References

- **Bishop, C.M.** (1994). _Mixture Density Networks._  
- **Sabour, S. et al.** (2017). _Dynamic Routing Between Capsules._  
- **Trabelsi, C. et al.** (2018). _Deep Complex Networks._  
- **Woo, S. et al.** (2018). _CBAM: Convolutional Block Attention Module._  
- **Liu, Z. et al.** (2022). _A ConvNet for the 2020s (ConvNeXt)._  
- **Ramsauer, H. et al.** (2020). _Hopfield Networks is All You Need._  
- **Kolmogorov, A.** (1957). _On the representation of continuous functions._  
- **Arnold, V.** (1963). _On functions of three variables._  

For a full list of references and extended documentation, see the docstrings in `analysis/`, `constraints/`, `initializers/`, and `layers/`. 

Enjoy exploring the **DL Techniques** library for your next advanced deep learning project!