# Deep Learning Techniques

<p align="center">
  <img alt="License: GPL-3.0" src="https://img.shields.io/badge/License-GPL_v3-blue.svg">
  <img alt="Python Version" src="https://img.shields.io/badge/python-3.11+-blue.svg">
  <img alt="TensorFlow Version" src="https://img.shields.io/badge/TensorFlow-2.18-orange">
  <img alt="Keras Version" src="https://img.shields.io/badge/Keras-3.x-red">
</p>

<p align="center">
  <img src="./imgs/minimalist-2d-logo-with-a-left-to-right-_sSVDZkeKR4C_eovommSCFQ_mJekanaZTB2Nbe5dBKOnPQ.png" alt="Deep Learning Techniques Logo" width="350">
</p>

**dl_techniques** is a research-focused Python library for Keras 3 and TensorFlow, providing a rich collection of cutting-edge deep learning components. This toolkit is designed for researchers and advanced practitioners to experiment with novel layers, build state-of-the-art architectures, and gain deeper insights into their models through advanced analysis tools.

The library is a curated collection of modern, SOTA, or otherwise interesting implementations, ranging from attention mechanisms and normalization layers to full model architectures and information-theoretic loss functions.

---

## Table of Contents
1.  [Key Features](#key-features)
2.  [Installation](#installation)
3.  [Usage Examples](#usage-examples)
4.  [Project Structure](#project-structure)
5.  [Documentation & Experiments](#documentation--experiments)
6.  [Contributing](#contributing)
7.  [License](#license)
8.  [References](#references)

---

## Key Features

A comprehensive suite of tools organized into four key areas:

<details>
<summary><b>1. Novel Layers & Architectures</b></summary>

-   **Modern Attention Mechanisms**: A collection of advanced attention layers, including `DifferentialMultiHeadAttention`, `HopfieldAttention`, `SparseAttention`, `GroupQueryAttention`, and `NonLocalAttention`.
-   **Advanced Convolutional Blocks**: SOTA convolutional blocks like `ConvNeXtV1` & `V2`, `Convolutional Block Attention Module` (CBAM), and `OrthoBlock`, a novel block combining orthonormal weights with centering.
-   **Cutting-edge Layer Implementations**: Faithful implementations of recent breakthroughs like `BitLinear`, `Kolmogorov-Arnold Networks` (KAN), and `Capsule Networks` (CapsNet).
-   **Specialized MLP Variants**: Performance-oriented MLP designs including `GatedMLP` (gMLP), `SwiGLU-FFN`, `Differential FFN`, and `PowerMLP`.
-   **Full Model Implementations**: Ready-to-use implementations of `Depth Anything`, `MobileNetV4`, `ConvNeXt`, `CoshNet`, `CapsNet`, and a modern CLIP model.
</details>

<details>
<summary><b>2. Advanced Normalization & Regularization</b></summary>

-   **Modern Normalization Layers**: A wide range of `RMSNorm` variants (including `BandRMS` and `BandLogitNorm`), `GlobalResponseNorm` (GRN), and `LogitNorm` for stable and efficient training.
-   **Orthogonal Constraints**: `SoftOrthogonal` and `SRIP` regularizers, along with `OrthonormalInitializer` variants, to improve model stability and gradient flow by maintaining weight matrix properties.
-   **Novel Regularizers**: `BinaryPreference` and `TriStatePreference` regularizers that encourage weights to adopt quantized states, useful for model compression and interpretation.
-   **Stochastic Techniques**: `StochasticDepth` (DropPath) for robust training of deep networks by creating an implicit ensemble of models with varying depths.
</details>

<details>
<summary><b>3. Specialized Losses & Optimization</b></summary>

-   **Metric-Driven Losses**: The `AnyLoss` framework, which transforms any confusion-matrix-based classification metric (like F1-score or Balanced Accuracy) into a directly optimizable, differentiable loss function.
-   **Information-Theoretic Losses**: `GoodhartAwareLoss`, a novel loss that mitigates metric gaming by penalizing spurious correlations through entropy and mutual information regularization.
-   **Specialized Losses**: Custom losses for `Clustering`, `Segmentation`, and contrastive learning (`CLIPContrastiveLoss`).
-   **Optimization Tools**: `WarmupSchedule` for fine-grained learning rate control and utilities for implementing `DeepSupervision` in complex architectures.
</details>

<details>
<summary><b>4. In-Depth Model Analysis & Visualization</b></summary>

-   **Weight & Activation Analysis**: Integration of `WeightWatcher` for SVD-based generalization analysis and an `ActivationDistributionAnalyzer` to inspect layer-by-layer statistics, sparsity, and effective dimensionality.
-   **Comprehensive Model Comparison**: A powerful `ModelAnalyzer` to compare performance, confidence distributions, calibration curves, and feature representations across multiple models.
-   **Advanced Calibration Tools**: Built-in functions to compute and visualize `Expected Calibration Error (ECE)`, `Brier Score`, and reliability diagrams.
-   **Robust Logging & Visualization**: A `VisualizationManager` and structured logging to consistently monitor training pipelines and generate high-quality plots for analysis and reporting.
</details>

---

## Installation

> **Note:** This library requires Python 3.11 or newer.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/nikolasmarkou/dl_techniques.git
    cd dl_techniques
    ```

2.  **Install dependencies:**
    For standard usage, install the dependencies from `pyproject.toml` (or `requirements.txt` for a legacy approach):
    ```bash
    pip install .
    ```

3.  **Editable Install (for developers):**
    If you plan to contribute or modify the code, an editable install is recommended. This allows your changes to be reflected immediately without reinstalling.
    ```bash
    pip install -e ".[dev]"
    ```
    This command also installs development tools like `pytest` and `pylint`.

---

## Usage Examples

### 1. Building a Model with Custom Layers

Here’s how to integrate advanced components like `OrthoBlock` and `GlobalResponseNorm` into a simple model:

```python
import tensorflow as tf
from dl_techniques.layers import OrthoBlock
from dl_techniques.layers.norms import GlobalResponseNorm

inputs = tf.keras.Input(shape=(64,))
x = OrthoBlock(units=128, activation='gelu')(inputs)
x = GlobalResponseNorm()(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

### 2. Using a Custom Loss Function (AnyLoss)

The `AnyLoss` framework lets you directly optimize for metrics like F1-score, which is especially useful for imbalanced datasets.

```python
from dl_techniques.losses.any_loss import F1Loss

# ... model definition ...

model.compile(
    optimizer='adam',
    loss=F1Loss(),  # Directly optimize for F1-score
    metrics=['accuracy', 'precision', 'recall']
)
```

### 3. Using a Pre-built SOTA Model

Instantiate a complete, state-of-the-art model like `ConvNeXtV1` with just a few lines of code.

```python
from dl_techniques.models import ConvNextV1Tiny

# Create the ConvNeXtV1 "Tiny" model for ImageNet
model = ConvNextV1Tiny(
    include_top=True,
    weights=None, # Can be 'imagenet'
    input_shape=(224, 224, 3),
    classes=1000
)

model.summary()
```

---

## Project Structure

The repository is organized to separate the core library code, documentation, experiments, and tests.

-   **`src/dl_techniques/`**: The main library source code.
    -   `layers/`: All custom layers, including attention, normalization, and specialized blocks.
    -   `losses/`: Custom loss functions like `AnyLoss` and `GoodhartAwareLoss`.
    -   `regularizers/`: Advanced regularizers like `SRIP` and `SoftOrthogonal`.
    -   `models/`: Full implementations of architectures like `ConvNeXt`, `Depth Anything`, etc.
    -   `analysis/`: Tools for model analysis, including `WeightWatcher` and activation statistics.
    -   `utils/`: Core utilities for training, visualization, and data handling.
-   **`docs/`**: In-depth documentation on key concepts, architectures, and theoretical foundations.
-   **`experiments/`**: Standalone scripts demonstrating the use of library components on real tasks (e.g., MNIST, CIFAR-10).
-   **`tests/`**: A comprehensive test suite using `pytest` to ensure reliability and correctness.

---

## Documentation & Experiments

Detailed explanations of key concepts, implementations, and experiments can be found in the `docs/` and `experiments/` directories.

### Documentation Highlights

-   **[Chronological Guide to Neural Architectures](./docs/neural_network_architectures.md)**: An extensive, chronological list of influential neural network architectures with key references.
-   **[The Complete Transformer Guide (2025)](./docs/2025_attention.md)**: A meticulous guide to implementing SOTA Transformer architectures with every detail that matters.
-   **[AnyLoss Framework](./docs/anyloss_classification_metrics_loss_functions.md)**: A deep dive into directly optimizing classification metrics like F1-score.
-   **[Orthonormal Regularization & OrthoBlock](./docs/orthoblock.md)**: A novel block combining orthonormal weights with mean-centering normalization.
-   **[Band-Constrained Normalization](./docs/bcn_thesis.md)**: A thesis on a novel normalization technique that preserves magnitude information within a bounded shell.
-   **[Mixture Density Networks (MDNs)](./docs/mdn.md)**: Theory, use cases, and best practices for probabilistic modeling.

### Experiments

The [`experiments/`](./src/experiments/) directory contains practical scripts and results for various techniques:

-   **Goodhart's Law**: Testing `GoodhartAwareLoss` on spurious correlations in CIFAR-10.
-   **OrthoBlock**: Comparing `OrthoBlock` against baseline dense layers on CIFAR-10.
-   **Differential FFN**: Evaluating the `DifferentialFFN` layer on synthetic data.
-   **Mixture Density Networks**: Forecasting time series data with `MDN`s.
-   **Custom Normalization**: Evaluating `BandRMS` and coupled normalization layers.

Each experiment provides a starting point for your own research and includes code for training, evaluation, and visualization.

---

## Contributing

Contributions are welcome! Please follow these steps:

1.  **Fork** & **Clone** the repository.
2.  **Create a Branch** for your feature or bugfix.
3.  **Write Tests** covering your changes in the `tests/` folder.
4.  **Run Checks**: This project uses `pre-commit` for code formatting and `pytest` for testing. Ensure all checks pass.
5.  **Open a Pull Request** with a clear description of your changes.

**Coding Standards**:
-   Follow [PEP 8](https://peps.python.org/pep-0008/).
-   Use **type hints** for all functions and methods.
-   Write clear, concise docstrings (Sphinx or NumPy style) for new layers, models, or utilities.

---

## License

This project is licensed under the **GNU General Public License v3.0**.

**Important**: The GPL-3.0 is a "copyleft" license. This means that any derivative work you create and distribute must also be licensed under GPL-3.0. Please review the [LICENSE](./LICENSE) file carefully before using this library in your projects, especially in a commercial context.

---

## References

This library is built upon an extensive body of research. The implementations are inspired by or directly based on numerous academic papers. The following is a categorized list of key references.

<details>
<summary><b>Foundational Architectures</b></summary>

-   **Attention Is All You Need** (Transformers): Vaswani, A., et al. (2017). In *Advances in Neural Information Processing Systems (NIPS)*.
-   **ImageNet Classification with Deep Convolutional Neural Networks** (AlexNet): Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). In *Advances in Neural Information Processing Systems (NIPS)*.
-   **Deep Residual Learning for Image Recognition** (ResNet): He, K., et al. (2016). In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
-   **A ConvNet for the 2020s** (ConvNeXt): Liu, Z., et al. (2022). In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.
-   **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** (ViT): Dosovitskiy, A., et al. (2020). In *International Conference on Learning Representations (ICLR)*.
</details>

<details>
<summary><b>Attention & Transformer Variants</b></summary>

-   **DIFFERENTIAL TRANSFORMER**: Zhu, J., et al. (2025). "Amplifying attention to the relevant context while canceling noise." In *CVPR*.
-   **Hopfield Networks is All You Need**: Ramsauer, H., et al. (2020). In *International Conference on Learning Representations (ICLR)*.
-   **GQA: Training Generalized Multi-Query Transformer Models**: Ainslie, J., et al. (2023). In *arXiv*.
-   **RoFormer: Enhanced Transformer with Rotary Position Embedding**: Su, J., et al. (2021). In *arXiv*.
-   **GLU Variants Improve Transformer**: Shazeer, N. (2020). In *arXiv*.
-   **Non-local Neural Networks**: Wang, X., et al. (2018). In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
</details>

<details>
<summary><b>Normalization & Regularization</b></summary>

-   **Batch Normalization**: Ioffe, S., & Szegedy, C. (2015). In *Proceedings of the 32nd International Conference on Machine Learning (ICML)*.
-   **Layer Normalization**: Ba, J. L., et al. (2016). In *arXiv*.
-   **Root Mean Square Layer Normalization**: Zhang, B., & Sennrich, R. (2019). In *Advances in Neural Information Processing Systems (NeurIPS)*.
-   **Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?**: Bansal, N., et al. (2018). In *Advances in Neural Information Processing Systems (NeurIPS)*.
-   **Decoupled Weight Decay Regularization** (AdamW): Loshchilov, I., & Hutter, F. (2019). In *International Conference on Learning Representations (ICLR)*.
-   **Deep Networks with Stochastic Depth**: Huang, G., et al. (2016). In *European Conference on Computer Vision (ECCV)*.
</details>

<details>
<summary><b>Specialized & Modern Layers</b></summary>

-   **Dynamic Routing Between Capsules** (CapsNet): Sabour, S., Frosst, N., & Hinton, G. E. (2017). In *Advances in Neural Information Processing Systems (NIPS)*.
-   **KAN: Kolmogorov-Arnold Networks**: Liu, Y., et al. (2024). In *arXiv*.
-   **The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits** (BitLinear): Ma, S., et al. (2024). In *arXiv*.
-   **Pay Attention to MLPs** (gMLP): Liu, H., et al. (2021). In *Advances in Neural Information Processing Systems (NeurIPS)*.
-   **PowerMLP: An Efficient Version of KAN**: Chen, Z., et al. (2024). In *arXiv*.
-   **Mish: A Self Regularized Non-Monotonic Neural Activation Function**: Misra, D. (2019). In *arXiv*.
</details>

<details>
<summary><b>Loss Functions & Optimization</b></summary>

-   **AnyLoss: Transforming Classification Metrics into Loss Functions**: Han, D., Moniz, N., & Chawla, N. V. (2024).
-   **Regularizing by Penalizing Confident Outputs**: Pereyra, G., et al. (2017).
-   **The Information Bottleneck Method**: Tishby, N., Pereira, F. C., & Bialek, W. (2000).
-   **Mitigating Neural Network Overconfidence with Logit Normalization**: Wei, H., et al. (2022). In *ICML*.
-   **On calibration of modern neural networks**: Guo, C., et al. (2017). In *ICML*.
</details>

<details>
<summary><b>Generative & Contrastive Learning</b></summary>

-   **Learning Transferable Visual Models From Natural Language Supervision** (CLIP): Radford, A., et al. (2021). In *ICML*.
-   **Denoising Diffusion Probabilistic Models**: Ho, J., Jain, A., & Abbeel, P. (2020). In *Advances in Neural Information Processing Systems (NeurIPS)*.
-   **Generative Adversarial Nets**: Goodfellow, I., et al. (2014). In *Advances in Neural Information Processing Systems (NIPS)*.
-   **Auto-Encoding Variational Bayes** (VAE): Kingma, D. P., & Welling, M. (2013). In *arXiv*.
-   **Mixture Density Networks**: Bishop, C. M. (1994). *Technical Report, Aston University*.
</details>