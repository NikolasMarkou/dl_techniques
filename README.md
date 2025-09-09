# A Curated Arsenal of Advanced Deep Learning Techniques

<p align="center">
  <a href="https://github.com/nikolasmarkou/dl_techniques/blob/main/LICENSE">
    <img alt="License: GPL-3.0" src="https://img.shields.io/badge/License-GPL_v3-blue.svg">
  </a>
  <img alt="Python Version" src="https://img.shields.io/badge/python-3.11+-blue.svg">
  <img alt="Framework" src="https://img.shields.io/badge/Keras-3.x-red">
  <img alt="Backend" src="https://img.shields.io/badge/TensorFlow-2.18-orange">
  <a href="https://electiconsulting.com">
    <img alt="Sponsored by Electi Consulting" src="https://img.shields.io/badge/Sponsored%20by-Electi%20Consulting-8B1C34">
  </a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/nikolasmarkou/dl_techniques/main/imgs/minimalist-2d-logo-with-a-left-to-right-_sSVDZkeKR4C_eovommSCFQ_mJekanaZTB2Nbe5dBKOnPQ.png" alt="Deep Learning Techniques Logo" width="350">
</p>

**dl_techniques** is a research-first Python library built for Keras 3 and TensorFlow, developed with the support of [Electi Consulting](https://electiconsulting.com)'s AI research initiatives. It's more than a collection of layers; it's a curated toolkit for researchers and advanced practitioners to design, train, and dissect state-of-the-art neural networks. This library bridges the gap between groundbreaking research papers and practical implementation, providing faithful and efficient components that are ready for experimentation and production deployment.

From novel attention mechanisms and orthonormal layers to information-theoretic loss functions and advanced model analysis tools, `dl_techniques` is your companion for pushing the boundaries of deep learning research and enterprise applications.

---

## Table of Contents
1.  [**Key Features**](#key-features): What makes this library stand out.
2.  [**Why `dl_techniques`?**](#why-dl_techniques): The philosophy behind the project.
3.  [**Installation**](#installation): Get up and running quickly.
4.  [**Quick Start**](#quick-start): See the library in action.
5.  [**In-Depth Documentation**](#in-depth-documentation): Go beyond the code.
6.  [**Project Structure**](#project-structure): How the repository is organized.
7.  [**Contributing**](#contributing): Join the development.
8.  [**License**](#license): Understanding the GPL-3.0 license.
9.  [**Acknowledgments**](#acknowledgments): Recognition and support.
10. [**Citations & References**](#citations--references): The research that inspired this library.

---

## Key Features

This library is a comprehensive suite of tools organized into four key pillars, developed through rigorous research and validated in real-world enterprise applications:

<details>
<summary><b>1. An Arsenal of SOTA Layers & Architectures (170+ Layers)</b></summary>

-   **Modern Attention Mechanisms**: A rich collection of advanced attention layers, including `DifferentialMultiHeadAttention`, `HopfieldAttention`, `GroupQueryAttention`, and `NonLocalAttention`, along with essentials like `RotaryPositionEmbedding` (RoPE).
-   **Advanced Convolutional Blocks**: State-of-the-art convolutional blocks like `ConvNeXtV1` & `V2`, the `Convolutional Block Attention Module` (CBAM), and our novel `OrthoBlock`, which combines orthonormal weights with mean-centering.
-   **Cutting-edge Layer Implementations**: Faithful and efficient implementations of recent breakthroughs like `BitLinear` layers, `Kolmogorov-Arnold Networks` (KAN), and `Capsule Networks` (CapsNet).
-   **Specialized MLP & FFN Variants**: Explore beyond the standard MLP with performance-oriented designs including `GatedMLP` (gMLP), `SwiGLU-FFN`, `DifferentialFFN`, and `PowerMLP`.
-   **Full Model Implementations**: Ready-to-use implementations of `Depth Anything`, `MobileNetV4`, `ConvNeXt`, `CoshNet`, `CapsNet`, and a modern `CLIP` model for vision-language tasks.
</details>

<details>
<summary><b>2. Advanced Normalization & Regularization</b></summary>

-   **Next-Generation Normalization**: Go beyond BatchNorm with a wide range of `RMSNorm` variants (including our novel `BandRMS` and `BandLogitNorm`), `GlobalResponseNorm` (GRN), and `LogitNorm` for stable and efficient training.
-   **Principled Orthogonal Constraints**: Improve model stability and gradient flow with `SoftOrthogonal` and `SRIP` regularizers, alongside `OrthonormalInitializer` variants that maintain critical weight matrix properties.
-   **Novel Regularizers**: Experiment with `BinaryPreference` and `TriStatePreference` regularizers that encourage weights to adopt quantized states, ideal for model compression and interpretability studies.
-   **Robust Stochastic Techniques**: A battle-tested `StochasticDepth` (DropPath) implementation for training deep networks, creating an implicit ensemble of models to prevent overfitting.
</details>

<details>
<summary><b>3. Specialized Loss Functions & Optimization</b></summary>

-   **Direct Metric Optimization**: The `AnyLoss` framework, which transforms any confusion-matrix-based classification metric (like F1-score or Balanced Accuracy) into a directly optimizable, differentiable loss function.
-   **Information-Theoretic Losses**: Mitigate Goodhart's Law and metric gaming with `GoodhartAwareLoss`, a novel loss that penalizes spurious correlations by regularizing entropy and mutual information.
-   **Task-Specific Losses**: Custom losses for `Clustering`, `Segmentation`, and contrastive learning (`CLIPContrastiveLoss`).
-   **Advanced Optimization Tools**: `WarmupSchedule` for fine-grained learning rate control and utilities for implementing `DeepSupervision` in complex, multi-scale architectures.
</details>

<details>
<summary><b>4. Deep Model Introspection & Analysis</b></summary>

-   **Comprehensive Model Analyzer**: A unified `ModelAnalyzer` toolkit for comparing multiple models across performance, weight health, calibration quality, information flow, and training dynamics with publication-ready visualizations and automated insights.
-   **Weight & Activation Analysis**: Integrated `WeightWatcher` for SVD-based generalization analysis and detailed layer-by-layer inspection of statistics, sparsity, and effective rank.
-   **Advanced Calibration Tools**: Built-in functions to compute and visualize `Expected Calibration Error (ECE)`, `Brier Score`, and reliability diagrams to ensure your model's confidence scores are trustworthy.
-   **Robust Logging & Visualization**: A `VisualizationManager` and structured logging to consistently monitor training pipelines and generate publication-quality plots for analysis and reporting.
</details>

---

## Why `dl_techniques`?

*   **Curated, Not Cluttered**: This isn't an unorganized collection of every paper implementation. Each component is selected for its significance, novelty, or potential for research and enterprise deployment.
*   **Built for Research & Production**: The library is designed to make it easy to swap components, test new ideas, and analyze the results with powerful, built-in tools. Components have been validated in enterprise environments through Electi Consulting's AI implementations.
*   **More Than Code**: We provide in-depth documentation in the `docs/` directory that explains the *why* behind the *what*. This includes deep dives into attention mechanisms, custom architectures, and the theory behind our novel loss functions.
*   **Reliable and Tested**: With an extensive test suite and real-world validation, you can trust that these advanced components are implemented correctly and robustly for both research and production use.
*   **Deep Introspection**: We believe that understanding *why* a model works is as important as its accuracy. Our integrated analysis tools are a first-class feature of the library, informed by practical experience in enterprise AI deployments.

---

## Installation

> **Note:** This library is built on Python 3.11+ and Keras 3.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/nikolasmarkou/dl_techniques.git
    cd dl_techniques
    ```

2.  **Install dependencies:**
    For standard usage, install the library and its dependencies directly. This will use `pyproject.toml` to handle the installation.
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

## Quick Start

### 1. Build a Model with Custom Layers

Here's how to integrate advanced components like our novel `OrthoBlock` and `GlobalResponseNorm` into a simple model:

```python
import tensorflow as tf
from dl_techniques.layers.experimental import OrthoBlock
from dl_techniques.layers.norms import GlobalResponseNormalization

inputs = tf.keras.Input(shape=(64,))
x = OrthoBlock(units=128, activation='gelu')(inputs)
x = GlobalResponseNormalization()(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.summary()
```

### 2. Optimize Directly for F1-Score

The `AnyLoss` framework lets you directly optimize for metrics like F1-score, which is especially useful for imbalanced datasets.

```python
from dl_techniques.losses.any_loss import F1Loss

# ... model definition ...

model.compile(
    optimizer='adam',
    loss=F1Loss(),  # Directly optimize for F1-score instead of a proxy like cross-entropy
    metrics=['accuracy', 'precision', 'recall']
)
```

### 3. Use a Pre-built SOTA Model

Instantiate a complete, state-of-the-art model like `ConvNeXtV1` with just a few lines of code.

```python
from dl_techniques.models import create_convnext_v1

# Create the ConvNeXtV1 "Tiny" model for ImageNet
model = create_convnext_v1(
    variant='tiny',
    num_classes=1000,
    input_shape=(224, 224, 3)
)

model.summary()
```

### 4. Analyze and Compare Models

Get deep insights into your models with comprehensive analysis covering training dynamics, calibration, and weight health:

```python
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig, DataInput

# Compare multiple models and their training histories
models = {'ResNet': resnet_model, 'ConvNext': convnext_model}
histories = {'ResNet': resnet_history, 'ConvNext': convnext_history}
config = AnalysisConfig(analyze_training_dynamics=True, analyze_calibration=True)
test_data = DataInput(x_test, y_test)

# Initialize the analyzer
analyzer = ModelAnalyzer(models, config=config, training_history=histories)

# Run all configured analyses
results = analyzer.analyze(test_data)

# This generates a full suite of publication-ready visualizations and detailed insights.
```

---

## In-Depth Documentation

This library is a knowledge base as much as it is a codebase. Detailed explanations of key concepts, implementations, and experiments can be found in the `docs/` and `experiments/` directories.

### Documentation Highlights

-   **[The Complete Transformer Guide (2025)](./docs/2025_attention.md)**: A meticulous, production-focused guide to implementing SOTA Transformer architectures with every detail that matters.
-   **[AnyLoss Framework](./docs/anyloss_classification_metrics_loss_functions.md)**: A deep dive into directly optimizing classification metrics like F1-score.
-   **[Chronological Guide to Neural Architectures](./docs/neural_network_architectures.md)**: An extensive, chronological list of influential neural network architectures with key references.
-   **[Band-Constrained Normalization](./docs/bcn_thesis.md)**: A thesis on a novel normalization technique that preserves magnitude information within a bounded shell.
-   **[OrthoBlock & Orthonormal Regularization](./docs/orthoblock.md)**: A novel block combining orthonormal weights with mean-centering normalization.
-   **[Mixture Density Networks (MDNs)](./docs/mdn.md)**: Theory, use cases, and best practices for probabilistic modeling.

### Experiments

The [`experiments/`](./src/experiments/) directory contains practical scripts and results for various techniques, providing a starting point for your own research:

-   **Goodhart's Law**: Testing `GoodhartAwareLoss` on spurious correlations in CIFAR-10.
-   **OrthoBlock**: Comparing `OrthoBlock` against baseline dense layers on CIFAR-10.
-   **Differential FFN**: Evaluating the `DifferentialFFN` layer on synthetic data.
-   **Mixture Density Networks**: Forecasting time series data with `MDN`s.

---

## Project Structure

The repository is organized to separate the core library code, documentation, experiments, and tests.

-   **`src/dl_techniques/`**: The main library source code.
    -   `layers/`: All custom layers, including attention, normalization, and specialized blocks.
    -   `losses/`: Custom loss functions like `AnyLoss` and `GoodhartAwareLoss`.
    -   `regularizers/`: Advanced regularizers like `SRIP` and `SoftOrthogonal`.
    -   `models/`: Full implementations of architectures like `ConvNeXt`, `Depth Anything`, etc.
    -   `utils/`: Core utilities for training, visualization, and data handling.
    -   `weightwatcher/`: Tools for deep model analysis, including SVD-based metrics.
    -   `analyzer/`: Comprehensive model analysis and comparison toolkit.
-   **`docs/`**: In-depth documentation, theoretical guides, and papers.
-   **`experiments/`**: Standalone scripts demonstrating library components on real tasks.
-   **`tests/`**: A comprehensive test suite using `pytest` to ensure reliability and correctness.

---

## Contributing

Contributions are highly welcome! If you'd like to add a new technique, improve documentation, or fix a bug, please follow these steps:

1.  **Fork** & **Clone** the repository.
2.  **Create a Branch** for your feature or bugfix.
3.  **Write Tests** covering your changes in the `tests/` folder.
4.  **Run Checks**: This project uses `pre-commit` for code formatting (`black`, `isort`) and `pytest` for testing. Ensure all checks pass.
5.  **Open a Pull Request** with a clear description of your changes and their significance.

**Coding Standards**:
-   Follow [PEP 8](https://peps.python.org/pep-0008/).
-   Use **type hints** for all functions and methods.
-   Write clear, concise docstrings for new layers, models, or utilities.

---

## License

This project is licensed under the **GNU General Public License v3.0**.

**Important**: The GPL-3.0 is a "copyleft" license. This means that any derivative work you create and distribute that uses code from this library must also be licensed under GPL-3.0. Please review the [LICENSE](./LICENSE) file carefully before using this library in your projects, especially in a commercial context.

---

## Acknowledgments

This project is proudly sponsored by **[Electi Consulting](https://electiconsulting.com)**, a leading AI and technology consultancy specializing in artificial intelligence, blockchain, and cryptography solutions for enterprise clients. The practical insights and enterprise validation of these techniques have been made possible through Electi's extensive experience in deploying AI solutions across various industries including finance, maritime, and healthcare.

Special thanks to the open-source community and the researchers whose groundbreaking work forms the foundation of this library.

---

## Citations & References

This library is built upon an extensive body of research. The implementations are inspired by or directly based on numerous academic papers. The following is a categorized, but not exhaustive, list of key references that have shaped this work.

<details>
<summary><b>Foundational Architectures</b></summary>

-   **Attention Is All You Need** (Transformers): Vaswani, A., et al. (2017).
-   **Deep Residual Learning for Image Recognition** (ResNet): He, K., et al. (2016).
-   **A ConvNet for the 2020s** (ConvNeXt): Liu, Z., et al. (2022).
-   **An Image is Worth 16x16 Words** (ViT): Dosovitskiy, A., et al. (2020).
-   **Dynamic Routing Between Capsules** (CapsNet): Sabour, S., Frosst, N., & Hinton, G. E. (2017).
</details>

<details>
<summary><b>Attention & Transformer Variants</b></summary>

-   **DIFFERENTIAL TRANSFORMER**: Zhu, J., et al. (2025).
-   **Hopfield Networks is All You Need**: Ramsauer, H., et al. (2020).
-   **GQA: Training Generalized Multi-Query Transformer Models**: Ainslie, J., et al. (2023).
-   **RoFormer: Enhanced Transformer with Rotary Position Embedding**: Su, J., et al. (2021).
-   **GLU Variants Improve Transformer**: Shazeer, N. (2020).
</details>

<details>
<summary><b>Normalization & Regularization</b></summary>

-   **Root Mean Square Layer Normalization**: Zhang, B., & Sennrich, R. (2019).
-   **Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?**: Bansal, N., et al. (2018).
-   **Decoupled Weight Decay Regularization** (AdamW): Loshchilov, I., & Hutter, F. (2019).
-   **Deep Networks with Stochastic Depth**: Huang, G., et al. (2016).
</details>

<details>
<summary><b>Specialized & Modern Layers</b></summary>

-   **KAN: Kolmogorov-Arnold Networks**: Liu, Y., et al. (2024).
-   **The Era of 1-bit LLMs** (BitLinear): Ma, S., et al. (2024).
-   **Pay Attention to MLPs** (gMLP): Liu, H., et al. (2021).
-   **PowerMLP: An Efficient Version of KAN**: Chen, Z., et al. (2024).
</details>

<details>
<summary><b>Loss Functions & Optimization</b></summary>

-   **AnyLoss: Transforming Classification Metrics into Loss Functions**: Han, D., Moniz, N., & Chawla, N. V. (2024).
-   **Mitigating Neural Network Overconfidence with Logit Normalization**: Wei, H., et al. (2022).
-   **Regularizing by Penalizing Confident Outputs**: Pereyra, G., et al. (2017).
</details>

<details>
<summary><b>Generative & Contrastive Learning</b></summary>

-   **Learning Transferable Visual Models From Natural Language Supervision** (CLIP): Radford, A., et al. (2021).
-   **Mixture Density Networks**: Bishop, C. M. (1994).
-   **Denoising Diffusion Probabilistic Models**: Ho, J., Jain, A., & Abbeel, P. (2020).
</details>