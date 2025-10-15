# A Curated Arsenal of Advanced Deep Learning Techniques

<p align="center">
  <a href="https://github.com/nikolasmarkou/dl_techniques/blob/main/LICENSE">
    <img alt="License: GPL-3.0" src="https://img.shields.io/badge/License-GPL_v3-blue.svg">
  </a>
  <img alt="Python Version" src="https://img.shields.io/badge/python-3.11+-blue.svg">
  <img alt="Framework" src="https://img.shields.io/badge/Keras-3.8-red">
  <img alt="Backend" src="https://img.shields.io/badge/TensorFlow-2.18-orange">
  <a href="https://electiconsulting.com">
    <img alt="Sponsored by Electi Consulting" src="https://img.shields.io/badge/Sponsored%20by-Electi%20Consulting-8B1C34">
  </a>
</p>

<p align="center">
  <img src="https://raw.github.com/nikolasmarkou/dl_techniques/main/imgs/minimalist-2d-logo-with-a-left-to-right-_sSVDZkeKR4C_eovommSCFQ_mJekanaZTB2Nbe5dBKOnPQ.png" alt="Deep Learning Techniques Logo" width="350">
</p>

**dl_techniques** is a research-first Python library built for Keras 3 and TensorFlow, developed with the support of [Electi Consulting](https://electiconsulting.com)'s AI research initiatives. It is more than a collection of layers; it is a curated toolkit for researchers and advanced practitioners to design, train, and dissect state-of-the-art neural networks. This library bridges the gap between groundbreaking research papers and practical implementation, providing faithful and efficient components that are ready for experimentation and production deployment.

From cutting-edge attention mechanisms and graph neural networks to information-theoretic loss functions and comprehensive model analysis tools, `dl_techniques` is your companion for pushing the boundaries of deep learning research and enterprise applications.

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

This library is a comprehensive suite of tools organized into five key pillars, developed through rigorous research and validated in real-world enterprise applications:

<details>
<summary><b>1. Cutting-Edge Architectures & Models (130+ Complete Models)</b></summary>
<p>

- **Modern Language Models**: Implementations of `Gemma 3`, `Qwen3`, `Mamba`, and modern `BERT` variants with Block-wise Latent Transformers (BLT) and Hierarchical Reasoning.
- **Vision & Multimodal Models**: A comprehensive suite including `CLIP`, `FastVLM`, `NanoVLM`, `DINOv1/v2/v3`, `SigLIP-ViT`, and the object detection model `DETR`.
- **State-of-the-Art CNNs**: Advanced convolutional architectures such as `ConvNeXtV1/V2`, `MobileNetV4`, `FractalNet`, `CoShNet`, and the efficient `SqueezeNet` family.
- **Specialized Vision Architectures**: Models for specific tasks like `DepthAnything` for monocular depth estimation and a complete `VAE` for generative modeling.
- **Time Series & Forecasting**: Advanced forecasting models like `TiRex` with quantile prediction capabilities and an enhanced implementation of `N-BEATS`.
- **Experimental Architectures**: Exploration of novel concepts including full `Capsule Networks` (CapsNet) with dynamic routing, `Graph Neural Networks` (GNNs), and Kolmogorov-Arnold Networks (`KAN`).
</p>
</details>

<details>
<summary><b>2. Advanced Layer Components (220+ Specialized Layers)</b></summary>
<p>

- **Next-Generation Attention**: `DifferentialMultiHeadAttention`, modern `HopfieldAttention`, `GroupQueryAttention`, and efficient alternatives like `FNetFourierTransform`.
- **Factory Pattern Architecture**: Unified factory systems for consistent creation and validation of over 15 attention mechanisms, 15+ normalization variants, and multiple Feed-Forward Network (FFN) types.
- **Graph & Structural Components**: Configurable GNN layers with multiple aggregation strategies, `Relational Graph Transformer` (RELGT) blocks, and `Entity-Graph Refinement` for hierarchical relationship learning.
- **Mixture of Experts (MoE)**: A complete MoE system featuring configurable expert networks, multiple gating strategies (including SoftMoE), and integrated training utilities.
- **Statistical & Probabilistic Layers**: `Mixture Density Networks` (MDN) for probabilistic predictions, `Normalizing Flows` for conditional density estimation, and time series analysis layers for residual autocorrelation.
</p>
</details>

<details>
<summary><b>3. Comprehensive Analysis & Introspection Toolkit</b></summary>
<p>

- **Multi-Dimensional Model Analysis**: A unified `ModelAnalyzer` to compare models across six key dimensions: training dynamics, weight health, calibration, information flow, and advanced spectral analysis.
- **Advanced Visualization Suite**: Automated generation of publication-quality visualizations, interactive summary dashboards, and comparative analysis tools with statistical significance testing.
- **Spectral Analysis (WeightWatcher Integration)**: Assess generalization potential and training quality by analyzing the spectral properties (eigenvalues) of weight matrices, often without requiring test data.
- **Specialized Analyzers**: Deep-dive modules for prediction confidence (`CalibrationAnalyzer`), information bottlenecks (`InformationFlowAnalyzer`), weight statistics (`WeightAnalyzer`), and learning efficiency (`TrainingDynamicsAnalyzer`).
</p>
</details>

<details>
<summary><b>4. Advanced Loss Functions & Optimization (20+ Specialized Losses)</b></summary>
<p>

- **Direct Metric Optimization**: The `AnyLoss` framework, which transforms any confusion-matrix-based metric (e.g., F1-score, Balanced Accuracy) into a differentiable loss function.
- **Information-Theoretic & Robust Losses**: `GoodhartAwareLoss` to combat spurious correlations, calibration-focused losses like `BrierScoreLoss`, and the uncertainty-aware `FocalUncertaintyLoss`.
- **Task-Specific Loss Functions**: Specialized losses for vision-language (`CLIPContrastiveLoss`), segmentation (`Dice`, `Focal`, `Tversky`), time series (`MASELoss`, `SMAPELoss`), and generative modeling (`WassersteinLoss` with gradient penalty).
- **Advanced Optimization Tools**: Smart learning rate schedulers like `WarmupSchedule`, utilities for `DeepSupervision` in multi-scale architectures, and a suite of advanced regularizers (`SoftOrthogonal`, `SRIP`).
</p>
</details>

<details>
<summary><b>5. Production-Ready Training Infrastructure</b></summary>
<p>

- **Complete Training Pipelines**: Over 25 ready-to-use training scripts for all major architectures, establishing standardized workflows for training, validation, and testing.
- **Utilities & Tools**: Advanced data loaders, augmentation pipelines, a structured visualization and logging manager, and enhanced model serialization with custom object support.
- **Comprehensive Testing**: An extensive suite of over 600 unit and integration tests ensures the reliability and correctness of all components.
- **Performance Benchmarks**: Validation against reference implementations and established benchmarks to ensure numerical accuracy and performance.
</p>
</details>

---

## Why `dl_techniques`?

*   **Research-Driven Excellence**: Each component is selected for its significance, novelty, and potential impact, not just popularity. Components are implemented based on a thorough understanding of the underlying research.
*   **Enterprise-Validated**: All components have been tested and validated in real-world enterprise environments through Electi Consulting's AI implementations across finance, maritime, and healthcare industries.
*   **Factory Pattern Architecture**: Innovative factory systems ensure consistency, reduce boilerplate code, and make component swapping effortless for rapid experimentation.
*   **Deep Introspection as a First-Class Citizen**: Understanding *why* a model works is as important as its accuracy. Our integrated analysis tools provide unprecedented insights into model behavior.
*   **Modern Keras 3 Design**: Built from the ground up for Keras 3 with proper type hints, comprehensive documentation, and modern Python practices.
*   **Extensible by Design**: The modular architecture allows for the easy integration of new components while maintaining backward compatibility.

---

## Installation

> **Note:** This library requires Python 3.11+ and Keras 3.8.0 with the TensorFlow 2.18.0 backend.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/nikolasmarkou/dl_techniques.git
    cd dl_techniques
    ```

2.  **Install dependencies:**
    For standard usage, install the library and its core dependencies:
    ```bash
    pip install .
    ```

3.  **Editable Install (for developers):**
    If you plan to contribute or modify the library's source code, install it in editable mode with development tools:
    ```bash
    pip install -e ".[dev]"
    ```
    This installs additional tools such as `pytest`, `pylint`, and `black`.

4.  **Verify Installation:**
    ```bash
    python -c "import dl_techniques; print('Installation successful!')"
    ```

---

## Quick Start

### 1. Build a Modern Architecture with Factory Patterns

Utilize the factory systems to easily create and interchange components for rapid prototyping.

```python
import keras
from dl_techniques.layers.attention.factory import create_attention_layer
from dl_techniques.layers.norms.factory import create_normalization_layer
from dl_techniques.layers.ffn.factory import create_ffn_layer

inputs = keras.Input(shape=(1024, 512))

# Use factories for consistent component creation
attention = create_attention_layer(
    'differential_mha',
    dim=512,
    num_heads=8,
    head_dim=64
)
norm = create_normalization_layer('rms_norm', epsilon=1e-6)
ffn = create_ffn_layer('swiglu_ffn', hidden_dim=2048)

# Build a modern transformer block
x = attention(inputs)
x = norm(x)
x = ffn(x)

model = keras.Model(inputs, x)
model.summary()
```

### 2. Advanced Time Series Forecasting with TiRex

Create a state-of-the-art time series model capable of generating probabilistic forecasts.

```python
from dl_techniques.models.tirex import create_tirex_model

# Create a TiRex model for multivariate forecasting
model = create_tirex_model(
    input_shape=(100, 10),  # 100 timesteps, 10 features
    forecast_horizon=24,
    quantiles=[0.1, 0.5, 0.9],  # Configure for probabilistic forecasting
    variant='base'
)

# The model is compatible with quantile loss for uncertainty estimation
model.compile(
    optimizer='adamw',
    loss='quantile_loss',
    metrics=['mae', 'mse']
)
```

### 3. Vision-Language Modeling with FastVLM

Construct a complete, efficient vision-language model for multimodal tasks.

```python
from dl_techniques.models.fastvlm import FastVLM

# Create a FastVLM instance from a predefined variant
vlm = FastVLM.from_variant(
    'base',
    vocab_size=32000,
    max_length=512,
    image_size=224
)

# The model accepts both image and text inputs
image_input = keras.Input(shape=(224, 224, 3))
text_input = keras.Input(shape=(512,), dtype='int32')

outputs = vlm([image_input, text_input])
model = keras.Model([image_input, text_input], outputs)
```

### 4. Comprehensive Model Analysis

Gain deep insights into model behavior with the advanced analysis toolkit.

```python
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig, DataInput

# Compare multiple models comprehensively
models = {'TiRex': tirex_model, 'LSTM': lstm_model}
histories = {'TiRex': tirex_history, 'LSTM': lstm_history}
test_data = DataInput(x_test, y_test)

# Configure a comprehensive analysis run
config = AnalysisConfig(
    analyze_training_dynamics=True,
    analyze_calibration=True,
    analyze_weight_health=True,
    analyze_spectral=True, # Enable spectral analysis for generalization insights
    save_plots=True,
    plot_style='publication'
)

analyzer = ModelAnalyzer(models, config=config, training_history=histories)

# Execute the complete analysis and generate visualizations
results = analyzer.analyze(test_data)

# Access detailed, structured metrics programmatically
print(f"Best calibrated model (by ECE): {min(results.calibration_metrics.items(), key=lambda x: x[1]['ece'])}")
print(f"Training efficiency ranking (epochs to converge): {results.training_metrics.convergence_epochs}")
```

### 5. Direct F1-Score Optimization

Use the `AnyLoss` framework to optimize a model directly for a target classification metric.

```python
from dl_techniques.losses.any_loss import F1Loss, BalancedAccuracyLoss

# For imbalanced datasets, optimize F1-score directly
model.compile(
    optimizer='adamw',
    loss=F1Loss(from_logits=True),
    metrics=['accuracy', 'precision', 'recall']
)

# Alternatively, use balanced accuracy for better class balance
model.compile(
    optimizer='adamw',
    loss=BalancedAccuracyLoss(from_logits=True),
    metrics=['accuracy']
)
```

### 6. Graph Neural Networks for Relational Data

Process graph-structured data using configurable GNN implementations.

```python
from dl_techniques.layers.graphs import GraphNeuralNetworkLayer

# Create a GNN layer with Graph Attention Networks (GAT) for message passing
gnn = GraphNeuralNetworkLayer(
    concept_dim=256,
    num_layers=3,
    message_passing='gat',
    aggregation='attention',
    dropout_rate=0.1
)

# Apply the layer to graph-structured inputs
node_features = keras.Input(shape=(None, 256))  # Variable number of nodes
adjacency_matrix = keras.Input(shape=(None, None))

node_embeddings = gnn([node_features, adjacency_matrix])
```

---

## In-Depth Documentation

This library serves as both a practical toolkit and a knowledge repository. Comprehensive documentation covers implementation details and their theoretical foundations. All research articles are located in the `research/` directory.

### Core Documentation

-   **[Model Analyzer Guide](./research/model_analyzer.md)**: A comprehensive tutorial for the advanced model analysis toolkit.
-   **[AnyLoss Framework](./research/anyloss_classification_metrics_loss_functions.md)**: A deep dive into the theory and practice of direct metric optimization.
-   **[Complete Transformer Guide (2025)](./research/2025_transformer_architectures.md)**: A production-focused guide to implementing state-of-the-art Transformer architectures.

### Architecture & Implementation Guides

-   **[Chronological Neural Architectures](./research/neural_network_architectures.md)**: An extensive chronological guide to influential architectures with implementation notes.
-   **[Band-Constrained Normalization](./research/bcn_thesis.md)**: A novel normalization technique that preserves magnitude information within bounded constraints.
-   **[OrthoBlock & Orthonormal Regularization](./research/orthoblock.md)**: A guide to structured feature learning with orthogonal constraints.
-   **[Mixture Density Networks](./research/mdn.md)**: The theory and best practices for implementing probabilistic models.

### API Reference
For detailed, auto-generated documentation on every module, class, and function, please refer to the `docs/` directory.

---

## Project Structure

The repository is organized for clarity and maintainability.

```
dl_techniques/
├── src/dl_techniques/
│   ├── layers/                # 220+ specialized layer implementations
│   │   ├── attention/         # Modern attention mechanisms with factory
│   │   ├── norms/             # Advanced normalization with factory
│   │   ├── ffn/               # Feed-forward networks with factory
│   │   ├── graphs/            # Graph neural network components
│   │   ├── moe/               # Mixture of Experts implementation
│   │   └── statistics/        # Statistical and probabilistic layers
│   ├── models/                # 130+ complete architecture implementations
│   │   ├── qwen/              # Qwen3 language model family
│   │   ├── fastvlm/           # Fast vision-language models
│   │   ├── tirex/             # Advanced time series forecasting
│   │   └── detr/              # DETR object detection model
│   ├── losses/                # 20+ specialized loss functions
│   ├── analyzer/              # Comprehensive model analysis toolkit
│   ├── regularizers/          # Advanced regularization techniques
│   └── utils/                 # Core utilities and data handlers
├── docs/                      # Auto-generated API documentation
├── research/                  # In-depth articles and theoretical guides
├── src/train/                 # 25+ ready-to-use training pipelines
└── tests/                     # 600+ unit and integration tests
```

---

## Contributing

We welcome contributions from the research community. Whether you are implementing a new technique, improving documentation, or fixing bugs, your input is valuable.

### Getting Started
1.  **Fork & Clone** the repository.
2.  **Set up the development environment**: `pip install -e ".[dev]"`.
3.  **Create a new branch** for your feature: `git checkout -b feature/new-technique`.
4.  **Adhere to our development standards**: Use type hints, write comprehensive tests, and document your code thoroughly.

### Development Standards
-   **Code Quality**: Follow PEP 8 guidelines. Use `black` for formatting and `isort` for import sorting.
-   **Testing**: Write comprehensive tests using `pytest`. Aim for coverage greater than 90%.
-   **Documentation**: Provide Sphinx-compliant docstrings and update relevant guides in the `research/` directory.
-   **Validation**: Include benchmarks or comparisons against reference implementations where applicable.

### Contribution Types
-   **New Architectures**: Implementations of recent, impactful research papers.
-   **Performance Improvements**: Optimizations that maintain numerical accuracy.
-   **New Analysis Tools**: Additional analyzers or visualizations for the `ModelAnalyzer` toolkit.
-   **Enhanced Documentation**: Tutorials, guides, or improved API documentation.

---

## License

This project is licensed under the **GNU General Public License v3.0**.

**Important Considerations:**
- **Copyleft License**: Any derivative works must also be licensed under GPL-3.0.
- **Enterprise Use**: Please contact us for commercial licensing options that may be better suited for enterprise environments.
- **Research Use**: The library is fully open for academic and non-commercial research applications.

See the [LICENSE](./LICENSE) file for complete details.

---

## Acknowledgments

This library is proudly sponsored by **[Electi Consulting](https://electiconsulting.com)**, a premier AI consultancy specializing in enterprise artificial intelligence, blockchain technology, and cryptographic solutions. The practical validation and enterprise-ready nature of these components has been made possible through Electi's extensive experience deploying AI solutions across:

- **Financial Services**: High-frequency trading, risk assessment, and fraud detection.
- **Maritime Industry**: Route optimization, predictive maintenance, and cargo management.
- **Healthcare**: Diagnostic assistance, treatment optimization, and clinical decision support.
- **Manufacturing**: Predictive maintenance, quality control, and supply chain optimization.

Special recognition is extended to the open-source community and the many researchers whose groundbreaking work forms the foundation of this library.

---

## Citations & References

This library builds upon extensive academic research. Our implementations are based on a rigorous study of the source papers.

<details>
<summary><b>Core Architectures & Transformers</b></summary>

-   **Attention Is All You Need** (Transformers): Vaswani, A., et al. (2017)
-   **DIFFERENTIAL TRANSFORMER**: Zhu, J., et al. (2025)
-   **Mamba: Linear-Time Sequence Modeling**: Gu, A., & Dao, T. (2023)
-   **Modern Hopfield Networks**: Ramsauer, H., et al. (2020)
-   **Gemma: Open Models for Responsible Innovation**: Team, G., et al. (2024)
-   **Qwen Technical Report**: Bai, J., et al. (2023)
</details>

<details>
<summary><b>Vision & Multimodal Models</b></summary>

-   **A ConvNet for the 2020s** (ConvNeXt): Liu, Z., et al. (2022)
-   **DinoV2: Learning Robust Visual Features without Supervision**: Oquab, M., et al. (2023)
-   **Sigmoid Loss for Language Image Pre-Training** (SigLIP): Zhai, X., et al. (2023)
-   **Learning Transferable Visual Models From Natural Language Supervision** (CLIP): Radford, A., et al. (2021)
-   **End-to-End Object Detection with Transformers** (DETR): Carion, N., et al. (2020)
</details>

<details>
<summary><b>Graph Neural Networks & Advanced Architectures</b></summary>

-   **Graph Neural Networks: A Review**: Wu, Z., et al. (2020)
-   **Graph Attention Networks**: Veličković, P., et al. (2018)
-   **Dynamic Routing Between Capsules**: Sabour, S., et al. (2017)
-   **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer**: Shazeer, N., et al. (2017)
</details>

<details>
<summary><b>Time Series & Forecasting</b></summary>

-   **N-BEATS: Neural basis expansion analysis for interpretable time series forecasting**: Oreshkin, B. N., et al. (2019)
-   **Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting**: Lim, B., et al. (2021)
-   **DeepAR: Probabilistic forecasting with autoregressive recurrent networks**: Salinas, D., et al. (2020)
</details>

<details>
<summary><b>Loss Functions & Optimization</b></summary>

-   **AnyLoss: A General and Differentiable Framework for Classification Metric Optimization**: Han, D., et al. (2024)
-   **Focal Loss for Dense Object Detection**: Lin, T. Y., et al. (2017)
-   **On calibration of modern neural networks**: Guo, C., et al. (2017)
-   **Wasserstein GAN**: Arjovsky, M., et al. (2017)
</details>

*Complete bibliographic information is available in the documentation for individual modules.*

---

*Built for the deep learning research community*