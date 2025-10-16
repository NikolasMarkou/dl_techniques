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
  <img src="https://raw.github.com/nikolasmarkou/dl_techniques/main/imgs/logo_v2.png" alt="Deep Learning Techniques Logo" width="350">
</p>

In the rapidly evolving landscape of AI research, groundbreaking techniques are often scattered across countless repositories, disparate implementations, and dense academic papers. **dl_techniques** emerges as a unified, curated, and production-ready arsenal for advanced deep learning. Pioneered and sponsored by [Electi Consulting](https://electiconsulting.com), this library is more than a collection of layers—it is the definitive toolkit for researchers and engineers to design, train, and dissect state-of-the-art neural networks.

We bridge the chasm between theoretical innovation and practical application, providing faithful, efficient, and enterprise-validated components. From next-generation attention mechanisms and graph neural networks to information-theoretic loss functions and an unparalleled model analysis suite, `dl_techniques` is your strategic advantage for pushing the boundaries of deep learning.

---

## Table of Contents
1.  [**Key Features**](#key-features): A Glimpse into the Arsenal.
2.  [**Why `dl_techniques`?**](#why-dl_techniques): Your Unfair Advantage in AI R&D.
3.  [**Installation**](#installation): Get Up and Running in Minutes.
4.  [**Quick Start**](#quick-start): See the Library in Action.
5.  [**In-Depth Documentation**](#in-depth-documentation): From Theory to Tensors.
6.  [**Project Structure**](#project-structure): A Tour of the Repository.
7.  [**Contributing**](#contributing): Join Our Research & Development Efforts.
8.  [**License**](#license): Understanding the GPL-3.0 License.
9.  [**Acknowledgments**](#acknowledgments): Recognition and Support.
10. [**Citations & References**](#citations--references): The Research That Inspired This Library.

---

## Key Features

This library is a comprehensive suite of tools organized into five key pillars, developed through rigorous research and validated in real-world enterprise applications:

<details>
<summary><b>1. State-of-the-Art Architectures & Models (130+ Implementations)</b></summary>
<p>

- **Battle-Tested Language Models**: Production-ready implementations of **`Gemma 3`**, **`Qwen3`**, **`Mamba`**, and modern **`BERT`** variants featuring Block-wise Latent Transformers (BLT) and Hierarchical Reasoning.
- **Vision & Multimodal Powerhouses**: A comprehensive suite including **`CLIP`**, **`FastVLM`**, **`NanoVLM`**, **`DINOv1/v2/v3`**, **`SigLIP-ViT`**, and the seminal object detection model **`DETR`**.
- **Next-Generation CNNs**: Advanced convolutional architectures such as **`ConvNeXtV1/V2`**, **`MobileNetV4`**, the recursively-defined **`FractalNet`**, **`CoShNet`**, and the ultra-efficient **`SqueezeNet`** family.
- **Specialized Architectures**: Task-specific models like **`DepthAnything`** for monocular depth estimation, a complete **`VAE`** for generative modeling, and full **`Capsule Networks`** (CapsNet) with dynamic routing.
- **Advanced Time Series & Forecasting**: State-of-the-art forecasting models including **`TiRex`** with quantile prediction and an enhanced implementation of **`N-BEATS`**.
- **Experimental Frontiers**: Explore novel concepts including **`Graph Neural Networks`** (GNNs) and Kolmogorov-Arnold Networks (**`KAN`**).
</p>
</details>

<details>
<summary><b>2. A Modular Arsenal of Advanced Layers (290+ Components)</b></summary>
<p>

- **Pioneering Attention Mechanisms**: Go beyond standard attention with **`DifferentialMultiHeadAttention`**, modern **`HopfieldAttention`**, **`GroupQueryAttention`**, and efficient alternatives like `FNetFourierTransform`.
- **Unified Factory Architecture**: A consistent, powerful factory system for creating and validating over **15 attention mechanisms**, **15+ normalization variants**, and multiple Feed-Forward Network (**FFN**) types with a single line of code.
- **Graph & Structural Primitives**: Configurable **GNN layers** with multiple aggregation strategies, **`Relational Graph Transformer`** (RELGT) blocks, and **`Entity-Graph Refinement`** for learning hierarchical relationships.
- **Mixture of Experts (MoE) System**: A complete MoE implementation with configurable FFN experts, multiple gating strategies (including **SoftMoE**), and integrated training utilities.
- **Probabilistic & Statistical Layers**: Build models that reason about uncertainty with **`Mixture Density Networks`** (MDN), **`Normalizing Flows`**, and time series analysis layers for residual autocorrelation.
</p>
</details>

<details>
<summary><b>3. A Unified Command Center for Model Analysis & Introspection</b></summary>
<p>

- **Holistic Model Analysis**: A powerful `ModelAnalyzer` to benchmark models across six critical dimensions: training dynamics, weight health, prediction calibration, information flow, and advanced spectral analysis.
- **Publication-Ready Visualizations**: Automatically generate insightful visualizations, interactive summary dashboards, and comparative analysis plots with integrated statistical significance testing.
- **Predictive Generalization with Spectral Analysis**: Integrate the power of **WeightWatcher** to assess generalization potential by analyzing the spectral properties (eigenvalues) of weight matrices—often without needing test data.
- **Specialized Diagnostic Tools**: Deep-dive modules for diagnosing overconfidence (`CalibrationAnalyzer`), information bottlenecks (`InformationFlowAnalyzer`), weight decay (`WeightAnalyzer`), and learning efficiency (`TrainingDynamicsAnalyzer`).
</p>
</details>

<details>
<summary><b>4. Next-Generation Loss Functions & Optimization (20+ Specialized Losses)</b></summary>
<p>

- **Optimize What Matters with `AnyLoss`**: A groundbreaking framework that transforms any confusion-matrix-based metric (e.g., F1-score, Balanced Accuracy) into a differentiable loss function for direct optimization.
- **Information-Theoretic & Robust Losses**: Train more robust models with `GoodhartAwareLoss` to combat spurious correlations, calibration-focused losses like `BrierScoreLoss`, and the uncertainty-aware `FocalUncertaintyLoss`.
- **Domain-Specific Loss Functions**: Specialized losses for vision-language (`CLIPContrastiveLoss`), segmentation (`Dice`, `Focal`, `Tversky`), time series (`MASELoss`, `SMAPELoss`), and generative modeling (`WassersteinLoss` with gradient penalty).
- **Advanced Optimization Suite**: Leverage smart learning rate schedulers like `WarmupSchedule`, utilities for `DeepSupervision` in multi-scale architectures, and a suite of advanced regularizers (`SoftOrthogonal`, `SRIP`).
</p>
</details>

<details>
<summary><b>5. Enterprise-Grade Training & Deployment Infrastructure</b></summary>
<p>

- **Accelerated Development with Training Pipelines**: Over 25 ready-to-use training scripts for all major architectures, establishing standardized and reproducible workflows for training, validation, and testing.
- **Production-Ready Utilities**: A suite of tools including advanced data loaders, augmentation pipelines, a structured visualization and logging manager, and enhanced model serialization with custom object support.
- **Assured Reliability**: An extensive suite of over 600 unit and integration tests ensures the correctness and stability of every component.
- **Validated Performance**: Rigorous benchmarks against reference implementations and established academic results to guarantee numerical accuracy and performance.
</p>
</details>

---

## Why `dl_techniques`?

*   **From Theory to Tensors, Instantly**: We consolidate the fragmented landscape of AI research. Instead of hunting down dozens of disparate GitHub repos, you get a single, cohesive framework with faithful implementations of cutting-edge research.
*   **Built for Battle, Validated in the Enterprise**: This is not a toy library. Every component has been hardened and validated in demanding enterprise applications, ensuring robustness, efficiency, and production-readiness.
*   **Unprecedented Introspection**: Move beyond accuracy scores. Our first-class analysis toolkit is designed to answer the *why* behind your model's behavior, providing deep insights that are critical for building trustworthy AI.
*   **Engineered for Experimentation**: Our innovative factory patterns and modular design are built for rapid prototyping. Swap attention mechanisms, normalization layers, or even entire architectural blocks with a single line of code.
*   **Modern, Maintainable, and Future-Proof**: Built from the ground up for Keras 3 and modern Python, `dl_techniques` adheres to the highest standards of software engineering, ensuring it's easy to use, extend, and maintain.

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

### 1. Compose a State-of-the-Art Transformer Block

Effortlessly construct and experiment with modern transformer components using our unified factory system.

```python
import keras
from dl_techniques.layers.attention.factory import create_attention_layer
from dl_techniques.layers.norms.factory import create_normalization_layer
from dl_techniques.layers.ffn.factory import create_ffn_layer

inputs = keras.Input(shape=(1024, 512))

# Use factories for consistent, validated component creation
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

### 2. Deploy a Probabilistic Time Series Forecaster

Instantiate a state-of-the-art time series model capable of generating robust, uncertainty-aware forecasts.

```python
from dl_techniques.models.tirex import create_tirex_model

# Create a TiRex model for multivariate probabilistic forecasting
model = create_tirex_model(
    input_shape=(100, 10),  # 100 timesteps, 10 features
    forecast_horizon=24,
    quantiles=[0.1, 0.5, 0.9],  # Generate 80% prediction intervals
    variant='base'
)

# Compile with a quantile loss to train for uncertainty estimation
model.compile(
    optimizer='adamw',
    loss='quantile_loss',
    metrics=['mae', 'mse']
)
```

### 3. Build a Complete Vision-Language Model

Construct a powerful, efficient vision-language model for complex multimodal tasks in just a few lines.

```python
from dl_techniques.models.fastvlm import FastVLM

# Create a FastVLM instance from a predefined, optimized variant
vlm = FastVLM.from_variant(
    'base',
    vocab_size=32000,
    max_length=512,
    image_size=224
)

# The model seamlessly handles both image and text inputs
image_input = keras.Input(shape=(224, 224, 3))
text_input = keras.Input(shape=(512,), dtype='int32')

outputs = vlm([image_input, text_input])
model = keras.Model([image_input, text_input], outputs)
```

### 4. Dissect Model Behavior with the Analysis Engine

Go beyond surface-level metrics and gain deep, actionable insights into your models' performance and behavior.

```python
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig, DataInput

# Compare multiple models across a suite of deep diagnostics
models = {'TiRex_Model': tirex_model, 'Baseline_LSTM': lstm_model}
histories = {'TiRex_Model': tirex_history, 'Baseline_LSTM': lstm_history}
test_data = DataInput(x_test, y_test)

# Configure a comprehensive analysis run
config = AnalysisConfig(
    analyze_training_dynamics=True,
    analyze_calibration=True,
    analyze_weight_health=True,
    analyze_spectral=True, # Unleash spectral analysis for generalization insights
    save_plots=True,
    plot_style='publication'
)

analyzer = ModelAnalyzer(models, config=config, training_history=histories)

# Execute the complete analysis and generate a full suite of visualizations
results = analyzer.analyze(test_data)

# Access detailed, structured metrics programmatically for automated reporting
print(f"Best calibrated model (by ECE): {min(results.calibration_metrics.items(), key=lambda x: x[1]['ece'])}")
print(f"Training efficiency ranking (epochs to converge): {results.training_metrics.convergence_epochs}")
```

### 5. Optimize Directly for F1-Score on Imbalanced Data

Stop tuning class weights and start optimizing your target metric directly with the `AnyLoss` framework.

```python
from dl_techniques.losses.any_loss import F1Loss, BalancedAccuracyLoss

# For imbalanced datasets, optimize F1-score directly
model.compile(
    optimizer='adamw',
    loss=F1Loss(from_logits=True),
    metrics=['accuracy', 'precision', 'recall']
)

# Alternatively, optimize for balanced accuracy
model.compile(
    optimizer='adamw',
    loss=BalancedAccuracyLoss(from_logits=True),
    metrics=['accuracy']
)
```

### 6. Harness Graph Neural Networks for Relational Data

Unlock insights from graph-structured data with our powerful and configurable GNN implementations.

```python
from dl_techniques.layers.graphs import GraphNeuralNetworkLayer

# Create a Graph Attention Network (GAT) to process relational data
gnn = GraphNeuralNetworkLayer(
    concept_dim=256,
    num_layers=3,
    message_passing='gat',  # Use attention for message passing
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

This library is engineered to be a living knowledge base, bridging the gap between academia and industry. Our documentation goes beyond API specs to provide the theoretical foundations behind each component.

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

This library is proudly sponsored and pioneered by **[Electi Consulting](https://electiconsulting.com)**, a premier AI consultancy specializing in enterprise artificial intelligence, blockchain technology, and cryptographic solutions. The practical validation and enterprise-ready nature of these components has been made possible through Electi's extensive experience deploying AI solutions across:

- **Financial Services**: High-frequency trading, risk assessment, and fraud detection.
- **Maritime Industry**: Route optimization, predictive maintenance, and cargo management.
- **Healthcare**: Diagnostic assistance, treatment optimization, and clinical decision support.
- **Manufacturing**: Predictive maintenance, quality control, and supply chain optimization.

Special recognition is extended to the open-source community and the many researchers whose groundbreaking work forms the foundation of this library.

---

## Citations & References

This library stands on the shoulders of giants. Our implementations are grounded in a rigorous study of the source papers that have defined the field of modern deep learning.

<details>
<summary><b>Transformers & Language Models</b></summary>

-   **Attention Is All You Need** (Transformer): Vaswani, A., et al. (2017). *NeurIPS*.
-   **BERT: Pre-training of Deep Bidirectional Transformers**: Devlin, J., et al. (2018). *NAACL*.
-   **RoFormer: Enhanced Transformer with Rotary Position Embedding**: Su, J., et al. (2021). *ACL*.
-   **DIFFERENTIAL TRANSFORMER**: Zhu, J., et al. (2025). *CVPR*.
-   **Mamba: Linear-Time Sequence Modeling**: Gu, A., & Dao, T. (2023).
-   **Gemma 3 Technical Report**: Google (2024).
-   **Qwen Technical Report**: Bai, J., et al. (2023).
-   **Byte Latent Transformer: Patches Scale Better Than Tokens**: Pagnoni, A., et al. (2024).
</details>

<details>
<summary><b>Vision & Multimodal Models</b></summary>

-   **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** (ViT): Dosovitskiy, A., et al. (2020). *ICLR*.
-   **A ConvNet for the 2020s** (ConvNeXt): Liu, Z., et al. (2022). *CVPR*.
-   **ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders**: Woo, S., et al. (2023). *CVPR*.
-   **MobileNetV2: Inverted Residuals and Linear Bottlenecks**: Sandler, M., et al. (2018). *CVPR*.
-   **Searching for MobileNetV3**: Howard, A., et al. (2019). *ICCV*.
-   **MobileNetV4: Universal Inverted Bottleneck and Mobile MQA**: Li, Y., et al. (2024).
-   **DINOv2: Learning Robust Visual Features without Supervision**: Oquab, M., et al. (2023).
-   **Sigmoid Loss for Language Image Pre-Training** (SigLIP): Zhai, X., et al. (2023). *ICCV*.
-   **Learning Transferable Visual Models From Natural Language Supervision** (CLIP): Radford, A., et al. (2021). *ICML*.
-   **End-to-End Object Detection with Transformers** (DETR): Carion, N., et al. (2020). *ECCV*.
-   **FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization**: Vasu, P. K. A., et al. (2023).
</details>

<details>
<summary><b>Advanced Attention & FFN Mechanisms</b></summary>

-   **Modern Hopfield Networks is All You Need**: Ramsauer, H., et al. (2020). *ICML*.
-   **GQA: Training Generalized Multi-Query Transformer Models**: Ainslie, J., et al. (2023).
-   **Ring Attention with Blockwise Transformers for Near-Infinite Context**: Liu, H., et al. (2024).
-   **Rethinking Attention with Performers**: Choromanski, K., et al. (2020).
-   **FNet: Mixing Tokens with Fourier Transforms**: Lee-Thorp, J., et al. (2021).
-   **GLU Variants Improve Transformer**: Shazeer, N. (2020).
-   **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer**: Shazeer, N., et al. (2017).
-   **Pay Attention to MLPs** (gMLP): Liu, H., et al. (2021). *NeurIPS*.
</details>

<details>
<summary><b>Graph Neural Networks & Special Architectures</b></summary>

-   **Graph Neural Networks: A Review**: Wu, Z., et al. (2020).
-   **Graph Attention Networks**: Veličković, P., et al. (2018). *ICLR*.
-   **Semi-Supervised Classification with Graph Convolutional Networks**: Kipf, T. N., & Welling, M. (2016).
-   **Dynamic Routing Between Capsules**: Sabour, S., et al. (2017). *NeurIPS*.
-   **KAN: Kolmogorov-Arnold Networks**: Liu, Z., et al. (2024).
-   **FractalNet: Ultra-Deep Neural Networks without Residuals**: Larsson, G., et al. (2016).
</details>

<details>
<summary><b>Time Series & Forecasting</b></summary>

-   **N-BEATS: Neural basis expansion analysis for interpretable time series forecasting**: Oreshkin, B. N., et al. (2019). *ICLR*.
-   **Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting**: Lim, B., et al. (2021).
-   **DeepAR: Probabilistic forecasting with autoregressive recurrent networks**: Salinas, D., et al. (2020).
-   **Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift**: Kim, T., et al. (2021). *ICLR*.
</details>

<details>
<summary><b>Loss Functions, Optimization, & Regularization</b></summary>

-   **AnyLoss: A General and Differentiable Framework for Classification Metric Optimization**: Han, D., et al. (2024).
-   **Focal Loss for Dense Object Detection**: Lin, T. Y., et al. (2017). *ICCV*.
-   **On calibration of modern neural networks**: Guo, C., et al. (2017). *ICML*.
-   **Wasserstein GAN**: Arjovsky, M., et al. (2017). *ICML*.
-   **Root Mean Square Layer Normalization**: Zhang, B., & Sennrich, R. (2019). *NeurIPS*.
-   **Can We Gain More from Orthogonality Regularizations in Training Deep Networks?**: Bansal, N., et al. (2018). *NeurIPS*.
-   **Predicting the Generalization Gap in Deep Networks with Margin Distributions**: Martin, C., & Mahoney, M. W. (2019). *ICLR*.
</details>

*Complete bibliographic information is available in the documentation for individual modules.*

---