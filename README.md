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
  <img src="https://raw.githubusercontent.com/nikolasmarkou/dl_techniques/main/imgs/minimalist-2d-logo-with-a-left-to-right-_sSVDZkeKR4C_eovommSCFQ_mJekanaZTB2Nbe5dBKOnPQ.png" alt="Deep Learning Techniques Logo" width="350">
</p>

**dl_techniques** is a research-first Python library built for Keras 3 and TensorFlow, developed with the support of [Electi Consulting](https://electiconsulting.com)'s AI research initiatives. It's more than a collection of layers; it's a curated toolkit for researchers and advanced practitioners to design, train, and dissect state-of-the-art neural networks. This library bridges the gap between groundbreaking research papers and practical implementation, providing faithful and efficient components that are ready for experimentation and production deployment.

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
<summary><b>1. Cutting-Edge Architectures & Models (25+ Complete Models)</b></summary>

**🤖 Modern Language Models:**
- **State Space Models**: Complete `Mamba` implementation with efficient sequence modeling
- **Modern BERT Variants**: `ModernBERT` with BLT (Block-wise Learning) and Hierarchical Reasoning integration
- **Gemma 3**: Faithful `Gemma3-270M` implementation with dual normalization patterns
- **Text Generation**: Advanced `TextDecoder` models with configurable architectures

**🖼️ Vision & Multimodal Models:**
- **Vision-Language Models**: `FastVLM`, `NanoVLM`, and comprehensive `CLIP` implementations
- **Advanced Vision Transformers**: `DinoV3`, `ViT-HMLP`, `SigLIP-ViT`, and specialized encoders
- **State-of-the-Art CNNs**: `ConvNeXtV1/V2`, `MobileNetV4`, `FractalNet`, and `CoShNet`
- **Specialized Vision**: `DepthAnything` for monocular depth estimation, `VAE` for generative modeling

**📈 Time Series & Forecasting:**
- **TiRex**: Advanced time series forecasting with quantile prediction and mixed sequential processing
- **N-BEATS**: Enhanced implementation with modern optimizations
- **Specialized Components**: Adaptive lag attention, residual autocorrelation analysis

**🧠 Experimental Architectures:**
- **Graph Neural Networks**: Complete GNN implementations with multiple message-passing variants
- **Capsule Networks**: Full `CapsNet` with dynamic routing
- **Holographic Networks**: `HolographicMPS` with entropy-guided architecture
</details>

<details>
<summary><b>2. Advanced Layer Components (200+ Specialized Layers)</b></summary>

**🔥 Next-Generation Attention:**
- **Differential Attention**: `DifferentialMultiHeadAttention` from recent transformer advances
- **Hopfield Networks**: Modern `HopfieldAttention` with iterative updates
- **Specialized Variants**: `GroupQueryAttention`, `MobileMQA`, `NonLocalAttention`, `PerceiverAttention`
- **Efficient Alternatives**: `FNetFourierTransform` for parameter-free mixing

**🏭 Factory Patterns for Consistency:**
- **Attention Factory**: Unified creation and validation of attention mechanisms
- **Normalization Factory**: Consistent access to 15+ normalization variants
- **FFN Factory**: Streamlined feed-forward network component management

**🌐 Graph & Structural Components:**
- **Graph Neural Networks**: Configurable GNN layers with multiple aggregation strategies
- **Relational Graph Transformers**: `RELGT` blocks for complex relational reasoning
- **Entity-Graph Refinement**: Hierarchical relationship learning in embedding space

**🔢 Mixture of Experts (MoE):**
- **Complete MoE System**: Configurable expert networks with multiple gating strategies
- **Expert Types**: FFN experts, cosine gating, SoftMoE implementations
- **Training Integration**: Specialized optimizers and auxiliary loss computation

**📊 Statistics & Analysis Layers:**
- **Mixture Density Networks**: `MDNLayer` for probabilistic predictions
- **Normalizing Flows**: Conditional density estimation with affine coupling
- **Time Series Analysis**: Residual ACF layers, moving statistics, quantile heads
</details>

<details>
<summary><b>3. Comprehensive Analysis & Introspection Toolkit</b></summary>

**🔬 Multi-Dimensional Model Analysis:**
- **Unified ModelAnalyzer**: Compare multiple models across 6 key dimensions simultaneously
- **Training Dynamics**: Convergence analysis, overfitting detection, learning curve insights
- **Weight Health Analysis**: SVD-based generalization metrics, weight distribution studies
- **Calibration Assessment**: ECE, Brier score, reliability diagrams with confidence analysis

**📈 Advanced Visualization Suite:**
- **Publication-Ready Plots**: Automated generation of research-quality visualizations
- **Interactive Dashboards**: Summary dashboards with pareto analysis for model selection
- **Information Flow Tracking**: Layer-by-layer activation and gradient flow analysis
- **Comparative Analysis**: Side-by-side model comparison with statistical significance testing

**🎯 Specialized Analyzers:**
- **Calibration Analyzer**: Deep dive into prediction confidence and reliability
- **Information Flow Analyzer**: Effective rank analysis and activation health metrics
- **Weight Analyzer**: Comprehensive weight statistics and distribution analysis
- **Training Dynamics Analyzer**: Learning efficiency and convergence pattern analysis
</details>

<details>
<summary><b>4. Advanced Loss Functions & Optimization (25+ Specialized Losses)</b></summary>

**🎯 Direct Metric Optimization:**
- **AnyLoss Framework**: Transform any confusion-matrix-based metric into differentiable loss
- **Specialized Implementations**: `F1Loss`, `BalancedAccuracyLoss`, `GeometricMeanLoss`

**🛡️ Information-Theoretic & Robust Losses:**
- **GoodhartAwareLoss**: Combat spurious correlations with entropy regularization
- **Calibration Losses**: `BrierScoreLoss`, `SpiegelhalterZLoss` for trustworthy predictions
- **Uncertainty-Aware**: `FocalUncertaintyLoss` combining focal loss with uncertainty quantification

**🔄 Task-Specific Loss Functions:**
- **Vision-Language**: `CLIPContrastiveLoss`, `SigLIPContrastiveLoss`, `NanoVLMLoss`
- **Segmentation**: Comprehensive segmentation loss suite with Dice, Focal, Tversky variants
- **Time Series**: `MASELoss`, `SMAPELoss`, `MQLoss` for forecasting applications
- **Generative**: `WassersteinLoss` with gradient penalty for GAN training

**⚙️ Advanced Optimization Tools:**
- **Smart Scheduling**: `WarmupSchedule` with configurable warmup strategies
- **DeepSupervision**: Multi-scale architecture training utilities
- **Regularization Suite**: `SoftOrthogonal`, `SRIP`, entropy-based regularizers
</details>

<details>
<summary><b>5. Production-Ready Training Infrastructure</b></summary>

**🏋️ Complete Training Pipelines:**
- **25+ Model Training Scripts**: Ready-to-use training pipelines for all major architectures
- **Standardized Workflows**: Consistent training, validation, and testing procedures
- **Hyperparameter Management**: Integrated configuration and experiment tracking

**🔧 Utilities & Tools:**
- **Data Handling**: Advanced data loaders, augmentation pipelines, normalization utilities
- **Visualization Manager**: Structured logging and plot generation system
- **Model Serialization**: Enhanced save/load utilities with custom object support

**📊 Comprehensive Testing:**
- **600+ Unit Tests**: Extensive test coverage ensuring reliability
- **Integration Tests**: End-to-end validation of training pipelines
- **Performance Benchmarks**: Validation against reference implementations
</details>

---

## Why `dl_techniques`?

*   **Research-Driven Excellence**: Each component is selected for its significance, novelty, and potential impact, not just popularity. Components are implemented based on thorough understanding of the underlying research.

*   **Enterprise-Validated**: All components have been tested and validated in real-world enterprise environments through Electi Consulting's AI implementations across finance, maritime, and healthcare industries.

*   **Factory Pattern Architecture**: Innovative factory systems ensure consistency, reduce boilerplate, and make component swapping effortless for experimentation.

*   **Deep Introspection First-Class**: Understanding *why* a model works is as important as its accuracy. Our integrated analysis tools provide unprecedented insights into model behavior.

*   **Modern Keras 3 Design**: Built from the ground up for Keras 3 with proper type hints, Sphinx documentation, and modern Python practices.

*   **Extensible by Design**: Modular architecture allows easy integration of new components while maintaining backward compatibility.

---

## Installation

> **Note:** This library requires Python 3.11+ and Keras 3.8.0 with TensorFlow 2.18.0 backend.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/nikolasmarkou/dl_techniques.git
    cd dl_techniques
    ```

2.  **Install dependencies:**
    For standard usage, install the library and its dependencies directly:
    ```bash
    pip install .
    ```

3.  **Editable Install (for developers):**
    If you plan to contribute or modify the code:
    ```bash
    pip install -e ".[dev]"
    ```
    This installs development tools like `pytest`, `pylint`, and `black`.

4.  **Verify Installation:**
    ```bash
    python -c "import dl_techniques; print('Installation successful!')"
    ```

---

## Quick Start

### 1. Build a Modern Architecture with Factory Patterns

Use the new factory systems to easily create and swap components:

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

Create a state-of-the-art time series model with quantile predictions:

```python
from dl_techniques.models.tirex import create_tirex_model

# Create a TiRex model for multivariate forecasting
model = create_tirex_model(
    input_shape=(100, 10),  # 100 timesteps, 10 features
    forecast_horizon=24,
    quantiles=[0.1, 0.5, 0.9],  # Probabilistic forecasting
    variant='base'
)

# The model supports quantile loss and uncertainty estimation
model.compile(
    optimizer='adamw',
    loss='quantile_loss',
    metrics=['mae', 'mse']
)
```

### 3. Vision-Language Modeling with FastVLM

Build a complete vision-language model:

```python
from dl_techniques.models.fastvlm import FastVLM

# Create a fast vision-language model
vlm = FastVLM.from_variant(
    'base',
    vocab_size=32000,
    max_length=512,
    image_size=224
)

# Supports both image and text inputs
image_input = keras.Input(shape=(224, 224, 3))
text_input = keras.Input(shape=(512,))

outputs = vlm([image_input, text_input])
model = keras.Model([image_input, text_input], outputs)
```

### 4. Comprehensive Model Analysis

Get deep insights with our advanced analysis toolkit:

```python
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig, DataInput

# Compare multiple models comprehensively  
models = {'TiRex': tirex_model, 'LSTM': lstm_model, 'Transformer': transformer_model}
histories = {'TiRex': tirex_history, 'LSTM': lstm_history, 'Transformer': transformer_history}

# Configure comprehensive analysis
config = AnalysisConfig(
    analyze_training_dynamics=True,
    analyze_calibration=True, 
    analyze_weight_health=True,
    analyze_information_flow=True,
    save_plots=True,
    plot_style='publication'
)

test_data = DataInput(x_test, y_test)
analyzer = ModelAnalyzer(models, config=config, training_history=histories)

# Run complete analysis with publication-ready visualizations
results = analyzer.analyze(test_data)

# Access detailed insights
print(f"Best calibrated model: {min(results.calibration_metrics.items(), key=lambda x: x[1]['ece'])}")
print(f"Training efficiency ranking: {results.training_metrics.convergence_epochs}")
```

### 5. Direct F1-Score Optimization

Use the AnyLoss framework to optimize directly for your target metric:

```python
from dl_techniques.losses.any_loss import F1Loss, BalancedAccuracyLoss

# For imbalanced datasets, optimize F1-score directly
model.compile(
    optimizer='adamw',
    loss=F1Loss(from_logits=True),  # Direct F1 optimization
    metrics=['accuracy', 'precision', 'recall']
)

# Or use balanced accuracy for better class balance
model.compile(
    optimizer='adamw',
    loss=BalancedAccuracyLoss(from_logits=True),
    metrics=['accuracy', 'f1_score']
)
```

### 6. Graph Neural Networks for Relational Data

Work with graph-structured data using our GNN implementations:

```python
from dl_techniques.layers.graphs import GraphNeuralNetworkLayer

# Create a configurable GNN layer
gnn = GraphNeuralNetworkLayer(
    concept_dim=256,
    num_layers=3,
    message_passing='gat',  # Graph Attention Networks
    aggregation='attention',
    dropout_rate=0.1
)

# Use with graph-structured inputs
node_features = keras.Input(shape=(None, 256))  # Variable number of nodes
adjacency_matrix = keras.Input(shape=(None, None))

node_embeddings = gnn([node_features, adjacency_matrix])
```

---

## In-Depth Documentation

This library serves as both a practical toolkit and a knowledge repository. Comprehensive documentation covers both implementation details and theoretical foundations:

### Core Documentation

-   **[Complete Transformer Guide (2025)](./docs/2025_attention.md)**: Production-focused guide to implementing SOTA Transformer architectures with every critical detail
-   **[Model Analyzer Guide](./docs/analyzer/model_analyzer_guide.md)**: Comprehensive tutorial for the advanced model analysis toolkit
-   **[AnyLoss Framework](./docs/anyloss_classification_metrics_loss_functions.md)**: Deep dive into direct metric optimization
-   **[Factory Pattern Usage](./docs/factories/normalization_factory_guide.md)**: How to leverage factory patterns for consistent component creation

### Architecture & Implementation Guides

-   **[Chronological Neural Architectures](./docs/neural_network_architectures.md)**: Extensive chronological guide to influential architectures with implementation notes  
-   **[Band-Constrained Normalization](./docs/bcn_thesis.md)**: Novel normalization preserving magnitude information within bounded constraints
-   **[OrthoBlock & Orthonormal Regularization](./docs/orthoblock.md)**: Structured feature learning with orthogonal constraints
-   **[Mixture Density Networks](./docs/mdn.md)**: Theory and best practices for probabilistic modeling

### Specialized Topics

-   **[Graph Neural Networks](./docs/graphs/gnn_guide.md)**: Implementation guide for relational data modeling
-   **[Time Series Forecasting](./docs/time_series/forecasting_guide.md)**: Advanced techniques for temporal data
-   **[Vision-Language Models](./docs/multimodal/vlm_guide.md)**: Building and training multimodal systems

### Experimental Results

The [`experiments/`](./src/experiments/) directory contains validation studies and research results:

-   **Goodhart's Law Mitigation**: Testing `GoodhartAwareLoss` on spurious correlation datasets
-   **OrthoBlock Validation**: Comparative studies against baseline architectures  
-   **TiRex Forecasting**: Time series prediction benchmarks across multiple domains
-   **Model Analysis Case Studies**: Real-world applications of the analysis toolkit

---

## Project Structure

The repository is organized for clarity and maintainability:

```
src/dl_techniques/
├── layers/                 # 200+ specialized layer implementations
│   ├── attention/         # Modern attention mechanisms with factory
│   ├── norms/             # Advanced normalization with factory  
│   ├── ffn/               # Feed-forward networks with factory
│   ├── graphs/            # Graph neural network components
│   ├── moe/               # Mixture of Experts implementation
│   ├── time_series/       # Temporal modeling layers
│   ├── statistics/        # Statistical and probabilistic layers
│   └── experimental/      # Research-stage implementations
├── models/                # 25+ complete architecture implementations
│   ├── tirex/            # Advanced time series forecasting
│   ├── fastvlm/          # Fast vision-language models  
│   ├── gemma3/           # Gemma 3 language model
│   ├── modern_bert/      # Modern BERT variants
│   └── mamba/            # State space models
├── losses/               # 25+ specialized loss functions
├── analyzer/             # Comprehensive model analysis toolkit
│   ├── analyzers/        # Individual analysis components
│   └── visualizers/      # Publication-ready visualization
├── utils/                # Core utilities and training infrastructure
└── weightwatcher/        # Deep model introspection tools

docs/                     # Comprehensive documentation
experiments/              # Validation studies and research
tests/                    # 600+ unit and integration tests
training_pipelines/       # Ready-to-use training scripts
```

---

## Contributing

We welcome contributions from the research community! Whether you're implementing a new technique, improving documentation, or fixing bugs:

### Getting Started
1.  **Fork & Clone** the repository
2.  **Set up development environment**: `pip install -e ".[dev]"`
3.  **Create a branch** for your feature: `git checkout -b feature/new-technique`
4.  **Follow our standards**: Use type hints, write tests, document thoroughly

### Development Standards
-   **Code Quality**: Follow PEP 8, use `black` formatting, `isort` imports
-   **Testing**: Write comprehensive tests with `pytest`, aim for >90% coverage
-   **Documentation**: Sphinx-compliant docstrings, update relevant guides
-   **Validation**: Include benchmarks or comparisons with reference implementations

### Contribution Types
-   **New Architectures**: Recent papers with solid theoretical foundations
-   **Performance Improvements**: Optimizations maintaining numerical accuracy  
-   **Analysis Tools**: New analyzers or visualizations for model understanding
-   **Documentation**: Tutorials, guides, or improved API documentation

---

## License

This project is licensed under **GNU General Public License v3.0**.

**Important Considerations:**
- **Copyleft License**: Derivative works must also use GPL-3.0
- **Enterprise Use**: Contact us for commercial licensing options
- **Research Use**: Fully open for academic and research applications

See [LICENSE](./LICENSE) for complete details.

---

## Acknowledgments

**Proudly sponsored by [Electi Consulting](https://electiconsulting.com)** - a premier AI consultancy specializing in enterprise artificial intelligence, blockchain technology, and cryptographic solutions. The practical validation and enterprise-ready nature of these components has been made possible through Electi's extensive experience deploying AI solutions across:

- **Financial Services**: High-frequency trading, risk assessment, fraud detection
- **Maritime Industry**: Route optimization, predictive maintenance, cargo management  
- **Healthcare**: Diagnostic assistance, treatment optimization, clinical decision support
- **Manufacturing**: Predictive maintenance, quality control, supply chain optimization

Special recognition to the open-source community and researchers whose groundbreaking work forms the foundation of this library.

---

## Citations & References

This library builds upon extensive academic research. Our implementations are based on rigorous study of the source papers:

<details>
<summary><b>Core Architectures & Transformers</b></summary>

-   **Attention Is All You Need** (Transformers): Vaswani, A., et al. (2017)
-   **DIFFERENTIAL TRANSFORMER**: Zhu, J., et al. (2025) 
-   **Mamba: Linear-Time Sequence Modeling**: Gu, A., & Dao, T. (2023)
-   **Modern Hopfield Networks**: Ramsauer, H., et al. (2020)
-   **Gemma: Open Weights and Strong Performance**: Team, G., et al. (2024)
</details>

<details>
<summary><b>Vision & Multimodal Models</b></summary>

-   **A ConvNet for the 2020s** (ConvNeXt): Liu, Z., et al. (2022)
-   **DinoV2: Learning Robust Visual Representations**: Oquab, M., et al. (2023)
-   **Sigmoid Loss for Language Image Pre-Training** (SigLIP): Zhai, X., et al. (2023)
-   **CLIP: Connecting Text and Images**: Radford, A., et al. (2021)
</details>

<details>
<summary><b>Graph Neural Networks & Advanced Architectures</b></summary>

-   **Graph Neural Networks: A Review**: Wu, Z., et al. (2020)
-   **Graph Attention Networks**: Veličković, P., et al. (2018)
-   **Dynamic Routing Between Capsules**: Sabour, S., et al. (2017)
-   **Mixture of Experts**: Shazeer, N., et al. (2017)
</details>

<details>
<summary><b>Time Series & Forecasting</b></summary>

-   **N-BEATS: Neural basis expansion**: Oreshkin, B. N., et al. (2019)
-   **Temporal Fusion Transformers**: Lim, B., et al. (2021)
-   **DeepAR: Probabilistic forecasting**: Salinas, D., et al. (2020)
</details>

<details>
<summary><b>Loss Functions & Optimization</b></summary>

-   **AnyLoss: Transforming Classification Metrics**: Han, D., et al. (2024)
-   **Focal Loss for Dense Object Detection**: Lin, T. Y., et al. (2017)
-   **Calibration of Probabilities**: Platt, J. (1999)
-   **Wasserstein GAN**: Arjovsky, M., et al. (2017)
</details>

*Complete bibliographic information available in individual module documentation.*

---

*Built with ❤️ for the deep learning research community*