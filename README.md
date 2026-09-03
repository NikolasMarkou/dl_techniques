# A Curated Arsenal of Advanced Deep Learning Techniques

<p align="center">
  <a href="https://github.com/nikolasmarkou/dl_techniques/blob/main/LICENSE">
    <img alt="License: GPL-3.0" src="https://img.shields.io/badge/License-GPL_v3-blue.svg">
  </a>
  <img alt="Python Version" src="https://img.shields.io/badge/python-3.11+-blue.svg">
  <img alt="Framework" src="https://img.shields.io/badge/Keras-3.8-red">
  <img alt="Backend" src="https://img.shields.io/badge/TensorFlow-2.18-orange">
  <a href="https://deepwiki.com/NikolasMarkou/dl_techniques/1-overview">
    <img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
  <a href="https://electiconsulting.com">
    <img alt="Sponsored by Electi Consulting" src="https://img.shields.io/badge/Sponsored%20by-Electi%20Consulting-8B1C34">
  </a>
</p>

<p align="center">
  <img src="https://raw.github.com/nikolasmarkou/dl_techniques/main/imgs/logo_v2.png" alt="Deep Learning Techniques Logo" width="350">
</p>

In the rapidly evolving landscape of AI research, groundbreaking techniques are often scattered across countless repositories, disparate implementations, and dense academic papers. **dl_techniques** emerges as a unified, curated, and production-ready arsenal for advanced deep learning. Pioneered and sponsored by [Electi Consulting](https://electiconsulting.com), this library is more than a collection of components—it is the definitive toolkit for researchers and engineers to design, train, and dissect state-of-the-art neural networks.

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
<summary><b>1. State-of-the-Art Architectures & Models (80 Model Packages)</b></summary>
<p>

- **Next-Generation Language Models**: Production-ready implementations of **`BERT`**, **`Gemma 3`**, **`Qwen3`** (dense, `Qwen3Next` and embeddings), **`Mamba`**, **`ModernBERT`**, **`DistilBERT`** and **`GPT-2`**, alongside the **`Byte Latent Transformer`** (BLT), the **`Hierarchical Reasoning Model`** (HRM), the **`Tiny Recursive Model`**, **`Tree Transformer`**, late-interaction retrieval with **`ColBERT`** (v1 and v2), and Fourier token mixing in **`FNet`** and **`FFTNet`**.
- **Vision & Multimodal Powerhouses**: A comprehensive suite including **`CLIP`**, **`MobileCLIP`** v1 and v2 in one package (v2 is the faithful port, on a faithful **`FastViT`** image tower; v1 is deliberately non-faithful on the image side and is kept, not deprecated), **`FastVLM`** (vision-only despite the name), **`NanoVLM`**, **`DINOv1/v2/v3`**, **`ViT-SigLIP`** (a ViT with a two-stage conv patch stem, not SigLIP's sigmoid contrastive objective), **`BEiT`**, **`Swin Transformer`**, **`MAE`**, **`Video-JEPA`**, the latent-energy world model **`LEWM`**, object detection with **`DETR`** and **`YOLOv12`**, keypoints with **`SuperPoint`**, and **`Segment Anything`** (SAM 1, 2 and 3). Three packages are named for something they are not; each is listed with its measured correction in [`src/dl_techniques/models/README.md`](./src/dl_techniques/models/README.md) § *Names that misattribute*.
- **Advanced CNNs**: A rich collection of advanced convolutional architectures such as **`ConvNeXtV1/V2`**, **`ConvUNeXt`**, **`MobileNetV1-V4`**, **`ResNet`**, the recursively-defined **`FractalNet`**, the complex shearlet-based **`CoShNet`**, attention-augmented **`CBAM`**, and the ultra-efficient **`SqueezeNet`** family.
- **Time Series & Forecasting**: State-of-the-art forecasting models including the probabilistic **`TiRex`** with quantile prediction, an enhanced implementation of **`N-BEATS`**, autoregressive **`DeepAR`**, **`PRISM`**, and the novel **`xLSTM`**.
- **Generative Modeling & Image Restoration**: Diffusion and flow-matching transformers (**`SD3 MMDiT`**, **`Ideogram4`**), a complete **`Variational Autoencoder (VAE)`** framework with **`VQ-VAE`** variants (including rotation-based codebook updates), arbitrary-scale super-resolution (**`THERA`**, **`PFT-SR`**), and restoration/denoising backbones (**`DarkIR`**, **`SCUNet`**, **`ACC-UNet`**, `PW-FNet`, and a family of bias-free denoisers).
- **Specialized Models**: Task-specific models like **`DepthAnything`** for monocular depth estimation, full **`Capsule Networks`** (CapsNet) with dynamic routing, **`TabM`** for tabular data, model-agnostic inference-time `power_sampling` for any causal LM/VLM, and unsupervised aligners like `Mini-Vec2Vec`.
- **Experimental Frontiers**: Explore novel concepts including a wide range of **`Graph Neural Networks`** (GNNs) in `RELGT` and the Simplified Hyperbolic GCN `SHGCN`, the **`Energy Transformer`** family (image and graph domains), geometric-algebra networks in **`CliffordNet`**, Kolmogorov-Arnold Networks (**`KAN`**) and `PowerMLP`, external-memory computers (**`NTM`**, the Neural Arithmetic **Module** `NAM`), **`Self-Organizing Maps`**, point-cloud `latent_gmm_registration`, and bio-mimetic models like `MothNet`.
</p>
</details>

<details>
<summary><b>2. A Modular Arsenal of Advanced Layers (275 modules, 21 subpackages)</b></summary>
<p>

- **Pioneering Attention Mechanisms**: Go beyond standard attention with **`DifferentialMultiHeadAttention`**, modern **`HopfieldAttention`**, **`GroupQueryAttention`**, `CapsuleRoutingAttention`, and efficient alternatives like `FNetFourierTransform` and `RingAttention`.
- **Unified Factory Architecture**: A consistent, powerful factory system for creating and validating **33 attention mechanisms**, **18 normalization variants** (including `BandRMS`, `LogitNorm`), and **21 Feed-Forward Network (FFN)** types (`SwiGLU`, `GeGLU`, `OrthoGLU`) with a single line of code — plus registries for **22 activations**, **13 embeddings**, and task heads dispatched by domain via `create_head`.
- **Graph & Structural Primitives**: Configurable **GNN layers** with multiple aggregation strategies (`GCN`, `GAT`, `GraphSAGE`), **`Relational Graph Transformer`** (RELGT) blocks, and **`Entity-Graph Refinement`** for learning hierarchical relationships.
- **Mixture of Experts (MoE) System**: A complete MoE implementation with configurable FFN experts, multiple gating strategies (including **SoftMoE** and Cosine Gating), and integrated training utilities.
- **Probabilistic & Statistical Layers**: Build models that reason about uncertainty with **`Mixture Density Networks`** (MDN), **`Normalizing Flows`**, and time series analysis layers for residual autocorrelation (`ResidualACFLayer`).
- **Advanced Embeddings**: A full suite of positional and semantic embeddings including standard, sinusoidal, **`Rotary Position Embedding (RoPE)`**, and its modern variants like `DualRotaryPositionEmbedding`.
</p>
</details>

<details>
<summary><b>3. A Unified Command Center for Model Analysis & Introspection</b></summary>
<p>

- **Holistic Model Analysis**: A powerful `ModelAnalyzer` to benchmark models across five critical dimensions: training dynamics, weight health, prediction calibration, information flow, and advanced spectral analysis. Its modular design includes specialized analyzers like `CalibrationAnalyzer`, `WeightAnalyzer`, and `InformationFlowAnalyzer`.
- **Publication-Ready Visualizations**: Automatically generate insightful visualizations, interactive summary dashboards, and comparative analysis plots with integrated statistical significance testing.
- **Predictive Generalization with Spectral Analysis**: Integrate the power of **WeightWatcher** through our `SpectralAnalyzer` to assess generalization potential by analyzing the spectral properties (eigenvalues) of weight matrices—often without needing test data.
- **Deep Diagnostic Toolkit**: Move beyond accuracy to diagnose overconfidence (`CalibrationAnalyzer`), information bottlenecks (`InformationFlowAnalyzer`), weight decay and similarity (`WeightAnalyzer`), and learning efficiency (`TrainingDynamicsAnalyzer`).
</p>
</details>

<details>
<summary><b>4. Next-Generation Loss Functions & Optimization (42 Loss Modules)</b></summary>
<p>

- **Optimize What Matters with `AnyLoss`**: A groundbreaking framework that transforms any confusion-matrix-based metric (e.g., F1-score, Balanced Accuracy, Matthews Correlation Coefficient) into a differentiable loss function for direct optimization on imbalanced data.
- **Calibration & Robust Losses**: Train better-calibrated models with `GoodhartAwareLoss` (cross-entropy plus a per-sample confidence penalty, with an optional anti-collapse term), calibration-focused losses like `BrierScoreLoss`, the uncertainty-aware `FocalUncertaintyLoss`, and `DINO`'s self-distillation loss.
- **Domain-Specific Loss Functions**: Specialized losses for vision-language (`CLIPContrastiveLoss`, `SigLIPLoss`), segmentation (`Dice`, `Focal`, `Tversky`), time series (`MASELoss`, `SMAPELoss`), and generative modeling (`WassersteinLoss` with gradient penalty).
- **Advanced Optimization Suite**: Leverage smart learning rate schedulers like `WarmupSchedule`, utilities for `DeepSupervision` in multi-scale architectures, and a suite of advanced regularizers (`SoftOrthogonal`, `SRIP`).
</p>
</details>

<details>
<summary><b>5. Enterprise-Grade Training & Deployment Infrastructure</b></summary>
<p>

- **Accelerated Development with Training Pipelines**: **70 ready-to-use `train_*.py` entry points across 47 trainer directories** (`src/train/`, plus the shared `src/train/common/` library), establishing standardized and reproducible workflows for training, validation, and testing across domains like NLP, Vision, and Time Series.
- **Production-Ready Utilities**: A suite of tools including advanced data loaders, augmentation pipelines, a structured visualization and logging manager (`VisualizationManager`), and enhanced model serialization with custom object support.
- **Assured Reliability**: An extensive **895-module test suite** (`tests/`) ensures the correctness and stability of every component, with dedicated fixtures for mixed-precision and TF32-sensitive regressions. It mirrors `src/dl_techniques/` directory-for-directory, with the deliberate exception of `tests/test_models/`, which stays flat rather than following the model family nesting.
- **Verified Against the Source**: Where a reference implementation exists, ports are checked against it numerically — several packages commit the reference itself as an executable oracle (e.g. `src/dl_techniques/layers/fastvit/reference.py`) rather than asserting parity in prose.
</p>
</details>

---

## Why `dl_techniques`?

*   **From Theory to Tensors, Instantly**: We consolidate the fragmented landscape of AI research. Instead of hunting down dozens of disparate GitHub repos, you get a single, cohesive framework with faithful implementations of cutting-edge research.
*   **Built for Battle, Validated in the Enterprise**: This is not a toy library. Every component has been hardened and validated in demanding enterprise applications, ensuring robustness, efficiency, and production-readiness.
*   **Unprecedented Introspection**: Move beyond accuracy scores. Our first-class analysis toolkit is designed to answer the *why* behind your model's behavior, providing deep insights that are critical for building trustworthy and reliable AI.
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
    This installs additional tools such as `pytest`, `pytest-cov`, `pylint`, and `pre-commit`.

4.  **Optional Extras:**
    The Streamlit front-end under `src/applications/bias_free_denoiser/` needs an extra that is *not* part of a core install:
    ```bash
    pip install ".[apps]"
    ```
    The tokenizer utilities and the HuggingFace/`tensorflow-datasets`-backed loaders under `dl_techniques.datasets` need their own extra:
    ```bash
    pip install ".[data]"
    ```

5.  **Verify Installation:**
    ```bash
    python -c "import dl_techniques; print('Installation successful!')"
    ```

### Development Commands

```bash
make test       # full pytest suite (~1.5 hours; also the pre-push hook)
make clean      # remove build artifacts and __pycache__
make structure  # print the src/ tree
```

Scope pytest to what you changed rather than running the whole suite as a regression check:

```bash
pytest tests/test_layers/test_norms/ -vvv
```

Training scripts need a non-interactive matplotlib backend on headless machines:

```bash
MPLBACKEND=Agg python -m train.<pipeline>.train_<script> [args]
```

> **No trained weights ship with this repository.** Training outputs live under `results/`, which is gitignored, so a fresh clone contains source, tests and prose only. Any `results/<run>/final_model.keras` path quoted in a trainer default, a paper or an application is a local artifact — you train it yourself first. See `REPO_MAP.md` § *What a fresh clone actually contains*.

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
    'differential',
    dim=512,
    num_heads=8,
    head_dim=64
)
norm = create_normalization_layer('rms_norm', epsilon=1e-6)
ffn = create_ffn_layer('swiglu', output_dim=512)

# Build a modern transformer block
x = attention(inputs)
x = norm(x)
x = ffn(x)

model = keras.Model(inputs, x)
model.summary()
```

### 2. Deploy a Probabilistic Time Series Forecaster

Instantiate a state-of-the-art univariate time series model capable of generating robust, uncertainty-aware forecasts.

```python
from dl_techniques.models.time_series.tirex.model import create_tirex_model
from dl_techniques.losses.quantile_loss import QuantileLoss

# TiRex: probabilistic univariate forecasting with quantile prediction
model = create_tirex_model(
    input_length=100,            # length of the input context window
    prediction_length=24,        # forecast horizon
    quantile_levels=[0.1, 0.5, 0.9],  # 80% prediction interval + median
)

# Train directly for calibrated uncertainty with a quantile loss.
# Predictions are (batch, prediction_length, n_quantiles), so a plain
# point-forecast metric like 'mae' would broadcast across the quantile axis.
model.compile(
    optimizer='adamw',
    loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
)
```

### 3. Contrastive Vision-Language Learning with CLIP

Map images and text into a single shared embedding space with a CLIP dual encoder, unlocking zero-shot classification and cross-modal retrieval.

```python
import keras
from dl_techniques.models.vision_language.clip.model import create_clip_variant

# CLIP ViT-B/32: a dual encoder mapping images and text into one space
model = create_clip_variant("ViT-B/32")
model.build({"image": (None, 224, 224, 3), "text": (None, 77)})

images = keras.random.normal((4, 224, 224, 3))
tokens = keras.ops.cast(keras.random.uniform((4, 77), 0, 49408), "int32")

# Full contrastive forward pass
outputs = model({"image": images, "text": tokens}, training=False)
# outputs["image_features"]   -> (4, 512), L2-normalised
# outputs["text_features"]    -> (4, 512), L2-normalised
# outputs["logits_per_image"] -> (4, 4),  similarity matrix

# Or encode each modality independently for retrieval
image_features = model.encode_image(images)
text_features = model.encode_text(tokens)
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
    analyze_weights=True,
    analyze_spectral=True, # Unleash spectral analysis for generalization insights
    save_plots=True,
    plot_style='publication'
)

analyzer = ModelAnalyzer(models, config=config, training_history=histories)

# Execute the complete analysis and generate a full suite of visualizations
results = analyzer.analyze(test_data)

# Access detailed, structured metrics programmatically for automated reporting
print(f"Best calibrated model (by ECE): {min(results.calibration_metrics.items(), key=lambda x: x[1]['ece'])}")
print(f"Training efficiency ranking (epochs to converge): {results.training_metrics.epochs_to_convergence}")
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
import keras
from dl_techniques.layers.graphs.graph_neural_network import GraphNeuralNetworkLayer

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

This library is engineered to be a living knowledge base, bridging the gap between academia and industry. Our documentation is split into three primary resources: a navigational map of the repository, in-depth theoretical guides, and a comprehensive API reference.

### Start Here: [`REPO_MAP.md`](./REPO_MAP.md)

Before anything else, read **[`REPO_MAP.md`](./REPO_MAP.md)** — a path-verified router for the repository. It answers *where code of a given kind lives*, *how the registry and factory dispatch is wired*, *which trainer trains which model*, and *which of the many in-tree docs answers your question*.

### Tutorials & Deep Dives (`research/`)

Our `research/` directory contains over 120 articles providing the theoretical foundations, implementation details, and best practices behind key components. Highlights include:

-   **[Complete Transformer Guide (2025)](./research/2025_transformer_architectures.md)**: A production-focused guide to implementing state-of-the-art Transformer architectures.
-   **[Model Analyzer Guide](./src/dl_techniques/analyzer/README.md)**: A comprehensive tutorial for the advanced model analysis toolkit.
-   **[AnyLoss Framework](./research/anyloss_classification_metrics_loss_functions.md)**: A deep dive into the theory and practice of direct metric optimization.
-   **[Chronological Neural Architectures](./research/neural_network_architectures.md)**: An extensive chronological guide to influential architectures with implementation notes.
-   **[Band-Constrained Normalization](./research/bcn_thesis.md)**: A novel normalization technique that preserves magnitude information within bounded constraints.
-   **[Mixture Density Networks](./research/mdn.md)**: The theory and best practices for implementing probabilistic models.

### Papers (`research/papers/`)

Five LaTeX manuscripts written against this codebase live under [`research/papers/`](./research/papers), several with built PDFs: `band_rms` (band-constrained RMS normalization), `bfunet` (bias-free denoisers as image priors), `cliffordnet_extensions`, `correlations`, and `logical_net`.

### API Reference (per-module docs)
There is **no committed documentation directory and no doc generator** — `generate_docs.py` and the `make docs` target were deleted as deprecated. For detailed documentation on every module, class, and function, browse the source tree directly: each subpackage ships a focused `README.md` (e.g. [`src/dl_techniques/analyzer/README.md`](./src/dl_techniques/analyzer/README.md)) and a per-package `CLAUDE.md` describing its conventions, patterns, and components. Every one of the 84 leaf model packages carries its own `README.md`. The `research/` guides above complement these with the underlying theory.

---

## Project Structure

The repository is organized for clarity, maintainability, and ease of contribution.

`src/` is the import root: the library is imported as `dl_techniques.*`, the training pipelines as `train.*`, and the applications as `applications.*`.

```
dl_techniques/
├── src/dl_techniques/         # THE LIBRARY — 13 subpackages
│   ├── models/                # 80 leaf model packages in 11 FAMILY directories
│   │   │                      # full catalogue: src/dl_techniques/models/README.md
│   │   ├── vision/            # 35 — resnet, convnext, vit, dino, swin_transformer, beit,
│   │   │                      #      yolo12, detr, vae, vq_vae, depth_anything, and the
│   │   │                      #      nested image_restoration/, super_resolution/, keypoints/
│   │   ├── language/          # 17 — bert, gemma, qwen, mamba, modern_bert, gpt2, colbert,
│   │   │                      #      byte_latent_transformer, hierarchical_reasoning_model, ...
│   │   ├── vision_language/   #  9 — clip, mobile_clip, fastvlm, nano_vlm, sd3_mmdit,
│   │   │                      #      ideogram4, and the nested sam/{sam1,sam2,sam3}
│   │   ├── time_series/       #  7 — tirex, nbeats, xlstm, deepar, prism, mdn, adaptive_ema
│   │   ├── general_purpose/   #  3 — kan, mothnet, power_mlp
│   │   ├── graph/             #  3 — relgt, graph_energy_transformer, shgcn
│   │   ├── neural_computer/   #  2 — ntm, nam
│   │   ├── common/            #  1 — power_sampling (model-agnostic inference machinery)
│   │   ├── memory/            #  1 — som
│   │   ├── point_cloud/       #  1 — latent_gmm_registration
│   │   └── tabular/           #  1 — tabm
│   ├── layers/                # 275 modules — 200 in 21 themed subpackages,
│   │   │                      # 75 loose at the top level
│   │   ├── attention/         # 33 registered attention mechanisms (factory)
│   │   ├── ffn/               # 21 registered feed-forward networks (factory)
│   │   ├── activations/       # 22 registered activations (factory)
│   │   ├── norms/             # 18 registered normalization layers (factory)
│   │   ├── embedding/         # 13 registered positional/semantic embeddings (factory)
│   │   ├── transformers/      # Assembled transformer/encoder/decoder blocks
│   │   ├── heads/             # Task heads dispatched by domain (nlp/vision/vlm)
│   │   ├── time_series/       # Forecasting-specific layers
│   │   ├── fastvit/           # FastViT primitives (+ a committed reference impl)
│   │   ├── graphs/            # Graph neural network components
│   │   ├── moe/               # Mixture of Experts (MoE) system
│   │   ├── memory/            # External / associative memory layers
│   │   ├── statistics/        # Statistical and probabilistic layers (MDN, Flows)
│   │   └── ...                # fusion, geometric, logic, mixtures, physics,
│   │                          # reasoning, sequence_pooling, tokenizers
│   ├── losses/                # 44 specialized loss modules (AnyLoss, Goodhart, etc.)
│   ├── metrics/               # Custom Keras metrics (PSNR, SSIM, perplexity, Brier)
│   ├── optimization/          # Optimizers (Muon, VSGD, SGLD, Gefen, WW-PGD), LR schedules
│   ├── analyzer/              # Comprehensive model analysis toolkit and visualizers
│   ├── visualization/         # Plotting helpers for training and evaluation
│   ├── datasets/              # Dataset loaders and synthetic generators
│   ├── callbacks/             # Reusable Keras callbacks
│   ├── regularizers/          # Advanced regularization techniques (SRIP, Orthogonal)
│   ├── initializers/          # Structured initializers (Gabor, Haar, orthonormal, KAN)
│   ├── constraints/           # Weight constraints
│   └── utils/                 # Core utilities, loggers, masking, and data handlers
├── src/train/                 # 70 train_*.py entry points across 47 trainer directories
├── src/applications/          # Deployable apps (today: bias_free_denoiser, Streamlit)
├── research/                  # 120+ in-depth articles, guides, and LaTeX papers
├── tests/                     # 895 test modules mirroring src/dl_techniques/
└── REPO_MAP.md                # Path-verified router — read this first
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
-   **Code Quality**: Follow PEP 8, use type hints, and rely on centralized logging via `dl_techniques.utils.logger` (no `print`).
-   **Testing**: Develop in the `.venv` environment and write comprehensive tests using `pytest`, scoped to the modules you change (`make test` runs the full ~1.5h suite). Set `MPLBACKEND=Agg` when running training scripts on headless machines.
-   **Documentation**: Document every public symbol, and update relevant guides in the `research/` directory. Docstring style is **not** uniform across this repo — `layers/` is predominantly Sphinx/reST (`:param:`), `models/` is measurably mixed with the Sphinx exemplar `models/language/bert/model.py` as the model for new packages, and `losses/`/`metrics/`/`utils/`/`optimization/`/`analyzer/`/`visualization/` are Google-`Args:`-majority. Match the package you are editing and never convert a file wholesale; the measured counts and the greps that re-derive them are in `src/dl_techniques/CLAUDE.md` § Core Conventions → Code Style.
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

This library is proudly sponsored and pioneered by **[Electi Consulting](https://electiconsulting.com)**, a premier AI consultancy specializing in enterprise artificial intelligence, blockchain technology, and cryptographic solutions. The practical validation and enterprise-ready nature of these components has been made possible through Electi's extensive experience deploying state-of-the-art AI solutions across diverse industries:

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
-   **Differential Transformer**: Ye, T., et al. (2024). *ICLR 2025*.
-   **Mamba: Linear-Time Sequence Modeling**: Gu, A., & Dao, T. (2023).
-   **Gemma 3 Technical Report**: Google DeepMind (2025).
-   **Qwen Technical Report**: Bai, J., et al. (2023).
-   **Byte Latent Transformer: Patches Scale Better Than Tokens**: Pagnoni, A., et al. (2024).
-   **ModernBERT: Smarter, Better, Faster, Longer**: Warner, B., et al. (2024).
-   **DistilBERT, a distilled version of BERT**: Sanh, V., et al. (2019).
-   **ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT**: Khattab, O., & Zaharia, M. (2020). *SIGIR*.
-   **ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction**: Santhanam, K., et al. (2022). *NAACL*.
-   **Tree Transformer: Integrating Tree Structures into Self-Attention**: Wang, Y.-S., Lee, H.-Y., & Chen, Y.-N. (2019). *EMNLP-IJCNLP*.
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
-   **BEiT: BERT Pre-Training of Image Transformers**: Bao, H., et al. (2021). *ICLR*.
-   **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows**: Liu, Z., et al. (2021). *ICCV*.
-   **MobileCLIP2: Improving Multi-Modal Reinforced Training**: Faghri, F., et al. (2025).
-   **Segment Anything**: Kirillov, A., et al. (2023). *ICCV*.
-   **SAM 2: Segment Anything in Images and Videos**: Ravi, N., et al. (2024).
-   **SuperPoint: Self-Supervised Interest Point Detection and Description**: DeTone, D., et al. (2018). *CVPR Workshops*.
-   **V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video**: Bardes, A., et al. (2024).
-   **Masked Autoencoders Are Scalable Vision Learners** (MAE): He, K., et al. (2022). *CVPR*.
-   **CBAM: Convolutional Block Attention Module**: Woo, S., et al. (2018). *ECCV*.
-   **Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data**: Yang, L., et al. (2024). *CVPR*.
</details>

<details>
<summary><b>Generative Modeling & Image Restoration</b></summary>

-   **Scaling Rectified Flow Transformers for High-Resolution Image Synthesis** (SD3 / MMDiT): Esser, P., et al. (2024). *ICML*.
-   **Flow Matching for Generative Modeling**: Lipman, Y., et al. (2023). *ICLR*.
-   **Auto-Encoding Variational Bayes** (VAE): Kingma, D. P., & Welling, M. (2013).
-   **Neural Discrete Representation Learning** (VQ-VAE): van den Oord, A., et al. (2017). *NeurIPS*.
-   **Robust and Interpretable Blind Image Denoising via Bias-Free Convolutional Neural Networks**: Mohan, S., et al. (2020). *ICLR*.
-   **Thera: Aliasing-Free Arbitrary-Scale Super-Resolution with Neural Heat Fields**: Becker, A., et al. (2025). *CVPR*.
-   **DarkIR: Robust Low-Light Image Restoration**: Feijoo, D., et al. (2025). *CVPR*.
-   **Global Modeling Matters: A Fast, Lightweight and Effective Baseline for Efficient Image Restoration** (PW-FNet): Jiang, X., et al. (2025).
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
-   **Energy Transformer**: Hoover, B., et al. (2023). *NeurIPS*.
-   **Neural Turing Machines**: Graves, A., et al. (2014).
-   **Geometric Clifford Algebra Networks**: Ruhe, D., et al. (2023). *ICML*.
-   **Hyperbolic Graph Convolutional Neural Networks**: Chami, I., et al. (2019). *NeurIPS*.
-   **sHGCN: Simplified hyperbolic graph convolutional neural networks**: Arevalo, P., Molina, A., & Ciudad, A. (2025).
-   **The Self-Organizing Map**: Kohonen, T. (1990). *Proceedings of the IEEE*, 78(9).
</details>

<details>
<summary><b>Time Series & Forecasting</b></summary>

-   **N-BEATS: Neural basis expansion analysis for interpretable time series forecasting**: Oreshkin, B. N., et al. (2019). *ICLR*.
-   **Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting**: Lim, B., et al. (2021).
-   **DeepAR: Probabilistic forecasting with autoregressive recurrent networks**: Salinas, D., et al. (2020).
-   **xLSTM: Extended Long Short-Term Memory**: Beck, M., et al. (2024). *NeurIPS*.
-   **Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift**: Kim, T., et al. (2021). *ICLR*.
-   **TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning**: Auer, A., et al. (2025).
-   **PRISM: A hierarchical multiscale approach for time series forecasting**: Chen, Z., et al. (2025).
</details>

<details>
<summary><b>Tabular & Inference-Time Methods</b></summary>

-   **TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling**: Gorishniy, Y., et al. (2024).
-   **Reasoning with Sampling: Your Base Model is Smarter Than You Think**: Karan, A., et al. (2025).
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