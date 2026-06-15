# Layers Package

The largest package in the library — custom Keras 3 layers organized by domain. All layers follow Keras 3 conventions with full `get_config()` serialization support.

> **New layers MUST follow `research/2026_keras_custom_models_instructions.md`.** Read it before creating a new layer — it is the canonical guide for Keras 3 custom layer authoring in this repo (`__init__`/`build`/`call`/`get_config`, serialization, factory registration, tests).

## Structure

### Attention (`attention/`)
Multi-head, cross, latent, differential, group-query, ring, performer, perceiver, Hopfield, capsule routing, anchor, channel, spatial, convolutional block (CBAM), progressive focused, wave field, window, mobile MQA, non-local, RPC, shared-weights cross, single-window, tripse, gated, FNet Fourier transform. Includes `factory.py` for config-driven construction.

### Feed-Forward Networks (`ffn/`)
MLP, SwiGLU, GeGLU, GLU, OrthoGLU, gated MLP, power MLP, counting FFN, diff FFN, logic FFN, Swin MLP, residual block. Includes `factory.py`.

### Normalization (`norms/`)
RMS norm, zero-centered RMS, band RMS, adaptive band RMS, logit norm, max logit norm, band logit norm, dynamic tanh, global response norm. Includes `factory.py`. Also hosts `PolarWeightNorm` — a polar-coordinate *weight* reparameterization (radius + hierarchical angles, exact per-unit norm; generalizes Weight Normalization). Not factory-registered; see the `PolarWeightNorm` module docstring in `norms/polar_weight_norm.py`.

### Embeddings (`embedding/`)
Patch embedding (1D/2D), learned positional, fixed 2D sinusoidal positional, rotary position (RoPE), dual rotary, continuous RoPE, continuous sin/cos, scalar/timestep sinusoidal, multi-axis (t/h/w) RoPE, BERT / ModernBERT / ALBERT-factorized token embeddings. Includes `factory.py` with **13 registered keys** (`patch_1d`, `patch_2d`, `positional_learned`, `rope`, `dual_rope`, `continuous_rope`, `continuous_sincos`, `bert_embeddings`, `modern_bert_embeddings`, `albert_factorized`, `positional_sine_2d`, `scalar_sinusoidal`, `mrope_ideogram4`). `HierarchicalCodebookEmbedding` is direct-import-only (not factory-registered). All `call()` paths are graph-safe (no eager ops); `positional_sine_2d` emits channels-first `(B, 2*num_pos_feats, H, W)`.

### Mixture of Experts (`moe/`)
Full MoE framework: `config.py` (MoE configuration), `experts.py` (expert networks), `gating.py` (routing/gating), `layer.py` (main MoE layer), `integration.py` (integration helpers).

### Mixtures (`mixtures/`)
Differentiable soft-clustering / mixture layers + factory: `radial_basis_function.py` (`RBFLayer`), `kmeans.py` (`KMeansLayer` — differentiable K-means), `gmm.py` (`GMMLayer` — differentiable Gaussian Mixture Model with isometric-kernel regularization). `factory.py` exposes `MixtureType` + `create_mixture_layer`/`create_mixture_from_config`. Import via `from dl_techniques.layers.mixtures import RBFLayer, KMeansLayer, GMMLayer, create_mixture_layer`.

### Transformers (`transformers/`)
Standard transformer, Swin transformer block, Swin conv block, perceiver transformer, progressive focused transformer, EoMT transformer, free transformer, text encoder/decoder, vision encoder.

### Graph Layers (`graphs/`)
Graph neural network, relational graph transformer, simplified hyperbolic GCN, entity graph refinement, Fermi-Dirac decoder.

### Activations (`activations/`)
GoLU, Mish, hard sigmoid, hard swish, ReLU-k, sparsemax, squash, thresh-max, adaptive softmax, differentiable step, expanded activations, monotonicity, probability output, routing probabilities (unified deterministic / trainable modes), basis function. Includes `factory.py`.

### Geometric (`geometric/`)
Clifford algebra block, point cloud autoencoder, supernode pooling, and `fields/` subpackage: connection layer, field embedding, gauge-invariant attention, holonomic transformer, holonomy layer, manifold stress, parallel transport.

### Fusion (`fusion/`)
Multimodal fusion layer.

### Memory (`memory/`)
Single canonical home for memory-augmented and topographic-memory layers
(merged from a previously-separate NTM subpackage). Exports four families
via `dl_techniques.layers.memory.*`:

- **NTM family** (`ntm_interface.py`, `baseline_ntm.py`) — `NTMConfig`,
  `NTMMemory`, `NTMReadHead`, `NTMWriteHead`, `NTMController`, `NTMCell`,
  `NeuralTuringMachine`, abstract `BaseMemory/BaseHead/BaseController/BaseNTM`,
  `AddressingMode` enum (CONTENT + HYBRID only), state dataclasses
  (`MemoryState`, `HeadState`, `NTMOutput`), and utility functions
  (`cosine_similarity`, `circular_convolution`, `sharpen_weights`).
- **MANN** (`mann.py`) — `MannLayer` standalone class (legacy entry point;
  new callers prefer the `create_mann(...)` factory).
- **SOM family** (`som_nd_layer.py`, `som_2d_layer.py`, `som_nd_soft_layer.py`) —
  `SOMLayer` (N-D hard winner), `SOM2dLayer` (2D specialization),
  `SoftSOMLayer` (differentiable / per-dim or global softmax).
- **NeuroGrid** (`neuro_grid.py`) — `NeuroGrid` topographic memory grid
  (differentiable soft-assignment grid; uses orthogonal hypersphere init +
  soft-orthonormal regularization).
- **Factory** (`factory.py`) — uniform construction surface:
  `create_ntm(...)`, `create_mann(...)` (returns a `NeuralTuringMachine`
  configured to preserve the historical MANN output shape), `create_som_2d(...)`.

### Logic (`logic/`)
Arithmetic operators, logic operators, neural circuit.

### Physics (`physics/`)
Lagrange layer, approximate Lagrange layer.

### Reasoning (`reasoning/`)
HRM reasoning core, HRM reasoning module, HRM sparse puzzle embedding.

### Statistics (`statistics/`)
Deep kernel PCA, invertible kernel PCA, MDN layer, moving std, normalizing flow, residual ACF, scaler.

### Time Series (`time_series/`)
Adaptive lag attention, DeepAR blocks, EMA layer, forecasting layers, mixed sequential block, N-BEATS/N-BEATSx blocks, PRISM blocks, quantile heads (fixed/variable IO), temporal convolutional network, temporal fusion, xLSTM blocks.

### Tokenizers (`tokenizers/`)
BPE tokenizer implementation.

### Task Heads (`heads/`)
Single merged package consolidating the formerly-separate `nlp_heads/`,
`vision_heads/`, and `vlm_heads/` packages into three sub-packages:
- `heads/nlp/` — NLP output heads (text/token classification, QA, similarity,
  generation, multiple-choice, multi-task). Sequence pooling reuses the shared
  `SequencePooling` layer for `cls`/`mean`/`max` (the learnable `attention`
  strategy stays inline — different mechanism + weights).
- `heads/vision/` — vision output heads (detection, segmentation, depth,
  classification, instance segmentation, enhancement, multi-task) +
  `VisionTaskType` (with a `TaskType` back-compat alias).
- `heads/vlm/` — vision-language model heads (captioning, VQA, visual grounding,
  image-text matching, multi-task).

Each sub-package keeps its own `factory.py` + `task_types.py`. A top-level
`heads/factory.py` exposes a `create_head(domain, ...)` dispatch facade over the
three single-head factories, and `heads/task_types.py` aggregates the task-type
enums/configs. Import via `from dl_techniques.layers.heads import create_head` or
`from dl_techniques.layers.heads.nlp import create_nlp_head` (likewise `.vision`,
`.vlm`). See `heads/CLAUDE.md` and `heads/README.md`.

### Experimental (`experimental/`)
Experimental/unstable layers: band RMS OOD, contextual counter FFN, contextual memory, field embeddings, graph MANN, hierarchical evidence LLM, hierarchical memory system, MST correlation filter.

### Standalone Layers (top-level files)
Bias-free Conv1D/Conv2D, BitLinear, BLT blocks/core, Canny edge detection, capsules, CLAHE, complex-valued layers, conditional output, Conv2D builder, ConvNeXt v1/v2 blocks, convolutional KAN, depthwise separable, downsample/upsample, dynamic Conv2D, EoMT mask, FiLM, FNet encoder, fractal block, FFT layers, Gaussian filter/pyramid, gated delta net, global sum pool, HANC block/layer, hierarchical MLP stem, inverted residual block, IO preparation, KAN linear, Laplacian filter, layer scale, mobile-one block, modality projection, MothNet blocks, MPS layer, multi-level feature compilation, one-hot encoding, orthoblock, orthogonal butterfly (exactly-orthogonal Givens butterfly, invertible; see module docstring), patch merging, pixel shuffle, random Fourier features, RepMixer block, res-path, restricted Boltzmann machine, rigid simplex, router, sampling (Gaussian-ball / thin-shell hypersphere / von Mises-Fisher reparameterization samplers + inline factory; `vmf` adds `VMFSampling` + the closed-form `vmf_kl_divergence`), selective gradient mask, sequence pooling, shearlet transform, sparse autoencoder, spatial layer, squeeze-excitation, standard blocks, stochastic depth/gradient, strong augmentation, TabM blocks, Tversky projection, universal inverted bottleneck, vector quantizer, YOLO12 blocks/heads.

## Conventions

- `__init__.py` is empty — import from submodules directly (e.g., `from dl_techniques.layers.attention.multi_head_attention import MultiHeadAttention`)
- Subpackages with `factory.py` support config-driven layer construction
- All layers must implement `get_config()` for Keras serialization
- Layers follow Keras 3 custom layer patterns: `__init__`, `build`, `call`, `get_config`

## Testing

Tests in `tests/test_layers/` organized by subdomain (attention, embeddings, ffn, norms, graphs, moe, etc.).
