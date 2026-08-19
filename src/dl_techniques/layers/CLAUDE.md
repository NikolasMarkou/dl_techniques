# Layers Package

The largest package in the library — custom Keras 3 layers organized by domain. All layers follow Keras 3 conventions with full `get_config()` serialization support.

> **New layers MUST follow `research/2026_keras_custom_models_instructions_v2.md`.** Read it before creating a new layer — it is the canonical guide for Keras 3 custom layer authoring in this repo (`__init__`/`build`/`call`/`get_config`, serialization, factory registration, tests).

## Structure

One row per subpackage. **F** = has a `factory.py`. Names carrying a trap are marked and explained
in § Naming traps below.

| Subpackage | F | Contents |
|---|:-:|---|
| `attention/` | ✅ | Multi-head, cross, latent, differential, group-query, ring, performer, perceiver, Hopfield, capsule routing, anchor, channel, spatial, CBAM, progressive focused, wave field, window, mobile MQA, non-local, RPC, shared-weights cross, single-window, tripse, gated, linear (Miyasawa-compliant O(N)), lighthouse, energy, FNet Fourier transform |
| `ffn/` | ✅ | MLP, SwiGLU, GeGLU, GLU, OrthoGLU, gated MLP, power MLP, counting FFN, diff FFN, logic FFN, Swin MLP, residual block |
| `norms/` | ✅ | RMS, zero-centered RMS, band RMS, adaptive band RMS, logit norm, max logit norm, band logit norm, dynamic tanh, global response norm, bias-free batch norm, energy layer norm. Also hosts `PolarWeightNorm` (`norms/polar_weight_norm.py`, not factory-registered) |
| `embedding/` | ✅ | Patch (1D/2D), learned positional, fixed 2D sinusoidal, RoPE, dual RoPE, continuous RoPE, continuous sin/cos, scalar/timestep sinusoidal, multi-axis (t/h/w) RoPE, BERT / ModernBERT / ALBERT-factorized token embeddings |
| `activations/` | ✅ | GoLU, Mish, hard sigmoid, hard swish, ReLU-k, sparsemax, squash, thresh-max, adaptive softmax, differentiable step, expanded activations, monotonicity, probability output, routing probabilities, basis function |
| `heads/` | ✅ | Task heads in three sub-packages — `nlp/`, `vision/`, `vlm/` (see below) |
| `logic/` | ✅ | Arithmetic operators, logic operators, neural circuit |
| `memory/` | ✅ | NTM family, SOM family, NeuroGrid (see below) |
| `mixtures/` | ✅ | `RBFLayer` (`radial_basis_function.py`), `KMeansLayer` (`kmeans.py`, differentiable K-means), `GMMLayer` (`gmm.py`, differentiable GMM with isometric-kernel regularization). `factory.py` exposes `MixtureType` + `create_mixture_layer` / `create_mixture_from_config`. Import via `from dl_techniques.layers.mixtures import RBFLayer, KMeansLayer, GMMLayer, create_mixture_layer` |
| `sequence_pooling/` | ✅ | Pool a `(B, T, D)` sequence to `(B, D)`: `SequencePooling` (`sequence_pooling.py` — `cls`/`mean`/`max`/positional, reused by `heads/nlp/`), `attention_pooling.py`, `weighted_pooling.py`. Carries its own `README.md` + `GUIDE.md` |
| `transformers/` | — | Standard transformer, Swin block, Swin conv block, perceiver, progressive focused, EoMT, free transformer, text encoder/decoder, vision encoder, `EnergyTransformer` + `HopfieldNetwork` (`energy_transformer.py`), `GatedLinearAttentionBlock` (`gated_linear_attention_block.py`) |
| `fastvit/` | — | Channels-last transcriptions of timm's FastViT **MCi** image-tower primitives, consumed by `models/fastvit/` (which assembles them into the MCi tower): `FastVitConvMlp`, `RepConditionalPosEnc`, `FastVitRepMixer`, `FastVitRepMixerBlock`, `ReparamLargeKernelConv`, `FastVitPatchEmbed`, `FastVitAttentionBlock`, `FastVitStage`. Curated `__init__` re-export, no factory. Train-time multi-branch form only — no reparameterization / fusion path. See `fastvit/README.md` |
| `moe/` | — | Full MoE framework: `config.py`, `experts.py`, `gating.py`, `layer.py`, `integration.py` |
| `graphs/` | — | Graph neural network, relational graph transformer, simplified hyperbolic GCN, entity graph refinement, Fermi-Dirac decoder |
| `geometric/` | — | Clifford algebra block, point cloud autoencoder, supernode pooling, and `fields/`: connection layer, field embedding, gauge-invariant attention, holonomic transformer, holonomy layer, manifold stress, parallel transport |
| `statistics/` | — | Deep kernel PCA, invertible kernel PCA, MDN layer, moving std, normalizing flow, residual ACF, scaler |
| `time_series/` | — | Adaptive lag attention, DeepAR blocks, EMA layer, forecasting layers, mixed sequential block, N-BEATS/N-BEATSx blocks, PRISM blocks, quantile heads, TCN, temporal fusion, xLSTM blocks |
| `reasoning/` | — | HRM reasoning core, HRM reasoning module, HRM sparse puzzle embedding |
| `physics/` | — | Lagrange layer, approximate Lagrange layer |
| `fusion/` | — | Multimodal fusion layer |
| `tokenizers/` | — | BPE tokenizer |
| `experimental/` | — | Unstable: band RMS OOD, contextual counter FFN, contextual memory, field embeddings, graph MANN, hierarchical evidence LLM, hierarchical memory system, MST correlation filter |

### Semantics worth knowing before you reuse

| Layer | Fact |
|---|---|
| `EnergyAttention` (`attention/`) | no value matrix; `call()` returns the closed-form negative gradient of a scalar token-mixing energy |
| `BiasFreeBatchNorm` (`norms/`) | no `moving_mean` / `beta`; degree-1 homogeneous at inference, for bias-free / Miyasawa denoisers |
| `EnergyLayerNorm` (`norms/`) | **scalar** gamma + **vector** delta; the output is `dL/dx` of a Lagrangian with a PSD Hessian — what makes the Energy Transformer's energy descent provable |
| `PolarWeightNorm` (`norms/`) | a polar-coordinate *weight* reparameterization (radius + hierarchical angles, exact per-unit norm); generalizes Weight Normalization. See its module docstring |
| `EnergyTransformer` (`transformers/`) | recurrent block replacing `attn -> FFN` with `T` steps of gradient descent on a single scalar energy (arXiv:2302.07253) |
| `GatedLinearAttentionBlock` (`transformers/`) | linear-time recurrent sequence mixer, state follows a gated outer-product rule; consumed 3x per Qwen3-Next block |
| `positional_sine_2d` (`embedding/`) | emits **channels-first** `(B, 2*num_pos_feats, H, W)` |
| embedding `call()` paths | all graph-safe — no eager ops |

### `embedding/` factory keys (13)

`patch_1d`, `patch_2d`, `positional_learned`, `rope`, `dual_rope`, `continuous_rope`,
`continuous_sincos`, `bert_embeddings`, `modern_bert_embeddings`, `albert_factorized`,
`positional_sine_2d`, `scalar_sinusoidal`, `mrope_ideogram4`.

`HierarchicalCodebookEmbedding` is direct-import-only (not factory-registered).

### `memory/` — the single canonical home

Merged from a previously-separate NTM subpackage. Four families via `dl_techniques.layers.memory.*`:

| Family | Modules | Exports |
|---|---|---|
| **NTM** | `ntm_interface.py`, `baseline_ntm.py` | `NTMConfig`, `NTMMemory`, `NTMReadHead`, `NTMWriteHead`, `NTMController`, `NTMCell`, `NeuralTuringMachine`, abstract `BaseMemory`/`BaseHead`/`BaseController`/`BaseNTM`, `AddressingMode` (CONTENT + HYBRID only), state dataclasses (`MemoryState`, `HeadState`, `NTMOutput`), utilities (`cosine_similarity`, `circular_convolution`, `sharpen_weights`) |
| **SOM** | `som_nd_layer.py`, `som_2d_layer.py`, `som_nd_soft_layer.py` | `SOMLayer` (N-D hard winner), `SOM2dLayer` (2D specialization), `SoftSOMLayer` (differentiable, per-dim or global softmax) |
| **NeuroGrid** | `neuro_grid.py` | `NeuroGrid` topographic memory grid — differentiable soft-assignment; orthogonal hypersphere init + soft-orthonormal regularization |
| **Factory** | `factory.py` | `create_mann(...)`, `create_som_2d(...)`. **`create_mann` is the ONLY MANN construction path** — there is no standalone MANN class; it returns a `NeuralTuringMachine` configured to preserve the historical MANN output shape |

### `heads/` — one merged package

Consolidates the formerly-separate `nlp_heads/`, `vision_heads/` and `vlm_heads/`:

| Sub-package | Heads |
|---|---|
| `heads/nlp/` | text/token classification, QA, similarity, generation, multiple-choice, multi-task. Pooling reuses the shared `SequencePooling` for `cls`/`mean`/`max`; the learnable `attention` strategy stays inline (different mechanism + weights) |
| `heads/vision/` | detection, segmentation, depth, classification, instance segmentation, enhancement, multi-task, plus `VisionTaskType` (with a `TaskType` back-compat alias) |
| `heads/vlm/` | captioning, VQA, visual grounding, image-text matching, multi-task |

Each sub-package keeps its own `factory.py` + `task_types.py`. A top-level `heads/factory.py` exposes
a `create_head(domain, ...)` dispatch facade; `heads/task_types.py` aggregates the enums/configs.
Import via `from dl_techniques.layers.heads import create_head`, or
`from dl_techniques.layers.heads.nlp import create_nlp_head` (likewise `.vision`, `.vlm`).
See `heads/CLAUDE.md` and `heads/README.md`.

### Standalone layers (top-level files)

Bias-free Conv1D/Conv2D, BitLinear, BLT blocks/core, Canny edge detection, capsules, CLAHE,
complex-valued layers, conditional output, Conv2D builder, ConvNeXt v1/v2 blocks, convolutional KAN,
depthwise separable, downsample/upsample, dynamic Conv2D, EoMT mask, FiLM, FNet encoder, fractal
block, FFT layers, Gaussian filter/pyramid, global sum pool, HANC block/layer, hierarchical MLP stem,
inverted residual block, IO preparation, KAN linear, Laplacian filter, layer scale, mobile-one block,
modality projection, MothNet blocks, MPS layer, multi-level feature compilation, one-hot encoding,
orthoblock, orthogonal butterfly (exactly-orthogonal Givens butterfly, invertible — see its module
docstring), patch merging, pixel shuffle, random Fourier features, RepMixer block (`repmixer_block.py`), res-path,
restricted Boltzmann machine, rigid simplex, router, sampling, scheduled dropout, selective gradient
mask, shearlet transform, sparse autoencoder, spatial layer, squeeze-excitation, standard blocks,
stochastic depth/gradient, strong augmentation, TabM blocks, Tversky projection, universal inverted
bottleneck, vector quantizer, YOLO12 blocks/heads.

`sampling.py` provides Gaussian-ball / thin-shell hypersphere / von Mises-Fisher reparameterization
samplers plus an inline factory; `vmf` adds `VMFSampling` and the closed-form `vmf_kl_divergence`.

### Naming traps

| Trap | Detail |
|---|---|
| **`RepMixerBlock` is two different architectures** | `fastvit/FastVitRepMixerBlock` (timm FastViT MCi, consumed by `models/fastvit/`) is **NOT** the top-level `repmixer_block.py::RepMixerBlock` (a different architecture sharing the name, consumed by `models/fastvlm/`). The FastViT names carry a `FastVit` prefix precisely because the serialization registry is keyed by bare class name |

## Conventions

### `__init__.py` policy varies by subpackage — check before assuming

The `layers/__init__.py` root **is** empty. Most subpackages are **not**.

| Shape | Subpackages | How to import |
|---|---|---|
| **Curated re-export with `__all__`** | `activations`, `attention` (43 names), `embedding`, `fastvit`, `ffn`, `heads`, `logic`, `memory`, `mixtures`, `moe`, `norms`, `sequence_pooling`, `time_series`, `transformers` | `from dl_techniques.layers.attention import MultiHeadAttention, create_attention_layer` |
| **Empty** | `fusion`, `geometric`, `graphs`, `physics`, `reasoning`, `statistics`, `tokenizers`, and the top-level standalone modules | `from dl_techniques.layers.graphs.graph_neural_network import GraphNeuralNetwork` |
| **No `__init__.py` at all** | `experimental/` — a namespace package | submodule imports only |

Submodule imports keep working in both cases — e.g.
`from dl_techniques.layers.attention.multi_head_attention import MultiHeadAttention` — but for a
subpackage with an `__all__`, prefer the package-level import.

### Other conventions

- Docstrings in `layers/` use **Sphinx/reST** (`:param:` / `:type:` / `:raises:`), not Google
  `Args:`, in the large majority of modules. The count, its date and the re-deriving grep are printed
  once, in `src/dl_techniques/CLAUDE.md` § Core Conventions → Code Style. In `attention/` it is
  mandatory; `attention/channel_attention.py` is the reference exemplar.
- Subpackages with `factory.py` support config-driven construction.
- All layers implement `get_config()` for Keras serialization.
- Layers follow the Keras 3 pattern: `__init__`, `build`, `call`, `get_config`.

> **This does not put `layers/` in opposition to a Google-style `models/`.** `models/` has no
> package-wide style at all — it is measurably mixed, and its normative exemplar for new packages
> (`models/bert/bert.py`) is itself entirely Sphinx/reST. The earlier phrasing here, "this differs
> from `models/`", rested on a blanket claim that measurement refuted. Where `layers/` genuinely
> differs is from the Google-majority `losses/`, `metrics/`, `utils/`, `optimization/`, `analyzer/`
> and `visualization/`.

## Layer Reuse Policy (factory-first)

> **Before implementing ANY new layer, you MUST first check for an existing one to reuse.** Authoring a bespoke layer is the last resort, not the first move — this package already ships a large, tested layer surface.

Check in this precedence order; only proceed to the next step when nothing fits:

1. **The relevant domain factory** — each factory exposes a `create_*_layer()` entry point backed by a registry of named types. Pass a `type` string + config; do not hand-roll what a factory already builds.

   | Domain | Factory entry point | Registered types |
   |--------|---------------------|------------------|
   | Normalization | `create_normalization_layer()` in `norms/factory.py` | 18 |
   | Attention | `create_attention_layer()` in `attention/factory.py` | 32 |
   | FFN / MLP | `create_ffn_layer()` in `ffn/factory.py` | 21 |
   | Embeddings | `create_embedding_layer()` in `embedding/factory.py` | 13 |
   | Activations | `create_activation_layer()` in `activations/factory.py` | 22 |
   | Sequence pooling | `create_sequence_pooling_layer()` in `sequence_pooling/factory.py` | 3 (`sequence`, `attention`, `weighted`) |
   | Logic | `create_logic_layer()` in `logic/factory.py` | 4 (`logic`, `arithmetic`, `neural_circuit`, `circuit_depth`) |
   | Mixtures | `create_mixture_layer()` in `mixtures/factory.py` | 3 (RBF / KMeans / GMM) |
   | Memory | `create_mann()` / `create_som_2d()` in `memory/factory.py` | n/a |
   | Task heads | `create_head(domain, ...)` in `heads/factory.py` (NLP / vision / VLM) | n/a |
   | Transformer blocks | `TransformerLayer` in `transformers/transformer.py` (direct import) | n/a |

   > **Note on transformer blocks**: `transformers/` has no `create_*_layer` factory. Use `TransformerLayer` directly — it is highly configurable (selectable attention / FFN / normalization types and normalization position via its config) and composes the domain factories above internally, so it covers most cases without a custom block. The package also offers higher-level `create_*_encoder` builders (`transformers/vision_encoder.py`, `transformers/text_encoder.py`).

2. **An existing standalone layer** — if no factory covers your need, search the subpackages and the top-level Standalone Layers list above before writing your own. The reuse surface is broad; a close match often already exists.

3. **Only then, a new custom layer** — if nothing above fits, implement it following `research/2026_keras_custom_models_instructions_v2.md` (full serialization, `build`, `get_config`, tests). Place it in the appropriate domain subpackage and, where that domain has a `factory.py`, register it there so the next author can reuse it via the factory too.

### The factory contract

**1. Every factory RAISES `ValueError` on an undeclared keyword.** Verified by execution 2026-08-19
for attention, ffn, norms, activations and embedding: each rejects a `bogus_key=1`. A new factory, or
any new registry-backed dispatch, does the same. **Never filter-and-drop** — each of these was
silent, and each passed its tests:

| Misspelling | Consequence |
|---|---|
| `dropout=` against a declared `dropout_rate` | positional dropout dead repo-wide |
| `qkv_bias=` against `use_bias` | attention built with zero bias weights |
| RoPE args into an attention type declaring none | keys evaporated |

**2. A registry's key set, `Literal` aliases and each entry's `required_params` / `optional_params`
are public API** — consumed by config-driven callers and asserted by
`tests/test_layers/test_factory_registry_drift.py`. Adding, renaming or removing one is a breaking
change. `attention/factory.py`'s module docstring is the exemplar for declaring this, including why
two of its keys deliberately map to functions rather than classes, and why two more configurations
are deliberately NOT registered.

**3. An "optional" parameter the layer derives is not safe to pass explicitly.** Consult the registry
entry's `required_params`: `hidden_dim` is required for 13 of the 21 FFN types and derived for
`swiglu` (a two-thirds rule plus `ffn_multiple_of`).

**4. When you audit "who calls factory X", also sweep "who builds X's argument dict without calling X
directly."** 9 files pass an `ffn_args=` dict to a wrapper rather than calling `create_ffn_layer`
themselves. Such a site is invisible to an AST call inventory of the factory, and invisible to a suite
sweep run at site defaults (the break needs a non-empty caller dict to appear). A `**kwargs`-splat
site has the identical blind spot. Use `assemble_ffn_config()` / `assemble_attention_config()` so
those dicts go through the same validation — `models/qwen/qwen3.py:416` is the repaired exemplar and
carries a comment explaining why it must not be simplified back to a bare dict literal.

**5. Normalization epsilon.** The factory sets `1e-6`; Keras' `LayerNormalization` /
`BatchNormalization` default to `1e-3` (both confirmed by execution 2026-08-19). Route normalization
through the factory. Constructing directly makes `epsilon=` mandatory, with a cited reference. Do not
blanket-fix — MobileNet's reference BN genuinely is `1e-3`.

## Testing

Tests in `tests/test_layers/` organized by subdomain (attention, embeddings, ffn, norms, graphs, moe, etc.).
