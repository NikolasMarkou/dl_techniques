# Layers Package

The largest package in the library — custom Keras 3 layers organized by domain, all with full
`get_config()` serialization.

> **New layers MUST follow `research/2026_keras_custom_models_instructions_v2.md`** — the canonical
> guide for Keras 3 custom layer authoring here (`__init__`/`build`/`call`/`get_config`,
> serialization, factory registration, tests). Read it before writing a new layer.

## Structure

One row per subpackage. **F** = has a `factory.py`. Several class names collide across packages —
see § Naming traps.

| Subpackage | F | Contents |
|---|:-:|---|
| `attention/` | Y | 34 registered attention types — multi-head, cross, latent, differential, group-query, ring, performer, perceiver, Hopfield, capsule routing, window / single-window, mobile MQA, non-local, CBAM, linear (Miyasawa-compliant O(N)), energy, area, FNet Fourier, and more |
| `ffn/` | Y | MLP, SwiGLU, GeGLU, GLU, OrthoGLU, gated MLP, power MLP, counting, diff, logic, Swin MLP, residual block |
| `norms/` | Y | RMS family (RMS, zero-centered, band, adaptive band), logit-norm family, dynamic tanh, GRN, bias-free batch norm, energy layer norm. Also hosts `PolarWeightNorm` (not factory-registered) |
| `embedding/` | Y | Patch (1D/2D), learned positional, sinusoidal (2D / scalar / timestep), RoPE family (plain, dual, continuous, multi-axis), BERT / ModernBERT / ALBERT-factorized token embeddings, class-label table with a classifier-free-guidance dropout row. `HierarchicalCodebookEmbedding` is direct-import-only |
| `activations/` | Y | GoLU, Mish, hard sigmoid/swish, ReLU-k, sparsemax, squash, thresh-max, adaptive softmax, differentiable step, expanded activations, monotonicity, probability / routing outputs, basis function |
| `heads/` | Y | Task heads in `nlp/`, `vision/`, `vlm/` — see below |
| `logic/` | Y | Arithmetic operators, logic operators, neural circuit |
| `memory/` | Y | NTM family, SOM family, NeuroGrid — see below |
| `mixtures/` | Y | `RBFLayer`, `KMeansLayer` (differentiable K-means), `GMMLayer` (differentiable GMM with isometric-kernel regularization) |
| `sequence_pooling/` | Y | `(B, T, D)` → `(B, D)`. `SequencePooling` covers 18 strategies (positional `cls`/`first`/`last`/`middle`, statistical, composite, learnable `attention`/`multi_head_attention`/`weighted`, top-k, `none`/`flatten`); four are reused by `heads/nlp/`. Plus `attention_pooling.py`, `weighted_pooling.py`. Carries its own `README.md` + `GUIDE.md` |
| `transformers/` | — | Standard transformer, Swin block / conv block, perceiver, progressive focused, EoMT, free transformer, text encoder/decoder, vision encoder, `EnergyTransformer` + `HopfieldNetwork`, `GatedLinearAttentionBlock`, `AreaAttentionBlock` |
| `fastvit/` | — | Channels-last transcriptions of timm's FastViT **MCi** image-tower primitives, consumed by `models/vision/fastvit/`. Curated `__init__`, no factory; train-time multi-branch form only, no reparameterization path. See `fastvit/README.md` |
| `moe/` | — | Full MoE framework: `config.py`, `experts.py`, `gating.py`, `layer.py`, `integration.py` |
| `graphs/` | — | Graph neural network, relational graph transformer, simplified hyperbolic GCN, entity graph refinement, Fermi-Dirac decoder |
| `geometric/` | — | Clifford algebra block, point cloud autoencoder, supernode pooling, and `fields/` (connection, field embedding, gauge-invariant attention, holonomic transformer, holonomy, manifold stress, parallel transport) |
| `statistics/` | — | Deep kernel PCA, invertible kernel PCA, MDN layer, moving std, normalizing flow, residual ACF, scaler, split-conformal prediction interval |
| `time_series/` | — | Adaptive lag attention, DeepAR blocks, EMA layer, forecasting layers, mixed sequential block, N-BEATS/N-BEATSx blocks, PRISM blocks, quantile heads, TCN, temporal fusion, xLSTM blocks |
| `reasoning/` | — | HRM reasoning core, reasoning module, sparse puzzle embedding |
| `physics/` | — | Lagrange layer, approximate Lagrange layer |
| `fusion/` | — | Multimodal fusion layer |
| `tokenizers/` | — | BPE tokenizer |
| `complex/` | — | Complex-valued (complex-number) layers: `ComplexConv2D`, `ComplexDense`, `ComplexReLU`, complex pooling/dropout, raw-TF backend only |
| `conv_blocks/` | — | Convolutional building blocks: bias-free Conv1D/2D, ConvNeXt v1/v2, depthwise separable, dynamic Conv2D, inverted residual, MobileOne, RepMixer, ResPath, universal inverted bottleneck, plus the shared `SqueezeExcitation` gate and `MatchChannels` skip helper |
| `signal_processing/` | — | Classical/fixed signal and image primitives: Canny, CLAHE, Gaussian filter/pyramid, Laplacian filter family, Haar wavelet decomposition, shearlet transform, FFT/IFFT, strong (color-jitter + CutMix) augmentation |
| `pooling/` | — | Spatial/channel layout changes with ~0 learned params: blur pool, pixel shuffle/unshuffle, patch merging, global sum pool, U-Net downsample-and-skip junction |
| `structured_linear/` | — | Alternative parameterizations to a dense weight matrix: BitLinear (1.58-bit), MPS/tensor-train, OrthoBlock, orthogonal butterfly, random Fourier features, rigid simplex, KANvolution |
| `regularization/` | — | Regularization and routing: layer scale, scheduled dropout, stochastic depth, stochastic gradient, selective gradient mask, router, FiLM, conditional output |
| `generative/` | — | Generative / latent-variable layers: restricted Boltzmann machine, sparse autoencoder, VAE reparameterization samplers, vector quantizer (+ rotation-trick variant) |
| `blt/` | — | Byte Latent Transformer stack: tokenizer, entropy model, dynamic patcher, patch pooling, local encoder/decoder, global transformer, plus the HRM-fused reasoning core |
| `acc_unet/` | — | ACC-UNet cluster: HANC block + layer (hierarchical-context aggregation), multi-level feature compilation (cross-scale fusion) |
| `yolo12/` | — | YOLOv12 backbone blocks (`Bottleneck`, `C3k2Block`, `A2C2fBlock`) and task heads (detection/segmentation/classification) |
| `tabular/` | — | TabM batched-ensemble MLP building blocks (D-007: named for the domain, not the paper acronym) |

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

### `memory/` — the single canonical home

Three families under `dl_techniques.layers.memory.*`: **NTM** (`NeuralTuringMachine` plus its
memory/head/controller/cell parts, the abstract bases, `AddressingMode` with CONTENT + HYBRID only,
the state dataclasses and the addressing utilities), **SOM** (`SOMLayer` N-D hard winner,
`SOM2dLayer`, `SoftSOMLayer` differentiable with per-dim or global softmax), and **NeuroGrid** (a
differentiable topographic memory grid, orthogonal hypersphere init + soft-orthonormal
regularization). `factory.py` exposes `create_mann(...)` and `create_som_2d(...)`; **`create_mann`
is the ONLY MANN construction path** — there is no standalone MANN class, it returns a
`NeuralTuringMachine` configured to preserve the historical MANN output shape.

### `heads/`

One merged package with `nlp/`, `vision/` and `vlm/` sub-packages, a `create_head(domain, ...)`
dispatch facade and per-domain factories. **`heads/CLAUDE.md` owns it** — read that, not this file.

### Standalone layers (top-level files)

`ls src/dl_techniques/layers/*.py` is the authoritative list. As of this reorg, 13 files remain at
`layers/` root — each a single-architecture or general-utility layer with no sibling file in scope
for a subpackage (see D-005: a 1-file subpackage adds a directory for zero organizational gain):

| File | Holds |
|---|---|
| `anchor_generator.py` | `AnchorGenerator` — precomputed multi-scale detection anchor/center grid |
| `capsules.py` | `PrimaryCapsule`, `RoutingCapsule`, `CapsuleBlock` — capsule layers with dynamic routing |
| `eomt_mask.py` | `EomtMask` — query-token class+mask segmentation head for Encoder-only Mask Transformer |
| `fnet_encoder_block.py` | `FNetEncoderBlock` — full FNet encoder block (Fourier token mixing + FFN) |
| `fractal_block.py` | `FractalBlock` — recursive FractalNet block |
| `hierarchical_mlp_stem.py` | `HierarchicalMLPStem` — hierarchical non-overlapping-conv ViT patch stem |
| `io_preparation.py` | `ClipLayer` (+ normalize/denormalize sibling) — tensor clipping/normalization pre/post-processing |
| `modality_projection.py` | `ModalityProjection` — pixel-shuffle + dense projection of vision tokens into a language embedding space |
| `mothnet_blocks.py` | `AntennalLobeLayer`, `MushroomBodyLayer`, `HebbianReadoutLayer` — insect-olfaction-inspired few-shot feature cascade |
| `one_hot_encoding.py` | `OneHotEncoding` — in-graph multi-column one-hot encoder |
| `spatial_layer.py` | `SpatialLayer` (+ `coordinate_grid`, `interpolate_grid`) — CoordConv-style coordinate grid + bilinear grid sampling |
| `standard_blocks.py` | `ConvBlock`, `DenseBlock`, `ResidualDenseBlock`, `BasicBlock`, `BottleneckBlock` — generic factory-driven building blocks; `ConvBlock` is a heavily-reused shared asset (attention/, transformers/, resnet, fractalnet, yolo12) |
| `thera_heat_field.py` | `ThermalActivation`, `HeatField` — THERA neural heat field for anti-aliased arbitrary-scale super-resolution |

### Naming traps

| Trap | Detail |
|---|---|
| **`RepMixerBlock` is two different architectures** | `fastvit/FastVitRepMixerBlock` (timm FastViT MCi, consumed by `models/vision/fastvit/`) is **NOT** the top-level `repmixer_block.py::RepMixerBlock` (consumed by `models/vision_language/fastvlm/`). The FastViT names carry a `FastVit` prefix precisely because the serialization registry is keyed by bare class name |
| **`MLPBlock` is `ffn/mlp.py`'s, and only its** | `tabm_blocks.py`'s two-Dense ensemble block is **`TabMMLPBlock`**. `ffn/mlp.py::MLPBlock` keeps the bare name because it is the FFN factory's `'mlp'` key, so moving it would move a public factory key. Same rule as `FastVitRepMixerBlock`: the narrower consumer takes the package prefix |
| **`Downsample` / `Upsample` are `ideogram4/vae.py`'s** | `models/vision/image_restoration/pw_fnet/model.py` spells its pair **`PWFNetDownsample`** / **`PWFNetUpsample`**. pw_fnet's is a strided `Conv2D` / `Conv2DTranspose`; ideogram4's is a kernel-4 conv with manual asymmetric padding / `UpSampling2D`+`Conv2D`. NOT interchangeable; do not merge them |

## Conventions

### `__init__.py` policy varies by subpackage — check before assuming

The `layers/__init__.py` root **is** empty. Most subpackages are **not**.

| Shape | Subpackages | How to import |
|---|---|---|
| **Curated re-export with `__all__`** | `activations`, `attention` (44 names), `embedding`, `fastvit`, `ffn`, `heads`, `logic`, `memory`, `mixtures`, `moe`, `norms`, `sequence_pooling`, `time_series`, `transformers` | `from dl_techniques.layers.attention import MultiHeadAttention, create_attention_layer` |
| **Empty** | `fusion`, `geometric`, `graphs`, `physics`, `reasoning`, `statistics`, `tokenizers`, `complex`, `conv_blocks`, `signal_processing`, `pooling`, `structured_linear`, `regularization`, `generative`, `blt`, `acc_unet`, `yolo12`, `tabular`, and the top-level standalone modules | `from dl_techniques.layers.graphs.graph_neural_network import GraphNeuralNetwork` |
| **No `__init__.py` at all** | `experimental/` — a namespace package | submodule imports only |

Submodule imports keep working in both cases — e.g.
`from dl_techniques.layers.attention.multi_head_attention import MultiHeadAttention` — but for a
subpackage with an `__all__`, prefer the package-level import.

### Other conventions

- Docstrings in `layers/` use **Sphinx/reST** (`:param:` / `:type:` / `:raises:`), not Google
  `Args:`, in the large majority of modules. In `attention/` it is mandatory;
  `attention/channel_attention.py` is the reference exemplar.
- Subpackages with `factory.py` support config-driven construction.
- All layers implement `get_config()` for Keras serialization.
- Layers follow the Keras 3 pattern: `__init__`, `build`, `call`, `get_config`.

## Layer Reuse Policy (factory-first)

> **Before implementing ANY new layer, check for an existing one to reuse.** A bespoke layer is the
> last resort, not the first move.

Check in this precedence order; proceed to the next step only when nothing fits.

1. **The relevant domain factory** — each exposes a `create_*_layer()` entry point backed by a registry of named types. Pass a `type` string + config; do not hand-roll what a factory already builds.

   | Domain | Factory entry point | Registered types |
   |--------|---------------------|------------------|
   | Normalization | `create_normalization_layer()` in `norms/factory.py` | 18 |
   | Attention | `create_attention_layer()` in `attention/factory.py` | 34 |
   | FFN / MLP | `create_ffn_layer()` in `ffn/factory.py` | 21 |
   | Embeddings | `create_embedding_layer()` in `embedding/factory.py` | 15 |
   | Activations | `create_activation_layer()` in `activations/factory.py` | 24 |
   | Sequence pooling | `create_sequence_pooling_layer()` in `sequence_pooling/factory.py` | 3 (`sequence`, `attention`, `weighted`) |
   | Logic | `create_logic_layer()` in `logic/factory.py` | 4 (`logic`, `arithmetic`, `neural_circuit`, `circuit_depth`) |
   | Mixtures | `create_mixture_layer()` in `mixtures/factory.py` | 3 (RBF / KMeans / GMM) |
   | Memory | `create_mann()` / `create_som_2d()` in `memory/factory.py` | n/a |
   | Task heads | `create_head(domain, ...)` in `heads/factory.py` (NLP / vision / VLM) | n/a |
   | Transformer blocks | `TransformerLayer` in `transformers/transformer.py` (direct import) | n/a |

   > **Transformer blocks have no `create_*_layer` factory.** Use `TransformerLayer` directly — its config selects attention / FFN / normalization types and normalization position, and it composes the domain factories internally, so it covers most cases without a custom block. Higher-level `create_*_encoder` builders live in `transformers/vision_encoder.py` and `transformers/text_encoder.py`.

2. **An existing standalone layer** — search the subpackages and the standalone list above before writing your own. The reuse surface is broad; a close match often already exists.

3. **Only then, a new custom layer**, following `research/2026_keras_custom_models_instructions_v2.md` (full serialization, `build`, `get_config`, tests). Place it in the right domain subpackage and, where that domain has a `factory.py`, register it there so the next author can reuse it.

### The factory contract

**1. Every factory RAISES `ValueError` on an undeclared keyword.** Attention, ffn, norms,
activations and embedding each reject a `bogus_key=1`; a new factory, or any new registry-backed
dispatch, does the same. **Never filter-and-drop** — each of these was silent, and passed its tests:

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

**4. When you audit "who calls factory X", also sweep "who builds X's argument dict without calling
X directly."** Files that pass an `ffn_args=` dict to a wrapper rather than calling
`create_ffn_layer` themselves are invisible to an AST call inventory of the factory, and invisible
to a suite sweep run at site defaults (the break needs a non-empty caller dict to appear). A
`**kwargs`-splat site has the identical blind spot. Use `assemble_ffn_config()` /
`assemble_attention_config()` so those dicts go through the same validation —
`models/language/qwen/qwen3.py` is the repaired exemplar and carries a comment explaining why it
must not be simplified back to a bare dict literal.

**5. Normalization epsilon.** The factory sets `1e-6`; Keras' `LayerNormalization` /
`BatchNormalization` default to `1e-3` — 1000x in every denominator, with no shape symptom and no
warning. Route normalization through the factory. Constructing directly makes `epsilon=` mandatory,
with a cited reference. Do not blanket-fix — MobileNet's reference BN genuinely is `1e-3`.

## Testing

Tests in `tests/test_layers/` organized by subdomain (attention, embeddings, ffn, norms, graphs, moe, etc.).
