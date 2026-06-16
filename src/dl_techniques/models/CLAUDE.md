# Models Package

Complete model architectures organized as subdirectories. Each subdirectory is a self-contained model implementation.

> **ALL model work MUST follow `research/2026_keras_custom_models_instructions.md`.** Read it before creating a new model directory *or authoring any new layer inside a model* — it is the canonical guide for Keras 3 custom authoring in this repo (serialization, `build`, `get_config`, factories, tests). This is non-negotiable for every model in this package, new or existing.

## Model Categories

### Vision
- `mobilenet/` — MobileNet variants (V1, V2, V3)
- `resnet/` — ResNet architectures
- `convnext/` — ConvNeXt
- `convunext/` — ConvUNeXt (U-Net + ConvNeXt)
- `squeezenet/` — SqueezeNet
- `vit/` — Vision Transformer
- `vit_hmlp/` — ViT with hierarchical MLP
- `vit_siglip/` — ViT with SigLIP
- `swin_transformer/` — Swin Transformer
- `dino/` — DINO self-supervised
- `jepa/` — Joint Embedding Predictive Architecture
- `masked_autoencoder/` — MAE
- `depth_anything/` — Depth estimation
- `sam/` — Segment Anything Model
- `detr/` — DEtection TRansformer
- `yolo12/` — YOLOv12 detection
- `pft_sr/` — Super-resolution
- `scunet/` — SCUNet denoiser
- `darkir/` — DarkIR image restoration
- `cbam/` — CBAM attention model
- `accunet/` — AccuNet
- `fractalnet/` — FractalNet
- `bias_free_denoisers/` — Bias-free denoiser models
- `convnext_patch_vae/` — ConvNeXt patch-level VAE with SIGReg anti-collapse
- `video_jepa/` — Video JEPA (joint embedding predictive)

### NLP / Language
- `bert/` — BERT
- `modern_bert/` — ModernBERT
- `distilbert/` — DistilBERT
- `gemma/` — Gemma LLM
- `qwen/` — Qwen LLM
- `masked_language_model/` — MLM training
- `byte_latent_transformer/` — Byte Latent Transformer (BLT)
- `gpt2/` — GPT-2 architecture
- `wave_field_llm/` — Wave-field LLM
- `memory_bank/` — Memory-bank language model components

### Vision-Language
- `clip/` — CLIP
- `mobile_clip/` — MobileCLIP
- `fastvlm/` — FastVLM
- `nano_vlm/` — NanoVLM
- `nano_vlm_world_model/` — NanoVLM world model

### Time Series (`models/time_series/`)
- `time_series/deepar/` — DeepAR probabilistic forecasting
- `time_series/nbeats/` — N-BEATS
- `time_series/prism/` — PRISM forecasting
- `time_series/tirex/` — TiReX time series
- `time_series/adaptive_ema/` — Adaptive EMA model
- `time_series/mdn/` — Mixture Density Networks
- `time_series/xlstm/` — xLSTM

### Sequence / State Space
- `mamba/` — Mamba (SSM)

### Tabular
- `tabm/` — TabM for tabular data

### Graph / Geometric
- `cliffordnet/` — Clifford algebra networks
- `relgt/` — Relational Graph Transformer
- `shgcn/` — Simplified Hyperbolic GCN
- `som/` — Self-Organizing Maps

### Specialized Architectures
- `capsnet/` — Capsule Networks
- `kan/` — Kolmogorov-Arnold Networks
- `ntm/` — Neural Turing Machine
- `vae/` — Variational Autoencoder (ResNet encoder/decoder; `sampling_type` ∈ {`gaussian`, `hypersphere`, `vmf`} — `vmf` is a true von Mises-Fisher Spherical VAE with the closed-form vMF→uniform-sphere KL; see the package README §16–17)
- `vq_vae/` — VQ-VAE
- `vq_vae_rotation/` — VQ-VAE with rotation-based codebook updates
- `lewm/` — Latent-energy world model
- `nam/` — Neural additive model
- `fnet/` — FNet (Fourier)
- `fftnet/` — FFTNet
- `pw_fnet/` — Patchwise FNet
- `power_mlp/` — Power MLP
- `mothnet/` — MothNet (bio-inspired)
- `ccnets/` — CCNets
- `coshnet/` — CoshNet
- `latent_gmm_registration/` — Latent GMM registration
- `mini_vec2vec/` — Mini Vec2Vec
- `hierarchical_reasoning_model/` — HRM
- `tiny_recursive_model/` — Tiny recursive model
- `tree_transformer/` — Tree Transformer

## Conventions

- `__init__.py` is empty — import from model subdirectories directly
- Each model subdirectory typically contains:
  - Model definition module(s)
  - Block/layer definitions specific to that architecture
  - Optional `train.py` or training utilities
- All models follow Keras 3 patterns with full `get_config()` serialization
- Models use config dicts (`Dict[str, Any]`) for construction parameters
- Factory patterns are common for creating model variants

## Layer Reuse Policy (factory-first)

> **Before implementing ANY new layer, you MUST first check for an existing one to reuse.** Authoring a bespoke layer is the last resort, not the first move — the library already ships a large, tested layer surface.

Check in this precedence order; only proceed to the next step when nothing fits:

1. **The relevant layer factory** — each factory exposes a `create_*_layer()` entry point backed by a registry of named types. Pass a `type` string + config; do not hand-roll what a factory already builds.

   | Domain | Factory entry point | Registered types |
   |--------|---------------------|------------------|
   | Normalization | `create_normalization_layer()` in `src/dl_techniques/layers/norms/factory.py` | ~16 |
   | Attention | `create_attention_layer()` in `src/dl_techniques/layers/attention/factory.py` | ~29 |
   | FFN / MLP | `create_ffn_layer()` in `src/dl_techniques/layers/ffn/factory.py` | ~15 |
   | Embeddings | `create_embedding_layer()` in `src/dl_techniques/layers/embedding/factory.py` | ~13 |
   | Activations | `create_activation_layer()` in `src/dl_techniques/layers/activations/factory.py` | ~22 |
   | Transformer blocks | `TransformerLayer` in `src/dl_techniques/layers/transformers/transformer.py` (direct import) | n/a |

   > **Note on transformer blocks**: `transformers/` has no `create_*_layer` factory. Use `TransformerLayer` directly — it is highly configurable (selectable attention / FFN / normalization types and normalization position via its config) and composes the factories above internally, so it covers most cases without a custom block. The package also offers higher-level `create_*_encoder` builders (`vision_encoder.py`, `text_encoder.py`).

2. **The broader `layers/` package** — if no factory covers your need, search `src/dl_techniques/layers/` (20+ subpackages of standalone layers) for an existing implementation before writing your own.

3. **Only then, a new custom layer** — if nothing above fits, implement it following `research/2026_keras_custom_models_instructions.md` (full serialization, `build`, `get_config`, tests). Prefer adding it to the appropriate `layers/` subpackage (and its factory registry, where one exists) over burying it inside the model directory, so the next author can reuse it too.

## Testing

Tests in `tests/test_models/` with one subdirectory per model (45+ test suites). Test pattern:
- Class-based organization: `class TestModelName`
- Tests cover: serialization, initialization, forward pass, gradient flow, training mode, variants, edge cases
- Pytest fixtures provide model configs
