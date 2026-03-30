# Models Package

150+ complete model architectures organized as subdirectories. Each subdirectory is a self-contained model implementation.

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

### NLP / Language
- `bert/` — BERT
- `modern_bert/` — ModernBERT
- `distilbert/` — DistilBERT
- `gemma/` — Gemma LLM
- `qwen/` — Qwen LLM
- `masked_language_model/` — MLM training
- `byte_latent_transformer/` — Byte Latent Transformer (BLT)

### Vision-Language
- `clip/` — CLIP
- `mobile_clip/` — MobileCLIP
- `fastvlm/` — FastVLM
- `nano_vlm/` — NanoVLM
- `nano_vlm_world_model/` — NanoVLM world model

### Time Series
- `deepar/` — DeepAR probabilistic forecasting
- `nbeats/` — N-BEATS
- `prism/` — PRISM forecasting
- `tirex/` — TiReX time series
- `adaptive_ema/` — Adaptive EMA model

### Sequence / State Space
- `mamba/` — Mamba (SSM)
- `xlstm/` — xLSTM

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
- `vae/` — Variational Autoencoder
- `vq_vae/` — VQ-VAE
- `mdn/` — Mixture Density Networks
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

## Testing

Tests in `tests/test_models/` with one subdirectory per model (45+ test suites). Test pattern:
- Class-based organization: `class TestModelName`
- Tests cover: serialization, initialization, forward pass, gradient flow, training mode, variants, edge cases
- Pytest fixtures provide model configs
