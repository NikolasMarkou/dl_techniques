# `dl_techniques.models`

Complete model architectures — **85 leaf packages** grouped into **12 family directories**.
A *leaf package* is a directory with an `__init__.py` and no `__init__.py`-bearing child; it
holds one architecture, its blocks, usually a factory, and a `README.md` (85 of 85 have one).
The family directory above it is a filing decision, not a namespace.

This file is the orientation map. For authoring rules, the per-leaf census, the house module
shape and the review findings, read [`CLAUDE.md`](CLAUDE.md) — it is the longer, normative
document and this README does not restate it.

## The one import rule

`models/__init__.py` is **empty on purpose**, and so is every family `__init__.py` except
`time_series/`. Always import from the leaf package:

```python
from dl_techniques.models.vision.resnet import create_resnet          # yes
from dl_techniques.models.language.bert import BertModel              # yes
from dl_techniques.models import resnet                               # no - nothing is re-exported
```

Family-level re-exports were considered and rejected: `import dl_techniques.models.vision`
would then eagerly construct all 35 vision packages — the whole Keras/TensorFlow import cost
of the family to reach one model — and it opens a circular-import surface between packages
that share layers. The family `__init__.py` files carry a docstring listing their members and
nothing else. `time_series/` is the single exception; it predates this layout, has 7 children
rather than 35, and its consumers rely on the re-exports.

## The 12 families

Counts are **leaf counts**, which for `vision` and `vision_language` are larger than the
direct-child count, because those two nest one level further. Re-derive with the `find` in
`CLAUDE.md` § Layout.

| Family | Leaves | What it holds |
|---|---|---|
| [`vision/`](vision/) | **35** | image backbones, detectors, segmenters, denoisers, generators |
| [`language/`](language/) | 17 | token-sequence models: encoders, decoders, SSMs, reasoning stacks |
| [`vision_language/`](vision_language/) | **9** | models consuming an image and a text stream (plus one that does not — see below) |
| [`time_series/`](time_series/) | 8 | forecasting, probabilistic and point |
| [`embeddings_experimental/`](embeddings_experimental/) | 4 | ASCII text-embedding encoders, built to be compared against each other |
| [`general_purpose/`](general_purpose/) | 3 | architecture-level MLP replacements, modality-agnostic |
| [`graph/`](graph/) | 3 | models over explicit graph inputs |
| [`neural_computer/`](neural_computer/) | 2 | external-memory / differentiable-computer architectures |
| [`common/`](common/) | 1 | model-agnostic inference machinery |
| [`memory/`](memory/) | 1 | learned codebook topologies |
| [`point_cloud/`](point_cloud/) | 1 | 3D point set models |
| [`tabular/`](tabular/) | 1 | tabular-data models |
| **Sum** | **85** | |

### `vision/` (35)

| Package | |
|---|---|
| `accunet/` | AccuNet |
| `beit/` | BEiT — masked image modeling over discrete visual tokens, plus classifier |
| `bias_free_denoisers/` | bias-free denoiser models |
| `capsnet/` | Capsule Networks |
| `cbam/` | CBAM attention model |
| `cliffordnet/` | Clifford-algebra networks |
| `convnext/` | ConvNeXt |
| `convunext/` | ConvUNeXt — U-Net + ConvNeXt |
| `coshnet/` | CoshNet |
| `depth_anything/` | depth estimation |
| `detr/` | DEtection TRansformer |
| `dino/` | DINO self-supervised |
| `energy_transformer/` | Energy Transformer — masked image completion, plus classifier |
| `fastvit/` | FastViT MCi image backbone, the assembled tower over `layers/fastvit/` |
| `fractalnet/` | FractalNet |
| `image_restoration/darkir/` | DarkIR low-light restoration |
| `image_restoration/pw_fnet/` | 2-level U-Net, FFT token mixing, multi-scale supervision. No wavelet op despite the paper's pyramid-wavelet design |
| `image_restoration/scunet/` | SCUNet denoiser |
| `keypoints/superpoint/` | SuperPoint keypoint detector + descriptor |
| `lewm/` | latent-energy world model |
| `masked_autoencoder/` | MAE |
| `mobilenet/` | MobileNet V1, V2, V3, V4 |
| `resnet/` | ResNet architectures |
| `squeezenet/` | SqueezeNet |
| `super_resolution/pft_sr/` | PFT-SR progressive focused transformer |
| `swin_transformer/` | Swin Transformer |
| `thera/` | THERA aliasing-free arbitrary-scale super-resolution |
| `vae/` | Variational Autoencoder, ResNet encoder/decoder, Gaussian / hypersphere / vMF sampling |
| `video_jepa/` | Video JEPA, joint embedding predictive |
| `vit/` | Vision Transformer |
| `vit_hmlp/` | ViT with hierarchical MLP |
| `vit_siglip/` | ViT with a two-stage conv patch-embedding stem. Name misattributes |
| `vq_vae/` | VQ-VAE |
| `vq_vae_rotation/` | VQ-VAE with rotation-based codebook updates |
| `yolo12/` | YOLOv12 detection |

### `language/` (17)

| Package | |
|---|---|
| `bert/` | BERT — `bert/model.py` is the normative exemplar for a new model package |
| `byte_latent_transformer/` | Byte Latent Transformer (BLT) |
| `colbert/` | ColBERT v1/v2, late-interaction retrieval |
| `distilbert/` | DistilBERT |
| `fftnet/` | FFTNet |
| `fnet/` | FNet, Fourier token mixing |
| `gemma/` | Gemma LLM |
| `gpt2/` | GPT-2 architecture |
| `hierarchical_reasoning_model/` | HRM |
| `mamba/` | Mamba, selective state space |
| `masked_language_model/` | MLM training |
| `mini_vec2vec/` | Mini Vec2Vec |
| `modern_bert/` | ModernBERT |
| `qwen/` | Qwen LLM |
| `tiny_recursive_model/` | tiny recursive model |
| `tree_transformer/` | Tree Transformer |
| `wave_field/` | wave-field LLM |

Packages here are filed by input modality — token sequences — which is not a claim that
`mamba`, `fnet` or `fftnet` is only a language model.

### `vision_language/` (9)

| Package | |
|---|---|
| `clip/` | CLIP |
| `fastvlm/` | vision-only hybrid backbone: MobileOne stem, RepMixer, attention stages. Name misattributes |
| `ideogram4/` | Ideogram4 text-to-image flow-matching DiT |
| `mobile_clip/` | MobileCLIP, both generations in one package: `mobile_clip_v1.py` is deliberately non-faithful on the image side, `mobile_clip_v2.py` is the faithful MobileCLIP2. Neither deprecates the other |
| `nano_vlm/` | NanoVLM |
| `sam/sam1/` | Segment Anything v1 |
| `sam/sam2/` | Segment Anything v2, memory bank for video |
| `sam/sam3/` | Segment Anything v3, text-promptable |
| `sd3_mmdit/` | SD3 MMDiT dual-stream text-to-image diffusion transformer |

### `time_series/` (8)

| Package | |
|---|---|
| `adaptive_ema/` | adaptive EMA model |
| `deepar/` | DeepAR probabilistic forecasting |
| `ets/` | Pure additive exponential smoothing (the one recursive forecaster) |
| `mdn/` | Mixture Density Networks |
| `nbeats/` | N-BEATS, plus the exogenous `nbeatsx` variant |
| `prism/` | PRISM forecasting |
| `tirex/` | TiReX |
| `xlstm/` | xLSTM |

### The small families

| Package | |
|---|---|
| `general_purpose/kan/` | Kolmogorov-Arnold Networks |
| `general_purpose/mothnet/` | MothNet, bio-inspired |
| `general_purpose/power_mlp/` | Power MLP |
| `graph/graph_energy_transformer/` | Graph Energy Transformer, node anomaly + graph classification |
| `graph/relgt/` | Relational Graph Transformer |
| `graph/shgcn/` | Simplified Hyperbolic GCN |
| `neural_computer/nam/` | Neural Arithmetic **Module**. Name misattributes |
| `neural_computer/ntm/` | Neural Turing Machine |
| `common/power_sampling/` | inference-time power sampling for any causal LM/VLM — model-agnostic, which is why it is not under `language/` |
| `memory/som/` | Self-Organizing Maps |
| `point_cloud/latent_gmm_registration/` | latent GMM registration |
| `tabular/tabm/` | TabM |
| `embeddings_experimental/shared/` | the study skeleton and block registry — not an architecture |
| `embeddings_experimental/ascii_bert/` | ASCII-vocabulary BERT, the study's transformer baseline |
| `embeddings_experimental/ascii_clifford_bert/` | same skeleton, attention-free Clifford mixing |
| `embeddings_experimental/ascii_convnext_bert/` | same skeleton, ConvNeXt V1 convolutional mixing |

## Second-level nesting

Four subdirectories group related architectures one level below a family. Each has a
docstring `__init__.py` and, like the families, exports nothing:

| Subfamily | Members |
|---|---|
| `vision/image_restoration/` | `darkir`, `pw_fnet`, `scunet` — plus `README.md` and `BENCHMARKS.md`, a transcribed literature survey whose PSNR/SSIM numbers all come from papers and none from this repository |
| `vision/keypoints/` | `superpoint` (one member today) |
| `vision/super_resolution/` | `pft_sr` (one member today) |
| `vision_language/sam/` | `sam1`, `sam2`, `sam3` |

`time_series/` is the only container that also owns a module of its own — `forecast.py`, the
shared `Forecast` / `ForecastMixin` — and the only one that re-exports its children.

## Names that misattribute

Three packages are named for something they are not. This is the full list; the evidence for
each correction is in the named package's own source, cited below, so nothing is duplicated
into `CLAUDE.md`.

| Package | Reads as | Actually is | Where the evidence sits |
|---|---|---|---|
| `vision_language/fastvlm/` | a vision-*language* model | **vision-only** — the package is `model.py` + `components.py`, with no text tower, no tokenizer and no vocabulary anywhere in it | `ls` the package; the only `token` hits in `model.py` are "token-mixing" and the stage-3 patch grid |
| `vision/vit_siglip/` | SigLIP | **a ViT with a two-stage conv patch-embedding stem**. SigLIP's contribution is a *loss* — the pairwise sigmoid objective — and this package contains no text tower and no loss at all, so nothing in it can be sigmoid-contrastive | `model.py` lines 5-12 and 75-76 say so at the site |
| `neural_computer/nam/` | Neural *Additive* Model | Neural Arithmetic **Module** — a differentiable-computer cell, not a GAM | `__init__.py` line 1 |

`vision/image_restoration/pw_fnet/` is **not** in that list: PW-FNet is the architecture's
own name in Jiang et al., *Global Modeling Matters: A Fast, Lightweight and Effective
Baseline for Efficient Image Restoration*. Its gap is an implementation one, not a naming
one — there is no wavelet or DWT operation in `model.py`, so the pyramid is built from the
encoder/decoder resolution ladder and multi-scale supervision alone. That package's own
`README.md` records the gap.

## What you may claim in these docs

Nothing under `models/` has been trained or benchmarked in this repository. Crediting the
*paper* with a measured result is accurate and belongs in a README; crediting *this code*
with one is not. Unverifiable maturity adjectives are rejected outright. Both rules are
enforced by `tests/test_docs_make_no_unearned_performance_claims.py`, which reads every `.md`
and `.py` under `models/` — including this file. Read its regexes before writing docs.

## Where to go next

| | |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | authoring rules, house module shape, per-leaf census, layer-reuse policy, review findings |
| `research/2026_keras_custom_models_instructions_v2.md` | **the mandatory guide** for creating any new model or any new layer inside one |
| `REPO_MAP.md` | repo-wide navigation: which trainer trains which model, and how registry/factory dispatch is wired |
| `tests/test_models/` | flat, one directory per leaf package; deliberately does not mirror this tree |
