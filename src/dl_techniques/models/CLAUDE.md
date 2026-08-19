# Models Package

Complete model architectures organized as subdirectories. Each subdirectory is a self-contained model implementation.

> **ALL model work MUST follow `research/2026_keras_custom_models_instructions.md`.** Read it before creating a new model directory *or authoring any new layer inside a model* — it is the canonical guide for Keras 3 custom authoring in this repo (serialization, `build`, `get_config`, factories, tests). This is non-negotiable for every model in this package, new or existing.

## Model Categories

### Vision
- `mobilenet/` — MobileNet variants (V1, V2, V3, V4)
- `resnet/` — ResNet architectures
- `convnext/` — ConvNeXt
- `fastvit/` — FastViT MCi image backbone (the assembled tower over `layers/fastvit/`; the image branch of MobileCLIP2, also usable standalone — architecture only, no pretrained weights)
- `convunext/` — ConvUNeXt (U-Net + ConvNeXt)
- `squeezenet/` — SqueezeNet
- `fastvlm/` — despite the `VLM` in its name and its former listing under Vision-Language, this is a **vision-only** hybrid backbone (MobileOne stem + RepMixer + attention stages). No text tower, no tokenizer, no language input — its own `README.md:1` says so: "A Fast Hybrid Vision Model"
- `vit/` — Vision Transformer
- `vit_hmlp/` — ViT with hierarchical MLP
- `vit_siglip/` — ViT with a two-stage conv patch-embedding stem. **The name misattributes**: SigLIP is a sigmoid contrastive LOSS, and its own tower has a single-conv stem, no CLS token and a MAP head — none of which this package shares. There is no text tower and no loss here. See `vit_siglip/model.py`'s module docstring
- `beit/` — BEiT (masked image modeling over discrete visual tokens + classifier; trainers in `src/train/beit/`)
- `swin_transformer/` — Swin Transformer
- `dino/` — DINO self-supervised
- `masked_autoencoder/` — MAE
- `depth_anything/` — Depth estimation
- `SAM/SAM1/`, `SAM/SAM2/`, `SAM/SAM3/` — Segment Anything Model (v1, v2, v3)
- `detr/` — DEtection TRansformer
- `yolo12/` — YOLOv12 detection
- `pft_sr/` — Super-resolution
- `scunet/` — SCUNet denoiser
- `darkir/` — DarkIR image restoration
- `cbam/` — CBAM attention model
- `accunet/` — AccuNet
- `fractalnet/` — FractalNet
- `bias_free_denoisers/` — Bias-free denoiser models
- `video_jepa/` — Video JEPA (joint embedding predictive)
- `superpoint/` — SuperPoint keypoint detector + descriptor
- `thera/` — THERA aliasing-free arbitrary-scale super-resolution
- `sd3_mmdit/` — SD3 MMDiT dual-stream text-to-image diffusion transformer
- `ideogram4/` — Ideogram4 text-to-image flow-matching DiT
- `energy_transformer/` — Energy Transformer (masked image completion + classifier)

### NLP / Language
- `bert/` — BERT
- `modern_bert/` — ModernBERT
- `distilbert/` — DistilBERT
- `gemma/` — Gemma LLM
- `qwen/` — Qwen LLM
- `masked_language_model/` — MLM training
- `byte_latent_transformer/` — Byte Latent Transformer (BLT)
- `gpt2/` — GPT-2 architecture
- `wave_field/` — Wave-field LLM
- `memory_bank/` — Memory-bank language model components
- `power_sampling/` — Inference-time power sampling for any causal LM/VLM

### Vision-Language
- `clip/` — CLIP
- `mobile_clip/` — MobileCLIP: **both generations in one package**. `mobile_clip_v1.py` is
  deliberately non-faithful on the image side (`keras.applications` substitutes, its own
  D-001); `mobile_clip_v2.py` is the faithful MobileCLIP2 (architecture only, no pretrained
  weights, no accuracy claim). They share `components.py`'s text tower AND one structure:
  a nested `MODEL_VARIANTS` CLASS attribute (`embed_dim` + `image_config` + `text_config`)
  and a constructor over those two sub-dicts. Neither deprecates the other — see the
  package README §17
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
- `graph_energy_transformer/` — Graph Energy Transformer (node anomaly + graph classification)

### Specialized Architectures
- `capsnet/` — Capsule Networks
- `kan/` — Kolmogorov-Arnold Networks
- `ntm/` — Neural Turing Machine
- `vae/` — Variational Autoencoder (ResNet encoder/decoder; `sampling_type` ∈ {`gaussian`, `hypersphere`, `vmf`} — `vmf` is a true von Mises-Fisher Spherical VAE with the closed-form vMF→uniform-sphere KL; see the package README §16–17)
- `vq_vae/` — VQ-VAE
- `vq_vae_rotation/` — VQ-VAE with rotation-based codebook updates
- `lewm/` — Latent-energy world model
- `nam/` — Neural Arithmetic **Module**: a tree-transformer parse + NTM memory + TRM
  halting stack that evaluates arithmetic expressions. **Not** a Neural Additive Model —
  there is no per-feature additive model anywhere in the package. This entry said
  "Neural additive model" until 2026-08-15; `nam/model.py`'s own module docstring has
  always said Module.
- `fnet/` — FNet (Fourier)
- `fftnet/` — FFTNet
- `pw_fnet/` — **PW-FNet, "Pyramid Wavelet-Fourier Network" for image restoration** (the package's own name: `pw_fnet/model.py:2`). It was listed here as "Patchwise FNet" until 2026-08-19 and is **not** patchwise and **not** FNet: it is a 2-level U-Net with FFT token mixing and multi-scale supervision, not FNet's token+feature-axis Fourier mixing. There is no wavelet transform either — the only spectral ops are `FFTLayer` / `IFFTLayer` (`grep -rn -i wavelet pw_fnet/` returns only the name itself), so two of the three words in the expanded name are unearned by the code
- `power_mlp/` — Power MLP
- `mothnet/` — MothNet (bio-inspired)
- `coshnet/` — CoshNet
- `latent_gmm_registration/` — Latent GMM registration
- `mini_vec2vec/` — Mini Vec2Vec
- `hierarchical_reasoning_model/` — HRM
- `tiny_recursive_model/` — Tiny recursive model
- `tree_transformer/` — Tree Transformer

## Conventions

- `src/dl_techniques/models/__init__.py` (the parent) is empty — always import
  from the model subpackage, never from `dl_techniques.models` itself.
- Per-model `<pkg>/__init__.py` is **mixed, and the empty case is no longer a
  safe default assumption**. Re-measured 2026-08-14 (second pass, after the `mobile_clip_v2/` split): 27 of the 73 model packages
  bind a `create_*` factory in their own `__init__.py` (an import, a `def` or an
  assignment). A plain `grep create_` now agrees at 27 — it used to give one more
  than the binding count, because `convnext_patch_vae/__init__.py` mentioned its
  factories in a docstring without binding them, and that package has since been
  deleted; the two figures can diverge again the moment any init mentions a
  factory it does not bind. Exemplars of a curated init with `__all__`:
  `energy_transformer/`, `dino/`, `vit/`. The remaining 46 are empty or
  near-empty and do require importing from the submodule directly. **Read the
  package init before assuming either shape.** (`REPO_MAP.md` § "The factory
  convention is not universal" carries the same derivation.)
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

   | Domain | Factory entry point |
   |--------|---------------------|
   | Normalization | `create_normalization_layer()` in `src/dl_techniques/layers/norms/factory.py` |
   | Attention | `create_attention_layer()` in `src/dl_techniques/layers/attention/factory.py` |
   | FFN / MLP | `create_ffn_layer()` in `src/dl_techniques/layers/ffn/factory.py` |
   | Embeddings | `create_embedding_layer()` in `src/dl_techniques/layers/embedding/factory.py` |
   | Activations | `create_activation_layer()` in `src/dl_techniques/layers/activations/factory.py` |
   | Transformer blocks | `TransformerLayer` in `src/dl_techniques/layers/transformers/transformer.py` (direct import) |

   > **Note on transformer blocks**: `transformers/` has no `create_*_layer` factory. Use `TransformerLayer` directly — it is highly configurable (selectable attention / FFN / normalization types and normalization position via its config) and composes the factories above internally, so it covers most cases without a custom block. The package also offers higher-level `create_*_encoder` builders (`vision_encoder.py`, `text_encoder.py`).

2. **The broader `layers/` package** — if no factory covers your need, search `src/dl_techniques/layers/` (20+ subpackages of standalone layers) for an existing implementation before writing your own.

3. **Only then, a new custom layer** — if nothing above fits, implement it following `research/2026_keras_custom_models_instructions.md` (full serialization, `build`, `get_config`, tests). Prefer adding it to the appropriate `layers/` subpackage (and its factory registry, where one exists) over burying it inside the model directory, so the next author can reuse it too.

## House Model Module Shape

> **Reference implementation: `models/resnet/model.py`.** Read the spec below rather than
> copying the file blind — the shape spread through this package by copy-paste, and that is
> exactly how its defects (placeholder weight URLs, mutable default args, a `logger.info`
> inside `call()`) reached `convnext/`, `mobilenet/` and others.

This shape applies to a package that implements **one architecture with named variants**. It
is a target, not a universal law: see "When the shape does not apply" at the end.

### Axis 1 — module skeleton

**The module docstring is substantive prose, not a template.** `models/resnet/model.py`
is the exemplar. Its shape:

1. **One opening sentence** naming the architecture and its distinguishing options —
   a sentence, not a title with an `====` underline.
2. **Prose explaining the principle**: what problem the architecture solves and *why its
   mechanism resolves it*, not just what the layers are. Inline math in backticks
   (`` `y = F(x) + x` ``) where an equation carries the idea.
3. **Prose on the architecture itself**: the stage/block structure, the design trade-offs,
   and — importantly — the places where the code does something non-obvious and why
   (ResNet's docstring explains exactly which shortcuts need a projection and which stay
   parameter-free, because that is the part a reader would otherwise get wrong).
4. **Any deliberate behavioural choice**, stated as a choice with its reason (e.g. why
   `pretrained=True` raises rather than warning and returning a random model).
5. **A `References:` section** listing papers as `- Author et al., YEAR. Title. (url)`.
   Include the papers the design actually draws on, not only the headline one.

What this replaces: terse `Model Variants:` / `Usage Examples:` boilerplate blocks that
restate the `MODEL_VARIANTS` dict and the factory signature. Those are already in the
code directly below; the docstring's job is the reasoning that is *not* in the code.

Read `models/resnet/model.py`'s docstring before writing one. Length follows the
architecture — ResNet's is ~70 lines because that much is genuinely worth saying. Do not
pad, and do not move real explanation to the README to hit a line budget; benchmark
tables and usage walkthroughs are what belongs in the README.

After the docstring: imports, the `# local imports` banner, `# -----` separator bars,
`@keras.saving.register_keras_serializable()`.

```python
import os
import keras
from typing import List, Optional, Union, Tuple, Dict, Any, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
...

# ---------------------------------------------------------------------
```

### Axis 2 — class API

```python
@keras.saving.register_keras_serializable()
class <Model>(keras.Model):
    MODEL_VARIANTS = {"<variant>": {...}}

    def __init__(self, ..., **kwargs):
        super().__init__(**kwargs)
        # 1. validate arguments, raising ValueError with the offending value
        # 2. resolve None-sentinel defaults (never a mutable default arg)
        # 3. store configuration on self
        # 4. call self._build_<part>() helpers
        # 5. ONE logger.info summarizing what was created
```

- `call(self, inputs, training=None)` — **no logging inside**; it fires on every trace.
- `get_config()` returns every constructor argument, with
  `keras.regularizers.serialize(...)` for regularizers; `from_config()` deserializes them.
- `from_variant(cls, variant, ..., pretrained=False, **kwargs)` looks the name up in
  `MODEL_VARIANTS`, raising `ValueError` listing the available keys when it misses.

`MODEL_VARIANTS` is the canonical name for **the registry of publicly named variants**.
Packages that predate this spec also use `VARIANT_CONFIGS`, `NAM_VARIANTS`, `NTM_VARIANTS`
or `MCI_VARIANTS` for that same role; where one of those is the package's *only* variant
table, add `MODEL_VARIANTS` as a class-level alias to the same dict.

**`SCALE_CONFIGS` is NOT a stale spelling of `MODEL_VARIANTS`, and the two must not be
merged where both appear.** They answer different questions, both defined at module
scope in `beit/model.py` (`SCALE_CONFIGS` and `MODEL_VARIANTS`):

- `SCALE_CONFIGS` is the **architecture table** — `'tiny' -> {hidden_size: 192,
  num_layers: 12, num_heads: 3, ...}`.
- `MODEL_VARIANTS` is the **public-name registry** — `'beit_tiny' -> {scale: 'tiny'}`,
  one row per name a caller may pass, resolving to a scale.

`beit`, `vit` and `energy_transformer` carry both, deliberately, and a
`_resolve_scale`-style helper accepts either spelling. (`fastvit` was named here as a
fourth until 2026-08-19 and carries NEITHER name in that role: its only table is the
module-level `MCI_VARIANTS`, and it has no `SCALE_CONFIGS` at all. That is the
single-table case, so the rule above applies to it and
`FastVitImageEncoder.MODEL_VARIANTS` is now the class-level alias — added 2026-08-19,
because tooling resolving `getattr(cls, "MODEL_VARIANTS")` raised `AttributeError`
there while this file asserted the opposite.) Unifying them would collapse a
name→scale indirection that exists so a variant can pin a patch size or an input
resolution alongside its scale. Only where a package has a single table does the alias
apply.

Prefer an alias over renaming in place: the old spelling is referenced by trainers under
`src/train/` and by tests, and the rename buys nothing the alias does not.

### Axis 3 — pretrained weights

`load_pretrained_weights(weights_path, skip_mismatch)` loads from a **local path**,
building the model with a dummy forward pass first if needed. There is **no `by_name`
parameter**: Keras 3 removed it from `Model.load_weights`, transfer here is
layer-by-layer via `utils/weight_transfer.py` and therefore always name-based, and
the argument survived for a while as an accepted no-op — deleted 2026-08-14 from all
nine packages that carried it (`resnet`, `bert`, `convnext_v1`, `convnext_v2`,
`fnet`, `modern_bert`, `cliffordnet`, `bias_free_denoisers/bfunet`, `kan`). Do not
reintroduce it.

`_download_weights(...)` **raises `NotImplementedError` naming the variant and showing the
local-path alternative.** No public checkpoints are distributed with `dl_techniques` for any
architecture in this package.

**Never** write a placeholder URL table plus a `try/except` in `from_variant` that logs a
warning and continues with random initialization. That combination means
`pretrained=True` silently returns an untrained model — a caller asking for pretrained
weights gets random ones and no error. This contract is pinned by tests in
`test_bert/`, `test_gpt2/`, `test_wave_field/`, `test_tree_transformer/`, `test_vit/`,
`test_cliffordnet/`, `test_xlstm/` and `test_capsnet/test_model_v2.py`.

### Axis 4 — factory and exports

A module-level `create_<name>(variant="<default>", ...)` that delegates to `from_variant` —
no logic of its own. The package `__init__.py` exports the class and the factory with an
`__all__`; `accunet/__init__.py` is the exemplar.

### Axis 5 — hygiene

- No comment that restates the line below it (`# Store configuration`, `# Squeeze`,
  `# Compute gradients`).
- No `# 1. / # 2. / # 3.` step ladders. A comment earns its place by explaining *why*, or by
  recording a non-obvious constraint — not by narrating *what*.
- No mutable default arguments; use `None` sentinels resolved in the body.
- No unused imports (an imported-but-never-called `logger` is the common case).
- Prefer `keras.ops`; `keras.config.floatx()` / `keras.config.epsilon()` over the
  `keras.backend.*` spellings.

### Things you must NOT do

- **Never delete or reword a `# DECISION <plan-id>/D-NNN` comment.** They resolve through the
  append-only manifest `plans/ANCHORS.md`; a comment-tidying sweep that removes one silently
  destroys the record. Files with a high comment density are usually dense *because of*
  these anchors — never target a file by comment density.
- **Never rename a module file to `model.py`.** `bert/bert.py`, `convnext/convnext_v1.py`
  and `mobilenet/mobilenet_v2.py` stay where they are; renaming breaks every import.
- **Never convert docstring style.** This package is measurably mixed; match the file you
  are editing (see `src/dl_techniques/CLAUDE.md` § Code Style).
- **Never delete the deep-supervision re-export shim** at the bottom of `resnet/model.py`,
  `convunext/model.py`, `bias_free_denoisers/bfunet.py` and
  `bias_free_denoisers/bfconvunext.py`. It is a deliberate late import with `# noqa: E402`.

### When the shape does not apply

- **No genuine named variants.** Do not invent a `MODEL_VARIANTS` table to satisfy the
  template. Apply axes 1, 4 and 5 only, and say why in the package `README.md`.
- **Functional builders** (`bias_free_denoisers/`, `convunext/`, `darkir/`) return
  `keras.Model(inputs, outputs)` and have no subclass. Keep them functional — converting
  them would break existing checkpoints, and the `bias_free_denoisers` ones are actively
  used by `src/train/bfunet/` and `src/applications/`. Axes 1, 4 and 5 still apply.
  (`detr/` was listed here in the first draft of this section and does **not** belong:
  `detr/model.py` defines `class DETR(models.Model)`. Verify with
  `grep -n "^class .*(.*Model)" <pkg>/*.py` before classifying a package — the census
  that produced the original list was grep-based and wrong about several packages.)
- **Multi-model families and nested packages** (`SAM/`, `time_series/`, `dino/`,
  `ideogram4/`, `sd3_mmdit/`) apply the shape per *inner architecture*, not per directory.

## Consumer-less packages: CARRY (ruling, 2026-08-19)

25 packages under `models/` (~19.3k LOC) have **no consumer outside their own package
and tests** — no trainer under `src/train/`, no application under `src/applications/`,
no import from another `models/` package:

`cbam` `detr` `distilbert` `fastvlm` `fftnet` `fractalnet` `gemma`
`latent_gmm_registration` `mamba` `memory_bank` `mini_vec2vec` `mothnet`
`nano_vlm_world_model` `pft_sr` `pw_fnet` `qwen` `relgt` `scunet` `shgcn` `som`
`squeezenet` `swin_transformer` `vit_hmlp` `vit_siglip` `vq_vae`

**They are CARRIED. Do not propose deleting them as dead code.** All 25 have test
suites, and a tested package with no trainer is *library surface*, not dead code: this
repo distributes reusable layers and models, and "has a trainer" was never the shipping
criterion. Deletion is not reversible for a downstream user who has pinned an import.

This question was re-opened by two separate review rounds (2026-08-17, 2026-08-18) and
ruled the same way both times. The ruling lives here rather than in a planning document
because `plans/` is gitignored and does not ship. Re-derive the list before acting on
it — it moves as trainers are added or packages are merged (e.g. `convunext`,
`bfconvunext` and `bfcnn` all left this list when their trainers were consolidated).

## Testing

Tests in `tests/test_models/` with one subdirectory per model — 81 test directories
as of 2026-08-14, more than the 73 model packages because several architectures get
more than one suite. One exception: `lewm` is tested by the loose
`tests/test_models/test_lewm.py`, so a directory-to-directory comparison will
wrongly report it as untested. Test pattern:
- Class-based organization: `class TestModelName`
- Tests cover: serialization, initialization, forward pass, gradient flow, training mode, variants, edge cases
- Pytest fixtures provide model configs
