# Models Package

Complete model architectures organized as subdirectories. Each subdirectory is a self-contained model implementation.

> **ALL model work MUST follow `research/2026_keras_custom_models_instructions_v2.md`.** Read it before creating a new model directory *or authoring any new layer inside a model* — it is the canonical guide for Keras 3 custom authoring in this repo (serialization, `build`, `get_config`, factories, tests). This is non-negotiable for every model in this package, new or existing.

## Model Categories

### Vision
- `mobilenet/` — MobileNet variants (V1, V2, V3, V4)
- `resnet/` — ResNet architectures
- `convnext/` — ConvNeXt
- `fastvit/` — FastViT MCi image backbone (the assembled tower over `layers/fastvit/`; the image branch of MobileCLIP2, also usable standalone — architecture only, no pretrained weights)
- `convunext/` — ConvUNeXt (U-Net + ConvNeXt)
- `squeezenet/` — SqueezeNet
- `fastvlm/` — vision-only hybrid backbone (MobileOne stem + RepMixer + attention stages). **Name misattributes** — see below
- `vit/` — Vision Transformer
- `vit_hmlp/` — ViT with hierarchical MLP
- `vit_siglip/` — ViT with a two-stage conv patch-embedding stem. **Name misattributes** — see below
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
- `nam/` — Neural Arithmetic **Module**: tree-transformer parse + NTM memory + TRM halting stack
  that evaluates arithmetic expressions. **Name misattributes** — see below
- `fnet/` — FNet (Fourier)
- `fftnet/` — FFTNet
- `pw_fnet/` — a 2-level U-Net with FFT token mixing and multi-scale supervision, for image
  restoration. **Name misattributes** — see below
- `power_mlp/` — Power MLP
- `mothnet/` — MothNet (bio-inspired)
- `coshnet/` — CoshNet
- `latent_gmm_registration/` — Latent GMM registration
- `mini_vec2vec/` — Mini Vec2Vec
- `hierarchical_reasoning_model/` — HRM
- `tiny_recursive_model/` — Tiny recursive model
- `tree_transformer/` — Tree Transformer

### Names that misattribute

Four packages are named for something they are not. Each was corrected in place after measurement;
the entry is kept so the correction is not silently re-reverted.

| Package | The name claims | What the code is | Evidence |
|---|---|---|---|
| `fastvlm/` | a vision-**language** model, and it was formerly listed under Vision-Language | **vision-only**: no text tower, no tokenizer, no language input | its own `README.md:1` — "A Fast Hybrid Vision Model" |
| `vit_siglip/` | SigLIP | a ViT with a two-stage conv patch-embedding stem. SigLIP is a sigmoid contrastive **loss**, and its own tower has a single-conv stem, no CLS token and a MAP head — none of which this package shares. There is no text tower and no loss here | `vit_siglip/model.py` module docstring |
| `nam/` | (read as) Neural **Additive** Model | Neural Arithmetic **Module** — there is no per-feature additive model anywhere in the package | `nam/model.py`'s module docstring has always said Module; this catalogue said "Neural additive model" until 2026-08-15 |
| `pw_fnet/` | "Pyramid **Wavelet**-Fourier Network" (the package's own name, `pw_fnet/model.py:2`), and it was listed here as "Patchwise FNet" until 2026-08-19 | neither patchwise nor FNet: a 2-level U-Net with FFT token mixing and multi-scale supervision, not FNet's token+feature-axis Fourier mixing. **No wavelet transform** — the only spectral ops are `FFTLayer` / `IFFTLayer` | `grep -rn -i wavelet pw_fnet/` returns only the name itself. Two of the three words in the expanded name are unearned |

> A class or package sharing a name with a published architecture is not necessarily that
> architecture. Check the composition rule, not the name.

## Conventions

- `src/dl_techniques/models/__init__.py` (the parent) is empty — always import
  from the model subpackage, never from `dl_techniques.models` itself.
- **Per-model `<pkg>/__init__.py` is now curated almost everywhere — the "mostly
  empty" era is over.** Re-measured 2026-08-19 over the 73 model packages:
  **73 of 73** have a non-empty `__init__.py`; **72 of 73** declare `__all__`
  (`SAM/` is the sole exception, and it is the multi-model family package);
  **69 of 73** actually *bind* a `create_*` name (an import, a `def` or an
  assignment). Import the public name straight from the package. Exemplars:
  `accunet/`, `energy_transformer/`, `dino/`, `vit/`.

  Re-derive with an **AST walk**, not a grep — a plain `grep -l create_` over the
  same inits returns **71**, because two of them mention a factory in a docstring
  or comment without binding it. The gap between those two numbers is the whole
  reason to prefer the AST:

  ```bash
  # non-empty inits / inits declaring __all__
  for d in src/dl_techniques/models/*/; do [ -s "$d/__init__.py" ] && echo "$d"; done | wc -l
  grep -l "^__all__" src/dl_techniques/models/*/__init__.py | wc -l
  ```

  This paragraph asserted "27 of 73 bind a factory, the remaining 46 are empty or
  near-empty" from a 2026-08-14 measurement. The package surface was curated
  repo-wide after that and the claim inverted; it was corrected on 2026-08-19.
  The census is guarded by `tests/test_models/test_package_api_contract.py`, so
  prefer running that over re-deriving by hand.
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

3. **Only then, a new custom layer** — if nothing above fits, implement it following `research/2026_keras_custom_models_instructions_v2.md` (full serialization, `build`, `get_config`, tests). Prefer adding it to the appropriate `layers/` subpackage (and its factory registry, where one exists) over burying it inside the model directory, so the next author can reuse it too.

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
- **A module rename must carry every referencing site in the same commit.** `bert/` now
  holds `model.py`, renamed from `bert/bert.py` on 2026-08-24 by commit `2c4a0ca7c`
  ("[models] cleaning up models documentation and structure") — not by the merge
  `4ed922781` that carried it in. `convnext/convnext_v1.py` and `mobilenet/mobilenet_v2.py`
  genuinely do stay where they are, and for a reason that does not generalize to `bert/`:
  they are *versioned variants* sitting beside their siblings (`convnext_v2.py`,
  `mobilenet_v1/v3/v4.py`), so a `model.py` in either package would name one version and
  hide the rest. What the rename got wrong was its blast radius, and not in the obvious way:
  `2c4a0ca7c` touched 19 files and *did* repoint `models/bert/__init__.py` alongside 10 test
  modules under `tests/test_models/test_bert/`. It grepped one directory. The 13 references
  it missed were 11 outside that directory — 5 convention/README documents and 6 test
  modules, including `tests/test_models/test_package_api_contract.py`, whose waiver dict is
  keyed by *file path* — plus 2 inside it that it repointed only partially. Measured cost:
  tree-wide collection sat at `23809 collected, 3 errors` and
  `tests/test_models/test_package_api_contract.py` at `520 passed, 4 failed` until commit
  `62d0b05cb` repointed all 13, which restored `23828 collected, 0 errors` and
  `524 passed, 0 failed`. So: grep the whole tree for the old path first — documents and
  path-keyed test data, not just `import` lines — and land every hit in the same commit.
  This bullet is **not** a licence to rename. It **reverses** the absolute prohibition that
  stood in its place until 2026-08-24 ("Never rename a module file to `model.py`.
  `bert/bert.py` ... stay where they are; renaming breaks every import") exactly as far as
  `bert/` and no further: `bert/model.py` is correct and its 13 references are now
  consistent, so do not read it as drift and rename it back. The only two other packages
  holding a single non-`__init__` module are `gpt2/gpt2.py` and `time_series/forecast.py`,
  and both are to be LEFT ALONE. `gpt2/gpt2.py` is cited by path in
  `src/dl_techniques/CLAUDE.md:79` and is a live waiver key
  `("models/gpt2/gpt2.py", "create_gpt2")` at `test_package_api_contract.py:3907` — the same
  construct whose `bert` twin one line above produced 4 of the failures counted above, and
  it is checked from both sides (an unwaived offender fails one test, a stale key fails
  another). `time_series/forecast.py` holds the shared `Forecast`/`ForecastMixin` imported
  by four `src/train/` modules and three test modules; it is not a model module at all.
  Recorded in `plan-2026-08-24T120026-64ffd751/D-006`.
- **Never convert docstring style.** This package is measurably mixed; match the file you
  are editing (see `src/dl_techniques/CLAUDE.md` § Code Style).
- **Never re-export the deep-supervision helpers from a `models/` package.**
  `get_model_output_info` and `create_inference_model_from_training_model` live in
  `dl_techniques/utils/deep_supervision.py` and are imported from there. No `models/*`
  module and no `models/*/__init__.py` passes either name through; do not re-add a
  pass-through. They are model-agnostic — they inspect any Keras model's outputs and slice
  a training model down to its primary head — so a model package re-exporting them
  advertised a surface it did not own and gave one function two import paths for no gain.
  `models/vit/__init__.py` is the documented shape, and `models/resnet/__init__.py` now
  carries the same docstring; read either before reaching for a shim.
  This bullet **reverses** the rule that stood in its place until 2026-08-24 ("Never delete
  the deep-supervision re-export shim ... It is a deliberate late import"), so do not read
  the current tree as drift and restore the shims. That rule was factually wrong on both
  counts. It named four files as carrying the shim, but `bias_free_denoisers/bfunet.py` had
  no import at all — only an orphaned comment describing one, left behind when commit
  `e40a13a86` deleted the import (re-verified 2026-08-24: no top-level import past line 900
  in that file; the comment is now gone too). And the "deliberate late import" premise was
  tested at every site this removal touched, where no circular import existed anywhere —
  `utils/deep_supervision.py` imports only `keras` and `utils.logger` and cannot cycle back
  into `models/` — so the tail-of-file placement was stylistic, not structural. Removal
  decided in `plan-2026-08-24T120026-64ffd751/D-001`; this rule inverted in
  `plan-2026-08-24T120026-64ffd751/D-002`.

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

## Checkpoint-affecting changes, 2026-08-19

`plan-2026-08-18T140459-7991552f` changed the saved weight layout of four packages. Recorded here
because `plans/` is gitignored — a note that lives only in a decision log does not ship.

| Package | Change | What still works | What does not |
|---|---|---|---|
| `vq_vae`, `vq_vae_rotation` | `use_ema=True` gains a non-trainable `ema_step` scalar (3 -> 4 weights), and `ema_embeddings` is now zero-initialized | nothing | **both** `load_weights` and `keras.models.load_model()` on a pre-fix artifact raise `ValueError: A total of 1 objects could not be loaded` — measured, not assumed |
| `gemma` | `use_bias=True` now really reaches the attention projections, so it ADDS bias tensors | `use_bias=False` (the default) is unaffected | by-name `load_weights` of a pre-fix `use_bias=True` artifact |
| `fastvlm` | stage-3 default `attention_type` flips `'multi_head'` -> `'group_query'`; same Q/K/V shapes, different registered class and sublayer names | full `load_model()` — `attention_type` is serialized, so an old artifact rebuilds its old topology | a bare `load_weights` into a default-constructed model |
| `tiny_recursive_model` | `attention_args` is now conditional on `attention_type` | `'group_query'` (the default) is byte-identical | a pre-2026-08-17 artifact carrying `'multi_head'` still fails to load — a remapping shim was deliberately REFUSED because it would rebuild a different weight tree than the file contains |

No such checkpoint exists under `results/` on the author's machine (checked read-only). If you hold
one elsewhere, re-train or convert rather than forcing a load.

## Checkpoint- and API-affecting changes, 2026-08-23

`plan-2026-08-23T091307-9a110062` corrected five variant tables that named a published config
while carrying different numbers, after fetching each upstream source. Recorded here for the same
reason as the block above: `plans/` is gitignored and does not ship.

| Package | Variant | Change | Consequence |
|---|---|---|---|
| `hierarchical_reasoning_model` | `small` | `h_layers`/`l_layers` 6 -> 4, `l_cycles` 3 -> 2, `halt_max_steps` 8 -> 16 (`sapientinc/HRM` `config/arch/hrm_v1.yaml`) | different weight tree; a pre-fix `"small"` checkpoint will not load |
| `sd3_mmdit` | `full` + dataclass default | `sample_size` 64 -> 128 (official `SD3Transformer2DModel`) | no weight-shape change; the positional grid scale doubles |
| `ntm` | `base` | `memory_dim` 32 -> 20 (Graves et al. 2014, Tables 1-2 use 128x20 in every row) | memory matrix reshapes; a pre-fix `"base"` checkpoint will not load |
| `pft_sr` | `light`, `base` | corrected to the paper's own two released configs (`CVL-UESTC/PFT-SR`), incl. `window_size` 8 -> 32; `large` **renamed** `repo_medium` | pre-fix `light`/`base` checkpoints will not load; `create_pft_sr(..., variant="large")` now raises |
| `relgt` | `base` | `embedding_dim` 128 -> 512, `num_global_centroids` 32 -> 4096, `num_transformer_blocks` 2 -> 1 (`snap-stanford/relgt` argparse defaults); `large` **renamed** `repo_medium`; `size_configs` hoisted to `RELGT.MODEL_VARIANTS` | pre-fix `base` checkpoints will not load; `create_relgt_model(..., model_size="large")` now raises |

No pretrained weights are distributed for any of these packages, and no checkpoint for them exists
under `results/` (checked read-only). Every corrected row is pinned, with its source URL, by
`tests/test_variant_tables_match_upstream_references.py`; rows with no upstream counterpart are
deliberately NOT pinned there, and both renames exist because a repo-invented row that had been
called "large" ended up *smaller* than the corrected `base` beside it.

**D-002 ruling (2026-08-19): CARRY.** The 25 packages with no consumer outside their own tests —
`cbam detr distilbert fastvlm fftnet fractalnet gemma latent_gmm_registration mamba memory_bank
mini_vec2vec mothnet nano_vlm_world_model pft_sr pw_fnet qwen relgt scunet shgcn som squeezenet
swin_transformer vit_hmlp vit_siglip vq_vae` — are kept. They all have tests; they are library
surface without a trainer, not dead code. This closes a question that had been re-asked across
three planning rounds.

## Verification conventions

The house shape above makes a model package *look* right. These make it *be* right; the full
rationale for each is in `research/2026_keras_custom_models_instructions_v2.md`.

- **Every constructor knob is pinned by a test that varies it and asserts a measured difference in
  weights or outputs.** `assert model.d_state == d_state` proves the constructor stored the argument
  and nothing else. Use `tests/test_models/knob_sensitivity_oracle.py` and pick the instrument by
  knob class — a **structural** knob (depth/heads/filters) must be pinned on the weight-SHAPE
  signature, because different shapes consume different RNG draws and an output-difference
  assertion is satisfied by random-init luck alone.
- **Every smoke test ships with the meta-test that proves its contract can reject**
  (`tests/test_models/smoke_contract_oracle.py`). Never wrap build+forward in a blanket
  `except Exception: pytest.xfail(...)` — a total build break then reports green.
- **Serialization round-trips compare weight VALUES at `atol=0.0`, sampled BEFORE the loaded
  model's first call**, and pass `training=False` explicitly. A shape-only round trip is satisfied
  by a model that restored zero weights; after a forward pass, fresh random weights fill the gap
  and the count matches either way.
- **`build()` materializes exactly the tree `call()` runs** — no more, no less. Overriding `build()`
  is not itself the hazard; failing to materialize is. Pin it with an explicit-vs-lazy weight-path
  parity test PLUS a direct no-sub-layer layout assertion for each `None`/`False` config, since
  parity alone is blind to over-building.
- **Never `.assign()` a constant table inside `build()`** — Keras discards it whenever the layer is
  first reached from a parent's `call()`. Compute it inside an `add_weight(initializer=<callable>)`,
  and test it by building through a parent, never by calling `.build()` directly.
- **Causal models carry a three-armed future-leak probe**: perturb token `t`, assert positions `< t`
  are bit-identical (exactly `0.0`), assert positions `>= t` still move, and add an all-attend
  negative control. `test_attention_mask_functionality` asserting only that masked ≠ unmasked is
  satisfied by any mask. Pass the causal mask at **rank 3** — a rank-2 mask is silently reinterpreted
  as a padding mask by `GroupedQueryAttention`.
- **Pooling strategy and attention causality are one decision.** Assert the pooled representation
  depends on more than one input token.
- **Every forward test asserts `ops.all(ops.isfinite(y))`**, never just `y.shape`. An all-NaN output
  of the correct shape has shipped green more than once.
- **Every guard is proven RED by an injection in the committed record**, and every "nothing changed"
  assertion has a "something changed" twin. A dead-component probe catching a vacuous guard is the
  dominant outcome of writing a new test here, not an edge case.
- **No new custom `train_step`** — stock `fit()` plus extra signals through `tf.data`.

## Testing

Tests in `tests/test_models/` with one subdirectory per model — 82 test directories
as of 2026-08-19 (`ls -d tests/test_models/*/ | wc -l`), more than the 73 model
packages because several architectures get more than one suite. One exception: `lewm` is tested by the loose
`tests/test_models/test_lewm.py`, so a directory-to-directory comparison will
wrongly report it as untested. Test pattern:
- Class-based organization: `class TestModelName`
- Tests cover: serialization, initialization, forward pass, gradient flow, training mode, variants, edge cases
- Pytest fixtures provide model configs
