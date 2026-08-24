# Models Package

Complete model architectures, grouped into **11 family directories**. Each *leaf* directory
is a self-contained model implementation; the family directory above it is a grouping, not a
namespace you import from.

> **ALL model work MUST follow `research/2026_keras_custom_models_instructions_v2.md`.** Read it before creating a new model directory *or authoring any new layer inside a model* — it is the canonical guide for Keras 3 custom authoring in this repo (serialization, `build`, `get_config`, factories, tests). This is non-negotiable for every model in this package, new or existing.

## Layout

The flat `models/<pkg>/` era ended on 2026-08-24 (commits `d0b599ff2`, `452d663d2`). The move
was a pure `git mv`: no model's *content* changed, so every defect and every rule below
survived the move — only the paths did not. Everything in this document quoting a count was
**re-derived against the live tree on 2026-08-25**, not carried over; where the old number is
now false, the old number is named so the correction is not silently re-reverted.

A **leaf package** is a directory with an `__init__.py` and no `__init__.py`-bearing child.
That definition, not "direct child of `models/`", is what the counts below use — two families
nest one level further.

```bash
# 95 __init__.py-bearing directories under models/, of which 79 are leaves and 16 are containers
find src/dl_techniques/models -name '__init__.py' -not -path '*__pycache__*' | sed 's|/__init__.py$||' | sort
```

| | Count | Members |
|---|---|---|
| Families | **11** | `common` `general_purpose` `graph` `language` `memory` `neural_computer` `point_cloud` `tabular` `time_series` `vision` `vision_language` |
| Subfamilies (a second nesting level) | **4** | `vision/image_restoration` `vision/keypoints` `vision/super_resolution` `vision_language/sam` |
| Leaf model packages | **79** | see the catalogue below |

Leaves per family — **these are leaf counts, and for `vision` and `vision_language` they are
NOT the direct-child count**; a `ls -d` of the family directory undercounts both:

| Family | Leaves | Note |
|---|---|---|
| `vision` | **35** | 30 direct + `image_restoration/{darkir,pw_fnet,scunet}` + `keypoints/superpoint` + `super_resolution/pft_sr` |
| `language` | 16 | |
| `vision_language` | **9** | 6 direct + `sam/{sam1,sam2,sam3}` |
| `time_series` | 7 | |
| `general_purpose` | 3 | |
| `graph` | 3 | |
| `neural_computer` | 2 | |
| `common` | 1 | |
| `memory` | 1 | |
| `point_cloud` | 1 | |
| `tabular` | 1 | |
| **Sum** | **79** | |

`time_series/` is the only container that also owns a module of its own —
`time_series/forecast.py`, holding the shared `Forecast` / `ForecastMixin`. It is not a model
module; see the rename bullet under "Things you must NOT do".

## Model Categories

The families below are the directories that exist on disk. They replace the ad-hoc headings
this catalogue used until 2026-08-24 ("Specialized Architectures", "Sequence / State Space",
"Graph / Geometric"), which mapped to nothing in the tree and had become a second, drifting
taxonomy. Where the restructure made a classification call that the old headings did not, it
is flagged inline — the directory is now the authority, and a package's family is a filing
decision, not a claim about the architecture.

### `vision/` (35)
- `vision/accunet/` — AccuNet
- `vision/beit/` — BEiT (masked image modeling over discrete visual tokens + classifier; trainers in `src/train/beit/`)
- `vision/bias_free_denoisers/` — Bias-free denoiser models
- `vision/capsnet/` — Capsule Networks
- `vision/cbam/` — CBAM attention model
- `vision/cliffordnet/` — Clifford algebra networks
- `vision/convnext/` — ConvNeXt
- `vision/convunext/` — ConvUNeXt (U-Net + ConvNeXt)
- `vision/coshnet/` — CoshNet
- `vision/depth_anything/` — Depth estimation
- `vision/detr/` — DEtection TRansformer
- `vision/dino/` — DINO self-supervised
- `vision/energy_transformer/` — Energy Transformer (masked image completion + classifier)
- `vision/fastvit/` — FastViT MCi image backbone (the assembled tower over `layers/fastvit/`; the image branch of MobileCLIP2, also usable standalone — architecture only, no pretrained weights)
- `vision/fractalnet/` — FractalNet
- `vision/image_restoration/darkir/` — DarkIR image restoration
- `vision/image_restoration/pw_fnet/` — a 2-level U-Net with FFT token mixing and multi-scale supervision, for image restoration. **Name misattributes** — see below
- `vision/image_restoration/scunet/` — SCUNet denoiser
- `vision/keypoints/superpoint/` — SuperPoint keypoint detector + descriptor
- `vision/lewm/` — Latent-energy world model
- `vision/masked_autoencoder/` — MAE
- `vision/mobilenet/` — MobileNet variants (V1, V2, V3, V4)
- `vision/resnet/` — ResNet architectures
- `vision/squeezenet/` — SqueezeNet
- `vision/super_resolution/pft_sr/` — Super-resolution
- `vision/swin_transformer/` — Swin Transformer
- `vision/thera/` — THERA aliasing-free arbitrary-scale super-resolution
- `vision/vae/` — Variational Autoencoder (ResNet encoder/decoder; `sampling_type` ∈ {`gaussian`, `hypersphere`, `vmf`} — `vmf` is a true von Mises-Fisher Spherical VAE with the closed-form vMF→uniform-sphere KL; see the package README §16–17)
- `vision/video_jepa/` — Video JEPA (joint embedding predictive)
- `vision/vit/` — Vision Transformer
- `vision/vit_hmlp/` — ViT with hierarchical MLP
- `vision/vit_siglip/` — ViT with a two-stage conv patch-embedding stem. **Name misattributes** — see below
- `vision/vq_vae/` — VQ-VAE
- `vision/vq_vae_rotation/` — VQ-VAE with rotation-based codebook updates
- `vision/yolo12/` — YOLOv12 detection

### `language/` (16)
- `language/bert/` — BERT
- `language/byte_latent_transformer/` — Byte Latent Transformer (BLT)
- `language/distilbert/` — DistilBERT
- `language/fftnet/` — FFTNet
- `language/fnet/` — FNet (Fourier)
- `language/gemma/` — Gemma LLM
- `language/gpt2/` — GPT-2 architecture
- `language/hierarchical_reasoning_model/` — HRM
- `language/mamba/` — Mamba (SSM)
- `language/masked_language_model/` — MLM training
- `language/mini_vec2vec/` — Mini Vec2Vec
- `language/modern_bert/` — ModernBERT
- `language/qwen/` — Qwen LLM
- `language/tiny_recursive_model/` — Tiny recursive model
- `language/tree_transformer/` — Tree Transformer
- `language/wave_field/` — Wave-field LLM

> Six of these sat under the old "Specialized Architectures" / "Sequence / State Space"
> headings rather than under NLP: `fnet`, `fftnet`, `mamba`, `hierarchical_reasoning_model`,
> `tiny_recursive_model`, `tree_transformer`, plus `mini_vec2vec`. They are token-sequence
> models and the restructure filed them by input modality. That is a filing decision, not a
> claim that `mamba` is a language model and nothing else.

### `vision_language/` (9)
- `vision_language/clip/` — CLIP
- `vision_language/fastvlm/` — vision-only hybrid backbone (MobileOne stem + RepMixer + attention stages). **Name misattributes** — see below
- `vision_language/ideogram4/` — Ideogram4 text-to-image flow-matching DiT
- `vision_language/mobile_clip/` — MobileCLIP: **both generations in one package**.
  `mobile_clip_v1.py` is deliberately non-faithful on the image side
  (`keras.applications` substitutes, its own D-001); `mobile_clip_v2.py` is the faithful
  MobileCLIP2 (architecture only, no pretrained weights, no accuracy claim). They share
  `components.py`'s text tower AND one structure: a nested `MODEL_VARIANTS` CLASS attribute
  (`embed_dim` + `image_config` + `text_config`) and a constructor over those two sub-dicts.
  Neither deprecates the other — see the package README §17
- `vision_language/nano_vlm/` — NanoVLM
- `vision_language/sam/sam1/` — Segment Anything Model v1
- `vision_language/sam/sam2/` — Segment Anything Model v2 (video / memory bank)
- `vision_language/sam/sam3/` — Segment Anything Model v3 (text-promptable)
- `vision_language/sd3_mmdit/` — SD3 MMDiT dual-stream text-to-image diffusion transformer

> `sd3_mmdit` and `ideogram4` were listed under Vision until the restructure; both consume a
> text stream, so `vision_language/` is the better fit. `fastvlm` is the opposite case and is
> the sharpest one in the tree — the directory now asserts exactly the thing the code does not
> do (see the table below).

### `time_series/` (7)
- `time_series/adaptive_ema/` — Adaptive EMA model
- `time_series/deepar/` — DeepAR probabilistic forecasting
- `time_series/mdn/` — Mixture Density Networks
- `time_series/nbeats/` — N-BEATS (plus the exogenous `nbeatsx` variant)
- `time_series/prism/` — PRISM forecasting
- `time_series/tirex/` — TiReX time series
- `time_series/xlstm/` — xLSTM

### `general_purpose/` (3)
- `general_purpose/kan/` — Kolmogorov-Arnold Networks
- `general_purpose/mothnet/` — MothNet (bio-inspired)
- `general_purpose/power_mlp/` — Power MLP

### `graph/` (3)
- `graph/graph_energy_transformer/` — Graph Energy Transformer (node anomaly + graph classification)
- `graph/relgt/` — Relational Graph Transformer
- `graph/shgcn/` — Simplified Hyperbolic GCN

### `neural_computer/` (2)
- `neural_computer/nam/` — Neural Arithmetic **Module**: tree-transformer parse + NTM memory + TRM halting stack that evaluates arithmetic expressions. **Name misattributes** — see below
- `neural_computer/ntm/` — Neural Turing Machine

### `common/` (1)
- `common/power_sampling/` — Inference-time power sampling for any causal LM/VLM. Model-agnostic machinery, which is why it is not filed under `language/`

### `memory/` (1)
- `memory/som/` — Self-Organizing Maps (filed under the old "Graph / Geometric" heading; its defining structure is a learned codebook topology, not a graph input)

### `point_cloud/` (1)
- `point_cloud/latent_gmm_registration/` — Latent GMM registration

### `tabular/` (1)
- `tabular/tabm/` — TabM for tabular data

### Deleted by the restructure

Two packages this catalogue listed until 2026-08-24 no longer exist and must not be
re-added to any document as if they did: `memory_bank/` (memory-bank language model
components) and `nano_vlm_world_model/` (NanoVLM world model). Their tests were removed in
`377ab9565` and two `train_step` freeze keys naming them were re-derived out of
`tests/test_models/test_package_api_contract.py` in the same commit.

### Names that misattribute

Four packages are named for something they are not. Each was corrected in place after measurement;
the entry is kept so the correction is not silently re-reverted.

| Package | The name claims | What the code is | Evidence |
|---|---|---|---|
| `vision_language/fastvlm/` | a vision-**language** model — and the 2026-08-24 restructure filed it under `vision_language/`, re-asserting at the directory level exactly the claim this row refutes | **vision-only**: no text tower, no tokenizer, no language input | its own `README.md:1` — "A Fast Hybrid Vision Model" |
| `vision/vit_siglip/` | SigLIP | a ViT with a two-stage conv patch-embedding stem. SigLIP is a sigmoid contrastive **loss**, and its own tower has a single-conv stem, no CLS token and a MAP head — none of which this package shares. There is no text tower and no loss here | `vision/vit_siglip/model.py` module docstring |
| `neural_computer/nam/` | (read as) Neural **Additive** Model | Neural Arithmetic **Module** — there is no per-feature additive model anywhere in the package | `neural_computer/nam/model.py`'s module docstring has always said Module; this catalogue said "Neural additive model" until 2026-08-15 |
| `vision/image_restoration/pw_fnet/` | "Pyramid **Wavelet**-Fourier Network" (the package's own name, `pw_fnet/model.py:2`), and it was listed here as "Patchwise FNet" until 2026-08-19 | neither patchwise nor FNet: a 2-level U-Net with FFT token mixing and multi-scale supervision, not FNet's token+feature-axis Fourier mixing. **No wavelet transform** — the only spectral ops are `FFTLayer` / `IFFTLayer` | `grep -rn -i wavelet src/dl_techniques/models/vision/image_restoration/pw_fnet/` returns only the name itself. Two of the three words in the expanded name are unearned |

> A class or package sharing a name with a published architecture is not necessarily that
> architecture. Check the composition rule, not the name. The same now goes for a package
> sharing a name with the **family directory** it sits in.

## Conventions

### Import from the leaf package, never from a family

- `src/dl_techniques/models/__init__.py` (the root) is **0 bytes** and stays that way — never
  import from `dl_techniques.models` itself.
- **The 11 family `__init__.py` files carry a docstring and NOTHING ELSE — no imports, no
  `__all__`.** This is a decision, not an unfinished job. Re-exporting a family would make
  `import dl_techniques.models.vision` eagerly pull all **35** vision packages behind one
  statement, on a Keras/TensorFlow tree where that is both a real import cost and a real
  circular-import surface. Recorded in
  `plan-2026-08-24T205033-8fd4f20d/D-002`. Do not "finish the job" by adding re-exports.
- **`time_series/__init__.py` is the one exception** — 43 lines, a docstring plus explicit
  re-imports from all 7 children plus a curated `__all__`. It predates the family layout, it
  has 7 children rather than 35, and its consumers already rely on it. Leave it exactly as it
  is. It is also why the three empty leaf inits below are not broken.
- This is a deliberate divergence from `layers/`, where 13 of 20 subpackages are curated. The
  discriminator is child count and import weight. Do not file it as an inconsistency to fix.
- The 4 **subfamily** inits are split: `vision/image_restoration/__init__.py` and
  `vision_language/sam/__init__.py` carry substantive docstrings that predate the restructure
  (the SAM one documents a measured name-shadowing failure — read it before adding a
  re-export there), while `vision/keypoints/__init__.py` and
  `vision/super_resolution/__init__.py` are still 0 bytes.

### Per-leaf `__init__.py` census — re-derived 2026-08-25

The old text here read "**73 of 73** have a non-empty `__init__.py`; **72 of 73** declare
`__all__`; **69 of 73** bind a `create_*`". Against the 79-leaf tree those are all false, and
the first one was hiding an exception:

| Property | Live value | Command |
|---|---|---|
| Leaf packages | **79** | the `find` at the top of this file, filtered to leaves |
| Non-empty `__init__.py` | **76 / 79** | `for d in $(find src/dl_techniques/models -name '__init__.py' -not -path '*__pycache__*'); do [ -s "$d" ] && echo "$d"; done` (then filter to leaves) |
| Declares `__all__` | **76 / 79** | `grep -l "^__all__" <leaf>/__init__.py` |
| Binds at least one `create_*` | **72 / 79** | AST walk over each leaf `__init__.py`, counting `Import`/`ImportFrom`/`def`/`class`/`Assign` bindings whose name starts with `create_` |
| Has a `README.md` | **79 / 79** | `test -e <leaf>/README.md` |

**The 3 empty inits are `time_series/deepar`, `time_series/prism`, `time_series/tirex`** — the
same three, in the same family, and they are the same three that lack `__all__`. They are not
broken: the curated `time_series/__init__.py` re-exports their public names, so
`from dl_techniques.models.time_series import DeepAR, create_deepar` works. This is a
pre-existing inconsistency that predates the restructure, and it is exactly what the old
"73 of 73" claim concealed — a total with no exception named is a total that has not been
checked.

The 7 leaves binding no `create_*` are `common/power_sampling`, `time_series/deepar`,
`time_series/mdn`, `time_series/prism`, `time_series/tirex`, `vision_language/sam/sam1` and
`vision_language/sam/sam3`.

**Re-derive the factory count with an AST walk, not a grep.** A plain `grep -l create_` over
the same inits over-counts: a file that mentions a factory in a docstring or a comment without
binding it is a match for grep and not for the AST. The gap between those two numbers is the
whole reason to prefer the AST. The census is guarded by
`tests/test_models/test_package_api_contract.py`, so prefer running that over re-deriving by
hand.

### Docstring style — measurably mixed, re-derived 2026-08-25

```bash
grep -rlE "^[[:space:]]*Args:[[:space:]]*$" src/dl_techniques/models --include=*.py | wc -l   # 90 = Google-only + both
grep -rl ":param " src/dl_techniques/models --include=*.py | wc -l                            # 83 = Sphinx-only + both
find src/dl_techniques/models -name '*.py' -not -path '*__pycache__*' | wc -l                 # 270 total
```

Over those 270 files: **Google-only 82, Sphinx-only 75, both 8, neither 105**. There is no
package-wide rule in either direction. **Match the file you are editing; never convert one
wholesale.** These four numbers are perishable — re-run the two greps rather than quoting them
from memory, and note that a different instrument gives a different answer (an unanchored
`Args:` grep returns 91, not 90, because it also matches the string inside prose).

### Everything else

- Each leaf directory typically contains: model definition module(s), block/layer definitions
  specific to that architecture, an optional `train.py` or training utilities, and a `README.md`.
- All models follow Keras 3 patterns with full `get_config()` serialization.
- Models use config dicts (`Dict[str, Any]`) for construction parameters.
- Factory patterns are common for creating model variants.

## Tests are FLAT and deliberately do not mirror this layout

`tests/test_models/` is **not** being restructured to match the 11 families, and this is a
ruling, not a backlog item. Recorded in `plan-2026-08-24T205033-8fd4f20d/D-001`.

- Measured today: `ls -d tests/test_models/*/ | wc -l` → **82** test directories, all one
  level deep, plus ~30 loose modules.
- `grep -rn "from \.\." --include='*.py' tests/test_models | wc -l` → **215** relative imports
  (it was 219 before `377ab9565` deleted 21 orphaned test files). They reach shared oracle
  modules that live at `tests/test_models/*.py` — `gradient_flow_oracle`,
  `knob_sensitivity_oracle`, `smoke_contract_oracle`, `precision_arm_oracle`,
  `test_sam/dead_component_oracle`.
- Nesting `test_beit/` under a `test_vision/` changes what `..` resolves to and forces every
  one of those to `...`, plus several absolute `tests.test_models.test_X.Y` forms — in a suite
  that takes ~1.5h and cannot be verified in a single process (it OOMs at ~36.6 GB RSS), for
  zero behavioural gain.

**The convention that matters is unchanged: leaf package `x` is tested by
`tests/test_<x>/`.** `test_beit/` still tests `vision/beit`. Only the phrasing "the test tree
mirrors the source tree" is falsified, and `REPO_MAP.md` § Tests carries the same correction.
Do not "fix" the divergence.

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

> Nothing in this section moved on 2026-08-24: `layers/` was untouched by the restructure and
> all six factory paths above resolve as written.

## House Model Module Shape

> **Reference implementation: `models/vision/resnet/model.py`.** Read the spec below rather than
> copying the file blind — the shape spread through this package by copy-paste, and that is
> exactly how its defects (placeholder weight URLs, mutable default args, a `logger.info`
> inside `call()`) reached `vision/convnext/`, `vision/mobilenet/` and others.

This shape applies to a package that implements **one architecture with named variants**. It
is a target, not a universal law: see "When the shape does not apply" at the end.

### Axis 1 — module skeleton

**The module docstring is substantive prose, not a template.** `models/vision/resnet/model.py`
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
   `pretrained=True` raises `NotImplementedError` rather than warning and returning a
   random model — no public pretrained weights are distributed for any package here, so
   see Axis 3 below).
5. **A `References:` section** listing papers as `- Author et al., YEAR. Title. (url)`.
   Include the papers the design actually draws on, not only the headline one.

What this replaces: terse `Model Variants:` / `Usage Examples:` boilerplate blocks that
restate the `MODEL_VARIANTS` dict and the factory signature. Those are already in the
code directly below; the docstring's job is the reasoning that is *not* in the code.

Read `models/vision/resnet/model.py`'s docstring before writing one. Length follows the
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
scope in `vision/beit/model.py` (`SCALE_CONFIGS` and `MODEL_VARIANTS`):

- `SCALE_CONFIGS` is the **architecture table** — `'tiny' -> {hidden_size: 192,
  num_layers: 12, num_heads: 3, ...}`.
- `MODEL_VARIANTS` is the **public-name registry** — `'beit_tiny' -> {scale: 'tiny'}`,
  one row per name a caller may pass, resolving to a scale.

`vision/beit`, `vision/vit` and `vision/energy_transformer` carry both, deliberately, and a
`_resolve_scale`-style helper accepts either spelling. (`vision/fastvit` was named here as a
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
nine packages that carried it (`vision/resnet`, `language/bert`, `vision/convnext`'s
`convnext_v1` and `convnext_v2`, `language/fnet`, `language/modern_bert`,
`vision/cliffordnet`, `vision/bias_free_denoisers/bfunet`, `general_purpose/kan`). Do not
reintroduce it.

`_download_weights(...)` **raises `NotImplementedError` naming the variant and showing the
local-path alternative.** No public checkpoints are distributed with `dl_techniques` for any
architecture in this package.

**Never** write a placeholder URL table plus a `try/except` in `from_variant` that logs a
warning and continues with random initialization. That combination means
`pretrained=True` silently returns an untrained model — a caller asking for pretrained
weights gets random ones and no error. This contract is pinned by tests in
`test_bert/`, `test_gpt2/`, `test_wave_field/`, `test_tree_transformer/`, `test_vit/`,
`test_cliffordnet/`, `test_xlstm/` and `test_capsnet/test_model_v2.py` — all directly under
`tests/test_models/`, which is flat (see above).

### Axis 4 — factory and exports

A module-level `create_<name>(variant="<default>", ...)` that delegates to `from_variant` —
no logic of its own. The package `__init__.py` exports the class and the factory with an
`__all__`; `vision/accunet/__init__.py` is the exemplar. The `__init__.py` that carries this
is the **leaf** one, never the family one.

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
  these anchors — never target a file by comment density. This applies to the two HTML-comment
  anchors in this very file: the 2026-08-24 restructure truncated this document from 545 lines
  to 5 and took both with it, orphaning two `plans/ANCHORS.md` entries until they were
  restored on 2026-08-25.
<!-- DECISION plan-2026-08-24T120026-64ffd751/D-006: the bullet below INVERTS the
     absolute "Never rename a module file to `model.py`" prohibition that stood here until
     2026-08-24. Do NOT rename `bert/model.py` back to `bert/bert.py` — its 13 referencing
     sites were repointed in `62d0b05cb` and are consistent. Do NOT read this bullet as
     permission to rename `gpt2/gpt2.py` or `time_series/forecast.py`: both are cited by
     path elsewhere in the tree, and `gpt2/gpt2.py` is a live waiver key in
     `tests/test_models/test_package_api_contract.py`.
     This anchor was placed retroactively per DECISION
     plan-2026-08-24T120026-64ffd751/D-008, which corrects the false claim that a Markdown
     convention document cannot carry one.
     Paths in the bullet below were re-prefixed for the 11-family layout on 2026-08-25
     (`bert/` -> `language/bert/`, `gpt2/` -> `language/gpt2/`, `convnext/` ->
     `vision/convnext/`, `mobilenet/` -> `vision/mobilenet/`). The ruling and the reasoning
     are unchanged. -->
- **A module rename must carry every referencing site in the same commit.** `language/bert/`
  now holds `model.py`, renamed from `bert/bert.py` on 2026-08-24 by commit `2c4a0ca7c`
  ("[models] cleaning up models documentation and structure") — not by the merge
  `4ed922781` that carried it in. `vision/convnext/convnext_v1.py` and
  `vision/mobilenet/mobilenet_v2.py`
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
  `language/bert/` and no further: `language/bert/model.py` is correct and its references are
  now consistent, so do not read it as drift and rename it back. The only two other packages
  holding a single non-`__init__` module are `language/gpt2/gpt2.py` and
  `time_series/forecast.py`, and both are to be LEFT ALONE. `language/gpt2/gpt2.py` is cited
  by path in `src/dl_techniques/CLAUDE.md:79` and is a live waiver key
  `("models/language/gpt2/gpt2.py", "create_gpt2")` at `test_package_api_contract.py:3905` —
  the same construct whose `bert` twin one line above produced 4 of the failures counted
  above, and it is checked from both sides (an unwaived offender fails one test, a stale key
  fails another). `time_series/forecast.py` holds the shared `Forecast`/`ForecastMixin`
  imported by four `src/train/` modules and three test modules; it is not a model module at
  all, and it is the only module a *container* directory owns.
  Recorded in `plan-2026-08-24T120026-64ffd751/D-006`.
  **The 2026-08-24 family restructure is the same lesson at scale**: it was a pure `git mv`
  that carried no referencing site, leaving 1636 stale dotted paths and 653 stale slash paths
  across 750 files, 27 of 95 packages unimportable, and 410 pytest collection errors — all
  repaired by `plan-2026-08-24T205033-8fd4f20d`. A rename's blast radius is the whole tree,
  every time.
- **Never convert docstring style.** This package is measurably mixed; match the file you
  are editing (see `src/dl_techniques/CLAUDE.md` § Code Style, and the census above).
<!-- DECISION plan-2026-08-24T120026-64ffd751/D-002: the bullet below INVERTS the
     "Never delete the deep-supervision re-export shim" rule that stood here until
     2026-08-24. Do NOT restore the pass-throughs at the tail of `models/*` modules or in
     `models/*/__init__.py`, and do NOT read the current tree as drift. This is NOT a
     licence to sweep cross-package imports generally: a registrar import (`# noqa: F401`,
     bound for its Keras registration side effect) is load-bearing and is carved out below.
     Removal decided in DECISION plan-2026-08-24T120026-64ffd751/D-001; this anchor placed
     retroactively per DECISION plan-2026-08-24T120026-64ffd751/D-008.
     Paths in the bullet below were re-prefixed for the 11-family layout on 2026-08-25
     (`models/vit/` -> `models/vision/vit/`, `models/resnet/` -> `models/vision/resnet/`,
     `models/bias_free_denoisers/` -> `models/vision/bias_free_denoisers/`). The ruling and
     the reasoning are unchanged. -->
- **Never re-export the deep-supervision helpers from a `models/` package.**
  `get_model_output_info` and `create_inference_model_from_training_model` live in
  `dl_techniques/utils/deep_supervision.py` and are imported from there. No `models/*`
  module and no `models/*/__init__.py` passes either name through; do not re-add a
  pass-through. They are model-agnostic — they inspect any Keras model's outputs and slice
  a training model down to its primary head — so a model package re-exporting them
  advertised a surface it did not own and gave one function two import paths for no gain.
  `models/vision/vit/__init__.py` is the documented shape, and
  `models/vision/resnet/__init__.py` now carries the same docstring; read either before
  reaching for a shim.
  What this forbids is a **surface re-export**: a name passed through so callers gain a
  second import path to something the package does not own. It does not touch a
  **registrar import** — a cross-package import bound for its
  `@keras.saving.register_keras_serializable` side effect, so `keras.models.load_model` can
  resolve every custom class a saved graph names. Those carry `# noqa: F401`, are
  load-bearing, and must not be swept.
  `models/vision/bias_free_denoisers/bfconvunext.py` (its `# local imports` block, lines
  ~44-87) is the reference case (registrar contract H-4, anchored under
  `plan-2026-08-19T163559-499b6f0e/D-080`): the last sweep that deleted its twelve "unused"
  imports failed 7 tests at
  `tests/test_models/test_bias_free_denoisers/test_bfconvunext_wrappers.py::TestRegistrarContract`.
  The test to apply is whether a consumer needs the *name* at runtime or only the import's
  side effect — if deleting it breaks `load_model`, it is a registrar, not a shim.
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
- **Never re-export a family.** See the Conventions section: the 11 family `__init__.py`
  files carry a docstring only, per `plan-2026-08-24T205033-8fd4f20d/D-002`.

### When the shape does not apply

- **No genuine named variants.** Do not invent a `MODEL_VARIANTS` table to satisfy the
  template. Apply axes 1, 4 and 5 only, and say why in the package `README.md`.
- **Functional builders** (`vision/bias_free_denoisers/`, `vision/convunext/`,
  `vision/image_restoration/darkir/`) return `keras.Model(inputs, outputs)` and have no
  subclass. Keep them functional — converting them would break existing checkpoints, and the
  `bias_free_denoisers` ones are actively used by `src/train/bfunet/` and
  `src/applications/`. Axes 1, 4 and 5 still apply.
  (`vision/detr/` was listed here in the first draft of this section and does **not** belong:
  `vision/detr/model.py` defines `class DETR(models.Model)`. Verify with
  `grep -n "^class .*(.*Model)" <pkg>/*.py` before classifying a package — the census
  that produced the original list was grep-based and wrong about several packages.)
- **Multi-model families and nested packages** (`vision_language/sam/`, `time_series/`,
  `vision/dino/`, `vision_language/ideogram4/`, `vision_language/sd3_mmdit/`) apply the shape
  per *inner architecture*, not per directory. Note that `vision_language/sam/` is now a
  **subfamily container** whose three generations are leaf packages in their own right.

## Consumer-less packages: CARRY (ruling, 2026-08-19; list re-derived 2026-08-25)

**23** leaf packages have **no consumer outside their own package and tests** — no trainer
under `src/train/`, no application under `src/applications/`, no import from another `models/`
package:

`general_purpose/mothnet` `graph/relgt` `graph/shgcn` `language/distilbert`
`language/fftnet` `language/gemma` `language/mamba` `language/mini_vec2vec` `language/qwen`
`memory/som` `point_cloud/latent_gmm_registration` `vision/cbam` `vision/detr`
`vision/fractalnet` `vision/image_restoration/pw_fnet` `vision/image_restoration/scunet`
`vision/squeezenet` `vision/super_resolution/pft_sr` `vision/swin_transformer`
`vision/vit_hmlp` `vision/vit_siglip` `vision/vq_vae` `vision_language/fastvlm`

The old text here said **25**. The set did not change composition: 23 is exactly those 25
minus `memory_bank` and `nano_vlm_world_model`, both deleted by the restructure. The ruling
therefore still describes the same population and holds structurally.

**They are CARRIED. Do not propose deleting them as dead code.** All 23 have test
suites, and a tested package with no trainer is *library surface*, not dead code: this
repo distributes reusable layers and models, and "has a trainer" was never the shipping
criterion. Deletion is not reversible for a downstream user who has pinned an import.

This question was re-opened by two separate review rounds (2026-08-17, 2026-08-18) and
ruled the same way both times. The ruling lives here rather than in a planning document
because `plans/` is gitignored and does not ship. Re-derive the list before acting on
it — it moves as trainers are added or packages are merged (e.g. `convunext`,
`bfconvunext` and `bfcnn` all left this list when their trainers were consolidated).

**Re-derive it with a word-boundary match, not a substring match.** A plain
`grep -rl "dl_techniques.models.vision.vq_vae"` reports `vision/vq_vae` as having three
consumers; all three are `vision/vq_vae_rotation` references that the prefix swallows, and
the same trap sits between `nam`/`nano_vlm`, `vit`/`vit_siglip` and `clip`/`mobile_clip`.
Also exclude the 15 container `__init__.py` files from the consumer set: since 2026-08-25
each family docstring *names* its leaf packages, so a naive grep reports every leaf as having
a consumer — that is a mention, not an import. Both mistakes were made while re-deriving this
list and both produced a plausible, wrong answer (21, then 79).

## Checkpoint-affecting changes, 2026-08-19

`plan-2026-08-18T140459-7991552f` changed the saved weight layout of four packages. Recorded here
because `plans/` is gitignored — a note that lives only in a decision log does not ship.

| Package | Change | What still works | What does not |
|---|---|---|---|
| `vision/vq_vae`, `vision/vq_vae_rotation` | `use_ema=True` gains a non-trainable `ema_step` scalar (3 -> 4 weights), and `ema_embeddings` is now zero-initialized | nothing | **both** `load_weights` and `keras.models.load_model()` on a pre-fix artifact raise `ValueError: A total of 1 objects could not be loaded` — measured, not assumed |
| `language/gemma` | `use_bias=True` now really reaches the attention projections, so it ADDS bias tensors | `use_bias=False` (the default) is unaffected | by-name `load_weights` of a pre-fix `use_bias=True` artifact |
| `vision_language/fastvlm` | stage-3 default `attention_type` flips `'multi_head'` -> `'group_query'`; same Q/K/V shapes, different registered class and sublayer names | full `load_model()` — `attention_type` is serialized, so an old artifact rebuilds its old topology | a bare `load_weights` into a default-constructed model |
| `language/tiny_recursive_model` | `attention_args` is now conditional on `attention_type` | `'group_query'` (the default) is byte-identical | a pre-2026-08-17 artifact carrying `'multi_head'` still fails to load — a remapping shim was deliberately REFUSED because it would rebuild a different weight tree than the file contains |

No such checkpoint exists under `results/` on the author's machine (checked read-only). If you hold
one elsewhere, re-train or convert rather than forcing a load.

## Checkpoint- and API-affecting changes, 2026-08-23

`plan-2026-08-23T091307-9a110062` corrected five variant tables that named a published config
while carrying different numbers, after fetching each upstream source. Recorded here for the same
reason as the block above: `plans/` is gitignored and does not ship.

| Package | Variant | Change | Consequence |
|---|---|---|---|
| `language/hierarchical_reasoning_model` | `small` | `h_layers`/`l_layers` 6 -> 4, `l_cycles` 3 -> 2, `halt_max_steps` 8 -> 16 (`sapientinc/HRM` `config/arch/hrm_v1.yaml`) | different weight tree; a pre-fix `"small"` checkpoint will not load |
| `vision_language/sd3_mmdit` | `full` + dataclass default | `sample_size` 64 -> 128 (official `SD3Transformer2DModel`) | no weight-shape change; the positional grid scale doubles |
| `neural_computer/ntm` | `base` | `memory_dim` 32 -> 20 (Graves et al. 2014, Tables 1-2 use 128x20 in every row) | memory matrix reshapes; a pre-fix `"base"` checkpoint will not load |
| `vision/super_resolution/pft_sr` | `light`, `base` | corrected to the paper's own two released configs (`CVL-UESTC/PFT-SR`), incl. `window_size` 8 -> 32; `large` **renamed** `repo_medium` | pre-fix `light`/`base` checkpoints will not load; `create_pft_sr(..., variant="large")` now raises |
| `graph/relgt` | `base` | `embedding_dim` 128 -> 512, `num_global_centroids` 32 -> 4096, `num_transformer_blocks` 2 -> 1 (`snap-stanford/relgt` argparse defaults); `large` **renamed** `repo_medium`; `size_configs` hoisted to `RELGT.MODEL_VARIANTS` | pre-fix `base` checkpoints will not load; `create_relgt_model(..., model_size="large")` now raises |

No pretrained weights are distributed for any of these packages, and no checkpoint for them exists
under `results/` (checked read-only). Every corrected row is pinned, with its source URL, by
`tests/test_variant_tables_match_upstream_references.py`; rows with no upstream counterpart are
deliberately NOT pinned there, and both renames exist because a repo-invented row that had been
called "large" ended up *smaller* than the corrected `base` beside it.

## Restructure, 2026-08-25

The user moved `models/` from ~78 flat packages into the 11 families above (`d0b599ff2`,
`452d663d2`), as a pure `git mv`. `plan-2026-08-24T205033-8fd4f20d` repaired the fallout:

| What | Before | After |
|---|---|---|
| Packages that fail to import | 27 of 95 | 0 |
| Stale dotted `dl_techniques.models.*` refs (outside `plans/`) | 1636 | 0 |
| Stale slash-form `models/<old>` refs (outside `plans/`) | 653 | 0 |
| `pytest` collection errors | 410 | 0 |
| Family `__init__.py` files with a docstring | 1 of 11 | 11 of 11 |
| This file | 545 lines, then truncated to 5 | rebuilt, every count re-derived |

Two packages were deleted outright (`memory_bank`, `nano_vlm_world_model`) and their 21
orphaned test files removed. 77 untracked empty directories left behind by the move were
pruned first, because PEP 420 namespace packages made every one of them report as importable
and would have reported a false GREEN over dead paths.

**Claims in this file that the restructure falsified** — corrected above, listed here so the
correction is not re-reverted: "73 of 73 non-empty `__init__.py`" (now 76 of 79, with three
named exceptions), "72 of 73 declare `__all__`, `SAM/` the sole exception" (now 76 of 79 and
the exceptions are three `time_series` leaves, not `SAM/`), "69 of 73 bind a `create_*`" (now
72 of 79), the "25 consumer-less packages" list (now 23), the docstring-style census, and
every path in the document.

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
- **No document under `models/` may credit *this port* with measured performance, or use an
  unverifiable maturity adjective.** Nothing here has been trained or benchmarked in this
  repository; crediting the *paper* is fine, crediting the code is not. Enforced by
  `tests/test_docs_make_no_unearned_performance_claims.py`, which reads every `.md` and `.py`
  under `models/` — read its two regexes before writing a README.

## Testing

Tests live in `tests/test_models/`, **flat**, one directory per leaf package —
`ls -d tests/test_models/*/ | wc -l` → **82** as of 2026-08-25. That is more than the 79 leaf
packages because several architectures get more than one suite, and it does not mirror the
family tree by design (see "Tests are FLAT" above). One exception to the directory rule:
`vision/lewm` is tested by the loose `tests/test_models/test_lewm.py`, so a
directory-to-directory comparison will wrongly report it as untested. Test pattern:
- Class-based organization: `class TestModelName`
- Tests cover: serialization, initialization, forward pass, gradient flow, training mode, variants, edge cases
- Pytest fixtures provide model configs
