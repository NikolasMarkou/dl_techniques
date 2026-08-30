# REPO_MAP — a verified router for `dl_techniques`

**What this file is.** A router and a verified skeleton. It tells you *where code of a
given kind lives*, *how the pieces are wired together*, and *which of the repo's many
in-tree docs answers your question*. Every filesystem path printed here resolved when this
file was written, with one declared exception: the stale-doc ledger in Part C names paths
precisely *because* they do not resolve, and each carries an explicit `allow-dead-path`
directive above that table. Every number lives in exactly one table, beside the command
that re-derives it.

**What this file deliberately is NOT.** It is not a replacement for the 20 in-tree
`CLAUDE.md` files — those are the authority on their own subtree, and this map points at
them rather than restating them. Every routing target below was checked to *exist*; none
was audited for its own accuracy, and a sample found two that are already stale (last row
of the ledger). It is not an API reference: it will not tell you what any
individual layer, loss or model does, only which document will. Restating subtree content
here is exactly how the repo's older maps drifted into describing code that no longer
exists, so the omission is the design, not an oversight. Parts B (the navigation spine)
and C (conventions, tests, doc routing, the stale-doc ledger) follow below.

---

## Top-level layout

```
.
├── src/
│   ├── dl_techniques/       # THE LIBRARY — layers, models, losses, everything reusable
│   ├── train/               # production training pipelines, one entry per trainer
│   └── applications/        # deployable end-user apps built on the library
├── tests/                   # pytest suite; mirrors src/dl_techniques/ EXCEPT test_models/ — see Part C
├── research/                # research notes; research/papers/ holds LaTeX sources
├── imgs/                    # committed image assets
├── scripts/                 # standalone maintenance scripts
├── results/                 # LOCAL ONLY — gitignored, training outputs + checkpoints
├── plans/                   # LOCAL ONLY — gitignored, planner state
├── data/                    # LOCAL ONLY — untracked, local datasets
├── CLAUDE.md                # repo-wide agent/contributor instructions
├── README.md                # project overview
├── Makefile                 # test / clean / structure / docs targets
├── pyproject.toml           # src-layout packaging + dependency pins
├── requirements.txt         # a SECOND, diverging pin set — see Part C
├── generate_docs.py         # doc generator; its output tree is NOT committed
├── Dockerfile
└── LICENSE
```

Two directories that older docs name do **not** exist here: a committed documentation
tree (it is generated on demand by `make docs` and never committed) and a second image
directory. A prototyping tree under `src/` was deleted in `4673fc88`. See the stale-doc
ledger in Part C.

`src/` is the import root. The editable install puts it on `sys.path` process-wide, so
the library is imported as `dl_techniques.*`, the trainers as `train.*`, and the apps as
`applications.*` — from any working directory. That is why documentation in this repo
legitimately writes a trainer path as `train/vit/` rather than `src/train/vit/`.

## src/dl_techniques/ — the 13 subpackages

Weighting matters more than completeness here: layers and models together are 574 of the
library's Python files, i.e. 56% of everything under `src/`. The other eleven subpackages
combined are smaller than either one.

> **Whole-table re-derivation 2026-08-25** by `plan-2026-08-24-8fd4f20d`, after the user moved
> `src/dl_techniques/models/` from ~78 flat packages into **11 family directories**
> (`d0b599ff2`, `452d663d2`). ALL enforceable rows were re-run with their own commands and
> **21 moved**. Six of them moved because their COMMAND had stopped measuring the quantity its
> label named: every "model packages" row walked `find src/dl_techniques/models -maxdepth 1
> -type d`, which used to enumerate the model packages and now enumerates the eleven families.
> Re-running such a row verbatim returns a correct-looking number for a different question — the
> single most dangerous failure mode this table has, because the sweep goes GREEN on it. Those
> rows now walk to the LEAVES and say so in their labels. See the dated entry in Part C
> § "2026-08-24/25 — `models/` restructured into families" for the rest.
>
> **How dated notes below spell paths.** Every historical note in this file names model packages
> in TODAY'S spelling (`models/vision/beit/`, not the `models/beit/` that existed on the date the
> note describes), so that every path printed here still resolves. Where a note is *about* the old
> name rather than about the package, the old spelling is kept and carries an `allow-dead-path`
> directive in the ledger. Do not read a dated note's path as evidence of where that package lived
> on that date.
>
> **`layers`/`models` re-derived 2026-08-14 (second pass), after `models/mobile_clip_v2/`
> was SPLIT into `models/vision/fastvit/` (the assembled MCi tower) + `models/vision_language/mobile_clip/mobile_clip_v2.py`
> (the CLIP model), leaving `layers/fastvit/` untouched. That pass also corrected an
> off-by-one in each of the two rows below — the pair total `561` had been right while its
> `294`/`267` split was not; at that HEAD the true split was `295`/`266`. Re-derive with
> `find src/dl_techniques/<pkg> -name '*.py' | wc -l`.
>
> **Whole-table re-derivation 2026-08-14 (first pass), after the `layers/fastvit/` +
> `models/mobile_clip_v2/` packages
> landed (every one of the ~68 Numbers-table rows was
> re-executed, not only the rows this change moved). Previously re-derived 2026-08-11,
> after the `models/vision/beit/` + `src/train/beit/` package landed, and 2026-08-10, after the
> multi-package deletion pass** (`convnext_patch_vae`, `cliffordnet` submodules,
> `bfcliffordunet`, `anomaly_detection`, `qwen3_som`, two `modern_bert` modules). Every
> `.py` FILE-COUNT digit in this section and in the corresponding Numbers-table rows moved
> in one edit together with the `561` / `55%` prose above — that is the row-group rule for
> this section: never half-correct it. (`561` and `55%` moved on 2026-08-14: `layers` gained 8
> files and `models` lost 8 in the whole-table pass, leaving the pair at `560`/55%, and a
> later same-day completion fix added `layers/fastvit/reference.py`, taking `layers` to 294
> and the pair to `561`; 561/1006 is still 55%.) The directory-COUNT rows (model
> packages, trainer dirs, test dirs, applications) were re-derived in the same pass and are
> current. The 2026-08-14 pass also absorbed 17 rows that had already been stale at HEAD.

| Subpackage | `.py` | Role |
|---|---|---|
| **`src/dl_techniques/layers/`** | 299 | **The largest package.** 21 themed subpackages (attention, ffn, norms, embedding, activations, transformers, heads, memory, moe, time_series, fastvit, …) plus 75 loose top-level modules of standalone building blocks. Most subpackages expose a factory module with a registry — see Part B. |
| **`src/dl_techniques/models/`** | 275 | **The second largest, and the only subpackage that is not flat.** Since the 2026-08-24 restructure (`d0b599ff2`, `452d663d2`) it is **12 family directories** — `vision`, `language`, `vision_language`, `time_series`, `general_purpose`, `graph`, `neural_computer`, `common`, `memory`, `point_cloud`, `tabular`, `embeddings_experimental` — holding **85 leaf model packages** between them, with **4 subfamilies** nesting a third level (`src/dl_techniques/models/vision/image_restoration/`, `src/dl_techniques/models/vision/keypoints/`, `src/dl_techniques/models/vision/super_resolution/`, `src/dl_techniques/models/vision_language/sam/`). A *leaf* is a directory that carries an `__init__.py` and has no `__init__.py`-bearing child; a *container* is one that has such children. `vision/` alone holds 35 leaves and `language/` 17. **Listing one level down gives 12, not 85** — so every model-package row in the Numbers table walks to the leaves, and prints that walk in full. Re-derived 2026-08-28: **78 of 85** leaves bind a `create_*` factory in their own package init, **82 of 85** declare a curated `__all__`, **82 of 85** have a non-empty init, and **85 of 85** carry a `README.md`. The 7 that bind no factory are the four `time_series` leaves (`mdn`, `deepar`, `prism`, `tirex` — the curated `src/dl_techniques/models/time_series/__init__.py` re-exports them instead, and it is the one container that does re-export), `src/dl_techniques/models/common/power_sampling/`, and `src/dl_techniques/models/vision_language/sam/sam1/` and `src/dl_techniques/models/vision_language/sam/sam3/`; the 3 with an empty init are `deepar`, `prism` and `tirex`. `sam/` exports nothing on purpose — re-exporting the class `SAM2` there binds the name `SAM2` and shadows the `sam2/` subpackage (its own init docstring carries the reasoning and the exact `ImportError`). **`image_restoration/` is no longer the documentation-only directory this row described until 2026-08-25**: the restructure moved `darkir/`, `pw_fnet/` and `scunet/` underneath it, so it is now a subfamily container that *also* carries `BENCHMARKS.md` and `README.md` — and those tables are still quoted from papers, never measured here, which matters more now that implementations sit beside them. (Before 2026-08-25 this cell read "74 *top-level* model packages ... **71 of 74** bind a `create_*` ... **72 of 74** export a curated `__all__`", and before that "69 of 73" and "70 of 73": four consecutive wrong lists. Every one of them came from quoting a command whose scope had drifted from the question. Re-derive from the Numbers table, never from memory.) The full catalogue is `src/dl_techniques/models/CLAUDE.md`; the family taxonomy is `src/dl_techniques/models/README.md`. See Part C. |
| `src/dl_techniques/losses/` | 43 | Loss families, one module each; `src/dl_techniques/losses/any_loss.py` holds the single dict-based loss registry. |
| `src/dl_techniques/utils/` | 41 | Cross-cutting helpers — `src/dl_techniques/utils/logger.py` (mandatory central logging), `src/dl_techniques/utils/masking/` (the canonical mask factory), plus tensor, export, alignment and geometry helpers. |
| `src/dl_techniques/datasets/` | 37 | Dataset loaders and synthetic generators, with arc, graphs, time_series and vision subtrees. |
| `src/dl_techniques/analyzer/` | 24 | Post-hoc model analysis — `src/dl_techniques/analyzer/model_analyzer.py` is the entry point; calibration and spectral metrics, plus its own visualizers. |
| `src/dl_techniques/metrics/` | 15 | Keras metrics (PSNR, SSIM, perplexity, depth, forecasting, Brier). |
| `src/dl_techniques/optimization/` | 14 | Custom optimizers (Muon, VSGD, SGLD, …), LR schedules, deep-supervision weighting. |
| `src/dl_techniques/callbacks/` | 11 | Reusable Keras callbacks — but **most callbacks in this repo are not here**; see Part B. |
| `src/dl_techniques/initializers/` | 11 | Structured initializers (Gabor, Haar, orthonormal, KAN, polar). |
| `src/dl_techniques/regularizers/` | 8 | Orthogonality (SRIP, soft-orthogonal), entropy, preference regularizers. |
| `src/dl_techniques/visualization/` | 7 | Plotting helpers for training curves, classification, regression, time series. |
| `src/dl_techniques/constraints/` | 2 | Weight constraints; just `src/dl_techniques/constraints/value_range_constraint.py`. The smallest package. |

Every one of the 13 has its own `CLAUDE.md` — start there, not here.

## What a fresh clone actually contains

**Source, tests, and prose. Nothing that was trained.**

| Directory | Present here | Ships in a clone | Consequence |
|---|---|---|---|
| `src/`, `tests/`, `research/`, `imgs/`, `scripts/` | yes | yes | The whole repo a newcomer gets. |
| `results/` | yes — 9 run dirs, 4.5 M | **no** (gitignored) | Zero checkpoints in a clone. These two digits are the most volatile in this file: they describe one machine's untracked scratch, and they fell from 55 dirs / 6.3 G to 6 / 3.4 M between 2026-08-11 and 2026-08-14, and stand at 9 / 4.5 M today. Re-derived by name and mtime 2026-08-24: exactly ONE of the nine, `REDPROOF_conftest_guard_20260814`, is the artifact of `plan-2026-08-14T042537-ff96c6c6` proving a repo-root-write guard RED, so removing that one returns the count to 8; the other eight are `sam3_tiny_*` smoke runs dated 2026-08-13/14. All nine are deliberately left in place because nothing under `results/` may be deleted. Do not read them as a repo property, and never act on them — `results/` is gitignored and untracked, so a deletion there is unrecoverable. |
| `plans/` | yes | **no** (gitignored) | Planner state is machine-local. |
| `data/` | yes — 2.6 G | **no** (untracked) | Local datasets only. |

This is the single most consequential fact for anyone reading the rest of this map.
Every checkpoint the papers in `research/papers/` measure, and every checkpoint the one
surviving app under `src/applications/` loads at startup, lives under `results/` — which is
gitignored. A `results/<run>/final_model.keras` path quoted in a paper, a docstring or a
trainer default is a reference to a local artifact that was never distributed. Nothing in
`src/applications/` runs end-to-end on a fresh clone without training something first.
Concretely: `results/20260717_convunext_denoiser/final_model.keras` exists on this
machine and is absent from every clone.

---

# Part B — the navigation spine

The four things below are what this repo's in-tree docs do not answer, because each
answer spans several packages at once. This is the one part of the map that goes deep.

## The registry / factory surface

**The house pattern, stated once:** where the library has many interchangeable
implementations of one idea, it puts them behind a single `create_*` dispatcher backed
by a module-level registry dict, and you select an implementation with a **string key**
rather than an import. So the question "which attention variants exist?" is answered by
reading one dict, not by listing a directory. New interchangeable variants extend an
existing registry; they do not get a new dispatch style.

There are 9 dicts named `*_REGISTRY`:

| Registry | File | Keys | Dispatcher |
|---|---|---|---|
| `ATTENTION_REGISTRY` | `src/dl_techniques/layers/attention/factory.py` | 33 | `create_attention_layer` |
| `ACTIVATION_REGISTRY` | `src/dl_techniques/layers/activations/factory.py` | 22 | `create_activation_layer` |
| `FFN_REGISTRY` | `src/dl_techniques/layers/ffn/factory.py` | 21 | `create_ffn_layer` |
| `ANYLOSS_REGISTRY` | `src/dl_techniques/losses/any_loss.py` | 16 | dispatched by the `AnyLoss` class |
| `EMBEDDING_REGISTRY` | `src/dl_techniques/layers/embedding/factory.py` | 13 | `create_embedding_layer` |
| `LOGIC_REGISTRY` | `src/dl_techniques/layers/logic/factory.py` | 4 | `create_logic_layer` |
| `SAMPLING_REGISTRY` | `src/dl_techniques/layers/sampling.py` | 3 | `create_sampling_layer` |
| `MIXTURE_REGISTRY` | `src/dl_techniques/layers/mixtures/factory.py` | 3 | `create_mixture_layer` |
| `SEQUENCE_POOLING_REGISTRY` | `src/dl_techniques/layers/sequence_pooling/factory.py` | 3 | `create_sequence_pooling_layer` |

Grepping for `_REGISTRY` will not find all of the surface, though. Three variants:

- **A registry under a different name.** `src/dl_techniques/layers/norms/factory.py`
  drives `create_normalization_layer` from a private `_TYPE_TO_CLASS` dict with 18 keys,
  alongside per-type parameter tables and a `create_normalization_from_config` entry.
  It is the most elaborate factory in the library and the one to copy when a layer family
  has per-type-optional constructor arguments.
- **Dispatchers with no dict at all.** `src/dl_techniques/layers/memory/factory.py`
  exposes two named constructors, `create_mann` and `create_som_2d`, and nothing else.
- **A three-tier dispatcher.** `src/dl_techniques/layers/heads/factory.py` defines a thin
  `create_head(domain, ...)` shim that forwards to one per-domain factory each:
  `src/dl_techniques/layers/heads/nlp/factory.py`,
  `src/dl_techniques/layers/heads/vision/factory.py`, and
  `src/dl_techniques/layers/heads/vlm/factory.py`. Task heads are selected by *domain
  first*, then by key.

And one deliberate absence: `src/dl_techniques/layers/transformers/` has **no factory
module and no registry** — transformer blocks are direct-imported by class, by design.
It is not factory-free, though:
`src/dl_techniques/layers/transformers/vision_encoder.py` and
`src/dl_techniques/layers/transformers/text_encoder.py` each define a family of
`create_*_encoder` helpers, all re-exported from the package init. Those are preset
constructors for two composite encoders, not a keyed dispatcher, and they are the only
`create_*` the package *defines*. Do not "fix" the absence by adding a registry.

The absence is about what it **publishes**. Internally the blocks are registry-*driven*:
10 modules here import a sibling dispatcher (`create_attention_layer`,
`create_ffn_layer`, `create_normalization_layer`, the `*_from_config` variants) and
select by string key, so `ATTENTION_REGISTRY`/`FFN_REGISTRY` keys are the vocabulary of a
transformer block's constructor arguments. For the options behind `attention_type` or
`ffn_type`, read those sibling registries — not this package.

## The model / trainer / test triangle

<!-- allow-dead-path: src/train/convunext/ - drift-note subject: deleted 2026-08-14 in the ConvUNext/bfconvunext merge; naming it is the point of the addendum below -->

Three trees are meant to line up by directory name: **84 leaf model packages** under
`src/dl_techniques/models/`, **48** trainer directories under `src/train/`, and **84** test
directories under `tests/test_models/`. **The first tree is two or three levels deep, the test tree is one level deep, and the
trainer tree is *mostly* one level deep**, which is the first thing to know here: since
2026-08-24 a model package is `models/<family>/<name>/` (and for five of them
`models/<family>/<subfamily>/<name>/`), while its tests are `tests/test_<name>/` and its
trainer is usually `train/<name>/`. **Two entries under `src/train/` are family containers
that nest one level instead** — `src/train/time_series/` (seven children, e.g.
`src/train/time_series/nbeats/train_nbeats.py`) and `src/train/language/` (one child,
`src/train/language/colbert/`, added 2026-08-25). The Numbers-table `src/train/` rows count
maxdepth-1 entries, so each container counts as **one**, not as its children. A name-to-name comparison still works; a *path*-to-path comparison never
did and now cannot. **The two 80s are a coincidence, not a bijection** — the sets differ by two
on each side, and points 4 and 5 below name all four. (The trainer digit was re-measured on
2026-08-14 in the whole-table re-derivation; the trainer count had drifted 48 -> 47
since 2026-08-10 without any row moving to record it. **ADDENDUM, 2026-08-14 (later
the same day): 47 -> 46.** `src/train/convunext/` was deleted when
`models/vision/convunext/` and `models/vision/bias_free_denoisers/bfconvunext.py` were merged onto one
`create_convunext(..., use_bias=...)` builder; its two scripts reached into the
deleted `ConvUNextModel`'s subclass-only internals and had zero importers, zero
tests and no row here. Both derivations move together, and both moved again on 2026-08-25 when
`src/train/language/` landed: 48 -> 47 -> **48** including `src/train/common/`,
47 -> 46 -> **47** excluding it. Re-run the Numbers-table
commands before quoting these; they are correct as of that measurement, not
permanently. The model and test digits were re-derived on 2026-08-25 in the restructure
sweep.) **47 counts `src/train/` entries EXCLUDING
`src/train/common/`, which is a shared library, not a trainer** — the Numbers table
carries both derivations (48 including it, 47 excluding it), because the two rows were
once read as contradicting each other. Comparing those name lists directly is the
obvious move and it produces a badly wrong picture. Five things break the correspondence:

**1. Trainers renamed away from their model package.** A model with no same-named trainer
is usually still trained — under another name:

| Model package | Actual trainer |
|---|---|
| `src/dl_techniques/models/vision/bias_free_denoisers/` | `src/train/bfunet/` — four `train_*.py` scripts, e.g. `src/train/bfunet/train_convunext_denoiser.py` |
| `src/dl_techniques/models/language/byte_latent_transformer/` | `src/train/blt/train_blt.py` |
| `src/dl_techniques/models/language/hierarchical_reasoning_model/` | `src/train/hrm/train_hrm.py` |

**2. Four entries under `src/train/` are not themselves model trainers.** Two are the
family containers just named (`src/train/time_series/`, `src/train/language/`), whose own
`__init__.py` is a marker and whose trainers live one level down. The other two are not
trainers at all: `src/train/logic/` is a
boolean-circuit and rule-learning research harness (numbered experiment scripts plus its
own `src/train/logic/PAPER.md`), and `src/train/rms_variants_train/` is a
normalization-layer ablation sweep (`src/train/rms_variants_train/sweep.py`,
`src/train/rms_variants_train/RESULTS.md`). Both are experiment code parked under
`src/train/`. Read `src/train/` as "runnable pipelines", not "one directory per model".

**3. Every model package nests, so a one-level listing of `models/` finds none of them.**
This used to be point 3 about `src/dl_techniques/models/time_series/` alone, whose seven
children made seven test directories *look* orphaned. After 2026-08-24 that is the rule rather
than the exception: `find src/dl_techniques/models -maxdepth 1 -type d` returns the eleven
families and **not one model package**, so a naive comparison reports all 80 test directories as
orphans and all 47 trainers as untargeted. Two collapsed test instruments were repaired on
2026-08-25 for exactly this — `tests/test_models/test_package_api_contract.py` was silently
running its contract classes against the eleven family names and skipping almost everything.
Both now share one leaf walk, `tests/test_models/model_package_discovery.py`; use it rather than
writing a third copy. `src/dl_techniques/models/time_series/` remains the one container that
did NOT move (it was already nested before the restructure) and the one that re-exports its
children.

**4. Two model packages have no same-named test directory, and both are covered anyway.**
`vision/lewm` is tested by the loose `tests/test_models/test_lewm.py` — the one model whose
tests are a file, not a directory. `vision_language/sam/sam1` is tested by
`tests/test_models/test_sam/`, which predates the `sam1` spelling; `sam2` and `sam3` do have
same-named directories. Any directory-to-directory comparison reports both as untested. Neither
is.

**5. Two test directories have no same-named model package.** `tests/test_models/test_sam/` is
point 4's other half, and `tests/test_models/test_time_series/` targets a *container* — 
`src/dl_techniques/models/time_series/` is the only one that owns a module of its own
(`src/dl_techniques/models/time_series/forecast.py`), and that shared forecasting machinery is
what that directory tests. Neither is an orphan.

Once those five are accounted for, 26 leaf model packages have neither a same-named directory
under `src/train/` nor any `models.<...>.<name>` import anywhere under it (command in the
Numbers table). That measures reachability from `src/train/` and nothing else — this map
did **not** check whether those 25 are covered by tests, so do not read "no trainer" as
"tested instead". Having no trainer is normal here, not a defect to fix. **That row's command
had to be rewritten, not just re-run**: it matched `models\.<name>` against `src/train/`, and a
trainer now imports `dl_techniques.models.vision.beit`, so the old regex matched nothing and the
row would have reported every package unreachable.

## Entry points

**Training.** Every trainer is a module, run with `-m`, never as a file path:

```
MPLBACKEND=Agg .venv/bin/python -m train.<model>.<script> [args]
```

Concretely, `MPLBACKEND=Agg .venv/bin/python -m train.bfunet.train_convunext_denoiser --help`.
For the two nested families the module path carries an extra segment — `-m train.time_series.nbeats.train_nbeats`, `-m train.language.colbert.train_colbert_v1`.
`MPLBACKEND=Agg` is mandatory — matplotlib's interactive backend crashes headless.

**Why it resolves from any working directory.** The editable install drops a `.pth` file
into the virtualenv's `site-packages` whose sole content is the absolute path of this
repo's `src/`. That puts `src/` on `sys.path` for *every* invocation of `.venv/bin/python`,
which is why `train.*`, `dl_techniques.*` and `applications.*` are all importable
top-level names regardless of where you stand. The `pythonpath = ["src"]` setting in
`pyproject.toml` is a separate, redundant mechanism that only affects pytest — do not
mistake it for the reason `-m train....` works.

**Output.** Trainers write to `results/<run_name>/` at the **repo root**, never to a
results directory nested under `src/`. A run directory carries its config, its CSV
training log, and the checkpoints; `src/train/common/compare_runs.py` reads exactly that
layout. Concretely, `results/20260717_convunext_denoiser/config.json` and
`results/20260717_convunext_denoiser/training_log.csv` on this machine. Remember Part A:
`results/` is gitignored, so a run directory exists only where it was produced.

**Applications.** **One** Streamlit app, following the GUI-free-core plus thin entry
split documented in `src/applications/CLAUDE.md`:
`src/applications/bias_free_denoiser/streamlit_app.py` (denoiser-as-prior inverse-problem
solver; also has a headless `src/applications/bias_free_denoiser/main.py`). A second app,
`src/applications/anomaly_detection/`, was deleted on 2026-08-10: the only architecture
its loader supported (`models/convnext_patch_vae/`) had been removed, and no surviving
model in the library implements the `sample_from(...)` decode API its scoring loop
called, so it was end-to-end non-functional. Launch:

```
CUDA_VISIBLE_DEVICES=1 .venv/bin/streamlit run src/applications/<app>/streamlit_app.py \
  --server.address 127.0.0.1 --server.port 8501
```

Streamlit is an **optional extra** (`[project.optional-dependencies].apps` in
`pyproject.toml`), not a core dependency. A plain install of the library will not have it,
and the apps additionally need a checkpoint that no clone contains.

## Where the callbacks actually are

`src/dl_techniques/callbacks/` cannot answer "where are the callbacks", and a map that
sent you there alone would be lying by omission. Of the 46 files under `src/` that
name `keras.callbacks.Callback`, only 10 live in that package. The answer has three
parts:

1. **`src/dl_techniques/callbacks/`** — the reusable, model-agnostic ones (curricula and
   annealing schedules, visualization callbacks, the analyzer hook).
2. **`src/train/common/`** — a *second*, trainer-side callback library that sits outside
   the core library entirely, shared across trainers
   (`src/train/common/callbacks.py`, `src/train/common/step_checkpoint.py`,
   `src/train/common/step_plots.py`, and siblings). 31 of the 46 files are under
   `src/train/`, and this package is where the shared ones concentrate.
3. **Beside the thing they serve** — a callback tightly coupled to one model or one
   optimizer lives with it, not in `src/dl_techniques/callbacks/`. Examples:
   `src/dl_techniques/models/vision/depth_anything/teacher_ema.py`,
   `src/dl_techniques/layers/statistics/residual_acf.py`, and
   `src/dl_techniques/optimization/ww_pgd_optimizer.py`. (This bullet named
   `models/memory_bank/phase_scheduler.py` until 2026-08-25; that package was deleted, not
   moved, in the family restructure — see the ledger.)

The reliable way to find one is a grep, not a directory listing:
`grep -rln "keras.callbacks.Callback" src --include=*.py`.

---

# Part C — conventions, tests, doc routing, and the stale-doc ledger

## Conventions, as measured rather than as asserted

The root `CLAUDE.md` states the house conventions. Below is the same list
*re-derived from the code*, so you can see how strongly each one actually holds.
Every command is in the Numbers table at the foot of this file.

| Convention | Files following it (of the library's `.py`) |
|---|---|
| `@register_dl_technique(...)` on custom classes (was `@keras.saving.register_keras_serializable()`, tabulated 476 until the 2026-08-29 migration; re-derived from the anchored Numbers row below) | 479 |
| A `get_config()` for round-trip serialization | 477 |
| Logging through `src/dl_techniques/utils/logger.py`, never `print` | 335 |

Three honest exceptions. None of them is a defect to go fix on sight; each is a
thing you will meet and should not be surprised by.

- **Raw TensorFlow has not been fully migrated.** 58 files under
  `src/dl_techniques/` still import `tensorflow` directly despite `keras.ops`
  being the stated backend-agnostic surface. The standing repo preference is to
  *migrate* such a site rather than document it as an accepted exception, unless
  it is genuinely unmigratable (FFT, SVD).
- **Docstring style is split repo-wide; both styles are in wide use.** Counted
  on the same scope — the library *outside*
  `src/dl_techniques/layers/attention/` — 370 modules carry Sphinx/reST
  `:param:` and 223 carry a Google-style `Args:` block, and the two sets are not
  disjoint: 16 modules carry both, so these do not sum to a partition. reST is
  therefore not a carve-out and is localized nowhere. The only thing true of
  `src/dl_techniques/layers/attention/` is that it is near-uniformly reST
  *within itself* — 34 of its 35 modules. The split reaches the shared test
  fixtures too: both `tests/conftest.py` and `tests/test_layers/conftest.py`
  document themselves in reST. Read the root `CLAUDE.md`'s "Google-style
  docstrings" line as a preference for new code, not a description of the tree.
- **The factory convention is now near-universal, but this entry is kept because
  the measurement trap that produced it is not.** Re-derived 2026-08-25 over the
  **84 leaf** packages: **77 of 84** bind a `create_*` in their own
  `<pkg>/__init__.py` and **81 of 84** declare a curated `__all__`, so the old
  "read the init before assuming importability" caveat is largely retired. The
  exceptions are named in Part A's `models/` row. Before the `1bfe89d08`
  curated-export pass the binding figure was 27 of 73 while 14 packages defined no
  `create_*` *anywhere* — two different numbers answering two different questions,
  which is the point. Both have since moved four times: 69 and 1 on 2026-08-19
  (`power_sampling` alone), 71 and 2 on 2026-08-24 when a package arrived by `git
  pull`, and 72 and 3 on 2026-08-25 when the restructure changed what the commands
  could even see. **That last move is the one worth studying.** The family
  restructure did not touch a single `__init__.py`, yet re-running the 2026-08-24
  commands verbatim returned **1** and **1**: they walked one level into `models/`
  and found the eleven family directories, whose inits are docstring-only by
  ruling. A number that fell 71 -> 1 is obvious; the danger is that the same class of
  drift usually produces a *plausible* number. **A command's scope is part of its
  claim** — when the tree's shape changes, re-run is not enough, the command has to
  be re-read. **Defining a factory and exporting one are different things**, and a
  plain mention-`grep create_` answers neither: it reads **78** against a binding
  count of 77 today, and until 2026-08-10 it gave 24 against a binding count of 25
  because `src/dl_techniques/models/convnext_patch_vae/__init__.py` was a pure
  docstring cross-referencing factories it never bound. Use an AST scan for the
  definition question and the binding command in the Numbers table for the export
  question; never the mention-grep. A grep-based census run on 2026-08-14 got this
  column wrong for `convnext`, `squeezenet`, `mobilenet`, `qwen`, `gemma` and
  `gpt2`, each time reporting a factory as missing when it existed.

**Before writing a new layer or model, read
`research/2026_keras_custom_models_instructions_v2.md`.** The root `CLAUDE.md` names
it as mandatory and at 2698 lines it is among the longest documents in the repo.
Four of its rules are load-bearing. **Nothing enforces them.** None fails at
layer-definition time: you find out at `.keras` save/load, at shape inference,
or on a weight transfer — and there is no CI to tell you (see Tests, below).
So they are worth knowing before you write, not after:

- **Create/build separation** (§ 1.1 *The Golden Rule: Create vs. Build*) —
  `__init__` only creates sub-layers and stores config; it must never call
  `add_weight` or inspect `input_shape`. `build()` creates weights, explicitly
  builds each sub-layer, and ends with `super().build(input_shape)`.
- **`compute_output_shape()` on every custom layer** (§ 3.4 *Implementing
  compute_output_shape*) — and it must work from stored config on an *unbuilt*
  layer, never from weight shapes.
- **Create all sub-layers unconditionally; gate USAGE in `call()`, never
  creation** (§ 1.2 *Separation of Layer Creation vs. Layer Usage*) —
  conditional creation makes the weight set depend on flags, which breaks
  loading and transfer.
- **Config is data, not code** (§ 1.3 *Configuration as Data*) — constructor
  arguments must be plain serializable values, because `get_config()` has to
  survive a round-trip (the guide's § 8.2 gives the test).

Jump by those section headings, not by line number — the guide is edited and line
anchors rot. Everything else stays there: restating it here is how a map goes stale.

## Tests

`tests/` mirrors `src/dl_techniques/`: package `x` is tested by `tests/test_<x>/`,
for example `tests/test_layers/`.

**`tests/test_models/` is FLAT on purpose, and that is a ruling rather than unfinished
work.** `src/dl_techniques/models/` moved to 11 family directories on 2026-08-24. The test
tree deliberately did NOT follow: there is no `tests/test_models/test_vision/`, and there
must not be one. Recorded as **D-001** of `plan-2026-08-24T205033-8fd4f20d` and restated in
`src/dl_techniques/models/CLAUDE.md`. The measured reason is **218** relative imports of the
form `from ..gradient_flow_oracle`, `from ..knob_sensitivity_oracle`,
`from ..smoke_contract_oracle`, `from ..precision_arm_oracle` (command in the Numbers table),
which reach shared oracle modules that live directly at `tests/test_models/*.py` — for example
`tests/test_models/gradient_flow_oracle.py`. Inserting a family level changes what `..`
resolves to and forces every one of those to `...`, plus the absolute
`from tests.test_models.test_X.Y` forms, across a suite that takes about 1.5 hours and cannot
be verified in a single process. Zero behavioural gain, large mechanical risk. **The rule that
actually matters is untouched: leaf package `x` is still tested by `tests/test_<x>/`** —
`vision/beit/` by `tests/test_models/test_beit/`. Only the phrase "mirrors the directory tree"
is false, and only for this one subtree. Do not "fix" it; if you think it needs fixing, read
D-001 first.

Three named places break the `tests/test_<x>/` rule, and each has cost someone a search:

- **`src/dl_techniques/visualization/` has no test directory at all.** It has a
  `CLAUDE.md`; it has no tests, and no test module anywhere imports it. The trap
  is that `tests/test_utils/test_visualization.py` exists and is a zero-byte
  file, so a name-based search finds something and learns nothing.
- **`tests/test_models/test_lewm.py` is a loose file**, where every other model
  gets a `tests/test_models/test_<name>/` directory — e.g.
  `tests/test_models/test_vit/`. Any directory-to-directory comparison
  reports `lewm` as untested. It is not.
- **`vision_language/sam/sam1/` is tested by `tests/test_models/test_sam/`**, a directory
  name that predates the `sam1` spelling. `sam2` and `sam3` do have same-named directories,
  so the family looks two-thirds consistent and is not. See Part B, point 4.

And one place that only *looks* broken: `src/dl_techniques/layers/sequence_pooling/`
has no test *directory*, but it is tested — by the loose
`tests/test_layers/test_sequence_pooling.py`. A loose module is the dominant
layout there, 78 of them against 21 subdirectories, so under `tests/test_layers/`
a missing directory means nothing at all. Under `tests/test_models/`, where the
directory is the norm, it usually does — which is why `lewm` is worth naming.

**The four fixtures a test author needs**, all restore-safe by construction:

| Fixture | Defined in | What it is for |
|---|---|---|
| `golden_reference_device` | `tests/conftest.py` | The single source of truth for the device a stored golden value is built AND compared on. Cross-device float32 comparison diverges far past the tolerance a golden guard needs, which makes the guard blind rather than merely noisy. |
| `dtype_policy` | `tests/test_layers/conftest.py` | Parametrizes one test over float32 / mixed_float16 / float64 by flipping Keras's *process-global* policy and restoring it. `mixed_float16` is a mandatory regression dtype here, not a nicety. |
| `tf32_disabled` | `tests/test_layers/conftest.py` | Module-scoped TF32 off for precision-sensitive float32 assertions. Opt in with `pytestmark = pytest.mark.usefixtures("tf32_disabled")`. |
| `_tf32_leak_canary` | `tests/test_layers/conftest.py` | Autouse. Fails the first test after *any* TF32 leak, because a leak makes every later measurement in the session order-dependent. No opt-in needed. |

**Never run the full suite as a routine check — it takes about 1.5 hours.** Scope
pytest to the module you touched plus whatever imports it. Two things about that
are worth knowing before you commit:

- **Configured is not installed.** `.pre-commit-config.yaml` *declares* one
  local hook running `python -m pytest` with `always_run: true`. What is
  *installed* under `.git/hooks/` on this machine is only `pre-push`; no
  `pre-commit` hook file exists. So the suite fires on push, not on commit —
  which is why the standing default is `git push --no-verify`. Hook
  installation is per-clone and untracked: run `ls .git/hooks/` to see yours.
- **There is no CI.** No `.github/` directory exists. Nothing runs these tests
  except you.

## Which doc answers which question

This is the part that keeps this file a router. The repo already carries 20
`CLAUDE.md` files and well over a hundred in-tree README and GUIDE files;
the map's job is to route you to the right one, not to paraphrase it.

| Your question | Read |
|---|---|
| How do I write a new layer or model? | `research/2026_keras_custom_models_instructions_v2.md` (mandatory) |
| What are the library-wide conventions? | `src/dl_techniques/CLAUDE.md` |
| What attention variants exist, and which do I pick? | `src/dl_techniques/layers/attention/README.md`, then `src/dl_techniques/layers/attention/GUIDE.md` |
| What activations / sequence-pooling options exist? | `src/dl_techniques/layers/activations/GUIDE.md`, `src/dl_techniques/layers/sequence_pooling/GUIDE.md` |
| How is a layer subpackage organized? | `src/dl_techniques/layers/CLAUDE.md`, plus that subpackage's own `README.md` |
| What task heads exist, and how is `create_head` dispatched? | `src/dl_techniques/layers/heads/CLAUDE.md` |
| What does model `<name>` do? | `src/dl_techniques/models/<family>/<name>/README.md` — every one of the 80 leaves has one. E.g. `src/dl_techniques/models/vision/bias_free_denoisers/README.md`, or `src/dl_techniques/models/vision_language/sam/sam2/README.md` for a third-level leaf. There is no flat `models/<name>/` any more; if you do not know the family, `find src/dl_techniques/models -maxdepth 3 -type d -name '<name>'` |
| Which family is a model in, and what else is in it? | `src/dl_techniques/models/README.md` (the family taxonomy), then that family's `__init__.py` docstring |
| How are model packages meant to be structured? | `src/dl_techniques/models/CLAUDE.md` |
| How do I add or run a trainer? | `src/train/CLAUDE.md`, then that trainer's own `README.md` |
| How do the Streamlit apps split GUI from core? | `src/applications/CLAUDE.md` |
| Which loss / metric / optimizer should I use? | the `CLAUDE.md` in `src/dl_techniques/losses/`, `src/dl_techniques/metrics/`, `src/dl_techniques/optimization/` |
| Which callback should I use — or where is it? | **Part B, "Where the callbacks actually are", first**; only then `src/dl_techniques/callbacks/CLAUDE.md`, which documents that one package and not the callbacks outside it |
| How do I analyze a trained model? | `src/dl_techniques/analyzer/CLAUDE.md` |
| What helpers already exist (masking, export, tensors)? | `src/dl_techniques/utils/CLAUDE.md` — check here before writing a helper |
| What datasets can I load? | `src/dl_techniques/datasets/CLAUDE.md` |
| What is the project about, at a glance? | `README.md` (marketing-style; its self-reported counts are not derived) |
| Which dependency pin is authoritative? | `pyproject.toml` — but see the ledger below |

## The stale-doc ledger

**This is the most valuable section of this file.** The repo has already shipped
documentation describing a directory deleted six weeks earlier *and* a package
that was never committed at all. Every row below was re-verified at the moment
this file was written, with the command or commit that proves it. Paths in this
table are named precisely *because* they do not resolve.

<!-- allow-dead-path: src/experiments/ - ledger subject: deleted in 4673fc88; naming it is the point of the row -->
<!-- allow-dead-path: src/experiments/undivided_attention/ - ledger subject: never committed in any reachable commit -->
<!-- allow-dead-path: docs/ - ledger subject: generated on demand by `make docs`, never committed -->
<!-- allow-dead-path: ww-img/ - ledger subject: named by the root CLAUDE.md, absent from disk -->
<!-- allow-dead-path: .github/ - asserted absent on purpose: this repo has no CI -->
<!-- allow-dead-path: src/dl_techniques/models/jepa/ - ledger subject: was named by models/CLAUDE.md until ba3ec3122; never existed under that name -->
<!-- allow-dead-path: tests/test_models/test_mobilenet_v1.py - ledger subject: cited as the mirroring exemplar; the real path is a directory -->
<!-- allow-dead-path: src/dl_techniques/models/memory_bank/ - ledger subject: DELETED by the user in the 2026-08-24 restructure, not moved; a path rewrite would have invented a home for it -->
<!-- allow-dead-path: src/dl_techniques/models/nano_vlm_world_model/ - ledger subject: DELETED by the user in the 2026-08-24 restructure, not moved -->
<!-- allow-dead-path: tools/ - ledger subject: deleted 2026-08-25 with its sole occupant tools/sam_move_probe.py -->
<!-- allow-dead-path: tools/sam_move_probe.py - ledger subject: self-declared throwaway from a closed plan whose own deletion step never ran -->
<!-- allow-dead-path: src/dl_techniques/models/mobile_clip_v2/ - historical: named by two dated notes above; split on 2026-08-14 into what is today models/vision/fastvit/ + models/vision_language/mobile_clip/ -->
<!-- allow-dead-path: src/dl_techniques/models/convnext_patch_vae/ - historical: deleted 2026-08-10; named by the applications note and the factory-convention bullet -->
<!-- allow-dead-path: src/dl_techniques/models/image_restoration/ - historical: the FLAT spelling used by the 4334b282d note in the Numbers preamble; today src/dl_techniques/models/vision/image_restoration/ -->
<!-- allow-dead-path: src/applications/anomaly_detection/ - ledger subject: deleted 2026-08-10; naming it is the point of the Entry points note -->
<!-- allow-dead-path: src/dl_techniques/models/memory_bank/phase_scheduler.py - ledger subject: the callback-beside-its-model example this map named until 2026-08-25; deleted with its package -->
<!-- allow-dead-path: src/dl_techniques/models/beit/ - convention example: the flat spelling quoted in Part A's boxed "How dated notes spell paths" note; today src/dl_techniques/models/vision/beit/ -->
<!-- allow-dead-path: src/dl_techniques/models/convnext_patch_vae/__init__.py - historical: a file inside a package deleted 2026-08-10; named by the factory-convention bullet -->
<!-- allow-dead-path: tests/test_models/test_vision/ - asserted absent ON PURPOSE: Part C section Tests names it to say it must never be created (D-001) -->

| The claim | Where it is made | Reality, and the proof |
|---|---|---|
| `src/experiments/` is a repo component | root `CLAUDE.md` (structure tree *and* a prose section), `plans/SYSTEM.md` § Components | Deleted in `4673fc88` on 2026-06-16. `git log -1 --format='%h %ad %s' --date=short 4673fc88`; `test -e src/experiments/` fails; `ls -d src/*/` lists only `src/applications/`, `src/dl_techniques/`, `src/train/` |
| `src/experiments/undivided_attention/`, its test suite and its research note exist | `plans/SYSTEM.md`, in a long, confident, entirely fictional section | **Never committed in any commit reachable from any ref.** `git log --all --oneline -- '*undivided*'` returns nothing; `find src tests -name '*undivided*'` returns nothing. The owning plan ran weeks *after* `src/experiments/` was deleted, wrote into a directory the repo no longer had, and its description was then promoted into the atlas as fact. *Caveat: `plans/` is gitignored, so a cloner cannot see this file — which is exactly why the fabrication survived* |
| `docs/` is a repo directory | root `CLAUDE.md` (tree and quick reference), `plans/SYSTEM.md` | Does not exist. `test -e docs/` fails. `Makefile` target `docs` runs `generate_docs.py` on demand; nothing is committed. Any `docs/` you find locally is your own build output |
| `ww-img/` is an assets directory | root `CLAUDE.md` structure tree | Does not exist. `test -e ww-img/` fails. Only `imgs/` is real |
| The module map is `{… optimizers, analyzers …}` | `plans/SYSTEM.md`, and — until a later commit in the same change as this file — the root `CLAUDE.md` § core library, which carried the identical two wrong names | Both names are wrong — the real packages are `src/dl_techniques/optimization/` and `src/dl_techniques/analyzer/` — and the map omits `src/dl_techniques/callbacks/`, `src/dl_techniques/constraints/`, `src/dl_techniques/initializers/` and `src/dl_techniques/regularizers/` entirely |
| "the callbacks live in `src/dl_techniques/callbacks/`" | implied by the structure | 46 files under `src/` name `keras.callbacks.Callback`; only 10 are in `src/dl_techniques/callbacks/` and 31 are under `src/train/`. `grep -rl "keras.callbacks.Callback" src --include=*.py`. See Part B |
| "Config-driven construction via factory functions" | root `CLAUDE.md` Core Conventions | Now holds for models too, though it did not when this row was written: **78 of the 85 leaf** model packages bind a `create_*` in their package init and only **3** (`common/power_sampling/`, `vision_language/sam/sam1/`, `vision_language/sam/sam3/`) define none anywhere. The row is kept because the pre-`1bfe89d08` figures — 27 and 14 — are what the root `CLAUDE.md` claim was measured against, and they moved without the map moving; and because on 2026-08-25 both of this row's commands stopped measuring what their labels said, returning 1 and 1 against the eleven family directories rather than the packages. Both commands are in the Numbers table and both were **rewritten**, not merely re-run |
| Any flat `src/dl_techniques/models/<name>/` path, in any document, test, trainer or docstring | everything written before 2026-08-24 — 2218 occurrences across 750 files at the moment of the move | **The flat layout is gone.** 85 leaf model packages now live under one of 12 families (`models/vision/beit/`, `models/language/bert/`), and five under a third level (`models/vision_language/sam/sam2/`). The user moved them in `d0b599ff2` and `452d663d2`; `plan-2026-08-24-8fd4f20d` rewrote the references on 2026-08-25. Two names were **deleted, not moved** — `models/memory_bank/` and `models/nano_vlm_world_model/` — and a rename map that "fixed" their paths would have manufactured a plausible pointer to nothing, which is why they are declared dead above instead. If you meet a flat `models/<name>/` path anywhere, it predates the restructure: resolve it with `find src/dl_techniques/models -maxdepth 3 -type d -name '<name>'`, never by guessing a family. Note also that `models/image_restoration/` did not merely move — it changed kind, from a documentation-only directory to a subfamily container holding `darkir/`, `pw_fnet/` and `scunet/` |
| `tools/` is a repo directory | nothing tracked claims it; it existed on disk and in one closed plan's records | Deleted 2026-08-25 together with its only file, `tools/sam_move_probe.py`, a self-declared throwaway probe whose owning plan's deletion step never ran. `test -e tools/` fails. It was never in Part A's top-level tree, which is why nothing here had to change when it went |
| `src/train/` is one directory per model architecture | implied by root `CLAUDE.md` and `plans/SYSTEM.md` | `src/train/logic/` and `src/train/rms_variants_train/` are research and ablation harnesses, not model trainers; and several model packages are trained under a *renamed* directory. See Part B |
| **The subtree `CLAUDE.md` files this map routes to are themselves unaudited** — every sample taken so far has found rot, and the rot is numeric as often as it is a dead path | `src/dl_techniques/models/CLAUDE.md` listed a package `src/dl_techniques/models/jepa/` and four more stale claims, three of them numeric (MobileNet "V1, V2, V3"; "23 of the 72" packages binding a `create_*`; "the remaining ~50"; "45+ test suites"). `src/dl_techniques/CLAUDE.md` cited `tests/test_models/test_mobilenet_v1.py` as the test-mirroring exemplar, claimed a pytest **pre-commit** hook runs on every commit, and gave a docstring split of "248 of 285" | **Every instance named here is now repaired** — the row is kept as a standing warning, not as a live indictment. `jepa/` in `ba3ec3122` (the real name is `src/dl_techniques/models/vision/video_jepa/`; a bare `jepa/` never existed); the `models/CLAUDE.md` numbers in `7680bdec0` (to V4; 26 of 73, with the remainder following from it; 81 test directories); the `src/dl_techniques/CLAUDE.md` claims in the same change as this edit — the exemplar is the *directory* `tests/test_models/test_mobilenet/`, only `pre-push` is installed on this machine so the suite fires on push and not on commit, and the docstring split was repaired to 255 of 294 — that file now asserts **256 of 296**, re-derived 2026-08-19, which is this row's own lesson happening to this row's own text. Two lessons the row's own history teaches. First, **it went stale for four days** between the `ba3ec3122` repair and the re-derivation that caught it, across two whole-table sweeps — the Numbers sweep only covers rows carrying a Value and a command, and ledger subjects carry neither, so **re-verify a ledger row before quoting it**. Second, "248 of 285" was *exactly right when written* on 2026-08-11 and was falsified by one package landing: a correct derived number is a perishable good. This map verifies only that its routing targets **exist** — never what they say — so this row is a found-by-sampling floor, not a count |
| `ModernBERT`'s `base` and `large` variants are "95M" / "280M" parameter hybrid local/global encoders | `models/language/modern_bert/model.py`'s own `MODEL_VARIANTS` descriptions, until 2026-08-21 | Both numbers and the architecture label were wrong, and the variants **could not run at all**. Measured 2026-08-21 on a 12 GB RTX 4070 at a sequence length of **8**: `ModernBERT.from_variant("base")` and `from_variant("large")` both raised `ResourceExhaustedError` inside `SingleWindowAttention.call`, because a `window` local layer pads every window to `window_size**2 = 16384` slots independent of `L`. Repaired on 2026-08-21 by shipping `global_attention_interval = 1` for those two variants (all-global attention), which measured **160,584,704 params / 268 weight tensors** and **409,522,176 params / 340 weight tensors**. **That repair was SUPERSEDED on 2026-08-25** by `plan-2026-08-25T053412-0f1fa04f`: the local layers now use the 1-D `window_band` layout instead of a square grid, the cause is gone, and `base`/`large` ship the paper's `global_attention_interval = 3` again — so the current counts are **152,720,384 params / 208 weight tensors** and **399,560,704 params / 264 weight tensors** (measured 2026-08-25, `from_variant(v)` then one forward at L=8). Quote a ModernBERT parameter count only with the interval it was measured at. `tiny` is untouched and still hybrid. Pinned by `tests/test_models/test_modern_bert/test_the_shipped_variants_can_run.py` |

Two of the sources above — `plans/SYSTEM.md` and the plan directories it summarizes —
are gitignored and will not be in your clone. The root `CLAUDE.md` is tracked, and the
claims this ledger attributes to it were corrected in `a3685599`, just after this file.

A closing note on dependencies, since the routing table defers to it. `pyproject.toml`
and `requirements.txt` never *contradict* each other on a shared pin — wherever both name
a package, `requirements.txt` is strictly narrower and fully contained (numpy `>=1.22,<3.0`
against `~=2.0.2`), i.e. redundancy, not conflict. The divergence is coverage:
`requirements.txt` asks for `tensorflow[and-cuda]` where `pyproject.toml` asks for plain
`tensorflow`, and adds `tensorflow-datasets`, `tiktoken` and `datasets`, which
`pyproject.toml` omits although `src/dl_techniques/datasets/vision/imagenet.py`,
`src/dl_techniques/datasets/nlp.py` and `src/dl_techniques/utils/tokenizer.py` import
them. `pyproject.toml` is authoritative for the library API; a bare `pip install -e .`
will not run the dataset loaders or the tokenizer, nor put you on a GPU wheel.


### 2026-08-24/25 — `models/` restructured into families

Recorded here because it is the largest structural change this file has had to absorb, and
because the way it broke this file is instructive rather than merely tedious.

**What the user did** (`d0b599ff2`, `452d663d2`): moved `src/dl_techniques/models/` from ~78
flat packages into **11 family directories**, plus 4 subfamilies at a third level. It was a pure
`git mv` — no source file's *content* changed, and there are zero relative imports anywhere
under `models/`, so every import in the library is absolute and all the damage was
**referential**: dotted module paths and slash-form filesystem paths naming the old flat
location. Two packages were deleted outright in the same work, `models/memory_bank/` and
`models/nano_vlm_world_model/`.

**No checkpoint is affected.** A `.keras` file stores the *registered serializable name*, which
is module-independent, not the defining module's dotted path. Nothing under `results/` had to be
touched and nothing was.

**What `plan-2026-08-24-8fd4f20d` repaired on 2026-08-25**: 2218 stale references across 750
tracked files; 21 orphaned test files for the two deleted packages; two collapsed
directory-walking test instruments; the gutted `src/dl_techniques/models/CLAUDE.md`; a new
`src/dl_techniques/models/README.md`; and this file. Three things are worth carrying forward:

- **77 untracked `__pycache__`-only directories survived at the old flat paths**, and because a
  directory with no `__init__.py` is a PEP 420 implicit namespace package, `import
  dl_techniques.models.vit` **succeeded** for all 77 dead paths while
  `from dl_techniques.models.vit import model` failed. Any check shaped "does the module import?"
  answered YES for every dead path. `make clean` does not close this — it removes the
  `__pycache__` children and leaves the empty parents. Prune to a fixpoint BEFORE measuring.
  The same residue reappeared under `tests/test_models/` after the two orphaned test directories
  were `git rm`-ed, where it kept the `Test dirs under tests/test_models/` row reading 81 and the
  orphan-directory row reading 2.
- **A silently-passing structural break.** `tests/test_models/test_package_api_contract.py`
  built its package list with a one-level listing, so after the move it ran its contract classes
  against the eleven family names and `pytest.skip`-ped almost every case on "declares no
  `__all__`". A 282 KB test file read GREEN while covering nothing. Its sibling
  `tests/test_models/test_roundtrip_instrument_family.py` failed loudly instead — for the wrong
  reason. Both now share `tests/test_models/model_package_discovery.py`.
- **Six rows of the Numbers table had the same defect as that test**, and are the reason this
  entry exists. See the sweep note at the head of § Numbers.

### Checkpoint-affecting changes

Recorded here because they change a weight tree, not just behaviour, and this file is
the tracked place a future reader will look.

* **2026-08-21 — `ModernBERT` `base` / `large` moved to `global_attention_interval = 1`.**
  Every layer is now a `group_query` global layer, so the local layers' fused-QKV subtree
  and its `relative_position_bias_table` are gone and the global layers' RoPE caches take
  their place: **every parameter path under a swapped layer changes**, and any pre-existing
  `base`/`large` checkpoint is unloadable. Impact check, run read-only BEFORE the edit:
  `find results -iname "*modern*bert*"` returned **nothing**, and `find results -name '*.keras'`
  returned **8 files, all `sam3_tiny_*/final_model.keras`** — so the local impact is **zero**.
  A clone contains no checkpoints at all (see the `results/` row above). The hybrid schedule
  remains reachable as `ModernBERT.from_variant("base", global_attention_interval=3)`, which
  at the time was the configuration that could not complete a forward pass on this hardware.
  Rationale and the rejected alternatives: the `D-019` / `D-027` / `D-135` anchors in
  `models/language/modern_bert/model.py`.

* **2026-08-25 — `ModernBERT` `base` / `large` moved BACK to `global_attention_interval = 3`,
  and the weight tree changed again.** `plan-2026-08-25T053412-0f1fa04f` gave
  `WindowAttention` a 1-D `partition_mode='band'` and routed the local layers to it, so
  D-135's reason for forcing `1` (the square-grid local layer raised `ResourceExhaustedError`
  at L=8) no longer exists. Two thirds of the layers are local again, so the RoPE caches of
  the layers that were global go away and the local layers' own subtree returns: `base` moves
  from 160,584,704 params / 268 weight tensors to **152,720,384 / 208**, `large` from
  409,522,176 / 340 to **399,560,704 / 264**. Any checkpoint written between 2026-08-21 and
  2026-08-25 is unloadable; the same impact check as above still returns nothing under
  `results/`. Note the band's local layers carry NO relative-position bias and no positional
  term of their own — that is the layout, not an omission. See the `D-012` / `D-016` anchors
  in `models/language/modern_bert/model.py`.

---

## Numbers, and how to re-derive them

Every *digit* in the prose above is a Value from this table, mechanically enforced. That
enforcement cannot see a quantity spelled out in words ("fewer than a third"), so those
are kept rare and each is decidable from a Value here. Run these from the repo root.

**Re-derive the WHOLE table, not the rows you think you changed.** The enforcement is
only as good as the last full sweep, and it has failed three times, twice in the same
direction:

- **A row can go GREEN while measuring the wrong thing.** This is the worst of the three and it
  is the newest. On 2026-08-24 the user moved `src/dl_techniques/models/` into 11 family
  directories. Six rows here walked `find src/dl_techniques/models -mindepth 1 -maxdepth 1 -type
  d`, an expression that had enumerated the model packages for as long as this table existed and
  now enumerates the families. Re-running them verbatim is not a re-derivation: it answers a
  DIFFERENT QUESTION and prints a plausible number. Two of those rows collapsed loudly (71 -> 1,
  74 -> 4) but one, `Model dirs nested in .../time_series/`, would have stayed at 7 and been
  correct by accident, and a family count near a package count would have gone unnoticed
  entirely. Those six rows now walk to the LEAVES — a directory with an `__init__.py` and no
  `__init__.py`-bearing child — and their Quantity text says "leaf" so the label and the command
  cannot drift apart again. **When the shape of a tree changes, re-READ every command whose scope
  touches it; re-running is not enough.**
- A change that adds one directory or one file moves rows in sections you did not open.
  One `src/train/` addition on 2026-08-01 moved **19 of the 68 numeric rows** — the
  trainer count, the callback counts, the serializable / `get_config` / logger counts,
  the docstring-style counts, the file and line totals, and the coverage row. **Both
  digits in that sentence are a 2026-08-01 record and neither is the table's present
  size** — 19 rows moved out of the 68 the table held on that day. It parses at **76**
  enforceable rows today (`_rows()` in `tests/test_repo_map_numbers.py`, re-derived
  2026-08-26), and 76 is the figure a fresh sweep must reproduce. That file runs **77** test
  cases, not 76: the extra one is `test_the_table_is_still_parseable`, an anti-vacuity guard
  that re-derives nothing. Do not quote the test count as the row count — the 2026-08-26 note
  below did exactly that and read 77, contradicting this bullet in the same file.
- **Fixing the prose without fixing the table silently breaks the invariant this section
  claims.** On 2026-08-01 the "22 of the 73 bind a `create_*`" prose was corrected to 23
  while the table row behind it stayed at 22, so the map contradicted itself for one
  commit. A prose digit and its Value must move in the SAME edit.

The cheap sweep: extract every `| Quantity | Value | \`command\` |` row whose Value is
all digits, run the command, and diff. That is ~15 lines of script and it is the only
thing that makes "mechanically enforced" true rather than aspirational.

**Two rows were REMOVED on 2026-08-01 rather than corrected: total Python LINE counts
under `src/` and `tests/`.** They were measured wrong at the commit that shipped them
(`417760` vs an actual `417836`; `241983` vs `242082`) — the sweep had run before that
same commit's later `.py` edits landed. The correction was not a re-derivation, because
the row class is unmaintainable by construction: a whole-tree line count is invalidated
by *any* edit to *any* `.py` file, including the commit that fixes it, and it has two
different true values at once (working tree vs committed tree) with nothing in the table
to say which is meant. A number that is stale the moment it is written cannot be
"mechanically enforced", and nothing else in this map depends on it. If you want a line
count, run `find src -name '*.py' -exec cat {} + | wc -l` yourself and treat the answer as
valid for that instant only. The FILE-count rows below are kept: they move only when a
file is added or deleted, which is a reviewable event rather than a side effect of typing.

> **The `tests/` file-count row was re-derived on 2026-08-23** (1026 -> 1036: ten guard
> modules added by `plan-2026-08-22-a11304c8`). At that same re-derivation ALL 66
> enforceable rows in this table were re-run with their OWN commands and every other row
> reproduced unchanged — notably `Python files under src/` held at 1007, which is the
> mechanical statement that that plan added ZERO files under `src/`.
>
> **Superseded 2026-08-23** by `plan-2026-08-23-9a110062`, which re-derived ALL 67 enforceable
> rows with their own commands; 10 moved. `Python files under src/` is now **1011** (+4: the
> shared activation-serialization, model-build, Caffe-reference and DINO-reference helpers), and
> `tests/` is **1042** (+8 guards). The +4 is the mechanical statement that THIS plan did add
> files under `src/` — deliberately, each one collapsing a rule that previously had many homes.
>
> **Superseded 2026-08-24** by `plan-2026-08-23-009b7ccf` (BERT + ResNet doc/API repair). ALL 66
> enforceable rows were re-run with their OWN commands; exactly ONE moved. `tests/` is now
> **1055** (+13), and every other row — including `Python files under src/` at **1011** — held
> unchanged, which is the mechanical statement that this plan added ZERO files under `src/`; it
> edited four existing ones. The +13 reconciles exactly to the guard modules the plan landed:
> five under `test_models/test_bert/`, five under `test_models/test_resnet/`, and three
> (including `__init__.py`) in the new `tests/test_train/test_resnet/` package.
>
> The row was GREEN at that plan's base commit `1a3a6e7e3` — `git ls-tree -r --name-only
> 1a3a6e7e3 tests | grep '\.py$' | wc -l` gives exactly 1042 — so all thirteen are that plan's
> own, and no earlier work was owed. (A carried claim that the row was already stale at 1048 was
> checked against the base tree and did not survive.)
>
> **Superseded 2026-08-24 (second pass that day)** by `plan-2026-08-24-247151fd` (the BEiT
> tidy-to-ResNet-shape plan). ALL 66 enforceable rows were re-run with their OWN commands and
> **11 moved** — which is why the whole table is rewritten here rather than the rows one plan
> touched. Only **4 of the 11 are this plan's**: `tests/` 1055 -> 1068 (+13, the seven-file test
> decomposition plus the shared geometry module and its `__init__`), and the three docstring-style
> rows that the `model.py` Google -> Sphinx conversion moved (`:param` 342 -> 343, `Args:`
> 247 -> 246) plus the layers+models pair it did not.
>
> The other **7 are inherited debt from a `git pull`** and are settled here rather than left RED:
> a `models/image_restoration/` package arrived in `4334b282d` — that is the FLAT spelling it had
> on that date; it lives at `src/dl_techniques/models/vision/image_restoration/` today, and it is
> no longer documentation-only — (`src/` 1011 -> 1012, models 269 ->
> 270, layers+models 566 -> 567, the share 55 -> 56%, model packages 73 -> 74, packages with no
> `create_` anywhere 1 -> 2, packages with no same-named trainer dir 26 -> 27) and five
> `research/` notes were deleted (126 -> 122). That package holds **no Python**: an empty
> `__init__.py` beside `README.md` and `BENCHMARKS.md`. Every count above treats it as a package
> because every command above defines a package as a directory — the row is correct and the
> prose in Part A says what it actually is.
>
> Four rows of the Part A per-subpackage table were ALSO stale and are corrected in the same
> edit (`layers` 296 -> 297, `models` 267 -> 270, `utils` 39 -> 41, `initializers` 10 -> 11).
> They are enforced by nothing — that table's rows carry no command, so the sweep never reads
> them — which is exactly how they drifted while the Numbers rows beside them stayed green.
>
> A **twelfth** row then moved as a direct consequence of the fix, and it is left in as a
> demonstration rather than dodged: giving `image_restoration/__init__.py` an honest docstring
> (which says, among other things, that the package binds no `create_*` factory) made the
> bare-mention-grep row read 74. That row's own Quantity text had predicted exactly this — "can
> do so again" — and the mention/binding gap is now 74 against 71. The table was re-derived a
> SECOND time after that edit, and all 66 rows reproduce.
>
> **The `.py` FILE-COUNT rows were re-derived on 2026-08-11**, after the `models/vision/beit/` +
> `src/train/beit/` package landed (and on 2026-08-10, after the multi-package deletion pass) — see the boxed note in Part A § "src/dl_techniques/ — the 13
> subpackages". They are one row-group with that section's prose and its per-subpackage
> table: move all three together or none. The directory-count rows are current as of the
> same date.
>
> **Superseded 2026-08-24 (third pass that day)** by `plan-2026-08-24-64ffd751`
> (the `bfconvunext` rewrite). ALL 66 enforceable rows were re-run with their OWN commands and
> **3 moved**; the other 63 reproduce. Only ONE is this plan's: `tests/` 1068 -> 1069, the single
> delegation-contract guard `tests/test_models/test_bias_free_denoisers/test_the_bfconvunext_delegation_contract.py`.
> The other two are inherited debt from the `git pull` merged in `4ed922781`, whose docstring
> rewrites in `accunet/`, `convnext/`, `coshnet/` and `modern_bert/` converted Google `Args:`
> blocks to Sphinx: `:param ` 343 -> 347 and `Args:` 246 -> 242. Both were ALREADY RED at this
> plan's base commit (the tabulated values are byte-identical in `git show 4ed922781:REPO_MAP.md`
> and at HEAD), so this plan did not break them; it is settling them rather than leaving them RED.
>
> **Seven UNENFORCED prose digits were re-derived in the same edit** — the sweep reads only rows
> that carry a Value *and* a command, so everything below is invisible to it and had drifted:
> the raw-`tensorflow` importer count in Part A's "three honest exceptions" (61 -> 60), that
> section's docstring split (339 -> 347 `:param `, 246 -> 242 `Args:`, 13 -> 18 carrying both),
> the curated-`__all__` figure in BOTH places it is stated (73 -> **72** of 74 — the digit
> contradicted its own sentence, which already named `SAM/` *and* `image_restoration/` as the two
> exceptions, and the Part A models cell additionally claimed `image_restoration` was the only one),
> the `src/train/` callback count in the Part D ledger (32 -> 31, against a green table row reading
> 31), and the `results/` run-dir count in the "ships in a clone" table (8 -> 9, against this
> file's own later note recording that same 8 -> 9 move). **The two `(local)` prose-valued rows
> (`4.5M`, `2.6G`) were also re-run and both reproduce exactly** — they are unenforced because
> their Values are not digits, not because they are unmeasured.

> **Those three rows were REPAIRED on 2026-08-25 by `plan-2026-08-25-704a9bcb/iter-1/step-7`,
> in a commit containing no source file** — which is the "own change" the superseded note below
> asked for. They had been left knowingly RED by `plan-2026-08-25-c71fc3ad/iter-1/step-9`
> (`decisions.md` D-028) to avoid mixing two populations in one commit. Their provenance, kept
> because the *cause* of a drift outlives the number: `` `.py` in `src/dl_techniques/layers/` ``
> (298 -> **299**) and `Modules in src/dl_techniques/layers/ using Sphinx :param` (258 -> **259**)
> were both moved by `4ab68e323` "[layers] cleaning up / normalizing", which added
> `src/dl_techniques/layers/activations/common.py` (3 `:param ` lines); `Library modules using a
> Google-style Args: block OUTSIDE layers/attention/` (230 -> **228**) was moved by the merge
> `a89a25808`, which reshaped the docstrings of
> `src/dl_techniques/models/neural_computer/ntm/model.py` and `model_multitask.py` so their
> anchored `Args:` blocks no longer match. The repair re-ran the **WHOLE** table, not the three
> rows: all 76 enforceable rows were re-derived with their own commands and **exactly these 3
> moved**, so the six ColBERT-review steps that preceded it in that plan moved none. Three
> unenforced prose digits sourced from those rows moved in the SAME edit — this file's Part A
> `layers/` subpackage cell (298 -> 299), its Part A docstring-split bullet (230 -> 228), and
> `src/dl_techniques/CLAUDE.md`'s `layers/` docstring-style row (`258 of 298` -> `259 of 299`).

> **Six rows moved again on 2026-08-26, by `plan-2026-08-25-d5a035ab` itself.** Five were
> moved by `iter-1/step-6.1` (`3475b4bf3`), which added ONE library module,
> `src/dl_techniques/layers/norms/_masking.py`, and that single file moves `Python files
> under src/` (1025 -> **1026**), `` `.py` in `src/dl_techniques/layers/` `` (299 -> **300**),
> `` `.py` in layers + models `` (574 -> **575**), `Modules in
> src/dl_techniques/layers/ using Sphinx :param` (259 -> **260**) and `Library modules using
> Sphinx :param OUTSIDE layers/attention/` (357 -> **358**) — it carries `:param ` lines, so
> it lands in both docstring-style populations. The sixth, `Python files under tests/`
> (1066 -> **1067**), was moved by `iter-1/step-9.2`, which added
> `tests/test_layers/test_norms/test_the_negative_alpha_init_is_checkpoint_visible.py`.
> As on 2026-08-25, the **WHOLE** table was re-derived rather than the named rows: all 76
> enforceable rows were re-run by `tests/test_repo_map_numbers.py` and **exactly these 6
> moved**. (That note said "77" when written, which is the file's TEST count, not its ROW
> count: the 77th case is `test_the_table_is_still_parseable`, the anti-vacuity guard, which
> re-derives nothing. Corrected 2026-08-26 — `_rows()` returns 76 and the paragraph above
> already said so.) Four unenforced prose digits sourced from them moved in the SAME edit — this
> file's Part A `layers/` subpackage cell (299 -> 300), its "layers and models together are
> N of the library's Python files" sentence (574 -> 575, still 56%), its docstring-split
> bullet's Sphinx count (357 -> 358), and `src/dl_techniques/CLAUDE.md`'s `layers/`
> docstring-style row (`259 of 299` -> `260 of 300`). The Google-style `Args:` counts did
> NOT move: `_masking.py` is reST-only.

> **Seven rows moved on 2026-08-26**, re-derived by `plan-2026-08-26-fb07cf4e/iter-1/step-5.1` in a
> commit containing no source file. Only **two of the seven populations** are this plan lineage's,
> and the split matters because the third is inherited debt that was already RED at our base commit:
>
> - **Four rows are one deletion.** `plan-2026-08-26-f3744602/iter-1/step-6` (`0e6b2ea5b`) deleted
>   `src/dl_techniques/layers/moe/integration.py`, and that single file moves `Python files under
>   src/` (1026 -> **1025**), `` `.py` in `src/dl_techniques/layers/` `` (300 -> **299**), `` `.py`
>   in layers + models `` (575 -> **574**), `Modules in src/dl_techniques/layers/ using Sphinx
>   :param` (260 -> **259**) and `Library modules using Sphinx :param OUTSIDE layers/attention/`
>   (358 -> **357**) — five rows, because it carried 14 `:param ` lines and so belonged to both
>   docstring-style populations. It is the exact inverse of the `_masking.py` move recorded
>   directly above: the same five rows, the same size, the opposite sign. The `Args:` rows did not
>   move; `integration.py` was reST-only. `layers+models share of src/ (%)` held at **56** (574 of
>   1025 = 55.99), so a green row here is not evidence that nothing moved.
> - **One row is a net +6 across three plans.** `Python files under tests/` (1067 -> **1073**) is
>   seven additions minus one deletion, and all eight are from this lineage:
>   `plan-2026-08-26-c515641a` added `test_the_dynamic_non_batch_axis_survives_a_concrete_call.py`
>   and `test_the_dead_centroid_keeps_its_distinctiveness.py` under `tests/test_layers/test_mixtures/`;
>   `plan-2026-08-26-f3744602` added `test_the_sparse_kernel_matches_the_dense_oracle.py`,
>   `test_end_to_end.py`, `test_config.py` and `test_experts.py` under `tests/test_layers/test_moe/`
>   and deleted `test_integration.py` with its subject in the same commit (`0e6b2ea5b`, which is also
>   the commit behind the five `src/` rows above); this plan added
>   `test_the_auxiliary_loss_survives_a_mixed_precision_fit.py`.
> - **One row is inherited debt.** `Lines in README.md` (564 -> **610**) was moved by `27d3e13bc`
>   "docs: re-derive every count in the root README", a doc-only commit 26 back from here that
>   updated the README without updating the row sourcing it. It was already RED at this plan's base
>   commit; it is settled here rather than left RED, per this section's own rule.
>
> The **WHOLE** table was re-derived, not the seven: all 76 enforceable rows were re-run with their
> own commands by `tests/test_repo_map_numbers.py` and exactly these 7 moved. Four unenforced prose
> digits sourced from them moved in the SAME edit — this file's Part A `layers/` subpackage cell
> (300 -> 299), its "layers and models together are N of the library's Python files" sentence
> (575 -> 574, still 56%), its Part A docstring-split bullet's Sphinx count (358 -> 357), and
> `src/dl_techniques/CLAUDE.md`'s `layers/` docstring-style row (`260 of 300` -> `259 of 299`). No
> command was found to be wrong; every one of the seven was a stale Value, not a drifted scope.

> **Four rows moved on 2026-08-28**, re-derived by `plan-2026-08-28-6de2095b/iter-1/step-6.1` in a
> commit containing no source file. The **WHOLE** table was re-derived, not the four: all 76
> enforceable rows were re-run with their own commands by `tests/test_repo_map_numbers.py`, exactly
> these four moved, and no command was found to be wrong — every one of the four was a stale Value,
> not a drifted scope. The split between what this plan caused and what it inherited is NOT the
> split the red rows suggested, and that is this entry's lesson:
>
> - **Three of the four are the same population and were ALREADY RED at this plan's base commit
>   `f43881697`, by a different amount.** Re-running the three docstring-style commands in a
>   detached worktree at that base returns `:param ` in `layers/` **261**, `:param ` outside
>   `attention/` **359**, and `Args:` outside `attention/` **226** — against tabulated 259 / 357 /
>   228. So the inherited drift is +2 / +2 / -2 and this plan added a further +3 / +3 / -3, giving
>   **264 / 362 / 223**. A plan that converts N files can therefore NOT read its own contribution
>   off the size of the failure: the red row is the sum of every population member since the last
>   sweep, and half of this one predated the plan.
> - **This plan's three** are `class_token.py`, `mask_token.py` and `register_tokens.py` under
>   `src/dl_techniques/layers/embedding/`, converted from Google `Args:` to Sphinx `:param ` by
>   `iter-1/step-1`. Each leaves the `Args:` population and joins BOTH `:param ` populations, which
>   is why the two Sphinx rows move +3 and the Google row -3 in lockstep. `Modules carrying BOTH
>   styles` held at **17**, and `` `.py` in `src/dl_techniques/layers/` `` held at **299** — this
>   plan added and deleted no file, so the numerator moved without the denominator.
> - **The inherited +2 / -2** is two files converted after the last sweep: `4b8dd2559`
>   (`plan-2026-08-27-0261908f/iter-1/step-7`) converted
>   `src/dl_techniques/layers/norms/polar_weight_norm.py`, and `7648dbc7c` "[layers/theta] cleaning
>   up / normalizing" converted `src/dl_techniques/layers/thera_heat_field.py`. Both were green at
>   `08faf3aba`, the last commit to touch this file.
> - **The fourth row is unrelated to any of this.** `Python files under tests/` (1086 -> **1090**)
>   is four guard modules landed by `plan-2026-08-27-60745fe0` under
>   `tests/test_layers/test_activations/` (`conftest.py`,
>   `test_the_dtype_floor_never_narrows.py`, `test_the_gelu_constant_follows_the_input_dtype.py`,
>   `test_the_tables_survive_a_parent_build.py`). `Python files under src/` held at **1025**, which
>   is the mechanical statement that this plan added zero files under `src/`.
>
> Three unenforced prose digits sourced from these rows moved in the SAME edit — this file's Part A
> docstring-split bullet (`357` -> `362` Sphinx and `228` -> `223` Google; its `17 modules carry
> both` and its `34 of its 35` were re-run and reproduce), and `src/dl_techniques/CLAUDE.md`'s
> `layers/` docstring-style row (`259 of 299` -> `264 of 299`). Every other figure in that file's
> docstring table and its printed command block was re-run in the same edit and reproduces exactly.
>
> **2026-08-29 — the registration migration (`MIGRATIONS.md`) moved five of these rows, and the
> ownership split is measured, not assumed.** Re-run at this plan's base commit `5ac4f5b71` and at
> its step-4 commit `36b27f3b3`: eight rows were red at `36b27f3b3`, of which **five turned red
> during this plan** and are repaired here — `Python files under src/` (1048 -> **1049**), `.py in
> src/dl_techniques/utils/` (41 -> **42**) and `layers+models share of src/ (%)` (56 -> **55**), all
> three from the single new file `src/dl_techniques/utils/keras_registration.py`; `Library modules
> using Sphinx :param OUTSIDE layers/attention/` (369 -> **370**), the same new file's docstring; and
> `Files using @keras.saving.register_keras_serializable` (483 -> **16**), which is not a drift but a
> change of meaning — the row no longer counts registrations at all, so it is relabelled and a new
> row counts `@register_dl_technique` (**480**) beside it.
>
> **2026-08-29, later the same day — the prose pass moved BOTH of those two rows again, and the
> second one had to change its command.** `Files using @keras.saving.register_keras_serializable`
> fell 16 -> **5** as the stale mentions were retired; the 5 that remain are deliberate (four
> superseded DECISION anchors that must name the old form to forbid it, plus the helper). The
> `@register_dl_technique` row was the sharper lesson: its unanchored `grep -rl` counts *prose*,
> so writing the helper's name into four docstrings pushed it 480 -> 484 without a single new
> registration. The Value was not adjusted to fit — the COMMAND was wrong for what the row claims
> to measure, and is now anchored at line start (`^@register_dl_technique`), re-deriving **479**.
> The 1-file gap against the old 480 is `utils/keras_registration.py`, which mentions the helper
> in its own docstring and registers nothing.
>
> **Three rows were ALREADY red at `5ac4f5b71` and are deliberately NOT repaired here**, because
> absorbing another plan's debt into this diff would make the diff lie about what changed:
> `Python files under tests/` (tabulated 1109, re-derives **1118**), `Library modules using a
> Google-style Args: block` (224 -> **223**) and `Library modules carrying BOTH styles`
> (17 -> **16**). They are named so the next reader does not re-diagnose them.

| Quantity | Value | Command |
|---|---|---|
| Python files under `src/` | 1048 | `find src -name '*.py' \| wc -l` |
| Python files under `tests/` | 1126 | `find tests -name '*.py' \| wc -l` |
| In-tree `CLAUDE.md` files (excl. `plans/`) | 19 | `find . -name 'CLAUDE.md' \| grep -v plans \| wc -l` |
| Subpackages of `src/dl_techniques/` | 13 | `find src/dl_techniques -mindepth 1 -maxdepth 1 -type d ! -name __pycache__ \| wc -l` |
| `.py` in `src/dl_techniques/layers/` | 300 | `find src/dl_techniques/layers -name '*.py' \| wc -l` |
| `.py` in `src/dl_techniques/models/` | 285 | `find src/dl_techniques/models -name '*.py' \| wc -l` |
| `.py` in `src/dl_techniques/losses/` | 44 | `find src/dl_techniques/losses -name '*.py' \| wc -l` |
| `.py` in `src/dl_techniques/utils/` | 42 | `find src/dl_techniques/utils -name '*.py' \| wc -l` |
| `.py` in `src/dl_techniques/datasets/` | 37 | `find src/dl_techniques/datasets -name '*.py' \| wc -l` |
| `.py` in `src/dl_techniques/analyzer/` | 24 | `find src/dl_techniques/analyzer -name '*.py' \| wc -l` |
| `.py` in `src/dl_techniques/metrics/` | 16 | `find src/dl_techniques/metrics -name '*.py' \| wc -l` |
| `.py` in `src/dl_techniques/optimization/` | 14 | `find src/dl_techniques/optimization -name '*.py' \| wc -l` |
| `.py` in `src/dl_techniques/callbacks/` | 11 | `find src/dl_techniques/callbacks -name '*.py' \| wc -l` |
| `.py` in `src/dl_techniques/initializers/` | 11 | `find src/dl_techniques/initializers -name '*.py' \| wc -l` |
| `.py` in `src/dl_techniques/regularizers/` | 8 | `find src/dl_techniques/regularizers -name '*.py' \| wc -l` |
| `.py` in `src/dl_techniques/visualization/` | 7 | `find src/dl_techniques/visualization -name '*.py' \| wc -l` |
| `.py` in `src/dl_techniques/constraints/` | 2 | `find src/dl_techniques/constraints -name '*.py' \| wc -l` |
| `.py` in layers + models | 585 | `find src/dl_techniques/layers src/dl_techniques/models -name '*.py' \| wc -l` |
| layers+models share of `src/` (%) | 55 | `echo $(( ( $(find src/dl_techniques/layers -name '*.py' \| wc -l) + $(find src/dl_techniques/models -name '*.py' \| wc -l) ) * 100 / $(find src -name '*.py' \| wc -l) ))` |
| Subpackages under `src/dl_techniques/layers/` | 21 | `find src/dl_techniques/layers -mindepth 1 -maxdepth 1 -type d ! -name __pycache__ \| wc -l` |
| Loose modules directly under `src/dl_techniques/layers/` | 75 | `find src/dl_techniques/layers -maxdepth 1 -name '*.py' \| grep -vc __init__` |
| Model FAMILIES directly under `src/dl_techniques/models/` — **this is NOT the model-package count**; it read 74 and answered that question until the 2026-08-24 restructure, and the label is spelled out because re-running the command was what hid the change | 12 | `find src/dl_techniques/models -mindepth 1 -maxdepth 1 -type d ! -name __pycache__ \| wc -l` |
| **LEAF** model packages under `src/dl_techniques/models/` (a directory carrying an `__init__.py` with no `__init__.py`-bearing child) | 84 | `find src/dl_techniques/models -mindepth 1 -type d ! -path '*__pycache__*' \| while read -r d; do [ -f "$d/__init__.py" ] && [ -z "$(find "$d" -mindepth 2 -maxdepth 2 -name __init__.py)" ] && echo "$d"; done \| wc -l` |
| Containers under `src/dl_techniques/models/` (12 families + 4 subfamilies; excludes `models/` itself) | 16 | `find src/dl_techniques/models -mindepth 1 -type d ! -path '*__pycache__*' \| while read -r d; do [ -f "$d/__init__.py" ] && [ -n "$(find "$d" -mindepth 2 -maxdepth 2 -name __init__.py)" ] && echo "$d"; done \| wc -l` |
| …of those, subfamily containers nested inside a family (`vision/image_restoration/`, `vision/keypoints/`, `vision/super_resolution/`, `vision_language/sam/`) | 4 | `find src/dl_techniques/models -mindepth 2 -type d ! -path '*__pycache__*' \| while read -r d; do [ -f "$d/__init__.py" ] && [ -n "$(find "$d" -mindepth 2 -maxdepth 2 -name __init__.py)" ] && echo "$d"; done \| wc -l` |
| Leaf model packages under `vision/` (the largest family) | 35 | `find src/dl_techniques/models -mindepth 1 -type d ! -path '*__pycache__*' \| while read -r d; do [ -f "$d/__init__.py" ] && [ -z "$(find "$d" -mindepth 2 -maxdepth 2 -name __init__.py)" ] && echo "$d"; done \| grep -c '^src/dl_techniques/models/vision/'` |
| Leaf model packages under `language/` (the second largest) | 17 | `find src/dl_techniques/models -mindepth 1 -type d ! -path '*__pycache__*' \| while read -r d; do [ -f "$d/__init__.py" ] && [ -z "$(find "$d" -mindepth 2 -maxdepth 2 -name __init__.py)" ] && echo "$d"; done \| grep -c '^src/dl_techniques/models/language/'` |
| Leaf model packages with a non-empty `__init__.py` | 81 | `find src/dl_techniques/models -mindepth 1 -type d ! -path '*__pycache__*' \| while read -r d; do [ -f "$d/__init__.py" ] && [ -z "$(find "$d" -mindepth 2 -maxdepth 2 -name __init__.py)" ] && echo "$d"; done \| while read -r p; do [ -s "$p/__init__.py" ] && echo "$p"; done \| wc -l` |
| Leaf model packages declaring `__all__` | 81 | `find src/dl_techniques/models -mindepth 1 -type d ! -path '*__pycache__*' \| while read -r d; do [ -f "$d/__init__.py" ] && [ -z "$(find "$d" -mindepth 2 -maxdepth 2 -name __init__.py)" ] && echo "$d"; done \| while read -r p; do grep -q '__all__' "$p/__init__.py" && echo "$p"; done \| wc -l` |
| Leaf model packages carrying a `README.md` | 84 | `find src/dl_techniques/models -mindepth 1 -type d ! -path '*__pycache__*' \| while read -r d; do [ -f "$d/__init__.py" ] && [ -z "$(find "$d" -mindepth 2 -maxdepth 2 -name __init__.py)" ] && echo "$d"; done \| while read -r p; do [ -f "$p/README.md" ] && echo "$p"; done \| wc -l` |
| Entries under `src/train/` | 49 | `find src/train -mindepth 1 -maxdepth 1 -type d ! -name __pycache__ \| wc -l` |
| Entries under `src/applications/` | 1 | `find src/applications -mindepth 1 -maxdepth 1 -type d ! -name __pycache__ \| wc -l` |
| Top-level dirs under `tests/` | 15 | `find tests -mindepth 1 -maxdepth 1 -type d \| wc -l` |
| Size of `results/` (local) | 4.5M | `LC_ALL=C du -sh results \| cut -f1` |

> **A `Run dirs under \`results/\` (local)` row stood here and was DELETED on 2026-08-17.**
> `results/` is gitignored and untracked, so that number described one machine at one
> moment and moved every time anybody started a training run — it went 8 -> 9 while this
> plan was executing, without a single tracked file changing. A row that cannot hold
> still is not a fact the map can enforce; it is a permanent RED that teaches readers to
> ignore the sweep. The two remaining `(local)` rows survive only because their Values
> are prose (`4.5M`, `2.6G`) and are therefore not enforced by
> `tests/test_repo_map_numbers.py` in the first place. Prefer deleting such a row over
> re-deriving it forever.
| Size of `data/` (local) | 2.6G | `LC_ALL=C du -sh data \| cut -f1` |
| Files under `data/` tracked by git | 0 | `git ls-files data \| wc -l` |
| Notes in `research/` | 122 | `find research -maxdepth 1 -type f -name '*.md' \| wc -l` |
| Paper dirs under `research/papers/` | 5 | `find research/papers -mindepth 1 -maxdepth 1 -type d \| wc -l` |
| Lines in `README.md` | 610 | `wc -l < README.md` |
| Dicts named `*_REGISTRY` under `src/dl_techniques/` | 10 | `grep -rn "^[A-Z_]*REGISTRY[[:space:]]*[:=]" src/dl_techniques --include=*.py \| wc -l` |
| Keys in `ATTENTION_REGISTRY` | 33 | `awk 'index($0,"ATTENTION_REGISTRY")==1{f=1} f&&$0=="}"{f=0} f' src/dl_techniques/layers/attention/factory.py \| grep -cE "^    ['\"][A-Za-z0-9_]+['\"]:"` |
| Keys in `ACTIVATION_REGISTRY` | 22 | `awk 'index($0,"ACTIVATION_REGISTRY")==1{f=1} f&&$0=="}"{f=0} f' src/dl_techniques/layers/activations/factory.py \| grep -cE "^    ['\"][A-Za-z0-9_]+['\"]:"` |
| Keys in `FFN_REGISTRY` | 21 | `awk 'index($0,"FFN_REGISTRY")==1{f=1} f&&$0=="}"{f=0} f' src/dl_techniques/layers/ffn/factory.py \| grep -cE "^    ['\"][A-Za-z0-9_]+['\"]:"` |
| Keys in `ANYLOSS_REGISTRY` | 16 | `awk 'index($0,"ANYLOSS_REGISTRY")==1{f=1} f&&$0=="}"{f=0} f' src/dl_techniques/losses/any_loss.py \| grep -cE "^    ['\"][A-Za-z0-9_]+['\"]:"` |
| Keys in `EMBEDDING_REGISTRY` | 13 | `awk 'index($0,"EMBEDDING_REGISTRY")==1{f=1} f&&$0=="}"{f=0} f' src/dl_techniques/layers/embedding/factory.py \| grep -cE "^    ['\"][A-Za-z0-9_]+['\"]:"` |
| Keys in `LOGIC_REGISTRY` | 4 | `awk 'index($0,"LOGIC_REGISTRY")==1{f=1} f&&$0=="}"{f=0} f' src/dl_techniques/layers/logic/factory.py \| grep -cE "^    ['\"][A-Za-z0-9_]+['\"]:"` |
| Keys in `SAMPLING_REGISTRY` | 3 | `awk 'index($0,"SAMPLING_REGISTRY")==1{f=1} f&&$0=="}"{f=0} f' src/dl_techniques/layers/sampling.py \| grep -cE "^    ['\"][A-Za-z0-9_]+['\"]:"` |
| Keys in `MIXTURE_REGISTRY` | 3 | `awk 'index($0,"MIXTURE_REGISTRY")==1{f=1} f&&$0=="}"{f=0} f' src/dl_techniques/layers/mixtures/factory.py \| grep -cE "^    ['\"][A-Za-z0-9_]+['\"]:"` |
| Keys in `SEQUENCE_POOLING_REGISTRY` | 3 | `awk 'index($0,"SEQUENCE_POOLING_REGISTRY")==1{f=1} f&&$0=="}"{f=0} f' src/dl_techniques/layers/sequence_pooling/factory.py \| grep -cE "^    ['\"][A-Za-z0-9_]+['\"]:"` |
| Keys in `_TYPE_TO_CLASS` (norms factory) | 18 | `awk 'index($0,"_TYPE_TO_CLASS")==1{f=1} f&&$0=="}"{f=0} f' src/dl_techniques/layers/norms/factory.py \| grep -cE "^    ['\"][A-Za-z0-9_]+['\"]:"` |
| Trainer dirs under `src/train/` (excl. `src/train/common/`) | 48 | `find src/train -mindepth 1 -maxdepth 1 -type d ! -name __pycache__ ! -name common \| wc -l` |
| Test dirs under `tests/test_models/` | 84 | `find tests/test_models -mindepth 1 -maxdepth 1 -type d ! -name __pycache__ \| wc -l` |
| Model dirs nested in `src/dl_techniques/models/time_series/` | 7 | `find src/dl_techniques/models/time_series -mindepth 1 -maxdepth 1 -type d ! -name __pycache__ \| wc -l` |
| Files under `src/` naming `keras.callbacks.Callback` | 46 | `grep -rl "keras.callbacks.Callback" src --include=*.py \| wc -l` |
| …of those, inside `src/dl_techniques/callbacks/` | 10 | `grep -rl "keras.callbacks.Callback" src/dl_techniques/callbacks --include=*.py \| wc -l` |
| …of those, under `src/train/` | 31 | `grep -rl "keras.callbacks.Callback" src/train --include=*.py \| wc -l` |
| Files MENTIONING `@keras.saving.register_keras_serializable` — **this is not the registration count and no longer even a decorator count.** It read 483 until 2026-08-29, when all 744 registration sites moved onto the `register_dl_technique` helper (`MIGRATIONS.md`); it then read 16, and 5 after the same day's prose pass retired the stale mentions. All 5 survivors are docstrings and comments — 4 superseded-in-place DECISION anchors that must keep naming the old form to forbid it, plus the helper itself. There are **0** live bare decorators (`grep -rc "^@keras.saving.register_keras_serializable()" src/`). The row below is the one that counts registrations | 5 | `grep -rl "@keras.saving.register_keras_serializable" src/dl_techniques --include=*.py \| wc -l` |
| Files using `@register_dl_technique` (the live registration count; anchored at line start, so prose mentions and the helper's own module do not inflate it — the unanchored form read 480 on 2026-08-29 and then 484 once four docstrings began naming the helper) | 479 | `grep -rlE "^@register_dl_technique" src/dl_techniques --include=*.py \| wc -l` |
| Files defining `get_config` | 484 | `grep -rl "def get_config" src/dl_techniques --include=*.py \| wc -l` |
| Files using the central logger | 348 | `grep -rl "utils.logger" src/dl_techniques --include=*.py \| wc -l` |
| Files importing raw `tensorflow` | 59 | `grep -rl "import tensorflow as tf" src/dl_techniques --include=*.py \| wc -l` |
| `.py` in `src/dl_techniques/layers/attention/` | 35 | `find src/dl_techniques/layers/attention -name '*.py' \| wc -l` |
| …of those using Sphinx `:param` docstrings | 34 | `grep -rl ":param " src/dl_techniques/layers/attention --include=*.py \| wc -l` |
| Modules in `src/dl_techniques/layers/` using Sphinx `:param` (the figure `src/dl_techniques/CLAUDE.md` asserts) | 265 | `grep -rl ":param " src/dl_techniques/layers --include=*.py \| wc -l` |
| Library modules using Sphinx `:param` OUTSIDE `src/dl_techniques/layers/attention/` | 370 | `grep -rl ":param " src/dl_techniques --include=*.py \| grep -vc "src/dl_techniques/layers/attention"` |
| Library modules using a Google-style `Args:` block OUTSIDE `src/dl_techniques/layers/attention/` (same scope as the row above) | 223 | `grep -rlE "^ +Args:$" src/dl_techniques --include=*.py \| grep -vc "src/dl_techniques/layers/attention"` |
| Library modules carrying BOTH styles (the two sets overlap) | 16 | `{ grep -rlE "^ +Args:$" src/dl_techniques --include=*.py; grep -rl ":param " src/dl_techniques --include=*.py; } \| sort \| uniq -d \| wc -l` |
| Modules in `src/dl_techniques/layers/transformers/` importing a sibling `create_*` dispatcher | 10 | `grep -rlE "^from .* import .*create_(attention\|ffn\|normalization)\|^ +create_(attention\|ffn\|normalization)_[a-z_]+,$" src/dl_techniques/layers/transformers --include=*.py \| wc -l` |
| Loose `test_*.py` directly under `tests/test_layers/` | 84 | `find tests/test_layers -maxdepth 1 -name 'test_*.py' \| wc -l` |
| Subdirectories under `tests/test_layers/` | 20 | `find tests/test_layers -mindepth 1 -maxdepth 1 -type d ! -name __pycache__ \| wc -l` |
| **Leaf** model packages with no `create_` function ANYWHERE in the package (`common/power_sampling/`, `vision_language/sam/sam1/`, `vision_language/sam/sam3/`) | 3 | `find src/dl_techniques/models -mindepth 1 -type d ! -path '*__pycache__*' \| while read -r d; do [ -f "$d/__init__.py" ] && [ -z "$(find "$d" -mindepth 2 -maxdepth 2 -name __init__.py)" ] && echo "$d"; done \| while read -r p; do grep -rq '^def create_' "$p" --include=*.py \|\| echo "$p"; done \| wc -l` |
| **Leaf** model packages BINDING a `create_` in their own package init (what a caller sees) | 77 | `find src/dl_techniques/models -mindepth 1 -type d ! -path '*__pycache__*' \| while read -r d; do [ -f "$d/__init__.py" ] && [ -z "$(find "$d" -mindepth 2 -maxdepth 2 -name __init__.py)" ] && echo "$d"; done \| while read -r p; do grep -qE '^(from\|import) .*create_\|^ +create_\|^def create_' "$p/__init__.py" && echo "$p"; done \| wc -l` |
| …the same thing counted by a bare mention-grep, which has overcounted at every measurement: by one until the docstring-only `convnext_patch_vae` init was deleted (2026-08-10), by three on 2026-08-24, and by one today — **73** against a binding count of 72, because `common/power_sampling/` merely MENTIONS `create_` in its init docstring while binding nothing. The gap between this row and the one above it is the whole point of keeping both | 78 | `find src/dl_techniques/models -mindepth 1 -type d ! -path '*__pycache__*' \| while read -r d; do [ -f "$d/__init__.py" ] && [ -z "$(find "$d" -mindepth 2 -maxdepth 2 -name __init__.py)" ] && echo "$d"; done \| while read -r p; do grep -q 'create_' "$p/__init__.py" && echo "$p"; done \| wc -l` |
| **Leaf** model packages with no same-named `src/train/` dir AND no `models.<...>.<name>` import under `src/train/`. The import half of this test had to be REWRITTEN, not re-run: a trainer now writes `dl_techniques.models.vision.beit`, so the old `models\.<name>` pattern matched nothing and would have reported every package unreachable | 26 | `t=$(find src/train -mindepth 1 -maxdepth 1 -type d -printf "%f\n"); find src/dl_techniques/models -mindepth 1 -type d ! -path '*__pycache__*' \| while read -r d; do [ -f "$d/__init__.py" ] && [ -z "$(find "$d" -mindepth 2 -maxdepth 2 -name __init__.py)" ] && echo "$d"; done \| sed 's#.*/##' \| while read -r n; do echo "$t" \| grep -qx "$n" \|\| grep -rqE "models\.([a-z0-9_]+\.)*$n([^a-zA-Z0-9_]\|$)" src/train --include=*.py \|\| echo "$n"; done \| wc -l` |
| Lines in the mandatory authoring guide | 2698 | `wc -l < research/2026_keras_custom_models_instructions_v2.md` |
| Relative imports reaching a shared oracle from `tests/test_models/` (the measured cost of nesting the test tree — see § Tests, D-001) | 218 | `grep -rn "from \.\." --include='*.py' tests/test_models \| wc -l` |
| Loose `test_*.py` directly under `tests/test_models/` | 24 | `find tests/test_models -maxdepth 1 -name 'test_*.py' \| wc -l` |
| Test files under `tests/test_analyzer/` | 2 | `find tests/test_analyzer -name 'test_*.py' \| wc -l` |
| Test directories holding an init module and **no** `test_*.py` (orphans of a deleted subject) | 0 | `for d in $(find tests -mindepth 1 -type d ! -name __pycache__); do [ -z "$(find $d -maxdepth 1 -name 'test_*.py')" ] && [ -z "$(find $d -mindepth 1 -maxdepth 1 -type d ! -name __pycache__)" ] && echo $d; done \| wc -l` |
