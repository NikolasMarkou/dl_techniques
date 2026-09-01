# REPO_MAP — a router for `dl_techniques`

A router: *where code of a given kind lives*, *how the pieces are wired together*, and
*which of the repo's many in-tree docs answers your question*.

It is not a replacement for the in-tree `CLAUDE.md` files — those are the authority on
their own subtree — and it is not an API reference. Restating subtree content here is how
the repo's older maps drifted into describing code that no longer exists, so the omission
is the design.

---

## Top-level layout

```
.
├── src/
│   ├── dl_techniques/       # THE LIBRARY — layers, models, losses, everything reusable
│   ├── train/               # production training pipelines, one entry per trainer
│   └── applications/        # deployable end-user apps built on the library
├── tests/                   # pytest suite; mirrors src/dl_techniques/ EXCEPT test_models/ — see Tests
├── research/                # research notes; research/papers/ holds LaTeX sources
├── imgs/                    # committed image assets
├── scripts/                 # standalone maintenance scripts
├── results/                 # LOCAL ONLY — gitignored, training outputs + checkpoints
├── plans/                   # LOCAL ONLY — gitignored, planner state
├── data/                    # LOCAL ONLY — untracked, local datasets
├── CLAUDE.md                # repo-wide agent/contributor instructions
├── README.md                # project overview
├── Makefile                 # test / clean / structure targets
├── pyproject.toml           # src-layout packaging + dependency pins
├── requirements.txt         # a SECOND, diverging pin set — see Dependency pins
├── Dockerfile
└── LICENSE
```

There is no committed documentation tree and no doc generator: `generate_docs.py` and the
`make docs` target were deleted as deprecated. Documentation lives in the in-tree
`CLAUDE.md` and `README.md` files, and this map routes to them.

`src/` is the import root. The editable install puts it on `sys.path` process-wide, so the
library is imported as `dl_techniques.*`, the trainers as `train.*`, and the apps as
`applications.*` — from any working directory. That is why documentation in this repo
legitimately writes a trainer path as `train/vit/` rather than `src/train/vit/`.

## src/dl_techniques/ — the 13 subpackages

Weighting matters more than completeness: `layers/` and `models/` together are more than
half of every Python file under `src/`. The other eleven combined are smaller than either.

| Subpackage | Role |
|---|---|
| **`layers/`** | The largest package. Themed subpackages (attention, ffn, norms, embedding, activations, transformers, heads, memory, moe, time_series, fastvit, …) plus a large set of loose top-level modules of standalone building blocks. Most subpackages expose a factory module with a registry — see *The registry / factory surface*. |
| **`models/`** | The second largest, and the only subpackage that is not flat. Leaf model packages are grouped into family directories: `vision`, `language`, `vision_language`, `time_series`, `general_purpose`, `graph`, `neural_computer`, `common`, `memory`, `point_cloud`, `tabular`, `embeddings_experimental`. Four of those nest a third level: `vision/image_restoration/`, `vision/keypoints/`, `vision/super_resolution/`, `vision_language/sam/`. Catalogue: `src/dl_techniques/models/CLAUDE.md`; family taxonomy: `src/dl_techniques/models/README.md`. |
| `losses/` | Loss families, one module each; `losses/any_loss.py` holds the single dict-based loss registry. |
| `utils/` | Cross-cutting helpers — `utils/logger.py` (mandatory central logging), `utils/masking/` (the canonical mask factory), plus tensor, alignment and geometry helpers. |
| `datasets/` | Dataset loaders and synthetic generators, with arc, graphs, time_series and vision subtrees. |
| `analyzer/` | Post-hoc model analysis — `analyzer/model_analyzer.py` is the entry point; calibration and spectral metrics, plus its own visualizers. |
| `metrics/` | Keras metrics (PSNR, SSIM, perplexity, depth, forecasting, Brier). |
| `optimization/` | Custom optimizers (Muon, VSGD, SGLD, …), LR schedules, deep-supervision weighting. |
| `callbacks/` | Reusable Keras callbacks — but **most callbacks in this repo are not here**; see *Where the callbacks actually are*. |
| `initializers/` | Structured initializers (Gabor, Haar, orthonormal, KAN, polar). |
| `regularizers/` | Orthogonality (SRIP, soft-orthogonal), entropy, preference regularizers. |
| `visualization/` | Plotting helpers for training curves, classification, regression, time series. |
| `constraints/` | Weight constraints; just `constraints/value_range_constraint.py`. The smallest package. |

Every one of the 13 has its own `CLAUDE.md` — start there, not here.

## What a fresh clone actually contains

**Source, tests, and prose. Nothing that was trained.**

| Directory | Ships in a clone |
|---|---|
| `src/`, `tests/`, `research/`, `imgs/`, `scripts/` | yes — the whole repo a newcomer gets |
| `results/` | **no** (gitignored) — zero checkpoints in a clone |
| `plans/` | **no** (gitignored) — planner state is machine-local |
| `data/` | **no** (untracked) — local datasets only |

Every checkpoint the papers in `research/papers/` measure, and every checkpoint the one
app under `src/applications/` loads at startup, lives under `results/` — which is
gitignored. A `results/<run>/final_model.keras` path quoted in a paper, a docstring or a
trainer default is a local artifact that was never distributed, so nothing in
`src/applications/` runs end-to-end on a fresh clone without training something first.

Nothing under `results/` may be deleted: gitignored and untracked means a deletion there is
unrecoverable.

---

# The navigation spine

The four things below are what the in-tree docs do not answer, because each answer spans
several packages at once. This is the one part of the map that goes deep.

## The registry / factory surface

**The house pattern:** where the library has many interchangeable implementations of one
idea, it puts them behind a single `create_*` dispatcher backed by a module-level registry
dict, and you select one with a **string key** rather than an import. "Which attention
variants exist?" is answered by reading one dict, not by listing a directory. New
interchangeable variants extend an existing registry; they do not get a new dispatch
style.

| Registry | File | Dispatcher |
|---|---|---|
| `ATTENTION_REGISTRY` | `src/dl_techniques/layers/attention/factory.py` | `create_attention_layer` |
| `ACTIVATION_REGISTRY` | `src/dl_techniques/layers/activations/factory.py` | `create_activation_layer` |
| `FFN_REGISTRY` | `src/dl_techniques/layers/ffn/factory.py` | `create_ffn_layer` |
| `ANYLOSS_REGISTRY` | `src/dl_techniques/losses/any_loss.py` | dispatched by the `AnyLoss` class |
| `EMBEDDING_REGISTRY` | `src/dl_techniques/layers/embedding/factory.py` | `create_embedding_layer` |
| `LOGIC_REGISTRY` | `src/dl_techniques/layers/logic/factory.py` | `create_logic_layer` |
| `SAMPLING_REGISTRY` | `src/dl_techniques/layers/sampling.py` | `create_sampling_layer` |
| `MIXTURE_REGISTRY` | `src/dl_techniques/layers/mixtures/factory.py` | `create_mixture_layer` |
| `SEQUENCE_POOLING_REGISTRY` | `src/dl_techniques/layers/sequence_pooling/factory.py` | `create_sequence_pooling_layer` |

**Grepping for `_REGISTRY` will not find all of the surface.** Three variants exist under
other shapes:

- **A registry under a different name.** `src/dl_techniques/layers/norms/factory.py` drives
  `create_normalization_layer` from a private `_TYPE_TO_CLASS` dict, alongside per-type
  parameter tables and a `create_normalization_from_config` entry. It is the most elaborate
  factory in the library and the one to copy when a layer family has per-type-optional
  constructor arguments.
- **Dispatchers with no dict at all.** `src/dl_techniques/layers/memory/factory.py` exposes
  two named constructors, `create_mann` and `create_som_2d`, and nothing else.
- **A three-tier dispatcher.** `src/dl_techniques/layers/heads/factory.py` defines a thin
  `create_head(domain, ...)` shim that forwards to one per-domain factory each:
  `heads/nlp/factory.py`, `heads/vision/factory.py`, `heads/vlm/factory.py`. Task heads are
  selected by *domain first*, then by key.

And one deliberate absence: **`src/dl_techniques/layers/transformers/` has no factory module
and no registry** — transformer blocks are direct-imported by class, by design. Do not
"fix" the absence by adding a registry. It is not factory-free, though:
`transformers/vision_encoder.py` and `transformers/text_encoder.py` each define a family of
`create_*_encoder` helpers, all re-exported from the package init. Those are preset
constructors for two composite encoders, not a keyed dispatcher, and they are the only
`create_*` the package *defines*.

The absence is about what the package **publishes**. Internally the blocks are
registry-*driven*: modules here import a sibling dispatcher (`create_attention_layer`,
`create_ffn_layer`, `create_normalization_layer`, the `*_from_config` variants) and select
by string key, so `ATTENTION_REGISTRY` / `FFN_REGISTRY` keys are the vocabulary of a
transformer block's constructor arguments.

## The model / trainer / test triangle

Three trees are meant to line up **by name**, and they have three different shapes:

| Tree | Shape |
|---|---|
| `src/dl_techniques/models/` | two levels (`models/<family>/<name>/`), sometimes three (`models/<family>/<subfamily>/<name>/`) |
| `src/train/` | mostly one level (`train/<name>/`); `train/time_series/` and `train/language/` nest one further |
| `tests/test_models/` | always one level (`tests/test_models/test_<name>/`) |

So `models/vision/beit/` is trained by `src/train/beit/` and tested by
`tests/test_models/test_beit/`. **A name-to-name comparison works; a path-to-path
comparison never did.** In particular, a one-level listing of `models/` returns the
families and *not one model package*; a naive comparison built on it reports every test
directory as an orphan and every trainer as untargeted. Walk to the leaves instead — a
directory carrying an `__init__.py` with no `__init__.py`-bearing child. That walk already
exists at `tests/test_models/model_package_discovery.py`; use it rather than writing a
third copy.

Five things to know before comparing the trees:

**1. Trainers renamed away from their model package.** A model with no same-named trainer
is usually still trained, under another name:

| Model package | Actual trainer |
|---|---|
| `models/vision/bias_free_denoisers/` | `src/train/bfunet/` — several `train_*.py` scripts |
| `models/language/byte_latent_transformer/` | `src/train/blt/train_blt.py` |
| `models/language/hierarchical_reasoning_model/` | `src/train/hrm/train_hrm.py` |

**2. Some entries under `src/train/` are not model trainers.** `src/train/time_series/` and
`src/train/language/` are family containers whose own `__init__.py` is a marker and whose
trainers live one level down. `src/train/logic/` is a boolean-circuit and rule-learning
research harness (numbered experiment scripts plus its own `PAPER.md`), and
`src/train/rms_variants_train/` is a normalization-layer ablation sweep (`sweep.py`,
`RESULTS.md`). `src/train/common/` is a shared library, not a trainer. Read `src/train/` as
"runnable pipelines", not "one directory per model".

**3. `models/time_series/` is the one family whose `__init__.py` re-exports its children.**
Every other family init is a docstring and nothing else, so always import from the leaf
package (`dl_techniques.models.vision.resnet.model`), never from a family.

**4. Three leaf packages have no same-named test directory, and all three are covered.**
`vision/lewm` is tested by the loose file `tests/test_models/test_lewm.py`;
`vision_language/sam/sam1` by `tests/test_models/test_sam/`, a directory name that predates
the `sam1` spelling (`sam2` and `sam3` do have same-named directories); and
`embeddings_experimental/shared` by `tests/test_models/test_embeddings_shared/`.

**5. `vision_language/sam/` exports nothing on purpose.** Re-exporting the class `SAM2`
there binds the name `SAM2` at the family level and shadows the `sam2/` subpackage; that
init's docstring carries the reasoning and the exact `ImportError`.

Nearly every leaf model package binds a `create_*` factory in its own `__init__.py`,
declares a curated `__all__`, and carries a `README.md`. The exceptions that bind no factory
are the four `time_series` leaves (`mdn`, `deepar`, `prism`, `tirex` — the family init
re-exports them instead), `common/power_sampling/`, and `sam/sam1/` + `sam/sam3/`.

Many leaf model packages have neither a same-named trainer nor any import from
`src/train/`. **Having no trainer is normal here, not a defect to fix**, and it says nothing
about test coverage.

## Entry points

**Training.** Every trainer is a module, run with `-m`, never as a file path:

```
MPLBACKEND=Agg .venv/bin/python -m train.<model>.<script> [args]
```

Concretely, `MPLBACKEND=Agg .venv/bin/python -m train.bfunet.train_convunext_denoiser --help`.
For the two nested families the module path carries an extra segment —
`-m train.time_series.nbeats.train_nbeats`, `-m train.language.colbert.train_colbert_v1`.
`MPLBACKEND=Agg` is mandatory; matplotlib's interactive backend crashes headless.

**Why it resolves from any working directory.** The editable install drops a `.pth` file
into the virtualenv's `site-packages` whose sole content is the absolute path of this repo's
`src/`. That puts `src/` on `sys.path` for *every* invocation of `.venv/bin/python`, which
is why `train.*`, `dl_techniques.*` and `applications.*` are importable top-level names
regardless of where you stand. The `pythonpath = ["src"]` in `pyproject.toml` is a separate
mechanism affecting pytest only — do not mistake it for the reason `-m train....` works.

**Output.** Trainers write to `results/<run_name>/` at the **repo root**, never to a results
directory nested under `src/`. A run directory carries its config, its CSV training log, and
the checkpoints; `src/train/common/compare_runs.py` reads exactly that layout. `results/` is
gitignored, so a run directory exists only where it was produced.

**Applications.** One Streamlit app, following the GUI-free-core plus thin-entry split
documented in `src/applications/CLAUDE.md`:
`src/applications/bias_free_denoiser/streamlit_app.py` (denoiser-as-prior inverse-problem
solver; also has a headless `main.py`). Launch:

```
CUDA_VISIBLE_DEVICES=1 .venv/bin/streamlit run src/applications/<app>/streamlit_app.py \
  --server.address 127.0.0.1 --server.port 8501
```

Streamlit is an **optional extra** (`[project.optional-dependencies].apps` in
`pyproject.toml`), not a core dependency. A plain install of the library will not have it,
and the apps additionally need a checkpoint that no clone contains.

## Where the callbacks actually are

`src/dl_techniques/callbacks/` cannot answer "where are the callbacks". Three parts:

1. **`src/dl_techniques/callbacks/`** — the reusable, model-agnostic ones (curricula and
   annealing schedules, visualization callbacks, the analyzer hook).
2. **`src/train/common/`** — a *second*, trainer-side callback library that sits outside the
   core library entirely and is shared across trainers (`callbacks.py`,
   `step_checkpoint.py`, `step_plots.py`, and siblings). Most callback-defining files in
   this repo are under `src/train/`, and this package is where the shared ones concentrate.
3. **Beside the thing they serve** — a callback tightly coupled to one model or one
   optimizer lives with it. Examples:
   `src/dl_techniques/models/vision/depth_anything/teacher_ema.py`,
   `src/dl_techniques/layers/statistics/residual_acf.py`,
   `src/dl_techniques/optimization/ww_pgd_optimizer.py`.

The reliable way to find one is a grep, not a directory listing:
`grep -rln "keras.callbacks.Callback" src --include=*.py`.

---

# Conventions, tests, and doc routing

## Conventions

The root `CLAUDE.md` and `src/dl_techniques/CLAUDE.md` state the house conventions. How
strongly they hold:

- **Serialization is near-universal.** Almost every module under `src/dl_techniques/` that
  defines a custom class carries `@register_dl_technique(...)` on it and a `get_config()`
  for round-trip serialization. Logging through `src/dl_techniques/utils/logger.py` rather
  than `print` is the strong majority.
- **Raw TensorFlow has not been fully migrated.** A minority of library modules still import
  `tensorflow` directly despite `keras.ops` being the stated backend-agnostic surface. The
  standing preference is to *migrate* such a site rather than document it as an accepted
  exception, unless it is genuinely unmigratable (FFT, SVD).
- **Docstring style is split repo-wide.** Sphinx/reST `:param:` and Google-style `Args:`
  are both in wide use and some modules carry both, so they are not a partition. Read the
  root `CLAUDE.md`'s "Google-style docstrings" line as a preference for new code, not a
  description of the tree. **Match the file you are editing; never convert one wholesale.**

**Before writing a new layer or model, read
`research/2026_keras_custom_models_instructions_v2.md`** — the mandatory authoring guide and
the owner of the golden rules (create/build separation, `compute_output_shape` on every
custom layer, unconditional sub-layer creation, config-as-data). Nothing enforces those
rules: you find out at `.keras` save/load, at shape inference, or on a weight transfer. Jump
by its section headings, not by line number.

## Tests

`tests/` mirrors `src/dl_techniques/`: package `x` is tested by `tests/test_<x>/`, for
example `tests/test_layers/`.

**`tests/test_models/` is FLAT on purpose, and that is a ruling rather than unfinished
work.** `src/dl_techniques/models/` is grouped into family directories; the test tree
deliberately did NOT follow. There is no `tests/test_models/test_vision/`, and there must
not be one. The reason is mechanical: many relative imports of the form
`from ..gradient_flow_oracle` reach shared oracle modules that live directly at
`tests/test_models/*.py`. Inserting a family level changes what `..` resolves to and forces
every one of them to `...`, plus the absolute `from tests.test_models.test_X.Y` forms,
across a suite that takes about 1.5 hours and cannot be verified in a single process. Zero
behavioural gain, large mechanical risk. **The rule that actually matters is untouched: leaf
package `x` is still tested by `tests/test_<x>/`** — `vision/beit/` by
`tests/test_models/test_beit/`. Only the phrase "mirrors the directory tree" is false, and
only for this one subtree.

Named exceptions to the `tests/test_<x>/` rule:

- **`src/dl_techniques/visualization/` has no test directory at all.** It has a `CLAUDE.md`;
  it has no tests, and no test module anywhere imports it.
- **`tests/test_models/test_lewm.py` is a loose file**, where every other model gets a
  directory. Any directory-to-directory comparison reports `lewm` as untested. It is not.
- **`vision_language/sam/sam1/` is tested by `tests/test_models/test_sam/`**, a directory
  name that predates the `sam1` spelling.
- **`embeddings_experimental/shared/` is tested by
  `tests/test_models/test_embeddings_shared/`.**

And one place that only *looks* broken: `src/dl_techniques/layers/sequence_pooling/` has no
test *directory*, but it is tested by the loose `tests/test_layers/test_sequence_pooling.py`.
Loose modules are the dominant layout under `tests/test_layers/`, so a missing directory
there means nothing. Under `tests/test_models/`, where the directory is the norm, it
usually does — which is why `lewm` is worth naming.

**The four fixtures a test author needs**, all restore-safe by construction:

| Fixture | Defined in | What it is for |
|---|---|---|
| `golden_reference_device` | `tests/conftest.py` | The single source of truth for the device a stored golden value is built AND compared on. Cross-device float32 comparison diverges far past the tolerance a golden guard needs, which makes the guard blind rather than merely noisy. |
| `dtype_policy` | `tests/test_layers/conftest.py` | Parametrizes one test over float32 / mixed_float16 / float64 by flipping Keras's *process-global* policy and restoring it. `mixed_float16` is a mandatory regression dtype here, not a nicety. |
| `tf32_disabled` | `tests/test_layers/conftest.py` | Module-scoped TF32 off for precision-sensitive float32 assertions. Opt in with `pytestmark = pytest.mark.usefixtures("tf32_disabled")`. |
| `_tf32_leak_canary` | `tests/test_layers/conftest.py` | Autouse. Fails the first test after *any* TF32 leak, because a leak makes every later measurement in the session order-dependent. No opt-in needed. |

**Never run the full suite as a routine check — it takes about 1.5 hours.** Scope pytest to
the module you touched plus whatever imports it. Two things about that are worth knowing
before you commit:

- **Configured is not installed.** `.pre-commit-config.yaml` *declares* a local hook running
  `python -m pytest` with `always_run: true`. What is *installed* under `.git/hooks/` on
  this machine is only `pre-push`; there is no `pre-commit` hook file. So the suite fires on
  push, not on commit — which is why the standing default is `git push --no-verify`. Hook
  installation is per-clone and untracked: run `ls .git/hooks/` to see yours.
- **There is no CI.** No `.github/` directory exists. Nothing runs these tests except you.

## Which doc answers which question

The repo carries a `CLAUDE.md` per subpackage and well over a hundred in-tree README and
GUIDE files; the map's job is to route you to the right one, not to paraphrase it.

| Your question | Read |
|---|---|
| How do I write a new layer or model? | `research/2026_keras_custom_models_instructions_v2.md` (mandatory) |
| What are the library-wide conventions? | `src/dl_techniques/CLAUDE.md` |
| What attention variants exist, and which do I pick? | `src/dl_techniques/layers/attention/README.md`, then `.../attention/GUIDE.md` |
| What activations / sequence-pooling options exist? | `src/dl_techniques/layers/activations/GUIDE.md`, `.../sequence_pooling/GUIDE.md` |
| How is a layer subpackage organized? | `src/dl_techniques/layers/CLAUDE.md`, plus that subpackage's own `README.md` |
| What task heads exist, and how is `create_head` dispatched? | `src/dl_techniques/layers/heads/CLAUDE.md` |
| What does model `<name>` do? | `src/dl_techniques/models/<family>/<name>/README.md` — every leaf has one. If you do not know the family: `find src/dl_techniques/models -maxdepth 3 -type d -name '<name>'` |
| Which family is a model in, and what else is in it? | `src/dl_techniques/models/README.md`, then that family's `__init__.py` docstring |
| How are model packages meant to be structured? | `src/dl_techniques/models/CLAUDE.md` |
| How do I add or run a trainer? | `src/train/CLAUDE.md`, then that trainer's own `README.md` |
| How do the Streamlit apps split GUI from core? | `src/applications/CLAUDE.md` |
| Which loss / metric / optimizer should I use? | the `CLAUDE.md` in `src/dl_techniques/losses/`, `.../metrics/`, `.../optimization/` |
| Which callback should I use — or where is it? | *Where the callbacks actually are*, above, **first**; only then `src/dl_techniques/callbacks/CLAUDE.md`, which documents that one package and not the callbacks outside it |
| How do I analyze a trained model? | `src/dl_techniques/analyzer/CLAUDE.md` |
| What helpers already exist (masking, tensors, geometry)? | `src/dl_techniques/utils/CLAUDE.md` — check here before writing a helper |
| What datasets can I load? | `src/dl_techniques/datasets/CLAUDE.md` |
| What is the project about, at a glance? | `README.md` |
| Which dependency pin is authoritative? | `pyproject.toml` — but see below |

This map records where things live. It does not audit what the documents it routes to
*say*, so treat a subtree `CLAUDE.md` as authoritative on its own subject and re-derive any
number you intend to quote out of one.

## Dependency pins

`pyproject.toml` and `requirements.txt` do not contradict each other on a shared pin —
wherever both name a package, `requirements.txt` is strictly narrower and fully contained
(numpy `>=1.22,<3.0` against `~=2.0.2`). The divergence is coverage: `requirements.txt` asks
for `tensorflow[and-cuda]` where `pyproject.toml` asks for plain `tensorflow`, and adds
`tensorflow-datasets`, `tiktoken` and `datasets`, which `pyproject.toml` omits although
`src/dl_techniques/datasets/vision/imagenet.py`, `src/dl_techniques/datasets/nlp.py` and
`src/dl_techniques/utils/tokenizer.py` import them.

`pyproject.toml` is authoritative for the library API; a bare `pip install -e .` will not
run the dataset loaders or the tokenizer, nor put you on a GPU wheel.
