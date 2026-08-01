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
them rather than restating them. It is not an API reference: it will not tell you what any
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
├── tests/                   # pytest suite, mirrors src/dl_techniques/
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

Weighting matters more than completeness here: layers and models together are 553 of the
library's Python files, i.e. 55% of everything under `src/`. The other eleven subpackages
combined are smaller than either one.

| Subpackage | `.py` | Role |
|---|---|---|
| **`src/dl_techniques/layers/`** | 283 | **The largest package.** 20 themed subpackages (attention, ffn, norms, embedding, activations, transformers, heads, memory, moe, time_series, …) plus 74 loose top-level modules of standalone building blocks. Most subpackages expose a factory module with a registry — see Part B. |
| **`src/dl_techniques/models/`** | 270 | **The second largest.** 73 self-contained model packages, one per architecture. Fewer than a third export a `create_*` factory from their package init — do not assume one exists; see Part C. |
| `src/dl_techniques/losses/` | 39 | Loss families, one module each; `src/dl_techniques/losses/any_loss.py` holds the single dict-based loss registry. |
| `src/dl_techniques/utils/` | 39 | Cross-cutting helpers — `src/dl_techniques/utils/logger.py` (mandatory central logging), `src/dl_techniques/utils/masking/` (the canonical mask factory), plus tensor, export, alignment and geometry helpers. |
| `src/dl_techniques/datasets/` | 35 | Dataset loaders and synthetic generators, with arc, graphs, time_series and vision subtrees. |
| `src/dl_techniques/analyzer/` | 24 | Post-hoc model analysis — `src/dl_techniques/analyzer/model_analyzer.py` is the entry point; calibration and spectral metrics, plus its own visualizers. |
| `src/dl_techniques/metrics/` | 15 | Keras metrics (PSNR, SSIM, perplexity, depth, forecasting, Brier). |
| `src/dl_techniques/optimization/` | 14 | Custom optimizers (Muon, VSGD, SGLD, …), LR schedules, deep-supervision weighting. |
| `src/dl_techniques/callbacks/` | 12 | Reusable Keras callbacks — but **most callbacks in this repo are not here**; see Part B. |
| `src/dl_techniques/initializers/` | 9 | Structured initializers (Gabor, Haar, orthonormal, KAN, polar). |
| `src/dl_techniques/regularizers/` | 8 | Orthogonality (SRIP, soft-orthogonal), entropy, preference regularizers. |
| `src/dl_techniques/visualization/` | 7 | Plotting helpers for training curves, classification, regression, time series. |
| `src/dl_techniques/constraints/` | 2 | Weight constraints; just `src/dl_techniques/constraints/value_range_constraint.py`. The smallest package. |

Every one of the 13 has its own `CLAUDE.md` — start there, not here.

## What a fresh clone actually contains

**Source, tests, and prose. Nothing that was trained.**

| Directory | Present here | Ships in a clone | Consequence |
|---|---|---|---|
| `src/`, `tests/`, `research/`, `imgs/`, `scripts/` | yes | yes | The whole repo a newcomer gets. |
| `results/` | yes — 56 run dirs, 6.2 G | **no** (gitignored) | Zero checkpoints in a clone. |
| `plans/` | yes | **no** (gitignored) | Planner state is machine-local. |
| `data/` | yes — 2.6 G | **no** (untracked) | Local datasets only. |

This is the single most consequential fact for anyone reading the rest of this map.
Every checkpoint the papers in `research/papers/` measure, and every checkpoint the two
apps under `src/applications/` load at startup, lives under `results/` — which is
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
| `ATTENTION_REGISTRY` | `src/dl_techniques/layers/attention/factory.py` | 31 | `create_attention_layer` |
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
`create_*` in the package. Do not "fix" the absence by adding a registry.

## The model / trainer / test triangle

Three trees are meant to line up by directory name: 73 model packages under
`src/dl_techniques/models/`, 45 trainer directories under `src/train/`, and 79 test
directories under `tests/test_models/`. Comparing those name lists directly is the
obvious move and it produces a badly wrong picture. Four things break the correspondence:

**1. Trainers renamed away from their model package.** A model with no same-named trainer
is usually still trained — under another name:

| Model package | Actual trainer |
|---|---|
| `src/dl_techniques/models/bias_free_denoisers/` | `src/train/bfunet/` — four `train_*.py` scripts, e.g. `src/train/bfunet/train_convunext_denoiser.py` |
| `src/dl_techniques/models/byte_latent_transformer/` | `src/train/blt/train_blt.py` |
| `src/dl_techniques/models/hierarchical_reasoning_model/` | `src/train/hrm/train_hrm.py` |

**2. Two entries under `src/train/` are not model trainers.** `src/train/logic/` is a
boolean-circuit and rule-learning research harness (numbered experiment scripts plus its
own `src/train/logic/PAPER.md`), and `src/train/rms_variants_train/` is a
normalization-layer ablation sweep (`src/train/rms_variants_train/sweep.py`,
`src/train/rms_variants_train/RESULTS.md`). Both are experiment code parked under
`src/train/`. Read `src/train/` as "runnable pipelines", not "one directory per model".

**3. `src/dl_techniques/models/time_series/` nests two levels deep.** Seven test
directories look orphaned — `tests/test_models/test_nbeats/`,
`tests/test_models/test_deepar/`, `tests/test_models/test_xlstm/`,
`tests/test_models/test_prism/`, `tests/test_models/test_tirex/`,
`tests/test_models/test_mdn/`, `tests/test_models/test_adaptive_ema/` — only because
their models live inside `src/dl_techniques/models/time_series/` rather than at the
top level.

**4. One model's tests are a file, not a directory.** `lewm` is tested by the loose
`tests/test_models/test_lewm.py`. Any directory-to-directory comparison will report it as
untested; it is not.

Once those four are accounted for, 27 model packages have neither a same-named directory
under `src/train/` nor any `models.<name>` import anywhere under it (command in the
Numbers table). That measures reachability from `src/train/` and nothing else — this map
did **not** check whether those 27 are covered by tests, so do not read "no trainer" as
"tested instead". Having no trainer is normal here, not a defect to fix.

## Entry points

**Training.** Every trainer is a module, run with `-m`, never as a file path:

```
MPLBACKEND=Agg .venv/bin/python -m train.<model>.<script> [args]
```

Concretely, `MPLBACKEND=Agg .venv/bin/python -m train.bfunet.train_convunext_denoiser --help`.
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

**Applications.** Two Streamlit apps, each following the GUI-free-core plus thin entry
split documented in `src/applications/CLAUDE.md`:
`src/applications/bias_free_denoiser/streamlit_app.py` (denoiser-as-prior inverse-problem
solver; also has a headless `src/applications/bias_free_denoiser/main.py`) and
`src/applications/anomaly_detection/streamlit_app.py`. Launch:

```
CUDA_VISIBLE_DEVICES=1 .venv/bin/streamlit run src/applications/<app>/streamlit_app.py \
  --server.address 127.0.0.1 --server.port 8501
```

Streamlit is an **optional extra** (`[project.optional-dependencies].apps` in
`pyproject.toml`), not a core dependency. A plain install of the library will not have it,
and the apps additionally need a checkpoint that no clone contains.

## Where the callbacks actually are

`src/dl_techniques/callbacks/` cannot answer "where are the callbacks", and a map that
sent you there alone would be lying by omission. Of the 56 files under `src/` that
name `keras.callbacks.Callback`, only 11 live in that package. The answer has three
parts:

1. **`src/dl_techniques/callbacks/`** — the reusable, model-agnostic ones (curricula and
   annealing schedules, visualization callbacks, the analyzer hook).
2. **`src/train/common/`** — a *second*, trainer-side callback library that sits outside
   the core library entirely, shared across trainers
   (`src/train/common/callbacks.py`, `src/train/common/step_checkpoint.py`,
   `src/train/common/step_plots.py`, and siblings). 38 of the 56 files are under
   `src/train/`, and this package is where the shared ones concentrate.
3. **Beside the thing they serve** — a callback tightly coupled to one model or one
   optimizer lives with it, not in `src/dl_techniques/callbacks/`. Examples:
   `src/dl_techniques/models/depth_anything/teacher_ema.py`,
   `src/dl_techniques/models/memory_bank/phase_scheduler.py`, and
   `src/dl_techniques/optimization/ww_pgd_optimizer.py`.

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
| `@keras.saving.register_keras_serializable()` on custom classes | 461 |
| A `get_config()` for round-trip serialization | 464 |
| Logging through `src/dl_techniques/utils/logger.py`, never `print` | 354 |

Three honest exceptions. None of them is a defect to go fix on sight; each is a
thing you will meet and should not be surprised by.

- **Raw TensorFlow has not been fully migrated.** 68 files under
  `src/dl_techniques/` still import `tensorflow` directly despite `keras.ops`
  being the stated backend-agnostic surface. The standing repo preference is to
  *migrate* such a site rather than document it as an accepted exception, unless
  it is genuinely unmigratable (FFT, SVD).
- **Docstring style is split repo-wide; both styles are in wide use.** 260
  library modules carry a Google-style `Args:` block, and 311 modules *outside*
  `src/dl_techniques/layers/attention/` carry Sphinx/reST `:param:`. reST is
  therefore not a carve-out and is localized nowhere. The only thing true of
  `src/dl_techniques/layers/attention/` is that it is near-uniformly reST
  *within itself* — 33 of its 34 modules. The split reaches the shared test
  fixtures too: both `tests/conftest.py` and `tests/test_layers/conftest.py`
  document themselves in reST. Read the root `CLAUDE.md`'s "Google-style
  docstrings" line as a preference for new code, not a description of the tree.
- **The factory convention is not universal, and the two ways of measuring it
  disagree sharply.** 16 of the 73 model packages define no `create_*` function
  *anywhere* in the package. But defining one and exporting one are different
  things: only 23 of the 73 name a `create_*` in their own `<pkg>/__init__.py`
  (concretely `src/dl_techniques/models/vit/__init__.py`), and that is the
  figure a caller actually experiences; the rest bury the factory in a submodule
  you must know to import. Read the package init before assuming importability.

**Before writing a new layer or model, read
`research/2026_keras_custom_models_instructions.md`.** The root `CLAUDE.md` names
it as mandatory and at 3105 lines it is among the longest documents in the repo.
Four of its rules are load-bearing and will reject your layer if broken:
create/build separation, a mandatory `compute_output_shape()`, creating every
sub-layer unconditionally (gate usage in `call()`, never creation), and config
being plain serializable data. Read them there — restating them here is how a
map goes stale.

## Tests

`tests/` mirrors `src/dl_techniques/`: package `x` is tested by `tests/test_<x>/`,
for example `tests/test_layers/`.
Three named places break that rule, and each has cost someone a search:

- **`src/dl_techniques/visualization/` has no test directory at all.** It has a
  `CLAUDE.md`; it has no tests, and no test module anywhere imports it. The trap
  is that `tests/test_utils/test_visualization.py` exists and is a zero-byte
  file, so a name-based search finds something and learns nothing.
- **`tests/test_analysis/` is a vestigial empty directory** — it holds nothing but an
  empty init module and 0 test files, and its name shadows the real, populated
  `tests/test_analyzer/` (2 files). Searching for "analysis tests" lands you in
  the wrong one.
- **`tests/test_models/test_lewm.py` is a loose file**, where every other model
  gets a `tests/test_models/test_<name>/` directory — e.g.
  `tests/test_models/test_ccnets/`. Any directory-to-directory comparison
  reports `lewm` as untested. It is not.

And one place that only *looks* broken: `src/dl_techniques/layers/sequence_pooling/`
has no test *directory*, but it is tested — by the loose
`tests/test_layers/test_sequence_pooling.py`. A loose module is the dominant
layout there, 79 of them against 20 subdirectories, so under `tests/test_layers/`
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
| How do I write a new layer or model? | `research/2026_keras_custom_models_instructions.md` (mandatory) |
| What are the library-wide conventions? | `src/dl_techniques/CLAUDE.md` |
| What attention variants exist, and which do I pick? | `src/dl_techniques/layers/attention/README.md`, then `src/dl_techniques/layers/attention/GUIDE.md` |
| What activations / sequence-pooling options exist? | `src/dl_techniques/layers/activations/GUIDE.md`, `src/dl_techniques/layers/sequence_pooling/GUIDE.md` |
| How is a layer subpackage organized? | `src/dl_techniques/layers/CLAUDE.md`, plus that subpackage's own `README.md` |
| What task heads exist, and how is `create_head` dispatched? | `src/dl_techniques/layers/heads/CLAUDE.md` |
| What does model `<name>` do? | `src/dl_techniques/models/<name>/README.md`, e.g. `src/dl_techniques/models/bias_free_denoisers/README.md` |
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

| The claim | Where it is made | Reality, and the proof |
|---|---|---|
| `src/experiments/` is a repo component | root `CLAUDE.md` (structure tree *and* a prose section), `plans/SYSTEM.md` § Components | Deleted in `4673fc88` on 2026-06-16. `git log -1 --format='%h %ad %s' --date=short 4673fc88`; `test -e src/experiments/` fails; `ls -d src/*/` lists only `src/applications/`, `src/dl_techniques/`, `src/train/` |
| `src/experiments/undivided_attention/`, its test suite and its research note exist | `plans/SYSTEM.md`, in a long, confident, entirely fictional section | **Never committed in any commit reachable from any ref.** `git log --all --oneline -- '*undivided*'` returns nothing; `find src tests -name '*undivided*'` returns nothing. The owning plan ran weeks *after* `src/experiments/` was deleted, wrote into a directory the repo no longer had, and its description was then promoted into the atlas as fact. *Caveat: `plans/` is gitignored, so a cloner cannot see this file — which is exactly why the fabrication survived* |
| `docs/` is a repo directory | root `CLAUDE.md` (tree and quick reference), `plans/SYSTEM.md` | Does not exist. `test -e docs/` fails. `Makefile` target `docs` runs `generate_docs.py` on demand; nothing is committed. Any `docs/` you find locally is your own build output |
| `ww-img/` is an assets directory | root `CLAUDE.md` structure tree | Does not exist. `test -e ww-img/` fails. Only `imgs/` is real |
| The module map is `{… optimizers, analyzers …}` | `plans/SYSTEM.md`, and — until a later commit in the same change as this file — the root `CLAUDE.md` § core library, which carried the identical two wrong names | Both names are wrong — the real packages are `src/dl_techniques/optimization/` and `src/dl_techniques/analyzer/` — and the map omits `src/dl_techniques/callbacks/`, `src/dl_techniques/constraints/`, `src/dl_techniques/initializers/` and `src/dl_techniques/regularizers/` entirely |
| "the callbacks live in `src/dl_techniques/callbacks/`" | implied by the structure | 56 files under `src/` name `keras.callbacks.Callback`; only 11 are in `src/dl_techniques/callbacks/` and 38 are under `src/train/`. `grep -rl "keras.callbacks.Callback" src --include=*.py`. See Part B |
| "Config-driven construction via factory functions" | root `CLAUDE.md` Core Conventions | Holds for the layer families, not for models: only 23 of the 73 model packages export a `create_*` from their package init, and 16 define none anywhere. Both commands are in the Numbers table |
| `src/train/` is one directory per model architecture | implied by root `CLAUDE.md` and `plans/SYSTEM.md` | `src/train/logic/` and `src/train/rms_variants_train/` are research and ablation harnesses, not model trainers; and several model packages are trained under a *renamed* directory. See Part B |

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

---

## Numbers, and how to re-derive them

Every *digit* in the prose above is a Value from this table, mechanically enforced. That
enforcement cannot see a quantity spelled out in words ("fewer than a third"), so those
are kept rare and each is decidable from a Value here. Run these from the repo root.

| Quantity | Value | Command |
|---|---|---|
| Python files under `src/` | 1002 | `find src -name '*.py' \| wc -l` |
| Python files under `tests/` | 692 | `find tests -name '*.py' \| wc -l` |
| Python lines under `src/` | 413693 | `find src -name '*.py' -exec cat {} + \| wc -l` |
| Python lines under `tests/` | 236612 | `find tests -name '*.py' -exec cat {} + \| wc -l` |
| In-tree `CLAUDE.md` files (excl. `plans/`) | 20 | `find . -name 'CLAUDE.md' \| grep -v plans \| wc -l` |
| Subpackages of `src/dl_techniques/` | 13 | `find src/dl_techniques -mindepth 1 -maxdepth 1 -type d ! -name __pycache__ \| wc -l` |
| `.py` in `src/dl_techniques/layers/` | 283 | `find src/dl_techniques/layers -name '*.py' \| wc -l` |
| `.py` in `src/dl_techniques/models/` | 270 | `find src/dl_techniques/models -name '*.py' \| wc -l` |
| `.py` in `src/dl_techniques/losses/` | 39 | `find src/dl_techniques/losses -name '*.py' \| wc -l` |
| `.py` in `src/dl_techniques/utils/` | 39 | `find src/dl_techniques/utils -name '*.py' \| wc -l` |
| `.py` in `src/dl_techniques/datasets/` | 35 | `find src/dl_techniques/datasets -name '*.py' \| wc -l` |
| `.py` in `src/dl_techniques/analyzer/` | 24 | `find src/dl_techniques/analyzer -name '*.py' \| wc -l` |
| `.py` in `src/dl_techniques/metrics/` | 15 | `find src/dl_techniques/metrics -name '*.py' \| wc -l` |
| `.py` in `src/dl_techniques/optimization/` | 14 | `find src/dl_techniques/optimization -name '*.py' \| wc -l` |
| `.py` in `src/dl_techniques/callbacks/` | 12 | `find src/dl_techniques/callbacks -name '*.py' \| wc -l` |
| `.py` in `src/dl_techniques/initializers/` | 9 | `find src/dl_techniques/initializers -name '*.py' \| wc -l` |
| `.py` in `src/dl_techniques/regularizers/` | 8 | `find src/dl_techniques/regularizers -name '*.py' \| wc -l` |
| `.py` in `src/dl_techniques/visualization/` | 7 | `find src/dl_techniques/visualization -name '*.py' \| wc -l` |
| `.py` in `src/dl_techniques/constraints/` | 2 | `find src/dl_techniques/constraints -name '*.py' \| wc -l` |
| `.py` in layers + models | 553 | `find src/dl_techniques/layers src/dl_techniques/models -name '*.py' \| wc -l` |
| layers+models share of `src/` (%) | 55 | `echo $(( ( $(find src/dl_techniques/layers -name '*.py' \| wc -l) + $(find src/dl_techniques/models -name '*.py' \| wc -l) ) * 100 / $(find src -name '*.py' \| wc -l) ))` |
| Subpackages under `src/dl_techniques/layers/` | 20 | `find src/dl_techniques/layers -mindepth 1 -maxdepth 1 -type d ! -name __pycache__ \| wc -l` |
| Loose modules directly under `src/dl_techniques/layers/` | 74 | `find src/dl_techniques/layers -maxdepth 1 -name '*.py' \| grep -vc __init__` |
| Model packages under `src/dl_techniques/models/` | 73 | `find src/dl_techniques/models -mindepth 1 -maxdepth 1 -type d ! -name __pycache__ \| wc -l` |
| Entries under `src/train/` | 46 | `find src/train -mindepth 1 -maxdepth 1 -type d ! -name __pycache__ \| wc -l` |
| Entries under `src/applications/` | 2 | `find src/applications -mindepth 1 -maxdepth 1 -type d ! -name __pycache__ \| wc -l` |
| Top-level dirs under `tests/` | 17 | `find tests -mindepth 1 -maxdepth 1 -type d \| wc -l` |
| Run dirs under `results/` (local) | 56 | `find results -mindepth 1 -maxdepth 1 -type d \| wc -l` |
| Size of `results/` (local) | 6.2G | `LC_ALL=C du -sh results \| cut -f1` |
| Size of `data/` (local) | 2.6G | `LC_ALL=C du -sh data \| cut -f1` |
| Files under `data/` tracked by git | 0 | `git ls-files data \| wc -l` |
| Notes in `research/` | 120 | `find research -maxdepth 1 -type f -name '*.md' \| wc -l` |
| Paper dirs under `research/papers/` | 5 | `find research/papers -mindepth 1 -maxdepth 1 -type d \| wc -l` |
| Lines in `README.md` | 485 | `wc -l < README.md` |
| Dicts named `*_REGISTRY` under `src/dl_techniques/` | 9 | `grep -rn "^[A-Z_]*REGISTRY[[:space:]]*[:=]" src/dl_techniques --include=*.py \| wc -l` |
| Keys in `ATTENTION_REGISTRY` | 31 | `awk 'index($0,"ATTENTION_REGISTRY")==1{f=1} f&&$0=="}"{f=0} f' src/dl_techniques/layers/attention/factory.py \| grep -cE "^    ['\"][A-Za-z0-9_]+['\"]:"` |
| Keys in `ACTIVATION_REGISTRY` | 22 | `awk 'index($0,"ACTIVATION_REGISTRY")==1{f=1} f&&$0=="}"{f=0} f' src/dl_techniques/layers/activations/factory.py \| grep -cE "^    ['\"][A-Za-z0-9_]+['\"]:"` |
| Keys in `FFN_REGISTRY` | 21 | `awk 'index($0,"FFN_REGISTRY")==1{f=1} f&&$0=="}"{f=0} f' src/dl_techniques/layers/ffn/factory.py \| grep -cE "^    ['\"][A-Za-z0-9_]+['\"]:"` |
| Keys in `ANYLOSS_REGISTRY` | 16 | `awk 'index($0,"ANYLOSS_REGISTRY")==1{f=1} f&&$0=="}"{f=0} f' src/dl_techniques/losses/any_loss.py \| grep -cE "^    ['\"][A-Za-z0-9_]+['\"]:"` |
| Keys in `EMBEDDING_REGISTRY` | 13 | `awk 'index($0,"EMBEDDING_REGISTRY")==1{f=1} f&&$0=="}"{f=0} f' src/dl_techniques/layers/embedding/factory.py \| grep -cE "^    ['\"][A-Za-z0-9_]+['\"]:"` |
| Keys in `LOGIC_REGISTRY` | 4 | `awk 'index($0,"LOGIC_REGISTRY")==1{f=1} f&&$0=="}"{f=0} f' src/dl_techniques/layers/logic/factory.py \| grep -cE "^    ['\"][A-Za-z0-9_]+['\"]:"` |
| Keys in `SAMPLING_REGISTRY` | 3 | `awk 'index($0,"SAMPLING_REGISTRY")==1{f=1} f&&$0=="}"{f=0} f' src/dl_techniques/layers/sampling.py \| grep -cE "^    ['\"][A-Za-z0-9_]+['\"]:"` |
| Keys in `MIXTURE_REGISTRY` | 3 | `awk 'index($0,"MIXTURE_REGISTRY")==1{f=1} f&&$0=="}"{f=0} f' src/dl_techniques/layers/mixtures/factory.py \| grep -cE "^    ['\"][A-Za-z0-9_]+['\"]:"` |
| Keys in `SEQUENCE_POOLING_REGISTRY` | 3 | `awk 'index($0,"SEQUENCE_POOLING_REGISTRY")==1{f=1} f&&$0=="}"{f=0} f' src/dl_techniques/layers/sequence_pooling/factory.py \| grep -cE "^    ['\"][A-Za-z0-9_]+['\"]:"` |
| Keys in `_TYPE_TO_CLASS` (norms factory) | 18 | `awk 'index($0,"_TYPE_TO_CLASS")==1{f=1} f&&$0=="}"{f=0} f' src/dl_techniques/layers/norms/factory.py \| grep -cE "^    ['\"][A-Za-z0-9_]+['\"]:"` |
| Trainer dirs under `src/train/` (excl. `src/train/common/`) | 45 | `find src/train -mindepth 1 -maxdepth 1 -type d ! -name __pycache__ ! -name common \| wc -l` |
| Test dirs under `tests/test_models/` | 79 | `find tests/test_models -mindepth 1 -maxdepth 1 -type d ! -name __pycache__ \| wc -l` |
| Model dirs nested in `src/dl_techniques/models/time_series/` | 7 | `find src/dl_techniques/models/time_series -mindepth 1 -maxdepth 1 -type d ! -name __pycache__ \| wc -l` |
| Files under `src/` naming `keras.callbacks.Callback` | 56 | `grep -rl "keras.callbacks.Callback" src --include=*.py \| wc -l` |
| …of those, inside `src/dl_techniques/callbacks/` | 11 | `grep -rl "keras.callbacks.Callback" src/dl_techniques/callbacks --include=*.py \| wc -l` |
| …of those, under `src/train/` | 38 | `grep -rl "keras.callbacks.Callback" src/train --include=*.py \| wc -l` |
| Files using `@keras.saving.register_keras_serializable` | 461 | `grep -rl "@keras.saving.register_keras_serializable" src/dl_techniques --include=*.py \| wc -l` |
| Files defining `get_config` | 464 | `grep -rl "def get_config" src/dl_techniques --include=*.py \| wc -l` |
| Files using the central logger | 354 | `grep -rl "utils.logger" src/dl_techniques --include=*.py \| wc -l` |
| Files importing raw `tensorflow` | 68 | `grep -rl "import tensorflow as tf" src/dl_techniques --include=*.py \| wc -l` |
| `.py` in `src/dl_techniques/layers/attention/` | 34 | `find src/dl_techniques/layers/attention -name '*.py' \| wc -l` |
| …of those using Sphinx `:param` docstrings | 33 | `grep -rl ":param " src/dl_techniques/layers/attention --include=*.py \| wc -l` |
| Library modules using Sphinx `:param` OUTSIDE `src/dl_techniques/layers/attention/` | 311 | `grep -rl ":param " src/dl_techniques --include=*.py \| grep -vc "src/dl_techniques/layers/attention"` |
| Library modules using a Google-style `Args:` block | 260 | `grep -rlE "^ +Args:$" src/dl_techniques --include=*.py \| wc -l` |
| Loose `test_*.py` directly under `tests/test_layers/` | 79 | `find tests/test_layers -maxdepth 1 -name 'test_*.py' \| wc -l` |
| Subdirectories under `tests/test_layers/` | 20 | `find tests/test_layers -mindepth 1 -maxdepth 1 -type d ! -name __pycache__ \| wc -l` |
| Model packages with no `create_` function ANYWHERE in the package | 16 | `for d in $(find src/dl_techniques/models -mindepth 1 -maxdepth 1 -type d ! -name __pycache__); do if ! grep -rq "^def create_" "$d" --include=*.py; then echo "$d"; fi; done \| wc -l` |
| Model packages naming a `create_` in their own package init (what a caller sees) | 23 | `for d in $(find src/dl_techniques/models -mindepth 1 -maxdepth 1 -type d ! -name __pycache__); do grep -q "create_" "$d/__init__.py" && echo "$d"; done \| wc -l` |
| Model packages with no same-named `src/train/` dir AND no `models.` import under `src/train/` | 27 | `t=$(find src/train -mindepth 1 -maxdepth 1 -type d -printf "%f\n"); for d in $(find src/dl_techniques/models -mindepth 1 -maxdepth 1 -type d ! -name __pycache__ -printf "%f\n"); do echo "$t" \| grep -qx "$d" \|\| grep -rq "models\.$d" src/train --include=*.py \|\| echo "$d"; done \| wc -l` |
| Lines in the mandatory authoring guide | 3105 | `wc -l < research/2026_keras_custom_models_instructions.md` |
| Test files under `tests/test_analysis/` | 0 | `find tests/test_analysis -name 'test_*.py' \| wc -l` |
| Test files under `tests/test_analyzer/` | 2 | `find tests/test_analyzer -name 'test_*.py' \| wc -l` |
