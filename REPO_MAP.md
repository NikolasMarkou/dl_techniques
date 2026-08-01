# REPO_MAP — a verified router for `dl_techniques`

**What this file is.** A router and a verified skeleton. It tells you *where code of a
given kind lives*, *how the pieces are wired together*, and *which of the repo's many
in-tree docs answers your question*. Every filesystem path printed here resolved when
this file was written; every number printed here lives in exactly one table together
with the shell command that re-derives it.

**What this file deliberately is NOT.** It is not a replacement for the 20 in-tree
`CLAUDE.md` files — those are the authority on their own subtree, and this map points at
them rather than restating them. It is not an API reference: it will not tell you what any
individual layer, loss or model does, only which document will. Restating subtree content
here is exactly how the repo's older maps drifted into describing code that no longer
exists, so the omission is the design, not an oversight.

Parts B (the navigation spine) and C (conventions, tests, doc routing, and the stale-doc
ledger) follow below.

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
| **`src/dl_techniques/models/`** | 270 | **The second largest.** 73 self-contained model packages in roughly eight architecture families. Most expose a `create_*` factory; a minority do not. |
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

## Numbers, and how to re-derive them

This is the only place in this document where a number is *stated*. Anything numeric in
the prose above is a Value from this table. Run these from the repo root.

| Quantity | Value | Command |
|---|---|---|
| Python files under `src/` | 1002 | `find src -name '*.py' \| wc -l` |
| Python files under `tests/` | 692 | `find tests -name '*.py' \| wc -l` |
| Python lines under `src/` | 413693 | `find src -name '*.py' -exec cat {} + \| wc -l` |
| Python lines under `tests/` | 236612 | `find tests -name '*.py' -exec cat {} + \| wc -l` |
| Commits at HEAD | 4216 | `git log --oneline \| wc -l` |
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
