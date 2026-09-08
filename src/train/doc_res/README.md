# `train/doc_res` — DocRes staging, training and inference

DocRes conditions one Restormer backbone on five document-restoration tasks by concatenating
three classical-CV "prompt" channels onto the RGB input. This directory holds the whole
non-model half of that port:

| Module | What it does |
|---|---|
| `prepare_doc_res_data.py` | Downloads, verifies, lays out and prompt-precomputes the corpora. |
| `common.py` | The config, the CLI flags, the path worklist, the tf.data pipeline, the table-driven loss, the optimizer and `fit()`. |
| `train_doc_res.py` | The training entry point. |
| `infer_doc_res.py` | The inference entry point: pad, predict, post-process, write. |

**Binarization is the only task with a staged training corpus today** — see
[What is staged, and what is not](#what-is-staged-and-what-is-not) for the reason each of the
other four is empty.

```bash
# train (one task per run)
MPLBACKEND=Agg .venv/bin/python -m train.doc_res.train_doc_res \
    --task binarization --epochs 2 --steps-per-epoch 50 --gpu 1

# restore one page, or a directory of them
MPLBACKEND=Agg .venv/bin/python -m train.doc_res.infer_doc_res \
    --task binarization --input page.png --output-dir results/docres_infer \
    --checkpoint results/doc_res_binarization_<stamp>/best_model.keras --gpu 1
```

`infer_doc_res.py` is where the padding lives. `DocRes` refuses an input whose height or width
is not a multiple of 8 rather than padding silently, so the shim pads the top and left
(upstream's `stride_integral`) and crops exactly that back off the prediction — an output page
has the input page's size to the pixel. Per-task post-processing is read off the `TASKS` table
(`dl_techniques/datasets/document_restoration/tasks.py`); no task string is compared anywhere
outside it. **Without `--checkpoint` the model is freshly initialised and the run says so at
WARNING level** — the output is noise, not a restoration.

## Staging

```bash
# print every URL and the byte budget; write nothing
MPLBACKEND=Agg .venv/bin/python -m train.doc_res.prepare_doc_res_data --dry-run

# stage everything that is fetchable, then precompute the DTSPrompt sidecars
MPLBACKEND=Agg .venv/bin/python -m train.doc_res.prepare_doc_res_data

# one task, or one dataset
... --tasks binarization
... --datasets dibco2019,noisy_office
```

Layout — matching both the sibling convention under `/media/arxwn/data0_4tb/datasets/` and
DocRes's own manifest-relative `data/train/<task>/<dataset>/` shape:

```
/media/arxwn/data0_4tb/datasets/doc_res/<task>/<dataset>/
    _archives/<file>            the downloaded archive and its .ok marker
    _prompts/<mirror>/<stem>.png the precomputed DTSPrompt sidecar
    <whatever the archive contained>
    README.md                   only for a slot the script cannot fill
```

The script is **additive only**: it deletes nothing, anywhere, and never writes outside
`<root>/doc_res/`. It refuses rather than filling the volume (`--min-free-gb`, default 100),
verifies every URL before downloading, reports a dead one *by name*, and is idempotent — a
second run makes zero HTTP requests.

## What is staged, and what is not

Measured 2026-09-08. Byte figures are the real `du -sh` after staging, prompt sidecars
included.

| Task | Dataset | Status | On disk | Input pages |
|---|---|---|---|---|
| binarization | DIBCO 2009 | staged | 31 M | 10 |
| binarization | H-DIBCO 2010 | staged | 18 M | 10 |
| binarization | DIBCO 2011 | staged | 50 M | 16 |
| binarization | H-DIBCO 2012 | staged | 66 M | 14 |
| binarization | DIBCO 2013 | staged | 138 M | 16 |
| binarization | H-DIBCO 2014 | staged | 36 M | 10 |
| binarization | H-DIBCO 2015 | **DEAD URL** | — | 0 |
| binarization | H-DIBCO 2016 | staged | 95 M | 10 |
| binarization | DIBCO 2017 | staged | 259 M | 20 |
| binarization | H-DIBCO 2018 | staged | 391 M | 10 |
| binarization | DIBCO 2019 | staged | 323 M | 20 |
| binarization | NoisyOffice | staged | 1.4 G | 361 |
| deshadowing | RDD | manual (Drive) | — | 0 |
| dewarping | DIR300 (**eval only**) | manual (Drive) | — | 0 |
| dewarping | Doc3D | **registration-gated** | — | 0 |
| deblurring | TDD | **dead host** | — | 0 |

**497 input pages, 2.8 GB total. Binarization is the only trainable task today.**

### There is no pooled DIBCO download

Each competition year lives on its own university Apache listing — Demokritos, Duth's
`utopia`, and Duth's `vc.ee`. Two candidate aggregator repos were checked and neither
republishes the data. The script therefore carries eleven separate entries. Formats vary by
year: six ship `.rar`, one ships `.7z`, the rest `.zip`.

**H-DIBCO 2015 is dead.** Five candidate hosts were probed and all five 404. It stays in the
manifest on purpose, so every run names it rather than quietly showing a ten-row table.

### Doc3D — the wired, empty dewarping slot

Doc3D is the only corpus that can train dewarping, and it is **not downloadable by script**.
Its HuggingFace home (`StonyBrook-CVLab/doc3D-dataset`) requires accepting a contact-info
agreement before the file tree is even visible, and the legacy GitHub path required a Google
Form for credentials. The full release is **549 GB** — 100k renders plus backward-map, UV,
normal, albedo and mesh components. Even granted, a full pull would dominate the shared
volume; the minimum for a supervised unwarping loss is `img` + `bm`.

`prepare_doc_res_data.py` never attempts it. It creates
`.../doc_res/dewarping/doc3d/` with a README stating the gate and the size. Dropping the
extracted corpus in that directory is the only step needed; the manifest entry, the task
binding and the trainer's startup check already point at it.

Asking to *train* dewarping without it fails at startup, loudly, naming the gate:

```python
from train.doc_res.prepare_doc_res_data import require_task_data
require_task_data(Path("/media/arxwn/data0_4tb/datasets"), "dewarping")
# MissingTrainingDataError: ... doc3d: [gated] registration-gated ... 549 GB ...
#   dir300 is staged/available for this task but is EVAL ONLY ...
```

### TDD — deblurring has no live source

The official host is **dead**, not missing a file: `fit.vutbr.cz` redirects to `fit.vut.cz`
and that 404s too. The only remaining lead is an unverified OneDrive link in a third-party
recommendations README. The error message says so, rather than "file not found".

### RDD and DIR300 — a browser, not a script

Both are Google Drive **folders** of loose files (RDD: `train/`+`test/`, ~4,916 images;
DIR300: `dist/`+`gt/`, 300 each). A Drive folder's HTML listing is paginated behind an
authenticated internal API — scraping the DIR300 `dist/` page yields ~51 file ids for a
folder of 300. `gdown` scrapes the same HTML and hits the same cap. A scraper would therefore
stage a *silently truncated* corpus, so the script refuses and prints the browser URL instead.
Unpack a manual download straight into the slot directory and re-run with `--only-sidecars`.

## Prompt sidecars

After staging, the script writes one DTSPrompt sidecar per **input** page (ground-truth pages
are skipped) under `_prompts/`, mirroring the input's relative path. The tf.data pipeline then
reads two files per sample and never runs Python per batch — the same design upstream uses for
binarization.

The binarization prompt is `[Sauvola threshold, Sobel gradient, Sauvola binary]` and is cheap:
**497 sidecars in 514.6 s** (~1.0 s/page). The deshadowing and appearance prompts go through
`estimate_background`, whose `median_filter(21)` at the 1024x1024 working resolution measured
~8 s/call in step 9 — budget roughly 10x per page for those tasks when their corpora arrive.

Dewarping sidecars are **not** precomputed: that prompt needs a per-page document mask, which
upstream gets from an MBD segmentation network this port does not include.

## Archive extraction

`.zip` uses stdlib `zipfile`. `.rar` and `.7z` go through `libarchive-c`, a ctypes binding to
the system `libarchive.so.13` — none of `unrar`, `unar`, `bsdtar` or `7z` is installed on this
machine. It is **not** a declared dependency; without it the script reports a named
`NoExtractorError` and keeps the downloaded bytes, and the `.zip` years still work. Install it
with `pip install libarchive-c` if the RAR years matter to you.

Completeness is proven by **decoding every member** of the archive and then atomically
renaming a `.part` file into place, with the `.ok` marker written last. A `Content-Length`
check would not do: UCI serves `noisyoffice.zip` with no length and no range support.
