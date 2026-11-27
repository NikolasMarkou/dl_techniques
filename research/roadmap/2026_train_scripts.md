# Task: a LightlyTrain-style front door for `src/train/`

Repo: `dl_techniques` (Keras 3 / TF 2.18). Reference: `lightly-ai/lightly-train` (PyTorch / Lightning Fabric).
Goal: every trainer reachable through ONE task-oriented Python API and ONE `key=value` CLI, with typed configs, `"auto"`
defaults resolved per model, strict validation and a fixed run-directory contract. Existing trainers keep working.

This is an interface migration, not a rewrite. Do not change training math, data protocols, losses, schedules or output
file schemas of any existing trainer.

---

## 0. Read first (mandatory, before any edit)

In this repo:

1. `AGENTS.md` (root): no Claude attribution, `.venv`, never delete under `results/`, scoped pytest,
   `git push --no-verify`.
2. `REPO_MAP.md`: registry/factory surface, model/trainer/test triangle, entry points.
3. `src/train/AGENTS.md`: the eight trainer patterns, `train.common` API, `create_callbacks`, monitor-mode registry,
   optimizer traps.
4. `src/train/GUIDE.md`: defect patterns and their guards (failed run exits non-zero, history after model save, GPU rule
   D-110, `steps_per_epoch` on finite datasets).
5. `tests/test_train/_cli_contract.py` and `tests/test_train/test_config_fields_are_live.py`: the flag-to-field and
   dead-field guards. The new CLI must satisfy the same contracts.
6. `src/train/convnext/common.py`: the pilot. It already has `TrainingConfig` (dataclass), `config_from_args`,
   `train(config) -> Dict`, and thin wrappers `train_convnext_v1.py` / `train_convnext_v2.py`.

In LightlyTrain (clone to a scratch dir OUTSIDE the repo:
`git clone --depth 1 https://github.com/lightly-ai/lightly-train /tmp/lightly-train`):

| File                                                           | What to take from it                                                                                                                               |
|----------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| `src/lightly_train/__init__.py`                                | Flat public API: `train_<task>`, `load_model`, `export`, `list_models`                                                                             |
| `src/lightly_train/_cli.py`                                    | `<command> key=value` CLI, nested dot keys, per-command `help`, help text that prints live defaults                                                |
| `src/lightly_train/_commands/train_task.py`                    | Keyword-only `train_<task>(*, out, data, model, ...)`, `kwargs = locals()` into one config class, one `_train_task_from_config`                    |
| `src/lightly_train/_commands/train_task_helpers.py`            | `get_out_dir` (refuse non-empty unless resume), `pretty_format_args`, `load_checkpoint` (resume vs weights-only), checkpoint/export path producers |
| `src/lightly_train/_configs/config.py`, `_configs/validate.py` | Strict config base: `extra="forbid"`, `"auto"` sentinel, `has_auto()`, `no_auto()`, readable dotted error paths                                    |
| `src/lightly_train/_task_models/train_model.py`                | `TrainModelArgs` with `ClassVar` `default_batch_size` / `default_steps` and `resolve_auto(...)`: defaults owned by the model, not the CLI          |
| `src/lightly_train/_loggers/task_logger_args.py`               | `*_args` sub-config with its own `resolve_auto`                                                                                                    |
| `src/lightly_train/_models/package_helpers.py`                 | `"package/model"` string namespace, `list_model_names`                                                                                             |
| `src/lightly_train/_task_models/task_model_helpers.py`         | `load_model(path_or_name)` returning an object with `.predict(...)`                                                                                |

---

## 1. What to copy

1. **Task-named, keyword-only Python API.**
   `train.train_image_classification(out=..., data=..., model=..., epochs="auto", ...)`. One function per task, not per
   model.
2. **Model as a namespaced string.** `model="convnext_v1/cifar10"`, `"vit/small"`, `"cliffordnet/<variant>"`, or a path
   to a `.keras` file (fine-tune from weights).
3. **One config class per task**, built from the call's kwargs, validated once, logged once, written once.
4. **`"auto"` values resolved from model-owned defaults** (batch size, epochs, image size, lr), never from scattered CLI
   defaults.
5. **Grouped `*_args` dicts** instead of dozens of flat flags: `model_args`, `optim_args`, `schedule_args`,
   `transform_args`, `loader_args`, `logger_args`, `callback_args`, `analysis_args`.
6. **Strict validation.** Unknown key at any depth raises, naming the dotted path and the closest valid key
   (`difflib.get_close_matches`).
7. **CLI = same config.**
   `python -m train image_classification out=... model=convnext_v1/cifar10 data=cifar10 optim_args.lr=1e-3 logger_args.tensorboard=null`.
   The CLI builds the same dict and calls the same function.
8. **`resume_interrupted` vs `checkpoint`.** Resume = same `out`, full state, no parameter changes allowed (compare
   against saved config and refuse on mismatch). Checkpoint = weights only, new run.
9. **`load_model(path_or_run_dir).predict(x)`**, `list_models(task=None)`, `list_tasks()`, `export(...)`.
10. **Help generated from the config** (`python -m train help`, `python -m train image_classification help`), printing
    current defaults from the dataclass, so docs cannot drift.

## 2. What NOT to copy (house rules win)

| LightlyTrain behaviour                                                        | Rule here                                                                                                                                                                                                                                                                    |
|-------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `overwrite=True` reuses/overwrites a non-empty `out`                          | **Do not implement.** Keep `refuse_existing_run`. Nothing under `results/` is ever overwritten or deleted.                                                                                                                                                                   |
| Pydantic + OmegaConf                                                          | Neither. Stdlib `dataclasses` + a small validator; existing AST guards read dataclass fields. PyYAML is the optional `train` extra of `pyproject.toml`, read only for a `config=` or `data=` YAML file.                                                                       |
| Step-based loop, `steps`, `val_every_num_steps`                               | Keep Keras `fit` and epochs. `epochs` only, plus `data.steps_per_epoch` in `depth_estimation` and `time_series_forecasting`. Respect the "repeat the dataset" rule.                                                                                                      |
| `accelerator`, `devices`, `strategy`, `num_nodes`                             | One key, `devices`: an int GPU index, or `auto` or null for every visible GPU, through `train.common.setup_gpu` (D-110). No distributed scope.                                                                                                                            |
| `precision="bf16-mixed"` strings                                              | `precision`: `float32`, `mixed_float16` or `mixed_bfloat16`, through `setup_mixed_precision`; `LossScaleOptimizer` at the call site for fp16.                                                                                                                              |
| Default `seed=0`                                                              | `seed` defaults to `DEFAULT_SEED` from `train/common/seed.py`, applied via `set_seeds`.                                                                                                                                                                                      |
| Output files `checkpoints/last.ckpt`, `exported_models/*.pt`, `metrics.jsonl` | Keep the existing contract: `config.json`, `run.log`, `training_log.csv`, `training_history.json`, `best_model.keras`, `final_model.keras`, `results_summary.json`, `visualizations/`, `model_analysis/`. `compare_runs` and guards read it. Only ADD files (see section 5). |
| Telemetry (`tracker.track_training_started`)                                  | None.                                                                                                                                                                                                                                                                        |
| `out` required                                                                | `out` required too; `out=auto` names a new `results/<task>_<family>_<variant>_<timestamp>`; relative paths anchored at repo root via the existing resolver.                                                                                                                  |

---

## 3. Target UX (as shipped)

The front door is `train.api` and `python -m train`. The package `train` itself exports nothing: `src/train/__init__.py`
is empty by ruling (plan a865fa08 invariant 7, legacy behaviour does not change; kept by the user 2026-11-24), so Python
callers import from `train.api`.

Python:

```python
from train.api import train_image_classification, load_model, export, list_models

result = train_image_classification(         # keyword-only; the keys `image_classification help` prints
    out="results/convnext_c10_baseline",      # required; "auto" names a new directory under results/
    model="convnext_v1/cifar10",
    data={"dataset": "cifar10"},              # a mapping, a data config object, or the path of a YAML file
    epochs=100,
    optim_args={"lr": 1e-3},
    transform_args={"image_size": 32},
)

predictor = load_model("results/convnext_c10_baseline")   # a run directory, or a .keras file beside its task.json
answer = predictor.predict("cat.png", top_k=5)            # {"labels", "scores"}; each task has its own predictor
export(model=predictor.model, out="convnext_c10.keras", format="keras", record=predictor.record)
names = list_models(task="image_classification")        # sorted "family/variant" names, e.g. "vit/vit_pico"
```

CLI (every key is one the command's `help` prints):

```bash
MPLBACKEND=Agg .venv/bin/python -m train help
MPLBACKEND=Agg .venv/bin/python -m train image_classification help
MPLBACKEND=Agg .venv/bin/python -m train image_classification \
    out=results/convnext_c10_baseline model=convnext_v1/cifar10 data.dataset=cifar10 \
    epochs=100 optim_args.lr=1e-3 transform_args.image_size=32
MPLBACKEND=Agg .venv/bin/python -m train list_models task=image_classification
MPLBACKEND=Agg .venv/bin/python -m train predict model=results/convnext_c10_baseline data=images/ out=predictions.json
MPLBACKEND=Agg .venv/bin/python -m train export model=results/convnext_c10_baseline out=convnext_c10.keras format=keras
```

Legacy entry points (`python -m train.convnext.train_convnext_v1 --dataset cifar10 ...`) stay. The user guide is
`src/train/README.md`.

---

## 4. Architecture

Package `src/train/api/` (never a module named `train.py`), as shipped. Not shipped, as first sketched here: a lazy
`import train` with `train.load_model`, a `LoadedModel` with `.keras_model` and `.config`, the modules `api/load.py`,
`api/export.py`, `api/help.py`, `dotlist.py`, `common_args.py` and `api/run.py`, a `train_args.json`, an
`exported_models/` directory, `export run=results/... which=best`, and a per-task `run(config)` adapter onto each legacy
trainer's own `train(config)`; the shipped design is a `TaskTrainer` sequence with task hooks.

| Path | Role |
|---|---|
| `src/train/__init__.py` | Empty (0 bytes) by ruling, see Section 3. |
| `src/train/__main__.py` | `python -m train`: `sys.exit(main(sys.argv[1:]))` with `main` from `src/train/api/cli.py`. |
| `src/train/api/__init__.py` | The lazy surface (PEP 562, the `_LAZY` table): `import train.api` loads no submodule; `load_model`, `export`, `list_models`, `list_tasks`, `main`, each `train_<task>` and the config classes resolve on first read. |
| `src/train/api/cli.py` | The commands (`help`, `list_tasks`, `list_models`, `export`, `predict`, `convert`, then each task), the `key=value` parser, `config=` files, help rendered from the config dataclasses; exit 0, 2 for a refused request (`DLTrainError`), 1 for a failed or unmet run. |
| `src/train/api/config.py`, `src/train/api/errors.py` | `StrictConfig` (frozen dataclasses): unknown keys refused at every depth with the dotted path and a suggestion, `assert_resolved`, the JSON and YAML readers; the exception types. |
| `src/train/api/args.py` | The seven argument groups (`OptimArgs`, `TransformArgs`, `MetricArgs`, `CheckpointArgs`, `LoggerArgs`, `AnalysisArgs`, `DebugArgs`) and how `auto` becomes a value. |
| `src/train/api/registry.py`, `src/train/api/recipes.py` | `TaskSpec` and `ModelSpec` rows stored as `"module:attr"` strings, variants read from each family's `MODEL_VARIANTS`; `list_tasks`, `list_models`, `parse_model_name`; the image-classification families and their recipes. |
| `src/train/api/trainer.py`, `src/train/api/run_sequence.py` | `TaskTrainer`: the one sequence every task runs; a task supplies hooks (`REQUIRED_HOOKS`). |
| `src/train/api/run_dir.py`, `src/train/api/resume_decisions.py` | `resolve_out`, `open_run` (claims `out`, never overwrites), `run.lock`, the run files of Section 5; what a resume may do. |
| `src/train/api/fit_callbacks.py`, `src/train/api/optim.py` | The callback list in the order resume needs; `OptimArgs` to an optimizer through `optimizer_builder`. |
| `src/train/api/tasks/` | One module per task: its config, its `TaskTrainer` subclass and the keyword-only `train_<task>(...)`. |
| `api/<task>_args.py`, `_models.py`, `_predictor.py` | Per task (denoising, depth, forecast, pretrain, segmentation; classification uses `data_args.py`, `recipes.py` and `predictor.py`): the data and argument groups, the families with their refusals and recipes, the predictor and export rules. |
| `src/train/api/predictor.py` | `load_model(path)`: a run directory or a `.keras` file beside its `task.json`, returning the task's predictor (`PREDICTORS`) with `.model`, `.record` and `.predict(x, ...)`; the schema-v2 `task.json` and the per-task tables (`TASK_TABLES`). |
| `src/train/api/exporting.py` | `export(*, model, out, format, input_shape=None, record=None)`: writes `out`, which must not exist, as `keras`, `saved_model`, `tflite` or `onnx` (onnx needs `tf2onnx` importable), and `<out>.task.json` beside it when `record` is given. |
| `src/train/api/init_from.py` | `init`: start a run from an earlier run's weights (Phase 4g). |
| `src/train/api/convert.py` | `python -m train convert`: ADE20K and COCO as index masks and listings. |

Config fields every task has (from `python -m train <task> help` of all ten; a task adds its own, such as `loss_args`
or `jit_compile`):

| Field | Type | Default |
|---|---|---|
| `out` | `str` | required; `auto` names a new directory under `results/` |
| `model` | `str` | required, `family/variant` (`python -m train list_models`) |
| `data` | a mapping, a data config, or the path of a YAML file | required; its keys are the task's |
| `epochs`, `batch_size` | `int` | `auto` (the family's recipe) |
| `seed` | `int` | `42` |
| `devices` | `int` or null | `auto` |
| `precision` | `float32`, `mixed_float16` or `mixed_bfloat16` | `float32` |
| `resume_interrupted` | `bool` | `False` |
| `init` | `str` or null | `None` |
| `model_args` | a mapping | `{}` |
| `optim_args`, `transform_args`, `metric_args`, `checkpoint_args`, `logger_args`, `analysis_args`, `debug_args` | the seven groups of `src/train/api/args.py` (a task may give its own transform, metric or optimizer group) | per field, printed by `<task> help` |

Rules for a task: written in shipped terms, each with the test that enforces it, in one place: `src/train/AGENTS.md`,
section "Front door: `train.api`" (Phase 6). This roadmap does not keep a second copy.

## 5. Run directory contract (additions only)

Existing files are untouched in name and schema. As shipped (`src/train/api/run_dir.py`), a `python -m train` run adds:

- `config_requested.json`: the request as the caller gave it, `auto` values kept. It is created empty with `O_EXCL` to
  claim `out`, then filled. `config.json` holds the config after `auto` resolution, plus what the run was built with
  beyond its fields under the key `used`.
- `task.json`: the schema-v2 record a model needs to be used without its config (`src/train/api/predictor.py`).
- `run.lock`: the run lock, held for the life of the run.
- `run.log` gains the blocks `Args:`, `Resolved args:` and `Used args:` (`src/train/api/run_sequence.py`).

`export` has no default place in a run: it writes its `out`, which must not exist, and `<out>.task.json` beside it.

`resume_interrupted=True`: needs `checkpoints/backup` under the same `out`; compares this request with the run's
`config_requested.json` and refuses, naming the keys, any difference outside `RESUME_UNCOMPARED_KEYS`
(`resume_interrupted`, `out`, `devices`, `analysis_args.eval_batch_size`); a finished run (with `results_summary.json`)
is refused. Every task takes the key; none refuses resume as unsupported.

---

## 6. Phases (stop and report after each)

Create the plan under `plans/plan-<timestamp>-<hash>/` following the existing convention (`progress.md`, `decisions.md`,
`changelog.md`). Record every non-obvious choice as a `D-NNN` decision.

**Phase 0: inventory (no code changes).** Produce `plans/.../inventory.md`: one row per entry point under `src/train/`
(about 118): module path, pattern (1 to 8 or "none"), has a config dataclass, has a `train(config)` function separable
from argparse, model package, `MODEL_VARIANTS` present, smoke preset present, resume support, proposed front-door task
name. Propose the task list (expected: `image_classification`, `semantic_segmentation`, `denoising`, `depth_estimation`,
`object_detection`, `language_model_pretraining`, `text_classification`, `time_series_forecasting`, plus "not migrated"
with reason). Wait for approval.

**Phase 1: infrastructure only.** The strict config, the `key=value` parser, the registry (empty task table), the shared
argument groups, the run preamble, the help and `__main__.py` (shipped under other names and with `src/train/__init__.py`
left empty: see Section 4). Tests in `tests/test_train/test_api/`. No trainer touched.

**Phase 2: pilot `image_classification` on ConvNeXt V1/V2.** Adapter builds `train.convnext.common.TrainingConfig` and
calls `train.convnext.common.train`. Registry keys from `ConvNeXtV1.MODEL_VARIANTS` / `ConvNeXtV2.MODEL_VARIANTS`.
Parity test: legacy argv and the equivalent API call produce field-for-field identical `TrainingConfig`.
(As shipped: no adapter; `TaskTrainer` runs the legacy ConvNeXt sequence cut at its seams; parity: Section 7, item 8.)

**Phase 3: rest of Pattern 1.** One trainer per commit: cliffordnet, vit, resnet, alexnet, mobilenet, ... For trainers
whose body takes an argparse `Namespace` (e.g. `cliffordnet.train_model(args)`), first split into `config_from_args` +
`train(config)` with no behaviour change, proven by the existing tests, then add the adapter. (As shipped: families of
`image_classification`, not adapted trainers, eight by `list_models task=image_classification`, no alexnet or mobilenet.)

**Phase 4: other tasks, one pattern at a time** in the order approved in Phase 0. Exemplars per `src/train/AGENTS.md`.
(As shipped, with no Phase 0 inventory, Section 9: `image_segmentation`, `image_denoising`, `pretrain`,
`time_series_forecasting`, `depth_estimation`, `language_model_pretraining` and `text_classification` (Phase 4i), `beit_tokenizer` (Phase 4j) and `dino` (Phase 4k), in that order; no
detection task.)

**Phase 4g: start from trained weights. Code done, GPU gate run** (branch `train-api/phase-4g`; the gate's numbers are
in `src/train/README.md`, "Measured: the `init` gate": every claim held except the overlap arm's planned exit 0; it
exited 1, the bar under `init` refusing a fine-tune that scored below its loaded weights).
One top-level key `init`, on every task's config, names an earlier `python -m train` run directory
and replaced `checkpoint` outright (no alias): the new run builds its own model, then takes the source's weights before
`compile`. Every weight is copied, resampled by a rule its family declares (one rule: the ViT position table, bicubic, to a
larger grid), or left fresh as a declared head or a new block of a deeper variant; anything else is refused by name before
any weight is copied. Covered: a new image size or lookback where the family keeps its weights (depth, the spatial families,
ViT through the rule; tirex, xlstm, ets, timestral), a new head (classes, horizon), a deeper variant (vit, resnet,
resnet_seg), and four cross-task pairs (resnet and convnext_v1 classifiers into their segmenters, segmentation ConvUNext to
and from biased denoising). Pretrain is refused as source and target until it writes a record (lifted by 4h). The run records the source,
its sha256, the pins, the transfer counts and the split relation in `task.json` and `results_summary.json`; a source whose
training data reaches this run's selection or test data is accepted, counted and marks the verdict as biased; under `init`
the classification, segmentation and denoising bars also need a strict improvement over the loaded weights' epoch-zero
score, and the forecasting and depth bars' epoch-zero term is the loaded weights. A resume never reads the source. It
supersedes item 8 of Section 1 and the `checkpoint` row of Section 4. Design, decisions and what changed on the way:
`research/roadmap/20261117_train_api_init_from.md`; user-facing rules: `src/train/README.md`, "Starting from an earlier run".

**Phase 4h: the pretrain record. Code done, GPU gate run** (branch `train-api/phase-4h`, plan 5c6cc105; the gate's numbers
are in `src/train/README.md`, "Measured: the Phase 4h gate": every claim held, and two predictions missed, reported with
their numbers: the `energy_transformer` classifier from its MIM source scored below its scratch control, and the `et_mim`
run from its source did not improve on it, the bar under `init` failing it). Straight after 4g and before the language tasks (user,
2026-11-17). The pretrain rows are renamed `et_mim` and `beit_mim`, with no alias; a pretrain run writes a schema-v2 `task.json`
(the sixth record kind), and `predict` and `export` give the backbone's patch-token embeddings (`et_mim` tflite refused:
TensorFlow 2.18's converter mistranslates the Hopfield update). image_classification gains `beit` and `energy_transformer`, the
library classifiers, which a pretrain run of the matching family seeds through `init` (six declared cross-task pairs now): the
BEiT relative position bias is resampled by unilm's rule, checked against unilm's own code run verbatim, and the Energy
Transformer table, which no reference resizes, is refused at another grid. Pretrain is an `init` target too, from its own family
only. What changed against the 4g design: `research/roadmap/20261117_train_api_init_from.md`, Section 0.

**Phase 4i: the language tasks. Both tasks' code done, GPU gate run** (branch `train-api/phase-4i`, plan 7806a290; two tasks in one plan, the first
finished, and kept as the local branch `train-api/phase-4i-lm`, before the second started). `language_model_pretraining` is built and tested on a CPU:
the seventh task, one family `gpt2_lm` (a functional model over the library GPT-2, trained by the stock `fit`), a text stream that is a
function of `(seed, step)` so that a resume continues it, a bar strictly below `min(unigram, epoch zero)`, the seventh record kind with
`predict` (per-document `mean_nll` and the next tokens) and `export` (float32 token ids), and `init` from its own family. Its tokenizer is
pinned outside the repository because tiktoken's own cache is in `/tmp`. `text_classification` is built and tested on a CPU too: the eighth task,
two families, `bert` and `gpt2` (functional classifiers over the library encoder and decoder with the sentiment head, the padding mask read from the
pad id), the language task's stream, split and tokenizer, a bar strictly above the majority share of the selection split plus three binomial standard errors of it (so
the gate reports the accuracies beside it), the eighth record kind with `predict` (`labels` and `scores`) and `export`, and `init` from its own family and,
for `gpt2`, from a `gpt2_lm` run (`bert` has no cross-task source). The GPU gate has run on the final code, after a first night whose findings (holes in
tests, a wrongly defined claim) were fixed; its numbers are in `src/train/README.md`, "Measured: the language gate". User-facing rules: `src/train/README.md`, the two task sections.

**Phase 4j: the BEiT tokenizer. Code done, GPU gate run** (branch `train-api/phase-4j`, plan fa94f59c; one task). `beit_tokenizer` is the ninth task, one family
`vq_vae_rotation` (the library `VQVAERotationTrick` itself, with its own `train_step`: the task adds none): a bar of two parts (the reconstruction strictly below the constant
image and the untrained model, and a codebook perplexity over every selection position of at least 64 pooled and at least 8 per cell), the ninth record kind with `predict` (the int32 code grid) and `export`
(all four formats), and `init` from its own family. The auto decoder's last layer is linear now (the sigmoid could not reach a standardized image). `beit_mim` takes the `best_model.keras`
of a tokenizer run and holds its `task.json` to its own patch, size and statistics before the claim. The tokenizer of the 4h `beit_mim` rows is lost and its figures are not
reproduced. The GPU gate has run on the final code (its numbers are in `src/train/README.md`, "Measured: the tokenizer gate"). User-facing rules: `src/train/README.md`, the `beit_tokenizer` section.

**Phase 4k: DINO. Code done, GPU gate not yet run** (branch `train-api/phase-4k`, plan e3381378; one task). `dino` is the tenth task, one family `dino_v1`
(the library `DINOTrainingModel` on stock `fit`, with no validation data and no `train_step`): a bar of two parts (the best k-NN accuracy above the larger of the constant-feature and the untrained model's plus a margin, and a clear collapse flag and a centred teacher entropy at every epoch), the tenth record kind with `predict` (a float32 CLS embedding per image) and `export` (all four formats), and `init`: a `dino` run seeds a `vit` classifier by its student (the eighth declared cross-task pair) and is refused as a target. The recipe is the improved arm of the measurement record with its EMA warmup counted in epochs; `cifar10` and the `small` variant are offered and not gate-measured. User-facing rules: `src/train/README.md`, the `dino` section (its gate block holds no measurement until the night has run).

**Phase 5: `load_model`, `export`, `list_models`, `list_tasks`. Done** (plan 98c9aac5, branch `train-api/phase-5-6`):
built inside Phases 1 to 4h (`api/predictor.py`, `api/exporting.py`, `api/registry.py`, `api/cli.py`), on `train.api`
only, with one fix: `train.api.list_tasks` and `list_models` now register the built-in tasks and families themselves, so
they answer in a fresh process (they returned empty until `main(...)` ran; plan 98c9aac5 D-013,
`tests/test_train/test_api/test_cli_help_allocates_nothing.py`); `src/train/__init__.py` stays empty (plan a865fa08
invariant 7, kept by the user 2026-11-24); Sections 3, 4, 7 and 9 now describe the shipped surface.

**Phase 6: docs. Done** (plan 98c9aac5, branch `train-api/phase-5-6`): Quick Start item 7 of the root `README.md`;
the section "Front door: `train.api` (`python -m train`)" of `src/train/AGENTS.md`, opening with the adapter rules as
shipped, each with its code home and test; a **Front door.** paragraph in § Entry points of `REPO_MAP.md`; a pointer in
the root `AGENTS.md`. Guards: `tests/test_the_guides_cite_only_paths_that_exist.py` checks the paths the two `GUIDE.md` cite;
`tests/test_train/test_api/test_the_forecasting_docs_state_the_numbers_of_the_code.py` checks the cited paths of
`src/train/README.md` and `src/train/AGENTS.md`; `tests/test_train/test_api/test_the_readme_commands_run_to_the_claim.py`
reads the root `README.md` and `AGENTS.md` only for `init=` and forecasting commands, which item 7 has none of (each of its
commands was run on 2026-11-24, plan 98c9aac5, the training line up to the claim; no test reads item 7); `REPO_MAP.md` is read only for its TopoLM numbers and sentence
(`tests/test_models/test_topolm/test_topolm_docs_agree.py`); no test reads this roadmap.

---

## 7. Required tests (`tests/test_train/test_api/`)

Each item as written first, then what covers it now (every file named is under `tests/test_train/test_api/` unless its
path says otherwise).

1. `dotlist`: nesting, types, `null`, lists, quoted strings, duplicate key error, malformed token error.
   Covered: `tests/test_train/test_api/test_dotlist_parser.py` (the parser lives in `src/train/api/cli.py`).
2. Strict validation: unknown top-level and nested keys raise with dotted path and suggestion; wrong type raises.
   Covered: `tests/test_train/test_api/test_strict_config.py`.
3. `"auto"`: every auto field resolved before the runner is called (`assert_resolved`), from the model's
   `ModelDefaults`. Covered: `tests/test_train/test_api/test_auto_resolution.py`; the defaults are the family's recipe
   on its `ModelSpec` (no `ModelDefaults` class shipped).
4. CLI contract, same three traps as `_cli_contract.py`: every key drives a NON-default value, the row table equals the
   config's field set (completeness), values distinct across rows. Assert on the resulting trainer config object, never
   on exit codes or help text. Covered: `tests/test_train/test_api/test_every_cli_key_reaches_the_config.py` (on the
   task config) and `tests/test_train/test_api/test_function_and_cli_agree.py` (the same `config.json` bytes either way).
5. Dead fields: extend `test_config_fields_are_live.py` coverage to front-door configs (a serialized field is not a
   consumed field). Covered: `tests/test_train/test_api/test_the_api_fields_are_read_in_production.py` (a field must
   be read by production code outside its own class); `tests/test_train/test_config_fields_are_live.py` lists the
   `api/args.py` groups too.
6. Import hygiene: `python -m train help` and `import train` in a subprocess leave `tensorflow` and `keras` out of
   `sys.modules` and set no GPU state. Covered for `python -m train help`, every command's and task's help, and
   `import train.api`: `tests/test_train/test_api/test_cli_help_allocates_nothing.py`. Not built for `import train`:
   it imports an empty package. GPU state has no test of its own beyond TensorFlow not being imported.
7. Registry: `MODELS` keys equal the union of the wired families' `MODEL_VARIANTS` keys; unknown model string raises
   with suggestions. Covered: `tests/test_train/test_api/test_registry.py` (`MODELS` is keyed by family, its variants
   read from `MODEL_VARIANTS` through `ModelSpec.variants_attr`; an unknown family or variant gets a suggestion).
8. Parity per migrated trainer (legacy argv vs API kwargs gives identical trainer config). Not built as written: no task
   builds a legacy trainer's config. Parity is of recipes and builds against the legacy trainers instead:
   `tests/test_train/test_api/test_every_family_builds_from_its_defaults.py` and the `legacy` tests of
   `test_denoising_families.py`, `test_image_segmentation_task.py`, `test_depth_task.py`, `test_pretrain_families.py`
   and `test_forecast_families.py`.
9. Failure: a stubbed raise inside the runner gives non-zero exit through the CLI and a raised exception through the
   API. Covered: `tests/test_train/test_api/test_a_failed_api_run_exits_nonzero.py`,
   `tests/test_train/test_api/test_the_cli_exit_codes.py`.
10. Output location: all writes under `out` (`tmp_path`, absolute). Never write under `results/` in tests. Covered:
   `tests/test_train/test_api/test_existing_run_is_refused.py` (`test_a_new_run_writes_only_inside_its_run_directory`)
   and `tests/test_train/test_api/test_function_and_cli_agree.py` (`test_a_run_adds_no_entry_to_the_repo_results_directory`).
11. Resume: mismatched parameters refused; unsupported trainer refused. Covered: `tests/test_train/test_api/test_resume_interrupted.py`
   (`test_a_changed_request_is_refused_naming_the_key`), `tests/test_train/test_api/test_resume_hardening.py` and the
   depth, forecast, pretrain and denoising resume files. Not built: an unsupported-trainer refusal (every task resumes).
12. One CPU smoke per task (tiny `max_samples`, 1 epoch), marked `slow`. Covered by a short run per task in
   `test_image_classification_end_to_end.py`, `test_image_segmentation_task.py`, `test_image_denoising_task.py`,
   `test_depth_task.py`, `test_forecast_task.py` and `test_pretrain_task.py`; none is marked `slow` (no file under
   `tests/test_train/test_api/` uses the marker, and `pyproject.toml` says the marker deselects nothing).

Run only the test files of what changed, one file or node id per command: never the full suite unless asked (root
`AGENTS.md`, Quick Reference), nor the whole `tests/test_train/test_api/` (the user's rule, plan 98c9aac5 D-009).

---

## 8. Non-negotiables (repeat of repo rules)

- No `Co-Authored-By: Claude`, no "Generated with Claude Code", no Claude/Anthropic attribution anywhere.
- Never delete anything under `results/`. Tests write to `tmp_path` only.
- `.venv/bin/python`, `MPLBACKEND=Agg`, modules run with `-m`.
- A failed run never exits 0. Broad `except` only around optional artifacts and listed in the census test.
- `save_training_history_json` after the model save, never inside `try/except`.
- `train/common/gpu.py` remains the only writer of `CUDA_VISIBLE_DEVICES`.
- Logging through `dl_techniques.utils.logger`, no `print` (CLI help output excepted, written to stdout explicitly).
- Match the docstring style of the file being edited; new files use Sphinx/reST like `models/language/bert/model.py`.
- A path or filename known in two places is a function.
- Commit per trainer/step; `git push --no-verify`.

## 9. Definition of done

What shipped, bullet by bullet:

- `python -m train help` lists tasks; `python -m train <task> help` lists every field with type and live default. Shipped.
- Every migrated trainer is reachable by API and CLI, its legacy entry point still passes its existing tests, and parity
  tests pass. Shipped by task and family, not by trainer: the tasks `python -m train list_tasks` prints, each with the
  families `list_models task=<task>` prints; the legacy entry points stay; parity is Section 7, item 8.
- Unknown keys, unsupported fields and failed runs fail loudly. Shipped: Section 7, items 2, 5 and 9.
- The request and the resolved config are recorded: `config_requested.json`, and `config.json` with `used` (Section 5),
  and the run log's `Args:`, `Resolved args:` and `Used args:` blocks.
- `load_model(run_dir).predict(x)` works for every task: `load_model` picks the task's predictor from
  `predictor.PREDICTORS`, which has one row per task.
- Inventory marks every entry point as migrated or "not migrated: <reason>". Not done: no Phase 0 inventory exists under
  `plans/` (searched 2026-11-24); the task list grew phase by phase instead.

## 10. Decisions to confirm with the user before Phase 1

Ask once, with these defaults (each now answered by what shipped):

1. Package location `src/train/api/` with CLI `python -m train` (default), vs. moving the front door into the shipped
   `dl_techniques` package (would need `train.common` relocated). Shipped: `src/train/api/` and `python -m train`;
   `train/` is not part of the redistributable wheel (`pyproject.toml`).
2. No pydantic/omegaconf/pyyaml (default); data configs as dict or JSON. YAML accepted only if `yaml` is importable.
   Shipped: no pydantic or omegaconf; PyYAML is the optional `train` extra, read only for a `config=` or `data=` YAML
   file, and a missing PyYAML is a `ConfigError` naming the extra.
3. `out` optional with timestamped default (default) vs. required as in LightlyTrain. Shipped: required; `out=auto`
   names a new timestamped directory under `results/`.
4. Task list and migration order from the Phase 0 inventory. Shipped without the inventory: the tasks were chosen
   phase by phase (Section 6).
5. Whether `export` should target ONNX (only if a converter is already a dependency). Shipped: `format=onnx`, which needs
   `tf2onnx` importable (it is not a dependency in `pyproject.toml`); otherwise a `ConfigError`.