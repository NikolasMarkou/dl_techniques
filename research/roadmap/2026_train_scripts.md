# Task: a LightlyTrain-style front door for `src/train/`

Repo: `dl_techniques` (Keras 3 / TF 2.18). Reference: `lightly-ai/lightly-train` (PyTorch / Lightning Fabric).
Goal: every trainer reachable through ONE task-oriented Python API and ONE `key=value` CLI, with typed configs, `"auto"` defaults resolved per model, strict validation and a fixed run-directory contract. Existing trainers keep working.

This is an interface migration, not a rewrite. Do not change training math, data protocols, losses, schedules or output file schemas of any existing trainer.

---

## 0. Read first (mandatory, before any edit)

In this repo:

1. `AGENTS.md` (root): no Claude attribution, `.venv`, never delete under `results/`, scoped pytest, `git push --no-verify`.
2. `REPO_MAP.md`: registry/factory surface, model/trainer/test triangle, entry points.
3. `src/train/AGENTS.md`: the eight trainer patterns, `train.common` API, `create_callbacks`, monitor-mode registry, optimizer traps.
4. `src/train/GUIDE.md`: defect patterns and their guards (failed run exits non-zero, history after model save, GPU rule D-110, `steps_per_epoch` on finite datasets).
5. `tests/test_train/_cli_contract.py` and `tests/test_train/test_config_fields_are_live.py`: the flag-to-field and dead-field guards. The new CLI must satisfy the same contracts.
6. `src/train/convnext/common.py`: the pilot. It already has `TrainingConfig` (dataclass), `config_from_args`, `train(config) -> Dict`, and thin wrappers `train_convnext_v1.py` / `train_convnext_v2.py`.

In LightlyTrain (clone to a scratch dir OUTSIDE the repo: `git clone --depth 1 https://github.com/lightly-ai/lightly-train /tmp/lightly-train`):

| File | What to take from it |
|---|---|
| `src/lightly_train/__init__.py` | Flat public API: `train_<task>`, `load_model`, `export`, `list_models` |
| `src/lightly_train/_cli.py` | `<command> key=value` CLI, nested dot keys, per-command `help`, help text that prints live defaults |
| `src/lightly_train/_commands/train_task.py` | Keyword-only `train_<task>(*, out, data, model, ...)`, `kwargs = locals()` into one config class, one `_train_task_from_config` |
| `src/lightly_train/_commands/train_task_helpers.py` | `get_out_dir` (refuse non-empty unless resume), `pretty_format_args`, `load_checkpoint` (resume vs weights-only), checkpoint/export path producers |
| `src/lightly_train/_configs/config.py`, `_configs/validate.py` | Strict config base: `extra="forbid"`, `"auto"` sentinel, `has_auto()`, `no_auto()`, readable dotted error paths |
| `src/lightly_train/_task_models/train_model.py` | `TrainModelArgs` with `ClassVar` `default_batch_size` / `default_steps` and `resolve_auto(...)`: defaults owned by the model, not the CLI |
| `src/lightly_train/_loggers/task_logger_args.py` | `*_args` sub-config with its own `resolve_auto` |
| `src/lightly_train/_models/package_helpers.py` | `"package/model"` string namespace, `list_model_names` |
| `src/lightly_train/_task_models/task_model_helpers.py` | `load_model(path_or_name)` returning an object with `.predict(...)` |

---

## 1. What to copy

1. **Task-named, keyword-only Python API.** `train.train_image_classification(out=..., data=..., model=..., epochs="auto", ...)`. One function per task, not per model.
2. **Model as a namespaced string.** `model="convnext_v1/cifar10"`, `"vit/small"`, `"cliffordnet/<variant>"`, or a path to a `.keras` file (fine-tune from weights).
3. **One config class per task**, built from the call's kwargs, validated once, logged once, written once.
4. **`"auto"` values resolved from model-owned defaults** (batch size, epochs, image size, lr), never from scattered CLI defaults.
5. **Grouped `*_args` dicts** instead of dozens of flat flags: `model_args`, `optim_args`, `schedule_args`, `transform_args`, `loader_args`, `logger_args`, `callback_args`, `analysis_args`.
6. **Strict validation.** Unknown key at any depth raises, naming the dotted path and the closest valid key (`difflib.get_close_matches`).
7. **CLI = same config.** `python -m train image_classification out=... model=convnext_v1/cifar10 data=cifar10 optim_args.lr=1e-3 logger_args.tensorboard=null`. The CLI builds the same dict and calls the same function.
8. **`resume_interrupted` vs `checkpoint`.** Resume = same `out`, full state, no parameter changes allowed (compare against saved config and refuse on mismatch). Checkpoint = weights only, new run.
9. **`load_model(path_or_run_dir).predict(x)`**, `list_models(task=None)`, `list_tasks()`, `export(...)`.
10. **Help generated from the config** (`python -m train help`, `python -m train image_classification help`), printing current defaults from the dataclass, so docs cannot drift.

## 2. What NOT to copy (house rules win)

| LightlyTrain behaviour | Rule here |
|---|---|
| `overwrite=True` reuses/overwrites a non-empty `out` | **Do not implement.** Keep `refuse_existing_run`. Nothing under `results/` is ever overwritten or deleted. |
| Pydantic + OmegaConf | No new dependencies. Use stdlib `dataclasses` + a small validator. Existing AST guards read dataclass fields. |
| Step-based loop, `steps`, `val_every_num_steps` | Keep Keras `fit` and epochs. `epochs` (+ optional `steps_per_epoch`) only. Respect the "repeat the dataset" rule. |
| `accelerator`, `devices`, `strategy`, `num_nodes` | Single `gpu: int | None` through `train.common.setup_gpu` (D-110). No distributed scope. |
| `precision="bf16-mixed"` strings | `precision: "float32" | "mixed_float16" | "mixed_bfloat16"` through `setup_mixed_precision`; `LossScaleOptimizer` at the call site for fp16. |
| Default `seed=0` | `seed` defaults to `DEFAULT_SEED` from `train/common/seed.py`, applied via `set_seeds`. |
| Output files `checkpoints/last.ckpt`, `exported_models/*.pt`, `metrics.jsonl` | Keep the existing contract: `config.json`, `run.log`, `training_log.csv`, `training_history.json`, `best_model.keras`, `final_model.keras`, `results_summary.json`, `visualizations/`, `model_analysis/`. `compare_runs` and guards read it. Only ADD files (see section 5). |
| Telemetry (`tracker.track_training_started`) | None. |
| `out` required | `out` optional; omitted means `results/<default_experiment_name(task, model)>`; relative paths anchored at repo root via the existing resolver. |

---

## 3. Target UX

Python:

```python
import train

train.train_image_classification(
    out="results/convnext_c10_baseline",
    data="cifar10",                         # or a dict / JSON path for a folder dataset
    model="convnext_v1/cifar10",
    epochs="auto",
    batch_size=128,
    gpu=0,
    optim_args={"lr": 1e-3, "weight_decay": 0.05},
    callback_args={"patience": 20, "epoch_analysis_every": 10},
)

model = train.load_model("results/convnext_c10_baseline")   # run dir or .keras path
probs = model.predict(images)
train.list_models(task="image_classification")
```

CLI:

```bash
MPLBACKEND=Agg .venv/bin/python -m train help
MPLBACKEND=Agg .venv/bin/python -m train image_classification help
MPLBACKEND=Agg .venv/bin/python -m train image_classification \
    out=results/convnext_c10_baseline data=cifar10 model=convnext_v1/cifar10 \
    epochs=100 optim_args.lr=1e-3 transform_args.image_size="[32, 32]" gpu=0
MPLBACKEND=Agg .venv/bin/python -m train list_models task=image_classification
MPLBACKEND=Agg .venv/bin/python -m train export run=results/convnext_c10_baseline format=keras
```

Legacy entry points (`python -m train.convnext.train_convnext_v1 --dataset cifar10 ...`) stay and keep passing their tests.

---

## 4. Architecture

New package `src/train/api/` (never a module named `train.py`):

| Path | Responsibility |
|---|---|
| `src/train/__init__.py` | Lazy public API via PEP 562 `__getattr__`. `import train` must not import TensorFlow or Keras. |
| `src/train/__main__.py` | CLI dispatch: `help`, `list_models`, `list_tasks`, `export`, `<task>`. Non-zero exit on any failure. |
| `src/train/api/config.py` | `StrictConfig` base (dataclass). `from_dict` rejecting unknown keys at every depth with suggestions; `AUTO = "auto"`; `has_auto()`, `assert_resolved()`; `to_dict()`; field help via `field(metadata={"help": ...})`. |
| `src/train/api/dotlist.py` | `a.b.c=value` tokens to nested dict. Values: `null`/`None`, `true`/`false`, numbers, `[..]`/`{..}` via `ast.literal_eval`, else string. Duplicate key raises. |
| `src/train/api/registry.py` | `TASKS: {task_name: TaskSpec(config_cls, runner, models)}`. `MODELS` keys `"<family>/<variant>"` DERIVED from each model class's `MODEL_VARIANTS` (one fact, one home). `ModelDefaults` per family (`default_batch_size`, `default_epochs`, `default_image_size`, `default_lr`). Imports of model modules happen inside functions only. |
| `src/train/api/common_args.py` | Shared sub-configs: `OptimArgs`, `ScheduleArgs`, `TransformArgs`, `LoaderArgs`, `LoggerArgs`, `CallbackArgs`, `AnalysisArgs`, each with `resolve_auto(...)`. |
| `src/train/api/run.py` | Shared run preamble: resolve `out`, `refuse_existing_run` (or resume checks), `attach_run_log`, log input args and resolved config (pretty JSON), `setup_gpu`, `set_seeds`, precision, write `train_args.json`. |
| `src/train/api/tasks/<task>.py` | `<Task>Config(StrictConfig)` + `train_<task>(**kwargs)` + `run(config)` adapter that builds the EXISTING trainer's config and calls its existing `train(config)`. |
| `src/train/api/load.py` | `load_model(path_or_run_dir)` returning a `LoadedModel` with `.predict(x, batch_size=...)`, `.keras_model`, `.config`, `.class_names`; preprocessing/normalization read from the run's `config.json`, never hardcoded. |
| `src/train/api/export.py` | `export(run, format="keras" | "saved_model", which="best" | "final", out=None)` into `<run>/exported_models/`. ONNX only if a converter already exists in the repo. |
| `src/train/api/help.py` | Help text rendered from config dataclasses and registry. |

Config field set for every task (task configs add their own):

| Field | Type | Default |
|---|---|---|
| `out` | `str | None` | `None` (timestamped under `results/`) |
| `data` | `str | dict | path` | required |
| `model` | `str` | required (`family/variant` or `.keras` path) |
| `epochs` | `int | "auto"` | `"auto"` |
| `batch_size` | `int | "auto"` | `"auto"` |
| `steps_per_epoch` | `int | None` | `None` |
| `seed` | `int` | `DEFAULT_SEED` |
| `gpu` | `int | None` | `None` |
| `precision` | `str` | `"float32"` |
| `resume_interrupted` | `bool` | `False` |
| `checkpoint` | `str | None` | `None` |
| `smoke` | `bool` | `False` (maps to existing smoke presets where present) |
| `model_args`, `optim_args`, `schedule_args`, `transform_args`, `loader_args`, `logger_args`, `callback_args`, `analysis_args` | `dict | None` | `None` |

Rules for the adapter layer:

- An adapter MAPS fields; it never re-implements data loading, callbacks, evaluation or figures.
- Every front-door field must reach a field of the underlying trainer config, or be consumed in `run.py`. A field the target trainer does not support raises `ConfigError` naming the field and task/model, never a silent no-op.
- Optimizers go through `dl_techniques.optimization.optimizer_builder()`. Map `optim_args.clipnorm` to `gradient_clipping_by_norm_local` and `optim_args.global_clipnorm` to `gradient_clipping_by_norm` explicitly (the builder silently ignores a literal `clipnorm`). Never combine `AdamW(weight_decay)` with an L2 regularizer.
- Monitor direction always via `resolve_monitor_mode`.
- No eager TF op at module scope anywhere under `src/train/api/` or `src/train/common/`.

## 5. Run directory contract (additions only)

Existing files are untouched in name and schema. Add:

- `train_args.json`: `{"task": ..., "input": <kwargs as passed>, "resolved": <front-door config after auto resolution>, "trainer_config_class": "<module.Class>"}`.
- `run.log` gains two blocks at the top: `Input args:` and `Resolved config:` (pretty JSON), then system info (TF/Keras versions, visible GPUs, precision).
- `exported_models/` only when `export` is called.

`resume_interrupted=True`: requires an existing run at `out`, loads its `train_args.json`, refuses if any resolved field other than `resume_interrupted`/`epochs` differs, resumes from the trainer's existing step/epoch checkpoint mechanism. If the target trainer has no resume support, raise `ConfigError`; do not invent one in this task.

---

## 6. Phases (stop and report after each)

Create the plan under `plans/plan-<timestamp>-<hash>/` following the existing convention (`progress.md`, `decisions.md`, `changelog.md`). Record every non-obvious choice as a `D-NNN` decision.

**Phase 0: inventory (no code changes).** Produce `plans/.../inventory.md`: one row per entry point under `src/train/` (about 118): module path, pattern (1 to 8 or "none"), has a config dataclass, has a `train(config)` function separable from argparse, model package, `MODEL_VARIANTS` present, smoke preset present, resume support, proposed front-door task name. Propose the task list (expected: `image_classification`, `semantic_segmentation`, `denoising`, `depth_estimation`, `object_detection`, `language_model_pretraining`, `text_classification`, `time_series_forecasting`, plus "not migrated" with reason). Wait for approval.

**Phase 1: infrastructure only.** `config.py`, `dotlist.py`, `registry.py` (empty task table), `common_args.py`, `run.py`, `help.py`, `__main__.py`, lazy `__init__.py`. Tests in `tests/test_train/test_api/`. No trainer touched.

**Phase 2: pilot `image_classification` on ConvNeXt V1/V2.** Adapter builds `train.convnext.common.TrainingConfig` and calls `train.convnext.common.train`. Registry keys from `ConvNeXtV1.MODEL_VARIANTS` / `ConvNeXtV2.MODEL_VARIANTS`. Parity test: legacy argv and the equivalent API call produce field-for-field identical `TrainingConfig`.

**Phase 3: rest of Pattern 1.** One trainer per commit: cliffordnet, vit, resnet, alexnet, mobilenet, ... For trainers whose body takes an argparse `Namespace` (e.g. `cliffordnet.train_model(args)`), first split into `config_from_args` + `train(config)` with no behaviour change, proven by the existing tests, then add the adapter.

**Phase 4: other tasks, one pattern at a time** in the order approved in Phase 0. Exemplars per `src/train/AGENTS.md`.

**Phase 5: `load_model`, `export`, `list_models`, `list_tasks`.**

**Phase 6: docs.** Section in root `README.md`; new section in `src/train/AGENTS.md` ("Front door: `train.api`") with the adapter rules from section 4; entry-points section of `REPO_MAP.md`. Paths cited in guides must exist (`tests/test_the_guides_cite_only_paths_that_exist.py`).

---

## 7. Required tests (`tests/test_train/test_api/`)

1. `dotlist`: nesting, types, `null`, lists, quoted strings, duplicate key error, malformed token error.
2. Strict validation: unknown top-level and nested keys raise with dotted path and suggestion; wrong type raises.
3. `"auto"`: every auto field resolved before the runner is called (`assert_resolved`), from the model's `ModelDefaults`.
4. CLI contract, same three traps as `_cli_contract.py`: every key drives a NON-default value, the row table equals the config's field set (completeness), values distinct across rows. Assert on the resulting trainer config object, never on exit codes or help text.
5. Dead fields: extend `test_config_fields_are_live.py` coverage to front-door configs (a serialized field is not a consumed field).
6. Import hygiene: `python -m train help` and `import train` in a subprocess leave `tensorflow` and `keras` out of `sys.modules` and set no GPU state.
7. Registry: `MODELS` keys equal the union of the wired families' `MODEL_VARIANTS` keys; unknown model string raises with suggestions.
8. Parity per migrated trainer (legacy argv vs API kwargs gives identical trainer config).
9. Failure: a stubbed raise inside the runner gives non-zero exit through the CLI and a raised exception through the API.
10. Output location: all writes under `out` (`tmp_path`, absolute). Never write under `results/` in tests.
11. Resume: mismatched parameters refused; unsupported trainer refused.
12. One CPU smoke per task (tiny `max_samples`, 1 epoch), marked `slow`.

Run only the scoped suites: `.venv/bin/python -m pytest tests/test_train/test_api/ tests/test_train/test_<touched_trainer>/ -q`. Never the full suite unless asked.

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

- `python -m train help` lists tasks; `python -m train <task> help` lists every field with type and live default.
- Every migrated trainer is reachable by API and CLI, its legacy entry point still passes its existing tests, and parity tests pass.
- Unknown keys, unsupported fields and failed runs fail loudly.
- `train_args.json` and the run log carry input and resolved config; all existing artifacts unchanged.
- `load_model(run_dir).predict(x)` works for every migrated task.
- Inventory marks every entry point as migrated or "not migrated: <reason>".

## 10. Decisions to confirm with the user before Phase 1

Ask once, with these defaults:

1. Package location `src/train/api/` with CLI `python -m train` (default), vs. moving the front door into the shipped `dl_techniques` package (would need `train.common` relocated).
2. No pydantic/omegaconf/pyyaml (default); data configs as dict or JSON. YAML accepted only if `yaml` is importable.
3. `out` optional with timestamped default (default) vs. required as in LightlyTrain.
4. Task list and migration order from the Phase 0 inventory.
5. Whether `export` should target ONNX (only if a converter is already a dependency).