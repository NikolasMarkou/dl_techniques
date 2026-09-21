# Training Scripts

Training pipelines for models in `dl_techniques/models/`. One directory per runnable pipeline; the
grouped families `time_series/` and `language/` keep their trainers one level further down.
`REPO_MAP.md` maps the model/trainer/test triangle, including the trainers whose directory name
differs from the model package they train.

`research/benchmarks/` (repo root) holds dated external SOTA snapshots (embeddings, LLM, vision,
VLM) plus a metric-definition reference. Consult them **before** starting a run at the relevant
scale, so the target is realistic rather than ten points below open weights.

## File naming

Name scripts `train_<model>.py`, **not** `train.py`. A file named `train.py` shadows the `train`
package and breaks `from train.common import ...`.

## Training script patterns

Six shapes. Pick the closest exemplar and copy it; do not re-derive the scaffold.

| Pattern | Used by | Exemplar to copy | Monitor |
|---|---|---|---|
| **Pattern 1: Vision classification** — `load_dataset()` + `create_base_argument_parser()` + the full evaluation pipeline | ConvNeXt, CapsNet, CoshNet, CliffordNet, PowerMLP (`load_dataset()` and the evaluation pipeline, but its own parser, not `create_base_argument_parser()`; see `src/train/power_mlp/README.md`), KAN, ViT, SOM, MobileNet | `src/train/vit/train_vit.py` | `val_accuracy` |
| **Pattern 2: Time-series / probabilistic** — synthetic generators, local argparse (the base parser's `--dataset` choices do not apply), `include_terminate_on_nan=True`, analyzer behind a `--deep-analysis` flag | N-BEATS, PRISM, TiRex, MDN | `src/train/time_series/nbeats/` | `val_loss` |
| **Pattern 3: NLP pretrain/finetune** — `train.common.nlp` for tokenization, text datasets, warmup LR, callbacks; model code stays local | BERT, FNet, tree_transformer, GPT-2, wave_field | `src/train/bert/pretrain.py`, `src/train/bert/finetune.py` | `val_loss` |
| **Pattern 4: Denoising / detection / dense prediction** — file-based (or in-memory dense-label) datasets, domain callbacks appended to a `create_callbacks()` wrapper | BFCNN, BFUNet, ConvUNeXt denoiser (`bfunet/`), ConvUNext segmentation (`convunext/`), YOLO12-COCO, ResNet, DarkIR | `src/train/bfunet/train_bfcnn_denoiser.py` | `val_loss` / `val_psnr` |
| **Pattern 5: Depth estimation** — `train.common.megadepth` pipeline, depth metrics + visualization callbacks from `dl_techniques` | Depth Anything | `src/train/depth_anything/train_depth_anything.py` | `val_loss` |
| **Pattern 6: Byte-level LM pretrain** — no tokenizer at all: `dl_techniques.datasets.byte_lm` packs raw UTF-8 into causal windows, and the auxiliary loss arrives through the model's `add_loss` rather than a custom `train_step` | H-Net | `src/train/hnet/common.py` | `val_loss` |

**`src/train/convunext/` (ConvUNext segmentation) is a sibling of `src/train/convnext/`, not a task switch inside it.**
It writes the same run-directory contract (`config.json`, `run.log`, `training_log.csv`,
`training_history.json`, `best_model.keras`, `final_model.keras`, `results_summary.json`,
`visualizations/`, `model_analysis/`) from the shared `train.common` pieces, with segmentation's own
orchestrator (`common.py`), figures (`segmentation_viz.py`) and `SegTrainingConfig`; its README lists
the flags and every summary key. Run it with `python -m train.convunext.train_convunext_segmentation`.
Its denoiser counterpart stays in `src/train/bfunet/` (`train_convunext_denoiser.py`). Notes that hold
for both ConvUNext trainers: neither puts the Keras layer table in `run.log` (the denoiser writes it to
`model_summary.txt`); a run whose last epoch is the best epoch scores its test data once and records it
(`final_reused_best` at the top level of the segmentation summary, `test_eval.final_reused_best` for the
denoiser); both write one `Visualizations: N written (...)` line to `run.log` and record `visualizations.{files, failed, seconds}` (`failed` is a list of figure names in both)
(the segmentation block also has `skipped`, the figures deliberately not drawn); the denoiser redraws its dashboard on
the shared cadence (every epoch up to 39 epochs), the segmentation trainer through `TrainingDashboardCallback`.

`create_base_argument_parser()` supplies `--dataset`, `--image-size`, `--epochs`, `--batch-size`,
`--learning-rate`, `--weight-decay`, `--lr-schedule`, `--patience`, `--gpu` and `--show-plots`; add
model-specific arguments on top of it.

> **Never double weight decay.** `AdamW` applies decoupled weight decay internally, so do not also
> pass `kernel_regularizer=L2(weight_decay)` — the penalty inflates the loss and the optimizer
> decays the parameter again. Use `AdamW(weight_decay=...)` alone (preferred, matches most paper
> recipes) or `Adam` + `kernel_regularizer=L2(...)`, never both.

### `train.common.nlp` API

| Function | Purpose |
|----------|---------|
| `create_tokenizer(encoding, max_len, ...)` | `TiktokenPreprocessor` with `cl100k_base` defaults |
| `load_text_dataset(name, split, max_samples, as_supervised)` | TFDS text dataset loading |
| `preprocess_mlm_dataset(ds, preprocessor, seq_len, batch)` | Tokenize + batch for MLM (deliberately no `.cache()`) |
| `preprocess_clm_dataset(ds, preprocessor, seq_len, batch)` | Concat-and-chunk packed CLM; the signature has **no** `streaming` parameter |
| `preprocess_clm_packed_dataset(...)` | Lower-level packed CLM with `repeat=True` for explicit step-budget loops |
| `preprocess_classification_dataset(ds, preprocessor, seq_len, batch)` | Tokenize + batch with labels |
| `estimate_clm_steps_per_epoch(num_articles, max_seq_length, batch_size, override=None, avg_tokens_per_article=440)` | **The canonical chunk-aware steps-per-epoch helper. Never roll a local `_estimate_steps_per_epoch`.** |
| `create_warmup_lr_schedule(lr, epochs, steps, warmup_ratio)` | Warmup + cosine decay (defined in `dl_techniques.optimization.schedule`, re-exported here) |
| `create_nlp_callbacks(name, prefix, ...)` | `create_callbacks` with NLP defaults and TensorBoard on |
| `evaluate_mlm_model(mlm_model, preprocessor, test_texts=None)` | Qualitative MLM probe + `visualize_mlm_predictions` |
| `run_finetune_post_training_analysis(config, model_name, create_initial_model, results_dir)` | Full `ModelAnalyzer` comparison over Initial/Best/Final. `results_dir` is REQUIRED with no fallback on purpose |
| `sentiment_final_model_filename(model_name)` | `f"{model_name}_sentiment_final_best.keras"` — called at both the save site and the read site |
| `best_checkpoint_path(results_dir)` (in `common/callbacks.py`) | `<results_dir>/best_model.keras` — the ONE producer of the best-checkpoint path |

> **A path or filename known in two places is a function, not a string typed twice.** Those last
> two exist because a write site and a read site once disagreed, and every default fine-tuning run
> crashed at the end loading a file nothing wrote. Both ends of the contract call the producer.

> **Pattern 6 shares the corpus, not the tokenizer.** `train.common.nlp` builds a
> `tiktoken.Encoding` inside its generator thread, so a `vocab_size = 256` byte model cannot
> use `preprocess_clm_dataset` at all. `dl_techniques.datasets.byte_lm` is the byte-unit
> rebuild of exactly those transforms — including `estimate_byte_clm_steps_per_epoch`, the
> sibling of `estimate_clm_steps_per_epoch`. Both pipelines consume the SAME
> `load_wikipedia_train_val` text stream, so the conventions below apply unchanged.

**Wikipedia/HF conventions** (`dl_techniques.datasets.nlp.load_wikipedia_train_val`):
`min_article_length` defaults to `0` (packed CLM uses every token; pass 500+ only when a consumer
treats one document as one example); `num_shards` enables parallel tokenization shards with
per-epoch reshuffle (`1` keeps deterministic single-thread behaviour, CLM consumers default to 4);
`return_counts=True` returns post-filter article counts to feed `estimate_clm_steps_per_epoch`.
Every CLM script exposes the same four flags so users can switch scripts without relearning:
`--steps-per-epoch`, `--seed`, `--min-article-length`, `--shuffle-shards`. On `--resume <ckpt>`
derive `data_seed = config.seed + initial_step`, so a resumed run sees new article ordering rather
than replaying the first N chunks.

### bert / fnet / tree_transformer drifts, deliberately not harmonized

`src/train/tree_transformer/` looks like a third Pattern-3 package and its README says it "mirrors"
bert. It is **not** folded into the shared bert/fnet scaffold, on purpose: its `finetune.py` has no
`post_training_analysis` and no `ModelAnalyzer` references, and its `pretrain.py` has no
`evaluate_model` — folding it in means inventing config toggles for code that does not exist.

| Drift | bert | fnet | tree_transformer |
|---|---|---|---|
| `finetune.py` `max_seq_length` | **256** | **128** | 128 |
| `finetune.py` `stage1_epochs` / `stage2_epochs` | 5 / 10 | 5 / 10 | **2 / 3** |
| `finetune.py` optimizer `clipnorm` | absent | absent | **1.0** |
| MLM `steps_per_epoch` fallback | `max_samples // batch_size if max_samples else 1000` | same | **`max(1, (max_samples or 10000) // batch_size)`** |

The 256-vs-128 split is the one a shared scaffold is most likely to tidy away. It is pinned by
`tests/test_train/test_bert_fnet/test_finetune_scripts.py::TestArgvToConfig`; harmonizing either
value turns that guard RED.

## `create_callbacks()`

Defined in `src/train/common/callbacks.py`; read it for the parameter docs.

```python
create_callbacks(
    model_name, results_dir_prefix="model", output_root="results", run_dir=None,
    monitor='val_accuracy', monitor_mode=None, patience=15, use_lr_schedule=True,
    analyzer_epoch_frequency=1, include_tensorboard=False, include_terminate_on_nan=False,
    include_analyzer=True, analyzer_config=None, analyzer_start_epoch=1,
) -> Tuple[List[Callback], str]      # (callbacks, results_dir)
```

Always included: EarlyStopping, ModelCheckpoint, CSVLogger. Optional: TensorBoard, TerminateOnNaN
(time-series / probabilistic), the epoch analyzer (off for sub-stages), and ReduceLROnPlateau via
`use_lr_schedule=False` when there is no external schedule.

**The monitor mode comes from a metric-name TOKEN REGISTRY, never a substring test.**
`train.common.resolve_monitor_mode(monitor, mode=None)` lowercases the key, splits it into
alphanumeric tokens, drops a leading `val`, matches `_MINIMIZE_METRIC_TOKENS` first (so
`val_dice_loss` is a loss, not a Dice coefficient), then `_MAXIMIZE_METRIC_TOKENS`. An unrecognized
name logs a WARNING and falls back to `'min'` — pass `monitor_mode='max'` or add the token, rather
than working around it at the call site. Two rules keep it correct: a Keras
multi-output monitor key embeds the OUTPUT LAYER's name (`val_<output>_<metric>`), so a generic
token like `residual`, `score`, `std` or `variance` in either set would resolve off a layer name
and invert the direction — keep both sets unambiguous; and Keras' own `mode='auto'` is not a
substitute, because it resolves at epoch end by matching a compiled metric object's `.name` and
**raises `ValueError`** when it finds none, which is exactly the multi-output case. Guarded by
`tests/test_train/test_common_callbacks.py`.

## What lives in `train.common`

| Symbol | Notes |
|---|---|
| `setup_gpu(gpu_id)` | Memory growth + device selection. Every script supports `--gpu` and calls `setup_gpu(args.gpu)`: it sets `CUDA_VISIBLE_DEVICES` for a specific GPU, or enables memory growth on all of them when `None` |
| `create_callbacks(...)` / `create_nlp_callbacks(...)` | See above |
| `resolve_monitor_mode(monitor, mode=None)` | The ONE producer of a checkpoint-selection direction |
| `create_base_argument_parser(description, default_dataset)` | Only for scripts using `load_dataset()` |
| `create_learning_rate_schedule(lr, type, epochs, steps_per_epoch)` | Cosine / exponential / constant. Defined in `dl_techniques.optimization.schedule` and re-exported here; both paths resolve to the same object |
| `load_dataset(...)` / `get_class_names(...)` | See Data loading below |
| `validate_model_loading(...)` / `run_model_analysis(...)` | Round-trip serialization check; full ModelAnalyzer pipeline |
| `run_summary.run_data_free_analysis(model, sample, labels, history, model_name, run_dir, enabled=True)` | The end-of-run weights + spectral analysis for trainers whose data is not per-image labels (the ConvUNeXt denoiser and segmentation trainers). Never raises; the returned `analyzer` block (`{status, analyzers, error, path, seconds}`) is read back from `model_analysis/analysis_results.json` by `read_data_free_analysis_status`, not taken from the analyzer's return value. After an `ok` read-back it deletes the library's empty `model_analysis/summary_dashboard.png` and adds `removed: ["summary_dashboard.png"]` to the block (absent otherwise), and only while `model_metrics`, `calibration_metrics`, `confidence_metrics` and `weight_pca` of the returned result are all empty: a dashboard a panel has data for is kept. `data_free_analysis_config()` is the one `AnalysisConfig` behind it. The analyzer call runs inside `threadpool_limits(limits=1)`, scoped to that call (12 BLAS threads oversubscribed the host: 14.7 to 21.0 s against 6.1 to 6.6 s on one saved model). Do not write a per-trainer copy |
| `run_summary.summary_head(config, run_dir, *, params, steps_per_epoch, devices)` | The summary keys the bfunet trainers (ConvUNeXt denoiser, unet, bfcnn) and the ConvUNext segmenter share: the config attributes named in `SUMMARY_CONFIG_KEYS`, `epochs_requested`, the sizes, the three device keys and `image_range` `[0.0, 1.0]` (the CLEAN images' range, not the noisy input, which `--no-clip` leaves unclipped; not `config.json`'s `data_range` string). A new shared fact is added there once; the ConvNeXt reference keeps its own `_summary_head` |
| `TrainingDashboardCallback(..., best_key=None)` / `render_training_dashboard(..., best_epoch=None)` (`classification_viz`) | Opt-in best-epoch marker: pass `best_key="val_loss"` to draw a dashed line and star at the lowest value so far. `None` draws nothing and leaves the PNG unchanged. A history that carries `miou` / `val_miou` columns (the segmentation trainer's) gets an extra mIoU panel; a history without them draws the same panels and the same PNG bytes as before |
| `discover_megadepth_pairs(root)` / `MegaDepthDataset(...)` | MegaDepth RGB+depth pipeline |
| `compare_runs(a_dir, b_dir, labels, output_dir)` | Two-run comparison; also `python -m train.common.compare_runs A B`. Emits `comparison.md` + curve PNGs |
| `StepCheckpointCallback(...)` | Step-indexed CSV logging, rolling `.keras` checkpoint window, optional periodic ModelAnalyzer, step-loss plots. Pass an external `step_counter` for resume setups. Use instead of a per-trainer step-checkpoint class |
| `set_seeds(seed)` | Canonical seeding (`PYTHONHASHSEED` + `random` + numpy + `keras.utils.set_random_seed`). The numpy/`random`-only seeders in `tabm/` and `nam/` are deliberately NOT routed through it — that would ADD TF/Keras seeding and change their init RNG stream |
| `save_config_json(...)`, `prepare_run_dir(config, output_dir=None)` | Run-directory preamble |
| `save_training_history_json(history, output_dir)` | Best-effort: warns and returns `None` rather than raising, since it runs after the weights are saved |
| `default_experiment_name(*parts)` | Underscore-join + run timestamp. A prefix already ending in `_` must be concatenated with the next fragment yourself |
| `log_gpu_peak_memory()`, `setup_mixed_precision(enabled, policy)` | Wrap the optimizer in `LossScaleOptimizer` at the call site for `mixed_float16` |
| `train.common.stats` | `mean_std`, `bootstrap_ci`, `paired_permutation_test`, `format_mean_std` — NaN-tolerant, degenerate-safe. Pass an explicit `rng=np.random.default_rng(SEED)`; `json_numpy_default` serializes numpy scalars for `json.dump` |
| `CIFAR10_MEAN` / `CIFAR10_STD` | Distinct from the OpenAI-CLIP `IMAGE_MEAN`/`IMAGE_STD` in `image_text.py`. Never conflate them |

> **Never run an eager TF op at module scope anywhere under `train/common/`.** It initializes TF's
> eager context and creates a GPU device, so every importer of every submodule — including a bare
> `--help` — allocates a GPU. The package `__init__` re-exports everything, so the cost is tree-wide.

From `dl_techniques` rather than `train.common`: `metrics.depth_metrics`,
`callbacks.depth_visualization`, and `optimization`'s `optimizer_builder()` /
`learning_rate_schedule_builder()` / `WarmupSchedule`.

**Build optimizers through `optimizer_builder()`** rather than `keras.optimizers.AdamW(...)`: it
handles clipping and weight-decay exclusions in the constructor, where Keras requires them. Never
set `optimizer.clipnorm` after construction. Two traps when you adopt it:

- It **renames the clipping keys** — `gradient_clipping_by_value` → `clipvalue`,
  `gradient_clipping_by_norm_local` → `clipnorm`, `gradient_clipping_by_norm` → `global_clipnorm`.
  A literal `"clipnorm"` key is silently ignored, so a naive migration from `AdamW(clipnorm=1.0)`
  **drops gradient clipping with no error and no warning**.
- It **hard-codes `"name": "AdamW"`** against Keras' own `"adamw"` default. The name is the
  optimizer's variable scope — a checkpoint-compatibility consideration for every trainer using it.

For weight transfer use `dl_techniques.utils.weight_transfer.load_weights_from_checkpoint`
(layer-by-layer, from a saved `.keras` model). Not `model.load_weights(..., by_name=True)`, which
Keras 3.8 supports only for legacy `.h5`/`.hdf5` and rejects on `.keras` with
`ValueError: Invalid keyword arguments: {'by_name': True}`.

Keep local: model creation and compilation, domain-specific losses, custom argparse where
`create_base_argument_parser()` does not fit, and the training summary.

## Data loading

`load_dataset()` covers mnist, cifar10, cifar100 and imagenet only, returning
`(x_train, y_train), (x_test, y_test), input_shape, num_classes` for the numpy datasets and
`train_ds, val_ds, input_shape, num_classes` for ImageNet (tf.data). For text use
`train.common.nlp`; for MegaDepth RGB+depth pairs `train.common.megadepth` (yielding
`(rgb, y_true)` with `y_true = [depth, mask]` on the last axis). For anything else write local
loading — do NOT force it through `load_dataset()`.

## Scripts that don't use `train.common` callbacks (and why)

| Script | Reason |
|--------|--------|
| `bert/wikipedia/*` | MirroredStrategy distributed training, BackupAndRestore for fault tolerance |
| `blt` | Multi-stage pipeline (entropy pretraining + main training), class-based trainer |
| `yolo12/train_multitask` | Per-task callbacks, losses and visualization |
| `tabm` | Custom `TabMTrainer`, not standard Keras `fit()` |
| `lewm` | Dict-shaped inputs (`{"pixels", "action"}`) + add_loss-only training; `EpochAnalyzerCallback` doesn't understand either |

When a new script genuinely cannot use `create_callbacks()`, document the reason in a comment at
the top of its callbacks section. **This table is about CALLBACKS and nothing else — it is not a licence to skip the CLI.** Every
entry point owes a `parse_arguments()` / `main(argv)` whose FIRST statement parses argv, so
`--help` exits 0 with a `usage:` line and allocates nothing. **Exit 0 is not a passing `--help`**:
a script with no parser ignores it, runs its whole job and exits 0 anyway — assert the `usage:`
line, not the exit code.

## Config fields must be live

A declared config field that nothing reads is a knob that silently does nothing — the same class as
a CLI argument `main()` forgets to forward. **Serialization is not consumption**: reaching
`save_config_json` / `asdict` / `prepare_run_dir` and nothing else does not make a field live.
DELETE such a field rather than wiring it, unless a caller actually wants the behaviour.
`tests/test_train/test_config_fields_are_live.py` enforces this for the classes in its `REGISTERED`
list — add a row when you add a config class; never add an exemption to green a newly dead field.
