# Training Scripts

Training pipelines for models in `dl_techniques/models/`. One directory per runnable pipeline; the
grouped families `time_series/` and `language/` keep their trainers one level further down.
`REPO_MAP.md` maps the model/trainer/test triangle, including the trainers whose directory name
differs from the model package they train.

`research/benchmarks/` (repo root) holds dated external SOTA snapshots (embeddings, LLM, vision,
VLM) plus a metric-definition reference. Consult them **before** starting a run at the relevant
scale, so the target is realistic rather than ten points below open weights.

> **Read [`GUIDE.md`](GUIDE.md) before changing a trainer.** It records the defect patterns found
> in these entry points (failed runs that exit 0, writes that cost the weights, GPU selection,
> CSV logs, seeding), each with the guard that now prevents it. Model-side patterns are in
> `src/dl_techniques/models/GUIDE.md`.

## File naming

Name scripts `train_<model>.py`, **not** `train.py`. A file named `train.py` shadows the `train`
package and breaks `from train.common import ...`.

## Training script patterns

Six shapes. Pick the closest exemplar and copy it; do not re-derive the scaffold.

| Pattern | Used by | Exemplar to copy | Monitor |
|---|---|---|---|
| **Pattern 1: Vision classification** — `load_dataset()` + `create_base_argument_parser()` + the full evaluation pipeline | ConvNeXt, CapsNet, CoshNet, CliffordNet, PowerMLP (`load_dataset()` and the evaluation pipeline, but its own parser, not `create_base_argument_parser()`; see `src/train/power_mlp/README.md`), KAN, ViT, SOM, MobileNet, DeepSORT embedding (`deepsort/`: PK identity batches over synthetic/Market1501 data, cosine-softmax or triplet loss, CMC/mAP retrieval eval; own parser since `load_dataset()` has no ReID source) | `src/train/vit/train_vit.py` | `val_accuracy` |
| **Pattern 2: Time-series / probabilistic** — synthetic generators, local argparse (the base parser's `--dataset` choices do not apply), `include_terminate_on_nan=True`, analyzer behind a `--deep-analysis` flag | N-BEATS, PRISM, TiRex, MDN | `src/train/time_series/nbeats/` | `val_loss` |
| **Pattern 3: NLP pretrain/finetune** — `train.common.nlp` for tokenization, text datasets, warmup LR, callbacks; model code stays local | BERT, FNet, tree_transformer, GPT-2, wave_field | `src/train/bert/pretrain.py`, `src/train/bert/finetune.py` | `val_loss` |
| **Pattern 4: Denoising / detection / dense prediction** — file-based (or in-memory dense-label) datasets, domain callbacks appended to a `create_callbacks()` wrapper | BFCNN, BFUNet, ConvUNeXt denoiser (`bfunet/`), ConvUNext segmentation (`convunext/`), YOLO12-COCO, ResNet, DarkIR, LightGlue (`lightglue/`: frozen SuperPoint front end, loss through `add_loss`, README lists the run directory and the flags), SiamFC (`siamfc/`: centered COCO/synthetic pairs, radius-label logistic loss), DaSiamRPN (`dasiamrpn/`: per-anchor cls/reg losses over the reference anchor grid) | `src/train/bfunet/train_bfcnn_denoiser.py` | `val_loss` / `val_psnr` |
| **Pattern 5: Depth estimation** — `train.common.megadepth` pipeline, depth metrics + visualization callbacks from `dl_techniques` | Depth Anything | `src/train/depth_anything/train_depth_anything.py` | `val_loss` |
| **Pattern 6: Byte-level LM pretrain** — no tokenizer at all: `dl_techniques.datasets.byte_lm` packs raw UTF-8 into causal windows, and the auxiliary loss arrives through the model's `add_loss` rather than a custom `train_step` | H-Net | `src/train/hnet/common.py` | `val_loss` |
| **Pattern 7: Sequence-generative with a paper's own metrics** — a GENERATOR over unlabelled sequences, so the validation signal is the objective and the run's real output is a set of paper columns rather than a single monitor. `dl_techniques.datasets` owns the data semantics, the model owns a dict output, a `keras.losses.Loss` composes the ELBO, and the trainer evaluates with domain metrics that no stock callback provides | Topographic VAE (`topographic_vae/`): transformation sequences from `datasets/vision/transform_sequences.py`, `TopographicVAELoss`, and an evaluation block carrying `log p(x)` (IWAE) + `equivariance_error` + `capcorr` — the last two kept together because the first cannot distinguish equivariance from invariance | `src/train/topographic_vae/train_topographic_vae.py` | `val_loss` |

| **Pattern 8: NLP pretrain whose auxiliary term rides `add_loss`, with a post-hoc analysis block** — Pattern 3's data path unchanged, with four deviations that are each load-bearing. (i) The model adds its auxiliary term through `add_loss`, so the head MUST be constructed with `aggregate_backbone_losses=True`; without it the term is computed every step, discarded, and the run produces a non-topographic model with a healthy curve. (ii) Pretraining is ~one pass over a stream and Keras only validates at epoch ends, so the validation cadence IS the virtual epoch length — `fit(steps_per_epoch=eval_every_steps)` — which makes the step count `eval_every_steps * epochs` rather than one pass, so `--train-steps` pins it and both numbers are logged before the first step. (iii) The run assembles its OWN callback list rather than calling `create_nlp_callbacks`, because that helper unconditionally installs `keras.callbacks.EarlyStopping` and this pattern's stopping rule is a different question. (iv) The run's real output is a paper figure set from `dl_techniques.metrics`, not a single monitor, and the reported loss must be SPLIT so a rising curve can be attributed | TopoLM (`topolm/`): `SpatialSmoothness` taps on both branch outputs, `ConsecutiveIncreaseEarlyStopping` for the paper's three-consecutive-increases rule, `SpatialLossLogger` for the task/spatial split, and an evaluation block running t-map -> joint BH-FDR -> cluster growing -> Moran's I | `src/train/topolm/common.py` + `src/train/topolm/pretrain.py` | `val_loss` |

**`src/train/mothnet/` fits none of the eight patterns either**, and is the reference for
*hand-assembling figures from `dl_techniques` plugins*. Its training loop is its own
(`train_hebbian`), and it is the only trainer that registers library classification
templates against a `ClassificationResults` it builds itself. Copy it for a bespoke loop
that wants the library's figure families: `_visualize_or_warn` is the house answer to
`visualize()` swallowing errors and returning `None`, and the trainer derives
`class_names` from the classes **actually present** rather than a hardcoded range —
required, because `ConfusionMatrixVisualization` calls sklearn's `confusion_matrix`
without a `labels=` argument (decision D-007). Its `final_{classification_report,
per_class_analysis,error_analysis}.png` trio renders on the LAST epoch only, reusing the
object already in hand rather than re-predicting; `roc_pr_curves` is deliberately absent
because MothNet carries no `y_prob`. It defines its own `run_model_analysis` — a local
four-argument function, NOT `train.common`'s five-argument one, which also takes
`test_data`.

**`src/train/hkan/` fits none of the six patterns.** HKAN is a tabular regression model whose default
training is a closed-form, layer-by-layer least-squares fit (`HKAN.fit_closed_form`), with stock `fit()`
as an option (`--training-mode closed_form | backprop | closed_form_then_backprop`). Copy it for a model
trained without an epoch loop: a purpose-built parser, `refuse_existing_run` + `attach_run_log` +
`write_summary_json`, repeats with derived seeds, median and IQR in `results_summary.json`. Its README
lists the flags, the per-mode run directory and the measured runs.

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

> **A failed run does not exit 0, and that includes a failure one call below `main`.** A broad
> `except` (bare, `Exception`, `BaseException`) anywhere under `src/train` re-raises, exits
> non-zero, or records the failure and fails the run after the rows and the report are written
> (D-095). Two kinds may log and go on: a handler around an optional artifact (a figure, a log
> line, a best-effort extra), which must log the exception and must not guard `fit`, `evaluate`,
> a model save or load, `export` or a `run_io` writer; and a named exemption with a reason (a
> fallback value, a cleanup, an optional import). Both are listed in
> `tests/test_train/test_a_broad_handler_outside_main_is_classified.py`; an unlisted handler
> fails that census. Handlers inside a `main` are censused by
> `tests/test_train/test_a_failed_trainer_run_does_not_exit_zero.py` (D-176, D-183, D-184 of
> plan-2026-10-14T062616-81938e81).

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
| `build_headed_model_from_encoder(model_cls, encoder_path, task_config)` | The fine-tuning model for any task (an `NLPTaskConfig` or a dict of its fields); the bert, fnet, tree_transformer and `bert/wikipedia` sentiment scripts pass `sentiment_task_config(num_classes, dropout_rate=0.1)`. `model_cls.from_config` over the HEADLESS encoder checkpoint's own config with `include_top=True` (nothing about the architecture is restated), then `load_pretrained_weights`. A failed load **raises** — never a random-init fallback. The head is the model's own `task_head` (its default pooling, the finalized output); the saved artifact reloads as the model class (`BERT`/`FNet`/`TreeTransformer`), not a Functional graph |
| `set_backbone_trainable(model, trainable)` | The two-stage freeze: every layer but the one NAMED `task_head` is set by loop. **Never** `model.trainable = False` + `task_head.trainable = True` (leaves 0 trainable weights) and never `model.trainable = True` to unfreeze after a per-layer freeze (a no-op) |
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
    analyzer_on_train_begin=True, analyzer_first_epoch=True,
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

## Image classification trainers: required analysis and visualizations

Every image classification trainer follows this pattern. Reference implementations:
`src/train/convnext/common.py` (v1/v2) and `src/train/cliffordnet/train_cliffordnet.py`. Both use
only shared helpers; do not write per-trainer plotting code.

**1. Model analysis schedule (the standard).** The epoch analyzer (data-free weight and spectral
analysis, `EpochAnalyzerCallback` through `create_callbacks`) runs

- once before training starts, into `epoch_analysis/epoch_000/` (the untrained model),
- after epoch 1, into `epoch_analysis/epoch_001/`,
- then every N epochs from N, into `epoch_analysis/epoch_NNN/`.

`create_callbacks` gives the first two by default (`analyzer_on_train_begin=True`,
`analyzer_first_epoch=True`); N is `analyzer_epoch_frequency`, and `analyzer_start_epoch=N` puts the
snapshots on N, 2N, ... (the default start of 1 would give 1, N+1, ...). Expose N as
`--epoch-analysis-every N` (default 0 = off; AlexNet and CliffordNet are the precedents) and pass
`include_analyzer=N > 0`. A model that is not built when training starts skips the `epoch_000`
snapshot with a warning; any analysis failure on a built model raises out of `fit` (D-117), so a
run never exits 0 with a snapshot missing. Pass `analyzer_on_train_begin=False,
analyzer_first_epoch=False` only to reproduce an older schedule. The full `ModelAnalyzer` pass
(`run_model_analysis`) still runs once after `fit`, on the best weights.

**2. Visualizations, into `<run>/visualizations/`.**

| File | Written by |
|---|---|
| `training_dashboard.png` | `TrainingDashboardCallback(out_path, baseline_fn=..., title=...)`, appended LAST; redrawn after epoch 1, about 20 times per run, and at the end; includes the epoch-0 baseline and the learning rate |
| `confusion_matrix.png`, `per_class_metrics.png`, `confidence_calibration.png`, `misclassifications.png`, `classification_report.json` | `run_summary.write_classification_figures(vis_dir, x_test, y_test, probs, class_names, ...)` after the best weights are loaded |

Rules that each cost a run once: insert `LearningRateLogger(at_epoch_start=True)` at the FRONT of
the callback list so `lr` reaches `training_log.csv` and the dashboard; pass the dashboard a
`baseline_fn` that measures the untrained model the way the trainer's own loss guard does; give
`write_classification_figures` PROBABILITIES (softmax of the logits for a logits head) in the
order of `y_test`, and `mean`/`std` only when `x_test` is standardized; `get_class_names(dataset,
num_classes)` supplies the labels. A figure that fails logs a warning and does not abort the run;
read `failed` in the helper's return value.

**Not yet conforming.** From a grep for `write_classification_figures` and
`TrainingDashboardCallback` (not an audit), these image classification trainers do not follow the
pattern yet: `alexnet`, `beit/train_classification.py`, `coshnet`,
`energy_transformer/train_classification.py`, `mobilenet`, `resnet`, `vit`, `head_comparison`,
`som_nd_soft/train_cifar.py`, `cliffordnet/train_downsampling_techniques.py`. `power_mlp` already writes the
figures and gets the schedule from the defaults whenever its `epoch_analysis` option is on. Bring a trainer up to the pattern when you touch it, and delete it
from this list in the same commit. No test enforces this list.

## What lives in `train.common`

| Symbol | Notes |
|---|---|
| `setup_gpu(gpu_id)` | Memory growth + device selection. Every script supports `--gpu` and calls `setup_gpu(args.gpu)`: it sets `CUDA_VISIBLE_DEVICES` for a specific GPU, or enables memory growth on all of them when `None`. When `CUDA_VISIBLE_DEVICES` is already set, `gpu_id` counts WITHIN that list (`CUDA_VISIBLE_DEVICES=1 ... --gpu 0` runs on physical GPU 1; `--gpu 1` there raises `ValueError`); only when it is unset does `gpu_id` name a physical GPU. `visible_device_for(gpu_id, visible)` is the one rule (D-110). `train/common/gpu.py` is the ONLY module that writes `CUDA_VISIBLE_DEVICES` (census in `tests/test_train/test_common_setup_gpu_respects_visible_devices.py`, D-116): a launcher resolves `--gpu` once with `visible_device_for` (or `child_visible_devices` when it has its own default GPU) and builds each child's environment with `env_with_visible_devices(value)`, never `str(gpu_id)`; a process that must stay CPU-only calls `hide_gpus()`, which returns the value it replaced |
| `create_callbacks(...)` / `create_nlp_callbacks(...)` | See above |
| `resolve_monitor_mode(monitor, mode=None)` | The ONE producer of a checkpoint-selection direction |
| `create_base_argument_parser(description, default_dataset)` | Only for scripts using `load_dataset()` |
| `create_learning_rate_schedule(lr, type, epochs, steps_per_epoch)` | Cosine / exponential / constant. Defined in `dl_techniques.optimization.schedule` and re-exported here; both paths resolve to the same object |
| `load_dataset(...)` / `get_class_names(...)` | See Data loading below |
| `validate_model_loading(...)` / `run_model_analysis(...)` | Round-trip serialization check; full ModelAnalyzer pipeline |
| `create_regression_results(...)` / `create_multi_model_regression(...)` / `create_timeseries_results(...)` | The constructors for `RegressionResults`, `MultiModelRegression` and `TimeSeriesEvaluationResults`. Every plugin in those two families type-checks its input with `isinstance`, so **without these, six plugins were unreachable by construction** — not merely unregistered. Flatten and length-check the prediction pair; the timeseries builder promotes a bare 1-D array on either axis to a batch of ONE (consistent across both axes, so one call means the same thing whichever array it touches) and refuses quantiles without `quantile_levels`, which the plugin needs to label the bands. **No trainer calls these yet** — they are available infrastructure; see the figures section below. Sibling of `create_classification_results`. Tested by `tests/test_train/test_common_evaluation_results.py` |
| `run_summary.run_data_free_analysis(model, sample, labels, history, model_name, run_dir, enabled=True)` | The end-of-run weights + spectral analysis for trainers whose data is not per-image labels (the ConvUNeXt denoiser and segmentation trainers). Never raises; the returned `analyzer` block (`{status, analyzers, error, path, seconds}`) is read back from `model_analysis/analysis_results.json` by `read_data_free_analysis_status`, not taken from the analyzer's return value. After an `ok` read-back it deletes the library's empty `model_analysis/summary_dashboard.png` and adds `removed: ["summary_dashboard.png"]` to the block (absent otherwise), and only while `model_metrics`, `calibration_metrics`, `confidence_metrics` and `weight_pca` of the returned result are all empty: a dashboard a panel has data for is kept. `data_free_analysis_config()` is the one `AnalysisConfig` behind it. The analyzer call runs inside `threadpool_limits(limits=limits)`, scoped to that call; `limits` defaults to `DATA_FREE_ANALYSIS_BLAS_THREADS` = 4 (12 BLAS threads oversubscribed the host: 14.7 to 21.0 s against 6.1 to 6.6 s at one thread on one saved model, D-046; four against one: AlexNet 177.8 s against 381.7 s, a 1.9M-parameter ConvUNext 5.7 s against 5.8 s, plan-2026-10-14T062616-81938e81 D-243). Do not write a per-trainer copy |
| `run_summary.summary_head(config, run_dir, *, params, steps_per_epoch, devices)` | The summary keys the bfunet trainers (ConvUNeXt denoiser, unet, bfcnn) and the ConvUNext segmenter share: the config attributes named in `SUMMARY_CONFIG_KEYS`, `epochs_requested`, the sizes, the three device keys and `image_range` `[0.0, 1.0]` (the CLEAN images' range, not the noisy input, which `--no-clip` leaves unclipped; not `config.json`'s `data_range` string). A new shared fact is added there once; the ConvNeXt reference keeps its own `_summary_head` |
| `TrainingDashboardCallback(..., best_key=None)` / `render_training_dashboard(..., best_epoch=None)` (`classification_viz`) | Opt-in best-epoch marker: pass `best_key="val_loss"` to draw a dashed line and star at the best value so far (the lowest, or the highest when `resolve_monitor_mode` says `max`, e.g. the segmentation trainer's `--monitor val_miou`). `None` draws nothing and leaves the PNG unchanged. A history that carries `miou` / `val_miou` columns (the segmentation trainer's) gets an extra mIoU panel; a history without them draws the same panels and the same PNG bytes as before |
| `discover_megadepth_pairs(root, max_files)` / `split_megadepth_pairs(root, train_split, max_train_files, max_val_files)` / `MegaDepthDataset(...)` | MegaDepth RGB+depth pipeline. A `max_files` cap is taken evenly across the scenes; `split_megadepth_pairs` is scene-disjoint (no scene in both splits) |
| `compare_runs(a_dir, b_dir, labels, output_dir)` | Two-run comparison; also `python -m train.common.compare_runs A B`. Emits `comparison.md` + curve PNGs |
| `StepCheckpointCallback(...)` | Step-indexed CSV logging, rolling `.keras` checkpoint window, optional periodic ModelAnalyzer, step-loss plots. Pass an external `step_counter` for resume setups. Use instead of a per-trainer step-checkpoint class. A failed `final.keras` save raises; a failed step save is retried at the next interval and `on_train_end` raises naming it after `final.keras` is written (D-183) |
| `set_seeds(seed)` | Canonical seeding (`PYTHONHASHSEED` + `random` + numpy + `keras.utils.set_random_seed`). Every entry point that trains takes `--seed`, defaulting to `DEFAULT_SEED` (42, in `train/common/seed.py`; a module that does not import `train.common` spells it as the literal 42, and the census holds the two equal), and passes it here, never a literal; a script that only evaluates, exports or prepares data is named with its reason instead. `tests/test_train/test_every_trainer_seeds_through_the_shared_helper.py` holds both by an AST call-graph census (D-247 of plan-2026-10-14T062616-81938e81). A fixed seed for a fixed evaluation set (`nam/train_dfsa_ste.py`'s quick eval, the TabM synthetic example) stays, on a private generator when the training stream draws from the same global module (D-248) |
| `save_config_json(...)`, `prepare_run_dir(config, output_dir=None)` | Run-directory preamble. Raises on failure (`OSError`, or `TypeError` for a value JSON cannot serialize; nothing is written then), never call it inside `try/except`; the same census checks every call of every `run_io` writer (D-177 of plan-2026-10-14T062616-81938e81) |
| `save_training_history_json(history, output_dir)` | Raises on failure (`OSError`, or `TypeError`/`ValueError` for a value `float()` rejects), so a run whose history was not written does not exit 0. Call it AFTER the model save, never inside `try/except`: a failed write then never costs the weights. `tests/test_train/test_a_history_write_follows_the_model_save.py` checks every call site by AST (a save before, none after on the same path, no enclosing `except`); a diverged run that raises before its final save writes the history just before its `raise`. Until D-174 of plan-2026-10-14T062616-81938e81 it warned and returned `None` |
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

## Figures: what `dl_techniques.visualization` offers a trainer, and how to call it safely

The library ships 25 plugin templates. **Prefer the house plotting modules in
`train.common` for the figure families they already own**, and reach for a library
plugin when you need a family nothing local draws (regression residuals, QQ, forecast
bands). `train/common/classification_viz.py` has **outgrown** the library's
classification plugins — `plot_confusion_matrix` bounds the image at `0.62 * n` inches
per panel because 100 classes previously produced a 15585x7571 px file, uses `PowerNorm`
for narrow count ranges, drops cell text below 0.5%, and adds a top-confused-pairs panel
the library plugin does not have. Rewiring a Pattern-1 trainer from the house module to
`ConfusionMatrixVisualization` would regress its figures.

**A plugin needs BOTH halves to be reachable**: a registration, *and* something that
constructs the container its `can_handle` type-checks against. The missing half is almost
always the container. `create_regression_results`, `create_multi_model_regression` and
`create_timeseries_results` exist for exactly that reason — before them, nothing in the
tree constructed `RegressionResults` or `TimeSeriesEvaluationResults`, so the five
regression plugins and `ForecastVisualization` were unreachable by construction and no
`register_template` call could have fixed it. They are reachable now; **as of this commit
no trainer calls them**, so treat them as available infrastructure rather than a live path.

| Want | Do |
|---|---|
| training curves | `generate_training_curves(history, dir)` — the house path, already used by `callbacks.py` and `timeseries.py`. Registers an *instance*. |
| classification dashboard / confusion matrix | `render_training_dashboard` + `plot_confusion_matrix` in `train/common/classification_viz.py`. Pattern 1's default. |
| per-class breakdown, error analysis | register `classification_report`, `per_class_analysis`, `error_analysis` against a `ClassificationResults` |
| regression residuals / QQ / prediction-error | `create_regression_results(...)`, then `prediction_error`, `residuals_plot`, `residual_distribution`, `qq_plot` |
| forecast bands | `create_timeseries_results(...)`, then `forecast_visualization` |
| a domain figure nothing provides | subclass `VisualizationPlugin` in the trainer and `register_template` it — `train/kan/train_kan.py` is the worked example |

Four traps, all measured:

- **`visualize()` swallows exceptions and returns `None`** (`visualization/core.py:508-512`).
  A plugin that raises leaves you with a **missing file and no exception**. Callers that do
  not check the return value will report a successful run with a missing figure. Check the
  return, or assert the file exists — `train/mothnet`'s `_visualize_or_warn` is the house
  pattern for exactly this, and it logs which figure was lost.
- **A template key must equal the class's own `name`.** `visualize` registers the instance
  under `plugin.name` and then re-reads `self.plugins[plugin_name]`, so a disagreeing key
  misses that lookup with a bare `KeyError`. `ErrorAnalysisDashboard` is named
  `"error_analysis"`, **not** `"error_analysis_dashboard"`.
- **`ROCPRCurves` yields nothing without `y_prob`.** Its per-model loop skips such a model,
  its `legend()` then raises, and `visualize` swallows it — so registering it on a
  probability-free model produces an absent figure and nothing else. Every other
  classification plugin degrades visibly instead.
- **Never call `ax.legend()` on unlabelled artists**; this repo runs
  `filterwarnings = ["error::UserWarning", ...]`, so that is a test failure.

For weight transfer use `dl_techniques.utils.weight_transfer.load_weights_from_checkpoint`
(layer-by-layer, from a saved `.keras` model). Not `model.load_weights(..., by_name=True)`, which
Keras 3.8 supports only for legacy `.h5`/`.hdf5` and rejects on `.keras` with
`ValueError: Invalid keyword arguments: {'by_name': True}`.

Keep local: model creation and compilation, domain-specific losses, custom argparse where
`create_base_argument_parser()` does not fit, and the training summary.

## Data loading

`load_dataset()` covers mnist, cifar10 and cifar100 only, returning
`(x_train, y_train), (x_test, y_test), input_shape, num_classes` (numpy). For ImageNet-1k use
`train.common.imagenet` (flags via `add_imagenet_arguments`, splits via `load_imagenet_splits`,
an in-memory val head via `eval_head`). For text use
`train.common.nlp`; for MegaDepth RGB+depth pairs `train.common.megadepth` (yielding
`(rgb, y_true)` with `y_true = [disparity, mask]` on the last axis, the disparity per-map
normalized to `[0, 1]`). For anything else write local
loading — do NOT force it through `load_dataset()`.

ImageNet-1k from the extracted `train/<wnid>/` + `val/` layout: `dl_techniques.datasets.vision.imagenet2012.load_imagenet2012`, wired through `train.common.imagenet` by `train_resnet`, `train_vit` and `train_cliffordnet` (`--imagenet-root` / `--devkit-dir`; val labels need the ILSVRC2012 devkit). `train.common.make_imagenet_filesystem_dataset` remains for `alexnet`.

## Scripts that don't use `train.common` callbacks (and why)

| Script | Reason |
|--------|--------|
| `bert/wikipedia/*` | MirroredStrategy distributed training, BackupAndRestore for fault tolerance |
| `blt` | Multi-stage pipeline (entropy pretraining + main training), class-based trainer |
| `yolo12/train_multitask` | Per-task callbacks, losses and visualization |
| `tabm` | Custom `TabMTrainer`, not standard Keras `fit()` |
| `lewm` | Dict-shaped inputs (`{"pixels", "action"}`) + add_loss-only training; `EpochAnalyzerCallback` doesn't understand either |
| `hkan` | Three training modes, one of them closed-form least squares with no epochs; `create_callbacks()` always adds early stopping, which would stop repeats at different epochs. The backprop modes hand-assemble `ModelCheckpoint` + `CSVLogger` |

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
