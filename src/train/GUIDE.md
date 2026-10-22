# Trainers: Lessons From Past Issues

Defect patterns found in `src/train/` entry points, and the rule or guard that now prevents each.
Model-side patterns stay in [`src/dl_techniques/models/GUIDE.md`](../dl_techniques/models/GUIDE.md)
(§ 10 points here for the trainer half). How to write a trainer, the patterns to copy and the
`train.common` API are in [`AGENTS.md`](AGENTS.md).

Every entry gives the mechanism, how it was detected, the guard or rule that now prevents it, and
the decision (D-NNN in `plans/plan-2026-10-14T062616-81938e81/decisions.md`) holding an example.
Test files named without a directory sit in the directory of the previous path in the same entry.
`tests/test_the_guides_cite_only_paths_that_exist.py` checks that every path cited here exists.

## 1. Entry points and run outcome

Moved from `models/GUIDE.md` § 10.3 (iteration 1 of that plan).

- **Entry points and default paths no test runs.** Trainers died before step one: a tokenizer in
  `tf.data.map`, `astype` on a symbolic tensor, a stale attribute on a default path the test
  skipped. *Detected:* running each trainer. *Rule:* test the default path. *Guard:*
  `tests/test_train/test_cliffordnet/test_the_trainer_entry_point_pretrains_both_towers.py`,
  `tests/test_train/test_alexnet/test_the_cifar_pipeline_builds_a_batch.py`. *Example:* D-034,
  D-069, D-098. Refines `models/GUIDE.md` § 8.
- **A failed run that exits 0.** A swallowed training error, save, history write, evaluation,
  analysis, reload check or Ctrl-C ended with the code of a finished run. *Detected:* stubbing each
  step to raise. *Guard:* `tests/test_train/test_a_failed_trainer_run_does_not_exit_zero.py`
  (with a census of `except KeyboardInterrupt`),
  `tests/test_callbacks/test_a_failed_epoch_analysis_fails_the_epoch.py`. *Example:* D-095,
  D-102, D-111, D-117, D-118.
- **`--output-dir` ignored, and tests writing into `results/`.** Callbacks went to a second
  directory; a RED proof created a run under `results/`. *Detected:* the `results/` name set grew.
  *Guard:* `tests/test_train/test_a_trainer_writes_under_its_output_dir.py`,
  `tests/test_the_results_session_guard_fails_a_session_that_writes.py`. *Example:* D-048, D-094.
- **`--gpu` overwrote `CUDA_VISIBLE_DEVICES`.** `CUDA_VISIBLE_DEVICES=1 ... --gpu 0` ran on GPU 0,
  launchers forwarded an index their child could not resolve, and a `--gpu 1` default cannot fit
  a one-entry export. *Detected:* the Step 13.5 smokes ran on the RTX 4090. *Rule:*
  `train/common/gpu.py` is the only writer; `--gpu` counts within an export. *Guard:*
  `tests/test_train/test_common_setup_gpu_respects_visible_devices.py`,
  `tests/test_train/test_launchers_select_their_gpu_by_the_d110_rule.py`. *Example:* D-110,
  D-116, D-118.
- **`steps_per_epoch` on a finite dataset.** Keras 3.8 keeps one iterator across epochs, so 3
  batches at 2 steps train 2, 1, 2. *Detected:* counting steps per epoch. *Rule:* repeat the
  dataset. *Guard:* `test_every_epoch_trains_its_steps_per_epoch_batches` in
  `tests/test_train/test_clip/test_the_trainer_entry_point_trains_end_to_end.py`. *Example:* D-036.
- **A measurement that measures nothing.** Depth metrics read 0.0 because a wrapper cut the mask
  channel; validation crops changed every epoch. *Detected:* exact zeros in the CSV log. *Guard:*
  `tests/test_train/test_depth_anything/test_the_compiled_depth_metrics_see_valid_pixels.py`.
  *Example:* D-100, D-109.
- **One fact in several places.** Three CIFAR-100 statistics pairs, three positive-int
  validators that refused different values. *Detected:* re-deriving the values. *Guard:*
  `tests/test_train/test_common_cifar100_statistics_have_one_home.py`,
  `tests/test_utils/test_the_positive_int_rule_has_one_home.py`. *Example:* D-088, D-112.

## 2. Patterns from iteration 2

- **A failed history write cost the weights.** `save_training_history_json` warned and returned,
  and 14 trainers wrote the history before the model save, so a write that raised would end a run
  with nothing on disk. *Rule:* every `train.common.run_io` writer raises; the history is written
  after the model save, never inside `try/except`. *Detected:* an AST census of every call site.
  *Guard:* `tests/test_train/test_a_history_write_follows_the_model_save.py`,
  `tests/test_train/test_common_run_io.py`. *Example:* D-174, D-177.
- **A broad handler in `main` let a failed run exit 0.** 80 handlers in a `main`, 8 of which
  returned normally after a failure (a sweep cell became a NaN row, a failed example was logged).
  *Rule:* in `main` a broad handler raises or exits non-zero; a sweep records each failed cell,
  writes its report, then raises naming them. *Detected:* an AST census of every handler.
  *Guard:* `tests/test_train/test_a_failed_trainer_run_does_not_exit_zero.py`. *Example:* D-176.
- **A broad handler below `main` that swallowed a failure.** 26 handlers outside `main` swallowed
  a failure (a stream error ended a corpus, a failed evaluation or save became a log line). *Rule:* every broad handler outside `main`
  raises, exits, records a failure the run then fails on, or is named in the census as optional
  output (it must log the exception and its `try` may call no fit, evaluate or save) or as an
  exemption with its reason. *Detected:* the same census, extended. *Guard:*
  `tests/test_train/test_a_broad_handler_outside_main_is_classified.py`,
  `tests/test_train/test_a_failure_below_main_is_not_swallowed.py`. *Example:* D-180, D-183,
  D-184.
- **A batch read the same file twice.** The OmniPoint loaders advanced their index by two per
  sample, so a batch held every other pair, and the shared refill then tried positions modulo the
  file count, so a data set smaller than a batch repeated files. *Rule:* one shared `fill_batch` reads each file at most once per batch. *Detected:*
  counting reads per batch. *Guard:* `tests/test_train/test_common_fill_batch.py`,
  `tests/test_train/test_omnipoint/test_an_omnipoint_batch_reads_each_pair_once.py`.
  *Example:* D-193, D-199.
- **`setup_gpu` logged and continued.** A memory-growth or device selection that did not take
  effect (TensorFlow already initialised) was a warning, and 8 scripts defaulted `--gpu` to an
  index. *Rule:* `setup_gpu` raises when the selection or the memory growth did not take effect;
  `--gpu` defaults to `None` everywhere. *Detected:* stubbing TensorFlow's device list. *Guard:*
  `tests/test_train/test_common_setup_gpu_respects_visible_devices.py`,
  `tests/test_train/test_an_in_process_setup_gpu_does_not_depend_on_test_order.py`,
  `tests/test_train/test_dino/test_importing_the_trainer_leaves_tensorflow_uninitialised.py`.
  *Example:* D-234.
- **CSV logs with phantom or missing validation columns.** Keras' `CSVLogger` pads `val_` columns
  at epoch 0, so every run without validation logged `NA` columns; fixing the header at epoch 0
  then dropped the `val_` columns when `validation_freq > 1`. *Rule:* one `EpochCSVLogger`
  (`train/common/callbacks.py`); no `keras.callbacks.CSVLogger(` outside `train/common`.
  *Detected:* reading the CSV of a run without validation. *Guard:*
  `tests/test_train/test_a_run_without_validation_logs_no_val_columns.py`,
  `tests/test_train/test_an_epoch_without_validation_leaves_its_val_cells_empty.py`.
  *Example:* D-236, D-241.
- **The best checkpoint saved the wrapper, or nothing.** The CLIP trainer trains a contrastive
  wrapper and had no validation, so no best model was kept. *Rule:* one `BestSubmodelCheckpoint`
  saves the inner model on the best `val_loss`. *Detected:* loading the saved file. *Guard:*
  `tests/test_train/test_clip/test_the_best_checkpoint_is_the_lowest_val_loss_epoch.py`.
  *Example:* D-237.
- **A trainer that seeds nothing.** 26 trainers reached no seeding call, 8 seeded a literal 42
  that no flag could change. *Rule:* every entry point that trains takes `--seed` (default 42) and
  passes it to `train.common.seed.set_seeds`; a script that does not train is named with its
  reason. A fixed evaluation seed uses a private generator when training draws from the same
  global module. *Detected:* an AST call graph from each `__main__` block. *Guard:*
  `tests/test_train/test_every_trainer_seeds_through_the_shared_helper.py`,
  `tests/test_train/test_nam/test_the_quick_eval_leaves_the_training_stream_alone.py`.
  *Example:* D-247, D-248.
- **A generator dataset of unknown length that shrank every logged loss.** With a
  `from_generator` source and no `steps_per_epoch`, Keras 3.8 finds the epoch end through one
  more train or test function call that raises OutOfRange, and the loss trackers' `count`
  update still runs in it: every logged loss is `n / (n + 1)` of the mean over `n` batches. A 2-clip SAM 2 run logged 0.48 for a model whose loss was 0.95, a constant's 0.96.
  *Rule:* a source that yields a known number of records states it with
  `tf.data.experimental.assert_cardinality`; otherwise `fit` gets `steps_per_epoch` (and
  `validation_steps`) on a repeated dataset. *Detected:* `evaluate` on the live dataset against
  the same batches materialised. *Guard:*
  `tests/test_train/test_sam2/test_the_pipeline_states_its_length_and_redraws_its_prompts.py`,
  `tests/test_train/test_sam/test_the_pipeline_states_its_length_and_redraws_its_prompts.py`,
  `tests/test_train/test_sam3/test_the_pipeline_states_its_length.py`. *Example:* D-258, D-262.
- **A random draw in a `tf.data` map that repeated every epoch.** A stateful `tf.random` op in
  a `map` restarts from its seed on every new iterator once a global seed is set, so every
  epoch drew the same SAM prompts and a shuffle only moved them between records. *Rule:* draw
  from a per-record seed zipped in from `tf.data.Dataset.random(...,
  rerandomize_each_iteration=True)` (`train.sam.data.with_prompt_seeds`). *Detected:* two passes
  over the dataset compared. *Guard:*
  `tests/test_train/test_sam2/test_the_pipeline_states_its_length_and_redraws_its_prompts.py`,
  `tests/test_train/test_sam/test_the_pipeline_states_its_length_and_redraws_its_prompts.py`.
  *Example:* D-259, D-262.
