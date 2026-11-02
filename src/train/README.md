# Training

One directory per runnable pipeline (see `AGENTS.md` for the patterns, `GUIDE.md` for the defect
guards, `REPO_MAP.md` at the repo root for which trainer trains which model). Most are run as
`python -m train.<model>.train_<script>`. One is run as `python -m train`.

## `python -m train` and `train.api`

A key=value command line over strict configs. It currently has one task, `image_classification`
(families `vit`, `resnet`, `convnext_v1`, `convnext_v2`, `cliffordnet`, `cliffordnet_hier`), and the
same code is a Python function. Run it from the repo root with `.venv`, with
`MPLBACKEND=Agg` on a headless machine.

```bash
python -m train help                         # commands and tasks; imports no TensorFlow
python -m train list_models task=image_classification
python -m train image_classification         # every key of the task, its type and default
python -m train image_classification out=auto model=vit/vit_pico data.dataset=cifar10 epochs=10
```

```python
from train.api import main, train_image_classification, load_model, export
result = train_image_classification(out="auto", model="vit/vit_pico", data={"dataset": "cifar10"})
main(["list_tasks"])      # the same as the command line; returns the exit code
```

**Commands**: `help`, `list_tasks`, `list_models`, `export`, `predict`, and each registered task.
`-v` anywhere logs at DEBUG. `<command> help` prints that command's keys.

**Keys**: `key=value` tokens; a dot reaches a group (`optim_args.lr=3e-4`, `data.dataset=cifar10`);
`config=run.yaml` loads a file first and the command line wins over it. A value is read by
the field's type (`true`/`false`, `3e-4`, `null`, JSON for a list or mapping; a string is verbatim). An unknown key, a wrong type or a bad choice is refused with the list of all
problems and a "did you mean". `auto` asks for the family's recipe default.
`data.format` is `builtin` (the library's datasets), `folder` (`data.train_dir`, one directory of
images per class, optional `data.test_dir`) or `csv` (`data.train_csv`, columns `path` and `label`).

**The run directory**: `out=auto` makes `results/<timestamp>...`; an existing `out` is refused,
nothing is overwritten or deleted. A run writes `config_requested.json` (what was asked),
`config.json` (resolved), `run.log`, `training_log.csv`, `results_summary.json`, `task.json`
(class names, input shape, mean and std: what a model needs to be used without its config),
`best_model.keras`, `final_model.keras`, the figures and analysis, and `checkpoints/`
(`backup/`, `fit_state.json`) and `run.lock` for resume. `run.lock` is a kernel `flock` held for the
life of the run and released when the process ends, a SIGKILL included, so two resumes cannot both run
and a reused pid blocks nobody; the pid in the file is a label for the refusal message.

**Resume**: `resume_interrupted=true` with the same `out` and the same request continues a killed
run. It is refused for a finished run (one with a `results_summary.json`), another request, other data, a live
process holding the run, or a torn backup; `devices` and `analysis_args.eval_batch_size` may differ. The training data
order replays epoch 0's and a half-done epoch is trained again (the key's help lists the rest).
A run whose **training finished and whose reporting died** (the sidecar says `fit_finished`, there is no summary: the
figures, the analyzer, the conformal sets or the final evaluation raised, for example out of memory) is not trained
again: the same command with `resume_interrupted=true` skips `fit` and runs only the reporting, from
`best_model.keras` (the best weights), `checkpoints/backup/latest.weights.h5` (the last epoch's weights, which were
only in memory) and `training_log.csv` (read, never written: its sha256 is the same before and after). The last
epoch's weights are used only when that backup is the epoch-end copy of the epochs that ran (metadata epoch and
batch 0, optimizer step count epochs times steps); a missing, torn, mid-epoch or older copy is refused with exit 2
naming `latest.weights.h5`, nothing is changed, and the previous `.bkp` copy is never put in its place, because a
final model one epoch old would look like the final model. An existing `final_model.keras` is written over. The
summary says `reporting_resumed: true`, `epoch_times` and `fit_wall_seconds` are null (no epoch ran in that segment),
`best_checkpoint_max_abs_diff` compares `best_model.keras` with itself, and a second resume is refused. The reason a
reporting died is an `ERROR` line of `run.log` (exception type and message), and the command exits 1. For folder and CSV data the sidecar holds a fingerprint of the
files: `data.fingerprint=listing` (the default) hashes every file's relative path, class and size, in listing
order, and opens no file (a CSV whose rows were reordered splits differently, so it is another listing), so two
same-sized files swapped, or a pixel edited, go unseen; `data.fingerprint=content` also hashes every
file's bytes, streamed, which reads all the data once (a pass over the disk tree, minutes on a large one on a
spinning disk; the seconds go to `run.log`). A resume refuses a changed mode or digest, and a sidecar written before the rows were hashed in listing order
(no `data_fingerprint_scheme`).

**Figures, analysis and TensorBoard**: five keys decide what is written besides the models and the
summary. The region between the markers below is read by
`tests/test_train/test_api/test_the_knob_docs_name_real_keys.py`, which checks the config keys (and any `=value`), the
defaults in the table, the hook names and signatures, the file paths and the output file names it mentions against the
code, and the cost table against its evidence file, so a stale one of those fails the test. It does not check the
wording of a description (what a knob writes or needs) or any number that is not a cost in that table; the test's
docstring lists what is unchecked.

<!-- knobs:begin -->
| Key | Default | What it does |
|---|---|---|
| `analysis_args.figures` | `true` | Writes `visualizations/`: `training_dashboard.png` (redrawn after epoch 1, about 20 times per run, and at the end), then after the best weights are loaded `confusion_matrix.png`, `per_class_metrics.png`, `confidence_calibration.png`, `misclassifications.png` and `classification_report.json` (these five need a test split). `false` writes none of them and `ece` in the summary is null |
| `analysis_args.model_analysis` | `true` | Runs the full model analyzer once, after training, on the best weights, into `model_analysis/` (`analysis_results.json` and seven PNGs: weights, spectral, information flow, calibration, training dynamics and the summary dashboard). Needs a test split. `false` leaves `analyzer.status` at `skipped` |
| `analysis_args.epoch_analysis_every` | `0` | `N > 0` runs the data-free epoch analyzer (weights and spectral only) before training into `epoch_analysis/epoch_000/` (the untrained model), after epoch 1 into `epoch_001/`, then at epochs N, 2N, ... into `epoch_NNN/`. `0` is off. A failure on a built model stops the fit, so no snapshot is silently missing. A resumed run does not redo `epoch_000` and `epoch_001` |
| `analysis_args.conformal_alpha` | `null` | A level strictly between 0 and 1 (0.1 aims at 90% coverage) writes `conformal.json` and `conformal_model.keras` (outputs logits, probabilities and set membership), calibrated on the test split. `null` skips it and the summary's `conformal` is null. A level with no test split is skipped with a note |
| `analysis_args.eval_batch_size` | `"auto"` | Images per pass of every evaluation and prediction that is not training: the in-memory (CIFAR, MNIST) validation and test splits, the figures' predictions, the model analyzer's predictions (when below its own 32) and the final-model load check. `auto` is the recipe's value, and 256 where the recipe says nothing, so a default run evaluates exactly as before. The cliffordnet recipes on ImageNet and Imagenette set the batch they train at (8 for `cliffordnet`, 16 for `cliffordnet_hier`), because 256 images of 224 px ask 18 GB. ImageNet, Imagenette, folder and csv splits are evaluated at the training `batch_size` and ignore it. It changes no weight, and it may differ when a run is resumed |
| `checkpoint_args.backup_every_steps` | `null` | `N >= 2` (and below the steps of an epoch) also writes the resume backup after every N-th batch inside an epoch, so a killed epoch loses at most N batches of weights; `null` backs up at each epoch end only. A resume trains the killed epoch again from its first batch on the weights of the last copy and puts the optimizer's step count back to the epoch's start. Each copy writes the weights and optimizer state twice (the previous copy is kept); the cost per save is in the backup table of the ImageNet section |
| `logger_args.tensorboard` | `{}` | `{}` writes TensorBoard event files into `tensorboard/` (`train/` and `validation/`); `null` turns them off; a dict with keys is refused, because no TensorBoard option is wired |

Measured cost, one run each, so a difference of a few seconds is inside the noise. Setup: `vit/vit_pico` on 240 fake
CIFAR-10 images (240 test images), 3 epochs, batch 16, CPU with one thread. Each row turns on the one setting, against a
base run with figures, model analysis and epoch analysis off and TensorBoard on (the `base` row, wall seconds of the whole
call); the TensorBoard row is the base run against one with TensorBoard off. The cost grows with the number of layers and
test images. The durations of the knobs are stated only in this table, and `test_the_knob_docs_name_real_keys.py` recomputes every row from
`tests/test_train/test_api/knob_cost_evidence.json` (a copy of the measurement in the untracked
`plans/plan-2026-10-31T180033-2faf890a/scratch_step13/measure_out.txt`; the test does not re-measure it).

<!-- costs:begin -->
| Setting | Extra wall seconds | Output, MB |
|---|---|---|
| `base` | 68.0 | - |
| `analysis_args.epoch_analysis_every=1` | 29.3 | 6.4 |
| `analysis_args.model_analysis=true` | 23.4 | 3.1 |
| `analysis_args.figures=true` | 6.9 | 0.4 |
| `analysis_args.conformal_alpha=0.1` | 4.2 | 32.6 |
| `logger_args.tensorboard={}` | 3.0 | 0.2 |
<!-- costs:end -->

`epoch_analysis_every=1` writes four snapshots (`epoch_000` to `epoch_003`). The `conformal_model.keras` of the conformal
row is a full copy of the model. The `base` row is the whole call, not an extra, and the TensorBoard row is inside the
noise of one run each.

**A failed analysis is named, with its reason.** The analyzer isolates a failing analysis and carries on, so
a run can write a normal-looking `model_analysis/` after one of its parts failed. For this task the summary's
`analyzer` block gains `summary.analyzer.failed_analyses`, an object that maps each name to its reason, only when
it is not empty (a clean run writes the same keys as before), and `run.log` carries a WARNING. A name is an
analysis that was attempted and left no result (`weights`, `calibration`, `information_flow`,
`training_dynamics`, `spectral`), a figure set that raised (`visualizer:calibration`, `visualizer:summary`, and so
on), `analysis` (a failure before any analyzer ran) or `model_analysis` (the whole call raised and returned
nothing). The reason is `ExcType: message` of the swallowed exception, or a sentence for an analysis that ran and
produced nothing (spectral on a model whose layers all have fewer than `spectral_min_evals` = 10 eigenvalues
says so). When any analysis is named, `summary.analyzer.status` reads `partial` instead of `success`; `error`,
`missing` and the other statuses are unchanged. The analyzer writes `model_analysis/analysis_results.json` before
it draws any figure, so a figure that crashed loses no computed number; the file gains an `analysis_failures`
object only when something failed. In the analyzer's summary dashboard a missing ECE or Brier prints `n/a`, not 0.0.

**For a task author**: the trainer calls two optional hooks and one required one for this output. They are
documented at their definitions in `api/trainer.py`: `TaskTrainer.figure_callbacks(config, data, vis_dir)`
returns per-epoch figure callbacks (called only while `analysis_args.figures` is true, placed after the
epoch log line and before the dashboard); `TaskTrainer.run_analysis(config, model, test_data, history, experiment_name, run_dir)`
returns what the model analyzer returns, `None` when it raised; `TaskTrainer.evaluate_and_report(config, data, fit)`
is the required end-of-run sequence that calls them. `src/train/AGENTS.md` has the rules.
<!-- knobs:end -->

**Predict**: `python -m train predict model=<run dir or .keras> data=<image or directory> out=pred.json`
writes a new JSON list of `{"path", "labels", "scores"}`, every class best first. A directory is
searched recursively. `out` must not exist. **On ImageNet the `labels` are WordNet ids** such as `n01806143`, not
words, and so are the `class_names` of `task.json`. The loader sorts the 1,000 ids and takes the position as the class
index; the devkit's `meta.mat` maps an id to its words:

```python
from scipy.io import loadmat
rows = [r for r in loadmat("<imagenet_root>/devkit/meta.mat", squeeze_me=True)["synsets"] if int(r["num_children"]) == 0]
words = {str(r["WNID"]): str(r["words"]) for r in rows}      # "n02012849" -> "crane"
```

The names stay ids because the words repeat, and the predictor refuses a record whose class names repeat (`'class_names'
repeats ...`). Measured on the real devkit (`meta.mat`, 1,000 leaf synsets): 998 distinct first words; `crane` is two classes
(`n02012849`, the bird, and `n03126707`, the machine; their whole `words` strings are equal too) and `maillot` is two
(`n03710637` and `n03710721`, whose strings differ: `maillot` and `maillot, tank suit`). Use the id as the key. Through the Python `Predictor`, a colour array or PIL image given to a
grey model (MNIST) is made grey on its 8-bit pixels with the loader's own rule, `(9797 R + 19234 G + 3737 B) >> 15`
(libpng's weights as `tf.io.decode_image(channels=1)` uses them), so a PNG gives the loader's pixels exactly (checked
within 1e-5, measured 0). **Measured limit:** a JPEG given as a path is read by the loader's decoder and is exact, but
a JPEG that was already decoded to an array or a PIL image has lost the Y plane the decoder takes its grey from, and
PIL's and TensorFlow's JPEG decoders differ, so its grey is off the loader's by up to 14 levels of 255 on random
colour noise at quality 75 (mean 0.56; 13 and 1.15 at quality 95) and by 1 to 2 levels (mean 0.5 to 0.6, about
0.002) on smooth images: above 1e-5, and not removable from the decoded pixels.
A 16-bit grey PIL image (`I;16` and its byte-order variants) is converted as the decoder reads the same file, each
sample's high byte, and is equal to the loader on it; PIL's `I` and `F` (32-bit integers and floats) are refused with a
`ConfigError` naming the mode, because the loader reads no file of that depth. A 16-bit RGB or RGBA PNG opens in PIL as
8-bit, which cannot be told from an 8-bit file, so for a grey model it is within 1 level of 255 of the loader (measured; a
path is exact). `load_model` refuses, at load, a `task.json` whose `task` is not `image_classification`, whose family or
variant is not the model's, or whose `input_shape` the model does not take, whose `mean` is outside 0 to 1 or whose `std` is
outside 1e-6 to 1 (the images are standardized from the range 0 to 1; a `mean` of 1e30 used to predict NaN without an error),
and a `.keras` zip with no `config.json` or no weights, each with a `ConfigError` naming the problem. **Limit:** the class
names are checked for their count and for repeats, never for their order, so a `task.json` whose names are a permutation of
the right ones loads and reports the wrong name for the swapped classes; nothing in the file or the model can tell. A run whose `model_args` change a key of its variant's table
(`depth` of `cliffordnet/nano`, `blocks_per_stage` of `resnet/resnet18`, `depths` of a convnext) records those keys in an
optional `variant_overrides` object of `task.json`, written only when there is one; `predict` and `export` then compare
the model with the variant plus those changes (and `export` copies the object into `<out>.task.json`). A ViT's `scale`
cannot be overridden: choose the variant (`model=vit/vit_tiny`); the request is refused before the run directory is made.

**Export**: `python -m train export model=<run dir or .keras> out=<path> format=keras|saved_model|tflite|onnx`
writes one new file (a directory for `saved_model`) and `<out>.task.json` beside it, and prints
only the path on stdout. Input contract: float32 `(batch, height, width, channels)` pixels scaled
to `[0, 1]` then standardized, `(x - mean) / std`, with the mean and std in `<out>.task.json`;
the output is raw logits, one per class name in that file. Other pixel ranges give numbers and
no error. A `tflite` file of a CliffordNet needs the Flex delegate (a WARNING says so). The check `export` makes of such a
file runs the Keras model on the CPU, as the interpreter does, so a GPU machine compares CPU with CPU (the tolerance is
unchanged: 1e-4 of the logits' scale, and a file that really differs is still refused). **Serving on a GPU is a different
computation from the CPU's**: it differs from the CPU's by about 1e-3 in the logits. Measured on the ImageNet gate's `cliffordnet/nano`, the same weights run on the GPU and on the CPU: up to 1.1e-3 on logits of scale 3.6 for 16 real validation images (5.8e-4 on scale 3.0 for 16 noise images), argmaxes equal on all 16 of each. No export format changes that.

**Exit codes**: 0 success; 2 a request that cannot be honoured (bad key, value, path, data,
resume), found before the run directory is claimed whenever it can be seen without loading the data (the few that need
the loaded data, such as a `checkpoint_args.backup_every_steps` not below the steps of an epoch, come after the claim and
say so), and before TensorFlow is imported only for the checks the command line makes itself (an unknown key, a value out
of range, a missing path of `predict` or `export`: 0.04 s). Everything that needs the model name or the data comes after
the import, because naming a model imports its family's module and that imports TensorFlow (measured, one run each, no
GPU: a missing `data.imagenet_root` exits 2 after 3.2 s, a bad `metric_args.watch_metric` after 4.0 s, a resume of a missing
run after 3.9 s). Checking the model name without importing the family was judged not worth the registry change. 1 a run
that finished without reaching its
success threshold, or any unexpected failure (a traceback). `data.dataset=imagenet` and `imagenette` look at
the disk before the run directory is made, with path tests only: a missing `data.imagenet_root`, a root
without `train/` or `val/`, a devkit (`data.devkit_dir`, or `devkit/` under the root) without `meta.mat` or
`ILSVRC2012_validation_ground_truth.txt`, and no `imagenette/320px-v2` records under `$TFDS_DATA_DIR` (or
`~/tensorflow_datasets`) each exit 2 with a message that names the key and the path looked at.

**Recipes on ImageNet and Imagenette**: each of the six families has an entry for `imagenet` and for
`imagenette` in its recipe, so the notice `No recipe overlay for <family> on <dataset>` is printed only for a
dataset a family has no entry for (`mnist` for `vit`, `resnet` and both cliffordnet families). What a run on one of
these pairs resolves to, with every value `auto`:

| Family | Dataset | Epochs | Batch | Optimizer | Peak lr | Weight decay | Warmup epochs | Augment | Smoothing | Image size | Patience | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vit | imagenet | 50 | 128 | adamw | 3e-4 | 0.05 | 5 | none | 0 | 224 | 15 | legacy: `vit/train_vit.py:67,83-89,99,103` |
| vit | imagenette | 200 | 128 | adamw | 3e-4 | 0.05 | 20 | none | 0 | 224 | 15 | new: rules E, W below |
| resnet | imagenet | 100 | 256 | sgd | 0.1 | 1e-4 as L2 | 5 | none, loader colour jitter | 0 | 224 | 15 | legacy: `resnet/train_resnet.py:61,74-81,89,347` |
| resnet | imagenette | 200 | 128 | sgd | 0.05 | 1e-4 as L2 | 10 | none | 0 | 224 | 15 | new: rules E, B, L, W |
| convnext_v1, convnext_v2 | imagenet | 100 | 64 | adamw | 1e-3 | 1e-4 | 0 | none | 0 | 224 | 50 | legacy: `convnext/common.py:155,160-165,258-265` |
| convnext_v1, convnext_v2 | imagenette | 100 | 64 | adamw | 1e-3 | 1e-4 | 0 | none | 0 | 224 | 50 | new (the legacy trainer refuses Imagenette): rules E, M |
| cliffordnet | imagenet | 200 | 8 | adamw | 1e-3 | 0.1 | 0 | none | 0 | 224 | 30 | legacy except the batch (legacy 128): `cliffordnet/train_cliffordnet.py:241-246,611,652-664,681`, `common/args.py:345`; batch measured, see Memory fit |
| cliffordnet | imagenette | 200 | 8 | adamw | 1e-3 | 0.1 | 0 | none | 0 | 224 | 30 | legacy except the batch (legacy 128): `cliffordnet/train_cliffordnet.py:247-251,611,652-664,681`; batch measured, see Memory fit |
| cliffordnet_hier | imagenet | 200 | 16 | adamw | 1e-3 | 0.1 | 0 | none | 0 | 224 | 30 | legacy except the batch (legacy 128): same lines as `cliffordnet`; batch measured, see Memory fit |
| cliffordnet_hier | imagenette | 200 | 16 | adamw | 1e-3 | 0.1 | 0 | none | 0 | 224 | 30 | legacy except the batch (legacy 128): same lines as `cliffordnet`; batch measured, see Memory fit |

"Augment none" adds nothing to the loader's own random-resized-crop and flip, which every ImageNet and Imagenette
run has (`resnet` on ImageNet adds the loader's colour jitter through `imagenet_augment_color`, which the loader
refuses for Imagenette). "Smoothing" is the label smoothing of the loss, 0 everywhere: the task builds a plain sparse
cross-entropy, `vit` refuses a nonzero value (`vit/train_vit.py:125`), the 0.1 of `resnet/train_resnet.py` never
reaches its loss, and the other two trainers default to 0. "Image size" is the dataset's own 224 (`transform_args.image_size`
can change it). Other values of the six recipes: ViT uses patch 16 at 224 with dropout 0.1 and attention dropout 0.1
(`vit/train_vit.py:79-80,451-453`); the sgd weight decay of `resnet` is an L2 on the model, adamw decays in the
optimizer; `convnext_v1` and `convnext_v2` use strides 4, drop path 0.1 and dropout 0 on both datasets; stochastic
depth is 0.3 for `cliffordnet` (0.4 for `cn_64_5`) and 0.1 for `cliffordnet_hier`. The cliffordnet batches are measured, not
legacy (Memory fit, below). The table is pinned, value by value, by
`tests/test_train/test_api/test_the_imagenet_and_imagenette_recipes.py`, which also reads the legacy trainers'
own defaults and compares.

**The four Imagenette pairs that have no legacy trainer** (`vit`, `resnet`, `convnext_v1`, `convnext_v2`) get values
from these rules, each a derivation from a legacy value or a cited recipe, not from a run: no run of these four pairs
has been made, so read them as starting points. Only `cliffordnet` has an Imagenette parity run planned.

| Rule | Value | Basis |
|---|---|---|
| E, epochs | 200 for `vit` and `resnet` at batch 128, 100 for `convnext` at batch 64 | The fit split is 8,523 images (9,469 train records less 946 held out), 66 steps an epoch at batch 128 and 133 at batch 64, so 200 and 100 epochs are 13,200 and 13,300 steps: the step count of the one legacy Imagenette recipe, `cliffordnet` at 200 epochs (`cliffordnet/train_cliffordnet.py:664`). The ImageNet epochs (50, 100) would be 3,300 and 6,600 steps |
| B, batch | `vit` 128 and `convnext` 64 unchanged; `resnet` 128 | `resnet`'s 256 is 33 steps an epoch on this split (`resnet/train_resnet.py:61`) |
| L, learning rate | `resnet` 0.05; the others unchanged | linear scaling of the SGD rate with the batch, 0.1 x 128 / 256 (Goyal et al. 2017, "Accurate, Large Minibatch SGD"); the adamw rates are the legacy ones at the legacy batch |
| W, warmup | `vit` 20 epochs, `resnet` 10, `convnext` 0 | the legacy share of the schedule: `vit` 5 of 50 epochs (`vit/train_vit.py:83,87`), `resnet` 5 of 100 (`resnet/train_resnet.py:74,78`), `convnext` none (`convnext/common.py:264`) |
| Weight decay, clipping, patience, dropout | the legacy values | no cited recipe changes them with the size of the data |
| M, `convnext` model arguments | strides 4, drop path 0.1, dropout 0 | the legacy table's only 224 px row (`convnext/common.py:155,160-165`), the ConvNeXt-T values; without an entry the factory would apply drop path 0 |

**Memory fit** (measured, GPU 0): for the smallest variant of each of the six families on both datasets, for
`cliffordnet/lite` (the factory's default variant) and `cliffordnet/cn_64_5` (its largest), and for `resnet/resnet50`
(the legacy ImageNet trainer's default variant), 20 real train steps ran through `python -m train image_classification`
on an RTX 4090 (24,564 MiB; 717 MiB in use when idle), with XLA as Keras sets it (`jit_compile` auto; the logs show the
XLA cluster compiled), the recipe's own values except `batch_size`, and `epochs=1` on a `data.max_samples` that gives 20
steps. "Allocator peak" is the run's own `GPU peak on` log line (`tf.config.experimental.get_memory_info`, read right
after `fit`, so the validation pass is in it); "Device" is the highest `nvidia-smi` `memory.used` sampled every half
second. The device figure is always above the peak plus the idle use (the allocator holds a pool larger than its peak, and
the CUDA context and kernels sit outside it) and, once a model has a big temporary, the pool grows to about 22,600 MiB,
92% of the card, so Device is a ceiling and not a headroom measure. **The rule:** a batch fits when it completes 20 steps
with the allocator peak at or under 20,000 MiB, the device at or under 23,000 MiB, and no allocator garbage-collection or
out-of-memory line in the log (TensorFlow degrades silently otherwise: `Garbage collection: deallocate free memory
regions` means the allocator reached the card's limit and recovered, which is a run with no margin). The batch is halved
from the recipe's until one fits. An OOM was observed in each failed row below, not inferred: the process died on
`Out of memory while trying to allocate N bytes` or on an allocator `ran out of memory` line. On ImageNet the rows near 100
images per second are limited by the disk (the first read of those files, below); the rows that repeat a
`data.max_samples` read the same files from the page cache.

<!-- memory:begin -->
| Model | Dataset | Image | Batch | Allocator peak, MiB | Device, MiB | Steps | Seconds per step | Images per second | Result |
|---|---|---|---|---|---|---|---|---|---|
| `vit/vit_pico` | imagenet | 224 | 128 | 3,235 | 6,293 | 20 | 1.330 | 96.3 | fits |
| `vit/vit_pico` | imagenette | 224 | 128 | 3,230 | 6,277 | 20 | 0.045 | 2,843.7 | fits |
| `resnet/resnet18` | imagenet | 224 | 256 | 5,326 | 18,165 | 20 | 2.624 | 97.5 | fits |
| `resnet/resnet18` | imagenette | 224 | 128 | 2,630 | 9,330 | 20 | 0.061 | 2,092.5 | fits |
| `resnet/resnet50` | imagenet | 224 | 256 | 18,239 | 22,591 | 20 | 0.326 | 785.3 | completed, allocator garbage collection, no margin |
| `resnet/resnet50` | imagenet | 224 | 128 | 10,579 | 22,589 | 20 | 0.164 | 780.4 | fits |
| `resnet/resnet50` | imagenette | 224 | 128 | 10,577 | 22,589 | 20 | 0.166 | 768.9 | fits |
| `convnext_v1/tiny` | imagenet | 224 | 64 | 4,184 | 9,337 | 20 | 0.598 | 106.9 | fits |
| `convnext_v1/tiny` | imagenette | 224 | 64 | 4,162 | 9,336 | 20 | 0.056 | 1,141.6 | fits |
| `convnext_v2/atto` | imagenet | 224 | 64 | 1,238 | 5,269 | 20 | 0.034 | 1,899.2 | fits |
| `convnext_v2/atto` | imagenette | 224 | 64 | 1,232 | 5,270 | 20 | 0.030 | 2,125.0 | fits |
| `cliffordnet/nano` | imagenet | 224 | 128 | - | 17,657 | - | - | - | OOM, asks 147.6 GiB |
| `cliffordnet/nano` | imagenet | 224 | 64 | - | 17,591 | - | - | - | OOM, asks 55.8 GiB |
| `cliffordnet/nano` | imagenet | 224 | 32 | - | 9,367 | - | - | - | OOM, asks 32.2 GiB |
| `cliffordnet/nano` | imagenet | 224 | 16 | 18,391 | 22,575 | 20 | 0.260 | 61.5 | completed, allocator garbage collection, no margin |
| `cliffordnet/nano` | imagenet | 224 | 8 | 10,804 | 19,587 | 20 | 0.133 | 60.2 | fits |
| `cliffordnet/nano` | imagenette | 224 | 128 | - | 17,654 | - | - | - | OOM, asks 147.6 GiB |
| `cliffordnet/nano` | imagenette | 224 | 64 | - | 17,590 | - | - | - | OOM, asks 55.8 GiB |
| `cliffordnet/nano` | imagenette | 224 | 32 | - | 9,366 | - | - | - | OOM, asks 32.2 GiB |
| `cliffordnet/nano` | imagenette | 224 | 16 | 18,388 | 22,574 | 20 | 0.260 | 61.6 | completed, allocator garbage collection, no margin |
| `cliffordnet/nano` | imagenette | 224 | 8 | 10,869 | 19,586 | 20 | 0.134 | 59.6 | fits |
| `cliffordnet/nano` | imagenette | 128 | 64 | 18,758 | 22,577 | 20 | 0.373 | 171.5 | completed, allocator garbage collection, no margin |
| `cliffordnet/nano` | imagenette | 128 | 32 | 14,315 | 19,589 | 20 | 0.183 | 174.4 | fits |
| `cliffordnet/lite` | imagenet | 224 | 128 | - | 22,565 | - | - | - | OOM, asks 112.3 GiB |
| `cliffordnet/lite` | imagenet | 224 | 64 | - | 17,591 | - | - | - | OOM, asks 60.0 GiB |
| `cliffordnet/lite` | imagenet | 224 | 32 | - | 14,487 | - | - | - | OOM, asks 38.3 GiB |
| `cliffordnet/lite` | imagenet | 224 | 16 | - | 9,351 | - | - | - | OOM, asks 24.9 GiB |
| `cliffordnet/lite` | imagenet | 224 | 8 | 16,458 | 22,575 | 20 | 0.200 | 40.0 | fits |
| `cliffordnet/lite` | imagenette | 224 | 128 | - | 17,645 | - | - | - | OOM |
| `cliffordnet/lite` | imagenette | 224 | 64 | - | 22,559 | - | - | - | OOM, asks 60.0 GiB |
| `cliffordnet/lite` | imagenette | 224 | 32 | - | 14,487 | - | - | - | OOM, asks 38.3 GiB |
| `cliffordnet/lite` | imagenette | 224 | 16 | - | 7,815 | - | - | - | OOM, asks 24.9 GiB |
| `cliffordnet/lite` | imagenette | 224 | 8 | 16,456 | 22,575 | 20 | 0.201 | 39.8 | fits |
| `cliffordnet/lite` | imagenette | 128 | 64 | - | 9,351 | - | - | - | OOM, asks 29.9 GiB |
| `cliffordnet/lite` | imagenette | 128 | 32 | 19,897 | 22,577 | 20 | 0.278 | 114.9 | completed, allocator garbage collection, no margin |
| `cliffordnet/lite` | imagenette | 128 | 16 | 15,148 | 19,859 | 20 | 0.146 | 109.8 | fits |
| `cliffordnet/cn_64_5` | imagenette | 224 | 8 | - | 3,249 | - | - | - | OOM, asks 32.9 GiB |
| `cliffordnet/cn_64_5` | imagenette | 224 | 4 | - | 2,225 | - | - | - | OOM, asks 21.9 GiB |
| `cliffordnet/cn_64_5` | imagenette | 224 | 2 | 18,166 | 22,619 | 21 | 0.207 | 9.7 | fits |
| `cliffordnet/cn_64_5` | imagenette | 224 | 1 | 6,405 | 9,613 | 22 | 0.129 | 7.8 | fits |
| `cliffordnet_hier/hier_345` | imagenet | 224 | 128 | - | 19,695 | - | - | - | OOM, asks 70.8 GiB |
| `cliffordnet_hier/hier_345` | imagenet | 224 | 64 | - | 17,589 | - | - | - | OOM, asks 33.4 GiB |
| `cliffordnet_hier/hier_345` | imagenet | 224 | 32 | 18,647 | 22,589 | 20 | 0.282 | 113.6 | completed, allocator garbage collection, no margin |
| `cliffordnet_hier/hier_345` | imagenet | 224 | 16 | 11,582 | 21,653 | 20 | 0.140 | 114.7 | fits |
| `cliffordnet_hier/hier_345` | imagenette | 224 | 128 | - | 17,652 | - | - | - | OOM, asks 70.8 GiB |
| `cliffordnet_hier/hier_345` | imagenette | 224 | 64 | - | 17,591 | - | - | - | OOM, asks 26.8 GiB |
| `cliffordnet_hier/hier_345` | imagenette | 224 | 32 | 18,442 | 22,585 | 20 | 0.289 | 110.7 | completed, allocator garbage collection, no margin |
| `cliffordnet_hier/hier_345` | imagenette | 224 | 16 | 11,601 | 21,657 | 20 | 0.142 | 112.7 | fits |
<!-- memory:end -->

What the table settles, in the recipes:

* `vit/vit_pico`, `resnet/resnet18`, `convnext_v1/tiny` and `convnext_v2/atto` fit at their recipes' batches on both
  datasets (allocator peaks 1,232 to 5,326 MiB). Their recipes did not change.
* `cliffordnet/nano` at 224 px does not fit at batch 128, 64 or 32 (the 32 asks 32.2 GiB, the legacy docstring's 34.6 GB),
  completes at 16 with a garbage collection (no margin) and fits at **8** (allocator peak 10,869 MiB). `cliffordnet/lite`
  also needs **8** (peak 16,456 MiB). `cliffordnet_hier/hier_345` fits at **16** (peak 11,601 MiB; 32 completes with a
  garbage collection). So the `cliffordnet` recipe on ImageNet and Imagenette is batch 8, the `cliffordnet_hier` recipe
  is 16, and both were 128. The CIFAR recipes keep 128 (32 px).
* **The learning rate is not scaled with the batch.** `cliffordnet/train_cliffordnet.py` has no batch rule (its learning rate
  is 1e-3 at batch 128 in the paper recipe and at batch 32 in its own README), so none is invented, and no run has
  checked that 1e-3 suits a batch of 8 or 16. (`resnet` is different: its Imagenette lr 0.05 is the linear rule of
  Goyal et al. 2017, as the rules above say.)
* **The image size stays the data's own 224.** The legacy README asks for `--image-size 128` on the isotropic variants,
  but in this API the data decides the size before the recipe is read, so a recipe cannot set it. The legacy README's
  form `--image-size 128 --batch-size 32` is `transform_args.image_size=128 batch_size=32` here, and it was measured: `cliffordnet/nano` fits at 32 there
  (peak 14,315 MiB, 174 images per second against 60 at 224 px), and `cliffordnet/lite` fits at 16. A parity run against the
  legacy trainer sets both keys itself.
* **The evaluation batch is a recipe key too.** The training batches above leave the passes AFTER `fit` at the module constant
  256 images, and 256 images of 224 px asked 18 GB of the same card: the gate's `cliffordnet/nano` trained and then died in the
  figures' `predict`. `analysis_args.eval_batch_size` is now the batch of those passes; the `cliffordnet` and
  `cliffordnet_hier` ImageNet and Imagenette recipes set it to the batch they train at (8 and 16). Measured with that run's
  weights, 512 images, allocator peak: 924 MiB at batch 8, 1,468 at 16, 2,555 at 32, 5,043 at 64, 17,775 at 256; the real
  reporting pass of that run (20,000 test images, figures, analyzer, final evaluation, load check) at batch 8 peaked at
  4,681.5 MiB. `cliffordnet` on CIFAR at 224 px is not reachable: `transform_args.image_size` is refused for in-memory data.

Not covered by a recipe, and measured: `resnet/resnet50` on ImageNet completes at the recipe's batch 256 but with a garbage
collection and the device at 22,591 MiB (no margin), and a bounded whole run of it (`data.max_samples=2560`, one epoch)
also reported to exit 0 with the evaluation batch left at 256 (the figures' `predict` alone peaks at 4,487 MiB); batch 128 fits (peak 10,579 MiB), with `optim_args.lr=0.05` by the same
linear rule. `cliffordnet/cn_64_5` does not fit at 8 or 4 on Imagenette and fits at 2 (peak 18,166 MiB); it was not run on
ImageNet (the tensors are the same but for the 1,000-way head). A recipe is per family and `per_variant` applies to every
dataset, so changing the batch for these two would change their CIFAR recipes; they are left, and
`tests/test_train/test_api/test_the_recipe_batches_are_the_measured_memory_fit.py` names them. The larger variants of the other families (`vit_tiny`
and up, `resnet101`, `convnext` above `tiny`) were not measured. A run of 20 steps is not a run of 200 epochs: nothing here
says the margin holds for the whole run. The table is pinned to `tests/test_train/test_api/memory_fit_evidence.json` by the
test, row by row.

**ImageNet on a spinning disk.** The ImageNet train images are on a hard disk (a 4 TB WD40EFAX, `ROTA=1`) on the development
machine. In the runs above the first read of files cost **96.3, 97.5 and 106.9 images per second** (`vit/vit_pico`,
`resnet/resnet18`, `convnext_v1/tiny`; cold random reads of single files measured 44 to 115), against about 1,150 images
per second of CPU decode, so the disk is the limit, not the GPU. One epoch is 1,271,167 images; that is 3.7 to 3.3 hours,
and a run of E epochs takes (validation not counted):

<!-- imagenet-time:begin -->
| Epochs | Days at 96.3 images per second | Days at 106.9 images per second |
|---|---|---|
| 50 | 7.6 | 6.9 |
| 100 | 15.3 | 13.8 |
| 200 | 30.6 | 27.5 |
<!-- imagenet-time:end -->

So the 50-epoch ImageNet recipe (`vit`) takes about a week, and the 100-epoch ones (`resnet`, `convnext`) about two. A
killed epoch is lost: the run goes back to the start of the epoch it was in (`resume_interrupted=true` trains it again), and
at these rates an epoch is hours; `checkpoint_args.backup_every_steps` bounds the loss (next section). A shorter trial is
`data.max_samples` (it caps the train pool and the test split; it does NOT cap the selection set, which is
`data.holdout_per_class` images per class, 10 by default, so 10,000 images of the 1,000 classes whatever the cap: the bounded
gate with `data.max_samples=20000` trained on 20,000 images, selected on 10,000 and tested on 20,000 official validation images
(`fit_state.json` `data_counts`: `n_train` 20000, `n_select` 10000, `n_test` 20000); set `data.holdout_per_class` to shrink it).

**Backing up inside an epoch.** `checkpoint_args.backup_every_steps` set to N (an integer from 2 to one below the steps of an epoch;
`null`, the default, backs up at each epoch end only, as before) also writes the resume backup (`checkpoints/backup/`, the
previous copy kept beside it) after every N-th batch of an epoch, so a kill loses at most N batches of weights instead of the
epoch. A value that is not below the steps of an epoch of the loaded data is refused (exit 2, before the model is built). What a
resume from such a copy is, measured with real `SIGKILL`s of a child process
(`tests/test_train/test_api/test_backup_every_steps.py`):

* the killed epoch restarts from its **first** batch on the weights, Adam moments and loss scale of the last copy, so the
  examples seen since that copy (up to N batches) are seen again, in the data order epoch 0 had (the data iterator of a resumed
  run is new and its position is not stored: measured, a fresh process with the same seed reshuffles from the start, so the
  first epoch it trains has epoch 0's order, the next epoch 1's, and so on);
* the optimizer's step count goes back to the epoch's start, so the run ends on exactly `epochs * steps_per_epoch` steps and
  every epoch's learning rate is the unkilled run's. Under `mixed_float16` the counter rewound is the loss-scale optimizer's
  inner one, which lags the batches by the steps skipped for non-finite gradients; the rewind keeps that lag;
* the CSV, the sidecar and the best checkpoint hold finished epochs only, so a resume reads those of the epoch before the killed
  one; a kill inside the **first** epoch resumes too, from its first copy;
* a kill between the weights and the metadata of a copy is answered by the previous copy (N batches older): the pair is
  checked against the optimizer's step count before `fit`, and a resume that cannot verify a pair refuses it by name;
* a resumed run is a valid run of the same recipe, not the unkilled run's trajectory: in the test setup (`vit/vit_pico`, 3
  epochs of 18 batches, a copy every 5) the final weights differ from the unkilled run's by about 1% of their norm, against
  about 70% for another seed, and Adam's moments differ more (they follow the data order). An unkilled run with the key set
  is weight for weight the run without it.

A save writes `latest.weights.h5` (the weights and the optimizer state) after copying the previous pair to `*.bkp`, so the disk
holds two files of the size below. Measured (CPU, no data, the ImageNet recipe's shapes and optimizers, the disk of `results/`
which is an SSD; the cost is the write into the page cache, nothing is flushed):

<!-- backup-cost:begin -->
| Model | Optimizer | Parameters | File, MB | Seconds per save |
|---|---|---|---|---|
| `vit/vit_pico` | adamw | 3,048,232 | 35.2 | 0.38 |
| `resnet/resnet18` | sgd | 11,699,112 | 89.4 | 0.67 |
| `resnet/resnet50` | sgd | 25,610,152 | 195.7 | 1.54 |
| `convnext_v1/tiny` | adamw | 33,233,992 | 380.9 | 2.61 |
| `convnext_v2/atto` | adamw | 4,516,640 | 52.1 | 0.66 |
| `cliffordnet/nano` | adamw | 1,552,616 | 18.2 | 0.39 |
| `cliffordnet_hier/hier_345` | adamw | 883,576 | 10.6 | 0.35 |
<!-- backup-cost:end -->

At `backup_every_steps=500`, `vit/vit_pico` at batch 128 (9,930 steps an epoch) takes 19 copies of 0.38 s inside an epoch (7.2 s)
and a kill loses at most 10.0 to 11.1 minutes of training; `resnet/resnet18` at batch 256 (4,965 steps) takes 9 copies of 0.67 s
(6.0 s) and loses at most 20.0 to 22.2 minutes. The epoch-end copy is written in either case. The numbers come from the rates
above and the table; `test_backup_every_steps.py` recomputes them.

The exact commands for a full 50-epoch and a full 100-epoch run, from the repo root with the project virtual environment.
**They were NOT run**, because each takes a week or more of the only working GPU; the test
`tests/test_train/test_api/test_the_help_and_the_commands_state_the_imagenet_cost.py` only checks that each line is a valid request
(every key exists, every value has its type), not that it trains.

<!-- imagenet-commands:begin -->
```bash
# NOT RUN. 50 epochs of the vit recipe on the full ImageNet: about a week.
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m train image_classification \
    out=results/imagenet_vit_pico_50ep model=vit/vit_pico data.dataset=imagenet epochs=50 \
    checkpoint_args.backup_every_steps=500

# NOT RUN. 100 epochs of the resnet recipe on the full ImageNet: about two weeks.
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m train image_classification \
    out=results/imagenet_resnet18_100ep model=resnet/resnet18 data.dataset=imagenet epochs=100 \
    checkpoint_args.backup_every_steps=500

# NOT RUN. After a kill, the same line with resume_interrupted=true continues it (the killed epoch is trained again from its first batch).
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m train image_classification \
    out=results/imagenet_resnet18_100ep model=resnet/resnet18 data.dataset=imagenet epochs=100 \
    checkpoint_args.backup_every_steps=500 resume_interrupted=true
```
<!-- imagenet-commands:end -->

**Parity**: the API run and the legacy trainer were compared on `vit_pico` on CIFAR-10 only
(10 epochs, three seeds each; the means differ by less than a tolerance measured from the legacy
trainer's own run-to-run spread). The other families' recipes are pinned by tests that build them,
not by a run. The legacy `train.<model>` scripts remain.
