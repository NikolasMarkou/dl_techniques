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
run. It is refused for a finished run, another request, other data, a live process holding the
run, or a torn backup; `devices` may differ. The training data order replays epoch 0's and a half-done epoch is
trained again (the key's help lists the rest). For folder and CSV data the sidecar holds a fingerprint of the
files: `data.fingerprint=listing` (the default) hashes every file's relative path, class and size and opens no
file, so two same-sized files swapped, or a pixel edited, go unseen; `data.fingerprint=content` also hashes every
file's bytes, streamed, which reads all the data once (a pass over the disk tree, minutes on a large one on a
spinning disk; the seconds go to `run.log`). A resume refuses a changed mode or digest.

**Figures, analysis and TensorBoard**: five keys decide what is written besides the models and the
summary. The region between the markers below is read by
`tests/test_train/test_api/test_the_knob_docs_name_real_keys.py`: every key, default and name in it is
checked against the code, so a stale one fails the test.

<!-- knobs:begin -->
| Key | Default | What it does |
|---|---|---|
| `analysis_args.figures` | `true` | Writes `visualizations/`: `training_dashboard.png` (redrawn after epoch 1, about 20 times per run, and at the end), then after the best weights are loaded `confusion_matrix.png`, `per_class_metrics.png`, `confidence_calibration.png`, `misclassifications.png` and `classification_report.json` (these five need a test split). `false` writes none of them and `ece` in the summary is null |
| `analysis_args.model_analysis` | `true` | Runs the full model analyzer once, after training, on the best weights, into `model_analysis/` (`analysis_results.json` and seven PNGs: weights, spectral, information flow, calibration, training dynamics and the summary dashboard). Needs a test split. `false` leaves `analyzer.status` at `skipped` |
| `analysis_args.epoch_analysis_every` | `0` | `N > 0` runs the data-free epoch analyzer (weights and spectral only) before training into `epoch_analysis/epoch_000/` (the untrained model), after epoch 1 into `epoch_001/`, then at epochs N, 2N, ... into `epoch_NNN/`. `0` is off. A failure on a built model stops the fit, so no snapshot is silently missing. A resumed run does not redo `epoch_000` and `epoch_001` |
| `analysis_args.conformal_alpha` | `null` | A level strictly between 0 and 1 (0.1 aims at 90% coverage) writes `conformal.json` and `conformal_model.keras` (outputs logits, probabilities and set membership), calibrated on the test split. `null` skips it and the summary's `conformal` is null. A level with no test split is skipped with a note |
| `logger_args.tensorboard` | `{}` | `{}` writes TensorBoard event files into `tensorboard/` (`train/` and `validation/`); `null` turns them off; a dict with keys is refused, because no TensorBoard option is wired |

Measured cost, one run each, `vit/vit_pico` on 240 fake CIFAR-10 images (240 test images), 3 epochs, batch 16,
CPU with one thread, wall seconds of the whole call against a run with figures, model analysis and epoch
analysis off (68.0 s; its fit took 53.7 s; TensorBoard on, no conformal): `epoch_analysis_every=1` +29.4 s to the fit (four snapshots, `epoch_000` to `epoch_003`, about
7 s each, 6.2 MB); `model_analysis` +23.4 s (3.1 MB); `figures` +6.9 s (0.4 MB); `conformal_alpha=0.1` +4.2 s
(`conformal_model.keras` is a full copy of the model, 32 MB here); TensorBoard on against off was 68.0 s against 65.0 s (0.2 MB), which one run each cannot tell
from noise. These are small-model, small-data numbers: the analyzer's cost
grows with the number of layers and test images, and the dashboard render is recorded at 1.0 to 1.7 s each
(`common/classification_viz.py`). The evidence is `plans/plan-2026-10-31T180033-2faf890a/scratch_step13/measure_out.txt`
(the plans directory is untracked).

**A failed analysis is named, but only some are.** The analyzer isolates a failing analysis and carries on, so
a run can write a normal-looking `model_analysis/` after one of its parts failed. For this task the summary's
`analyzer` block gains `summary.analyzer.failed_analyses`, a list, only when it is not empty (a clean run
writes the same keys as before), and `run.log` carries a WARNING. It covers exactly two things:
`calibration` (requested, and the analyzer produced no calibration entry for the model) and `model_analysis`
(the whole analysis raised and returned nothing). It does not check weights, spectral, information flow or
training dynamics. `summary.analyzer.status` is read back from the analyzer's own result file and does not see
calibration, so it can read `success` beside a non-empty `failed_analyses`: read both. In the analyzer's
summary dashboard a missing ECE or Brier prints `n/a`, not 0.0.

**For a task author**: the trainer calls two optional hooks and one required one for this output. They are
documented at their definitions in `api/trainer.py`: `TaskTrainer.figure_callbacks(config, data, vis_dir)`
returns per-epoch figure callbacks (called only while `analysis_args.figures` is true, placed after the
epoch log line and before the dashboard); `TaskTrainer.run_analysis(config, model, test_data, history, experiment_name, run_dir)`
returns what the model analyzer returns, `None` when it raised; `TaskTrainer.evaluate_and_report(config, data, fit)`
is the required end-of-run sequence that calls them. `src/train/AGENTS.md` has the rules.
<!-- knobs:end -->

**Predict**: `python -m train predict model=<run dir or .keras> data=<image or directory> out=pred.json`
writes a new JSON list of `{"path", "labels", "scores"}`, every class best first. A directory is
searched recursively. `out` must not exist. Through the Python `Predictor`, a colour array or PIL image given to a
grey model (MNIST) is made grey on its 8-bit pixels with the loader's own rule, `(9797 R + 19234 G + 3737 B) >> 15`
(libpng's weights as `tf.io.decode_image(channels=1)` uses them), so a PNG gives the loader's pixels exactly (checked
within 1e-5, measured 0). **Measured limit:** a JPEG given as a path is read by the loader's decoder and is exact, but
a JPEG that was already decoded to an array or a PIL image has lost the Y plane the decoder takes its grey from, and
PIL's and TensorFlow's JPEG decoders differ, so its grey is off the loader's by up to 14 levels of 255 on random
colour noise at quality 75 (mean 0.56; 13 and 1.15 at quality 95) and by 1 to 2 levels (mean 0.5 to 0.6, about
0.002) on smooth images: above 1e-5, and not removable from the decoded pixels.

**Export**: `python -m train export model=<run dir or .keras> out=<path> format=keras|saved_model|tflite|onnx`
writes one new file (a directory for `saved_model`) and `<out>.task.json` beside it, and prints
only the path on stdout. Input contract: float32 `(batch, height, width, channels)` pixels scaled
to `[0, 1]` then standardized, `(x - mean) / std`, with the mean and std in `<out>.task.json`;
the output is raw logits, one per class name in that file. Other pixel ranges give numbers and
no error. A `tflite` file of a CliffordNet needs the Flex delegate (a WARNING says so).

**Exit codes**: 0 success; 2 a request that cannot be honoured (bad key, value, path, data,
resume), found before TensorFlow loads when it can be; 1 a run that finished without reaching its
success threshold, or any unexpected failure (a traceback).

**Parity**: the API run and the legacy trainer were compared on `vit_pico` on CIFAR-10 only
(10 epochs, three seeds each; the means differ by less than a tolerance measured from the legacy
trainer's own run-to-run spread). The other families' recipes are pinned by tests that build them,
not by a run. The legacy `train.<model>` scripts remain.
