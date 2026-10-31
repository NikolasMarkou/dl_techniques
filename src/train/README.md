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
