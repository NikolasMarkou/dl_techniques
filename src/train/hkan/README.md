# HKAN trainer

Trains `HKAN` (`src/dl_techniques/models/general_purpose/hkan/`), the Hierarchical
Kolmogorov-Arnold Network of Dudek and Rodak (arXiv:2501.18199), on a synthetic
target of the paper or on your own CSV files. The model package README explains the
model; this file covers the trainer.

```bash
MPLBACKEND=Agg .venv/bin/python -m train.hkan.train_hkan --help
```

The model package lives at `src/dl_techniques/models/general_purpose/hkan/`, not at
`src/dl_techniques/models/hkan/`: this repository files every model under a family
directory. The trainer is at `src/train/hkan/`.

## 1. Training modes

`--training-mode` selects how the weights are found.

| Mode | What runs | Rows fitted on |
|---|---|---|
| `closed_form` (default) | `HKAN.fit_closed_form`: the paper's layer-by-layer least squares, in float64, no gradients, no epochs | every train row |
| `backprop` | stock `compile()` + `fit()` with Adam and MSE from the model's initial state | train rows minus the validation part |
| `closed_form_then_backprop` | the closed-form fit, then stock `fit()` starting from the closed-form weights | train rows minus the validation part |

The basis centers are fixed in every mode. In `backprop` mode with `--centers data`
they are drawn from the data before training starts.

In the two backprop modes `--val-fraction` (default 0.1) of the TRAIN split is held
out for checkpoint selection. The test split is never used for fitting or selection.
With `--val-fraction 0` nothing is held out and the fit rows themselves are passed to
`fit()` as its validation data, so the checkpoint still selects on an end-of-epoch
MSE (`val_loss`, here on the fit rows) and never on Keras' `loss`, which is a running
mean over the epoch's batches while the weights move. That costs one extra
evaluation pass over the fit rows per epoch, in every repeat.
`--epochs`, `--batch-size`, `--learning-rate` and `--val-fraction` are ignored in
`closed_form` mode.

There is no custom `train_step`. `train.common.create_callbacks()` is not used: it
always adds early stopping, which would stop repeats at different epochs, and it has
no role in the closed-form mode. The backprop modes use `ModelCheckpoint` and
`CSVLogger` directly.

## 2. Data

`--dataset` is one of `tf1`, `tf2`, `tf3`, `tf4`, `tf5`, `tf5_5` (generated locally
from `--seed` with the sample counts of the paper's Table VI) or `csv`.

| Name | Function | Inputs | Train / test rows |
|---|---|---|---|
| `tf1` | `(2 x1 - 1)(2 x2 - 1)` on `[0, 1]^2` | 2 | 5000 / 10000 |
| `tf2` | `sum_i sin(20 exp(x_i)) x_i^2` on `[0, 1]^2`, noise `U(-0.2, 0.2)` on the train targets | 2 | 5000 / 10000 |
| `tf3` | `-sum_i x_i sin(sqrt(abs(x_i)))` on `[-500, 500]^2` | 2 | 5000 / 10000 |
| `tf4` | `1 - cos(2 pi r) + 0.1 r`, `r = sqrt(sum_i x_i^2)`, on `[-4, 4]^10` | 10 | 3750 / 1250 |
| `tf5` | `-sum_i sin(x_i) sin^20(i x_i^2 / pi)` on `[0, pi]^2` | 2 | 5000 / 10000 |
| `tf5_5` | the same with 5 inputs | 5 | 7500 / 2500 |

Every input and every target is min-max scaled to `[0, 1]` with the train split's
statistics; a test value outside the train range maps outside `[0, 1]`. Four readings
of the paper are this trainer's own, stated in `src/train/hkan/data.py`:

- **Scaling of TF1 and TF2.** The paper's sentence on normalization names TF3, TF4
  and TF5 only. Its Fig. 3 draws TF1 and TF2 on a 0 to 1 axis and its notation table
  gives the target as `y in [0, 1]`, so every target is scaled here. The paper's
  TF2 train RMSE (0.116, the standard deviation of the noise) cannot tell the two
  readings apart.
- **TF2 noise** is added to the scaled train targets, after scaling; the scaling
  statistics come from the noiseless train targets; the test targets are noiseless.
- **TF2 formula**: the exponent 2 is this trainer's reading of a garbled PDF line.
- **TF4 sign**: the paper prints `- 0.1 r`; the trainer uses `+ 0.1 r`, the standard
  Salomon function, reading the printed minus as a typo.

`--dataset csv --train-csv A --test-csv B` reads two comma-separated files with no
header, one sample per row, the last column the target. `--scale-csv` min-max scales
with the train file's statistics. The basis centers of the `random` and
`equally_spaced` modes lie on `[0, 1]`: with unscaled inputs outside that range, use
`--scale-csv` or `--centers data` for the first layer; the trainer logs a warning
otherwise. The paper's 18 benchmark CSV files are not shipped and are not downloaded:
the upstream repository carries no licence.

## 3. Flags

Per-layer flags take one value (used for every layer) or a comma-separated value per
layer. The number of layers is the number of `--hidden-units` entries plus one; the
last layer always has one output. Invalid values are refused before anything is
written.

| Flag | Default | Meaning |
|---|---|---|
| `--training-mode` | `closed_form` | see section 1 |
| `--dataset` | `tf5` | see section 2 |
| `--train-csv`, `--test-csv` | none | files for `--dataset csv` |
| `--scale-csv` | off | min-max scale the CSV data with train statistics |
| `--num-train-samples`, `--num-test-samples` | paper counts | rows of a synthetic target (at least 2) |
| `--repeats` | 1 | independent fits with derived seeds (the paper uses 50) |
| `--seed` | 0 | seed of the data and of the per-repeat seeds |
| `--hidden-units` | `64` | hidden widths; empty string for a one-layer model |
| `--num-basis` | `10` | basis functions per block, per layer |
| `--basis` | `sigmoid` | `sigmoid`, `gaussian`, `relu`, `tanh`, `softplus`, `identity`, per layer |
| `--slope` | `5` | basis slope (the paper's sigma), finite and positive, per layer; `identity` ignores it |
| `--centers` | `random` | `random`, `equally_spaced`, `data`, per layer |
| `--l2-block` | `0.01` | ridge strength of the block fits, per layer (closed-form modes) |
| `--l2-mix` | `0.01` | ridge strength of the connecting fits, per layer (closed-form modes); `0` is the paper's plain least squares |
| `--no-block-bias`, `--no-bias` | off | drop the block or the output intercepts (the paper's bias-free equations) |
| `--chunk-size` | auto | outputs solved at once in the closed-form fit (bounds the memory of the solve and of the fit's closing forward pass) |
| `--epochs` | 100 | backprop epochs |
| `--batch-size` | 64 | backprop batch size |
| `--learning-rate` | 1e-3 | Adam learning rate |
| `--val-fraction` | 0.1 | part of the train split held out in the backprop modes |
| `--predict-batch-size` | 256 | batch size of every prediction pass |
| `--gpu` | none | GPU index |
| `--output-dir` | `<repo>/results` | base directory |
| `--experiment-name` | `hkan_<timestamp>` | run directory name |

The `--slope`, `--l2-block` and `--l2-mix` defaults are the model's constructor
defaults. They differ from the paper and its reference code, which use slope 1 and
no ridge: with those, a float32 model with a hidden layer does not compute its own
closed-form fit (the model package README, section 6, has the measurements). The
paper's configurations are reproduced by passing their values explicitly, as the
measured runs below do.

## 4. Run directory

```
results/<experiment-name>/
  config.json               the parsed arguments
  run.log                   the run's log
  results_summary.json      strict JSON, see below
  final_model.keras         repeat 0, weights at the end of training
  visualizations/
    test_predictions.png    predicted against target and residuals, test split, repeat 0, final weights
    input_importance.png    closed-form modes: mean R^2 of the first-layer blocks per input
    rmse_over_repeats.png   when --repeats > 1
    loss_curve.png          backprop modes; the legend names the rows of the monitored line
  best_model.keras          backprop modes, when an epoch was the best model (see below)
  closed_form_model.keras   closed_form_then_backprop: the closed-form fit before training
  training_log.csv          backprop modes: per-epoch loss and val_loss of repeat 0
  training_history.json     backprop modes
```

A reused `--experiment-name` is refused before anything is written. Nothing in an
existing run directory is overwritten or deleted.

`results_summary.json` keys: `model`, `training_mode`, `dataset`, `num_inputs`,
`train_rows`, `fit_rows`, `test_rows`, `params`, `seed`, `repeats`; `train_rmse`,
`test_rmse` and `fit_seconds`, each with `median`, `iqr`, `min`, `max` over the
repeats; `closed_form_phase` (the RMSE after the closed-form fit and before `fit()`,
two-phase mode only, otherwise null); `best_checkpoint`; `input_importance` (repeat
0, null in `backprop` mode); `repeat_results`; `visualizations` (`files` written and
`failed` figure names).

`repeat_results` holds one entry per repeat: its seed, `fit_rows` (the rows it was
actually fitted on), train and test RMSE, fit seconds, and in the closed-form modes
the per-layer train RMSE of the float64 fit and `forward_deviation_rms`, the RMS
difference between the float32 model and that fit on the training rows.

`best_checkpoint` is null in `closed_form` mode. In the backprop modes it has the
keys `monitor`, `monitor_rows`, `source`, `file`, `epoch`, `monitor_value`,
`train_rmse`, `test_rmse` and `reason`. `monitor` is always `val_loss`, the MSE at
the end of each epoch; `monitor_rows` says which rows it is measured on:
`validation` (the held-out part of the train split) or `fit` (with
`--val-fraction 0`, when the fit rows are the validation data; `training_log.csv`,
`training_history.json` and the loss curve's monitored line then describe the fit
rows). `source` is `epoch` when an epoch was the best model (`file` is
`best_model.keras`), `closed_form` in the two-phase mode when no epoch beat the
closed-form start (`file` is `closed_form_model.keras`, `epoch` is 0, and no
`best_model.keras` is written), or `none` when the monitored loss was never finite
(a diverged run still finishes and writes its summary). `monitor_value` is the MSE of
the model `file` names on the `monitor_rows`, float32 predictions against float32
targets (`null` for `none`).

Reported RMSEs come from the weights at the end of training in every repeat. They
are computed in float64 from the model's float32 predictions. Train RMSE is on the
rows the model was fitted on. With `--repeats N`, the data is generated once and
each repeat draws new centers and new initial weights.

## 5. Measured runs

Six runs on one RTX 4070, 2026-09-30, 5 repeats each, `--seed 0`. The settings and
the rules for reading the results were written down before the runs
(`decisions.md` D-030 and D-046 of plan-2026-09-30T082355-4d999dbc). The backprop
arms are untuned: one learning rate, 100 epochs, no schedule. Every number below is
recomputed by script from the `results_summary.json` and `training_log.csv` files of
`results/hkan_{tf5,tf2}_<mode>_20260930c`.

Common part of every command:

```bash
MPLBACKEND=Agg .venv/bin/python -m train.hkan.train_hkan --gpu 1 --repeats 5 --seed 0 \
    --epochs 100 --batch-size 64 --val-fraction 0.1 --l2-mix 0 <CONFIG> --training-mode <MODE>
```

with `--learning-rate 1e-3` for `backprop` and `1e-4` for `closed_form_then_backprop`.

**TF5**, the configuration of the authors' tutorial (108529 parameters):
`--dataset tf5 --hidden-units 912 --basis tanh,identity --slope 50 --num-basis 23,10 --centers random,data --l2-block 0.01,0.1`

| Mode | Train RMSE median (IQR) | Test RMSE median (IQR) | Fit seconds, min to max |
|---|---|---|---|
| `closed_form` | 1.369e-07 (1.7e-08) | 1.351e-07 (1.3e-08) | 7.4 to 9.2 |
| `backprop` | 2.340e-03 (2.2e-03) | 2.327e-03 (2.2e-03) | 23.7 to 25.8 |
| `closed_form_then_backprop`, closed-form phase | 1.129e-07 (2.3e-08) | 1.172e-07 (2.0e-08) | |
| `closed_form_then_backprop`, after `fit()` | 4.391e-03 (3.9e-03) | 4.396e-03 (3.9e-03) | 31.1 to 38.1 |

**TF2**, the paper's Table V row with this trainer's choice for the identity top
layer (26990 parameters):
`--dataset tf2 --hidden-units 48,11 --basis tanh,softplus,identity --slope 22,18,1 --num-basis 17,21,10 --centers random,data,data --l2-block 0.1,1,0.1`

| Mode | Train RMSE median (IQR) | Test RMSE median (IQR) | Fit seconds, min to max |
|---|---|---|---|
| `closed_form` | 0.1157 (6.2e-05) | 0.01556 (1.6e-03) | 3.0 to 3.9 |
| `backprop` | 0.1183 (3.2e-03) | 0.03739 (1.4e-02) | 27.2 to 65.4 |
| `closed_form_then_backprop`, closed-form phase | 0.1156 (3.5e-04) | 0.01655 (3.7e-04) | |
| `closed_form_then_backprop`, after `fit()` | 0.1144 (3.8e-03) | 0.02878 (1.4e-02) | 30.9 to 39.3 |

Fit seconds cover the whole fit of a repeat: in the two-phase mode, the closed-form
fit plus 100 epochs. The 65.4 s maximum of TF2 `backprop` is one repeat; the other
four took 27.2 to 37.7 s.

What these runs show, by the rules written beforehand:

- **TF5 closed form sits at the float32 limit.** The float32 model is 1.2e-07 to
  1.5e-07 from its own float64 fit on the train rows (`forward_deviation_rms`), and
  that is the whole of its error. The float64 fit itself is far below it. The paper
  reports 4.56e-15 test RMSE for TF5 on its own data file; this port's float64 fit
  on its own generated data reaches 3.2e-13 to 5.4e-13 train RMSE, a gap that was not investigated.
- **Fine-tuning the TF5 closed-form fit with Adam made it worse in all 5 repeats**,
  by factors of 16400 to 49900 on the train rows. The rule written beforehand calls
  this case "not informative at the float32 floor" (the closed-form error is already
  below 1e-5), so it claims no direction; the measured fact is stated anyway. A
  one-seed probe on the same configuration (not a run in `results/`) shows the
  damage happens at the first optimizer step and scales with the learning rate: one
  Adam step takes the train RMSE from 1.3e-07 to 2.1e-03 at learning rate 1e-4, to
  2.0e-05 at 1e-6 and to 1.5e-07 at 1e-8, while SGD at 1e-4 leaves it at 1.3e-07
  after 79 steps. In repeat 0, the repeat whose models the run directory keeps and
  the only one with a recorded best checkpoint, no epoch beat the start, so the run
  directory holds `closed_form_model.keras` and no `best_model.keras`, and
  `best_checkpoint.source` is `closed_form`.
- **On TF2 the fine-tune is a tie or mixed on the train rows and worse on test.**
  Train RMSE after the fine-tune is 0.987 to 1.036 of the closed-form phase ("tie
  or mixed" by the rule); the test RMSE ratio is 1.39 to 2.94, above 1 in all 5
  repeats. The train RMSE of about 0.116 is the noise level (the standard deviation
  of `U(-0.2, 0.2)` is 0.1155), so fitting the train rows more closely is fitting
  noise. In repeat 0, the only repeat with a recorded best checkpoint, no epoch beat
  the closed-form start on the validation rows.
- **Backprop alone against closed form.** TF5: median test RMSE is 17200 times the
  closed-form one ("worse"). TF2: 2.40 times ("worse"; the quartiles are 0.0255 to
  0.0396 for `backprop` against 0.0151 to 0.0166 for `closed_form`). Confound:
  `closed_form` fits on all 5000 train rows and `backprop` on 4500. Against the
  closed-form phase of the two-phase run, which uses the same 4500 rows, the ratios
  are 19900 (TF5) and 2.26 (TF2).
- **Cost.** The closed-form fit took 3.0 to 9.2 s per repeat in the `closed_form`
  runs and 2.7 to 12.1 s inside the two-phase runs (`closed_form_seconds`: 6.8 to
  12.1 s on TF5, 2.7 to 7.5 s on TF2); 100 epochs of `backprop` took 23.7 to 65.4 s.
- **TF2 against the paper.** The paper reports train RMSE 0.116 and test RMSE 0.0185
  for HKAN on TF2, on its own data file with its own identity-layer settings; this
  trainer measures 0.1157 and 0.01556 on its own generated data. The two are not the
  same data and are not compared further.

Two earlier batches of the same six runs, `results/hkan_*_20260930` and
`results/hkan_*_20260930b`, are superseded and kept only as records. The first ran
with a defect that rounded the inputs to float32 before the closed-form fit (its TF5
closed-form RMSE read 1.4e-04); the second ran before a review round changed the
seed streams, the scaling of TF1 and TF2, and the defaults (`decisions.md` D-031,
D-033 to D-046). In the reported batch the RMSE box plot of the TF5 `backprop` run
has a single labelled tick on its log axis; the plot now switches to a linear axis
inside one decade.

A second review round (`decisions.md` D-048 to D-055) changed the batch size of the
closed-form fit's closing forward pass, the dtype handling, the constant-target test
and, with `--val-fraction 0` only, the checkpoint monitor; none of these touches the
reported configurations. One repeat of the TF5 `closed_form` run was repeated on the
changed code: seed, train RMSE, test RMSE, per-layer RMSE and
`forward_deviation_rms` equal repeat 0 of `hkan_tf5_closed_form_20260930c` bit for
bit. The loss curves of the reported runs carry the legend labels of the code that
drew them ("train (fit rows)", "validation").

## 6. Tests

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_train/test_hkan -q
```
