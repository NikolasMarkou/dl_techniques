# HKAN trainer

Trains `HKAN` (`src/dl_techniques/models/general_purpose/hkan/`), the Hierarchical
Kolmogorov-Arnold Network of Dudek and Rodak (arXiv:2501.18199), on a synthetic
target of the paper or on your own CSV files. The model package README explains the
model; this file covers the trainer.

```bash
MPLBACKEND=Agg .venv/bin/python -m train.hkan.train_hkan --help
```

## 1. Training modes

`--training-mode` selects how the weights are found.

| Mode | What runs | Rows fitted on |
|---|---|---|
| `closed_form` (default) | `HKAN.fit_closed_form`: the paper's layer-by-layer least squares, in float64, no gradients, no epochs | every train row |
| `backprop` | stock `compile()` + `fit()` with Adam and MSE from the model's initial state | train rows minus the validation part |
| `closed_form_then_backprop` | the closed-form fit, then stock `fit()` starting from the closed-form weights | train rows minus the validation part |

The basis centers are fixed in every mode. In `backprop` mode with
`--centers data` they are drawn from the data before training starts.

In the two backprop modes `--val-fraction` (default 0.1) of the TRAIN split is held
out for checkpoint selection. The test split is never used for fitting or selection.
`--epochs`, `--batch-size`, `--learning-rate` and `--val-fraction` are ignored in
`closed_form` mode.

There is no custom `train_step`. `train.common.create_callbacks()` is not used: it
always adds early stopping, which would stop repeats at different epochs, and it has
no role in the closed-form mode. The backprop modes use `ModelCheckpoint` and
`CSVLogger` directly.

## 2. Data

`--dataset` is one of `tf1`, `tf2`, `tf3`, `tf4`, `tf5`, `tf5_5` (generated locally
from `--seed` with the sample counts of the paper's Table VI) or `csv`.

| Name | Function | Inputs | Train / test rows | Scaled to [0, 1] |
|---|---|---|---|---|
| `tf1` | `(2 x1 - 1)(2 x2 - 1)` on `[0, 1]^2` | 2 | 5000 / 10000 | no |
| `tf2` | `sum_i sin(20 exp(x_i)) x_i^2` on `[0, 1]^2`, noise `U(-0.2, 0.2)` on the train targets | 2 | 5000 / 10000 | no |
| `tf3` | `-sum_i x_i sin(sqrt(abs(x_i)))` on `[-500, 500]^2` | 2 | 5000 / 10000 | yes |
| `tf4` | `1 - cos(2 pi r) + 0.1 r`, `r = sqrt(sum_i x_i^2)`, on `[-4, 4]^10` | 10 | 3750 / 1250 | yes |
| `tf5` | `-sum_i sin(x_i) sin^20(i x_i^2 / pi)` on `[0, pi]^2` | 2 | 5000 / 10000 | yes |
| `tf5_5` | the same with 5 inputs | 5 | 7500 / 2500 | yes |

Three readings of the paper are this trainer's own and are stated in
`src/train/hkan/data.py`: only TF3, TF4 and TF5 are scaled (the paper names only
those as normalized); the TF2 noise is added to the unscaled train targets and the
test targets are noiseless; the exponent 2 in TF2 is a reading of a garbled PDF line.
Scaling statistics always come from the train split.

`--dataset csv --train-csv A --test-csv B` reads two comma-separated files with no
header, one sample per row, the last column the target. `--scale-csv` min-max scales
with the train file's statistics. The paper's 18 benchmark CSV files are not shipped
here and are not downloaded: the upstream repository carries no licence.

## 3. Flags

Per-layer flags take one value (used for every layer) or a comma-separated value per
layer. The number of layers is the number of `--hidden-units` entries plus one; the
last layer always has one output.

| Flag | Default | Meaning |
|---|---|---|
| `--training-mode` | `closed_form` | see section 1 |
| `--dataset` | `tf5` | see section 2 |
| `--train-csv`, `--test-csv` | none | files for `--dataset csv` |
| `--scale-csv` | off | min-max scale the CSV data with train statistics |
| `--num-train-samples`, `--num-test-samples` | paper counts | rows of a synthetic target |
| `--repeats` | 1 | independent fits with derived seeds (the paper uses 50) |
| `--seed` | 0 | seed of the data and of the per-repeat seeds |
| `--hidden-units` | `64` | hidden widths; empty string for a one-layer model |
| `--num-basis` | `10` | basis functions per block, per layer |
| `--basis` | `sigmoid` | `sigmoid`, `gaussian`, `relu`, `tanh`, `softplus`, `identity`, per layer |
| `--slope` | `5` | basis slope (the paper's sigma), per layer; `identity` ignores it |
| `--centers` | `random` | `random`, `equally_spaced`, `data`, per layer |
| `--l2-block` | `0.01` | ridge strength of the block fits, per layer (closed-form modes) |
| `--l2-mix` | `0` | ridge strength of the connecting fits, per layer (closed-form modes) |
| `--no-block-bias`, `--no-bias` | off | drop the block or the output intercepts (the paper's bias-free equations) |
| `--chunk-size` | auto | outputs solved at once in the closed-form fit (bounds memory) |
| `--epochs` | 100 | backprop epochs |
| `--batch-size` | 64 | backprop batch size |
| `--learning-rate` | 1e-3 | Adam learning rate |
| `--val-fraction` | 0.1 | part of the train split held out in the backprop modes |
| `--predict-batch-size` | 256 | batch size of every prediction pass |
| `--gpu` | none | GPU index |
| `--output-dir` | `<repo>/results` | base directory |
| `--experiment-name` | `hkan_<timestamp>` | run directory name |

## 4. Run directory

```
results/<experiment-name>/
  config.json              the parsed arguments
  run.log                  the run's log
  results_summary.json     strict JSON, see below
  final_model.keras        repeat 0, weights at the end of training
  visualizations/
    test_predictions.png   predicted against target and residuals, test split, repeat 0
    input_importance.png   closed-form modes: mean R^2 of the first-layer blocks per input
    rmse_over_repeats.png  when --repeats > 1
    loss_curve.png         backprop modes
  best_model.keras         backprop modes: best validation epoch of repeat 0
  training_log.csv         backprop modes: per-epoch loss of repeat 0
  training_history.json    backprop modes
```

A reused `--experiment-name` is refused before anything is written. Nothing in an
existing run directory is overwritten or deleted.

`results_summary.json` keys: `training_mode`, `dataset`, `num_inputs`, `train_rows`,
`fit_rows`, `test_rows`, `params`, `seed`, `repeats`; `train_rmse`, `test_rmse` and
`fit_seconds`, each with `median`, `iqr`, `min`, `max` over the repeats;
`closed_form_phase` (the RMSE after the closed-form fit and before `fit()`, two-phase
mode only, otherwise null); `best_checkpoint` (repeat 0, backprop modes, otherwise
null); `input_importance` (repeat 0, null in `backprop` mode); `repeat_results` (one
entry per repeat with its seed and scores); `visualizations` (`files` written and
`failed` figure names).

Reported RMSEs come from the weights at the end of training in every repeat. They
are computed in float64 from the model's float32 predictions. Train RMSE is on the
rows the model was fitted on. With `--repeats N`, the data is generated once and
each repeat draws new centers and new initial weights.

## 5. Measured runs

Six runs on one RTX 4070, 2026-09-30, 5 repeats each, `--seed 0`. The settings and
the rules for reading the results were written down before the runs
(`decisions.md` D-030 of plan-2026-09-30T082355-4d999dbc). The backprop arms are
untuned: one learning rate, 100 epochs, no schedule. Every number below is recomputed
from the `results_summary.json` files of `results/hkan_{tf5,tf2}_<mode>_20260930b`.

Common part of every command:

```bash
MPLBACKEND=Agg .venv/bin/python -m train.hkan.train_hkan --gpu 1 --repeats 5 --seed 0 \
    --epochs 100 --batch-size 64 --val-fraction 0.1 <CONFIG> --training-mode <MODE>
```

with `--learning-rate 1e-3` for `backprop` and `1e-4` for `closed_form_then_backprop`.

**TF5**, the configuration of the authors' tutorial (108529 parameters):
`--dataset tf5 --hidden-units 912 --basis tanh,identity --slope 50 --num-basis 23,10 --centers random,data --l2-block 0.01,0.1`

| Mode | Train RMSE median (IQR) | Test RMSE median (IQR) | Fit seconds, min to max |
|---|---|---|---|
| `closed_form` | 1.291e-07 (2.0e-09) | 1.293e-07 (7.9e-10) | 8.6 to 12.7 |
| `backprop` | 6.175e-03 (3.8e-03) | 6.169e-03 (3.7e-03) | 23.8 to 26.5 |
| `closed_form_then_backprop` | 3.081e-03 (3.1e-03) | 3.081e-03 (3.1e-03) | 31.0 to 40.8 |

**TF2**, the paper's Table V row with this trainer's choice for the identity top
layer (26990 parameters):
`--dataset tf2 --hidden-units 48,11 --basis tanh,softplus,identity --slope 22,18,1 --num-basis 17,21,10 --centers random,data,data --l2-block 0.1,1,0.1`

| Mode | Train RMSE median (IQR) | Test RMSE median (IQR) | Fit seconds, min to max |
|---|---|---|---|
| `closed_form` | 0.1198 (0.0009) | 0.0350 (0.0040) | 3.5 to 4.8 |
| `backprop` | 0.1261 (0.0067) | 0.0600 (0.0139) | 27.7 to 32.7 |
| `closed_form_then_backprop` | 0.1146 (0.0017) | 0.0412 (0.0078) | 29.5 to 34.0 |

What these runs show, by the rules written beforehand:

- **TF5 closed form sits at the float32 limit.** 1.3e-07 is the accuracy of a
  float32 forward pass on this model; the float64 fit itself is far below it. The
  paper's 1e-15 figure for TF5 is a float64 number and is not reachable here.
- **Fine-tuning the TF5 closed-form fit with Adam made it worse in all 5 repeats.**
  The closed-form phase scored a train RMSE of 1.35e-07 to 1.66e-07 on the fit rows;
  after 100 epochs at learning rate 1e-4 it was 2900 to 62500 times larger. The
  loss curve shows why: Adam's first steps move every weight by about the learning
  rate whatever the gradient, which breaks an interpolating solution at once (mean
  train MSE of epoch 1 of repeat 0: 5.1e-04); the loss then falls to 1.6e-10 at
  epoch 15 and becomes unstable again (up to 1.8e-04 after epoch 20). The
  best-validation checkpoint of repeat 0 (epoch 14) has a test RMSE of 1.6e-05,
  still about 100 times the closed-form value. The pre-written rule calls this case
  "not informative at the float32 floor" (the closed-form error is already below
  1e-5), so no claim is made about whether a gentler optimizer could help.
- **On TF2 the fine-tune is a tie or mixed.** Train RMSE after the fine-tune is
  0.945 to 0.995 of the closed-form phase (below 0.95 in 2 of 5 repeats, so not
  "lowers" by the rule); the test RMSE ratio ranges from 0.84 to 1.50. The train
  RMSE of about 0.12 is the noise level (the standard deviation of `U(-0.2, 0.2)`
  is 0.1155), so fitting the train rows more closely is fitting noise.
- **Backprop alone against closed form.** TF5: median test RMSE is 47700 times the
  closed-form one ("worse"). TF2: 1.72 times ("comparable" by the rule, which treats
  anything within a factor of 2 as no difference; the quartiles do not overlap:
  0.0553 to 0.0693 for `backprop`, 0.0348 to 0.0389 for `closed_form`). Confound: `closed_form` fits on all 5000
  train rows and `backprop` on 4500. Against the closed-form phase of the two-phase
  run, which uses the same 4500 rows, the ratios are 39900 (TF5) and 1.64 (TF2).
- **Cost.** The closed-form fit took 3.5 to 12.7 s per repeat; 100 epochs of
  backprop took 23.8 to 40.8 s.

An earlier batch of the same six runs (`results/hkan_*_20260930`, no `b`) was made
with a defect in this trainer that rounded the inputs to float32 before the
closed-form fit; its TF5 closed-form RMSE read 1.4e-04. Those directories are
superseded and kept only as the record of that defect (`decisions.md` D-031).

One figure in the second batch has a known flaw: in
`hkan_tf5_closed_form_then_backprop_20260930b/visualizations/loss_curve.png` the
legend box covers part of the closed-form line. The legend has since been moved
outside the axes.

## 6. Tests

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_train/test_hkan -q
```
