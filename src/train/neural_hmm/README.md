# Neural HMM trainer

Trains a `NeuralHMM` (`src/dl_techniques/models/time_series/neural_hmm/`) on sequences
sampled from a known hidden Markov model, then scores it against that ground truth.
The model package README explains the model, its variants and its inference; this file
covers the trainer.

```bash
MPLBACKEND=Agg .venv/bin/python -m train.neural_hmm.train_neural_hmm --help
```

The model package lives at `src/dl_techniques/models/time_series/neural_hmm/`; the
trainer is at `src/train/neural_hmm/`, not under `src/train/time_series/`.

## 1. What a run does

1. Samples a train, a validation and a test split from one ground-truth HMM
   (`src/train/neural_hmm/data.py`). The three splits come from three disjoint random
   streams of `--seed`.
2. Builds the `--variant` preset with `create_neural_hmm`, sized to the data
   (`--n-states`, `--obs-type` with `--obs-dim` or `--vocab-size`, `--hidden-dim`;
   `--encoder` overrides the preset's encoder type).
3. Trains with stock `compile()` + `fit()`. The loss is the model's own `compute_loss`:
   the negative log-likelihood per valid step. The generative presets get `y=None`; the
   supervised `neural-crf-tagger` gets the true state labels. Adam with a global
   gradient-norm clip, built by `optimizer_builder`.
4. Keeps the epoch with the lowest `val_loss` as `best_model.keras`. Early stopping
   uses the same monitor and restores those weights; the trainer reloads the archive,
   checks it reproduces the in-memory model exactly, and scores the reloaded model.
5. Scores the test split, which nothing else uses:
   - the NLL per valid step, next to the same quantity under the true HMM;
   - the Viterbi state accuracy, next to the true HMM's own Viterbi accuracy and the
     majority-state rate (the accuracy of a constant predictor).

There is no custom `train_step`. `train.common.create_callbacks()` is not used: its
analyzer and default monitor are built for classification. The trainer assembles
`ModelCheckpoint`, `CSVLogger`, `EarlyStopping`, `TerminateOnNaN` and a one-line
per-epoch log itself.

## 2. The ground-truth HMM

The true HMM has no random parameter; `--seed` drives only the sampling.

| Part | Value |
|---|---|
| Initial distribution | uniform over the `K` states |
| Transitions | `--stay-prob` on the diagonal, `(1 - stay_prob) / (K - 1)` elsewhere |
| Gaussian emissions | state `k` emits `N(mu_k, I)` in `--obs-dim` dimensions; the means lie on the all-ones direction, consecutive states `--separation` standard deviations apart, centred on 0 |
| Categorical emissions | symbol `v` belongs to state `v mod K`; a state puts 0.8 of its mass (`OWN_SYMBOL_MASS`) uniformly on its own symbols and the rest uniformly on the others |

Valid lengths are uniform on `[--min-seq-len, --seq-len]`; shorter sequences are
right-padded (mask 0, state `-1`, observation 0).

## 3. What the scores mean

| Summary key | Meaning |
|---|---|
| `nll_kind` | `marginal` for the generative presets (`-log p(x)`), `conditional` for `neural-crf-tagger` (`-log p(states \| x)`) |
| `test_nll_per_step` | the model's NLL of that kind, summed over the test split and divided by its valid steps (the training loss's reduction) |
| `true_nll_per_step` | the same quantity under the true HMM, by an independent numpy scaled forward algorithm |
| `nll_gap` | `test_nll_per_step - true_nll_per_step` |
| `viterbi_acc_matched` | Viterbi accuracy after relabelling the model's states by the assignment that maximizes agreement with the true states (`scipy.optimize.linear_sum_assignment` on the confusion matrix). An unsupervised model names its states arbitrarily, so this is its accuracy |
| `state_mapping` | that assignment: entry `p` is the true state matched to model state `p` |
| `viterbi_acc` | direct Viterbi accuracy, no relabelling; `null` for the generative presets, whose labels are arbitrary |
| `true_viterbi_acc` | the accuracy of Viterbi decoding with the TRUE parameters, through the model package's own `inference.viterbi_decode`: the ceiling |
| `majority_rate` | the share of valid test steps holding the most frequent state: the floor |

## 4. Flags

| Flag | Default | Read by |
|---|---|---|
| `--variant` | `neural-hmm` | `create_neural_hmm`; one of `NeuralHMM.MODEL_VARIANTS` |
| `--obs-type` | `gaussian` | the data and the model (`gaussian` or `categorical`) |
| `--n-states` | 4 | the data and the model |
| `--obs-dim` | 2 (gaussian), `null` (categorical) | the data and the model, gaussian only; refused with categorical |
| `--vocab-size` | 8 (categorical), `null` (gaussian) | the data and the model, categorical only; must be `>= --n-states`; refused with gaussian |
| `--encoder` | the preset's | the model (`mlp`, `transformer`, `lstm`, `gru`, `conv`); refused for a preset that builds no encoder (`neural-hmm`) |
| `--hidden-dim` | 32 | the model (encoder, observation embedding, emission head); divisible by 4 for a transformer encoder |
| `--seq-len` | 50 | the data (padded length) and the model (transformer position table) |
| `--min-seq-len` | 25 | the data (shortest valid length) |
| `--num-train`, `--num-val`, `--num-test` | 2000, 200, 500 | the data |
| `--separation` | 4.0 (gaussian, `K > 1`), `null` otherwise | the data, gaussian with `K > 1` only; refused with categorical or `--n-states 1` |
| `--stay-prob` | 0.9 (`K > 1`), `null` (`K = 1`) | the data; in `(0, 1)`; refused with `--n-states 1` |
| `--epochs` | 60 | `fit` (maximum; early stopping may end sooner) |
| `--batch-size` | 64 | `fit` and the evaluation passes |
| `--learning-rate` | 0.003 | the Adam optimizer |
| `--gradient-clip-norm` | 1.0 | the optimizer's global-norm clip |
| `--patience` | 10 | early stopping on `val_loss` |
| `--seed` | 0 | the three data streams and the model initialization (`set_seeds`) |
| `--gpu` | none | `setup_gpu`, which OVERWRITES `CUDA_VISIBLE_DEVICES`; to pick a GPU from the environment instead, omit it |
| `--output-dir` | `<repo>/results` | the run directory's parent |
| `--experiment-name` | `neural_hmm_<variant>_<timestamp>` | the run directory's name |

A bad value is refused before anything is written: range errors at parse time (exit 2),
a combination the model refuses with a `ValueError` before the run directory exists. A
reused `--experiment-name` is refused the same way, and an existing run is never
overwritten.

**A flag the configuration would ignore is refused, not dropped.** The rule is the same
for every flag: given explicitly where nothing reads it, the run stops before anything is
written, even when the value given equals the default (`--vocab-size 8` on a gaussian
run is refused; `--vocab-size=8` and the abbreviation `--vocab 8` too).

| Flag | Refused when | How |
|---|---|---|
| `--obs-dim` | `--obs-type categorical` | parse time, exit 2 |
| `--vocab-size` | `--obs-type gaussian` (the default) | parse time, exit 2 |
| `--separation` | `--obs-type categorical`, or `--n-states 1` | parse time, exit 2 |
| `--stay-prob` | `--n-states 1` | parse time, exit 2 |
| `--encoder` | a preset that builds no encoder (`neural-hmm`) | `ValueError` from the model build, before the run directory exists |

The four data flags come from one table, `unread_arguments` in `data.py`, the same
function that decides what `make_true_hmm` reads. When such a flag is not given it is
set to `null`, so `config.json` and `results_summary.json` record `null` rather than a
default that played no part in the run. Every other flag is read by every
configuration (`--hidden-dim` sizes the emission head even without an encoder). The
model's own transformer knobs (`num_heads`, `ffn_type` and the like) are not trainer
flags.

## 5. Run directory

| File | Content |
|---|---|
| `config.json` | the parsed arguments |
| `run.log` | the run's log, one line per epoch |
| `training_log.csv` | `epoch`, `loss`, `val_loss` per epoch (`CSVLogger`) |
| `training_history.json` | the `fit` history |
| `best_model.keras` | the epoch with the lowest `val_loss`; loads with `keras.models.load_model` and no `custom_objects` once `dl_techniques.models.time_series.neural_hmm` is imported (the import registers the classes) |
| `results_summary.json` | strict JSON (a non-finite number is written as `null`): the keys of §3, the resolved arguments, `params`, `jit_compile` (the resolved value after `compile`), `epochs_run`, `best_checkpoint` (`file`, `epoch`, `val_loss`, `reload_max_abs_diff`, `reload_exact`; a NaN in one output facing a number in the other counts as an infinite difference, written `null`, with `reload_exact` false), `fit_seconds`, `total_seconds` and `visualizations` (`files`, `failed`) |
| `visualizations/loss_curve.png` | train and validation loss per epoch, with the true HMM's NLL per step on the test split as a dashed line |
| `visualizations/transitions.png` | the learned transition matrix (mean of `exp(log_A)` over the valid test steps; for a static transition, the matrix itself) beside the true one, both in true-label order |
| `visualizations/state_strip.png` | true and decoded states of the first 4 test sequences, in true-label order; padding blank |

A figure that fails to render is logged, listed under `visualizations.failed`, and does
not stop the run.

## 6. XLA

On a GPU, Keras compiles `fit` with XLA by default (`jit_compile="auto"`), except for a
model holding an `lstm`/`gru` encoder (the model package README, §10). The summary's
`jit_compile` records what the run used. The three runs of §7 (no encoder, a `conv`
encoder and a `transformer` encoder) all resolved it to `true` on GPU 1.

## 7. Measured runs

Three runs on 2026-10-14, one at a time on GPU 1 (RTX 4070), from the repository root:

```bash
# Run A: unsupervised, gaussian, K=4, well separated
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m train.neural_hmm.train_neural_hmm \
    --variant neural-hmm --obs-type gaussian --n-states 4 --obs-dim 2 --separation 4.0 \
    --stay-prob 0.9 --seq-len 50 --min-seq-len 25 --num-train 2000 --num-val 200 \
    --num-test 500 --epochs 60 --batch-size 64 --learning-rate 3e-3 --seed 0 \
    --experiment-name neural_hmm_runA_gaussian_k4_sep4_seed0_20261014
# Run B: unsupervised, categorical, K=3, 8 symbols, every other flag at its default
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m train.neural_hmm.train_neural_hmm \
    --variant io-hmm --obs-type categorical --n-states 3 --vocab-size 8 --stay-prob 0.9 \
    --seq-len 50 --min-seq-len 25 --num-train 2000 --num-val 200 --num-test 500 \
    --epochs 60 --seed 0 --experiment-name neural_hmm_runB_categorical_k3_v8_seed0_20261014
# Run C: supervised, gaussian, K=4, half the separation of run A
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m train.neural_hmm.train_neural_hmm \
    --variant neural-crf-tagger --obs-type gaussian --n-states 4 --obs-dim 2 --separation 2.0 \
    --stay-prob 0.9 --seq-len 50 --min-seq-len 25 --num-train 2000 --num-val 200 \
    --num-test 500 --epochs 30 --batch-size 64 --learning-rate 3e-3 --seed 0 \
    --experiment-name neural_hmm_runC_crf_gaussian_k4_sep2_seed0_20261014
```

Run directories (under the repository's `results/`, which git does not track):

- Run A: `results/neural_hmm_runA_gaussian_k4_sep4_seed0_20261014/`
- Run B: `results/neural_hmm_runB_categorical_k3_v8_seed0_20261014/`
- Run C: `results/neural_hmm_runC_crf_gaussian_k4_sep2_seed0_20261014/`

Every number below was copied by a script from those runs' `results_summary.json`, then
compared back against them. Values are rounded to 4 decimals; the JSON holds them in
full. One seed per configuration, so there is no seed-noise estimate.

| Summary key | Run A | Run B | Run C |
|---|---|---|---|
| `variant` | `neural-hmm` | `io-hmm` | `neural-crf-tagger` |
| `obs_type` | gaussian | categorical | gaussian |
| `n_states` | 4 | 3 | 4 |
| `encoder_type` | none | `conv` | `transformer` |
| `params` | 760 | 8740 | 27320 |
| `jit_compile` | true | true | true |
| `epochs_run` | 27 | 19 | 22 |
| `best_checkpoint.epoch` | 17 | 9 | 12 |
| `best_checkpoint.reload_exact` | true | true | true |
| `nll_kind` | marginal | marginal | conditional |
| `test_nll_per_step` | 3.2735 | 1.8340 | 0.1016 |
| `true_nll_per_step` | 3.2675 | 1.8329 | 0.0964 |
| `nll_gap` | 0.0060 | 0.0010 | 0.0052 |
| `viterbi_acc` | n/a | n/a | 0.9414 |
| `viterbi_acc_matched` | 0.9952 | 0.8871 | 0.9414 |
| `true_viterbi_acc` | 0.9952 | 0.9068 | 0.9452 |
| `majority_rate` | 0.2592 | 0.3573 | 0.2592 |
| `fit_seconds` | 22.2 | 29.1 | 45.7 |

The criteria were fixed before the runs (plan `plan-2026-10-13T192257-1bb3bf1f`, decision
D-012). They were not adjusted afterwards:

| Run | Criterion | Result |
|---|---|---|
| A | `nll_gap <= 0.05` | pass |
| A | `viterbi_acc_matched >= max(0.80, majority_rate + 0.30)` | pass |
| A | `viterbi_acc_matched >= true_viterbi_acc - 0.05` | pass |
| A | `best_checkpoint.reload_exact` | pass |
| B | `nll_gap <= 0.05` | pass |
| B | `viterbi_acc_matched >= majority_rate + 0.20` | pass |
| B | `viterbi_acc_matched >= true_viterbi_acc - 0.05` | pass |
| C | `viterbi_acc >= true_viterbi_acc - 0.03` | pass |
| C | `viterbi_acc >= majority_rate + 0.30` | pass |

All three runs ended by early stopping (patience 10) before reaching `--epochs`.

The figures show the following (epochs counted from 1, as `best_checkpoint.epoch`;
`training_log.csv` counts from 0):

- **Run A.** The learned transition matrix matches the true one to within 0.005 on every
  entry, after relabelling with `state_mapping`, so all four states are separate and none
  are merged. The decoded strips follow the true states except for a one-step boundary
  shift.
- **Run B.** `val_loss` is flat from epoch 2 and rises slightly after its minimum at
  epoch 9, while the training loss keeps falling. The learned transition diagonal is
  0.74 to 0.81 against the true 0.90. The `io-hmm` transitions depend on the input, so
  this heatmap is an average of per-step matrices, not one static matrix. The decoded
  strips lag the true state changes by 1 to 3 steps.
- **Run C.** `val_loss` reaches its minimum at epoch 12 and then rises while the
  training loss keeps falling. The learned transition matrix has a 0.86 to 0.87
  diagonal.

The transition numbers re-derive from the archives with the trainer's own `evaluate`.
The heatmaps were drawn on GPU 1; this command runs on the CPU and printed a largest
entry error of 0.0043 for run A, and diagonals of 0.744 to 0.808 for run B and 0.863 to
0.872 for run C. From the repository root:

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python - <<'PY'
import argparse, json, keras, numpy as np
import dl_techniques.models.time_series.neural_hmm  # registers the classes
from train.neural_hmm.train_neural_hmm import load_splits, evaluate, _relabel_matrix
for run in ("runA_gaussian_k4_sep4", "runB_categorical_k3_v8", "runC_crf_gaussian_k4_sep2"):
    d = f"results/neural_hmm_{run}_seed0_20261014"
    args = argparse.Namespace(**json.load(open(f"{d}/config.json")))
    s = json.load(open(f"{d}/results_summary.json"))
    test = load_splits(args)["test"]
    model = keras.models.load_model(f"{d}/best_model.keras")
    order = np.arange(args.n_states) if s["mode"] == "discriminative" else np.array(s["state_mapping"])
    learned = _relabel_matrix(evaluate(model, test, args.batch_size)["transition"], order)
    print(run, "max|learned-true| = %.4f" % np.abs(learned - test.true_hmm.transition).max(),
          "diagonal =", np.diag(learned).round(3).tolist())
PY
```

**Sequence length on the GPU.** Each run's `best_model.keras` was reloaded on GPU 1 and
called with `predict` at three lengths in one process: T=50, 30 and 70 for runs A and B,
and T=50, 30 and 40 for run C, whose `transformer` position table has 50 rows. Every call
returned the expected shapes and finite outputs:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python - <<'PY'
import keras, numpy as np
import dl_techniques.models.time_series.neural_hmm  # registers the classes
rng = np.random.default_rng(0)
for run, lengths in (("runA_gaussian_k4_sep4", (50, 30, 70)), ("runB_categorical_k3_v8", (50, 30, 70)),
                     ("runC_crf_gaussian_k4_sep2", (50, 30, 40))):
    model = keras.models.load_model(f"results/neural_hmm_{run}_seed0_20261014/best_model.keras")
    for t in lengths:
        obs = (rng.integers(0, model.vocab_size, (5, t)).astype("int32") if model.obs_type == "categorical"
               else rng.normal(size=(5, t, model.obs_dim)).astype("float32"))
        out = model.predict({"obs": obs, "mask": np.ones((5, t), "float32")}, batch_size=5, verbose=0)
        print(run, t, {k: v.shape for k, v in out.items()}, all(np.isfinite(v).all() for v in out.values()))
PY
```

**TF32.** On GPU 1, `predict` (XLA) and an eager call on the same run-A batch (5
sequences of T=50) differ by up to 3.5e-2 in `log_emit`. That difference comes from
TensorFlow's default TF32 matmuls: with
`tf.config.experimental.enable_tensor_float_32_execution(False)` it falls to 1.5e-5. The
runs above used the default (TF32 on). The argument after `-` is `on` or `off`:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python - off <<'PY'
import sys, keras, numpy as np, tensorflow as tf
import dl_techniques.models.time_series.neural_hmm  # registers the classes
tf.config.experimental.enable_tensor_float_32_execution(sys.argv[1] == "on")
model = keras.models.load_model("results/neural_hmm_runA_gaussian_k4_sep4_seed0_20261014/best_model.keras")
x = {"obs": np.random.default_rng(0).normal(size=(5, 50, 2)).astype("float32"), "mask": np.ones((5, 50), "float32")}
pred, eager = model.predict(x, batch_size=5, verbose=0), model(x, training=False)
print("TF32", sys.argv[1], "max|predict - eager| log_emit = %.2e"
      % np.abs(pred["log_emit"] - keras.ops.convert_to_numpy(eager["log_emit"])).max())
PY
```

## 8. Tests

`tests/test_train/test_neural_hmm/`:

- `test_data.py`: the sampler's layout and statistics, and the references against a
  brute-force enumeration of every state path and against the model package's
  `inference.forward_log_likelihood` and `inference.viterbi_decode`, on an HMM with an
  asymmetric transition matrix and a non-uniform initial distribution (so a transposed
  matrix fails); `unread_arguments` against what `make_true_hmm` actually reads;
- `test_cli_contract.py`: one contract row per flag, the `--gpu` sentinel, two real
  runs recording what the sampler, the model factory, the optimizer and `fit` received,
  every ignored-flag refusal of §4 and the `null` they leave in `config.json`;
- `test_train_neural_hmm.py`: real tiny runs of `main()` for gaussian `neural-hmm`,
  categorical `io-hmm` and gaussian `neural-crf-tagger`, every artifact, the archive
  reload (including a run whose best epoch is not its last, so that the exact reload
  depends on `restore_best_weights=True`), the refusals and a failing figure.

```bash
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_train/test_neural_hmm/ -q
```
