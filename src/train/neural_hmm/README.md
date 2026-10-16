# Neural HMM trainer

Trains a `NeuralHMM` (`src/dl_techniques/models/probabilistic/neural_hmm/`) on sequences
sampled from a known hidden Markov model, then scores it against that ground truth.
The model package README explains the model, its variants and its inference; this file
covers the trainer.

```bash
MPLBACKEND=Agg .venv/bin/python -m train.neural_hmm.train_neural_hmm --help
```

The model package lives at `src/dl_techniques/models/probabilistic/neural_hmm/`; the
trainer is at `src/train/neural_hmm/`.

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

**The matched-accuracy convention.** `viterbi_acc_matched` is one-to-one matched accuracy: the mapping from model states to true states is the one-to-one (Hungarian) assignment fitted on the same test labels it is then scored on. It is therefore not a held-out measurement: fitting the assignment on those labels can only raise the score relative to a mapping fitted elsewhere, by an amount that shrinks with the number of scored steps against the `K!` candidate assignments (`K <= 4` here, and 18673 to 18979 scored test steps per run, `test_valid_steps`). The true model is held to the same rule: its states are named after the true ones, so `true_viterbi_acc` is its direct accuracy, and the same Hungarian assignment fitted on its own test path is the identity on all nine runs in §7 (re-derived read-only from each run's `config.json` through `load_splits` and `true_viterbi_path`), so its matched and direct accuracies are equal and the comparison is like for like.

## 4. Flags

| Flag | Default | Read by |
|---|---|---|
| `--variant` | `neural-hmm` | `create_neural_hmm`; one of `NeuralHMM.MODEL_VARIANTS` |
| `--obs-type` | `gaussian` | the data and the model (`gaussian` or `categorical`) |
| `--n-states` | 4 | the data and the model; at least 2 (parse time, exit 2) |
| `--obs-dim` | 2 (gaussian), `null` (categorical) | the data and the model, gaussian only; refused with categorical |
| `--vocab-size` | 8 (categorical), `null` (gaussian) | the data and the model, categorical only; must be `>= --n-states`; refused with gaussian |
| `--encoder` | the preset's | the model (`mlp`, `transformer`, `lstm`, `gru`, `conv`); refused for a preset that builds no encoder (`neural-hmm`) |
| `--hidden-dim` | 32 | the model (encoder, observation embedding, emission head); divisible by 4 for a transformer encoder |
| `--seq-len` | 50 | the data (padded length) and, with a `transformer` encoder only, the model (its position table, `max_seq_len`) |
| `--min-seq-len` | 25 | the data (shortest valid length) |
| `--num-train`, `--num-val`, `--num-test` | 2000, 200, 500 | the data |
| `--separation` | 4.0 (gaussian), `null` (categorical) | the data, gaussian only; refused with categorical |
| `--stay-prob` | 0.9 | the data; in `(0, 1)` |
| `--epochs` | 60 | `fit` (maximum; early stopping may end sooner) |
| `--batch-size` | 64 | `fit` and the evaluation passes |
| `--learning-rate` | 0.003 | the Adam optimizer |
| `--gradient-clip-norm` | 1.0 | the optimizer's global-norm clip |
| `--patience` | 10 | early stopping on `val_loss` |
| `--seed` | 0 | the three data streams and the model initialization (`set_seeds`) |
| `--gpu` | none | `setup_gpu`: counts within an exported `CUDA_VISIBLE_DEVICES`, names a physical GPU only when it is unset |
| `--output-dir` | `<repo>/results` | the run directory's parent |
| `--experiment-name` | `neural_hmm_<variant>_<timestamp>` | the run directory's name |

A bad value is refused before anything is written: range errors at parse time (exit 2),
`--n-states 1` among them for every preset (the model refuses a one-state HMM too: its
initial and transition distributions would be constants that never train), and a
combination the model refuses with a `ValueError` before the run directory exists. A
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
| `--separation` | `--obs-type categorical` | parse time, exit 2 |
| `--encoder` | a preset that builds no encoder (`neural-hmm`) | `ValueError` from the model build, before the run directory exists |

The three data flags come from one table, `unread_arguments` in `data.py`, the same
function that decides what `make_true_hmm` reads (its one-state rows, `separation` and
`stay_prob` at `K = 1`, are never reached from the CLI, which refuses `--n-states 1`
first). When such a flag is not given it is
set to `null`, so `config.json` and `results_summary.json` record `null` rather than a
default that played no part in the run. Every other flag is read by every
configuration (`--hidden-dim` sizes the emission head even without an encoder). The
model's own transformer knobs (`num_heads`, `ffn_type` and the like) are not trainer
flags.
The model refuses a non-default value for an argument its configuration does not read
(`NeuralHMM.unread_arguments`), so the trainer hands `--seq-len` to the model as
`max_seq_len` only when the model builds a `transformer` encoder.

## 5. Run directory

| File | Content |
|---|---|
| `config.json` | the parsed arguments |
| `run.log` | the run's log, one line per epoch |
| `training_log.csv` | `epoch`, `loss`, `val_loss` per epoch (`CSVLogger`) |
| `training_history.json` | the `fit` history |
| `best_model.keras` | the epoch with the lowest `val_loss`; loads with `keras.models.load_model` and no `custom_objects` once `dl_techniques.models.probabilistic.neural_hmm` is imported (the import registers the classes) |
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

Nine runs on 2026-10-14, one at a time on GPU 1 (RTX 4070), from the repository root.
Runs A, D and E train three generative presets on the same data (same truth, same
`--seed`, so the same test split); runs B and C were each repeated with `--seed` 1 and 2.

```bash
# Run A: unsupervised, gaussian, K=4, well separated
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m train.neural_hmm.train_neural_hmm \
    --variant neural-hmm --obs-type gaussian --n-states 4 --obs-dim 2 --separation 4.0 \
    --stay-prob 0.9 --seq-len 50 --min-seq-len 25 --num-train 2000 --num-val 200 \
    --num-test 500 --epochs 60 --batch-size 64 --learning-rate 3e-3 --seed 0 \
    --experiment-name neural_hmm_runA_gaussian_k4_sep4_seed0_20261014
# Run D: run A's data, autoregressive emission (a superset of the true model)
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m train.neural_hmm.train_neural_hmm \
    --variant autoregressive-neural-hmm --obs-type gaussian --n-states 4 --obs-dim 2 \
    --separation 4.0 --stay-prob 0.9 --seq-len 50 --min-seq-len 25 --num-train 2000 \
    --num-val 200 --num-test 500 --epochs 60 --batch-size 64 --learning-rate 3e-3 --seed 0 \
    --experiment-name neural_hmm_runD_autoregressive_gaussian_k4_sep4_seed0_20261014
# Run E: run A's data, left-to-right segmenter (misspecified, see below)
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m train.neural_hmm.train_neural_hmm \
    --variant left-to-right-segmenter --obs-type gaussian --n-states 4 --obs-dim 2 \
    --separation 4.0 --stay-prob 0.9 --seq-len 50 --min-seq-len 25 --num-train 2000 \
    --num-val 200 --num-test 500 --epochs 60 --batch-size 64 --learning-rate 3e-3 --seed 0 \
    --experiment-name neural_hmm_runE_segmenter_gaussian_k4_sep4_seed0_20261014
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

Seeds 1 and 2 of runs B and C repeat those commands with `--seed 1` or `--seed 2` and
`seed0` replaced by `seed1` or `seed2` in `--experiment-name`; nothing else changes. The
seed drives the data streams as well as the initialization (section 2), so each seed
draws its own training, validation and test sequences, and its own `true_nll_per_step`
and `true_viterbi_acc`.

Run directories (under the repository's `results/`, which git does not track):

- Run A: `results/neural_hmm_runA_gaussian_k4_sep4_seed0_20261014/`
- Run D: `results/neural_hmm_runD_autoregressive_gaussian_k4_sep4_seed0_20261014/`
- Run E: `results/neural_hmm_runE_segmenter_gaussian_k4_sep4_seed0_20261014/`
- Run B: `results/neural_hmm_runB_categorical_k3_v8_seed0_20261014/`,
  `results/neural_hmm_runB_categorical_k3_v8_seed1_20261014/`,
  `results/neural_hmm_runB_categorical_k3_v8_seed2_20261014/`
- Run C: `results/neural_hmm_runC_crf_gaussian_k4_sep2_seed0_20261014/`,
  `results/neural_hmm_runC_crf_gaussian_k4_sep2_seed1_20261014/`,
  `results/neural_hmm_runC_crf_gaussian_k4_sep2_seed2_20261014/`

Every number in the tables below was printed by a script from those runs'
`results_summary.json`, then compared back against them by a second script. Values are
rounded to 4 decimals; the JSON holds them in full.

Run E is misspecified by construction: the `left-to-right-segmenter` preset starts in
state 0 and can only stay or move to the next state, so it cannot return to a state it
has left, while the true HMM is ergodic with a uniform start; its accuracy below is
descriptive and is not a recovery result.

| Summary key | Run A | Run D | Run E |
|---|---|---|---|
| `variant` | `neural-hmm` | `autoregressive-neural-hmm` | `left-to-right-segmenter` |
| `obs_type` | gaussian | gaussian | gaussian |
| `n_states` | 4 | 4 | 4 |
| `seed` | 0 | 0 | 0 |
| `encoder_type` | none | `lstm` | `transformer` |
| `params` | 760 | 18584 | 28952 |
| `jit_compile` | true | false | true |
| `epochs_run` | 27 | 32 | 30 |
| `best_checkpoint.epoch` | 17 | 22 | 20 |
| `best_checkpoint.reload_exact` | true | true | true |
| `nll_kind` | marginal | marginal | marginal |
| `test_nll_per_step` | 3.2735 | 3.2705 | 3.6376 |
| `true_nll_per_step` | 3.2675 | 3.2675 | 3.2675 |
| `nll_gap` | 0.0060 | 0.0029 | 0.3701 |
| `viterbi_acc` | n/a | n/a | n/a |
| `viterbi_acc_matched` | 0.9952 | 0.9950 | 0.4316 |
| `true_viterbi_acc` | 0.9952 | 0.9952 | 0.9952 |
| `majority_rate` | 0.2592 | 0.2592 | 0.2592 |
| `fit_seconds` | 22.2 | 133.8 | 47.7 |

Runs B and C over three seeds, as mean, min and max of the per-seed values:

| Summary key | Run B seed 0 | Run B seed 1 | Run B seed 2 |
|---|---|---|---|
| `variant` | `io-hmm` | `io-hmm` | `io-hmm` |
| `obs_type` | categorical | categorical | categorical |
| `n_states` | 3 | 3 | 3 |
| `seed` | 0 | 1 | 2 |
| `encoder_type` | `conv` | `conv` | `conv` |
| `params` | 8740 | 8740 | 8740 |
| `jit_compile` | true | true | true |
| `epochs_run` | 19 | 14 | 21 |
| `best_checkpoint.epoch` | 9 | 4 | 11 |
| `best_checkpoint.reload_exact` | true | true | true |
| `nll_kind` | marginal | marginal | marginal |
| `test_nll_per_step` | 1.8340 | 1.8508 | 1.8584 |
| `true_nll_per_step` | 1.8329 | 1.8496 | 1.8560 |
| `nll_gap` | 0.0010 | 0.0012 | 0.0025 |
| `viterbi_acc` | n/a | n/a | n/a |
| `viterbi_acc_matched` | 0.8871 | 0.8987 | 0.8927 |
| `true_viterbi_acc` | 0.9068 | 0.9086 | 0.8988 |
| `majority_rate` | 0.3573 | 0.3394 | 0.3553 |
| `fit_seconds` | 29.1 | 28.8 | 29.6 |

| Summary key | Run C seed 0 | Run C seed 1 | Run C seed 2 |
|---|---|---|---|
| `variant` | `neural-crf-tagger` | `neural-crf-tagger` | `neural-crf-tagger` |
| `obs_type` | gaussian | gaussian | gaussian |
| `n_states` | 4 | 4 | 4 |
| `seed` | 0 | 1 | 2 |
| `encoder_type` | `transformer` | `transformer` | `transformer` |
| `params` | 27320 | 27320 | 27320 |
| `jit_compile` | true | true | true |
| `epochs_run` | 22 | 18 | 21 |
| `best_checkpoint.epoch` | 12 | 8 | 11 |
| `best_checkpoint.reload_exact` | true | true | true |
| `nll_kind` | conditional | conditional | conditional |
| `test_nll_per_step` | 0.1016 | 0.1079 | 0.1146 |
| `true_nll_per_step` | 0.0964 | 0.1023 | 0.1076 |
| `nll_gap` | 0.0052 | 0.0056 | 0.0069 |
| `viterbi_acc` | 0.9414 | 0.9388 | 0.9336 |
| `viterbi_acc_matched` | 0.9414 | 0.9388 | 0.9336 |
| `true_viterbi_acc` | 0.9452 | 0.9415 | 0.9384 |
| `majority_rate` | 0.2592 | 0.2648 | 0.2693 |
| `fit_seconds` | 45.7 | 38.4 | 46.0 |

| Variant | Summary key | seed 0 | seed 1 | seed 2 | mean | min | max |
|---|---|---|---|---|---|---|---|
| `io-hmm` | `nll_gap` | 0.0010 | 0.0012 | 0.0025 | 0.0016 | 0.0010 | 0.0025 |
| `io-hmm` | `viterbi_acc_matched` | 0.8871 | 0.8987 | 0.8927 | 0.8928 | 0.8871 | 0.8987 |
| `io-hmm` | `true_viterbi_acc` | 0.9068 | 0.9086 | 0.8988 | 0.9047 | 0.8988 | 0.9086 |
| `neural-crf-tagger` | `nll_gap` | 0.0052 | 0.0056 | 0.0069 | 0.0059 | 0.0052 | 0.0069 |
| `neural-crf-tagger` | `viterbi_acc` | 0.9414 | 0.9388 | 0.9336 | 0.9379 | 0.9336 | 0.9414 |
| `neural-crf-tagger` | `true_viterbi_acc` | 0.9452 | 0.9415 | 0.9384 | 0.9417 | 0.9384 | 0.9452 |

The criteria were fixed before the runs (plan `plan-2026-10-13T192257-1bb3bf1f`,
decisions D-012, D-022 and D-025) and were not adjusted afterwards. The bars for runs B
and C were set on seed 0; seeds 1 and 2 test them. The leak canary applies to every run:
no model may beat the generating distribution's NLL by more than 0.01 nats per step, so a
negative gap beyond that would mean information from the future reached the score. Run E
is held to the canary only:

| Run | Criterion | Value | Bound | Result |
|---|---|---|---|---|
| A | canary: `nll_gap >= -0.01` | 0.0060 | -0.0100 | pass |
| D | canary: `nll_gap >= -0.01` | 0.0029 | -0.0100 | pass |
| E | canary: `nll_gap >= -0.01` | 0.3701 | -0.0100 | pass |
| B seed 0 | canary: `nll_gap >= -0.01` | 0.0010 | -0.0100 | pass |
| B seed 1 | canary: `nll_gap >= -0.01` | 0.0012 | -0.0100 | pass |
| B seed 2 | canary: `nll_gap >= -0.01` | 0.0025 | -0.0100 | pass |
| C seed 0 | canary: `nll_gap >= -0.01` | 0.0052 | -0.0100 | pass |
| C seed 1 | canary: `nll_gap >= -0.01` | 0.0056 | -0.0100 | pass |
| C seed 2 | canary: `nll_gap >= -0.01` | 0.0069 | -0.0100 | pass |
| A | `nll_gap <= 0.05` | 0.0060 | 0.0500 | pass |
| A | `viterbi_acc_matched >= max(0.80, majority_rate + 0.30)` | 0.9952 | 0.8000 | pass |
| A | `viterbi_acc_matched >= true_viterbi_acc - 0.05` | 0.9952 | 0.9452 | pass |
| A | `best_checkpoint.reload_exact` | true | true | pass |
| D | `nll_gap <= 0.05` | 0.0029 | 0.0500 | pass |
| D | `viterbi_acc_matched >= max(0.80, majority_rate + 0.30)` | 0.9950 | 0.8000 | pass |
| D | `viterbi_acc_matched >= true_viterbi_acc - 0.05` | 0.9950 | 0.9452 | pass |
| D | `best_checkpoint.reload_exact` | true | true | pass |
| B seed 0 | `nll_gap <= 0.05` | 0.0010 | 0.0500 | pass |
| B seed 0 | `viterbi_acc_matched >= majority_rate + 0.20` | 0.8871 | 0.5573 | pass |
| B seed 0 | `viterbi_acc_matched >= true_viterbi_acc - 0.05` | 0.8871 | 0.8568 | pass |
| B seed 1 | `nll_gap <= 0.05` | 0.0012 | 0.0500 | pass |
| B seed 1 | `viterbi_acc_matched >= majority_rate + 0.20` | 0.8987 | 0.5394 | pass |
| B seed 1 | `viterbi_acc_matched >= true_viterbi_acc - 0.05` | 0.8987 | 0.8586 | pass |
| B seed 2 | `nll_gap <= 0.05` | 0.0025 | 0.0500 | pass |
| B seed 2 | `viterbi_acc_matched >= majority_rate + 0.20` | 0.8927 | 0.5553 | pass |
| B seed 2 | `viterbi_acc_matched >= true_viterbi_acc - 0.05` | 0.8927 | 0.8488 | pass |
| C seed 0 | `viterbi_acc >= true_viterbi_acc - 0.03` | 0.9414 | 0.9152 | pass |
| C seed 0 | `viterbi_acc >= majority_rate + 0.30` | 0.9414 | 0.5592 | pass |
| C seed 1 | `viterbi_acc >= true_viterbi_acc - 0.03` | 0.9388 | 0.9115 | pass |
| C seed 1 | `viterbi_acc >= majority_rate + 0.30` | 0.9388 | 0.5648 | pass |
| C seed 2 | `viterbi_acc >= true_viterbi_acc - 0.03` | 0.9336 | 0.9084 | pass |
| C seed 2 | `viterbi_acc >= majority_rate + 0.30` | 0.9336 | 0.5693 | pass |

All nine runs ended by early stopping (patience 10) before reaching `--epochs`.

The figures show the following (epochs counted from 1, as `best_checkpoint.epoch`;
`training_log.csv` counts from 0):

- **Run A.** The learned transition matrix matches the true one to within 0.005 on every
  entry, after relabelling with `state_mapping`, so all four states are separate and none
  are merged. The decoded strips follow the true states except for a one-step boundary
  shift.
- **Run D.** The loss curves flatten by about epoch 8, just above the true-model line.
  The learned transition matrix matches the true one to within 0.003 on every entry, and
  the decoded strips follow the true states except for one boundary shifted by a step.
- **Run E.** Both loss curves level off well above the true-model line. After
  relabelling, the learned matrix is a chain through the true states 1, 3, 0 and 2, with
  state 2 absorbing (diagonal 1.0), so every decoded strip visits the states in that
  order and never returns: a sequence that revisits a state is decoded as a different
  state the second time.
- **Run B.** On seed 0, `val_loss` is flat from epoch 2 and rises slightly after its
  minimum at epoch 9, while the training loss keeps falling, and the decoded strips lag
  the true state changes by 1 to 3 steps. The learned transition diagonal is 0.74 to
  0.83 against the true 0.90 on all three seeds. The `io-hmm` transitions depend on the
  input, so each heatmap is an average of per-step matrices, not one static matrix.
- **Run C.** On seed 0, `val_loss` reaches its minimum at epoch 12 and then rises while
  the training loss keeps falling. The learned transition diagonal is 0.84 to 0.87 on all
  three seeds.

The transition numbers re-derive from the archives with the trainer's own `evaluate`.
The heatmaps were drawn on GPU 1; this command runs on the CPU and printed a largest
entry error of 0.0043 for run A and 0.0024 for run D, diagonals of 0.744 to 0.830 for
run B and 0.839 to 0.872 for run C over the three seeds, and diagonal
`[0.824, 0.93, 1.0, 0.936]` for run E. From the repository root:

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python - <<'PY'
import argparse, json, keras, numpy as np
import dl_techniques.models.probabilistic.neural_hmm  # registers the classes
from train.neural_hmm.train_neural_hmm import load_splits, evaluate, _relabel_matrix
for run in ("runA_gaussian_k4_sep4_seed0", "runD_autoregressive_gaussian_k4_sep4_seed0",
            "runE_segmenter_gaussian_k4_sep4_seed0", *(f"runB_categorical_k3_v8_seed{s}" for s in range(3)),
            *(f"runC_crf_gaussian_k4_sep2_seed{s}" for s in range(3))):
    d = f"results/neural_hmm_{run}_20261014"
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
import dl_techniques.models.probabilistic.neural_hmm  # registers the classes
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
import dl_techniques.models.probabilistic.neural_hmm  # registers the classes
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
