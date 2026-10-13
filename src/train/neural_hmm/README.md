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
| `--obs-dim` | 2 | the data and the model, gaussian only |
| `--vocab-size` | 8 | the data and the model, categorical only; must be `>= --n-states` |
| `--encoder` | the preset's | the model (`mlp`, `transformer`, `lstm`, `gru`, `conv`); refused for a preset that builds no encoder (`neural-hmm`) |
| `--hidden-dim` | 32 | the model (encoder, observation embedding, emission head); divisible by 4 for a transformer encoder |
| `--seq-len` | 50 | the data (padded length) and the model (transformer position table) |
| `--min-seq-len` | 25 | the data (shortest valid length) |
| `--num-train`, `--num-val`, `--num-test` | 2000, 200, 500 | the data |
| `--separation` | 4.0 | the data, gaussian only |
| `--stay-prob` | 0.9 | the data; in `(0, 1)` |
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
a combination the model refuses (or an ignored `--encoder`) with a `ValueError` before
the run directory exists. A reused `--experiment-name` is refused the same way, and an
existing run is never overwritten.

## 5. Run directory

| File | Content |
|---|---|
| `config.json` | the parsed arguments |
| `run.log` | the run's log, one line per epoch |
| `training_log.csv` | `epoch`, `loss`, `val_loss` per epoch (`CSVLogger`) |
| `training_history.json` | the `fit` history |
| `best_model.keras` | the epoch with the lowest `val_loss`; loads with `keras.models.load_model` and no `custom_objects` once `dl_techniques.models.time_series.neural_hmm` is imported (the import registers the classes) |
| `results_summary.json` | strict JSON (a non-finite number is written as `null`): the keys of §3, the resolved arguments, `params`, `jit_compile` (the resolved value after `compile`), `epochs_run`, `best_checkpoint` (`file`, `epoch`, `val_loss`, `reload_max_abs_diff`, `reload_exact`), `fit_seconds`, `total_seconds` and `visualizations` (`files`, `failed`) |
| `visualizations/loss_curve.png` | train and validation loss per epoch, with the true HMM's NLL per step on the test split as a dashed line |
| `visualizations/transitions.png` | the learned transition matrix (mean of `exp(log_A)` over the valid test steps; for a static transition, the matrix itself) beside the true one, both in true-label order |
| `visualizations/state_strip.png` | true and decoded states of the first 4 test sequences, in true-label order; padding blank |

A figure that fails to render is logged, listed under `visualizations.failed`, and does
not stop the run.

## 6. XLA

On a GPU, Keras compiles `fit` with XLA by default (`jit_compile="auto"`), except for a
model holding an `lstm`/`gru` encoder (the model package README, §10). The summary's
`jit_compile` records what the run used. A tiny `neural-hmm` run on GPU 1 resolved
`jit_compile` to `true`.

## 7. Measured runs

None yet.

## 8. Tests

`tests/test_train/test_neural_hmm/`:

- `test_data.py`: the sampler's layout and statistics, and the references against a
  brute-force enumeration of every state path and against the model package's
  `inference.forward_log_likelihood`;
- `test_cli_contract.py`: one contract row per flag, the `--gpu` sentinel, and two real
  runs recording what the sampler, the model factory, the optimizer and `fit` received;
- `test_train_neural_hmm.py`: real tiny runs of `main()` for gaussian `neural-hmm`,
  categorical `io-hmm` and gaussian `neural-crf-tagger`, every artifact, the archive
  reload, the refusals and a failing figure.

```bash
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_train/test_neural_hmm/ -q
```
