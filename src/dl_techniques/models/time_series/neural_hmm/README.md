# Neural HMM: Hidden Markov Models with Neural Transitions and Emissions

![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg) ![Python 3.11](https://img.shields.io/badge/Python-3.11+-blue.svg) ![TF 2.18](https://img.shields.io/badge/TF-2.18-orange.svg)

A Keras 3 **Neural Hidden Markov Model**: a `K`-state HMM whose initial distribution, transition matrices and emission densities are produced by neural layers, trained by exact maximum likelihood through the forward algorithm, and decoded exactly by Viterbi and forward-backward.

> **Identity, up front.** `NeuralHMM` scores and decodes sequences; it is not a forecaster and does not implement the `time_series` family's `ForecastMixin`. Its forward pass returns the HMM's three log-potentials as a dict, `{"log_pi": (B, K), "log_A": (B, T, K, K), "log_emit": (B, T, K)}`; the loss, the likelihood, the most likely state path and the state posteriors are all computed from that dict by one exact inference module. No pretrained weights are distributed: asking `from_variant` / `create_neural_hmm` for pretrained weights raises `NotImplementedError`.

---

## 1. Overview

`NeuralHMM` is a `keras.Model` over right-padded sequences, `{"obs": ..., "mask": ...}`. Four knobs decide what it is:

| Knob | Values | What it changes |
|------|--------|-----------------|
| `mode` | `generative` (default), `discriminative` | unsupervised `log p(x)`, or a linear-chain CRF `log p(y \| x)` trained on state labels |
| `transition` | `static` (default), `input_dependent` | one learned matrix, or per-step matrices driven by the encoder (an input-output HMM) |
| `emission_context` | `none` (default), `autoregressive` | the generative emission reads only the state, or also the encoder's summary of `x_{<t}` |
| `obs_type` | `gaussian` (default), `poisson`, `categorical` | diagonal Gaussian over `obs_dim` reals, Poisson over `obs_dim` counts, or one symbol in `[0, vocab_size)` |

The sequence encoder (`mlp`, `transformer`, `lstm`, `gru` or `conv`) is built **only when something reads its output**: an autoregressive emission, input-dependent transitions, or the discriminative mode. The default configuration has no encoder at all.

---

## 2. The Problem It Solves

A classical HMM has exact, cheap inference (likelihood, decoding, posteriors in `O(T K^2)`), but its emission and transition distributions are tables or fixed parametric families. A neural sequence model is flexible but gives no discrete latent state and no exact posterior over it. A neural HMM keeps the HMM's exact inference and lets neural layers parametrize the distributions:

```
┌─────────────────────────────────────────────────────────────┐
│  1. Discrete latent states with exact inference: the         │
│     likelihood, the Viterbi path and the posteriors are      │
│     computed exactly, never sampled or approximated.         │
│                                                              │
│  2. Neural parametrization: emissions come from learned      │
│     state embeddings through a feed-forward head (Tran et    │
│     al., 2016), optionally conditioned on the history;       │
│     transitions can depend on the input (Bengio and          │
│     Frasconi, 1995).                                         │
│                                                              │
│  3. One model, two regimes: unsupervised (segment and        │
│     cluster sequences) or supervised (a CRF tagger, Lample   │
│     et al., 2016).                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. How It Works: Core Concepts

**Log-potentials.** The model emits three tensors (shapes with batch `B`, time `T`, states `K`):

- `log_pi[b, k]`: log-probability of starting in state `k`;
- `log_A[b, t, i, j]`: log-probability of moving from state `i` at step `t - 1` INTO state `j` at step `t`. Rows are the source, columns the destination, and every row of `exp(log_A)` sums to 1. The `t = 0` slot is never read;
- `log_emit[b, t, k]`: `log p(x_t | z_t = k)` in generative mode (a normalized density), or an unnormalized CRF potential in discriminative mode.

**Forward algorithm** (Rabiner, 1989), in log space:
`alpha_0 = log_pi + log_emit[0]`, `alpha_t[j] = logsumexp_i(alpha_{t-1}[i] + log_A[t, i, j]) + log_emit[t, j]`, `log Z = logsumexp(alpha_last)`. In generative mode `log Z = log p(x)`.

**Padding.** The mask marks a contiguous valid PREFIX (right padding). A padded step contributes no emission and no transition: the recursion freezes. So the log-likelihood of a padded sequence equals that of the truncated sequence, Viterbi paths hold `-1` at padded steps, and posteriors are 0 there. A sequence with no valid step has `log Z = 0`.

**Causality.** In generative mode the encoder is causal and reads the input shifted right by one step, so `h_t` summarizes `x_{<t}` only and `p(x_t | z_t, x_{<t})` stays a proper factorization of `p(x)`. Categorical symbols are shifted after the embedding, so symbol 0 stays a real symbol.

**Arbitrary sequence length.** Every recursion runs on `keras.ops.while_loop`, so `predict` and `evaluate` work at any sequence length in one process (Keras retraces them with an unknown time axis from the second length on), and the loss differentiates under XLA.

---

## 4. Architecture Deep Dive

```
Input:  {"obs": (B, T, obs_dim) | (B, T) int,  "mask": (B, T)}
                     │
     padded steps neutralized (floats -> 0, symbols -> -1)
                     │
     ┌───────────────┴───────────────── only when an encoder is built ─┐
     │  categorical: Embedding(vocab_size, hidden_dim)                  │
     │  generative:  shift right one step (step t sees x_{<t})          │
     │  NeuralHMMEncoder: mlp | transformer | lstm | gru | conv          │
     │     causal in generative mode, non-causal in discriminative       │
     └───────────────┬──────────────────────────────────────────────────┘
                     │  h (B, T, hidden_dim)
          ┌──────────┴───────────┐
          ▼                      ▼
   transition                 emission
   StaticTransition           GenerativeEmission: state embedding e_k
   (reads B and T only)         [+ h_t if autoregressive] -> FFN head
   or InputDependentTransition  -> Gaussian / Poisson / categorical log-density
   (static logits + W h_t)    or DiscriminativeEmission: Dense(K)(h_t)
          │                      │
          ▼                      ▼
   log_pi (B, K)              log_emit (B, T, K)
   log_A  (B, T, K, K)
          └──────────┬───────────┘
                     ▼
   inference.py: forward_log_likelihood, path_log_score,
                 viterbi_decode, state_posteriors
```

All three outputs are float32 under both `float32` and `mixed_float16` (float64 under a float64 policy). Under mixed precision the encoder and the emission head run in float16 and the log-space math does not: a float16 Gaussian log-density overflows once `|x - mu| / sd` exceeds about 362, and the forward recursion accumulates.

Topology masks are applied by selection (`keras.ops.where` with a finite sentinel), never by adding `-1e9`, so a forbidden transition has probability exactly 0 under every policy.

---

## 5. Modes (Generative vs Discriminative)

| Mode | `y` passed to `fit` | Loss per valid step | Encoder | Emission |
|------|---------------------|---------------------|---------|----------|
| `generative` | `None` (labels raise `ValueError`) | `-log p(x)` | causal, optional | normalized density `log p(x_t \| z_t)` |
| `discriminative` | `(B, T)` int states; padded values ignored | `log Z - score(y)` = `-log p(y \| x)` | non-causal, always built | `Dense(K)` potentials |

Both are trained by stock `fit()`: the loss is `NeuralHMM.compute_loss`, so `compile(loss=...)` is ignored. The reduction is `sum_b w_b nll_b / sum_b w_b len_b`, where `w` is the optional per-sequence `sample_weight`; a batch of empty sequences has loss 0. `discriminative` refuses `emission_context="autoregressive"`.

---

## 6. Quick Start

```python
import numpy as np
from dl_techniques.models.time_series.neural_hmm import create_neural_hmm

rng = np.random.default_rng(0)
obs = rng.normal(size=(8, 20, 2)).astype("float32")
mask = np.ones((8, 20), dtype="float32")
mask[4:, 15:] = 0.0  # the last four sequences have 15 valid steps
inputs = {"obs": obs, "mask": mask}

model = create_neural_hmm("neural-hmm", n_states=3, obs_type="gaussian", obs_dim=2)
model.compile(optimizer="adam")
model.fit(inputs, None, batch_size=4, epochs=1, verbose=0)

path, score = model.viterbi(inputs)    # (8, 20) int32 with -1 at padding, (8,)
log_px = model.log_likelihood(inputs)  # (8,) log p(x)
gamma = model.posteriors(inputs)       # (8, 20, 3); rows sum to 1 at valid steps
```

---

## 7. Component Reference

### `NeuralHMM` (`model.py`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_states` | `int` | required | Number of hidden states `K`; positive. |
| `mode` | `str` | `"generative"` | `generative` or `discriminative` (§5). |
| `obs_type` | `str` | `"gaussian"` | `gaussian`, `poisson` or `categorical`. |
| `obs_dim` | `Optional[int]` | `None` | Features of a `gaussian`/`poisson` observation; required for those, must be `None` for `categorical`. |
| `vocab_size` | `Optional[int]` | `None` | Symbols of a `categorical` observation; required for it, must be `None` otherwise. |
| `emission_context` | `str` | `"none"` | `autoregressive` makes the generative emission also read `h_t`. |
| `transition` | `str` | `"static"` | `input_dependent` adds a per-step offset `W h_t` (zero-initialized). |
| `topology` | `str` | `"ergodic"` | `ergodic`, `left_to_right` or `banded`. |
| `band_width` | `int` | `1` | Largest forward skip under `banded`. |
| `self_loop_init` | `float` | `2.0` | Initial diagonal of the transition logits (sticky start). |
| `min_scale` | `float` | `1e-3` | Floor of the Gaussian scale and the Poisson rate. |
| `hidden_dim` | `int` | `64` | Encoder, symbol-embedding and emission-head width. |
| `state_emb_dim` | `int` | `16` | Width of the learned state embeddings. |
| `encoder_type` | `str` | `"lstm"` | `mlp`, `transformer`, `lstm`, `gru` or `conv`; read only when an encoder is built. |
| `num_layers` | `int` | `2` | Encoder blocks or stacked recurrent layers. |
| `num_heads` | `int` | `4` | Attention heads (`transformer`); must divide `hidden_dim`. |
| `intermediate_size` | `int` | `128` | Feed-forward width inside the `mlp`/`transformer` encoder. |
| `max_seq_len` | `int` | `512` | Rows of the `transformer` encoder's learned position table. |
| `kernel_size` | `int` | `3` | Convolution width (`conv`). |
| `ffn_type` | `str` | `"mlp"` | Feed-forward type of the `transformer` encoder's blocks (`create_ffn_layer`); read only when that encoder is built. The emission head is always an `mlp` block. |
| `normalization_type` | `str` | `"layer_norm"` | Normalization inside the encoder (`create_normalization_layer`). |
| `dropout_rate` | `float` | `0.0` | Encoder dropout, `[0, 1)`; active only with `training=True`. |
| `attention_dropout_rate` | `float` | `0.0` | Attention-probability dropout (`transformer`), `[0, 1)`. |

Methods: `call(inputs, training=None) -> dict`; `compute_loss(x, y, y_pred, sample_weight, training)`; `log_likelihood(inputs, states=None) -> (B,)` (`log p(x)`, or `log p(states | x)` in discriminative mode); `viterbi(inputs) -> (path, score)`; `posteriors(inputs) -> (B, T, K)`; `compute_output_shape` (valid before `build`); `from_variant(variant, pretrained=False, **kwargs)`. Module-level `create_neural_hmm(variant="neural-hmm", pretrained=False, **kwargs)` delegates to `from_variant`.

### `NeuralHMMEncoder` (`encoders.py`)

`(B, T, F) -> (B, T, hidden_dim)`, with a `(B, T)` mask. Padded inputs are replaced by zeros before anything reads them, and the output passes through a final `create_normalization_layer`. Paths: `mlp` (residual FFN blocks, per step), `transformer` (learned positions + pre-norm `TransformerLayer`s, causal through `create_causal_attend_mask`), `lstm`/`gru` (stock Keras layers, `Bidirectional(merge_mode="sum")` when non-causal), `conv` (`Conv1D` with `causal` or `same` padding, padded steps re-zeroed before every convolution).

### Transitions (`transitions.py`)

`StaticTransition(n_states, topology="ergodic", band_width=1, self_loop_init=2.0)` returns `(log_pi (B, K), log_A (B, T, K, K))` broadcast from one learned matrix. `InputDependentTransition` (same arguments) adds a zero-initialized, bias-free `Dense(K * K)` of a `(B, T, H)` context. `topology_masks(n_states, topology, band_width)` returns the allowed start states and transitions.

### Emissions (`emissions.py`)

`GenerativeEmission(n_states, obs_type, obs_dim=None, vocab_size=None, state_emb_dim=16, hidden_dim=64, context_dim=0, min_scale=1e-3)`: a fixed `create_ffn_layer("mlp")` head (`hidden_dim` wide) maps each state embedding, with `h_t` appended when `context_dim > 0`, to the distribution's parameters. Gaussian (`sd = softplus(raw) + min_scale`), Poisson (`rate = softplus(raw) + min_scale`, including `-log x!`) or categorical (`log_softmax`; any symbol outside `[0, V)` scores 0). `DiscriminativeEmission(n_states)`: one `Dense(K)`.

### Inference functions (`inference.py`)

All take `log_pi (B, K)`, `log_A (B, T, K, K)`, `log_emit (B, T, K)` and `mask (B, T)`, and compute in float32 (float64 for float64 input):

| Function | Returns |
|----------|---------|
| `forward_log_likelihood(log_pi, log_A, log_emit, mask)` | `(B,)` `log Z` |
| `path_log_score(log_pi, log_A, log_emit, states, mask)` | `(B,)` score of a given path; padded state values are ignored |
| `viterbi_decode(log_pi, log_A, log_emit, mask)` | `(path (B, T) int32, score (B,))`, `-1` at padding |
| `state_posteriors(log_pi, log_A, log_emit, mask)` | `(B, T, K)`, 0 at padding |

The package `__init__.py` exports `NeuralHMM` and `create_neural_hmm`; the family re-exports both, so `from dl_techniques.models.time_series import NeuralHMM, create_neural_hmm` also works.

---

## 8. Configuration & Presets

`NeuralHMM.MODEL_VARIANTS` holds the structural choices only; `n_states` and the observation shape (`obs_type` with `obs_dim` or `vocab_size`) come from the caller, and any argument overrides the preset. Every component a preset builds is read by the forward pass, so every built weight receives a gradient.

| variant | mode | transition | emission | encoder | source |
|:---|:---|:---|:---|:---|:---|
| `neural-hmm` | generative | static, ergodic | state embedding -> FFN head | none | Tran et al., 2016 |
| `autoregressive-neural-hmm` | generative | static, ergodic | head also reads `h_t` from `x_{<t}` | causal `lstm` | Tran et al., 2016, conditioned on history |
| `io-hmm` | generative | input-dependent, ergodic | state embedding -> FFN head | causal `conv` | Bengio and Frasconi, 1995 |
| `left-to-right-segmenter` | generative | static, `banded` (`band_width=1`) | head also reads `h_t` | causal `transformer` | Rabiner, 1989 (left-to-right models) |
| `neural-crf-tagger` | discriminative | static, ergodic | `Dense(K)` potentials from `h` | non-causal `transformer` | Lample et al., 2016 |

`neural-crf-tagger` uses a non-causal transformer; the reference encoder is a BiLSTM, which `encoder_type="lstm"` gives in discriminative mode.

---

## 9. Comprehensive Usage Examples

### Example 1 — A CRF tagger trained on state labels

```python
states = rng.integers(0, 3, size=(8, 20)).astype("int32")

tagger = create_neural_hmm(
    "neural-crf-tagger", n_states=3, obs_type="gaussian", obs_dim=2,
    hidden_dim=16, num_heads=2, intermediate_size=32, num_layers=1,
)
tagger.compile(optimizer="adam")
tagger.fit(inputs, states, batch_size=4, epochs=1, verbose=0)
log_p_y = tagger.log_likelihood(inputs, states)  # (8,) log p(y | x)
tagged, _ = tagger.viterbi(inputs)               # (8, 20)
```

### Example 2 — Categorical symbols with input-dependent transitions

```python
symbols = rng.integers(0, 6, size=(8, 20)).astype("int32")
symbols[mask == 0] = -1  # padding may hold anything; -1 is conventional

io_hmm = create_neural_hmm(
    "io-hmm", n_states=3, obs_type="categorical", vocab_size=6,
    hidden_dim=16, num_layers=1,
)
io_hmm.compile(optimizer="adam")
io_hmm.fit({"obs": symbols, "mask": mask}, None, batch_size=4, epochs=1, verbose=0)
per_step_nll = -io_hmm.log_likelihood({"obs": symbols, "mask": mask}) / mask.sum(axis=1)
```

### Example 3 — A left-to-right segmenter on Poisson counts

```python
counts = rng.poisson(3.0, size=(8, 20, 2)).astype("float32")

segmenter = create_neural_hmm(
    "left-to-right-segmenter", n_states=4, obs_type="poisson", obs_dim=2,
    hidden_dim=16, num_heads=2, intermediate_size=32, num_layers=1, max_seq_len=64,
)
segments, _ = segmenter.viterbi({"obs": counts, "mask": mask})  # never decreases along valid steps
```

### Example 4 — Per-sequence weights, and the inference module directly

```python
from dl_techniques.models.time_series.neural_hmm import inference

weights = np.array([1, 1, 1, 1, 2, 2, 2, 2], dtype="float32")
model.fit(inputs, None, sample_weight=weights, batch_size=4, epochs=1, verbose=0)

out = model(inputs, training=False)
log_z = inference.forward_log_likelihood(out["log_pi"], out["log_A"], out["log_emit"], mask)
```

---

## 10. Training & Best Practices

The trainer lives in [`src/train/neural_hmm/`](../../../../train/neural_hmm/): it samples sequences from a known ground-truth HMM, trains with stock `fit()`, and scores the learned model against the true model. Run it with `MPLBACKEND=Agg .venv/bin/python -m train.neural_hmm.train_neural_hmm --help` for its flags; measured results are recorded in that directory's README, not here.

- **Pad on the right.** The mask must be a contiguous valid prefix; interior gaps are not supported and are not checked (§13).
- **Label switching.** Unsupervised states are identifiable only up to a permutation. Compare decoded paths with ground truth after matching states (Hungarian matching on the confusion matrix).
- **Sticky starts.** `self_loop_init` (default `2.0`) makes states sticky at initialization, which keeps early decoding from flickering. Lower it for data whose states switch often.
- **`min_scale`.** The Gaussian scale and Poisson rate never drop below it. A tiny floor lets a state collapse onto a single point (a likelihood spike); raise it if the loss diverges toward minus infinity on real-valued data.
- **XLA.** On a GPU, Keras compiles with XLA by default, except for a model holding an `lstm`/`gru` encoder: Keras' cuDNN recurrent layers report `supports_jit=False`, and an explicit `jit_compile=True` warns ("Model doesn't support `jit_compile=True`") and trains without XLA. Assert `model.jit_compile` after `compile` when the regime matters.

---

## 11. Serialization & Deployment

```python
import os
import tempfile

import keras

with tempfile.TemporaryDirectory() as directory:
    archive = os.path.join(directory, "neural_hmm.keras")
    model.save(archive)
    restored = keras.models.load_model(archive)  # no custom_objects
    assert np.array_equal(
        keras.ops.convert_to_numpy(restored.log_likelihood(inputs)),
        keras.ops.convert_to_numpy(model.log_likelihood(inputs)),
    )
```

Every class registers through `register_dl_technique` under `dl_techniques.models.neural_hmm.<module>`: `model>NeuralHMM`, `encoders>NeuralHMMEncoder`, `transitions>StaticTransition`, `transitions>InputDependentTransition`, `emissions>GenerativeEmission`, `emissions>DiscriminativeEmission`. `NeuralHMM.build` takes the dict of input shapes and builds every sublayer, so `load_model` restores every weight before the first call. `get_config` returns every constructor argument.

---

## 12. Interpretability

The HMM's parameters are readable directly from the forward pass. With static transitions every time step carries the same matrix:

```python
out = model(inputs, training=False)
initial = np.exp(keras.ops.convert_to_numpy(out["log_pi"])[0])        # (K,) start distribution
transition = np.exp(keras.ops.convert_to_numpy(out["log_A"])[0, 1])   # (K, K), rows sum to 1
dwell = 1.0 / (1.0 - np.diag(transition))                              # expected steps per visit
```

- `viterbi(inputs)` gives the single most likely state path (a segmentation of each sequence).
- `posteriors(inputs)` gives `p(z_t = k | x)` per step: the decoder's confidence, and soft boundaries between segments.
- With `transition="input_dependent"`, `log_A[:, t]` varies per step: plotting `exp(log_A[b, t, i, i])` over `t` shows where the model expects to leave state `i`.
- With `emission_context="none"`, the emission parameters of state `k` are a fixed function of its embedding, so the per-state Gaussian means or symbol distributions can be read off by evaluating `log_emit` on a grid of observations.

---

## 13. Limitations, Troubleshooting & FAQs

### Limitations

- **L-1. Right padding only.** A padded step freezes the recursion; a mask with a hole in the middle is treated as a different sequence. This is documented, not checked: XLA drops runtime assertions.
- **L-2. `max_seq_len` bounds the `transformer` encoder.** Its learned position table has `max_seq_len` rows. A statically known `T > max_seq_len` is refused at build time; a longer dynamic `T` fails inside the table slice. The other encoder types have no length limit.
- **L-3. Two cuDNN traps are worked around in the recurrent encoders** (Keras 3.8, visible on a GPU only). (a) `LSTM/GRU(dropout=...)` applies its input-dropout mask on the cuDNN path whatever `training` is, so the encoder never passes `dropout=` to them and runs a separate `Dropout` (one mask per sequence) before each recurrent layer. (b) cuDNN refuses a mask row with no valid step in graph mode (`fit`, `predict`, `tf.function`), so an all-padding row gets step 0 marked valid; that step reads zeroed input and every output of the row is padding.
- **L-4. Constant parameters under constrained topologies.** Under `banded` or `left_to_right` the chain starts in state 0, so the initial-state logits are constant and never train. With `n_states=1`, `log_A` is identically 0.
- **L-5. `io-hmm` starts at the static solution.** Its context projection is zero-initialized, so the encoder receives its first nonzero gradient on the second optimizer step.
- **L-6. Mixed precision rounds the inputs.** Keras casts a model's float inputs to the compute dtype, so under `mixed_float16` the observations are float16 before the float32 density math; values above 65504 overflow.
- **L-7. The key part of a fused attention bias does not train.** The `transformer` encoder's attention projects `qkv` with one biased `Dense`; the key bias adds a per-query constant that the softmax removes, so that part receives no gradient, while the fused variable as a whole does.

### Troubleshooting / FAQs

**Q. `ValueError: mode='generative' is trained without labels`.** Pass `y=None` to `fit` in generative mode; labels are for `mode="discriminative"`.

**Q. `compile(loss=...)` seems to have no effect.** It has none: the objective is `compute_loss` (§5).

**Q. Padded steps hold NaN.** That is allowed: padded content is replaced before any layer reads it, so neither the loss nor the gradients change.

**Q. A warning about a softmax over an axis of size 1.** A `transformer` encoder called on a sequence of length 1 attends over one key; Keras warns, and the result (weight 1) is correct.

**Q. My decoded states do not match the true labels.** In generative mode the state indices are arbitrary; match them first (§10).

**Q. Training on a GPU is slower with `lstm`/`gru`.** Keras disables XLA for cuDNN recurrent layers (§10); the `conv`, `mlp` and `transformer` encoders compile.

---

## 14. References

- Tran, K., Bisk, Y., Vaswani, A., Marcu, D. and Knight, K. (2016) — *Unsupervised Neural Hidden Markov Models*. Proceedings of the Workshop on Structured Prediction for NLP. arXiv:1609.09007.
- Bengio, Y. and Frasconi, P. (1995) — *An Input Output HMM Architecture*. Advances in Neural Information Processing Systems 7, pp. 427-434.
- Rabiner, L. R. (1989) — *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition*. Proceedings of the IEEE 77(2), pp. 257-286.
- Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K. and Dyer, C. (2016) — *Neural Architectures for Named Entity Recognition*. NAACL-HLT. arXiv:1603.01360.

**Related code**

- Model: `dl_techniques/models/time_series/neural_hmm/model.py`
- Layers: `encoders.py`, `transitions.py`, `emissions.py`; inference: `inference.py`
- Trainer: `train/neural_hmm/`
- Tests: `tests/test_models/test_neural_hmm/`; this README's code blocks run in `tests/test_models/test_readme_examples.py`
