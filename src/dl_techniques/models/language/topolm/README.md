# TopoLM: a language model whose units are laid out on a sheet of tissue

> Rathi, Mehrer, AlKhamissi, Binhuraib, Blauch & Schrimpf, *TopoLM: Brain-like
> spatio-functional organization in a topographic language model*, ICLR 2025.
> ([arXiv:2410.11516](https://arxiv.org/abs/2410.11516))

---

## 1. Overview

A GPT-2 block indexes its units by nothing but their position on the last axis of
a weight matrix. Two units that learn to encode the same thing have no reason to
sit near each other, so the emergent clustering in a trained model is a property
of the initialisation rather than of anything the objective asked for.

TopoLM gives the units of every attention and feed-forward branch a distinct cell
on an `h x w` grid — with `h * w == embed_dim` — and adds a penalty alongside the
next-token cross-entropy that makes nearby units co-activate. Functional clusters
then emerge as a consequence of the objective rather than as a post-hoc
clustering of whatever the model happened to learn.

```bash
python - <<'PY'
from dl_techniques.models.language.topolm import create_topolm

topographic = create_topolm("paper")                # alpha = 2.5
control     = create_topolm("paper", alpha=0.0)     # identical weights, no loss
ablation    = create_topolm("paper", permute=False) # the paper's Fig. 12
PY
```

**No pretrained weights are distributed.** `pretrained=True` raises
`NotImplementedError` at every entry point, and the message names the alternative.

---

## 2. The objective

For one square neighbourhood of radius `rho` centred on the grid — covering
`P = (2*rho + 1)**2` units — the penalty compares two things that should agree:

- `r`, the Pearson correlation of every unit pair's activations across the `M`
  samples in the batch;
- `d`, the inverse distance `1 / (dist + 1)` of the same pairs, mean-centred and
  unit-normalised.

```text
SL = 0.5 * (1 - corr(r, d))          in [0, 1], 0 == smooth
```

The `0.5` is what puts the range at `[0, 1]`, since `corr` is in `[-1, 1]`.
Minimising `SL` makes near units co-activate and pushes distant units apart in
correlation, an efficient proxy for short wiring length.

Implemented in
[`layers/regularization/spatial_smoothness.py`](../../../layers/regularization/spatial_smoothness.py).

### The whole grid is never scored at once

`r` for all `N` units is `N*(N-1)/2` numbers — 306,936 of them at the paper's
`N = 784` — computed from a `K x M x P` gather every step. The loss instead
samples `K` neighbourhoods per step, which is what makes it affordable, and is
why centres are restricted to those whose patch fits entirely inside the grid: a
constant `P` means `d` is identical for every neighbourhood and can be
precomputed once, at build time, and shared.

At the paper's settings: `P = 121`, `Q = 7260` pairs each, `C = 324` admissible
centres.

### Measured behaviour

784 units on a 28x28 grid, radius 5, 5 sampled neighbourhoods, 256 samples:

| stimulus | `SL` |
|---|---|
| spatially smooth field (lowest Fourier modes) | 0.172 |
| same field, distance prior negated | 0.826 |
| i.i.d. noise | 0.496 |
| wavelength finer than the neighbourhood (`f=9`) | 0.526 |

Two of those rows are load-bearing. The negation row is the **sign-discriminating**
probe: 0.174 and 0.826 are not reachable by loosening a tolerance, so nothing
about `d`'s direction or scale is unconstrained. And the `f=9` row is the
anti-smooth arm — a field whose wavelength is much finer than the neighbourhood
is anti-smooth *at that neighbourhood's scale*. A checkerboard does **not**
demonstrate this: at l-infinity separation 1 it has three offset classes, two of
them negative, so it averages to zero correlation and scores exactly what noise
scores.

---

## 3. Architecture

```text
input_ids [B, S]
     │
     ▼
┌─────────────────────────┐
│ token + learned position│
└─────────────────────────┘
     │ [B, S, D]
     ▼
┌─────────────────────────┐
│ TopoLMBlock x depth     │◄── rank-3 causal mask
│  taps on both branches  │
└─────────────────────────┘
     │
     ▼  last_hidden_state [B, S, D]
┌─────────────────────────┐
│ final LayerNorm         │
└─────────────────────────┘
     │
     ├──► {"last_hidden_state"}
     ▼
tied matmul(token emb.T)  |  lm_head Dense
     │
     ▼  logits [B, S, vocab_size]
```

Both leaves come back together, so
[`CausalLanguageModel`](../../common/masked_language_model/) and the standard CLM
data wrapper work unchanged. No `None` is echoed into the output dict, so
`predict({"input_ids": ...})` does not trip Keras' nested-structure check.

### The block owns its taps, and why

Every architectural piece a TopoLM block needs already exists in
`layers/transformers/`, and none of them can be used here.
`TransformerLayer.call` computes the attention and feed-forward branches and then
adds each straight into the residual stream; in the pre-norm path the branch
tensor is a local overwritten by stochastic depth and layer scale before the
residual add, so it is gone by the time `call` returns. The spatial loss has to be
computed on that tensor, at exactly that point. There is no flag, no hook and no
`activity_regularizer` that reaches it — and `TextDecoder` has no per-block config
to hang a tap on, and only `gpt2/` uses it anyway.

So [`TopoLMBlock`](components.py) is written here, and it **composes** the shared
`create_attention_layer` / `create_ffn_layer` / `create_normalization_layer`
factories rather than reimplementing any of them. The reuse order is unchanged:
factories first, a bespoke layer last. Precedent: `hnet`, `wave_field`,
`tree_transformer`, `qwen3_next`.

The taps sit **on** the branch outputs, before the residual add. Their forward
pass is the identity, so the residual arithmetic is byte-identical to a plain
GPT-2 block and any difference in the loss curve is attributable to the added term
alone.

---

## 4. Two choices that are not the obvious ones

**Each tap permutes its units independently** (`permute=True`, the default).
Without it the residual stream hands every layer the same spatial pattern, the
loss is satisfied by copying one map forward rather than by each layer learning
its own, and the model ends up with one map instead of `depth` of them. That is
the paper's Fig. 12 ablation; `--no-permute` reproduces it.

**`alpha = 0` is a real control, not an absence of one.** The taps are still
created, still built, and still derive their layouts from the same seeds, so the
control has a byte-identical weight set and differs from the trained arm in its
objective alone. `tests/test_models/test_topolm/` pins the identical weight paths
*and* the identical permutations; without that, "identical hyperparameters" would
be a claim rather than a measured fact.

---

## 5. Variants

| variant | embed_dim | depth | heads | max_seq_len | grid | provenance |
|---|---|---|---|---|---|---|
| `paper` | 784 | 12 | 16 | 1024 | 28 x 28 | Rathi et al. 2025, Sec. 3 |
| `small` | 512 | 6 | 8 | 512 | 16 x 32 | repo-authored |
| `tiny` | 256 | 4 | 4 | 256 | 16 x 16 | repo-authored |

Only `paper` is quoted data, and only `paper` is pinned by
`tests/test_variant_tables_match_upstream_references.py`. `small` and `tiny` are
scales of the paper's shape for smoke runs and CI; they exist nowhere in the paper
and are deliberately not pinned, because quoting a derived number as if it were
published is exactly the false citation that guard exists to prevent.

784 is not a free parameter — it is the only plausible hidden size near GPT-2
small that factors into a **square** unit grid (784 = 28 x 28), which is what the
spatial loss needs. `small`'s 16 x 32 grid is deliberately non-square so the
orientation probes have a transposed stride to miss.

Every variant is validated at import (`validate_all_variants`): a width that
cannot host a radius-`r` patch, or that does not divide among the heads, fails on
the import that introduced it rather than on whichever run selects that row first.

---

## 6. Training

The auxiliary term arrives through `add_loss`, not a custom `train_step`, so stock
`fit` still scales the loss under mixed precision and still reports the **pure
task loss** at validation time. A training head therefore needs:

```python
from dl_techniques.models.common.masked_language_model import CausalLanguageModel

head = CausalLanguageModel(
    backbone=create_topolm("paper"),
    vocab_size=100277,
    skip_head=True,
    output_key="logits",
    aggregate_backbone_losses=True,   # load-bearing
)
```

`aggregate_backbone_losses=True` is not a default worth overriding. The taps reach
the objective only through `backbone.losses`; without the flag the head computes
every penalty, discards it, and the run produces a non-topographic model with a
perfectly healthy loss curve. Measured on the same weights at `alpha = 2.5`: a
reported training loss of **15.31** aggregated against **5.36** not, with an
evaluation loss of 5.31 either way.

[`train/topolm/`](../../../../train/topolm/) is the trainer. Two things about it
are worth knowing before reading its code, both of which would otherwise look like
mistakes: the validation cadence *is* the virtual epoch length, and the head must
be told to aggregate the backbone's losses.

---

## 7. Post-hoc analysis

The trainer scores a trained backbone through four steps, and each has a way to
go quietly wrong:

1. a two-sample t-map between two stimulus conditions, on activations
   **mean-pooled** over token positions;
2. BH-FDR corrected **across all taps**, then split back;
3. clusters grown from the surviving cells, both polarities;
4. Moran's I on the **unthresholded** map, plus the islands variant.

```text
activations        contrast_tmap       fdr_across_layers    grow_clusters
(stimuli x units)  -> (t per unit)     -> (q per unit)      -> islands
                     |                  |                    |
                     |                  +- ONE correction over
                     |                     every tap at once
                     +- independent across
                        stimuli, NOT across units
```

Two rules in that list are load-bearing and not interchangeable:

- **Correction is joint across taps.** A twelve-layer model taps 9,408 units;
  correcting each layer's 784 against 5% in isolation admits roughly 470 false
  positives per layer, and the paper's figures read as one organism with one map
  per layer rather than as twelve independent families.
- **Moran's I is scored on the raw map.** Thresholding produces contiguous
  patches of zero, and a patch of zeros is a patch of agreement, so the statistic
  jumps without any change in the underlying organisation.

Two things the metrics do *not* do, deliberately:

- **A simulator is not a result.** `GaussianReadout` exists so a model can be built
  to emit voxel-like signals, and the evaluation runs a raw arm and a readout arm
  side by side. The readout's default is edge-replicating (`padding_mode='nearest'`)
  rather than Keras' zero-padded `'same'`, which lets a constant field decay at
  the grid border.
- **The islands statistic is not a strict improvement.** On a small island it is
  high-variance and sometimes *negative* — a compact blob carries more same-sign
  diagonal pairs than the queen weighting expects (Anselin's queen-contiguity
  instability). It is reported alongside the standard value, never instead of it.

---

## 8. Component reference

| symbol | location | role |
|---|---|---|
| `TopoLM` | `model.py` | the model; `{"logits", "last_hidden_state"}` |
| `TopoLMBlock` | `components.py` | decoder block carrying the taps |
| `create_topolm` | `model.py` | module-level factory |
| `MODEL_VARIANTS` | `config.py` | the variant table |
| `SpatialSmoothness` | `layers/regularization/spatial_smoothness.py` | the identity tap |
| `SpatialLayout` | same | the unit-to-grid bijection |
| `GaussianReadout` | `layers/regularization/gaussian_readout.py` | fixed-width sensor simulation |
| `morans_i_summary` | `metrics/spatial_autocorrelation.py` | standard + islands |
| `contrast_tmap` / `fdr_across_layers` / `grow_clusters` | `metrics/topographic_selectivity.py` | the measurement chain |

### `SpatialSmoothness`

An identity forward pass. All the work happens in the training-only loss, so the
tap is invisible to the residual arithmetic and to validation. Place it on a
tensor whose last axis is the model's unit dimension and where spatial neighbours
*should* respond alike — on the residual-path output of an attention or
feed-forward branch, before normalization and before the add.

Its tables (`perm`, `patch_table`, `triu_flat`, `d_unit`) travel as
**non-trainable weights**, not as configuration. `get_config` carries the seed
that makes a fresh construction reproducible; it does not carry the permutation
itself. That asymmetry is only safe because the permutation is restored
bit-exactly, and a trainable `perm` would let the optimizer silently re-map every
unit while the checkpoint still looked loadable.

Two numerical rules, both measured rather than assumed:

- the correlation block runs at `numpy.promote_types(input_dtype, "float32")` —
  a hard cast to `float32` *narrows* the reduction under a `float64` policy;
- `add_loss` is skipped for a symbolic `KerasTensor`. Keras traces `call` once
  before the first real batch, `add_loss` accepts a `KerasTensor` without
  complaint, and the stored tensor never becomes a value — so an
  otherwise-correct model would report a loss with no spatial term in it and
  nothing would raise.

---

## 9. Reproducing the paper's numbers

```bash
MPLBACKEND=Agg python -m train.topolm.pretrain \
    --variant paper --alpha 2.5 --train-steps 100000
```

The paper trained on FineWeb-Edu and evaluated against an fMRI stimulus set.
**Neither is in this repository.** The trainer's data path is Wikipedia (the
shared `ClmPretrainConfig` default) and its evaluation falls back to a small
built-in smoke stimulus set, with every report stamped
`"stimuli_are_smoke_set": true` and a warning logged. Pass `--variant paper` plus
your own `stimuli=` config for a real analysis.

---

## 10. Tests

```bash
pytest tests/test_models/test_topolm/ tests/test_train/test_topolm/ \
       tests/test_layers/test_regularization/ tests/test_metrics/test_spatial*.py \
       tests/test_metrics/test_topographic*.py tests/test_callbacks/test_topographic_callbacks.py
```

Do not run the full suite as a routine check — it takes about 1.5 hours and is
also the pre-push hook. Scope pytest to the modules you touched.