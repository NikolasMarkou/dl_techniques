# TabM — parameter-efficient batched MLP ensembles for tabular data

This file was **0 bytes** until 2026-08-18. `model.py`'s module docstring is the
long-form reference and remains the authority on the mechanism; this README is the
orientation layer.

## What it is

A deep ensemble of `k` MLPs that share almost all of their weights. One kernel `W`
is used by every member; member `i` scales the layer input by a learned vector `r_i`
and the output by `s_i`, so its effective weight is `diag(r_i) W diag(s_i)` — a
genuinely different linear map per member, bought with `d_in + d_out` parameters
instead of `d_in * d_out`.

Reference: Gorishniy et al., 2024, *TabM: Advancing Tabular Deep Learning with
Parameter-Efficient Ensembling* (https://arxiv.org/abs/2410.24210).

## The member axis

Tensors carry an explicit member axis: `(batch, k, features)`. `call` returns
per-member predictions `(batch, k, d_out)`, **un-aggregated**, and the `'plain'`
path expands to `(batch, 1, d_out)` so output rank does not depend on architecture.
Aggregation is explicit and separate:

- `model.predict_with_uncertainty(x)` — mean plus across-member standard deviation.
- `ensemble_predict(model, x)` — the mean alone.

The spread is a *disagreement* statistic over whatever `call` emits (logits, for a
classifier), not a calibrated predictive variance.

## Architecture types (`arch_type`)

| value | backbone | head | diversity from |
|:---|:---|:---|:---|
| `plain` | ordinary MLP (`k` must be `None`) | `Dense` | none — this is the baseline |
| `tabm` | `LinearEfficientEnsemble` per layer | `NLinear` (k independent matrices) | rank-1 scaling, random-sign init |
| `tabm-normal` | as `tabm` | `NLinear` | rank-1 scaling, `N(1, 0.1)` init |
| `tabm-packed` | `k` independent kernels per layer | `NLinear` | full deep ensemble — the honest upper bound, `k`x backbone parameters |
| `tabm-mini` | shared, unperturbed | `NLinear` | a single input-side `ScaleEnsemble` adapter |
| `tabm-mini-normal` | as `tabm-mini` | `NLinear` | same adapter, normal init |

## Variants

Measured 2026-08-18 at `n_num_features=10, cat_cardinalities=[3, 5], n_classes=2`:

| variant | `hidden_dims` | `k` | `arch_type` | params | `call` output |
|:---|:---|:---:|:---|---:|:---|
| `micro` | `[64, 32]` | 4 | `tabm-mini` | 3,920 | `(B, 4, 2)` |
| `tiny` | `[128, 64]` | 8 | `tabm-mini` | 13,216 | `(B, 8, 2)` |
| `small` | `[256, 128]` | 8 | `tabm` | 47,776 | `(B, 8, 2)` |
| `base` | `[512, 256, 128]` | 8 | `tabm` | 195,744 | `(B, 8, 2)` |
| `large` | `[1024, 512, 256]` | 16 | `tabm` | 764,224 | `(B, 16, 2)` |
| `xlarge` | `[2048, 1024, 512, 256]` | 32 | `tabm` | 3,166,848 | `(B, 32, 2)` |

Parameter counts depend on the input width (one-hot encoding is inside the model),
so re-derive rather than quoting these for another dataset:

```python
TabMModel.from_variant(variant, n_num_features=..., cat_cardinalities=[...],
                       n_classes=...).count_params()
```

The `description` strings inside `MODEL_VARIANTS` carry round parameter figures of
their own; treat them as indicative of scale, not as counts for your feature width.

## Usage

```python
import numpy as np
from dl_techniques.models.tabm import TabMModel, ensemble_predict

model = TabMModel.from_variant(
    "small", n_num_features=10, cat_cardinalities=[3, 5], n_classes=2
)
batch = {
    "x_num": np.random.rand(4, 10).astype("float32"),
    "x_cat": np.random.randint(0, 3, (4, 2)).astype("int32"),
}
per_member = model(batch, training=False)   # (4, 8, 2)
mean = ensemble_predict(model, batch)       # (4, 2)
```

`create_tabm_model` / `create_tabm_plain` / `create_tabm_ensemble` /
`create_tabm_mini` / `create_tabm_for_dataset` are the non-variant factories; they
take the same feature description plus an explicit `hidden_dims`.

## Two things that will bite you

1.  **Preprocessing is inside the model.** Numerical features pass through unchanged;
    categorical features are one-hot encoded against the declared cardinalities. The
    effective input width therefore grows with the sum of cardinalities — a
    high-cardinality column is the case where an external embedding is preferable.
2.  **`share_training_batches=False` requires you to prepare the batch.** In that
    mode an incoming batch of `B * k` rows is *reshaped* so each member gets a
    disjoint slice. The model does not resample for you, and a batch size that is
    not a multiple of `k` will not reshape correctly. When `True` (the default, and
    always at inference) one batch is tiled `k` ways.

## Loss

`dl_techniques.losses.tabm_loss.TabMLoss`, re-exported from this package, is the
member-axis-aware loss. A stock Keras loss applied to `call`'s `(B, k, d_out)` output
will not reduce over the member axis the way you expect.
