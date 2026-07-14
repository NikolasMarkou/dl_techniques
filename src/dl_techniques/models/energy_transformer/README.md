# Energy Transformer — Image Models

Keras 3 implementation of the image models from **Hoover et al., "Energy Transformer", NeurIPS 2023** ([arXiv:2302.07253](https://arxiv.org/abs/2302.07253)): a masked-image-completion model (the paper's §3 model) and a classifier that shares the same trunk.

The Energy Transformer replaces the usual stack of transformer blocks with **one** block that defines a single scalar energy `E(g)` over the token states and then runs `T` steps of gradient descent on it. Attention and the Hopfield feed-forward are not two sequential sub-layers; they are two additive terms of that one energy. The forward pass *is* the optimization.

These two models exist to give the `EnergyTransformer` block (`layers/transformers/energy_transformer.py`) real consumers, and to demonstrate that a trunk pretrained by masked completion transfers into a classifier.

---

## Contents

1. [Architecture](#1-architecture)
2. [The data contract](#2-the-data-contract-read-this-before-writing-a-trainer)
3. [The 90/10 rule](#3-the-9010-rule)
4. [No custom `train_step`](#4-no-custom-train_step)
5. [Variants and measured cost](#5-variants-and-measured-cost)
6. [Hyperparameters](#6-hyperparameters-paper-table-4)
7. [Usage](#7-usage)
8. [Known limitations](#8-known-limitations)
9. [Tests](#9-tests)
10. [Citation](#10-citation)

---

## 1. Architecture

One shared trunk, two heads.

```
image (B, H, W, C)              input_mask (B, N) bool   [MIM only]
      |                                |
PatchEmbedding2D  ---------------------+
      |                                |
      v                                v
(B, N, D)  ----------------->  MaskTokenApply   (skipped when no mask is passed,
      |                                |         but ALWAYS created AND built)
      +--------------------------------+
                      |
              PositionalEmbedding (learned)
                      |
              EnergyTransformer  (ONE block, T=12 internal descent steps
                      |           on ONE scalar energy)
                (B, N, D) tokens
                /              \
    decoder_norm                head_norm
    decoder_proj                head_pool (mean over tokens)
        |                       head_dense
        v                           |
(B, N, P*P*C)                       v
   patch pixels             (B, num_classes) logits
```

| Class | Role |
|---|---|
| `EnergyTransformerBackbone` | patch-embed -> (optional learnable MASK token) -> learnable positional embedding -> ONE `EnergyTransformer` block unrolling `T=12` descent steps -> `(B, N, D)` |
| `EnergyTransformerMIM` | backbone -> `LayerNorm` -> **affine** `Dense(P*P*C)` -> `(B, N, P*P*C)` raw (normalized) patch pixels |
| `EnergyTransformerClassifier` | the SAME backbone -> `LayerNorm` -> mean-pool over tokens -> `Dense(num_classes)` **logits** |

The MIM decoder is a single affine projection on purpose: the reconstruction quality is meant to come from the energy descent in the trunk, not from a deep decoder.

### Why the two trunks are weight-identical

`MaskTokenApply` is created **and built** by every backbone, including the classifier's, which never calls it. That is the authoring guide's §9 "ALWAYS CREATE / CONDITIONALLY USE" rule, and it is what keeps the MIM trunk and the classifier trunk weight-identical so that `load_weights_from_checkpoint(..., skip_prefixes=("decoder_", "head_"))` transfers the trunk 1:1. Removing the "dead" mask token from the classifier would silently break the warm-start.

### Why mean-pool and not a CLS token

The classifier pools with a mean over tokens, not a CLS token.

A CLS token makes `N = 197` in the classifier versus `N = 196` in the MIM model. That changes the shape of the learnable positional-embedding table and **breaks the trunk transfer outright** — i.e. it destroys the exact capability the classifier exists to demonstrate. On top of that, the ET block has no CLS concept at all: nothing in `EnergyTransformer` / `EnergyAttention` / `HopfieldNetwork` special-cases a distinguished token, and `attn_self=False` (the paper's image default) masks the diagonal, so a CLS token could not even attend to itself. The paper uses a CLS token only in the *graph* variant.

Mean-pool costs zero parameters and transfers cleanly.

---

## 2. The data contract (read this before writing a trainer)

**This is the load-bearing part.** The masked-completion objective needs the occlusion mask to reach the **loss**, not just the model. It does so as a Keras `sample_weight`, supplied as the third element of each `tf.data` batch by `dl_techniques.datasets.vision.masked_patches`:

```python
((image, input_mask), target_patches, loss_weight)

image           (B, H, W, C)   float32   the UNMODIFIED image
input_mask      (B, N)         bool      True -> substitute the learnable MASK token
target_patches  (B, N, P)      float32   P = patch_size * patch_size * C
loss_weight     (B, N)         float32   1{i in S} * (N / n_loss)
```

with `sum_i loss_weight_i == N` per sample. Compile with a stock `loss="mse"`.

### Why the `N / n_loss` scale is exact

`keras.losses.mse` **reduces the last (patch-pixel) axis first**, leaving a `(B, N)` per-token loss. Keras' default `sum_over_batch_size` reduction then divides the weighted sum by `B * N` — **not** by `B * N * P`. Since the weights sum to `N` per sample and are zero off the loss set `S`:

```
sum_{b,i} w_i * l_{b,i} / (B * N)  ==  mean_b [ mean_{i in S} MSE(recon_i, target_i) ]
```

Verified at `N=196` against a numpy reference: delta `0.000e+00`. The dead-mask control (replace `loss_weight` with all-ones) changes the reported loss (1.012808 -> 1.083901), so the weights are demonstrably not being dropped, and `d(loss)/d(recon_i)` is **exactly 0.0** for every `i` outside `S`.

> **WARNING — do not swap in a loss that does not reduce the last axis.** The `N / n_loss` scale factor is derived from the fact that `mse` collapses the `P` axis *before* the reduction. A loss that leaves the `P` axis intact (or one with `reduction="sum"`, or a per-pixel weighting) changes the denominator and **silently rescales the loss**. Nothing errors; the curve still descends; the number just means something else. If you change the loss, re-derive the scale factor and re-run the numpy control.

---

## 3. The 90/10 rule

Two distinct sets are involved, and they are **not** the same set (paper §3):

* the **loss set** `S` — `n_loss = round(mask_ratio * N)` tokens drawn per sample. These are the tokens the reconstruction loss is computed on.
* the **input mask** — a random `n_input = round(mask_token_frac * n_loss)` subset of `S` (default `mask_token_frac = 0.9`). Only these tokens are actually occluded by the learnable MASK token in the model input.

The remaining ~10% of `S` **keep their true patch embedding but still count in the loss**. The paper reports that the Hopfield network only learns meaningful filters when un-occluded patches are present in the loss.

So `input_mask` is a **strict subset** of the loss set by construction (`input_mask ⊊ loss_set`), and the two masks are not interchangeable. A config where rounding collapses `n_input == n_loss` (which deletes the 10% signal) is **rejected at construction time**, not silently accepted.

At imagenette 224/16 (`N=196`, `mask_ratio=0.5`): `n_loss = 98`, `n_input = 88`, 10 un-occluded loss tokens.

---

## 4. No custom `train_step`

There is **no** `train_step`, `test_step` or `compute_loss` override anywhere in this model package, in the trainers, or in the dataset transform — and none may be added. The masking travels through `sample_weight`, which is Keras' sanctioned channel for "which elements count in the loss", so stock `compile` + `fit` does the whole job and every stock callback, metric and checkpoint works unmodified.

This is deliberate and it is **enforced by a test**: a grep for `def train_step|def test_step|def compute_loss` over the model package, the trainers and `masked_patches.py` must return **0 hits**.

---

## 5. Variants and measured cost

All variants share the paper's image defaults: `T=12` descent steps, `step_size (alpha) = 0.1`, `beta = None` (resolved to `1/sqrt(head_dim)`), `attn_self=False` (ET-Full), `hopfield_activation='relu'`. Note `head_dim` is fixed at 64 across scales — it is **not** `embed_dim // num_heads`, because ET attention has no value matrix and the head dimension is a free parameter.

| Variant | `embed_dim` | `num_heads` | `head_dim` | `hopfield_dim` | Params (224/16, N=196, MIM) | Peak GPU memory |
|---|---|---|---|---|---|---|
| `tiny` | 192 | 3 | 64 | 768 | 555,457 | 1167.8 MiB |
| `small` | 384 | 6 | 64 | 1536 | 1,552,513 | 2040.8 MiB |
| `base` | 768 | 12 | 64 | 3072 | 4,873,729 | 3998.9 MiB |

> **Execution mode matters — never quote a bare number.** The memory figures above were **measured** under **stock `fit` in graph mode**, batch size 32, `N=196` (224/16), on an **RTX 4070 (12 GB)**. All three variants fit with headroom (`base` uses ~1/3 of the card). An eager or `jit_compile` figure would differ; a `run_eagerly` figure is a debug number and must be labelled as such.

Backward memory is **linear in `T`** because the descent is unrolled. Raising `--num-steps` raises memory proportionally.

Reference throughput (same mode, `tiny`, batch 32, RTX 4070): 91 ms/step; input pipeline 1142 img/s vs training 352 img/s, so the run is GPU-bound, not I/O-bound, even with imagenette records on a spinning disk.

---

## 6. Hyperparameters (paper Table 4)

The trainers' defaults are the paper's image settings:

| Hyperparameter | Default |
|---|---|
| Optimizer | AdamW |
| Learning rate | `5e-4` |
| Weight decay | `0.05` (via `optimizer_builder` only — **never** also a `kernel_regularizer=L2`, that is double decay) |
| LR schedule | cosine decay |
| Warmup epochs | `2` |
| **Gradient clipping (global norm)** | **`1.0`** |
| Descent steps `T` | `12` |
| Step size `alpha` | `0.1` |
| Mask ratio | `0.5` |
| Mask-token fraction | `0.9` |

### Gradient clipping is load-bearing

The paper reports the model **fails to train at learning rates above `1e-4` without gradient clipping**. The reason is structural: the energy descent is unrolled `T=12` times, so the backward pass composes 12 Jacobians and the gradient norm spikes. Do **not** set `--gradient-clipping 0` "to simplify" and then raise the learning rate — that exact combination is the paper's documented failure mode.

---

## 7. Usage

### Model construction

```python
from dl_techniques.models.energy_transformer import (
    create_energy_transformer_mim,
    create_energy_transformer_classifier,
)

mim = create_energy_transformer_mim('tiny', input_shape=(224, 224, 3), patch_size=16)
mim.compile(optimizer='adamw', loss='mse')   # the sample_weight does the masking

clf = create_energy_transformer_classifier(
    'tiny', input_shape=(224, 224, 3), patch_size=16, num_classes=10,
)
clf.compile(
    optimizer='adamw',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # head emits LOGITS
    metrics=['accuracy'],
)
```

### 1. Masked-completion pretraining

```bash
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m train.energy_transformer.train_masked_completion \
    --dataset imagenette --variant tiny --image-size 224 --patch-size 16 \
    --epochs 100 --batch-size 32
```

Writes `results/<experiment>/{config.json, best_model.keras, training_log.csv, energy_trace.csv, training_history.json}`.

`energy_trace.csv` holds the per-epoch `(T+1)` energy trace, read **out of graph** by `EnergyTraceCallback` (a probe backbone rebuilt with `return_energy=True` and re-synced from the live weights every epoch). The energy is `float32` by design and must never reach a possibly-fp16 head — both model classes **raise** if handed a `return_energy=True` backbone.

### 2. Classification, warm-started from the pretrained trunk

```bash
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m train.energy_transformer.train_classification \
    --dataset imagenette --variant tiny --image-size 224 --patch-size 16 \
    --epochs 100 --batch-size 32 \
    --pretrained-encoder results/et_mim_imagenette_tiny_.../best_model.keras
```

`--pretrained-encoder` transfers the `et_backbone` trunk from the MIM checkpoint by name, skipping the MIM `decoder_*` head. The transfer is **asserted, not logged**:

* `et_backbone` must appear in `report.loaded` — a 0-layer transfer aborts the run rather than training from random init while the command line says "pretrained";
* a trunk shape mismatch (the two runs were built from different variant / image-size / patch-size / num-steps configs) is **fatal**, because `strict=False` would otherwise merely record it and leave the trunk at init.

The backbone config must match between the two runs, or the transfer is refused.

---

## 8. Known limitations

Read these before trusting a number from this package.

### 8.1 `--gpu` is INERT — use the shell instead

The `--gpu N` flag on these trainers (and on **every** trainer in this repo) **does nothing**. `src/train/common/gpu.py`'s `setup_gpu()` runs *after* TensorFlow has already been imported and has already initialized the device, so both its `set_memory_growth` call and its `CUDA_VISIBLE_DEVICES` write arrive too late. Ops land on `GPU:0` regardless.

**Use the shell-level environment variable:**

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m train.energy_transformer.train_masked_completion ...
```

This is a **pre-existing, repo-wide defect**, not something introduced here, and it was left unfixed because `gpu.py` is a shared file outside this work's scope. Note also that the CLI-wiring test guards that a flag **reaches** `TrainingConfig` — it does not and cannot guard that the value has any **effect**.

### 8.2 The warm-start is wired and bit-exact, but its BENEFIT is unproven

The transfer itself is proven: the classifier's trunk weights come out **bit-exact** equal to the MIM checkpoint's (max |Δ| = 0.0), the head is untouched, and all four failure paths raise.

Whether the pretraining **helps** is **not established**. One measured pair, 1 epoch of MIM pretraining on imagenette, `tiny`:

| Run | val_accuracy |
|---|---|
| warm (`--pretrained-encoder`) | 0.28202 |
| cold (random init) | 0.26101 |

A +2.1 point delta that is **not evidence**: one seed, negligible pretraining (a single epoch on 9,469 images), the whole run sits inside the LR warmup, and the *cold* run's training loss was marginally lower. **Do not read this as validation of the pretraining objective.** A real answer needs a proper pretraining budget and multiple seeds.

### 8.3 A 1-epoch run never leaves LR warmup

`warmup_epochs` defaults to `2`. A 1-epoch run therefore trains entirely inside the warmup ramp and never reaches the nominal learning rate. That is fine for a smoke test and **wrong for judging quality**. Any quality claim needs a run long enough to clear warmup.

### 8.4 Graph variants are NOT implemented

The paper's graph anomaly detection and graph classification models are **not** here, and are not a small addition. `EnergyAttention` supports only a **binary 0/1 keep-mask**; the paper's eq.-25 weighted adjacency tensor is a real-valued per-edge bias with no path into the block's `_project` / `energy` / `update`. Expressing it requires a source change to the attention layer **plus a new hand-derived closed-form gradient** (the block has no autodiff path for the energy — the update is a closed form). That work is deferred to a follow-up.

---

## 9. Tests

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
    tests/test_datasets/test_masked_patches.py \
    tests/test_models/test_energy_transformer/ \
    tests/test_train/test_energy_transformer/ -q
```

Covers: the 90/10 rule and `sum w == N`; **patch-order agreement** (an identity-kernel `PatchEmbedding2D` reproduces `patchify_targets` **bit-exactly** — a scrambled patch order would still descend, so this is proven, not assumed); the `.keras` round-trip (deterministic output **and** equal weight count — an output-only compare can miss a dropped weight on a dead path); the energy trace is `float32`, finite and non-increasing at `N=196` even under `mixed_float16`; the masked loss through stock `fit`; and every CLI flag of both trainers reaching `TrainingConfig` (fail-closed — a flag added later is RED by default).

Every guard in that suite was **proven RED** under an injected defect before being accepted, including a dead-component injection: bypassing the ET block makes the overfit loss 7.0x worse, so the block is demonstrably load-bearing.

One test is a `strict` xfail: "a trainable ET block overfits at least 2x faster than a frozen one" is **false** — a frozen block overfits ~2x *faster*, because a random-init ET block is already a strong token mixer and training it concurrently makes the trunk non-stationary under the decoder. That criterion measured memorization speed, not gradient usefulness; the dead-component control is the test that actually answers the question. The xfail is kept rather than deleted so the falsification stays visible.

---

## 10. Citation

```bibtex
@inproceedings{hoover2023energy,
  title     = {Energy Transformer},
  author    = {Hoover, Benjamin and Liang, Yuchen and Pham, Bao and
               Panda, Rameswar and Strobelt, Hendrik and Chau, Duen Horng and
               Zaki, Mohammed J. and Krotov, Dmitry},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023},
  eprint    = {2302.07253}
}
```
