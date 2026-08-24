# Graph Energy Transformer — Node Anomaly (B) + Graph Classification (C-lite)

Keras 3 implementation of the **graph-domain** models from **Hoover et al., "Energy Transformer", NeurIPS 2023** ([arXiv:2302.07253](https://arxiv.org/abs/2302.07253)): variant **B** (per-node anomaly detection, paper §4 / App C) and a variant **C-lite** (whole-graph classification, paper §5 / App D).

Like the image models in `models/energy_transformer/`, these replace a stack of transformer blocks with **one** (or a few) `EnergyTransformer` block(s) that define a single scalar energy `E(g)` over the node states and run `T` steps of gradient descent on it. The forward pass *is* the optimization. By default both variants ride the block's existing **binary 0/1 keep-mask** — supplied as a rank-3 `(B, N, N)` `attention_mask` that literally *is* the graph adjacency — with **zero `layers/` behavior change and zero new hand-derived gradients** on that default path. An **opt-in** `use_weighted_adjacency=True` path (§3) additionally learns the paper's **eq.-25 per-edge weighted adjacency** `Ŵ`; it is the one feature that adds a real (oracle-verified) closed-form gradient term to `EnergyAttention`. With the flag **off** (the default) every existing image and graph consumer is byte-identical.

These two models exist to give the `EnergyTransformer` block a *graph* consumer and to prove that the block's proven image-side fp16/XLA fix transfers verbatim to the graph domain.

---

## Contents

1. [Architecture](#1-architecture)
2. [The two variants](#2-the-two-variants)
3. [Adjacency: binary by default, opt-in eq.-25 weighted (D-001)](#3-adjacency-binary-by-default-opt-in-eq-25-weighted-d-001)
4. [The data contracts](#4-the-data-contracts)
5. [No custom `train_step`](#5-no-custom-train_step)
6. [fp16 / XLA safety is consumer-side (D-002)](#6-fp16--xla-safety-is-consumer-side-d-002)
7. [Hyperparameters](#7-hyperparameters)
8. [Usage](#8-usage)
9. [Known limitations](#9-known-limitations)
10. [Tests](#10-tests)
11. [Citation](#11-citation)

---

## 1. Architecture

One shared trunk, two heads.

```
node_features (B, N, F)   adjacency (B, N, N)   node_mask (B, N)   pe (B, N, k)  [C only]
      |                        |                     |                  |
node projection Dense ---------+---------------------+------------------+
      |                        |                     |                  |
      + (optional) Laplacian-PE-projection Dense  ...................... (C only)
      |
   node-mask token   (graph analog of MaskTokenApply — ALWAYS created, conditionally used)
      |
   EnergyTransformer x num_blocks   (1 block for B, S=4 for C; each unrolls T descent steps
      |                              on ONE scalar energy; built in variable_dtype)
   (B, N, D) node states
      /                         \
  variant B head             variant C-lite head
  concat(g_1 || g_T)         CLS-token readout
  at the target node         (N -> N+1, full CLS row/col in the mask)
  two LayerNorms + MLP       Dense
      |                         |
  (B, 1) logit              (B, num_classes) logits
```

| Class | Role |
|---|---|
| `GraphEnergyTransformerBackbone` | node-projection → (optional Laplacian-PE add) → always-created node-mask token → `num_blocks` × `EnergyTransformer` (each `T` descent steps, `variable_dtype`) → `(B, N, D)`. Exposes `embed` / `call` / `descend_capture`. Stable name `GRAPH_BACKBONE_NAME = "graph_et_backbone"`. |
| `GraphAnomalyDetector` (B) | backbone (`num_blocks=1`) → `descend_capture` returns `g_1` and `g_T` → per-target-node `concat(g_1 ‖ g_T)` (both LayerNormed) → MLP → **single logit** (`from_logits=True`, no in-graph sigmoid). |
| `GraphClassifier` (C-lite) | prepend a CLS token (`N→N+1`) → CLS-augmented adjacency mask (adjacency block + full CLS row/col) → Laplacian-PE features → backbone (`num_blocks=S=4`, `noise_std=0.02`) → CLS-token → `Dense` → **class logits**. |

### Why variant B runs the descent by hand

The `EnergyTransformer` block returns only its *final* state, but variant B needs both the post-step-1 state `g_1` and the final state `g_T`. `descend_capture(g, keep, node_mask, T)` therefore runs the descent **manually** through the block's PUBLIC `.attention.update` / `.hopfield.update` / `.norm` (static Python `range(T)`, `keras.ops` only) and returns the LayerNormed states after step 1 and after step T — **without copy-pasting block internals and without any `layers/` change** (D-002). The manual descent is verified bit-exact against the block's own `call()` (max |Δ| = 0.0).

### Why the two trunks are NOT weight-compatible

Variant C prepends a CLS token, so its `N` is one larger than B's. The two trunks are intentionally **not** warm-start compatible — a B→C transfer would name-match 0 useful layers. `GRAPH_BACKBONE_NAME` is shared so a B→B or C→C pretext transfer *does* match by name, but cross-variant transfer is expected to do nothing. (This mirrors the image README's "a CLS token changes N and breaks the trunk transfer" note.)

---

## 2. The two variants

**Variant B — `GraphAnomalyDetector`.** Given a node, the fraud loader extracts a bounded, size-capped k-hop subgraph (the target node pinned at index 0), runs the ET block over it with attention restricted to the binary adjacency, forms the target's representation as `concat(g_{t=1} ‖ g_{t=T})`, and emits a binary logit trained with **class-weighted** cross-entropy (`ω = benign/anomalous ratio`, carried through stock `fit` via `sample_weight` / `class_weight`).

**Variant C-lite — `GraphClassifier`.** Prepends a CLS token, feeds Laplacian-eigenvector positional features, runs `S=4` stacked ET blocks (each with `noise_std=0.02` saddle-escape noise, eq.-27) over the CLS-augmented binary adjacency, and classifies from the CLS token via a linear head + **label-smoothed** (0.05) softmax cross-entropy on one-hot targets.

Both compose the SAME `GraphEnergyTransformerBackbone`, serialize via `.keras`, and train with stock `fit` — no custom `train_step`.

---

## 3. Adjacency: binary by default, opt-in eq.-25 weighted (D-001)

### 3.1 The default — binary adjacency (C-lite)

**By default both variants ride the existing rank-3 binary `(B, N, N)` `attention_mask`.** The mask is a 0/1 keep-mask: `mask[b, n, m] = 1` iff node `n` attends to node `m` (i.e. there is an edge). This is a `g`-independent constant, so it needs **no new gradient**. This is the "C-lite" model — variant C minus the learned per-edge weight — and it remains the default, byte-identical to before this feature.

### 3.2 Opt-in — the paper's eq.-25 weighted adjacency (`use_weighted_adjacency=True`, Branch A)

The paper's variant C uses a real-valued, *learned* per-edge adjacency `Ŵ = Conv2D(X ⊗ X) ⊙ A′`. Setting `use_weighted_adjacency=True` (default `False`) turns this on:

* **What it computes.** Each `EnergyTransformer` block builds a per-head edge weight `Ŵ_{hnm}` **once per block** — from the block's *input* tokens `X` (a `WeightedAdjacencyProjector`: `X ⊗ X → Conv2D → (B, H, N, N)`, gated by the binary adjacency `A′`), **not** from the evolving state `g`. This is **Branch A**: because the paper computes `Â` once per block (eq.-27 iterates the tokens over `T` steps but never re-derives `Â`), `Ŵ` is a **per-call constant**, hoisted out of the `T`-step descent loop exactly like the existing keep-mask.
* **How it enters the energy.** `Ŵ` modulates the attention score **multiplicatively**: `logit = β·A·Ŵ + M`, where `A = K·Q` is today's raw score and `M` is the unchanged `−∞`/0 keep-bias built from the binary `A′`. `Ŵ` carries **only** the finite learned weight; the non-edge masking is still `M`, so the two roles never mix.
* **The gradient is hand-derived and oracle-verified.** The block's descent is a closed form (there is no autodiff path for the energy), so `Ŵ`'s effect on `-dE/dg` was derived by hand: `omega_eff = softmax(logit, axis=key) · keep · Ŵ` substitutes for `omega` in `EnergyAttention`'s `term_q` / `term_k`. `update() == -dE^ATT/dg` is verified against a `tf.GradientTape` oracle at **N = 64** (f32) and **N = 1024** (f64), and the guard is proven RED (deleting the `·Ŵ` factor turns the oracle red). The projector's own weights train through the ordinary outer `fit()` backward pass; only `omega_eff` is hand-coded.
* **Cost knobs.** `adjacency_kernel_size` (default `1`) sets the Conv2D receptive field over the pairing grid. `adjacency_proj_dim` (default `None`) reduces the token dim `D` **before** the `X ⊗ X` outer product, cutting the `D²` pairing-channel cost that would otherwise dominate for large graphs — set it when the `D²`-channel Conv OOMs.
* **CLI.** `train_classification.py` exposes `--use-weighted-adjacency` (plus `--adjacency-kernel-size`, `--adjacency-proj-dim`) to flip the flag from the shell.

**Scope note — faithful, not benchmarked.** This is a faithful interpretation of eq.-25's *multiplicative* form (verified gradient + energy-descent monotonicity + fp16/XLA fit guard, all proven RED). It is **not** an end-to-end accuracy claim: matching the paper's reported variant-C classification numbers on the TUDatasets is a separate campaign and is **not** benchmarked here.

> With `use_weighted_adjacency=False` (the default), `git diff --stat src/dl_techniques/layers/` is behaviorally inert (`y(flag off) == y(no flag)`, byte-identical), and a grep finds **no** `train_step` / `test_step` / `compute_loss` / `GradientTape` in the model or trainer `src/` code — the descent stays a closed form, never autodiff. Those facts are the D-001 boundary; the only `layers/` sources the weighted path touches are `energy_attention.py` and `energy_transformer.py`.

---

## 4. The data contracts

Two `tf.data` builders in `dl_techniques.datasets.graphs` emit the two input dicts. Read these before writing a trainer.

**Variant B — `build_fraud_subgraph_dataset`** (`datasets/graphs/fraud.py`):

```python
({"node_features": (B, N, F),   # per-node features of the sampled subgraph
  "adjacency":     (B, N, N),   # binary, SYMMETRIC (A == A.T is a caller obligation)
  "node_mask":     (B, N),      # rank-2 validity, 1 = real node, 0 = PAD row
  "target_index":  (B,)},       # kept for forward-compat; the head reads index 0 (see D-003)
 label)                         # (B,) binary node label
```

**Variant C-lite — `build_tudataset_graph_dataset`** (`datasets/graphs/tudataset.py`):

```python
({"node_features": (B, N, F),   # one-hot node labels
  "adjacency":     (B, N, N),   # binary, SYMMETRIC
  "pe":            (B, N, 15),  # top-k=15 Laplacian eigenvectors (per-epoch random sign-flip)
  "node_mask":     (B, N)},     # rank-2 validity
 label)                         # (B, num_classes) one-hot graph label
```

* **Data sources.** TUDataset (MUTAG / PROTEINS / ...) is parsed with a stdlib `zipfile` + `numpy` + `scipy.sparse` parser (no PyTorch-Geometric); the real Amazon / YelpChi CARE-GNN fraud graphs are read from `.mat` via `scipy.io.loadmat` (classic) with an `h5py` fallback (v7.3). Both loaders cache under `DEFAULT_GRAPH_CACHE = /media/arxwn/data0_4tb/datasets/graphs/` — **never** the repo, **never** `data_fast`. A network-free synthetic fraud generator (`make_synthetic_fraud_graph`) backs the offline smoke path and every test.
* **Adjacency symmetry is a CALLER obligation.** The mask is applied literally `(n=key, m=query)` and is **not** auto-symmetrized; the loaders emit `A == A.T`.
* **PAD rows are the fp16 hazard surface.** Dense-batched graphs pad to `N` with all-zero rows; those near-constant tokens are the exact `fp16 × near-constant × XLA` triple that silently freezes `EnergyLayerNorm` backward — see §6. The rank-2 `node_mask` also excludes PAD from the Hopfield energy `E_HN`.
* **Trainers** live under `src/train/graph_energy_transformer/` (`train_anomaly.py`, `train_classification.py`); results always go to the repo-root `results/` directory, GPU1 only (`CUDA_VISIBLE_DEVICES=1`), `MPLBACKEND=Agg`, via `.venv`.

---

## 5. No custom `train_step`

There is **no** `train_step`, `test_step` or `compute_loss` override anywhere in this model package or in the trainers — and none may be added. Variant B's class imbalance travels through `sample_weight` / `class_weight`, Keras' sanctioned channel for "which elements count in the loss", so stock `compile` + `fit` does the whole job and every stock callback, metric and checkpoint works unmodified. This is enforced by the SC3 grep (0 hits) and the frozen-`layers/` diff.

---

## 6. fp16 / XLA safety is consumer-side (D-002)

`GraphEnergyTransformerBackbone` builds each `EnergyTransformer` block with `dtype=self.dtype_policy.variable_dtype` (via an overridable `_make_block` seam) and casts the node states into and back out of it in `call()`. Under `float32` those casts are provable no-ops; under `mixed_float16` they force the block to compute in `float32`.

**Why**, and what NOT to do instead (identical to the image README §5 / §8.4):

* `EnergyLayerNorm`'s backward forms `(var + eps)^(-3/2)`. In fp16 that intermediate **overflows 65504** whenever `eps < 6.1e-4`; at the default `eps = 1e-5` it is ~`3.2e7` → `inf`, and `0 * inf` → `NaN`. **PAD rows** (`var ≈ 0`) supply the near-constant tokens that reach the cliff, and **XLA** (which `fit` enables by default) keeps the intermediate in fp16.
* The result is a **silent dead trainer**: every gradient non-finite → `LossScaleOptimizer` rejects 100 % of steps (scale collapses) → **all weights move by exactly 0.0** while the loss stays finite. Nothing errors.
* **Do NOT remove the cast** "to save a copy" — it reinstates the overflow. **Do NOT raise `norm_epsilon` to `1e-3`** — it clears the overflow but silently trains a *different* network (a fix that makes fp16 and fp32 disagree silently is worse than the bug).

The defect itself is still in `layers/norms/energy_layer_norm.py`, untouched (0-`layers/`-change budget). **Any OTHER consumer of `EnergyLayerNorm` / `EnergyTransformer` that runs `mixed_float16` + XLA on near-constant tokens has this bug today.**

**The guard is proven RED.** `test_model.py`'s fp16/XLA guard runs a real backward `fit` at `N ≥ 64` under `mixed_float16` + `jit_compile=True` for **both** variants and asserts weights MOVE and the loss scale does not collapse (GREEN: scale held 32768, trunk moves). It then injects the dead component — a fp16-unsafe block via a `_make_block` subclass override — and asserts the guard goes RED (scale → 0.03125, all weights frozen). **A forward-only fp16 test certifies nothing** (that is exactly the trap that let the image-side dead trainer ship); this guard is backward, at realistic `N`, and dead-component-injected.

---

## 7. Hyperparameters

### Variant B — paper Table 6 (training)

| Hyperparameter | Default | Note |
|---|---|---|
| Optimizer | Adam | plain Adam, `weight_decay = 0` |
| Learning rate | `1e-3` | |
| LR schedule | cosine decay | |
| Epochs | `100` | |
| Descent steps `T` | `2` | paper `T ∈ {1, 2, 3}` |
| Step size `alpha` | `0.1` | |
| Gradient clipping (global norm) | `1.0` | the unrolled descent spikes the gradient norm |
| Loss | `BinaryCrossentropy(from_logits=True)` | class imbalance via `sample_weight = ω` |

### Variant C-lite — paper Table 9 (training)

| Hyperparameter | Default | Note |
|---|---|---|
| Optimizer | AdamW | `weight_decay = 1e-4` |
| Learning rate | `1e-3` | |
| LR schedule | cosine decay + warmup (`5` epochs) | |
| Epochs | `300`, batch `32` | |
| Stacked blocks `S` | `4` | |
| Descent steps `T` | `12` | |
| Step size `alpha` | `0.01` | |
| `beta` | `1/sqrt(64)` | resolved from `head_dim = 64` |
| Token dim / heads / head_dim | `128` / `12` / `64` | |
| Saddle-escape `noise_std` | `0.02` | eq.-27, **training only** |
| Label smoothing | `0.05` | one-hot + `CategoricalCrossentropy(from_logits=True)` |
| Gradient clipping (global norm) | `1.0` | |

> **Gradient centralization is omitted.** The paper's variant-C optimizer applies gradient centralization; the repo's `optimizer_builder` has no such option, so it is intentionally **not** wired (documented, not silently dropped). This is the one Table-9 knob this implementation does not reproduce.

---

## 8. Usage

### Model construction

```python
from dl_techniques.models.graph_energy_transformer import (
    create_graph_anomaly_detector,
    create_graph_classifier,
)
import keras

# Variant B — node anomaly
det = create_graph_anomaly_detector(
    node_feature_dim=25, embed_dim=64, num_heads=4, head_dim=16,
    hopfield_dim=128, mlp_hidden_dim=64, num_steps=2,
)
det.compile(optimizer="adam",
            loss=keras.losses.BinaryCrossentropy(from_logits=True))  # sample_weight does the ω

# Variant C-lite — graph classification
clf = create_graph_classifier(node_feature_dim=7, num_classes=2)     # Table-9 defaults
clf.compile(optimizer="adamw",
            loss=keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.05),
            metrics=["accuracy"])
```

### Training

```bash
# Variant B (Amazon primary; synthetic fallback is network-free)
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
    -m train.graph_energy_transformer.train_anomaly --dataset amazon --epochs 100

# Variant C-lite (TUDataset MUTAG / PROTEINS)
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
    -m train.graph_energy_transformer.train_classification --dataset mutag --epochs 300 --batch-size 32
```

Both write `results/<experiment>/{config.json, best_model.keras, training_log.csv, ...}`.

### `.keras` round-trip is the acceptance gate

`GraphAnomalyDetector` and `GraphClassifier` both survive a `.keras` save/load with **deterministic output equal AND equal weight count** (backbone 8/8, B 16/16, C 30/30) — an output-only compare can miss a dropped weight on a dead path, so both are checked. Variant B is designed for this: the target readout is a **static index-0 slice** (`caps[t][:, 0, :]`, the same XLA-safe readout C-lite uses for CLS), not a dynamic gather — the fraud sampler always pins the target at index 0, so the slice is byte-identical for every input the sampler produces and keeps the whole B path XLA-compilable (D-003).

---

## 9. Known limitations

* **`--gpu N` is INERT — use the shell.** As with every trainer in this repo, `setup_gpu()` runs after TF has initialized the device, so the flag arrives too late. Always prefix `CUDA_VISIBLE_DEVICES=1`. The CLI-wiring test guards that a flag *reaches* `TrainingConfig`, not that it has an *effect*.
* **eq.-25 weighted adjacency is opt-in, not benchmarked** — available via `use_weighted_adjacency=True` (§3.2, Branch A: gradient oracle-verified, default-off byte-identical). The default is still the binary C-lite path. Matching the paper's reported variant-C accuracy on the TUDatasets is a separate, un-run campaign.
* **Variant B needs bounded subgraphs.** A full dense `N×N` over the whole Amazon (~12k) / YelpChi (~45k) graph is infeasible and would OOM; B always trains on size-capped k-hop subgraphs, never a single dense adjacency.
* **Gradient centralization omitted** (§7) — the one Table-9 optimizer knob not reproduced.
* **Cross-variant warm-start does nothing** — B and C are not trunk-weight-compatible (CLS token changes `N`). Only B→B / C→C match by name.
* **fp16 forward-only tests certify nothing** — the load-bearing guard is the backward, dead-component-injected one in §6.

---

## 10. Tests

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
    tests/test_models/test_graph_energy_transformer \
    tests/test_datasets/test_graphs \
    tests/test_train/test_graph_energy_transformer -vvv
```

Covers: backbone / B / C init, forward, `.keras` round-trip (deterministic output **and** equal weight count); the **proven-RED** fp16/XLA backward guard (green with the `variable_dtype` cast, red under the dead-component injection); the TUDataset parser matching published MUTAG stats (188 graphs, 2 classes, ~17.9 avg nodes); Laplacian PE shape `(N, 15)`, column orthonormality, and sign-flip (changes sign not magnitude); the fraud subgraph sampler's size cap + symmetric adjacency; dense-pad + node-mask correctness; and the **fail-closed CLI-to-config wiring** for both trainers (a flag added later is RED by default, and a dropped wiring line is proven to bite). All network-free — an autouse fixture blocks network access.

---

## 11. Citation

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
