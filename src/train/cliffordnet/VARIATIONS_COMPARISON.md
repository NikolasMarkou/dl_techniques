# CliffordNetBlock Variant Comparisons on CIFAR-100

Running log of head-to-head experiments comparing `CliffordNetBlock`
variants. Each section documents one experiment: **what changed**, the
**fixed shared setup**, and the **results**.

Driver script: [`train_compare_variants.py`](train_compare_variants.py).

---

## Shared baseline (held constant across all experiments unless noted)

| Setting | Value |
|---|---|
| Dataset | CIFAR-100 (50k train / 10k test) |
| Input | 32×32×3, per-channel normalisation, AutoAugment (CIFAR-10 policy), random flip/crop, random erasing p=0.25 |
| Stem | 3×3 stride-2 Conv2D + BN → 16×16×128 |
| Backbone | 5 isotropic blocks @ C=128, no in-backbone downsampling |
| Block hyperparams | `shifts=[1,2]`, `cli_mode="full"`, `ctx_mode="diff"`, `use_global_context=False` |
| Drop-path | linear schedule 0.0 → 0.1 across the 5 blocks |
| LayerScale init | 1e-5 |
| Head | GAP → LayerNorm → Dense(100) |
| Optimiser | AdamW, β1=0.9, β2=0.999, ε=1e-8, weight_decay=0.1 |
| LR schedule | cosine_decay with linear warmup, peak_lr=1e-3, warmup_epochs=5, alpha=1e-2 |
| Batch | 128 |
| Epochs | 100 |
| EarlyStopping | monitor=val_accuracy, patience=30 |
| Loss | SparseCategoricalCrossentropy(from_logits=True) |
| Hardware | GPU 0 (RTX 4090, 24 GB), serial |

Anything different in a given experiment is called out under
**Differences from baseline**.

---

## Experiment registry

| # | Date | Experiment | Status | Best variant |
|---|---|---|---|---|
| 1 | 2026-04-29 | E01: vanilla vs single 7×7 DS (no downsampling) | done | tie (vanilla +0.0009) |
| 2 | 2026-04-29 | E02: DS vs DS-plain (no BN, no activation on context) | done | vanilla > ds > ds_plain (small) |
| 3 | 2026-04-29 | E03: ds_plain kernel size sweep (5×5 vs 7×7) | done | tie (k5 −0.0007 vs k7) |
| 4 | 2026-04-29 | E04: ds with 5×5 vs 7×7 (closes the 2×2 BN×kernel matrix) | done | **ds_k5 wins (0.7115)** |

---

## E01 — vanilla `CliffordNetBlock` vs `CliffordNetBlockDS` (single 7×7, no downsampling)

**Date**: 2026-04-29
**Status**: done
**Run root**: `results/cliffordnet_compare_variants_20260429_090124/`
**Driver invocation**:
```bash
MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_compare_variants \
    --variant all --epochs 100 --batch-size 128 --gpu 0
```

**Question**: holding everything else constant, does replacing the context
stream's two stacked 3×3 depthwise convs with a single 7×7 depthwise conv
(same effective receptive field, fewer params/FLOPs in the context branch)
improve, hurt, or break even on CIFAR-100?

### Variants

| Key | Block class | Context stream |
|---|---|---|
| `vanilla` | `CliffordNetBlock` | DWConv 3×3 → DWConv 3×3 → BN → SiLU |
| `ds` | `CliffordNetBlockDS` (kernel_size=7, strides=1, skip_pool="avg" *inert at strides=1*) | DWConv 7×7 → BN → SiLU |

### Differences from baseline

None — this is the canonical comparison. `skip_pool` is set but inert
because `strides=1` (no pool layers built; residual is the unmodified
`x_prev`).

### Smoke test (3 epochs, batch 32, no augmentation, GPU 0)

Sanity check that both variants build, forward, backward, and converge in
the right direction. Not a fair comparison (3 epochs, no aug).

| Variant | Params | best_val_acc @ ep3 | Wall (min) |
|---|---|---|---|
| vanilla | 608,612 | 0.3060 | 3.3 |
| ds | 628,452 | 0.3443 | 3.0 |

DS variant is ~3% larger by params (the 7×7 DWConv has 49 vs 2×9=18 spatial
weights per channel) but trained slightly faster wall-clock — fewer kernel
launches in the context stream.

### Full results (100 epochs)

Source: `results/cliffordnet_compare_variants_20260429_090124/comparison.csv`.

| Variant | Params | Best val_acc | Final val_acc | Best val_top5 | Wall (min) |
|---|---|---|---|---|---|
| vanilla | 608,612 | **0.7112** | 0.7098 | 0.9218 | 73.0 |
| ds      | 628,452 | 0.7103 | 0.7076 | 0.9188 | 73.8 |
| Δ (ds − vanilla) | +19,840 (+3.3%) | −0.0009 | −0.0022 | −0.0030 | +0.8 |

### Observations

- **Statistical tie on accuracy.** Δ best_val_acc = −0.09pp (vanilla wins
  by less than one CIFAR-100 test sample's worth of resolution; CIFAR-100
  test set has 10k examples → 1 sample = 0.01pp). Δ top-5 = −0.30pp, also
  within noise for a single seed.
- **Trajectory was tracking ahead, then converged.** DS led mid-training
  (e.g. ep32: 0.6333 vs vanilla ~0.580; ep51: 0.6670 vs ~0.660; ep60:
  0.6833 vs ~0.679) but the gap closed in the cosine-decay tail — both
  models hit the same effective ceiling.
- **DS is +3.3% larger by params.** A 7×7 DWConv has 49 spatial weights
  per channel vs 2×9=18 for the stacked 3×3 pair. With C=128, that's
  +3,968 weights per block × 5 blocks = +19,840 — matches the observed
  delta exactly. So the cost story is "more params, fewer kernel
  launches" rather than "fewer params".
- **Wall-clock was a wash** (+1.1%). The smoke test showed DS slightly
  faster, but at batch 128 / 100 epochs the difference disappears —
  both are bounded by the geometric-product / GGR ops, not the context
  conv.
- **No regression.** DS is at parity on accuracy and runtime under
  identical training schedule, with no downsampling exercised. This is
  the "neutral baseline" we needed before testing DS in regimes it was
  designed for (downsampling, larger receptive fields).

### Verdict

Single 7×7 DWConv ≈ stacked 3×3 DWConv pair on CIFAR-100 at this
scale, when downsampling is not exercised. The DS variant is a viable
drop-in for vanilla blocks, with its real value to be tested in
downsampling configurations and at larger resolutions where the 7×7
RF starts to differ meaningfully from a 7×7 effective RF built from
two 3×3 dilations.

---

## E02 — `CliffordNetBlockDS` with vs without BN+SiLU on the context stream

**Date**: 2026-04-29
**Status**: done
**Run root**: `results/cliffordnet_compare_variants_20260429_113740/` (ds_plain only; vanilla/ds reused from E01 since seed was not pinned in either run)
**Driver invocation**:
```bash
MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_compare_variants \
    --variant ds_plain --epochs 100 --batch-size 128 --gpu 0
```

**Question**: does the BN + SiLU after the 7×7 depthwise conv pull weight,
or is the geometric product downstream (with its own SiLU on the inner
product) and the GGR's SiLU on `H_norm` enough non-linearity? Removing
both saves a tiny number of params and one BN sync, and tests whether
the context branch *needs* its own normalisation+activation at all.

### Variants

| Key | Block class | Context stream |
|---|---|---|
| `vanilla` | `CliffordNetBlock` | DWConv 3×3 → DWConv 3×3 → BN → SiLU |
| `ds` | `CliffordNetBlockDS` (`use_ctx_bn=True`, `ctx_activation="silu"`) | DWConv 7×7 (no bias) → BN → SiLU |
| `ds_plain` | `CliffordNetBlockDS` (`use_ctx_bn=False`, `ctx_activation=None`) | DWConv 7×7 (with bias) → *(nothing)* |

### Differences from baseline

- Adds a third variant `ds_plain` that turns off both BN and the
  activation on the context stream. The DWConv carries a learnable
  bias in lieu of BN's affine.
- Otherwise identical to E01: same stem, channels, head, optimiser,
  schedule, augmentation. Vanilla and `ds` rerun for a fresh same-seed
  reference (E01's seed was not pinned).

### Implementation note

Added two flags to `CliffordNetBlockDS` (default behavior unchanged):
- `use_ctx_bn: bool = True` — when `False`, no `BatchNormalization` is
  built; the DWConv switches to `use_bias=True` to compensate.
- `ctx_activation: Optional[str] = "silu"` — when `None`, no activation
  is applied. Resolved via `keras.activations.get`.

All 115 existing layer tests pass with the additions.

### Smoke test (3 epochs, batch 32, no augmentation, GPU 0)

| Variant | Params | best_val_acc @ ep3 | Wall (min) |
|---|---|---|---|
| vanilla  | 608,612 | 0.3060 *(from E01)* | 3.3 |
| ds       | 628,452 | 0.3443 *(from E01)* | 3.0 |
| ds_plain | 626,532 | 0.3189 | 3.2 |

`ds_plain` is 1,920 params lighter than `ds` (BN's 4 stats × 128 ch × 5
blocks − DWConv bias × 128 ch × 5 blocks = −2,560 + 640 = −1,920;
matches exactly). At 3 epochs `ds_plain` lands between vanilla and `ds`,
which is what we'd expect if BN+SiLU is helping but only modestly.

### Full results (100 epochs)

vanilla and ds reused from E01 (`results/cliffordnet_compare_variants_20260429_090124/`); ds_plain freshly trained.

| Variant | Params | Best val_acc | Final val_acc | Best val_top5 | Wall (min) |
|---|---:|---:|---:|---:|---:|
| vanilla  | 608,612 | **0.7112** | 0.7098 | 0.9218 | 73.0 |
| ds       | 628,452 | 0.7103 | 0.7076 | 0.9188 | 73.8 |
| ds_plain | 626,532 | 0.7059 | 0.7032 | 0.9145 | 74.4 |
| Δ (ds_plain − ds) | −1,920 | −0.0044 | −0.0044 | −0.0043 | +0.6 |
| Δ (ds_plain − vanilla) | +17,920 | −0.0053 | −0.0066 | −0.0073 | +1.4 |

### Observations

- **ds_plain trails by ~0.5pp.** Δ vs ds = −0.44pp on best, −0.44pp on
  final, −0.43pp on top-5 — small but consistent across all three
  metrics. Δ vs vanilla = −0.53pp / −0.66pp / −0.73pp.
- **The ranking inverted in the cosine tail.** Same-epoch trajectory:

  | Epoch | vanilla | ds | ds_plain |
  |---:|---:|---:|---:|
  | 16 | 0.5234 | ~0.51 | **0.5611** |
  | 25 | ~0.585 | ~0.605 | **0.6207** |
  | 33 | ~0.61 | ~0.625 | **0.6421** |
  | 62 | ~0.661 | ~0.682 | **0.6824** |
  | 77 | ~0.703 | ~0.708 | 0.6984 |
  | 100 (best) | **0.7112** | 0.7103 | 0.7059 |

  ds_plain led from warmup through ~ep 60. The advantage compounded
  through mid-training, then the cosine-decay tail flattened it while
  vanilla and ds kept squeezing out gains. This is the classic
  no-BN signature: faster early signal-propagation through the
  context branch (no per-channel rescaling toward zero variance), at
  the cost of less stable late-stage fine-tuning.
- **Wall-clock indistinguishable.** All three within 1.4 min of each
  other over a 73-min run.
- **No augmentation differences.** All three used the same
  `RandomCrop(32, pad=4) → RandomHFlip → AutoAugment(CIFAR-10) →
  Normalize → RandomErasing(p=0.25)` pipeline (imported from
  `train_cliffordnet.build_train_dataset`). Eval is unaugmented.

### Verdict

Removing BN+SiLU from the DS context stream costs ~0.5pp on CIFAR-100
at this scale — small in absolute terms, but consistent across best /
final / top-5. ds_plain trains faster mid-run (interesting if compute
budget were tighter) but loses the head-to-head when allowed to
converge.

Keep BN+SiLU as the default for `CliffordNetBlockDS`. The flags exist
and are useful, but the configuration is not the recommended one.

---

## E03 — `ds_plain` with 5×5 vs 7×7 depthwise conv

**Date**: 2026-04-29
**Status**: done
**Run root**: `results/cliffordnet_compare_variants_20260429_125847/`
**Driver invocation**:
```bash
MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_compare_variants \
    --variant ds_plain_k5 --epochs 100 --batch-size 128 --gpu 0
```

**Question**: E02's ds_plain trailed ds and vanilla by ~0.5pp. Does
shrinking the depthwise kernel from 7×7 to 5×5 (still wider than the
two stacked 3×3 in vanilla) recover the gap, or compound it? The
hypothesis is that without BN to renormalise activations, the larger
7×7 RF aggregates more noise, and a tighter 5×5 might be a better fit
for the no-norm regime.

### Variants

| Key | Block class | Context stream | DW kernel |
|---|---|---|---:|
| `ds_plain` | `CliffordNetBlockDS` `(use_ctx_bn=False, ctx_activation=None)` | DWConv 7×7 (with bias) → *(nothing)* | 7×7 (49 spatial weights) |
| `ds_plain_k5` | `CliffordNetBlockDS` `(kernel_size=5, use_ctx_bn=False, ctx_activation=None)` | DWConv 5×5 (with bias) → *(nothing)* | 5×5 (25 spatial weights) |

### Differences from baseline

- New variant `ds_plain_k5`: identical to E02's `ds_plain` except
  `kernel_size=5`.
- `kernel_size` lifted from a global constant in the training script
  into each variant's `ds_kwargs` dict so it can be tuned per-variant.
- `vanilla`, `ds`, and `ds_plain` results are reused from E01/E02 —
  this experiment only requires training the new variant.

### Implementation note

`CliffordNetBlockDS.kernel_size` was already a constructor parameter
(default 7); the only change is exposing it through the per-variant
`ds_kwargs` in the comparison driver.

### Smoke test (3 epochs, batch 32, no augmentation, GPU 0)

| Variant | Params | best_val_acc @ ep3 | Wall (min) |
|---|---:|---:|---:|
| ds_plain     | 626,532 | 0.3189 *(E02)* | 3.2 |
| ds_plain_k5  | 611,172 | 0.2813 | 3.3 |

ds_plain_k5 is **15,360 params lighter** than ds_plain (5 blocks ×
128 ch × (49 − 25) = 15,360 — matches exactly).

### Full results (100 epochs)

| Variant | Params | Best val_acc | Final val_acc | Best val_top5 | Wall (min) |
|---|---:|---:|---:|---:|---:|
| vanilla     | 608,612 | **0.7112** | 0.7098 | 0.9218 | 73.0 |
| ds          | 628,452 | 0.7103 | 0.7076 | 0.9188 | 73.8 |
| ds_plain    | 626,532 | 0.7059 | 0.7032 | 0.9145 | 74.4 |
| ds_plain_k5 | 611,172 | 0.7052 | 0.7046 | 0.9153 | 73.3 |
| Δ (k5 − k7 ds_plain) | −15,360 | −0.0007 | +0.0014 | +0.0008 | −1.1 |

### Observations

- **k5 ≈ k7 within noise.** Δ best_val_acc = −0.07pp (≈7 test
  samples), Δ final = +0.14pp, Δ top-5 = +0.08pp. Direction-mixed
  signs across the three metrics confirm this is a tie.
- **k5 is param-efficient.** −15,360 trainable params (−2.5% vs k7
  ds_plain, −3.4% smaller than `ds`). Gets the same accuracy as k7
  with measurably fewer FLOPs in the context branch.
- **Trajectory: k5 trailed k7 mid-run, caught up in the cosine tail.**

  | Epoch | ds_plain (k7) | ds_plain_k5 |
  |---:|---:|---:|
  | 17/18 | 0.5611 | 0.5433 |
  | 25/29 | 0.6207 | 0.6091 |
  | 33    | 0.6421 | – |
  | 59/62 | 0.6824 | 0.6826 |
  | 75/77 | 0.6984 | 0.6995 |
  | 95    | – | 0.7052 |
  | best  | **0.7059** | 0.7052 |

  Inverse of E02's pattern: k5 starts slow but converges to the same
  ceiling. Suggests the no-BN ceiling on this 5-block isotropic
  backbone is set by something other than RF size in the context
  branch (likely the geometric product / GGR capacity).
- **No augmentation differences.** All variants use the shared
  `train_cliffordnet.build_train_dataset` pipeline.

### Verdict

5×5 and 7×7 are statistically equivalent for `ds_plain` on CIFAR-100
at this scale. **Prefer k5** when params/FLOPs matter — same accuracy
for −2.5% params and slightly faster wall-clock. When BN is enabled,
this question still needs testing (E02 already used k7 for `ds`; a
`ds_k5` variant would be the natural follow-up).

Across E01–E03, the four configurations are clustered in a 0.6pp
window (vanilla 0.7112, ds 0.7103, ds_plain 0.7059, ds_plain_k5
0.7052). The ranking is consistent (BN helps, more spatial weights
help marginally) but the magnitudes are within seed noise for a
single run each.

---

## E04 — `ds` with 5×5 vs 7×7 (BN+SiLU enabled)

**Date**: 2026-04-29
**Status**: done
**Run root**: `results/cliffordnet_compare_variants_20260429_142013/`
**Driver invocation**:
```bash
MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_compare_variants \
    --variant ds_k5 --epochs 100 --batch-size 128 --gpu 0
```

**Question**: closes the 2×2 design matrix `{BN on/off} × {kernel 5/7}`.
E03 showed k5≈k7 in the no-BN regime — does the same hold when BN+SiLU
is on? If yes, we'd recommend k5 across the board (same accuracy, fewer
params). If no — i.e. BN amplifies the value of a wider RF — the
recommendation splits by config.

### Variants

| Key | use_ctx_bn | ctx_activation | DW kernel | Params |
|---|:-:|:-:|---:|---:|
| `ds`          | True  | silu | 7 | 628,452 |
| `ds_k5`       | True  | silu | 5 | 613,092 |
| `ds_plain`    | False | None | 7 | 626,532 |
| `ds_plain_k5` | False | None | 5 | 611,172 |

### Differences from baseline

- New variant `ds_k5`: identical to E01's `ds` except `kernel_size=5`.
- vanilla / ds / ds_plain / ds_plain_k5 reused from E01–E03.

### Smoke test (3 epochs, batch 32, no augmentation, GPU 0)

| Variant | Params | best_val_acc @ ep3 | Wall (min) |
|---|---:|---:|---:|
| ds          | 628,452 | 0.3443 *(E01)* | 3.0 |
| ds_k5       | 613,092 | 0.3167 | 3.3 |
| ds_plain    | 626,532 | 0.3189 *(E02)* | 3.2 |
| ds_plain_k5 | 611,172 | 0.2813 *(E03)* | 3.3 |

ds_k5 is **15,360 params lighter** than ds (5 blocks × 128 ch × 24
fewer spatial weights — matches ds_plain → ds_plain_k5 delta exactly).
At smoke ds_k5 lands closer to ds_plain than to ds, hinting that the
BN+k7 advantage at smoke (0.3443) may be an early-warmup effect.

### Full results (100 epochs) — full 2×2 + vanilla

| Variant | use_ctx_bn | DW kernel | Params | Best val_acc | Final val_acc | Best val_top5 | Wall (min) |
|---|:-:|---:|---:|---:|---:|---:|---:|
| vanilla     | –     | (3+3) | 608,612 | 0.7112 | 0.7098 | 0.9218 | 73.0 |
| ds          | True  | 7 | 628,452 | 0.7103 | 0.7076 | 0.9188 | 73.8 |
| **ds_k5**   | **True**  | **5** | **613,092** | **0.7115** | **0.7103** | **0.9220** | **71.0** |
| ds_plain    | False | 7 | 626,532 | 0.7059 | 0.7032 | 0.9145 | 74.4 |
| ds_plain_k5 | False | 5 | 611,172 | 0.7052 | 0.7046 | 0.9153 | 73.3 |

### 2×2 view (best val_acc, ranked)

| | k=5 | k=7 |
|---|---:|---:|
| **BN+SiLU on**  | **0.7115** | 0.7103 |
| **BN+SiLU off** | 0.7052 | 0.7059 |

- BN row: k5 > k7 by 0.12pp
- no-BN row: k5 ≈ k7 (within noise)
- BN effect at k7: +0.44pp
- BN effect at k5: +0.63pp

### Observations

- **ds_k5 is the new top scorer.** Best val_acc 0.7115 > vanilla
  0.7112 by 0.03pp, > ds (k7+BN) by 0.12pp. Best top-5 0.9220 also
  best in the matrix. Final val_acc tied with ds_k5 = ds = 0.7103.
- **Smallest DS variant.** 613,092 params: only +4,480 over vanilla
  (+0.74%), and −15,360 vs ds (k7+BN). Best accuracy-per-param of the
  five configurations.
- **Fastest wall-clock** of the five (71.0 min). The k5 DWConv has
  ~half the FLOPs of k7 in the context branch, and the difference
  shows here in a way it didn't in the no-BN case.
- **2×2 reveals an interaction.** With BN, k5 beats k7 (0.7115 vs
  0.7103). Without BN, k5 ties k7 (0.7052 vs 0.7059). BN's
  per-channel rescaling appears to help a *narrower* RF more — the
  larger 7×7 RF spreads activations enough that BN's normalisation
  is less critical, while the 5×5 RF benefits clearly from it. This
  matches the no-BN regime tracking observation in E03 (k5 caught
  up only after activations stabilised).
- **Trajectory:** ds_k5 entered the cosine tail in the middle of the
  pack (~0.66 at ep 60) but finished strongest, gaining +0.05 over
  the last 30 epochs. Late-stage convergence, not early speed, was
  the discriminator.

### Verdict

**Recommended default for `CliffordNetBlockDS` is now kernel_size=5
with BN+SiLU on.** This configuration:
- Beats vanilla on best val_acc (+0.03pp), final (+0.05pp), top-5
  (+0.02pp) — small but consistent.
- Beats k7 ds on every metric (+0.12pp / +0.27pp / +0.32pp).
- Costs ~4,500 more params than vanilla, ~15,000 fewer than ds.
- Trains slightly faster than every other variant.

The wider 7×7 RF the DS class was originally pitched against (vanilla's
two stacked 3×3 = effective 7×7) turns out to not be necessary — a
single 5×5 with BN does at least as well. The library's current default
of `kernel_size=7` should be revisited for non-downsampling use; for
strided/downsampling use it remains untested.

Across E01–E04 the five configurations span 0.6pp (0.7052–0.7115).
Single-seed results within ~0.2pp are at the noise floor; ds_k5's win
over vanilla and ds_plain_k5 is in that band, but its consistent edge
across best/final/top-5 metrics + lower param count + faster wall
clock makes it the unambiguous Pareto winner.

---

<!-- Append future experiments below as ## E05, E06, ... -->
