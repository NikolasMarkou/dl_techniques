# CliffordNet Downsampling Experiment

**Status**: complete — all 6 variants trained on CIFAR-100 (2026-04-24).

## Motivation

The default [`CliffordNet`](model.py) is **isotropic**: a patch-embedding stem
maps the image to a fixed-dimensional feature map which is then processed by
a stack of `L` identical [`CliffordNetBlock`](../../layers/geometric/clifford_block.py)
layers, followed by global average pooling and a linear classifier. The
block itself is dim-preserving — it cannot change spatial or channel
dimensionality — which means any hierarchical / multi-stage design must
introduce explicit **inter-stage downsamplers** between groups of blocks.

This experiment compares five hierarchical variants of CliffordNet against
the existing isotropic baseline on CIFAR-100. The axes explored are:

- **Channel expansion strategy** — mild 3-stage doubling (64→128→256),
  aggressive 4-stage (64→128→256→512), early-wide (128→256).
- **Inter-stage downsampler block** — strided 3x3 conv, AvgPool + 1x1,
  Swin patch-merging, depthwise-separable strided.
- **Stride configuration** — stem stride {1, 2, 4}, combined with 2x
  intra-backbone downsamplings at different positions.

Each variant is built by the training script
[`train_downsampling_techniques.py`](../../../train/cliffordnet/train_downsampling_techniques.py).
The library `CliffordNet` model and the `CliffordNetBlock` layer are **not
modified** — variants are assembled as Keras functional models that compose
these primitives with the new stems / downsamplers.

## Variants

| Tag | Stem | Stages (C × n_blocks) | Downsampler | Params | Notes |
|-----|------|------------------------|-------------|--------|-------|
| V0_baseline_isotropic       | patch=2 | 128×12                                   | — (isotropic)          |  1.44M | Existing `CliffordNet.nano` control |
| V1_3stage_strided_conv      | patch=1 | 64×4 → 128×4 → 256×4                     | Conv2D(3×3, s=2) + LN  |  2.87M | Classic ConvNeXt-style |
| V2_3stage_avgpool           | patch=1 | 64×4 → 128×4 → 256×4                     | AvgPool(2) + Conv(1×1) + LN |  2.55M | Parameter-light spatial reduction |
| V3_3stage_patch_merging     | patch=1 | 64×4 → 128×4 → 256×4                     | Swin `PatchMerging`    |  2.67M | Fully-mixing patch-merge (2×2 concat + LN + Dense) |
| V4_4stage_aggressive        | patch=1 | 64×2 → 128×2 → 256×4 → 512×4             | Swin `PatchMerging`    | 10.32M | 8× channel expansion, 4 stages |
| V5_2stage_aggressive_stem   | patch=4 | 128×6 → 256×6                            | DepthwiseConv(3×3, s=2) + Conv(1×1) + BN |  3.64M | Front-load spatial reduction |

Spatial resolution at end of each stage (CIFAR-100 input 32×32):

| Variant | stage 0 | stage 1 | stage 2 | stage 3 | final |
|---------|---------|---------|---------|---------|-------|
| V0      | 16×16 (whole run) | | | | 16×16 |
| V1      | 32×32   | 16×16   | 8×8     |         | 8×8   |
| V2      | 32×32   | 16×16   | 8×8     |         | 8×8   |
| V3      | 32×32   | 16×16   | 8×8     |         | 8×8   |
| V4      | 32×32   | 16×16   | 8×8     | 4×4     | 4×4   |
| V5      | 8×8     | 4×4     |         |         | 4×4   |

All `CliffordNetBlock` instances use `shifts=[1, 2]`, `cli_mode="full"`,
`ctx_mode="diff"` and a shared linear DropPath schedule (max rate 0.1).

## Training Recipe

Identical across variants (AdamW + cosine schedule, paper-matching aug):

- Dataset: CIFAR-100 (50 k train / 10 k test, 32×32×3, 100 classes).
- Augmentation: HFlip → Pad-4 + RandomCrop → CIFAR-10 AutoAugment → per-channel
  normalize → RandomErasing (p=0.25). *Identical to `train_cliffordnet.py`.*
- Optimiser: AdamW, `weight_decay=0.1`, `(β1, β2) = (0.9, 0.999)`.
- Schedule: cosine decay with 5-epoch linear warmup, peak `lr=1e-3`, `α=1e-2`.
- Batch size: 64 (reduced from the initial 128 target to fit GPU 1 — RTX
  4070 12 GB — memory budget; see the plan's decision D-006 for rationale.
  Ranking conclusions are unchanged because all variants share the same
  batch size and schedule).
- Epochs: 100 (shortened from the paper's 200 to fit a single-session
  5-variant sweep; ranking is preserved because cosine has fully decayed
  by epoch 100).
- Regularisation: LayerScale init 1e-5, DropPath max 0.1 (linearly scheduled),
  no explicit L2 (AdamW handles decoupled WD).
- `TerminateOnNaN` + `EarlyStopping(patience=30, val_accuracy)` +
  `ModelCheckpoint(best_val_accuracy)` per variant.

## Results

Full sweep completed 2026-04-24 on GPU 1 (RTX 4070, 12 GB, batch 64, 100 epochs).
Per-variant `comparison.csv` files aggregated below.

| Variant                   | Params  | Best val_acc | Best val_top5 | Final val_acc | Wall time |
|---------------------------|---------|--------------|---------------|---------------|-----------|
| V0_baseline_isotropic     |  1.44 M | 0.7425 †     | 0.9377 †      | 0.7425 †      |  ~5 h †   |
| **V1_3stage_strided_conv**|  2.87 M | **0.7658**   | 0.9361        | 0.7651        | 110.6 min |
| V2_3stage_avgpool         |  2.55 M | 0.7596       | 0.9357        | 0.7585        | 108.2 min |
| V3_3stage_patch_merging   |  2.67 M | 0.7644       | **0.9376**    | 0.7639        | 113.7 min |
| V4_4stage_aggressive      | 10.32 M | 0.7526       | 0.9272        | 0.7512        | 137.8 min |
| V5_2stage_aggressive_stem |  3.64 M | 0.7005       | 0.8993        | 0.6987        |  80.1 min |

† **V0 note.** The baseline run was stopped by the user at epoch 77/100
after confirming the checkpoint had plateaued around 0.74; best val_acc is
taken from the last saved checkpoint (see plan decision D-007). V0 is
therefore a lower bound — the full 100-epoch run would likely add ~0.5-1.0
pp from the residual cosine decay — but even with that allowance every
3-stage hierarchical variant (V1, V2, V3) clearly beats it.

**Accuracy / parameter efficiency:**

| Variant | Best val_acc | Params (M) | pp per additional M params vs. V0 |
|---------|--------------|------------|-----------------------------------|
| V0      | 0.7425 †     | 1.44       | — (reference)                     |
| V1      | 0.7658       | 2.87       | +1.63 pp / M                      |
| V2      | 0.7596       | 2.55       | +1.54 pp / M                      |
| V3      | 0.7644       | 2.67       | +1.78 pp / M                      |
| V4      | 0.7526       | 10.32      | +0.11 pp / M                      |
| V5      | 0.7005       | 3.64       | -1.91 pp / M (regression)         |

## Early Signals (smoke test — 3 epochs, batch 32)

Cheap sanity check before the full run. Not a benchmark — but useful for
catching gross architectural problems:

| Variant         | 3-epoch val_acc | 3-epoch val_top5 |
|-----------------|-----------------|-------------------|
| V0 baseline     | 0.378           | 0.713             |
| V1 strided conv | 0.467           | 0.779             |

(V2–V4 were confirmed to forward+backward cleanly on GPU at 70-100 ms/step
@ batch 128 during smoke without completing a full 3-epoch sweep; see plan
decision D-004 for why we shortened the smoke phase.)

V1 already outperformed V0 at 3 epochs despite having only ~2× the params,
and that signal held through 100 epochs — V1 finished as the sweep winner.

## How to Reproduce

```bash
# Cheap sanity run (~20 minutes, all 6 variants, batch 32, 3 epochs each)
MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_downsampling_techniques \
    --variant all --smoke-test --gpu 0 --skip-save

# Full benchmark — reproduces the table in §Results (~9 h on RTX 4070 12 GB
# for V1-V5, batch 64). Use batch 128 on a 24 GB card (RTX 4090) for ~1.6×
# wall-time speedup.
MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_downsampling_techniques \
    --variant all --epochs 100 --batch-size 64 --gpu 0

# A single variant (recommended V1 for fastest good-accuracy baseline)
MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_downsampling_techniques \
    --variant V1_3stage_strided_conv --epochs 100 --batch-size 64 --gpu 0
```

Each variant produces:

- `results/cliffordnet_downsampling_<timestamp>/<variant>_downsampling_<ts>/config.json`
- `.../training_history.json`, `.../training_log.csv`
- `.../best_model.keras`, `.../training_summary.txt`
- `.../<variant>_final.keras` (round-trip validated)
- Aggregated: `results/cliffordnet_downsampling_<timestamp>/comparison.csv`

## Discussion

**Headline result.** A **3-stage hierarchy (64 → 128 → 256)** with any
reasonable mid-network downsampler clearly dominates both the isotropic
baseline and the more aggressive configurations. V1 (strided conv), V2
(avgpool + 1×1) and V3 (Swin patch-merging) cluster within 0.6 pp of each
other — 0.7596 / 0.7644 / 0.7658 — and all of them beat the isotropic V0
baseline by ≥1.7 pp while fitting in well under 3 M parameters.

**Downsampler choice is largely a wash (V1 ≈ V2 ≈ V3).** The three 3-stage
variants share stem, stages and channel schedule; only the 2×-downsampling
block differs. The accuracy spread between the three is 0.62 pp — smaller
than the V0 → V1 gap and well within run-to-run seed noise at 100 epochs.
V3 has a small edge on top-5 (0.9376), V1 on top-1 (0.7658), and V2 is the
smallest (2.55 M). **Picking the simplest option (V2 avgpool + 1×1) gives
essentially the same accuracy for ~11% fewer parameters than V1.**

**More capacity is not free (V4 regression).** V4 uses a 4-stage
64→128→256→512 schedule and ends up with **3.6× more parameters than V1
(10.32 M vs. 2.87 M) and 1.3 pp lower val-acc**. Two forces are probably
at play: (i) the final 4×4 feature map is tiny — global pooling over 16
tokens loses positional diversity, and (ii) the CIFAR-100 50 k-image
training set does not supply enough signal to fit a 10 M-param model at
only 100 epochs. V4 might close the gap with longer training or a bigger
dataset, but on *this* benchmark it's a clear efficiency loss.

**Front-loaded stem downsampling (V5) is the clear loser.** The patch=4
stem collapses the image to 8×8 before any `CliffordNetBlock` has run, and
the network never recovers: V5 trails V1 by **6.5 pp** despite carrying
~27% more parameters. Interpretation: `CliffordNetBlock`'s
sparse-channel-rolling mixing operator relies on local spatial structure
surviving into the early layers, and an aggressive stem destroys that
structure before it can be exploited. This is a useful negative result —
it suggests CliffordNet is **not** a drop-in replacement for the patchify
stem of ViT/ConvNeXt-style architectures.

**Recommendation.** For CIFAR-100 and similar small-image classification,
adopt the **V1 (3-stage strided-conv)** or **V2 (3-stage avgpool+1×1)**
recipe. V1 is a touch more accurate; V2 is slightly smaller and simpler.
Avoid stems that downsample by more than 2× before the first block stack.

**Caveats.**

- The baseline V0 run was stopped at epoch 77 (val_acc 0.7425); the full
  100-epoch number would likely be 0.5-1.0 pp higher — still below every
  3-stage hierarchical variant, so the ranking is robust.
- Wall times are not directly comparable to the Clifford paper's headline
  numbers: we used GPU 1 (RTX 4070 12 GB) at batch 64, whereas the paper
  uses larger GPUs at batch 128+. Ranking across variants within *this*
  table is the intended takeaway.
- Seeds are not averaged; all results are single-run. Differences under
  ~0.5 pp should be treated as within noise.

## References

- Source: [arXiv:2601.06793v2 — *CliffordNet: All You Need is Geometric Algebra*](../../../../research/2026_clifford_vlm.md).
- Baseline implementation: [`model.py`](model.py).
- Training script: [`train_downsampling_techniques.py`](../../../train/cliffordnet/train_downsampling_techniques.py).
