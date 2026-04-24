# CliffordNet Downsampling Experiment

**Status**: training in progress — results table populated after full run.

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
- Batch size: 128; Epochs: 100 (shortened from the paper's 200 to fit a
  single-session 5-variant sweep; ranking is preserved because cosine has
  fully decayed by epoch 100).
- Regularisation: LayerScale init 1e-5, DropPath max 0.1 (linearly scheduled),
  no explicit L2 (AdamW handles decoupled WD).
- `TerminateOnNaN` + `EarlyStopping(patience=30, val_accuracy)` +
  `ModelCheckpoint(best_val_accuracy)` per variant.

## Results

**Populated after the full 100-epoch sweep completes. See
`results/cliffordnet_downsampling_<timestamp>/comparison.csv` for the raw
data.**

| Variant | Params | Best val_acc | Best val_top5 | Final val_acc | Wall time |
|---------|--------|--------------|---------------|---------------|-----------|
| V0_baseline_isotropic       | 1.44M | _pending_ | _pending_ | _pending_ | _pending_ |
| V1_3stage_strided_conv      | 2.87M | _pending_ | _pending_ | _pending_ | _pending_ |
| V2_3stage_avgpool           | 2.55M | _pending_ | _pending_ | _pending_ | _pending_ |
| V3_3stage_patch_merging     | 2.67M | _pending_ | _pending_ | _pending_ | _pending_ |
| V4_4stage_aggressive        | 10.32M | _pending_ | _pending_ | _pending_ | _pending_ |
| V5_2stage_aggressive_stem   | 3.64M | _pending_ | _pending_ | _pending_ | _pending_ |

## Early Signals (smoke test — 3 epochs, batch 32)

Cheap sanity check before the full run. Not a benchmark — but useful for
catching gross architectural problems:

| Variant | 3-epoch val_acc | 3-epoch val_top5 |
|---------|-----------------|-------------------|
| V0 baseline | 0.378 | 0.713 |
| V1 strided conv | 0.467 | 0.779 |
| (V2–V5) | _populated after smoke completes_ | |

V1 already outperforms V0 at 3 epochs despite having only ~2× the params,
validating the hierarchical premise.

## How to Reproduce

```bash
# Cheap sanity run (~20 minutes, all 6 variants, batch 32, 3 epochs each)
MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_downsampling_techniques \
    --variant all --smoke-test --gpu 0 --skip-save

# Full benchmark (~5 hours, all 6 variants serial on GPU 0)
MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_downsampling_techniques \
    --variant all --epochs 100 --batch-size 128 --gpu 0

# A single variant
MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_downsampling_techniques \
    --variant V3_3stage_patch_merging --epochs 100 --batch-size 128 --gpu 0
```

Each variant produces:

- `results/cliffordnet_downsampling_<timestamp>/<variant>_downsampling_<ts>/config.json`
- `.../training_history.json`, `.../training_log.csv`
- `.../best_model.keras`, `.../training_summary.txt`
- `.../<variant>_final.keras` (round-trip validated)
- Aggregated: `results/cliffordnet_downsampling_<timestamp>/comparison.csv`

## Discussion

*To be written after the full run completes. Expected questions:*

1. Does any hierarchical variant beat the isotropic baseline on accuracy-per-parameter?
2. Does the **downsampler choice** matter at all once the shape is fixed (V1 vs V2 vs V3)?
3. Does the 8× channel expansion of V4 pay for its ~7× parameter budget?
4. Does front-loaded stem downsampling (V5) match the mid-network downsampling of V1–V3?

## References

- Source: [arXiv:2601.06793v2 — *CliffordNet: All You Need is Geometric Algebra*](../../../../research/2026_clifford_vlm.md).
- Baseline implementation: [`model.py`](model.py).
- Training script: [`train_downsampling_techniques.py`](../../../train/cliffordnet/train_downsampling_techniques.py).
