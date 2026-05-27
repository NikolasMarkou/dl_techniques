# Train CliffordNetPatchVAEV2

Training script for the hierarchical Clifford-block VAE
(`dl_techniques.models.cliffordnet_patch_vae_v2.CliffordNetPatchVAEV2`).
Mirrors `train_convnext_patch_vae_v2.py` 1:1; the only differences are
the model factory, the hierarchical CLI flags (`--stage-dims`,
`--stage-depths`, `--stage-shifts`, `--downsample-kind`, `--cli-mode`,
`--ctx-mode`, ...) and the output-directory prefix
(`results/cliffordnet_patch_vae_v2_<exp_name>_<timestamp>/`).

## Quick start

```bash
# Smoke (CPU, tiny model, 2 epochs, synthetic data)
.venv/bin/python -m train.cliffordnet_patch_vae_v2.train_cliffordnet_patch_vae_v2 --smoke

# CIFAR-10 base preset with MAE + classification head
CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m \
    train.cliffordnet_patch_vae_v2.train_cliffordnet_patch_vae_v2 \
    --dataset cifar10 --variant base \
    --recon-loss-type bce \
    --mae-mask-ratio 0.5 --lambda-mae 1.0 \
    --use-classification-head --num-classes-cls 10 --lambda-cls 1.0 \
    --epochs 50

# Custom hierarchical config (override preset stages)
MPLBACKEND=Agg .venv/bin/python -m \
    train.cliffordnet_patch_vae_v2.train_cliffordnet_patch_vae_v2 \
    --dataset cifar10 --variant base \
    --stage-dims 64 128 --stage-depths 2 2 \
    --stage-shifts '[[1,2],[1,2,4]]' --downsample-kind blur
```

## Hierarchical CLI flags (cliffordnet-specific)

| Flag | Type | Default | Meaning |
|------|------|---------|---------|
| `--variant {tiny,base,large,xl}` | str | `base` | Preset (see model README). |
| `--stage-dims D1 D2 [D3 ...]` | int+ | preset | Channel width per stage. |
| `--stage-depths N1 N2 [N3 ...]` | int+ | preset | Block count per stage. |
| `--stage-shifts '[[1,2],...]'` | JSON | preset | Per-stage shift schedules. |
| `--downsample-kind {avg,max,blur,gaussian_dw,pixel_unshuffle,resnetd}` | str | `blur` | Pool inside `CliffordNetBlockDSv2`. |
| `--downsample-kernel-size N` | int | 7 | DW conv kernel inside transitions. |
| `--cli-mode {inner,wedge,full}` | str | `full` | Algebraic components per block. |
| `--ctx-mode {diff,abs}` | str | `diff` | Context stream mode. |
| `--no-global-context` | flag | off (=enabled) | Disable GAP branch in deepest stage. |
| `--layer-scale-init FLOAT` | float | 1e-5 | GGR LayerScale init. |
| `--block-drop-path-rate FLOAT` | float | 0.0 | Max DropPath rate (linear schedule). |
| `--latent-dim INT` | int | preset | Per-bottleneck-cell latent width. |

All other flags (datasets, optimizer, MAE/LPIPS/cls/seg weights,
beta-annealing, warmup, output) are identical to
`train.convnext_patch_vae_v2.train_convnext_patch_vae_v2` — see that
script's `--help` and README.

## Datasets

Same as v2: `cifar10`, `cifar100`, `ade20k`, `coco`. CIFAR uses the
in-memory Keras loader; ADE20K/COCO use a `patches_per_image`-crop
filesystem loader. Segmentation-mask loading is deferred on
ADE20K/COCO; only VAE/LPIPS/MAE-recon supervision is used on those.

## Outputs

Goes to repo-root `results/cliffordnet_patch_vae_v2_<exp>_<timestamp>/`:

- `final_model.keras` — saved + reload-checked (mu round-trip < 1e-4).
- `training_curves/` — PNG plots from `TrainingCurvesCallback`.
- `masked_recon/` — periodic visualization when `--mae-mask-ratio > 0`.
- `config.json` — full dataclass.
- Standard `create_callbacks` outputs (CSV log, best-checkpoint).
