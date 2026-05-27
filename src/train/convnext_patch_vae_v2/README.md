# Training: ConvNeXtPatchVAEV2

Multi-task pretraining pipeline for `dl_techniques.models.convnext_patch_vae_v2.ConvNeXtPatchVAEV2`.

> Plan: `plans/plan_2026-05-27_4a444b14/`.

---

## Quick start

### Smoke test (CPU, < 60 s, no GPU required)

```bash
.venv/bin/python -m train.convnext_patch_vae_v2.train_convnext_patch_vae_v2 --smoke
```

Tiny model, 2 epochs, 2 steps each, 16×16 synthetic data. Validates the
end-to-end pipeline (model build → fit → save → reload check on `mu`).

### CIFAR-10 baseline (VAE-only, V1-equivalent)

```bash
CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m \
    train.convnext_patch_vae_v2.train_convnext_patch_vae_v2 \
    --dataset cifar10 --variant base \
    --recon-loss-type bce --epochs 50
```

### CIFAR-10 with all-the-things (LPIPS + MAE + classification)

```bash
CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m \
    train.convnext_patch_vae_v2.train_convnext_patch_vae_v2 \
    --dataset cifar10 --variant base \
    --recon-loss-type bce --lambda-lpips 0.1 \
    --mae-mask-ratio 0.5 --lambda-mae 1.0 \
    --use-classification-head --num-classes-cls 10 --lambda-cls 1.0 \
    --epochs 50
```

`--num-classes-cls` auto-defaults to 10 / 100 for cifar10 / cifar100 when
`--use-classification-head` is set.

### ADE20K / COCO at 256×256 (VAE + LPIPS + MAE only)

```bash
CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m \
    train.convnext_patch_vae_v2.train_convnext_patch_vae_v2 \
    --dataset ade20k --image-size 256 --patch-size 8 \
    --variant xl --batch-size 16 --epochs 1 --steps-per-epoch 10 \
    --recon-loss-type bce --lambda-lpips 0.1 \
    --mae-mask-ratio 0.5 --lambda-mae 1.0
```

ADE20K seg-mask label loading is deferred (D-006). The seg head is
built / unit-tested but not wired to real ADE20K masks in iter-1.

---

## Training Menu (recipes ranked by intent)

| # | Goal | Dataset | Recipe | Cost |
|---|------|---------|--------|------|
| **R1** | V1 parity check | CIFAR-10 | `--variant base --recon-loss-type bce --epochs 50` | ~30 min RTX 4090 |
| **R2** | V1 + LPIPS | CIFAR-10 | R1 + `--lambda-lpips 0.1` | ~35 min |
| **R3** | V1 + MAE pretext | CIFAR-10 | R1 + `--mae-mask-ratio 0.5 --lambda-mae 1.0` | ~35 min |
| **R4** | Full multi-task | CIFAR-10 | R2 + R3 + `--use-classification-head --num-classes-cls 10` | ~40 min |
| **R5** | Scale-up validation | CIFAR-10 | R4 + `--variant xl --batch-size 128` | ~2 h |
| **R6** | 256×256 patch VAE | ADE20K | `--variant xl --image-size 256 --patch-size 8 --batch-size 16 --recon-loss-type bce --lambda-lpips 0.1 --mae-mask-ratio 0.5` | ~8-12 h |
| **R7** | Multi-dataset (256×256) | ADE20K + COCO | R6 + custom data composition (out of iter-1 scope) | — |

R1 is the falsification cheapest first — confirms V2 reaches V1's
recon quality without regressions. R4 is the headline recipe for
producing a multi-purpose pretrained backbone.

---

## CLI Reference

### V2-specific flags
| Flag | Default | Meaning |
|------|---------|---------|
| `--variant {tiny,base,large,xl}` | base | Preset (xl is new in V2: 256/8/8/64) |
| `--mae-mask-ratio FLOAT` | 0.0 | SimMIM mask ratio (0 disables) |
| `--lambda-mae FLOAT` | 1.0 | Multiplier on masked-patch recon |
| `--lambda-lpips FLOAT` | 0.0 | Multiplier on LPIPS perceptual (0 disables) |
| `--use-classification-head` | False | Enable attention-pool cls head |
| `--num-classes-cls INT` | 0 | Cls head output dim (auto from dataset) |
| `--lambda-cls FLOAT` | 1.0 | Multiplier on cls CE |
| `--use-segmentation-head` | False | Enable bilinear-upsample seg head |
| `--num-classes-seg INT` | 0 | Seg head class count |
| `--lambda-seg FLOAT` | 1.0 | Multiplier on seg CE |

### Inherited from V1 (same names + semantics)
`--dataset`, `--image-size`, `--patch-size`, `--epochs`, `--batch-size`,
`--learning-rate`, `--weight-decay`, `--patience`, `--gpu`,
`--beta-kl`, `--lambda-sigreg`, `--sigreg-knots`, `--sigreg-num-proj`,
`--recon-loss-type`, `--dropout`, `--gamma-clip`, `--warmup-epochs`,
`--beta-kl-start`, `--beta-anneal-epochs`, `--ade20k-dir`, `--coco-dir`,
`--steps-per-epoch`, `--no-augment`, `--no-color-augment`.

---

## Output

Each run creates `results/convnext_patch_vae_v2_<experiment>_<ts>/`:

- `config.json` — full training config snapshot
- `best_model.keras` — best val_loss checkpoint
- `final_model.keras` — model at end of training
- `training_log.csv` — per-epoch metrics
- `training_curves/loss.png`, `other.png` — auto-rendered curves
- `masked_recon/recon_epoch_NNNN.png` — visualizations (when MAE active)

The reload check at the end of training compares `model.encode(dummy).mu`
between the in-memory and reloaded models; exits 1 with an error if
`max|delta| >= 1e-4`.

---

## Known limitations (iter-1 scope)

1. **ADE20K seg-mask loader is not wired** (D-006). The seg head is
   built and unit-tested with synthetic labels; integration with real
   ADE20K palette decoding is a follow-up plan.
2. **No hierarchical V2 mode** (D-005). Single-scale only in iter-1.
3. **No `xxl` preset** (D-005). Validate `xl` first.
4. **No distillation / GAN / CLIP heads** (D-005). LPIPS + MAE + cls +
   seg cover the "recommended path" from the design discussion; more
   heads add hyperparameter-balancing cost.
5. **No real-data training was performed in iter-1.** The code ships
   ready-to-run, but no benchmark numbers exist yet. R1 (CIFAR-10
   baseline) is the cheapest first-validation run.
