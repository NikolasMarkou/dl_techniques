# Patch-Entropy Anomaly Detection

A lightweight anomaly detector that reuses a trained
**`HierarchicalConvNeXtPatchVAE`** to flag *high-entropy patches* — regions the
model finds surprising / hard to compress. It runs the **encoder only** (the
decoder is never executed) and scores every patch by its **KL divergence**.

## Key idea

For each patch, the encoder produces a posterior `q(z|x) = N(mu, sigma^2)`. The
KL divergence from the prior measures how many nats the model must spend to
describe that patch beyond what the prior already predicts. **High KL = high
entropy = anomalous.**

Two scales are computed on every image:

| Level | Grid (128px) | Patch | Prior | Use |
|-------|--------------|-------|-------|-----|
| **L2** (primary) | 16×16 | 8px | learned conditional `p(z_l2\|z_l1)` | fine localization |
| **L1** | 4×4 | 32px | `N(0, I)` | coarse regions |

L2 uses the **conditional** KL `KL(q(z_l2|x) ‖ p(z_l2|z_l1))` — exactly the
objective the checkpoint (`learnable_l2_prior=True`) was trained to minimize, so
the scores reflect true surprise given global context. The formula mirrors
`model_hierarchical._compute_kl_l2_conditional` (same `[-10, +10]` log-var clip).

## Requirements

- The `dl_techniques` env (`.venv`), Keras 3 / TF 2.18.
- `gradio` (for the GUI): `\.venv/bin/pip install gradio`.
- A trained checkpoint, e.g.
  `results/hierarchical_convnext_patch_vae_ade20k+coco_large_20260528_205245/best_model.keras`.

The core `PatchEntropyAnomalyDetector` has **no gradio dependency** and works
headless / programmatically; only `app.py` imports gradio.

## GUI

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m applications.anomaly_detection.app \
    --model results/hierarchical_convnext_patch_vae_ade20k+coco_large_20260528_205245/best_model.keras
```

Open the printed `http://127.0.0.1:7860`. Upload an image, hit **Analyze**, then
move the threshold sliders — they re-render instantly (the KL maps are cached;
no re-encode, no decoder).

Options: `--host`, `--port`, `--share` (public link). On a headless box, either
use `--share` or SSH-forward the port: `ssh -L 7860:127.0.0.1:7860 host`.

Controls:
- **Process size**: square side fed to the model (multiple of `patch_size_l1`).
- **Mask / score level**: `l2` (fine) or `l1` (coarse) drives the mask + scores.
- **Threshold method**:
  - `zscore` — flag patches above `mean + k·std` (per-image, calibration-free; default).
  - `percentile` — flag the top `(100 − p)%` of patches.
  - `absolute` — fixed nats cut-off (use if you calibrate a value offline).

## Programmatic use

```python
from applications.anomaly_detection import PatchEntropyAnomalyDetector

det = PatchEntropyAnomalyDetector.from_pretrained(".../best_model.keras")
x = det.preprocess("photo.jpg")              # (1, 128, 128, 3) in [0, 1]
maps = det.kl_maps(x)                         # {"l1": (4,4), "l2": (16,16)}
mask, thr = det.anomaly_mask(maps["l2"], method="zscore", k=3.0)
scores = det.score(maps["l2"], mask)          # mean/max/p95 KL, frac anomalous
overlay = det.overlay(x[0], maps["l2"])       # uint8 (H, W, 3) heatmap overlay
```

## Notes & tuning

- Inputs are scaled to `[0, 1]` (BCE checkpoint); a non-square image is resized
  to a square `--process-size` (default = training size 128).
- `zscore` is relative per image — not comparable across images. For
  cross-image comparison, calibrate an `absolute` threshold on known-normal data.
- Lighter is faster: the decoder weights load into RAM but never run; inference
  is a single encoder forward (+ the small conditional-prior head for L2).
