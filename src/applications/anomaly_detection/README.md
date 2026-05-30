# Patch-Entropy Anomaly Detection

A lightweight anomaly detector that reuses a trained
**`ConvNeXtPatchVAE`** to flag *high-entropy patches* — regions the model finds
surprising / hard to compress. It runs the **encoder only** (the decoder is
never executed) and scores every patch by its **KL divergence**.

## Key idea

For each patch, the encoder produces a posterior `q(z|x) = N(mu, sigma^2)`. The
KL divergence from the prior measures how many nats the model must spend to
describe that patch beyond what the prior already predicts. **High KL = high
entropy = anomalous.**

A single per-patch KL map is computed on every image:

| Grid (128px) | Patch | Prior | Use |
|--------------|-------|-------|-----|
| `ceil(H/patch) x ceil(W/patch)` | `patch_size` (e.g. 4px) | `N(0, I)` | per-patch localization |

The score is the standard KL `KL(q(z|x) ‖ N(0, I))` summed over the latent
dimension (same `[-10, +10]` log-var clip as training).

> **Note**: a prior version used the hierarchical model's *conditional* KL
> `KL(q(z_l2|x) ‖ p(z_l2|z_l1))` as the primary signal. The hierarchical variant
> was removed; single-scale KL-vs-`N(0,I)` is an off-objective, weaker signal
> (it measures absolute encoding cost, not surprise-given-context).

## Requirements

- The `dl_techniques` env (`.venv`), Keras 3 / TF 2.18.
- GUI: `streamlit` + `streamlit-webrtc` (live webcam) —
  `.venv/bin/pip install streamlit streamlit-webrtc`.
- A trained single-scale checkpoint, e.g.
  `results/convnext_patch_vae_ade20k+coco_large_20260527_130515/best_model.keras`.

The core `PatchEntropyAnomalyDetector` has **no GUI dependency** and works
headless / programmatically; only `streamlit_app.py` imports streamlit.

## GUI (Streamlit)

```bash
CUDA_VISIBLE_DEVICES=1 \
ANOMALY_MODEL=results/convnext_patch_vae_ade20k+coco_large_20260527_130515/best_model.keras \
  .venv/bin/streamlit run src/applications/anomaly_detection/streamlit_app.py \
  --server.address 127.0.0.1 --server.port 8501
```

Open `http://127.0.0.1:8501`. Two tabs:

- **Live (webcam)** — real-time webcam via `streamlit-webrtc`; each frame is one
  encoder forward, overlaid live with the KL heatmap or anomaly mask. Click
  **Start** and allow camera access. Lower **Max side** for higher FPS.
- **Image** — upload a still image; shows original, the KL overlay, anomaly
  mask, and the score JSON.

The checkpoint path comes from the `ANOMALY_MODEL` env var (or the sidebar
**Model checkpoint** box). On a headless box, SSH-forward the port:
`ssh -L 8501:127.0.0.1:8501 host` (localhost is a secure context, so the browser
grants camera access). The webcam runs in the **browser**, so the camera must be
on the machine running the browser.

Sidebar controls:
- **Max side px**: caps the longer side (aspect-preserving) to bound GPU memory
  and raise FPS; `0` = native. The image is never squashed to a square — it keeps
  its aspect ratio and is reflect-padded to a multiple of `patch_size`.
- **Threshold method**:
  - `zscore` — flag patches above `mean + k·std` (per-image, calibration-free; default).
  - `percentile` — flag the top `(100 − p)%` of patches.
  - `absolute` — fixed nats cut-off (use if you calibrate a value offline).

## Programmatic use

```python
from applications.anomaly_detection import PatchEntropyAnomalyDetector

det = PatchEntropyAnomalyDetector.from_pretrained(".../best_model.keras")
x, (h, w) = det.preprocess("photo.jpg")       # native res, padded to /patch
kl = det.kl_maps(x, orig_hw=(h, w))["kl"]     # (ceil(h/patch), ceil(w/patch))
mask, thr = det.anomaly_mask(kl, method="zscore", k=3.0)
scores = det.score(kl, mask)                  # mean/max/p95 KL, frac anomalous
overlay = det.overlay(x[0][:h, :w], kl)       # uint8 (h, w, 3) heatmap overlay
```

## Notes & tuning

- Inputs are scaled to `[0, 1]` (BCE checkpoint) and kept at native resolution
  and aspect ratio — reflect-padded to a multiple of `patch_size` (the model is
  resolution-agnostic). The KL map is sized `ceil(H/patch) x ceil(W/patch)`;
  padded patches are cropped out of scoring.
- `zscore` is relative per image — not comparable across images. For
  cross-image comparison, calibrate an `absolute` threshold on known-normal data.
- Lighter is faster: the decoder weights load into RAM but never run; inference
  is a single encoder forward.
