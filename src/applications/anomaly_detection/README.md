# Patch-Reconstruction Anomaly Detection

A lightweight anomaly detector that reuses a trained
**`ConvNeXtPatchVAE`** to flag *poorly reconstructed patches* — regions the
model cannot faithfully reproduce. It scores every patch by its **reconstruction
error**: the model deterministically reconstructs the image and the per-patch
squared pixel error becomes the anomaly score.

## Key idea

The model deterministically reconstructs the input via
`sample_from(x, temperature=0.0)` (no sampling noise). The squared pixel error
between the input and the reconstruction is averaged over the channels and then
average-pooled over each `patch_size x patch_size` block, producing a
`(Hp, Wp)` anomaly map. **High value = poorly reconstructed = anomalous.**

This signal is **sampler-agnostic** — it works identically for `gaussian` and
`vmf` checkpoints, because it only touches the input/output pixels, never the
posterior parameters. It is also **sign-correct by construction**: larger error
always means more anomalous, with no monotonicity caveats.

A single per-patch reconstruction-error map is computed on every image:

| Grid (256px) | Patch | Signal | Use |
|--------------|-------|--------|-----|
| `ceil(H/patch) x ceil(W/patch)` | `patch_size` (e.g. 8px) | mean squared pixel error | per-patch localization |

For a 256px input with `patch_size = 8` the anomaly map is `32 x 32`. Scores are
MSE on `[0, 1]` pixels, so in-distribution values are small (≈ 0.001–0.05).

## Requirements

- The `dl_techniques` env (`.venv`), Keras 3 / TF 2.18.
- GUI: `streamlit` + `streamlit-webrtc` (live webcam) —
  `.venv/bin/pip install streamlit streamlit-webrtc`.
- A trained checkpoint, e.g.
  `results/convnext_patch_vae_ade20k+coco_custom_20260606_214647/best_model.keras`
  (vmf, `patch_size=8`, `img_size=256`, `latent_dim=32`, BCE).

The core `PatchReconstructionAnomalyDetector` has **no GUI dependency** and works
headless / programmatically; only `streamlit_app.py` imports streamlit.

## GUI (Streamlit)

```bash
CUDA_VISIBLE_DEVICES=1 \
ANOMALY_MODEL=results/convnext_patch_vae_ade20k+coco_custom_20260606_214647/best_model.keras \
  .venv/bin/streamlit run src/applications/anomaly_detection/streamlit_app.py \
  --server.address 127.0.0.1 --server.port 8501
```

Open `http://127.0.0.1:8501`. Two tabs:

- **Live (webcam)** — real-time webcam via `streamlit-webrtc`; each frame is one
  encode + deterministic decode, overlaid live with the reconstruction-error
  heatmap or anomaly mask. Click **Start** and allow camera access. Lower
  **Max side** for higher FPS.
- **Image** — upload a still image; shows original, the reconstruction-error
  overlay, anomaly mask, and the score JSON.

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
  - `absolute` — fixed reconstruction-error (MSE) cut-off, in `[0, 1]` pixel
    units (small values like `0.01`); use if you calibrate a value offline.

## Programmatic use

```python
from applications.anomaly_detection import PatchReconstructionAnomalyDetector

det = PatchReconstructionAnomalyDetector.from_pretrained(
    "results/convnext_patch_vae_ade20k+coco_custom_20260606_214647/best_model.keras")
x, orig_hw = det.preprocess("image.jpg", max_size=384)
amap = det.anomaly_maps(x, orig_hw=orig_hw)["anomaly"]   # (Hp, Wp)
mask, thr = det.anomaly_mask(amap, method="zscore", k=3.0)
print(det.score(amap, mask))                             # mean/max/p95 score, frac anomalous
```

## Notes & tuning

- Inputs are scaled to `[0, 1]` (BCE checkpoint) and kept at native resolution
  and aspect ratio — reflect-padded to a multiple of `patch_size` (the model is
  resolution-agnostic). The anomaly map is sized `ceil(H/patch) x ceil(W/patch)`;
  padded patches are cropped out of scoring.
- `zscore` is relative per image — not comparable across images. For
  cross-image comparison, calibrate an `absolute` (MSE) threshold on
  known-normal data.
- The decoder runs at inference: each frame is one encode + one deterministic
  decode. This is heavier than a single encode forward, so expect lower webcam
  FPS than a pure-encode detector.
- Set `MPLBACKEND=Agg` for any headless matplotlib use (avoids X11 crashes).
  The app does **not** use Gradio (its webcam streaming is broken in 6.x); the
  GUI is Streamlit + `streamlit-webrtc`.
