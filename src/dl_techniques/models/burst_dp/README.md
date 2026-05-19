# BurstDP

Multi-view reference-conditioned vision model. Consumes one **reference image**
plus a **variable-size set of auxiliary views** (1-5 in practice; 0..`n_max`
supported) of the same scene under different viewpoints, noise, or
transformations, and produces two dense outputs about the reference:

1. **reconstruction** — clean RGB of the (possibly corrupted) reference
2. **segmentation** — per-pixel class logits

## Architecture

```
ref  (B, H, W, 3) ─┐
                    ├─► shared ViT encoder ─► ref tokens (B, T, D)
aux  (B, N, H, W, 3) ─►                       aux tokens (B, N, T, D)
                                              + aux_mask (B, N)
                          │
                          ▼
            stack of BurstFusionBlock × M
              self-attn(ref) + cross-attn(ref → flat(aux), masked) + FFN
                          │
                          ▼
            reshape tokens to (B, h, w, D)
                          │
        ┌─────────────────┴─────────────────┐
        ▼                                   ▼
     recon head                          seg head
    (DPT, sigmoid)                    (DPT, logits)
```

## Key properties

- **Permutation-invariant over the aux set** by construction: cross-attention
  over flattened key/value tokens with a boolean mask carries no positional
  bias across views.
- **Variable N at runtime**: mask zeroes padding aux slots; per-sample
  cross-attention output is gated to zero when no aux view is present (N=0),
  so the model degrades gracefully to single-image inference.
- **Reference is privileged**: the cross-attention is asymmetric (ref = query,
  aux = key/value) — the task heads only ever read fused reference tokens.

## Usage

```python
from dl_techniques.models.burst_dp import create_burst_dp

model = create_burst_dp(preset="burst_dp_small", image_size=256, n_max=5)

ref = keras.random.uniform((2, 256, 256, 3))
aux = keras.random.uniform((2, 5, 256, 256, 3))
aux_mask = keras.ops.convert_to_tensor([[1, 1, 1, 0, 0],
                                        [1, 1, 0, 0, 0]], dtype="float32")
out = model({"ref": ref, "aux": aux, "aux_mask": aux_mask})
out["recon"].shape         # (2, 256, 256, 3)
out["segmentation"].shape  # (2, 256, 256, 81)
```

## Presets

| Preset | Encoder | Fusion blocks | Approx params |
|--------|---------|---------------|---------------|
| `burst_dp_pico`  | ViT-Pico  | 2 | ~6 M  |
| `burst_dp_tiny`  | ViT-Tiny  | 3 | ~12 M |
| `burst_dp_small` | ViT-Small | 4 | ~50 M |
| `burst_dp_base`  | ViT-Base  | 6 | ~140 M |

## Training

See `src/train/burst_dp/train_burst_dp.py`. Data is COCO 2017 with synthetic
aux-view generation.

## Files

- `config.py` — `BurstDPConfig`, presets, factory `get_preset`
- `fusion.py` — `BurstFusionBlock` (self-attn + masked cross-attn + FFN)
- `heads.py` — DPT-style reconstruction / segmentation heads
- `model.py` — `BurstDP` model + `create_burst_dp` factory
