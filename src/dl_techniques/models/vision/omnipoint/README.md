# OmniPoint

`OmniPoint` predicts a per-pixel metric 3D point cloud `P = s* * d_hat * r_hat` from a single
RGB image: `r_hat` is a unit ray direction, `d_hat` a positive radial distance, and `s*` a
single global metric scale factor. `P = d_hat * r_hat` is computed by exactly one shared
combinator (`layers/geometric/ray_point_combinator.py::RayPointCombinator`) regardless of the
input camera model — camera-specific geometry only ever enters through ray-map conditioning
(`conditioning.py`), never through architecture branching in this package. That is the sense in
which the model is **camera-agnostic by construction**: the same weights, the same `call()`
signature, produce a valid unit-ray/positive-distance output for a pinhole, fisheye, or
equirectangular input alike, because nothing in `model.py` or `heads.py` ever inspects which
camera model produced the image.

## READ THIS BEFORE CITING THIS PACKAGE

**This package's fisheye and equirectangular code paths are proven only by synthetic
reprojection tests, not trained on real fisheye or panoramic data.** No fisheye or
equirectangular imagery exists anywhere locally (`decisions.md` D-001) — the only evidence the
non-pinhole code paths work is `tests/test_models/test_omnipoint/test_cross_camera_synthetic.py`
(Step 6) and the staged dataset from
`src/train/omnipoint/generate_synthetic_camera_eval_set.py` (Step 10), both of which reproject an
existing **pinhole** RGB+depth pair through the fisheye/equirectangular ray-map functions in
`utils/camera_models.py` and check that the model's output on the reprojected image is finite,
unit-norm and positive-distance. Neither is a trained result and neither uses a genuine
fisheye/panorama photograph. **Do not cite this package as evidence for the OmniPoint paper's
cross-camera zero-shot generalization claim** — that claim is about weight generalization to
unseen real camera geometries, and this repository has never trained on, or evaluated against,
any. What this package DOES demonstrate is a narrower, code-path-level claim: the architecture
does not branch on camera model, and the non-pinhole ray-map/inverse-projection math runs
correctly and produces well-formed output. See Problem Statement invariant 5 in
`plans/plan-2026-09-11T050223-1b47bcf6/plan.md` for the exact wording this caveat is restating.

## Architecture

```
Input [B, H, W, 3] (+ optional intrinsics ray map, + optional sparse depth)
      |
      v
+--------------------------------+
| encoder: ViT (patch_size,      |
| include_top=False,             |
| pooling=None)                  |
+---------------+----------------+
                |  [B, N+1, D] (CLS-prefixed patch sequence)
    +-----------+-----------+
    |                       |
    v                       v
cls_token = seq[:, 0]   _features_to_spatial:
    |                   drop CLS, reshape -> [B, h, w, D]
    |                       |
    v                 +-----+-----+
MetricScaleHead        |           |
(pooled MLP)           v           v
    |            RayDistanceHead  MaskHead
    |            (DPTDecoder x4   (DPTDecoder x1
    |             + combinator)    logits)
    v                  |           |
  scale        (ray, distance,   mask_logit
                 point)
                    |
                    v
   Output: (ray, distance, point, mask_logit, scale)
```

- **Backbone**: `ViT.from_variant("vit_large"|"vit_base", patch_size=14, include_top=False,
  pooling=None)` (`decisions.md` D-009 — chosen over `DINOv2VisionTransformer` because this
  repo's `DPTDecoder` is single-scale, so DINOv2's multi-scale/register-token apparatus buys
  nothing here, and neither backbone carries pretrained weights regardless).
- **Heads** (`heads.py`): `RayDistanceHead` and `MaskHead` are each one `DPTDecoder` instance
  (head-specific `output_channels`/`output_activation`) feeding `layers/geometric/
  ray_point_combinator.py::RayPointCombinator` for the ray+distance head (L2-normalize `r_hat`,
  positive-activation `d_hat`, `P = d_hat * r_hat`); `MetricScaleHead` is a small pooled MLP over
  the ViT's own CLS token.
- **Loss** (`losses/omnipoint_losses.py`): `OmniPointCombinedLoss` decouples ray direction,
  point distance (projected along the GT ray, never the predicted ray), online metric scale
  alignment, mask BCE, and simplified normal/local-consistency terms into separate, independently
  testable terms — see Success Criterion 5 in the plan.
- **Optional conditioning** (`conditioning.py`, `enable_conditioning=True`): an intrinsics ray
  map and/or a densified sparse-depth prior fuse as extra input channels before the (unmodified)
  ViT encoder, and two per-sample `StateIndicatorEmbedding` vectors (present/absent, one per
  modality) fuse additively onto the encoder's own output token sequence. `enable_conditioning=
  False` (the default) is byte-for-byte identical to the unconditioned model (`decisions.md`
  D-013).

## Construction

```python
from dl_techniques.models.vision.omnipoint import create_omnipoint, OmniPoint, MODEL_VARIANTS

model = create_omnipoint("omnipoint_base", image_shape=(224, 224, 3))
# or, with the optional conditioning path enabled:
model = OmniPoint.from_variant(
    "omnipoint_large", image_shape=(224, 224, 3), enable_conditioning=True,
)
```

`MODEL_VARIANTS` keys: `omnipoint_base` (`vit_scale="base"`), `omnipoint_large`
(`vit_scale="large"`) — each maps to a `ViT` scale config plus this package's own decoder/metric
head sizing. `OmniPoint(..., pretrained=True)` raises `NotImplementedError`: no pretrained
weights exist for the backbone or for this model as a whole, anywhere in this repository
(`decisions.md` D-001).

## Known limitations

- **Output resolution**: both dense heads (`ray`, `distance`, `point`, `mask_logit`) emit at the
  encoder's native patch-grid resolution `(H // patch_size, W // patch_size)`, not full pixel
  resolution — `DPTDecoder` requires a power-of-2 `upsample_factor` and `patch_size=14` (the
  paper's ViT-L/14 convention) is not a power of 2 (`decisions.md` D-011). A caller wanting
  full-resolution output must add an external upsample/interpolation step; this package does not
  provide one.
- **Derived, not calibrated, intrinsics in training**: the trainer under `src/train/omnipoint/`
  derives a per-image pinhole `K` from an assumed field-of-view rather than a stored camera
  calibration, because none is available locally for KITTI or MegaDepth (`decisions.md` D-006).
  Every ray/distance GT value the model is trained against inherits that approximation's bias.
  See `src/train/omnipoint/README.md` for the full statement.
- **Simplified normal/local-consistency losses**: `NormalConsistencyLoss` uses a finite-difference
  cross-product surface-normal approximation and `LocalConsistencyLoss` a discrete-Laplacian
  smoothness term, in place of the paper's PCA/eigendecomposition-based normal estimator and its
  own (unspecified) local-consistency mechanism — a deliberate, documented simplification for
  compute cost and unit-testability (`decisions.md` D-004).
