# OmniPoint

`OmniPoint` predicts a per-pixel metric 3D point cloud `P = s* * d_hat * r_hat` from a single
RGB image: `r_hat` is a unit ray direction, `d_hat` a positive radial distance, and `s*` a single
global metric scale factor. `P = d_hat * r_hat` is computed by exactly one shared combinator
(`layers/geometric/ray_point_combinator.py::RayPointCombinator`) regardless of the input camera
model — camera-specific geometry only ever enters through ray-map conditioning (Step 5), never
through architecture branching in this package. That is the sense in which the model is
**camera-agnostic by construction**: the same weights, the same `call()` signature, produce a
valid unit-ray/positive-distance output for a pinhole, fisheye, or equirectangular input alike,
because nothing in `model.py` or `heads.py` ever inspects which camera model produced the image.

As of this step (Step 4 of the plan under `plans/plan-2026-09-11T050223-1b47bcf6/`), the model is
encoder + three heads with **no conditioning input**: a `ViT` backbone
(`ViT.from_variant("vit_large", patch_size=14, include_top=False, pooling=None)`, see
`decisions.md` D-009 for why `ViT` over `DINOv2VisionTransformer`) feeds a shared spatial feature
map to `RayDistanceHead` and `MaskHead` (each one `DPTDecoder` instance) and its own CLS token to
`MetricScaleHead` (a small pooled MLP). Step 5 adds the optional intrinsics/sparse-depth
conditioning path (`conditioning.py`) that lets the model use, but not require, known camera
geometry or a sparse depth prior.

**Camera-agnosticism is a code-path property, not a trained/validated generalization claim.**
This model is trained (Steps 7-9) on pinhole imagery only, using a documented derived-intrinsics
approximation (`decisions.md` D-006) — no fisheye or equirectangular/panoramic training data exists
locally (`decisions.md` D-001). The fisheye and equirectangular ray-map code paths in
`utils/camera_models.py` are exercised by direct computation (synthetic reprojection, forward-pass
finiteness/unit-norm/positive-distance checks — Step 6) and are unit-tested, but the model's
behavior on genuine fisheye or panoramic photographs has never been measured and is not claimed
here. Read this as: "the architecture does not branch on camera model, and the non-pinhole code
paths run correctly," not "the model generalizes to non-pinhole cameras it was never trained on."
