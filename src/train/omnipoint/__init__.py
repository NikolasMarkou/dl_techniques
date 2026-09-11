"""Training pipeline for OmniPoint (camera-agnostic monocular metric point-cloud model).

See `src/dl_techniques/models/vision/omnipoint/` for the model itself and
`src/train/omnipoint/README.md` (added in a later plan step) for the scope caveat this
package's data loaders restate individually: training here is pinhole-only, on locally
available KITTI depth + MegaDepth data, using DERIVED (not calibrated) camera intrinsics
(see `kitti_depth.py`'s module docstring).
"""
