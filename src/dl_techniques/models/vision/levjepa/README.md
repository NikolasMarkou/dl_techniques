# LeVJEPA

A Keras 3 port of the LeVJEPA Vision Transformer encoder -- a shared ViT-style
backbone for video (tubelet patches) or images (2D patches), with either a
frozen 3D sin-cos positional table or a 3-axis (frame/height/width) rotary
position embedding, and an optional block-causal attention mask (bidirectional
within a frame, causal across frames) for autoregressive-over-time
pretraining.

## What's here (this step)

- `blocks.py::LeVJEPABlock` -- pre-norm self-attention + MLP block, no
  `LayerScale` (a faithful port of the reference's plain-residual `Block`).
- `encoder.py::LeVJEPAEncoder` -- the full ViT-style encoder: patch embed
  (video or image dispatch), CLS token, sincos-or-RoPE position handling,
  train-time token dropping, optional block-causal masking, block stack,
  final norm.
- `model.py` -- `SCALE_CONFIGS` (`vit_tiny` .. `vit_gigantic`),
  `MODEL_VARIANTS`, `from_variant(...)`, `create_levjepa(...)`.
- `masking.py` -- `build_block_causal_mask`, `random_token_drop` (Step 3).
- `projector.py::LeVJEPAProjector` -- the SIGReg projection head
  (`Dense -> BatchNorm -> GELU -> Dense`); no reshape-around
  `BatchNormalization`, see `decisions.md` D-014.
- `dl_techniques/callbacks/ema_shadow_callback.py::EMAShadowCallback` --
  checkpoint-only EMA-shadow driver (lives in `callbacks/`, not this
  package -- see `decisions.md` D-006); reuses `teacher_ema.py`'s decay
  schedules.
- `training.py::LeVJEPATrainingModel` -- the multiview training wrapper
  (Step 6): shared encoder run on 1 global + N local views, `LeVJEPAProjector`
  head, `SIGRegLayer(normalize_by_n=True)`, both loss terms added via
  `self.add_loss(...)` inside `call()` (no custom `train_step`), plus
  `update_ema_shadow(decay)` for `EMAShadowCallback`'s duck-typed contract.
  See `decisions.md` D-017 (no ImageNet mean/std normalization -- data
  already arrives as bounded float32), D-018 (round-trip via nested encoder
  serialization), D-020 (EMA shadow seeded lazily on first
  `update_ema_shadow` call, not at `build()` time -- `ops.convert_to_numpy`
  is not always available under `model.fit()`'s tracing).
- `dl_techniques/datasets/vision/multi_crop_video.py` -- the video-shaped
  multi-crop `tf.data` transform (1 global + N local views, same crop box
  across every frame of one clip via a stateless-RNG replay -- see
  `decisions.md` D-019). Reuses `multi_crop.py`'s crop/augmentation
  primitives, does not re-derive them (`decisions.md` D-005).
- `src/train/levjepa/train_levjepa.py` -- the training CLI (`--dataset
  {synthetic_drone,bdd100k}`, `--smoke` preset, `optimizer_builder()` +
  `WarmupSchedule(primary_schedule=FlatSchedule(...))`, `EMAShadowCallback`
  wired as a callback). Verified with a REAL end-to-end smoke run, not an
  import check.

## Usage

```python
from dl_techniques.models.vision.levjepa import create_levjepa

# Image mode, frozen sincos positions, full attention.
encoder = create_levjepa(variant="vit_tiny", input_shape=(64, 64, 3))

# Video mode, RoPE, block-causal attention.
encoder = create_levjepa(
    variant="vit_tiny",
    input_shape=(32, 32, 3),
    num_frames=4,
    tubelet_size=2,
    use_rope=True,
    attn_mode="block_causal",
)
```

`LeVJEPAEncoder.call()` returns `(batch, 1 + num_patches_kept, embed_dim)`,
with the CLS token at index 0.

## Deliberate scope simplifications (not gaps)

- No multi-output-feature (`out_layers`) branch -- only the final CLS token is
  ever consumed downstream, so only the last layer's normalized sequence is
  returned.
- No dynamic positional-embedding interpolation -- the frozen sincos table is
  sized once, for the constructed `input_shape`/`num_frames`.
- `use_rope` is the ONLY position-mode toggle, matching the reference's own
  constructor exactly -- there is no separate `pos_embed=` argument to
  conflict with it.

See `decisions.md` D-011 through D-013 in this plan's directory for the
reasoning behind each of these and the LayerScale-removal correction.

## House-convention gap (RESOLVED, step 6.2 completion-fix)

`encoder.py` and `blocks.py` (Step 4), plus `layers/embedding/patch_embed_3d.py`
and `layers/embedding/video_rope.py` (Step 2), previously imported
`from keras import ops` rather than the house convention `import keras` +
qualify at the call site (`src/dl_techniques/CLAUDE.md` § Core Conventions).
Flagged by `plan.md` Success Criterion 12's own check
(`grep -rn "from keras import ops" ...`), which found 4 sites, not the 2
self-reported at the end of Step 6. Fixed in the iter-1 completion-fix round
(step 6.2): all 4 files now use `import keras` and qualify every call site as
`keras.ops.<fn>(...)`. Re-verified empty:
`grep -rn "from keras import ops" src/dl_techniques/models/vision/levjepa/
src/dl_techniques/layers/embedding/patch_embed_3d.py
src/dl_techniques/layers/embedding/video_rope.py`.
