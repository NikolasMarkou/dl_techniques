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

Not yet implemented (later plan steps): the projector head, the EMA-shadow
checkpoint callback, the multiview training wrapper (`SIGReg` + `add_loss`),
and the `src/train/levjepa/` training script.

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
