"""Vision-language models — architectures that consume or relate both an image and a text
stream, plus the promptable segmenters that grew out of them.

- `bit_diffusion/` — BiT/BiB bidirectional text<->image diffusion bridge (DiTXA)
- `clip/` — CLIP
- `dit/` — DiT, the class-conditional latent Diffusion Transformer (Peebles & Xie),
  plus the DDPM sampler it needs
- `fastvlm/` — a vision-only hybrid backbone (MobileOne stem + RepMixer + attention
  stages). The name misattributes; see `models/CLAUDE.md`
- `ideogram4/` — Ideogram4 text-to-image flow-matching DiT
- `mobile_clip/` — MobileCLIP, both generations in one package: `mobile_clip_v1.py` is
  deliberately non-faithful on the image side, `mobile_clip_v2.py` is the faithful port
- `nano_vlm/` — NanoVLM
- `sam/sam1/` — Segment Anything Model v1
- `sam/sam2/` — Segment Anything Model v2 (video / memory)
- `sam/sam3/` — Segment Anything Model v3 (text-promptable)
- `sd3_mmdit/` — SD3 MMDiT dual-stream text-to-image diffusion transformer

The three SAM generations sit one level deeper, under `sam/`.

Import from the leaf package, not from here — this family package carries no re-exports
by design (the reasoning is written out in `models/vision/__init__.py`):

    from dl_techniques.models.vision_language.sam.sam2 import SAM2
"""
