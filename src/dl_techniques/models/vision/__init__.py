"""Vision models — image backbones, detectors, segmenters, denoisers and generators.

The largest family: 35 leaf packages, four of which sit one level deeper under a
task subdirectory (`image_restoration/`, `keypoints/`, `super_resolution/`).

- `accunet/` — AccuNet
- `beit/` — BEiT (masked image modeling over discrete visual tokens + classifier)
- `bias_free_denoisers/` — bias-free denoiser models
- `capsnet/` — Capsule Networks
- `cbam/` — CBAM attention model
- `cliffordnet/` — Clifford-algebra networks
- `convnext/` — ConvNeXt
- `convunext/` — ConvUNeXt (U-Net + ConvNeXt)
- `coshnet/` — CoshNet
- `depth_anything/` — depth estimation
- `detr/` — DEtection TRansformer
- `dino/` — DINO self-supervised
- `energy_transformer/` — Energy Transformer (masked image completion + classifier)
- `fastvit/` — FastViT MCi image backbone, the assembled tower over `layers/fastvit/`
- `fractalnet/` — FractalNet
- `image_restoration/darkir/` — DarkIR image restoration
- `image_restoration/pw_fnet/` — a 2-level U-Net with FFT token mixing and multi-scale
  supervision, for image restoration. The name misattributes; see `models/CLAUDE.md`
- `image_restoration/scunet/` — SCUNet denoiser
- `keypoints/superpoint/` — SuperPoint keypoint detector + descriptor
- `lewm/` — latent-energy world model
- `masked_autoencoder/` — MAE
- `mobilenet/` — MobileNet variants (V1, V2, V3, V4)
- `resnet/` — ResNet architectures
- `squeezenet/` — SqueezeNet
- `super_resolution/pft_sr/` — super-resolution
- `swin_transformer/` — Swin Transformer
- `thera/` — THERA aliasing-free arbitrary-scale super-resolution
- `vae/` — Variational Autoencoder with a ResNet encoder/decoder
- `video_jepa/` — Video JEPA (joint embedding predictive)
- `vit/` — Vision Transformer
- `vit_hmlp/` — ViT with hierarchical MLP
- `vit_siglip/` — ViT with a two-stage conv patch-embedding stem. The name
  misattributes; see `models/CLAUDE.md`
- `vq_vae/` — VQ-VAE
- `vq_vae_rotation/` — VQ-VAE with rotation-based codebook updates
- `yolo12/` — YOLOv12 detection

This module is deliberately free of re-exports, and so are the other family packages.
Import from the leaf package:

    from dl_techniques.models.vision.resnet import create_resnet

Re-exporting a family here would mean that `import dl_techniques.models.vision` eagerly
constructs every one of these 35 packages — the whole Keras/TensorFlow import cost of the
family for one model — and it opens a circular-import surface between packages that share
layers. `time_series/` is the one family that does re-export; it predates this layout and
its consumers rely on it, so it is left as it is rather than made consistent.
"""
