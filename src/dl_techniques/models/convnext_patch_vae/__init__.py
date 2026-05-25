"""ConvNeXt patch-level VAE package.

A resolution-agnostic ConvNeXt-based variational autoencoder operating
on per-patch latents ``z(B, Hp, Wp, latent_dim)`` with SIGReg-driven
anti-patch-collapse. See ``plans/plan_2026-05-25_fb57d478/`` for the
original design rationale and ``plans/plan_2026-05-25_8faec5b6/`` for
the factory surface.

Core components:

- :class:`ConvNeXtPatchVAEConfig` — dataclass config (:mod:`.config`).
- :class:`ConvNeXtPatchEncoder` — flat single-stage ConvNeXtV2 encoder
  (:mod:`.encoder`).
- :class:`ConvNeXtPatchDecoder` — symmetric flat ConvNeXtV2 decoder
  (:mod:`.decoder`).
- :class:`ConvNeXtPatchVAE` — top-level model with encode/decode/sample
  API and custom ``train_step`` (:mod:`.model`).
- :func:`create_convnext_patch_vae` — module-level factory wrapping
  :meth:`ConvNeXtPatchVAE.from_variant` (:mod:`.model`).

Per ``models/CLAUDE.md`` this ``__init__.py`` exports nothing — import
from submodules directly.
"""
