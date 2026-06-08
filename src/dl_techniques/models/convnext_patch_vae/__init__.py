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

Hierarchical (2-level) Patch-Ladder-VAE components
(``plans/plan_2026-06-08_e3917bd5/``, :mod:`.model_hierarchical` and
:mod:`.config`):

- :class:`HierarchicalConvNeXtPatchVAEConfig` — dataclass config for the
  hierarchical model (:mod:`.config`).
- :class:`HierarchicalConvNeXtPatchVAE` — 2-level sibling model: pool-derived
  coarse latent ``z2`` + learned top-down conditional prior ``p(z1|z2)`` with
  VDVAE delta-parameterization and free-bits gating (:mod:`.model_hierarchical`).
- :class:`_L2ConditionalPrior` — learned convolutional top-down conditional
  prior layer (:mod:`.model_hierarchical`).
- :class:`_CoarseLatentHead` — pool-derived coarse-latent head (:mod:`.model_hierarchical`).
- :func:`create_hierarchical_convnext_patch_vae` — module-level factory wrapping
  :meth:`HierarchicalConvNeXtPatchVAE.from_variant` (:mod:`.model_hierarchical`).
- ``HIERARCHICAL_PRESETS`` — tiny/base/large preset dict (:mod:`.model_hierarchical`).

Per ``models/CLAUDE.md`` this ``__init__.py`` exports nothing — import
from submodules directly.
"""
