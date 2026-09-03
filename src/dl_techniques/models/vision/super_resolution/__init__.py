"""Super-resolution — reconstructing a higher-resolution image from a low-resolution one.

Restoration (``image_restoration/``) recovers a clean image at the same resolution as its
degraded input. Super-resolution changes the output grid, so models whose scale factor is an
argument belong here.

It currently holds one package:

* ``pft_sr/`` — PFT-SR, the Progressive Focused Transformer for single-image
  super-resolution, whose attention maps are inherited and refined across layers.

``vision/thera/`` is also arbitrary-scale super-resolution but sits one level up rather than
here, a filing choice rather than an architectural distinction.

This module holds no re-exports, like every other container under ``models/``. Import from
the leaf package:

    from dl_techniques.models.vision.super_resolution.pft_sr import create_pft_sr

See ``models/CLAUDE.md`` for why containers under ``models/`` do not re-export.
"""
