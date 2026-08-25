"""Super-resolution — reconstructing a higher-resolution image from a low-resolution one.

The task is separated from ``image_restoration/`` on purpose: restoration recovers a clean
image at the SAME resolution as its degraded input, while super-resolution changes the
output grid. Models whose scale factor is an argument belong here.

It currently holds exactly one package:

* ``pft_sr/`` — PFT-SR, the Progressive Focused Transformer for single-image
  super-resolution, whose attention maps are inherited and refined across layers.

One member is not a mistake and not a reason to flatten the directory back into
``vision/``: the grouping is by task, and a second model added here should not force a move
of the first. Note that ``vision/thera/`` is also arbitrary-scale super-resolution and sits
one level up rather than here — a filing decision made by the restructure, not a claim
about the architecture.

This module is deliberately free of re-exports, like every other container under
``models/``. Import from the leaf package:

    from dl_techniques.models.vision.super_resolution.pft_sr import create_pft_sr

Re-exporting here would buy one saved import line and cost an eager import of every
package in the subfamily; the reasoning is recorded in
``plan-2026-08-24T205033-8fd4f20d/D-002`` and summarised in ``models/CLAUDE.md``.
"""
