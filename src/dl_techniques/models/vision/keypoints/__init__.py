"""Keypoints — local interest-point detection and description.

Models that emit sparse, repeatable image locations together with a descriptor for each,
the front end of matching, tracking, homography estimation and structure-from-motion
pipelines. The subfamily exists to keep that task separate from the classification and
segmentation backbones filling most of ``vision/``.

It currently holds exactly one package:

* ``superpoint/`` — SuperPoint keypoint detector + descriptor: one shared encoder, two
  heads, producing a detection heatmap and a full-resolution descriptor field in a single
  forward pass.

One member is not a mistake and not a reason to flatten the directory back into
``vision/``: the grouping is by task, and a second detector added here should not force a
move of the first.

This module is deliberately free of re-exports, like every other container under
``models/``. Import from the leaf package:

    from dl_techniques.models.vision.keypoints.superpoint import create_superpoint

Re-exporting here would buy one saved import line and cost an eager import of every
package in the subfamily; the reasoning is recorded in
``plan-2026-08-24T205033-8fd4f20d/D-002`` and summarised in ``models/CLAUDE.md``.
"""
