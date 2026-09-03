"""Keypoints — local interest-point detection and description.

Models that emit sparse, repeatable image locations together with a descriptor for each,
the front end of matching, tracking, homography estimation and structure-from-motion
pipelines. This subfamily keeps that task separate from the classification and
segmentation backbones filling most of ``vision/``.

It currently holds one package:

* ``superpoint/`` — SuperPoint keypoint detector and descriptor: one shared encoder, two
  heads, producing a detection heatmap and a full-resolution descriptor field in a single
  forward pass.

A single member is not a reason to flatten this directory back into ``vision/``: the
grouping is by task, and a second detector added here should not force a move of the
first.

This module holds no re-exports, like every other container under ``models/``. Import
from the leaf package:

    from dl_techniques.models.vision.keypoints.superpoint import create_superpoint

Re-exporting here would save one import line at the cost of an eager import of every
package in the subfamily. See ``plan-2026-08-24T205033-8fd4f20d/D-002`` and
``models/CLAUDE.md`` for the reasoning.
"""
