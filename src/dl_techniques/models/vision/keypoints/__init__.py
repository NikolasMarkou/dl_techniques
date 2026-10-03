"""Keypoints — local interest-point detection and description.

Models that emit sparse, repeatable image locations together with a descriptor for each,
and models that match such keypoints across images: the front end and the matcher of
tracking, homography estimation and structure-from-motion pipelines. This subfamily keeps
that task separate from the classification and segmentation backbones filling most of
``vision/``.

It currently holds two packages:

* ``lightglue/`` — LightGlue keypoint matcher: a transformer over the keypoints and
  descriptors of two images that returns per-layer soft assignments with a dustbin. It is
  detector-agnostic; its generic blocks live in ``layers/matching/``.
* ``superpoint/`` — SuperPoint keypoint detector and descriptor: one shared encoder, two
  heads, producing a detection heatmap and a full-resolution descriptor field in a single
  forward pass.

The grouping is by task, not by count: a further detector or matcher added here should not
force a move of the existing ones.

This module holds no re-exports, like every other container under ``models/``. Import
from the leaf package:

    from dl_techniques.models.vision.keypoints.superpoint import create_superpoint
    from dl_techniques.models.vision.keypoints.lightglue import create_lightglue

Re-exporting here would save one import line at the cost of an eager import of every
package in the subfamily. See ``plan-2026-08-24T205033-8fd4f20d/D-002`` and
``models/AGENTS.md`` for the reasoning.
"""
