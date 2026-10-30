"""Mean shift Phase B experiment harness.

``corruption`` holds the occlusion and noise functions shared by every arm, ``common``
the cell runner (idle-GPU gate, manifest, fsynced rows, resume). Both import without
TensorFlow.
"""
