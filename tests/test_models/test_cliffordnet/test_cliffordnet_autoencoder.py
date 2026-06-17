"""
Test suite for the explicit-Laplacian-pyramid CliffordNet autoencoder
(autoencoder.py).

This file currently covers the standalone ``LaplacianPyramidLevel`` helper: the
falsifiable reconstruction identity ``merge(split(x)) == x`` (atol 1e-5), the
split band shapes, and ``compute_output_shape``. Model-level tests (config
round-trip, forward, .keras save/load, variants, fit-smoke) are added in later
steps.
"""

import keras
import numpy as np

from dl_techniques.models.cliffordnet.autoencoder import LaplacianPyramidLevel


# ---------------------------------------------------------------------
# LaplacianPyramidLevel
# ---------------------------------------------------------------------

class TestLaplacianIdentity:

    def test_reconstruction_identity(self):
        x = np.random.RandomState(0).randn(2, 64, 64, 3).astype("float32")
        layer = LaplacianPyramidLevel()
        low, high = layer(x)
        recon = layer.merge(low, high)
        recon_np = keras.ops.convert_to_numpy(recon)
        np.testing.assert_allclose(
            recon_np, x, atol=1e-5,
            err_msg="LaplacianPyramidLevel merge(split(x)) != x",
        )

    def test_split_shapes(self):
        x = np.random.RandomState(0).randn(2, 64, 64, 3).astype("float32")
        layer = LaplacianPyramidLevel()
        low, high = layer(x)
        assert tuple(low.shape) == (2, 32, 32, 3)
        assert tuple(high.shape) == (2, 64, 64, 3)

    def test_compute_output_shape(self):
        layer = LaplacianPyramidLevel()
        low_s, high_s = layer.compute_output_shape((2, 64, 64, 3))
        assert low_s == (2, 32, 32, 3)
        assert high_s == (2, 64, 64, 3)
