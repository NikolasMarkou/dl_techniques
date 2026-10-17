"""The numpy DINOv2 reference can fail: it has torch's kernel and reads every tensor.

RED proof of the shared instrument ``dinov2_numpy_reference.py``
(plan-2026-10-14T062616-81938e81 iteration 2 step 1.7). The official-weights
guards compare the Keras port against it, so it must (1) carry torch's
bicubic kernel and not another one, and (2) read every official tensor the
forward uses, so a mis-mapped tensor in the port cannot match it by accident.
No downloaded file is needed: the tensors here are synthetic.
"""

import numpy as np
import pytest

from .dinov2_numpy_reference import embed, forward, official_manifest, torch_bicubic_2d


def test_the_kernel_is_torchs_a_minus_0_75():
    # Downsampling 8 -> 4 at the output-size scale puts output o at source
    # 2o + 0.5 (t = 0.5), taps 2o-1 .. 2o+2. torch's coefficients at t = 0.5 for
    # A = -0.75, by hand: near(0.5) = (1.25 * 0.5 - 2.25) * 0.25 + 1 = 0.59375,
    # far(1.5) = ((-0.75 * 1.5 + 3.75) * 1.5 - 6) * 1.5 + 3 = -0.09375 (Keys'
    # A = -0.5, TF's bicubic, gives 0.5625 / -0.0625). A delta at 3 is tap 3 of
    # output 1 and tap 1 of output 2.
    signal = np.zeros((1, 1, 8))
    signal[0, 0, 3] = 1.0
    np.testing.assert_allclose(torch_bicubic_2d(signal, 1, 4)[0, 0],
                               [0.0, 0.59375, -0.09375, 0.0], rtol=0, atol=1e-12)


def test_a_constant_stays_constant_and_scale_one_is_the_identity():
    image = np.random.default_rng(0).normal(size=(3, 5, 7))
    np.testing.assert_allclose(torch_bicubic_2d(np.ones((2, 5, 5)), 3, 4, 0.7, 0.9), 1.0,
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(torch_bicubic_2d(image, 5, 7, 1.0, 1.0), image, rtol=0, atol=1e-12)


def _synthetic(dim=8, depth=2, grid=3, seed=0):
    rng = np.random.default_rng(seed)
    return {name: rng.normal(size=shape).astype(np.float32)
            for name, shape in official_manifest(dim, depth, grid=grid, patch=2).items()}


IMAGE = np.random.default_rng(1).normal(size=(4, 6, 3))   # a 2 x 3 grid: resampled


def test_the_forward_reads_every_tensor_but_the_mask_token_and_the_key_biases():
    # The key bias adds the same q . b_k to every score of one query, which the
    # softmax removes: no implementation's output depends on it (so the loader's
    # mapping of it is pinned by the shape and naming tests, not by value).
    tensors = _synthetic()
    base, layers = forward(tensors, IMAGE, num_heads=2, last=2, patch=2)
    assert len(layers) == 2
    unread = []
    for name in tensors:
        bumped = dict(tensors, **{name: tensors[name] + 0.5})
        if np.allclose(forward(bumped, IMAGE, num_heads=2, patch=2)[0], base, rtol=0, atol=1e-9):
            unread.append(name)
    assert unread == ["embeddings.mask_token",
                      "encoder.layer.0.attention.attention.key.bias",
                      "encoder.layer.1.attention.attention.key.bias"], unread


def test_the_offset_changes_the_resampled_table():
    tensors = _synthetic()
    assert np.abs(embed(tensors, IMAGE, 0.1, patch=2) - embed(tensors, IMAGE, 0.0, patch=2)).max() > 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
