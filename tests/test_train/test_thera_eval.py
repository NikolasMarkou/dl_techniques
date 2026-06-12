"""Fast unit tests for the THERA multi-scale evaluation module.

No real training, no DIV2K: a tiny :class:`Thera` is built and dummy-forwarded,
then ``super_resolve`` / ``evaluate_multiscale`` are exercised on random inputs.
The tests assert SHAPE + FINITENESS + correct dict structure only -- an untrained
model will NOT beat bicubic, so no quality assertion is made.
"""

import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.thera import (
    Thera,
    EDSRBackbone,
    build_thera_tail,
)
from train.thera.eval_thera import super_resolve, evaluate_multiscale


DEFAULT_K_INIT = float(np.sqrt(np.log(4.0)) / (np.pi ** 2 * 2.0))


@pytest.fixture(scope="module")
def tiny_thera():
    """A minimal Thera model, dummy-forwarded to build all sublayers."""
    model = Thera(
        hidden_dim=16,
        out_dim=3,
        backbone=EDSRBackbone(num_feats=32, num_blocks=2),
        tail=build_thera_tail("air"),
        k_init=DEFAULT_K_INIT,
        components_init_scale=16.0,
    )
    # Dummy forward to build (source, coords, t).
    dummy_source = tf.zeros((1, 16, 16, 3))
    dummy_coords = tf.zeros((1, 8, 8, 2))
    dummy_t = tf.ones((1, 1))
    model((dummy_source, dummy_coords, dummy_t), training=False)
    return model


def test_super_resolve_shape_and_range(tiny_thera):
    rng = np.random.default_rng(0)
    lr = rng.random((32, 32, 3)).astype(np.float32)

    sr = super_resolve(tiny_thera, lr, (64, 64))

    assert sr.shape == (64, 64, 3)
    assert np.all(np.isfinite(sr))
    assert sr.min() >= 0.0 and sr.max() <= 1.0


def _write_random_pngs(tmp_path, n=3, size=24):
    rng = np.random.default_rng(123)
    paths = []
    for i in range(n):
        arr = (rng.random((size, size, 3)) * 255).astype(np.uint8)
        png = tf.io.encode_png(tf.convert_to_tensor(arr))
        p = tmp_path / f"img_{i}.png"
        tf.io.write_file(str(p), png)
        paths.append(str(p))
    return paths


def test_evaluate_multiscale_structure(tiny_thera, tmp_path):
    paths = _write_random_pngs(tmp_path, n=3, size=24)

    results = evaluate_multiscale(
        tiny_thera, paths, scales=(2, 3), max_images=3
    )

    assert set(results.keys()) == {"x2", "x3"}
    for scale_key, metrics in results.items():
        assert set(metrics.keys()) == {
            "psnr", "ssim", "bicubic_psnr", "bicubic_ssim"
        }
        for name, value in metrics.items():
            assert np.isfinite(value), f"{scale_key}/{name} not finite: {value}"


def test_evaluate_multiscale_y_only(tiny_thera, tmp_path):
    paths = _write_random_pngs(tmp_path, n=2, size=24)

    results = evaluate_multiscale(
        tiny_thera, paths, scales=(2,), max_images=2, y_only=True
    )

    assert set(results.keys()) == {"x2"}
    for value in results["x2"].values():
        assert np.isfinite(value)
