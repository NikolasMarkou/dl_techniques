"""The SAM 2 trainer clips the GLOBAL gradient norm at 0.1, as upstream does (D-264).

Upstream SAM 2 (``sam2.1_hiera_b+_MOSE_finetune.yaml`` at 2b90b9f5,
``optim.gradient_clip``: ``training.optimizer.GradientClipper`` with
``max_norm: 0.1, norm_type: 2``) calls
``torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)``: ONE norm
over every gradient, all of them scaled by the same factor. Keras' stock match
is ``global_clipnorm``; ``clipnorm`` clips each variable on its own and changes
the update's direction. These tests read the optimizer the trainer's own
``create_sam2_training_model`` compiles, and check the clip by value.
"""

import numpy as np
import pytest

from train.sam2.train_sam2 import config_from_argv, create_sam2_training_model

UPSTREAM_MAX_NORM = 0.1


@pytest.fixture(scope="module")
def optimizer():
    return create_sam2_training_model(config_from_argv(["--smoke"])).optimizer


def test_the_default_optimizer_carries_upstreams_global_clip(optimizer):
    assert optimizer.global_clipnorm == pytest.approx(UPSTREAM_MAX_NORM)
    assert optimizer.clipnorm is None, "per-variable clipping is not upstream's"
    assert optimizer.clipvalue is None


def test_the_clip_scales_every_gradient_by_one_global_factor(optimizer):
    """By value: the clipped list has global norm 0.1 and the SAME direction.

    Per-variable clipping (``clipnorm``) would give each gradient norm 0.1 on
    its own (global norm 0.1 * sqrt(2)) and change the ratio between them.
    """
    grads = [np.full((2,), 3.0, "float32"), np.full((1,), 4.0, "float32")]
    clipped = [np.asarray(g) for g in optimizer._clip_gradients(grads)]
    global_norm = np.sqrt(sum(float(np.sum(g ** 2)) for g in clipped))
    assert global_norm == pytest.approx(UPSTREAM_MAX_NORM, rel=1e-5)
    scale = UPSTREAM_MAX_NORM / np.sqrt(34.0)
    for before, after in zip(grads, clipped):
        np.testing.assert_allclose(after, before * scale, rtol=1e-5)


def test_a_gradient_inside_the_norm_is_left_alone(optimizer):
    grads = [np.full((2,), 0.01, "float32")]
    clipped = np.asarray(optimizer._clip_gradients(grads)[0])
    np.testing.assert_allclose(clipped, grads[0], rtol=1e-6)


def test_zero_turns_the_clip_off():
    config = config_from_argv(["--smoke", "--gradient-clip-norm", "0"])
    assert create_sam2_training_model(config).optimizer.global_clipnorm is None


def test_a_negative_norm_is_refused():
    with pytest.raises(ValueError, match="gradient_clip_norm"):
        config_from_argv(["--smoke", "--gradient-clip-norm", "-0.1"])
