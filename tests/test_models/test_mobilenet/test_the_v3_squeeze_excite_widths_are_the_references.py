"""MobileNetV3's squeeze-excite bottleneck widths are the reference's.

Which reference: Keras Applications ``MobileNetV3`` at every width since
plan-2026-10-14T062616-81938e81 step 10.3 (the expanded widths, too, which
followed torchvision's ``adjust_channels`` before). For the SE bottleneck
Keras Applications and torchvision agree: ``_se_block`` uses
``_depth(filters * se_ratio)`` (divisor 8, at least 8) and torchvision
``SqueezeExcitation`` uses ``_make_divisible(expanded // 4, 8)``; every
expanded width here is a multiple of 8, so ``expanded * 0.25`` and
``expanded // 4`` are equal. Both are re-typed below as oracles, and the
widths are also read from a live ``keras.applications`` model built with
``weights=None`` at every width the formula test covers.

Until plan-2026-10-14T062616-81938e81 step 9.3 the port rounded the bottleneck
with ``max(1, int(expanded * 0.25))``: 2 channels at Small's first SE block at
``width_multiplier=0.25``, whose ReLU died on about one draw in eight (seeds 0
and 10 of 0..15 left both SE kernels with an identically-zero gradient).
"""

import re

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.mobilenet.mobilenet_v3 import MobileNetV3

from .test_oracle_adoption import (
    SMALL,
    _built,
    _images,
    _one_adam_step,
    _v3,
    gradient_report,
    ramp_loss,
)

_SE_REDUCE = re.compile(r"/block_(\d+)/.*reduce/kernel$")


def _keras_applications_depth(v: float, divisor: int = 8) -> int:
    """Keras Applications ``mobilenet_v3._depth`` (``min_value`` = divisor)."""
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _torchvision_make_divisible(v: float, divisor: int = 8) -> int:
    """torchvision ``models/_utils._make_divisible`` (``min_value`` = divisor)."""
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _se_kernels(model):
    """``(block index, expanded width, bottleneck width)`` per SE block."""
    rows = []
    for w in model.weights:
        m = _SE_REDUCE.search(w.path)
        if m:
            rows.append((int(m.group(1)), int(w.shape[-2]), int(w.shape[-1])))
    return sorted(rows)


def _model(variant: str, width: float) -> MobileNetV3:
    m = MobileNetV3(num_classes=1000, variant=variant, width_multiplier=width)
    m(np.zeros((1, 32, 32, 3), "float32"))
    return m


@pytest.mark.parametrize("variant", ["small", "large"])
@pytest.mark.parametrize("width", [1.0, 0.75, 0.5, 0.25])
def test_each_se_width_is_the_reference_formula(variant, width):
    rows = _se_kernels(_model(variant, width))
    assert len(rows) == (9 if variant == "small" else 8)
    for block, expanded, bottleneck in rows:
        assert expanded % 8 == 0
        keras_ref = _keras_applications_depth(expanded * 0.25)
        torch_ref = _torchvision_make_divisible(expanded // 4)
        assert keras_ref == torch_ref
        assert bottleneck == keras_ref, (
            f"{variant} x{width} block_{block}: SE width {bottleneck} for "
            f"{expanded} expanded channels, reference {keras_ref}")


@pytest.mark.parametrize("width", [1.0, 0.75, 0.5, 0.25])
@pytest.mark.parametrize("variant,ctor", [
    ("small", "MobileNetV3Small"), ("large", "MobileNetV3Large")])
def test_the_widths_equal_a_live_keras_applications_model(variant, ctor, width):
    reference = getattr(keras.applications, ctor)(
        input_shape=(224, 224, 3), alpha=width, weights=None,
        include_preprocessing=False)
    ref_widths = [
        layer.filters for layer in reference.layers
        if layer.name.endswith("squeeze_excite_conv")]
    assert [b for _, _, b in _se_kernels(_model(variant, width))] == ref_widths


def _one_by_one_betas(variant: str) -> set:
    """``project_norm/beta`` of every block whose output is 1 x 1 at 32 px.

    A per-channel shift of a 1 x 1 map is spatially uniform, and every path
    from it reaches a 1x1 conv followed by a training-mode BatchNorm, which
    subtracts it exactly; its gradient is float32 rounding (measured 2e-9 to
    3e-7 of the median at 32 px, about 1e-4 at 64 and 128 px), so the oracle's
    exact-zero test reads it as live or dead by the draw (small, seed 3,
    block_8 at d929850f3). The MobileNetV2 arm of the oracle-adoption file pins
    the same geometry artifact.
    """
    config = (MobileNetV3.SMALL_CONFIG if variant == "small"
              else MobileNetV3.LARGE_CONFIG)
    size, betas = -(-SMALL[0] // 2), set()
    for i, row in enumerate(config):
        size = -(-size // row[3])
        if size == 1:
            betas.add(f"block_{i}/project_norm/beta")
    return betas


@pytest.mark.parametrize("variant", ["small", "large"])
def test_the_se_path_is_live_at_every_seed(variant):
    """The gradient-flow oracle, seeds 0..15, one Adam step, ramp loss, at the
    narrowest width the oracle-adoption file uses (0.25, 32 px): no weight
    reads an exactly-zero gradient (the 1 x 1-stage betas above excepted), and
    every SE kernel's gradient is at least 1e-4 of the model's median."""
    artifacts = _one_by_one_betas(variant)
    assert artifacts, "anti-vacuity: 32 px reaches a 1 x 1 stage"
    failures = {}
    for seed in range(16):
        model = _built(lambda: _v3(variant=variant), SMALL, seed=seed)
        x = _images(SMALL)
        _one_adam_step(model, x)
        report = {p.split("/", 1)[1]: v for p, v in
                  gradient_report(model, x, loss_fn=ramp_loss).items()}
        median = float(np.median([v for v in report.values() if v]))
        dead = sorted(p for p, v in report.items()
                      if (v is None or v == 0.0) and p not in artifacts)
        weak_se = sorted(p for p, v in report.items() if "/se/" in p
                         and (v is None or v < 1e-4 * median))
        if dead or weak_se:
            failures[seed] = dead + weak_se
    assert failures == {}


# The step 9.3 hand-sum pin (the count moved from the d929850f3 totals by
# exactly the SE widths and biases) was retired at step 10.3 (D-232): its
# baseline widths no longer exist below 1.0, and the stronger claim, count and
# every weight shape equal to a live keras.applications model at every width,
# is test_the_v3_widths_follow_keras_applications_at_every_width.py.
