"""Golden-value regression guard for :class:`SCUNet`.

Why this file exists
--------------------

``tests/test_models/test_scunet/test_model.py`` asserts block counts, output
shapes, padding behaviour and ``.keras`` round-trips. It asserts **no numeric
value**. That gap was measured, not assumed: the plan
``plan-2026-07-31-ddc92265`` changed shifted-window attention inside
``SwinTransformerBlock`` (it now builds and applies the SW-MSA mask that was
missing, and it follows reference Swin's single-window rule at
``H == window_size``), which moved this model's output on **12255 of 12288**
elements -- and that suite reported **zero movement**, 148 passed / 1 xfailed
on both sides of the change.

So a shipped denoiser's numerics changed with no test that could have noticed.
This module closes that hole for the NEXT change.

What the reference numbers are, and what they are not
-----------------------------------------------------

They were generated from **current HEAD, after** the plan's fixes landed. They
are a *forward-looking tripwire*, not a claim that the pre-plan values were
right -- the pre-plan values were provably wrong (no SW-MSA mask at all). If a
future change moves them, that is not automatically a defect; it is a signal
that this path's numerics moved and someone must decide, deliberately, whether
that was intended. Regenerate them only together with that decision, and record
it.

Determinism and tolerance
-------------------------

Every weight is overwritten from a seeded ``numpy`` generator after build, so
the values do not depend on Keras initializer internals, and the forward pass
runs at ``training=False`` with ``stochastic_depth_rate=0.0``. Measured
bit-identical across three separate interpreter processes on CPU. ``atol=1e-8``
keeps two orders of magnitude of headroom over that floor while staying far
under the ``3.2e-05`` movement this plan actually produced.

Device
------

The probe is pinned to the ``golden_reference_device`` fixture
(``tests/conftest.py``), which carries the measurement and the rationale.
Unpinned on GPU 1 this model's output lands ``4.062057e-05`` from the CPU
reference (whole-array statistics ``1.186132e-05`` off), which is ordinary
reduced-precision matmul reassociation -- dominated by TF32 -- and not a
device-dependent code path, but it is far outside ``atol`` and the tolerance is
not negotiable here. So: this module pins **CPU** numerics. The GPU answer
differs by the amount above and is guarded by nothing.

What this guard does NOT cover
------------------------------

The D-006 single-window rule (``H == window_size`` -> ``shift_size = 0``) is
**structurally unreachable from SCUNet** and no tolerance here can change that.
``swin_conv_block.py:177-185`` downgrades ``block_type`` ``"SW" -> "W"``
whenever ``input_resolution <= window_size``, so the block never carries a shift
for the rule to suppress. Measured over all 14 ``SwinConvBlock``s of this
config: at ``input_resolution=256`` the smallest stage is 32 (no ``H <= ws``
stage exists at all), and at ``input_resolution=64`` the 8x8 bottleneck is
downgraded before construction. Instrumenting ``_resolve_shift_size`` across a
full forward pass returns "resolved shift differs from configured shift" for
**zero** of the 28 blocks. That rule's shipped consumer is
``models/swin_transformer`` alone, and its guard lives in
``tests/test_models/test_swin_transformer/test_golden_values.py``.
"""

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.scunet.model import SCUNet

# --- Frozen probe configuration. Do not tune; the references below are tied
#     to every one of these values, including the two seeds. ----------------
CONFIG = dict(
    in_nc=3,
    config=[2] * 7,
    dim=16,
    head_dim=8,
    window_size=8,
    input_resolution=64,
    stochastic_depth_rate=0.0,
)
INPUT_SHAPE = (1, 64, 64, 3)
INPUT_SEED = 7
WEIGHT_SEED = 20260731
WEIGHT_SCALE = 0.05

# Sampled at ``np.linspace(0, size - 1, 12).astype(int)`` over the flattened
# output. Sampling (rather than an inlined 12288-element array) is deliberate:
# the change this guard exists to catch moved 12255/12288 elements, so twelve
# spread-out probes plus the four whole-array statistics below cannot miss it,
# and the file stays readable.
GOLDEN_INDICES = (0, 1117, 2234, 3351, 4468, 5585, 6702, 7819, 8936, 10053, 11170, 12287)
GOLDEN_VALUES = (
    0.025833524763584137,
    -0.21490876376628876,
    0.14970672130584717,
    0.15265358984470367,
    0.18284118175506592,
    0.049421459436416626,
    0.0020036837086081505,
    0.05485982447862625,
    -0.4749571681022644,
    -0.00747334212064743,
    -0.3105686604976654,
    -0.020963262766599655,
)
# mean, std, min, max over the FULL output -- these see every element, so a
# change confined to the elements the sampler happens to skip is still caught.
GOLDEN_STATS = (
    -0.06287622451782227,
    0.16517893970012665,
    -0.6904259920120239,
    0.6506773233413696,
)

ATOL = 1e-8
RTOL = 0.0


def _run_probe() -> np.ndarray:
    """Build the probe model with seeded weights and run one forward pass.

    The caller is responsible for the device scope; see ``golden_output``.

    :return: the ``(1, 64, 64, 3)`` denoised output, as ``float32`` numpy.
    """
    model = SCUNet(**CONFIG)
    x = np.asarray(
        np.random.default_rng(INPUT_SEED).standard_normal(INPUT_SHAPE), "float32"
    )
    model(x)  # materialize every sublayer before assigning
    rng = np.random.default_rng(WEIGHT_SEED)
    for w in model.weights:
        w.assign(np.asarray(rng.normal(size=w.shape) * WEIGHT_SCALE, dtype=w.dtype))
    return np.asarray(ops.convert_to_numpy(model(x, training=False)))


@pytest.fixture(scope="module")
def golden_output(golden_reference_device) -> np.ndarray:
    """The pinned probe run. See this module's "Device" section."""
    # The scope must cover CONSTRUCTION as well as the forward pass: it is what
    # places the variables, and a model whose weights live on GPU computes on
    # GPU no matter where the call happens.
    with keras.device(golden_reference_device):
        return _run_probe()


class TestSCUNetGoldenValues:
    """Pin the shipped forward-pass numerics of the Swin-carrying denoiser."""

    def test_output_shape_is_the_one_the_references_were_taken_at(
        self, golden_output
    ):
        """A shape change would make every comparison below meaningless."""
        assert golden_output.shape == INPUT_SHAPE
        assert np.isfinite(golden_output).all()

    def test_sampled_elements_match_the_reference(self, golden_output):
        """Twelve spread-out output elements must still hold their values."""
        flat = golden_output.ravel()
        assert flat.size == 12288, (
            f"the probe output has {flat.size} elements, not the 12288 the "
            "reference indices were sampled from"
        )
        got = flat[list(GOLDEN_INDICES)]
        want = np.asarray(GOLDEN_VALUES, dtype="float64")
        # Non-vacuity: a model that emitted all-zeros would otherwise have to
        # be caught by the tolerance alone.
        assert float(np.abs(want).max()) > 1e-3
        np.testing.assert_allclose(
            got,
            want,
            rtol=RTOL,
            atol=ATOL,
            err_msg=(
                "SCUNet forward-pass numerics moved. This is a REAL change to a "
                "shipped denoiser's output, not a flaky test: max|diff|="
                f"{float(np.abs(got - want).max()):.6e} at sampled indices "
                f"{GOLDEN_INDICES}. Decide whether it was intended, then "
                "regenerate the reference in the SAME commit and say why."
            ),
        )

    def test_whole_array_statistics_match_the_reference(self, golden_output):
        """mean/std/min/max over every element, so nothing hides between samples."""
        got = np.asarray(
            [
                float(golden_output.mean()),
                float(golden_output.std()),
                float(golden_output.min()),
                float(golden_output.max()),
            ]
        )
        np.testing.assert_allclose(
            got,
            np.asarray(GOLDEN_STATS, dtype="float64"),
            rtol=RTOL,
            atol=ATOL,
            err_msg=(
                "SCUNet whole-output statistics (mean, std, min, max) moved: "
                f"got {got.tolist()}, expected {list(GOLDEN_STATS)}."
            ),
        )

    def test_the_probe_is_reproducible_within_a_process(
        self, golden_output, golden_reference_device
    ):
        """Two identical builds must agree bit-exactly.

        Without this, a failure above could equally be nondeterminism rather
        than a numerics change, and the guard would be untrustworthy in exactly
        the situation it exists for.

        Pinned to the same device as the fixture on purpose: this comparison is
        against ``golden_output``, so leaving it unpinned would measure the
        device difference rather than reproducibility.
        """
        with keras.device(golden_reference_device):
            again = _run_probe()
        assert np.array_equal(again, golden_output), (
            "the probe is not reproducible within one process: max|diff|="
            f"{float(np.abs(again - golden_output).max()):.6e}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
