"""Golden-value regression guard for :class:`SwinTransformer`.

Sibling of ``tests/test_models/test_scunet/test_golden_values.py``; read that
module's header for the full rationale. In short: ``test_model.py`` and
``test_round_trip.py`` assert structure, shapes and serialization and **no
numeric value**, and the plan ``plan-2026-07-31-ddc92265`` moved this
classifier's logits on **20 of 20** elements while both suites reported zero
movement.

The probe config is chosen so that BOTH behaviour changes that plan made are on
the path:

* stages 1-3 run at resolutions ``16 > 8 > 4``, all strictly greater than
  ``window_size = 2``, so their odd blocks take the shifted-window path and
  apply the SW-MSA mask that previously did not exist;
* stage 4 runs at resolution ``2 == window_size``, which is exactly the
  reference-Swin single-window rule (D-006, made runtime-conditional by D-012)
  that replaced the plan's falsified ``H < 2 * window_size`` guard. Verified by
  instrumenting ``_resolve_shift_size``: ``stage_3_block_1`` is configured with
  ``shift_size=1`` and resolves to ``0``, and it is the ONLY block that does.

The whole logit vector is inlined -- there are only eight numbers -- so unlike
the SCUNet guard this one needs no sampling and cannot miss anything.

Tolerance
---------

``atol=1e-8``, and that number is measured rather than chosen for comfort.
Neutering the SW-MSA mask (``ops.ones_like(...)`` at the block's call site)
moves these logits far outside any plausible tolerance, but neutering the D-006
single-window rule (``_resolve_shift_size`` -> ``return self.shift_size``) moves
them by only **1.04e-07** -- the affected block is one 2x2 stage whose four
tokens are then global-pooled. A ``1e-6`` tolerance would have let that probe
pass, i.e. the guard would have been blind to half of what it is here to watch.
``1e-8`` is still two orders of magnitude above the measured floor: the probe is
bit-identical across separate interpreter processes on CPU.

Device
------

Both the probe below and the reference it is compared against are pinned to the
``golden_reference_device`` fixture (``tests/conftest.py``), which is where the
justification and the measurement table live. Short version: unpinned on GPU 1
these logits land ``2.254173e-05`` away from the reference (8/8 elements), which
is 216x the 1.04e-07 signal the tolerance is calibrated for. That gap is
reduced-precision matmul reassociation, ~99.93% of it TF32 -- with TF32 off the
GPU is ``1.490116e-08`` away and at float64 ``2.08e-17`` -- so it is a precision
artifact, not a device-dependent Swin code path. It is nevertheless larger than
``atol``, so the probe is pinned rather than the tolerance widened.

Consequence, stated so it is not misread: this module pins the shipped
classifier's **CPU** numerics. The GPU answer differs by the amount above and is
guarded by nothing.
"""

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.vision.swin_transformer.model import SwinTransformer

# --- Frozen probe configuration. The references are tied to every value. ----
CONFIG = dict(
    num_classes=4,
    embed_dim=16,
    depths=[2, 2, 2, 2],
    num_heads=[1, 2, 4, 8],
    window_size=2,
    patch_size=4,
    drop_path_rate=0.0,
    input_shape=(64, 64, 3),
)
INPUT_SHAPE = (2, 64, 64, 3)
INPUT_SEED = 11
WEIGHT_SEED = 20260731
WEIGHT_SCALE = 0.05

#: Re-derived 2026-10-13. The previous values were taken against the bespoke
#: top (GlobalAveragePooling2D + a LINEAR Dense), and are kept nowhere: the
#: factory ``ClassificationHead`` replaced it, and that head's ``classifier``
#: applies ``activation='softmax'`` -- so this is a change of FUNCTION, not of
#: the last few bits, and the two sets share no digits. The tell is the row
#: sums: the pinned set summed to nothing in particular (a logit vector need
#: not), this one sums to 1.0 to 1e-8 because every entry is a probability.
#: The head's own docstring says both keys hold the same softmax output, so
#: ``'logits'`` is a misnomer there; what is pinned is the tensor ``call``
#: returns under that key, which is what the rest of this file compares.
GOLDEN_LOGITS = (
    (
        0.23747052252292633,
        0.25725874304771423,
        0.26556873321533203,
        0.2397020012140274,
    ),
    (
        0.23747017979621887,
        0.25725799798965454,
        0.26556596159935,
        0.2397058755159378,
    ),
)

ATOL = 1e-8
RTOL = 0.0


def _run_probe() -> np.ndarray:
    """Build the probe model with seeded weights and run one forward pass.

    The caller is responsible for the device scope; see ``golden_output``.

    :return: the ``(2, 4)`` array under the head's ``'logits'`` key, float32.
    """
    model = SwinTransformer(**CONFIG)
    x = np.asarray(
        np.random.default_rng(INPUT_SEED).standard_normal(INPUT_SHAPE), "float32"
    )
    model(x)  # materialize every sublayer before assigning
    rng = np.random.default_rng(WEIGHT_SEED)
    for w in model.weights:
        w.assign(np.asarray(rng.normal(size=w.shape) * WEIGHT_SCALE, dtype=w.dtype))
    out = model(x, training=False)
    # The factory ClassificationHead returns a dict. Its classifier applies
    # softmax, so both keys are the same probability tensor; 'logits' is the
    # one everything else in this file compares.
    return np.asarray(
        ops.convert_to_numpy(out["logits"] if isinstance(out, dict) else out)
    )


@pytest.fixture(scope="module")
def golden_output(golden_reference_device) -> np.ndarray:
    """The pinned probe run. See this module's "Device" section."""
    # The scope must cover CONSTRUCTION as well as the forward pass: it is what
    # places the variables, and a model whose weights live on GPU computes on
    # GPU no matter where the call happens.
    with keras.device(golden_reference_device):
        return _run_probe()


class TestSwinTransformerGoldenValues:
    """Pin the shipped forward-pass logits of the Swin classifier."""

    def test_output_shape_is_the_one_the_references_were_taken_at(
        self, golden_output
    ):
        assert golden_output.shape == (2, 4)
        assert np.isfinite(golden_output).all()

    def test_logits_match_the_reference(self, golden_output):
        """Every logit, both batch rows. Nothing is sampled away."""
        want = np.asarray(GOLDEN_LOGITS, dtype="float64")
        # Non-vacuity: an all-zero head would otherwise pass by tolerance.
        assert float(np.abs(want).max()) > 1e-3
        np.testing.assert_allclose(
            golden_output,
            want,
            rtol=RTOL,
            atol=ATOL,
            err_msg=(
                "SwinTransformer logits moved. This is a REAL change to a "
                "shipped classifier's output: max|diff|="
                f"{float(np.abs(golden_output - want).max()):.6e}. Decide "
                "whether it was intended, then regenerate the reference in the "
                "SAME commit and say why."
            ),
        )

    def test_the_two_batch_rows_are_not_identical(self, golden_output):
        """Guards the guard: if the head collapsed, the values above would be
        stable for a reason that has nothing to do with the attention path.

        The threshold was 1e-5 and is now 1e-6, re-derived 2026-10-13. The
        factory head's ``classifier`` applies softmax, so this compares two
        probability vectors rather than two logit vectors, and a softmax
        compresses the spread: measured row spread fell from above 1e-5 to
        3.874e-06, and re-running the probe with BOTH rows fed the SAME image
        gives exactly 0.0. So 3.874e-06 is entirely input-driven, with nothing
        of it arithmetic -- the floor is zero, not merely small, and a
        threshold anywhere below the measured spread catches a collapse. The
        probes are also bit-identical across separate constructions, so there
        is no run-to-run drift for a tighter bound to hide in.
        """
        spread = float(np.abs(golden_output[0] - golden_output[1]).max())
        assert spread > 1e-6, (
            f"the two rows differ by only {spread:.3e}; the probe is not "
            "discriminating between inputs"
        )

    def test_the_probe_is_reproducible_within_a_process(
        self, golden_output, golden_reference_device
    ):
        """Two identical builds must agree bit-exactly.

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
