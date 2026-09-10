"""R-088 / R-141 / R-142 precision arms for ``doc_res``.

Two arms, both on the shared ``precision_arm_oracle``:

* ``mixed_float16`` -- all four parts (forward runs, every float output carries
  the compute dtype, the values stay finite, the backward produces a gradient
  for every variable), with a float32 control.
* ``float64`` -- the policy took, the INPUT took, the OUTPUT took. R-142's
  charge is that five packages named ``float64`` in a test while never calling
  ``set_floatx``, so the "two different precisions" agreed because they were
  the same precision.

Why this model is a real fp16 subject rather than a formality
-------------------------------------------------------------
MDTA L2-normalises over the flattened spatial axis, and the reference divides
by ``sqrt(sum + 1e-12)``. ``1e-12`` is BELOW the float16 subnormal floor
(``6e-08``), so a literal epsilon underflows to exactly 0 in half precision and
the normalisation divides by zero for an all-zero row. The layer uses
``dl_techniques.utils.dtype_policy.stability_floor(compute_dtype, 1e-12)``
instead, which is the mechanism this arm exercises end-to-end through the
assembled model rather than through the layer alone -- ``test_mdta.py`` covers
the layer.

The arms live in their own file, beside ``darkir``'s and ``scunet``'s, because
that is where the family looks for a package-specific precision arm; the
package is also a member of the shared round-trip family through
``precision_arm_subjects._b_doc_res``.
"""

import numpy as np

from ..precision_arm_oracle import assert_float64_arm, assert_precision_arm

# The same geometry as ``test_model.py``'s subject and for the same reasons:
# `heads=[1, 2, 4, 8]` keeps the multi-head path live, and 32x32 is the
# smallest spatial size that survives three downsampling stages (D-016).
_SMALL = dict(dim=8, num_blocks=[1, 1, 1, 1], num_refinement_blocks=1,
              heads=[1, 2, 4, 8])


def _build():
    from dl_techniques.models.vision.image_restoration.doc_res.model import DocRes
    return DocRes(**_SMALL)


def _inputs():
    return np.random.RandomState(0).randn(1, 32, 32, 6).astype("float32")


def test_doc_res_runs_under_mixed_float16():
    """All four parts, with a float32 control at a half-precision tolerance.

    ``rtol_against_float32=2e-2`` is half precision's own resolution
    (``eps_f16 = 9.77e-04``) carried through a 30-layer-deep residual stack,
    not a tolerance widened until the test passed; ``darkir``, the other
    restoration U-Net in this family, uses the same value.
    """
    reports = assert_precision_arm(
        build=_build,
        make_inputs=_inputs,
        rtol_against_float32=2e-2,
    )
    # One output tensor, and it must really be float16 -- part 2 of the arm is
    # the one that catches a hard-coded ``dtype="float32"`` island.
    assert reports["mixed_float16"]["dtypes"] == ["float16"]
    # No waived ``None`` gradient: every variable is on the fp16 backward
    # graph, exactly as it is on the float32 one.
    assert reports["backward_mixed_float16"]["n_none"] == 0


def test_doc_res_runs_under_float64():
    """The scope, the input and the output are all really ``float64``."""
    report = assert_float64_arm(_build, _inputs, training=False)
    assert report["dtypes"] == ["float64"]
    assert sum(report["n_nan"]) == 0 and sum(report["n_inf"]) == 0
