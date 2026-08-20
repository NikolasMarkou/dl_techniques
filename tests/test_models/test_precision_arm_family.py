"""
The R-088 / R-141 mixed-precision arm, for every charged ``models/`` package.

Rules R-088 ("a mixed-precision arm exists") and R-141 ("that arm has all four
parts") were charged against ~55 of the 73 ``models/`` test directories. The
four parts and the reason each is not optional are argued once, in
``precision_arm_oracle.py``; the per-package build/input pairs live once, in
``precision_arm_subjects.py``. This file is only the parameterization that
joins them, plus the guard that keeps the table complete.

What this family FOUND, not just instrumented
---------------------------------------------
Writing the arm was not bookkeeping. Six packages were NOT green and five of
those were product defects, each fixed at its source and each with its own
``test_precision_arm.py`` regression pin in the package's own test directory:

======================  =============================================
package                 defect (decisions.md)
======================  =============================================
``SAM``                 float32 ``pixel_mean`` / ``pixel_std``  D-063
``qwen`` (MoE layer)    ``ops.one_hot`` returns float32        D-064
``som``                 autocast variable vs float32 grid      D-062
``shgcn`` (Poincare)    single-``where`` NaN gradient          D-061
``superpoint``          no float16 ``ResizeBicubic`` gradient  D-060
======================  =============================================

Two further readings were REVERSED by the instrument's own controls and are
recorded as instrument defects, not model defects: a plain ``mean(square(.))``
loss has an exactly-zero gradient on a uniform softmax (D-059), and an
untrained BatchNorm makes an inference-mode fp16 arm measure the initializer
rather than the dtype (D-065).
"""

import pytest

from .precision_arm_oracle import assert_precision_arm
from .precision_arm_subjects import CHARGED_PACKAGES, SUBJECTS, subject_names


def test_every_charged_package_has_a_subject():
    """The registry must cover exactly the charged set -- no silent dropouts.

    Without this, deleting a ``_sub(...)`` line would delete a package's arm
    and the suite would stay green with one fewer test, which is the failure
    mode this whole family exists to prevent.
    """
    missing = sorted(set(CHARGED_PACKAGES) - set(SUBJECTS))
    extra = sorted(set(SUBJECTS) - set(CHARGED_PACKAGES))
    assert not missing, f"charged packages with no precision arm: {missing}"
    assert not extra, f"subjects not in CHARGED_PACKAGES: {extra}"


@pytest.mark.parametrize("name", subject_names())
def test_the_package_runs_under_mixed_float16(name):
    """All four parts of the arm, for one package."""
    build, make_inputs, kwargs = SUBJECTS[name]
    reports = assert_precision_arm(build=build, make_inputs=make_inputs,
                                   **kwargs)
    # Anti-vacuity: the arm must really have run under the policy it names.
    # `assert_precision_arm` asserts this too; repeating it here means a future
    # edit that loosens the oracle cannot make this file silently trivial.
    assert reports["mixed_float16"]["model_policy"] == "mixed_float16"
    assert reports["float32"]["model_policy"] == "float32"
