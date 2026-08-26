"""B7: the factory validator and the class validator must agree in BOTH directions.

`validate_normalization_config(t, **kw)` is documented as the pre-flight check for
`create_normalization_layer(t, **kw)`. At HEAD it disagreed with the class in both
directions for two parameters:

- ``decoupled_max_logit`` / ``constant``: the factory checked only that the value is a
  number, while ``DecoupledMaxLogit._validate_inputs`` requires ``constant > 0``. So
  ``validate_normalization_config('decoupled_max_logit', constant=-1)`` returned ``True``
  and the ``ValueError`` only surfaced later, from inside the class.
- ``dynamic_tanh`` / ``alpha_init_value``: the factory required
  ``alpha_init_value > 0`` while ``DynamicTanh.__init__`` checked only ``isinstance``.
  So ``DynamicTanh(alpha_init_value=-0.5)`` constructed happily and produced
  ``tanh(-0.5 * x)`` (and at ``0.0``, the constant-zero layer ``tanh(0 * x)``), a
  configuration the factory refuses.

`test_review_pass2.py::TestFactoryValidateCreateAgreement` already pins the
``dynamic_tanh`` extra-params and ``max_band_width`` bounds direction; this module adds
the two remaining sign checks and pins the message wording so the two sites read
identically.
"""

import pytest

from dl_techniques.layers.norms.dynamic_tanh import DynamicTanh
from dl_techniques.layers.norms.max_logit_norm import DecoupledMaxLogit
from dl_techniques.layers.norms.factory import (
    validate_normalization_config,
    create_normalization_layer,
)


# The two non-positive values that matter: a negative value flips the sign of the
# transform, and exactly 0.0 makes DynamicTanh the constant-zero map (tanh(0 * x) == 0).
NON_POSITIVE = [-1, -0.5, 0, 0.0]


def _validate_accepts(ntype, **kw):
    """True if the factory-level validator accepts the config."""
    try:
        validate_normalization_config(ntype, **kw)
        return True
    except ValueError:
        return False


def _create_accepts(ntype, **kw):
    """True if the factory can actually construct the layer."""
    try:
        create_normalization_layer(ntype, **kw)
        return True
    except Exception:
        return False


class TestConstantSignAgreement:
    """Direction 1: the CLASS already had ``constant > 0``; the factory did not."""

    @pytest.mark.parametrize("constant", NON_POSITIVE)
    def test_validate_rejects_non_positive_constant(self, constant):
        """RED at HEAD: validate_normalization_config returned True for constant=-1."""
        with pytest.raises(ValueError, match="constant must be positive"):
            validate_normalization_config('decoupled_max_logit', constant=constant)

    @pytest.mark.parametrize("constant", NON_POSITIVE)
    def test_class_rejects_non_positive_constant(self, constant):
        """Control (GREEN at HEAD): the class-side rule this pass copies upward."""
        with pytest.raises(ValueError, match="constant must be positive"):
            DecoupledMaxLogit(constant=constant)

    @pytest.mark.parametrize("constant", NON_POSITIVE)
    def test_validate_and_create_agree_on_non_positive_constant(self, constant):
        """Both sides must REFUSE — the disagreement itself is the defect."""
        assert _validate_accepts('decoupled_max_logit', constant=constant) is False
        assert _create_accepts('decoupled_max_logit', constant=constant) is False

    @pytest.mark.parametrize("constant", [0.8, 1, 1.0, 3.5])
    def test_a_positive_constant_is_still_accepted_by_both(self, constant):
        """The tightening must not reject the live, positive call sites."""
        assert validate_normalization_config('decoupled_max_logit', constant=constant) is True
        assert _create_accepts('decoupled_max_logit', constant=constant) is True

    def test_a_non_numeric_constant_still_reports_the_type_error(self):
        """The pre-existing type check keeps its own distinct message."""
        with pytest.raises(ValueError, match="constant must be a number"):
            validate_normalization_config('decoupled_max_logit', constant="big")


class TestAlphaInitValueSignAgreement:
    """Direction 2: the FACTORY already had ``alpha_init_value > 0``; the class did not."""

    @pytest.mark.parametrize("alpha", NON_POSITIVE)
    def test_class_rejects_non_positive_alpha(self, alpha):
        """RED at HEAD: DynamicTanh(alpha_init_value=-0.5) constructed successfully."""
        with pytest.raises(ValueError, match="alpha_init_value must be a positive number"):
            DynamicTanh(alpha_init_value=alpha)

    @pytest.mark.parametrize("alpha", NON_POSITIVE)
    def test_validate_rejects_non_positive_alpha(self, alpha):
        """Control (GREEN at HEAD): the factory-side rule this pass copies downward."""
        with pytest.raises(ValueError, match="alpha_init_value must be a positive number"):
            validate_normalization_config('dynamic_tanh', alpha_init_value=alpha)

    @pytest.mark.parametrize("alpha", NON_POSITIVE)
    def test_validate_and_create_agree_on_non_positive_alpha(self, alpha):
        assert _validate_accepts('dynamic_tanh', alpha_init_value=alpha) is False
        assert _create_accepts('dynamic_tanh', alpha_init_value=alpha) is False

    @pytest.mark.parametrize("alpha", [1e-6, 0.1, 0.5, 0.7, 2.0])
    def test_a_positive_alpha_is_still_accepted_by_both(self, alpha):
        """Every live call site in src/ and tests/ passes a positive value."""
        assert validate_normalization_config('dynamic_tanh', alpha_init_value=alpha) is True
        assert _create_accepts('dynamic_tanh', alpha_init_value=alpha) is True
        assert DynamicTanh(alpha_init_value=alpha).alpha_init_value == pytest.approx(alpha)

    @pytest.mark.parametrize("bad", ["invalid", None, [0.5]])
    def test_a_non_numeric_alpha_still_reports_the_type_error(self, bad):
        """The pre-existing type check keeps its own distinct message (pinned by
        ``test_dynamic_tanh.py`` and reached BEFORE the new sign check)."""
        with pytest.raises(ValueError, match="alpha_init_value must be a number"):
            DynamicTanh(alpha_init_value=bad)


class TestValidateStillReturnsTrue:
    """`validate_normalization_config` returns True or raises. Do not change that
    contract — live callers test its return value."""

    def test_return_value_is_literally_true(self):
        assert validate_normalization_config('dynamic_tanh', alpha_init_value=0.5) is True
        assert validate_normalization_config('decoupled_max_logit', constant=1.0) is True
