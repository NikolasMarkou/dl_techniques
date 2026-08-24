"""
The two ConvNeXt siblings must agree about the ``compute_output_shape`` contract.

MEASURED before this guard, on the SAME geometry and with both real forwards at
``(1, 4)``:

=============  =====================================================
class          ``compute_output_shape((None, 32, 32, 3))``
=============  =====================================================
``ConvNeXtV2`` ``(None, 4)``
``ConvNeXtV1`` **RAISE** ``NotImplementedError: Layer ConvNeXtV1 does
               not have a compute_output_shape method implemented``
=============  =====================================================

This is the defect class a per-package suite structurally cannot see: each
sibling's tests only ever exercise the sibling they were written for, so an
asymmetry between two classes in ONE package is nobody's test. See decisions.md
D-069.

The assertion is parity AND correctness -- the declared shape must equal the
real forward's shape, not merely exist. A ``compute_output_shape`` that returns
a plausible-looking wrong tuple is worse than one that raises, because Keras
will build a functional graph on it.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.convnext import ConvNeXtV1, ConvNeXtV2

_INPUT = (1, 32, 32, 3)


@pytest.mark.parametrize("cls", [ConvNeXtV1, ConvNeXtV2],
                         ids=["v1", "v2"])
@pytest.mark.parametrize("include_top", [True, False])
def test_the_declared_output_shape_matches_the_real_forward(cls, include_top):
    keras.utils.set_random_seed(0)
    model = cls(num_classes=4, depths=[1, 1], dims=[8, 16],
                include_top=include_top)
    real = tuple(model(np.random.RandomState(0).randn(*_INPUT)
                       .astype("float32"), training=False).shape)
    declared = tuple(model.compute_output_shape((None,) + _INPUT[1:]))
    assert declared[1:] == real[1:], (
        f"{cls.__name__} (include_top={include_top}) declares {declared} but "
        f"really returns {real}")
    assert declared[0] is None, (
        f"{cls.__name__} declared a concrete batch dimension {declared[0]!r}")


def test_both_siblings_define_the_method_themselves():
    """Parity, asserted structurally -- inheritance from Keras would raise."""
    for cls in (ConvNeXtV1, ConvNeXtV2):
        assert "compute_output_shape" in cls.__dict__, (
            f"{cls.__name__} does not define compute_output_shape; its sibling "
            "does, and the two are used interchangeably")


def test_the_parity_check_can_see_a_missing_method():
    """Liveness: a class WITHOUT the method must fail the structural check.

    Without this, ``'compute_output_shape' in cls.__dict__`` could silently
    become true for every class (Keras' base defines one that raises) and the
    test above would pass against the very defect it names.
    """
    class _NoShape(keras.Model):
        pass

    assert "compute_output_shape" not in _NoShape.__dict__
    with pytest.raises(NotImplementedError):
        _NoShape().compute_output_shape((None, 4))
