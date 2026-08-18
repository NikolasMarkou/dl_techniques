"""ConvNeXt's documented `pretrained=<local path>` call must work at the default
config, and a genuine failure must still name its cause.

Guard for C-13 (plan-2026-08-14T233721-d4f9beb2, step 35).

``load_pretrained_weights`` built its pre-load dummy forward as
``keras.random.normal((1,) + tuple(self.input_shape))``, while the default
``input_shape`` is ``(None, None, 3)`` and a freshly constructed subclassed model
is not built. ``create_convnext_v1("tiny", pretrained="/path/weights.keras")`` --
the exact call in the factory's own docstring -- therefore tried
``(1, None, None, 3)``, raised, and the surrounding ``except Exception``
re-raised it as ``ValueError: Failed to load weights from ...``, hiding the cause.
It worked only if the caller also passed a concrete ``input_shape``.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.convnext.convnext_v1 import ConvNeXtV1, create_convnext_v1
from dl_techniques.models.convnext.convnext_v2 import ConvNeXtV2, create_convnext_v2


TINY_KW = dict(depths=[1, 1], dims=[8, 16], num_classes=4)

#: The probe every load is judged on. Fixed, not random, so donor and loaded
#: model are compared on IDENTICAL bits.
PROBE = np.linspace(-1.0, 1.0, 1 * 32 * 32 * 3, dtype="float32").reshape(1, 32, 32, 3)


def _write_checkpoint(tmp_path, cls, name):
    """A real .keras file plus the donor's OUTPUT on :data:`PROBE`.

    Four tests in this file used to assert only ``model.built`` after loading.
    That is true regardless of whether any weight was transferred, because
    ``load_pretrained_weights`` runs a dummy forward -- which builds the model --
    BEFORE it loads anything. The donor was constructed, built and saved, and
    then never compared to anything. Returning its logits here makes the
    comparison possible; the shape is copied from
    ``test_gpt2/test_gpt2.py::TestPretrainedLocalPathIsVerified::test_matching_checkpoint_still_loads``.
    """
    donor = cls(input_shape=(32, 32, 3), **TINY_KW)
    donor(PROBE, training=False)
    path = str(tmp_path / f"{name}.keras")
    donor.save(path)
    expected = keras.ops.convert_to_numpy(donor(PROBE, training=False))
    return path, expected


def _assert_reproduces_donor(model, expected):
    """The loaded model must reproduce the donor's logits, not merely be built."""
    actual = keras.ops.convert_to_numpy(model(PROBE, training=False))
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "cls,factory,name",
    [
        (ConvNeXtV1, create_convnext_v1, "v1"),
        (ConvNeXtV2, create_convnext_v2, "v2"),
    ],
)
class TestLocalPathLoadAtTheDefaultShape:
    def test_load_at_the_default_none_spatial_input_shape(self, cls, factory, name, tmp_path):
        """The failing call: default input_shape, local weights path."""
        ckpt, expected = _write_checkpoint(tmp_path, cls, name)
        model = cls(**TINY_KW)  # default input_shape == (None, None, 3)
        assert model.input_shape == (None, None, 3)
        model.load_pretrained_weights(ckpt, skip_mismatch=True)
        assert model.built
        _assert_reproduces_donor(model, expected)

    def test_explicit_input_shape_still_works(self, cls, factory, name, tmp_path):
        """Anti-vacuity: the path that already worked must keep working."""
        ckpt, expected = _write_checkpoint(tmp_path, cls, name)
        model = cls(input_shape=(32, 32, 3), **TINY_KW)
        model.load_pretrained_weights(ckpt, skip_mismatch=True)
        assert model.built
        _assert_reproduces_donor(model, expected)

    def test_a_genuine_failure_names_its_cause(self, cls, factory, name, tmp_path):
        """A corrupt file must not be reported as an unrelated shape problem:
        the raised message has to carry the underlying exception."""
        bad = tmp_path / f"{name}-corrupt.keras"
        bad.write_bytes(b"not a keras archive")
        model = cls(**TINY_KW)
        with pytest.raises(ValueError) as excinfo:
            model.load_pretrained_weights(str(bad))
        message = str(excinfo.value)
        assert "Failed to load weights" in message
        assert excinfo.value.__cause__ is not None, (
            "the original exception was swallowed instead of chained"
        )
        assert "None" not in message.split("Failed to load weights")[-1].split(":")[0]


def test_factory_docstring_call_v1(tmp_path):
    """`create_convnext_v1("tiny", pretrained="path/to/weights.keras")` verbatim."""
    donor = create_convnext_v1("tiny", num_classes=4, input_shape=(32, 32, 3))
    expected = keras.ops.convert_to_numpy(donor(PROBE, training=False))
    path = str(tmp_path / "convnext_tiny.keras")
    donor.save(path)

    model = create_convnext_v1("tiny", num_classes=4, pretrained=path)
    assert model.built
    _assert_reproduces_donor(model, expected)


def test_factory_docstring_call_v2(tmp_path):
    donor = create_convnext_v2("atto", num_classes=4, input_shape=(32, 32, 3))
    expected = keras.ops.convert_to_numpy(donor(PROBE, training=False))
    path = str(tmp_path / "convnext_v2_atto.keras")
    donor.save(path)

    model = create_convnext_v2("atto", num_classes=4, pretrained=path)
    assert model.built
    _assert_reproduces_donor(model, expected)
