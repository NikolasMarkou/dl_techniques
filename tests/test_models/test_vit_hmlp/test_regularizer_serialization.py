"""``kernel_regularizer``/``bias_regularizer`` must be resolved eagerly in
``__init__``, mirroring ``kernel_initializer``/``bias_initializer`` -- a real
functional regression found by adversarial review of the D-006 migration
(see decisions.md D-013).

The deleted ``from_config`` override used to run ``regularizers.deserialize()``
on these two constructor args. ``__init__`` only replicated the initializer
half after that migration, so ``model.kernel_regularizer``/``bias_regularizer``
came back as a raw, unresolved config dict (``TrackedDict``) after a
``from_config`` round trip -- sublayers still applied the correct L2 penalty
via their own internal ``regularizers.get()`` calls, but the model's own
stored attribute had the wrong type.

RED proof: revert the ``regularizers.get(...)`` calls in ``__init__`` back to
storing the raw argument and this test fails with
``type(restored.kernel_regularizer) is not keras.regularizers.L2``.
"""

import keras

from dl_techniques.models.vision.vit_hmlp.model import ViTHMLP


def _build(kernel_regularizer, bias_regularizer=None):
    keras.utils.set_random_seed(1234)
    return ViTHMLP(
        input_shape=(32, 32, 3),
        num_classes=4,
        scale="tiny",
        patch_size=16,
        include_top=False,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
    )


class TestViTHMLPRegularizerSerialization:

    def test_kernel_regularizer_is_a_resolved_instance_before_any_round_trip(self):
        model = _build(keras.regularizers.L2(0.01))
        assert isinstance(model.kernel_regularizer, keras.regularizers.L2)

    def test_from_config_restores_kernel_regularizer_as_a_resolved_instance(self):
        model = _build(keras.regularizers.L2(0.01))
        restored = ViTHMLP.from_config(model.get_config())
        assert isinstance(restored.kernel_regularizer, keras.regularizers.L2), (
            f"kernel_regularizer came back as "
            f"{type(restored.kernel_regularizer).__name__} "
            f"{restored.kernel_regularizer!r}, not a resolved L2 instance"
        )
        assert restored.kernel_regularizer.l2 == 0.01

    def test_from_config_restores_bias_regularizer_as_a_resolved_instance(self):
        model = _build(
            kernel_regularizer=None, bias_regularizer=keras.regularizers.L2(0.02)
        )
        restored = ViTHMLP.from_config(model.get_config())
        assert isinstance(restored.bias_regularizer, keras.regularizers.L2), (
            f"bias_regularizer came back as "
            f"{type(restored.bias_regularizer).__name__} "
            f"{restored.bias_regularizer!r}, not a resolved L2 instance"
        )
        assert restored.bias_regularizer.l2 == 0.02

    def test_none_regularizer_still_round_trips_as_none(self):
        # Control: the default path must not be disturbed by the fix.
        model = _build(kernel_regularizer=None, bias_regularizer=None)
        restored = ViTHMLP.from_config(model.get_config())
        assert restored.kernel_regularizer is None
        assert restored.bias_regularizer is None
