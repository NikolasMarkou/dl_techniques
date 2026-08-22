"""``get_config()`` must emit a serializable ``activation``, and ``from_config``
must undo it -- ViT sibling of the ``vit_siglip`` defect (D-012, N-1).

Measured before the fix (D-205), with the config value stored raw:

- the default ``"gelu"``: round-trips bit-identically. **The string arm is a
  CONTROL, not the claim** -- the defect is latent for every shipped config.
- a REGISTERED callable also already round-tripped, because Keras's generic
  object encoder resolves it. **A guard built on a registered callable is
  VACUOUS** and would ship green while proving nothing. Kept below only as a
  control.
- ``json.dumps(get_config()["activation"])`` for a callable:
  ``TypeError: Object of type function is not JSON serializable``.
- an **UNREGISTERED** callable -- the only arm that discriminates -- saves fine
  and then RAISES at ``keras.models.load_model``:
  ``ValueError: Could not interpret activation function identifier: {...}``.
  A save-side check cannot see a load-side loss.
- ``ViT`` had **no ``from_config`` at all**, so there was nothing to undo it with.

The repair is a SYMMETRIC pair: ``keras.activations.serialize`` in
``get_config``, ``keras.activations.deserialize`` in ``from_config``. One side
alone breaks the other direction.
"""

import json

import keras
import numpy as np
import pytest

from dl_techniques.models.vit.model import ViT


@keras.saving.register_keras_serializable(package="dl_techniques_test")
def registered_scaled_relu(x):
    """A callable activation that IS registered -- the vacuous-guard control."""
    return keras.ops.relu(x) * 0.5


def unregistered_scaled_relu(x):
    """A callable activation that is NOT registered anywhere.

    This is the discriminating arm. Step 12 MEASURED that a registered callable
    already round-trips, so a guard using one passes with and without the fix.
    """
    return keras.ops.relu(x) * 0.5


def _build(activation):
    keras.utils.set_random_seed(1234)
    return ViT(
        input_shape=(32, 32, 3),
        num_classes=4,
        scale="tiny",
        patch_size=16,
        include_top=False,
        activation=activation,
    )


def _inputs():
    return np.random.RandomState(0).randn(2, 32, 32, 3).astype("float32")


def _forward(model, x):
    return np.asarray(keras.ops.convert_to_numpy(model(x, training=False)))


class TestViTActivationSerialization:

    def test_get_config_emits_a_json_safe_activation_for_a_callable(self):
        value = _build(unregistered_scaled_relu).get_config()["activation"]
        json.dumps(value)

    def test_from_config_restores_an_unregistered_callable_after_a_json_hop(self):
        """The config must survive a REAL json hop, not an in-memory dict that
        still holds the live function object."""
        config = _build(unregistered_scaled_relu).get_config()
        config["activation"] = json.loads(json.dumps(config["activation"]))
        restored = ViT.from_config(
            config,
            custom_objects={"unregistered_scaled_relu": unregistered_scaled_relu},
        )
        assert restored.activation is unregistered_scaled_relu, (
            f"activation came back as {restored.activation!r} "
            f"({type(restored.activation).__name__}), not the callable"
        )

    def test_an_unregistered_callable_survives_save_and_load_with_custom_objects(
        self, tmp_path
    ):
        x = _inputs()
        model = _build(unregistered_scaled_relu)
        before = _forward(model, x)

        path = str(tmp_path / "unregistered_activation.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path,
            custom_objects={"unregistered_scaled_relu": unregistered_scaled_relu},
        )
        assert callable(loaded.activation), (
            "the reconstructed activation is not callable: "
            f"{type(loaded.activation).__name__} {loaded.activation!r}"
        )
        delta = float(np.max(np.abs(before - _forward(loaded, x))))
        assert delta == 0.0, f"round trip changed the output: max|delta| = {delta:.6e}"

    def test_a_registered_callable_still_round_trips(self, tmp_path):
        """CONTROL -- passed before the fix too. Proves the fix broke nothing."""
        x = _inputs()
        model = _build(registered_scaled_relu)
        before = _forward(model, x)
        path = str(tmp_path / "registered_activation.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        delta = float(np.max(np.abs(before - _forward(loaded, x))))
        assert delta == 0.0

    def test_the_default_string_activation_still_round_trips_at_zero(self, tmp_path):
        """CONTROL -- the path every shipped config takes must be untouched."""
        x = _inputs()
        model = _build("gelu")
        assert model.get_config()["activation"] == "gelu"
        assert isinstance(model.get_config()["activation"], str)
        before = _forward(model, x)
        path = str(tmp_path / "string_activation.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        delta = float(np.max(np.abs(before - _forward(loaded, x))))
        assert delta == 0.0, f"default 'gelu' path changed: max|delta| = {delta:.6e}"

    @pytest.mark.parametrize("activation", ["gelu", "relu"])
    def test_from_config_leaves_string_activations_as_strings(self, activation):
        restored = ViT.from_config(_build(activation).get_config())
        assert restored.activation == activation
        assert isinstance(restored.activation, str)
