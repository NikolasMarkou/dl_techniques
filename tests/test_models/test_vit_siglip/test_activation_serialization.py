"""
``get_config()`` must emit a serializable ``activation``, and ``from_config``
must undo it.

Measured at HEAD before the fix, with the config value stored raw:

- default ``"gelu"``: round-trips bit-identically (``max|delta| = 0.0``). The
  defect is latent for every shipped config, so the string arm below is a
  CONTROL, not the claim.
- ``json.dumps(get_config()["activation"])`` for a callable:
  ``TypeError: Object of type function is not JSON serializable``.
- an UNREGISTERED callable: ``model.save()`` SUCCEEDS and
  ``keras.models.load_model()`` RAISES
  ``ValueError: Could not interpret activation function identifier:
  {'module': 'builtins', 'class_name': 'function', 'config': 'my_act',
  'registered_name': 'function'}`` -- a hard failure at LOAD, so a save-side
  check is blind to it.
- the same callable loaded WITH ``custom_objects``: loads, output delta 0.0,
  but ``loaded.activation`` is the raw ``{'module': 'builtins', ...}`` DICT,
  which its own ``get_config`` and ``get_feature_extractor`` then propagate.

A REGISTERED callable already round-tripped at HEAD (Keras's generic object
encoder resolves it), which is why that arm is kept as a control rather than
quoted as the repair's proof.

The repair is a SYMMETRIC pair -- ``activations.serialize`` in ``get_config``,
``activations.deserialize`` in ``from_config``. One side alone breaks the other
direction. It cannot make an unregistered, custom-objects-less callable load;
nothing can. What it fixes is the config's serializability and the type of the
reconstructed attribute.

RED proof: revert the ``activations.serialize`` branch in ``get_config`` and
``test_get_config_emits_a_json_safe_activation_for_a_callable`` fails with that
``TypeError``; revert ``from_config`` and
``test_from_config_restores_a_callable_from_a_json_round_tripped_config`` fails
because the activation comes back as the serialized dict.
"""

import json
import numpy as np
import keras
import pytest

from dl_techniques.models.vit_siglip.model import SigLIPVisionTransformer


@keras.saving.register_keras_serializable(package="dl_techniques_test")
def scaled_relu_activation(x):
    """A genuine callable activation, registered so ``load_model`` can find it."""
    return keras.ops.relu(x) * 0.5


def unregistered_activation(x):
    """A genuine callable activation that is NOT registered anywhere."""
    return keras.ops.relu(x) * 0.5


def _build(activation):
    keras.utils.set_random_seed(1234)
    return SigLIPVisionTransformer(
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


class TestActivationSerialization:
    def test_get_config_emits_a_json_safe_activation_for_a_callable(self):
        # RED at HEAD: TypeError: Object of type function is not JSON serializable.
        cfg_value = _build(scaled_relu_activation).get_config()["activation"]
        json.dumps(cfg_value)

    def test_from_config_restores_a_callable_from_a_json_round_tripped_config(self):
        # The config has to survive a real JSON hop, not just an in-memory dict
        # that still holds the live function object.
        cfg = _build(scaled_relu_activation).get_config()
        cfg["activation"] = json.loads(json.dumps(cfg["activation"]))
        restored = SigLIPVisionTransformer.from_config(cfg)
        assert restored.activation is scaled_relu_activation, (
            f"activation came back as {restored.activation!r}"
        )

    def test_an_unregistered_callable_is_restored_as_a_callable_with_custom_objects(
        self, tmp_path
    ):
        # RED at HEAD: loads, but `loaded.activation` is the raw config dict.
        x = _inputs()
        model = _build(unregistered_activation)
        before = _forward(model, x)

        path = str(tmp_path / "unregistered_activation.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"unregistered_activation": unregistered_activation}
        )
        assert callable(loaded.activation), (
            "the reconstructed activation is not callable: "
            f"{type(loaded.activation).__name__} {loaded.activation!r}"
        )
        assert loaded.activation is unregistered_activation
        delta = float(np.max(np.abs(before - _forward(loaded, x))))
        assert delta == 0.0, f"round trip changed the output: max|delta| = {delta:.6e}"

    def test_a_registered_callable_survives_the_round_trip(self, tmp_path):
        # Control: this arm already passed at HEAD; the fix must not break it.
        x = _inputs()
        model = _build(scaled_relu_activation)
        before = _forward(model, x)

        path = str(tmp_path / "callable_activation.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        delta = float(np.max(np.abs(before - _forward(loaded, x))))
        assert delta == 0.0, (
            f"registered callable did not survive save->load: max|delta| = {delta:.6e}"
        )

    def test_the_default_string_activation_still_round_trips_at_zero(self, tmp_path):
        # Control. The fix must not change the path every shipped config takes.
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
        model = _build(activation)
        restored = SigLIPVisionTransformer.from_config(model.get_config())
        assert restored.activation == activation
        assert isinstance(restored.activation, str)
