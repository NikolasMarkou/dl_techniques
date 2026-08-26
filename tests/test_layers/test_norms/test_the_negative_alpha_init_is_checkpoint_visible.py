"""Step 5's ``alpha_init_value > 0`` guard is a CHECKPOINT break, pinned deliberately.

`plan-2026-08-25T195813-d5a035ab` DROPPED review-finding B14 on the grounds that it
"breaks every existing `.keras` checkpoint", then shipped a validator tightening in step 5
that breaks a (much narrower) class of checkpoint. `decisions.md` D-014 records why the
asymmetry is acceptable; this module is the part of that record a future reader cannot
skip, because it fails if either half of the claim stops being true.

The two halves, both MEASURED before being written down:

1. **The break is real.** ``get_config()`` writes ``alpha_init_value``, so a config
   produced by a version that accepted a non-positive value cannot be rebuilt. A
   functional model containing ``DynamicTanh(alpha_init_value=-0.5)``, saved from a
   worktree at ``a8a042f53``, fails at HEAD with ``TypeError: <class
   'keras.src.models.functional.Functional'> could not be deserialized properly``, whose
   root cause is this module's ``ValueError``.
2. **The break is narrow.** Only the config's INIT value is validated - never the restored
   ``alpha`` weight. A checkpoint written with the default ``alpha_init_value=0.5`` whose
   *learned* ``alpha`` is ``-0.5`` still loads and reproduces its outputs exactly. This is
   what bounds the exposure, and it is the half most likely to be broken by a
   well-meaning future "validate alpha on load too" change.

The archive is reconstructed from ``get_config()`` rather than committed as a binary
fixture: a `.keras` file checked into the tree would rot against Keras versions, and the
deserialization path under test (``from_config`` -> ``__init__``) is identical either way.
"""

import numpy as np
import keras
import pytest

from dl_techniques.layers.norms.dynamic_tanh import DynamicTanh


def _legacy_config_with(alpha_init_value):
    """Return the config an older version would have serialized for this alpha.

    Built by asking the CURRENT class for a valid config and then overwriting the one
    key, so every other key stays whatever this Keras version actually writes.

    :param alpha_init_value: The non-positive value to plant in the config.
    :type alpha_init_value: float
    :return: A ``DynamicTanh`` config dict carrying that value.
    :rtype: dict
    """
    config = DynamicTanh(alpha_init_value=0.5).get_config()
    assert "alpha_init_value" in config, (
        "get_config() no longer writes alpha_init_value; if that is intentional the "
        "checkpoint exposure this module pins has GONE and D-014 should be revisited."
    )
    config["alpha_init_value"] = alpha_init_value
    return config


@pytest.mark.parametrize("alpha_init_value", [-1, -0.5, 0, 0.0])
def test_the_constructor_refuses_a_non_positive_alpha_with_a_value_error(
    alpha_init_value,
):
    """Constructed from source, the guard raises its own ``ValueError``."""
    with pytest.raises(ValueError, match="alpha_init_value must be a positive number"):
        DynamicTanh(alpha_init_value=alpha_init_value)


@pytest.mark.parametrize("alpha_init_value", [-1, -0.5, 0, 0.0])
def test_a_legacy_config_with_a_non_positive_alpha_no_longer_deserializes(
    alpha_init_value,
):
    """The accepted break, and the error a LOADER actually sees.

    MEASURED, and not what one would guess: ``Operation.from_config`` catches the
    ``ValueError`` and re-raises it as a ``TypeError`` ("Error when deserializing class
    'DynamicTanh' ... Exception encountered: <the ValueError's message>"). Pinning
    ``ValueError`` here would be pinning the constructor, not the load path.
    """
    config = _legacy_config_with(alpha_init_value)
    with pytest.raises(TypeError) as excinfo:
        DynamicTanh.from_config(config)
    assert "alpha_init_value must be a positive number" in str(excinfo.value)
    assert "DynamicTanh" in str(excinfo.value)


def test_the_refusal_survives_a_full_keras_deserialization_round_trip():
    """The same refusal on the generic ``deserialize_keras_object`` path."""
    serialized = keras.saving.serialize_keras_object(
        DynamicTanh(alpha_init_value=0.5)
    )
    serialized["config"]["alpha_init_value"] = -0.5
    with pytest.raises(TypeError) as excinfo:
        keras.saving.deserialize_keras_object(serialized)
    assert "alpha_init_value must be a positive number" in str(excinfo.value)


def test_a_negative_LEARNED_alpha_still_saves_and_loads_exactly(tmp_path):
    """The bound on the exposure: only the INIT value is validated, never the weight.

    If this test ever fails, the checkpoint exposure is far wider than D-014 assumes and
    the guard must be reconsidered - not this test relaxed.
    """
    inputs = keras.Input(shape=(8,))
    layer = DynamicTanh(alpha_init_value=0.5)
    model = keras.Model(inputs, layer(inputs))

    layer.alpha.assign(keras.ops.convert_to_tensor(-0.5, dtype=layer.alpha.dtype))

    x = np.random.default_rng(0).normal(size=(2, 8)).astype("float32")
    before = keras.ops.convert_to_numpy(model(x))

    path = tmp_path / "negative_learned_alpha.keras"
    model.save(path)
    reloaded = keras.saving.load_model(path)

    after = keras.ops.convert_to_numpy(reloaded(x))
    np.testing.assert_array_equal(before, after)

    restored_alpha = [w for w in reloaded.weights if "alpha" in w.name]
    assert len(restored_alpha) == 1
    assert float(keras.ops.convert_to_numpy(restored_alpha[0])) == -0.5
