"""Guard on the ``activation``-in-``get_config`` repair (N-1, decisions.md D-400).

Why this file uses an UNREGISTERED callable
===========================================

MEASURED at HEAD 2026-08-23 on a minimal reproduction, then reproduced on the
real classes exercised below:

============================  ================  =====================  ==========
``activation`` passed in       forward ``max
                               |delta|`` after
                               save->load        ``get_config()``       ``.activation``
                                                                        after load
============================  ================  =====================  ==========
``"gelu"`` (a string)          0.0               JSON-safe              ``str``
a **registered** callable      0.0               **NOT JSON-safe**      raw ``dict``
an **unregistered** callable   load **RAISES** unless ``custom_objects`` is given
============================  ================  =====================  ==========

Two traps follow, and both have bitten a previous guard in this family:

* **The forward delta is 0.0 in every configuration, broken or not.** A
  save/load/compare test that only asserts ``max|delta| == 0.0`` passes with the
  repair fully reverted. It is kept below because it must hold, not because it
  discriminates.
* **A registered callable round-trips.** Keras' generic object encoder resolves
  it, so a guard written with ``keras.activations.gelu`` or with a
  ``@register_keras_serializable`` function is vacuous on the load-time raise.
  Only ``_unregistered_activation`` -- deliberately NOT registered -- reaches it.

The two discriminating observables, and the RED-proof
=====================================================

The repair is a pair, and each half was reverted separately and MEASURED to
fail a **different** assertion:

* revert ``serialize_activation`` in ``get_config`` ->
  ``test_get_config_is_json_serializable`` fails
  (``TypeError: Object of type function is not JSON serializable``); the loaded
  attribute is still fine and the delta is still 0.0.
* revert ``deserialize_activation`` in ``__init__`` ->
  ``test_the_loaded_activation_is_still_callable`` fails for all six classes
  (the attribute comes back a raw ``{'class_name': 'function', ...}`` dict);
  ``get_config()`` is still JSON-safe. The forward delta survives for five of
  the six -- only ``MDNLayer``, which *calls* ``keras.activations.get`` on the
  attribute inside ``call``, degrades far enough to raise. Do not read that one
  incidental failure as the delta being a discriminator.

Neither half-revert is detected by the delta alone, and neither is detected by
the assertion the other one fires. That separation is the point of splitting
them into two named tests.
"""

import json
import os
from typing import Any, Callable, Tuple

import keras
import numpy as np
import pytest

from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)

# ---------------------------------------------------------------------


def _unregistered_activation(x):
    """An activation deliberately NOT registered with Keras.

    Do not decorate this with ``@keras.saving.register_keras_serializable``.
    Registration is exactly what makes the defect invisible: a registered
    callable is resolved by Keras' generic encoder and round-trips even with the
    repair reverted.
    """
    return keras.ops.tanh(x) * 1.3


CUSTOM_OBJECTS = {"_unregistered_activation": _unregistered_activation}


# ---------------------------------------------------------------------
# Unit level: the helper pair itself.
# ---------------------------------------------------------------------


class TestTheHelperPair:

    @pytest.mark.parametrize(
        "value",
        ["gelu", "mish", "sparsemax", None, True, False, 3],
        ids=["gelu", "mish", "sparsemax", "none", "true", "false", "int"],
    )
    def test_non_callables_pass_through_both_directions_unchanged(self, value):
        """Strings must survive verbatim.

        ``keras.activations.serialize`` REJECTS a bare string, and ``'mish'`` /
        ``'sparsemax'`` are dl_techniques activation-FACTORY keys that are not
        Keras activations at all -- routing them through
        ``keras.activations.deserialize`` would either raise or silently swap the
        factory key for a Keras function. ``True``/``3`` cover the sites where
        ``activation`` is a bool flag or an unrelated scalar.
        """
        assert serialize_activation(value) is value
        assert deserialize_activation(value) is value

    def test_an_unregistered_callable_round_trips_to_the_same_function(self):
        blob = serialize_activation(_unregistered_activation)
        json.dumps(blob)  # must not raise
        with keras.utils.custom_object_scope(CUSTOM_OBJECTS):
            restored = deserialize_activation(blob)
        assert restored is _unregistered_activation

    def test_a_keras_builtin_serializes_to_its_plain_name(self):
        assert serialize_activation(keras.activations.gelu) == "gelu"

    def test_an_already_live_callable_survives_deserialize_unchanged(self):
        """``deserialize_activation`` runs in ``__init__``, so it sees both the
        config dict (on reload) and the live callable (on ordinary
        construction). The second must be a no-op."""
        assert deserialize_activation(_unregistered_activation) is _unregistered_activation


# ---------------------------------------------------------------------
# Integration level: real classes, one per package family.
# ---------------------------------------------------------------------


def _hierarchical_mlp_stem(activation):
    from dl_techniques.layers.hierarchical_mlp_stem import HierarchicalMLPStem
    return HierarchicalMLPStem(
        embed_dim=16, img_size=(16, 16), patch_size=(4, 4), in_channels=3,
        activation=activation,
    )


def _mdn_layer(activation):
    from dl_techniques.layers.statistics.mdn_layer import MDNLayer
    return MDNLayer(
        output_dimension=2, num_mixtures=2, intermediate_activation=activation,
    )


def _gated_linear_attention_block(activation):
    from dl_techniques.layers.transformers.gated_linear_attention_block import (
        GatedLinearAttentionBlock,
    )
    return GatedLinearAttentionBlock(
        dim=16, num_heads=2, max_seq_len=8, activation=activation,
    )


def _temporal_block(activation):
    from dl_techniques.layers.time_series.temporal_convolutional_network import (
        TemporalBlock,
    )
    return TemporalBlock(
        filters=8, kernel_size=2, dilation_rate=1, activation=activation,
    )


def _mdn_model(activation):
    from dl_techniques.models.time_series.mdn.model import MDNModel
    return MDNModel(
        hidden_layers=[8], output_dimension=2, num_mixtures=2,
        hidden_activation=activation,
    )


def _vit(activation):
    from dl_techniques.models.vit.model import ViT
    return ViT(
        input_shape=(16, 16, 3), num_classes=3, scale="tiny", patch_size=4,
        activation=activation,
    )


#: ``(id, factory, attribute-name, input-shape, is_model)``. Six classes across
#: five packages -- ``layers/``, ``layers/statistics``, ``layers/transformers``,
#: ``layers/time_series``, ``models/time_series`` and ``models/vit`` (the last is
#: one of D-205's three, migrated onto the shared helper by D-400).
CASES = [
    ("HierarchicalMLPStem", _hierarchical_mlp_stem, "activation", (2, 16, 16, 3), False),
    ("MDNLayer", _mdn_layer, "intermediate_activation", (4, 6), False),
    ("GatedLinearAttentionBlock", _gated_linear_attention_block, "activation", (2, 8, 16), False),
    ("TemporalBlock", _temporal_block, "activation", (2, 8, 4), False),
    ("MDNModel", _mdn_model, "hidden_activation", (4, 6), True),
    ("ViT", _vit, "activation", (2, 16, 16, 3), True),
]
CASE_IDS = [case[0] for case in CASES]


def _build(factory: Callable[[Any], Any], activation: Any, shape: Tuple[int, ...],
           is_model: bool):
    """Return ``(saveable_model, config_owner, input_batch, reference_output)``."""
    keras.utils.set_random_seed(0)
    x = np.random.RandomState(0).randn(*shape).astype("float32")
    if is_model:
        owner = factory(activation)
        model = owner
    else:
        owner = factory(activation)
        inputs = keras.Input(shape=shape[1:])
        model = keras.Model(inputs, owner(inputs))
    return model, owner, x, np.asarray(model(x))


def _reload(model, tmp_path, custom_objects):
    path = os.path.join(str(tmp_path), "m.keras")
    model.save(path)
    return keras.models.load_model(path, custom_objects=custom_objects)


def _owner_of(reloaded, name, is_model):
    if is_model:
        return reloaded
    for layer in reloaded.layers:
        if hasattr(layer, name) and not isinstance(layer, keras.layers.InputLayer):
            return layer
    raise AssertionError(f"no reloaded layer carries `{name}`")


@pytest.mark.parametrize("name,factory,attr,shape,is_model", CASES, ids=CASE_IDS)
class TestAnUnregisteredCallableActivationSurvivesSaveLoad:

    def test_get_config_is_json_serializable(self, name, factory, attr, shape,
                                             is_model):
        """RED when ``serialize_activation`` is removed from ``get_config``.

        MEASURED failure without it: ``TypeError: Object of type function is not
        JSON serializable``. This assertion, not the forward delta, is what the
        ``get_config`` half of the pair is proven by.
        """
        _, owner, _, _ = _build(factory, _unregistered_activation, shape, is_model)
        json.dumps(owner.get_config())

    def test_the_forward_output_is_bit_identical_after_reload(
            self, name, factory, attr, shape, is_model, tmp_path):
        """Necessary, NOT sufficient.

        This delta is 0.0 with the repair fully reverted too (measured). It is
        here to catch a repair that silently swaps the activation for a
        different function, which the two discriminating assertions would not
        see.
        """
        model, _, x, y0 = _build(factory, _unregistered_activation, shape, is_model)
        reloaded = _reload(model, tmp_path, CUSTOM_OBJECTS)
        assert float(np.max(np.abs(np.asarray(reloaded(x)) - y0))) == 0.0

    def test_the_loaded_activation_is_still_callable(
            self, name, factory, attr, shape, is_model, tmp_path):
        """RED when ``deserialize_activation`` is removed from ``__init__``.

        MEASURED failure without it: the attribute comes back a ``TrackedDict``
        (``{'module': ..., 'class_name': 'function', ...}``) which the next
        ``get_config()`` then propagates onward, so the corruption compounds
        across successive save/load cycles.
        """
        model, _, _, _ = _build(factory, _unregistered_activation, shape, is_model)
        reloaded = _reload(model, tmp_path, CUSTOM_OBJECTS)
        restored = getattr(_owner_of(reloaded, attr, is_model), attr)
        assert not isinstance(restored, dict), (
            f"{name}.{attr} came back a raw config dict, not an activation: "
            f"{restored!r}"
        )
        assert callable(restored), f"{name}.{attr} is {type(restored).__name__}"

    def test_the_default_string_activation_is_the_control(
            self, name, factory, attr, shape, is_model, tmp_path):
        """The control: a string must be untouched in both directions.

        A repair that routed strings through ``keras.activations.serialize``
        would raise here, and one that routed them through
        ``keras.activations.deserialize`` would silently swap a dl_techniques
        factory key for a Keras function.
        """
        model, owner, x, y0 = _build(factory, "gelu", shape, is_model)
        json.dumps(owner.get_config())
        assert getattr(owner, attr) == "gelu"
        reloaded = _reload(model, tmp_path, None)
        assert float(np.max(np.abs(np.asarray(reloaded(x)) - y0))) == 0.0
        assert getattr(_owner_of(reloaded, attr, is_model), attr) == "gelu"
