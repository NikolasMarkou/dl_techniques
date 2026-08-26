"""``EnergyLayerNorm`` and ``BandLogitNorm`` rely on the BASE ``from_config``.

Both classes used to carry a hand-written ``from_config``. ``BandLogitNorm``'s
was literally ``return cls(**config)`` — the base implementation, retyped.
``EnergyLayerNorm``'s additionally ran ``initializers.deserialize`` /
``constraints.deserialize`` over three keys before the same ``cls(**config)``,
which is redundant because its ``__init__`` already routes every one of those
arguments through ``initializers.get`` / ``constraints.get``, and those accept a
serialized dict directly.

Both were removed by `plan-2026-08-25-d5a035ab/iter-1/step-10` (decisions.md
D-011) only after the measurements below were shown to be BIT-IDENTICAL with and
without the methods. This module is that measurement, committed — so that the
next reader who thinks "a custom layer needs a custom ``from_config``" has to
falsify a running test rather than re-add 35 lines on a hunch.

The dangerous case is ``EnergyLayerNorm``'s ``gamma_constraint``, which has THREE
distinct reload regimes that must stay distinguishable (see the ``get_config``
comment in `energy_layer_norm.py`, and the
``# DECISION plan_2026-07-13_57c9833e/D-010`` anchor in its ``__init__``):

* the ``"__default__"`` string sentinel — a caller who never mentioned the
  argument gets a ``ValueRangeConstraint(min_value=1e-3)`` positivity floor;
* an explicit constraint object — round-trips as that object;
* an explicit ``None`` — round-trips as ``None``, deliberately unconstrained, and
  must NOT silently reacquire the floor.

Reloading regime 3 as regime 1 (or vice versa) would be invisible to a shape or
dtype assertion, so every case here compares the FULL serialized config and the
constraint's own clipping behaviour, not just the class name.
"""

from typing import Any, Callable, Dict, List, Tuple

import json
import os

import keras
import numpy as np
import pytest

from dl_techniques.layers.norms.band_logit_norm import BandLogitNorm
from dl_techniques.layers.norms.energy_layer_norm import EnergyLayerNorm


def _cfg_json(layer: keras.layers.Layer) -> str:
    """Order-independent representation of a layer's CONSTRUCTOR configuration.

    Deliberately ``get_config()`` and not ``serialize_keras_object(layer)``: the
    latter also carries a ``build_config`` key that appears only once a layer has
    been called, so it would compare build state rather than configuration. No
    ``default=`` fallback either — every value here must already be JSON-native
    (that is what ``initializers.serialize`` / ``constraints.serialize`` are for),
    and a stray live object must raise instead of being ``repr``'d into a string
    that silently compares equal or unequal for the wrong reason.
    """
    return json.dumps(layer.get_config(), sort_keys=True)


def _constraint_effect(layer: keras.layers.Layer) -> Any:
    """What the layer's ``gamma_constraint`` actually DOES, or ``None``.

    Comparing serialized configs is not enough on its own: a constraint that was
    restored as its own raw config dict instead of a live object serializes back
    to the SAME dict and so compares equal, while being unusable at training
    time. Calling it is what tells the two apart. Measured: injecting exactly
    that defect (dropping the ``constraints.get`` in ``__init__``) leaves every
    config comparison in this module green.
    """
    constraint = getattr(layer, "gamma_constraint", None)
    if constraint is None:
        return None
    probe = keras.ops.convert_to_tensor(np.array([-5.0, 0.5, 5.0], dtype="float32"))
    return keras.ops.convert_to_numpy(constraint(probe)).tolist()


# Each case builds a layer with NON-DEFAULT arguments. A round trip that only
# ever sees defaults cannot tell a restored value from a re-defaulted one.
CASES: List[Tuple[str, Callable[[], keras.layers.Layer]]] = [
    (
        "energy_default_sentinel_constraint",
        lambda: EnergyLayerNorm(
            epsilon=3e-4,
            gamma_initializer="he_normal",
            delta_initializer="random_normal",
            name="eln_sentinel",
        ),
    ),
    (
        "energy_explicit_gamma_constraint",
        lambda: EnergyLayerNorm(
            epsilon=7e-3,
            gamma_initializer=keras.initializers.Constant(0.75),
            delta_initializer=keras.initializers.Constant(-0.25),
            gamma_constraint=keras.constraints.MaxNorm(2.0),
            name="eln_explicit",
        ),
    ),
    (
        "energy_explicit_none_constraint",
        lambda: EnergyLayerNorm(
            epsilon=1e-2,
            gamma_initializer="ones",
            delta_initializer="zeros",
            gamma_constraint=None,
            name="eln_none",
        ),
    ),
    (
        "band_logit_non_trailing_axis",
        lambda: BandLogitNorm(
            axis=1, epsilon=1e-3, max_band_width=0.37, name="bln_axis1"
        ),
    ),
    (
        "band_logit_trailing_axis",
        lambda: BandLogitNorm(
            axis=-1, epsilon=5e-6, max_band_width=0.05, name="bln_axis_last"
        ),
    ),
]

_IDS = [case[0] for case in CASES]


@pytest.fixture(scope="module")
def sample_input() -> np.ndarray:
    rng = np.random.default_rng(1234)
    return rng.standard_normal((3, 5, 8)).astype("float32")


@pytest.mark.parametrize("_name,maker", CASES, ids=_IDS)
def test_model_save_load_round_trips_without_a_custom_from_config(
    _name: str,
    maker: Callable[[], keras.layers.Layer],
    sample_input: np.ndarray,
    tmp_path: Any,
) -> None:
    """`.keras` round trip: identical config AND bit-identical output."""
    layer = maker()
    inputs = keras.Input(shape=sample_input.shape[1:], dtype="float32")
    model = keras.Model(inputs, layer(inputs))

    before = keras.ops.convert_to_numpy(model(sample_input, training=False))
    config_before = _cfg_json(layer)

    path = os.path.join(str(tmp_path), "model.keras")
    model.save(path)
    restored = keras.models.load_model(path)

    reloaded = next(lay for lay in restored.layers if lay.name == layer.name)
    assert type(reloaded) is type(layer)
    assert _cfg_json(reloaded) == config_before, (
        "the reloaded layer's serialized config differs from the original — the "
        "base `from_config` is NOT sufficient for this class and a hand-written "
        "one must be restored"
    )

    assert _constraint_effect(reloaded) == _constraint_effect(layer)

    after = keras.ops.convert_to_numpy(restored(sample_input, training=False))
    assert float(np.max(np.abs(before - after))) == 0.0


@pytest.mark.parametrize("_name,maker", CASES, ids=_IDS)
def test_direct_from_config_and_registry_deserialize_agree(
    _name: str,
    maker: Callable[[], keras.layers.Layer],
    sample_input: np.ndarray,
) -> None:
    """The two non-`model.save` reload paths must match the original exactly.

    ``cls.from_config(cfg)`` is what a caller reaches for directly;
    ``deserialize_keras_object(serialize_keras_object(layer))`` is what the
    registry path (and any enclosing layer's own ``get_config``) uses. Neither
    goes through the `.keras` archive, so neither is covered by the test above.
    """
    original = maker()
    # Serialize BEFORE the first call: a built layer carries a `build_config`
    # key that a freshly constructed one has not grown yet, and comparing
    # across that difference would measure build state, not configuration.
    config_before = _cfg_json(original)
    expected = keras.ops.convert_to_numpy(original(sample_input))

    rebuilt = [
        type(original).from_config(dict(original.get_config())),
        keras.saving.deserialize_keras_object(
            keras.saving.serialize_keras_object(original)
        ),
    ]
    for candidate in rebuilt:
        assert type(candidate) is type(original)
        assert _cfg_json(candidate) == config_before
        assert _constraint_effect(candidate) == _constraint_effect(original)
        candidate(sample_input)  # build, so the weights are assignable
        candidate.set_weights(original.get_weights())
        actual = keras.ops.convert_to_numpy(candidate(sample_input))
        assert float(np.max(np.abs(expected - actual))) == 0.0


def test_a_serialized_none_constraint_does_not_reacquire_the_default_floor() -> None:
    """Regime 3 must not be reloaded as regime 1.

    ``gamma_constraint=None`` means "deliberately unconstrained". If the base
    ``from_config`` ever routed a serialized ``None`` back through the
    ``"__default__"`` sentinel branch, the reloaded layer would silently clamp
    gamma at ``1e-3`` — a behaviour change no shape or dtype assertion sees.
    """
    original = EnergyLayerNorm(epsilon=1e-2, gamma_constraint=None)
    reloaded = EnergyLayerNorm.from_config(dict(original.get_config()))
    assert reloaded.gamma_constraint is None


def test_the_default_sentinel_survives_a_config_that_never_carried_the_key() -> None:
    """Regime 1 via a legacy config, and via a normal one.

    A configuration written before ``gamma_constraint`` was serialized has no
    such key at all; ``cls(**config)`` must then hit the ``_DEFAULT_CONSTRAINT``
    sentinel and restore the positivity floor rather than leaving gamma free.
    """
    legacy: Dict[str, Any] = dict(EnergyLayerNorm(epsilon=5e-4).get_config())
    legacy.pop("gamma_constraint")

    from_legacy = EnergyLayerNorm.from_config(legacy)
    from_full = EnergyLayerNorm.from_config(
        dict(EnergyLayerNorm(epsilon=5e-4).get_config())
    )

    probe = keras.ops.convert_to_tensor(np.array([-5.0, 5.0], dtype="float32"))
    for layer in (from_legacy, from_full):
        assert layer.gamma_constraint is not None
        clipped = keras.ops.convert_to_numpy(layer.gamma_constraint(probe))
        # The floor is `_GAMMA_FLOOR`; the point is that the NEGATIVE value is
        # lifted above zero, which is the whole reason the constraint exists.
        assert clipped[0] > 0.0
        assert clipped[1] == pytest.approx(5.0)


@pytest.mark.parametrize(
    "cls", [EnergyLayerNorm, BandLogitNorm], ids=["EnergyLayerNorm", "BandLogitNorm"]
)
def test_neither_class_reintroduces_a_hand_written_from_config(cls: type) -> None:
    """Anti-regrowth guard, with a way out that is a measurement.

    This is not a style rule. Re-adding ``from_config`` is only wrong while the
    tests above still pass without it; if a future change makes the base
    implementation genuinely insufficient, one of them fails FIRST and this
    assertion is then the thing to delete — in the same commit, not before.
    """
    assert "from_config" not in cls.__dict__, (
        f"{cls.__name__} defines its own `from_config` again. Every reload path "
        "in this module was measured bit-identical without it "
        "(plan-2026-08-25-d5a035ab, decisions.md D-011). If it is needed now, a "
        "test above must be failing — fix that first and remove this guard "
        "alongside it."
    )
