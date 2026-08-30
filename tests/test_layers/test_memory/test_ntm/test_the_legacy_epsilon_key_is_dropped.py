"""Guards for the retirement of `NTMMemory`'s inert `epsilon` knob.

`BaseMemory` used to accept, store and *emit* an `epsilon` that neither
`NTMMemory.read` nor `NTMMemory.write` ever read. It was removed
(`decisions.md` D-002 of `plan-2026-08-30T120217-7f6cedd1`), which makes any
config dict written by the OLD code an unloadable config: Keras' default
`from_config` is `cls(**config)`, so the stale key reaches
`keras.layers.Layer.__init__` as an unrecognized keyword.

The two guards here are a matched pair and must be read together:

* `TestTheLegacyEpsilonKeyIsDropped` proves the named-key shim WORKS -- an
  old-shape config still loads.
* `TestATypoKeyIsStillARejectedKey` proves the shim is NOT a blanket
  unknown-key filter. This is the assertion that discriminates a D-003-shaped
  named-key pop from the forbidden generalization; it goes RED against a
  blanket filter while every other assertion in this file stays green.

`TestTheHeadsEpsilonIsStillLive` guards the opposite claim: the
identically-named `epsilon` on `NTMReadHead` / `NTMWriteHead` is load-bearing
and was deliberately NOT touched. It is a VALUE measurement, not a source
inspection, because a source check cannot tell a live constant from a stored
one.
"""

import logging

import keras
import numpy as np
import pytest

from dl_techniques.layers.memory.baseline_ntm import NTMMemory, NTMReadHead
from dl_techniques.layers.memory.ntm_interface import AddressingMode, MemoryState

# The OLD-SHAPE config, transcribed VERBATIM from
#     NTMMemory(memory_size=32, memory_dim=16, epsilon=1e-5).get_config()
# executed at BASE (948fd90cd), BEFORE the removal landed. It is written here as
# a literal on purpose: a dict produced by calling `get_config()` on the CURRENT
# code can no longer contain `"epsilon"`, so a guard built that way would be
# tautological and could never see the regression it exists to catch.
OLD_SHAPE_CONFIG = {
    "name": "ntm_memory",
    "trainable": True,
    "dtype": {
        "module": "keras",
        "class_name": "DTypePolicy",
        "config": {"name": "float32"},
        "registered_name": None,
    },
    "memory_size": 32,
    "memory_dim": 16,
    "epsilon": 1e-05,
    "memory_init_seed": 42,
}


class TestTheLegacyEpsilonKeyIsDropped:
    """A config saved by the pre-removal code must still reconstruct."""

    def test_an_old_shape_config_still_loads(self, caplog):
        """The shim drops the key, warns once, and keeps every other value."""
        with caplog.at_level(logging.WARNING, logger="dl"):
            memory = NTMMemory.from_config(dict(OLD_SHAPE_CONFIG))

        # (a) no exception -- reaching this line at all is the first assertion.
        assert isinstance(memory, NTMMemory)

        # (b) exactly one warning, naming the dropped key.
        epsilon_warnings = [
            record
            for record in caplog.records
            if record.levelno == logging.WARNING and "epsilon" in record.getMessage()
        ]
        assert len(epsilon_warnings) == 1, (
            "expected exactly one WARNING naming 'epsilon', got "
            f"{[r.getMessage() for r in caplog.records]}"
        )

        # (c) the key was DROPPED, not absorbed as an attribute.
        assert not hasattr(memory, "epsilon")

        # (d) the surviving keys arrive at their transcribed values.
        assert memory.memory_size == OLD_SHAPE_CONFIG["memory_size"]
        assert memory.memory_dim == OLD_SHAPE_CONFIG["memory_dim"]
        assert memory.memory_init_seed == OLD_SHAPE_CONFIG["memory_init_seed"]
        assert memory.name == OLD_SHAPE_CONFIG["name"]

    def test_the_key_is_gone_from_the_emitted_config(self):
        """The removal must be real: nothing re-emits `epsilon`."""
        config = NTMMemory(memory_size=32, memory_dim=16).get_config()
        assert "epsilon" not in config
        assert config["memory_size"] == 32
        assert config["memory_dim"] == 16
        assert config["memory_init_seed"] == 42

    def test_the_constructor_no_longer_accepts_epsilon(self):
        """`epsilon` is not a knob on this class any more, only a legacy key."""
        with pytest.raises((TypeError, ValueError)):
            NTMMemory(memory_size=32, memory_dim=16, epsilon=1e-5)


class TestATypoKeyIsStillARejectedKey:
    """The discriminating guard: the shim drops ONE NAMED key, not any key.

    A blanket unknown-key filter would keep every other assertion in this file
    green while silently swallowing a misspelled parameter, turning a loud
    constructor error into a wrong-config-that-runs. This test is the only one
    that goes RED against that wrong fix.
    """

    def test_a_misspelled_key_still_raises(self):
        config = dict(OLD_SHAPE_CONFIG)
        config.pop("epsilon")
        config["memory_dimm"] = 16
        with pytest.raises((TypeError, ValueError)):
            NTMMemory.from_config(config)

    def test_a_misspelled_key_raises_even_beside_the_legacy_key(self):
        """Dropping `epsilon` must not license dropping anything else."""
        config = dict(OLD_SHAPE_CONFIG)
        config["memory_dimm"] = 16
        with pytest.raises((TypeError, ValueError)):
            NTMMemory.from_config(config)


class TestTheHeadsEpsilonIsStillLive:
    """`NTMReadHead.epsilon` is LIVE and was deliberately not removed.

    Identical field name, disjoint liveness: the head hands its `epsilon` to
    `cosine_similarity` and to `sharpen_weights`. The probe holds the head's
    WEIGHTS fixed (it mutates `epsilon` on one built instance rather than
    building two) so the measured difference can only come from `epsilon`.
    """

    def test_two_epsilons_give_two_different_addressings(self):
        memory_size, memory_dim, controller_dim, batch = 8, 6, 5, 2

        head = NTMReadHead(
            memory_size=memory_size,
            memory_dim=memory_dim,
            addressing_mode=AddressingMode.HYBRID,
            shift_range=3,
        )
        head.build((batch, controller_dim))

        rng = np.random.default_rng(0)
        controller_output = keras.ops.convert_to_tensor(
            rng.normal(size=(batch, controller_dim)).astype("float32")
        )
        # Near-zero-length memory slots are exactly where the epsilon inside the
        # L2 norm is load-bearing, so the probe is run there.
        state = MemoryState(
            memory=keras.ops.convert_to_tensor(
                (1e-3 * rng.normal(size=(batch, memory_size, memory_dim))).astype(
                    "float32"
                )
            )
        )
        prev_weights = keras.ops.convert_to_tensor(
            np.full((batch, memory_size), 1.0 / memory_size, dtype="float32")
        )

        head.epsilon = 1e-8
        weights_small, _ = head.compute_addressing(
            controller_output, state, prev_weights
        )
        head.epsilon = 1.0
        weights_large, _ = head.compute_addressing(
            controller_output, state, prev_weights
        )

        delta = float(
            np.max(
                np.abs(
                    keras.ops.convert_to_numpy(weights_small)
                    - keras.ops.convert_to_numpy(weights_large)
                )
            )
        )
        assert delta > 0.0, (
            "NTMReadHead.epsilon measured INERT (max abs delta exactly 0.0 "
            "between epsilon=1e-8 and epsilon=1.0) -- it must stay live"
        )
