"""C-45(c)/(d): NTM determinism, and `training=` actually reaching the layer.

(c) With ``use_memory_init=False`` the initial memory was an UNSEEDED
``keras.random.normal`` drawn per call, so ``model.predict(x)`` twice returned
different values. stddev is 1e-3 -- small, and exactly the size that makes a
downstream numeric test flaky rather than red.

(d) ``NTMMultiTask.call`` neither accepted nor forwarded ``training``. Keras 3
propagates it through a single mutable ``CallContext`` slot that every nested
``__call__`` overwrites, so the NTM layer read whatever the slot last held --
``None``, i.e. inference -- and any dropout configured in ``NTMConfig`` never
activated during ``fit()``.
"""

import inspect

import numpy as np
import keras
import pytest

from dl_techniques.layers.memory.ntm_interface import NTMConfig
from dl_techniques.models.ntm.model_multitask import NTMMultiTask


def _config(**overrides) -> NTMConfig:
    base = dict(
        memory_size=8, memory_dim=4, controller_dim=8,
        num_read_heads=1, num_write_heads=1, use_memory_init=False,
    )
    base.update(overrides)
    return NTMConfig(**base)


class TestTheRandomInitialMemoryIsDeterministic:

    def test_two_forward_passes_agree(self):
        model = NTMMultiTask(
            ntm_config=_config(), num_tasks=2, output_dim=3)
        sequence = np.random.rand(2, 5, 6).astype("float32")
        task = np.eye(2, dtype="float32")

        first = np.asarray(model([sequence, task], training=False))
        second = np.asarray(model([sequence, task], training=False))

        assert np.max(np.abs(first - second)) == 0.0, (
            f"ASSERT-PREDICT-IS-DETERMINISTIC: two identical inference calls "
            f"differ by {np.max(np.abs(first - second)):.3e}; the initial "
            f"memory is being redrawn from the global stateful stream."
        )

    def test_the_seed_is_recorded_in_the_config(self):
        config = _config(memory_init_seed=7)
        assert config.memory_init_seed == 7
        assert config.to_dict()["memory_init_seed"] == 7

    def test_the_memory_is_still_symmetry_broken(self):
        """Anti-vacuity: determinism must not mean a constant memory."""
        from dl_techniques.layers.memory.baseline_ntm import NTMMemory

        memory_state = NTMMemory(memory_size=8, memory_dim=4).initialize_state(2)
        slots = np.asarray(memory_state.memory)

        assert slots.std() > 0.0, (
            "the draw exists to differentiate memory slots; a constant memory "
            "would collapse content-based addressing"
        )
        # Two slots of the same batch element must not be identical.
        assert np.max(np.abs(slots[0, 0] - slots[0, 1])) > 0.0

    def test_a_different_seed_gives_a_different_memory(self):
        """Liveness arm: the seed is READ, not merely stored."""
        from dl_techniques.layers.memory.baseline_ntm import NTMMemory

        a = np.asarray(NTMMemory(
            memory_size=8, memory_dim=4, memory_init_seed=1
        ).initialize_state(2).memory)
        b = np.asarray(NTMMemory(
            memory_size=8, memory_dim=4, memory_init_seed=2
        ).initialize_state(2).memory)

        assert np.max(np.abs(a - b)) > 0.0


class TestTrainingIsForwardedToTheNTMLayer:

    def test_call_declares_the_parameter(self):
        signature = inspect.signature(NTMMultiTask.call)
        assert "training" in signature.parameters, (
            "ASSERT-CALL-TAKES-TRAINING: without the parameter, Keras 3's "
            "CallContext slot is the only channel and it is not reliable."
        )

    def test_it_is_forwarded_explicitly(self):
        source = inspect.getsource(NTMMultiTask.call)
        assert "training=training" in source, (
            "ASSERT-TRAINING-FORWARDED: declaring the parameter and dropping "
            "it on the floor is the same defect."
        )

    @pytest.mark.parametrize("training", [True, False, None])
    def test_the_value_reaches_the_ntm_layer(self, training):
        """Record the propagated value -- there is nothing numeric to measure.

        MEASURED and CORRECTING the finding: ``NTMConfig`` has NO dropout knob
        (``NTMConfig(dropout_rate=...)`` raises ``TypeError``) and nothing in
        the NTM stack is training-sensitive, so the "configured dropout never
        activated during ``fit()``" harm does not exist today and a
        two-passes-differ assertion would have no mechanism to fire. The fix is
        latent; the guard records the value instead of inventing an effect.
        """
        model = NTMMultiTask(
            ntm_config=_config(), num_tasks=2, output_dim=3)
        sequence = np.random.rand(2, 5, 6).astype("float32")
        task = np.eye(2, dtype="float32")
        model([sequence, task], training=False)  # build

        recorded = []
        original_call = model.ntm_layer.call

        def recording_call(*args, **kwargs):
            recorded.append(kwargs.get("training", "<ABSENT>"))
            return original_call(*args, **kwargs)

        model.ntm_layer.call = recording_call
        try:
            model([sequence, task], training=training)
        finally:
            model.ntm_layer.call = original_call

        assert recorded, "the NTM layer was never reached"
        assert all(value == training for value in recorded), (
            f"ASSERT-TRAINING-VALUE-PROPAGATES: called with training="
            f"{training!r}, the NTM layer saw {recorded!r}."
        )

    def test_the_config_has_no_dropout_knob(self):
        """Pin the corrected premise so it is not re-derived from the review."""
        with pytest.raises(TypeError):
            NTMConfig(memory_size=8, memory_dim=4, dropout_rate=0.5)
