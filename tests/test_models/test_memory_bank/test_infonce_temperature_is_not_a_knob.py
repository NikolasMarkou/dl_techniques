"""F-22: `infonce_temperature` was a dead knob, plumbed through two classes
and serialized by both.

``MemoryReadController.__init__`` accepted it, stored it and emitted it from
``get_config()``; ``WaveFieldMemoryLLM`` accepted it, stored it, forwarded it to
the controller and emitted it too. Seven grep hits, all store / forward /
serialize, and not one read. The temperature the InfoNCE term actually uses is
the learned weight ``log_temp_nce``: ``tau = softplus(log_temp_nce) + 1e-3``,
which starts at ``softplus(0) + 1e-3 == 0.694``. So a caller asking for
``infonce_temperature=0.05`` trained at 0.694 -- and then saw 0.05 echoed back
in the reloaded config, which is what made it look honoured.

The ruling (decisions.md plan-2026-08-18T140459-7991552f/D-033) is to remove the
argument rather than seed the initializer from it: seeding would move the
shipped default init from 0.694 to 0.1 and silently change every future training
run, for a knob nobody could have been relying on.

These tests pin all three halves: the argument is gone from both public
surfaces, a pre-removal config still loads, and the temperature that IS live is
the weight.
"""

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.memory_bank.read_controller import MemoryReadController
from dl_techniques.models.memory_bank.wave_field_memory_llm import (
    WaveFieldMemoryLLM,
)

from .test_read_controller import _build_bundle, _forward


_LLM_KWARGS = dict(
    vocab_size=32, embed_dim=32, depth=3, num_heads=4, max_seq_len=8,
    field_size=16, d_k=8, d_v=16, s_lt=16, top_k=4,
    diversity_subsample=8, infonce_negatives=8,
)


class TestTheArgumentIsGoneFromBothSurfaces:

    def test_the_read_controller_refuses_it_by_name(self):
        with pytest.raises((TypeError, ValueError), match="infonce_temperature"):
            MemoryReadController(
                embed_dim=32, num_heads=4, d_k=8, d_v=16,
                s_lt=16, max_seq_len=10, top_k=4,
                infonce_temperature=0.05,
            )

    def test_the_model_refuses_it_by_name(self):
        with pytest.raises((TypeError, ValueError), match="infonce_temperature"):
            WaveFieldMemoryLLM(infonce_temperature=0.05, **_LLM_KWARGS)

    def test_it_is_absent_from_both_serialized_configs(self):
        read, _, _ = _build_bundle()
        assert "infonce_temperature" not in read.get_config()
        assert "infonce_temperature" not in WaveFieldMemoryLLM(
            **_LLM_KWARGS
        ).get_config()


class TestPreRemovalConfigsStillLoad:
    """Dropping the key is behaviour-preserving by construction -- the value
    never reached the InfoNCE term -- so a legacy config must load, not raise."""

    def test_read_controller_from_config_drops_the_legacy_key(self):
        read, _, _ = _build_bundle()
        legacy = read.get_config()
        legacy["infonce_temperature"] = 0.05
        restored = MemoryReadController.from_config(legacy)
        assert not hasattr(restored, "infonce_temperature")
        assert restored.top_k == read.top_k

    def test_model_from_config_drops_the_legacy_key(self):
        model = WaveFieldMemoryLLM(**_LLM_KWARGS)
        legacy = model.get_config()
        legacy["infonce_temperature"] = 0.05
        restored = WaveFieldMemoryLLM.from_config(legacy)
        assert not hasattr(restored, "infonce_temperature")
        assert restored.read_controller.top_k == model.read_controller.top_k


class TestTheLiveTemperatureIsTheWeight:
    """Why the knob was dead, measured rather than asserted from the source."""

    @staticmethod
    def _bundle():
        from dl_techniques.models.memory_bank.memory_banks import (
            LongTermMemoryBank,
        )
        from dl_techniques.models.memory_bank.write_controller import (
            MemoryWriteController,
        )
        read = MemoryReadController(
            embed_dim=32, num_heads=4, d_k=8, d_v=16,
            s_lt=16, max_seq_len=10, top_k=4,
            enable_infonce=True, infonce_negatives=8,
        )
        lt = LongTermMemoryBank(s_lt=16, d_k=8, d_v=16)
        lt.build()
        write = MemoryWriteController(
            d_k=8, d_v=16, embed_dim=32, max_seq_len=10,
        )
        return read, lt, write

    @staticmethod
    def _total(read):
        return float(sum(float(ops.convert_to_numpy(l)) for l in read.losses))

    @classmethod
    def _infonce_loss_at(cls, log_temp_value):
        """Aux-loss contribution of ONE forward pass on a fresh, identically
        seeded bundle whose learned temperature has been set to
        ``log_temp_value``.

        The first pass exists only to build the layer (``log_temp_nce`` is
        created in ``build``). ``add_loss`` accumulates across eager calls, so
        the measurement is the DIFFERENCE the second pass contributed, not the
        running total -- otherwise the first pass's terms, computed at the
        default temperature, would dilute both arms equally and could hide a
        real difference.
        """
        keras.utils.set_random_seed(0)
        read, lt, write = cls._bundle()
        x = np.random.RandomState(0).randn(2, 6, 32).astype(np.float32)
        k_lt, v_lt = lt(None)
        k_wm, v_wm, mask = write(x)
        read(x, k_lt, v_lt, k_wm, v_wm, mask, aux_scale=1.0, training=True)
        before = cls._total(read)

        read.log_temp_nce.assign(log_temp_value)
        keras.utils.set_random_seed(0)
        read(x, k_lt, v_lt, k_wm, v_wm, mask, aux_scale=1.0, training=True)
        return cls._total(read) - before, read

    def test_the_default_tau_is_softplus_zero_plus_eps(self):
        _, read = self._infonce_loss_at(0.0)
        tau = float(ops.convert_to_numpy(
            ops.softplus(read.log_temp_nce) + 1e-3
        ))
        assert tau == pytest.approx(np.log(2.0) + 1e-3, abs=1e-6), (
            "the InfoNCE temperature at init is softplus(0) + 1e-3 == 0.694, "
            "which is what a caller asking for 0.1 (the old default) or 0.05 "
            "actually trained at"
        )

    def test_moving_the_weight_moves_the_infonce_term(self):
        """Liveness arm: the weight is not decorative. Without this, 'the knob
        did nothing' would be consistent with 'nothing does anything here'."""
        loss_a, _ = self._infonce_loss_at(0.0)
        loss_b, _ = self._infonce_loss_at(3.0)
        assert loss_a != loss_b, (
            f"the learned InfoNCE temperature does not move the aux loss "
            f"({loss_a} vs {loss_b}); this test cannot prove which "
            f"temperature is live"
        )
