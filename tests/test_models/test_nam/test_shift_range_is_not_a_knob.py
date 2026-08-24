"""R-3: `NAMConfig.shift_range` configured nothing.

`nam/cell.py` builds every read and write head with
``addressing_mode=AddressingMode.CONTENT`` -- and has done since NAM's first
commit -- while passing ``shift_range=config.shift_range`` alongside it. Commit
``4972b1bd1`` (D-073) made the location-addressing projections conditional on
``AddressingMode.HYBRID``, so under CONTENT no ``shift_dense`` is created and no
circular shift ever runs. `shift_range` was left documented in the config
docstring and serialized by ``to_dict()``, reaching a parameter that is read
only inside a branch NAM never takes.

The ruling (decisions.md D-014) is that CONTENT is NAM's deliberate design, not
an accident, so the knob is removed rather than the addressing mode changed.
These tests pin BOTH halves: the knob is gone from the public config surface,
AND the heads it used to feed are genuinely content-only.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import keras
import numpy as np
import pytest

from dl_techniques.layers.memory.baseline_ntm import NTMReadHead, NTMWriteHead
from dl_techniques.layers.memory.ntm_interface import AddressingMode, MemoryState
from dl_techniques.models.nam.config import NAMConfig


class TestShiftRangeIsGoneFromNAMConfig:

    def test_nam_config_no_longer_accepts_shift_range(self):
        with pytest.raises(TypeError, match="shift_range"):
            NAMConfig(shift_range=7)

    def test_shift_range_is_absent_from_the_serialized_config(self):
        assert "shift_range" not in NAMConfig().to_dict()

    def test_an_old_serialized_config_carrying_shift_range_still_loads(self):
        """`to_dict()` output from before this change must not become
        unloadable -- `from_dict` filters to declared fields."""
        legacy = NAMConfig().to_dict()
        legacy["shift_range"] = 3
        restored = NAMConfig.from_dict(legacy)
        assert not hasattr(restored, "shift_range")
        assert restored.memory_size == NAMConfig().memory_size


class TestContentHeadsIgnoreShiftRange:
    """The reason the knob was dead, measured rather than asserted from the
    source: under CONTENT the shift projection does not exist and the value of
    `shift_range` moves neither the parameter count nor the output."""

    @staticmethod
    def _built(head_cls, shift_range):
        head = head_cls(
            memory_size=8,
            memory_dim=16,
            addressing_mode=AddressingMode.CONTENT,
            shift_range=shift_range,
        )
        head.build((None, 16))
        return head

    @pytest.mark.parametrize("head_cls", [NTMReadHead, NTMWriteHead])
    def test_no_shift_projection_exists_under_content(self, head_cls):
        head = self._built(head_cls, 3)
        assert head.shift_dense is None
        assert head.gate_dense is None
        assert head.gamma_dense is None

    @pytest.mark.parametrize("head_cls", [NTMReadHead, NTMWriteHead])
    def test_shift_range_changes_neither_weights_nor_output(self, head_cls):
        keras.utils.set_random_seed(0)
        a = self._built(head_cls, 3)
        keras.utils.set_random_seed(0)
        b = self._built(head_cls, 9)

        assert a.count_params() == b.count_params()

        x = np.random.default_rng(0).random((2, 16)).astype("float32")
        memory = np.random.default_rng(1).random((2, 8, 16)).astype("float32")
        prev = np.full((2, 8), 1.0 / 8, dtype="float32")

        state = MemoryState(memory=keras.ops.convert_to_tensor(memory))
        out_a = keras.ops.convert_to_numpy(a.compute_addressing(x, state, prev)[0])
        out_b = keras.ops.convert_to_numpy(b.compute_addressing(x, state, prev)[0])
        np.testing.assert_array_equal(out_a, out_b)

    def test_a_hybrid_head_is_the_control(self):
        """Anti-vacuity: the same parameter DOES do something under HYBRID, so
        the equality above is a property of CONTENT and not of the probe."""
        keras.utils.set_random_seed(0)
        a = NTMReadHead(memory_size=8, memory_dim=16,
                        addressing_mode=AddressingMode.HYBRID, shift_range=3)
        a.build((None, 16))
        keras.utils.set_random_seed(0)
        b = NTMReadHead(memory_size=8, memory_dim=16,
                        addressing_mode=AddressingMode.HYBRID, shift_range=9)
        b.build((None, 16))

        assert a.shift_dense is not None
        assert a.count_params() != b.count_params()
