"""F-39: `NAMConfig.num_write_heads` was a serialized, documented knob that
reached no code.

`nam/cell.py` constructs exactly one `NTMWriteHead` and binds it to a single
attribute (`self.write_head`). `num_read_heads`, immediately above it, drives a
list comprehension; the write side never had one. So `num_write_heads` was a
default, three identical variant entries (all ``1``), a `:param:` line and a
`to_dict()` key that no code path read -- byte-for-byte the `shift_range`
defect the same package removed one day earlier, missed by that audit because
it sits one field away (see `test_shift_range_is_not_a_knob.py`).

The ruling (decisions.md plan-2026-08-18T140459-7991552f/D-035) is that one
write head is NAM's deliberate design, so the field is removed rather than
honoured -- honouring it would change the memory semantics and the weight tree
of every shipped checkpoint.

These tests pin BOTH halves: the knob is gone from the public config surface,
AND the write side it used to describe is genuinely singular while the read
side (the anti-vacuity control) genuinely is not.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import keras
import pytest

from dl_techniques.models.nam.cell import NAMCell
from dl_techniques.models.nam.config import NAMConfig, NAM_VARIANTS


def _tiny(**overrides) -> NAMConfig:
    base = dict(
        hidden_size=32,
        num_heads=4,
        num_tree_layers=1,
        intermediate_size=64,
        memory_size=8,
        num_read_heads=2,
        max_expression_len=16,
        halt_max_steps=4,
    )
    base.update(overrides)
    return NAMConfig(**base)


class TestNumWriteHeadsIsGoneFromNAMConfig:

    def test_nam_config_no_longer_accepts_num_write_heads(self):
        with pytest.raises(TypeError, match="num_write_heads"):
            NAMConfig(num_write_heads=2)

    def test_num_write_heads_is_absent_from_the_serialized_config(self):
        assert "num_write_heads" not in NAMConfig().to_dict()

    def test_no_shipped_variant_still_carries_it(self):
        for name, variant in NAM_VARIANTS.items():
            assert "num_write_heads" not in variant, name

    def test_an_old_serialized_config_carrying_it_still_loads(self):
        """`to_dict()` output from before this change must not become
        unloadable -- `from_dict` filters to declared fields. The value never
        reached anything, so dropping it is behaviour-preserving by
        construction."""
        legacy = NAMConfig().to_dict()
        legacy["num_write_heads"] = 4
        restored = NAMConfig.from_dict(legacy)
        assert not hasattr(restored, "num_write_heads")
        assert restored.memory_size == NAMConfig().memory_size


class TestTheWriteSideIsSingularAndTheReadSideIsNot:
    """The reason the knob was dead, measured at the cell rather than asserted
    from the source."""

    def test_the_cell_has_exactly_one_write_head(self):
        cell = NAMCell(_tiny())
        assert not isinstance(cell.write_head, (list, tuple))
        write_heads = [
            v for v in vars(cell).values()
            if type(v).__name__ == "NTMWriteHead"
        ]
        assert len(write_heads) == 1

    def test_num_read_heads_is_the_control_and_is_live(self):
        """Anti-vacuity: a head count on this cell CAN matter -- the read one
        does, changing both the head list length and the parameter count. The
        write side simply never had the same wiring."""
        keras.utils.set_random_seed(0)
        a = NAMCell(_tiny(num_read_heads=2))
        a.build((None, 16))
        keras.utils.set_random_seed(0)
        b = NAMCell(_tiny(num_read_heads=4))
        b.build((None, 16))

        assert len(a.read_heads) == 2
        assert len(b.read_heads) == 4
        assert a.count_params() != b.count_params()
