"""Literal pins on `layers/memory`'s public surface.

`MemoryAccessType` was exported for years with ZERO consumers, and nothing
noticed -- no test asserted the length of `__all__`, enumerated its names, or
watched either state dataclass's field set. That removal was spent precisely on
the grounds that nothing watched the surface; leaving it unwatched afterwards
would make the next agent's re-add exactly as invisible as this removal was.
See `decisions.md` D-003 of `plan-2026-08-30T120217-7f6cedd1`.

Each pin carries a LITERAL, not a computed value. A length alone is blind to a
rename (25 names with one renamed is still 25), so the exact sorted tuple sits
beside the length; and a literal forces a future intentional export to update
this file consciously rather than watching a self-recomputing assertion agree
with itself.

These pin the field SET, not value FLOW. A `dataclasses.fields()` guard cannot
see a field that is declared, wired and then never written -- which is the very
defect that put `read_vector`, `temporal_links` and `precedence` here. Do not
mistake these for behavioural guards.
"""

import dataclasses

import dl_techniques.layers.memory as memory_package
from dl_techniques.layers.memory.ntm_interface import HeadState, MemoryState

EXPECTED_ALL = (
    "AddressingMode",
    "BaseController",
    "BaseHead",
    "BaseMemory",
    "BaseNTM",
    "HeadState",
    "MemoryState",
    "NTMCell",
    "NTMConfig",
    "NTMController",
    "NTMMemory",
    "NTMOutput",
    "NTMReadHead",
    "NTMWriteHead",
    "NeuralTuringMachine",
    "NeuroGrid",
    "SOM2dLayer",
    "SOMLayer",
    "SoftSOMLayer",
    "circular_convolution",
    "cosine_similarity",
    "create_mann",
    "create_ntm",
    "create_som_2d",
    "sharpen_weights",
)

EXPECTED_MEMORY_STATE_FIELDS = (
    "memory",
    "usage",
    "write_weights",
    "read_weights",
    "metadata",
)

EXPECTED_HEAD_STATE_FIELDS = (
    "weights",
    "key",
    "beta",
    "gate",
    "shift",
    "gamma",
    "erase_vector",
    "add_vector",
    "metadata",
)


class TestThePackageSurfaceIsPinned:
    """`__all__` is pinned by LENGTH and by exact membership."""

    def test_the_surface_has_exactly_twenty_five_names(self):
        assert len(memory_package.__all__) == 25, (
            "layers/memory/__all__ changed length. This number is restated in "
            "layers/memory/__init__.py's module docstring and in "
            "plans/SYSTEM.md; move all three or the atlas goes stale."
        )

    def test_the_surface_holds_exactly_these_names(self):
        assert tuple(sorted(memory_package.__all__)) == EXPECTED_ALL

    def test_the_retired_enum_is_absent(self):
        """`MemoryAccessType` was retired; it must not come back silently."""
        assert "MemoryAccessType" not in memory_package.__all__
        assert not hasattr(memory_package, "MemoryAccessType")

    def test_every_exported_name_actually_resolves(self):
        """A pinned name that does not import is a pin on a lie."""
        missing = [n for n in memory_package.__all__ if not hasattr(memory_package, n)]
        assert missing == []


class TestTheStateDataclassFieldSetsArePinned:
    """The two runtime state carriers hold exactly these fields.

    Guards against a silent RE-ADD of a declared-but-never-written field --
    the shape that put `read_vector`, `temporal_links` and `precedence` in the
    tree in the first place.
    """

    def test_memory_state_fields(self):
        names = tuple(f.name for f in dataclasses.fields(MemoryState))
        assert names == EXPECTED_MEMORY_STATE_FIELDS

    def test_head_state_fields(self):
        names = tuple(f.name for f in dataclasses.fields(HeadState))
        assert names == EXPECTED_HEAD_STATE_FIELDS
