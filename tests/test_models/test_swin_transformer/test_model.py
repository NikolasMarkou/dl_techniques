"""Permanent build+forward smoke test for the swin_transformer family.

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).
REPORT-ONLY: a build/forward break is documented via xfail, never fixed.

`create_swin_transformer(variant, num_classes, input_shape)` verified from
source (model.py:627 -> SwinTransformer.from_variant). The tiny variant uses
``patch_size=4`` and ``window_size=7`` (measured off the built model; this
docstring previously said 8). Returns logits ``(B, num_classes)``.

The claim this docstring used to make -- "input H/W MUST be divisible by
``patch_size * 8``" -- is FALSE and was removed rather than softened. That
expression is a compute note, not a requirement: divisibility by
``patch_size * 8`` is never enforced and never needs to be, because
``PatchMerging`` ceil-pads an odd grid dimension. The only hard requirement is
``H % patch_size == 0``, raised by ``PatchEmbedding2D``. 32x32 remains the input
used below because it is the documented example, not because it is a minimum.
See ``TestWarnedGeometriesAreCorrect`` for the executable form.
"""

import numpy as np
import pytest


def _assert_finite(value):
    arr = np.asarray(value)
    assert arr is not None
    assert not np.any(np.isnan(arr))
    assert not np.any(np.isinf(arr))


def test_smoke_build_and_forward():
    try:
        from dl_techniques.models.swin_transformer.model import (
            create_swin_transformer,
        )

        # 32x32 is the DOCUMENTED example input (model.py:30), not a minimum --
        # see the module docstring; smaller and non-divisible inputs also work.
        model = create_swin_transformer("tiny", 10, input_shape=(32, 32, 3))

        images = np.random.rand(2, 32, 32, 3).astype("float32")
        out = model(images, training=False)
    except Exception as exc:  # noqa: BLE001
        pytest.xfail(
            f"swin_transformer build/forward failed: "
            f"{type(exc).__name__}: {exc}"
        )

    _assert_finite(out)


# ---------------------------------------------------------------------------
# G-03 -- the executable content of the reworded `patch_size * 8` note.
#
# The old warning claimed non-divisibility "may cause issues in deeper stages".
# An 80-cell sweep found ZERO true positives, so the warning was reworded to a
# compute note (`models/swin_transformer/model.py`, DECISION
# plan-2026-07-31T210633-b63a35aa/D-003) and NO `window_size` raise was added.
# A prose reword is not RED-provable on its own; this class is its testable
# content. It goes red if anyone ever makes a warned geometry a correctness
# failure.
#
# `include_top=False` is REQUIRED, not incidental: with the classification head
# the declared shape is `(num_classes,)` at every geometry, so `declared ==
# actual` would be trivially true both ways -- a vacuous guard. With
# `include_top=False` the declared shape is SPATIAL `(H/32, W/32, C)` and
# actually depends on the merge arithmetic under test.
# ---------------------------------------------------------------------------

#: ``(input_hw, patch_size, window_size)`` for the two geometries the
#: ``patch_size * 8`` predicate WARNS about. Both build correctly; that is the
#: point. Measured deepest-stage grids: 56/ps=4 -> 2x2, 30/ps=2 -> 2x2.
WARNED_GEOMETRIES = ((56, 4, 7), (30, 2, 4))


class TestWarnedGeometriesAreCorrect:

    @pytest.mark.parametrize("input_hw,patch_size,window_size", WARNED_GEOMETRIES)
    def test_warned_geometry_builds_with_declared_shape_equal_to_actual(
            self, input_hw, patch_size, window_size):
        """A WARNED geometry must build and report its true output shape."""
        from dl_techniques.models.swin_transformer.model import SwinTransformer

        # Pin that this cell really is one the predicate warns about, derived
        # from the same expression the model uses -- not restated as a literal.
        assert input_hw % (patch_size * 8) != 0, (
            f"{input_hw}/ps={patch_size} is NOT a warned geometry; this test "
            f"would be about something else entirely."
        )
        # And that it satisfies the ONE hard requirement, so a failure below is
        # never `PatchEmbedding2D`'s divisibility raise in disguise.
        assert input_hw % patch_size == 0

        model = SwinTransformer(
            num_classes=3, input_shape=(input_hw, input_hw, 3),
            patch_size=patch_size, window_size=window_size,
            embed_dim=24, depths=[2, 2, 2, 2], num_heads=[2, 2, 2, 2],
            include_top=False,
        )
        images = np.random.rand(1, input_hw, input_hw, 3).astype("float32")
        out = model(images, training=False)

        declared = tuple(model.output_shape[1:])
        actual = tuple(int(v) for v in out.shape[1:])
        assert len(declared) == 3, (
            f"declared shape {declared} is not spatial -- the guard would be "
            f"vacuous. Did include_top=False stop taking effect?"
        )
        assert declared == actual, (
            f"input {input_hw}x{input_hw}, patch_size={patch_size}, "
            f"window_size={window_size}: model.output_shape[1:] {declared} != "
            f"actual {actual}. The reworded compute note in "
            f"SwinTransformer.__init__ asserts these geometries are CORRECT; "
            f"if that is no longer true the note is a false claim."
        )
        _assert_finite(out)
