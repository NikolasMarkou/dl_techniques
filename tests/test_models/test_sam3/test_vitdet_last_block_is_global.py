"""F-17: ``Sam3ViTDetBackbone`` must not build blocks it never executes.

``call`` returns at ``index == max(global_att_blocks)``. ``__init__`` validated
only that each index lies in ``[0, depth)``, never that the LAST one is
``depth - 1``, so ``Sam3ViTDetBackbone(depth=6, global_att_blocks=(1, 3))``
constructed silently, reported the full 6-block parameter count, and left
blocks 4 and 5 dead: no gradient, no contribution, and two AdamW moment buffers
per weight for nothing.

MEASURED at ``11f971ed1``, ``depth=6, global_att_blocks=(1, 3)``: zeroing every
weight of blocks 4 and 5 -- 1,744 parameters -- moved the output by
``max|delta| == 0.0`` EXACTLY. No shape error, no warning, and invisible to a
parameter-count assertion, which is why the guard belongs in ``__init__``.

Every shipped variant already satisfies the invariant (``sam3`` 31/32, ``small``
5/6, ``tiny`` 1/2), so this adds no variant breakage.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest

from dl_techniques.models.SAM.SAM3.vitdet import Sam3ViTDetBackbone
from dl_techniques.models.SAM.SAM3.sam3_image import Sam3Image

BASE = dict(img_size=16, patch_size=2, in_channels=3, embed_dim=8, depth=4,
            num_heads=2, mlp_ratio=4.0, window_size=4, global_att_blocks=(1, 3),
            pretrain_img_size=8, drop_path_rate=0.0, dropout_rate=0.0)


class TestTheLastBlockMustBeGlobal:

    @pytest.mark.parametrize("depth,globals_", [
        (6, (1, 3)),      # the finding's exact case: blocks 4, 5 dead
        (4, (0,)),        # blocks 1, 2, 3 dead
        (32, (7, 15, 23)),  # the finding's headline: 8 of 32 dead
    ])
    def test_a_trunk_with_dead_tail_blocks_raises(self, depth, globals_):
        with pytest.raises(ValueError, match="must name the LAST block"):
            Sam3ViTDetBackbone(**{**BASE, "depth": depth,
                                  "global_att_blocks": globals_})

    def test_the_message_names_the_blocks_that_would_be_dead(self):
        with pytest.raises(ValueError) as excinfo:
            Sam3ViTDetBackbone(**{**BASE, "depth": 6,
                                  "global_att_blocks": (1, 3)})
        message = str(excinfo.value)
        assert "(4, 5)" in message, message
        assert "depth - 1 = 5" in message, message

    def test_order_does_not_matter_only_the_maximum(self):
        """``(3, 1)`` is the same set as ``(1, 3)``; the guard reads ``max``."""
        trunk = Sam3ViTDetBackbone(**{**BASE, "global_att_blocks": (3, 1)})
        assert trunk.last_global_block == 3
        with pytest.raises(ValueError, match="must name the LAST block"):
            Sam3ViTDetBackbone(**{**BASE, "depth": 6,
                                  "global_att_blocks": (3, 1)})

    def test_a_single_global_block_at_the_last_index_is_accepted(self):
        trunk = Sam3ViTDetBackbone(**{**BASE, "depth": 3,
                                      "global_att_blocks": (2,)})
        assert trunk.last_global_block == 2
        assert len(trunk.blocks) == 3


class TestTheExistingGuardsStillFireFirst:
    """The new check must not shadow the two that preceded it."""

    def test_empty_still_raises_its_own_message(self):
        with pytest.raises(ValueError, match="at least one block"):
            Sam3ViTDetBackbone(**{**BASE, "global_att_blocks": ()})

    def test_out_of_range_still_raises_its_own_message(self):
        """``(0, 99)`` at ``depth=4`` is out of range, NOT a last-block error.

        ``max`` of it is 99, so a guard placed BEFORE the range check would
        report the wrong problem.
        """
        with pytest.raises(ValueError, match=r"\[0, depth"):
            Sam3ViTDetBackbone(**{**BASE, "global_att_blocks": (0, 99)})


class TestEveryShippedVariantSatisfiesTheInvariant:
    """The guard is a new ``ValueError``; nothing shipped may trip it."""

    @pytest.mark.parametrize("variant", sorted(Sam3Image.MODEL_VARIANTS))
    def test_variant_table_names_the_last_block(self, variant):
        table = Sam3Image.MODEL_VARIANTS[variant]
        assert max(table["global_att_blocks"]) == table["depth"] - 1, (
            f"variant {variant!r} would now fail to construct"
        )

    @pytest.mark.parametrize("variant", sorted(Sam3Image.MODEL_VARIANTS))
    def test_the_trunk_of_each_variant_actually_constructs(self, variant):
        table = Sam3Image.MODEL_VARIANTS[variant]
        trunk = Sam3ViTDetBackbone(
            img_size=table["img_size"], patch_size=table["patch_size"],
            embed_dim=table["embed_dim"], depth=table["depth"],
            num_heads=table["num_heads"], mlp_ratio=table["mlp_ratio"],
            window_size=table["window_size"],
            global_att_blocks=table["global_att_blocks"],
            pretrain_img_size=table["pretrain_img_size"],
        )
        assert trunk.last_global_block == table["depth"] - 1


class TestTheInvariantMakesTheTailReachable:
    """With the guard in place, EVERY block contributes."""

    def test_zeroing_the_last_block_changes_the_output(self):
        """The positive control for the RED proof's own instrument.

        Pre-fix the same procedure on blocks 4+5 of a ``depth=6,
        global=(1, 3)`` trunk measured EXACTLY 0.0. Here the trunk is legal and
        the last block is live, so the same instrument must read non-zero --
        otherwise it proves nothing.
        """
        import keras
        from keras import ops

        trunk = Sam3ViTDetBackbone(**BASE)
        trunk.build((None, 16, 16, 3))
        image = np.random.RandomState(5).randn(1, 16, 16, 3).astype("float32")
        base = keras.ops.convert_to_numpy(trunk(image, training=False))
        for var in trunk.blocks[-1].weights:
            var.assign(ops.zeros_like(var))
        after = keras.ops.convert_to_numpy(trunk(image, training=False))
        assert float(np.max(np.abs(after - base))) > 1e-6
