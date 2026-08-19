"""Tests for the FractalBlock (FractalNet) layer."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.fractal_block import FractalBlock
from dl_techniques.layers.standard_blocks import ConvBlock

B, H, W, C = 2, 8, 8, 4
F = 8


def _block_config():
    return ConvBlock(filters=F, kernel_size=3).get_config()


@pytest.fixture
def sample():
    return np.random.default_rng(0).standard_normal((B, H, W, C)).astype("float32")


class TestFractalBlock:

    def test_construction(self):
        layer = FractalBlock(block_config=_block_config(), depth=2)
        assert layer.depth == 2

    @pytest.mark.parametrize("bad", [
        {"depth": 0},
        {"drop_path_rate": 1.5},
    ])
    def test_invalid_args_raise(self, bad):
        kwargs = {"block_config": _block_config(), **bad}
        with pytest.raises(ValueError):
            FractalBlock(**kwargs)

    def test_invalid_block_config_raises(self):
        with pytest.raises(ValueError):
            FractalBlock(block_config="not-a-dict")

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_forward_pass(self, sample, depth):
        layer = FractalBlock(block_config=_block_config(), depth=depth)
        out = layer(sample)
        assert tuple(out.shape) == (B, H, W, F)

    def test_compute_output_shape(self):
        layer = FractalBlock(block_config=_block_config(), depth=2)
        assert layer.compute_output_shape((B, H, W, C)) == (B, H, W, F)

    def test_serialization_round_trip(self, sample, tmp_path):
        inp = keras.Input(shape=(H, W, C))
        out = FractalBlock(block_config=_block_config(), depth=2, name="fractal")(inp)
        model = keras.Model(inp, out)
        y0 = model(sample, training=False)
        path = os.path.join(tmp_path, "fractal.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"FractalBlock": FractalBlock}
        )
        y1 = loaded(sample, training=False)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )


class TestFractalExpansionRule:
    """The block must implement the paper's rule, not a parallel average.

    Until 2026-08-14 ``FractalBlock`` at depth ``k`` built two sub-blocks of
    depth ``k-1`` and applied BOTH to the SAME input::

        F_k(x) = 0.5 * (F_{k-1}(x) + F_{k-1}(x))

    Recursion on that rule terminates with every leaf receiving the block's own
    input, so every input-to-output path traversed exactly ONE convolution no
    matter how large ``depth`` was. A depth-``k`` block was an average of
    ``2^(k-1)`` independent parallel convolutions -- ``depth`` bought width and
    doubled parameters while buying no depth at all.

    The paper's rule COMPOSES the deep branch::

        f_{C+1}(z) = [f_C(f_C(z))] join [conv(z)]

    so the longest path is ``2^(C)`` blocks and the shortest stays 1.

    The instrument is the receptive field, which measures composition directly
    and cannot be satisfied by parameter count, layer count or output shape --
    all three of which matched while the block was wrong. With 3x3 ``same``
    convolutions a path of ``L`` composed blocks has a receptive field of
    ``1 + 2L``, so a correct depth-``k`` block spans ``1 + 2 * 2^(k-1)``. Under
    the old rule this measured 3 at EVERY depth.
    """

    @staticmethod
    def _receptive_field(depth: int) -> int:
        """Width of the output region a single centre input pixel can affect."""
        cfg = ConvBlock(filters=4, kernel_size=3, strides=1, padding="same",
                        use_pooling=False).get_config()
        block = FractalBlock(block_config=cfg, depth=depth, drop_path_rate=0.0)

        size = 81
        base = np.zeros((1, size, size, 4), dtype="float32")
        poked = base.copy()
        poked[0, size // 2, size // 2, :] = 1.0

        out_base = keras.ops.convert_to_numpy(block(base, training=False))
        out_poked = keras.ops.convert_to_numpy(block(poked, training=False))

        delta = np.abs(out_base - out_poked).sum(axis=-1)[0]
        rows = np.nonzero(delta > 1e-9)[0]
        return int(rows.max() - rows.min() + 1) if rows.size else 0

    @pytest.mark.parametrize("depth", [1, 2, 3, 4])
    def test_receptive_field_grows_with_depth(self, depth: int):
        """RF must be ``1 + 2 * 2^(depth-1)``: 3, 5, 9, 17."""
        expected = 1 + 2 * (2 ** (depth - 1))
        assert self._receptive_field(depth) == expected, (
            f"depth={depth} should span {expected} pixels. A measurement of 3 at "
            f"every depth means the branches are applied in PARALLEL to the same "
            f"input instead of being composed."
        )

    def test_depth_actually_deepens(self):
        """Guard the trend itself, independent of the exact arithmetic."""
        fields = [self._receptive_field(d) for d in (1, 2, 3)]
        assert fields[0] < fields[1] < fields[2], (
            f"receptive field must grow with depth, measured {fields}"
        )

    @pytest.mark.parametrize("depth,expected", [(1, 1), (2, 3), (3, 7), (4, 15)])
    def test_leaf_count_is_two_to_the_k_minus_one(self, depth: int, expected: int):
        """The docstrings claimed ``2^(k-1)`` leaves until 2026-08-19.

        That is the count under the OLD parallel rule, where the shallow branch
        does not exist. The composed rule's recurrence is ``L(1) = 1``,
        ``L(k) = 2 * L(k-1) + 1`` -- both halves of the deep branch plus the
        shallow one -- which is ``2^k - 1``. The wrong number is not detectable
        from the receptive field, so it is pinned separately.
        """
        cfg = ConvBlock(filters=4, kernel_size=3, strides=1, padding="same",
                        use_pooling=False).get_config()
        block = FractalBlock(block_config=cfg, depth=depth, drop_path_rate=0.0)

        def leaves(b: FractalBlock) -> int:
            total = 0
            for sub in (b.block, b.deep_first, b.deep_second, b.shallow):
                if sub is None:
                    continue
                total += 1 if isinstance(sub, ConvBlock) else leaves(sub)
            return total

        assert leaves(block) == expected == 2 ** depth - 1

    def test_stride_inside_the_block_is_refused(self):
        """A strided base block cannot compose, so it must raise, not mis-build.

        The deep branch applies the base block ``2^(depth-1)`` times; at stride 2
        it would downsample that many times against the shallow branch's once,
        and the join would receive mismatched shapes.
        """
        cfg = ConvBlock(filters=4, kernel_size=3, strides=2,
                        padding="same", use_pooling=False).get_config()
        with pytest.raises(ValueError, match="constant resolution|strides"):
            FractalBlock(block_config=cfg, depth=3)


class TestLocalDropPathNeverZeroesTheBlock:
    """The join must always keep at least one live path.

    The previous implementation applied an independent ``StochasticDepth`` to
    each branch and then scaled both by a fixed 0.5, so when both draws dropped
    -- probability ``drop_path_rate ** 2``, about 2.3% per sample at the 0.15
    default -- the block emitted EXACTLY ZERO for that sample and destroyed its
    signal. The join now revives one branch by a fair coin in that case.
    """

    def test_output_is_never_all_zero_under_training(self):
        cfg = ConvBlock(filters=F, kernel_size=3, strides=1, padding="same",
                        use_pooling=False).get_config()
        # A high rate makes the both-dropped case near-certain if unhandled:
        # at 0.9 it is 81% per sample, so 64 samples over 20 draws would hit it
        # thousands of times.
        block = FractalBlock(block_config=cfg, depth=2, drop_path_rate=0.9)
        x = np.random.default_rng(0).standard_normal((64, H, W, C)).astype("float32")

        for _ in range(20):
            y = keras.ops.convert_to_numpy(block(x, training=True))
            per_sample = np.abs(y).sum(axis=(1, 2, 3))
            assert np.all(per_sample > 0.0), (
                "at least one branch must survive every draw; a zero row means "
                "both paths were dropped and the block became a zero map"
            )

    def test_inference_is_the_plain_mean(self):
        """At ``training=False`` the join is deterministic regardless of rate."""
        cfg = ConvBlock(filters=F, kernel_size=3, strides=1, padding="same",
                        use_pooling=False).get_config()
        block = FractalBlock(block_config=cfg, depth=2, drop_path_rate=0.9)
        x = np.random.default_rng(1).standard_normal((4, H, W, C)).astype("float32")
        a = keras.ops.convert_to_numpy(block(x, training=False))
        b = keras.ops.convert_to_numpy(block(x, training=False))
        np.testing.assert_allclose(a, b, rtol=0, atol=0)
