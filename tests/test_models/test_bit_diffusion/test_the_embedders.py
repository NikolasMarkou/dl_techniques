"""The two package-local embedders: the timestep MLP and the 2D sin-cos table.

Both are numerically specified and both have a plausible substitute that is
WRONG. This file is the guard that makes each substitution red rather than
silent.

What is pinned, and why a cheaper assertion would not do:

1. **The timestep embedder is not** ``ScalarSinusoidalEmbedding``. Three
   independent numeric axes plus one structural one. A shape test, a config
   round trip and a finiteness test all pass under every one of the three
   swaps, because a sinusoid of the wrong argument is still a finite number of
   the right width. So each difference gets its own value-level arm, with the
   measured delta written into the assertion's message.
2. **The 2D positional table's halves are ordered.** ``embed_dim // 2`` columns
   encode the COLUMN index, then ``embed_dim // 2`` encode the ROW index. On a
   square grid the swapped table has the same shape, the same dtype, the same
   set of rows, the same per-column mean and the same Frobenius norm -- it is a
   permutation of the rows. Only an ELEMENTWISE comparison against an
   independently written formula sees it, which is what
   ``test_the_2d_table_matches_a_hand_written_formula_elementwise`` is.

The expected values here are written from the mathematical definition
(``sin(c * omega_j)``, ``cos(r * omega_j)`` with ``omega = [1.0, 0.01]``), never
by calling the code under test. Re-deriving the implementation's own arithmetic
would make the guard vacuous.
"""

import math

import keras
import numpy as np
import pytest

from dl_techniques.layers.embedding.scalar_sinusoidal_embedding import (
    ScalarSinusoidalEmbedding,
)
from dl_techniques.models.vision_language.bit_diffusion.blocks import (
    DiTXATimestepEmbedder,
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
    get_2d_sincos_pos_embed_from_grid,
)

# The upstream time scale. The MODEL multiplies by this before calling the
# embedder; the embedder itself must not.
TIME_SCALE = 1000.0


def _np(x):
    return keras.ops.convert_to_numpy(x)


# =====================================================================
# The timestep embedder's own contract
# =====================================================================


class TestTimestepEmbedderBasis:
    """The sinusoidal basis, isolated from the MLP."""

    def test_the_concat_is_cos_first_not_sin_first(self):
        # At t = 0 every argument is 0, so cos = 1 and sin = 0. Cos-first means
        # the FIRST half is ones. This single reading separates the two orders
        # with the largest gap a bounded sinusoid admits (1.0).
        emb = DiTXATimestepEmbedder(hidden_size=4, frequency_embedding_size=8)
        emb.build((None,))
        basis = _np(emb.timestep_embedding(keras.ops.zeros((1,))))
        half = 4
        np.testing.assert_allclose(basis[0, :half], np.ones(half), atol=0.0)
        np.testing.assert_allclose(basis[0, half:], np.zeros(half), atol=0.0)

    def test_the_frequency_ladder_divides_by_half_not_by_half_minus_one(self):
        # freqs_i = exp(-log(max_period) * i / half). Written out here from the
        # definition, not read back from the layer.
        freq_size = 8
        half = freq_size // 2
        emb = DiTXATimestepEmbedder(
            hidden_size=4, frequency_embedding_size=freq_size
        )
        emb.build((None,))
        got = _np(emb.freqs)

        expected = np.exp(
            -math.log(10000.0) * np.arange(half, dtype="float32") / half
        )
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=0.0)

        # The ladder never reaches 1 / max_period: that is the whole difference
        # from a `half - 1` denominator.
        assert float(got[-1]) > 1e-4
        house_ladder = np.exp(
            np.arange(half) * -(math.log(1e4) / (half - 1))
        )
        assert float(house_ladder[-1]) == pytest.approx(1e-4, rel=1e-6)
        assert float(got[-1]) / float(house_ladder[-1]) == pytest.approx(
            10.0, rel=1e-5
        )

    def test_the_layer_does_not_rescale_its_input(self):
        # args = t * freqs, with t used AS GIVEN. The house layer would map
        # t = 0.25 onto 2500.0 first.
        freq_size = 8
        emb = DiTXATimestepEmbedder(
            hidden_size=4, frequency_embedding_size=freq_size
        )
        emb.build((None,))
        freqs = _np(emb.freqs)

        t = np.array([0.25], dtype="float32")
        got = _np(emb.timestep_embedding(keras.ops.convert_to_tensor(t)))
        args = t[:, None] * freqs[None]
        expected = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-7)

        # And a rescaled input is a materially different embedding, so the
        # absence of the rescale is observable rather than cosmetic.
        rescaled = _np(
            emb.timestep_embedding(keras.ops.convert_to_tensor(1e4 * t))
        )
        assert float(np.max(np.abs(rescaled - got))) > 0.5

    def test_an_odd_frequency_size_pads_one_trailing_zero(self):
        emb = DiTXATimestepEmbedder(hidden_size=4, frequency_embedding_size=7)
        emb.build((None,))
        basis = _np(emb.timestep_embedding(keras.ops.ones((3,))))
        assert basis.shape == (3, 7)
        # half = 3, so 6 real columns then one zero column. Padding, not a
        # dropped frequency.
        np.testing.assert_allclose(basis[:, 6], np.zeros(3), atol=0.0)
        assert float(np.min(np.abs(basis[:, :6]))) > 0.0


class TestTimestepEmbedderIsNotTheHouseLayer:
    """The three numeric differences and the structural one, side by side.

    Each arm constructs BOTH layers at the same width and shows the actual
    divergence. The point is not that they differ -- it is that they differ in
    three separately identifiable ways, so a partial "simplification" (say,
    flipping the concat but keeping the ladder) is caught too.
    """

    WIDTH = 8

    def _both(self):
        house = ScalarSinusoidalEmbedding(dim=self.WIDTH)
        house.build((None,))
        ours = DiTXATimestepEmbedder(
            hidden_size=self.WIDTH, frequency_embedding_size=self.WIDTH
        )
        ours.build((None,))
        return house, ours

    def test_the_house_layer_still_exposes_the_internals_this_guard_reads(self):
        # Anti-rot arm. If `_sinusoidal` or `freq` is renamed, this file's
        # comparison would silently stop comparing; it must fail loudly first.
        house, _ = self._both()
        assert hasattr(house, "_sinusoidal")
        assert hasattr(house, "freq")

    def test_difference_1_concat_order(self):
        house, ours = self._both()
        half = self.WIDTH // 2
        zeros = keras.ops.zeros((1,))
        house_basis = _np(house._sinusoidal(zeros))
        ours_basis = _np(ours.timestep_embedding(zeros))
        # sin-first vs cos-first, at the one input where the two halves are
        # maximally separated.
        np.testing.assert_allclose(house_basis[0, :half], 0.0, atol=0.0)
        np.testing.assert_allclose(house_basis[0, half:], 1.0, atol=0.0)
        np.testing.assert_allclose(ours_basis[0, :half], 1.0, atol=0.0)
        np.testing.assert_allclose(ours_basis[0, half:], 0.0, atol=0.0)
        assert float(np.max(np.abs(house_basis - ours_basis))) == pytest.approx(
            1.0, abs=1e-6
        )

    def test_difference_2_frequency_ladder(self):
        house, ours = self._both()
        house_freq = _np(house.freq)
        our_freq = _np(ours.freqs)
        assert house_freq.shape == our_freq.shape  # same width, different values
        assert float(np.max(np.abs(house_freq - our_freq))) > 0.0
        # The measured last-entry ratio at half = 4.
        assert float(our_freq[-1] / house_freq[-1]) == pytest.approx(
            10.0, rel=1e-4
        )

    def test_difference_3_input_rescale(self):
        house, ours = self._both()
        t = keras.ops.convert_to_tensor(
            np.array([0.0, 0.25, 0.5, 1.0], dtype="float32")
        )
        # `call` applies the house rescale; `_sinusoidal` does not. Feeding the
        # SAME scaled value to both bases isolates the rescale from the ladder.
        our_freq = _np(ours.freqs)
        t_np = np.array([0.0, 0.25, 0.5, 1.0], dtype="float32")
        unrescaled = np.concatenate(
            [
                np.cos(t_np[:, None] * our_freq[None]),
                np.sin(t_np[:, None] * our_freq[None]),
            ],
            axis=-1,
        )
        rescaled = np.concatenate(
            [
                np.cos(1e4 * t_np[:, None] * our_freq[None]),
                np.sin(1e4 * t_np[:, None] * our_freq[None]),
            ],
            axis=-1,
        )
        # The house rescale alone -- same ladder, same concat order -- moves the
        # embedding by more than a unit.
        assert float(np.max(np.abs(rescaled - unrescaled))) > 1.0

        got = _np(ours.timestep_embedding(t))
        np.testing.assert_allclose(got, unrescaled, rtol=1e-6, atol=1e-6)

    def test_difference_4_the_frequency_width_is_decoupled_from_hidden_size(self):
        hidden = 64
        ours = DiTXATimestepEmbedder(
            hidden_size=hidden, frequency_embedding_size=256
        )
        ours.build((None,))
        assert tuple(ours.mlp_in.kernel.shape) == (256, hidden)
        assert tuple(ours.mlp_out.kernel.shape) == (hidden, hidden)

        house = ScalarSinusoidalEmbedding(dim=hidden)
        house.build((None,))
        # One `dim` drives both, so the house layer cannot express 256 -> 64.
        assert tuple(house.mlp_in.kernel.shape) == (hidden, hidden)


class TestTimestepEmbedderMechanics:

    def test_output_shape_and_rank_1_or_2_input(self):
        emb = DiTXATimestepEmbedder(hidden_size=16, frequency_embedding_size=32)
        t = keras.ops.convert_to_tensor(
            np.array([0.0, 0.5, 1.0], dtype="float32") * TIME_SCALE
        )
        flat = _np(emb(t))
        column = _np(emb(keras.ops.reshape(t, (3, 1))))
        assert flat.shape == (3, 16)
        np.testing.assert_allclose(flat, column, atol=0.0)
        assert emb.compute_output_shape((None,)) == (None, 16)
        assert emb.compute_output_shape((3, 1)) == (3, 16)

    def test_the_frequency_ladder_is_a_non_trainable_weight(self):
        emb = DiTXATimestepEmbedder(hidden_size=8, frequency_embedding_size=8)
        emb.build((None,))
        names = [w.name for w in emb.weights]
        trainable = [w.name for w in emb.trainable_weights]
        assert "freqs" in names
        assert "freqs" not in trainable

    def test_the_two_dense_layers_do_not_share_an_initializer_instance(self):
        emb = DiTXATimestepEmbedder(hidden_size=8)
        assert (
            emb.mlp_in.kernel_initializer is not emb.mlp_out.kernel_initializer
        )

    def test_config_round_trip_preserves_every_knob(self):
        emb = DiTXATimestepEmbedder(
            hidden_size=12,
            frequency_embedding_size=10,
            max_period=5000.0,
            kernel_stddev=0.05,
        )
        restored = DiTXATimestepEmbedder.from_config(emb.get_config())
        assert restored.hidden_size == 12
        assert restored.frequency_embedding_size == 10
        assert restored.max_period == 5000.0
        assert restored.kernel_stddev == 0.05

    def test_the_ladder_survives_a_keras_round_trip(self, tmp_path):
        # The reason `freqs` is a weight and not a plain attribute.
        inputs = keras.Input(shape=(), dtype="float32")
        outputs = DiTXATimestepEmbedder(
            hidden_size=8, frequency_embedding_size=8, name="temb"
        )(inputs)
        model = keras.Model(inputs, outputs)
        t = keras.ops.convert_to_tensor(
            np.array([0.0, 0.3, 0.9], dtype="float32")
        )
        before = _np(model(t, training=False))

        path = tmp_path / "temb.keras"
        model.save(path)
        reloaded = keras.models.load_model(path)
        after = _np(reloaded(t, training=False))
        np.testing.assert_allclose(before, after, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(
            _np(reloaded.get_layer("temb").freqs),
            _np(model.get_layer("temb").freqs),
            atol=0.0,
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(hidden_size=0),
            dict(hidden_size=8, frequency_embedding_size=1),
            dict(hidden_size=8, max_period=1.0),
            dict(hidden_size=8, kernel_stddev=0.0),
        ],
    )
    def test_invalid_arguments_raise(self, kwargs):
        with pytest.raises(ValueError):
            DiTXATimestepEmbedder(**kwargs)


# =====================================================================
# The 2D sin-cos positional table
# =====================================================================

#: The 1D angular frequencies at ``embed_dim // 2 = 4``: ``omega_j = 1 /
#: 10000 ** (j / 2)`` for ``j in {0, 1}``. Written as literals so the expected
#: table below owes nothing to the code under test.
OMEGA_AT_HALF_4 = (1.0, 0.01)


def _expected_table_embed8_grid4():
    """The (16, 8) table at ``embed_dim=8, grid_size=4``, from the definition.

    Row ``m`` is grid position ``(row=m // 4, col=m % 4)`` and holds::

        [sin(col*w0), sin(col*w1), cos(col*w0), cos(col*w1),
         sin(row*w0), sin(row*w1), cos(row*w0), cos(row*w1)]

    COLUMN first, ROW second -- that is the whole content of the ordering claim.
    """
    w0, w1 = OMEGA_AT_HALF_4
    rows = []
    for m in range(16):
        r, c = m // 4, m % 4
        rows.append(
            [
                math.sin(c * w0), math.sin(c * w1),
                math.cos(c * w0), math.cos(c * w1),
                math.sin(r * w0), math.sin(r * w1),
                math.cos(r * w0), math.cos(r * w1),
            ]
        )
    return np.array(rows, dtype="float64")


class TestTwoDimensionalSinCosTable:

    def test_the_2d_table_matches_a_hand_written_formula_elementwise(self):
        got = get_2d_sincos_pos_embed(embed_dim=8, grid_size=4)
        expected = _expected_table_embed8_grid4()
        assert got.shape == expected.shape == (16, 8)
        np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-15)

    def test_a_swapped_half_order_is_a_different_table(self):
        # Anti-vacuity for the arm above: prove the assertion it makes could
        # fail, and that a SHAPE test could not. Both halves are 4 wide, the
        # grid is square, so the swap is a row permutation -- same shape, same
        # sorted contents, same Frobenius norm.
        expected = _expected_table_embed8_grid4()
        swapped = np.concatenate([expected[:, 4:], expected[:, :4]], axis=1)
        assert swapped.shape == expected.shape
        assert np.linalg.norm(swapped) == pytest.approx(
            np.linalg.norm(expected), rel=1e-12
        )
        np.testing.assert_allclose(
            np.sort(swapped, axis=None), np.sort(expected, axis=None), atol=1e-15
        )
        assert float(np.max(np.abs(swapped - expected))) > 1.0

    def test_the_first_half_moves_with_the_column_and_the_second_with_the_row(
        self,
    ):
        # A second, independent reading of the same fact, stated as a
        # difference rather than as absolute values.
        table = get_2d_sincos_pos_embed(embed_dim=8, grid_size=4)
        origin = table[0]       # (row 0, col 0)
        same_row = table[1]     # (row 0, col 1) -- only the column moved
        same_col = table[4]     # (row 1, col 0) -- only the row moved

        np.testing.assert_allclose(same_row[4:], origin[4:], atol=1e-15)
        assert float(np.max(np.abs(same_row[:4] - origin[:4]))) > 0.5

        np.testing.assert_allclose(same_col[:4], origin[:4], atol=1e-15)
        assert float(np.max(np.abs(same_col[4:] - origin[4:]))) > 0.5

    def test_the_meshgrid_is_w_first(self):
        # `np.meshgrid(grid_w, grid_h)` -> grid[0] is the COLUMN index. Read the
        # consequence off the table rather than the source: at (row 0, col 1)
        # the first half must encode 1 and the second half 0.
        table = get_2d_sincos_pos_embed(embed_dim=8, grid_size=4)
        w0, w1 = OMEGA_AT_HALF_4
        np.testing.assert_allclose(
            table[1],
            [
                math.sin(w0), math.sin(w1), math.cos(w0), math.cos(w1),
                0.0, 0.0, 1.0, 1.0,
            ],
            rtol=0.0,
            atol=1e-15,
        )

    def test_the_1d_helper_is_sin_first(self):
        # Deliberately the OPPOSITE of the timestep embedder. Both orders are
        # upstream; unifying them would break one of the two.
        emb = get_1d_sincos_pos_embed_from_grid(4, np.zeros((3,)))
        np.testing.assert_allclose(emb[:, :2], np.zeros((3, 2)), atol=0.0)
        np.testing.assert_allclose(emb[:, 2:], np.ones((3, 2)), atol=0.0)

    def test_shapes_and_the_cls_token_branch(self):
        assert get_2d_sincos_pos_embed(16, 8).shape == (64, 16)
        with_cls = get_2d_sincos_pos_embed(
            8, 4, cls_token=True, extra_tokens=2
        )
        assert with_cls.shape == (18, 8)
        np.testing.assert_allclose(with_cls[:2], np.zeros((2, 8)), atol=0.0)
        np.testing.assert_allclose(
            with_cls[2:], get_2d_sincos_pos_embed(8, 4), atol=0.0
        )
        # Upstream prepends only when BOTH flags are set.
        assert get_2d_sincos_pos_embed(8, 4, cls_token=True).shape == (16, 8)
        assert get_2d_sincos_pos_embed(
            8, 4, cls_token=False, extra_tokens=2
        ).shape == (16, 8)

    @pytest.mark.parametrize(
        "call",
        [
            lambda: get_2d_sincos_pos_embed(8, 0),
            lambda: get_2d_sincos_pos_embed_from_grid(7, np.zeros((2, 4))),
            lambda: get_1d_sincos_pos_embed_from_grid(3, np.zeros((4,))),
            lambda: get_1d_sincos_pos_embed_from_grid(0, np.zeros((4,))),
        ],
    )
    def test_invalid_arguments_raise(self, call):
        with pytest.raises(ValueError):
            call()

    def test_the_table_installs_as_a_non_trainable_constant_weight(self, tmp_path):
        # The documented install pattern, exercised end to end: Constant
        # initializer into add_weight(trainable=False). Neither a plain tensor
        # attribute nor an .assign() in build().
        table = get_2d_sincos_pos_embed(embed_dim=8, grid_size=4)

        class _Holder(keras.layers.Layer):
            def build(self, input_shape):
                self.pos_embed = self.add_weight(
                    name="pos_embed",
                    shape=table.shape,
                    initializer=keras.initializers.Constant(
                        table.astype("float32")
                    ),
                    trainable=False,
                )
                super().build(input_shape)

            def call(self, inputs):
                return inputs + self.pos_embed

        holder = _Holder()
        holder.build((None, 16, 8))
        assert holder.trainable_weights == []
        np.testing.assert_allclose(
            _np(holder.pos_embed), table.astype("float32"), rtol=1e-6, atol=1e-7
        )
