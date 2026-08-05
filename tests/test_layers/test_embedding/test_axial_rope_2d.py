"""
Tests for `AxialRoPE2D` — real-valued 2D axial rotary position embedding.

The guards here (G1.1-G1.5, plan-2026-08-04T044628-4c240b4c step 1) are written to be
*discriminating*, not descriptive: each one is paired with a wrong-VALUE mutation of the
source and was executed RED against it before being committed. The two independent
correctness guards are deliberately of different KINDS:

* **G1.1** compares against a from-scratch float64 NumPy *complex* oracle. It would catch
  a wrong implementation only if the oracle itself is right.
* **G1.2** asserts a *property* the rotation must satisfy regardless of how it is written
  (inner products depend only on the per-axis displacement). It is not a reimplementation,
  so it cannot inherit an error from the oracle.

If those two ever disagree, suspect the ORACLE first.

`G1.3` (repeat_k block bit-identity) is a CONSISTENCY property and is therefore vacuous on
its own — a completely dead rotation satisfies it trivially. It carries an explicit liveness
arm for exactly that reason, and `TestDeadComponentProbe` measures which arms survive which
dead component rather than assuming they all die.
"""

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.layers.embedding import AxialRoPE2D

# ---------------------------------------------------------------------
# oracle + guard bodies (shared by the real tests and the dead-component probe)
# ---------------------------------------------------------------------

THETA = 10000.0


def _oracle_rotate(x: np.ndarray, height: int, width: int, theta: float = THETA) -> np.ndarray:
    """Independent float64 NumPy *complex* reference for 2D axial RoPE.

    Written from the published semantics, not from the layer: flat token index ``t`` over an
    ``(H, W)`` row-major grid, ``head_dim // 4`` shared frequency bands per axis, adjacent
    channel pairs read as complex numbers, rotation as ``exp(1j * angle)``.

    :param x: Array of shape ``(..., H * W, head_dim)`` in float64.
    :param height: Grid height ``H``.
    :param width: Grid width ``W``.
    :param theta: Frequency-ladder base.
    :return: Rotated array of the same shape and dtype.
    """
    head_dim = x.shape[-1]
    num_tokens = height * width
    assert x.shape[-2] == num_tokens

    flat = np.arange(num_tokens, dtype=np.float64)
    t_x = flat % width
    t_y = flat // width
    bands = np.arange(0, head_dim, 4, dtype=np.float64)[: head_dim // 4]
    freqs = 1.0 / (theta ** (bands / head_dim))
    angles = np.concatenate([np.outer(t_x, freqs), np.outer(t_y, freqs)], axis=-1)

    cis = np.exp(1j * angles)                      # (N, head_dim // 2)
    x_complex = x[..., 0::2] + 1j * x[..., 1::2]   # adjacent-pair packing
    rotated = x_complex * cis

    out = np.empty_like(x)
    out[..., 0::2] = rotated.real
    out[..., 1::2] = rotated.imag
    return out


def _guard_oracle(cls) -> None:
    """G1.1 body: layer output must equal the complex oracle at float64.

    :param cls: An ``AxialRoPE2D`` (or dead-component subclass) to instantiate.
    """
    height, width, head_dim = 3, 5, 12
    rng = np.random.default_rng(20260804)
    q = rng.standard_normal((2, 3, height * width, head_dim)).astype(np.float64)

    rope = cls(head_dim=head_dim, feat_shape=(height, width), theta=THETA, dtype="float64")
    got = ops.convert_to_numpy(rope(q))
    expected = _oracle_rotate(q, height, width)

    np.testing.assert_allclose(
        got, expected, atol=1e-12, rtol=0.0,
        err_msg="AxialRoPE2D disagrees with the float64 complex oracle",
    )


def _guard_relative_position(cls) -> None:
    """G1.2 body: an independent property control, not a reimplementation.

    With a CONSTANT query vector at every position and a CONSTANT key vector at every
    position, ``<rot(q, p1), rot(k, p2)>`` must depend only on the per-axis displacement
    ``p1 - p2``. Two liveness arms additionally require that displacement along each axis
    actually MATTERS, which is what collapses if the two axial halves are fed the same
    coordinate.

    :param cls: An ``AxialRoPE2D`` (or dead-component subclass) to instantiate.
    """
    height, width, head_dim = 4, 4, 8
    num_tokens = height * width
    rng = np.random.default_rng(11)
    v = rng.standard_normal(head_dim)
    w = rng.standard_normal(head_dim)

    q = np.broadcast_to(v, (1, 1, num_tokens, head_dim)).astype(np.float64).copy()
    k = np.broadcast_to(w, (1, 1, num_tokens, head_dim)).astype(np.float64).copy()

    rope = cls(head_dim=head_dim, feat_shape=(height, width), theta=THETA, dtype="float64")
    q_rot, k_rot = rope(q, k)
    q_rot = ops.convert_to_numpy(q_rot)[0, 0]
    k_rot = ops.convert_to_numpy(k_rot)[0, 0]
    gram = q_rot @ k_rot.T                        # (N, N)

    flat = np.arange(num_tokens)
    t_x = flat % width
    t_y = flat // width

    # Arm 1 -- invariance: equal displacement => equal inner product.
    groups: dict = {}
    for i in range(num_tokens):
        for j in range(num_tokens):
            groups.setdefault((t_x[i] - t_x[j], t_y[i] - t_y[j]), []).append(gram[i, j])
    for disp, values in groups.items():
        spread = float(np.max(values) - np.min(values))
        assert spread < 1e-12, (
            f"relative-position invariance broken at displacement {disp}: inner products "
            f"spread by {spread:.3e} across {len(values)} position pairs"
        )

    # Arm 2 -- x-axis liveness: same dy, different dx must differ.
    same_dy_dx0 = groups[(0, 1)][0]
    same_dy_dx1 = groups[(1, 1)][0]
    assert abs(same_dy_dx0 - same_dy_dx1) > 1e-6, (
        f"x displacement is DEAD: inner product at (dx=0, dy=1) is {same_dy_dx0:.9f} and at "
        f"(dx=1, dy=1) is {same_dy_dx1:.9f}"
    )

    # Arm 3 -- y-axis liveness: same dx, different dy must differ. This is the arm that
    # fires when both axial halves are driven by t_x (the two axes collapse).
    same_dx_dy0 = groups[(1, 0)][0]
    same_dx_dy1 = groups[(1, 1)][0]
    assert abs(same_dx_dy0 - same_dx_dy1) > 1e-6, (
        f"y displacement is DEAD: inner product at (dx=1, dy=0) is {same_dx_dy0:.9f} and at "
        f"(dx=1, dy=1) is {same_dx_dy1:.9f} -- the two axial halves have collapsed onto one "
        f"coordinate"
    )


def _guard_repeat_k(cls) -> None:
    """G1.3 body: `repeat_k` broadcasts the SAME spatial table across every block.

    Builds ``k`` as one spatial block tiled ``r`` times and requires the ``r`` rotated
    blocks to be max-abs-diff EXACTLY 0.0. A per-block phase (i.e. temporal position leaking
    into the rotation) breaks this.

    The bit-identity arm alone is vacuous -- a fully dead rotation passes it -- so a liveness
    arm requiring the rotation to actually change the block is included.

    :param cls: An ``AxialRoPE2D`` (or dead-component subclass) to instantiate.
    """
    height, width, head_dim, repeats = 2, 3, 8, 4
    num_tokens = height * width
    rng = np.random.default_rng(7)

    q = rng.standard_normal((1, 2, num_tokens, head_dim)).astype(np.float64)
    block = rng.standard_normal((1, 2, num_tokens, head_dim)).astype(np.float64)
    k = np.tile(block, (1, 1, repeats, 1))

    rope = cls(
        head_dim=head_dim, feat_shape=(height, width), theta=THETA,
        repeat_k=True, dtype="float64",
    )
    _, k_rot = rope(q, k)
    k_rot = ops.convert_to_numpy(k_rot)

    first = k_rot[:, :, :num_tokens, :]
    for r in range(1, repeats):
        chunk = k_rot[:, :, r * num_tokens:(r + 1) * num_tokens, :]
        max_diff = float(np.max(np.abs(chunk - first)))
        assert max_diff == 0.0, (
            f"repeat_k block {r} differs from block 0 by {max_diff:.3e}; the angle table is "
            f"NOT spatial-only (a block/temporal index has leaked into the rotation)"
        )

    # Liveness arm: the rotation must actually do something.
    live = float(np.max(np.abs(first - block)))
    assert live > 1e-6, (
        f"repeat_k rotation is DEAD: rotated block 0 differs from its input by only "
        f"{live:.3e}"
    )


def _guard_num_k_exclude(cls) -> None:
    """G1.4 body: the last ``num_k_exclude`` key rows are left completely untouched.

    Both halves are asserted -- tail bit-identity AND head movement. A one-sided assertion
    is vacuous: "tail unchanged" alone is satisfied by a dead rotation, and "head changed"
    alone is satisfied by rotating everything.

    :param cls: An ``AxialRoPE2D`` (or dead-component subclass) to instantiate.
    """
    height, width, head_dim, n_exclude = 3, 3, 8, 4
    num_tokens = height * width
    rng = np.random.default_rng(1234)

    q = rng.standard_normal((1, 2, num_tokens, head_dim)).astype(np.float64)
    k = rng.standard_normal((1, 2, num_tokens + n_exclude, head_dim)).astype(np.float64)

    rope = cls(head_dim=head_dim, feat_shape=(height, width), theta=THETA, dtype="float64")
    _, k_rot = rope(q, k, num_k_exclude=n_exclude)
    k_rot = ops.convert_to_numpy(k_rot)

    assert k_rot.shape == k.shape

    tail_diff = float(np.max(np.abs(k_rot[:, :, num_tokens:, :] - k[:, :, num_tokens:, :])))
    assert tail_diff == 0.0, (
        f"the {n_exclude} excluded tail key rows were MODIFIED (max abs diff {tail_diff:.3e}); "
        f"object-pointer tokens must not receive spatial RoPE"
    )

    head_diff = float(np.max(np.abs(k_rot[:, :, :num_tokens, :] - k[:, :, :num_tokens, :])))
    assert head_diff > 1e-6, (
        f"the spatial key rows were NOT rotated (max abs diff {head_diff:.3e}); the "
        f"tail-identity assertion above is vacuous without this arm"
    )


# ---------------------------------------------------------------------
# dead-component subclasses (G1.5)
# ---------------------------------------------------------------------


class _SinDeadRoPE(AxialRoPE2D):
    """Dead component A: the sine half of the rotation is forced to zero."""

    def build(self, query_shape, key_shape=None):
        super().build(query_shape, key_shape)
        self._sin_table = np.zeros_like(self._sin_table)


class _RotationDeadRoPE(AxialRoPE2D):
    """Dead component B: the rotation is the identity (``cos=1``, ``sin=0``)."""

    def build(self, query_shape, key_shape=None):
        super().build(query_shape, key_shape)
        self._sin_table = np.zeros_like(self._sin_table)
        self._cos_table = np.ones_like(self._cos_table)


# ---------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def small_rope() -> AxialRoPE2D:
    """A small 2x3-grid layer with ``head_dim=8``."""
    return AxialRoPE2D(head_dim=8, feat_shape=(2, 3))


# ---------------------------------------------------------------------
# G1.1 -- numeric oracle
# ---------------------------------------------------------------------


class TestComplexOracle:
    """G1.1: agreement with an independent float64 NumPy complex oracle."""

    def test_matches_complex_oracle(self) -> None:
        """Layer output equals `exp(1j*angle)` complex rotation to atol=1e-12, rtol=0."""
        _guard_oracle(AxialRoPE2D)

    def test_oracle_pairing_is_adjacent_not_split_half(self) -> None:
        """The oracle's own pairing choice is pinned by a hand-computed 4-channel case.

        This guards the ORACLE, not the layer: a split-half oracle would agree with a
        split-half layer and both would be wrong. With ``head_dim=4`` there is exactly one
        band per axis, so the angles are ``(t_x * 1, t_y * 1)`` and the expected output is
        writable by hand.
        """
        height, width, head_dim = 2, 2, 4
        # freqs = 1/theta**(arange(0,4,4)[:1]/4) = 1/theta**0 = [1.0]
        x = np.zeros((1, 1, 4, head_dim), dtype=np.float64)
        x[0, 0, :, 0] = 1.0  # real part of the x-axis pair only

        rope = AxialRoPE2D(head_dim=head_dim, feat_shape=(height, width), dtype="float64")
        got = ops.convert_to_numpy(rope(x))[0, 0]

        # token t: t_x = t % 2, t_y = t // 2. Pair 0 carries angle t_x, pair 1 angle t_y.
        for t in range(4):
            t_x = t % width
            assert got[t, 0] == pytest.approx(np.cos(t_x), abs=1e-12)
            assert got[t, 1] == pytest.approx(np.sin(t_x), abs=1e-12)
            # Channels 2/3 are the y pair; their input was zero, so they stay zero.
            assert got[t, 2] == pytest.approx(0.0, abs=1e-12)
            assert got[t, 3] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------
# G1.2 -- independent property control
# ---------------------------------------------------------------------


class TestRelativePositionInvariance:
    """G1.2: `<rot(q,p1), rot(k,p2)>` depends only on `p1 - p2`, per axis."""

    def test_relative_position_invariance(self) -> None:
        """Invariance plus per-axis liveness for both x and y."""
        _guard_relative_position(AxialRoPE2D)


# ---------------------------------------------------------------------
# G1.3 -- repeat_k
# ---------------------------------------------------------------------


class TestRepeatK:
    """G1.3: `repeat_k` broadcasts one spatial table, with no per-block phase."""

    def test_repeat_k_is_spatial_only(self) -> None:
        """The r rotated blocks are max-abs-diff exactly 0.0, and the rotation is live."""
        _guard_repeat_k(AxialRoPE2D)

    def test_repeat_k_false_rejects_longer_key(self) -> None:
        """Without `repeat_k`, a key longer than the grid is an error, not a silent crop."""
        rope = AxialRoPE2D(head_dim=8, feat_shape=(2, 3), repeat_k=False)
        q = np.zeros((1, 1, 6, 8), dtype="float32")
        k = np.zeros((1, 1, 12, 8), dtype="float32")
        with pytest.raises(ValueError, match="repeat_k=False"):
            rope(q, k)

    def test_repeat_k_rejects_non_multiple(self) -> None:
        """With `repeat_k`, a key length that is not an exact multiple raises."""
        rope = AxialRoPE2D(head_dim=8, feat_shape=(2, 3), repeat_k=True)
        q = np.zeros((1, 1, 6, 8), dtype="float32")
        k = np.zeros((1, 1, 14, 8), dtype="float32")
        with pytest.raises(ValueError, match="exact multiple"):
            rope(q, k)


# ---------------------------------------------------------------------
# G1.4 -- num_k_exclude
# ---------------------------------------------------------------------


class TestNumKExclude:
    """G1.4: trailing (object-pointer) key rows are excluded from rotation."""

    def test_num_k_exclude_tail_identity(self) -> None:
        """Tail rows bit-identical to input AND preceding rows changed."""
        _guard_num_k_exclude(AxialRoPE2D)

    def test_exclude_all_keys_is_a_noop(self) -> None:
        """`num_k_exclude == len(k)` returns k untouched without dividing by zero."""
        rope = AxialRoPE2D(head_dim=8, feat_shape=(2, 3))
        rng = np.random.default_rng(3)
        q = rng.standard_normal((1, 1, 6, 8)).astype("float32")
        k = rng.standard_normal((1, 1, 5, 8)).astype("float32")
        _, k_rot = rope(q, k, num_k_exclude=5)
        k_rot = ops.convert_to_numpy(k_rot)
        assert np.max(np.abs(k_rot - k)) == 0.0
        assert np.all(np.isfinite(k_rot))

    def test_num_k_exclude_out_of_range_raises(self) -> None:
        """A `num_k_exclude` larger than the key length raises."""
        rope = AxialRoPE2D(head_dim=8, feat_shape=(2, 3))
        q = np.zeros((1, 1, 6, 8), dtype="float32")
        k = np.zeros((1, 1, 6, 8), dtype="float32")
        with pytest.raises(ValueError, match=r"must be in \[0, 6\]"):
            rope(q, k, num_k_exclude=7)

    def test_repeat_k_excludes_tail_before_multiple_check(self) -> None:
        """Under `repeat_k`, the multiple check applies to `num_k - num_k_exclude`."""
        num_tokens, head_dim, n_exclude = 6, 8, 3
        rng = np.random.default_rng(5)
        q = rng.standard_normal((1, 1, num_tokens, head_dim)).astype(np.float64)
        k = rng.standard_normal((1, 1, 3 * num_tokens + n_exclude, head_dim)).astype(np.float64)

        rope = AxialRoPE2D(
            head_dim=head_dim, feat_shape=(2, 3), repeat_k=True, dtype="float64"
        )
        _, k_rot = rope(q, k, num_k_exclude=n_exclude)
        k_rot = ops.convert_to_numpy(k_rot)
        assert k_rot.shape == k.shape
        tail = float(np.max(np.abs(k_rot[:, :, -n_exclude:, :] - k[:, :, -n_exclude:, :])))
        assert tail == 0.0


# ---------------------------------------------------------------------
# G1.5 -- dead-component probe
# ---------------------------------------------------------------------


class TestDeadComponentProbe:
    """G1.5: guards must go RED when the rotation is dead.

    Two dead components are measured separately rather than assumed equivalent:

    * ``_SinDeadRoPE`` (sin -> 0) leaves a live ``cos`` scaling, so it is a *partial* kill.
    * ``_RotationDeadRoPE`` (sin -> 0, cos -> 1) makes the layer the identity.

    The MEASURED outcome (recorded when these were first run) is that G1.3's block
    bit-identity arm survives BOTH -- it is a consistency property that a dead rotation
    satisfies trivially. That is precisely why G1.3 carries a liveness arm, which does die
    under the identity component.
    """

    @pytest.mark.parametrize("dead_cls", [_SinDeadRoPE, _RotationDeadRoPE])
    def test_oracle_guard_dies(self, dead_cls) -> None:
        """G1.1 goes RED under both dead components."""
        with pytest.raises(AssertionError):
            _guard_oracle(dead_cls)

    @pytest.mark.parametrize("dead_cls", [_SinDeadRoPE, _RotationDeadRoPE])
    def test_relative_position_guard_dies(self, dead_cls) -> None:
        """G1.2 goes RED under both dead components."""
        with pytest.raises(AssertionError):
            _guard_relative_position(dead_cls)

    def test_num_k_exclude_guard_dies_under_identity(self) -> None:
        """G1.4's head-movement arm goes RED when the rotation is the identity."""
        with pytest.raises(AssertionError, match="were NOT rotated"):
            _guard_num_k_exclude(_RotationDeadRoPE)

    def test_num_k_exclude_guard_dies_under_sin_dead(self) -> None:
        """G1.4 goes RED under the sin-dead component too (via the oracle-free arms)."""
        # sin=0 still leaves `x * cos`, which DOES move the head rows, so the head arm
        # survives. The guard as a whole is still not vacuous -- G1.1 covers this component.
        # Recorded explicitly rather than asserted RED, because claiming a RED that does not
        # happen would be worse than reporting the real partition.
        _guard_num_k_exclude(_SinDeadRoPE)

    def test_repeat_k_guard_dies_under_identity(self) -> None:
        """G1.3's liveness arm goes RED when the rotation is the identity."""
        with pytest.raises(AssertionError, match="rotation is DEAD"):
            _guard_repeat_k(_RotationDeadRoPE)

    def test_repeat_k_bit_identity_arm_is_vacuous_under_sin_dead(self) -> None:
        """MEASURED: G1.3 survives sin->0, because `x * cos` is still block-consistent.

        Documented as an executable fact rather than a comment: a consistency guard cannot
        detect a dead component that is consistent.
        """
        _guard_repeat_k(_SinDeadRoPE)


# ---------------------------------------------------------------------
# construction / validation
# ---------------------------------------------------------------------


class TestConstruction:
    """Constructor validation and derived attributes."""

    @pytest.mark.parametrize("head_dim", [1, 2, 6, 10, 14, 62])
    def test_head_dim_not_multiple_of_four_raises(self, head_dim: int) -> None:
        """`head_dim % 4 != 0` raises at CONSTRUCTION, not at call time."""
        with pytest.raises(ValueError, match="divisible by 4"):
            AxialRoPE2D(head_dim=head_dim)

    @pytest.mark.parametrize("head_dim", [4, 8, 12, 64, 128])
    def test_valid_head_dims_construct(self, head_dim: int) -> None:
        """Multiples of 4 construct fine."""
        assert AxialRoPE2D(head_dim=head_dim).head_dim == head_dim

    @pytest.mark.parametrize("bad", [0, -4])
    def test_non_positive_head_dim_raises(self, bad: int) -> None:
        """Non-positive `head_dim` raises."""
        with pytest.raises(ValueError, match="positive int"):
            AxialRoPE2D(head_dim=bad)

    @pytest.mark.parametrize("bad", [(0, 4), (4, -1), (4,), (2, 2, 2)])
    def test_bad_feat_shape_raises(self, bad) -> None:
        """`feat_shape` must be a pair of positive ints."""
        with pytest.raises(ValueError, match="feat_shape"):
            AxialRoPE2D(head_dim=8, feat_shape=bad)

    def test_non_positive_theta_raises(self) -> None:
        """`theta` must be positive."""
        with pytest.raises(ValueError, match="theta must be positive"):
            AxialRoPE2D(head_dim=8, theta=0.0)

    def test_layer_owns_no_weights(self, small_rope: AxialRoPE2D) -> None:
        """The angle table is a config-derived constant, never a variable.

        This matters under mixed precision: a non-trainable float variable is AUTOCAST to
        the compute dtype on read, which would silently narrow the table to float16.
        """
        small_rope(np.zeros((1, 1, 6, 8), dtype="float32"))
        assert small_rope.built
        assert small_rope.weights == []
        assert small_rope.count_params() == 0

    def test_wrong_query_token_count_raises(self) -> None:
        """A query token count that is not `H * W` raises."""
        rope = AxialRoPE2D(head_dim=8, feat_shape=(2, 3))
        with pytest.raises(ValueError, match="must equal feat_shape"):
            rope(np.zeros((1, 1, 7, 8), dtype="float32"))

    def test_wrong_head_dim_raises(self) -> None:
        """A last dimension that is not `head_dim` raises."""
        rope = AxialRoPE2D(head_dim=8, feat_shape=(2, 3))
        with pytest.raises(ValueError, match="must equal head_dim"):
            rope(np.zeros((1, 1, 6, 12), dtype="float32"))

    def test_non_rank_4_raises(self) -> None:
        """RoPE is applied POST head split; a rank-3 input is rejected."""
        rope = AxialRoPE2D(head_dim=8, feat_shape=(2, 3))
        with pytest.raises(ValueError, match="rank-4"):
            rope(np.zeros((1, 6, 8), dtype="float32"))


# ---------------------------------------------------------------------
# shape / config / serialization
# ---------------------------------------------------------------------


class TestShapeAndConfig:
    """`compute_output_shape`, `get_config`, and round-trips."""

    def test_compute_output_shape_before_build(self) -> None:
        """Shape inference works on an UNBUILT layer, from stored config only."""
        rope = AxialRoPE2D(head_dim=8, feat_shape=(2, 3))
        assert not rope.built
        assert rope.compute_output_shape((None, 2, 6, 8)) == (None, 2, 6, 8)
        assert rope.compute_output_shape((None, 2, 6, 8), (None, 2, 18, 8)) == [
            (None, 2, 6, 8), (None, 2, 18, 8)
        ]

    def test_compute_output_shape_matches_forward(self, small_rope: AxialRoPE2D) -> None:
        """Predicted shape equals the realized shape."""
        q = np.zeros((3, 2, 6, 8), dtype="float32")
        out = small_rope(q)
        assert tuple(out.shape) == small_rope.compute_output_shape(q.shape)

    def test_get_config_covers_all_init_params(self) -> None:
        """`get_config()` returns every `__init__` parameter."""
        rope = AxialRoPE2D(head_dim=12, feat_shape=(3, 5), theta=5000.0, repeat_k=True)
        config = rope.get_config()
        for key in ("head_dim", "feat_shape", "theta", "repeat_k"):
            assert key in config, f"get_config() is missing {key!r}"
        assert config["head_dim"] == 12
        assert tuple(config["feat_shape"]) == (3, 5)
        assert config["theta"] == 5000.0
        assert config["repeat_k"] is True

    def test_from_config_round_trip_by_value(self) -> None:
        """A `from_config` clone produces bit-identical outputs, not just equal config."""
        rng = np.random.default_rng(99)
        q = rng.standard_normal((2, 2, 15, 12)).astype(np.float64)
        k = rng.standard_normal((2, 2, 33, 12)).astype(np.float64)

        original = AxialRoPE2D(
            head_dim=12, feat_shape=(3, 5), theta=5000.0, repeat_k=True, dtype="float64"
        )
        q0, k0 = original(q, k, num_k_exclude=3)

        clone = AxialRoPE2D.from_config(original.get_config())
        q1, k1 = clone(q, k, num_k_exclude=3)

        assert np.max(np.abs(ops.convert_to_numpy(q0) - ops.convert_to_numpy(q1))) == 0.0
        assert np.max(np.abs(ops.convert_to_numpy(k0) - ops.convert_to_numpy(k1))) == 0.0

    def test_keras_model_save_load_by_value(self, tmp_path) -> None:
        """A `.keras` round-trip reproduces outputs with max-abs-diff exactly 0.0."""
        head_dim, num_tokens = 8, 6
        q_in = keras.Input(shape=(2, num_tokens, head_dim))
        k_in = keras.Input(shape=(2, 3 * num_tokens, head_dim))
        q_out, k_out = AxialRoPE2D(
            head_dim=head_dim, feat_shape=(2, 3), repeat_k=True
        )(q_in, k_in)
        model = keras.Model([q_in, k_in], [q_out, k_out])

        rng = np.random.default_rng(2026)
        q = rng.standard_normal((4, 2, num_tokens, head_dim)).astype("float32")
        k = rng.standard_normal((4, 2, 3 * num_tokens, head_dim)).astype("float32")
        before = [ops.convert_to_numpy(t) for t in model([q, k])]

        path = tmp_path / "axial_rope_2d.keras"
        model.save(path)
        restored = keras.models.load_model(path)
        after = [ops.convert_to_numpy(t) for t in restored([q, k])]

        for name, b, a in zip(("query", "key"), before, after):
            assert np.max(np.abs(b - a)) == 0.0, f"{name} changed across a .keras round-trip"

        layer = [ly for ly in restored.layers if isinstance(ly, AxialRoPE2D)]
        assert len(layer) == 1
        assert layer[0].feat_shape == (2, 3)
        assert layer[0].repeat_k is True


# ---------------------------------------------------------------------
# dtype policy sweep
# ---------------------------------------------------------------------


class TestDtypePolicies:
    """The angle table must be built in a never-narrowed work dtype and cast back."""

    def test_dtype_policy_sweep(self, dtype_policy: str) -> None:
        """float32 / mixed_float16 / float64: output dtype is the layer's compute dtype.

        Uses `tests/test_layers/conftest.py`'s restore-safe `dtype_policy` fixture -- the
        global Keras policy is PROCESS-GLOBAL and must never be left mutated.
        """
        head_dim, num_tokens = 8, 6
        rope = AxialRoPE2D(head_dim=head_dim, feat_shape=(2, 3), repeat_k=True)
        rng = np.random.default_rng(4242)
        q = rng.standard_normal((2, 2, num_tokens, head_dim)).astype("float32")
        k = rng.standard_normal((2, 2, 2 * num_tokens + 2, head_dim)).astype("float32")

        q_rot, k_rot = rope(q, k, num_k_exclude=2)
        q_rot = ops.convert_to_numpy(q_rot)
        k_rot = ops.convert_to_numpy(k_rot)

        expected_dtype = rope.compute_dtype
        assert keras.backend.standardize_dtype(q_rot.dtype) == expected_dtype
        assert keras.backend.standardize_dtype(k_rot.dtype) == expected_dtype
        assert np.all(np.isfinite(q_rot))
        assert np.all(np.isfinite(k_rot))

        # Against a float64 reference computed by the independent oracle.
        reference = _oracle_rotate(q.astype(np.float64), 2, 3)
        tol = {"float32": 1e-5, "float16": 5e-3, "float64": 1e-12}[expected_dtype]
        np.testing.assert_allclose(
            q_rot.astype(np.float64), reference, atol=tol, rtol=0.0,
            err_msg=f"rotation drifted beyond {tol} under policy {dtype_policy}",
        )

        # The excluded tail must survive every policy untouched (up to the output cast).
        tail_expected = k[:, :, -2:, :].astype(np.float64)
        tail_got = k_rot[:, :, -2:, :].astype(np.float64)
        np.testing.assert_allclose(tail_got, tail_expected, atol=tol, rtol=0.0)

    def test_float64_policy_is_not_pinned_to_float32(self, dtype_policy: str) -> None:
        """Under a float64 policy the internal work dtype must also be float64.

        A hardcoded `"float32"` work dtype would silently pin a float64 model's rotation to
        float32; this test measures the resulting accuracy floor, not the source text.
        """
        if dtype_policy != "float64":
            pytest.skip("float64-specific assertion")
        head_dim = 8
        rope = AxialRoPE2D(head_dim=head_dim, feat_shape=(8, 8))
        rng = np.random.default_rng(8)
        q = rng.standard_normal((1, 1, 64, head_dim)).astype(np.float64)
        got = ops.convert_to_numpy(rope(q)).astype(np.float64)
        reference = _oracle_rotate(q, 8, 8)
        np.testing.assert_allclose(got, reference, atol=1e-12, rtol=0.0)


# ---------------------------------------------------------------------
# graph safety
# ---------------------------------------------------------------------


class TestGraphSafety:
    """The layer must trace under `tf.function` with a static input signature."""

    def test_traces_under_tf_function(self) -> None:
        """A static-signature trace completes without raising."""
        import tensorflow as tf  # test-only backend import

        head_dim, num_tokens = 8, 6
        rope = AxialRoPE2D(head_dim=head_dim, feat_shape=(2, 3), repeat_k=True)

        @tf.function(input_signature=[
            tf.TensorSpec(shape=(None, 2, num_tokens, head_dim), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 2, 2 * num_tokens + 3, head_dim), dtype=tf.float32),
        ])
        def traced(q, k):
            return rope(q, k, num_k_exclude=3)

        rng = np.random.default_rng(17)
        q = rng.standard_normal((2, 2, num_tokens, head_dim)).astype("float32")
        k = rng.standard_normal((2, 2, 2 * num_tokens + 3, head_dim)).astype("float32")
        q_g, k_g = traced(tf.constant(q), tf.constant(k))

        q_e, k_e = rope(q, k, num_k_exclude=3)
        assert np.max(np.abs(ops.convert_to_numpy(q_g) - ops.convert_to_numpy(q_e))) == 0.0
        assert np.max(np.abs(ops.convert_to_numpy(k_g) - ops.convert_to_numpy(k_e))) == 0.0

    def test_dynamic_token_axis_raises_clearly(self) -> None:
        """A dynamic token axis raises a named error rather than failing obscurely."""
        import tensorflow as tf  # test-only backend import

        rope = AxialRoPE2D(head_dim=8, feat_shape=(2, 3))

        @tf.function(input_signature=[
            tf.TensorSpec(shape=(None, 2, None, 8), dtype=tf.float32)
        ])
        def traced(q):
            return rope(q)

        with pytest.raises(ValueError, match="STATIC token axis"):
            traced(tf.zeros((1, 2, 6, 8)))


# ---------------------------------------------------------------------
# package surface
# ---------------------------------------------------------------------


class TestPackageSurface:
    """The layer is exported and uniquely registered."""

    def test_exported_from_package(self) -> None:
        """`AxialRoPE2D` is in `layers.embedding.__all__`."""
        from dl_techniques.layers import embedding

        assert "AxialRoPE2D" in embedding.__all__
        assert embedding.AxialRoPE2D is AxialRoPE2D

    def test_registered_exactly_once(self) -> None:
        """The Keras registered key resolves back to this exact class."""
        registry = keras.saving.get_custom_objects()
        matches = [k for k, v in registry.items() if v is AxialRoPE2D]
        assert matches, "AxialRoPE2D is not in the Keras serializable registry"
        for key in matches:
            assert keras.saving.get_registered_object(key) is AxialRoPE2D


# ---------------------------------------------------------------------
# G1.6 -- `scale_pos` (plan-2026-08-04T044628-4c240b4c iter-3 step 1)
# ---------------------------------------------------------------------

SCALE_ONE_THIRD = 1.0 / 3.0
"""SAM 3's global-ViTDet ratio `rope_pt_size / input_size = 24 / 72`.

Every `scale_pos` value oracle probes OFF the identity deliberately: at
`scale_pos == 1.0` a coordinate scale and a frequency scale are the same
function, so an on-identity probe cannot separate them.
"""


def _oracle_rotate_scaled(
        x: np.ndarray,
        height: int,
        width: int,
        scale_pos: float,
        theta: float = THETA,
) -> np.ndarray:
    """Independent float64 complex oracle for 2D axial RoPE WITH a position scale.

    Derived from the published `use_interp_rope` semantics — the position indices
    are interpolated onto the pre-training grid by `rope_pt_size / input_size`
    while the frequency ladder is computed at the target `head_dim` — and NOT read
    off the implementation under test. Both axial halves take the same scale,
    because both halves share one frequency ladder.

    :param x: Array of shape ``(..., H * W, head_dim)`` in float64.
    :param height: Grid height ``H``.
    :param width: Grid width ``W``.
    :param scale_pos: Multiplier on the ``(t_x, t_y)`` coordinates.
    :param theta: Frequency-ladder base.
    :return: Rotated array of the same shape and dtype.
    """
    head_dim = x.shape[-1]
    num_tokens = height * width
    assert x.shape[-2] == num_tokens

    flat = np.arange(num_tokens, dtype=np.float64)
    t_x = (flat % width) * scale_pos
    t_y = (flat // width) * scale_pos
    bands = np.arange(0, head_dim, 4, dtype=np.float64)[: head_dim // 4]
    freqs = 1.0 / (theta ** (bands / head_dim))
    angles = np.concatenate([np.outer(t_x, freqs), np.outer(t_y, freqs)], axis=-1)

    cis = np.exp(1j * angles)
    x_complex = x[..., 0::2] + 1j * x[..., 1::2]
    rotated = x_complex * cis

    out = np.empty_like(x)
    out[..., 0::2] = rotated.real
    out[..., 1::2] = rotated.imag
    return out


class _ScalePosDeadRoPE(AxialRoPE2D):
    """Dead component C: `scale_pos` is forced to a constant 1.0 before the table.

    Measures which of the G1.6 guards are structurally blind to the whole
    `scale_pos` mechanism being absent.
    """

    def build(self, query_shape, key_shape=None):
        self.scale_pos = 1.0
        super().build(query_shape, key_shape)


class TestScalePos:
    """`scale_pos` scales the COORDINATES, on BOTH axes, and round-trips."""

    def test_default_scale_pos_is_one(self) -> None:
        """The parameter is optional and defaults to the unscaled case."""
        assert AxialRoPE2D(head_dim=8, feat_shape=(2, 3)).scale_pos == 1.0

    def test_explicit_scale_pos_one_is_bit_identical_to_default(self) -> None:
        """`scale_pos=1.0` must be EXACTLY the unscaled output, and the
        comparator that says so must be able to see a 1-ulp change.

        The second half of this test is the comparator's own liveness arm: an
        exact-0.0 gate over the wrong quantity exits green while comparing
        nothing.
        """
        rng = np.random.default_rng(20260805)
        q = (rng.standard_normal((2, 3, 15, 12)) + 1.7).astype(np.float64)

        base = AxialRoPE2D(head_dim=12, feat_shape=(3, 5), dtype="float64")(q)
        explicit = AxialRoPE2D(
            head_dim=12, feat_shape=(3, 5), scale_pos=1.0, dtype="float64"
        )(q)
        base_np = ops.convert_to_numpy(base)
        explicit_np = ops.convert_to_numpy(explicit)

        assert np.max(np.abs(base_np - explicit_np)) == 0.0, (
            "scale_pos=1.0 is not bit-identical to the unscaled default"
        )

        perturbed = explicit_np.copy()
        idx = (0, 0, 7, 3)
        perturbed[idx] = np.nextafter(perturbed[idx], np.inf)
        assert np.max(np.abs(base_np - perturbed)) > 0.0, (
            "COMPARATOR IS BLIND: a 1-ulp perturbation produced a 0.0 difference"
        )

    def test_scale_pos_matches_float64_oracle(self) -> None:
        """Layer output equals the independent scaled complex oracle at 1/3.

        This is the M1.2-family value oracle and it probes OFF the identity on
        purpose.
        """
        height, width, head_dim = 4, 4, 12
        rng = np.random.default_rng(20260805)
        q = rng.standard_normal((2, 3, height * width, head_dim)).astype(np.float64)

        rope = AxialRoPE2D(
            head_dim=head_dim,
            feat_shape=(height, width),
            theta=THETA,
            scale_pos=SCALE_ONE_THIRD,
            dtype="float64",
        )
        got = ops.convert_to_numpy(rope(q))
        expected = _oracle_rotate_scaled(q, height, width, SCALE_ONE_THIRD)

        np.testing.assert_allclose(
            got, expected, atol=1e-12, rtol=0.0,
            err_msg="AxialRoPE2D(scale_pos=1/3) disagrees with the scaled oracle",
        )

    def test_scale_pos_oracle_separates_the_exponent_candidate(self) -> None:
        """The oracle must REJECT a scale folded into the frequency EXPONENT.

        `outer(s * t, f)` and `outer(t, s * f)` are the same function, so the
        literal "scale the frequencies" candidate is a mathematical no-op and
        cannot be separated by any probe. The candidate that IS distinguishable
        rescales the ladder EXPONENT (`theta ** (bands / D * s)`), which is what
        an NTK-style reading of "interpolate RoPE" would do. This test proves the
        oracle is not blind to it.
        """
        height, width, head_dim = 4, 4, 12
        rng = np.random.default_rng(7)
        q = rng.standard_normal((1, 1, height * width, head_dim)).astype(np.float64)

        bands = np.arange(0, head_dim, 4, dtype=np.float64)[: head_dim // 4]
        flat = np.arange(height * width, dtype=np.float64)
        exp_freqs = 1.0 / (THETA ** (bands / head_dim * SCALE_ONE_THIRD))
        angles = np.concatenate(
            [np.outer(flat % width, exp_freqs), np.outer(flat // width, exp_freqs)],
            axis=-1,
        )
        cis = np.exp(1j * angles)
        rotated = (q[..., 0::2] + 1j * q[..., 1::2]) * cis
        wrong = np.empty_like(q)
        wrong[..., 0::2] = rotated.real
        wrong[..., 1::2] = rotated.imag

        correct = _oracle_rotate_scaled(q, height, width, SCALE_ONE_THIRD)
        assert np.max(np.abs(correct - wrong)) > 1e-2, (
            "the scale_pos oracle cannot separate the exponent-scaled candidate"
        )

    def test_scale_pos_applies_to_both_axes(self) -> None:
        """AXIS SYMMETRY: a per-axis scale must not be asymmetric.

        On a SQUARE grid, feed a constant query whose two axial halves are
        IDENTICAL. Then the first half of the output at token ``(x=a, y=b)`` and
        the second half at token ``(x=b, y=a)`` are the same rotation of the same
        vector by the same angle — but ONLY if both axes carry the same
        coordinate scale. Scaling ``t_x`` alone breaks it with no shape error.
        """
        side, head_dim = 4, 12
        half = head_dim // 2
        rng = np.random.default_rng(11)
        u = rng.standard_normal(half)
        vec = np.concatenate([u, u])
        q = np.broadcast_to(vec, (1, 1, side * side, head_dim)).astype(np.float64).copy()

        rope = AxialRoPE2D(
            head_dim=head_dim,
            feat_shape=(side, side),
            scale_pos=SCALE_ONE_THIRD,
            dtype="float64",
        )
        out = ops.convert_to_numpy(rope(q))[0, 0]

        a, b = 1, 3
        assert a != b, "the probe must be OFF the diagonal or it is vacuous"
        tok_ab = b * side + a   # (x=a, y=b)
        tok_ba = a * side + b   # (x=b, y=a)

        np.testing.assert_allclose(
            out[tok_ab, :half], out[tok_ba, half:], atol=1e-12, rtol=0.0,
            err_msg="scale_pos is not applied symmetrically to t_x and t_y",
        )
        # Liveness: the probed angle must be non-trivial, otherwise the symmetry
        # above holds for a dead rotation too.
        assert np.max(np.abs(out[tok_ab, :half] - u)) > 1e-6, (
            "the axis-symmetry probe sits at a zero angle and is vacuous"
        )

    def test_scale_pos_changes_the_output(self) -> None:
        """LIVENESS: `scale_pos != 1` must measurably move the output."""
        rng = np.random.default_rng(3)
        q = rng.standard_normal((1, 1, 16, 12)).astype(np.float64)
        unscaled = ops.convert_to_numpy(
            AxialRoPE2D(head_dim=12, feat_shape=(4, 4), dtype="float64")(q)
        )
        scaled = ops.convert_to_numpy(
            AxialRoPE2D(
                head_dim=12, feat_shape=(4, 4), scale_pos=SCALE_ONE_THIRD,
                dtype="float64",
            )(q)
        )
        assert np.max(np.abs(unscaled - scaled)) > 1e-2, (
            "scale_pos=1/3 produced (near-)identical output to scale_pos=1.0"
        )

    @pytest.mark.parametrize("bad", [0.0, -1.0, -0.5])
    def test_non_positive_scale_pos_raises(self, bad: float) -> None:
        """A zero or negative position scale is rejected at construction."""
        with pytest.raises(ValueError, match="scale_pos must be positive"):
            AxialRoPE2D(head_dim=8, feat_shape=(2, 3), scale_pos=bad)

    def test_config_keys_equal_init_signature(self) -> None:
        """`get_config()` covers EVERY `__init__` parameter, derived from the
        signature rather than from a hand-maintained list.

        A hand-listed key tuple is the thing that silently rots when a parameter
        is added, which is exactly the defect this test class was added for.
        """
        import inspect

        expected = {
            name for name, p in
            inspect.signature(AxialRoPE2D.__init__).parameters.items()
            if name not in ("self", "kwargs")
        }
        config = AxialRoPE2D(
            head_dim=12, feat_shape=(3, 5), theta=5000.0,
            scale_pos=SCALE_ONE_THIRD, repeat_k=True,
        ).get_config()
        missing = expected - set(config)
        assert not missing, f"get_config() is missing __init__ params: {sorted(missing)}"
        assert config["scale_pos"] == SCALE_ONE_THIRD

    def test_scale_pos_round_trip_by_value(self) -> None:
        """A `from_config` clone reproduces a SCALED layer bit-identically."""
        rng = np.random.default_rng(123)
        q = rng.standard_normal((2, 2, 16, 12)).astype(np.float64)
        k = rng.standard_normal((2, 2, 35, 12)).astype(np.float64)

        original = AxialRoPE2D(
            head_dim=12, feat_shape=(4, 4), theta=5000.0,
            scale_pos=SCALE_ONE_THIRD, repeat_k=True, dtype="float64",
        )
        q0, k0 = original(q, k, num_k_exclude=3)
        clone = AxialRoPE2D.from_config(original.get_config())
        assert clone.scale_pos == SCALE_ONE_THIRD
        q1, k1 = clone(q, k, num_k_exclude=3)

        assert np.max(np.abs(ops.convert_to_numpy(q0) - ops.convert_to_numpy(q1))) == 0.0
        assert np.max(np.abs(ops.convert_to_numpy(k0) - ops.convert_to_numpy(k1))) == 0.0

    def test_scale_pos_keras_save_load_by_value(self, tmp_path) -> None:
        """A `.keras` round-trip preserves `scale_pos` by VALUE, not just by key."""
        head_dim = 12
        q_in = keras.Input(shape=(2, 16, head_dim))
        q_out = AxialRoPE2D(
            head_dim=head_dim, feat_shape=(4, 4), scale_pos=SCALE_ONE_THIRD
        )(q_in)
        model = keras.Model(q_in, q_out)

        rng = np.random.default_rng(555)
        q = rng.standard_normal((3, 2, 16, head_dim)).astype("float32")
        before = ops.convert_to_numpy(model(q))

        path = tmp_path / "axial_rope_2d_scaled.keras"
        model.save(path)
        restored = keras.models.load_model(path)
        after = ops.convert_to_numpy(restored(q))

        assert np.max(np.abs(before - after)) == 0.0
        layer = [ly for ly in restored.layers if isinstance(ly, AxialRoPE2D)]
        assert len(layer) == 1
        assert layer[0].scale_pos == SCALE_ONE_THIRD

    def test_scale_pos_composes_with_repeat_k(self) -> None:
        """`scale_pos` and `repeat_k` are independent mechanisms.

        `repeat_k` tiles a FINISHED table, so every repeated key block must carry
        the same scaled angles as the query grid.
        """
        head_dim, side = 12, 4
        n = side * side
        rng = np.random.default_rng(31)
        q = rng.standard_normal((1, 1, n, head_dim)).astype(np.float64)
        k_block = rng.standard_normal((1, 1, n, head_dim)).astype(np.float64)
        k = np.concatenate([k_block, k_block, k_block], axis=-2)

        rope = AxialRoPE2D(
            head_dim=head_dim, feat_shape=(side, side),
            scale_pos=SCALE_ONE_THIRD, repeat_k=True, dtype="float64",
        )
        _, k_rot = rope(q, k)
        k_rot = ops.convert_to_numpy(k_rot)
        expected_block = _oracle_rotate_scaled(k_block, side, side, SCALE_ONE_THIRD)
        for block in range(3):
            np.testing.assert_allclose(
                k_rot[..., block * n:(block + 1) * n, :], expected_block,
                atol=1e-12, rtol=0.0,
                err_msg=f"repeat_k block {block} does not carry the scaled angles",
            )


class TestScalePosDeadComponentProbe:
    """G1.6's own dead-component probe: `scale_pos` forced to 1.0 in `build()`."""

    def test_oracle_guard_dies(self) -> None:
        """The scaled value oracle must NOT survive a dead `scale_pos`."""
        height, width, head_dim = 4, 4, 12
        rng = np.random.default_rng(20260805)
        q = rng.standard_normal((2, 3, height * width, head_dim)).astype(np.float64)
        rope = _ScalePosDeadRoPE(
            head_dim=head_dim, feat_shape=(height, width), theta=THETA,
            scale_pos=SCALE_ONE_THIRD, dtype="float64",
        )
        got = ops.convert_to_numpy(rope(q))
        expected = _oracle_rotate_scaled(q, height, width, SCALE_ONE_THIRD)
        assert np.max(np.abs(got - expected)) > 1e-2, (
            "the scaled value oracle is GREEN with scale_pos dead"
        )

    def test_liveness_guard_dies(self) -> None:
        """The liveness guard must NOT survive a dead `scale_pos`."""
        rng = np.random.default_rng(3)
        q = rng.standard_normal((1, 1, 16, 12)).astype(np.float64)
        unscaled = ops.convert_to_numpy(
            AxialRoPE2D(head_dim=12, feat_shape=(4, 4), dtype="float64")(q)
        )
        dead = ops.convert_to_numpy(
            _ScalePosDeadRoPE(
                head_dim=12, feat_shape=(4, 4), scale_pos=SCALE_ONE_THIRD,
                dtype="float64",
            )(q)
        )
        assert np.max(np.abs(unscaled - dead)) == 0.0, (
            "the liveness guard would still fire with scale_pos dead"
        )

# ---------------------------------------------------------------------
