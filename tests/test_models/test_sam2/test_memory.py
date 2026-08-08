"""
Guards for SAM 2's memory-attention stack (plan step 2, guards G2.1-G2.5).

The defect class this file exists for is *silence*: every one of the four
``pos_enc_at_*`` booleans, the ``relu``-vs-``gelu`` choice, and the
``num_k_exclude`` threading can be ported wrong and still produce a model that
runs, trains, and returns plausible numbers with no shape error anywhere. So
every guard here is a BEHAVIOURAL discriminator on values, never a restatement
of the config that produced them.

Guard map:

    G2.1  ``TestPosEncBooleansAreEachLive``  -- 16-way distinctness
    G2.2  ``TestActivationIsRelu``           -- FFN hidden exact-zero count
    G2.3  ``TestNumKExcludeThreading``       -- object-pointer tail unrotated
    G2.4  ``TestGraphTrace``                 -- ``tf.function`` traceability
    G2.5  ``TestDeadComponentPartition``     -- the MEASURED RED partition
"""

import itertools
from typing import Any, Dict, List, Optional, Tuple

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.models.SAM.SAM2.memory_attention import (
    SAM2MemoryAttention,
    SAM2MemoryAttentionLayer,
)

from ..test_sam.dead_component_oracle import zeroed_variables

# ---------------------------------------------------------------------
# Test geometry.
#
# Small but STRUCTURALLY faithful to the shipped stack: a square query grid, a
# key sequence that is an exact integer multiple of it (so `repeat_k` is
# exercised), a non-empty object-pointer tail (so `num_k_exclude` is exercised),
# a memory width that DIFFERS from d_model (so the asymmetric `kv_in_dim`
# projection is exercised), and a head width divisible by 4 (RoPE's constraint).
# ---------------------------------------------------------------------

GRID: Tuple[int, int] = (4, 4)
NUM_QUERY_TOKENS = GRID[0] * GRID[1]
D_MODEL = 32
KV_IN_DIM = 8
DIM_FEEDFORWARD = 64
NUM_LAYERS = 2
NUM_MEMORY_FRAMES = 3
NUM_OBJ_PTR_TOKENS = 5
NUM_MEMORY_TOKENS = NUM_MEMORY_FRAMES * NUM_QUERY_TOKENS + NUM_OBJ_PTR_TOKENS
BATCH = 2

# The shipped SAM 2.1 four-tuple, in the order used throughout this module.
SHIPPED_BOOLEANS: Tuple[bool, bool, bool, bool] = (
    True,   # pos_enc_at_input
    False,  # pos_enc_at_attn
    False,  # pos_enc_at_cross_attn_queries
    True,   # pos_enc_at_cross_attn_keys
)
BOOLEAN_NAMES: Tuple[str, ...] = (
    "pos_enc_at_input",
    "pos_enc_at_attn",
    "pos_enc_at_cross_attn_queries",
    "pos_enc_at_cross_attn_keys",
)

SEED = 1234


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _inputs() -> Dict[str, np.ndarray]:
    """Build the seeded, NON-ZERO input bundle shared by every guard.

    The positional tensors are deliberately non-zero and non-constant: an
    all-zero positional encoding would make several of the sixteen boolean
    combinations genuinely identical, and the 16-way guard would then be
    measuring the fixture rather than the layer.

    :return: Mapping of ``features``/``memory``/``features_pos``/``memory_pos``.
    :rtype: Dict[str, np.ndarray]
    """
    rng = np.random.RandomState(SEED)
    return {
        "features": rng.randn(BATCH, NUM_QUERY_TOKENS, D_MODEL).astype("float32"),
        "memory": rng.randn(BATCH, NUM_MEMORY_TOKENS, KV_IN_DIM).astype("float32"),
        "features_pos": rng.randn(BATCH, NUM_QUERY_TOKENS, D_MODEL).astype("float32"),
        "memory_pos": rng.randn(BATCH, NUM_MEMORY_TOKENS, KV_IN_DIM).astype("float32"),
    }


def _build_stack(
        pos_enc_at_input: bool = SHIPPED_BOOLEANS[0],
        pos_enc_at_attn: bool = SHIPPED_BOOLEANS[1],
        pos_enc_at_cross_attn_queries: bool = SHIPPED_BOOLEANS[2],
        pos_enc_at_cross_attn_keys: bool = SHIPPED_BOOLEANS[3],
        activation: str = "relu",
        num_layers: int = NUM_LAYERS,
) -> SAM2MemoryAttention:
    """Construct and build a stack under a fixed seed.

    Re-seeding immediately before construction is what makes two stacks that
    differ only in a boolean weight-identical, which is the precondition for the
    16-way distinctness guard to be about the booleans and nothing else.

    :param pos_enc_at_input: Stack-level positional injection.
    :type pos_enc_at_input: bool
    :param pos_enc_at_attn: Per-block self-attention injection.
    :type pos_enc_at_attn: bool
    :param pos_enc_at_cross_attn_queries: Per-block cross-attn query injection.
    :type pos_enc_at_cross_attn_queries: bool
    :param pos_enc_at_cross_attn_keys: Per-block cross-attn key injection.
    :type pos_enc_at_cross_attn_keys: bool
    :param activation: FFN hidden activation.
    :type activation: str
    :param num_layers: Number of stacked blocks.
    :type num_layers: int
    :return: A built stack.
    :rtype: SAM2MemoryAttention
    """
    keras.utils.set_random_seed(SEED)
    stack = SAM2MemoryAttention(
        d_model=D_MODEL,
        num_layers=num_layers,
        pos_enc_at_input=pos_enc_at_input,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=0.1,
        activation=activation,
        pos_enc_at_attn=pos_enc_at_attn,
        pos_enc_at_cross_attn_queries=pos_enc_at_cross_attn_queries,
        pos_enc_at_cross_attn_keys=pos_enc_at_cross_attn_keys,
        num_heads=1,
        downsample_rate=1,
        feat_sizes=GRID,
        kv_in_dim=KV_IN_DIM,
    )
    stack.build(
        (None, NUM_QUERY_TOKENS, D_MODEL),
        (None, NUM_MEMORY_TOKENS, KV_IN_DIM),
    )
    return stack


def _forward(stack: SAM2MemoryAttention, data: Dict[str, np.ndarray]) -> np.ndarray:
    """Run a deterministic (``training=False``) forward pass.

    :param stack: A built stack.
    :type stack: SAM2MemoryAttention
    :param data: The input bundle from :func:`_inputs`.
    :type data: Dict[str, np.ndarray]
    :return: The stack output as NumPy.
    :rtype: np.ndarray
    """
    out = stack(
        data["features"],
        data["memory"],
        features_pos=data["features_pos"],
        memory_pos=data["memory_pos"],
        num_obj_ptr_tokens=NUM_OBJ_PTR_TOKENS,
        training=False,
    )
    return np.asarray(ops.convert_to_numpy(out))


class _CallSpy:
    """Record every ``(args, kwargs, result)`` of a bound layer ``call``.

    Used instead of asserting on a config field: what a guard must observe is
    the tensor that actually reached the sub-layer, not the number the config
    says should have reached it.
    """

    def __init__(self, owner: Any, method_name: str = "call") -> None:
        """Install the spy.

        :param owner: The object whose method is wrapped.
        :type owner: Any
        :param method_name: Name of the bound method to wrap.
        :type method_name: str
        """
        self.owner = owner
        self.method_name = method_name
        self.original = getattr(owner, method_name)
        self.calls: List[Tuple[Tuple[Any, ...], Dict[str, Any], Any]] = []

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            result = self.original(*args, **kwargs)
            self.calls.append((args, kwargs, result))
            return result

        setattr(owner, method_name, _wrapped)

    def restore(self) -> None:
        """Remove the spy, restoring the original bound method."""
        setattr(self.owner, self.method_name, self.original)


def _max_abs_diff(a: Any, b: Any) -> float:
    """Return the maximum absolute elementwise difference of two tensors.

    :param a: First tensor or array.
    :type a: Any
    :param b: Second tensor or array.
    :type b: Any
    :return: ``max(|a - b|)``.
    :rtype: float
    """
    a = np.asarray(ops.convert_to_numpy(a), dtype=np.float64)
    b = np.asarray(ops.convert_to_numpy(b), dtype=np.float64)
    return float(np.max(np.abs(a - b)))


# ---------------------------------------------------------------------
# construction / configuration
# ---------------------------------------------------------------------


class TestShippedConfiguration:
    """The settled SAM 2.1 numbers are the DEFAULTS, not merely reachable."""

    def test_stack_defaults_match_shipped_yaml(self) -> None:
        stack = SAM2MemoryAttention()
        assert stack.d_model == 256
        assert stack.num_layers == 4
        assert stack.pos_enc_at_input is True

    def test_layer_defaults_match_shipped_yaml(self) -> None:
        layer = SAM2MemoryAttentionLayer()
        assert layer.d_model == 256
        assert layer.dim_feedforward == 2048
        assert layer.dropout == pytest.approx(0.1)
        # relu, NOT the transformer-default gelu.
        assert layer.activation == "relu"
        assert layer.pos_enc_at_attn is False
        assert layer.pos_enc_at_cross_attn_queries is False
        assert layer.pos_enc_at_cross_attn_keys is True
        assert layer.kv_in_dim == 64
        assert layer.feat_sizes == (64, 64)
        assert layer.rope_theta == pytest.approx(10000.0)
        assert layer.num_heads == 1
        assert layer.downsample_rate == 1

    def test_cross_attention_repeats_the_spatial_table_self_attention_does_not(
            self) -> None:
        layer = SAM2MemoryAttentionLayer()
        assert layer.cross_attn_image.repeat_k is True
        assert layer.self_attn.repeat_k is False

    def test_projection_widths_are_asymmetric(self) -> None:
        """q projects from d_model; k/v project from the narrower kv_in_dim."""
        layer = SAM2MemoryAttentionLayer(
            d_model=D_MODEL, dim_feedforward=DIM_FEEDFORWARD,
            feat_sizes=GRID, kv_in_dim=KV_IN_DIM,
        )
        layer.build(
            (None, NUM_QUERY_TOKENS, D_MODEL),
            (None, NUM_MEMORY_TOKENS, KV_IN_DIM),
        )
        cross = layer.cross_attn_image
        assert tuple(cross.q_proj.kernel.shape) == (D_MODEL, D_MODEL)
        assert tuple(cross.k_proj.kernel.shape) == (KV_IN_DIM, D_MODEL)
        assert tuple(cross.v_proj.kernel.shape) == (KV_IN_DIM, D_MODEL)
        assert tuple(cross.out_proj.kernel.shape) == (D_MODEL, D_MODEL)
        # Self-attention consumes the full-width frame features on all three.
        assert tuple(layer.self_attn.k_proj.kernel.shape) == (D_MODEL, D_MODEL)

    def test_stack_holds_num_layers_blocks(self) -> None:
        stack = _build_stack(num_layers=3)
        assert len(stack.layers) == 3
        assert all(isinstance(b, SAM2MemoryAttentionLayer) for b in stack.layers)


class TestConfigAndShapes:
    """``get_config`` completeness, config round-trip, and output shapes."""

    def test_layer_config_contains_every_init_parameter(self) -> None:
        layer = SAM2MemoryAttentionLayer()
        config = layer.get_config()
        for key in (
                "d_model", "dim_feedforward", "dropout", "activation",
                "pos_enc_at_attn", "pos_enc_at_cross_attn_queries",
                "pos_enc_at_cross_attn_keys", "num_heads", "downsample_rate",
                "rope_theta", "feat_sizes", "kv_in_dim", "layer_norm_epsilon",
        ):
            assert key in config, f"get_config() is missing '{key}'"

    def test_stack_config_contains_every_init_parameter(self) -> None:
        stack = SAM2MemoryAttention()
        config = stack.get_config()
        for key in (
                "d_model", "num_layers", "pos_enc_at_input", "dim_feedforward",
                "dropout", "activation", "pos_enc_at_attn",
                "pos_enc_at_cross_attn_queries", "pos_enc_at_cross_attn_keys",
                "num_heads", "downsample_rate", "rope_theta", "feat_sizes",
                "kv_in_dim", "layer_norm_epsilon",
        ):
            assert key in config, f"get_config() is missing '{key}'"

    def test_config_round_trip_is_value_exact(self) -> None:
        """A rebuilt stack given the same weights reproduces the output exactly."""
        data = _inputs()
        stack = _build_stack()
        expected = _forward(stack, data)

        rebuilt = SAM2MemoryAttention.from_config(stack.get_config())
        rebuilt.build(
            (None, NUM_QUERY_TOKENS, D_MODEL),
            (None, NUM_MEMORY_TOKENS, KV_IN_DIM),
        )
        rebuilt.set_weights(stack.get_weights())
        assert _max_abs_diff(expected, _forward(rebuilt, data)) == 0.0

    def test_compute_output_shape_matches_the_forward_pass(self) -> None:
        stack = _build_stack()
        declared = stack.compute_output_shape(
            (BATCH, NUM_QUERY_TOKENS, D_MODEL),
            (BATCH, NUM_MEMORY_TOKENS, KV_IN_DIM),
        )
        assert declared == (BATCH, NUM_QUERY_TOKENS, D_MODEL)
        assert _forward(stack, _inputs()).shape == declared

    def test_layer_compute_output_shape(self) -> None:
        layer = SAM2MemoryAttentionLayer(
            d_model=D_MODEL, dim_feedforward=DIM_FEEDFORWARD,
            feat_sizes=GRID, kv_in_dim=KV_IN_DIM,
        )
        assert layer.compute_output_shape(
            (BATCH, NUM_QUERY_TOKENS, D_MODEL)
        ) == (BATCH, NUM_QUERY_TOKENS, D_MODEL)


class TestConstructionErrors:
    """Invalid geometry raises at construction, not at the first call."""

    def test_head_width_not_divisible_by_four_raises(self) -> None:
        with pytest.raises(ValueError, match="divisible by 4"):
            SAM2MemoryAttentionLayer(d_model=8, num_heads=4)

    def test_non_positive_num_layers_raises(self) -> None:
        with pytest.raises(ValueError, match="num_layers must be positive"):
            SAM2MemoryAttention(num_layers=0)

    def test_non_positive_dim_feedforward_raises(self) -> None:
        with pytest.raises(ValueError, match="dim_feedforward must be positive"):
            SAM2MemoryAttentionLayer(dim_feedforward=0)

    def test_num_heads_not_dividing_internal_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="divisible by num_heads"):
            SAM2MemoryAttentionLayer(d_model=32, num_heads=5)

    def test_memory_width_mismatch_raises(self) -> None:
        layer = SAM2MemoryAttentionLayer(
            d_model=D_MODEL, dim_feedforward=DIM_FEEDFORWARD,
            feat_sizes=GRID, kv_in_dim=KV_IN_DIM,
        )
        with pytest.raises(ValueError, match="must equal kv_in_dim"):
            layer.build(
                (None, NUM_QUERY_TOKENS, D_MODEL),
                (None, NUM_MEMORY_TOKENS, KV_IN_DIM + 1),
            )

    def test_key_length_not_a_multiple_of_the_grid_raises(self) -> None:
        """`repeat_k` requires an exact multiple after the excluded tail."""
        stack = _build_stack()
        data = _inputs()
        with pytest.raises(ValueError, match="exact multiple"):
            stack(
                data["features"],
                data["memory"],
                features_pos=data["features_pos"],
                memory_pos=data["memory_pos"],
                # One token short of the object-pointer tail: the remaining key
                # length is no longer r * H * W.
                num_obj_ptr_tokens=NUM_OBJ_PTR_TOKENS - 1,
                training=False,
            )


# ---------------------------------------------------------------------
# G2.1 -- the four positional-encoding booleans
# ---------------------------------------------------------------------


class TestPosEncBooleansAreEachLive:
    """G2.1: all 2^4 boolean settings produce PAIRWISE DISTINCT outputs.

    A dead boolean collapses the sixteen outputs into eight equal pairs -- the
    pairs that differ only in the dead flag -- with no shape error and no
    exception. The message names the collapsed pair so the failure identifies
    WHICH boolean died, not merely that one did.
    """

    @staticmethod
    def _all_outputs() -> Dict[Tuple[bool, ...], np.ndarray]:
        data = _inputs()
        outputs: Dict[Tuple[bool, ...], np.ndarray] = {}
        for combo in itertools.product([False, True], repeat=4):
            stack = _build_stack(*combo)
            outputs[combo] = _forward(stack, data)
        return outputs

    def test_four_pos_enc_booleans_are_each_live(self) -> None:
        outputs = self._all_outputs()
        assert len(outputs) == 16

        collapsed: List[str] = []
        for left, right in itertools.combinations(sorted(outputs), 2):
            diff = _max_abs_diff(outputs[left], outputs[right])
            if diff <= 1e-5:
                differing = [
                    BOOLEAN_NAMES[i] for i in range(4) if left[i] != right[i]
                ]
                collapsed.append(
                    f"{left} vs {right} (differ only in {differing}): "
                    f"max-abs-diff={diff:.3e}"
                )
        assert not collapsed, (
            "positional-encoding boolean combinations collapsed -- at least one "
            "boolean is DEAD:\n  " + "\n  ".join(collapsed)
        )

    def test_shipped_four_tuple_is_among_the_sixteen(self) -> None:
        outputs = self._all_outputs()
        assert SHIPPED_BOOLEANS in outputs
        assert np.all(np.isfinite(outputs[SHIPPED_BOOLEANS]))

    def test_fixture_positional_encodings_are_non_zero(self) -> None:
        """The 16-way guard is only about the layer if the fixture is non-zero.

        An all-zero positional tensor makes several combinations identical BY
        CONSTRUCTION, so the distinctness guard above would then be measuring
        the fixture. This asserts the precondition explicitly.
        """
        data = _inputs()
        assert float(np.abs(data["features_pos"]).min()) > 0.0
        assert float(np.abs(data["memory_pos"]).min()) > 0.0


class TestPosEncInjectionSites:
    """Each boolean moves the output on its OWN injection path.

    Complements the 16-way guard: distinctness proves the booleans are not
    interchangeable, these prove each one reaches the site its name claims.
    """

    @pytest.mark.parametrize("index,name", list(enumerate(BOOLEAN_NAMES)))
    def test_flipping_one_boolean_moves_the_output(
            self, index: int, name: str) -> None:
        data = _inputs()
        base = list(SHIPPED_BOOLEANS)
        flipped = list(SHIPPED_BOOLEANS)
        flipped[index] = not flipped[index]

        out_base = _forward(_build_stack(*base), data)
        out_flip = _forward(_build_stack(*flipped), data)
        diff = _max_abs_diff(out_base, out_flip)
        assert diff > 1e-5, (
            f"flipping {name} changed nothing (max-abs-diff={diff:.3e}); "
            f"that boolean is not read on any forward path"
        )

    def test_query_pos_of_none_disables_every_query_injection(self) -> None:
        """With no positional tensor, the query-side booleans cannot matter."""
        data = _inputs()
        kwargs = dict(
            memory_pos=data["memory_pos"],
            num_obj_ptr_tokens=NUM_OBJ_PTR_TOKENS,
            training=False,
        )
        a = _build_stack(True, True, True, True)(
            data["features"], data["memory"], features_pos=None, **kwargs)
        b = _build_stack(False, False, False, True)(
            data["features"], data["memory"], features_pos=None, **kwargs)
        assert _max_abs_diff(a, b) == 0.0


# ---------------------------------------------------------------------
# G2.2 -- the FFN activation is relu, proven by value
# ---------------------------------------------------------------------


def _capture_ffn_hidden(
        stack: SAM2MemoryAttention,
        data: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Capture the FFN pre-activation and the tensor the FFN actually consumes.

    The hidden tensor is read as the INPUT to ``linear2`` rather than by
    re-applying ``layer.activation_fn`` in the test: re-applying the layer's own
    activation would go green even if ``call()`` used a hardcoded different one.

    :param stack: A built stack.
    :type stack: SAM2MemoryAttention
    :param data: The input bundle.
    :type data: Dict[str, np.ndarray]
    :return: ``(pre_activation, ffn_hidden)`` for the first block.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    block = stack.layers[0]
    pre_spy = _CallSpy(block.linear1)
    hidden_spy = _CallSpy(block.linear2)
    try:
        _forward(stack, data)
    finally:
        pre_spy.restore()
        hidden_spy.restore()

    pre = np.asarray(ops.convert_to_numpy(pre_spy.calls[0][2]))
    hidden = np.asarray(ops.convert_to_numpy(hidden_spy.calls[0][0][0]))
    return pre, hidden


class TestActivationIsRelu:
    """G2.2: the FFN hidden activation is relu, asserted on VALUES."""

    def test_ffn_hidden_zeros_exactly_where_pre_activations_are_negative(
            self) -> None:
        pre, hidden = _capture_ffn_hidden(_build_stack(), _inputs())

        negatives = int(np.sum(pre < 0.0))
        assert negatives > 0, (
            "no negative pre-activations in the fixture -- a relu probe on an "
            "all-positive tensor passes for gelu, silu and identity too"
        )
        exact_zeros = int(np.sum(hidden == 0.0))
        assert exact_zeros == negatives, (
            f"FFN hidden has {exact_zeros} exact zeros but there are "
            f"{negatives} negative pre-activations; relu maps each negative to "
            f"EXACTLY 0.0 -- gelu and silu map them to small non-zero values"
        )
        assert np.allclose(hidden, np.maximum(pre, 0.0), atol=0.0, rtol=0.0)

    def test_gelu_build_differs_from_relu_build(self) -> None:
        data = _inputs()
        relu_out = _forward(_build_stack(activation="relu"), data)
        gelu_out = _forward(_build_stack(activation="gelu"), data)
        diff = _max_abs_diff(relu_out, gelu_out)
        assert diff > 1e-4, (
            f"a gelu build is indistinguishable from a relu build "
            f"(max-abs-diff={diff:.3e}); the activation config is not read"
        )

    def test_gelu_hidden_has_no_exact_zeros(self) -> None:
        """The control for the assertion above: gelu genuinely differs here."""
        pre, hidden = _capture_ffn_hidden(
            _build_stack(activation="gelu"), _inputs())
        assert int(np.sum(pre < 0.0)) > 0
        assert int(np.sum(hidden == 0.0)) == 0


# ---------------------------------------------------------------------
# G2.3 -- num_k_exclude threading
# ---------------------------------------------------------------------


def _capture_rope_calls(
        stack: SAM2MemoryAttention,
        data: Dict[str, np.ndarray],
        num_obj_ptr_tokens: int = NUM_OBJ_PTR_TOKENS,
) -> List[Tuple[Any, Dict[str, Any], Any]]:
    """Capture every cross-attention RoPE invocation in the stack.

    :param stack: A built stack.
    :type stack: SAM2MemoryAttention
    :param data: The input bundle.
    :type data: Dict[str, np.ndarray]
    :param num_obj_ptr_tokens: Value passed to the stack.
    :type num_obj_ptr_tokens: int
    :return: One ``(args, kwargs, result)`` triple per block, in block order.
    :rtype: List[Tuple[Any, Dict[str, Any], Any]]
    """
    spies = [_CallSpy(block.cross_attn_image.rope) for block in stack.layers]
    try:
        stack(
            data["features"],
            data["memory"],
            features_pos=data["features_pos"],
            memory_pos=data["memory_pos"],
            num_obj_ptr_tokens=num_obj_ptr_tokens,
            training=False,
        )
    finally:
        for spy in spies:
            spy.restore()
    return [spy.calls[0] for spy in spies]


class TestNumKExcludeThreading:
    """G2.3: the object-pointer tail reaches attention UNROTATED.

    Object pointers sit at the tail of the memory sequence and carry no spatial
    position. Rotating them would attach a position they do not have -- and
    would not change a single tensor shape.
    """

    def test_every_block_receives_the_object_pointer_count(self) -> None:
        calls = _capture_rope_calls(_build_stack(), _inputs())
        assert len(calls) == NUM_LAYERS
        for index, (_, kwargs, _result) in enumerate(calls):
            assert kwargs.get("num_k_exclude") == NUM_OBJ_PTR_TOKENS, (
                f"block {index} received num_k_exclude="
                f"{kwargs.get('num_k_exclude')!r}, expected "
                f"{NUM_OBJ_PTR_TOKENS}; the object-pointer count is not threaded"
            )

    def test_tail_key_rows_are_bit_identical_to_the_k_projection(self) -> None:
        args, _kwargs, result = _capture_rope_calls(_build_stack(), _inputs())[0]
        key_in = np.asarray(ops.convert_to_numpy(args[1]))
        key_out = np.asarray(ops.convert_to_numpy(result[1]))

        tail_in = key_in[:, :, -NUM_OBJ_PTR_TOKENS:, :]
        tail_out = key_out[:, :, -NUM_OBJ_PTR_TOKENS:, :]
        assert _max_abs_diff(tail_in, tail_out) == 0.0, (
            "the object-pointer tail of the key sequence was ROTATED; those "
            "tokens have no spatial position"
        )

    def test_spatial_key_rows_are_rotated(self) -> None:
        """The other half: a one-sided tail-identity assertion is vacuous.

        Returning k untouched would satisfy the tail assertion above perfectly.
        """
        args, _kwargs, result = _capture_rope_calls(_build_stack(), _inputs())[0]
        key_in = np.asarray(ops.convert_to_numpy(args[1]))
        key_out = np.asarray(ops.convert_to_numpy(result[1]))

        head_in = key_in[:, :, :-NUM_OBJ_PTR_TOKENS, :]
        head_out = key_out[:, :, :-NUM_OBJ_PTR_TOKENS, :]
        diff = _max_abs_diff(head_in, head_out)
        assert diff > 1e-4, (
            f"the spatial key rows were NOT rotated (max-abs-diff={diff:.3e}); "
            f"RoPE is a no-op on this path"
        )

    def test_excluding_the_tail_changes_the_output(self) -> None:
        """The threading is observable end-to-end, not only at the spy."""
        data = _inputs()
        stack = _build_stack()
        with_exclusion = _forward(stack, data)
        without = np.asarray(ops.convert_to_numpy(stack(
            data["features"],
            data["memory"],
            features_pos=data["features_pos"],
            memory_pos=data["memory_pos"],
            # Not a valid geometry for repeat_k unless the tail happens to be a
            # multiple of the grid; use a full extra frame's worth so it is.
            num_obj_ptr_tokens=NUM_OBJ_PTR_TOKENS + NUM_QUERY_TOKENS,
            training=False,
        )))
        assert _max_abs_diff(with_exclusion, without) > 1e-5

    def test_tail_unrotated_when_dropping_the_exclusion_would_still_fit(
            self) -> None:
        """The DISCRIMINATING value proof for the threading.

        With the shipped fixture the object-pointer tail is 5 tokens, so setting
        ``num_k_exclude=0`` leaves a key length that is not a multiple of the
        query grid and ``repeat_k`` raises -- a real guard, but a blunt one: it
        detects the mutation through a SHAPE constraint, not through the
        rotation. Here the tail is exactly one grid's worth, so
        ``num_k_exclude=0`` is geometrically valid and the only thing that can
        keep the tail unrotated is the exclusion actually being applied.
        """
        tail = NUM_QUERY_TOKENS
        num_memory = NUM_MEMORY_FRAMES * NUM_QUERY_TOKENS + tail

        keras.utils.set_random_seed(SEED)
        stack = SAM2MemoryAttention(
            d_model=D_MODEL, num_layers=NUM_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD, dropout=0.1,
            num_heads=1, downsample_rate=1, feat_sizes=GRID,
            kv_in_dim=KV_IN_DIM,
        )
        stack.build((None, NUM_QUERY_TOKENS, D_MODEL),
                    (None, num_memory, KV_IN_DIM))

        rng = np.random.RandomState(SEED + 2)
        data = {
            "features": rng.randn(BATCH, NUM_QUERY_TOKENS, D_MODEL).astype("float32"),
            "memory": rng.randn(BATCH, num_memory, KV_IN_DIM).astype("float32"),
            "features_pos": rng.randn(BATCH, NUM_QUERY_TOKENS, D_MODEL).astype("float32"),
            "memory_pos": rng.randn(BATCH, num_memory, KV_IN_DIM).astype("float32"),
        }
        args, kwargs, result = _capture_rope_calls(
            stack, data, num_obj_ptr_tokens=tail)[0]
        key_in = np.asarray(ops.convert_to_numpy(args[1]))
        key_out = np.asarray(ops.convert_to_numpy(result[1]))

        # Value first, spy second: the value assertion is the discriminator, and
        # asserting the spy first would mask it in the failure report.
        assert _max_abs_diff(key_in[:, :, -tail:, :],
                             key_out[:, :, -tail:, :]) == 0.0, (
            "the object-pointer tail was ROTATED even though the exclusion "
            "count was available -- num_k_exclude is not reaching RoPE"
        )
        assert _max_abs_diff(key_in[:, :, :-tail, :],
                             key_out[:, :, :-tail, :]) > 1e-4
        assert kwargs.get("num_k_exclude") == tail

    def test_rope_broadcasts_one_table_across_memory_frames(self) -> None:
        """H-13: identical spatial content in two frames rotates identically.

        This is the discriminator between `repeat_k` (spatial, frame-agnostic)
        and a temporal RoPE (frame-indexed phases): if any block index entered
        the angle table, the two frames' rotated rows would differ.
        """
        stack = _build_stack()
        rng = np.random.RandomState(SEED + 1)
        frame = rng.randn(BATCH, NUM_QUERY_TOKENS, KV_IN_DIM).astype("float32")
        frame_pos = rng.randn(BATCH, NUM_QUERY_TOKENS, KV_IN_DIM).astype("float32")
        tail = rng.randn(BATCH, NUM_OBJ_PTR_TOKENS, KV_IN_DIM).astype("float32")
        tail_pos = rng.randn(BATCH, NUM_OBJ_PTR_TOKENS, KV_IN_DIM).astype("float32")
        data = dict(_inputs())
        # Both the content AND the positional tensor are tiled per frame: the
        # only thing that could make the rotated rows differ is the rotation
        # itself. (In the real model the per-frame difference arrives through
        # `memory_pos` as an ADDITIVE temporal embedding, which is precisely the
        # mechanism this test asserts RoPE does not duplicate.)
        data["memory"] = np.concatenate(
            [frame] * NUM_MEMORY_FRAMES + [tail], axis=1)
        data["memory_pos"] = np.concatenate(
            [frame_pos] * NUM_MEMORY_FRAMES + [tail_pos], axis=1)
        args, _kwargs, result = _capture_rope_calls(stack, data)[0]
        key_out = np.asarray(ops.convert_to_numpy(result[1]))

        first = key_out[:, :, :NUM_QUERY_TOKENS, :]
        for block_index in range(1, NUM_MEMORY_FRAMES):
            other = key_out[
                :, :,
                block_index * NUM_QUERY_TOKENS:(block_index + 1) * NUM_QUERY_TOKENS,
                :,
            ]
            assert _max_abs_diff(first, other) == 0.0, (
                f"memory frame {block_index} rotated differently from frame 0; "
                f"a frame index leaked into the RoPE angle table (temporal "
                f"position must be additive, not rotary)"
            )


# ---------------------------------------------------------------------
# G2.4 -- graph traceability
# ---------------------------------------------------------------------


# DECISION plan-2026-08-04T044628-4c240b4c/D-009
# The plan's named RED mutation for G2.4 -- "insert an `ops.image.resize` on a
# dynamic size" -- was MEASURED to be INERT on this stack. Both bilinear and
# bicubic `keras.ops.image.resize` with a size built from `ops.shape(x)[0] + 1`
# trace cleanly under `tf.function`, producing a `(None, None, None, C)` output
# spec. The guard below stayed GREEN under it.
#
# Do NOT re-introduce that mutation as this guard's RED proof, and do NOT read
# H-2 ("`SAM.call` cannot be traced because of `ops.image.resize`") as "a
# dynamic-size resize raises under trace" -- it does not, here. The substituted
# mutation that DOES fire is an eager `float()` of a traced tensor
# (`float(ops.mean(features_pos))` in `SAM2MemoryAttention.call`), which raises
# `TypeError`, not the `ValueError`/`OperatorNotAllowedInGraphError` one might
# predict. See decisions.md D-009.
class TestGraphTrace:
    """G2.4: the stack traces under ``tf.function`` with static signatures."""

    def test_call_traces_with_static_input_signature(self) -> None:
        stack = _build_stack()
        signature = [
            tf.TensorSpec((None, NUM_QUERY_TOKENS, D_MODEL), tf.float32),
            tf.TensorSpec((None, NUM_MEMORY_TOKENS, KV_IN_DIM), tf.float32),
            tf.TensorSpec((None, NUM_QUERY_TOKENS, D_MODEL), tf.float32),
            tf.TensorSpec((None, NUM_MEMORY_TOKENS, KV_IN_DIM), tf.float32),
        ]

        @tf.function(input_signature=signature)
        def traced(features, memory, features_pos, memory_pos):
            return stack(
                features, memory,
                features_pos=features_pos,
                memory_pos=memory_pos,
                num_obj_ptr_tokens=NUM_OBJ_PTR_TOKENS,
                training=False,
            )

        concrete = traced.get_concrete_function()
        assert concrete is not None

        data = _inputs()
        traced_out = np.asarray(traced(
            data["features"], data["memory"],
            data["features_pos"], data["memory_pos"],
        ))
        assert _max_abs_diff(traced_out, _forward(stack, data)) < 1e-5

    def test_dynamic_token_axis_raises_a_named_error(self) -> None:
        """A dynamic token axis is refused loudly, not silently mis-rotated."""
        stack = _build_stack()

        @tf.function(input_signature=[
            tf.TensorSpec((None, None, D_MODEL), tf.float32),
            tf.TensorSpec((None, NUM_MEMORY_TOKENS, KV_IN_DIM), tf.float32),
        ])
        def traced(features, memory):
            return stack(features, memory, training=False)

        with pytest.raises(ValueError, match="STATIC token axis"):
            traced.get_concrete_function()


# ---------------------------------------------------------------------
# G2.5 -- the MEASURED dead-component RED partition
# ---------------------------------------------------------------------


class TestDeadComponentPartition:
    """G2.5: which half of each guard actually dies under a dead component.

    The prediction "all guards go RED under any dead component" is a hypothesis
    and it is FALSE here. These tests encode the partition that was MEASURED,
    including the halves that stay green -- because a guard that cannot go red
    is the thing worth knowing about.
    """

    def test_dead_cross_attention_collapses_the_16_way_distinctness(self) -> None:
        """Killing the cross-attention output projection kills 2 of 4 booleans.

        ``pos_enc_at_cross_attn_queries`` and ``pos_enc_at_cross_attn_keys`` can
        only act through the cross-attention residual, so with that residual
        forced to zero the sixteen combinations collapse into four groups of
        four. ``pos_enc_at_input`` and ``pos_enc_at_attn`` still separate them.
        """
        data = _inputs()
        outputs: Dict[Tuple[bool, ...], np.ndarray] = {}
        for combo in itertools.product([False, True], repeat=4):
            stack = _build_stack(*combo)
            dead = [
                variable
                for block in stack.layers
                for variable in block.cross_attn_image.out_proj.weights
            ]
            with zeroed_variables(dead):
                outputs[combo] = _forward(stack, data)

        collapsed = [
            (left, right)
            for left, right in itertools.combinations(sorted(outputs), 2)
            if _max_abs_diff(outputs[left], outputs[right]) <= 1e-5
        ]
        assert collapsed, (
            "the 16-way distinctness guard stayed GREEN with the cross-attention "
            "residual forced to zero -- it cannot detect a dead cross-attention"
        )
        # Every collapsed pair must agree on the two NON-cross-attention flags:
        # those two remain live even with cross-attention dead.
        for left, right in collapsed:
            assert left[0] == right[0] and left[1] == right[1], (
                f"{left} and {right} collapsed while differing in "
                f"pos_enc_at_input/pos_enc_at_attn -- more died than the "
                f"cross-attention path"
            )

    def test_dead_ffn_input_kills_the_gelu_discriminator_not_the_zero_count(
            self) -> None:
        """The measured asymmetry: one half of G2.2 survives a dead FFN.

        With ``linear1`` zeroed, every pre-activation is exactly 0.0, so relu
        and gelu both emit all-zero hidden tensors. The relu zero-count
        assertion therefore CANNOT distinguish them (its precondition -- that
        negatives exist -- is what dies), while the relu-vs-gelu output
        comparison collapses to an exact 0.0 difference.
        """
        data = _inputs()
        relu_stack = _build_stack(activation="relu")
        gelu_stack = _build_stack(activation="gelu")

        dead_relu = [
            v for block in relu_stack.layers for v in block.linear1.weights]
        dead_gelu = [
            v for block in gelu_stack.layers for v in block.linear1.weights]

        with zeroed_variables(dead_relu):
            pre, hidden = _capture_ffn_hidden(relu_stack, data)
            relu_out = _forward(relu_stack, data)
        with zeroed_variables(dead_gelu):
            gelu_out = _forward(gelu_stack, data)

        # The zero-count assertion's PRECONDITION dies -- there are no negatives
        # left to discriminate on. That is the guard's blind spot, stated.
        assert int(np.sum(pre < 0.0)) == 0
        assert int(np.sum(hidden == 0.0)) == hidden.size
        # And the gelu discriminator goes fully red.
        assert _max_abs_diff(relu_out, gelu_out) == 0.0

    def test_dead_key_projection_kills_the_rotation_half_not_the_tail_half(
            self) -> None:
        """The measured asymmetry: G2.3's tail-identity half survives.

        With the key projection zeroed, every key row is 0.0. Rotating zero is
        zero, so the tail-identity assertion stays GREEN while the "spatial rows
        were rotated" assertion goes RED. This is exactly why G2.3 asserts both
        halves.
        """
        data = _inputs()
        stack = _build_stack()
        dead = [
            v for block in stack.layers
            for v in block.cross_attn_image.k_proj.weights
        ]
        with zeroed_variables(dead):
            args, _kwargs, result = _capture_rope_calls(stack, data)[0]

        key_in = np.asarray(ops.convert_to_numpy(args[1]))
        key_out = np.asarray(ops.convert_to_numpy(result[1]))

        tail_diff = _max_abs_diff(
            key_in[:, :, -NUM_OBJ_PTR_TOKENS:, :],
            key_out[:, :, -NUM_OBJ_PTR_TOKENS:, :],
        )
        head_diff = _max_abs_diff(
            key_in[:, :, :-NUM_OBJ_PTR_TOKENS, :],
            key_out[:, :, :-NUM_OBJ_PTR_TOKENS, :],
        )
        assert tail_diff == 0.0, "tail-identity half unexpectedly went red"
        assert head_diff == 0.0, (
            "the rotation half stayed GREEN with a dead key projection -- it is "
            "not observing the key tensor at all"
        )
