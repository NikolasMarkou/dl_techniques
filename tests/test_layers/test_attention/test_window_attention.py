import pytest
import numpy as np
import keras
import os
import tempfile
import tensorflow as tf


from dl_techniques.layers.attention.window_attention import (
    WindowAttention,
    create_grid_window_attention,
    create_zigzag_window_attention,
    create_kan_key_window_attention,
    create_adaptive_softmax_window_attention,
)
from dl_techniques.layers.attention.single_window_attention import SingleWindowAttention


def build_transformer_block(inputs, dim, window_size, num_heads, **kwargs):
    """Helper function to build a Transformer block with the unified WindowAttention."""
    x1 = keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    # Pass all additional kwargs to the unified layer
    attn_out = WindowAttention(
        dim=dim, window_size=window_size, num_heads=num_heads, **kwargs
    )(x1)
    x = keras.layers.Add()([inputs, attn_out])
    x2 = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    mlp_out = keras.layers.Dense(int(dim * 4.0), activation="gelu")(x2)
    mlp_out = keras.layers.Dense(dim)(mlp_out)
    return keras.layers.Add()([x, mlp_out])


# Shared configurations for forward pass tests
common_configs = [
    # Grid mode variations
    {"partition_mode": "grid", "attention_mode": "linear", "probability_type": "softmax",
     "use_relative_position_bias": True},
    {"partition_mode": "grid", "attention_mode": "kan_key", "probability_type": "softmax",
     "use_relative_position_bias": True},
    {"partition_mode": "grid", "attention_mode": "linear", "probability_type": "adaptive",
     "use_relative_position_bias": False},
    # Zigzag mode variations
    {"partition_mode": "zigzag", "attention_mode": "linear", "probability_type": "softmax",
     "use_relative_position_bias": False},
    # hierarchical_routing is disallowed at score level; substitute sparsemax for
    # generic alt-to-softmax coverage.
    {"partition_mode": "zigzag", "attention_mode": "kan_key", "probability_type": "sparsemax",
     "use_relative_position_bias": False},
    {"partition_mode": "zigzag", "attention_mode": "linear", "probability_type": "softmax",
     "use_relative_position_bias": True},
]


class TestWindowAttention:
    """Comprehensive test suite for the unified WindowAttention layer."""

    @pytest.fixture
    def input_tensor(self):
        """Create a standard test input tensor."""
        return keras.random.normal([4, 50, 96])

    # 1. Initialization and Configuration Tests
    # =================================================================

    def test_initialization_defaults(self):
        """Test layer initializes with grid mode and standard linear attention."""
        layer = WindowAttention(dim=96, window_size=7, num_heads=4)
        assert layer.partition_mode == "grid"

        inner_attn = layer.attention
        assert isinstance(inner_attn, SingleWindowAttention)
        assert inner_attn.attention_mode == "linear"
        assert inner_attn.probability_type == "softmax"
        assert inner_attn.use_relative_position_bias is True
        assert hasattr(inner_attn, "qkv")
        assert not hasattr(inner_attn, "query")

    def test_initialization_zigzag_mode(self):
        """Test initialization with zigzag partition mode."""
        layer = WindowAttention(dim=96, window_size=7, num_heads=4, partition_mode="zigzag")
        assert layer.partition_mode == "zigzag"
        # Check that it prepares attributes for zigzag indices
        assert hasattr(layer, "zigzag_indices")

    def test_initialization_kan_mode(self):
        """Test initialization with KAN-based attention."""
        layer = WindowAttention(dim=96, window_size=7, num_heads=4, attention_mode="kan_key")
        inner_attn = layer.attention
        assert inner_attn.attention_mode == "kan_key"
        assert hasattr(inner_attn, "query")
        assert hasattr(inner_attn, "key")  # KANLinear
        assert hasattr(inner_attn, "value")
        assert not hasattr(inner_attn, "qkv")
        assert inner_attn.kan_grid_size == 5  # Default value

    @pytest.mark.parametrize("prob_type", ["adaptive", "sparsemax"])
    def test_initialization_advanced_probability(self, prob_type):
        """Test initialization with advanced probability normalization schemes."""
        layer = WindowAttention(dim=96, window_size=7, num_heads=4, probability_type=prob_type)
        inner_attn = layer.attention
        assert inner_attn.probability_type == prob_type
        assert hasattr(inner_attn, "attn_prob")

    @pytest.mark.parametrize("prob_type", ["hierarchical_routing", "routing",
                                            "deterministic_routing", "hierarchical"])
    def test_initialization_rejects_score_level_routing(self, prob_type):
        """Score-level routing variants are structurally incompatible with windowed scores."""
        with pytest.raises(ValueError):
            WindowAttention(dim=96, window_size=7, num_heads=4, probability_type=prob_type)

    def test_initialization_no_relative_bias(self):
        """Test explicit disabling of relative position bias."""
        layer = WindowAttention(dim=96, window_size=7, num_heads=4, use_relative_position_bias=False)
        assert layer.use_relative_position_bias is False
        assert layer.attention.use_relative_position_bias is False

    # 2. Build and Weight Creation Tests
    # =================================================================

    @pytest.mark.parametrize(
        "config, expected_attrs, forbidden_attrs",
        [
            # Standard grid mode: expects relative bias table, no zigzag indices
            (
                    {"partition_mode": "grid", "use_relative_position_bias": True, "window_size": 7},
                    ["relative_position_bias_table"],
                    ["zigzag_indices"],
            ),
            # Grid mode without bias, different window size
            (
                    {"partition_mode": "grid", "use_relative_position_bias": False, "window_size": 4},
                    [],
                    ["relative_position_bias_table", "zigzag_indices"],
            ),
            # Zigzag mode with bias
            (
                    {"partition_mode": "zigzag", "use_relative_position_bias": True, "window_size": 7,
                     "input_shape": (4, 10, 96)},
                    ["zigzag_indices", "inverse_zigzag_indices", "relative_position_bias_table"],
                    [],
            ),
            # Zigzag mode without bias, different window size
            (
                    {"partition_mode": "zigzag", "use_relative_position_bias": False, "window_size": 6,
                     "input_shape": (4, 10, 96)},
                    ["zigzag_indices", "inverse_zigzag_indices"],
                    ["relative_position_bias_table"],
            ),
        ],
    )
    def test_build_process_combinations(self, config, expected_attrs, forbidden_attrs):
        """Test that build creates the correct weights/attributes for each mode and window size."""
        input_shape = config.pop("input_shape", (4, 50, 96))
        window_size = config.pop("window_size")
        layer = WindowAttention(dim=96, window_size=window_size, num_heads=4, **config)
        layer.build(input_shape)

        assert layer.built

        inner_attn = layer.attention
        for attr in expected_attrs:
            if attr.endswith("_table"):  # It's a weight on the inner layer
                assert hasattr(inner_attn, attr)
            else:  # It's an attribute on the outer layer
                assert hasattr(layer, attr)
                assert getattr(layer, attr) is not None

        for attr in forbidden_attrs:
            if attr.endswith("_table"):
                assert not hasattr(inner_attn, attr)
            else:
                assert not hasattr(layer, attr) or getattr(layer, attr) is None

    # 3. Forward Pass and Functional Correctness Tests
    # =================================================================

    @pytest.mark.parametrize("config", common_configs)
    @pytest.mark.parametrize("window_size", [4, 7])
    @pytest.mark.parametrize("seq_len", [49, 50, 64])
    def test_forward_pass_combinations(self, config, window_size, seq_len):
        """Test forward pass for a matrix of configurations, window sizes, and sequence lengths."""
        dim, num_heads = 96, 4
        input_tensor = keras.random.normal([4, seq_len, dim])
        layer = WindowAttention(dim=dim, window_size=window_size, num_heads=num_heads, **config)

        # Test training and inference
        output_train = layer(input_tensor, training=True)
        output_infer = layer(input_tensor, training=False)

        assert output_train.shape == input_tensor.shape
        assert output_infer.shape == input_tensor.shape
        assert not np.any(np.isnan(output_train.numpy()))
        assert not np.any(np.isnan(output_infer.numpy()))

    @pytest.mark.parametrize("partition_mode", ["grid", "zigzag"])
    @pytest.mark.parametrize("window_size", [4, 5, 8])
    @pytest.mark.parametrize("seq_len", [55, 60])
    def test_attention_mask_integration(self, partition_mode, window_size, seq_len):
        """Test that the attention mask is correctly applied across modes, window sizes, and lengths."""
        dim, num_heads = 32, 4
        layer = WindowAttention(dim=dim, window_size=window_size, num_heads=num_heads, partition_mode=partition_mode)
        input_data = keras.random.normal((2, seq_len, dim))

        mask = np.ones((2, seq_len), dtype="int32")
        mask[:, -10:] = 0  # Mask out the last 10 tokens

        output = layer(input_data, attention_mask=keras.ops.convert_to_tensor(mask))
        assert output.shape == input_data.shape
        assert not np.any(np.isnan(output.numpy()))

    @pytest.mark.parametrize("partition_mode", ["grid", "zigzag"])
    @pytest.mark.parametrize("seq_len", [1, 15, 16, 63, 64, 100])
    @pytest.mark.parametrize("window_size", [3, 4, 8])
    def test_arbitrary_shapes_and_windows(self, partition_mode, seq_len, window_size):
        """Test robustness to various sequence lengths and window sizes."""
        dim, num_heads = 32, 2
        layer = WindowAttention(dim=dim, window_size=window_size, num_heads=num_heads, partition_mode=partition_mode)
        input_data = keras.random.normal((4, seq_len, dim))
        output = layer(input_data)
        assert output.shape == input_data.shape
        assert not np.any(np.isnan(output.numpy()))

    @pytest.mark.parametrize(
        "config",
        [
            {"attention_mode": "linear", "use_relative_position_bias": True},
            {"attention_mode": "kan_key", "use_relative_position_bias": False},
            {"probability_type": "adaptive"},
            {"probability_type": "sparsemax"},
        ]
    )
    @pytest.mark.parametrize("window_size", [3, 4])
    def test_gradient_flow(self, config, window_size):
        """Ensure gradients flow correctly for all trainable modes and window sizes."""
        layer = WindowAttention(dim=32, window_size=window_size, num_heads=2, **config)
        input_data = tf.Variable(keras.random.normal((2, 10, 32)))

        with tf.GradientTape() as tape:
            output = layer(input_data)
            loss = keras.ops.sum(output)

        grads = tape.gradient(loss, layer.trainable_variables)
        assert len(grads) == len(layer.trainable_variables)
        for grad in grads:
            assert grad is not None
            assert keras.ops.any(keras.ops.not_equal(grad, 0))

    # 4. Keras Ecosystem Integration Tests
    # =================================================================

    @pytest.mark.parametrize(
        "config",
        [
            # Grid mode with KAN and regularization, different window sizes
            {
                "partition_mode": "grid", "attention_mode": "kan_key", "window_size": 8,
                "kan_grid_size": 8, "kernel_regularizer": "l2"
            },
            # Zigzag mode with adaptive softmax and different window size
            {
                "partition_mode": "zigzag", "probability_type": "adaptive", "window_size": 5,
                "use_relative_position_bias": False,
                "probability_config": {"min_temp": 0.1}
            },
            # Standard default config with a non-default window size
            {"window_size": 6},
        ]
    )
    def test_serialization_comprehensive(self, config):
        """Test get_config and from_config for complex configurations and window sizes."""
        base_config = {"dim": 64, "num_heads": 4}
        full_config = {**base_config, **config}

        layer = WindowAttention(**full_config)
        input_shape = (None, 20, 64)
        layer.build(input_shape)

        config_dict = layer.get_config()
        recreated_layer = WindowAttention.from_config(config_dict)
        recreated_layer.build(input_shape)

        assert recreated_layer.get_config() == config_dict
        assert len(recreated_layer.weights) == len(layer.weights)

    @pytest.mark.parametrize(
        "config",
        [
            {"partition_mode": "grid", "name": "grid_attn", "window_size": 7, "num_heads": 3},
            {"partition_mode": "zigzag", "attention_mode": "kan_key", "name": "zigzag_kan_attn", "window_size": 5,
             "num_heads": 4},
            {"partition_mode": "grid", "probability_type": "adaptive", "name": "grid_adaptive",
             "window_size": 4, "num_heads": 2}
        ]
    )
    def test_model_save_load(self, config, input_tensor):
        """Test saving and loading a model containing the layer with various configs."""
        dim = 96
        inputs = keras.Input(shape=input_tensor.shape[1:])
        outputs = WindowAttention(dim=dim, **config)(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        original_prediction = model.predict(input_tensor, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.keras")
            model.save(path)
            loaded_model = keras.models.load_model(path)
            loaded_prediction = loaded_model.predict(input_tensor, verbose=0)

            assert np.allclose(original_prediction, loaded_prediction, atol=1e-6)
            assert isinstance(loaded_model.get_layer(config["name"]), WindowAttention)

    # 5. Factory Function Tests
    # =================================================================

    def test_factory_grid_window_attention(self):
        """Test the grid attention factory function and kwarg passthrough."""
        layer = create_grid_window_attention(
            dim=96, window_size=7, num_heads=4, dropout_rate=0.1, qkv_bias=False
        )
        assert isinstance(layer, WindowAttention)
        assert layer.partition_mode == "grid"
        assert layer.use_relative_position_bias is True  # Default for grid
        assert layer.dropout_rate == 0.1
        assert layer.qkv_bias is False

    def test_factory_zigzag_window_attention(self):
        """Test the zigzag attention factory function and kwarg passthrough."""
        layer = create_zigzag_window_attention(
            dim=96, window_size=7, num_heads=4, proj_bias=False
        )
        assert isinstance(layer, WindowAttention)
        assert layer.partition_mode == "zigzag"
        assert layer.use_relative_position_bias is False  # Default for zigzag
        assert layer.proj_bias is False

    def test_factory_kan_key_window_attention(self):
        """Test the KAN key attention factory function and kwarg passthrough."""
        # Test with default grid mode and KAN params
        layer_grid = create_kan_key_window_attention(
            dim=96, window_size=7, num_heads=4, kan_grid_size=10
        )
        assert layer_grid.attention.attention_mode == "kan_key"
        assert layer_grid.partition_mode == "grid"
        assert layer_grid.kan_grid_size == 10

        # Test with zigzag mode and KAN params
        layer_zigzag = create_kan_key_window_attention(
            dim=96, window_size=7, num_heads=4, partition_mode="zigzag", kan_spline_order=2
        )
        assert layer_zigzag.attention.attention_mode == "kan_key"
        assert layer_zigzag.partition_mode == "zigzag"
        assert layer_zigzag.kan_spline_order == 2

    def test_factory_adaptive_softmax_attention(self):
        """Test the adaptive softmax attention factory function and kwarg passthrough."""
        config = {"min_temp": 0.5}
        layer = create_adaptive_softmax_window_attention(
            dim=96, window_size=7, num_heads=4, probability_config=config
        )
        assert layer.attention.probability_type == "adaptive"
        assert layer.partition_mode == "grid"  # Default partition mode
        assert layer.probability_config == config


class TestAllOnesKeyMaskIsANoOp:
    """An all-ones ``(B, N)`` key mask masks NOTHING and must change NOTHING.

    ``attention_mask`` is a KEEP predicate (``1 = attend``, pinned by
    ``SingleWindowAttention``'s D-002 anchor and by ``common.apply_attention_mask``,
    which performs no polarity inference). An all-ones mask therefore keeps every one
    of the ``N`` real tokens: it is the identity, and the layer's output must be
    literally the same bits it produces with ``attention_mask=None``.

    **Why this guard exists.** Before plan ``plan-2026-08-25T053412-0f1fa04f`` step 3
    it was FALSE, by a lot. ``WindowAttention._call_grid`` lays a short sequence into a
    ``ceil(sqrt(N)) x ceil(sqrt(N))`` grid, zero-pads that grid up to a whole number of
    ``window_size x window_size`` tiles, and hands each tile to
    ``SingleWindowAttention`` as a full window. The inner layer's own padding mask is
    derived from ``N_actual`` versus ``N_target``, which are equal by then, so it is
    ALL ONES and masks nothing: the zero-filled pad slots entered every real token's
    softmax as ordinary keys and values. Passing an explicit all-ones ``(B, N)`` mask
    was the one thing that revealed it, because that mask IS zero-padded alongside the
    sequence and so does mask the pads -- turning a mathematical no-op into a visible
    output change. MEASURED on the pre-fix code (commit ``4cb5f47c9``), max |delta|
    between the no-mask and all-ones-mask outputs::

        window_size=4, N=9   ->  0.341491      window_size=8, N=64  ->  0.0  (N == ws**2)
        window_size=7, N=25  ->  0.255033      window_size=2, N=4   ->  0.0  (N == ws**2)
        window_size=8, N=50  ->  0.0876751

    The two zeros are the control: the leak appears if and only if there are pad slots.
    This was not merely an efficiency defect -- in the ``N < window_size ** 2`` regime
    the layer computed the WRONG attention, and that is exactly the regime ModernBERT's
    local layers ran in. See decisions.md D-007
    (``plan-2026-08-25T053412-0f1fa04f``).

    **The comparison is exact, not ``allclose``.** Both sides run the same kernels on
    the same weights in the same process; the mask, being all ones, contributes
    ``0 * -1e9 == 0.0`` to every score, so there is no legitimate nonzero delta to
    absorb. MEASURED: every cell in :data:`NO_OP_CELLS` is bit-for-bit ``0.0``,
    including ``window_size=8, N=63``, where a single pad slot would otherwise leak.
    A tolerance would let exactly that one-slot case through.
    """

    #: ``(window_size, seq_len)`` where step 3's short-circuit, or the absence of any
    #: padding, makes the identity mask a genuine no-op TODAY.
    NO_OP_CELLS = [
        # Ragged single window -- the regime step 3 rewrote. Pre-fix these leaked.
        ("grid", 4, 9),
        ("grid", 7, 25),
        ("grid", 8, 50),
        ("grid", 8, 63),
        # N == window_size ** 2 -- one window, nothing to pad. This is how
        # SwinTransformerBlock, FastVLM and TiRex call the layer.
        ("grid", 2, 4),
        ("grid", 8, 64),
        ("zigzag", 2, 4),
        ("zigzag", 8, 64),
        # N > window_size ** 2 and the grid tiles exactly -- nothing to pad either.
        ("grid", 2, 16),
        ("grid", 4, 64),
        ("grid", 7, 196),
        ("grid", 8, 256),
        ("zigzag", 2, 16),
        ("zigzag", 4, 64),
        ("zigzag", 7, 196),
        ("zigzag", 8, 256),
    ]

    # DECISION plan-2026-08-25T053412-0f1fa04f/D-008
    #: The SAME property, in the two regimes step 3 deliberately did NOT touch. These
    #: are ``xfail(strict=True)``, not deletions and not comments: they are live
    #: measurements that will turn XPASS -- and so fail the suite -- the moment either
    #: regime is fixed, which is the only way a known-open defect stays visible. Do NOT
    #: relax them to plain ``xfail``.
    #:
    #: 1. ``grid`` with ``N > window_size ** 2`` where the grid does not tile exactly:
    #:    ``ceil(sqrt(N))**2 - N`` sequence pads and/or ``pad_h``/``pad_w`` tile pads
    #:    still enter the softmax unmasked. Step 3's short-circuit covers only the
    #:    single-window case ``N < window_size ** 2``.
    #: 2. ``zigzag`` at ANY ragged ``N``: the short-circuit lives in ``_call_grid``
    #:    only, so ``_call_zigzag`` still pads to a whole number of windows.
    #:
    #: The fourth field is the delta MEASURED by :meth:`_delta` on the current tree
    #: (2026-08-25). It is reported in the failure message and is NOT asserted on:
    #: a magnitude assertion inside an ``xfail`` would keep the cell red -- and the
    #: ``xfail`` satisfied -- even after the leak was fixed, which is precisely the
    #: guard-that-cannot-pass this repo has been burned by before. The ONLY assertion
    #: is ``delta == 0.0``, so XPASS means fixed and nothing else can produce it.
    #:
    #: WHAT NOT TO DO: do not "fix" a red cell here by moving it out of this list, and
    #: do not widen step 3's short-circuit to `N <= window_size ** 2` hoping to cover
    #: them -- these two regimes need a pad mask inside the MULTI-window grid path and
    #: a slot map for the permuted zigzag layout, and fixing zigzag additionally breaks
    #: the 18 zigzag cells that `test_window_attention_restructure_is_inert.py` still
    #: pins BITWISE to the leaky pre-restructure reference (D-007 narrowed the grid
    #: cells only). See decisions.md D-008 (plan-2026-08-25T053412-0f1fa04f).
    LEAKY_CELLS = [
        ("grid", 2, 15, 0.298543),
        ("grid", 4, 20, 0.678637),
        ("grid", 8, 100, 1.000270),
        ("zigzag", 4, 9, 0.271005),
        ("zigzag", 7, 25, 0.382822),
        ("zigzag", 8, 50, 0.086691),
    ]

    @staticmethod
    def _delta(partition_mode, window_size, seq_len):
        """max |output(no mask) - output(all-ones mask)| for one configuration."""
        dim, num_heads, batch = 32, 4, 2
        keras.utils.set_random_seed(7)
        layer = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            partition_mode=partition_mode,
            use_relative_position_bias=True,
            dropout_rate=0.0,
        )
        rng = np.random.default_rng(abs(hash((partition_mode, window_size, seq_len))) % (2 ** 32))
        x = rng.standard_normal((batch, seq_len, dim)).astype("float32")
        ones = np.ones((batch, seq_len), dtype="int32")

        unmasked = np.asarray(keras.ops.convert_to_numpy(layer(x, training=False)))
        masked = np.asarray(
            keras.ops.convert_to_numpy(
                layer(x, attention_mask=keras.ops.convert_to_tensor(ones), training=False)
            )
        )
        # Non-vacuity: two constant (or NaN-free-but-dead) outputs would compare equal
        # no matter how broken the layer is.
        assert np.all(np.isfinite(unmasked)), "the unmasked output is not finite"
        assert float(np.std(unmasked)) > 1e-6, (
            f"the unmasked output is effectively constant "
            f"(std={float(np.std(unmasked))}) -- this comparison would be vacuous"
        )
        assert unmasked.shape == masked.shape == (batch, seq_len, dim)
        return float(np.abs(unmasked.astype("float64") - masked.astype("float64")).max())

    @pytest.mark.parametrize(
        "partition_mode,window_size,seq_len",
        NO_OP_CELLS,
        ids=[f"{m}-ws{w}-N{n}" for (m, w, n) in NO_OP_CELLS],
    )
    def test_an_all_ones_key_mask_does_not_change_the_output(
        self, partition_mode, window_size, seq_len
    ):
        delta = self._delta(partition_mode, window_size, seq_len)
        assert delta == 0.0, (
            f"WindowAttention(window_size={window_size}, "
            f"partition_mode={partition_mode!r}) on (2, {seq_len}, 32): an all-ones "
            f"(B, N) key mask masks no real token and must be a NO-OP, but it moved "
            f"the output by {delta}. That is the signature of unmasked PAD slots "
            f"reaching the softmax -- the defect D-007 records "
            f"(plan-2026-08-25T053412-0f1fa04f). Do NOT fix this by widening the "
            f"assertion: find the pad slots."
        )

    @pytest.mark.parametrize(
        "partition_mode,window_size,seq_len,measured",
        LEAKY_CELLS,
        ids=[f"{m}-ws{w}-N{n}" for (m, w, n, _) in LEAKY_CELLS],
    )
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "KNOWN OPEN: the unmasked-pad leak D-007 describes survives outside the "
            "single-window grid regime step 3 fixed -- grid with a non-tiling N, and "
            "zigzag at any ragged N. XPASS here means it was fixed; delete the cell "
            "from LEAKY_CELLS and move it to NO_OP_CELLS."
        ),
    )
    def test_the_all_ones_mask_leak_is_still_open_outside_the_fixed_regime(
        self, partition_mode, window_size, seq_len, measured
    ):
        delta = self._delta(partition_mode, window_size, seq_len)
        assert delta == 0.0, (
            f"KNOWN OPEN (D-007): an all-ones (B, N) key mask still moves "
            f"WindowAttention(window_size={window_size}, "
            f"partition_mode={partition_mode!r}) on (2, {seq_len}, 32) by {delta} "
            f"(recorded {measured} on 2026-08-25). Unmasked pad slots are still "
            f"reaching the softmax in this regime."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])