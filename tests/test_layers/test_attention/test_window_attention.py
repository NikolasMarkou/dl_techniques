import pytest
import zlib

import numpy as np
import keras
import os
import tempfile
import tensorflow as tf


from dl_techniques.layers.attention.window_attention import (
    WindowAttention,
    create_grid_window_attention,
    create_zigzag_window_attention,
    create_band_window_attention,
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

    Those five magnitudes, and the six in the D-011 anchor below, were sampled while
    ``_delta`` seeded from ``hash(...)`` -- salted by ``PYTHONHASHSEED``, which pytest
    does not pin -- so they are INDICATIVE OF SCALE, not reproducible constants. The
    tell is in this file: ``zigzag ws=4, N=9`` reads ``0.341491`` here and ``0.3826489``
    fifty lines down, from the same helper on the same defect. ``_delta`` seeds from
    ``zlib.crc32`` since 2026-08-25 (D-013), so magnitudes recorded from here on ARE
    reproducible; these historical ones cannot be recovered and are kept for their
    order of magnitude and their sign, nothing more.

    The two zeros are the control: the leak appears if and only if there are pad slots.
    This was not merely an efficiency defect -- in the ``N < window_size ** 2`` regime
    the layer computed the WRONG attention, and that is exactly the regime ModernBERT's
    local layers ran in. See decisions.md D-007
    (``plan-2026-08-25T053412-0f1fa04f``).

    **The leak was WIDER than D-007 scoped, and step 4.1 closed the rest.** Step 3
    fixed only the degenerate single-window grid case (``1 < N < window_size ** 2``).
    Two regimes survived it: ``grid`` with ``N > window_size ** 2`` where the padded
    grid side is not a multiple of ``window_size`` (the sequence pad and the tile pad
    still entered the softmax), and ``zigzag`` at ANY ragged ``N`` (the short-circuit
    lived in ``_call_grid`` only). Those six cells sat here as ``xfail(strict=True)``
    between step 3.1 and step 4.1. MEASURED on ``8435dcc2f``, the commit immediately
    before the fix::

        grid    ws=8, N=100 -> 1.0258900      zigzag  ws=4, N=9  -> 0.3826489
        grid    ws=4, N=20  -> 0.7214095      zigzag  ws=7, N=25 -> 0.2675675
        grid    ws=2, N=15  -> 0.2340506      zigzag  ws=8, N=50 -> 0.0807583

    ``grid ws=8, N=100`` is the worst case by construction: the grid side is 10 and
    pads to 16, so the last tile holds more pad slots than real tokens and the
    softmax is dominated by zeros. Step 4.1 SYNTHESIZES the all-ones key mask
    whenever the geometry creates pads, so ``None`` and an explicit all-ones mask now
    travel the same pipeline by construction. All six read exactly ``0.0`` after it.
    See decisions.md D-011 (``plan-2026-08-25T053412-0f1fa04f``).

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
        # DECISION plan-2026-08-25T053412-0f1fa04f/D-011
        # The six cells below were `LEAKY_CELLS` -- strict xfails -- between step
        # 3.1 and step 4.1. Step 4.1 closed the general leak (the grid path with a
        # non-tiling N, and the zigzag path at any ragged N) by SYNTHESIZING an
        # all-ones key mask whenever the geometry creates pad slots, so `None` and
        # an explicit all-ones mask now take the same pipeline. They assert
        # `delta == 0.0` like every other cell; nothing about them is special any
        # more except their history. MEASURED leak on 8435dcc2f, immediately
        # before the fix, in this same helper: grid ws=8 N=100 -> 1.0258900,
        # grid ws=4 N=20 -> 0.7214095, grid ws=2 N=15 -> 0.2340506,
        # zigzag ws=4 N=9 -> 0.3826489, zigzag ws=7 N=25 -> 0.2675675,
        # zigzag ws=8 N=50 -> 0.0807583. All six read exactly 0.0 after it.
        # Those six are INDICATIVE, not reproducible: they were sampled while
        # `_delta` seeded from the PYTHONHASHSEED-salted `hash`. See D-013 and
        # the note in this class's docstring.
        # `grid ws=8 N=100` is the worst case by construction: the grid side is 10
        # and pads to 16, so the last tile holds more pad slots than real tokens.
        ("grid", 2, 15),
        ("grid", 4, 20),
        ("grid", 8, 100),
        ("zigzag", 4, 9),
        ("zigzag", 7, 25),
        ("zigzag", 8, 50),
        # 'band' NEVER pads -- there is no tile to pad up to -- so the D-007
        # property must hold at EVERY N, ragged or not, including the ragged
        # lengths where 'grid' and 'zigzag' used to leak.
        ("band", 3, 17),
        ("band", 4, 9),
        ("band", 8, 50),
        ("band", 8, 100),
        ("band", 2, 15),
        ("band", 64, 33),
    ]

    #: RED-PROOF of the six moved cells, and of every other cell in
    #: :data:`NO_OP_CELLS`. ``WindowAttention._pads_exist`` was made to
    #: ``return False`` unconditionally and the ``zigzag`` synthesis condition was
    #: ``and``-ed with ``False`` -- the D-011 defect itself put back, with every
    #: shape, weight and code path otherwise untouched. Observed, verbatim::
    #:
    #:     E  AssertionError: WindowAttention(window_size=8, partition_mode='grid') on (2, 100, 32): an all-ones (B, N) key mask masks no real token and must be a NO-OP, but it moved the output by 1.1499272659420967. That is the signature of unmasked PAD slots reaching the softmax -- the defect D-007 records (plan-2026-08-25T053412-0f1fa04f). Do NOT fix this by widening the assertion: find the pad slots.
    #:     E  assert 1.1499272659420967 == 0.0
    #:
    #: Exactly six cells of this class went red -- ``grid`` ws2-N15 / ws4-N20 /
    #: ws8-N100 and ``zigzag`` ws4-N9 / ws7-N25 / ws8-N50 -- and no other, which is
    #: the correct split: every remaining cell either pads nothing or is a ``band``
    #: cell, and neither reaches the synthesis. Reverted immediately.
    #:
    #: TWO injections that do NOT red this class, recorded because the reason is
    #: instructive: permuting the synthesized zigzag mask by
    #: ``inverse_zigzag_indices``, and flipping the window pad to
    #: ``constant_values=1``. Both corrupt ``key_mask``, which the ``None`` side and
    #: the all-ones side now SHARE, so both sides move together and the relation
    #: this class measures survives. An injection that moves both sides of a
    #: comparison proves nothing about that comparison. Those two are RED-proven in
    #: ``test_window_attention_restructure_is_inert.py``, which compares against an
    #: EXTERNAL reference and therefore does see them. Neither instrument is
    #: redundant.
    #:
    #: WAS ``LEAKY_CELLS`` (D-008): six strict-xfail cells pinning the residual
    #: unmasked-pad leak in the two regimes step 3.1 deliberately did not touch.
    #: Step 4.1 fixed both (D-011), so all six moved into :data:`NO_OP_CELLS` above
    #: and assert ``delta == 0.0`` like every other cell. The list is gone rather
    #: than emptied: a strict xfail list with no members is a guard that can never
    #: speak, and the property it guarded is now stated positively.

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
            # 'band' REFUSES the relative-position bias (it indexes a 2-D tile a
            # 1-D band does not have), so it is asked for only where it exists.
            # 'grid'/'zigzag' keep the True this guard has always used.
            use_relative_position_bias=partition_mode != "band",
            dropout_rate=0.0,
        )
        # DECISION plan-2026-08-25T053412-0f1fa04f/D-013
        # `zlib.crc32`, NOT `hash`. `hash` on a tuple containing a `str` is
        # salted by PYTHONHASHSEED, which pytest does not pin, so this seed --
        # and therefore the input draw, and therefore every magnitude this
        # helper has ever "recorded" -- differed between runs. The proof is in
        # this module's own history: the class docstring records
        # `zigzag ws=4, N=9 -> 0.341491` and the D-011 anchor below records
        # `0.3826489` for the SAME cell measured by the SAME helper. Neither is
        # wrong; they are two samples of a distribution that looked like a
        # constant. `delta == 0.0` survives any seed, so the assertions were
        # never at risk -- the NUMBERS IN THE PROSE were. WHAT NOT TO DO: do not
        # go back to `hash` "because the seed is arbitrary anyway". An arbitrary
        # but STABLE seed is what makes a recorded magnitude a fact rather than
        # a sample. See decisions.md D-013.
        seed = zlib.crc32(repr((partition_mode, window_size, seq_len)).encode()) % (2 ** 32)
        rng = np.random.default_rng(seed)
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


class TestTheDegenerateWindowIsShortCircuited:
    """When ``N < window_size ** 2`` the layer must attend ``N`` slots, not ``ws ** 2``.

    This class grades the COST DECISION, which no value comparison in this repo can
    see. Both partition paths are correct either way -- the padding path masks its
    pads (D-011) and the short-circuit never creates them -- so an implementation that
    silently stopped taking the cheap branch would return the same numbers to within a
    float32 ulp and every existing guard would stay green while the layer went back to
    costing 33x plain global attention. MEASURED on ``(1, 128, 64)`` at
    ``window_size=128``, CPU peak RSS, before step 7.1 and after::

        mode            before      after
        window   (grid)  0.649 GB   0.649 GB   (short-circuited since step 3)
        window_band      0.648 GB   0.648 GB   (never tiles, so never inflated)
        multi_head       0.643 GB   0.643 GB   (the reference: full global attention)
        window_zigzag   21.695 GB   0.649 GB   <- step 7.1

    So the assertion is on the SHAPE the inner ``SingleWindowAttention`` is handed --
    the one directly observable proxy for "how many slots enter the softmax" -- and it
    is made for BOTH modes, because ``'zigzag'`` spent steps 3 through 7 as the one
    path the short-circuit had never reached.

    RED-PROOF (a) -- DISABLE THE SHORT-CIRCUIT. ``_call_zigzag``'s ``degenerate``
    condition was ``and``-ed with ``False``, so the cheap branch is never taken while
    every value, weight and mask stays as it was. Observed, verbatim::

        E           AssertionError: WindowAttention(window_size=4, partition_mode='zigzag') on N=9 handed SingleWindowAttention a sequence of 16 slots to answer a question about 9 real tokens (window_slots=None). The degenerate single-window short-circuit is not being taken: at window_size=128 that inflation is 0.649 GB -> 21.695 GB, i.e. 33x the cost of plain global attention.
        E           assert 16 == 9

    **8 failed, 7 passed** in this class -- the three ``zigzag`` slot-count cells plus
    all five slot-map cells (which cannot grade a slot map that was never handed over),
    while all three ``grid`` slot-count cells and all four non-degenerate cells stayed
    green. That is the correct split for an injection confined to ``_call_zigzag``.

    Note what did NOT move, because it is the reason this class exists:
    ``TestAllOnesKeyMaskIsANoOp`` **28 passed** and
    ``test_window_attention_restructure_is_inert.py`` **42 passed** -- both fully GREEN
    under an injection that puts a 33x cost regression back. With the short-circuit off
    the padding path masks its own pads (D-011), so it agrees with the pad-masked
    golden BITWISE and with the all-ones mask EXACTLY. Every VALUE instrument in this
    plan is blind to the cost decision. Reverted.

    RED-PROOF (b) -- WRONG ADJACENCY, the forward/inverse confusion. The slot vector
    was changed to ``self.zigzag_indices[: int(static_n)]`` (the SCATTER order) from
    ``self.inverse_zigzag_indices[: int(static_n)]`` (the token's own position in the
    scan). Same shape, same values as a set, wrong pairing: token ``i`` is then told it
    sits where the ``i``-th SCANNED token sits, so every relative-position bias row is
    gathered for the wrong pair of slots. Observed, verbatim::

        E       AssertionError: zigzag window_slots for window_size=4, N=9 are [0, 1, 3, 6, 4, 2, 5, 7, 8], but token i must be told the position it occupies in the zigzag SCAN, which is [0, 1, 5, 2, 4, 6, 3, 7, 8]. These two vectors are permutations of each other, so nothing about the shapes, the weights or the attention PATTERN can tell them apart -- only the relative-position bias, which is gathered by slot, moves.

    **5 failed, 10 passed** in this class -- all five slot-map cells, and no other, so
    the slot-count arm correctly stayed green (the short-circuit IS being taken; it is
    taken with the wrong map). And the external instrument saw it too:
    ``test_window_attention_restructure_is_inert.py`` **3 failed, 39 passed** --
    exactly the three ``use_relative_position_bias=True`` ragged ``zigzag`` cells, at
    ``max |delta|`` 0.02612234279513359 (``ws=4, N=9``), 0.01416805386543274
    (``ws=7, N=25``) and 0.011001929640769958 (``ws=8, N=50``), i.e. four to five
    orders of magnitude above that file's ``RAGGED_ATOL`` of 5e-7. Their three ``rpb=False`` twins
    stayed green, and that split is itself the evidence the injection acts through the
    BIAS GATHER and nowhere else: with no bias table there is no slot map to get wrong,
    because a single window attends every token to every other regardless of the order
    they are listed in. Reverted.

    Why (b) is not a symmetry: step 4.1 recorded an injection that permuted the
    synthesized zigzag MASK and proved nothing at two of three cells, because the
    vector it permuted was all ones and every permutation of all-ones is the identity.
    This injection permutes a vector of DISTINCT slot ids that indexes a table of
    distinct rows, so no permutation of it is a symmetry -- except the identity, which
    is the correct answer.
    """

    #: ``(partition_mode, window_size, seq_len)``. Every cell has
    #: ``1 < N < window_size ** 2``, the regime the short-circuit owns, in both modes.
    DEGENERATE_CELLS = [
        ("grid", 4, 9),
        ("grid", 7, 25),
        ("grid", 8, 50),
        ("zigzag", 4, 9),
        ("zigzag", 7, 25),
        ("zigzag", 8, 50),
    ]

    #: ``(partition_mode, window_size, seq_len)`` where the short-circuit must NOT be
    #: taken. ``N == window_size ** 2`` is the exact boundary -- one window with
    #: nothing to pad, so there is nothing to save and it is how ``SwinTransformerBlock``
    #: calls this layer -- and ``N > window_size ** 2`` genuinely tiles.
    NON_DEGENERATE_CELLS = [
        ("grid", 8, 64),
        ("zigzag", 8, 64),
        ("grid", 4, 64),
        ("zigzag", 4, 64),
    ]

    @staticmethod
    def _record(monkeypatch):
        """Spy on every ``SingleWindowAttention.call``, returning the record list.

        Patched on the CLASS, not the instance, and with the identical signature:
        Keras 3 introspects ``call``'s signature to decide how to route ``training``
        and the mask, so an instance attribute or a ``*args`` wrapper changes the
        behaviour being measured.
        """
        seen = []
        original = SingleWindowAttention.call

        def spy(self, inputs, attention_mask=None, training=None,
                pad_to_window=True):
            # The slot map is INSTANCE state (`set_window_slots`), not a call
            # argument -- it must not ride the traced-kwarg channel, or
            # `predict()`/`fit()` raise NotImplementedError (D-015). Read it
            # here exactly where the layer itself reads it.
            window_slots = self._window_slots
            seen.append(
                {
                    "slots_in": int(inputs.shape[1]),
                    "window_slots": (
                        None if window_slots is None else np.asarray(window_slots)
                    ),
                }
            )
            return original(
                self,
                inputs,
                attention_mask=attention_mask,
                training=training,
                pad_to_window=pad_to_window,
            )

        monkeypatch.setattr(SingleWindowAttention, "call", spy)
        return seen

    @staticmethod
    def _run(partition_mode, window_size, seq_len):
        keras.utils.set_random_seed(11)
        layer = WindowAttention(
            dim=32,
            window_size=window_size,
            num_heads=4,
            partition_mode=partition_mode,
            use_relative_position_bias=True,
            dropout_rate=0.0,
        )
        x = np.zeros((2, seq_len, 32), dtype="float32")
        layer(x, training=False)
        return layer

    @pytest.mark.parametrize(
        "partition_mode,window_size,seq_len",
        DEGENERATE_CELLS,
        ids=[f"{m}-ws{w}-N{n}" for (m, w, n) in DEGENERATE_CELLS],
    )
    def test_a_degenerate_window_attends_N_slots_not_window_size_squared(
        self, partition_mode, window_size, seq_len, monkeypatch
    ):
        """Why this can fail if the implementation is wrong: this is the ONLY guard in
        the plan that can. The padded path and the short-circuit agree to a float32
        ulp, so a regression that stopped taking the cheap branch is invisible to every
        value comparison and shows up only as 33x the memory."""
        seen = self._record(monkeypatch)
        self._run(partition_mode, window_size, seq_len)

        assert len(seen) == 1, (
            f"expected exactly one SingleWindowAttention call for a degenerate "
            f"single-window sequence, saw {len(seen)}"
        )
        assert seen[0]["slots_in"] == seq_len, (
            f"WindowAttention(window_size={window_size}, "
            f"partition_mode={partition_mode!r}) on N={seq_len} handed "
            f"SingleWindowAttention a sequence of {seen[0]['slots_in']} slots to "
            f"answer a question about {seq_len} real tokens "
            f"(window_slots={seen[0]['window_slots']}). The degenerate "
            f"single-window short-circuit is not being taken: at window_size=128 "
            f"that inflation is 0.649 GB -> 21.695 GB, i.e. 33x the cost of plain "
            f"global attention."
        )
        assert seen[0]["window_slots"] is not None, (
            "the short-circuit was taken but no window_slots vector was supplied, so "
            "the relative-position bias is being gathered at slot arange(N) -- the "
            "identity map, which is the RIGHT answer only when the grid side equals "
            "the window side"
        )

    @pytest.mark.parametrize(
        "partition_mode,window_size,seq_len",
        NON_DEGENERATE_CELLS,
        ids=[f"{m}-ws{w}-N{n}" for (m, w, n) in NON_DEGENERATE_CELLS],
    )
    def test_the_short_circuit_is_not_taken_when_there_is_nothing_to_save(
        self, partition_mode, window_size, seq_len, monkeypatch
    ):
        """``N >= window_size ** 2`` keeps the ordinary partition path.

        Why this can fail if the implementation is wrong: widening the condition to
        ``N <= window_size ** 2`` costs nothing at run time and would look like a
        harmless generalization, but ``N == window_size ** 2`` is the exact call
        ``SwinTransformerBlock`` makes -- with a rank-3 pairwise mask in
        already-partitioned coordinates that the short-circuit does not accept -- and
        it is the case pinned BITWISE against the pre-restructure layer.
        """
        seen = self._record(monkeypatch)
        self._run(partition_mode, window_size, seq_len)

        assert all(r["slots_in"] == window_size * window_size for r in seen), (
            f"WindowAttention(window_size={window_size}, "
            f"partition_mode={partition_mode!r}) on N={seq_len} did not use the "
            f"ordinary partition path: inner sequence lengths were "
            f"{[r['slots_in'] for r in seen]}, expected every one to be "
            f"window_size**2 = {window_size * window_size}"
        )
        assert all(r["window_slots"] is None for r in seen), (
            "window_slots was supplied on a non-degenerate call; it means 'these are "
            "the only real tokens in the tile', which is false here"
        )

    @pytest.mark.parametrize(
        "window_size,seq_len",
        [(4, 9), (7, 25), (8, 50), (4, 5), (8, 33)],
        ids=lambda v: str(v),
    )
    def test_the_zigzag_slot_map_is_the_tokens_own_position_in_the_scan(
        self, window_size, seq_len, monkeypatch
    ):
        """The short-circuit must attend the tokens at the slots the LAYOUT gives them.

        Why this can fail if the implementation is wrong: a single window attends every
        token to every other token whatever order they are listed in, so a wrong
        permutation is invisible in the attention pattern. It is visible in exactly one
        place -- the relative-position bias, which is gathered by SLOT. The forward and
        inverse zigzag permutations have the same shape and the same values as a set,
        and confusing them is the natural mistake here, so the expected slot vector is
        re-derived below from the SCAN ORDER ITSELF rather than from either array the
        layer holds.

        The scan is boustrophedon over anti-diagonals of the ``S x S`` grid the zigzag
        path folds the sequence into (``S = ceil(sqrt(N))``): anti-diagonals in
        increasing ``s = r + c``, and within one anti-diagonal the row index ascends
        when ``s`` is odd and descends when ``s`` is even.
        """
        side = int(np.ceil(np.sqrt(seq_len)))
        scan = sorted(
            ((r, c) for r in range(side) for c in range(side)),
            key=lambda rc: (
                rc[0] + rc[1],
                rc[0] if (rc[0] + rc[1]) % 2 == 1 else side - 1 - rc[0],
            ),
        )
        # Position in the scan of each row-major grid slot; token `i` occupies grid
        # slot `i` (the sequence is laid out row-major before the permutation).
        position_of = {r * side + c: p for p, (r, c) in enumerate(scan)}
        expected = np.array(
            [position_of[i] for i in range(seq_len)], dtype="int32"
        )

        seen = self._record(monkeypatch)
        self._run("zigzag", window_size, seq_len)

        assert len(seen) == 1 and seen[0]["window_slots"] is not None, (
            "the zigzag short-circuit was not taken, so there is no slot map to "
            "grade -- see "
            "test_a_degenerate_window_attends_N_slots_not_window_size_squared"
        )
        actual = seen[0]["window_slots"]
        assert np.array_equal(actual, expected), (
            f"zigzag window_slots for window_size={window_size}, N={seq_len} are "
            f"{actual.tolist()}, but token i must be told the position it occupies "
            f"in the zigzag SCAN, which is {expected.tolist()}. These two vectors "
            f"are permutations of each other, so nothing about the shapes, the "
            f"weights or the attention PATTERN can tell them apart -- only the "
            f"relative-position bias, which is gathered by slot, moves."
        )


class TestBandPartitionMode:
    """``partition_mode='band'`` — a 1-D SYMMETRIC band, proven by PERTURBATION.

    ``window_size`` is a HALF-WIDTH IN TOKENS here, not a 2-D edge length: query
    ``i`` attends key ``j`` iff ``abs(i - j) <= window_size``. That is the
    semantics every 1-D reference implementation uses (Longformer, Mistral,
    ModernBERT), and upstream ModernBERT derives the half-width from its config as
    ``sliding_window = local_attention // 2`` — so ``local_attention=128`` means
    "64 tokens either side", never a 128x128 tile. See D-003 / D-009
    (``plan-2026-08-25T053412-0f1fa04f``).

    **How the band is measured, and how it is NOT.** Every assertion below
    perturbs ONE token of the input and looks at ONE output row. A key outside
    the band must move that row by EXACTLY ``0.0`` (``np.array_equal``, not
    ``allclose``); a key at the band EDGE must move it by a nonzero amount. The
    mask is never read back and never compared against a recomputed copy of
    itself — a test that rebuilds the same expression it is checking passes
    identically when that expression is wrong, which is the self-referential
    oracle this repo has been burned by (``plans/LESSONS.md``). Both directions
    (``j = i + d`` and ``j = i - d``) are measured separately, because a
    one-sided test cannot tell a symmetric band from a causal one — and the
    idiom this implementation follows,
    ``gemma3_transformer.py:_create_attention_mask``, IS causal.
    """

    DIM, HEADS, BATCH = 32, 4, 2

    @classmethod
    def _layer(cls, window_size, **kwargs):
        keras.utils.set_random_seed(11)
        return create_band_window_attention(
            dim=cls.DIM, window_size=window_size, num_heads=cls.HEADS, **kwargs
        )

    @classmethod
    def _inputs(cls, seq_len, seed=3):
        rng = np.random.default_rng(seed)
        return rng.standard_normal(
            (cls.BATCH, seq_len, cls.DIM)
        ).astype("float32")

    @staticmethod
    def _row_delta(layer, x, query, key, **call_kwargs):
        """max |output[:, query] - output_perturbed[:, query]| after moving token ``key``."""
        base = np.asarray(
            keras.ops.convert_to_numpy(layer(x, training=False, **call_kwargs))
        )
        perturbed_x = x.copy()
        perturbed_x[:, key, :] += 5.0
        moved = np.asarray(
            keras.ops.convert_to_numpy(
                layer(perturbed_x, training=False, **call_kwargs)
            )
        )
        assert np.all(np.isfinite(base)) and np.all(np.isfinite(moved))
        # Non-vacuity: a dead layer emitting a constant would satisfy every
        # "did not move" assertion below no matter what the mask did.
        assert float(np.std(base)) > 1e-6, (
            f"the band layer's output is effectively constant "
            f"(std={float(np.std(base))}) -- every delta measured against it "
            f"would be vacuously zero"
        )
        if np.array_equal(base[:, query, :], moved[:, query, :]):
            return 0.0
        return float(
            np.abs(
                base[:, query, :].astype("float64")
                - moved[:, query, :].astype("float64")
            ).max()
        )

    @pytest.mark.parametrize("window_size,seq_len,query", [(3, 17, 8), (5, 33, 16), (1, 9, 4)])
    def test_a_key_outside_the_band_has_exactly_zero_influence(
        self, window_size, seq_len, query
    ):
        """Out-of-band keys are INERT, in BOTH directions, bit-for-bit.

        Why this can fail if the implementation is wrong: a band that is dead
        (the mask contributes nothing, so the layer is plain full attention),
        shifted by one, or inverted in polarity all produce a finite,
        well-shaped, non-constant output. Only the exact-zero delta at
        ``abs(i - j) > window_size``, paired with the NONZERO delta at
        ``abs(i - j) == window_size`` in the next test, distinguishes them.
        """
        layer = self._layer(window_size)
        x = self._inputs(seq_len)
        for distance in (window_size + 1, window_size + 2, seq_len - 1 - query):
            for sign in (+1, -1):
                key = query + sign * distance
                if not (0 <= key < seq_len) or abs(key - query) <= window_size:
                    continue
                delta = self._row_delta(layer, x, query, key)
                assert delta == 0.0, (
                    f"WindowAttention(partition_mode='band', "
                    f"window_size={window_size}) on (2, {seq_len}, 32): "
                    f"perturbing token {key} moved query row {query} by {delta}, "
                    f"but abs({query} - {key}) = {abs(query - key)} > "
                    f"{window_size}, so that key is OUTSIDE the band and must "
                    f"have EXACTLY zero influence. A nonzero value here means "
                    f"the band is dead, shifted, or inverted -- do NOT relax "
                    f"this to allclose."
                )

    @pytest.mark.parametrize("window_size,seq_len,query", [(3, 17, 8), (5, 33, 16), (1, 9, 4)])
    def test_a_key_at_the_band_edge_moves_the_output_in_both_directions(
        self, window_size, seq_len, query
    ):
        """The band EDGE is inclusive and SYMMETRIC — this is the non-vacuity half.

        Why this can fail if the implementation is wrong: a band shifted by one
        (``<`` instead of ``<=``) leaves the previous test green and reds here on
        the edge key. A CAUSAL band — what
        ``gemma3_transformer.py:_create_attention_mask`` builds, and what a
        copy-paste of it would produce — reds here on the ``+`` direction only,
        which is why the two signs are asserted separately rather than as an
        ``any``.
        """
        layer = self._layer(window_size)
        x = self._inputs(seq_len)
        for sign, label in ((+1, "future"), (-1, "past")):
            key = query + sign * window_size
            assert 0 <= key < seq_len, "fixture drift: edge key fell off the sequence"
            delta = self._row_delta(layer, x, query, key)
            assert delta > 1e-5, (
                f"WindowAttention(partition_mode='band', "
                f"window_size={window_size}) on (2, {seq_len}, 32): perturbing "
                f"token {key} moved query row {query} by only {delta}, but "
                f"abs({query} - {key}) = {window_size} is EXACTLY the half-width "
                f"and the band is INCLUSIVE and SYMMETRIC, so the {label} edge "
                f"key must influence the query. A zero here means the band is "
                f"off by one, or causal."
            )

    def test_a_band_wider_than_the_sequence_is_full_attention(self):
        """``window_size >= N - 1`` covers everything: no token is inert.

        Why this can fail if the implementation is wrong: an implementation that
        still folded the sequence into a grid, or padded up to
        ``window_size ** 2`` slots, would either raise or change which tokens
        reach a query. Here every one of them must.
        """
        seq_len = 13
        layer = self._layer(100)
        x = self._inputs(seq_len)
        for key in range(seq_len):
            delta = self._row_delta(layer, x, 6, key)
            assert delta > 1e-5, (
                f"a band of half-width 100 over {seq_len} tokens covers the whole "
                f"sequence, so token {key} must influence query 6; it moved it by "
                f"{delta}"
            )

    def test_the_caller_mask_composes_with_the_band_instead_of_replacing_it(self):
        """A caller ``(B, N)`` key mask AND-s with the band; neither wins outright.

        Why this can fail if the implementation is wrong: if the band REPLACED
        the caller's mask, masking an in-band key would change nothing (first
        assertion). If the caller's mask replaced the BAND, the out-of-band key
        would suddenly become live (second assertion). Both are silent: same
        shape, finite, non-constant.
        """
        window_size, seq_len, query = 3, 17, 8
        layer = self._layer(window_size)
        x = self._inputs(seq_len)
        in_band_key, out_of_band_key = query + 2, query + 6

        mask = np.ones((self.BATCH, seq_len), dtype="int32")
        mask[:, in_band_key] = 0
        with_mask = np.asarray(
            keras.ops.convert_to_numpy(
                layer(x, attention_mask=keras.ops.convert_to_tensor(mask), training=False)
            )
        )
        without = np.asarray(keras.ops.convert_to_numpy(layer(x, training=False)))
        moved = float(
            np.abs(
                with_mask[:, query, :].astype("float64")
                - without[:, query, :].astype("float64")
            ).max()
        )
        assert moved > 1e-5, (
            f"masking in-band key {in_band_key} did not change query row {query} "
            f"(delta {moved}) -- the band REPLACED the caller's mask instead of "
            f"composing with it, so a padded key is still being attended"
        )

        # ...and the band still holds under a caller mask that keeps everything
        # except one far key: the far key was already inert, so nothing moves.
        far_mask = np.ones((self.BATCH, seq_len), dtype="int32")
        far_mask[:, out_of_band_key] = 0
        far = np.asarray(
            keras.ops.convert_to_numpy(
                layer(
                    x,
                    attention_mask=keras.ops.convert_to_tensor(far_mask),
                    training=False,
                )
            )
        )
        assert np.array_equal(far[:, query, :], without[:, query, :]), (
            f"masking key {out_of_band_key}, which is already OUTSIDE the band "
            f"(abs({query} - {out_of_band_key}) = {abs(query - out_of_band_key)} "
            f"> {window_size}), changed query row {query}. The caller's mask "
            f"replaced the band instead of composing with it."
        )

    def test_a_pairwise_caller_mask_composes_with_the_band(self):
        """The rank-3 ``(B, N, N)`` branch is AND-ed with the band too.

        Why this can fail if the implementation is wrong: a band that ignored a
        rank-3 mask (forwarding it verbatim, as ``_call_grid`` does for its own
        reasons) would leave the first delta at zero.
        """
        window_size, seq_len, query = 3, 17, 8
        layer = self._layer(window_size)
        x = self._inputs(seq_len)
        pairwise = np.ones((self.BATCH, seq_len, seq_len), dtype="int32")
        pairwise[:, query, query + 2] = 0
        without = np.asarray(keras.ops.convert_to_numpy(layer(x, training=False)))
        with_mask = np.asarray(
            keras.ops.convert_to_numpy(
                layer(
                    x,
                    attention_mask=keras.ops.convert_to_tensor(pairwise),
                    training=False,
                )
            )
        )
        assert (
            float(
                np.abs(
                    with_mask[:, query, :].astype("float64")
                    - without[:, query, :].astype("float64")
                ).max()
            )
            > 1e-5
        ), "a rank-3 pairwise mask did not compose with the band"
        # A query row the pairwise mask left untouched must not move at all.
        assert np.array_equal(with_mask[:, 0, :], without[:, 0, :]), (
            "the pairwise mask leaked into a query row it did not name"
        )

    def test_band_never_folds_the_sequence_into_a_square_grid(self):
        """Adjacency is by TOKEN INDEX, not by a synthetic ``ceil(sqrt(N))`` grid.

        Why this can fail if the implementation is wrong: this is the whole point
        of the mode. ``'grid'`` at ``N=17`` lays tokens into a 5x5 grid, so token
        0 and token 5 are vertical neighbours and attend together while token 0
        and token 4 may not. A band must show the opposite: influence decays with
        ``abs(i - j)`` alone. Measured at ``window_size=2, N=17, query=0``: keys
        0,1,2 live, keys 3..16 (INCLUDING 5, the grid's vertical neighbour) dead.
        """
        layer = self._layer(2)
        x = self._inputs(17)
        live = [k for k in range(17) if self._row_delta(layer, x, 0, k) > 1e-5]
        assert live == [0, 1, 2], (
            f"query 0 of a half-width-2 band over 17 tokens is influenced by "
            f"{live}, expected exactly [0, 1, 2]. Anything else -- notably key 5, "
            f"which is token 0's VERTICAL neighbour in the ceil(sqrt(17)) = 5 "
            f"grid 'grid' mode would build -- means the sequence is still being "
            f"folded into a square."
        )

    def test_partition_mode_is_validated_and_names_all_three_values(self):
        """An unknown mode raises, and the message lists what is valid."""
        with pytest.raises(ValueError) as excinfo:
            WindowAttention(dim=32, window_size=4, num_heads=4, partition_mode="sliding")
        message = str(excinfo.value)
        for valid in ("grid", "zigzag", "band"):
            assert valid in message, (
                f"the partition_mode rejection message does not name {valid!r}: "
                f"{message}"
            )

    def test_band_refuses_the_relative_position_bias_rather_than_forcing_it_off(self):
        """An explicit ``use_relative_position_bias=True`` RAISES under ``'band'``.

        Why this can fail if the implementation is wrong: forcing it off silently
        would leave ``get_config()['use_relative_position_bias'] is True`` on a
        layer that has no such bias, and a caller who asked for it would never
        learn it did not arrive. D-009 rules this a refusal.
        """
        with pytest.raises(ValueError, match="use_relative_position_bias"):
            WindowAttention(
                dim=32,
                window_size=4,
                num_heads=4,
                partition_mode="band",
                use_relative_position_bias=True,
            )
        # The wrapper's DEFAULT-off is what makes the factory path usable, and it
        # is a default, not a silent override: an explicit True still raises.
        assert (
            create_band_window_attention(
                dim=32, window_size=4, num_heads=4
            ).use_relative_position_bias
            is False
        )
        with pytest.raises(ValueError, match="use_relative_position_bias"):
            create_band_window_attention(
                dim=32, window_size=4, num_heads=4, use_relative_position_bias=True
            )

    def test_the_two_no_padding_spellings_are_mutually_exclusive(self):
        """A slot map and ``pad_to_window=False`` both mean "do not pad".

        Why this can fail if the implementation is wrong: two independent
        spellings of one instruction is the drift D-005 exists to prevent. The
        check is an explicit raise, not a precedence rule.
        """
        inner = SingleWindowAttention(
            dim=32, window_size=4, num_heads=4, use_relative_position_bias=False
        )
        x = np.zeros((2, 5, 32), dtype="float32")
        inner.set_window_slots(np.arange(5, dtype="int32"))
        with pytest.raises(ValueError, match="pad_to_window"):
            inner(x, pad_to_window=False)
        inner.set_window_slots(None)

    def test_pad_to_window_false_refuses_a_relative_position_bias(self):
        """The inner layer refuses the combination too, not only the outer one."""
        inner = SingleWindowAttention(
            dim=32, window_size=4, num_heads=4, use_relative_position_bias=True
        )
        x = np.zeros((2, 5, 32), dtype="float32")
        with pytest.raises(ValueError, match="use_relative_position_bias"):
            inner(x, pad_to_window=False)


class TestAllThreePartitionModes:
    """Serialization and mixed-precision coverage, for every partition mode.

    ``'band'`` is a third VALUE of an existing flag, not a new class, so it must
    round-trip through the SAME ``get_config`` / ``keras.saving`` surface the
    other two already use — and the check is on the reloaded model's OUTPUT, not
    on its config dict, because a config that round-trips into a layer that
    computes something else is exactly the failure a config comparison cannot
    see.
    """

    MODES = [("grid", True), ("zigzag", False), ("band", False)]
    IDS = ["grid", "zigzag", "band"]

    @staticmethod
    def _model(partition_mode, use_rpb, seq_len=16, dim=32, window_size=2):
        keras.utils.set_random_seed(5)
        inputs = keras.Input(batch_shape=(2, seq_len, dim))
        layer = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=4,
            partition_mode=partition_mode,
            use_relative_position_bias=use_rpb,
        )
        return keras.Model(inputs, layer(inputs))

    @pytest.mark.parametrize("partition_mode,use_rpb", MODES, ids=IDS)
    def test_a_saved_model_reloads_and_computes_the_same_values(
        self, partition_mode, use_rpb
    ):
        """Full ``keras.saving`` round-trip, compared by VALUE and bit-for-bit."""
        model = self._model(partition_mode, use_rpb)
        x = np.random.default_rng(0).standard_normal((2, 16, 32)).astype("float32")
        before = np.asarray(keras.ops.convert_to_numpy(model(x, training=False)))
        assert float(np.std(before)) > 1e-6, "vacuous: the model output is constant"
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)
        after = np.asarray(keras.ops.convert_to_numpy(reloaded(x, training=False)))
        assert np.array_equal(before, after), (
            f"partition_mode={partition_mode!r} did not survive a keras.saving "
            f"round-trip: max |delta| "
            f"{float(np.abs(before.astype('float64') - after.astype('float64')).max())}"
        )

    @pytest.mark.parametrize("partition_mode,use_rpb", MODES, ids=IDS)
    def test_get_config_carries_the_partition_mode(self, partition_mode, use_rpb):
        """``from_config(get_config())`` reproduces the flag, and the OUTPUT."""
        layer = WindowAttention(
            dim=32,
            window_size=2,
            num_heads=4,
            partition_mode=partition_mode,
            use_relative_position_bias=use_rpb,
        )
        config = layer.get_config()
        assert config["partition_mode"] == partition_mode
        assert WindowAttention.from_config(config).partition_mode == partition_mode

    @pytest.mark.parametrize("partition_mode,use_rpb", MODES, ids=IDS)
    def test_mixed_float16_stays_finite(self, partition_mode, use_rpb):
        """No NaN under ``mixed_float16``, with no mask and with an all-ones mask.

        Why this can fail if the implementation is wrong: a hand-rolled additive
        ``-1e9`` sentinel is ``-inf`` in float16 and ``0 * -inf = NaN``; this repo
        has a recorded 10-site family of exactly that. The band routes through
        ``common.apply_attention_mask`` for this reason, and this is the guard
        that would see a regression to the arithmetic form.
        """
        previous = keras.mixed_precision.global_policy()
        keras.mixed_precision.set_global_policy("mixed_float16")
        try:
            keras.utils.set_random_seed(5)
            layer = WindowAttention(
                dim=32,
                window_size=2,
                num_heads=4,
                partition_mode=partition_mode,
                use_relative_position_bias=use_rpb,
            )
            x = np.random.default_rng(1).standard_normal((2, 16, 32)).astype("float32")
            plain = np.asarray(keras.ops.convert_to_numpy(layer(x, training=False)))
            ones = keras.ops.convert_to_tensor(np.ones((2, 16), dtype="int32"))
            masked = np.asarray(
                keras.ops.convert_to_numpy(
                    layer(x, attention_mask=ones, training=False)
                )
            )
        finally:
            keras.mixed_precision.set_global_policy(previous)
        for label, out in (("no mask", plain), ("all-ones mask", masked)):
            non_finite = int(np.sum(~np.isfinite(out)))
            assert non_finite == 0, (
                f"partition_mode={partition_mode!r} under mixed_float16 with "
                f"{label}: {non_finite}/{out.size} non-finite values"
            )
            assert float(np.std(out.astype("float64"))) > 1e-6, (
                f"partition_mode={partition_mode!r} under mixed_float16 with "
                f"{label}: the output is constant, so the finiteness check above "
                f"is vacuous"
            )


class TestBandFactoryRegistration:
    """``'window_band'`` is reachable through the public attention factory."""

    def test_the_registry_key_builds_a_band_layer(self):
        from dl_techniques.layers.attention.factory import (
            create_attention_layer,
            ATTENTION_REGISTRY,
        )

        assert "window_band" in ATTENTION_REGISTRY
        layer = create_attention_layer(
            "window_band", dim=32, window_size=3, num_heads=4
        )
        assert isinstance(layer, WindowAttention)
        assert layer.partition_mode == "band"
        assert layer.use_relative_position_bias is False
        x = np.random.default_rng(2).standard_normal((2, 17, 32)).astype("float32")
        assert layer(x).shape == (2, 17, 32)

    def test_the_registry_cost_claim_is_not_the_inverted_kind(self):
        """The ``complexity`` field must say ``O(N^2)``, not promise ``O(N*W)``.

        Why this can fail if the implementation is wrong: D-027 records TEN
        previously-inverted cost claims about this exact layer. A dense N x N
        banded mask is the same order as full attention; a registry entry
        advertising a linear-in-N saving would be the eleventh, and prose is the
        one thing no numeric test catches. This pins the claim's SHAPE, and the
        entry also carries the command that measures it.
        """
        from dl_techniques.layers.attention.factory import ATTENTION_REGISTRY

        entry = ATTENTION_REGISTRY["window_band"]
        complexity = entry["complexity"]
        assert "O(N^2)" in complexity, (
            f"the 'window_band' complexity field no longer states O(N^2): {complexity}"
        )
        assert "ru_maxrss" in complexity, (
            "the 'window_band' complexity field no longer carries the command that "
            "measures it (I-6): a cost claim without its instrument is the D-027 "
            "failure mode"
        )
        assert "HALF-WIDTH IN TOKENS" in entry["description"], (
            "the 'window_band' description no longer states that window_size is a "
            "half-width in tokens -- that is the single fact a caller coming from "
            "the 'window' key will get wrong"
        )


class TestTheShortCircuitIsCorrectAtWindowSize128:
    """The one regime the external-golden harness structurally cannot grade.

    ``test_window_attention_restructure_is_inert.py`` compares against the
    PRE-restructure code, and that code cannot run ``window_size=128`` cheaply
    enough to produce a reference (it inflates N real tokens to 16,384 slots and
    peaked at 17.69 GB). So the plan's own record listed "correctness of the
    short-circuit at ``ws=128``" under **Not Verified**, backed only by a MEMORY
    measurement. This class supplies the missing arm with a DIFFERENT oracle: a
    from-scratch float64 dense reference written against the Swin definition, not
    against the layer's own index code.

    Why this can fail if the implementation is wrong: the short-circuit attends the
    N real tokens directly, so it must gather the relative-position bias at each
    token's TILE SLOT ``(i // ceil(sqrt(N))) * ws + (i % ceil(sqrt(N)))`` -- which
    is NOT ``i`` whenever the grid side is smaller than ``ws``, i.e. always in this
    regime. RED-PROVED: replacing the slot map with the identity ``arange(N)`` --
    the "just call dense attention" shortcut the D-007 anchor forbids -- moves the
    ``rpb=True`` cells to 1.618e-02 / 7.614e-03 / 5.080e-03 (N = 17 / 100 / 300)
    against a noise floor of 2.7e-07, five orders of magnitude, while leaving the
    ``rpb=False`` cells at 2.086e-07 exactly as they should be.
    """

    # TF32 OFF for this class. `ATOL` below is 5e-7 against a float64 reference,
    # and on Ampere+ the TF32 tensor-core matmul carries ~1e-3 relative precision:
    # measured, these six cells fail on an RTX 4070 with TF32 on and pass with
    # `NVIDIA_TF32_OVERRIDE=0`. The bound must NOT be widened to accommodate that
    # -- see the note on `ATOL` -- so the precision regime is fixed instead, the
    # same opt-in `test_linear_attention.py` uses. Scoped to this class rather than
    # the module so the other ~250 tests in this file keep running in the default
    # regime.
    pytestmark = pytest.mark.usefixtures("tf32_disabled")

    WINDOW_SIZE = 128
    #: float32 reduction noise against a float64 reference. MEASURED worst case
    #: over all six cells: 2.682e-07. Do NOT widen it -- the RED injection above
    #: lands five orders of magnitude higher, so any real defect is unmissable.
    ATOL = 5e-7

    @staticmethod
    def _dense_reference(layer, x_f64, use_relative_position_bias):
        """Dense attention over the N real tokens, in float64, from the weights."""
        import math

        inner = layer.attention
        assert inner.q_norm is None, "this reference assumes no q/k normalization"
        batch, seq_len, _ = x_f64.shape
        heads, head_dim = inner.num_heads, inner.head_dim

        kernel, bias = [np.asarray(w, dtype=np.float64)
                        for w in inner.qkv.get_weights()]
        qkv = x_f64 @ kernel + bias
        qkv = qkv.reshape(batch, seq_len, 3, heads, head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * inner.scale) @ k.transpose(0, 1, 3, 2)

        if use_relative_position_bias:
            ws = layer.window_size
            grid_side = int(math.ceil(math.sqrt(seq_len)))
            tokens = np.arange(seq_len)
            slots = (tokens // grid_side) * ws + (tokens % grid_side)
            # The Swin reference form: stack the two coordinate axes, difference
            # them pairwise, shift to non-negative, encode row-major.
            coords = np.stack([slots // ws, slots % ws])
            rel = (coords[:, :, None] - coords[:, None, :]).transpose(1, 2, 0)
            rel = rel.astype(np.int64)
            rel[:, :, 0] += ws - 1
            rel[:, :, 1] += ws - 1
            rel[:, :, 0] *= 2 * ws - 1
            index = rel.sum(-1)
            table = np.asarray(
                inner.relative_position_bias_table, dtype=np.float64
            )
            attn = attn + table[index].transpose(2, 0, 1)[None]

        attn = np.clip(attn, -30.0, 30.0)
        attn = attn - attn.max(-1, keepdims=True)
        weights = np.exp(attn)
        weights = weights / weights.sum(-1, keepdims=True)

        out = (weights @ v).transpose(0, 2, 1, 3)
        out = out.reshape(batch, seq_len, heads * head_dim)
        proj_kernel, proj_bias = [np.asarray(w, dtype=np.float64)
                                  for w in inner.proj.get_weights()]
        return out @ proj_kernel + proj_bias

    @pytest.mark.parametrize("use_relative_position_bias", [True, False])
    @pytest.mark.parametrize("seq_len", [17, 100, 300])
    def test_matches_an_independent_dense_reference(
        self, seq_len, use_relative_position_bias
    ):
        keras.utils.set_random_seed(11)
        layer = WindowAttention(
            dim=64,
            window_size=self.WINDOW_SIZE,
            num_heads=4,
            partition_mode="grid",
            use_relative_position_bias=use_relative_position_bias,
            dropout_rate=0.0,
        )
        x = np.random.default_rng(seq_len).normal(
            size=(2, seq_len, 64)
        ).astype("float32")

        got = np.asarray(layer(x, training=False))
        reference = self._dense_reference(
            layer, x.astype(np.float64), use_relative_position_bias
        ).astype(np.float32)

        delta = float(np.abs(got - reference).max())
        assert delta <= self.ATOL, (
            f"window_size={self.WINDOW_SIZE}, N={seq_len}, "
            f"rpb={use_relative_position_bias}: the short-circuit disagrees with an "
            f"independent float64 dense reference by {delta:g} > {self.ATOL:g}. "
            "WIDEN NOTHING -- the slot map or the bias gather is wrong."
        )

    def test_the_reference_actually_sees_the_bias(self):
        """Anti-vacuity: the two rpb settings must not produce the same answer.

        Why this can fail if the implementation is wrong: if the bias were dropped
        (the failure mode the slot map exists to prevent) both cells above would
        agree with the same reference and the parametrization would be graded
        twice over the same computation.
        """
        outputs = []
        for use_relative_position_bias in (True, False):
            keras.utils.set_random_seed(11)
            layer = WindowAttention(
                dim=64,
                window_size=self.WINDOW_SIZE,
                num_heads=4,
                partition_mode="grid",
                use_relative_position_bias=use_relative_position_bias,
                dropout_rate=0.0,
            )
            x = np.random.default_rng(17).normal(size=(2, 17, 64)).astype("float32")
            outputs.append(np.asarray(layer(x, training=False)))

        spread = float(np.abs(outputs[0] - outputs[1]).max())
        assert spread > 1e-3, (
            "the relative-position bias changes the output by only "
            f"{spread:g} at window_size=128 -- it is not reaching the scores, so "
            "the parametrized cells above are grading one computation twice"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])