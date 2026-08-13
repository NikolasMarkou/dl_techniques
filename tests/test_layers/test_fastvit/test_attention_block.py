"""
Test suite for FastVitAttentionBlock (FastViT / MobileCLIP2 MCi).

Covers initialization + stored config, constructor validation, forward shape over
several spatial sizes (square AND non-square) and channel widths,
`compute_output_shape` pre- and post-build, training-mode behaviour, gradient flow,
a `.keras` VALUE round trip, and the six mandated behavioural pins:

1. `test_non_square_input_roundtrips_orientation` — THE reason this block exists as
   its own step. The block owns a rank-4 -> rank-3 flatten and the reshape back;
   an orientation error there is invisible to every shape assertion whenever
   `H == W`, and this repo has shipped that defect class twice before.
2. `test_attention_receives_rank3`
3. `test_attention_has_no_bias_weights` (pins the recorded deviation X-2)
4. `test_norm_epsilon_is_reference_value`
5. `test_layer_scale_constraints_are_none`
6. `test_graph_mode_matches_eager`
"""

import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.layers.fastvit.attention_block import FastVitAttentionBlock


class TestFastVitAttentionBlock:
    """Comprehensive test suite for the FastViT attention block."""

    # ------------------------------------------------------------------
    # fixtures
    # ------------------------------------------------------------------

    @pytest.fixture
    def basic_config(self):
        """Standard configuration used across the suite."""
        return {'dim': 64, 'head_dim': 32, 'mlp_ratio': 2.0}

    @pytest.fixture
    def sample_input(self):
        """Deterministic NON-SQUARE rank-4 input, (B, H, W, C) = (2, 8, 4, 64)."""
        rng = np.random.default_rng(20260813)
        return rng.normal(size=(2, 8, 4, 64)).astype('float32')

    # ------------------------------------------------------------------
    # initialization / config
    # ------------------------------------------------------------------

    def test_initialization_stores_config(self, basic_config):
        block = FastVitAttentionBlock(**basic_config)

        assert block.dim == 64
        assert block.head_dim == 32
        assert block.num_heads == 2
        assert block.mlp_ratio == 2.0
        assert block.hidden_dim == 128
        assert block.normalization_type == 'batch_norm'
        assert block.dropout_rate == 0.0
        assert block.attention_dropout_rate == 0.0
        assert block.drop_path_rate == 0.0
        assert block.layer_scale_init_value == pytest.approx(1e-5)
        assert not block.built

    def test_sublayers_created_in_init(self, basic_config):
        """Every sub-layer must exist before build (H-1)."""
        block = FastVitAttentionBlock(**basic_config)
        for attribute in (
                'norm', 'attn', 'layer_scale_1', 'layer_scale_2',
                'drop_path_1', 'drop_path_2', 'mlp',
        ):
            assert getattr(block, attribute) is not None, attribute

    def test_layer_scale_none_disables_both_scales(self, basic_config):
        block = FastVitAttentionBlock(**{**basic_config,
                                         'layer_scale_init_value': None})
        assert block.layer_scale_1 is None
        assert block.layer_scale_2 is None
        y = block(np.zeros((2, 4, 4, 64), dtype='float32'), training=False)
        assert tuple(y.shape) == (2, 4, 4, 64)

    def test_get_config_round_trips_through_from_config(self, basic_config):
        block = FastVitAttentionBlock(**basic_config, drop_path_rate=0.1,
                                      normalization_type='layer_norm')
        config = block.get_config()
        clone = FastVitAttentionBlock.from_config(config)

        for key in ('dim', 'head_dim', 'mlp_ratio', 'normalization_type',
                    'dropout_rate', 'attention_dropout_rate', 'drop_path_rate',
                    'layer_scale_init_value'):
            assert getattr(clone, key) == getattr(block, key), key

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("kwargs,match", [
        ({'dim': 0}, "dim must be positive"),
        ({'dim': -8}, "dim must be positive"),
        ({'dim': 64, 'head_dim': 0}, "head_dim must be positive"),
        ({'dim': 64, 'head_dim': 48}, "dim must be divisible by head_dim"),
        ({'dim': 64, 'mlp_ratio': 0.0}, "mlp_ratio must be positive"),
        ({'dim': 64, 'mlp_ratio': -1.0}, "mlp_ratio must be positive"),
        ({'dim': 64, 'dropout_rate': 1.0}, r"dropout_rate must be in \[0, 1\)"),
        ({'dim': 64, 'attention_dropout_rate': 1.5},
         r"attention_dropout_rate must be in \[0, 1\)"),
        ({'dim': 64, 'drop_path_rate': 1.0},
         r"drop_path_rate must be in \[0, 1\)"),
        ({'dim': 64, 'normalization_type': 7},
         "normalization_type must be a string"),
        ({'dim': 64, 'layer_scale_init_value': 'small'},
         "layer_scale_init_value must be a real number or None"),
    ])
    def test_invalid_config_raises(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            FastVitAttentionBlock(**kwargs)

    def test_divisibility_error_names_both_numbers(self):
        """The message must name dim AND head_dim, not just fail."""
        with pytest.raises(ValueError) as excinfo:
            FastVitAttentionBlock(dim=64, head_dim=48)
        message = str(excinfo.value)
        assert 'dim=64' in message
        assert 'head_dim=48' in message

    def test_build_rejects_wrong_rank(self, basic_config):
        block = FastVitAttentionBlock(**basic_config)
        with pytest.raises(ValueError, match="rank-4"):
            block.build((None, 16, 64))

    def test_build_rejects_wrong_channel_count(self, basic_config):
        block = FastVitAttentionBlock(**basic_config)
        with pytest.raises(ValueError, match="must equal dim=64"):
            block.build((None, 8, 8, 32))

    # ------------------------------------------------------------------
    # forward / shapes
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("shape,dim,head_dim", [
        ((2, 8, 8, 64), 64, 32),
        ((2, 8, 4, 64), 64, 32),
        ((1, 4, 16, 32), 32, 32),
        ((3, 5, 7, 96), 96, 32),
    ])
    def test_forward_shape(self, shape, dim, head_dim):
        block = FastVitAttentionBlock(dim=dim, head_dim=head_dim, mlp_ratio=2.0)
        rng = np.random.default_rng(7)
        x = rng.normal(size=shape).astype('float32')
        y = block(x, training=False)
        assert tuple(y.shape) == shape
        assert np.all(np.isfinite(ops.convert_to_numpy(y)))

    def test_compute_output_shape_pre_and_post_build(self, basic_config):
        block = FastVitAttentionBlock(**basic_config)
        # pre-build
        assert block.compute_output_shape((None, 8, 4, 64)) == (None, 8, 4, 64)
        block.build((None, 8, 4, 64))
        # post-build
        assert block.compute_output_shape((2, 8, 4, 64)) == (2, 8, 4, 64)

    def test_training_true_and_false_both_run(self, basic_config, sample_input):
        block = FastVitAttentionBlock(**basic_config, drop_path_rate=0.2,
                                      dropout_rate=0.1)
        y_train = block(sample_input, training=True)
        y_eval = block(sample_input, training=False)
        assert tuple(y_train.shape) == sample_input.shape
        assert tuple(y_eval.shape) == sample_input.shape
        assert np.all(np.isfinite(ops.convert_to_numpy(y_eval)))

    def test_training_false_is_deterministic(self, basic_config, sample_input):
        block = FastVitAttentionBlock(**basic_config, drop_path_rate=0.5,
                                      dropout_rate=0.5)
        first = ops.convert_to_numpy(block(sample_input, training=False))
        second = ops.convert_to_numpy(block(sample_input, training=False))
        np.testing.assert_allclose(first, second, atol=1e-6, rtol=0)

    @pytest.mark.parametrize("normalization_type", ['batch_norm', 'layer_norm'])
    def test_both_reference_norm_types_run(self, normalization_type):
        block = FastVitAttentionBlock(dim=64, head_dim=32, mlp_ratio=2.0,
                                      normalization_type=normalization_type)
        x = np.random.default_rng(3).normal(size=(2, 4, 4, 64)).astype('float32')
        y = block(x, training=False)
        assert tuple(y.shape) == (2, 4, 4, 64)

    # ------------------------------------------------------------------
    # gradients
    # ------------------------------------------------------------------

    def test_gradients_reach_every_trainable_weight(self, basic_config,
                                                    sample_input):
        block = FastVitAttentionBlock(**basic_config)
        x = tf.constant(sample_input)
        with tf.GradientTape() as tape:
            y = block(x, training=True)
            loss = ops.mean(ops.square(y))
        grads = tape.gradient(loss, block.trainable_variables)

        assert len(block.trainable_variables) > 0
        missing = [
            v.path for v, g in zip(block.trainable_variables, grads)
            if g is None
        ]
        assert not missing, f"no gradient reached: {missing}"

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    def test_keras_round_trip_matches_by_value(self, basic_config, sample_input):
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = FastVitAttentionBlock(**basic_config,
                                        drop_path_rate=0.1)(inputs)
        model = keras.Model(inputs, outputs)
        before = model(sample_input, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'attention_block.keras')
            model.save(path)
            restored = keras.models.load_model(path)
            after = restored(sample_input, training=False)

        np.testing.assert_allclose(
            ops.convert_to_numpy(before),
            ops.convert_to_numpy(after),
            atol=1e-6, rtol=0,
        )

    # ==================================================================
    # PIN 1 — flatten/reshape ORIENTATION (the step's reason for existing)
    # ==================================================================

    def test_non_square_input_roundtrips_orientation(self):
        """PIN 1: the reshape back must be the EXACT inverse of the flatten.

        Uses a NON-SQUARE ``(B, 8, 4, C)`` input, because a square input cannot
        distinguish ``(B, H, W, C)`` from ``(B, W, H, C)``. Two arms:

        A. The block's output equals an oracle in which the H-major flatten and
           its inverse are written HERE, independently of the implementation.
        B. The same oracle computed with a W-major (transposed) flatten and an
           H-major reshape back — a shape-legal orientation defect — gives a
           DIFFERENT result. Without arm B, arm A could be satisfied by an
           orientation the test simply cannot see.
        """
        batch, height, width, dim = 2, 8, 4, 64
        block = FastVitAttentionBlock(
            dim=dim, head_dim=32, mlp_ratio=2.0,
            layer_scale_init_value=1.0,  # residual branches must DOMINATE
        )
        rng = np.random.default_rng(4242)
        x = rng.normal(size=(batch, height, width, dim)).astype('float32')

        y = ops.convert_to_numpy(block(x, training=False))
        assert y.shape == (batch, height, width, dim)

        normed = block.norm(x, training=False)

        # --- arm A: independent H-major oracle --------------------------
        sequence = ops.reshape(normed, (batch, height * width, dim))
        attended = ops.reshape(block.attn(sequence, training=False),
                               (batch, height, width, dim))
        mid = ops.add(x, block.layer_scale_1(attended, training=False))
        expected = ops.convert_to_numpy(
            ops.add(mid, block.layer_scale_2(block.mlp(mid, training=False),
                                             training=False))
        )
        np.testing.assert_allclose(y, expected, atol=1e-5, rtol=0)

        # --- arm B: the W-major orientation defect must be DISTINGUISHABLE
        transposed = ops.transpose(normed, (0, 2, 1, 3))
        bad_sequence = ops.reshape(transposed, (batch, width * height, dim))
        bad_attended = ops.reshape(block.attn(bad_sequence, training=False),
                                   (batch, height, width, dim))
        bad_mid = ops.add(x, block.layer_scale_1(bad_attended, training=False))
        broken = ops.convert_to_numpy(
            ops.add(bad_mid,
                    block.layer_scale_2(block.mlp(bad_mid, training=False),
                                        training=False))
        )
        assert not np.allclose(expected, broken, atol=1e-4), (
            "the W-major orientation defect produces the SAME output as the "
            "H-major one, so this pin cannot see a transposed reshape"
        )

    def test_orientation_preserves_spatial_alignment(self):
        """PIN 1 (structural arm): a W-constant input must stay W-constant.

        With the second residual branch switched off (``layer_scale_2`` gamma set
        to zero) the block reduces to ``x + gamma1 * ATTN(norm(x))``. Softmax
        attention is permutation-equivariant, so an input that is constant along
        W has an attention output that is constant along W too — but ONLY if the
        reshape back is the exact inverse of the flatten. Under a W-major flatten
        followed by an H-major reshape, token ``i`` of the sequence lands at
        spatial position ``(i // W, i % W)`` while carrying the value of row
        ``i % H``, so variation appears along W.
        """
        batch, height, width, dim = 2, 8, 4, 64
        block = FastVitAttentionBlock(dim=dim, head_dim=32, mlp_ratio=2.0,
                                      layer_scale_init_value=1.0)
        block.build((batch, height, width, dim))
        # Kill the ConvMlp branch: its depthwise 7x7 'same' convolution pads along
        # W and would break W-constancy for reasons unrelated to orientation.
        block.layer_scale_2.gamma.assign(
            np.zeros(block.layer_scale_2.gamma.shape, dtype='float32'))

        rng = np.random.default_rng(99)
        # Varies along H and along C, CONSTANT along W.
        rows = rng.normal(size=(batch, height, 1, dim)).astype('float32')
        x = np.repeat(rows, width, axis=2)

        y = ops.convert_to_numpy(block(x, training=False))
        spread_along_w = np.max(np.abs(y - y[:, :, :1, :]))
        assert spread_along_w < 1e-4, (
            f"a W-constant input produced a W-varying output (max deviation "
            f"{spread_along_w}); the flatten/reshape is not orientation-preserving"
        )
        # Control: the input really does vary along H, so W-constancy is not
        # trivially satisfied by a constant tensor.
        assert np.max(np.abs(y - y[:, :1, :, :])) > 1e-3

    # ==================================================================
    # PIN 2 — the attention sub-layer sees rank-3
    # ==================================================================

    def test_attention_receives_rank3(self, basic_config):
        """PIN 2: the attention sub-layer is built on the FLATTENED shape.

        Its projection kernels are ``(dim, dim*3)`` and ``(dim, dim)``; that is
        only reachable through a rank-3 build, because the shared layer rejects a
        rank-4 input outright.
        """
        block = FastVitAttentionBlock(**basic_config)
        block.build((None, 8, 4, 64))

        kernels = {w.path.split('/')[-2]: tuple(w.shape)
                   for w in block.attn.weights}
        assert kernels['qkv'] == (64, 64 * 3), kernels
        assert kernels['proj'] == (64, 64), kernels

        # And the 4-D hand-off really is impossible (MEASURED, F-6 P-1).
        fresh = FastVitAttentionBlock(**basic_config)
        with pytest.raises(ValueError):
            fresh.attn.build((None, 8, 4, 64))

    # ==================================================================
    # PIN 3 — recorded deviation X-2: no bias anywhere in attention
    # ==================================================================

    def test_attention_has_no_bias_weights(self, basic_config):
        """PIN 3: the attention sub-layer has EXACTLY 2 weights, both kernels.

        This pins recorded deviation **X-2**. timm's ``Attention`` is
        ``qkv_bias=False`` (matched) but its output projection is a plain
        ``nn.Linear(dim, dim)``, i.e. BIASED (not matched). The repo's shared
        ``MultiHeadAttention`` has a single ``use_bias`` flag governing both
        projections, so the reference's asymmetry is not expressible without
        editing a shared layer. ``use_bias=False`` costs one missing bias vector
        of length ``dim``; ``use_bias=True`` would instead ADD a spurious
        ``3 * dim`` qkv bias. The deviation must not drift silently.
        """
        block = FastVitAttentionBlock(**basic_config)
        block.build((None, 8, 4, 64))

        names = [w.path for w in block.attn.weights]
        assert len(names) == 2, names
        assert all(n.endswith('/kernel') for n in names), names
        assert not any('bias' in n for n in names), names

    # ==================================================================
    # PIN 4 — norm epsilon
    # ==================================================================

    @pytest.mark.parametrize("normalization_type", ['batch_norm', 'layer_norm'])
    def test_norm_epsilon_is_reference_value(self, normalization_type):
        """PIN 4: epsilon must be 1e-5, not the norms factory's 1e-6 setdefault."""
        block = FastVitAttentionBlock(dim=64, head_dim=32, mlp_ratio=2.0,
                                      normalization_type=normalization_type)
        assert block.norm.epsilon == pytest.approx(1e-5, rel=0, abs=0), (
            f"{normalization_type} norm epsilon is {block.norm.epsilon}; the "
            f"reference uses 1e-5 and the factory setdefaults 1e-6"
        )

    # ==================================================================
    # PIN 5 — LayerScale constraints
    # ==================================================================

    def test_layer_scale_constraints_are_none(self, basic_config):
        """PIN 5: BOTH LayerScale gammas must be free to go negative.

        ``LearnableMultiplier`` defaults to ``constraint='non_neg'``. MEASURED:
        the constraint is enforced by the OPTIMIZER, not by ``Variable.assign``,
        so the second arm takes a real SGD step whose gradient drives gamma
        negative.
        """
        block = FastVitAttentionBlock(**basic_config)
        block.build((None, 8, 4, 64))

        assert block.layer_scale_1.constraint is None
        assert block.layer_scale_2.constraint is None

        for scale in (block.layer_scale_1, block.layer_scale_2):
            gamma = scale.gamma
            optimizer = keras.optimizers.SGD(learning_rate=1.0)
            optimizer.build([gamma])
            grad = ops.convert_to_tensor(
                np.ones(gamma.shape, dtype='float32'))
            optimizer.apply_gradients([(grad, gamma)])
            gamma_value = ops.convert_to_numpy(gamma)
            assert np.all(gamma_value < 0.0), (
                f"gamma was clamped at zero by a weight constraint after an "
                f"optimizer step; got {gamma_value}"
            )

    # ==================================================================
    # PIN 6 — graph mode
    # ==================================================================

    def test_graph_mode_matches_eager(self, basic_config):
        """PIN 6: the dynamic-shape path must trace under ``@tf.function``.

        ``fit()`` runs in this regime. A Python ``int()``/``tuple()`` coercion of
        a traced spatial dimension raises here while every eager test passes.
        """
        block = FastVitAttentionBlock(**basic_config)
        rng = np.random.default_rng(555)
        x = rng.normal(size=(2, 8, 4, 64)).astype('float32')
        eager = ops.convert_to_numpy(block(x, training=False))

        @tf.function(input_signature=[
            tf.TensorSpec(shape=(None, None, None, 64), dtype=tf.float32)])
        def traced(tensor):
            return block(tensor, training=False)

        graph = np.asarray(traced(tf.constant(x)))
        assert graph.shape == eager.shape
        np.testing.assert_allclose(graph, eager, atol=1e-6, rtol=0)

        # A second, differently-shaped call must reuse the SAME dynamic trace.
        other = rng.normal(size=(1, 5, 7, 64)).astype('float32')
        assert tuple(np.asarray(traced(tf.constant(other))).shape) == (1, 5, 7, 64)
