"""
Test suite for FastVitStage (FastViT / MobileCLIP2 MCi).

Covers initialization + stored config, constructor validation, forward shape for
BOTH token mixers, `compute_output_shape` pre- and post-build (square AND
non-square), training-mode behaviour, gradient flow, a `.keras` VALUE round trip,
and the five mandated behavioural pins:

1. `test_per_block_drop_path_wiring` — THE step's key pin. Block ``i`` must receive
   ``drop_path_rates[i]``. A reversed or off-by-one list produces an
   identically-shaped, identically-parameterized, subtly-wrong model, so the
   schedule used here is NON-DEGENERATE and DISTINCT (an all-zeros or all-equal
   schedule is invariant under both defects and would be vacuous).
2. `test_downsample_changes_spatial_shape` (square + NON-SQUARE).
3. `test_pos_emb_wired` — both arms: the positional encoding CHANGES the output,
   and zeroing its depthwise kernel makes it exactly the identity (which is what
   ``conv(x) + x`` means), on transplanted identical weights.
4. `test_blocks_are_the_requested_type`.
5. `test_roundtrip_preserves_block_weights_elementwise` — the nested-sub-layer-list
   trap: counts, paths and parameter totals can all match while the restored
   kernels are FRESH. Only an elementwise value diff sees it.
"""

import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.layers.fastvit.attention_block import FastVitAttentionBlock
from dl_techniques.layers.fastvit.patch_embed import FastVitPatchEmbed
from dl_techniques.layers.fastvit.rep_conditional_pos_enc import (
    RepConditionalPosEnc,
)
from dl_techniques.layers.fastvit.rep_mixer import FastVitRepMixerBlock
from dl_techniques.layers.fastvit.stage import FastVitStage


class TestFastVitStage:
    """Comprehensive test suite for one FastViT stage."""

    # ------------------------------------------------------------------
    # fixtures
    # ------------------------------------------------------------------

    @pytest.fixture
    def basic_config(self):
        """Small repmixer stage with a downsample."""
        return {
            'dim': 64,
            'depth': 2,
            'token_mixer': 'repmixer',
            'downsample': True,
            'mlp_ratio': 2.0,
        }

    @pytest.fixture
    def sample_input(self):
        """Deterministic NON-SQUARE rank-4 input, (B, H, W, C) = (2, 16, 8, 32)."""
        rng = np.random.default_rng(20260813)
        return rng.normal(size=(2, 16, 8, 32)).astype('float32')

    # ------------------------------------------------------------------
    # initialization / config
    # ------------------------------------------------------------------

    def test_initialization_stores_config(self, basic_config):
        stage = FastVitStage(**basic_config)

        assert stage.dim == 64
        assert stage.depth == 2
        assert stage.token_mixer == 'repmixer'
        assert stage.downsample_enabled is True
        assert stage.use_pos_emb is False
        assert stage.down_patch_size == 7
        assert stage.down_stride == 2
        assert stage.lkc_use_act is True
        assert stage.drop_path_rates == [0.0, 0.0]
        assert stage.layer_scale_init_value == pytest.approx(1e-5)
        assert not stage.built

    def test_sublayers_created_in_init(self, basic_config):
        """Every sub-layer must exist before build (H-1)."""
        stage = FastVitStage(**basic_config, use_pos_emb=True)
        assert isinstance(stage.downsample, FastVitPatchEmbed)
        assert isinstance(stage.pos_emb, RepConditionalPosEnc)
        assert len(stage.blocks) == 2
        assert all(block is not None for block in stage.blocks)

    def test_optional_sublayers_are_none_when_disabled(self):
        stage = FastVitStage(dim=64, depth=1, downsample=False, use_pos_emb=False)
        assert stage.downsample is None
        assert stage.pos_emb is None

    def test_block_names_are_stable_across_token_mixers(self):
        repmixer = FastVitStage(dim=64, depth=3, token_mixer='repmixer')
        attention = FastVitStage(dim=64, depth=3, token_mixer='attention')
        assert [b.name for b in repmixer.blocks] == ['block_0', 'block_1', 'block_2']
        assert [b.name for b in attention.blocks] == \
               ['block_0', 'block_1', 'block_2']

    def test_get_config_round_trips_through_from_config(self, basic_config):
        stage = FastVitStage(**basic_config, use_pos_emb=True,
                             drop_path_rates=[0.1, 0.2], se_downsample=True,
                             lkc_use_act=False)
        config = stage.get_config()
        clone = FastVitStage.from_config(config)

        for key in ('dim', 'depth', 'token_mixer', 'downsample_enabled',
                    'se_downsample', 'use_pos_emb', 'mlp_ratio',
                    'repmixer_kernel_size', 'head_dim', 'normalization_type',
                    'down_patch_size', 'down_stride', 'lkc_use_act',
                    'dropout_rate', 'drop_path_rates', 'layer_scale_init_value'):
            assert getattr(clone, key) == getattr(stage, key), key

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("kwargs,match", [
        ({'dim': 0, 'depth': 1}, "dim must be positive"),
        ({'dim': 64, 'depth': 0}, "depth must be positive"),
        ({'dim': 64, 'depth': -2}, "depth must be positive"),
        ({'dim': 64, 'depth': 1, 'repmixer_kernel_size': 0},
         "repmixer_kernel_size must be positive"),
        ({'dim': 64, 'depth': 1, 'down_patch_size': 0},
         "down_patch_size must be positive"),
        ({'dim': 64, 'depth': 1, 'down_stride': 0},
         "down_stride must be positive"),
        ({'dim': 64, 'depth': 1, 'dropout_rate': 1.0},
         r"dropout_rate must be in \[0, 1\)"),
    ])
    def test_invalid_config_raises(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            FastVitStage(**kwargs)

    @pytest.mark.parametrize("token_mixer", ['repmix', 'attn', 'RepMixer', ''])
    def test_invalid_token_mixer_raises_listing_both_options(self, token_mixer):
        with pytest.raises(ValueError) as excinfo:
            FastVitStage(dim=64, depth=2, token_mixer=token_mixer)
        message = str(excinfo.value)
        assert "'repmixer'" in message
        assert "'attention'" in message
        assert repr(token_mixer) in message

    @pytest.mark.parametrize("rates,depth", [
        ([0.0, 0.1, 0.2], 2),
        ([0.0], 3),
        ([], 1),
        ([0.0, 0.1, 0.2, 0.3], 3),
    ])
    def test_wrong_drop_path_rates_length_raises_naming_both_numbers(
            self, rates, depth):
        with pytest.raises(ValueError) as excinfo:
            FastVitStage(dim=64, depth=depth, drop_path_rates=rates)
        message = str(excinfo.value)
        assert f"depth={depth}" in message
        assert f"len(drop_path_rates)={len(rates)}" in message

    def test_non_numeric_drop_path_rate_raises(self):
        with pytest.raises(ValueError, match="must be real numbers"):
            FastVitStage(dim=64, depth=2, drop_path_rates=[0.0, 'high'])

    def test_build_rejects_wrong_rank(self, basic_config):
        stage = FastVitStage(**basic_config)
        with pytest.raises(ValueError, match="rank-4"):
            stage.build((None, 16, 64))

    def test_build_without_downsample_rejects_channel_mismatch(self):
        stage = FastVitStage(dim=64, depth=1, downsample=False)
        with pytest.raises(ValueError, match="must equal dim=64"):
            stage.build((None, 8, 8, 32))

    # ------------------------------------------------------------------
    # forward / shapes
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("token_mixer", ['repmixer', 'attention'])
    def test_forward_shape_both_token_mixers(self, token_mixer):
        stage = FastVitStage(dim=64, depth=2, token_mixer=token_mixer,
                             downsample=True, mlp_ratio=2.0, head_dim=32)
        rng = np.random.default_rng(11)
        x = rng.normal(size=(2, 16, 16, 32)).astype('float32')
        y = stage(x, training=False)
        assert tuple(y.shape) == (2, 8, 8, 64)
        assert np.all(np.isfinite(ops.convert_to_numpy(y)))

    @pytest.mark.parametrize("token_mixer", ['repmixer', 'attention'])
    def test_forward_shape_without_downsample(self, token_mixer):
        stage = FastVitStage(dim=64, depth=2, token_mixer=token_mixer,
                             downsample=False, mlp_ratio=2.0)
        rng = np.random.default_rng(12)
        x = rng.normal(size=(2, 8, 8, 64)).astype('float32')
        y = stage(x, training=False)
        assert tuple(y.shape) == (2, 8, 8, 64)

    def test_forward_with_pos_emb_at_small_feature_map(self):
        """RepCPE's 7x7 depthwise kernel must survive a 4x4 map (5-stage tail)."""
        stage = FastVitStage(dim=64, depth=1, token_mixer='attention',
                             downsample=True, use_pos_emb=True, mlp_ratio=2.0)
        x = np.random.default_rng(5).normal(size=(2, 8, 8, 32)).astype('float32')
        y = stage(x, training=False)
        assert tuple(y.shape) == (2, 4, 4, 64)

    def test_compute_output_shape_pre_and_post_build(self, basic_config):
        stage = FastVitStage(**basic_config)
        # pre-build
        assert stage.compute_output_shape((None, 16, 8, 32)) == (None, 8, 4, 64)
        stage.build((None, 16, 8, 32))
        # post-build
        assert stage.compute_output_shape((2, 16, 8, 32)) == (2, 8, 4, 64)

    @pytest.mark.parametrize("shape,downsample,expected", [
        ((2, 16, 16, 32), True, (2, 8, 8, 64)),
        ((2, 16, 8, 32), True, (2, 8, 4, 64)),
        ((2, 7, 5, 32), True, (2, 4, 3, 64)),
        ((2, 16, 8, 64), False, (2, 16, 8, 64)),
    ])
    def test_compute_output_shape_matches_actual(self, shape, downsample,
                                                 expected):
        stage = FastVitStage(dim=64, depth=1, downsample=downsample,
                             mlp_ratio=2.0)
        x = np.random.default_rng(6).normal(size=shape).astype('float32')
        y = stage(x, training=False)
        assert stage.compute_output_shape(shape) == expected
        assert tuple(y.shape) == expected

    def test_training_true_and_false_both_run(self, basic_config, sample_input):
        stage = FastVitStage(**basic_config, dropout_rate=0.1,
                             drop_path_rates=[0.1, 0.2])
        y_train = stage(sample_input, training=True)
        y_eval = stage(sample_input, training=False)
        assert tuple(y_train.shape) == (2, 8, 4, 64)
        assert tuple(y_eval.shape) == (2, 8, 4, 64)
        assert np.all(np.isfinite(ops.convert_to_numpy(y_eval)))

    def test_training_false_is_deterministic(self, basic_config, sample_input):
        stage = FastVitStage(**basic_config, dropout_rate=0.5,
                             drop_path_rates=[0.5, 0.5])
        first = ops.convert_to_numpy(stage(sample_input, training=False))
        second = ops.convert_to_numpy(stage(sample_input, training=False))
        np.testing.assert_allclose(first, second, atol=1e-6, rtol=0)

    # ------------------------------------------------------------------
    # gradients
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("token_mixer", ['repmixer', 'attention'])
    def test_gradients_reach_every_trainable_weight(self, token_mixer):
        stage = FastVitStage(dim=64, depth=2, token_mixer=token_mixer,
                             downsample=True, use_pos_emb=True, mlp_ratio=2.0,
                             layer_scale_init_value=1.0)
        rng = np.random.default_rng(21)
        x = tf.constant(rng.normal(size=(2, 16, 16, 32)).astype('float32'))
        with tf.GradientTape() as tape:
            y = stage(x, training=True)
            loss = ops.mean(ops.square(y))
        grads = tape.gradient(loss, stage.trainable_variables)

        assert len(stage.trainable_variables) > 0
        missing = [
            v.path for v, g in zip(stage.trainable_variables, grads) if g is None
        ]
        assert not missing, f"no gradient reached: {missing}"

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("token_mixer", ['repmixer', 'attention'])
    def test_keras_round_trip_matches_by_value(self, token_mixer, sample_input):
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = FastVitStage(dim=64, depth=2, token_mixer=token_mixer,
                               downsample=True, use_pos_emb=True, mlp_ratio=2.0,
                               drop_path_rates=[0.05, 0.1])(inputs)
        model = keras.Model(inputs, outputs)
        before = model(sample_input, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'stage.keras')
            model.save(path)
            restored = keras.models.load_model(path)
            after = restored(sample_input, training=False)

        np.testing.assert_allclose(
            ops.convert_to_numpy(before),
            ops.convert_to_numpy(after),
            atol=1e-6, rtol=0,
        )

    # ==================================================================
    # PIN 1 — per-block drop-path WIRING (the step's key pin)
    # ==================================================================

    @pytest.mark.parametrize("token_mixer", ['repmixer', 'attention'])
    def test_per_block_drop_path_wiring(self, token_mixer):
        """PIN 1: block ``i`` receives ``drop_path_rates[i]`` — not shifted, not reversed.

        The schedule is NON-DEGENERATE and pairwise DISTINCT on purpose: an
        all-zeros or all-equal schedule is invariant under BOTH a reversal and an
        off-by-one shift, so such a pin could never go red. Both the block's own
        stored rate and the ``StochasticDepth`` sub-layer's stored rate are read,
        because the block could store the right number and hand the wrong one to
        the sub-layer that actually uses it.
        """
        rates = [0.0, 0.1, 0.2, 0.3]
        stage = FastVitStage(dim=64, depth=len(rates), token_mixer=token_mixer,
                             downsample=False, mlp_ratio=2.0,
                             drop_path_rates=rates)

        # The schedule must be able to SEE both defects.
        assert rates != rates[::-1]
        assert len(set(rates)) == len(rates)

        assert stage.drop_path_rates == rates
        for index, expected in enumerate(rates):
            block = stage.blocks[index]
            assert block.drop_path_rate == pytest.approx(expected), (
                f"block_{index} stores drop_path_rate={block.drop_path_rate}, "
                f"expected {expected} (schedule {rates})"
            )
            # ...and the StochasticDepth sub-layer that actually applies it.
            if token_mixer == 'repmixer':
                sub_layers = [block.drop_path]
            else:
                sub_layers = [block.drop_path_1, block.drop_path_2]
            for sub_layer in sub_layers:
                assert sub_layer.drop_path_rate == pytest.approx(expected), (
                    f"block_{index}.{sub_layer.name} stores "
                    f"{sub_layer.drop_path_rate}, expected {expected}"
                )

    def test_drop_path_rates_default_to_zeros(self):
        stage = FastVitStage(dim=64, depth=3, downsample=False)
        assert stage.drop_path_rates == [0.0, 0.0, 0.0]
        assert all(b.drop_path_rate == 0.0 for b in stage.blocks)

    def test_drop_path_rates_survive_serialization_in_order(self):
        rates = [0.0, 0.1, 0.2, 0.3]
        stage = FastVitStage(dim=64, depth=4, downsample=False, mlp_ratio=2.0,
                             drop_path_rates=rates)
        clone = FastVitStage.from_config(stage.get_config())
        assert clone.drop_path_rates == rates
        assert [b.drop_path_rate for b in clone.blocks] == rates

    # ==================================================================
    # PIN 2 — downsample on/off changes the spatial shape as tabulated
    # ==================================================================

    def test_downsample_changes_spatial_shape(self):
        """PIN 2: downsample=True halves H and W; downsample=False preserves them."""
        rng = np.random.default_rng(1234)

        # --- square, the tabulated 64 -> 32 case -------------------------
        x_square = rng.normal(size=(2, 64, 64, 32)).astype('float32')
        with_down = FastVitStage(dim=64, depth=1, downsample=True, mlp_ratio=2.0)
        y_down = with_down(x_square, training=False)
        assert tuple(y_down.shape) == (2, 32, 32, 64)

        x_same = rng.normal(size=(2, 64, 64, 64)).astype('float32')
        without_down = FastVitStage(dim=64, depth=1, downsample=False,
                                    mlp_ratio=2.0)
        y_same = without_down(x_same, training=False)
        assert tuple(y_same.shape) == (2, 64, 64, 64)

    def test_downsample_changes_spatial_shape_non_square(self):
        """PIN 2 (non-square arm): H and W must be halved INDEPENDENTLY.

        A square-only assertion cannot distinguish ``(H/2, W/2)`` from a
        transposed ``(W/2, H/2)``.
        """
        rng = np.random.default_rng(4321)
        x = rng.normal(size=(2, 32, 16, 32)).astype('float32')
        stage = FastVitStage(dim=64, depth=1, downsample=True, mlp_ratio=2.0)
        y = stage(x, training=False)
        assert tuple(y.shape) == (2, 16, 8, 64)

        x_same = rng.normal(size=(2, 32, 16, 64)).astype('float32')
        no_down = FastVitStage(dim=64, depth=1, downsample=False, mlp_ratio=2.0)
        assert tuple(no_down(x_same, training=False).shape) == (2, 32, 16, 64)

    # ==================================================================
    # PIN 3 — the positional encoding is WIRED
    # ==================================================================

    def test_pos_emb_wired(self):
        """PIN 3: the RepCPE is actually applied, and it is ``conv(x) + x``.

        Weight transplanting IS practical here — the two stages differ only by the
        ``pos_emb`` sub-layer, so every shared sub-layer can be copied
        sub-layer-by-sub-layer — and it is the stronger instrument, so it is what
        this pin uses. Two arms:

        A. With identical weights everywhere else, the ``use_pos_emb=True`` stage
           and the ``use_pos_emb=False`` stage DISAGREE. This is the arm an
           unwired ``pos_emb`` fails.
        B. Zeroing the RepCPE's depthwise kernel and bias makes them AGREE
           exactly. Without arm B, arm A could be satisfied by any spurious extra
           layer; arm B pins that the extra term is precisely ``conv(x) + x`` and
           nothing else (a wrong skip, or a missing skip, breaks it).
        """
        config = dict(dim=64, depth=2, token_mixer='repmixer', downsample=True,
                      mlp_ratio=2.0)
        with_pos = FastVitStage(**config, use_pos_emb=True)
        without_pos = FastVitStage(**config, use_pos_emb=False)

        rng = np.random.default_rng(777)
        x = rng.normal(size=(2, 16, 16, 32)).astype('float32')
        with_pos(x, training=False)
        without_pos(x, training=False)

        assert with_pos.pos_emb is not None
        assert without_pos.pos_emb is None

        # Transplant every SHARED sub-layer so the ONLY difference is pos_emb.
        without_pos.downsample.set_weights(with_pos.downsample.get_weights())
        for target, source in zip(without_pos.blocks, with_pos.blocks):
            target.set_weights(source.get_weights())

        y_with = ops.convert_to_numpy(with_pos(x, training=False))
        y_without = ops.convert_to_numpy(without_pos(x, training=False))

        # --- arm A: the positional encoding changes the output ----------
        assert not np.allclose(y_with, y_without, atol=1e-5), (
            "use_pos_emb=True produced the same output as use_pos_emb=False on "
            "transplanted identical weights: the pos_emb is not wired into call()"
        )

        # --- arm B: with a zeroed kernel the RepCPE is exactly identity --
        for variable in with_pos.pos_emb.pos_conv.weights:
            variable.assign(np.zeros(variable.shape, dtype='float32'))
        y_zeroed = ops.convert_to_numpy(with_pos(x, training=False))
        np.testing.assert_allclose(y_zeroed, y_without, atol=1e-5, rtol=0)

    # ==================================================================
    # PIN 4 — the blocks are of the requested type
    # ==================================================================

    @pytest.mark.parametrize("token_mixer,expected_type", [
        ('repmixer', FastVitRepMixerBlock),
        ('attention', FastVitAttentionBlock),
    ])
    def test_blocks_are_the_requested_type(self, token_mixer, expected_type):
        """PIN 4: EVERY block is the requested type — not just the first."""
        stage = FastVitStage(dim=64, depth=4, token_mixer=token_mixer,
                             downsample=False, mlp_ratio=2.0)
        assert len(stage.blocks) == 4
        for index, block in enumerate(stage.blocks):
            assert isinstance(block, expected_type), (
                f"block_{index} is {type(block).__name__}, "
                f"expected {expected_type.__name__}"
            )

    # ==================================================================
    # PIN 5 — the round trip restores block weights ELEMENTWISE
    # ==================================================================

    @pytest.mark.parametrize("token_mixer", ['repmixer', 'attention'])
    def test_roundtrip_preserves_block_weights_elementwise(self, token_mixer):
        """PIN 5: compare a weight tensor from EACH block elementwise.

        The blocks are held in a Python list attribute. On this stack, a nested
        ``List[List[Layer]]`` loses its weights on a ``.keras`` round trip while
        the layer count, the variable paths AND the parameter total all still
        match — so an output or a count comparison is not evidence. Only an
        elementwise value diff sees it.
        """
        sample = np.random.default_rng(31).normal(
            size=(2, 16, 16, 32)).astype('float32')
        inputs = keras.Input(shape=sample.shape[1:])
        stage = FastVitStage(dim=64, depth=3, token_mixer=token_mixer,
                             downsample=True, use_pos_emb=True, mlp_ratio=2.0)
        model = keras.Model(inputs, stage(inputs))

        # Move every weight OFF its initialization so a fresh re-initialization
        # cannot coincide with the saved values.
        rng = np.random.default_rng(32)
        for variable in model.trainable_variables:
            variable.assign(
                rng.normal(size=variable.shape).astype('float32') * 0.1)

        expected = [
            [ops.convert_to_numpy(w) for w in block.weights]
            for block in stage.blocks
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'stage_blocks.keras')
            model.save(path)
            restored_model = keras.models.load_model(path)

        restored_stages = [
            layer for layer in restored_model.layers
            if isinstance(layer, FastVitStage)
        ]
        assert len(restored_stages) == 1
        restored = restored_stages[0]

        assert len(restored.blocks) == len(stage.blocks)
        for index, (block, reference) in enumerate(
                zip(restored.blocks, expected)):
            actual = [ops.convert_to_numpy(w) for w in block.weights]
            assert len(actual) == len(reference), (
                f"block_{index} restored with {len(actual)} weights, "
                f"expected {len(reference)}"
            )
            assert len(actual) > 0, f"block_{index} restored with NO weights"
            for weight_index, (got, want) in enumerate(zip(actual, reference)):
                np.testing.assert_array_equal(
                    got, want,
                    err_msg=(
                        f"block_{index} weight {weight_index} "
                        f"({block.weights[weight_index].path}) is not "
                        f"elementwise identical after the round trip"
                    ),
                )
