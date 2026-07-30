import pytest
import numpy as np
import tensorflow as tf
import keras
from keras import ops, layers, models
import tempfile
import os
import warnings
from typing import Any, Dict

from dl_techniques.layers.transformers.transformer import TransformerLayer
from dl_techniques.layers.moe import MoEConfig, ExpertConfig, GatingConfig


# --- Test Class ---
class TestTransformerLayer:
    """
    Comprehensive and modern test suite for the TransformerLayer.
    This suite follows modern Keras 3 testing best practices and includes MoE support.
    """

    # --- Fixtures for Reusability ---
    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Provides a standard configuration for a small, testable layer."""
        return {
            'hidden_size': 64,
            'num_heads': 4,
            'intermediate_size': 256,
        }

    @pytest.fixture
    def moe_config(self) -> MoEConfig:
        """Provides a standard MoE configuration for testing, using the new ffn_config structure."""
        return MoEConfig(
            num_experts=4,
            expert_config=ExpertConfig(
                ffn_config={
                    "type": "mlp",
                    "output_dim": 64,      # Matches hidden_size
                    "hidden_dim": 256,     # The intermediate size for the expert
                }
            ),
            gating_config=GatingConfig(
                gating_type='linear',
                top_k=2
            )
        )

    @pytest.fixture
    def moe_config_dict(self) -> Dict[str, Any]:
        """Provides MoE configuration as a dictionary for testing dict conversion."""
        return {
            'num_experts': 4,
            'expert_config': {
                'ffn_config': {
                    "type": "swiglu",
                    "output_dim": 64,
                    "ffn_expansion_factor": 4
                }
            },
            'gating_config': {
                'gating_type': 'linear',
                'top_k': 2
            }
        }

    @pytest.fixture
    def sample_input(self) -> tf.Tensor:
        """Provides a standard sample input tensor for testing."""
        # seq_len=16 is a perfect square (4*4), making it compatible
        # with WindowAttention(window_size=4).
        return tf.random.normal(shape=(4, 16, 64))

    # ===============================================
    # 1. Initialization and Build Tests
    # ===============================================
    def test_initialization_defaults(self, layer_config):
        """Tests layer initialization with default parameters."""
        layer = TransformerLayer(**layer_config)
        assert not layer.built
        assert layer.attention_type == 'multi_head'
        assert layer.normalization_type == 'layer_norm'
        assert layer.ffn_type == 'mlp'
        assert layer.moe_config is None

    def test_initialization_with_moe_config(self, layer_config, moe_config):
        """Tests initialization with MoE configuration."""
        layer = TransformerLayer(**layer_config, moe_config=moe_config)
        assert layer.moe_config is not None
        assert isinstance(layer.moe_config, MoEConfig)
        assert layer.moe_config.num_experts == 4
        assert layer.moe_config.expert_config.ffn_config['output_dim'] == 64

    def test_initialization_with_moe_dict(self, layer_config, moe_config_dict):
        """Tests initialization with MoE configuration as dictionary."""
        layer = TransformerLayer(**layer_config, moe_config=moe_config_dict)
        assert layer.moe_config is not None
        assert isinstance(layer.moe_config, MoEConfig)
        assert layer.moe_config.num_experts == 4
        assert layer.moe_config.expert_config.ffn_config['type'] == 'swiglu'

    def test_moe_overrides_ffn_params_with_warning(self, layer_config, moe_config):
        """Tests that MoE config overrides FFN parameters and issues warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            layer = TransformerLayer(
                hidden_size=layer_config['hidden_size'],
                num_heads=layer_config['num_heads'],
                intermediate_size=512,
                moe_config=moe_config,
                ffn_type='swiglu',
                ffn_args={'some_arg': 'value'},
            )
            assert any("moe_config is provided" in str(warn.message) for warn in w)
            assert any("`ffn_type`" in str(warn.message) for warn in w)
            assert any("`ffn_args`" in str(warn.message) for warn in w)

    def test_moe_hidden_dim_synchronization(self, layer_config, moe_config):
        """Tests that MoE expert output_dim is synchronized with transformer hidden_size."""
        moe_config.expert_config.ffn_config['output_dim'] = 128  # Mismatched
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            layer = TransformerLayer(**layer_config, moe_config=moe_config)
            assert any("Adjusting moe_config.expert_config.ffn_config['output_dim']" in str(warn.message) for warn in w)
        assert layer.moe_config.expert_config.ffn_config['output_dim'] == layer.hidden_size

    def test_moe_intermediate_size_inheritance(self, layer_config):
        """Tests that MoE expert inherits intermediate_size from TransformerLayer when not set."""
        moe_config_no_intermediate = MoEConfig(
            num_experts=4,
            expert_config=ExpertConfig(ffn_config={"type": "mlp", "output_dim": 64})
        )
        layer = TransformerLayer(**layer_config, moe_config=moe_config_no_intermediate)
        assert 'hidden_dim' in layer.moe_config.expert_config.ffn_config
        assert layer.moe_config.expert_config.ffn_config['hidden_dim'] == layer.intermediate_size

    def test_moe_preserves_explicit_intermediate_size(self):
        """Tests that explicitly set expert intermediate_size is preserved."""
        moe_config = MoEConfig(
            num_experts=4,
            expert_config=ExpertConfig(
                ffn_config={"type": "mlp", "output_dim": 64, "hidden_dim": 512}
            )
        )
        layer = TransformerLayer(hidden_size=64, num_heads=4, intermediate_size=256, moe_config=moe_config)
        assert layer.moe_config.expert_config.ffn_config['hidden_dim'] == 512

    def test_build_process_with_moe(self, layer_config, moe_config, sample_input):
        """Tests that the layer with MoE and all its sub-layers are built correctly."""
        layer = TransformerLayer(**layer_config, moe_config=moe_config)
        assert not layer.built
        layer(sample_input)
        assert layer.built
        assert hasattr(layer.ffn_layer, 'experts')

    # ===============================================
    # 2. Forward Pass and Core Behavior Tests
    # ===============================================
    @pytest.mark.parametrize("expert_ffn_type", ['mlp', 'swiglu', 'glu', 'geglu'])
    @pytest.mark.parametrize("gating_type", ['linear', 'cosine'])
    @pytest.mark.parametrize("top_k", [1, 2])
    def test_forward_pass_with_moe_variations(self, expert_ffn_type, gating_type, top_k, sample_input):
        """Tests forward pass with various MoE configurations."""
        ffn_config = {"type": expert_ffn_type, "output_dim": 64}
        if expert_ffn_type in ['mlp', 'glu', 'geglu']:
            ffn_config['hidden_dim'] = 256
        elif expert_ffn_type == 'swiglu':
            ffn_config['ffn_expansion_factor'] = 4

        moe_config = MoEConfig(
            num_experts=4,
            expert_config=ExpertConfig(ffn_config=ffn_config),
            gating_config=GatingConfig(gating_type=gating_type, top_k=top_k)
        )
        layer = TransformerLayer(hidden_size=64, num_heads=4, intermediate_size=256, moe_config=moe_config)
        output = layer(sample_input, training=False)
        assert output.shape == sample_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    def test_moe_forward_pass_training_vs_inference(self, layer_config, moe_config, sample_input):
        """Tests that MoE behaves differently in training vs inference mode."""
        layer = TransformerLayer(**layer_config, moe_config=moe_config, dropout_rate=0.0)
        output_train = layer(sample_input, training=True)
        output_infer = layer(sample_input, training=False)
        assert output_train.shape == output_infer.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output_train)))
        assert not np.any(np.isnan(ops.convert_to_numpy(output_infer)))

    # ===============================================
    # 3. Serialization Test (The Gold Standard)
    # ===============================================

    # The following comprehensive test replaces previous, more focused tests for
    # attention forward pass and serialization. It covers a wider range of
    # component combinations in a single, efficient test case.
    @pytest.mark.parametrize(
        "attention_config",
        [
            {'attention_type': 'multi_head', 'attention_args': {}},
            {'attention_type': 'window', 'attention_args': {'window_size': 4}},
            # `num_kv_heads` is the GroupedQueryAttention parameter. This fixture used to say
            # `n_kv_head` (TransformerLayer's OWN ctor name), which the attention factory
            # filtered out silently -- so it built with num_kv_heads=4 (the default), not 2,
            # and the test passed anyway. Verified: attention_args={'n_kv_head': 2} ->
            # num_kv_heads == 4; attention_args={'num_kv_heads': 2} -> num_kv_heads == 2.
            {'attention_type': 'group_query', 'attention_args': {'num_kv_heads': 2}},
            # hidden_size (64) / num_heads (4) = 16
            {'attention_type': 'differential', 'attention_args': {'head_dim': 16}},
        ]
    )
    @pytest.mark.parametrize("ffn_type", ['mlp', 'swiglu', 'glu', 'geglu'])
    @pytest.mark.parametrize("normalization_type", ['layer_norm', 'rms_norm', 'zero_centered_rms_norm', 'dynamic_tanh'])
    @pytest.mark.parametrize("normalization_position", ['pre', 'post'])
    def test_component_combinations_forward_pass_and_serialization(
        self,
        layer_config,
        sample_input,
        attention_config,
        ffn_type,
        normalization_type,
        normalization_position
    ):
        """
        Tests forward pass and serialization for various combinations of core components.
        This is a comprehensive 'gold standard' test that covers multiple configurations
        of attention, FFN, and normalization to ensure robust interoperability.
        """
        # --- 1. Setup Layer Configuration ---
        full_config = {
            **layer_config,
            **attention_config,
            'ffn_type': ffn_type,
            'normalization_type': normalization_type,
            'normalization_position': normalization_position,
            'dropout_rate': 0.0,
            'attention_dropout_rate': 0.0,
        }

        # --- 2. Model Creation and Forward Pass ---
        inputs = layers.Input(shape=sample_input.shape[1:])
        outputs = TransformerLayer(**full_config)(inputs)
        model = models.Model(inputs, outputs)

        original_prediction = model(sample_input, training=False)
        assert original_prediction.shape == sample_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(original_prediction)))

        # --- 3. Serialization and Deserialization ---
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_combo_model.keras")
            model.save(filepath)
            loaded_model = models.load_model(filepath)

            # --- 4. Verification ---
            loaded_prediction = loaded_model(sample_input, training=False)
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg=f"Mismatch for combo: {full_config}"
            )

    def test_full_serialization_cycle_with_moe(self, moe_config, sample_input):
        """Tests full serialization cycle with MoE configuration."""
        layer_config = {
            'hidden_size': 64, 'num_heads': 4, 'intermediate_size': 256,
            'moe_config': moe_config, 'normalization_position': 'pre', 'use_stochastic_depth': True,
        }
        inputs = layers.Input(shape=sample_input.shape[1:])
        outputs = TransformerLayer(**layer_config)(inputs)
        model = models.Model(inputs, outputs)
        original_prediction = model(sample_input, training=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_moe_model.keras")
            model.save(filepath)
            loaded_model = models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input, training=False)
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6
            )

    def test_full_serialization_cycle_with_moe_dict(self, moe_config_dict, sample_input):
        """Tests full serialization cycle with MoE configuration provided as dict."""
        layer_config = {
            'hidden_size': 64, 'num_heads': 4, 'intermediate_size': 256,
            'moe_config': moe_config_dict, 'normalization_position': 'post',
        }
        inputs = layers.Input(shape=sample_input.shape[1:])
        outputs = TransformerLayer(**layer_config)(inputs)
        model = models.Model(inputs, outputs)
        original_prediction = model(sample_input, training=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_moe_dict_model.keras")
            model.save(filepath)
            loaded_model = models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input, training=False)
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6
            )

    # ===============================================
    # 4. Gradient and Training Integration Tests
    # ===============================================
    def test_gradient_flow_with_moe(self, layer_config, moe_config, sample_input):
        """Tests gradient flow through MoE layer."""
        layer = TransformerLayer(**layer_config, moe_config=moe_config)
        x_var = tf.Variable(sample_input)
        with tf.GradientTape() as tape:
            output = layer(x_var, training=True)
            loss = ops.mean(ops.square(output))
        gradients = tape.gradient(loss, layer.trainable_variables)
        assert len(gradients) > 0, "No gradients were computed for MoE."
        assert all(g is not None for g in gradients), "A gradient is None in MoE."

    def test_model_training_loop_integration_with_moe(self, layer_config, moe_config):
        """Ensures MoE layer can be used in a standard training loop."""
        model = models.Sequential([
            layers.InputLayer(shape=(16, 64)),
            TransformerLayer(**layer_config, moe_config=moe_config),
            layers.GlobalAveragePooling1D(),
            layers.Dense(10)
        ])
        model.compile("adam", keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        x_train = tf.random.normal((32, 16, 64))
        y_train = tf.random.uniform([32], 0, 10, dtype=tf.int32)
        history = model.fit(x_train, y_train, epochs=1, batch_size=8, verbose=0)
        assert 'loss' in history.history
        assert not np.isnan(history.history['loss'][0]), "Loss became NaN during MoE training."

    def test_mixed_stacked_layers_with_moe(self, sample_input, moe_config):
        """Tests stacking TransformerLayers with mixed standard and MoE FFNs."""
        config_standard = {'hidden_size': 64, 'num_heads': 4, 'intermediate_size': 256}
        config_moe = {'hidden_size': 64, 'num_heads': 4, 'intermediate_size': 256, 'moe_config': moe_config}
        inputs = layers.Input(shape=sample_input.shape[1:])
        x = TransformerLayer(**config_standard)(inputs)
        x = TransformerLayer(**config_moe)(x)
        outputs = layers.GlobalAveragePooling1D()(x)
        model = models.Model(inputs, outputs)
        prediction = model(sample_input, training=False)
        assert prediction.shape == (sample_input.shape[0], 64)
        assert not np.any(np.isnan(ops.convert_to_numpy(prediction)))

    # ===============================================
    # 5. MoE-Specific Tests
    # ===============================================
    def test_moe_config_get_config(self, layer_config, moe_config):
        """Tests that get_config properly serializes MoE configuration."""
        layer = TransformerLayer(**layer_config, moe_config=moe_config)
        config = layer.get_config()
        assert 'moe_config' in config
        assert config['moe_config'] is not None
        assert isinstance(config['moe_config']['expert_config'], dict)
        assert 'ffn_config' in config['moe_config']['expert_config']

    def test_moe_config_from_config(self, layer_config, moe_config):
        """Tests that layer can be reconstructed from config with MoE."""
        original_layer = TransformerLayer(**layer_config, moe_config=moe_config)
        config = original_layer.get_config()
        reconstructed_layer = TransformerLayer.from_config(config)
        assert reconstructed_layer.moe_config is not None
        assert isinstance(reconstructed_layer.moe_config, MoEConfig)
        assert reconstructed_layer.moe_config.num_experts == original_layer.moe_config.num_experts

    @pytest.mark.parametrize("num_experts", [2, 8])
    @pytest.mark.parametrize("top_k", [1, 2])
    def test_moe_scaling_properties(self, num_experts, top_k, sample_input):
        """Tests MoE with different numbers of experts and top-k values."""
        moe_config = MoEConfig(
            num_experts=num_experts,
            expert_config=ExpertConfig(
                ffn_config={"type": "mlp", "output_dim": 64, "hidden_dim": 128}
            ),
            gating_config=GatingConfig(
                gating_type='linear', top_k=min(top_k, num_experts)
            )
        )
        layer = TransformerLayer(hidden_size=64, num_heads=4, intermediate_size=256, moe_config=moe_config)
        output = layer(sample_input, training=False)
        assert output.shape == sample_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))


# ===================================================================
# F4: newly-wired attention types (plan_2026-06-12_0bb1729b)
# multi_head_latent, anchor, lighthouse, fnet
# ===================================================================
class TestTransformerLayerExtendedAttention:
    """Covers the attention types added to TransformerLayer._get_attention_params.

    These four factory keys were previously rejected at construction with
    ``ValueError: Unknown attention type``. This suite asserts they now
    construct, forward, carry gradients, and round-trip through ``.keras`` —
    while a companion regression test confirms the original 4 types are
    byte-identical in their sub-layer wiring.
    """

    HIDDEN = 64
    HEADS = 4
    INTER = 128

    @pytest.fixture
    def sample_input(self) -> tf.Tensor:
        return tf.random.normal(shape=(2, 16, self.HIDDEN))

    def _make(self, attention_type, attention_args=None, normalization_position='pre'):
        return TransformerLayer(
            hidden_size=self.HIDDEN,
            num_heads=self.HEADS,
            intermediate_size=self.INTER,
            attention_type=attention_type,
            attention_args=attention_args or {},
            normalization_position=normalization_position,
            dropout_rate=0.0,
            attention_dropout_rate=0.0,
        )

    @pytest.mark.parametrize("attention_type, attention_args", [
        ('multi_head_latent', {}),
        ('multi_head_latent', {'kv_latent_dim': 8}),
        ('anchor', {}),
        ('lighthouse', {}),
        ('fnet', {}),
    ])
    @pytest.mark.parametrize("normalization_position", ['pre', 'post'])
    def test_construct_and_forward(self, sample_input, attention_type, attention_args, normalization_position):
        """Each new type constructs and forwards (B, seq, H) -> same shape, no NaN."""
        layer = self._make(attention_type, attention_args, normalization_position)
        out = layer(sample_input, training=False)
        assert out.shape == sample_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(out)))

    @pytest.mark.parametrize("attention_type, attention_args", [
        ('multi_head_latent', {'kv_latent_dim': 8}),
        ('anchor', {}),
        ('lighthouse', {}),
        ('fnet', {}),
    ])
    def test_serialization_round_trip(self, sample_input, attention_type, attention_args):
        """`.keras` save/load is numerically identical at atol=1e-6."""
        inputs = layers.Input(shape=sample_input.shape[1:])
        outputs = self._make(attention_type, attention_args)(inputs)
        model = models.Model(inputs, outputs)
        original = model(sample_input, training=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "extattn.keras")
            model.save(filepath)
            loaded = models.load_model(filepath)
            reloaded = loaded(sample_input, training=False)
        np.testing.assert_allclose(
            ops.convert_to_numpy(original),
            ops.convert_to_numpy(reloaded),
            rtol=1e-6, atol=1e-6,
            err_msg=f"round-trip mismatch for {attention_type} {attention_args}",
        )

    @pytest.mark.parametrize("attention_type", ['multi_head_latent', 'anchor', 'lighthouse', 'fnet'])
    def test_gradient_flow(self, sample_input, attention_type):
        """Gradients flow end-to-end through the layer.

        Most variables must receive a non-None gradient. A documented
        exception: ``AnchorAttention`` carries a ``query_token_proj`` that is
        structurally unused under the default (no explicit anchor-token) config,
        so its gradient is legitimately None. Any OTHER None gradient is a bug.
        """
        layer = self._make(attention_type)
        x = tf.Variable(sample_input)
        with tf.GradientTape() as tape:
            out = layer(x, training=True)
            loss = tf.reduce_mean(tf.square(out))
        grads = tape.gradient(loss, layer.trainable_variables)
        assert len(layer.trainable_variables) > 0
        none_paths = [v.path for v, g in zip(layer.trainable_variables, grads) if g is None]
        # Tolerate only the anchor layer's known structurally-unused projection.
        assert all('query_token_proj' in p for p in none_paths), \
            f"unexpected None grad(s) for {attention_type}: {none_paths}"
        assert any(g is not None for g in grads), f"no gradient flow for {attention_type}"

    def test_multi_head_latent_default_kv_latent_dim(self, sample_input):
        """MLA constructs with the documented hidden_size//4 fallback (no user arg)."""
        layer = self._make('multi_head_latent', attention_args={})
        out = layer(sample_input, training=False)
        assert out.shape == sample_input.shape

    def test_fnet_is_maskless_and_ignores_layer_mask(self, sample_input):
        """fnet forwards without an attention_mask reaching the sub-layer, and
        passing a mask at the TransformerLayer level does not crash (mask is
        simply not forwarded to the parameter-free Fourier mixer)."""
        assert 'fnet' in TransformerLayer._MASKLESS_ATTENTION_TYPES
        layer = self._make('fnet')
        mask = tf.ones((sample_input.shape[0], sample_input.shape[1], sample_input.shape[1]))
        out_no_mask = layer(sample_input, training=False)
        out_with_mask = layer(sample_input, attention_mask=mask, training=False)
        # Mask is ignored for fnet -> identical output.
        np.testing.assert_allclose(
            ops.convert_to_numpy(out_no_mask),
            ops.convert_to_numpy(out_with_mask),
            rtol=1e-6, atol=1e-6,
        )

    @pytest.mark.parametrize("attention_type", ['anchor', 'lighthouse'])
    def test_maskless_attention_types_membership(self, attention_type):
        """anchor and lighthouse are registered maskless (their call rejects mask)."""
        assert attention_type in TransformerLayer._MASKLESS_ATTENTION_TYPES

    def test_regression_existing_types_unchanged(self):
        """The 4 pre-existing types keep their exact attention sub-layer wiring.

        Guards against a refactor silently altering param mapping (the
        catastrophic checkpoint-break failure mode). Asserts the param dict
        produced by ``_get_attention_params`` for each legacy type.
        """
        layer_mh = TransformerLayer(64, 4, 128, attention_type='multi_head')
        p = layer_mh._get_attention_params('attn')
        assert p == {'dim': 64, 'num_heads': 4, 'dropout_rate': 0.1,
                     'use_bias': True, 'kernel_initializer': layer_mh.kernel_initializer,
                     'name': 'attn'}

        layer_win = TransformerLayer(64, 4, 128, attention_type='window', window_size=4)
        p = layer_win._get_attention_params('attn')
        assert p == {'dim': 64, 'num_heads': 4, 'window_size': 4,
                     'dropout_rate': 0.1, 'name': 'attn'}

        layer_gqa = TransformerLayer(64, 4, 128, attention_type='group_query', n_kv_head=2)
        p = layer_gqa._get_attention_params('attn')
        assert p == {'dim': 64, 'num_heads': 4, 'num_kv_heads': 2,
                     'dropout_rate': 0.1, 'use_bias': True, 'name': 'attn'}

        layer_diff = TransformerLayer(64, 4, 128, attention_type='differential')
        p = layer_diff._get_attention_params('attn')
        assert p == {'dim': 64, 'num_heads': 4, 'head_dim': 16,
                     'dropout_rate': 0.1, 'lambda_init': 0.8, 'name': 'attn'}

    def test_unknown_attention_type_still_raises(self):
        """An unsupported factory key still fails fast at construction."""
        with pytest.raises(ValueError, match="Unknown attention type"):
            TransformerLayer(64, 4, 128, attention_type='capsule_routing')

# ---------------------------------------------------------------------
# Step-4 guards: the shared FFN_REGISTRY pre-filter.
#
# `TransformerLayer` used to inject its OWN generic conveniences (`activation`,
# `dropout_rate`, the initializers, the dims) without checking whether the
# selected `ffn_type` accepts them. `create_ffn_layer` discarded the surplus and
# only logged it, so 6 of 21 types were about to become HARD construction
# failures the moment that warning becomes a raise.
#
# The instrument below attaches to the logger named 'dl' -- `utils/logger.py`
# does `logging.getLogger("dl")`, NOT "dl_techniques". A handler on the wrong
# name captures ZERO records and every assertion here would pass vacuously,
# which is why `test_..._harness_bites` exists and must never be deleted.
# ---------------------------------------------------------------------

import logging as _logging

from dl_techniques.layers.ffn.factory import FFN_REGISTRY, assemble_ffn_config
from dl_techniques.layers.transformers.transformer import (
    build_transformer_ffn_config,
)
from dl_techniques.layers.transformers.transformer_decoder import (
    TransformerDecoderLayer,
)

_GRID_HIDDEN = 32
_GRID_HEADS = 4
_GRID_INTER = 64
_ALL_FFN_TYPES = sorted(FFN_REGISTRY)


class _DropRecorder(_logging.Handler):
    """Captures `create_ffn_layer`'s dropped-key warnings off the 'dl' logger."""

    def __init__(self) -> None:
        super().__init__(level=_logging.WARNING)
        self.dropped: list = []

    def emit(self, record: _logging.LogRecord) -> None:
        msg = record.getMessage()
        if 'dropping' in msg and 'unsupported parameter' in msg:
            self.dropped.append(msg)


def _capture_drops(fn):
    """Run `fn` with a recorder attached to 'dl'; return the captured messages."""
    handler = _DropRecorder()
    dl_logger = _logging.getLogger('dl')
    dl_logger.addHandler(handler)
    try:
        fn()
    finally:
        dl_logger.removeHandler(handler)
    return handler.dropped


def _required_ffn_args(ffn_type: str) -> Dict[str, Any]:
    """The minimal caller-supplied `ffn_args` covering `ffn_type`'s required params.

    This is the 'caller-supplies-required' grid condition: the realistic shape a
    caller must use to even reach construction for a type the wrapper does not
    size itself. It is the condition under which the pre-filter matters -- at
    pure site defaults most of these types die earlier in `validate_ffn_config`.
    """
    sized = {
        'hidden_dim': _GRID_INTER,
        'output_dim': _GRID_HIDDEN, 'units': _GRID_HIDDEN,
        'features': _GRID_HIDDEN, 'filters': _GRID_HIDDEN,
    }
    return {
        p: sized.get(p, 8)
        for p in FFN_REGISTRY[ffn_type]['required_params']
    }


class TestFFNTypeGridEncoder:
    """0 of 21 types may lose a key silently, at either grid condition."""

    def _build(self, ffn_type: str, ffn_args: Dict[str, Any]) -> None:
        layer = TransformerLayer(
            hidden_size=_GRID_HIDDEN, num_heads=_GRID_HEADS,
            intermediate_size=_GRID_INTER, ffn_type=ffn_type,
            activation='relu', ffn_args=ffn_args,
        )
        layer(np.zeros((2, 5, _GRID_HIDDEN), dtype='float32'))

    def test_ffn_type_grid_harness_bites(self) -> None:
        """RED-proof the instrument BEFORE trusting any zero it reports."""
        dropped = _capture_drops(
            lambda: self._build('mlp', {'hiden_dim_typo': 3})
        )
        assert any('hiden_dim_typo' in m for m in dropped), (
            f"the 'dl'-logger recorder saw {dropped} for a deliberately "
            f"misspelled ffn_args key. It is blind, so its silence in the "
            f"parametrized tests below would prove nothing."
        )

    def test_ffn_type_grid_covers_every_registry_type(self) -> None:
        """Anti-vacuity: the parametrization must not quietly shrink."""
        assert len(_ALL_FFN_TYPES) == 21, (
            f"FFN_REGISTRY now has {len(_ALL_FFN_TYPES)} types, not 21; "
            f"re-derive the grid numbers in decisions.md D-018 rather than "
            f"editing this assertion"
        )

    @pytest.mark.parametrize('ffn_type', _ALL_FFN_TYPES)
    @pytest.mark.parametrize('condition', ['site-default-only', 'caller-supplies-required'])
    def test_ffn_type_grid_has_no_silent_drops(self, ffn_type, condition) -> None:
        args = (
            {} if condition == 'site-default-only'
            else _required_ffn_args(ffn_type)
        )

        def build():
            try:
                self._build(ffn_type, args)
            except Exception:
                # A RAISE is already loud, so strictness cannot newly break it.
                # Only a construction that SUCCEEDS while losing a key counts.
                pass

        dropped = _capture_drops(build)
        assert dropped == [], (
            f"TransformerLayer(ffn_type={ffn_type!r}) [{condition}] silently "
            f"dropped {dropped}. Once create_ffn_layer raises, this becomes a "
            f"hard construction failure."
        )


class TestFFNArgsSurviveThePreFilter:
    """PRE-MORTEM #3: the pre-filter must NEVER touch the caller's `ffn_args`.

    The pre-filter exists to strip the WRAPPER's generic conveniences. If it is
    ever applied AFTER `ffn_args` is merged in -- the one-line "simplification"
    a future reader will reach for -- it eats the caller's keys too. A valid one
    then stops reaching the FFN, and an invalid one becomes invisible to the
    factory, so the strict raise this plan is building toward can never fire and
    the caller's typo is silently discarded exactly as before.
    """

    def test_valid_caller_key_reaches_the_constructed_ffn(self) -> None:
        """A key the type accepts, that the wrapper never sends, must arrive."""
        # `mlp` accepts `use_bias`; `TransformerLayer` does not emit it.
        layer = TransformerLayer(
            hidden_size=_GRID_HIDDEN, num_heads=_GRID_HEADS,
            intermediate_size=_GRID_INTER, ffn_type='mlp',
            ffn_args={'use_bias': False},
        )
        assert layer.ffn_layer.use_bias is False, (
            "ffn_args={'use_bias': False} did not reach MLPBlock. The "
            "pre-filter ate a CALLER key -- see assemble_ffn_config's D-017 "
            "contract: it must filter only the wrapper's own dict."
        )

    def test_caller_key_overrides_the_wrapper_default(self) -> None:
        """`ffn_args` is merged LAST, so it still wins over our conveniences."""
        layer = TransformerLayer(
            hidden_size=_GRID_HIDDEN, num_heads=_GRID_HEADS,
            intermediate_size=_GRID_INTER, ffn_type='mlp',
            activation='relu', ffn_args={'activation': 'tanh'},
        )
        assert layer.ffn_layer.activation_name == 'tanh'

    def test_invalid_caller_key_stays_visible_to_the_factory(self) -> None:
        """An unaccepted CALLER key must still reach `create_ffn_layer`.

        Today that surfaces as the dropped-key warning; after the factory is
        made strict it becomes the ValueError. Either way the key must not be
        swallowed here -- if the pre-filter removes it, there is nothing left
        for the factory to complain about.
        """
        dropped = _capture_drops(lambda: TransformerLayer(
            hidden_size=_GRID_HIDDEN, num_heads=_GRID_HEADS,
            intermediate_size=_GRID_INTER, ffn_type='mlp',
            ffn_args={'branch_activation': 'relu'},
        ))
        assert any('branch_activation' in m for m in dropped), (
            "a caller's inapplicable ffn_args key produced NO signal. The "
            "pre-filter removed it before the factory could see it, which "
            "defeats the strict raise entirely."
        )

    def test_the_helper_itself_owns_the_merge_order(self) -> None:
        """Unit-level: `assemble_ffn_config` filters arg 2, never arg 3.

        This is the structural guarantee. Because the merge happens INSIDE the
        helper, a call site cannot express the wrong order; only editing the
        helper can.
        """
        out = assemble_ffn_config(
            'mlp',
            {'hidden_dim': 8, 'output_dim': 4, 'branch_activation': 'relu'},
            {'branch_activation': 'tanh', 'totally_bogus': 1},
        )
        # arg 2's inapplicable key WAS filtered ...
        assert 'branch_activation' in out and out['branch_activation'] == 'tanh'
        # ... and arg 3's inapplicable keys were NOT.
        assert out['totally_bogus'] == 1


class TestEncoderDecoderFFNConfigParity:
    """Guard against a THIRD divergent copy of the per-type injection table.

    `TransformerLayer` and `TransformerDecoderLayer` maintained two independent
    copies of it. That is what produced the `differential`/`activation` silent
    drop (D-016) and five decoder-only coverage gaps. Both now call
    `build_transformer_ffn_config`, so this compares them over EVERY registry
    key and names its exceptions instead of hiding them.
    """

    #: Deliberate exceptions, stated rather than silently excluded. There are
    #: none: the two blocks differ only in their attention wiring, and the FFN
    #: sub-layer sees the identical model/inner widths. If a real exception ever
    #: appears, add it HERE with its reason -- do not narrow the parametrization.
    DELIBERATE_EXCEPTIONS: Dict[str, str] = {}

    @staticmethod
    def _comparable(cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize values so two equivalent configs compare equal.

        Keras initializer INSTANCES do not define `__eq__`, so two identically
        configured `GlorotUniform` objects are unequal by identity. Comparing
        raw dicts would make this guard fail for all 21 types regardless of the
        thing it exists to detect -- a guard that is always red is as useless as
        one that is always green.
        """
        return {k: keras.saving.serialize_keras_object(v)
                for k, v in cfg.items()}

    @staticmethod
    def _spy_config(monkeypatch, module, cls, ffn_type: str) -> Dict[str, Any]:
        """Capture the dict the dispatcher hands the FFN factory.

        A spy, not a direct `_get_ffn_config` call: both blocks construct their
        FFN inside `__init__`, and 7 of the 21 types legitimately RAISE there
        (missing a required param the block cannot supply, e.g. `kan`'s
        `features`). Observing the config at the factory boundary compares the
        two dispatchers over ALL 21 types instead of only the ones that build.
        """
        seen: Dict[str, Any] = {}

        def _fake(config):
            seen.update(config)
            return keras.layers.Identity(name=config.get('name'))

        monkeypatch.setattr(module, 'create_ffn_from_config', _fake)
        cls(
            hidden_size=_GRID_HIDDEN, num_heads=_GRID_HEADS,
            intermediate_size=_GRID_INTER, ffn_type=ffn_type,
            activation='relu',
        )
        assert seen, f"the spy captured nothing for {ffn_type!r}"
        return seen

    @pytest.mark.parametrize('ffn_type', _ALL_FFN_TYPES)
    def test_both_dispatchers_emit_the_identical_config(
        self, ffn_type, monkeypatch
    ) -> None:
        import dl_techniques.layers.transformers.transformer as _enc_mod
        import dl_techniques.layers.transformers.transformer_decoder as _dec_mod

        enc_cfg = self._comparable(self._spy_config(
            monkeypatch, _enc_mod, TransformerLayer, ffn_type))
        dec_cfg = self._comparable(self._spy_config(
            monkeypatch, _dec_mod, TransformerDecoderLayer, ffn_type))

        if ffn_type in self.DELIBERATE_EXCEPTIONS:
            pytest.skip(self.DELIBERATE_EXCEPTIONS[ffn_type])

        assert enc_cfg == dec_cfg, (
            f"the encoder and decoder FFN dispatchers disagree for "
            f"ffn_type={ffn_type!r}: encoder-only keys "
            f"{ {k: v for k, v in enc_cfg.items() if dec_cfg.get(k) != v} }, "
            f"decoder-only keys "
            f"{ {k: v for k, v in dec_cfg.items() if enc_cfg.get(k) != v} }. "
            f"Someone re-inlined the table -- see D-018."
        )

    def test_the_policy_lives_in_exactly_one_function(self) -> None:
        """Both dispatchers must delegate, not re-implement."""
        import inspect
        for cls in (TransformerLayer, TransformerDecoderLayer):
            src = inspect.getsource(cls._get_ffn_config)
            assert 'build_transformer_ffn_config' in src, (
                f"{cls.__name__}._get_ffn_config no longer delegates to the "
                f"shared policy function. A second hand-maintained copy of the "
                f"per-type table is exactly what D-016/D-018 removed."
            )


class TestTransformerFFNPolicyPreserved:
    """The three things a pure registry intersection CANNOT express."""

    def _cfg(self, ffn_type: str) -> Dict[str, Any]:
        return build_transformer_ffn_config(
            ffn_type=ffn_type, name='ffn', hidden_size=_GRID_HIDDEN,
            intermediate_size=_GRID_INTER, activation='relu',
            dropout_rate=0.1, kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
        )

    def test_differential_renames_activation_to_branch_activation(self) -> None:
        cfg = self._cfg('differential')
        assert cfg['branch_activation'] == 'relu'
        assert 'activation' not in cfg
        assert 'gate_activation' not in cfg, (
            "the sigmoid gate is DifferentialFFN's defining feature (D-016)"
        )

    @pytest.mark.parametrize('ffn_type', ['reglu', 'bilinear'])
    def test_d005_activation_withholding_survives_the_pre_filter(self, ffn_type) -> None:
        """RED-proof target: these types ACCEPT `activation`, so the filter keeps it.

        `reglu`/`bilinear` list `activation` in `optional_params`, so a pure
        intersection would let the block's generic activation through and defeat
        the alias's fixed gate. Only the explicit withholding stops it.
        """
        assert 'activation' in FFN_REGISTRY[ffn_type]['optional_params'], (
            "anti-vacuity: if the registry stops accepting `activation` for "
            f"{ffn_type}, the pre-filter would drop it anyway and this test "
            "would no longer be testing the withholding"
        )
        assert 'activation' not in self._cfg(ffn_type)

    def test_swiglu_still_sizes_itself(self) -> None:
        """`hidden_dim` is OPTIONAL for swiglu, so the filter would keep it."""
        assert 'hidden_dim' in FFN_REGISTRY['swiglu']['optional_params']
        cfg = self._cfg('swiglu')
        assert 'hidden_dim' not in cfg, (
            "passing hidden_dim overrides swiglu's own 2/3 derivation"
        )
        assert cfg['ffn_expansion_factor'] == 4
        assert cfg['ffn_multiple_of'] == 256

    def test_unknown_type_raises_from_the_helper(self) -> None:
        with pytest.raises(ValueError, match="Unknown ffn_type"):
            self._cfg('no_such_ffn')
