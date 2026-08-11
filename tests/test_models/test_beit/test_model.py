"""``models/beit/`` -- the BEiT trunk, its variant table and its wiring.

The assertions here read the ACTUAL sub-layer objects and weights, never the kwargs
dict that was passed in: a kwarg that never arrived and a kwarg that arrived and was
honoured are indistinguishable from the caller's side, and
``create_attention_layer`` drops undeclared kwargs SILENTLY.

Two costs are traded deliberately:

* Forward passes, gradients and ``.keras`` round trips run at ``tiny`` and ``small``
  on a 32x32 image (a 2x2 patch grid, 5 tokens). ``base`` is CONSTRUCTED and BUILT and
  its parameter count asserted, but not forwarded. ``large`` (~300M parameters) is
  asserted only at the level of its :data:`SCALE_CONFIGS` row and its constructibility
  is left to the ``base`` path, which exercises the identical code with different
  numbers -- a 1.2 GB build in a unit test buys nothing the ``base`` build does not.
* Everything that must be deterministic passes ``training=False`` EXPLICITLY.
  ``training=None`` is NOT inference for ``StochasticDepth`` -- it short-circuits only
  on ``training is False`` -- and every block here carries a non-zero drop-path rate.
"""

import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.layers.attention.beit_attention import BeitAttention
from dl_techniques.layers.embedding.class_token import ClassTokenPrepend
from dl_techniques.layers.embedding.mask_token import MaskTokenApply
from dl_techniques.layers.transformers import TransformerLayer
from dl_techniques.models.beit import (
    BACKBONE_NAME,
    MODEL_VARIANTS,
    SCALE_CONFIGS,
    BeitModel,
    create_beit_backbone,
)
from dl_techniques.models.beit.model import _resolve_scale

# A 32x32 image at patch 16 -> a 2x2 patch grid -> 4 patches + 1 cls = 5 tokens.
IMG = (32, 32, 3)
PATCH = 16
GRID = (2, 2)
NUM_PATCHES = 4
SEQ_LEN = NUM_PATCHES + 1
EPS = 1e-12


def _tiny(**overrides) -> BeitModel:
    """A `tiny` backbone at the small test geometry."""
    config = dict(input_shape=IMG, patch_size=PATCH, scale='tiny')
    config.update(overrides)
    return BeitModel(**config)


def _images(batch: int = 2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(batch,) + IMG).astype('float32')


def _mask(batch: int = 2, num_patches: int = NUM_PATCHES) -> np.ndarray:
    m = np.zeros((batch, num_patches), dtype=bool)
    m[:, 0] = True
    m[0, -1] = True
    return m


# ==============================================================================
# 1. The variant table
# ==============================================================================

class TestBeitScaleConfigs:
    """`SCALE_CONFIGS` is DATA fetched from primary sources; assert it as data."""

    def test_base_matches_the_hf_config_json_verbatim(self):
        """microsoft/beit-base-patch16-224 config.json, fetched 2026-08-11."""
        assert SCALE_CONFIGS['base']['hidden_size'] == 768
        assert SCALE_CONFIGS['base']['num_layers'] == 12
        assert SCALE_CONFIGS['base']['num_heads'] == 12
        assert SCALE_CONFIGS['base']['intermediate_size'] == 3072

    def test_large_matches_the_hf_config_json_verbatim(self):
        """microsoft/beit-large-patch16-224 config.json, fetched 2026-08-11."""
        assert SCALE_CONFIGS['large']['hidden_size'] == 1024
        assert SCALE_CONFIGS['large']['num_layers'] == 24
        assert SCALE_CONFIGS['large']['num_heads'] == 16
        assert SCALE_CONFIGS['large']['intermediate_size'] == 4096

    def test_layer_scale_init_value_split_is_timms(self):
        """D-003 / X-2: timm's split, NOT HF's uniform 0.1.

        Why this can fail if the implementation is wrong: HF's shipped config.json
        reports 0.1 for BOTH sizes, so "correcting" the large entry to 0.1 to make the
        table agree with HF is a one-character edit that looks like a bug fix. It is
        the recorded deviation, and this pins it.
        """
        assert SCALE_CONFIGS['tiny']['layer_scale_init_value'] == 0.1
        assert SCALE_CONFIGS['small']['layer_scale_init_value'] == 0.1
        assert SCALE_CONFIGS['base']['layer_scale_init_value'] == 0.1
        assert SCALE_CONFIGS['large']['layer_scale_init_value'] == 1e-5

    def test_every_scale_has_a_variant_and_divisible_heads(self):
        assert set(SCALE_CONFIGS) == {'tiny', 'small', 'base', 'large'}
        assert set(MODEL_VARIANTS) == {
            'beit_tiny', 'beit_small', 'beit_base', 'beit_large'
        }
        for scale, cfg in SCALE_CONFIGS.items():
            assert cfg['hidden_size'] % cfg['num_heads'] == 0, scale
            # BEiT's FFN is the standard 4x expansion at every size.
            assert cfg['intermediate_size'] == 4 * cfg['hidden_size'], scale

    @pytest.mark.parametrize("spelling", ['base', 'beit_base'])
    def test_resolve_scale_accepts_both_spellings(self, spelling):
        assert _resolve_scale(spelling) == 'base'

    def test_resolve_scale_rejects_an_unknown_variant(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            _resolve_scale('beit_enormous')


# ==============================================================================
# 2. Construction and validation
# ==============================================================================

class TestBeitModelInitialization:
    """Every invalid configuration must raise at CONSTRUCTION, not at first call."""

    def test_defaults_come_from_the_hf_config(self):
        model = BeitModel()
        assert model.input_shape_config == (224, 224, 3)
        assert model.patch_size == 16
        assert model.layer_norm_eps == 1e-12
        assert model.drop_path_rate == 0.1
        assert model.hidden_dropout_prob == 0.0
        assert model.attention_probs_dropout_prob == 0.0
        assert model.use_absolute_position_embeddings is False
        assert model.use_relative_position_bias is True
        assert model.use_shared_relative_position_bias is False
        assert model.use_mean_pooling is True
        assert model.initializer_range == 0.02
        assert model.name == BACKBONE_NAME
        # 224/16 = 14 -> the paper's 14x14 = 196 patch grid.
        assert model.grid_size == (14, 14)
        assert model.num_patches == 196
        assert model.seq_len == 197

    def test_derived_geometry_at_the_test_size(self):
        model = _tiny()
        assert model.grid_size == GRID
        assert model.num_patches == NUM_PATCHES
        assert model.seq_len == SEQ_LEN

    def test_non_square_grid(self):
        model = BeitModel(input_shape=(32, 64, 3), patch_size=16, scale='tiny')
        assert model.grid_size == (2, 4)
        assert model.num_patches == 8
        for layer in model.encoder_layers:
            assert layer.attention.window_size == (2, 4)

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            (dict(input_shape=(0, 32, 3)), "must be positive"),
            (dict(input_shape=(32, 32)), "3-tuple"),
            (dict(input_shape=(30, 32, 3)), "divisible by patch height"),
            (dict(input_shape=(32, 30, 3)), "divisible by patch width"),
            (dict(patch_size=0), "must be positive"),
            (dict(hidden_size=0), "hidden_size must be positive"),
            (dict(num_layers=0), "num_layers must be positive"),
            (dict(num_heads=0), "num_heads must be positive"),
            (dict(intermediate_size=0), "intermediate_size must be positive"),
            (dict(hidden_size=192, num_heads=5), "divisible by num_heads"),
            (dict(drop_path_rate=1.5), r"drop_path_rate must be in \[0, 1\]"),
            (dict(hidden_dropout_prob=-0.1), r"must be in \[0, 1\]"),
            (dict(attention_probs_dropout_prob=2.0), r"must be in \[0, 1\]"),
            (dict(layer_norm_eps=0.0), "layer_norm_eps must be positive"),
            (dict(initializer_range=0.0), "initializer_range must be positive"),
            (dict(scale='enormous'), "Unknown variant"),
        ],
    )
    def test_invalid_config_raises(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            _tiny(**kwargs)

    def test_shared_relative_position_bias_is_refused_not_ignored(self):
        """G-2: the shared-table mode is out of scope and must say so LOUDLY.

        Why this can fail if the implementation is wrong: silently accepting the flag
        and building per-layer tables anyway would train a different architecture than
        the caller asked for, with no error and no shape change.
        """
        with pytest.raises(ValueError, match="use_shared_relative_position_bias"):
            _tiny(use_shared_relative_position_bias=True)

    def test_a_single_layer_model_does_not_divide_by_zero(self):
        """The drop-path ramp's degenerate case."""
        model = _tiny(num_layers=1, drop_path_rate=0.3)
        assert model.drop_path_rates == [0.0]

    def test_scale_overrides_are_honoured_and_resolved(self):
        model = _tiny(scale='base', hidden_size=64, num_heads=4, num_layers=2)
        assert model.hidden_size == 64
        assert model.num_heads == 4
        assert model.num_layers == 2
        # Un-overridden fields still come from the scale.
        assert model.intermediate_size == SCALE_CONFIGS['base']['intermediate_size']
        assert len(model.encoder_layers) == 2


# ==============================================================================
# 3. Build
# ==============================================================================

class TestBeitModelBuild:

    def test_build_marks_every_sublayer_built(self):
        model = _tiny()
        model.build((None,) + IMG)
        assert model.built
        assert model.patch_embed.built
        assert model.cls_token.built
        assert model.embed_dropout.built
        for layer in model.encoder_layers:
            assert layer.built

    def test_build_is_idempotent(self):
        model = _tiny()
        model.build((None,) + IMG)
        n = model.count_params()
        model.build((None,) + IMG)
        assert model.count_params() == n

    def test_base_variant_builds_at_the_expected_parameter_count(self):
        """`base` is built (not forwarded) so the 768d/12L path is really exercised."""
        model = create_beit_backbone('base', IMG, PATCH)
        model.build((None,) + IMG)
        # 12 blocks x (4*768^2 attention + 2*768*3072 FFN) = 84,934,656 weights of
        # kernel alone; the assertion below is on the whole trunk, derived by hand
        # from the shipped configuration rather than transcribed from a run.
        d, ffn, layers_n, heads = 768, 3072, 12, 12
        # The cls-augmented relative-position table: (2Wh-1)(2Ww-1)+3 rows per head.
        bias_table = ((2 * GRID[0] - 1) * (2 * GRID[1] - 1) + 3) * heads
        per_block = (
            4 * d * d          # q, k, v, proj kernels
            + 3 * d            # q, v, proj biases (k has NO bias -- BEiT)
            + bias_table       # the relative-position bias table
            + 2 * d            # attention_norm gamma+beta
            + 2 * d            # output_norm gamma+beta
            + 2 * d            # two LayerScale gammas
            + d * ffn + ffn    # FFN up
            + ffn * d + d      # FFN down
        )
        patch_embed = PATCH * PATCH * 3 * d + d
        expected = layers_n * per_block + patch_embed + d + d  # + cls + mask tokens
        assert model.count_params() == expected


# ==============================================================================
# 4. Forward pass
# ==============================================================================

class TestBeitModelForward:

    @pytest.mark.parametrize("variant", ['tiny', 'small'])
    def test_output_shape(self, variant):
        model = create_beit_backbone(variant, IMG, PATCH)
        out = model(_images(), training=False)
        assert tuple(out.shape) == (2, SEQ_LEN, SCALE_CONFIGS[variant]['hidden_size'])

    def test_output_is_finite(self):
        out = ops.convert_to_numpy(_tiny()(_images(), training=False))
        assert np.all(np.isfinite(out))

    def test_masked_and_unmasked_forwards_differ(self):
        """The mask must MOVE the output, not merely be accepted.

        Why this can fail if the implementation is wrong: a backbone that accepted the
        mask and then never called ``MaskTokenApply`` -- or called it with the polarity
        flipped on an all-False mask -- would return a perfectly healthy tensor.
        """
        model = _tiny()
        x = _images()
        unmasked = ops.convert_to_numpy(model(x, training=False))
        masked = ops.convert_to_numpy(model((x, _mask()), training=False))
        assert masked.shape == unmasked.shape
        assert np.all(np.isfinite(masked))
        assert not np.allclose(masked, unmasked, atol=1e-6)

    def test_an_all_false_mask_is_a_no_op(self):
        """Polarity control: True means REPLACE, so all-False must change nothing."""
        model = _tiny()
        x = _images()
        unmasked = ops.convert_to_numpy(model(x, training=False))
        no_mask = ops.convert_to_numpy(
            model((x, np.zeros((2, NUM_PATCHES), dtype=bool)), training=False)
        )
        np.testing.assert_allclose(unmasked, no_mask, atol=1e-6, rtol=0)

    def test_fully_and_zero_masked_grids_are_finite(self):
        model = _tiny()
        x = _images()
        for m in (
                np.ones((2, NUM_PATCHES), dtype=bool),
                np.zeros((2, NUM_PATCHES), dtype=bool),
        ):
            out = ops.convert_to_numpy(model((x, m), training=False))
            assert np.all(np.isfinite(out))

    def test_dict_and_tuple_inputs_agree(self):
        model = _tiny()
        x, m = _images(), _mask()
        a = ops.convert_to_numpy(model((x, m), training=False))
        b = ops.convert_to_numpy(model({'images': x, 'mask': m}, training=False))
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=0)

    def test_a_malformed_sequence_input_raises(self):
        model = _tiny()
        with pytest.raises(ValueError, match="sequence of length 3"):
            model((_images(), _mask(), _mask()), training=False)

    def test_training_true_and_false_differ_under_stochastic_depth(self):
        """`training=None` is NOT inference; only `training=False` is deterministic."""
        model = _tiny(drop_path_rate=0.9)
        x = _images(batch=8)
        a = ops.convert_to_numpy(model(x, training=False))
        b = ops.convert_to_numpy(model(x, training=False))
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=0)
        train_out = ops.convert_to_numpy(model(x, training=True))
        assert not np.allclose(train_out, a, atol=1e-6)

    def test_compute_output_shape(self):
        model = _tiny()
        assert model.compute_output_shape((None,) + IMG) == (None, SEQ_LEN, 192)
        assert model.compute_output_shape((4,) + IMG) == (4, SEQ_LEN, 192)
        assert model.compute_output_shape(
            [(4,) + IMG, (4, NUM_PATCHES)]
        ) == (4, SEQ_LEN, 192)

    def test_gradients_reach_every_trainable_weight(self):
        # drop_path_rate=0.0: with a live ramp, StochasticDepth can drop a whole
        # block for every sample in a small batch and zero that block's gradient.
        # That is the layer working correctly, so it must not be able to make this
        # assertion flaky.
        model = _tiny(drop_path_rate=0.0)
        model.build((None,) + IMG)
        x = tf.constant(_images())
        mask = tf.constant(_mask())
        with tf.GradientTape() as tape:
            out = model((x, mask), training=True)
            loss = tf.reduce_mean(tf.square(out))
        grads = tape.gradient(loss, model.trainable_variables)
        assert len(grads) == len(model.trainable_variables)
        dead = [
            v.path for g, v in zip(grads, model.trainable_variables) if g is None
        ]
        assert dead == [], f"no gradient reached: {dead}"

    def test_the_relative_position_bias_tables_receive_gradient(self):
        """The signature weight of this architecture must not be inert."""
        model = _tiny(drop_path_rate=0.0)  # see the note above on flakiness
        model.build((None,) + IMG)
        tables = [
            layer.attention.relative_position_bias_table
            for layer in model.encoder_layers
        ]
        assert all(t is not None for t in tables)
        x = tf.constant(_images())
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(model(x, training=True)))
        grads = tape.gradient(loss, tables)
        for i, g in enumerate(grads):
            assert g is not None, f"encoder_layer_{i} bias table has no gradient"
            assert float(tf.reduce_max(tf.abs(g))) > 0.0, f"encoder_layer_{i} zero grad"


# ==============================================================================
# 5. Serialization
# ==============================================================================

class TestBeitModelSerialization:

    def test_get_config_round_trips_every_constructor_param(self):
        model = _tiny(
            drop_path_rate=0.2,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.05,
            use_absolute_position_embeddings=True,
            use_mean_pooling=False,
            layer_norm_eps=1e-10,
            initializer_range=0.01,
        )
        config = model.get_config()
        clone = BeitModel.from_config(config)
        for key, value in config.items():
            if key in ('name', 'trainable', 'dtype'):
                continue
            assert getattr(clone, 'input_shape_config' if key == 'input_shape' else key) == value

    def test_keras_roundtrip_preserves_values(self):
        """Shapes/counts agreeing is NOT evidence; compare the OUTPUT values."""
        model = _tiny()
        model.build((None,) + IMG)
        x = _images()
        m = _mask()
        before = ops.convert_to_numpy(model((x, m), training=False))

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "beit_backbone.keras")
            model.save(path)
            restored = keras.models.load_model(path)
            after = ops.convert_to_numpy(restored((x, m), training=False))

        np.testing.assert_allclose(before, after, atol=1e-6, rtol=0)

    def test_keras_roundtrip_preserves_the_bias_tables_elementwise(self):
        """The nested-sub-layer-list trap: counts/paths/params all match while the
        restored kernels are FRESH. Only a value diff sees it."""
        model = _tiny()
        model.build((None,) + IMG)
        saved = [
            ops.convert_to_numpy(l.attention.relative_position_bias_table)
            for l in model.encoder_layers
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "beit_backbone.keras")
            model.save(path)
            restored = keras.models.load_model(path)
        loaded = [
            ops.convert_to_numpy(l.attention.relative_position_bias_table)
            for l in restored.encoder_layers
        ]
        assert len(loaded) == len(saved) == model.num_layers
        for i, (a, b) in enumerate(zip(saved, loaded)):
            np.testing.assert_array_equal(a, b, err_msg=f"encoder_layer_{i}")

    def test_keras_roundtrip_preserves_the_patch_embedding_kernel(self):
        """A second, non-attention weight -- so the check above is not the only one."""
        model = _tiny()
        model.build((None,) + IMG)
        saved = ops.convert_to_numpy(model.patch_embed.proj.kernel)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "beit_backbone.keras")
            model.save(path)
            restored = keras.models.load_model(path)
        np.testing.assert_array_equal(
            saved, ops.convert_to_numpy(restored.patch_embed.proj.kernel)
        )


# ==============================================================================
# 6. Architecture validation -- read the objects, not the config
# ==============================================================================

class TestBeitArchitectureValidation:
    """What the composition ACTUALLY wired, asserted by inspection."""

    def test_every_block_is_a_transformer_layer_with_beit_attention(self):
        model = _tiny()
        assert len(model.encoder_layers) == model.num_layers == 12
        for i, layer in enumerate(model.encoder_layers):
            assert isinstance(layer, TransformerLayer), i
            assert layer.name == f"encoder_layer_{i}"
            assert isinstance(layer.attention, BeitAttention), i
            assert layer.attention.window_size == GRID
            assert layer.attention.num_tokens == SEQ_LEN
            assert layer.attention.use_relative_position_bias is True
            # BEiT's asymmetric bias survived the whole factory chain.
            assert layer.attention.q_dense.use_bias is True
            assert layer.attention.k_dense.use_bias is False
            assert layer.attention.v_dense.use_bias is True

    def test_the_encoder_layer_list_is_stored_flat(self):
        """`List[List[Layer]]` loses weights on a `.keras` round trip while every
        structural check still passes."""
        model = _tiny()
        assert isinstance(model.encoder_layers, list)
        assert all(
            isinstance(l, TransformerLayer) for l in model.encoder_layers
        ), "encoder_layers must be a FLAT list of layers"

    def test_every_norm_epsilon_is_1e_12(self):
        model = _tiny()
        model.build((None,) + IMG)
        for i, layer in enumerate(model.encoder_layers):
            assert layer.attention_norm.epsilon == EPS, i
            assert layer.output_norm.epsilon == EPS, i

    def test_the_epsilon_assertion_is_falsifiable(self):
        """Control: a model asked for a different epsilon must report it.

        Why this can fail if the implementation is wrong: if the epsilon were being
        ignored, every norm would carry the factory default and the assertion above
        would be pinning that default rather than the config. These two models must
        disagree.
        """
        other = _tiny(layer_norm_eps=1e-7)
        other.build((None,) + IMG)
        assert other.encoder_layers[0].attention_norm.epsilon == 1e-7
        assert other.encoder_layers[0].output_norm.epsilon == 1e-7

    def test_normalization_is_pre_norm(self):
        for layer in _tiny().encoder_layers:
            assert layer.normalization_position == 'pre'
            assert layer.normalization_type == 'layer_norm'
            assert layer.ffn_type == 'mlp'

    def test_layer_scale_is_on_at_the_scale_value_and_signed(self):
        model = _tiny(scale='large', num_layers=2)
        model.build((None,) + IMG)
        for layer in model.encoder_layers:
            assert layer.use_layer_scale is True
            assert layer.layer_scale_init_value == 1e-5
            # BEiT's gamma is a SIGNED scale; LearnableMultiplier's own default is a
            # non_neg constraint, which TransformerLayer overrides with None.
            assert layer.attention_layer_scale.constraint is None

    def test_stochastic_depth_is_the_exact_linear_ramp(self):
        """Assert the WHOLE list, not just that the rates differ.

        Why this can fail if the implementation is wrong: a reversed ramp, a constant
        rate, or an off-by-one denominator all produce a list of distinct-looking
        floats that trains fine and is wrong. The oracle below is the schedule
        transcribed independently (linear 0 -> drop_path_rate over num_layers).
        """
        rate = 0.1
        model = _tiny(drop_path_rate=rate)
        n = model.num_layers
        expected = [round(i * (rate / (n - 1)), 6) for i in range(n)]
        assert expected[0] == 0.0
        assert expected[-1] == pytest.approx(rate)
        assert model.drop_path_rates == expected
        actual = [l.stochastic_depth_rate for l in model.encoder_layers]
        assert actual == expected
        # And the rate really reached the sub-layer, not just the block's attribute.
        assert [
            l.attention_stochastic_depth.drop_path_rate for l in model.encoder_layers
        ] == expected
        for l in model.encoder_layers:
            assert l.use_stochastic_depth is True

    def test_mask_token_is_present_and_built_even_on_an_unmasked_forward(self):
        """The warm-start contract: ALWAYS CREATE, CONDITIONALLY USE.

        Why this can fail if the implementation is wrong: dropping the "dead" mask
        token from a backbone that never masks makes the classifier's trunk a
        different layer set from the MIM trunk, and the warm start then transfers a
        strict subset with no error at all.
        """
        model = _tiny()
        model(_images(), training=False)  # never passes a mask
        assert isinstance(model.mask_token, MaskTokenApply)
        assert model.mask_token.built
        assert model.mask_token.mask_token is not None
        assert tuple(model.mask_token.mask_token.shape) == (1, 1, model.hidden_size)
        assert "mask_token" in {l.name for l in model.layers}

    def test_cls_token_is_a_class_token_prepend(self):
        model = _tiny()
        model.build((None,) + IMG)
        assert isinstance(model.cls_token, ClassTokenPrepend)
        assert tuple(model.cls_token.cls_token.shape) == (1, 1, model.hidden_size)

    def test_absolute_position_embedding_is_off_by_default(self):
        """BEiT uses RELATIVE bias; the absolute table must not exist by default."""
        assert _tiny().pos_embed is None
        enabled = _tiny(use_absolute_position_embeddings=True)
        assert enabled.pos_embed is not None
        enabled.build((None,) + IMG)
        # And it actually changes the forward pass.
        x = _images()
        a = ops.convert_to_numpy(_tiny()(x, training=False))
        b = ops.convert_to_numpy(enabled(x, training=False))
        assert not np.allclose(a, b, atol=1e-6)

    def test_final_norm_follows_the_mean_pooling_fork(self):
        """D-007: `use_mean_pooling=True` -> the trunk applies NO final norm.

        Why this can fail if the implementation is wrong: always applying a final norm
        here is the obvious "cleanup", and at the default config it silently inserts a
        normalization the reference does not have in front of BOTH heads -- no error,
        no shape change, a plausible loss curve.
        """
        pooled = _tiny(use_mean_pooling=True)
        assert pooled.final_norm is None
        assert "final_norm" not in {l.name for l in pooled.layers}

        cls_mode = _tiny(use_mean_pooling=False)
        assert cls_mode.final_norm is not None
        assert cls_mode.final_norm.epsilon == EPS
        cls_mode.build((None,) + IMG)
        assert cls_mode.final_norm.built
        # The fork is observable in the OUTPUT, not just in the layer list: a normed
        # sequence has ~unit variance along the feature axis.
        out = ops.convert_to_numpy(cls_mode(_images(), training=False))
        assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-4)
        assert np.allclose(out.std(axis=-1), 1.0, atol=1e-2)

    def test_the_backbone_is_named_for_the_warm_start(self):
        assert create_beit_backbone('tiny', IMG, PATCH).name == BACKBONE_NAME
        assert BeitModel.from_variant('beit_tiny', IMG, PATCH).name == BACKBONE_NAME

    def test_from_variant_and_the_factory_agree(self):
        a = BeitModel.from_variant('beit_small', IMG, PATCH)
        b = create_beit_backbone('small', IMG, PATCH)
        for key in ('hidden_size', 'num_layers', 'num_heads', 'intermediate_size',
                    'layer_scale_init_value', 'scale'):
            assert getattr(a, key) == getattr(b, key)
