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
    DEFAULT_VOCAB_SIZE,
    MODEL_VARIANTS,
    SCALE_CONFIGS,
    BeitForImageClassification,
    BeitForMaskedImageModeling,
    BeitModel,
    create_beit_backbone,
    create_beit_classifier,
    create_beit_mim,
)
from dl_techniques.models.beit.model import _resolve_scale
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint

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


# ==============================================================================
# 7. The masked-image-modeling head
# ==============================================================================

VOCAB = 64  # a toy codebook; the real default is DEFAULT_VOCAB_SIZE == 8192


def _mim(variant: str = 'tiny', **overrides) -> BeitForMaskedImageModeling:
    return create_beit_mim(variant, IMG, PATCH, vocab_size=VOCAB, **overrides)


def _classifier(variant: str = 'tiny', num_classes: int = 7, **overrides):
    return create_beit_classifier(
        variant, IMG, PATCH, num_classes=num_classes, **overrides
    )


class TestBeitForMaskedImageModeling:

    def test_default_vocab_size_is_the_dalle_codebook(self):
        assert DEFAULT_VOCAB_SIZE == 8192
        assert _mim().backbone.name == BACKBONE_NAME

    @pytest.mark.parametrize("variant", ['tiny', 'small'])
    def test_output_shape_excludes_the_cls_position(self, variant):
        """(B, N, vocab) -- NOT (B, N+1, vocab).

        Why this can fail if the implementation is wrong: forgetting the ``[:, 1:, :]``
        slice yields ``N + 1`` logits, which still trains against an ``(B, N)`` target
        only if the loss silently broadcasts -- otherwise it puts every target off by
        one patch. Either way there is no architectural error to see.
        """
        model = _mim(variant)
        out = model((_images(), _mask()), training=False)
        assert tuple(out.shape) == (2, NUM_PATCHES, VOCAB)
        assert model.compute_output_shape(
            [(2,) + IMG, (2, NUM_PATCHES)]
        ) == (2, NUM_PATCHES, VOCAB)

    def test_forward_without_a_mask_is_accepted(self):
        out = _mim()(_images(), training=False)
        assert tuple(out.shape) == (2, NUM_PATCHES, VOCAB)

    def test_output_is_logits_not_probabilities(self):
        """BOTH halves: a value outside [0, 1] is reachable AND no softmax exists."""
        model = _mim()
        model.build((None,) + IMG)
        # Zero the kernel and PIN the bias, so the head's output is exactly the bias.
        # A constant kernel would NOT work: `decoder_norm`'s output is zero-mean over
        # the feature axis, so a constant kernel maps every token to ~0.0 -- inside
        # [0, 1] up to float noise, which is a coin flip, not a test.
        pinned = np.linspace(-5.0, 5.0, VOCAB).astype('float32')
        model.decoder_head.set_weights([
            np.zeros_like(ops.convert_to_numpy(model.decoder_head.kernel)),
            pinned,
        ])
        out = ops.convert_to_numpy(model(_images(), training=False))
        np.testing.assert_allclose(
            out, np.broadcast_to(pinned, out.shape), atol=1e-5, rtol=0
        )
        assert out.min() < 0.0 and out.max() > 1.0, "head does not emit logits"
        # And structurally: nothing in the head applies a softmax.
        assert model.decoder_head.activation is keras.activations.linear
        for layer in model._flatten_layers(include_self=False):
            assert not isinstance(layer, keras.layers.Softmax), layer.name
            act = getattr(layer, 'activation', None)
            assert act is not keras.activations.softmax, layer.name
        # Nor does the output already sum to 1 over the vocab axis.
        probs = ops.convert_to_numpy(
            keras.activations.softmax(model(_images(), training=False))
        )
        assert not np.allclose(out.sum(axis=-1), 1.0, atol=1e-3)
        np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-5)
        assert not np.allclose(out, probs, atol=1e-3), (
            "the head output is already a probability distribution"
        )

    def test_head_layers_all_carry_the_decoder_prefix(self):
        model = _mim()
        model.build((None,) + IMG)
        head_names = {l.name for l in model.layers} - {BACKBONE_NAME}
        assert head_names == {"decoder_norm", "decoder_head"}
        assert all(n.startswith("decoder_") for n in head_names)

    def test_decoder_norm_uses_the_backbone_epsilon(self):
        model = _mim()
        assert model.decoder_norm.epsilon == EPS

    def test_gradients_reach_the_head_and_the_trunk(self):
        model = _mim(drop_path_rate=0.0)
        model.build((None,) + IMG)
        x, m = tf.constant(_images()), tf.constant(_mask())
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(model((x, m), training=True)))
        grads = tape.gradient(loss, model.trainable_variables)
        dead = [v.path for g, v in zip(grads, model.trainable_variables) if g is None]
        assert dead == [], f"no gradient reached: {dead}"

    def test_invalid_vocab_size_raises(self):
        with pytest.raises(ValueError, match="vocab_size must be a positive integer"):
            BeitForMaskedImageModeling(backbone=_tiny(), vocab_size=0)

    def test_a_non_backbone_is_refused(self):
        with pytest.raises(TypeError, match="backbone must be a BeitModel"):
            BeitForMaskedImageModeling(backbone="not a model")

    def test_keras_roundtrip_preserves_values(self):
        model = _mim()
        model.build((None,) + IMG)
        x, m = _images(), _mask()
        before = ops.convert_to_numpy(model((x, m), training=False))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "beit_mim.keras")
            model.save(path)
            restored = keras.models.load_model(path)
            after = ops.convert_to_numpy(restored((x, m), training=False))
        np.testing.assert_allclose(before, after, atol=1e-6, rtol=0)
        assert restored.vocab_size == VOCAB
        assert restored.backbone.name == BACKBONE_NAME


# ==============================================================================
# 8. The classification head
# ==============================================================================

class TestBeitForImageClassification:

    @pytest.mark.parametrize("variant", ['tiny', 'small'])
    def test_output_shape(self, variant):
        model = _classifier(variant)
        out = model(_images(), training=False)
        assert tuple(out.shape) == (2, 7)
        assert model.compute_output_shape((2,) + IMG) == (2, 7)

    def test_output_is_logits_not_probabilities(self):
        model = _classifier()
        model.build((None,) + IMG)
        # Same reasoning as the MIM head: `head_norm` is zero-mean over the feature
        # axis, so a constant kernel produces ~0.0 and the [0,1] check becomes a coin
        # flip. Zero the kernel and pin the bias instead.
        pinned = np.linspace(-5.0, 5.0, 7).astype('float32')
        model.head_classifier.set_weights([
            np.zeros_like(ops.convert_to_numpy(model.head_classifier.kernel)),
            pinned,
        ])
        out = ops.convert_to_numpy(model(_images(), training=False))
        np.testing.assert_allclose(
            out, np.broadcast_to(pinned, out.shape), atol=1e-5, rtol=0
        )
        assert out.min() < 0.0 and out.max() > 1.0, "head does not emit logits"
        assert model.head_classifier.activation is keras.activations.linear
        for layer in model._flatten_layers(include_self=False):
            assert not isinstance(layer, keras.layers.Softmax), layer.name
            act = getattr(layer, 'activation', None)
            assert act is not keras.activations.softmax, layer.name
        assert not np.allclose(out.sum(axis=-1), 1.0, atol=1e-3)

    def test_mean_pooling_excludes_the_cls_token(self):
        """A-7 / BEiT's own convention: pool the PATCH tokens only.

        Why this can fail if the implementation is wrong: pooling over all N+1 tokens
        is a one-character change (`exclude_positions=[]`) that changes nothing
        observable in a shape or a loss curve. Here the cls-token weight is perturbed
        and the pooled representation must NOT move.
        """
        model = _classifier(dropout_rate=0.0)
        model.build((None,) + IMG)
        assert model.head_pool.exclude_positions == [0]
        assert model.head_pool.strategy == ['mean']

        x = _images()
        # Pool the trunk output directly: perturbing position 0 must be invisible.
        tokens = ops.convert_to_numpy(model.backbone(x, training=False))
        pooled_a = ops.convert_to_numpy(model.head_pool(tokens))
        tokens_b = tokens.copy()
        tokens_b[:, 0, :] += 100.0
        pooled_b = ops.convert_to_numpy(model.head_pool(tokens_b))
        np.testing.assert_allclose(pooled_a, pooled_b, atol=1e-5, rtol=0)
        # Control: perturbing a PATCH position must move it.
        tokens_c = tokens.copy()
        tokens_c[:, 1, :] += 100.0
        assert not np.allclose(
            pooled_a, ops.convert_to_numpy(model.head_pool(tokens_c)), atol=1e-3
        )

    def test_cls_pooling_mode_has_no_head_norm(self):
        """D-007's other branch: use_mean_pooling=False -> the trunk norms, not us."""
        model = _classifier(use_mean_pooling=False)
        model.build((None,) + IMG)
        assert model.head_pool is None
        assert model.head_norm is None
        assert model.backbone.final_norm is not None
        assert tuple(model(_images(), training=False).shape) == (2, 7)
        names = {l.name for l in model.layers}
        assert "head_norm" not in names
        assert "head_pool" not in names

    def test_head_layers_all_carry_the_head_prefix(self):
        model = _classifier()
        model.build((None,) + IMG)
        head_names = {l.name for l in model.layers} - {BACKBONE_NAME}
        assert head_names == {
            "head_pool", "head_norm", "head_dropout", "head_classifier"
        }
        assert all(n.startswith("head_") for n in head_names)

    def test_head_norm_uses_the_backbone_epsilon(self):
        assert _classifier().head_norm.epsilon == EPS

    def test_gradients_reach_everything_except_the_dead_mask_token(self):
        """The mask token is the ONE weight with no gradient here -- by design.

        The classifier never calls ``MaskTokenApply``, so its mask token receives no
        gradient; it exists only to keep this trunk weight-identical to the MIM trunk.
        Asserting it is the SOLE exception turns that contract into a positive claim
        rather than a blanket exemption: any OTHER unreachable weight still fails.
        """
        model = _classifier(drop_path_rate=0.0)
        model.build((None,) + IMG)
        x = tf.constant(_images())
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(model(x, training=True)))
        grads = tape.gradient(loss, model.trainable_variables)
        dead = [v.path for g, v in zip(grads, model.trainable_variables) if g is None]
        assert len(dead) == 1, f"unexpected dead weights: {dead}"
        assert dead[0].endswith("mask_token/mask_token"), dead

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            (dict(num_classes=0), "num_classes must be a positive integer"),
            (dict(num_classes=3, dropout_rate=1.5), r"dropout_rate must be in \[0, 1\]"),
        ],
    )
    def test_invalid_config_raises(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            BeitForImageClassification(backbone=_tiny(), **kwargs)

    def test_keras_roundtrip_preserves_values(self):
        model = _classifier()
        model.build((None,) + IMG)
        x = _images()
        before = ops.convert_to_numpy(model(x, training=False))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "beit_classifier.keras")
            model.save(path)
            restored = keras.models.load_model(path)
            after = ops.convert_to_numpy(restored(x, training=False))
        np.testing.assert_allclose(before, after, atol=1e-6, rtol=0)
        assert restored.num_classes == 7
        assert restored.backbone.name == BACKBONE_NAME


# ==============================================================================
# 9. SC-10 -- the MIM -> classifier warm start
# ==============================================================================

class TestBeitWarmStart:
    """The property the whole two-head prefix discipline exists to deliver."""

    def test_the_two_heads_use_disjoint_prefixes(self):
        """Assert the property DIRECTLY, not via its consequence.

        Why this can fail if the implementation is wrong: any `head_`-prefixed layer
        inside the MIM model (or `decoder_` inside the classifier) would be silently
        skipped by the OTHER model's transfer, and the symptom would be a partially
        random trunk rather than an error.
        """
        mim = _mim()
        clf = _classifier()
        mim.build((None,) + IMG)
        clf.build((None,) + IMG)

        mim_head = {l.name for l in mim.layers} - {BACKBONE_NAME}
        clf_head = {l.name for l in clf.layers} - {BACKBONE_NAME}
        assert mim_head and clf_head
        assert mim_head.isdisjoint(clf_head)
        assert not any(n.startswith("head_") for n in mim_head)
        assert not any(n.startswith("decoder_") for n in clf_head)
        # And nothing inside the shared trunk claims either prefix.
        trunk_names = {l.name for l in mim.backbone.layers}
        assert not any(
            n.startswith(("head_", "decoder_")) for n in trunk_names
        ), trunk_names

    def test_the_trunks_are_weight_identical_in_structure(self):
        """Including the mask token, which the classifier never calls."""
        mim = _mim()
        clf = _classifier()
        mim.build((None,) + IMG)
        clf.build((None,) + IMG)
        mim_w = [tuple(w.shape) for w in mim.backbone.get_weights()]
        clf_w = [tuple(w.shape) for w in clf.backbone.get_weights()]
        assert mim_w == clf_w
        assert clf.backbone.mask_token.built
        assert clf.backbone.mask_token.mask_token is not None

    def test_mim_to_classifier_transfers_the_trunk_values(self):
        """SC-10. A ZERO-LAYER transfer must FAIL this test, not pass it."""
        mim = _mim()
        mim.build((None,) + IMG)

        clf = _classifier()
        # H-12: the TARGET must be built BEFORE the transfer.
        clf.build((None,) + IMG)
        assert clf.built

        source = [ops.convert_to_numpy(w) for w in mim.backbone.get_weights()]
        before = [ops.convert_to_numpy(w) for w in clf.backbone.get_weights()]
        # Precondition: the two trunks start DIFFERENT, otherwise "equal after" is
        # vacuous (both are randomly initialized, so this is a real check).
        assert any(
            not np.allclose(a, b) for a, b in zip(source, before)
        ), "trunks were already identical -- the transfer assertion would be vacuous"

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "beit_mim.keras")
            mim.save(path)
            report = load_weights_from_checkpoint(
                target=clf,
                ckpt_path=path,
                skip_prefixes=("decoder_", "head_"),
            )

        # (b) the report shows the backbone ACTUALLY loaded -- a zero-layer transfer
        # would leave `loaded` empty and this is what makes the test non-vacuous.
        assert BACKBONE_NAME in report.loaded, report.summary_string()
        assert report.num_loaded >= 1
        assert BACKBONE_NAME not in [name for name, _, _ in report.shape_mismatch]
        assert BACKBONE_NAME not in report.missing_in_source
        assert set(report.skipped_by_prefix) == {"decoder_norm", "decoder_head"}

        # (a) trunk weight VALUES are equal post-transfer.
        after = [ops.convert_to_numpy(w) for w in clf.backbone.get_weights()]
        assert len(after) == len(source)
        for i, (a, b) in enumerate(zip(source, after)):
            np.testing.assert_array_equal(a, b, err_msg=f"trunk weight {i}")

    def test_the_classifier_head_is_not_touched_by_the_transfer(self):
        mim = _mim()
        mim.build((None,) + IMG)
        clf = _classifier()
        clf.build((None,) + IMG)
        head_before = ops.convert_to_numpy(clf.head_classifier.kernel)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "beit_mim.keras")
            mim.save(path)
            load_weights_from_checkpoint(
                target=clf, ckpt_path=path, skip_prefixes=("decoder_", "head_")
            )
        np.testing.assert_array_equal(
            head_before, ops.convert_to_numpy(clf.head_classifier.kernel)
        )

    def test_a_mismatched_backbone_config_is_reported_not_silently_loaded(self):
        """A trunk of a different width must NOT quietly train from scratch."""
        mim = _mim('tiny')
        mim.build((None,) + IMG)
        clf = _classifier('small')          # 384d trunk vs the checkpoint's 192d
        clf.build((None,) + IMG)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "beit_mim.keras")
            mim.save(path)
            report = load_weights_from_checkpoint(
                target=clf, ckpt_path=path, skip_prefixes=("decoder_", "head_")
            )
        assert BACKBONE_NAME not in report.loaded
        assert BACKBONE_NAME in [name for name, _, _ in report.shape_mismatch]
