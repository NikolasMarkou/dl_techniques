"""``models/vision/beit/`` -- the BEiT trunk itself: construction, build, forward, round trip.

Once the 1173-line original was decomposed into one behaviour-named file per concern
(plan-2026-08-24T074054-247151fd, step 8), this file keeps the irreducible core -- the
four things that must hold before any head, any variant table row or any warm start
means anything. The concerns that left, and where they went:

* ``TestBeitScaleConfigs`` -> ``test_scale_configs.py``
* ``TestBeitArchitectureValidation`` -> ``test_architecture_invariants.py``
* ``TestBeitForMaskedImageModeling`` -> ``test_masked_image_modeling_head.py``
* ``TestBeitForImageClassification`` -> ``test_image_classification_head.py``
* ``TestBeitWarmStart`` -> ``test_warm_start.py``
* ``TestBeitUnbuiltFit`` -> ``test_unbuilt_fit.py``

The shared small geometry and the ``_tiny`` / ``_images`` / ``_mask`` / ``_mim`` /
``_classifier`` factories moved to ``beit_test_geometry.py``, so the seven files cannot
drift apart on what ``tiny`` means.

The assertions here read the ACTUAL sub-layer objects and weights, never the kwargs
dict that was passed in: a kwarg that never arrived and a kwarg that arrived and was
honoured are indistinguishable from the caller's side. ``create_attention_layer``
used to drop undeclared kwargs SILENTLY; since 2026-08-17
(plan-2026-08-17T183311-79c63e38/D-011) it raises on them, but the reason to read the
objects stands -- a kwarg the registry DECLARES and the constructor never wires is
still invisible to the raise.

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

import json
import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.models.vision.beit import (
    BACKBONE_NAME,
    SCALE_CONFIGS,
    BeitModel,
    create_beit_backbone,
)
from tests.test_models.test_beit.beit_test_geometry import (
    IMG,
    PATCH,
    GRID,
    NUM_PATCHES,
    SEQ_LEN,
    _tiny,
    _images,
    _mask,
)


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
        assert model.hidden_dropout_rate == 0.0
        assert model.attention_probs_dropout_rate == 0.0
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
            (dict(hidden_dropout_rate=-0.1), r"must be in \[0, 1\]"),
            (dict(attention_probs_dropout_rate=2.0), r"must be in \[0, 1\]"),
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

        The message content is asserted too, because README §15.6 (deviation X-5)
        promises an ACTIONABLE refusal: the reference shares one table across all
        layers during pre-training, and stage 1 of this package's own trainer IS
        pre-training, so a caller who hits this needs to be told what is unsupported
        and why — not merely that something is wrong.
        """
        with pytest.raises(ValueError, match="use_shared_relative_position_bias"):
            _tiny(use_shared_relative_position_bias=True)

        with pytest.raises(ValueError) as excinfo:
            _tiny(use_shared_relative_position_bias=True)
        message = str(excinfo.value)
        assert "is not implemented" in message, message
        # ... and it says WHY, naming the shared-block signature change that is out
        # of scope, so the reader does not have to guess whether it is a bug.
        assert "TransformerLayer" in message, message
        assert "out of scope" in message, message

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
            hidden_dropout_rate=0.1,
            attention_probs_dropout_rate=0.05,
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

    def test_from_config_restores_patch_size_tuple_after_a_real_json_cycle(self):
        """``from_config`` earns its place on exactly one field: ``patch_size``.

        The cycle is a REAL one -- ``json.dumps``/``json.loads`` -- because that is what
        strips tuple-ness; handing ``get_config()``'s dict straight back to
        ``from_config`` would pass with or without the override and prove nothing.
        Measured without the override at 59011a9d: ``patch_size`` came back a
        ``keras.src.utils.tracking.TrackedList`` and ``get_config()`` stopped being a
        fixed point. ``input_shape`` is deliberately NOT the subject: ``__init__``
        already coerces it, so that arm is vacuous by measurement.
        """
        model = _tiny(patch_size=(PATCH, PATCH))
        config = model.get_config()
        assert type(config['patch_size']) is tuple

        reloaded = BeitModel.from_config(json.loads(json.dumps(config)))
        assert type(reloaded.patch_size) is tuple, (
            f"patch_size came back as {type(reloaded.patch_size).__name__}"
        )
        # get_config is a fixed point again: round 2 emits what round 1 emitted.
        assert reloaded.get_config()['patch_size'] == config['patch_size']
        assert type(reloaded.get_config()['patch_size']) is tuple

    def test_from_config_leaves_an_int_patch_size_an_int(self):
        """The override must not "helpfully" widen an int into a pair.

        A directly-constructed ``patch_size=16`` model emits ``16``; a deserialized one
        must emit ``16`` too, or two models with identical behaviour disagree on their
        own config.
        """
        config = _tiny(patch_size=PATCH).get_config()
        reloaded = BeitModel.from_config(json.loads(json.dumps(config)))
        assert reloaded.patch_size == PATCH
        assert type(reloaded.patch_size) is int

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
