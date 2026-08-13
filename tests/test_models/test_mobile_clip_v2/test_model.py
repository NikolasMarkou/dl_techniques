"""
Test suite for `MobileClipV2Model` (the MobileCLIP2 dual encoder).

The ten mandated pins:

1.  `test_variant_table_transcription` — all 6 rows, all fields, against literals
    written HERE. Includes `use_causal_mask`, which is the ONLY reason both the
    `mobileclip2_*` and the `mobileclip_*` families are in the table.
2.  `test_output_dict_contract` — keys and shapes.
3.  `test_features_are_l2_normalized` — and that `normalize=False` differs.
4.  `test_logit_scale_is_clipped` — a raw weight of 50.0 must yield exactly
    `logit_scale_max`, not `exp(50)`.
5.  `test_logits_are_transposes`.
6.  `test_from_variant_equals_factory` — all 6 variants (config only, never built).
7.  `test_gradients_reach_every_trainable_weight` — BOTH towers, failures named.
8.  `test_keras_roundtrip_preserves_values` — plus an ELEMENTWISE weight check on
    one weight from EACH tower.
9.  `test_causal_mask_flag_changes_text_output` — with TRANSPLANTED identical
    text-tower weights, so the difference can only come from the flag.
10. `test_text_tower_widths` — 768/12/3072 and 512/8/2048.

Every model except the pure-config ones (1, 6) uses a REDUCED-DEPTH image tower:
a full mci4 does not fit on a 12GB GPU alongside other work.
"""

import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.models.mobile_clip_v2.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_VOCAB_SIZE,
    MODEL_VARIANTS,
    MobileClipV2Model,
    _resolve_model_variant,
    create_mobile_clip_v2,
)


# ---------------------------------------------------------------------
# The six supplied JSON configs, transcribed a SECOND time here.
#
# `use_causal_mask` is `not no_causal_mask`: the MobileCLIP2 series sets
# `"no_causal_mask": true` (bidirectional text tower), the earlier MobileCLIP
# S3/S4 configs leave it false (classic causal CLIP text tower).
# ---------------------------------------------------------------------

_REFERENCE_VARIANTS = {
    'mobileclip2_s0': {
        'embed_dim': 512,
        'image_backbone': 'mci0',
        'text_width': 512,
        'text_heads': 8,
        'text_layers': 12,
        'use_causal_mask': False,
    },
    'mobileclip2_s2': {
        'embed_dim': 512,
        'image_backbone': 'mci2',
        'text_width': 512,
        'text_heads': 8,
        'text_layers': 12,
        'use_causal_mask': False,
    },
    'mobileclip2_s3': {
        'embed_dim': 768,
        'image_backbone': 'mci3',
        'text_width': 768,
        'text_heads': 12,
        'text_layers': 12,
        'use_causal_mask': False,
    },
    'mobileclip2_s4': {
        'embed_dim': 768,
        'image_backbone': 'mci4',
        'text_width': 768,
        'text_heads': 12,
        'text_layers': 12,
        'use_causal_mask': False,
    },
    'mobileclip_s3': {
        'embed_dim': 768,
        'image_backbone': 'mci3',
        'text_width': 768,
        'text_heads': 12,
        'text_layers': 12,
        'use_causal_mask': True,
    },
    'mobileclip_s4': {
        'embed_dim': 768,
        'image_backbone': 'mci4',
        'text_width': 768,
        'text_heads': 12,
        'text_layers': 12,
        'use_causal_mask': True,
    },
}

_FIVE_STAGE_BACKBONES = ('mci3', 'mci4')

# Cheap-model constants.
_VOCAB = 64
_SEQ = 8
_IMG = 64
_EMBED = 16
_BATCH = 4


def _tiny_image_kwargs(num_stages: int) -> dict:
    """Reduced-depth overrides for the image tower (1 block per stage)."""
    return {
        'layers': (1,) * num_stages,
        'embed_dims': tuple(8 * 2 ** i for i in range(num_stages)),
    }


def _tiny_model(variant: str = 'mobileclip2_s0', **overrides) -> MobileClipV2Model:
    """A cheap but structurally faithful model for the behavioural pins."""
    backbone = MODEL_VARIANTS[variant]['image_backbone']
    num_stages = 5 if backbone in _FIVE_STAGE_BACKBONES else 4
    config = dict(
        image_size=_IMG,
        vocab_size=_VOCAB,
        context_length=_SEQ,
        text_width=32,
        text_heads=4,
        text_layers=1,
        text_intermediate=64,
        embed_dim=_EMBED,
        image_encoder_kwargs=_tiny_image_kwargs(num_stages),
    )
    config.update(overrides)
    return MobileClipV2Model.from_variant(variant, **config)


def _images(batch: int = _BATCH, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((batch, _IMG, _IMG, 3)).astype('float32')


def _tokens(batch: int = _BATCH, seed: int = 23) -> np.ndarray:
    """Token ids whose EOT (the numeric maximum) sits at the LAST position.

    `MobileClipTextEncoder` pools by `argmax` of the ids, so a batch whose
    maximum id landed at position 0 would make the causal and the non-causal
    tower agree BY CONSTRUCTION (position 0 attends to itself either way) and
    silently defuse pin 9.
    """
    rng = np.random.default_rng(seed)
    tokens = rng.integers(1, _VOCAB - 1, size=(batch, _SEQ)).astype('int32')
    tokens[:, -1] = _VOCAB - 1
    return tokens


def _inputs(batch: int = _BATCH) -> dict:
    return {'image': _images(batch), 'text': _tokens(batch)}


# ---------------------------------------------------------------------


class TestModelVariants:
    """Pure-config pins — all six variants, nothing built."""

    def test_variant_table_transcription(self):
        """PIN 1: every field of all six rows, against literals written here."""
        assert set(MODEL_VARIANTS) == set(_REFERENCE_VARIANTS), (
            f"variant name set differs: module={sorted(MODEL_VARIANTS)} "
            f"reference={sorted(_REFERENCE_VARIANTS)}"
        )
        for name, expected in _REFERENCE_VARIANTS.items():
            actual = MODEL_VARIANTS[name]
            assert set(actual) == set(expected), (
                f"{name}: field set differs, module={sorted(actual)} "
                f"reference={sorted(expected)}"
            )
            for field, expected_value in expected.items():
                assert actual[field] == expected_value, (
                    f"{name}.{field}: module has {actual[field]!r}, this test's "
                    f"transcription of the supplied JSON has "
                    f"{expected_value!r}. Exactly one of the two is wrong — "
                    f"resolve it against the JSON config, not by editing "
                    f"whichever is easier."
                )
            # A bool that is really an int (or vice versa) compares equal above.
            assert isinstance(actual['use_causal_mask'], bool)

    def test_causal_mask_splits_the_two_families(self):
        """The flag is the WHOLE reason both families are tabulated.

        `mobileclip2_s3` and `mobileclip_s3` agree on every other field; if the
        flag ever agreed too, one of the two rows would be pure duplication.
        """
        for name, row in MODEL_VARIANTS.items():
            expected = not name.startswith('mobileclip2_')
            assert row['use_causal_mask'] is expected, (
                f"{name} has use_causal_mask={row['use_causal_mask']}; the "
                f"MobileCLIP2 series is NON-causal (no_causal_mask: true) and "
                f"MobileCLIP S3/S4 are causal"
            )

        for suffix in ('s3', 's4'):
            v2 = MODEL_VARIANTS[f'mobileclip2_{suffix}']
            v1 = MODEL_VARIANTS[f'mobileclip_{suffix}']
            differing = {k for k in v1 if v1[k] != v2[k]}
            assert differing == {'use_causal_mask'}, (
                f"mobileclip_{suffix} and mobileclip2_{suffix} must differ in "
                f"use_causal_mask ONLY, but differ in {sorted(differing)}"
            )

    def test_shared_fields_are_constants(self):
        """vocab / context / image size are shared by every row."""
        assert DEFAULT_VOCAB_SIZE == 49408
        assert DEFAULT_CONTEXT_LENGTH == 77
        assert DEFAULT_IMAGE_SIZE == 256

    def test_resolve_rejects_unknown_and_returns_a_copy(self):
        row = _resolve_model_variant('mobileclip2_s0')
        row['embed_dim'] = 1
        assert MODEL_VARIANTS['mobileclip2_s0']['embed_dim'] == 512
        with pytest.raises(ValueError, match="Unknown MobileCLIP2 variant"):
            _resolve_model_variant('mobileclip2_s9')

    @pytest.mark.parametrize('variant', sorted(_REFERENCE_VARIANTS))
    def test_from_variant_equals_factory(self, variant):
        """PIN 6: the factory is exactly `from_variant`, for all six rows.

        Config only — these models are never built, so a full mci4 tower costs
        nothing but unbuilt layer objects.
        """
        a = MobileClipV2Model.from_variant(variant)
        b = create_mobile_clip_v2(variant)

        fields = (
            'embed_dim', 'image_backbone', 'image_size', 'vocab_size',
            'context_length', 'text_width', 'text_heads', 'text_layers',
            'text_intermediate', 'use_causal_mask', 'logit_scale_init',
            'logit_scale_max', 'variant',
        )
        for field in fields:
            assert getattr(a, field) == getattr(b, field), field

        expected = _REFERENCE_VARIANTS[variant]
        assert a.embed_dim == expected['embed_dim']
        assert a.image_backbone == expected['image_backbone']
        assert a.text_width == expected['text_width']
        assert a.text_heads == expected['text_heads']
        assert a.text_layers == expected['text_layers']
        assert a.use_causal_mask is expected['use_causal_mask']
        assert a.text_intermediate == 4 * expected['text_width']
        assert a.image_size == DEFAULT_IMAGE_SIZE
        assert a.vocab_size == DEFAULT_VOCAB_SIZE
        assert a.context_length == DEFAULT_CONTEXT_LENGTH

        # The towers really carry the variant's architecture, not just the
        # scalars: the image tower must be the tabulated MCi backbone and both
        # towers must project into the SAME joint space.
        assert a.image_encoder.variant == expected['image_backbone']
        assert a.image_encoder.projection_dim == expected['embed_dim']
        assert a.text_encoder.projection_dim == expected['embed_dim']
        assert a.text_encoder.use_causal_mask is expected['use_causal_mask']
        assert a.text_encoder.embed_dim == expected['text_width']
        assert a.text_encoder.max_seq_len == DEFAULT_CONTEXT_LENGTH


class TestMobileClipV2Model:
    """Behavioural pins on the assembled dual encoder."""

    # ------------------------------------------------------------------
    # PIN 2 — output contract
    # ------------------------------------------------------------------

    def test_output_dict_contract(self):
        """PIN 2: keys and shapes of the returned dict."""
        model = _tiny_model()
        out = model(_inputs(), training=False)

        assert set(out) == {
            'image_features', 'text_features',
            'logits_per_image', 'logits_per_text', 'logit_scale',
        }
        assert tuple(out['image_features'].shape) == (_BATCH, _EMBED)
        assert tuple(out['text_features'].shape) == (_BATCH, _EMBED)
        assert tuple(out['logits_per_image'].shape) == (_BATCH, _BATCH)
        assert tuple(out['logits_per_text'].shape) == (_BATCH, _BATCH)
        assert tuple(out['logit_scale'].shape) == ()

        # `compute_output_shape` must agree, pre-built.
        fresh = _tiny_model()
        predicted = fresh.compute_output_shape({
            'image': (_BATCH, _IMG, _IMG, 3),
            'text': (_BATCH, _SEQ),
        })
        assert predicted['image_features'] == (_BATCH, _EMBED)
        assert predicted['text_features'] == (_BATCH, _EMBED)
        assert predicted['logits_per_image'] == (_BATCH, _BATCH)

    def test_single_modality_returns_only_that_modality(self):
        """Image-only / text-only calls must not fabricate logits."""
        model = _tiny_model()
        model.build({'image': (None, _IMG, _IMG, 3), 'text': (None, _SEQ)})

        image_only = model({'image': _images()}, training=False)
        assert set(image_only) == {'image_features'}
        text_only = model({'text': _tokens()}, training=False)
        assert set(text_only) == {'text_features'}

    # ------------------------------------------------------------------
    # PIN 3 — normalization
    # ------------------------------------------------------------------

    def test_features_are_l2_normalized(self):
        """PIN 3: `normalize=True` gives unit rows; `normalize=False` differs.

        `compute_clip_logits` documents PRE-NORMALIZED inputs and does not
        normalize internally, so this is the model's own responsibility.
        """
        model = _tiny_model()
        images, tokens = _images(), _tokens()

        for name, normalized, raw in (
                (
                    'image',
                    model.encode_image(images, normalize=True, training=False),
                    model.encode_image(images, normalize=False, training=False),
                ),
                (
                    'text',
                    model.encode_text(tokens, normalize=True, training=False),
                    model.encode_text(tokens, normalize=False, training=False),
                ),
        ):
            norms = np.linalg.norm(ops.convert_to_numpy(normalized), axis=-1)
            np.testing.assert_allclose(
                norms, np.ones(_BATCH), atol=1e-5,
                err_msg=f"{name} features are not L2-normalized: {norms}",
            )

            raw_np = ops.convert_to_numpy(raw)
            raw_norms = np.linalg.norm(raw_np, axis=-1)
            assert np.max(np.abs(raw_norms - 1.0)) > 1e-3, (
                f"{name}: the un-normalized features already have unit rows, so "
                f"this pin cannot see whether normalization happens at all "
                f"(norms={raw_norms})"
            )
            assert np.max(
                np.abs(raw_np - ops.convert_to_numpy(normalized))
            ) > 1e-4, f"{name}: normalize=False returned the normalized features"

        # And the features that reach the logits are the NORMALIZED ones: with
        # unit rows every logit is bounded by the temperature.
        out = model(_inputs(), training=False)
        scale = float(ops.convert_to_numpy(out['logit_scale']))
        logits = np.abs(ops.convert_to_numpy(out['logits_per_image']))
        assert np.max(logits) <= scale * (1.0 + 1e-5), (
            f"|logits| max {np.max(logits)} exceeds the temperature {scale}, so "
            f"the features feeding compute_clip_logits are NOT unit-norm"
        )

    # ------------------------------------------------------------------
    # PIN 4 — temperature clipping
    # ------------------------------------------------------------------

    def test_logit_scale_is_clipped(self):
        """PIN 4: exp(50) must come out as exactly `logit_scale_max`.

        Without the clip the scale is 5.18e21 and the loss quietly becomes
        `inf`/`nan` on a diverging run.
        """
        model = _tiny_model()
        model.build({'image': (None, _IMG, _IMG, 3), 'text': (None, _SEQ)})

        assert model.logit_scale_max == 100.0
        model.logit_scale.assign(50.0)

        used = float(ops.convert_to_numpy(model.compute_logit_scale()))
        assert used == pytest.approx(100.0, abs=0.0, rel=0.0), (
            f"compute_logit_scale returned {used!r}; unclipped exp(50) is "
            f"{np.exp(50.0):.3e}"
        )

        out = model(_inputs(), training=False)
        reported = float(ops.convert_to_numpy(out['logit_scale']))
        assert reported == pytest.approx(100.0, abs=0.0, rel=0.0)

        # The clip must reach the LOGITS, not merely the reported scalar: with
        # unit-norm features every |logit| <= the clipped temperature.
        logits = np.abs(ops.convert_to_numpy(out['logits_per_image']))
        assert np.max(logits) <= 100.0 * (1.0 + 1e-5), (
            f"the reported logit_scale is clipped but the logits are not "
            f"(max |logit| = {np.max(logits):.3e})"
        )

    def test_logit_scale_default_is_openai_temperature(self):
        model = _tiny_model()
        model.build({'image': (None, _IMG, _IMG, 3), 'text': (None, _SEQ)})
        assert model.logit_scale_init == pytest.approx(np.log(1.0 / 0.07))
        assert float(
            ops.convert_to_numpy(model.compute_logit_scale())
        ) == pytest.approx(1.0 / 0.07, rel=1e-5)

    # ------------------------------------------------------------------
    # PIN 5 — symmetry
    # ------------------------------------------------------------------

    def test_logits_are_transposes(self):
        """PIN 5: `logits_per_text == transpose(logits_per_image)`."""
        model = _tiny_model()
        out = model(_inputs(), training=False)
        per_image = ops.convert_to_numpy(out['logits_per_image'])
        per_text = ops.convert_to_numpy(out['logits_per_text'])
        np.testing.assert_array_equal(per_text, per_image.T)
        # A symmetric matrix would satisfy the above vacuously.
        assert np.max(np.abs(per_image - per_image.T)) > 1e-6, (
            "logits_per_image happens to be symmetric, so the transpose "
            "assertion above is vacuous on this batch"
        )

    # ------------------------------------------------------------------
    # PIN 7 — gradients
    # ------------------------------------------------------------------

    def test_gradients_reach_every_trainable_weight(self):
        """PIN 7: both towers plus `logit_scale`, with failures NAMED.

        This is what catches a tower that is constructed and tracked but never
        actually called.
        """
        model = _tiny_model()
        inputs = _inputs()

        with tf.GradientTape() as tape:
            out = model(inputs, training=True)
            loss = (
                ops.mean(ops.square(out['logits_per_image']))
                + ops.mean(out['image_features'])
                + ops.mean(out['text_features'])
            )
        grads = tape.gradient(loss, model.trainable_weights)

        assert len(model.trainable_weights) > 0
        missing = [
            w.path for w, g in zip(model.trainable_weights, grads) if g is None
        ]
        assert not missing, f"{len(missing)} weights got a None gradient: {missing}"

        zero = [
            w.path for w, g in zip(model.trainable_weights, grads)
            if float(np.max(np.abs(ops.convert_to_numpy(g)))) == 0.0
        ]
        assert not zero, f"{len(zero)} weights got an all-zero gradient: {zero}"

        # BOTH towers must be represented, not merely "some weights".
        paths = [w.path for w in model.trainable_weights]
        assert any('image_encoder' in p for p in paths), paths[:5]
        assert any('text_encoder' in p for p in paths), paths[:5]
        assert any(p.endswith('logit_scale') for p in paths), paths[:5]

    # ------------------------------------------------------------------
    # PIN 8 — serialization
    # ------------------------------------------------------------------

    def test_keras_roundtrip_preserves_values(self):
        """PIN 8: value round trip, plus an ELEMENTWISE weight check per tower.

        Shapes, layer counts and parameter totals all agreeing after a reload is
        NOT evidence — the repo has measured restored-fresh-kernel failures that
        pass every one of those checks.
        """
        model = _tiny_model()
        inputs = _inputs()
        before = model(inputs, training=False)

        # One weight from EACH tower, chosen for being list-held per-block state.
        image_weight_before = ops.convert_to_numpy(
            model.image_encoder.stages[0].blocks[0].weights[0]
        )
        text_weight_before = ops.convert_to_numpy(
            model.text_encoder.projection_weights
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'mobile_clip_v2.keras')
            model.save(path)
            restored = keras.models.load_model(path)
            after = restored(inputs, training=False)

            image_weight_after = ops.convert_to_numpy(
                restored.image_encoder.stages[0].blocks[0].weights[0]
            )
            text_weight_after = ops.convert_to_numpy(
                restored.text_encoder.projection_weights
            )

        assert set(after) == set(before)
        for key in before:
            np.testing.assert_allclose(
                ops.convert_to_numpy(after[key]),
                ops.convert_to_numpy(before[key]),
                atol=1e-6, rtol=0,
                err_msg=f"output '{key}' changed across the round trip",
            )

        np.testing.assert_array_equal(
            image_weight_after, image_weight_before,
            err_msg="image tower stage_0 block_0 weight was NOT restored",
        )
        np.testing.assert_array_equal(
            text_weight_after, text_weight_before,
            err_msg="text tower projection_weights was NOT restored",
        )
        assert restored.use_causal_mask == model.use_causal_mask
        assert restored.text_encoder.use_causal_mask == model.use_causal_mask

    def test_roundtrip_keeps_the_reduced_image_tower(self):
        """The towers round-trip as THEMSELVES, not as the variant's defaults.

        `get_config` serializes both tower objects; rebuilding them from the
        scalar fields alone would silently restore a full-depth mci0.
        """
        model = _tiny_model()
        model(_inputs(), training=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'mobile_clip_v2.keras')
            model.save(path)
            restored = keras.models.load_model(path)
        assert (
            tuple(restored.image_encoder.layers_per_stage)
            == tuple(model.image_encoder.layers_per_stage) == (1, 1, 1, 1)
        )
        assert (
            tuple(restored.image_encoder.embed_dims)
            == tuple(model.image_encoder.embed_dims)
        )

    # ------------------------------------------------------------------
    # PIN 9 — the causal-mask flag is WIRED
    # ------------------------------------------------------------------

    def test_causal_mask_flag_changes_text_output(self):
        """PIN 9: same weights, different flag -> different text features.

        `mobileclip_s3` (causal) vs `mobileclip2_s3` (non-causal), with the
        causal model's text-tower weights TRANSPLANTED into the non-causal one,
        so the flag is the only remaining difference. A test that merely read
        back `model.use_causal_mask` would pass with the flag stored and never
        forwarded.
        """
        # TWO text layers, not one. MEASURED: at depth 1 this pin reads exactly
        # 0.0 and is VACUOUS. The pooled token is the EOT at the LAST position,
        # and a causal mask leaves the last row of the attention matrix
        # unmasked — so a single layer's output AT THAT POSITION is identical
        # either way. The masked positions only reach the EOT through a SECOND
        # layer. Do not reduce this back to one layer to save time.
        causal = _tiny_model('mobileclip_s3', text_layers=2)
        non_causal = _tiny_model('mobileclip2_s3', text_layers=2)
        assert causal.use_causal_mask is True
        assert non_causal.use_causal_mask is False

        tokens = _tokens()
        causal.build({'image': (None, _IMG, _IMG, 3), 'text': (None, _SEQ)})
        non_causal.build({'image': (None, _IMG, _IMG, 3), 'text': (None, _SEQ)})

        source = [
            ops.convert_to_numpy(w) for w in causal.text_encoder.weights
        ]
        target = non_causal.text_encoder.weights
        assert len(source) == len(target) and len(source) > 0
        for weight, value in zip(target, source):
            weight.assign(value)

        a = ops.convert_to_numpy(
            causal.encode_text(tokens, normalize=False, training=False))
        b = ops.convert_to_numpy(
            non_causal.encode_text(tokens, normalize=False, training=False))

        # The transplant really happened.
        for weight, value in zip(non_causal.text_encoder.weights, source):
            np.testing.assert_array_equal(ops.convert_to_numpy(weight), value)

        delta = float(np.max(np.abs(a - b)))
        assert delta > 1e-5, (
            f"the causal and non-causal text towers produce IDENTICAL features "
            f"on identical weights (max |delta| = {delta:.3e}), so "
            f"`use_causal_mask` is stored but never reaches the attention"
        )

    # ------------------------------------------------------------------
    # PIN 10 — text tower widths
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        'text_width,text_heads,text_intermediate',
        [(768, 12, 3072), (512, 8, 2048)],
    )
    def test_text_tower_widths(self, text_width, text_heads, text_intermediate):
        """PIN 10: both reference text-tower widths produce `(B, embed_dim)`.

        No existing test in the repo builds `MobileClipTextEncoder` at a width
        other than 512. `text_layers` is reduced to 1 (depth is not what this
        pin is about) but the WIDTHS are the reference ones.
        """
        model = _tiny_model(
            'mobileclip2_s3',
            text_width=text_width,
            text_heads=text_heads,
            text_intermediate=text_intermediate,
        )
        features = model.encode_text(_tokens(2), normalize=True, training=False)
        assert tuple(features.shape) == (2, _EMBED)
        assert model.text_encoder.embed_dim == text_width
        assert model.text_encoder.num_heads == text_heads
        assert model.text_encoder.intermediate_size == text_intermediate

        out = model(_inputs(2), training=False)
        assert tuple(out['logits_per_image'].shape) == (2, 2)

    def test_default_text_intermediate_is_four_times_width(self):
        model = MobileClipV2Model(text_width=384, text_heads=6)
        assert model.text_intermediate == 4 * 384

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    def test_heads_must_divide_width(self):
        with pytest.raises(ValueError, match="text_heads must divide text_width"):
            MobileClipV2Model(text_width=512, text_heads=7)

    @pytest.mark.parametrize(
        'kwargs',
        [
            {'embed_dim': 0},
            {'text_layers': -1},
            {'context_length': 0},
            {'logit_scale_max': 0.0},
            {'dropout_rate': 1.0},
        ],
    )
    def test_invalid_config_raises(self, kwargs):
        with pytest.raises(ValueError):
            MobileClipV2Model(**kwargs)
