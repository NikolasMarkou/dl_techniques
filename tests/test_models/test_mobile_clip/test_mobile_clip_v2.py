"""
Test suite for `MobileClipV2Model` (the MobileCLIP2 dual encoder).

The ten mandated pins:

1.  `test_variant_table_transcription` — all 6 rows, all fields, against literals
    written HERE. Includes `use_causal_mask`, which is the ONLY reason both the
    `mobileclip2_*` and the `mobileclip_*` families are in the table.
1b. `test_model_variants_match_supplied_json_configs` — the same 6 rows against
    the COMMITTED upstream open_clip JSONs at
    `research/mobileclip2_reference/model_configs/`, read with `json.load`. That
    is the REAL oracle; pin 1 above is only a second transcription.
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

Every model except the pure-config ones (1, 1b, 6) uses a REDUCED image tower:
a full mci4 does not fit on a 12GB GPU alongside other work.

WHAT THE REDUCTION PRESERVES, and what it does not
--------------------------------------------------
Reduced: block DEPTH (1 per stage), channel WIDTH (8, 16, 32, ...), text depth,
vocabulary, sequence length and the joint embedding width.

NOT reduced, because reducing them would make the mechanism under test
structurally unobservable:

* The number of STAGES, the per-stage token mixers, the downsampling pattern and
  the positional-embedding pattern — all taken from the real variant, so a
  5-stage model still ends in two real attention stages.
* The spatial GRID. `_IMG` is chosen per stage count (`_IMG_4STAGE` /
  `_IMG_5STAGE`) so the deepest attention stage keeps more than one token. At a
  single token, softmax is identically 1.0 and attention degenerates.
* Text DEPTH for the causal-mask pin: 2 layers, not 1 (measured vacuous at 1).
"""

import inspect
import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.models.mobile_clip.mobile_clip_v2 import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_VOCAB_SIZE,
    MobileClipV2Model,
    _resolve_model_variant,
    create_mobile_clip_v2,
)
from tests.test_models.test_fastvit.reference_oracle import (
    load_supplied_json,
)

#: The table is a CLASS attribute (v1's convention), not a module-level dict.
#: `test_variant_table_lives_on_the_class` pins that; this is just a shorthand.
_VARIANTS = MobileClipV2Model.MODEL_VARIANTS


# ---------------------------------------------------------------------
# The six supplied JSON configs, transcribed a SECOND time here, in the
# model's own NESTED shape.
#
# `use_causal_mask` is `not no_causal_mask`: the MobileCLIP2 series sets
# `"no_causal_mask": true` (bidirectional text tower), the earlier MobileCLIP
# S3/S4 configs leave it false (classic causal CLIP text tower).
#
# NB `text_config['embed_dim']` is the TEXT WIDTH; the joint space is the row's
# top-level `embed_dim`. `image_config['variant']` is the MCi backbone, a
# different "variant" from the row's own key.
# ---------------------------------------------------------------------

def _reference_row(backbone, width, heads, causal, *, embed_dim, layers=12):
    return {
        'embed_dim': embed_dim,
        'image_config': {
            'variant': backbone,
            'input_shape': (256, 256, 3),
        },
        'text_config': {
            'vocab_size': 49408,
            'max_seq_len': 77,
            'embed_dim': width,
            'num_layers': layers,
            'num_heads': heads,
            'intermediate_size': 4 * width,
            'use_causal_mask': causal,
        },
    }


_REFERENCE_VARIANTS = {
    'mobileclip2_s0': _reference_row('mci0', 512, 8, False, embed_dim=512),
    'mobileclip2_s2': _reference_row('mci2', 512, 8, False, embed_dim=512),
    'mobileclip2_s3': _reference_row('mci3', 768, 12, False, embed_dim=768),
    'mobileclip2_s4': _reference_row('mci4', 768, 12, False, embed_dim=768),
    'mobileclip_s3': _reference_row('mci3', 768, 12, True, embed_dim=768),
    'mobileclip_s4': _reference_row('mci4', 768, 12, True, embed_dim=768),
}

#: The ten flat names both the JSON oracle and the transcription pin compare on.
#: ONE locator, used for a raw table row AND for a `get_config()` dict, so the
#: two paths cannot drift apart. Without this, a nested `set()` comparison would
#: pass while an extra key hid inside a sub-dict.
_FLAT_FIELDS = (
    'embed_dim', 'image_backbone', 'image_size', 'image_channels',
    'text_width', 'text_heads', 'text_layers', 'context_length',
    'vocab_size', 'use_causal_mask', 'text_intermediate',
)


def _flatten(row: dict) -> dict:
    """Flatten a nested variant row (or a `get_config()` dict) to `_FLAT_FIELDS`."""
    image = row['image_config']
    text = row['text_config']
    input_shape = tuple(image['input_shape'])
    return {
        'embed_dim': row['embed_dim'],
        'image_backbone': image['variant'],
        'image_size': input_shape[0],
        'image_channels': input_shape[2],
        'text_width': text['embed_dim'],
        'text_heads': text['num_heads'],
        'text_layers': text['num_layers'],
        'context_length': text['max_seq_len'],
        'vocab_size': text['vocab_size'],
        'use_causal_mask': text['use_causal_mask'],
        'text_intermediate': text['intermediate_size'],
    }


def _assert_same_key_sets(actual: dict, expected: dict, label: str) -> None:
    """Compare key sets RECURSIVELY through `image_config` / `text_config`.

    A flat `set(actual) == set(expected)` over nested rows would pass while an
    extra field hid one level down, which is exactly what this pin exists to
    catch.
    """
    assert set(actual) == set(expected), (
        f"{label}: field set differs, module={sorted(actual)} "
        f"reference={sorted(expected)}"
    )
    for sub in ('image_config', 'text_config'):
        assert set(actual[sub]) == set(expected[sub]), (
            f"{label}.{sub}: field set differs, module={sorted(actual[sub])} "
            f"reference={sorted(expected[sub])}"
        )


_FIVE_STAGE_BACKBONES = ('mci3', 'mci4')

# Cheap-model constants.
_VOCAB = 64
_SEQ = 8
_EMBED = 16
_BATCH = 4

# Input resolution, per stage count. The tower halves the grid in the stem (x2)
# and once per DOWNSAMPLING stage, so the token count of the DEEPEST stage is
# `(img / 4 / 2**(num_stages - 1))**2`.
#
# These numbers are chosen so the deepest ATTENTION stage still has more than
# one token. MEASURED: at 64px the 5-stage ladder is 16/8/4/2/1, i.e. the last
# attention stage runs on a SINGLE token — a softmax over an axis of size 1 is
# identically 1.0 and the token mixer degenerates to a per-token linear map, so
# every pin in this module would exercise a DEAD last stage (Keras says so out
# loud: "UserWarning: You are using a softmax over axis -1 of a tensor of shape
# (2, 4, 1, 1)"). This is the same "a reduced config makes the mechanism
# structurally unobservable" defect already recorded for `text_layers=1` in
# `test_causal_mask_flag_changes_text_output`. Do not lower these to save time;
# reduce WIDTH (`_tiny_image_kwargs`) instead, which does not collapse the grid.
_IMG_4STAGE = 64    # ladder 16 / 8 / 4 / 2  -> deepest attention stage: 4 tokens
_IMG_5STAGE = 128   # ladder 32 / 16 / 8 / 4 / 2 -> deepest stage: 4 tokens

#: Default for the 4-stage models the majority of the pins use.
_IMG = _IMG_4STAGE


def _num_stages(variant: str) -> int:
    backbone = _VARIANTS[variant]['image_config']['variant']
    return 5 if backbone in _FIVE_STAGE_BACKBONES else 4


def _image_size(num_stages: int) -> int:
    return _IMG_5STAGE if num_stages == 5 else _IMG_4STAGE


def _tiny_image_kwargs(num_stages: int) -> dict:
    """Reduced-depth overrides for the image tower (1 block per stage).

    DEPTH and WIDTH are reduced; the number of stages, the token mixers, the
    downsampling pattern and the positional-embedding pattern are NOT — those
    come from the variant, so the reduced tower still runs a real attention
    stage on a real 2-D grid.
    """
    return {
        'layers': (1,) * num_stages,
        'embed_dims': tuple(8 * 2 ** i for i in range(num_stages)),
    }


def _tiny_model(
        variant: str = 'mobileclip2_s0',
        image: dict = None,
        text: dict = None,
        **overrides,
) -> MobileClipV2Model:
    """A cheap but structurally faithful model for the behavioural pins.

    `from_variant` overrides at the TOP level, so passing `text_config=` would
    REPLACE the row's sub-dict wholesale. `image=` / `text=` are per-field
    overrides that this helper MERGES onto the row instead — the idiom the class
    docstring documents for callers.
    """
    num_stages = _num_stages(variant)
    row = _VARIANTS[variant]

    image_config = {
        **row['image_config'],
        'input_shape': (_image_size(num_stages),) * 2 + (3,),
        **_tiny_image_kwargs(num_stages),
        **(image or {}),
    }
    text_config = {
        **row['text_config'],
        'vocab_size': _VOCAB,
        'max_seq_len': _SEQ,
        'embed_dim': 32,
        'num_heads': 4,
        'num_layers': 1,
        'intermediate_size': 64,
        **(text or {}),
    }
    config = dict(
        embed_dim=_EMBED,
        image_config=image_config,
        text_config=text_config,
    )
    config.update(overrides)
    return MobileClipV2Model.from_variant(variant, **config)


def _minimal_image_config(**overrides) -> dict:
    """The smallest legal `image_config`, for the validation pins.

    The constructor now REQUIRES both sub-dicts, so the old bare
    `MobileClipV2Model(text_width=...)` calls have no successor.
    """
    return {
        'variant': 'mci0',
        'input_shape': (_IMG_4STAGE, _IMG_4STAGE, 3),
        'layers': (1, 1, 1, 1),
        'embed_dims': (8, 16, 32, 64),
        **overrides,
    }


def _minimal_text_config(**overrides) -> dict:
    """The smallest legal `text_config`, for the validation pins."""
    return {
        'vocab_size': _VOCAB,
        'max_seq_len': _SEQ,
        'embed_dim': 32,
        'num_layers': 1,
        'num_heads': 4,
        'intermediate_size': 64,
        'use_causal_mask': False,
        **overrides,
    }


def _images(
        batch: int = _BATCH, seed: int = 11, image_size: int = _IMG
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(
        (batch, image_size, image_size, 3)).astype('float32')


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


def _inputs(batch: int = _BATCH, image_size: int = _IMG) -> dict:
    return {
        'image': _images(batch, image_size=image_size),
        'text': _tokens(batch),
    }


# ---------------------------------------------------------------------


class TestModelVariants:
    """Pure-config pins — all six variants, nothing built."""

    def test_variant_table_lives_on_the_class(self):
        """The table is a CLASS attribute, matching v1's convention.

        Both models are now reached the same way (`<Model>.MODEL_VARIANTS`), and
        the module deliberately exposes no `MODEL_VARIANTS` global that could be
        imported and then confused with v1's.
        """
        import dl_techniques.models.mobile_clip.mobile_clip_v2 as module
        assert 'MODEL_VARIANTS' in vars(MobileClipV2Model)
        assert not hasattr(module, 'MODEL_VARIANTS'), (
            "a module-level MODEL_VARIANTS reappeared; the table belongs on the "
            "class, so v1 and v2 are reached identically"
        )

    def test_variant_table_transcription(self):
        """PIN 1: every field of all six rows, against literals written here."""
        assert set(_VARIANTS) == set(_REFERENCE_VARIANTS), (
            f"variant name set differs: module={sorted(_VARIANTS)} "
            f"reference={sorted(_REFERENCE_VARIANTS)}"
        )
        for name, expected in _REFERENCE_VARIANTS.items():
            actual = _VARIANTS[name]
            # RECURSIVE key-set equality — a flat set() over nested rows would
            # pass while an extra field hid inside a sub-dict.
            _assert_same_key_sets(actual, expected, name)

            flat_actual = _flatten(actual)
            flat_expected = _flatten(expected)
            for field, expected_value in flat_expected.items():
                assert flat_actual[field] == expected_value, (
                    f"{name}.{field}: module has {flat_actual[field]!r}, this "
                    f"test's transcription of the supplied JSON has "
                    f"{expected_value!r}. Exactly one of the two is wrong — "
                    f"resolve it against the JSON config, not by editing "
                    f"whichever is easier."
                )
            # A bool that is really an int (or vice versa) compares equal above.
            assert isinstance(actual['text_config']['use_causal_mask'], bool)

    def test_model_variants_match_supplied_json_configs(self):
        """PIN 1b: all six rows vs the COMMITTED upstream open_clip JSONs.

        The oracle is the third-party config files themselves, at
        `research/mobileclip2_reference/model_configs/`, read with `json.load`.
        Nothing is restated here.

        Both the raw table row AND `from_variant(...).get_config()` are checked,
        through the SAME `_flatten` locator, so the table, the wiring and the
        serialized config cannot drift apart. Every row now states all ten
        fields (they used to be six tabulated plus four shared constants), so
        this covers all ten on both paths.
        """
        for name, row in _VARIANTS.items():
            family, _, size = name.partition('_')
            json_name = (
                f"{'MobileCLIP2' if family == 'mobileclip2' else 'MobileCLIP'}"
                f"-{size.upper()}"
            )
            supplied = load_supplied_json(json_name)
            config = MobileClipV2Model.from_variant(name).get_config()

            vision = supplied['vision_cfg']
            text = supplied['text_cfg']
            backbone = vision['timm_model_name']
            assert backbone.startswith('fastvit_'), (
                f"{json_name}: timm_model_name is {backbone!r}, expected a "
                f"`fastvit_*` name"
            )

            expected = {
                'embed_dim': supplied['embed_dim'],
                'image_backbone': backbone[len('fastvit_'):],
                'image_size': vision['image_size'],
                # `input_shape` carries the channel count too, which the old
                # scalar `image_size` field could not express.
                'image_channels': 3,
                'text_width': text['width'],
                'text_heads': text['heads'],
                'text_layers': text['layers'],
                'context_length': text['context_length'],
                'vocab_size': text['vocab_size'],
                'use_causal_mask': not text['no_causal_mask'],
                'text_intermediate': 4 * text['width'],
            }
            assert set(expected) == set(_FLAT_FIELDS), (
                "the JSON-derived field set and _FLAT_FIELDS disagree; one of "
                "them was extended without the other"
            )

            flat_config = _flatten(config)
            for field, expected_value in expected.items():
                assert flat_config[field] == expected_value, (
                    f"{name}.{field} DISAGREES with the supplied "
                    f"{json_name}.json: port has {flat_config[field]!r}, the "
                    f"config file gives {expected_value!r}."
                )

            # The TABLE must ALSO agree, so a row that is right only because
            # `from_variant` overrode it is caught.
            flat_row = _flatten(row)
            for field, expected_value in expected.items():
                assert flat_row[field] == expected_value, (
                    f"MODEL_VARIANTS[{name!r}] {field}={flat_row[field]!r} but "
                    f"{json_name}.json gives {expected_value!r}"
                )

    def test_causal_mask_splits_the_two_families(self):
        """The flag is the WHOLE reason both families are tabulated.

        `mobileclip2_s3` and `mobileclip_s3` agree on every other field; if the
        flag ever agreed too, one of the two rows would be pure duplication.
        """
        for name, row in _VARIANTS.items():
            expected = not name.startswith('mobileclip2_')
            actual = row['text_config']['use_causal_mask']
            assert actual is expected, (
                f"{name} has use_causal_mask={actual}; the MobileCLIP2 series "
                f"is NON-causal (no_causal_mask: true) and MobileCLIP S3/S4 "
                f"are causal"
            )

        for suffix in ('s3', 's4'):
            # Diff the FLATTENED views. Diffing the nested rows directly would
            # report `{'text_config'}` — true, but it would say nothing about
            # WHICH sub-field differs, and would keep passing if a second field
            # started differing too.
            v2 = _flatten(_VARIANTS[f'mobileclip2_{suffix}'])
            v1 = _flatten(_VARIANTS[f'mobileclip_{suffix}'])
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

    def test_resolve_rejects_unknown_and_returns_a_deep_copy(self):
        """The copy must be DEEP, not shallow.

        A shallow copy shares the class table's nested sub-dicts, so a caller
        mutating `row['image_config'][...]` would rewrite MODEL_VARIANTS for the
        whole process. Mutating only the TOP-level `embed_dim` — as this pin used
        to — cannot detect that, and passed either way.
        """
        row = _resolve_model_variant('mobileclip2_s0')
        row['embed_dim'] = 1
        row['image_config']['variant'] = 'BOGUS'
        row['text_config']['num_layers'] = 999

        table_row = _VARIANTS['mobileclip2_s0']
        assert table_row['embed_dim'] == 512
        assert table_row['image_config']['variant'] == 'mci0', (
            "the resolver returned a SHALLOW copy: mutating the returned "
            "image_config rewrote the class-level MODEL_VARIANTS"
        )
        assert table_row['text_config']['num_layers'] == 12

        with pytest.raises(ValueError, match="Unknown MobileCLIP2 variant"):
            _resolve_model_variant('mobileclip2_s9')

    def test_constructor_does_not_alias_the_class_table(self):
        """Same aliasing hazard, reached through the constructor instead."""
        model = MobileClipV2Model.from_variant('mobileclip2_s0')
        model.image_config['variant'] = 'BOGUS'
        model.text_config['num_layers'] = 999
        assert _VARIANTS['mobileclip2_s0']['image_config']['variant'] == 'mci0'
        assert _VARIANTS['mobileclip2_s0']['text_config']['num_layers'] == 12

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

        expected = _flatten(_REFERENCE_VARIANTS[variant])
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

        # `projection_dim` is INJECTED from embed_dim, never tabulated — a row
        # carrying its own would be a second, unfaithful projection.
        assert 'projection_dim' not in a.image_config
        assert 'projection_dim' not in a.text_config


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

    def test_get_config_names_every_constructor_parameter(self):
        """`get_config()` must be COMPLETE, checked by introspection.

        Not a patch for one missing field: the parameter list is read off
        `__init__` with `inspect.signature`, so a field added to the constructor
        tomorrow and forgotten in `get_config()` fails here without anyone
        remembering to extend a hand-written list. `self` and `**kwargs` are
        skipped; nothing else is exempt.
        """
        model = _tiny_model()
        config = model.get_config()

        signature = inspect.signature(MobileClipV2Model.__init__)
        expected = [
            name for name, parameter in signature.parameters.items()
            if name != 'self'
            and parameter.kind is not inspect.Parameter.VAR_KEYWORD
            and parameter.kind is not inspect.Parameter.VAR_POSITIONAL
        ]
        # An EXACT set pin, not a `len(...) > N` heuristic. The old guard was
        # `> 10`, which the v1-shaped constructor (9 params) would fail; lowering
        # the number to whatever passes would be silent weakening. This is
        # strictly stronger — it also catches an ADDED parameter, which a length
        # bound never could.
        assert set(expected) == {
            'embed_dim', 'image_config', 'text_config', 'logit_scale_init',
            'output_dict', 'logit_scale_max', 'image_encoder', 'text_encoder',
            'variant',
        }, (
            f"MobileClipV2Model.__init__'s parameter list changed to {expected}."
            f" Update this set AND get_config() — do not delete the assertion."
        )

        missing = [name for name in expected if name not in config]
        assert not missing, (
            f"get_config() omits {missing}; every constructor parameter of "
            f"MobileClipV2Model must appear in its config (H-1). Add the "
            f"field(s) to get_config(), do not shorten this list."
        )

        # Introspection can only see the NINE names; the real payload now lives
        # inside two dicts it cannot look into. Pin their values too, or this
        # test degrades in kind even while it keeps passing.
        assert config['image_config'] == model.image_config
        assert config['text_config'] == model.text_config
        assert tuple(config['image_config']['layers']) == (1, 1, 1, 1), (
            "this fixture builds a REDUCED tower, so image_config must carry "
            "its `layers` override — the default tower would make the value "
            "checks above vacuous"
        )

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
        causal = _tiny_model('mobileclip_s3', text={'num_layers': 2})
        non_causal = _tiny_model('mobileclip2_s3', text={'num_layers': 2})
        assert causal.use_causal_mask is True
        assert non_causal.use_causal_mask is False

        tokens = _tokens()
        img = causal.image_size
        assert img == _IMG_5STAGE and non_causal.image_size == img
        causal.build({'image': (None, img, img, 3), 'text': (None, _SEQ)})
        non_causal.build({'image': (None, img, img, 3), 'text': (None, _SEQ)})

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
            text={
                'embed_dim': text_width,
                'num_heads': text_heads,
                'intermediate_size': text_intermediate,
            },
        )
        features = model.encode_text(_tokens(2), normalize=True, training=False)
        assert tuple(features.shape) == (2, _EMBED)
        assert model.text_encoder.embed_dim == text_width
        assert model.text_encoder.num_heads == text_heads
        assert model.text_encoder.intermediate_size == text_intermediate

        out = model(_inputs(2, image_size=model.image_size), training=False)
        assert tuple(out['logits_per_image'].shape) == (2, 2)

    def test_default_text_intermediate_is_four_times_width(self):
        """An omitted `intermediate_size` is FILLED, and the fill round-trips.

        It is written into the stored `text_config` before `get_config()` can
        see it, so a reload does not have to re-derive it.
        """
        text_config = dict(_minimal_text_config(), embed_dim=384, num_heads=6)
        text_config.pop('intermediate_size', None)
        model = MobileClipV2Model(
            embed_dim=_EMBED,
            image_config=_minimal_image_config(),
            text_config=text_config,
        )
        assert model.text_intermediate == 4 * 384
        assert model.get_config()['text_config']['intermediate_size'] == 4 * 384

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    def test_heads_must_divide_width(self):
        with pytest.raises(
                ValueError,
                match=r"text_config\['num_heads'\] must divide",
        ):
            MobileClipV2Model(
                embed_dim=_EMBED,
                image_config=_minimal_image_config(),
                text_config=dict(
                    _minimal_text_config(), embed_dim=512, num_heads=7),
            )

    @pytest.mark.parametrize(
        'embed_dim,image_overrides,text_overrides',
        [
            (0, {}, {}),                              # embed_dim
            (_EMBED, {}, {'num_layers': -1}),         # text depth
            (_EMBED, {}, {'max_seq_len': 0}),         # context length
            (_EMBED, {}, {'dropout_rate': 1.0}),      # rate out of range
            (_EMBED, {'input_shape': (32, 32)}, {}),  # malformed input_shape
            (_EMBED, {'input_shape': (32, 32, 0)}, {}),
        ],
    )
    def test_invalid_config_raises(self, embed_dim, image_overrides, text_overrides):
        with pytest.raises(ValueError):
            MobileClipV2Model(
                embed_dim=embed_dim,
                image_config=dict(_minimal_image_config(), **image_overrides),
                text_config=dict(_minimal_text_config(), **text_overrides),
            )

    def test_invalid_logit_scale_max_raises(self):
        with pytest.raises(ValueError, match="logit_scale_max must be positive"):
            MobileClipV2Model(
                embed_dim=_EMBED,
                image_config=_minimal_image_config(),
                text_config=_minimal_text_config(),
                logit_scale_max=0.0,
            )

    @pytest.mark.parametrize('bad', [None, 5, 'mci0'])
    def test_sub_configs_must_be_dicts(self, bad):
        with pytest.raises(TypeError, match="must be a dictionary"):
            MobileClipV2Model(
                embed_dim=_EMBED,
                image_config=bad,
                text_config=_minimal_text_config(),
            )

    @pytest.mark.parametrize(
        'label,config_kwargs',
        [
            ('image_config', {'image_config': {'variant': 'mci0'}}),
            ('text_config', {'text_config': {'vocab_size': 8}}),
        ],
    )
    def test_missing_required_sub_config_keys_raise_a_named_error(
            self, label, config_kwargs):
        """A missing key must name itself, not surface as a bare KeyError."""
        kwargs = dict(
            embed_dim=_EMBED,
            image_config=_minimal_image_config(),
            text_config=_minimal_text_config(),
        )
        kwargs.update(config_kwargs)
        with pytest.raises(ValueError, match=f"{label} is missing required key"):
            MobileClipV2Model(**kwargs)

    def test_projection_dim_must_not_be_tabulated(self):
        """It is injected from `embed_dim`; a tabulated one would be a SECOND,
        unfaithful projection on top of the tower's own terminal Dense."""
        with pytest.raises(ValueError, match="projection_dim must NOT appear"):
            MobileClipV2Model(
                embed_dim=_EMBED,
                image_config=dict(_minimal_image_config(), projection_dim=32),
                text_config=_minimal_text_config(),
            )

    # ------------------------------------------------------------------
    # `output_dict` — adopted from v1, but NOT with v1's tuple contract
    # ------------------------------------------------------------------

    def test_output_dict_false_returns_the_five_tuple(self):
        """v1 returns a 3-tuple; v2 must NOT.

        `(image, text, logit_scale)` would silently discard `logits_per_image`
        and `logits_per_text`, which v1 never computes and v2 always does. The
        tuple is the documented key order, so it stays aligned with the dict.
        """
        model = _tiny_model(output_dict=False)
        assert model.output_dict is False

        out = model(_inputs(), training=False)
        assert isinstance(out, tuple)
        assert len(out) == 5, (
            f"expected the 5-tuple (image_features, text_features, "
            f"logits_per_image, logits_per_text, logit_scale), got {len(out)} "
            f"entries — v1's 3-tuple drops both logits matrices"
        )

        image_features, text_features, per_image, per_text, scale = out
        assert tuple(image_features.shape) == (_BATCH, _EMBED)
        assert tuple(text_features.shape) == (_BATCH, _EMBED)
        assert tuple(per_image.shape) == (_BATCH, _BATCH)
        assert tuple(per_text.shape) == (_BATCH, _BATCH)
        assert tuple(scale.shape) == ()

        # Same numbers as the dict form, positionally aligned.
        as_dict = _tiny_model()(_inputs(), training=False)
        assert set(as_dict) == {
            'image_features', 'text_features',
            'logits_per_image', 'logits_per_text', 'logit_scale',
        }

    def test_output_dict_false_pads_absent_modalities_with_none(self):
        model = _tiny_model(output_dict=False)
        out = model({'image': _images()}, training=False)
        assert len(out) == 5
        assert out[0] is not None
        assert out[1:] == (None, None, None, None)

    def test_compute_output_shape_follows_output_dict(self):
        """It returned a dict unconditionally before `output_dict` existed.

        Leaving it that way would contradict a model built with
        `output_dict=False` — an internal disagreement nothing else tests.
        """
        spec = {'image': (_BATCH, _IMG, _IMG, 3), 'text': (_BATCH, _SEQ)}

        as_dict = _tiny_model().compute_output_shape(spec)
        assert isinstance(as_dict, dict)
        assert as_dict['image_features'] == (_BATCH, _EMBED)
        assert as_dict['logits_per_image'] == (_BATCH, _BATCH)

        as_tuple = _tiny_model(output_dict=False).compute_output_shape(spec)
        assert isinstance(as_tuple, tuple) and len(as_tuple) == 5
        assert as_tuple[0] == (_BATCH, _EMBED)
        assert as_tuple[2] == (_BATCH, _BATCH)

    def test_output_dict_survives_a_keras_roundtrip(self):
        model = _tiny_model(output_dict=False)
        model.build({'image': (None, _IMG, _IMG, 3), 'text': (None, _SEQ)})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'tuple_mode.keras')
            model.save(path)
            restored = keras.models.load_model(path)
        assert restored.output_dict is False
        assert isinstance(restored(_inputs(), training=False), tuple)

    def test_summary_reports_the_resolved_config(self):
        """v1's `summary()` reads key names v2 does not have.

        Ported verbatim it would print 'Unknown' for the backbone and the image
        size; it must route through the properties instead.
        """
        model = _tiny_model()
        model.build({'image': (None, _IMG, _IMG, 3), 'text': (None, _SEQ)})

        records = []
        import dl_techniques.models.mobile_clip.mobile_clip_v2 as module
        original = module.logger.info
        module.logger.info = lambda message, *a, **k: records.append(str(message))
        try:
            model.summary()
        finally:
            module.logger.info = original

        joined = '\n'.join(records)
        assert 'Unknown' not in joined, (
            f"summary() printed 'Unknown' — it is reading v1's key names:\n"
            f"{joined}"
        )
        assert f"Image backbone: {model.image_backbone}" in joined
        assert f"Image size: {model.image_size}" in joined
        assert f"Text width: {model.text_width}" in joined

    # ------------------------------------------------------------------
    # config round-trip fixed point
    # ------------------------------------------------------------------

    def test_get_config_is_a_fixed_point_across_a_roundtrip(self):
        """Tuples in, tuples out.

        `input_shape` / `layers` / `embed_dims` are written as tuples but come
        back from JSON as LISTS. Without the normalization in `__init__`, a
        restored model's config compares unequal to the one it was saved from
        even though the network is identical.
        """
        model = _tiny_model()
        model.build({'image': (None, _IMG, _IMG, 3), 'text': (None, _SEQ)})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'fixed_point.keras')
            model.save(path)
            restored = keras.models.load_model(path)

        assert restored.image_config == model.image_config, (
            f"image_config changed type or value across the round trip:\n"
            f"  before={model.image_config}\n  after ={restored.image_config}"
        )
        assert restored.text_config == model.text_config
        assert isinstance(restored.image_config['input_shape'], tuple)
        assert isinstance(restored.image_config['layers'], tuple)
