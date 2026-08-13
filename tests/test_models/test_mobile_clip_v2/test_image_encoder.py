"""
Test suite for `FastVitImageEncoder` (MobileCLIP2's FastViT / MCi image tower).

The nine mandated pins:

1. `test_variant_table_matches_reference` — every field of all five rows, against
   values written LITERALLY here. This is a SECOND, INDEPENDENT hand transcription
   of the same upstream fetch; it can catch a typo in the port, but it CANNOT
   validate the transcription itself. `timm` is not installed (constraint H-7), so
   there is no local oracle for the mci0/mci1/mci2 rows at all.
2. `test_mci3_mci4_match_supplied_source` — the ONLY local cross-check that exists:
   mci3/mci4 against the user-supplied `mobileclip2/mobileclip2.py` values as
   reproduced verbatim in the plan's finding F-1. (The supplied file itself is not
   checked into this repo, so the oracle is F-1's reproduction of it.) This is the
   falsification test for the plan's assumption A-1.
3. `test_per_stage_geometry_at_256` — the REAL 256px geometry for a 4-stage and a
   5-stage variant, asserted on the actual forward intermediates, with reduced
   depths/widths so it stays cheap.
4. `test_global_drop_path_schedule_is_stagewise_slice` — the schedule is ONE global
   ramp cut at the stage boundaries, read back from each BLOCK.
5. `test_output_shape_and_projection`.
6. `test_from_variant_equals_factory` — all five variants.
7. `test_gradients_reach_every_trainable_weight`.
8. `test_keras_roundtrip_preserves_values`.
9. `test_stage_count_mismatch_raises`.
"""

import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.models.mobile_clip_v2.image_encoder import (
    MCI_VARIANTS,
    FastVitImageEncoder,
    _resolve_mci_variant,
    create_fastvit_image_encoder,
)


# ---------------------------------------------------------------------
# The reference tables, transcribed a SECOND time, independently of the module.
# mci0/mci1/mci2: timm upstream `timm/models/fastvit.py` (no local oracle, H-7).
# mci3/mci4: the user-supplied `mobileclip2/mobileclip2.py`.
# ---------------------------------------------------------------------

_REFERENCE_TABLE = {
    'mci0': {
        'layers': (2, 6, 10, 2),
        'embed_dims': (64, 128, 256, 512),
        'mlp_ratios': (3.0, 3.0, 3.0, 3.0),
        'se_downsamples': (False, False, True, True),
        'downsamples': (False, True, True, True),
        'pos_embs': (None, None, None, (7, 7)),
        'token_mixers': ('repmixer', 'repmixer', 'repmixer', 'attention'),
        'stem_use_scale_branch': True,
        'norm_layer': 'batch_norm',
        'lkc_use_act': True,
    },
    'mci1': {
        'layers': (4, 12, 20, 4),
        'embed_dims': (64, 128, 256, 512),
        'mlp_ratios': (3.0, 3.0, 3.0, 3.0),
        'se_downsamples': (False, False, True, True),
        'downsamples': (False, True, True, True),
        'pos_embs': (None, None, None, (7, 7)),
        'token_mixers': ('repmixer', 'repmixer', 'repmixer', 'attention'),
        'stem_use_scale_branch': True,
        'norm_layer': 'batch_norm',
        'lkc_use_act': True,
    },
    'mci2': {
        'layers': (4, 12, 24, 4),
        'embed_dims': (80, 160, 320, 640),
        'mlp_ratios': (3.0, 3.0, 3.0, 3.0),
        'se_downsamples': (False, False, True, True),
        'downsamples': (False, True, True, True),
        'pos_embs': (None, None, None, (7, 7)),
        'token_mixers': ('repmixer', 'repmixer', 'repmixer', 'attention'),
        'stem_use_scale_branch': True,
        'norm_layer': 'batch_norm',
        'lkc_use_act': True,
    },
    'mci3': {
        'layers': (2, 12, 24, 4, 2),
        'embed_dims': (96, 192, 384, 768, 1536),
        'mlp_ratios': (4.0, 4.0, 4.0, 4.0, 4.0),
        'se_downsamples': (False, False, False, False, False),
        'downsamples': (False, True, True, True, True),
        'pos_embs': (None, None, None, (7, 7), (7, 7)),
        'token_mixers': (
            'repmixer', 'repmixer', 'repmixer', 'attention', 'attention'),
        'stem_use_scale_branch': False,
        'norm_layer': 'layer_norm',
        'lkc_use_act': True,
    },
    'mci4': {
        'layers': (2, 12, 24, 4, 4),
        'embed_dims': (128, 256, 512, 1024, 2048),
        'mlp_ratios': (4.0, 4.0, 4.0, 4.0, 4.0),
        'se_downsamples': (False, False, False, False, False),
        'downsamples': (False, True, True, True, True),
        'pos_embs': (None, None, None, (7, 7), (7, 7)),
        'token_mixers': (
            'repmixer', 'repmixer', 'repmixer', 'attention', 'attention'),
        'stem_use_scale_branch': False,
        'norm_layer': 'layer_norm',
        'lkc_use_act': True,
    },
}

#: The number of stages per variant, asserted separately: a 4-vs-5 stage mixup is
#: the most likely transcription error and is invisible to a field-by-field
#: comparison that iterates over whichever tuple happens to be shorter.
_REFERENCE_STAGE_COUNTS = {
    'mci0': 4, 'mci1': 4, 'mci2': 4, 'mci3': 5, 'mci4': 5,
}

_PER_STAGE_FIELDS = (
    'layers', 'embed_dims', 'mlp_ratios', 'se_downsamples',
    'downsamples', 'pos_embs', 'token_mixers',
)


class TestMciVariantTable:
    """Pins on `MCI_VARIANTS` itself — pure config, so all five variants."""

    def test_variant_table_matches_reference(self):
        """PIN 1: every field of all five rows, plus the tuple LENGTHS."""
        assert set(MCI_VARIANTS) == set(_REFERENCE_TABLE), (
            f"variant name set differs: module={sorted(MCI_VARIANTS)} "
            f"reference={sorted(_REFERENCE_TABLE)}"
        )
        for name, expected in _REFERENCE_TABLE.items():
            actual = MCI_VARIANTS[name]
            assert set(actual) == set(expected), (
                f"{name}: field set differs, module={sorted(actual)} "
                f"reference={sorted(expected)}"
            )

            # Lengths FIRST and explicitly: a 4-vs-5 stage mixup must be
            # reported as such, not as a mismatched element.
            n = _REFERENCE_STAGE_COUNTS[name]
            for field in _PER_STAGE_FIELDS:
                assert len(actual[field]) == n, (
                    f"{name}.{field} has {len(actual[field])} entries, "
                    f"expected {n} (one per stage)"
                )
                assert len(expected[field]) == n, (
                    f"the TEST's own reference row {name}.{field} has "
                    f"{len(expected[field])} entries, expected {n}"
                )

            for field, expected_value in expected.items():
                actual_value = actual[field]
                if isinstance(expected_value, tuple):
                    actual_value = tuple(actual_value)
                assert actual_value == expected_value, (
                    f"{name}.{field}: module has {actual_value!r}, this test's "
                    f"transcription has {expected_value!r}. Exactly one of the "
                    f"two is a mis-transcription of the reference — resolve it "
                    f"against the source, not by editing whichever is easier."
                )

    def test_mci3_mci4_match_supplied_source(self):
        """PIN 2: mci3/mci4 vs the USER-SUPPLIED `mobileclip2.py`.

        This is the only local cross-check that exists (H-7: `timm` is not
        installed). The values below are the supplied source's `fastvit_mci3` /
        `fastvit_mci4` model_args as reproduced verbatim in finding F-1, written
        out here in the supplied file's own vocabulary rather than restructured,
        so a restructuring error in the port is visible.
        """
        # From the supplied `mobileclip2.py`, `fastvit_mci3`:
        #   layers=(2, 12, 24, 4, 2)
        #   embed_dims=(96, 192, 384, 768, 1536)
        #   mlp_ratios=(4, 4, 4, 4, 4)
        #   downsamples=(False, True, True, True, True)
        #   pos_embs=(None, None, None, RepCPE(7x7), RepCPE(7x7))
        #   token_mixers=("repmixer", "repmixer", "repmixer", "attention",
        #                 "attention")
        #   se_downsamples: none  ->  all False
        #   stem: monkey-patched `convolutional_stem_timm(use_scale_branch=False)`
        #   norm_layer: LayerNormChannel  ->  channels-last 'layer_norm'
        supplied_mci3 = {
            'layers': (2, 12, 24, 4, 2),
            'embed_dims': (96, 192, 384, 768, 1536),
            'mlp_ratios': (4.0, 4.0, 4.0, 4.0, 4.0),
            'se_downsamples': (False,) * 5,
            'downsamples': (False, True, True, True, True),
            'pos_embs': (None, None, None, (7, 7), (7, 7)),
            'token_mixers': (
                'repmixer', 'repmixer', 'repmixer', 'attention', 'attention'),
            'stem_use_scale_branch': False,
            'norm_layer': 'layer_norm',
            'lkc_use_act': True,
        }
        # `fastvit_mci4` differs from mci3 in EXACTLY two fields: the last
        # stage's depth (2 -> 4) and the widths (4/3 of mci3, i.e. 128/256/512/
        # 1024/2048).
        supplied_mci4 = dict(supplied_mci3)
        supplied_mci4['layers'] = (2, 12, 24, 4, 4)
        supplied_mci4['embed_dims'] = (128, 256, 512, 1024, 2048)

        for name, supplied in (('mci3', supplied_mci3), ('mci4', supplied_mci4)):
            actual = MCI_VARIANTS[name]
            for field, expected_value in supplied.items():
                actual_value = actual[field]
                if isinstance(expected_value, tuple):
                    actual_value = tuple(actual_value)
                assert actual_value == expected_value, (
                    f"{name}.{field} DISAGREES with the user-supplied "
                    f"mobileclip2.py: port has {actual_value!r}, supplied source "
                    f"has {expected_value!r}."
                )

        # mci4 is mci3 scaled: assert the structural relationship too, so a
        # copy-paste that duplicates mci3's widths into mci4 is caught even if
        # both literals above were edited together.
        assert MCI_VARIANTS['mci4']['embed_dims'] != MCI_VARIANTS['mci3']['embed_dims']
        assert (
            MCI_VARIANTS['mci4']['layers'][:4]
            == MCI_VARIANTS['mci3']['layers'][:4]
        )

    def test_resolve_accepts_timm_prefix_and_rejects_unknown(self):
        """`_resolve_mci_variant` handles timm's `fastvit_` model-name prefix."""
        assert _resolve_mci_variant('fastvit_mci3')['layers'] == (2, 12, 24, 4, 2)
        assert _resolve_mci_variant('mci3')['layers'] == (2, 12, 24, 4, 2)
        with pytest.raises(ValueError, match="Unknown MCi variant"):
            _resolve_mci_variant('mci9')

    def test_resolve_returns_a_copy(self):
        """Mutating the returned row must not corrupt the module table."""
        row = _resolve_mci_variant('mci0')
        row['embed_dims'] = (1, 1, 1, 1)
        assert MCI_VARIANTS['mci0']['embed_dims'] == (64, 128, 256, 512)


class TestFastVitImageEncoder:
    """Behavioural pins on the assembled tower."""

    # ------------------------------------------------------------------
    # fixtures / helpers
    # ------------------------------------------------------------------

    @pytest.fixture
    def tiny_encoder(self):
        """A cheap mci0-shaped tower: 1 block per stage, 64px input."""
        return FastVitImageEncoder(
            variant='mci0',
            layers=(1, 1, 1, 1),
            embed_dims=(32, 64, 128, 256),
            input_shape=(64, 64, 3),
            projection_dim=48,
        )

    @pytest.fixture
    def tiny_input(self):
        np.random.seed(17)
        return np.random.randn(4, 64, 64, 3).astype('float32')

    # ------------------------------------------------------------------
    # PIN 3 — spatial geometry
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        'variant,embed_dims,expected_sizes',
        [
            # 4-stage: stem /4 -> 64, then 64, 32, 16, 8.
            ('mci0', (32, 64, 128, 256), (64, 32, 16, 8)),
            # 5-stage: stem /4 -> 64, then 64, 32, 16, 8, 4.
            ('mci3', (32, 64, 128, 256, 512), (64, 32, 16, 8, 4)),
        ],
    )
    def test_per_stage_geometry_at_256(self, variant, embed_dims, expected_sizes):
        """PIN 3: the REAL 256px per-stage geometry, on actual forward outputs.

        Depths and widths are reduced to keep this cheap; the SPATIAL geometry is
        the reference one at the reference input resolution.
        """
        depths = (1,) * len(embed_dims)
        encoder = FastVitImageEncoder(
            variant=variant,
            layers=depths,
            embed_dims=embed_dims,
            input_shape=(256, 256, 3),
            projection_dim=32,
        )

        # (a) `compute_output_shape` composed in sequence, valid pre-build.
        predicted = encoder.stage_output_shapes((None, 256, 256, 3))
        assert encoder.stem_output_shape((None, 256, 256, 3))[1:3] == (64, 64), (
            "the stem must be net stride 4"
        )
        assert tuple(s[1] for s in predicted) == expected_sizes
        assert tuple(s[2] for s in predicted) == expected_sizes
        assert tuple(s[3] for s in predicted) == tuple(embed_dims)

        # (b) the ACTUAL forward intermediates — walking the real sub-layers, so a
        # `compute_output_shape` that lies about the forward pass is caught.
        x = np.zeros((1, 256, 256, 3), dtype='float32')
        for block in encoder.stem:
            x = block(x, training=False)
        assert tuple(x.shape[1:3]) == (64, 64)
        actual_sizes = []
        actual_channels = []
        for stage in encoder.stages:
            x = stage(x, training=False)
            assert x.shape[1] == x.shape[2], f"stage output not square: {x.shape}"
            actual_sizes.append(int(x.shape[1]))
            actual_channels.append(int(x.shape[3]))
        assert tuple(actual_sizes) == expected_sizes, (
            f"{variant} per-stage spatial sizes at 256px: got {actual_sizes}, "
            f"expected {list(expected_sizes)}"
        )
        assert tuple(actual_channels) == tuple(embed_dims)

    # ------------------------------------------------------------------
    # PIN 4 — the global drop-path schedule
    # ------------------------------------------------------------------

    def test_global_drop_path_schedule_is_stagewise_slice(self):
        """PIN 4: ONE ramp over sum(layers), cut at the stage boundaries.

        Read back from the BLOCKS, not from the encoder's own bookkeeping, so the
        schedule is proven to have reached the stochastic-depth layers. A
        per-stage independent ramp (the natural wrong implementation) restarts at
        0.0 in every stage and fails here.
        """
        depths = (2, 3, 4, 2)
        encoder = FastVitImageEncoder(
            variant='mci0',
            layers=depths,
            embed_dims=(32, 64, 128, 256),
            input_shape=(64, 64, 3),
            projection_dim=16,
            drop_path_rate=0.3,
        )

        observed = []
        for stage in encoder.stages:
            for block in stage.blocks:
                observed.append(block.drop_path_rate)

        expected = linear_drop_path_rates(sum(depths), 0.3)
        assert observed == expected, (
            f"per-block drop-path rates {observed} != the single global ramp "
            f"{expected}. A per-stage ramp would restart at 0.0 in every stage."
        )
        # Non-degenerate by construction: 11 blocks, 11 distinct rates. An
        # all-equal or all-zero schedule would make this pin vacuous.
        assert len(set(observed)) == len(observed) == sum(depths)
        assert observed[0] == 0.0 and observed[-1] == pytest.approx(0.3)
        # And the encoder's own stagewise bookkeeping must agree with the blocks.
        assert [len(s) for s in encoder.drop_path_rates] == list(depths)
        assert [r for stage in encoder.drop_path_rates for r in stage] == expected

    # ------------------------------------------------------------------
    # PIN 5 — output shape / projection
    # ------------------------------------------------------------------

    def test_output_shape_and_projection(self, tiny_input):
        """PIN 5: `(B, projection_dim)`, or the pooled width when None."""
        with_proj = FastVitImageEncoder(
            variant='mci0', layers=(1, 1, 1, 1), embed_dims=(32, 64, 128, 256),
            input_shape=(64, 64, 3), projection_dim=48,
        )
        y = with_proj(tiny_input, training=False)
        assert y.shape == (4, 48)
        assert with_proj.compute_output_shape((4, 64, 64, 3)) == (4, 48)
        assert with_proj.projection is not None

        without_proj = FastVitImageEncoder(
            variant='mci0', layers=(1, 1, 1, 1), embed_dims=(32, 64, 128, 256),
            input_shape=(64, 64, 3), projection_dim=None,
        )
        z = without_proj(tiny_input, training=False)
        expected_width = int(256 * 2.0)  # embed_dims[-1] * cls_ratio
        assert without_proj.final_features == expected_width
        assert z.shape == (4, expected_width)
        assert without_proj.compute_output_shape((4, 64, 64, 3)) == (4, expected_width)
        assert without_proj.projection is None

    def test_cls_ratio_widens_final_conv(self):
        """`cls_ratio` is applied to the LAST stage's width, not the first."""
        encoder = FastVitImageEncoder(
            variant='mci0', layers=(1, 1, 1, 1), embed_dims=(32, 64, 128, 256),
            input_shape=(64, 64, 3), projection_dim=None, cls_ratio=1.5,
        )
        assert encoder.final_features == 384
        assert encoder.final_conv.out_channels == 384

    # ------------------------------------------------------------------
    # PIN 6 — from_variant == factory
    # ------------------------------------------------------------------

    @pytest.mark.parametrize('variant', sorted(MCI_VARIANTS))
    def test_from_variant_equals_factory(self, variant):
        """PIN 6: the classmethod and the factory agree, for all five variants."""
        a = FastVitImageEncoder.from_variant(variant)
        b = create_fastvit_image_encoder(variant)

        cfg_a = dict(a.get_config())
        cfg_b = dict(b.get_config())
        # `name` is auto-generated and differs by construction order.
        cfg_a.pop('name', None)
        cfg_b.pop('name', None)
        assert cfg_a == cfg_b

        # And both really are that variant's architecture.
        row = MCI_VARIANTS[variant]
        assert a.layers_per_stage == row['layers']
        assert a.embed_dims == row['embed_dims']
        assert a.token_mixers == row['token_mixers']
        assert a.norm_layer == row['norm_layer']
        assert a.stem_use_scale_branch == row['stem_use_scale_branch']
        assert len(a.stages) == len(row['layers'])

    def test_explicit_arguments_override_the_variant_row(self):
        """A supplied field wins over the variant table."""
        encoder = FastVitImageEncoder(
            variant='mci0', layers=(1, 1, 1, 1), embed_dims=(32, 64, 128, 256),
            input_shape=(64, 64, 3), projection_dim=16, norm_layer='layer_norm',
        )
        assert encoder.norm_layer == 'layer_norm'
        assert encoder.layers_per_stage == (1, 1, 1, 1)
        # ...while the untouched fields still come from the row.
        assert encoder.token_mixers == MCI_VARIANTS['mci0']['token_mixers']

    # ------------------------------------------------------------------
    # PIN 7 — gradients
    # ------------------------------------------------------------------

    def test_gradients_reach_every_trainable_weight(self, tiny_encoder, tiny_input):
        """PIN 7: no `None` and no all-zero gradient, naming any weight that fails."""
        with tf.GradientTape() as tape:
            y = tiny_encoder(tiny_input, training=True)
            loss = tf.reduce_mean(tf.square(y))
        grads = tape.gradient(loss, tiny_encoder.trainable_variables)

        assert len(grads) == len(tiny_encoder.trainable_variables)
        missing = [
            v.path for g, v in zip(grads, tiny_encoder.trainable_variables)
            if g is None
        ]
        assert not missing, f"gradient is None for: {missing}"
        dead = [
            v.path for g, v in zip(grads, tiny_encoder.trainable_variables)
            if float(tf.reduce_max(tf.abs(g))) == 0.0
        ]
        assert not dead, f"gradient is identically zero for: {dead}"

    # ------------------------------------------------------------------
    # PIN 8 — serialization round trip BY VALUE
    # ------------------------------------------------------------------

    def test_keras_roundtrip_preserves_values(self, tiny_encoder, tiny_input):
        """PIN 8: save -> load -> forward, compared elementwise at training=False."""
        original = tiny_encoder(tiny_input, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'fastvit_image_encoder.keras')
            tiny_encoder.save(path)
            restored = keras.models.load_model(path)
            reloaded = restored(tiny_input, training=False)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original),
            keras.ops.convert_to_numpy(reloaded),
            atol=1e-6, rtol=0,
            err_msg="the restored tower does not reproduce the original outputs",
        )

    def test_config_roundtrip_preserves_architecture(self, tiny_encoder):
        """`from_config(get_config())` rebuilds the same architecture."""
        rebuilt = FastVitImageEncoder.from_config(tiny_encoder.get_config())
        assert rebuilt.layers_per_stage == tiny_encoder.layers_per_stage
        assert rebuilt.embed_dims == tiny_encoder.embed_dims
        assert rebuilt.pos_embs == tiny_encoder.pos_embs
        assert rebuilt.token_mixers == tiny_encoder.token_mixers
        assert rebuilt.norm_layer == tiny_encoder.norm_layer
        assert rebuilt.stem_use_scale_branch == tiny_encoder.stem_use_scale_branch
        assert rebuilt.final_features == tiny_encoder.final_features
        assert rebuilt.projection_dim == tiny_encoder.projection_dim

    # ------------------------------------------------------------------
    # PIN 9 — validation
    # ------------------------------------------------------------------

    def test_stage_count_mismatch_raises(self):
        """PIN 9: disagreeing per-stage tuple lengths raise, naming the lengths.

        Without this check nothing raises at all: the constructor iterates over
        `range(len(layers))` and a 5-entry `embed_dims` is SILENTLY truncated to a
        4-stage model. That silence is exactly what this pin exists to prevent.
        """
        with pytest.raises(ValueError) as exc:
            FastVitImageEncoder(
                layers=(1, 1, 1, 1),
                embed_dims=(32, 64, 128, 256, 512),  # 5 entries vs 4
                mlp_ratios=(3.0, 3.0, 3.0, 3.0),
                se_downsamples=(False, False, True, True),
                downsamples=(False, True, True, True),
                pos_embs=(None, None, None, (7, 7)),
                token_mixers=('repmixer', 'repmixer', 'repmixer', 'attention'),
                stem_use_scale_branch=True,
                norm_layer='batch_norm',
                lkc_use_act=True,
                input_shape=(64, 64, 3),
            )
        message = str(exc.value)
        assert 'layers=4' in message and 'embed_dims=5' in message, message

    def test_missing_field_without_variant_raises(self):
        """Omitting a per-stage tuple with no variant names the missing field."""
        with pytest.raises(ValueError, match="'embed_dims'"):
            FastVitImageEncoder(layers=(1, 1))

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown MCi variant"):
            FastVitImageEncoder(variant='mci7')

    @pytest.mark.parametrize(
        'kwargs,match',
        [
            ({'cls_ratio': 0.0}, 'cls_ratio'),
            ({'projection_dim': 0}, 'projection_dim'),
            ({'drop_path_rate': 1.0}, 'drop_path_rate'),
            ({'dropout_rate': -0.1}, 'dropout_rate'),
            ({'head_dropout_rate': 1.5}, 'head_dropout_rate'),
        ],
    )
    def test_invalid_scalars_raise(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            FastVitImageEncoder(
                variant='mci0', layers=(1, 1, 1, 1),
                embed_dims=(32, 64, 128, 256), input_shape=(64, 64, 3),
                **kwargs,
            )

    # ------------------------------------------------------------------
    # structural pins on the pieces the reference fixes
    # ------------------------------------------------------------------

    def test_stem_is_three_mobileone_blocks_with_variant_scale_branch(self):
        """Stem = dense k3/s2, DEPTHWISE k3/s2, pointwise k1/s1, /4 net."""
        for variant, expect_scale_branch in (('mci0', True), ('mci3', False)):
            dims = (32, 64, 128, 256, 512)[:len(MCI_VARIANTS[variant]['layers'])]
            encoder = FastVitImageEncoder(
                variant=variant, layers=(1,) * len(dims), embed_dims=dims,
                input_shape=(64, 64, 3), projection_dim=16,
            )
            assert len(encoder.stem) == 3
            kernels = [b.kernel_size for b in encoder.stem]
            strides = [b.stride for b in encoder.stem]
            assert kernels == [3, 3, 1]
            assert strides == [2, 2, 1]
            # `group_size=1` is timm's spelling of DEPTHWISE; only the middle
            # block is depthwise.
            assert [b.group_size for b in encoder.stem] == [0, 1, 0]
            assert all(
                b.use_scale_branch is expect_scale_branch for b in encoder.stem
            ), f"{variant}: stem use_scale_branch should be {expect_scale_branch}"
            assert all(b.out_channels == dims[0] for b in encoder.stem)

    def test_final_conv_matches_reference(self):
        """`final_conv`: k3, depthwise, SE at 1/16 with bias, BEFORE the act."""
        encoder = FastVitImageEncoder(
            variant='mci0', layers=(1, 1, 1, 1), embed_dims=(32, 64, 128, 256),
            input_shape=(64, 64, 3), projection_dim=16,
        )
        conv = encoder.final_conv
        assert conv.kernel_size == 3
        assert conv.stride == 1
        assert conv.group_size == 1
        assert conv.use_se is True
        assert conv.num_conv_branches == 1
        assert conv.out_channels == 512
        # timm builds this as `SqueezeExcite(out_chs, rd_divisor=1)` and never
        # overrides `rd_ratio`, whose default is 1/16 — NOT the 0.25 that
        # ReparamLargeKernelConv passes explicitly at its own call site.
        assert conv.se_reduction_ratio == pytest.approx(1.0 / 16.0)
        assert conv.se_use_bias is True
        # `out = self.act(self.se(out))` — SE runs BEFORE the activation.
        assert conv.se_position == 'pre_act'

    def test_norm_layer_reaches_the_attention_stages_only(self):
        """`norm_layer` is threaded into attention stages; repmixer has no norm arg."""
        encoder = FastVitImageEncoder(
            variant='mci3', layers=(1, 1, 1, 1, 1),
            embed_dims=(32, 64, 128, 256, 512),
            input_shape=(64, 64, 3), projection_dim=16,
        )
        assert encoder.norm_layer == 'layer_norm'
        attention_stages = [
            s for s in encoder.stages if s.token_mixer == 'attention'
        ]
        assert len(attention_stages) == 2
        for stage in attention_stages:
            for block in stage.blocks:
                assert block.normalization_type == 'layer_norm'

    def test_pos_emb_present_exactly_where_the_table_says(self):
        """A RepCPE exists iff the variant's `pos_embs` entry is not None."""
        for variant in sorted(MCI_VARIANTS):
            row = MCI_VARIANTS[variant]
            n = len(row['layers'])
            dims = (32, 64, 128, 256, 512)[:n]
            encoder = FastVitImageEncoder(
                variant=variant, layers=(1,) * n, embed_dims=dims,
                input_shape=(64, 64, 3), projection_dim=16,
            )
            actual = [s.pos_emb is not None for s in encoder.stages]
            expected = [p is not None for p in row['pos_embs']]
            assert actual == expected, f"{variant}: RepCPE placement {actual} != {expected}"

    def test_stages_are_a_flat_list(self):
        """Sub-models are stored FLAT — a nested list silently drops weights."""
        encoder = FastVitImageEncoder(
            variant='mci0', layers=(1, 1, 1, 1), embed_dims=(32, 64, 128, 256),
            input_shape=(64, 64, 3), projection_dim=16,
        )
        assert all(isinstance(s, keras.layers.Layer) for s in encoder.stages)
        assert all(isinstance(b, keras.layers.Layer) for b in encoder.stem)
