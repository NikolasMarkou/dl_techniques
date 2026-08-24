"""``SCALE_CONFIGS`` and ``MODEL_VARIANTS`` as DATA -- no model is constructed here.

Moved verbatim from ``test_model.py`` (class ``TestBeitScaleConfigs``, section 1) when
that module was decomposed into one behaviour-named file per concern
(plan-2026-08-24T074054-247151fd, step 8).

This file owns exactly one question: do the two shipped variant tables still hold the
numbers the BEiT paper and the repo's own ``tiny``/``small`` inventions specify, and
does ``_resolve_scale`` still map a variant name onto a scale row. It builds nothing,
so it is the only BEiT test file that stays fast when the model is broken -- which is
precisely when you want to know whether the tables or the wiring moved.
"""


import pytest

from dl_techniques.models.beit.model import _resolve_scale
from dl_techniques.models.beit import (
    MODEL_VARIANTS,
    SCALE_CONFIGS,
)

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
