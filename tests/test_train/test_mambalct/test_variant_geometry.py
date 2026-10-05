"""Variant/geometry contract: mislabeled sizes raise without explicit opt-out."""

import pytest

import train.mambalct.train_mambalct as train_mambalct


def test_variant_geometry_mismatch_raises_without_opt_out() -> None:
    with pytest.raises(ValueError, match="allow_custom_geometry"):
        train_mambalct.MambaLCTTrainingConfig(
            variant="mambalct-384", template_size=128, search_size=256
        )
    config = train_mambalct.MambaLCTTrainingConfig(
        variant="mambalct-384",
        template_size=128,
        search_size=256,
        allow_custom_geometry=True,
    )
    assert config.allow_custom_geometry is True


def test_variant_defaults_agree() -> None:
    config = train_mambalct.MambaLCTTrainingConfig()
    assert (config.template_size, config.search_size) == (128, 256)
    config384 = train_mambalct.MambaLCTTrainingConfig(
        variant="mambalct-384", template_size=192, search_size=384
    )
    assert (config384.template_size, config384.search_size) == (192, 384)
