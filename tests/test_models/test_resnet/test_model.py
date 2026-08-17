"""
Test suite for the ResNet model with focus on the step-4
``normalization_kwargs`` plumbing introduced by
plan_2026-05-18_6776f8ba (D-003).

Scope (intentionally narrow — adjacent ResNet tests did not previously
exist; this file is the new floor for the bit-exact preservation
invariant in Pre-Mortem #2 of the rms_variants_train Phase 3 refinement
plan):

1. Default-off plumbing (`normalization_kwargs=None`) is byte-identical
   to the pre-plumbing factory call shape: the stem and every block's
   internal norm receive `**{}`.
2. Explicit `normalization_kwargs={"use_scale": False}` propagates to
   the stem and all in-block norm layers when paired with a norm class
   that supports the kwarg (e.g. `rms_norm`).
3. `get_config` round-trips the kwarg so existing serialization
   contract is preserved.
4. End-to-end forward pass with a non-`batch_norm` normalization type
   does not raise.
"""

import keras
import numpy as np

from dl_techniques.models.resnet.model import ResNet
from dl_techniques.layers.standard_blocks import BasicBlock, BottleneckBlock


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _small_resnet18(**overrides):
    """A CIFAR-shaped ResNet18 that builds quickly on CPU."""
    cfg = dict(
        num_classes=10,
        blocks_per_stage=[2, 2, 2, 2],
        filters_per_stage=[64, 128, 256, 512],
        block_type="basic",
        normalization_type="batch_norm",
        input_shape=(32, 32, 3),
    )
    cfg.update(overrides)
    return ResNet(**cfg)


# ---------------------------------------------------------------------
# Default-off (bit-exact) plumbing
# ---------------------------------------------------------------------


class TestResNetNormKwargsDefault:
    def test_default_off_stores_empty_dict(self):
        model = _small_resnet18()
        assert model.normalization_kwargs == {}

    def test_default_off_stem_bn_factory_call(self):
        """stem_bn was created with no extra kwargs — get_config('name') == 'stem_bn'."""
        model = _small_resnet18()
        assert model.stem_bn.name == "stem_bn"

    def test_default_off_blocks_have_empty_kwargs(self):
        """Every BasicBlock in the stages stores `{}` for normalization_kwargs."""
        model = _small_resnet18()
        for stage_blocks in model.stages:
            for blk in stage_blocks:
                assert isinstance(blk, (BasicBlock, BottleneckBlock))
                assert blk.normalization_kwargs == {}

    def test_default_off_forward_pass(self):
        model = _small_resnet18()
        x = np.zeros((1, 32, 32, 3), dtype=np.float32)
        out = model(x, training=False).numpy()
        assert out.shape == (1, 10)
        assert not np.isnan(out).any()


# ---------------------------------------------------------------------
# Explicit kwargs propagation
# ---------------------------------------------------------------------


class TestResNetNormKwargsRmsNoScale:
    """When `normalization_kwargs={'use_scale': False}` is passed AND the
    `normalization_type` supports the kwarg (RMSNorm), every norm layer
    in the model honors it — stem + all in-block norms. This is the
    contract Step 5 of the plan exercises in E2 `param_matched` mode."""

    def test_stem_uses_scale_false(self):
        model = _small_resnet18(
            normalization_type="rms_norm",
            normalization_kwargs={"use_scale": False},
        )
        assert getattr(model.stem_bn, "use_scale", None) is False
        assert getattr(model.stem_bn, "gamma", None) is None

    def test_basic_block_internal_norms_use_scale_false(self):
        model = _small_resnet18(
            normalization_type="rms_norm",
            normalization_kwargs={"use_scale": False},
        )
        # Build via a forward pass so sub-layer weights materialize.
        _ = model(np.zeros((1, 32, 32, 3), dtype=np.float32), training=False)
        for stage_blocks in model.stages:
            for blk in stage_blocks:
                for norm_attr in ("bn1", "bn2"):
                    norm_layer = getattr(blk, norm_attr)
                    assert norm_layer.use_scale is False
                    assert getattr(norm_layer, "gamma", None) is None

    def test_bottleneck_block_internal_norms_use_scale_false(self):
        model = ResNet(
            num_classes=10,
            blocks_per_stage=[1, 1, 1, 1],
            filters_per_stage=[64, 128, 256, 512],
            block_type="bottleneck",
            normalization_type="rms_norm",
            normalization_kwargs={"use_scale": False},
            input_shape=(32, 32, 3),
        )
        _ = model(np.zeros((1, 32, 32, 3), dtype=np.float32), training=False)
        for stage_blocks in model.stages:
            for blk in stage_blocks:
                assert isinstance(blk, BottleneckBlock)
                for norm_attr in ("bn1", "bn2", "bn3"):
                    norm_layer = getattr(blk, norm_attr)
                    assert norm_layer.use_scale is False
                    assert getattr(norm_layer, "gamma", None) is None


# ---------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------


class TestResNetNormKwargsSerialization:
    def test_get_config_carries_kwargs(self):
        model = _small_resnet18(
            normalization_type="rms_norm",
            normalization_kwargs={"use_scale": False},
        )
        cfg = model.get_config()
        assert cfg["normalization_kwargs"] == {"use_scale": False}

    def test_from_config_restores_kwargs(self):
        model = _small_resnet18(
            normalization_type="rms_norm",
            normalization_kwargs={"use_scale": False},
        )
        cfg = model.get_config()
        restored = ResNet.from_config(cfg)
        assert restored.normalization_kwargs == {"use_scale": False}

    def test_default_empty_dict_round_trips(self):
        model = _small_resnet18()
        cfg = model.get_config()
        # `normalization_kwargs` was stored as `{}` and the config exposes it.
        assert cfg["normalization_kwargs"] == {}
        restored = ResNet.from_config(cfg)
        assert restored.normalization_kwargs == {}


# ---------------------------------------------------------------------
# Block-level (non-model) plumbing
# ---------------------------------------------------------------------


class TestBlockNormKwargsDirect:
    """Direct tests of BasicBlock / BottleneckBlock — same plumbing,
    addressed via the block surface (the plan E2 trainer reaches it
    via the ResNet model, but other dl_techniques models also use these
    blocks directly)."""

    def test_basic_block_accepts_kwargs(self):
        blk = BasicBlock(
            filters=32,
            normalization_type="rms_norm",
            normalization_kwargs={"use_scale": False},
        )
        assert blk.normalization_kwargs == {"use_scale": False}
        cfg = blk.get_config()
        assert cfg["normalization_kwargs"] == {"use_scale": False}

    def test_bottleneck_block_accepts_kwargs(self):
        blk = BottleneckBlock(
            filters=32,
            normalization_type="rms_norm",
            normalization_kwargs={"use_scale": False},
        )
        assert blk.normalization_kwargs == {"use_scale": False}
        cfg = blk.get_config()
        assert cfg["normalization_kwargs"] == {"use_scale": False}

    def test_basic_block_default_is_empty(self):
        blk = BasicBlock(filters=32)
        assert blk.normalization_kwargs == {}

    def test_bottleneck_block_default_is_empty(self):
        blk = BottleneckBlock(filters=32)
        assert blk.normalization_kwargs == {}


if __name__ == "__main__":
    import pytest, sys

    sys.exit(pytest.main([__file__, "-vvv"]))


# ---------------------------------------------------------------------
# The README's accuracy claims (C-5)
# ---------------------------------------------------------------------


class TestReadmeQuotesNoAccuracy:
    """`models/resnet/README.md` used to print a per-variant ImageNet Top-1/Top-5
    table, footnoted only "*ImageNet validation accuracy with single crop", plus a
    `Performance (ImageNet)` block per variant and an "Empirical Results (ResNet-50 on
    CIFAR-10)" comparison.

    Every one of those was the published paper's number (or, for the CIFAR block, no
    number at all — no such run exists) presented as though it described this
    implementation. NO pretrained weights exist for any architecture in this repo:
    `_download_weights` raises `NotImplementedError` by design. A prose claim is only
    testable by executing something, so this test executes the refusal and then reads
    the file.
    """

    @staticmethod
    def _readme() -> str:
        import pathlib

        return (
            pathlib.Path(__file__).resolve().parents[3]
            / "src" / "dl_techniques" / "models" / "resnet" / "README.md"
        ).read_text()

    def test_the_premise_holds_no_pretrained_weights_are_obtainable(self):
        """The fact the deletions rest on. If this ever goes RED the README may
        legitimately quote a number again — but only a MEASURED one."""
        import pytest

        from dl_techniques.models.resnet.model import ResNet as _ResNet

        with pytest.raises(NotImplementedError):
            _ResNet.from_variant("resnet18", num_classes=10, pretrained=True)

    def test_no_top1_or_top5_claim_survives_outside_the_explanation(self):
        readme = self._readme()
        # The one surviving mention is the blockquote explaining what was removed and
        # why; it is deliberately quoted there and nowhere else.
        assert readme.count("Top-1 Acc") <= 1
        assert readme.count("Top-5 Acc") <= 1
        assert "Top-1 accuracy:" not in readme
        assert "Performance (ImageNet):" not in readme

    def test_the_explanation_is_present_rather_than_a_silent_deletion(self):
        readme = self._readme()
        assert "No accuracy numbers are quoted anywhere in this README" in readme
        assert "No pretrained weights exist" in readme
