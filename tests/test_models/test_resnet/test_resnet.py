"""
Test suite for the ResNet model.

Scope:

1. Default-off `normalization_kwargs` plumbing (`normalization_kwargs=None`)
   resolves to `{}` for every normalization type EXCEPT `batch_norm`, where
   plan-2026-08-23T091307-9a110062/D-480 injects the reference
   `momentum=0.9` (torchvision `BatchNorm2d`'s `momentum=0.1`; the two
   frameworks define momentum oppositely, `keras = 1 - torch`). That
   injection is training-only: it changes no weight shape and does not
   enter the `training=False` forward pass, so the bit-exactness
   invariant this file was written to protect still holds.
2. Explicit `normalization_kwargs={"use_scale": False}` propagates to
   the stem and all in-block norm layers when paired with a norm class
   that supports the kwarg (e.g. `rms_norm`).
3. `get_config` round-trips the kwarg so existing serialization
   contract is preserved.
4. End-to-end forward pass with a non-`batch_norm` normalization type
   does not raise.
5. `compute_output_shape` equals the forward shape on an unbuilt model,
   `classification_head` is defined when headless, `from_variant` takes one
   contract, a local checkpoint loads through the shared loader, and
   `kernel_initializer` reaches every convolution without replaying a draw.
"""

import copy
import inspect
import json
import os

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.resnet.model import ResNet, create_resnet
from dl_techniques.layers.conv_blocks.basic_block import BasicBlock
from dl_techniques.layers.conv_blocks.bottleneck_block import BottleneckBlock

from ..initializer_replay_oracle import replayed_pairs


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
    def test_batch_norm_default_injects_the_reference_momentum(self):
        """D-480: `batch_norm` with no explicit kwargs gets `momentum=0.9`.

        0.9 is torchvision `BatchNorm2d`'s `momentum=0.1` expressed in Keras'
        opposite convention (`keras_momentum = 1 - torch_momentum`); Keras'
        own bare default is 0.99. This assertion replaced an
        `== {}` one, which pinned the pre-D-480 behaviour.
        """
        model = _small_resnet18()
        assert model.normalization_kwargs == {"momentum": 0.9}

    def test_the_injection_is_conditional_on_batch_norm(self):
        """The negative arm: `momentum` is not an accepted kwarg of the other
        factory types, so a non-`batch_norm` default must still be `{}`."""
        model = _small_resnet18(normalization_type="rms_norm")
        assert model.normalization_kwargs == {}

    def test_an_explicit_momentum_wins_over_the_injection(self):
        model = _small_resnet18(normalization_kwargs={"momentum": 0.99})
        assert model.normalization_kwargs == {"momentum": 0.99}

    def test_default_off_stem_bn_factory_call(self):
        """The stem norm was created with no extra kwargs — its name is 'stem_norm1'."""
        model = _small_resnet18()
        assert model.stem.norm1.name == "stem_norm1"

    def test_default_off_blocks_receive_the_resolved_kwargs(self):
        """Every block gets the SAME resolved dict the stem does — the
        injection happens once, in `__init__`, not per construction site."""
        model = _small_resnet18()
        for stage_blocks in model.stages:
            for blk in stage_blocks:
                assert isinstance(blk, (BasicBlock, BottleneckBlock))
                assert blk.normalization_kwargs == {"momentum": 0.9}
        rms = _small_resnet18(normalization_type="rms_norm")
        for stage_blocks in rms.stages:
            for blk in stage_blocks:
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
    contract the norm-variants sweep relies on in its `param_matched` mode."""

    def test_stem_uses_scale_false(self):
        model = _small_resnet18(
            normalization_type="rms_norm",
            normalization_kwargs={"use_scale": False},
        )
        assert getattr(model.stem.norm1, "use_scale", None) is False
        assert getattr(model.stem.norm1, "gamma", None) is None

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

    def test_the_resolved_default_round_trips(self):
        """The RESOLVED dict is what `get_config` reports, so a reloaded model
        keeps the momentum the artifact was saved with rather than re-deriving
        it from whatever the current default happens to be."""
        model = _small_resnet18()
        cfg = model.get_config()
        assert cfg["normalization_kwargs"] == {"momentum": 0.9}
        restored = ResNet.from_config(cfg)
        assert restored.normalization_kwargs == {"momentum": 0.9}

        rms = _small_resnet18(normalization_type="rms_norm")
        assert rms.get_config()["normalization_kwargs"] == {}
        assert ResNet.from_config(rms.get_config()).normalization_kwargs == {}


# ---------------------------------------------------------------------
# Block-level (non-model) plumbing
# ---------------------------------------------------------------------


class TestBlockNormKwargsDirect:
    """Direct tests of BasicBlock / BottleneckBlock — same plumbing,
    addressed via the block surface (the norm-variants sweep reaches it
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


# ---------------------------------------------------------------------
# compute_output_shape, the headless attribute, from_variant
# ---------------------------------------------------------------------


def _tiny(**overrides):
    """A two-stage ResNet small enough to build in well under a second."""
    cfg = dict(
        num_classes=5,
        blocks_per_stage=[1, 1],
        filters_per_stage=[8, 16],
        block_type="basic",
        input_shape=(32, 32, 3),
    )
    cfg.update(overrides)
    return ResNet(**cfg)


def _forward_shape(model, side):
    """The shape `call` produces for a `side` x `side` image, batch axis None."""
    out = model(np.zeros((1, side, side, 3), "float32"), training=False)
    if isinstance(out, list):
        return [(None, *tuple(o.shape)[1:]) for o in out]
    return (None, *tuple(out.shape)[1:])


class TestResNetComputeOutputShape:
    @pytest.mark.parametrize("stem_type", ["imagenet", "cifar"])
    @pytest.mark.parametrize("block_type", ["basic", "bottleneck"])
    @pytest.mark.parametrize("size", [32, 33, None])
    @pytest.mark.parametrize("include_top", [True, False])
    def test_it_equals_the_forward_shape_on_an_unbuilt_model(
        self, stem_type, block_type, size, include_top
    ):
        model = _tiny(stem_type=stem_type, block_type=block_type, include_top=include_top)
        got = model.compute_output_shape((None, size, size, 3))
        assert not model.built, "compute_output_shape built the model"

        want = _forward_shape(model, 32 if size is None else size)
        if size is None and not include_top:
            want = want[:1] + (None, None) + want[3:]
        assert tuple(got) == want

    def test_the_batch_axis_is_passed_through(self):
        assert _tiny().compute_output_shape((4, 32, 32, 3)) == (4, 5)
        assert _tiny(include_top=False).compute_output_shape((4, 32, 32, 3))[0] == 4

    def test_deep_supervision_reports_one_shape_per_head(self):
        model = _tiny(
            blocks_per_stage=[1, 1, 1],
            filters_per_stage=[8, 16, 32],
            enable_deep_supervision=True,
        )
        got = model.compute_output_shape((None, 33, 33, 3))
        want = _forward_shape(model, 33)
        assert isinstance(got, list) and len(got) == 3
        assert [tuple(s) for s in got] == want


class TestResNetHeadlessAttribute:
    def test_a_headless_model_defines_its_head_as_none(self):
        model = _tiny(include_top=False)
        assert hasattr(model, "classification_head")
        assert model.classification_head is None

    def test_a_headed_model_holds_its_head(self):
        assert _tiny().classification_head is not None


class TestResNetFromVariant:
    def test_an_unknown_variant_lists_the_keys(self):
        with pytest.raises(ValueError, match="Unknown variant 'nope'") as raised:
            ResNet.from_variant("nope")
        assert all(key in str(raised.value) for key in ResNet.MODEL_VARIANTS)

    @pytest.mark.parametrize("build", [ResNet.from_variant, create_resnet])
    def test_pretrained_true_names_the_variant(self, build):
        with pytest.raises(NotImplementedError, match="variant 'resnet34'"):
            build("resnet34", pretrained=True)

    def test_a_missing_checkpoint_raises_before_the_model_is_built(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ResNet.from_variant("resnet18", pretrained=str(tmp_path / "absent.keras"))

    @pytest.mark.parametrize("build", [ResNet.from_variant, create_resnet])
    def test_the_signature_is_the_house_contract(self, build):
        assert list(inspect.signature(build).parameters) == [
            "variant", "num_classes", "input_shape", "pretrained", "kwargs"
        ]
        assert inspect.signature(build).parameters["input_shape"].default is None

    @pytest.mark.parametrize(
        "knob", ["weights_dataset", "weights_input_shape", "cache_dir"]
    )
    def test_a_removed_download_knob_is_refused(self, knob):
        with pytest.raises(ValueError, match=knob):
            ResNet.from_variant("resnet18", **{knob: "imagenet"})

    def test_an_override_reaches_the_model_and_leaves_the_preset_alone(self):
        before = {k: dict(v) for k, v in ResNet.MODEL_VARIANTS.items()}
        model = ResNet.from_variant(
            "resnet18", num_classes=5, blocks_per_stage=[1, 1, 1, 1], input_shape=(32, 32, 3)
        )
        assert model.blocks_per_stage == [1, 1, 1, 1]
        assert ResNet.MODEL_VARIANTS == before

    def test_the_unused_input_attributes_and_the_download_stub_are_gone(self):
        model = _tiny()
        for name in ("input_height", "input_width", "input_channels"):
            assert not hasattr(model, name)
        assert not hasattr(ResNet, "_download_weights")


class TestResNetPretrainedLoad:
    def test_a_checkpoint_of_the_same_architecture_loads_its_tensors(self, tmp_path):
        donor = _tiny()
        donor.build((None, 32, 32, 3))
        path = os.path.join(tmp_path, "donor.keras")
        donor.save(path)

        receiver = ResNet.from_variant(
            "resnet18", num_classes=5, pretrained=path,
            blocks_per_stage=[1, 1], filters_per_stage=[8, 16], block_type="basic",
            input_shape=(32, 32, 3),
        )
        for a, b in zip(donor.weights, receiver.weights):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_a_checkpoint_that_restores_no_layer_raises(self, tmp_path):
        donor = _tiny()
        donor.build((None, 32, 32, 3))
        path = os.path.join(tmp_path, "donor.keras")
        donor.save(path)

        receiver = _tiny(filters_per_stage=[16, 32], num_classes=7)
        with pytest.raises(ValueError, match="stage1_block1"):
            receiver.load_pretrained_weights(path)

    @staticmethod
    def _saved(tmp_path, **overrides):
        donor = _tiny(**overrides)
        donor.build((None, 32, 32, 3))
        path = os.path.join(tmp_path, "donor.keras")
        donor.save(path)
        return donor, path

    def test_a_resnet50_checkpoint_does_not_load_into_resnet18(self, tmp_path):
        """It used to restore the stem alone, report the rest and return."""
        donor = ResNet.from_variant("resnet50", num_classes=10, input_shape=(32, 32, 3))
        donor.build((None, 32, 32, 3))
        path = os.path.join(tmp_path, "resnet50.keras")
        donor.save(path)

        with pytest.raises(ValueError) as caught:
            ResNet.from_variant(
                "resnet18", num_classes=10, input_shape=(32, 32, 3), pretrained=path
            )

        message = str(caught.value)
        assert "stage1_block1" in message, message
        assert "shape_mismatch" in message and "missing_in_source" in message, message

    def test_a_trunk_of_another_block_type_raises_and_writes_nothing(self, tmp_path):
        _, path = self._saved(tmp_path, block_type="bottleneck")
        receiver = _tiny()
        before = [np.asarray(w).copy() for w in receiver.weights]

        with pytest.raises(ValueError, match="stage1_block1"):
            receiver.load_pretrained_weights(path)

        for a, b in zip(before, receiver.weights):
            np.testing.assert_array_equal(a, np.asarray(b))

    def test_another_class_count_loads_the_trunk_and_skips_the_head(self, tmp_path):
        donor, path = self._saved(tmp_path, num_classes=10)

        receiver = ResNet.from_variant(
            "resnet18", num_classes=5, pretrained=path,
            blocks_per_stage=[1, 1], filters_per_stage=[8, 16], block_type="basic",
            input_shape=(32, 32, 3),
        )

        donor_weights = {w.path.split("/", 1)[1]: np.asarray(w) for w in donor.weights}
        trunk = [w for w in receiver.weights if "classification_head" not in w.path]
        assert trunk, "the receiver holds no trunk weight"
        for w in trunk:
            np.testing.assert_array_equal(
                np.asarray(w), donor_weights[w.path.split("/", 1)[1]], err_msg=w.path
            )
        head = [w for w in receiver.weights if "classification_head" in w.path]
        assert head and all(w.shape[-1] == 5 or w.shape == (5,) for w in head)

    def test_a_headless_model_loads_the_trunk_of_a_headed_checkpoint(self, tmp_path):
        donor, path = self._saved(tmp_path)

        receiver = ResNet.from_variant(
            "resnet18", include_top=False, pretrained=path,
            blocks_per_stage=[1, 1], filters_per_stage=[8, 16], block_type="basic",
            input_shape=(32, 32, 3),
        )

        donor_weights = {w.path.split("/", 1)[1]: np.asarray(w) for w in donor.weights}
        assert receiver.weights
        for w in receiver.weights:
            np.testing.assert_array_equal(
                np.asarray(w), donor_weights[w.path.split("/", 1)[1]], err_msg=w.path
            )


class TestResNetKernelInitializer:
    @staticmethod
    def _convs(model):
        return [l for l in model._flatten_layers() if isinstance(l, keras.layers.Conv2D)]

    @pytest.mark.parametrize("block_type, n_convs", [("basic", 6), ("bottleneck", 9)])
    def test_the_default_is_he_normal_at_every_convolution(self, block_type, n_convs):
        model = _tiny(block_type=block_type)
        model.build((None, 32, 32, 3))
        convs = self._convs(model)
        assert len(convs) == n_convs
        assert all(isinstance(c.kernel_initializer, keras.initializers.HeNormal) for c in convs)
        assert model.get_config()["kernel_initializer"] == "he_normal"

    @pytest.mark.parametrize("block_type, n_convs", [("basic", 6), ("bottleneck", 9)])
    def test_the_knob_reaches_the_stem_and_every_block_convolution(self, block_type, n_convs):
        model = _tiny(block_type=block_type, kernel_initializer="lecun_normal")
        model.build((None, 32, 32, 3))
        convs = self._convs(model)
        assert len(convs) == n_convs
        assert all(isinstance(c.kernel_initializer, keras.initializers.LecunNormal) for c in convs)

    @pytest.mark.parametrize("block_type", ["basic", "bottleneck"])
    def test_an_initializer_instance_replays_no_draw_and_the_config_is_json_safe(self, block_type):
        model = _tiny(block_type=block_type, kernel_initializer=keras.initializers.HeNormal())
        json.dumps(model.get_config())
        model.build((None, 32, 32, 3))
        pairs, compared = replayed_pairs(model)
        assert compared > 0
        assert pairs == [], f"{len(pairs)} of {compared} kernel pairs replay one draw: {pairs[:2]}"

    def test_the_config_round_trips_an_initializer_instance(self):
        model = _tiny(kernel_initializer=keras.initializers.LecunNormal())
        config = model.get_config()
        rebuilt = ResNet.from_config(json.loads(json.dumps(config)))
        assert json.loads(json.dumps(rebuilt.get_config())) == json.loads(json.dumps(config))

    def test_an_unknown_initializer_name_is_refused_by_name(self):
        with pytest.raises(ValueError, match="no_such_init"):
            _tiny(kernel_initializer="no_such_init")

    def test_a_seeded_instance_reuses_its_seed_at_every_site_by_contract(self):
        """The documented exception to 'no two kernels replay one draw'.

        A seeded ``Initializer`` instance is serialized with its seed, and every
        convolution resolves that spec: same-shaped kernels start equal. The
        docstrings say so; this test pins the statement they make.
        """
        model = _tiny(kernel_initializer=keras.initializers.HeNormal(seed=7))
        model.build((None, 32, 32, 3))
        pairs, compared = replayed_pairs(model)
        assert compared > 0
        assert pairs, "a seeded instance is documented to tie same-shaped kernels"


class TestResNetFromConfig:
    """``from_config`` is the inherited one: ``__init__`` accepts a serialized regularizer."""

    def test_a_serialized_regularizer_is_rebuilt_and_the_dict_is_left_alone(self):
        model = _tiny(kernel_regularizer=keras.regularizers.L2(1e-3))
        config = json.loads(json.dumps(model.get_config()))
        before = copy.deepcopy(config)
        rebuilt = ResNet.from_config(config)
        assert config == before, "from_config rewrote the dict it was given"
        assert isinstance(rebuilt.kernel_regularizer, keras.regularizers.L2)
        assert float(rebuilt.kernel_regularizer.l2) == pytest.approx(1e-3)

    def test_the_class_defines_no_from_config_of_its_own(self):
        assert "from_config" not in vars(ResNet)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-vvv"]))


# ---------------------------------------------------------------------
# The README's accuracy claims
# ---------------------------------------------------------------------


class TestReadmeQuotesNoAccuracy:
    """`models/vision/resnet/README.md` used to print a per-variant ImageNet Top-1/Top-5
    table, footnoted only "*ImageNet validation accuracy with single crop", plus a
    `Performance (ImageNet)` block per variant and an "Empirical Results (ResNet-50 on
    CIFAR-10)" comparison.

    Every one of those was the published paper's number (or, for the CIFAR block, no
    number at all — no such run exists) presented as though it described this
    implementation. NO pretrained weights exist for any architecture in this repo:
    `from_variant(pretrained=True)` raises `NotImplementedError` by design. A prose claim is only
    testable by executing something, so this test executes the refusal and then reads
    the file.
    """

    @staticmethod
    def _readme() -> str:
        import pathlib

        return (
            pathlib.Path(__file__).resolve().parents[3]
            / "src" / "dl_techniques" / "models" / "vision" / "resnet" / "README.md"
        ).read_text()

    def test_the_premise_holds_no_pretrained_weights_are_obtainable(self):
        """The fact the deletions rest on. If this ever goes RED the README may
        legitimately quote a number again — but only a MEASURED one."""
        import pytest

        from dl_techniques.models.vision.resnet.model import ResNet as _ResNet

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
