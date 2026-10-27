"""``CliffordNet`` and ``HierarchicalCliffordNet`` with ``include_top=False``
return the final feature map.

The switch mirrors ResNet and ConvNeXt. ``include_top=True`` (the default) is the
headed model, so the head recipe, the parameter pins in
``test_paper_v2_variants.py`` and the logits contract do not depend on it;
``include_top=False`` drops the head and returns the block stack's output
``(B, ceil(H/p), ceil(W/p), channels)``. ``HierarchicalCliffordNet`` has the same
switch; its headless output is the last stage's map, after
``len(stage_depths) - 1`` stride-2 downsamplers.

``from_variant(..., pretrained=<path>, input_shape=...)`` of both models loads the
checkpoint through the shared loader, which builds an unbuilt model at the
default probe shape before the transfer, so a checkpoint loads through
``from_variant`` headed or not, and a load that restores nothing raises.

Every expectation below is derived from the model itself or from the geometry
(``ceil(size / patch)`` for the 'same'-padded strided stems), never from a
number pasted out of a run. The input is non-square and not a multiple of any
patch size, so an axis mix-up or a floor/ceil mix-up changes the answer.
"""

import json
import math
import os

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.cliffordnet.hierarchical import (
    HierarchicalCliffordNet,
)
from dl_techniques.models.vision.cliffordnet.model import CliffordNet
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint

H, W, C_IN = 18, 26, 3
CHANNELS = 16
NUM_CLASSES = 7
BATCH = 2

# Patch 1, 2 and 4 have their own stems; 3 takes the kernel = stride branch.
PATCHES = [1, 2, 3, 4]

HEAD_LAYER_MARKERS = ("classification_head", "pooled_norm", "classifier")


def _model(**overrides) -> CliffordNet:
    kwargs = dict(
        num_classes=NUM_CLASSES,
        channels=CHANNELS,
        depth=2,
        patch_size=2,
        shifts=[1, 2],
    )
    kwargs.update(overrides)
    return CliffordNet(**kwargs)


def _x(seed: int = 0, height: int = H, width: int = W) -> np.ndarray:
    return np.random.default_rng(seed).normal(
        size=(BATCH, height, width, C_IN)
    ).astype("float32")


def _np(t) -> np.ndarray:
    return keras.ops.convert_to_numpy(t)


def _expected_feature_shape(patch: int) -> tuple:
    return (BATCH, math.ceil(H / patch), math.ceil(W / patch), CHANNELS)


def _perturb(model: CliffordNet, seed: int = 11) -> None:
    """Move every weight off its initializer so a transfer is observable."""
    rng = np.random.default_rng(seed)
    for w in model.weights:
        w.assign(
            keras.ops.convert_to_tensor(
                (_np(w) + rng.normal(size=w.shape) * 0.05).astype(w.dtype)
            )
        )


def _spy_on_the_transfer(monkeypatch) -> list:
    """Record the report of every layer-by-layer transfer the shared loader runs."""
    import dl_techniques.utils.weight_transfer as weight_transfer

    reports = []
    real = weight_transfer.load_weights_from_checkpoint

    def spy(*args, **kwargs):
        report = real(*args, **kwargs)
        reports.append(report)
        return report

    monkeypatch.setattr(weight_transfer, "load_weights_from_checkpoint", spy)
    return reports


def _weights_by_path(model) -> dict:
    """Weights keyed by their path below the model name.

    The leading component is the instance name (``clifford_net_12``), which
    differs between two models built in one process, so it is dropped.
    """
    return {w.path.split("/", 1)[1]: _np(w) for w in model.weights}


class TestHeadlessOutput:

    @pytest.mark.parametrize("patch", PATCHES)
    def test_headless_returns_the_feature_map(self, patch):
        model = _model(patch_size=patch, include_top=False)
        out = model(_x(1), training=False)
        out_np = _np(out)

        assert out_np.shape == _expected_feature_shape(patch)
        assert out_np.dtype == np.float32
        assert np.isfinite(out_np).all()
        # A model that annihilated its signal would emit a constant.
        assert float(np.std(out_np)) > 0.0

    @pytest.mark.parametrize("patch", PATCHES)
    def test_compute_output_shape_matches_the_real_output(self, patch):
        model = _model(patch_size=patch, include_top=False)
        real = tuple(_np(model(_x(2), training=False)).shape)

        assert tuple(model.compute_output_shape((BATCH, H, W, C_IN))) == real
        # An unknown batch and unknown spatial sizes stay unknown, not crash.
        assert tuple(model.compute_output_shape((None, H, W, C_IN))) == (
            (None,) + real[1:]
        )
        unknown = model.compute_output_shape((None, None, None, C_IN))
        assert tuple(unknown) == (None, None, None, CHANNELS)

    def test_the_headless_output_is_the_input_of_the_head(self):
        """Headed and headless models share their trunk: same weights, same map."""
        headed = _model(patch_size=2)
        headless = _model(patch_size=2, include_top=False)
        x = _x(3)
        headed(x, training=False)
        headless(x, training=False)
        by_path = {v.path.split("/", 1)[1]: v for v in headed.weights}
        for w in headless.weights:
            w.assign(by_path[w.path.split("/", 1)[1]])

        feature_map = headless(x, training=False)
        logits = _np(headed(x, training=False))
        # Recompute the logits from the headless map through the headed head.
        # The operation sequence is the same one, so the difference is exactly
        # zero: measured on CPU max|diff| = 0.0
        # for input seeds 3, 4 and 5 (max|logit| 2.3 to 2.9). A tolerance would
        # hide a real change of the head's input, so none is allowed.
        recomputed = _np(
            headed.classification_head(feature_map, training=False)["logits"]
        )
        np.testing.assert_allclose(recomputed, logits, rtol=0, atol=0.0)


class TestNoHeadWhenHeadless:

    def test_classification_head_is_none_and_the_attribute_exists(self):
        model = _model(include_top=False)
        assert hasattr(model, "classification_head")
        assert model.classification_head is None
        assert model.include_top is False

    def test_no_head_weights(self):
        model = _model(include_top=False)
        model(_x(4), training=False)
        offenders = [
            w.path
            for w in model.weights
            if any(m in w.path for m in HEAD_LAYER_MARKERS)
        ]
        assert not offenders, f"head weights in a headless model: {offenders}"
        assert model.weights, "the headless model holds no weights at all"
        assert all(
            not any(m in layer.name for m in HEAD_LAYER_MARKERS)
            for layer in model.layers
        )

    @pytest.mark.parametrize("patch", PATCHES)
    def test_param_count_is_the_headed_count_minus_the_head(self, patch):
        headed = _model(patch_size=patch)
        headless = _model(patch_size=patch, include_top=False)
        headed(_x(5), training=False)
        headless(_x(5), training=False)

        head_params = headed.classification_head.count_params()
        assert head_params > 0, "the head carries no parameters to subtract"
        assert headless.count_params() == headed.count_params() - head_params

        head_paths = {
            w.path.split("/", 1)[1] for w in headed.classification_head.weights
        }
        assert head_paths, "the head holds no weights"
        assert set(_weights_by_path(headed)) - set(
            _weights_by_path(headless)
        ) == head_paths


class TestSerialization:

    def test_get_config_round_trip_headless(self):
        model = _model(include_top=False)
        config = model.get_config()
        assert config["include_top"] is False

        clone = CliffordNet.from_config(config)
        assert clone.include_top is False
        assert clone.classification_head is None
        assert clone.get_config() == config
        out = _np(clone(_x(6), training=False))
        assert out.shape == _expected_feature_shape(2)

    def test_get_config_writes_include_top_when_headed(self):
        assert _model().get_config()["include_top"] is True

    def test_keras_round_trip_headless_is_bit_exact(self, tmp_path):
        model = _model(include_top=False)
        x = _x(7)
        model(x, training=False)
        _perturb(model)
        before = _np(model(x, training=False))

        path = os.path.join(str(tmp_path), "headless.keras")
        model.save(path)
        loaded = keras.models.load_model(path)

        assert loaded.include_top is False
        assert loaded.classification_head is None
        after = _np(loaded(x, training=False))
        np.testing.assert_allclose(after, before, rtol=0, atol=0)

        src, dst = _weights_by_path(model), _weights_by_path(loaded)
        assert set(src) == set(dst)
        for path_ in src:
            np.testing.assert_allclose(dst[path_], src[path_], rtol=0, atol=0)


class TestHeadedToHeadlessTransfer:
    """The checkpoint of a headed model feeds a headless one."""

    @staticmethod
    def _checkpoint(tmp_path):
        headed = _model()
        headed(_x(8), training=False)
        _perturb(headed)
        path = os.path.join(str(tmp_path), "headed.keras")
        headed.save(path)
        return headed, path

    def test_the_loader_copies_the_trunk_and_reports_a_nonzero_count(
        self, tmp_path
    ):
        headed, path = self._checkpoint(tmp_path)
        headless = _model(include_top=False)
        headless(_x(9), training=False)

        report = load_weights_from_checkpoint(
            target=headless, ckpt_path=path, skip_prefixes=(), strict=True
        )

        # This loader reports rather than raises on zero matches, so the
        # count is asserted, and it must cover every trunk layer.
        trunk_layers = [layer for layer in headless.layers if layer.weights]
        assert len(report.loaded) > 0
        assert len(report.loaded) == len(trunk_layers)
        assert not report.shape_mismatch
        # The head exists only in the checkpoint.
        assert report.unused_in_source == ["classification_head"]

        src, dst = _weights_by_path(headed), _weights_by_path(headless)
        assert dst, "the headless model holds no weights"
        for path_, value in dst.items():
            np.testing.assert_allclose(value, src[path_], rtol=0, atol=0)

        # A block kernel and the stem kernel, named, so a failure reads clearly.
        for needle in ("stem/stem_conv1/", "block_0/"):
            hits = [p for p in dst if p.startswith(needle) and p.endswith("kernel")]
            assert hits, f"no {needle} kernel among {sorted(dst)[:6]}"
            for p in hits:
                np.testing.assert_allclose(dst[p], src[p], rtol=0, atol=0)

    def test_the_loaded_headless_model_reproduces_the_headed_trunk(
        self, tmp_path
    ):
        headed, path = self._checkpoint(tmp_path)
        x = _x(10)
        headless = _model(include_top=False)
        headless(x, training=False)
        headless.load_pretrained_weights(path)

        feature_map = headless(x, training=False)
        recomputed = _np(
            headed.classification_head(feature_map, training=False)["logits"]
        )
        # Same operation sequence on identical weights: measured on CPU
        # max|diff| = 0.0, so the bound is zero.
        np.testing.assert_allclose(
            recomputed, _np(headed(x, training=False)), rtol=0, atol=0.0
        )

    @pytest.mark.parametrize("num_classes", [NUM_CLASSES, 3])
    def test_from_variant_loads_a_checkpoint_of_another_class_count(
        self, tmp_path, monkeypatch, num_classes
    ):
        """The head of another width is skipped and the trunk still copies.

        The transfer works per layer and skips a shape-mismatched one, whatever
        class count the checkpoint was trained on, so ``from_variant`` has no
        rule of its own about it.
        """
        reports = _spy_on_the_transfer(monkeypatch)
        headed, path = self._checkpoint(tmp_path)

        loaded = CliffordNet.from_variant(
            "nano", num_classes=num_classes, pretrained=path,
            channels=CHANNELS, depth=2,
        )

        (report,) = reports
        trunk = {p: v for p, v in _weights_by_path(loaded).items()
                 if not p.startswith("classification_head")}
        src = _weights_by_path(headed)
        for path_, value in trunk.items():
            np.testing.assert_allclose(value, src[path_], rtol=0, atol=0)
        assert trunk, "the trunk holds no weights"
        assert bool(report.shape_mismatch) == (num_classes != NUM_CLASSES)


class TestNumClasses:

    def test_zero_classes_is_accepted_headless(self):
        model = _model(num_classes=0, include_top=False)
        out = model(_x(12), training=False)
        assert tuple(_np(out).shape) == _expected_feature_shape(2)

    @pytest.mark.parametrize("bad", [0, -3])
    def test_a_non_positive_class_count_is_refused_when_headed(self, bad):
        with pytest.raises(ValueError, match="num_classes"):
            _model(num_classes=bad, include_top=True)

    def test_num_classes_defaults_to_1000_like_resnet(self):
        model = CliffordNet(channels=CHANNELS, depth=1, include_top=False)
        assert model.num_classes == 1000
        assert CliffordNet.from_variant(
            "nano", channels=CHANNELS, depth=1
        ).num_classes == 1000

    def test_other_arguments_are_still_validated_headless(self):
        with pytest.raises(ValueError, match="channels"):
            _model(channels=0, include_top=False)

    def test_summary_runs_headless(self):
        _model(include_top=False).summary()


class TestDefaultIsUnchanged:
    """``include_top=True`` is today's model; the pins run in their own file."""

    def test_the_default_is_headed_with_logits(self):
        model = _model()
        assert model.include_top is True
        assert model.classification_head is not None
        out = _np(model(_x(13), training=False))
        assert out.shape == (BATCH, NUM_CLASSES)
        assert tuple(model.compute_output_shape((BATCH, H, W, C_IN))) == (
            BATCH,
            NUM_CLASSES,
        )

    def test_the_head_keeps_its_name_and_owners(self):
        model = _model()
        model(_x(14), training=False)
        head = model.classification_head
        assert head.name == "classification_head"
        owners = {
            name for name in ("pooled_norm", "classifier")
            if getattr(head, name, None) is not None
        }
        assert owners == {"pooled_norm", "classifier"}

    def test_the_headed_config_round_trips(self):
        model = _model()
        clone = CliffordNet.from_config(model.get_config())
        assert clone.get_config() == model.get_config()


class TestFromVariantBuildsBeforeLoading:
    """``from_variant(pretrained=<path>, input_shape=...)`` loads.

    The checkpoint is a perturbed headed model, so every trunk tensor differs
    from a fresh initialisation: equality at ``rtol=0, atol=0`` proves the
    copy, and the loader is spied on because it reports rather than raises on
    zero matched layers.
    """

    VARIANT_KWARGS = dict(channels=CHANNELS, depth=2)

    @staticmethod
    def _spy(monkeypatch):
        return _spy_on_the_transfer(monkeypatch)

    def _checkpoint(self, tmp_path):
        headed = CliffordNet.from_variant(
            "nano", num_classes=NUM_CLASSES, **self.VARIANT_KWARGS
        )
        headed(_x(20), training=False)
        _perturb(headed)
        path = os.path.join(str(tmp_path), "headed.keras")
        headed.save(path)
        return headed, path

    @pytest.mark.parametrize("input_shape", [(H, W, C_IN), None])
    def test_headless_from_variant_loads_the_trunk(
        self, tmp_path, monkeypatch, input_shape
    ):
        reports = self._spy(monkeypatch)
        headed, path = self._checkpoint(tmp_path)
        extra = {} if input_shape is None else {"input_shape": input_shape}

        headless = CliffordNet.from_variant(
            "nano", include_top=False, pretrained=path,
            **self.VARIANT_KWARGS, **extra,
        )

        assert headless.built
        (report,) = reports
        trunk_layers = [layer for layer in headless.layers if layer.weights]
        assert len(report.loaded) > 0
        assert len(report.loaded) == len(trunk_layers)
        assert not report.shape_mismatch
        src, dst = _weights_by_path(headed), _weights_by_path(headless)
        assert dst, "the headless model holds no weights"
        for path_, value in dst.items():
            np.testing.assert_allclose(value, src[path_], rtol=0, atol=0)

    def test_headed_from_variant_loads_the_head_too(
        self, tmp_path, monkeypatch
    ):
        reports = self._spy(monkeypatch)
        headed, path = self._checkpoint(tmp_path)

        loaded = CliffordNet.from_variant(
            "nano", num_classes=NUM_CLASSES, pretrained=path,
            input_shape=(H, W, C_IN), **self.VARIANT_KWARGS,
        )

        (report,) = reports
        assert len(report.loaded) == len(
            [layer for layer in loaded.layers if layer.weights]
        )
        assert "classification_head" in report.loaded
        src, dst = _weights_by_path(headed), _weights_by_path(loaded)
        assert set(src) == set(dst)
        head_paths = [p for p in dst if "classifier" in p]
        assert head_paths, "no classifier weight to compare"
        for path_ in src:
            np.testing.assert_allclose(dst[path_], src[path_], rtol=0, atol=0)
        x = _x(21)
        np.testing.assert_allclose(
            _np(loaded(x, training=False)), _np(headed(x, training=False)),
            rtol=0, atol=0,
        )

    def test_without_pretrained_or_input_shape_the_model_stays_unbuilt(self):
        model = CliffordNet.from_variant("nano", **self.VARIANT_KWARGS)
        assert not model.built

    @pytest.mark.parametrize("include_top", [True, False])
    def test_input_shape_builds_the_model_without_pretrained(self, include_top):
        """``input_shape`` always reaches the model, as it does for ResNet."""
        model = CliffordNet.from_variant(
            "nano", num_classes=NUM_CLASSES, include_top=include_top,
            input_shape=(H, W, C_IN), **self.VARIANT_KWARGS,
        )
        assert model.built
        assert model.weights, "built but holds no weights"
        expected = (
            (BATCH, NUM_CLASSES) if include_top
            else _expected_feature_shape(model.patch_size)
        )
        assert tuple(
            model.compute_output_shape((BATCH, H, W, C_IN))
        ) == expected
        assert tuple(_np(model(_x(23), training=False)).shape) == expected

    def test_the_probe_shape_constant_is_public_and_shared(self):
        import dl_techniques.models.vision.cliffordnet.hierarchical as hier
        import dl_techniques.models.vision.cliffordnet.model as module

        assert module.DEFAULT_INPUT_SHAPE == (32, 32, 3)
        assert hier.DEFAULT_INPUT_SHAPE is module.DEFAULT_INPUT_SHAPE
        assert not hasattr(module, "_DEFAULT_INPUT_SHAPE")
        assert not hasattr(hier, "_DEFAULT_INPUT_SHAPE")

    @pytest.mark.parametrize("patch", PATCHES)
    def test_the_default_probe_shape_is_valid_for_every_patch_size(
        self, tmp_path, patch
    ):
        headed = CliffordNet.from_variant(
            "nano", num_classes=NUM_CLASSES, patch_size=patch,
            **self.VARIANT_KWARGS,
        )
        headed(_x(22), training=False)
        path = os.path.join(str(tmp_path), "headed.keras")
        headed.save(path)

        headless = CliffordNet.from_variant(
            "nano", include_top=False, pretrained=path, patch_size=patch,
            **self.VARIANT_KWARGS,
        )
        assert headless.built


# ---------------------------------------------------------------------
# HierarchicalCliffordNet
# ---------------------------------------------------------------------

STAGE_DEPTHS = [1, 1, 1]
STAGE_DIMS = [8, 16, 32]
HIER_PATCHES = [1, 2]
#: Even and odd extents: a floor/ceil mix-up in a stride-2 chain shows on odd.
HIER_SIZES = [(18, 26), (17, 23)]


def _hmodel(**overrides) -> HierarchicalCliffordNet:
    kwargs = dict(
        num_classes=NUM_CLASSES,
        stage_depths=list(STAGE_DEPTHS),
        stage_dims=list(STAGE_DIMS),
        patch_size=1,
        shifts=[1, 2],
    )
    kwargs.update(overrides)
    return HierarchicalCliffordNet(**kwargs)


def _hier_expected_shape(
    height: int, width: int, patch: int, dims=STAGE_DIMS
) -> tuple:
    """Geometry: a 'same' stride-p stem, then one 'same' stride-2 per downsampler."""
    h, w = math.ceil(height / patch), math.ceil(width / patch)
    for _ in range(len(dims) - 1):
        h, w = math.ceil(h / 2), math.ceil(w / 2)
    return (BATCH, h, w, dims[-1])


class TestHierarchicalHeadlessOutput:

    @pytest.mark.parametrize("patch", HIER_PATCHES)
    @pytest.mark.parametrize("height, width", HIER_SIZES)
    def test_headless_returns_the_last_stage_map(self, patch, height, width):
        model = _hmodel(patch_size=patch, include_top=False)
        x = _x(1, height, width)
        out = _np(model(x, training=False))

        assert out.shape == _hier_expected_shape(height, width, patch)
        assert out.dtype == np.float32
        assert np.isfinite(out).all()
        assert float(np.std(out)) > 0.0
        # It is the last stage's output before pooling, not a re-derivation.
        np.testing.assert_allclose(
            out,
            _np(model.extract_stage_features(x, training=False)[-1]),
            rtol=0, atol=0,
        )

    @pytest.mark.parametrize("patch", HIER_PATCHES)
    @pytest.mark.parametrize("height, width", HIER_SIZES)
    def test_compute_output_shape_matches_the_real_output(
        self, patch, height, width
    ):
        model = _hmodel(patch_size=patch, include_top=False)
        real = tuple(_np(model(_x(2, height, width), training=False)).shape)

        assert tuple(
            model.compute_output_shape((BATCH, height, width, C_IN))
        ) == real
        assert tuple(
            model.compute_output_shape((None, height, width, C_IN))
        ) == (None,) + real[1:]
        unknown = model.compute_output_shape((None, None, None, C_IN))
        assert tuple(unknown) == (None, None, None, STAGE_DIMS[-1])

    def test_a_one_stage_model_has_no_downsampler(self):
        model = _hmodel(
            stage_depths=[2], stage_dims=[8], patch_size=2, include_top=False
        )
        out = _np(model(_x(3), training=False))
        assert out.shape == _hier_expected_shape(H, W, 2, dims=[8])
        assert tuple(model.compute_output_shape((BATCH, H, W, C_IN))) == (
            out.shape
        )
        assert tuple(
            model.compute_output_shape((None, None, None, C_IN))
        ) == (None, None, None, 8)

    def test_equal_widths_skip_the_pointwise_conv(self):
        model = _hmodel(
            stage_depths=[1, 1], stage_dims=[8, 8], include_top=False
        )
        out = _np(model(_x(4), training=False))
        assert out.shape == _hier_expected_shape(H, W, 1, dims=[8, 8])


class TestHierarchicalNoHeadWhenHeadless:

    def test_classification_head_is_none_and_the_attribute_exists(self):
        model = _hmodel(include_top=False)
        assert hasattr(model, "classification_head")
        assert model.classification_head is None
        assert model.include_top is False

    def test_no_head_weights(self):
        model = _hmodel(include_top=False)
        model(_x(5), training=False)
        offenders = [
            w.path
            for w in model.weights
            if any(m in w.path for m in HEAD_LAYER_MARKERS)
        ]
        assert not offenders, f"head weights in a headless model: {offenders}"
        assert model.weights
        assert all(
            not any(m in layer.name for m in HEAD_LAYER_MARKERS)
            for layer in model.layers
        )

    @pytest.mark.parametrize("patch", HIER_PATCHES)
    def test_param_count_is_the_headed_count_minus_the_head(self, patch):
        headed = _hmodel(patch_size=patch)
        headless = _hmodel(patch_size=patch, include_top=False)
        headed(_x(6), training=False)
        headless(_x(6), training=False)

        head_params = headed.classification_head.count_params()
        assert head_params > 0
        assert headless.count_params() == headed.count_params() - head_params

        head_paths = {
            w.path.split("/", 1)[1] for w in headed.classification_head.weights
        }
        assert head_paths
        assert set(_weights_by_path(headed)) - set(
            _weights_by_path(headless)
        ) == head_paths


class TestHierarchicalSerialization:

    def test_get_config_round_trip_headless(self):
        model = _hmodel(include_top=False)
        config = model.get_config()
        assert config["include_top"] is False

        clone = HierarchicalCliffordNet.from_config(config)
        assert clone.include_top is False
        assert clone.classification_head is None
        assert clone.get_config() == config
        out = _np(clone(_x(7), training=False))
        assert out.shape == _hier_expected_shape(H, W, 1)

    def test_get_config_writes_include_top_when_headed(self):
        assert _hmodel().get_config()["include_top"] is True

    def test_keras_round_trip_headless_is_bit_exact(self, tmp_path):
        model = _hmodel(include_top=False)
        x = _x(8)
        model(x, training=False)
        _perturb(model)
        before = _np(model(x, training=False))

        path = os.path.join(str(tmp_path), "headless.keras")
        model.save(path)
        loaded = keras.models.load_model(path)

        assert loaded.include_top is False
        assert loaded.classification_head is None
        np.testing.assert_allclose(
            _np(loaded(x, training=False)), before, rtol=0, atol=0
        )
        src, dst = _weights_by_path(model), _weights_by_path(loaded)
        assert set(src) == set(dst)
        for path_ in src:
            np.testing.assert_allclose(dst[path_], src[path_], rtol=0, atol=0)


class TestHierarchicalHeadedToHeadlessTransfer:

    @staticmethod
    def _checkpoint(tmp_path):
        headed = _hmodel()
        headed(_x(9), training=False)
        _perturb(headed)
        path = os.path.join(str(tmp_path), "hier_headed.keras")
        headed.save(path)
        return headed, path

    def test_the_loader_copies_the_trunk_and_reports_a_nonzero_count(
        self, tmp_path
    ):
        headed, path = self._checkpoint(tmp_path)
        headless = _hmodel(include_top=False)
        headless(_x(10), training=False)

        report = load_weights_from_checkpoint(
            target=headless, ckpt_path=path, skip_prefixes=(), strict=True
        )

        trunk_layers = [layer for layer in headless.layers if layer.weights]
        assert len(report.loaded) > 0
        assert len(report.loaded) == len(trunk_layers)
        assert not report.shape_mismatch
        assert report.unused_in_source == ["classification_head"]

        src, dst = _weights_by_path(headed), _weights_by_path(headless)
        assert dst
        for path_, value in dst.items():
            np.testing.assert_allclose(value, src[path_], rtol=0, atol=0)
        for needle in ("stem/stem_conv1/", "stage1_block1/", "downsample_0_depthwise/"):
            hits = [p for p in dst if p.startswith(needle) and p.endswith("kernel")]
            assert hits, f"no {needle} kernel among {sorted(dst)[:6]}"

    def test_from_variant_builds_then_loads_headless_and_headed(
        self, tmp_path, monkeypatch
    ):
        reports = _spy_on_the_transfer(monkeypatch)
        headed, path = self._checkpoint(tmp_path)
        kwargs = dict(stage_depths=list(STAGE_DEPTHS), stage_dims=list(STAGE_DIMS))
        src = _weights_by_path(headed)

        headless = HierarchicalCliffordNet.from_variant(
            "hier_345", include_top=False, pretrained=path,
            input_shape=(H, W, C_IN), **kwargs,
        )
        assert headless.built
        assert len(reports[-1].loaded) == len(
            [layer for layer in headless.layers if layer.weights]
        ) > 0
        for path_, value in _weights_by_path(headless).items():
            np.testing.assert_allclose(value, src[path_], rtol=0, atol=0)

        # The default probe shape, and the head loaded when it is wanted.
        loaded = HierarchicalCliffordNet.from_variant(
            "hier_345", num_classes=NUM_CLASSES, pretrained=path, **kwargs,
        )
        assert loaded.built
        assert "classification_head" in reports[-1].loaded
        for path_, value in _weights_by_path(loaded).items():
            np.testing.assert_allclose(value, src[path_], rtol=0, atol=0)
        x = _x(11)
        np.testing.assert_allclose(
            _np(loaded(x, training=False)), _np(headed(x, training=False)),
            rtol=0, atol=0,
        )

    def test_without_pretrained_or_input_shape_the_model_stays_unbuilt(self):
        model = HierarchicalCliffordNet.from_variant(
            "hier_345", num_classes=NUM_CLASSES,
            stage_depths=list(STAGE_DEPTHS), stage_dims=list(STAGE_DIMS),
        )
        assert not model.built

    @pytest.mark.parametrize("include_top", [True, False])
    def test_input_shape_builds_the_model_without_pretrained(self, include_top):
        """``input_shape`` always reaches the model, as it does for ResNet."""
        model = HierarchicalCliffordNet.from_variant(
            "hier_345", num_classes=NUM_CLASSES, include_top=include_top,
            input_shape=(H, W, C_IN),
            stage_depths=list(STAGE_DEPTHS), stage_dims=list(STAGE_DIMS),
        )
        assert model.built
        assert model.weights, "built but holds no weights"
        expected = (
            (BATCH, NUM_CLASSES) if include_top
            else _hier_expected_shape(H, W, model.patch_size)
        )
        assert tuple(
            model.compute_output_shape((BATCH, H, W, C_IN))
        ) == expected
        assert tuple(_np(model(_x(24), training=False)).shape) == expected


class TestHierarchicalNumClasses:

    def test_zero_classes_is_accepted_headless(self):
        model = _hmodel(num_classes=0, include_top=False)
        out = model(_x(12), training=False)
        assert tuple(_np(out).shape) == _hier_expected_shape(H, W, 1)

    @pytest.mark.parametrize("bad", [0, -3])
    def test_a_non_positive_class_count_is_refused_when_headed(self, bad):
        with pytest.raises(ValueError, match="num_classes"):
            _hmodel(num_classes=bad, include_top=True)

    def test_num_classes_defaults_to_1000_like_resnet(self):
        model = HierarchicalCliffordNet(include_top=False)
        assert model.num_classes == 1000
        assert HierarchicalCliffordNet.from_variant("hier_345").num_classes == 1000

    def test_other_arguments_are_still_validated_headless(self):
        with pytest.raises(ValueError, match="stage_depths"):
            _hmodel(stage_depths=[], stage_dims=[], include_top=False)

    def test_summary_runs_headless(self):
        _hmodel(include_top=False).summary()


class TestHierarchicalDefaultIsUnchanged:

    def test_the_default_is_headed_with_logits(self):
        model = _hmodel()
        assert model.include_top is True
        assert model.classification_head is not None
        out = _np(model(_x(13), training=False))
        assert out.shape == (BATCH, NUM_CLASSES)
        assert tuple(model.compute_output_shape((BATCH, H, W, C_IN))) == (
            BATCH,
            NUM_CLASSES,
        )

    def test_the_head_keeps_its_name_and_owners(self):
        model = _hmodel()
        model(_x(14), training=False)
        head = model.classification_head
        assert head.name == "classification_head"
        owners = {
            name for name in ("pooled_norm", "classifier")
            if getattr(head, name, None) is not None
        }
        assert owners == {"pooled_norm", "classifier"}

    def test_the_headed_config_round_trips(self):
        model = _hmodel()
        clone = HierarchicalCliffordNet.from_config(model.get_config())
        assert clone.get_config() == model.get_config()


# ---------------------------------------------------------------------
# Positional argument order
# ---------------------------------------------------------------------


class TestIncludeTopIsNotASecondPositionalArgument:
    """``include_top`` once sat second, so ``CliffordNet(10, 64)`` bound 64 to it.

    Measured on the committed code: ``CliffordNet(10, 64)`` built with
    ``include_top=64, channels=128`` and ``HierarchicalCliffordNet(10, [1, 1])``
    with ``include_top=[1, 1]`` and the default stage depths, with no error.
    Every pre-existing parameter keeps its original positional slot and
    ``include_top`` follows them all, and it must be a real ``bool``.
    """

    def test_the_second_positional_argument_is_channels(self):
        model = CliffordNet(10, 64)
        assert model.channels == 64
        assert model.include_top is True

    def test_the_second_positional_argument_is_stage_depths(self):
        model = HierarchicalCliffordNet(10, [1, 1])
        assert model.stage_depths == [1, 1]
        assert model.include_top is True

    @pytest.mark.parametrize("bad", [64, [1, 1], 1, 0, "yes", None])
    def test_a_non_bool_include_top_is_refused(self, bad):
        with pytest.raises(TypeError, match="include_top"):
            CliffordNet(num_classes=10, channels=8, depth=1, include_top=bad)

    @pytest.mark.parametrize("bad", [64, [1, 1], 1, 0, "yes", None])
    def test_a_non_bool_include_top_is_refused_hierarchical(self, bad):
        with pytest.raises(TypeError, match="include_top"):
            HierarchicalCliffordNet(
                num_classes=10, stage_depths=[1], stage_dims=[8],
                include_top=bad,
            )

    def test_the_refusal_names_the_offending_value(self):
        with pytest.raises(TypeError, match=r"\[1, 1\]"):
            HierarchicalCliffordNet(10, include_top=[1, 1])

    @pytest.mark.parametrize("flag", [True, False])
    def test_a_real_bool_is_accepted_by_keyword(self, flag):
        assert CliffordNet(10, depth=1, include_top=flag).include_top is flag
        assert HierarchicalCliffordNet(
            10, include_top=flag
        ).include_top is flag


# ---------------------------------------------------------------------
# numpy.bool_
# ---------------------------------------------------------------------


def _both(flag, **overrides):
    """Both models, built with ``include_top=flag`` and a fixed name."""
    clifford = CliffordNet(
        num_classes=10, channels=8, depth=1, include_top=flag,
        name="numpy_flag_clifford", **overrides)
    hierarchical = HierarchicalCliffordNet(
        num_classes=10, stage_depths=[1], stage_dims=[8], include_top=flag,
        name="numpy_flag_hierarchical", **overrides)
    return clifford, hierarchical


class TestIncludeTopAcceptsANumpyBool:
    """``numpy.bool_`` is a real boolean, produced by every comparison on an array.

    Measured on the committed code: ``CliffordNet(10, depth=1, channels=8,
    include_top=np.bool_(False))`` raised "include_top must be a bool, got
    np.False_ (bool)": under numpy 2 ``type(np.False_).__name__`` is ``'bool'``,
    so the message contradicted itself. The accepted value is coerced to a
    Python ``bool`` so ``get_config`` stays JSON-serializable.
    """

    @pytest.mark.parametrize("flag", [True, False])
    def test_a_numpy_bool_builds_the_same_model_as_the_python_bool(self, flag):
        for numpy_model, python_model in zip(
                _both(np.bool_(flag)), _both(flag)):
            assert numpy_model.include_top is flag
            assert numpy_model.get_config() == python_model.get_config()
            assert (numpy_model.classification_head is None) is (not flag)

    @pytest.mark.parametrize("flag", [True, False])
    def test_the_config_stores_a_python_bool(self, flag):
        for model in _both(np.bool_(flag)):
            stored = model.get_config()["include_top"]
            assert type(stored) is bool and stored is flag
            json.dumps(model.get_config())  # a numpy bool would raise here

    @pytest.mark.parametrize("flag", [True, False])
    def test_a_numpy_bool_model_round_trips_its_config(self, flag):
        for model in _both(np.bool_(flag)):
            clone = type(model).from_config(model.get_config())
            assert clone.include_top is flag

    @pytest.mark.parametrize(
        "bad",
        [64, [1, 1], 0, 1, "yes", None, np.int64(1), np.array(True)],
        ids=["64", "list", "0", "1", "yes", "None", "np.int64", "0-d array"],
    )
    def test_everything_else_is_still_refused_and_named_unambiguously(self, bad):
        """The refusal prints the repr AND the qualified type of the value.

        ``np.int64(1)`` and a 0-d ``np.array(True)`` guard against loosening
        the check to ``bool(...)``: only a ``bool`` or ``numpy.bool_`` passes.
        """
        kind = type(bad)
        qualified = f"{kind.__module__}.{kind.__qualname__}"
        constructors = (
            lambda: CliffordNet(num_classes=10, channels=8, depth=1,
                                include_top=bad),
            lambda: HierarchicalCliffordNet(
                num_classes=10, stage_depths=[1], stage_dims=[8],
                include_top=bad),
        )
        for construct in constructors:
            with pytest.raises(TypeError) as caught:
                construct()
            message = str(caught.value)
            assert "include_top" in message
            assert repr(bad) in message
            assert qualified in message, (qualified, message)

    def test_the_message_does_not_call_a_numpy_bool_a_bool_it_refuses(self):
        """The old contradiction: "got np.False_ (bool)" on a refused value.

        No ``bool`` or ``numpy.bool_`` is refused any more, so the message can
        never print a ``bool`` type for a value it rejected; the qualified name
        of a refused numpy scalar is its own (``numpy.int64``), not ``bool``.
        """
        with pytest.raises(TypeError) as caught:
            CliffordNet(num_classes=10, channels=8, depth=1,
                        include_top=np.int64(1))
        assert "numpy.int64" in str(caught.value)
        assert "(bool)" not in str(caught.value)
