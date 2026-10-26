"""``CliffordNet`` and ``HierarchicalCliffordNet`` with ``include_top=False``
return the final feature map.

Plan plan-2026-10-26T125959-7af851e5 step 4 (D-011): the switch mirrors ResNet
and ConvNeXt. ``include_top=True`` (the default) is the model as it was, so the
head recipe, the parameter pins in ``test_paper_v2_variants.py`` and the logits
contract are untouched; ``include_top=False`` drops the head and returns the
block stack's output ``(B, ceil(H/p), ceil(W/p), channels)``.

Step 5 gave ``HierarchicalCliffordNet`` the same switch (its headless output is
the last stage's map, after ``len(stage_depths) - 1`` stride-2 downsamplers) and,
by D-017, made ``from_variant(..., pretrained=<path>, input_shape=...)`` of BOTH
models build the model before the weight transfer: ``from_variant`` builds
nothing and the loader refuses an unbuilt target, so before this a checkpoint
could not be loaded through ``from_variant`` at all, headed or not.

Every expectation below is derived from the model itself or from the geometry
(``ceil(size / patch)`` for the 'same'-padded strided stems), never from a
number pasted out of a run. The input is non-square and not a multiple of any
patch size, so an axis mix-up or a floor/ceil mix-up changes the answer.
"""

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
        recomputed = _np(
            headed.classification_head(feature_map, training=False)["logits"]
        )
        np.testing.assert_allclose(recomputed, logits, rtol=0, atol=1e-6)


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

        # The loader reports rather than raises on zero matches (Pre-Mortem 3),
        # so the count is asserted, and it must cover every trunk layer.
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
        for needle in ("stem_conv1", "clifford_block_0"):
            hits = [p for p in dst if needle in p and p.endswith("kernel")]
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
        np.testing.assert_allclose(
            recomputed, _np(headed(x, training=False)), rtol=0, atol=1e-6
        )

    @pytest.mark.parametrize(
        "kwargs, expect_skip",
        [
            # Headless: no classifier exists, so a class count that differs
            # from the hosted 100 (here the default 1000) must not relax the
            # loader.
            ({"include_top": False}, False),
            ({"include_top": False, "num_classes": 7}, False),
            # Headed: a width that differs from the hosted 100 skips the
            # classifier, as before.
            ({"num_classes": 7}, True),
            ({"num_classes": 100}, False),
        ],
    )
    def test_from_variant_gates_the_classifier_skip_on_include_top(
        self, monkeypatch, kwargs, expect_skip
    ):
        seen = {}

        def record(self, weights_path, skip_mismatch=True):
            seen["skip_mismatch"] = skip_mismatch

        monkeypatch.setattr(CliffordNet, "load_pretrained_weights", record)
        CliffordNet.from_variant(
            "nano",
            pretrained="unused.keras",
            channels=CHANNELS,
            depth=1,
            **kwargs,
        )
        assert seen == {"skip_mismatch": expect_skip}


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
    """D-017: ``from_variant(pretrained=<path>, input_shape=...)`` loads.

    The checkpoint is a perturbed headed model, so every trunk tensor differs
    from a fresh initialisation: equality at ``rtol=0, atol=0`` proves the
    copy, and the loader is spied on because it reports rather than raises on
    zero matched layers.
    """

    VARIANT_KWARGS = dict(channels=CHANNELS, depth=2)

    @staticmethod
    def _spy(monkeypatch, module):
        reports = []
        real = module.load_weights_from_checkpoint

        def spy(*args, **kwargs):
            report = real(*args, **kwargs)
            reports.append(report)
            return report

        monkeypatch.setattr(module, "load_weights_from_checkpoint", spy)
        return reports

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
        import dl_techniques.models.vision.cliffordnet.model as module

        reports = self._spy(monkeypatch, module)
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
        import dl_techniques.models.vision.cliffordnet.model as module

        reports = self._spy(monkeypatch, module)
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

    def test_without_pretrained_the_model_stays_unbuilt(self):
        model = CliffordNet.from_variant(
            "nano", input_shape=(H, W, C_IN), **self.VARIANT_KWARGS
        )
        assert not model.built

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
        for needle in ("stem_conv1", "stage0_block_0", "downsample_0_depthwise"):
            hits = [p for p in dst if needle in p and p.endswith("kernel")]
            assert hits, f"no {needle} kernel among {sorted(dst)[:6]}"

    def test_from_variant_builds_then_loads_headless_and_headed(
        self, tmp_path, monkeypatch
    ):
        import dl_techniques.models.vision.cliffordnet.hierarchical as module

        reports = []
        real = module.load_weights_from_checkpoint

        def spy(*args, **kwargs):
            report = real(*args, **kwargs)
            reports.append(report)
            return report

        monkeypatch.setattr(module, "load_weights_from_checkpoint", spy)
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

    def test_without_pretrained_the_model_stays_unbuilt(self):
        model = HierarchicalCliffordNet.from_variant(
            "hier_345", num_classes=NUM_CLASSES, input_shape=(H, W, C_IN),
            stage_depths=list(STAGE_DEPTHS), stage_dims=list(STAGE_DIMS),
        )
        assert not model.built


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
