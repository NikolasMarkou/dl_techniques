"""The CliffordNet public contract: factory, pretrained refusal, API surface, initializers.

These tests pin:
1. `create_cliffordnet` returns a model that builds, and `from_variant` takes
   the one house signature with no download or shortcut surface left.
2. `pretrained=True` raises `NotImplementedError` naming the variant (no silent
   fallback).
3. The public API of `dl_techniques.models.vision.cliffordnet`: the four names
   and the two modules' shared builders.
4. The model stores each initializer as a spec and every random kernel follows
   the seed, so no draw is shared between layers or fixed at import time.
5. `dropout_rate` and `stochastic_depth_rate` are validated by the model itself,
   with its own argument name in the message.
"""

import inspect
import json

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.cliffordnet.model import CliffordNet

#: ``from_variant`` takes exactly these named parameters, then ``**kwargs``.
FROM_VARIANT_PARAMETERS = ["variant", "num_classes", "input_shape", "pretrained"]


class TestCliffordNetFactoryAndSurface:
    """The factory, the pretrained refusal and the exported names."""

    def test_create_cliffordnet_factory_returns_instance(self):
        from dl_techniques.models.vision.cliffordnet import (
            CliffordNet,
            create_cliffordnet,
        )

        model = create_cliffordnet(variant="nano", num_classes=10)
        assert isinstance(model, CliffordNet)

        # Build with a small input to verify the factory wires through.
        model.build((None, 32, 32, 3))
        assert model.built

    def test_pretrained_true_raises_not_implemented(self):
        from dl_techniques.models.vision.cliffordnet import create_cliffordnet

        with pytest.raises(NotImplementedError):
            create_cliffordnet(
                variant="nano", num_classes=10, pretrained=True
            )

    def test_public_api_surface(self):
        import dl_techniques.models.vision.cliffordnet as pkg

        # The surface is the two classifiers and their factories. The embedding
        # and Laplacian U-Net names that used to be listed went with the DSv2
        # block and the modules behind them, and CliffordCLIP lives in
        # vision_language/clifford_clip, so none of them is re-exported here. The
        # hierarchical pair is exported on purpose: the pretrained behavioural
        # sweep in test_package_api_contract.py finds factories by walking the
        # package exports.
        assert set(pkg.__all__) == {
            "CliffordNet",
            "create_cliffordnet",
            "HierarchicalCliffordNet",
            "create_hierarchical_cliffordnet",
        }


class TestCliffordNetHouseSurface:
    """One ``from_variant`` contract; no download, shortcut or log-dump surface."""

    def test_from_variant_takes_the_house_signature(self):
        parameters = list(inspect.signature(CliffordNet.from_variant).parameters.values())
        assert [p.name for p in parameters if p.kind is not p.VAR_KEYWORD] == FROM_VARIANT_PARAMETERS
        assert parameters[-1].kind is inspect.Parameter.VAR_KEYWORD

    @pytest.mark.parametrize(
        "name", ["nano", "lite", "lite_g", "_download_weights", "_apply_stem", "summary"]
    )
    def test_the_removed_members_are_gone(self, name):
        assert name not in vars(CliffordNet), name

    def test_an_unknown_variant_lists_the_variants(self):
        with pytest.raises(ValueError, match="Unknown variant 'nope'. Available variants: "):
            CliffordNet.from_variant("nope")

    def test_pretrained_true_names_the_variant(self):
        with pytest.raises(NotImplementedError, match="variant 'lite'"):
            CliffordNet.from_variant("lite", pretrained=True)


class TestCliffordNetValidatesItsRates:
    """The model refuses a bad rate itself and names its own argument and the value."""

    @pytest.mark.parametrize(
        "knob, value",
        [("dropout_rate", 2.0), ("dropout_rate", 1.0), ("dropout_rate", -0.1),
         ("stochastic_depth_rate", -1.0), ("stochastic_depth_rate", 1.0)],
    )
    def test_a_rate_outside_zero_one_is_refused(self, knob, value):
        with pytest.raises(ValueError) as raised:
            CliffordNet(num_classes=3, channels=8, depth=2, **{knob: value})
        message = str(raised.value)
        assert f"{knob} must be in [0, 1), got {value}" in message
        assert f"classifier_{knob}" not in message and f"drop_path_{knob}" not in message

    def test_a_rate_inside_the_range_is_accepted(self):
        model = CliffordNet(
            num_classes=3, channels=8, depth=2, dropout_rate=0.5, stochastic_depth_rate=0.9
        )
        assert (model.dropout_rate, model.stochastic_depth_rate) == (0.5, 0.9)


class TestCliffordNetDropPathRamp:
    """The drop-path rates ramp from 0 to ``stochastic_depth_rate`` over the ``depth`` blocks."""

    @pytest.mark.parametrize("depth", [2, 5])
    def test_block_i_has_the_rate_linspace_gives_it(self, depth):
        rate = 0.3
        model = CliffordNet(num_classes=3, channels=8, depth=depth, stochastic_depth_rate=rate)
        seen = [entry["drop_path"].drop_path_rate for entry in model.blocks]
        assert [entry["drop_path"].name for entry in model.blocks] == [
            f"block_{i}_drop_path" for i in range(depth)]
        np.testing.assert_allclose(seen, np.linspace(0.0, rate, depth), rtol=0, atol=1e-6)
        assert seen[0] == 0.0 and seen[-1] == pytest.approx(rate)


def _kernels(model):
    """The kernels of a built model keyed by their path below the instance name."""
    return {w.path.split("/", 1)[1]: keras.ops.convert_to_numpy(w)
            for w in model.weights if w.path.endswith("/kernel")}


class TestCliffordNetInitializers:
    """Each initializer is a spec, so a kernel is drawn from the seed and not replayed."""

    @staticmethod
    def _built(seed):
        keras.utils.set_random_seed(seed)
        model = CliffordNet.from_variant("nano", num_classes=5, depth=3)
        model.build((None, 16, 16, 3))
        return model

    def test_every_random_kernel_follows_the_seed(self):
        """A kernel that a module-level initializer draws at import is the same at any seed."""
        first, second = _kernels(self._built(1)), _kernels(self._built(2))
        assert first.keys() == second.keys() and len(first) > 10
        fixed = sorted(path for path in first if np.array_equal(first[path], second[path]))
        assert not fixed, f"{len(fixed)} of {len(first)} kernels ignore the seed: {fixed[:3]}"

    def test_the_default_initializer_is_a_spec_not_an_instance(self):
        from dl_techniques.models.vision.cliffordnet.model import DEFAULT_KERNEL_INITIALIZER

        assert isinstance(DEFAULT_KERNEL_INITIALIZER, dict)
        for name, preset in CliffordNet.MODEL_VARIANTS.items():
            assert isinstance(preset["kernel_initializer"], dict), name

    @pytest.mark.parametrize(
        "argument",
        ["kernel_initializer", "bias_initializer"],
    )
    def test_an_instance_is_stored_as_json_safe_spec(self, argument):
        model = CliffordNet(
            num_classes=3, channels=8, depth=2,
            **{argument: keras.initializers.RandomNormal(stddev=0.1)},
        )
        stored = getattr(model, argument)
        assert isinstance(stored, (str, dict)), type(stored)
        assert json.loads(json.dumps(model.get_config()))[argument] == stored
        clone = CliffordNet.from_config(model.get_config())
        assert clone.get_config() == model.get_config()


class TestCliffordNetBiasFreeClassifier:
    """A model-wide bias initializer or regularizer next to a bias-free classifier.

    ``ClassificationHead`` refuses a non-default bias argument when none of its
    ``Dense`` layers has a bias. The model's own ``bias_initializer`` also governs
    the trunk, so the model leaves the two bias arguments out of the head config
    when the overrides leave the head without any bias. The head is untouched.
    """

    BIAS = dict(
        bias_initializer=keras.initializers.RandomNormal(stddev=0.01, seed=3),
        bias_regularizer=keras.regularizers.L2(1e-3),
    )

    def _model(self, **overrides):
        model = CliffordNet(
            num_classes=3, channels=8, depth=1, patch_size=2, **self.BIAS,
            head_config_overrides=overrides,
        )
        model.build((None, 16, 16, 3))
        return model

    def test_a_bias_free_classifier_builds_and_has_no_bias_weight(self):
        model = self._model(classifier_use_bias=False)
        paths = [w.path for w in model.weights if "classification_head" in w.path]
        assert any(p.endswith("classifier/kernel") for p in paths), paths
        assert not any(p.endswith("classifier/bias") for p in paths), paths

    def test_the_trunk_keeps_the_model_wide_bias_arguments(self):
        model = self._model(classifier_use_bias=False)
        assert isinstance(model.stem.conv1.bias_initializer, keras.initializers.RandomNormal)
        assert isinstance(model.stem.conv1.bias_regularizer, keras.regularizers.L2)
        config = json.loads(json.dumps(model.get_config()))
        assert config["bias_initializer"]["class_name"] == "RandomNormal"
        assert config["bias_regularizer"]["class_name"] == "L2"

    def test_a_hidden_block_with_a_bias_still_reads_them(self):
        model = self._model(classifier_use_bias=False, block_dims=(8,))
        assert isinstance(model.classification_head.bias_initializer,
                          keras.initializers.RandomNormal)

    def test_a_bias_free_hidden_block_leaves_the_head_without_any_bias(self):
        model = self._model(classifier_use_bias=False, block_dims=(8,), block_use_bias=False)
        assert model.classification_head.bias_initializer is None

    def test_a_bias_argument_the_overrides_carry_is_still_refused(self):
        with pytest.raises(ValueError, match="no layer reads"):
            self._model(classifier_use_bias=False, bias_initializer="ones")

    def test_the_default_head_still_receives_the_bias_arguments(self):
        from dl_techniques.models.vision.cliffordnet.model import clifford_head_config

        args = dict(num_classes=3, dropout_rate=0.0, kernel_initializer="he_normal",
                    kernel_regularizer=None, **{k: v for k, v in self.BIAS.items()})
        assert {"bias_initializer", "bias_regularizer"} <= set(clifford_head_config(**args))
        bias_free = clifford_head_config(**args, head_config_overrides={"classifier_use_bias": False})
        assert not {"bias_initializer", "bias_regularizer"} & set(bias_free)
