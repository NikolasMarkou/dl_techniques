"""Tests for CapsNetV2 model (Stage 1 + Stage 2 + Stage 3a).

Smoke-level: builds, forward passes, one-step fit, reconstruction helper,
serialization round-trip.
"""

import json
import os
import tempfile

import numpy as np
import keras
import pytest

from dl_techniques.layers.conv_blocks.stem import ConvStem
from dl_techniques.models.vision.resnet.model import ResNet
from dl_techniques.models.vision.capsnet.model_v2 import (
    CapsNetV2,
    create_capsnet_v2,
    create_capsnet_v2_pretrained,
)


# ---------------------------------------------------------------------


def _onehot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


# ---------------------------------------------------------------------


class TestCapsNetV2:
    """End-to-end smoke tests for CapsNetV2."""

    @pytest.fixture
    def synth_mnist(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(4, 28, 28, 1)).astype("float32")
        y = _onehot(rng.integers(0, 10, size=4), num_classes=10)
        return x, y

    @pytest.fixture
    def synth_cifar(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(4, 32, 32, 3)).astype("float32")
        y = _onehot(rng.integers(0, 10, size=4), num_classes=10)
        return x, y

    # ---- init / validation ----

    def test_invalid_num_classes(self):
        with pytest.raises(ValueError, match="num_classes"):
            CapsNetV2(num_classes=0, input_shape=(28, 28, 1))

    def test_invalid_input_shape(self):
        with pytest.raises(ValueError, match="input_shape"):
            CapsNetV2(num_classes=10, input_shape=(28, 28))

    def test_invalid_stem(self):
        with pytest.raises(ValueError, match="stem_type must be"):
            CapsNetV2(num_classes=10, input_shape=(28, 28, 1), stem_type="bogus")

    def test_invalid_loss_type(self):
        with pytest.raises(ValueError, match="loss_type"):
            CapsNetV2(num_classes=10, input_shape=(28, 28, 1), loss_type="bogus")

    # ---- build / forward ----

    def test_build_legacy_stem(self, synth_mnist):
        x, _ = synth_mnist
        model = CapsNetV2(num_classes=10, input_shape=(28, 28, 1), stem_type="legacy")
        out = model(x)
        assert out.shape == (4, 10)
        assert not np.any(np.isnan(out.numpy()))
        # Lengths from sigmoid magnitude × unit direction → ‖v‖ ∈ (0, 1).
        assert np.all(out.numpy() > 0.0)
        assert np.all(out.numpy() < 1.0)

    def test_build_resnet18_stem(self, synth_cifar):
        x, _ = synth_cifar
        model = CapsNetV2(
            num_classes=10,
            input_shape=(32, 32, 3),
            stem_type="resnet18",
            stem_pretrained=False,
        )
        out = model(x)
        assert out.shape == (4, 10)
        assert not np.any(np.isnan(out.numpy()))

    def test_get_capsules_returns_pose(self, synth_mnist):
        x, _ = synth_mnist
        model = CapsNetV2(
            num_classes=10,
            input_shape=(28, 28, 1),
            stem_type="legacy",
            digit_capsule_dim=16,
        )
        digit = model.get_capsules(x)
        assert digit.shape == (4, 10, 16)

    # ---- factory + compile/fit (Stage 1 recipe) ----

    def test_create_capsnet_v2_returns_compiled_adamw(self):
        model = create_capsnet_v2(
            num_classes=10,
            input_shape=(28, 28, 1),
            stem_type="legacy",
            decay_steps=100,
        )
        assert model.optimizer.__class__.__name__ == "AdamW"
        assert bool(getattr(model.optimizer, "use_ema", False))
        # weight_decay attribute is exposed on the optimizer.
        assert getattr(model.optimizer, "weight_decay", None) is not None

    def test_factory_fit_one_step_margin_loss(self, synth_mnist):
        x, y = synth_mnist
        model = create_capsnet_v2(
            num_classes=10,
            input_shape=(28, 28, 1),
            stem_type="legacy",
            decay_steps=100,
        )
        history = model.fit(x, y, epochs=1, verbose=0)
        # Loss must be a finite scalar.
        loss = history.history["loss"][0]
        assert np.isfinite(loss)

    def test_factory_fit_one_step_cce_label_smoothing(self, synth_mnist):
        x, y = synth_mnist
        model = create_capsnet_v2(
            num_classes=10,
            input_shape=(28, 28, 1),
            stem_type="legacy",
            loss_type="categorical_crossentropy",
            label_smoothing=0.1,
            decay_steps=100,
        )
        history = model.fit(x, y, epochs=1, verbose=0)
        loss = history.history["loss"][0]
        assert np.isfinite(loss)

    # ---- Stage 2: pretrained-backbone factory ----

    def test_create_capsnet_v2_pretrained_true_raises(self):
        """`pretrained=True` must RAISE, never return a random-init backbone.

        No public ResNet weights ship with dl_techniques. Before this contract,
        `ResNet.PRETRAINED_WEIGHTS` held placeholder URLs pointing at a
        non-existent host, `from_variant` caught the download failure, logged a
        warning and returned an untrained backbone — so a caller asking for a
        pretrained stem silently got random weights. Do not reinstate that.
        """
        with pytest.raises(NotImplementedError, match="ResNet variant 'resnet18' weights"):
            create_capsnet_v2_pretrained(
                backbone="resnet18",
                num_classes=10,
                input_shape=(32, 32, 3),
                pretrained=True,
                decay_steps=100,
            )

    def test_create_capsnet_v2_pretrained_false_builds(self, synth_cifar):
        """`pretrained=False` still builds a working random-init backbone."""
        x, _ = synth_cifar
        model = create_capsnet_v2_pretrained(
            backbone="resnet18",
            num_classes=10,
            input_shape=(32, 32, 3),
            pretrained=False,
            decay_steps=100,
        )
        out = model(x)
        assert out.shape == (4, 10)

    def test_create_capsnet_v2_pretrained_invalid_backbone(self):
        with pytest.raises(ValueError, match="backbone must be"):
            create_capsnet_v2_pretrained(
                backbone="resnet999",
                num_classes=10,
                input_shape=(32, 32, 3),
            )

    # ---- reconstruction helper ----

    def test_reconstruct_helper(self, synth_mnist):
        x, _ = synth_mnist
        model = CapsNetV2(
            num_classes=10,
            input_shape=(28, 28, 1),
            stem_type="legacy",
            reconstruction=True,
        )
        recon = model.reconstruct(x)
        assert recon.shape == (4, 28, 28, 1)
        # Sigmoid output: in (0, 1).
        assert np.all(recon.numpy() >= 0.0)
        assert np.all(recon.numpy() <= 1.0)

    def test_reconstruct_with_explicit_mask(self, synth_mnist):
        x, _ = synth_mnist
        model = CapsNetV2(
            num_classes=10,
            input_shape=(28, 28, 1),
            stem_type="legacy",
            reconstruction=True,
        )
        mask = np.zeros((4, 10), dtype=np.float32)
        mask[:, 3] = 1.0  # always reconstruct as class 3
        recon = model.reconstruct(x, mask=keras.ops.array(mask))
        assert recon.shape == (4, 28, 28, 1)

    def test_reconstruct_raises_when_disabled(self, synth_mnist):
        x, _ = synth_mnist
        model = CapsNetV2(
            num_classes=10,
            input_shape=(28, 28, 1),
            stem_type="legacy",
            reconstruction=False,
        )
        with pytest.raises(ValueError, match="reconstruction=True"):
            model.reconstruct(x)

    # ---- serialization ----

    def test_save_load_round_trip(self, synth_mnist):
        x, _ = synth_mnist
        model = CapsNetV2(
            num_classes=10,
            input_shape=(28, 28, 1),
            stem_type="legacy",
            attention_top_k=8,
            use_load_balancing=True,
        )
        ref = model(x).numpy()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "v2_model.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)
            new = reloaded(x).numpy()
            assert new.shape == ref.shape
            assert np.allclose(new, ref, atol=1e-5)


# ---------------------------------------------------------------------


def _weights(model: keras.Model) -> dict:
    return {weight.path: keras.ops.convert_to_numpy(weight) for weight in model.weights}


class TestCapsNetV2Stem:
    """``stem_type`` selects the trunk; ``stem`` holds the ResNet backbone, ``conv_blocks`` the legacy trunk."""

    def test_the_legacy_trunk_is_one_conv_stem_per_filter_entry(self):
        model = CapsNetV2(
            num_classes=10, input_shape=(28, 28, 1), stem_type="legacy",
            legacy_conv_filters=[8, 16, 24],
        )
        assert model.stem is None
        assert [type(block) for block in model.conv_blocks] == [ConvStem] * 3
        assert [block.name for block in model.conv_blocks] == [
            "conv_block_0", "conv_block_1", "conv_block_2"
        ]
        assert [block.conv1.kernel_size for block in model.conv_blocks] == [(9, 9), (5, 5), (5, 5)]
        for block in model.conv_blocks:
            assert block.conv1.padding == "valid" and block.conv1.use_bias is True
            assert block.norm1.epsilon == pytest.approx(1e-3)

    def test_a_resnet_stem_is_held_as_stem_and_the_legacy_trunk_is_empty(self):
        model = CapsNetV2(num_classes=10, input_shape=(32, 32, 3), stem_type="resnet18")
        assert isinstance(model.stem, ResNet)
        assert model.stem.name == "stem"
        assert model.stem.classification_head is None
        assert model.conv_blocks == []
        assert model.stem_type == "resnet18"

    def test_a_training_call_moves_the_block_batch_norm_statistics_and_an_inference_call_does_not(self):
        model = CapsNetV2(
            num_classes=10, input_shape=(28, 28, 1), legacy_conv_filters=[8, 8], primary_capsules=4
        )
        x = np.random.default_rng(0).uniform(size=(4, 28, 28, 1)).astype("float32")
        model(x, training=False)
        before = [keras.ops.convert_to_numpy(b.norm1.moving_mean) for b in model.conv_blocks]
        model(x, training=False)
        model(x)
        unchanged = [keras.ops.convert_to_numpy(b.norm1.moving_mean) for b in model.conv_blocks]
        model(x, training=True)
        moved = [keras.ops.convert_to_numpy(b.norm1.moving_mean) for b in model.conv_blocks]
        for was, same, now in zip(before, unchanged, moved):
            np.testing.assert_array_equal(was, same)
            assert float(np.max(np.abs(now - was))) > 1e-4

    def test_the_resnet_stem_gets_the_training_flag_too(self):
        model = CapsNetV2(num_classes=10, input_shape=(32, 32, 3), stem_type="resnet18", primary_capsules=4)
        x = np.random.default_rng(0).uniform(size=(4, 32, 32, 3)).astype("float32")
        model(x, training=False)
        norm = model.stem.stem.norm1
        before = keras.ops.convert_to_numpy(norm.moving_mean)
        model(x, training=False)
        np.testing.assert_array_equal(keras.ops.convert_to_numpy(norm.moving_mean), before)
        model(x, training=True)
        assert float(np.max(np.abs(keras.ops.convert_to_numpy(norm.moving_mean) - before))) > 1e-4

    def test_the_old_selector_spelling_and_attribute_names_are_gone(self):
        with pytest.raises(ValueError, match="Unrecognized keyword arguments.*stem"):
            CapsNetV2(num_classes=10, input_shape=(28, 28, 1), stem="legacy")
        model = CapsNetV2(num_classes=10, input_shape=(28, 28, 1))
        for gone in ("stem_layers", "resnet_stem", "_input_shape"):
            assert not hasattr(model, gone), gone
        assert model.input_shape_config == (28, 28, 1)

    def test_the_config_key_is_stem_type(self):
        config = CapsNetV2(num_classes=10, input_shape=(28, 28, 1)).get_config()
        assert config["stem_type"] == "legacy" and "stem" not in config

    def test_the_weight_paths_of_the_legacy_trunk_name_each_block_and_its_parts(self):
        model = CapsNetV2(
            num_classes=10, input_shape=(28, 28, 1), legacy_conv_filters=[8], name="capsnet_v2"
        )
        model.build((None, 28, 28, 1))
        assert [weight.path for weight in model.weights][:6] == [
            "capsnet_v2/conv_block_0/conv_block_0_conv1/kernel",
            "capsnet_v2/conv_block_0/conv_block_0_conv1/bias",
            "capsnet_v2/conv_block_0/conv_block_0_norm1/gamma",
            "capsnet_v2/conv_block_0/conv_block_0_norm1/beta",
            "capsnet_v2/conv_block_0/conv_block_0_norm1/moving_mean",
            "capsnet_v2/conv_block_0/conv_block_0_norm1/moving_variance",
        ]


class TestCapsNetV2Build:
    """``build`` materializes everything the model owns, the decoder included."""

    @pytest.mark.parametrize("reconstruction", [True, False], ids=["decoder", "no-decoder"])
    @pytest.mark.parametrize("stem_type", ["legacy", "resnet18"])
    def test_explicit_build_equals_a_lazy_build(self, stem_type, reconstruction):
        shape = (28, 28, 1) if stem_type == "legacy" else (32, 32, 3)

        def make():
            return CapsNetV2(
                num_classes=10, input_shape=shape, stem_type=stem_type,
                reconstruction=reconstruction, legacy_conv_filters=[8, 8], primary_capsules=4,
            )

        explicit = make()
        explicit.build((None, *shape))
        lazy = make()
        lazy(np.zeros((1, *shape), "float32"))
        if reconstruction:
            lazy.reconstruct(np.zeros((1, *shape), "float32"))

        def relative(model):
            return [weight.path.split("/", 1)[1] for weight in model.weights]

        assert len(explicit.weights) > 0 and explicit.count_params() > 0
        assert relative(explicit) == relative(lazy)
        assert explicit.count_params() == lazy.count_params()
        if reconstruction:
            assert any("reconstruction_decoder" in weight.path for weight in explicit.weights)

    def test_build_is_idempotent_and_refuses_a_non_4d_shape(self):
        model = CapsNetV2(num_classes=10, input_shape=(28, 28, 1), legacy_conv_filters=[8])
        model.build((None, 28, 28, 1))
        paths = [weight.path for weight in model.weights]
        model.build((None, 28, 28, 1))
        assert [weight.path for weight in model.weights] == paths
        with pytest.raises(ValueError, match="Expected 4D input"):
            CapsNetV2(num_classes=10, input_shape=(28, 28, 1)).build((None, 28, 28))

    @pytest.mark.parametrize("stem_type", ["legacy", "resnet18"])
    @pytest.mark.parametrize("batch", [None, 3])
    def test_compute_output_shape_equals_the_forward_shape_and_builds_nothing(self, stem_type, batch):
        shape = (28, 28, 1) if stem_type == "legacy" else (32, 32, 3)
        model = CapsNetV2(num_classes=10, input_shape=shape, stem_type=stem_type)
        reported = model.compute_output_shape((batch, *shape))
        assert model.built is False
        assert tuple(reported) == (batch, 10)
        out = model(np.zeros((batch or 2, *shape), "float32"))
        assert tuple(out.shape)[1:] == tuple(reported)[1:]

    def test_compute_output_shape_refuses_a_non_4d_shape(self):
        with pytest.raises(ValueError, match="Expected 4D input"):
            CapsNetV2(num_classes=10, input_shape=(28, 28, 1)).compute_output_shape((None, 28, 28))


class TestCapsNetV2Reload:
    """A ``.keras`` round trip carries every weight, the decoder's included."""

    @pytest.fixture
    def x(self):
        return np.random.default_rng(0).normal(size=(2, 28, 28, 1)).astype("float32")

    def _donor(self, x, call_reconstruct: bool) -> CapsNetV2:
        model = CapsNetV2(
            num_classes=10, input_shape=(28, 28, 1), reconstruction=True, legacy_conv_filters=[8, 8],
            primary_capsules=4, decoder_architecture=[16, 32],
        )
        model(x)
        if call_reconstruct:
            model.reconstruct(x)
        for weight in model.weights:  # move every tensor off its initializer
            weight.assign(weight + 0.02)
        return model

    @pytest.mark.parametrize("call_reconstruct", [False, True], ids=["only-called", "reconstructed"])
    def test_a_reloaded_reconstruction_model_keeps_its_decoder(self, x, tmp_path, call_reconstruct):
        donor = self._donor(x, call_reconstruct)
        donor_weights = _weights(donor)
        assert any("reconstruction_decoder" in path for path in donor_weights), (
            "the donor holds no decoder weight, so the reload proves nothing"
        )
        expected = keras.ops.convert_to_numpy(donor.reconstruct(x))

        path = str(tmp_path / "v2.keras")
        donor.save(path)
        loaded = keras.models.load_model(path)

        loaded_weights = _weights(loaded)  # BEFORE the first call
        assert set(loaded_weights) == set(donor_weights), (
            f"{len(loaded_weights)} weights reloaded against {len(donor_weights)}"
        )
        for weight_path, value in donor_weights.items():
            np.testing.assert_array_equal(loaded_weights[weight_path], value, err_msg=weight_path)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(loaded.reconstruct(x)), expected, rtol=0, atol=1e-6
        )


class TestCapsNetV2Config:
    def test_get_config_is_json_safe_with_an_initializer_instance_and_round_trips(self):
        model = CapsNetV2(
            num_classes=10, input_shape=(28, 28, 1), legacy_conv_filters=[8],
            kernel_initializer=keras.initializers.HeNormal(),
            kernel_regularizer=keras.regularizers.L2(1e-4),
        )
        config = model.get_config()
        json.dumps(config)
        rebuilt = CapsNetV2.from_config(json.loads(json.dumps(config)))
        assert json.dumps(rebuilt.get_config(), sort_keys=True) == json.dumps(config, sort_keys=True)

    def test_from_config_leaves_the_given_dict_alone(self):
        config = CapsNetV2(num_classes=10, input_shape=(28, 28, 1)).get_config()
        config = json.loads(json.dumps(config))
        snapshot = json.loads(json.dumps(config))
        CapsNetV2.from_config(config)
        assert config == snapshot

    def test_from_config_never_reads_stem_weights_from_a_path(self):
        """The archive already holds the stem's weights: a saved ``stem_pretrained`` path is not read."""
        config = CapsNetV2(num_classes=10, input_shape=(28, 28, 1)).get_config()
        config["stem_pretrained"] = "/this/path/does/not/exist.keras"
        rebuilt = CapsNetV2.from_config(config)
        assert rebuilt.stem_pretrained is False
        assert config["stem_pretrained"] == "/this/path/does/not/exist.keras"

    def test_a_string_regularizer_is_resolved_to_an_object(self):
        model = CapsNetV2(num_classes=10, input_shape=(28, 28, 1), kernel_regularizer="l2")
        assert isinstance(model.kernel_regularizer, keras.regularizers.Regularizer)
        json.dumps(model.get_config())

    def test_a_list_input_shape_is_accepted_and_stored_as_a_tuple(self):
        model = CapsNetV2(num_classes=10, input_shape=[28, 28, 1])
        assert model.input_shape_config == (28, 28, 1)
