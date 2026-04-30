"""Tests for CapsNetV2 model (Stage 1 + Stage 2 + Stage 3a).

Smoke-level: builds, forward passes, one-step fit, reconstruction helper,
serialization round-trip.
"""

import os
import tempfile

import numpy as np
import keras
import pytest

from dl_techniques.models.capsnet.model_v2 import (
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
            CapsNetV2(num_classes=10, input_shape=(28, 28))  # type: ignore[arg-type]

    def test_invalid_stem(self):
        with pytest.raises(ValueError, match="stem must be"):
            CapsNetV2(num_classes=10, input_shape=(28, 28, 1), stem="bogus")  # type: ignore[arg-type]

    def test_invalid_loss_type(self):
        with pytest.raises(ValueError, match="loss_type"):
            CapsNetV2(num_classes=10, input_shape=(28, 28, 1), loss_type="bogus")  # type: ignore[arg-type]

    # ---- build / forward ----

    def test_build_legacy_stem(self, synth_mnist):
        x, _ = synth_mnist
        model = CapsNetV2(num_classes=10, input_shape=(28, 28, 1), stem="legacy")
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
            stem="resnet18",
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
            stem="legacy",
            digit_capsule_dim=16,
        )
        digit = model.get_capsules(x)
        assert digit.shape == (4, 10, 16)

    # ---- factory + compile/fit (Stage 1 recipe) ----

    def test_create_capsnet_v2_returns_compiled_adamw(self):
        model = create_capsnet_v2(
            num_classes=10,
            input_shape=(28, 28, 1),
            stem="legacy",
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
            stem="legacy",
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
            stem="legacy",
            loss_type="categorical_crossentropy",
            label_smoothing=0.1,
            decay_steps=100,
        )
        history = model.fit(x, y, epochs=1, verbose=0)
        loss = history.history["loss"][0]
        assert np.isfinite(loss)

    # ---- Stage 2: pretrained-backbone factory ----

    def test_create_capsnet_v2_pretrained_falls_back_to_random_init(self, synth_cifar):
        """Placeholder URL → ResNet's existing fallback returns a random-init
        backbone with a logged warning. Wrapper must not raise."""
        x, _ = synth_cifar
        model = create_capsnet_v2_pretrained(
            backbone="resnet18",
            num_classes=10,
            input_shape=(32, 32, 3),
            pretrained=True,  # exercises the download fallback path
            decay_steps=100,
        )
        out = model(x)
        assert out.shape == (4, 10)

    def test_create_capsnet_v2_pretrained_invalid_backbone(self):
        with pytest.raises(ValueError, match="backbone must be"):
            create_capsnet_v2_pretrained(
                backbone="resnet999",  # type: ignore[arg-type]
                num_classes=10,
                input_shape=(32, 32, 3),
            )

    # ---- reconstruction helper ----

    def test_reconstruct_helper(self, synth_mnist):
        x, _ = synth_mnist
        model = CapsNetV2(
            num_classes=10,
            input_shape=(28, 28, 1),
            stem="legacy",
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
            stem="legacy",
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
            stem="legacy",
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
            stem="legacy",
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
