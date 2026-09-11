import os

import numpy as np
import keras
import pytest

from dl_techniques.models.vision.omnipoint import (
    OmniPoint,
    create_omnipoint,
    MODEL_VARIANTS,
)

from ..gradient_flow_oracle import assert_gradients_reach_every_trainable_weight

# A deliberately tiny image so a ViT-Base/14 forward pass (this test's variant)
# runs fast on CPU: 56 % 14 == 0 -> a 4x4 = 16-token patch grid.
_TEST_IMAGE_SHAPE = (56, 56, 3)


class TestOmniPoint:
    """Test suite for the Step-4 OmniPoint forward pass (no conditioning)."""

    @pytest.fixture
    def model(self) -> OmniPoint:
        return OmniPoint.from_variant(
            "omnipoint_base", image_shape=_TEST_IMAGE_SHAPE
        )

    @pytest.fixture
    def batch(self) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.uniform(
            0.0, 1.0, size=(2,) + _TEST_IMAGE_SHAPE
        ).astype("float32")

    # ------------------------------------------------------------------
    # Variant registry
    # ------------------------------------------------------------------

    def test_model_variants_has_base_and_large(self):
        assert "omnipoint_base" in MODEL_VARIANTS
        assert "omnipoint_large" in MODEL_VARIANTS
        assert MODEL_VARIANTS["omnipoint_base"]["vit_scale"] == "base"
        assert MODEL_VARIANTS["omnipoint_large"]["vit_scale"] == "large"
        # Class attribute and module-level alias must be the same object.
        assert MODEL_VARIANTS is OmniPoint.MODEL_VARIANTS

    def test_from_variant_rejects_unknown_variant(self):
        with pytest.raises(ValueError):
            OmniPoint.from_variant("omnipoint_huge")

    # ------------------------------------------------------------------
    # Forward pass: shapes and Success Criterion 2's invariants
    # ------------------------------------------------------------------

    def test_forward_pass_output_shapes(self, model, batch):
        ray, distance, point, mask_logit, scale = model(batch)
        grid = _TEST_IMAGE_SHAPE[0] // 14  # patch_size=14 default
        assert tuple(ray.shape) == (2, grid, grid, 3)
        assert tuple(distance.shape) == (2, grid, grid, 1)
        assert tuple(point.shape) == (2, grid, grid, 3)
        assert tuple(mask_logit.shape) == (2, grid, grid, 1)
        assert tuple(scale.shape) == (2,)

    def test_ray_is_unit_norm_and_distance_is_positive(self, model, batch):
        ray, distance, point, _, _ = model(batch)
        norms = keras.ops.convert_to_numpy(
            keras.ops.sqrt(keras.ops.sum(keras.ops.square(ray), axis=-1))
        )
        np.testing.assert_allclose(
            norms, np.ones_like(norms), atol=1e-5, rtol=0
        )
        distance_np = keras.ops.convert_to_numpy(distance)
        assert np.all(distance_np > 0.0)
        # point == distance * ray, end to end through the full model.
        point_np = keras.ops.convert_to_numpy(point)
        expected = distance_np * keras.ops.convert_to_numpy(ray)
        np.testing.assert_allclose(point_np, expected, atol=1e-6, rtol=0)

    def test_scale_is_strictly_positive(self, model, batch):
        _, _, _, _, scale = model(batch)
        assert np.all(keras.ops.convert_to_numpy(scale) > 0.0)

    def test_all_outputs_are_finite(self, model, batch):
        outputs = model(batch)
        for tensor in outputs:
            assert np.all(np.isfinite(keras.ops.convert_to_numpy(tensor)))

    # ------------------------------------------------------------------
    # Config round-trip (Success Criterion 3)
    # ------------------------------------------------------------------

    def test_config_round_trip(self, model):
        config = model.get_config()
        restored = OmniPoint.from_config(config)
        restored_config = restored.get_config()

        # The nested encoder is serialized as a fresh random-init sub-Model
        # config each time (get_config -> serialize_keras_object embeds a
        # fresh dict with its own name); compare every OTHER field exactly,
        # and compare the encoder configs by their own architecture-defining
        # fields rather than a full dict `==` (names differ if unset).
        for key in (
                "image_shape", "vit_scale", "patch_size", "decoder_dims",
                "metric_hidden_dim", "epsilon",
        ):
            assert config[key] == restored_config[key], key

        assert restored.image_shape == model.image_shape
        assert restored.vit_scale == model.vit_scale
        assert restored.patch_size == model.patch_size
        assert restored.embed_dim == model.embed_dim
        assert restored.grid_h == model.grid_h
        assert restored.grid_w == model.grid_w

    # ------------------------------------------------------------------
    # `.keras` save/load round-trip (Success Criterion 9)
    # ------------------------------------------------------------------

    def test_save_load_keras_round_trip(self, model, batch, tmp_path):
        """Full `.keras` archive round-trip: registration key survives, and the
        reloaded model reproduces the SAME model's forward pass bit-identically
        on the SAME input (not just a `get_config`/`from_config` structural
        comparison, which `test_config_round_trip` above already covers)."""
        expected_key = "dl_techniques.models.omnipoint.model>OmniPoint"
        assert keras.saving.get_registered_name(OmniPoint) == expected_key

        outputs_before = model(batch)

        save_path = os.path.join(tmp_path, "omnipoint.keras")
        model.save(save_path)
        reloaded = keras.models.load_model(save_path)

        assert keras.saving.get_registered_name(type(reloaded)) == expected_key

        outputs_after = reloaded(batch)
        for before, after in zip(outputs_before, outputs_after):
            np.testing.assert_array_equal(
                keras.ops.convert_to_numpy(before),
                keras.ops.convert_to_numpy(after),
            )

    # DECISION plan-2026-09-11T050223-1b47bcf6/D-023 (extended, completion-fix
    # step 4.1): the review's CRITICAL #3 found this exact gap -- the test above
    # only exercised the default `enable_conditioning=False` model, so D-023's
    # original heads-only build() fix silently left the 2 conditioning layers
    # (`conditioning_input_encoder`, `conditioning_state_embedding`) 0-weight on
    # `.keras` reload (8 of 212 weights re-randomized). This parametrized test
    # covers BOTH values, including a weight-by-weight comparison (not just
    # output comparison, which the review found insufficient on its own to
    # prove no weight was silently swapped for a compatible-shaped random one).
    # See decisions.md.
    @pytest.mark.parametrize("enable_conditioning", [False, True])
    def test_save_load_keras_round_trip_with_and_without_conditioning(
            self, enable_conditioning, batch, tmp_path
    ):
        model = OmniPoint.from_variant(
            "omnipoint_base",
            image_shape=_TEST_IMAGE_SHAPE,
            enable_conditioning=enable_conditioning,
        )
        outputs_before = model(batch)

        save_path = os.path.join(
            tmp_path, f"omnipoint_cond_{enable_conditioning}.keras"
        )
        model.save(save_path)
        reloaded = keras.models.load_model(save_path)

        outputs_after = reloaded(batch)
        for before, after in zip(outputs_before, outputs_after):
            np.testing.assert_array_equal(
                keras.ops.convert_to_numpy(before),
                keras.ops.convert_to_numpy(after),
            )

        # Weight-by-weight: every weight must survive the round trip exactly,
        # not merely produce a matching forward pass (a compatible-shaped but
        # differently-valued weight could coincidentally agree on one input).
        # MEASURED: `keras.Variable.path` inconsistently includes the outer
        # model's own name as a leading path segment depending on whether the
        # model was freshly constructed-and-called or reloaded via
        # `load_model` -- strip that one optional segment so the comparison
        # is keyed on the sublayer-relative path both sides actually share.
        def _relative_path(path: str, model_name: str) -> str:
            parts = path.split("/")
            if parts and parts[0] == model_name:
                parts = parts[1:]
            return "/".join(parts)

        weights_before = {
            _relative_path(w.path, model.name): keras.ops.convert_to_numpy(w)
            for w in model.weights
        }
        weights_after = {
            _relative_path(w.path, reloaded.name): keras.ops.convert_to_numpy(w)
            for w in reloaded.weights
        }
        assert set(weights_before.keys()) == set(weights_after.keys())
        mismatched = [
            path
            for path, value in weights_before.items()
            if not np.array_equal(value, weights_after[path])
        ]
        assert not mismatched, f"weights lost on reload: {mismatched}"

    # ------------------------------------------------------------------
    # pretrained=True raise (Success Criterion 4)
    # ------------------------------------------------------------------

    def test_pretrained_true_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            OmniPoint.from_variant("omnipoint_base", pretrained=True)

    # ------------------------------------------------------------------
    # Gradient flow (repo convention, shared oracle)
    # ------------------------------------------------------------------

    def test_gradients_reach_every_trainable_weight(self, model, batch):
        _ = model(batch)  # force-build: a subclassed keras.Model is unbuilt until its first call
        assert_gradients_reach_every_trainable_weight(model, batch)

    # ------------------------------------------------------------------
    # create_omnipoint factory
    # ------------------------------------------------------------------

    def test_create_omnipoint_builds_and_runs(self):
        model = create_omnipoint(
            "omnipoint_base", image_shape=_TEST_IMAGE_SHAPE
        )
        assert model.built
        rng = np.random.default_rng(1)
        batch = rng.uniform(
            0.0, 1.0, size=(1,) + _TEST_IMAGE_SHAPE
        ).astype("float32")
        outputs = model(batch)
        assert len(outputs) == 5
