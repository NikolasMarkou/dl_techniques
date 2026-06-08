"""
Test suite for the mixtures factory (create_mixture_layer + helpers).

Closes the coverage gap flagged in review-iter-1 WARNING #2: the factory was
the user-requested deliverable but had zero test coverage, and the D-001 latent
bug surfaced precisely via create_mixture_layer('gmm', n_components=4).
"""

import pytest
import keras

from dl_techniques.layers.mixtures import (
    RBFLayer,
    KMeansLayer,
    GMMLayer,
    MIXTURE_REGISTRY,
    create_mixture_layer,
    create_mixture_from_config,
    get_mixture_info,
    validate_mixture_config,
)


class TestCreateMixtureLayer:

    @pytest.mark.parametrize("mtype,kwargs,cls", [
        ("rbf", {"units": 8}, RBFLayer),
        ("kmeans", {"n_clusters": 4}, KMeansLayer),
        ("gmm", {"n_components": 4}, GMMLayer),
    ])
    def test_create_each_type_with_defaults(self, mtype, kwargs, cls) -> None:
        """All three build with their DEFAULT 'orthonormal' initializer (D-001 regression)."""
        layer = create_mixture_layer(mtype, **kwargs)
        assert isinstance(layer, cls)

    def test_name_passthrough(self) -> None:
        layer = create_mixture_layer("gmm", n_components=3, name="my_gmm")
        assert layer.name == "my_gmm"

    def test_unknown_kwargs_filtered(self) -> None:
        """Unknown kwargs are filtered out, not passed to the constructor."""
        layer = create_mixture_layer("gmm", n_components=4, not_a_real_param=123)
        assert isinstance(layer, GMMLayer)

    def test_kwargs_override_defaults(self) -> None:
        layer = create_mixture_layer("gmm", n_components=4, temperature=0.5)
        assert layer.temperature == 0.5

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown mixture type|Failed to create"):
            create_mixture_layer("not_a_type", units=4)

    def test_missing_required_param_raises(self) -> None:
        with pytest.raises(ValueError):
            create_mixture_layer("gmm")  # n_components missing


class TestValidateMixtureConfig:

    def test_valid_passes(self) -> None:
        validate_mixture_config("gmm", n_components=4)  # no raise

    def test_unknown_type(self) -> None:
        with pytest.raises(ValueError, match="Unknown mixture type"):
            validate_mixture_config("bogus", n_components=4)

    def test_missing_required(self) -> None:
        with pytest.raises(ValueError, match="Required parameters missing"):
            validate_mixture_config("kmeans")

    @pytest.mark.parametrize("kwargs,msg", [
        ({"n_components": 0}, "positive"),
        ({"n_components": 4, "temperature": 0.0}, "temperature must be positive"),
        ({"n_components": 4, "variance_floor": 0.0}, "variance_floor must be positive"),
        ({"n_components": 4, "isometric_regularizer_strength": -1.0}, "non-negative"),
        ({"n_components": 4, "output_mode": "bad"}, "output_mode must be"),
    ])
    def test_invalid_params_raise(self, kwargs, msg) -> None:
        with pytest.raises(ValueError, match=msg):
            validate_mixture_config("gmm", **kwargs)


class TestCreateFromConfigAndInfo:

    def test_create_from_config(self) -> None:
        layer = create_mixture_from_config({"type": "gmm", "n_components": 5})
        assert isinstance(layer, GMMLayer)
        assert layer.n_components == 5

    def test_create_from_config_missing_type(self) -> None:
        with pytest.raises(ValueError, match="must include 'type'"):
            create_mixture_from_config({"n_components": 4})

    def test_create_from_config_not_dict(self) -> None:
        with pytest.raises(ValueError, match="must be a dictionary"):
            create_mixture_from_config(["gmm"])

    def test_get_mixture_info_keys(self) -> None:
        info = get_mixture_info()
        assert set(info.keys()) == {"rbf", "kmeans", "gmm"}
        for key in ("rbf", "kmeans", "gmm"):
            assert "class" in info[key]
            assert "required_params" in info[key]

    def test_registry_classes(self) -> None:
        assert MIXTURE_REGISTRY["rbf"]["class"] is RBFLayer
        assert MIXTURE_REGISTRY["kmeans"]["class"] is KMeansLayer
        assert MIXTURE_REGISTRY["gmm"]["class"] is GMMLayer
