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


# ------------------------------------------------------- diagnostics (step 5)

class TestFactoryDiagnostics:
    """Guards for the factory's diagnostic surface: registry isolation, the
    dropped-kwarg warning, the `_KERAS_BASE_PARAMS` lockstep pin, and the
    import-time registry/Literal consistency check.
    """

    def test_get_mixture_info_cannot_corrupt_the_registry(self) -> None:
        """A5: `get_mixture_info()` hands out a DEEP copy.

        Pre-fix the outer dict was `.copy()`-ed, so `optional_params` was the
        SAME object as the module global and a caller reading the "info" dict
        could silently re-default every subsequent `create_mixture_layer` call
        in the process.
        """
        sentinel = 999.0
        original = MIXTURE_REGISTRY["gmm"]["optional_params"]["temperature"]

        info = get_mixture_info()
        info["gmm"]["optional_params"]["temperature"] = sentinel

        assert MIXTURE_REGISTRY["gmm"]["optional_params"]["temperature"] == original, (
            "get_mixture_info() aliased the module-global MIXTURE_REGISTRY: mutating "
            "the returned nested dict changed the registry itself."
        )
        assert create_mixture_layer("gmm", n_components=4).temperature == original

    def test_get_mixture_info_preserves_class_identity(self) -> None:
        """A5's counter-risk: the deep copy must NOT clone the layer classes.

        `create_mixture_from_config` and Keras serializable registration both
        key on class IDENTITY, so a copied class would be a worse bug than the
        aliasing it fixes.
        """
        info = get_mixture_info()
        assert info["rbf"]["class"] is RBFLayer
        assert info["kmeans"]["class"] is KMeansLayer
        assert info["gmm"]["class"] is GMMLayer

    def test_unknown_kwarg_warns_but_still_returns_a_layer(self, caplog) -> None:
        """B1 + H-2: the drop is ANNOUNCED, never rejected.

        `plans/SYSTEM.md:185,253` record a deliberate cross-plan decision that
        this factory stays NON-STRICT (unlike `ffn`/`norms`/`embedding`/
        `attention`). Both halves are load-bearing: a layer comes back, AND one
        warning names the misspelled key.
        """
        import logging

        with caplog.at_level(logging.WARNING, logger="dl"):
            layer = create_mixture_layer("gmm", n_components=8, temperture=0.5)

        assert isinstance(layer, GMMLayer), (
            "H-2: the mixtures factory must not raise on an unknown kwarg."
        )
        assert layer.temperature == 1.0  # the registry default, unchanged

        warnings = [
            record for record in caplog.records
            if record.levelno == logging.WARNING and "temperture" in record.message
        ]
        assert len(warnings) == 1, (
            f"expected exactly one warning naming the dropped key 'temperture', "
            f"got {len(warnings)}: {[r.message for r in caplog.records]}"
        )

    def test_keras_base_param_set_matches_the_norms_factory(self) -> None:
        """E7 (regression pin, GREEN on arrival — not a RED proof).

        `mixtures/factory.py:_KERAS_BASE_PARAMS` is a deliberate copy of the
        `norms` one (its own comment says so). Zero drift measured today; this
        pins it, mirroring
        `tests/test_layers/test_activations/test_activation_factory.py`'s
        `test_keras_base_param_set_matches_the_norms_factory`.
        """
        from dl_techniques.layers.mixtures.factory import _KERAS_BASE_PARAMS
        from dl_techniques.layers.norms.factory import (
            _KERAS_BASE_PARAMS as norms_base_params,
        )
        assert _KERAS_BASE_PARAMS == norms_base_params

    def test_registry_and_literal_agree(self) -> None:
        """B5: the shipped `MixtureType` Literal and the registry keys agree."""
        from typing import get_args
        from dl_techniques.layers.mixtures.factory import MixtureType

        assert set(get_args(MixtureType)) == set(MIXTURE_REGISTRY.keys())

    @pytest.mark.parametrize("literal_types,registry_keys,fragment", [
        (("rbf", "kmeans", "gmm", "ghost"), ("rbf", "kmeans", "gmm"), "ghost"),
        (("rbf", "kmeans"), ("rbf", "kmeans", "gmm"), "gmm"),
    ])
    def test_the_consistency_check_actually_fires(
        self, literal_types, registry_keys, fragment
    ) -> None:
        """B5's own RED: the import-time check must reject a real mismatch.

        Exercised through the extracted pure function so no module global is
        perturbed for the rest of the session. `RuntimeError`, not `assert` --
        a bare `assert` is stripped by `python -O`, which would make the check
        silently absent in exactly the deployment that runs optimized.
        """
        from dl_techniques.layers.mixtures.factory import (
            _check_registry_literal_consistency,
        )

        _check_registry_literal_consistency(("rbf",), ("rbf",))  # no raise

        with pytest.raises(RuntimeError, match=fragment):
            _check_registry_literal_consistency(literal_types, registry_keys)


class TestValidOutputModesIsDeclaredOnce:
    """B4 regression pins for the per-class ``VALID_OUTPUT_MODES`` frozensets.

    These are **regression pins, not RED proofs**: they describe a refactor
    (four hand-maintained copies of two sets collapsed to one declaration per
    owning class), not a behaviour change, so most of them were green before
    the refactor too and are labelled as such in ``verification.md``. The one
    thing that measurably changed is covered by
    ``test_the_factory_learns_the_legal_set_from_the_class``.
    """

    @pytest.mark.parametrize("cls,expected", [
        (GMMLayer, {'assignments', 'mixture'}),
        (KMeansLayer, {'assignments', 'mixture'}),
        (RBFLayer, {'basis', 'normalized'}),
    ])
    def test_each_class_declares_its_own_legal_set(self, cls, expected) -> None:
        assert isinstance(cls.VALID_OUTPUT_MODES, frozenset)
        assert set(cls.VALID_OUTPUT_MODES) == expected

    def test_the_two_vocabularies_stay_disjoint(self) -> None:
        """`plan-2026-07-20T160907-7de371a1/D-003`'s actual invariant.

        RBF reuses the kwarg NAME only. If these ever become one set, the
        factory starts accepting 'mixture' for an RBF layer (and the RBF
        constructor then rejects it), which is the regression D-003 exists to
        prevent. Pinning inequality is what keeps the D-012 refactor honest.
        """
        assert RBFLayer.VALID_OUTPUT_MODES != GMMLayer.VALID_OUTPUT_MODES
        assert RBFLayer.VALID_OUTPUT_MODES.isdisjoint(GMMLayer.VALID_OUTPUT_MODES)
        assert GMMLayer.VALID_OUTPUT_MODES == KMeansLayer.VALID_OUTPUT_MODES

    @pytest.mark.parametrize("mtype", sorted(MIXTURE_REGISTRY.keys()))
    def test_the_factory_learns_the_legal_set_from_the_class(self, mtype) -> None:
        """The drift guard: the factory's accepted set IS the class's set.

        Before B4 the factory could only learn a type's legal modes from a
        hardcoded ``if mixture_type == 'rbf'`` branch, so a fourth registered
        type was silently validated against GMM's vocabulary. This asserts by
        exercise, not by reading the source: every legal value is accepted and
        every value legal for the OTHER vocabulary is rejected.
        """
        cls = MIXTURE_REGISTRY[mtype]['class']
        base = {"rbf": {"units": 8}, "kmeans": {"n_clusters": 4},
                "gmm": {"n_components": 4}}[mtype]

        for mode in sorted(cls.VALID_OUTPUT_MODES):
            validate_mixture_config(mtype, output_mode=mode, **base)

        foreign = (
            {'assignments', 'mixture', 'basis', 'normalized'}
            - set(cls.VALID_OUTPUT_MODES)
        )
        for mode in sorted(foreign):
            with pytest.raises(ValueError, match="output_mode must be one of"):
                validate_mixture_config(mtype, output_mode=mode, **base)

    @pytest.mark.parametrize("mtype,base", [
        ("rbf", {"units": 8}),
        ("kmeans", {"n_clusters": 4}),
        ("gmm", {"n_components": 4}),
    ])
    def test_an_illegal_mode_raises_through_both_entry_points(self, mtype, base) -> None:
        """Direct constructor AND ``create_mixture_layer``, offending value named."""
        cls = MIXTURE_REGISTRY[mtype]['class']

        with pytest.raises(ValueError, match="nonsense"):
            cls(output_mode="nonsense", **base)

        with pytest.raises(ValueError, match="nonsense"):
            create_mixture_layer(mtype, output_mode="nonsense", **base)

        # ...and the message names the legal set, sorted, so it is stable across runs.
        legal = str(sorted(cls.VALID_OUTPUT_MODES)).replace("[", r"\[").replace("]", r"\]")
        with pytest.raises(ValueError, match=f"output_mode must be one of {legal}"):
            create_mixture_layer(mtype, output_mode="nonsense", **base)
