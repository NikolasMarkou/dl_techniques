"""`arch_type` must select a genuinely different model, and aggregation must be explicit.

Before 2026-08-14, `arch_type` accepted `'tabm'`, `'tabm-packed'` and
`'tabm-normal'` and `_create_layers` branched on none of them: all three built
byte-identical models (612 params, max|dy| = 0.0 from the same seed), so an
ablation over the three could be "run" without ever varying anything.
`'tabm-mini'` also carried the per-layer rank-1 perturbation its own module
docstring said it did not, which made it "tabm plus an adapter" rather than the
adapter-only limit case.

These tests pin the property that fix has to preserve: every `arch_type` names a
distinct build. They fail against a table where any two rows collapse.
"""

import numpy as np
import keras
import pytest

from dl_techniques.models.tabm.model import TabMModel

ENSEMBLE_ARCHS = [
    'tabm',
    'tabm-normal',
    'tabm-packed',
    'tabm-mini',
    'tabm-mini-normal',
]


def _build(arch_type: str, seed: int = 1234) -> TabMModel:
    keras.utils.set_random_seed(seed)
    return TabMModel(
        n_num_features=6,
        cat_cardinalities=[],
        n_classes=3,
        hidden_dims=[16, 8],
        arch_type=arch_type,
        k=4,
    )


@pytest.fixture()
def x() -> np.ndarray:
    return np.random.RandomState(0).randn(4, 6).astype("float32")


class TestArchTypeIsRead:
    """Each accepted `arch_type` must reach layer construction."""

    def test_every_arch_type_has_a_spec_row(self) -> None:
        assert set(TabMModel.ARCH_SPECS) == set(ENSEMBLE_ARCHS) | {'plain'}

    def test_no_two_spec_rows_are_equal(self) -> None:
        rows = [tuple(sorted(v.items())) for v in TabMModel.ARCH_SPECS.values()]
        assert len(set(rows)) == len(rows), (
            "two arch_type values build the same thing; one of them is inert"
        )

    def test_unknown_arch_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown arch_type"):
            TabMModel(
                n_num_features=4,
                cat_cardinalities=[],
                n_classes=2,
                hidden_dims=[8],
                arch_type='tabm-does-not-exist',
                k=2,
            )

    @pytest.mark.parametrize("arch_type", ENSEMBLE_ARCHS)
    def test_forward_shape(self, arch_type: str, x: np.ndarray) -> None:
        out = np.asarray(_build(arch_type)(x, training=False))
        assert out.shape == (4, 4, 3)
        assert np.all(np.isfinite(out))

    def test_outputs_differ_across_arch_types(self, x: np.ndarray) -> None:
        """The regression guard: identical seed, identical input, different output."""
        outs = {a: np.asarray(_build(a)(x, training=False)) for a in ENSEMBLE_ARCHS}
        base = outs['tabm']
        for arch_type in ENSEMBLE_ARCHS[1:]:
            delta = float(np.abs(outs[arch_type] - base).max())
            assert delta > 1e-4, (
                f"arch_type={arch_type!r} is bit-identical to 'tabm' "
                f"(max|dy|={delta:.3e}) — the knob does nothing"
            )

    def test_packed_costs_k_independent_backbone_kernels(self) -> None:
        """`'tabm-packed'` must actually pay for independent kernels."""
        efficient = _build('tabm')
        packed = _build('tabm-packed')
        efficient(np.zeros((1, 6), "float32"))
        packed(np.zeros((1, 6), "float32"))
        assert packed.count_params() > efficient.count_params()

    def test_mini_has_no_per_layer_perturbation(self) -> None:
        """`'tabm-mini'` diversity comes only from the input adapter."""
        mini = _build('tabm-mini')
        mini(np.zeros((1, 6), "float32"))
        assert mini.minimal_ensemble_adapter is not None
        for block in mini.backbone.blocks:
            assert not block.linear.ensemble_scaling_in
            assert not block.linear.ensemble_scaling_out

    def test_full_tabm_does_have_per_layer_perturbation(self) -> None:
        full = _build('tabm')
        full(np.zeros((1, 6), "float32"))
        assert full.minimal_ensemble_adapter is None
        for block in full.backbone.blocks:
            assert block.linear.ensemble_scaling_in
            assert block.linear.ensemble_scaling_out

    def test_scaling_vectors_are_not_all_ones_for_tabm(self) -> None:
        """`init_distribution='random-signs'` must reach the weights.

        With the previous hard-coded `'ones'` initializer every member shared one
        effective weight matrix at init, so 'tabm' and 'tabm-normal' could not
        have differed even if the branch had existed.
        """
        full = _build('tabm')
        full(np.zeros((1, 6), "float32"))
        r = np.asarray(full.backbone.blocks[0].linear.r)
        assert not np.allclose(r, 1.0)
        assert set(np.unique(r)) <= {-1.0, 1.0}


class TestUncertaintyIsExplicit:
    """`call` returns the raw member axis; aggregation is an opt-in method."""

    def test_call_is_not_aggregated(self, x: np.ndarray) -> None:
        out = np.asarray(_build('tabm')(x, training=False))
        assert out.ndim == 3 and out.shape[1] == 4

    def test_predict_with_uncertainty_shapes_and_values(self, x: np.ndarray) -> None:
        model = _build('tabm')
        mean, std = model.predict_with_uncertainty(x, verbose=0)
        assert mean.shape == (4, 3)
        assert std.shape == (4, 3)

        raw = model.predict(x, verbose=0)
        np.testing.assert_allclose(mean, raw.mean(axis=1), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(std, raw.std(axis=1), rtol=1e-6, atol=1e-6)

    def test_plain_has_zero_spread(self) -> None:
        keras.utils.set_random_seed(1234)
        model = TabMModel(
            n_num_features=6,
            cat_cardinalities=[],
            n_classes=3,
            hidden_dims=[16, 8],
            arch_type='plain',
        )
        _, std = model.predict_with_uncertainty(
            np.random.RandomState(0).randn(4, 6).astype("float32"), verbose=0
        )
        np.testing.assert_allclose(std, 0.0, atol=0.0)
