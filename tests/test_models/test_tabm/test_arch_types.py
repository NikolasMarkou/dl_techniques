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

from dl_techniques.models.tabular.tabm.model import TabMModel

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


class TestSharedInitializerCoincidence:
    """A SEEDED `kernel_initializer` aliases the output layer with the last block.

    `model.py` hands ONE `self.kernel_initializer` instance to both `TabMBackbone`
    and the output layer. That is exact for a SEEDLESS initializer -- the backbone
    clones per block, leaving the shared instance with one other `add_weight`
    consumer, and the two kernels then differ (measured `max|diff| = 0.42` at the
    config below). The exception is a caller-supplied SEEDED initializer:
    `clone_initializer` round-trips through `get_config()`/`from_config()` and so
    reproduces the seed ON PURPOSE (`initializers/clone.py:60-65`), which makes the
    output layer's kernel bit-identical to the last backbone block's wherever the
    two shapes coincide.

    This class pins BOTH directions of that documented contract. It is not a
    regression guard for a bug -- adding `clone_initializer` at `model.py`'s three
    fan-out sites was MEASURED to be a no-op against the seeded case, which is why
    the shipped action was a docstring plus this test rather than a fourth clone.
    See decisions.md D-004.
    """

    CONFIG = dict(
        n_num_features=8,
        cat_cardinalities=[],
        n_classes=16,
        hidden_dims=[16, 16],
        arch_type='tabm-packed',
        k=4,
    )

    @staticmethod
    def _kernels(kernel_initializer):
        model = TabMModel(**TestSharedInitializerCoincidence.CONFIG,
                          kernel_initializer=kernel_initializer)
        model(np.zeros((2, 8), "float32"))
        return (
            np.asarray(keras.ops.convert_to_numpy(model.output_layer.kernels)),
            np.asarray(keras.ops.convert_to_numpy(
                model.backbone.blocks[-1].linear.kernels)),
        )

    def test_a_seeded_initializer_aliases_the_output_layer_by_contract(self) -> None:
        out, last = self._kernels(keras.initializers.GlorotUniform(seed=7))

        # ANTI-VACUITY, and it must come first: an equality claim between two
        # arrays of different shapes is not a claim at all. This config was chosen
        # precisely so `n_classes == hidden_dims[-1] == 16` makes the output
        # layer's `(k, in, out)` kernel the same shape as the last block's.
        assert out.shape == last.shape == (4, 16, 16), (
            f"setup broken: shapes {out.shape} vs {last.shape}; the equality "
            "assertion below would be meaningless"
        )

        assert np.array_equal(out, last), (
            "RED here means `clone_initializer`'s SEEDED contract changed: it no "
            "longer reproduces a caller-supplied seed "
            "(`initializers/clone.py:60-65`). That is a behaviour change, not a "
            "bug in this test -- and it makes the `kernel_initializer` docstrings "
            "on TabMModel and create_tabm_model WRONG in the same direction, "
            "since both state that a seeded instance aliases the output layer "
            "with the last backbone block. Fix the docstrings and D-004 too."
        )

    def test_the_seedless_default_does_not_alias(self) -> None:
        # The other direction, and the reason the docstrings say "exact for a
        # SEEDLESS initializer; the exception is a SEEDED one" rather than an
        # unqualified claim in either direction. Without this arm the class above
        # would read as "the initializer always aliases", which is false.
        keras.utils.set_random_seed(1234)
        out, last = self._kernels('glorot_uniform')
        assert out.shape == last.shape == (4, 16, 16)
        assert not np.array_equal(out, last), (
            "the seedless default aliased: the backbone's per-block "
            "clone_initializer (tabm_backbone.py D-005) has stopped working, so "
            "the shared instance is being replayed at the output layer too."
        )

    def test_no_clone_was_added_at_the_model_level(self) -> None:
        # D-004 is a decision NOT to act, so the non-action needs a guard of its
        # own -- a comment cannot stop a later reader from "finishing the job".
        # A clone at model.py's three fan-out sites is a no-op against the seeded
        # case (the arm above measures exactly that), so adding one would only
        # look like a fix. If this goes red, re-derive the measurement before
        # keeping the clone: it must be justified by something D-004 did not see.
        # Checked as the MECHANISM (an imported name plus a call), not as the
        # word -- D-004's own anchor in that module names `clone_initializer`
        # several times in prose, and a bare substring search would just be
        # asserting that the anchor was deleted.
        import inspect
        from dl_techniques.models.tabular.tabm import model as tabm_model
        assert not hasattr(tabm_model, "clone_initializer"), (
            "model.py imported clone_initializer; D-004 says a clone here is a "
            "no-op against the seeded failure mode"
        )
        source = inspect.getsource(tabm_model)
        assert "clone_initializer(" not in source, (
            "model.py now CALLS clone_initializer; see D-004 -- it changes "
            "nothing for a seeded initializer and only looks like a fix"
        )
        # Anti-vacuity: the two assertions above are only meaningful if the
        # module really is the one carrying the fan-out this decision is about.
        assert source.count("kernel_initializer=self.kernel_initializer") == 3
