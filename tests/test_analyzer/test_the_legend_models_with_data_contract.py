"""`BaseVisualizer._get_models_with_data` must not be a lenient default (F-087).

`BaseVisualizer._create_figure_legend` selects which models a legend covers by a three-way
choice: `specific_models` if given, else `self.model_order` if `include_all_models`,
else `self._get_models_with_data()`.

The base implementation of `_get_models_with_data` returned `self.model_order` — the
*same list* the `include_all_models=True` branch already used. So for any subclass that
had not overridden it, the `include_all_models=False` path returned all models, under a
method name asserting they all had data. The parameter was silently inert.

All six concrete visualizers did override it, so the defect produced no wrong output; the
risk was a future eighth visualizer inheriting a base default that cannot report a
shortfall. The method is now `@abstractmethod`, so the contract is enforced by
instantiation rather than by convention.

This module pins that, plus the branch itself, which the plan to close F-087 proposed
deleting. It is not dead: it is the only consumer of `include_all_models=False`.
"""

import inspect

import matplotlib
import pytest

matplotlib.use('Agg')

from dl_techniques.analyzer.visualizers import (  # noqa: E402
    BaseVisualizer,
    CalibrationVisualizer,
    InformationFlowVisualizer,
    SpectralVisualizer,
    SummaryVisualizer,
    TrainingDynamicsVisualizer,
    WeightVisualizer,
)

CONCRETE = [
    WeightVisualizer,
    CalibrationVisualizer,
    InformationFlowVisualizer,
    TrainingDynamicsVisualizer,
    SpectralVisualizer,
    SummaryVisualizer,
]


class TestTheMethodIsAbstract:
    def test_the_base_is_abstract(self):
        assert inspect.isabstract(BaseVisualizer), (
            "BaseVisualizer became concrete; a lenient default is reachable again"
        )

    def test_the_base_lists_both_contracts(self):
        assert BaseVisualizer.__abstractmethods__ == frozenset(
            {'create_visualizations', '_get_models_with_data'}), (
            f"abstract methods are {sorted(BaseVisualizer.__abstractmethods__)}"
        )

    @pytest.mark.parametrize("cls", CONCRETE, ids=lambda c: c.__name__)
    def test_every_concrete_visualizer_implements_it(self, cls):
        assert '_get_models_with_data' not in cls.__abstractmethods__
        assert not inspect.isabstract(cls), (
            f"{cls.__name__} is still abstract: "
            f"{sorted(cls.__abstractmethods__)}"
        )

    @pytest.mark.parametrize("cls", CONCRETE, ids=lambda c: c.__name__)
    def test_no_visualizer_inherits_the_base_body(self, cls):
        """A subclass must define its OWN, not resolve to the base's."""
        assert '_get_models_with_data' in vars(cls), (
            f"{cls.__name__} does not define _get_models_with_data and relies on an "
            f"inherited one"
        )
        own = cls.__dict__['_get_models_with_data']
        assert own is not BaseVisualizer._get_models_with_data

    def test_a_new_visualizer_cannot_forget_it(self):
        """The RED proof: forgetting the override now fails at construction, not silently.

        A concrete subclass that omits the method cannot be instantiated, which is the
        whole point of the change — the failure is loud and immediate.
        """
        class _Forgets(BaseVisualizer):
            def create_visualizations(self):
                pass

        with pytest.raises(TypeError):
            _Forgets()

    def test_the_base_body_raises_rather_than_returning_everything(self):
        """Guard against a `return self.model_order` creeping back in."""
        source = inspect.getsource(BaseVisualizer._get_models_with_data)
        assert 'return self.model_order' not in source, (
            "the base _get_models_with_data returned the full model_order again, which "
            "is the defect: it asserts a model has data when it may not"
        )


class _Stub(BaseVisualizer):
    """Minimal concrete subclass with a controllable data set."""

    def __init__(self, model_order, with_data):
        self.model_order = model_order
        self.model_colors = {name: '#000000' for name in model_order}
        self._with_data = list(with_data)

    def create_visualizations(self):
        pass

    def _get_models_with_data(self):
        return self._with_data


class TestTheIncludeAllModelsBranch:
    """The branch the plan proposed deleting, pinned as load-bearing."""

    def _legend_labels(self, stub):
        """`include_all_models=False` — the branch under test."""
        import matplotlib.pyplot as plt

        fig = plt.figure()
        try:
            stub._create_figure_legend(fig, include_all_models=False)
            legend = fig.legends[0]
            return [text.get_text() for text in legend.get_texts()]
        finally:
            plt.close(fig)

    def test_false_selects_only_models_with_data(self):
        stub = _Stub(['a', 'b', 'c'], ['a', 'c'])
        assert self._legend_labels(stub) == ['a', 'c']

    def test_true_selects_every_model_in_order(self):
        stub = _Stub(['a', 'b', 'c'], ['a', 'c'])
        assert self._legend_labels_always(stub) == ['a', 'b', 'c']

    def _legend_labels_always(self, stub):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        try:
            stub._create_figure_legend(fig, include_all_models=True)
            legend = fig.legends[0]
            return [text.get_text() for text in legend.get_texts()]
        finally:
            plt.close(fig)

    def test_the_two_settings_genuinely_differ(self):
        """Without this, the branch is indistinguishable and `False` means nothing."""
        stub = _Stub(['a', 'b', 'c'], ['a', 'c'])
        assert self._legend_labels(stub) != self._legend_labels_always(stub)

    def test_specific_models_still_win(self):
        stub = _Stub(['a', 'b', 'c'], ['a'])
        import matplotlib.pyplot as plt

        fig = plt.figure()
        try:
            stub._create_figure_legend(fig, specific_models=['b', 'zzz'])
            assert [t.get_text() for t in fig.legends[0].get_texts()] == ['b'], (
                "an unknown model name must be dropped, and the list filtered"
            )
        finally:
            plt.close(fig)
