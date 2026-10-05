"""A truncated model name must not cost the row its colour (F-077).

The defect. Both performance tables truncate the model name in `row_data[0]` so it does
not overflow the cell, and both then used that same TRUNCATED string as the colour key.
`_get_model_color` looks the name up in `self.model_colors`, which is keyed by the FULL
name, so any model longer than the truncation length missed and fell back to the literal
`'#333333'`.

Why it is worse than cosmetic: the fallback grey is also what every UNKNOWN model gets.
So in a sweep whose names all exceed 12 characters -- `'ResNet_v1'`, `'ConvNext_v2'`,
`'efficientnet_b0'` and so on, i.e. the normal case -- every row rendered in the SAME
grey and the models were indistinguishable, in a tool whose entire purpose is comparing
them. Two long names rendering identically rather than wrongly is the failure mode that
survives review, because each row looks perfectly reasonable on its own.

`training_dynamics_visualizer.py` had three separate inline copies of the truncation
length; this file pins one length across all three tables.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

from dl_techniques.analyzer.config import AnalysisConfig
from dl_techniques.analyzer.data_types import AnalysisResults, TrainingMetrics
from dl_techniques.analyzer.utils import truncate_model_name
from dl_techniques.analyzer.visualizers.summary_visualizer import SummaryVisualizer
from dl_techniques.analyzer.visualizers.training_dynamics_visualizer import (
    TrainingDynamicsVisualizer,
    MODEL_NAME_TRUNCATE_LENGTH,
)

#: Long enough that `truncate_model_name` and the training table both shorten them.
LONG_NAMES = ['ResNet_v1_large', 'ConvNext_v2_huge', 'efficientnet_b0']


def _results(names, with_training):
    results = AnalysisResults()
    results.model_metrics = {
        name: {'loss': 0.5, 'accuracy': 0.8, 'status': 'success'} for name in names
    }
    if with_training:
        results.training_history = {
            name: {'loss': [1.0, 0.7, 0.5], 'val_loss': [1.1, 0.8, 0.6],
                   'accuracy': [0.4, 0.7, 0.8], 'val_accuracy': [0.35, 0.65, 0.78]}
            for name in names
        }
        results.training_metrics = TrainingMetrics()
        results.training_metrics.peak_performance = {
            name: {'epoch': 2, 'val_accuracy': 0.78, 'val_loss': 0.6}
            for name in names
        }
        results.training_metrics.epochs_to_convergence = {name: 2 for name in names}
        results.training_metrics.training_stability_score = {name: 0.02 for name in names}
        results.training_metrics.overfitting_index = {name: 0.1 for name in names}
        results.training_metrics.final_gap = {name: 0.1 for name in names}
    return results


def _colors(names):
    return {name: f'#{i:06x}' for i, name in enumerate(names, start=0x112233)}


def _cell_facecolors(table, n_cols):
    """Read the facecolour of every cell in each body row of a matplotlib table."""
    rows = []
    keys = sorted(k for k in table.get_celld().keys() if k[0] > 0)
    by_row = {}
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            continue
        by_row.setdefault(row, []).append((col, cell))
    for row in sorted(by_row):
        cells = [c for _, c in sorted(by_row[row])]
        rows.append(cells)
    return rows


class TestTruncatedNamesKeepTheirModelColour:
    @pytest.mark.parametrize("with_training", [False, True],
                             ids=["standard_table", "training_table"])
    def test_each_long_name_gets_its_own_colour(self, tmp_path, with_training):
        results = _results(LONG_NAMES, with_training)
        colors = _colors(LONG_NAMES)

        viz = SummaryVisualizer(results, AnalysisConfig(save_plots=True),
                                tmp_path, colors)
        fig, ax = plt.subplots()
        viz._plot_performance_table(ax)

        table = ax.tables[0] if hasattr(ax, 'tables') else None
        assert table is not None, "no table was rendered"

        # Every long name IS truncated, so a row keyed on the display string cannot
        # resolve a colour. Assert the precondition, then the consequence.
        for name in LONG_NAMES:
            assert len(truncate_model_name(name)) <= 12

        body_rows = set()
        for (row, _col), cell in table.get_celld().items():
            if row > 0:
                body_rows.add(tuple(np.round(np.asarray(cell.get_facecolor()), 6)))
        assert len(body_rows) == len(LONG_NAMES), (
            f"expected {len(LONG_NAMES)} distinct row colours, got {len(body_rows)}: "
            f"{body_rows}. F-077 regressed -- the truncated display name is being used "
            f"as the colour key."
        )
        plt.close(fig)

    def test_a_short_name_is_unaffected(self, tmp_path):
        """One name under the limit, one over: only the long one was ever at risk."""
        names = ['tiny', 'ResNet_v1_large']
        viz = SummaryVisualizer(_results(names, False), AnalysisConfig(save_plots=True),
                                tmp_path, _colors(names))
        fig, ax = plt.subplots()
        viz._plot_performance_table(ax)
        table = ax.tables[0]
        body_rows = set()
        for (row, _col), cell in table.get_celld().items():
            if row > 0:
                body_rows.add(tuple(np.round(np.asarray(cell.get_facecolor()), 6)))
        assert len(body_rows) == 2
        plt.close(fig)


class TestTheTruncationLengthIsDeclaredOnce:
    def test_the_training_table_uses_the_named_constant(self):
        """F-077: three inline copies of `12` existed across the three tables.

        `weight_visualizer.py` declared a pair of constants, this file spelled the numbers
        inline, and `information_flow_visualizer.py` spelled them inline again. A
        behaviour change to one table then left the others disagreeing.
        """
        import inspect
        source = inspect.getsource(TrainingDynamicsVisualizer)
        assert f"MODEL_NAME_TRUNCATE_LENGTH]" in source, (
            "the training table no longer truncates with the named constant; it is "
            "re-introducing an inline copy of the length"
        )
        assert "MODEL_NAME_ELLIPSIS" in source, (
            "the training table must use the named ellipsis, not a bare '...'"
        )
        assert "+ '...'" not in source, (
            "a bare '...' literal reappeared in training_dynamics_visualizer.py"
        )
        # No bare `[:12]` left in this module.
        assert "[:12]" not in source, (
            "an inline `[:12]` reappeared in training_dynamics_visualizer.py"
        )

    def test_the_summary_table_delegates_to_the_shared_helper(self):
        import inspect
        source = inspect.getsource(SummaryVisualizer)
        assert "truncate_model_name(model_name)" in source, (
            "the summary table must call the shared truncate_model_name helper rather "
            "than re-implementing the truncation"
        )


import matplotlib.pyplot as plt  # noqa: E402  (imported late: needs the Agg backend set)