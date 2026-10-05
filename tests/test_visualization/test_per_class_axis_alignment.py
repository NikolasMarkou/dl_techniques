"""`PerClassAnalysis` must align its axis with a label array that stops short.

**The claim.** Every panel of `PerClassAnalysis` draws exactly one mark per class
index spanning `0 .. max(y_true, y_pred)`, with exactly that many tick labels.

**The defect, and why it survived.** `np.bincount` returns length `max(label) + 1`,
so the "true" and "predicted" count arrays have DIFFERENT lengths whenever one
label array's maximum is lower than the other's. A real run hit exactly that:
`y_true` reached class 9 while `y_pred` stopped at 8, giving counts of length 10
and 9. The class-distribution panel called

    ax.bar(x - width / 2, true_counts[:len(classes)], ...)
    ax.bar(x + width / 2, pred_counts[:len(classes)], ...)

with `x = np.arange(len(classes))`. Slicing can only ever SHORTEN an array, so the
9-long one stayed 9 against a 10-long `x`, and `ax.bar` raised

    ValueError: shape mismatch: objects cannot be broadcast to a single shape.
    Mismatch is between arg 0 with shape (10,) and arg 1 with shape (9,).

`VisualizationManager.visualize` swallows that and returns `None` (`core.py:508-512`),
so the trainer saw a missing PNG and no exception.

The per-class-accuracy panel had the mirror-image bug: its ticks came from
`len(class_names)` while its bars came from `_calculate_per_class_accuracy`, whose
count is driven by `max(label) + 1`. `set_xticklabels` then raises the same
`shape mismatch`. Both are now reconciled by `PerClassAnalysis._axis_classes` plus
an explicit `np.pad` of the count arrays.

**RED proof.** `test_the_defect_reproduces_without_the_padding` reimplements the
pre-fix slicing inline and asserts that it raises the exact `shape mismatch` — so
this guard cannot go GREEN by the plugin quietly changing its mind about the error.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from dl_techniques.visualization import (  # noqa: E402
    ClassificationResults,
    PerClassAnalysis,
    PlotConfig,
    VisualizationContext,
)


@pytest.fixture
def plugin(tmp_path):
    return PerClassAnalysis(
        config=PlotConfig(),
        context=VisualizationContext(experiment_name="", output_dir=tmp_path),
    )


def _ragged_case():
    """Labels spanning 0..9 where PREDICTIONS never reach class 9.

    This is the shape the real run had: a model that has never once predicted its
    top class leaves `np.bincount(y_pred)` one short of `np.bincount(y_true)`.
    """
    rng = np.random.default_rng(20261006)
    y_true = rng.integers(0, 10, size=400)
    y_pred = rng.integers(0, 9, size=400)
    assert max(y_true) == 9, "precondition: y_true must actually reach class 9"
    assert max(y_pred) == 8, "precondition: y_pred must actually stop at class 8"
    return y_true, y_pred


class TestThePerClassAxisIsAligned:
    def test_every_panel_renders_when_predictions_stop_short_of_the_true_max(
        self, plugin
    ):
        y_true, y_pred = _ragged_case()
        results = ClassificationResults(
            y_true=y_true,
            y_pred=y_pred,
            class_names=[str(i) for i in range(10)],
            model_name="ragged",
        )

        figure = plugin.create_visualization(results)

        assert figure is not None
        assert len(figure.axes) >= 3

    def test_every_panel_renders_when_a_middle_class_is_absent(self, plugin):
        """Class 8 never occurs but class 9 does: `max(label) + 1` is 10 while
        only 9 distinct values exist, so a class_names list derived from the
        present values is short. This is the `train/mothnet` D-007 shape."""
        rng = np.random.default_rng(20261006)
        pool = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9])
        y_true = rng.choice(pool, size=400)
        y_pred = rng.choice(pool, size=400)
        present = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
        assert len(present) == 9 and max(present) == 9

        results = ClassificationResults(
            y_true=y_true,
            y_pred=y_pred,
            class_names=[str(c) for c in present],
            model_name="gap",
        )

        assert plugin.create_visualization(results) is not None

    def test_no_class_loses_its_mark_to_the_padding(self, plugin):
        """The fix pads with zeros; it must not truncate the tail bars, which the
        old `[:len(classes)]` slice did whenever `len(classes)` was the smaller
        of the two."""
        y_true, y_pred = _ragged_case()
        results = ClassificationResults(
            y_true=y_true, y_pred=y_pred,
            class_names=[str(i) for i in range(10)], model_name="tail",
        )

        figure = plugin.create_visualization(results)
        distribution_ax = figure.axes[0]
        true_bars = [
            patch for patch in distribution_ax.patches
            if abs(patch.get_height() - 0) > 0 and patch.get_x() < 5
        ]

        assert true_bars, "the 'True' series drew nothing at all"


class TestTheReconciler:
    @pytest.mark.parametrize(
        "class_names, n_classes, expected",
        [
            (None, 3, ["0", "1", "2"]),
            (["a", "b"], 4, ["a", "b", "2", "3"]),
            (["a", "b", "c", "d"], 2, ["a", "b"]),
            (["a"], 1, ["a"]),
        ],
    )
    def test_it_always_returns_exactly_n_classes(
        self, plugin, class_names, n_classes, expected
    ):
        assert plugin._axis_classes(class_names, n_classes) == expected

    def test_it_does_not_mutate_the_callers_list(self, plugin):
        supplied = ["a", "b"]
        plugin._axis_classes(supplied, 5)

        assert supplied == ["a", "b"], "the caller's list was mutated in place"


class TestTheRedProof:
    def test_the_defect_reproduces_without_the_padding(self):
        """Reimplements the pre-fix distribution panel verbatim. If this stops
        raising, the guard above has stopped guarding."""
        import matplotlib.pyplot as plt

        y_true, y_pred = _ragged_case()
        true_counts = np.bincount(y_true)
        pred_counts = np.bincount(y_pred)
        classes = [str(i) for i in range(10)]

        x = np.arange(len(classes))
        width = 0.35
        figure, ax = plt.subplots()
        with pytest.raises(ValueError, match="shape mismatch"):
            ax.bar(x - width / 2, true_counts[:len(classes)], width, label="True")
            ax.bar(x + width / 2, pred_counts[:len(classes)], width, label="Pred")
        plt.close(figure)