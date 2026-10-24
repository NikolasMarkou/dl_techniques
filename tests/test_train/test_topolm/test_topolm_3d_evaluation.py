"""The TopoLM trainer on a 3-D grid: evaluation, figures and one real run.

Written RED-first (plan-2026-10-23T190557-972a9ac5, step 1), before any 3-D code
existed. The baseline refuses a 3-D variant by name and every metric and plot it
would reach takes a ``(height, width)`` map, so each test below fails today for
one of those reasons and is expected to turn green in the step that supplies the
piece it exercises:

====================================  ==========================================
test class                            turns green with
====================================  ==========================================
``TestThreeDimensionalPlacement``     the N-d metrics and ``to_grid`` (steps 2, 3)
``TestThreeDimensionalReadout``       the readout refusal and skip (step 5)
``TestThreeDimensionalFigures``       the 3-D drawers (step 4) and step 5
``TestTheTrainerTrainsA3DVariant``    the whole chain, ``tiny3d`` on CPU (step 5)
====================================  ==========================================

The oracles share nothing with the code under test: the expected maps are built
by a loop through ``SpatialLayout.positions`` (never ``to_grid``,
``cell_to_unit`` or a reshape), the cluster counts are planted boxes, and the
Moran's I is the triple loop of ``tests/test_metrics``. The forward pass is
replaced by planted activations, so a failure here is about placement and
metrics and never about a model.
"""

import json
import os

import numpy as np
import pytest

from tests.test_metrics.test_spatial_autocorrelation import _brute_force_morans_i
from train.topolm import (
    SMOKE_STIMULI,
    evaluate_topography,
    plot_topography,
    run_topographic_evaluation,
    train_topolm,
)

from .test_topolm_trainer import (  # noqa: F401  (the autouse fixture must be in scope)
    StubTokenizer,
    _block_effect,
    _config,
    _evaluate_planted,
    _expected_arm,
    _never_write_to_repo_results,
    _PLANT_STIMULI,
    _on_grid_by_positions,
    _patch_data,
    _plant,
    _stub_backbone,
)

#: The imported autouse fixture is what keeps run directories out of ``results/``
#: for this module. The name is imported only to be in scope for pytest, which
#: pyflakes cannot know (it does not read the noqa marker on the import), so this
#: tuple is the one use of the name it can see.
_AUTOUSE_FIXTURES_IN_SCOPE = (_never_write_to_repo_results,)

#: The non-cubic grid every placement test uses: 7 x 8 x 14 = 784 units, so an
#: implementation that swaps two axes cannot also keep the shape.
VOLUME = (7, 8, 14)


def _stub_3d(**overrides):
    settings = dict(num_units=784, grid_dim=3, seeds=(21, 22, 23))
    settings.update(overrides)
    backbone = _stub_backbone(**settings)
    assert tuple(backbone.grid_shape) == VOLUME
    return backbone


def _boxes(backbone):
    """A different 2 x 3 x 6 = 36 voxel box per tap, as ``(lows, highs)``."""
    return {
        tap.path: ((1 + k, 2, 6), (3 + k, 5, 12))
        for k, tap in enumerate(backbone.tap_layers)
    }


class TestThreeDimensionalPlacement:
    def test_the_raw_arm_puts_t_and_significance_on_the_voxels_of_their_units(
        self, monkeypatch
    ):
        backbone = _stub_3d()
        boxes = _boxes(backbone)
        effects = {
            tap.path: _block_effect(tap.layout, *boxes[tap.path])
            for tap in backbone.tap_layers
        }
        activations = _plant(monkeypatch, backbone, effects)

        report, arrays = _evaluate_planted(backbone)

        assert report["grid_shape"] == list(VOLUME)
        expected = _expected_arm(backbone, activations)
        for tap in backbone.tap_layers:
            t_expected, sig_expected = expected[tap.path]
            t_drawn = arrays["arms"]["raw"]["t_maps"][tap.path]
            sig_drawn = arrays["arms"]["raw"]["sig_grids"][tap.path]
            assert t_drawn.shape == VOLUME and sig_drawn.shape == VOLUME
            np.testing.assert_allclose(t_drawn, t_expected, rtol=1e-9, atol=1e-9)
            np.testing.assert_array_equal(sig_drawn, sig_expected)
            lows, highs = boxes[tap.path]
            block = np.zeros(VOLUME, dtype=bool)
            block[tuple(slice(lo, hi) for lo, hi in zip(lows, highs))] = True
            np.testing.assert_array_equal(sig_drawn, block)

    def test_each_planted_box_is_one_cluster_of_36_with_the_right_bounding_box(
        self, monkeypatch
    ):
        """36 voxels whatever the order of the axes; the bounding box of the
        labelled voxels is not (it pins axis 0 as the slowest, axis 2 the fastest),
        and neither are the unit ids, which go back through ``positions``."""
        backbone = _stub_3d()
        boxes = _boxes(backbone)
        effects = {
            tap.path: _block_effect(tap.layout, *boxes[tap.path])
            for tap in backbone.tap_layers
        }
        _plant(monkeypatch, backbone, effects)

        report, arrays = _evaluate_planted(backbone)

        for tap in backbone.tap_layers:
            entry = report["arms"]["raw"]["per_tap"][tap.path]
            assert entry["cluster_sizes_a"] == [36]
            assert entry["cluster_sizes_b"] == []
            assert entry["num_significant_units"] == 36
            lows, highs = boxes[tap.path]
            labelled = np.argwhere(arrays["arms"]["raw"]["label_grids"][tap.path] > 0)
            assert labelled.min(axis=0).tolist() == list(lows)
            assert (labelled.max(axis=0) + 1).tolist() == list(highs)

    def test_a_field_smooth_on_the_layout_scores_like_the_triple_loop(self, monkeypatch):
        """Moran's I of the placed map is that of the field on the layout grid;
        placed by reshape it would be near chance. The expected value is the
        triple loop of the metrics tests, queen contiguity (26 neighbours)."""
        from scipy import ndimage

        backbone = _stub_3d()
        field = ndimage.gaussian_filter(
            np.random.default_rng(4).normal(size=VOLUME), 1.5, mode="nearest"
        )
        effects = {
            tap.path: np.array(
                [field[tuple(int(c) for c in tap.layout.positions[u])]
                 for u in range(784)]
            )
            for tap in backbone.tap_layers
        }
        _plant(monkeypatch, backbone, effects)

        report, arrays = _evaluate_planted(backbone)

        target = _brute_force_morans_i(field, "queen")
        assert target > 0.5, "the fixture is not smooth enough to discriminate"
        for tap in backbone.tap_layers:
            summary = report["arms"]["raw"]["per_tap"][tap.path]["morans_i"]
            assert summary["standard"] == pytest.approx(target, abs=1e-9)
            assert summary["num_units"] == 784
            assert summary["connectivity"] == "queen"

    def test_an_identity_layout_agrees_with_the_plain_reshape(self, monkeypatch):
        """``permute=False`` in 3-D: unit ``i`` is cell ``i`` in row-major order,
        so the old reshape and the placement are the same map."""
        from dl_techniques.metrics.topographic_selectivity import contrast_tmap

        backbone = _stub_3d(permute=False)
        boxes = _boxes(backbone)
        effects = {
            tap.path: _block_effect(tap.layout, *boxes[tap.path])
            for tap in backbone.tap_layers
        }
        activations = _plant(monkeypatch, backbone, effects)

        _, arrays = _evaluate_planted(backbone)

        for tap in backbone.tap_layers:
            t_units = contrast_tmap(
                activations["a"][tap.path], activations["b"][tap.path]
            )[0]
            drawn = arrays["arms"]["raw"]["t_maps"][tap.path]
            np.testing.assert_allclose(drawn, t_units.reshape(VOLUME), atol=1e-12)
            np.testing.assert_allclose(
                drawn, _on_grid_by_positions(tap.layout, t_units), atol=1e-12
            )


class TestThreeDimensionalReadout:
    """D-005: the Gaussian readout is a ``Conv2D`` and stays 2-D, so a 3-D run has
    a raw arm only. A typed ``--readout-fwhm`` is refused at the CLI (see
    ``test_cli_contract``); here, the evaluation itself."""

    def test_asking_the_evaluation_for_a_readout_on_a_3d_backbone_is_refused_by_name(
        self, monkeypatch
    ):
        backbone = _stub_3d()
        effects = {t.path: np.zeros(784) for t in backbone.tap_layers}
        _plant(monkeypatch, backbone, effects)
        with pytest.raises(ValueError) as caught:
            _evaluate_planted(backbone, readout_fwhm=2.0)
        message = str(caught.value).lower()
        assert "readout" in message
        assert any(word in message for word in ("3d", "3-d", "grid_dim", "rank"))

    def test_the_run_level_evaluation_skips_the_arm_and_says_so(
        self, monkeypatch, tmp_path
    ):
        """A programmatic config left at the dataclass default ``readout_fwhm=2.0``
        still runs on a 3-D backbone: one arm, and a top-level ``readout_skipped``
        in the report (3-D only; a 2-D report gains no key)."""
        backbone = _stub_3d()
        boxes = _boxes(backbone)
        effects = {
            tap.path: _block_effect(tap.layout, *boxes[tap.path])
            for tap in backbone.tap_layers
        }
        _plant(monkeypatch, backbone, effects)
        config = _config(
            model_variant="tiny3d", spatial_radius=None, readout_fwhm=2.0,
            stimuli=_PLANT_STIMULI,
        )

        report = run_topographic_evaluation(
            backbone, config, StubTokenizer(), str(tmp_path)
        )

        assert list(report["arms"]) == ["raw"]
        assert report.get("readout_skipped"), report.keys()
        written = json.loads((tmp_path / "topography_report.json").read_text())
        assert list(written["arms"]) == ["raw"]
        assert "t_maps_readout.png" not in os.listdir(tmp_path)


class TestAThreeDimensionalReportAlwaysSaysItsReadoutIsAbsent:
    """Completion fix step 5.1: the CLI leaves ``readout_fwhm`` unset for a 3-D
    variant, so no readout was REQUESTED and the report used to carry no
    ``readout_skipped`` at all. RED on the previous evaluation: the key was set only
    on the path that requested a readout."""

    def _planted_3d(self, monkeypatch):
        backbone = _stub_3d()
        boxes = _boxes(backbone)
        effects = {
            tap.path: _block_effect(tap.layout, *boxes[tap.path])
            for tap in backbone.tap_layers
        }
        _plant(monkeypatch, backbone, effects)
        return backbone

    def test_evaluate_topography_without_a_readout_still_carries_the_key(
        self, monkeypatch
    ):
        backbone = self._planted_3d(monkeypatch)
        report, _ = _evaluate_planted(backbone)
        assert list(report["arms"]) == ["raw"]
        assert report["readout_skipped"].startswith("3D: the Gaussian readout is 2-D only")

    def test_the_run_level_evaluation_without_a_readout_still_carries_the_key(
        self, monkeypatch, tmp_path
    ):
        backbone = self._planted_3d(monkeypatch)
        config = _config(
            model_variant="tiny3d", spatial_radius=None, readout_fwhm=None,
            stimuli=_PLANT_STIMULI,
        )
        report = run_topographic_evaluation(
            backbone, config, StubTokenizer(), str(tmp_path)
        )
        written = json.loads((tmp_path / "topography_report.json").read_text())
        assert report["readout_skipped"] == written["readout_skipped"]
        assert "2-D only" in written["readout_skipped"]

    def test_a_requested_readout_keeps_its_own_more_specific_reason(
        self, monkeypatch, tmp_path
    ):
        backbone = self._planted_3d(monkeypatch)
        config = _config(
            model_variant="tiny3d", spatial_radius=None, readout_fwhm=2.0,
            stimuli=_PLANT_STIMULI,
        )
        report = run_topographic_evaluation(
            backbone, config, StubTokenizer(), str(tmp_path)
        )
        assert "readout_fwhm=2.0 was not applied" in report["readout_skipped"]

    def test_a_short_cpu_tiny3d_run_with_no_readout_reports_the_key(self, monkeypatch):
        _patch_data(monkeypatch)
        config = _config(
            model_variant="tiny3d", spatial_radius=None, num_layers=2,
            readout_fwhm=None, save_dir="run3d_none",
        )
        result = train_topolm(config, preprocessor=StubTokenizer())
        report = json.loads(open(
            os.path.join(result["results_dir"], "topography_report.json")).read())
        assert list(report["arms"]) == ["raw"]
        assert "2-D only" in report["readout_skipped"], report.keys()

    def test_a_two_dimensional_report_has_no_such_key(self, monkeypatch):
        """The step-1 pin, restated on both 2-D paths: with or without a readout."""
        _patch_data(monkeypatch)
        result = train_topolm(
            _config(readout_fwhm=None, save_dir="run2d_none"),
            preprocessor=StubTokenizer(),
        )
        report = json.loads(open(
            os.path.join(result["results_dir"], "topography_report.json")).read())
        assert list(report["arms"]) == ["raw"]
        assert "readout_skipped" not in report, report.keys()


class TestThreeDimensionalFigures:
    def test_the_figure_set_of_a_3d_evaluation(self, monkeypatch, tmp_path):
        backbone = _stub_3d()
        boxes = _boxes(backbone)
        effects = {
            tap.path: _block_effect(tap.layout, *boxes[tap.path])
            for tap in backbone.tap_layers
        }
        _plant(monkeypatch, backbone, effects)
        report, arrays = _evaluate_planted(backbone)

        written = plot_topography(report, arrays, str(tmp_path))

        assert sorted(os.path.basename(p) for p in written) == [
            "clusters_raw.png", "morans_i_raw.png", "t_maps_raw.png",
        ]
        for path in written:
            assert os.path.getsize(path) > 0, path


class TestTheTrainerTrainsA3DVariant:
    """Through the REAL ``train_topolm``: model, fit, report and figures.

    Step 5 turns this green; today ``_require_two_dimensional_variant`` refuses
    ``tiny3d`` by name before anything is built. ``spatial_radius=None`` asks for
    the variant row's own radius (2, the only one a 7x7x7 grid can hold with a
    cubic L-infinity patch), and the dataclass-default ``readout_fwhm=2.0`` is
    left alone so the skip path is exercised too.
    """

    def test_a_tiny3d_run_trains_evaluates_and_draws(self, monkeypatch):
        _patch_data(monkeypatch)
        config = _config(
            model_variant="tiny3d", spatial_radius=None, num_layers=2,
            save_dir="run3d",
        )

        result = train_topolm(config, preprocessor=StubTokenizer())

        results_dir = result["results_dir"]
        written = set(os.listdir(results_dir))
        assert {"config.json", "training_history.json", "topography_report.json",
                "t_maps_raw.png", "morans_i_raw.png"} <= written, sorted(written)
        assert "t_maps_readout.png" not in written

        report = json.loads(open(os.path.join(results_dir, "topography_report.json")).read())
        assert report["grid_shape"] == [7, 7, 7]
        assert list(report["arms"]) == ["raw"]
        assert report.get("readout_skipped")
        assert result["backbone"].radius == 2

        # The maps themselves must be real: finite, and not all zero.
        _, arrays = evaluate_topography(
            result["backbone"], SMOKE_STIMULI, ("a", "b"), StubTokenizer(),
            readout_fwhm=None, min_cluster_size=2,
        )
        for name, t_map in arrays["arms"]["raw"]["t_maps"].items():
            assert t_map.shape == (7, 7, 7), name
            assert np.all(np.isfinite(t_map)), name
            assert np.any(t_map != 0.0), name

    def test_a_tiny3d_run_leaves_real_maps_and_a_consistent_figure_set(
        self, monkeypatch
    ):
        """Added with step 5 (the test above is the step-1 RED oracle, kept as it
        was). What a person opening the run directory needs to be true: the
        config records the radius the model used, every t-map is a finite
        7x7x7 volume that is not all zero (a dead map would plot fine and mean
        nothing), the figure set matches the report (a cluster figure exactly
        when a cluster was found), and no readout file exists."""
        _patch_data(monkeypatch)
        config = _config(
            model_variant="tiny3d", spatial_radius=None, num_layers=2,
            save_dir="run3d_real",
        )

        result = train_topolm(config, preprocessor=StubTokenizer())

        results_dir = result["results_dir"]
        written = set(os.listdir(results_dir))
        recorded = json.loads(open(os.path.join(results_dir, "config.json")).read())
        assert recorded["spatial_radius"] == 2
        assert recorded["model_variant"] == "tiny3d"

        report = json.loads(
            open(os.path.join(results_dir, "topography_report.json")).read()
        )
        assert report["grid_shape"] == [7, 7, 7]
        assert report["num_taps"] == len(result["backbone"].tap_layers) > 0
        raw = report["arms"]["raw"]
        found = raw["total_clusters_a"] + raw["total_clusters_b"]
        assert ("clusters_raw.png" in written) == (found > 0), (found, sorted(written))
        assert not any(name.endswith("_readout.png") for name in written)

        _, arrays = evaluate_topography(
            result["backbone"], SMOKE_STIMULI, ("a", "b"), StubTokenizer(),
            readout_fwhm=None, min_cluster_size=2,
        )
        t_maps = arrays["arms"]["raw"]["t_maps"]
        assert len(t_maps) == report["num_taps"]
        for name, t_map in t_maps.items():
            assert t_map.shape == (7, 7, 7), name
            assert np.all(np.isfinite(t_map)), name
            assert np.any(t_map != 0.0), f"{name}: an all-zero t-map"
