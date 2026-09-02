"""Executable guards for the prose claims in ``dl_techniques.analyzer``.

Every test here turns a sentence somebody wrote into something that runs. The package
has a recorded failure mode of documentation drifting away from the code it describes
(the README told users the information-flow analyzer used a PyTorch hook and to switch
the feature off, months after the capture path had been rewritten and was working), and
a second one of a doc-repair pass inventing *new* false claims that eyeballing passes
twice. A grep proves a line exists; only execution proves it is true.

Scope, one class per documented claim:

* ``TestInformationFlowIsProduced`` -- README "Information flow" section (plan step 35).
* ``TestAlphaWeightedIsTheCanonicalName`` -- the ``alpha_weighted`` / ``alpha_hat`` rows and
  the cross-architecture comparability caveat (plan step 36).
* ``TestTheSummaryDashboardSurvivesADegenerateWeightPca``,
  ``TestGetSummaryStatisticsIsKeyedByCategory``, ``TestMpSoftrankIsDocumentedAsARealMetric``,
  ``TestPlPvalueSemanticsAreDocumented``, ``TestTheDocumentedSpectralColumnsMatchTheFrame``
  -- the output table, the API table and the spectral column census (plan step 37).
* ``TestTheModuleDocstringResolves`` -- every import path, attribute and call keyword the
  ``__init__`` docstring and the README quick start demonstrate (plan step 38).
* ``TestCorrelationTrapsDocMatchesTheImplementation`` -- ``CORRELATION_TRAPS.md`` against the
  shipped trap detector, including the divergences it now declares (plan step 39).
* ``TestEveryQuotedPytestSelectorMatchesSomething`` -- the doc pointers themselves.
"""

import ast
import re
from pathlib import Path

import keras
import numpy as np
import pytest

import dl_techniques.analyzer as analyzer_pkg
from dl_techniques.analyzer.config import AnalysisConfig
from dl_techniques.analyzer.data_types import DataInput
from dl_techniques.analyzer.model_analyzer import ModelAnalyzer

PACKAGE_ROOT = Path(analyzer_pkg.__file__).parent
README_PATH = PACKAGE_ROOT / "README.md"

N_SAMPLES = 40
N_FEATURES = 8
N_CLASSES = 3


def _read_readme() -> str:
    """Return the package README as text.

    Returns:
        str: The full contents of ``src/dl_techniques/analyzer/README.md``.
    """
    return README_PATH.read_text(encoding="utf-8")


def _readme_section(title: str) -> str:
    """Return the body of one ``###`` section of the README.

    Args:
        title: The section heading text, without the leading ``###``.

    Returns:
        str: Everything from the heading up to the next heading of the same or a
        higher level.

    Raises:
        AssertionError: If the section is not present, so a renamed heading reddens
        rather than silently yielding an empty (and therefore claim-free) string.
    """
    text = _read_readme()
    match = re.search(
        rf"^### {re.escape(title)}$(.*?)(?=^#{{1,3}} )",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"README has no '### {title}' section"
    body = match.group(1)
    assert body.strip(), f"README section '### {title}' is empty"
    return body


def _package_python_files() -> list[Path]:
    """Every ``.py`` file shipped in the analyzer package."""
    files = sorted(PACKAGE_ROOT.rglob("*.py"))
    assert len(files) > 10, f"the package source sweep found only {len(files)} files"
    return files


def _build_probe_model(name: str = "docs_model") -> keras.Model:
    """A tiny two-Dense softmax classifier with deterministic weights."""
    keras.utils.set_random_seed(3)
    inputs = keras.Input(shape=(N_FEATURES,), name=f"{name}_in")
    hidden = keras.layers.Dense(16, activation="relu", name=f"{name}_d1")(inputs)
    outputs = keras.layers.Dense(N_CLASSES, activation="softmax", name=f"{name}_out")(hidden)
    return keras.Model(inputs=inputs, outputs=outputs, name=name)


@pytest.fixture()
def probe_data() -> DataInput:
    """Deterministic ``(x, y)`` with integer labels over ``N_CLASSES``."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((N_SAMPLES, N_FEATURES)).astype("float32")
    y = rng.integers(0, N_CLASSES, size=N_SAMPLES)
    return DataInput(x_data=x, y_data=y)


# ---------------------------------------------------------------------
# D1 -- the README told users to disable a feature that works
# ---------------------------------------------------------------------

class TestInformationFlowIsProduced:
    """The README's information-flow section must describe the shipped capture path.

    Defect guarded (D1): the section said "**Broken.** ``information_flow_analyzer.py``
    captures activations with ``layer.register_forward_hook(...)``, which is a PyTorch
    API ... Every model raises ``AttributeError`` ... Set ``analyze_information_flow=
    False`` until it is fixed", and the output-file table called
    ``information_flow_analysis.png`` "currently never produced". The shipped code wraps
    ``layer.call`` and runs one eager forward pass; a run of this module's own probe
    produced ``information_flow = {'docs_model': ['docs_model_d1', 'docs_model_out']}``
    and wrote the PNG.
    """

    def test_information_flow_is_populated_and_the_png_is_written(self, tmp_path, probe_data):
        """PIN, not RED evidence: this passes against the shipped code, which is the
        point -- it is what makes the README's "Broken" claim falsifiable, and it
        reddens if the capture path regresses."""
        model = _build_probe_model()
        output_dir = tmp_path / "info_flow"
        analyzer = ModelAnalyzer(
            models={"docs_model": model},
            config=AnalysisConfig(
                analyze_weights=False,
                analyze_calibration=False,
                analyze_information_flow=True,
                analyze_training_dynamics=False,
                analyze_spectral=False,
                n_samples=N_SAMPLES,
                save_plots=True,
                verbose=False,
            ),
            output_dir=str(output_dir),
        )
        results = analyzer.analyze(probe_data, analysis_types={"information_flow"})

        assert results.information_flow, (
            "results.information_flow is empty -- the README's 'Broken' claim would be true"
        )
        per_layer = results.information_flow["docs_model"]
        assert per_layer, "no layer was captured for the probe model"

        # Every metric the README's section names must actually be present.
        documented = {
            "layer_type",
            "output_shape",
            "mean_activation",
            "std_activation",
            "sparsity",
            "positive_ratio",
            "effective_rank",
            "capture_index",
        }
        for layer_name, entry in per_layer.items():
            missing = documented - set(entry)
            assert not missing, (
                f"layer '{layer_name}' is missing README-documented keys: {sorted(missing)}"
            )

        png = output_dir / "information_flow_analysis.png"
        assert png.exists(), (
            "information_flow_analysis.png was not written; the README's output table "
            f"lists it under analyze_information_flow. Files present: "
            f"{sorted(p.name for p in output_dir.iterdir())}"
        )

    def test_the_readme_does_not_teach_the_pytorch_hook(self):
        """The stale mechanism, the stale status and the stale workaround are all gone."""
        text = _read_readme()
        for stale in (
            "register_forward_hook",
            "currently never produced",
            "analyze_information_flow=False",
        ):
            assert stale not in text, (
                f"README still carries the stale information-flow claim {stale!r}; "
                "the shipped analyzer wraps `layer.call` and the feature works"
            )

    def test_the_readme_describes_the_shipped_capture_mechanism(self):
        """The replacement prose names the real mechanism, not just the absence of the old one."""
        section = _readme_section("Information flow (`results.information_flow`, per layer)")
        for required in ("layer.call", "eager", "finally", "capture_index", "memory_limit_mb"):
            assert required in section, (
                f"the information-flow section does not mention {required!r}, which is "
                "part of the mechanism it now claims to describe"
            )

    def test_no_source_file_calls_register_forward_hook(self):
        """No ``.py`` in the package touches the PyTorch-only hook API.

        Parsed with ``ast`` and matched on ATTRIBUTE ACCESS, so the historical note in
        ``information_flow_analyzer.py``'s comment (which names the API precisely to
        stop it coming back) neither satisfies nor trips this guard -- the same
        text-vs-code trap that a bare substring scan fell into elsewhere in this suite.
        """
        offenders = []
        for path in _package_python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr == "register_forward_hook":
                    offenders.append(f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno}")
        assert not offenders, (
            f"register_forward_hook is a PyTorch API and is used at: {offenders}"
        )

    def test_the_hook_name_is_still_findable_as_prose(self):
        """Anti-vacuity: the AST guard above must be measuring something.

        ``register_forward_hook`` DOES appear in the package as text (a comment
        explaining why it must never return). If this arm ever fails, the AST guard has
        become vacuous for a different reason than "the code is clean" and needs
        re-pointing.
        """
        hits = [
            path.name
            for path in _package_python_files()
            if "register_forward_hook" in path.read_text(encoding="utf-8")
        ]
        assert hits, (
            "no source file mentions register_forward_hook even as prose; the AST guard "
            "above can no longer distinguish 'clean code' from 'nothing to scan'"
        )


# ---------------------------------------------------------------------
# D2 -- the alpha_weighted / alpha_hat alias direction
# ---------------------------------------------------------------------

def _spectral_frame(tmp_path):
    """Run one real spectral analysis and return its details DataFrame.

    Args:
        tmp_path: pytest temporary directory for the analyzer's output.

    Returns:
        pandas.DataFrame: ``results.spectral_analysis``, one row per admitted layer.
    """
    keras.utils.set_random_seed(3)
    inputs = keras.Input(shape=(20,), name="alpha_in")
    hidden = keras.layers.Dense(32, activation="relu", name="alpha_d1")(inputs)
    outputs = keras.layers.Dense(16, activation="softmax", name="alpha_out")(hidden)
    model = keras.Model(inputs=inputs, outputs=outputs, name="alpha_model")

    analyzer = ModelAnalyzer(
        models={"alpha_model": model},
        config=AnalysisConfig(
            analyze_weights=False,
            analyze_calibration=False,
            analyze_information_flow=False,
            analyze_training_dynamics=False,
            analyze_spectral=True,
            save_plots=False,
            verbose=False,
        ),
        output_dir=str(tmp_path / "spectral"),
    )
    results = analyzer.analyze(analysis_types={"spectral"})
    frame = results.spectral_analysis
    assert frame is not None and not frame.empty, "the spectral probe admitted no layer"
    return frame


class TestAlphaWeightedIsTheCanonicalName:
    """`alpha_weighted` is the WeightWatcher name; `alpha_hat` is the SETOL alias.

    Defect guarded (D2): the README said "``alpha_weighted`` | deprecated alias of
    ``alpha_hat``", inverting the direction stated by ``spectral_metrics.py:974-994``
    and by WeightWatcher's own documentation ("alpha_weighted metric, also called
    AlphaHat"). A second, larger defect sat in the same rows: the README called
    ``alpha_hat`` a "within-model layer ranking only" quantity and listed it among the
    columns "not comparable across architectures", which contradicts both.
    """

    def test_the_two_columns_are_bit_identical(self, tmp_path):
        """Measured, not asserted from the source: the alias carries the same value."""
        frame = _spectral_frame(tmp_path)
        assert len(frame) >= 2, (
            f"anti-vacuity: only {len(frame)} layer(s) admitted, so an equality over the "
            "column could hold trivially"
        )
        # The values must be non-degenerate, otherwise two all-zero columns would agree.
        assert frame["alpha_weighted"].nunique() > 1, (
            f"anti-vacuity: alpha_weighted is constant at {frame['alpha_weighted'].tolist()}"
        )
        assert (frame["alpha_weighted"] == frame["alpha_hat"]).all(), (
            "alpha_hat is documented as an alias of alpha_weighted but the columns "
            f"differ: {frame[['alpha_weighted', 'alpha_hat']].to_dict('list')}"
        )

    def test_alpha_hat_normalized_is_a_different_quantity(self, tmp_path):
        """Anti-vacuity for the arm above: not every alpha column is the same column."""
        frame = _spectral_frame(tmp_path)
        assert not (frame["alpha_weighted"] == frame["alpha_hat_normalized"]).all(), (
            "alpha_hat_normalized equals alpha_weighted, so the identity test above "
            "would pass for any pair of columns in this frame"
        )

    def test_the_readme_does_not_invert_the_alias_direction(self):
        """The README must not call the canonical WeightWatcher name a deprecated alias."""
        text = _read_readme()
        assert "deprecated alias of `alpha_hat`" not in text, (
            "README still calls alpha_weighted a deprecated alias of alpha_hat; "
            "WeightWatcher's own name for this quantity IS alpha_weighted"
        )
        row = next(
            line for line in text.splitlines() if line.startswith("| `alpha_weighted` |")
        )
        assert "canonical" in row, (
            f"the alpha_weighted row does not identify it as the canonical WW name: {row}"
        )
        alias_row = next(
            line for line in text.splitlines() if line.startswith("| `alpha_hat` |")
        )
        assert "alias of `alpha_weighted`" in alias_row, (
            f"the alpha_hat row does not state which way the alias runs: {alias_row}"
        )

    def test_the_readme_does_not_claim_alpha_hat_is_within_model_only(self):
        """The comparability caveat must be the real one, not a blanket prohibition."""
        text = _read_readme()
        assert "for **within-model** layer ranking only" not in text, (
            "README still restricts alpha_hat to within-model ranking, contradicting "
            "both spectral_metrics.py and the WeightWatcher literature"
        )
        assert "layer-averaged" in text, (
            "README does not state the real caveat: WeightWatcher's cross-model claim "
            "is for the layer-AVERAGED alpha-hat, while this column is per-layer"
        )

    def test_the_source_comment_and_the_readme_agree(self):
        """The README's direction must match the citation-grade comment it summarises."""
        source = (PACKAGE_ROOT / "spectral_metrics.py").read_text(encoding="utf-8")
        assert "CANONICAL WeightWatcher AlphaHat" in source, (
            "spectral_metrics.py no longer states which name is canonical; the README "
            "row this guard mirrors has lost its source"
        )
        assert "SETOL-paper notation" in source, (
            "spectral_metrics.py no longer records alpha_hat as the SETOL alias"
        )


# ---------------------------------------------------------------------
# D-A -- "summary_dashboard.png | always" was false for two models
# ---------------------------------------------------------------------

def _build_sequential(name: str) -> keras.Model:
    """A tiny Sequential classifier, the shape the review's smoke run used."""
    return keras.Sequential(
        [
            keras.Input(shape=(6,)),
            keras.layers.Dense(8, activation="relu", name=f"{name}_d1"),
            keras.layers.Dense(3, activation="softmax", name=f"{name}_out"),
        ],
        name=name,
    )


class TestTheSummaryDashboardSurvivesADegenerateWeightPca:
    """`summary_dashboard.png` must be written for a two-model comparison.

    Defect guarded (D-A): the README's output table said this file is produced
    "always". A two-model run produced everything else and logged
    ``ERROR base.py:_save_figure:205] Could not save figure summary_dashboard:
    Singular matrix``. Root cause, measured: the weight-PCA of exactly two models spans
    rank 1, so the model-similarity panel's PC2 values are `+/-5.09e-16`;
    `set_aspect('equal', adjustable='box')` then collapsed that axes box to height
    EXACTLY 0.0, `transAxes.inverted()` raised `LinAlgError: Singular matrix` inside
    `_update_title_position`, and `savefig` lost the WHOLE figure -- not just the panel.
    """

    def test_the_two_model_dashboard_is_written(self, tmp_path):
        keras.utils.set_random_seed(3)
        models = {"A": _build_sequential("A"), "B": _build_sequential("B")}
        rng = np.random.default_rng(0)
        x = rng.standard_normal((60, 6)).astype("float32")
        y = rng.integers(0, 3, size=60)

        output_dir = tmp_path / "dashboard"
        analyzer = ModelAnalyzer(
            models=models,
            config=AnalysisConfig(
                analyze_information_flow=False,
                analyze_spectral=False,
                analyze_training_dynamics=False,
                n_samples=60,
                save_plots=True,
                verbose=False,
            ),
            output_dir=str(output_dir),
        )
        results = analyzer.analyze(DataInput(x_data=x, y_data=y))

        # Anti-vacuity: the probe must actually hit the degenerate branch, otherwise a
        # pass proves nothing about the defect. Two models => rank-1 PCA => zero PC2.
        components = np.asarray(results.weight_pca["components"], dtype=float)
        assert components.shape[0] == 2, "the probe did not produce a two-model PCA"
        pc1_span = float(np.ptp(components[:, 0]))
        pc2_span = float(np.ptp(components[:, 1]))
        assert pc1_span > 0.0 and pc2_span / pc1_span < 1e-9, (
            "anti-vacuity: the probe's PC2 spread is not degenerate "
            f"(PC1 span {pc1_span}, PC2 span {pc2_span}), so the equal-aspect collapse "
            "this guard exists for was never reached"
        )

        dashboard = output_dir / "summary_dashboard.png"
        assert dashboard.exists(), (
            "summary_dashboard.png was not written for a two-model run. Files present: "
            f"{sorted(p.name for p in output_dir.iterdir())}"
        )

    def test_a_failed_figure_is_logged_and_not_raised(self):
        """PIN, not RED evidence: documents WHY a missing figure is silent.

        `_save_figure` swallows every exception into an ERROR log line, which is why
        the dashboard could go missing for months without a single failing run. The
        README now tells readers to check the log rather than the exit status; this
        arm keeps that sentence true.
        """
        source = (PACKAGE_ROOT / "visualizers" / "base.py").read_text(encoding="utf-8")
        assert "Could not save figure" in source, (
            "base.py no longer logs 'Could not save figure'; the README tells readers "
            "to grep the log for exactly that string"
        )

    def test_the_readme_states_the_save_failure_behaviour(self):
        text = _read_readme()
        assert "Could not save figure" in text, (
            "the README's output section does not tell the reader that a figure "
            "failure is logged rather than raised"
        )


# ---------------------------------------------------------------------
# D-B -- get_summary_statistics is keyed by analysis category
# ---------------------------------------------------------------------

class TestGetSummaryStatisticsIsKeyedByCategory:
    """`get_summary_statistics()` is not "headline numbers per model".

    Defect guarded (D-B): the README's API table called it a "dict of headline numbers
    per model". It is keyed by analysis CATEGORY, with per-model sub-dicts one level
    down; `summary['A']` measured `{}` (i.e. `KeyError`-adjacent absence), while
    `summary['calibration_summary']['A']` is the real address.
    """

    def test_the_summary_is_not_keyed_by_model_name(self, tmp_path, probe_data):
        model = _build_probe_model("cat_model")
        analyzer = ModelAnalyzer(
            models={"cat_model": model},
            config=AnalysisConfig(
                analyze_weights=False,
                analyze_information_flow=False,
                analyze_training_dynamics=False,
                analyze_spectral=False,
                n_samples=N_SAMPLES,
                save_plots=False,
                verbose=False,
            ),
            output_dir=str(tmp_path / "summary"),
        )
        analyzer.analyze(probe_data, analysis_types={"calibration"})
        summary = analyzer.get_summary_statistics()

        assert "cat_model" not in summary, (
            "get_summary_statistics is keyed by model name after all; the README's "
            "category description would then be the wrong correction"
        )
        assert summary["calibration_summary"]["cat_model"], (
            "the per-model numbers are not one level down under their category either"
        )

    def test_the_readme_lists_exactly_the_categories_returned(self, tmp_path, probe_data):
        """Both directions: no undocumented key, no documented key that is absent."""
        model = _build_probe_model("cat_model2")
        analyzer = ModelAnalyzer(
            models={"cat_model2": model},
            config=AnalysisConfig(
                analyze_weights=False,
                analyze_information_flow=False,
                analyze_training_dynamics=False,
                analyze_spectral=False,
                n_samples=N_SAMPLES,
                save_plots=False,
                verbose=False,
            ),
            output_dir=str(tmp_path / "summary2"),
        )
        analyzer.analyze(probe_data, analysis_types={"calibration"})
        returned = set(analyzer.get_summary_statistics())

        row = next(
            line
            for line in _read_readme().splitlines()
            if line.startswith("| `.get_summary_statistics()`")
        )
        documented = set(re.findall(r"`([a-z_]+)`", row))

        assert len(documented) > 5, (
            f"the README row parsed to {len(documented)} category names; the parser, "
            "not the docs, is what failed"
        )
        assert returned - documented == set(), (
            f"get_summary_statistics returns undocumented keys: {sorted(returned - documented)}"
        )
        assert documented - returned == set(), (
            f"the README documents keys the summary does not return: "
            f"{sorted(documented - returned)}"
        )


# ---------------------------------------------------------------------
# D4 -- mp_softrank and pl_pvalue caveats
# ---------------------------------------------------------------------

class TestMpSoftrankIsDocumentedAsARealMetric:
    """`mp_softrank` is a theoretical-MP-edge ratio, not the constant 1.0.

    Defect guarded (D4a): the README listed `mp_softrank` in a bare run-on list with no
    definition and no caveat, at a time when the column measured `[1.0] * 6` at shipped
    defaults. Plan step 14 rewired it to `calc_lambda_plus / lambda_max`; the probe
    values moved `1.0 -> 1.141264` and `1.0 -> 0.708324`.
    """

    def test_the_column_is_documented_with_its_definition(self):
        row = next(
            line
            for line in _read_readme().splitlines()
            if line.startswith("| `mp_softrank` |")
        )
        for required in ("Marchenko-Pastur", "not clamped"):
            assert required in row, (
                f"the mp_softrank row does not mention {required!r}: {row}"
            )

    def test_the_column_is_not_the_constant_one(self, tmp_path):
        frame = _spectral_frame(tmp_path)
        values = frame["mp_softrank"].to_numpy(dtype=float)
        assert not np.allclose(values, 1.0), (
            f"mp_softrank is back to a constant masquerading as a metric: {values.tolist()}"
        )


class TestPlPvalueSemanticsAreDocumented:
    """The README's `pl_pvalue` caveats must describe the shipped function.

    Defect guarded (D4b): the README documented only "`-1.0` means the test did not
    run" and carried no bias caveat, while `powerlaw_goodness_of_fit` diverges from
    Clauset et al. 2009 in two measurable ways, both of which push the p-value down.
    """

    def test_the_sentinel_is_the_value_the_readme_names(self):
        from dl_techniques.analyzer.constants import SPECTRAL_PVALUE_NOT_COMPUTED

        assert SPECTRAL_PVALUE_NOT_COMPUTED == -1.0, (
            "the not-computed sentinel is no longer -1.0, which is the value the "
            f"README documents: {SPECTRAL_PVALUE_NOT_COMPUTED}"
        )
        assert "`-1.0` (`SPECTRAL_PVALUE_NOT_COMPUTED`) means the test **did not run**" in _read_readme(), (
            "the README no longer explains the -1.0 sentinel"
        )

    def test_a_short_tail_returns_the_sentinel_not_zero(self):
        """Executed: the case the README calls out, run rather than asserted in prose."""
        from dl_techniques.analyzer.constants import SPECTRAL_PVALUE_NOT_COMPUTED
        from dl_techniques.analyzer.spectral_metrics import powerlaw_goodness_of_fit

        evals = np.concatenate([np.linspace(0.01, 1.0, 40), np.linspace(2.0, 6.0, 6)])
        tail = evals[evals >= 2.0]
        assert 5 <= len(tail) < 10, (
            f"anti-vacuity: the probe tail holds {len(tail)} points, so it does not "
            "exercise the short-tail exit"
        )
        pvalue = powerlaw_goodness_of_fit(
            evals, alpha=2.5, xmin=2.0, n_bootstraps=5,
            rng=np.random.default_rng(0),
        )
        assert pvalue == SPECTRAL_PVALUE_NOT_COMPUTED, (
            f"a tail too short to fit was reported as p={pvalue}, not the sentinel"
        )

    def test_the_decisive_zero_exit_the_readme_names_is_real(self):
        """The one exit that still returns a decisive 0.0, documented as such."""
        from dl_techniques.analyzer.spectral_metrics import powerlaw_goodness_of_fit

        evals = np.linspace(1.0, 10.0, 100)
        assert powerlaw_goodness_of_fit(evals, alpha=1.0, xmin=1.0, n_bootstraps=5) == 0.0, (
            "alpha <= 1.0 no longer returns 0.0; the README's caveat about the one "
            "remaining decisive-zero exit is now wrong"
        )
        assert "`alpha <= 1.0` or `xmin <= 0`" in _read_readme(), (
            "the README no longer names the remaining decisive-zero exit"
        )

    def test_the_two_documented_asymmetries_are_still_in_the_code(self):
        """The README calls these divergences; if they are ever fixed, this reddens."""
        import inspect

        from dl_techniques.analyzer import spectral_metrics

        source = inspect.getsource(spectral_metrics.powerlaw_goodness_of_fit)
        assert "fit_powerlaw(synthetic)" in source, (
            "the synthetic bootstrap fits no longer use a free xmin search; the "
            "README's first documented asymmetry is stale"
        )
        assert "return count_ge / n_bootstraps" in source, (
            "the p-value denominator is no longer n_bootstraps; the README's second "
            "documented asymmetry is stale"
        )
        assert "n_valid" in source, (
            "the function no longer counts valid draws, so the numerator/denominator "
            "asymmetry the README documents cannot be what it describes"
        )

    def test_the_readme_states_the_direction_of_the_bias(self):
        text = _read_readme()
        assert "push the p-value **downward**" in text, (
            "the README documents the two asymmetries without saying which way they "
            "bias the result, which is the part a reader needs"
        )


# ---------------------------------------------------------------------
# The spectral column census -- README vs the frame, both directions
# ---------------------------------------------------------------------

class TestTheDocumentedSpectralColumnsMatchTheFrame:
    """Every produced column is documented, and every documented column is produced.

    This is the mechanical half of plan step 37(e): the config table already has a
    two-directional guard (`test_model_analyzer.py -k readme`), and the spectral column
    census needs the same, because steps 13, 19, 21 and 18 all added columns
    (`alpha_unreliable`, `spectrum_truncated`, `trap_severity_label`, ...) that the
    README's run-on list never learned about.
    """

    @staticmethod
    def _produced_columns(tmp_path, **config_overrides) -> set:
        keras.utils.set_random_seed(3)
        inputs = keras.Input(shape=(20,), name="census_in")
        hidden = keras.layers.Dense(32, activation="relu", name="census_d1")(inputs)
        outputs = keras.layers.Dense(16, activation="softmax", name="census_out")(hidden)
        model = keras.Model(inputs=inputs, outputs=outputs, name="census")
        defaults = dict(
            analyze_weights=False,
            analyze_calibration=False,
            analyze_information_flow=False,
            analyze_training_dynamics=False,
            analyze_spectral=True,
            save_plots=False,
            verbose=False,
        )
        defaults.update(config_overrides)
        analyzer = ModelAnalyzer(
            models={"census": model},
            config=AnalysisConfig(**defaults),
            output_dir=str(tmp_path),
        )
        frame = analyzer.analyze(analysis_types={"spectral"}).spectral_analysis
        assert frame is not None and not frame.empty, "the census probe admitted no layer"
        return set(frame.columns)

    def test_every_produced_column_is_named_in_the_readme(self, tmp_path):
        produced = self._produced_columns(tmp_path / "default")
        produced |= self._produced_columns(
            tmp_path / "randomized",
            spectral_randomize=True,
            spectral_n_randomizations=2,
        )
        assert len(produced) > 40, (
            f"anti-vacuity: the census produced only {len(produced)} columns"
        )
        text = _read_readme()
        undocumented = sorted(name for name in produced if f"`{name}`" not in text)
        assert not undocumented, (
            f"the spectral frame carries columns the README never names: {undocumented}"
        )

    def test_every_column_the_readme_tabulates_exists(self, tmp_path):
        produced = self._produced_columns(tmp_path / "default2")
        produced |= self._produced_columns(
            tmp_path / "randomized2",
            spectral_randomize=True,
            spectral_n_randomizations=2,
        )
        section = _readme_section(
            "Spectral (`results.spectral_analysis`, a `pandas.DataFrame`, one row per layer)"
        )
        tabulated = {
            match.group(1)
            for line in section.splitlines()
            for match in [re.match(r"^\| `([a-z_0-9]+)` \|", line)]
            if match
        }
        assert len(tabulated) > 8, (
            f"anti-vacuity: only {len(tabulated)} table rows parsed out of the spectral "
            "section; the parser, not the docs, is what failed"
        )
        missing = sorted(tabulated - produced)
        assert not missing, (
            f"the README tabulates spectral columns the frame does not carry: {missing}"
        )


# ---------------------------------------------------------------------
# D3 -- the __init__ docstring advertised two nonexistent names
# ---------------------------------------------------------------------

def _fenced_python_blocks(text: str) -> list[str]:
    """Return every fenced ```python block in ``text``.

    Args:
        text: Markdown or a docstring containing fenced code blocks.

    Returns:
        list[str]: The block bodies, in order of appearance.
    """
    import textwrap

    return [
        textwrap.dedent(block)
        for block in re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)
    ]


def _check_example_block(block: str, seeds: dict) -> list[str]:
    """Resolve every import, attribute and call signature used in one example.

    This is the executable half of "the docs are true": a grep proves a line exists,
    while importing the module, ``hasattr``-ing the attribute and binding the call's
    keywords against ``inspect.signature`` proves the line would run. It is shared by
    the ``__init__`` docstring guard and the README quick-start guard, so the two
    cannot drift into checking different things.

    Args:
        block: One example's source text. Undefined free names are ignored, since an
            example legitimately refers to the reader's own ``models`` / ``x_test``.
        seeds: Names already known to the caller (e.g. classes it imported itself),
            mapped to the object they denote.

    Returns:
        list[str]: Human-readable problems found; empty means the example resolves.
    """
    import importlib
    import inspect

    problems: list[str] = []
    known = dict(seeds)

    tree = ast.parse(block)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            try:
                module = importlib.import_module(node.module)
            except ImportError as exc:
                problems.append(f"`from {node.module} import ...` fails: {exc}")
                continue
            for alias in node.names:
                if not hasattr(module, alias.name):
                    problems.append(f"{node.module} has no attribute {alias.name!r}")
                else:
                    known[alias.asname or alias.name] = getattr(module, alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                try:
                    known[alias.asname or alias.name] = importlib.import_module(alias.name)
                except ImportError as exc:
                    problems.append(f"`import {alias.name}` fails: {exc}")

    # Bind simple `name = Known(...)` / `name = Known.method(...)` assignments so that
    # later attribute use on the result is checkable.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value = node.value
            if isinstance(target, ast.Name) and isinstance(value, ast.Call):
                func = value.func
                if isinstance(func, ast.Name) and func.id in known:
                    # `data = DataInput(...)` -- the value IS an instance of the class.
                    known[target.id] = known[func.id]
                elif (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id in known
                ):
                    # `results = analyzer.analyze(...)` -- bind the RETURN type, not the
                    # owner. Binding the owner made `results.spectral_analysis` resolve
                    # against ModelAnalyzer and produced a false positive.
                    method = getattr(known[func.value.id], func.attr, None)
                    returned = getattr(
                        inspect.signature(method).return_annotation, "__name__", None
                    ) if callable(method) else None
                    if returned:
                        resolved = getattr(
                            importlib.import_module("dl_techniques.analyzer.data_types"),
                            returned,
                            None,
                        )
                        if resolved is not None:
                            known[target.id] = resolved

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            owner = known.get(node.value.id)
            if owner is not None and not hasattr(owner, node.attr):
                problems.append(
                    f"{node.value.id} ({owner!r}) has no attribute {node.attr!r}"
                )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        target = None
        if isinstance(func, ast.Name):
            target = known.get(func.id)
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            owner = known.get(func.value.id)
            if owner is not None:
                target = getattr(owner, func.attr, None)
        if target is None or not callable(target):
            continue
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            continue
        keywords = [kw.arg for kw in node.keywords if kw.arg is not None]
        try:
            signature.bind_partial(**{name: None for name in keywords})
        except TypeError as exc:
            problems.append(f"{target!r} does not accept {keywords}: {exc}")

    return problems


class TestTheModuleDocstringResolves:
    """Every name the package docstring shows must exist at runtime.

    Defect guarded (D3): the docstring's multi-input example read
    ``from dl_techniques.utils.analyzer import DataInput`` (a module that does not
    exist: measured ``ModuleNotFoundError``) and called
    ``DataInput.from_multi_input([input1, input2], targets)``
    (``hasattr`` measured ``False``). Both sat above the real imports, so
    ``import dl_techniques.analyzer`` succeeded and nothing ever reddened. The repo has
    a matching lesson: a doc-repair pass documented four unexported names as importable
    and only a mechanical ``hasattr`` sweep caught it -- eyeballing passed twice.
    """

    def test_every_init_docstring_example_resolves(self):
        blocks = _fenced_python_blocks(analyzer_pkg.__doc__ or "")
        assert blocks, "the package docstring no longer carries a python example"

        problems: list[str] = []
        for block in blocks:
            problems.extend(_check_example_block(block, seeds={}))
        assert not problems, (
            "the dl_techniques.analyzer module docstring shows names that do not "
            f"resolve: {problems}"
        )

    def test_the_docstring_no_longer_names_the_phantom_module(self):
        doc = analyzer_pkg.__doc__ or ""
        example_text = "\n".join(_fenced_python_blocks(doc))
        assert "dl_techniques.utils.analyzer import" not in example_text, (
            "the docstring example still imports from dl_techniques.utils.analyzer, "
            "which does not exist"
        )
        assert "DataInput.from_multi_input" not in example_text, (
            "the docstring example still calls DataInput.from_multi_input, which "
            "DataInput does not define"
        )

    def test_the_two_phantoms_really_are_phantoms(self):
        """Anti-vacuity: the resolver above must be able to see these two failures."""
        import importlib

        from dl_techniques.analyzer.data_types import DataInput

        assert not hasattr(DataInput, "from_multi_input"), (
            "DataInput now HAS from_multi_input, so the docstring's old example was "
            "not a phantom after all and this guard is measuring nothing"
        )
        with pytest.raises(ImportError):
            importlib.import_module("dl_techniques.utils.analyzer")

    def test_the_resolver_catches_the_original_defect(self):
        """Anti-vacuity: feed the resolver the ORIGINAL example and require it to red.

        A checker that reports "no problems" on everything would pass
        `test_every_init_docstring_example_resolves` for the wrong reason.
        """
        original = (
            "from dl_techniques.utils.analyzer import DataInput\n"
            "data = DataInput.from_multi_input([input1, input2], targets)\n"
        )
        problems = _check_example_block(original, seeds={})
        assert problems, "the example resolver did not flag the known-bad example"

        # And it must see through one level of indirection, which is where a naive
        # resolver goes vacuous: `results` is the RETURN of a method, not the analyzer.
        indirect = (
            "from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig, DataInput\n"
            "analyzer = ModelAnalyzer(models=models, config=AnalysisConfig(no_such_knob=1))\n"
            "results = analyzer.analyze(DataInput(x_data=x, y_data=y))\n"
            "print(results.no_such_field)\n"
        )
        found = _check_example_block(indirect, seeds={})
        assert any("no_such_field" in problem for problem in found), (
            f"the resolver did not follow analyze()'s return type: {found}"
        )
        assert any("no_such_knob" in problem for problem in found), (
            f"the resolver did not check call keywords against the signature: {found}"
        )

    def test_the_docstring_shows_the_real_constructors(self):
        doc = analyzer_pkg.__doc__ or ""
        for constructor in ("DataInput.from_tuple", "DataInput.from_object"):
            assert constructor in doc, (
                f"the docstring does not show {constructor}, which is one of the two "
                "helpers DataInput actually defines"
            )

    def test_every_exported_name_resolves(self):
        """The ``__all__`` sweep: every advertised export must be importable."""
        missing = [name for name in analyzer_pkg.__all__ if not hasattr(analyzer_pkg, name)]
        assert not missing, f"__all__ advertises names the package does not expose: {missing}"

    def test_every_readme_quick_start_example_resolves(self):
        """The same machinery over the README's own demonstrated API."""
        blocks = _fenced_python_blocks(_read_readme())
        assert blocks, "the README no longer carries a python example"

        problems: list[str] = []
        for block in blocks:
            problems.extend(_check_example_block(block, seeds={}))
        assert not problems, (
            f"the README shows API usage that does not resolve: {problems}"
        )


# ---------------------------------------------------------------------
# S4 -- CORRELATION_TRAPS.md vs the shipped detector
# ---------------------------------------------------------------------

TRAPS_PATH = PACKAGE_ROOT / "CORRELATION_TRAPS.md"


def _wishart_spectrum(seed: int = 3, rows: int = 200, cols: int = 50) -> np.ndarray:
    """Eigenvalues of ``WᵀW / rows`` for a Gaussian ``W``, descending.

    This is the probe the document's §0 quotes its numbers from.

    Args:
        seed: RNG seed for the Gaussian draw.
        rows: Number of rows of ``W`` (the larger dimension).
        cols: Number of columns of ``W``.

    Returns:
        numpy.ndarray: The eigenvalues, sorted descending.
    """
    rng = np.random.default_rng(seed)
    weights = rng.standard_normal((rows, cols))
    correlation = (weights.T @ weights) / rows
    return np.sort(np.linalg.eigvalsh(correlation))[::-1]


class TestCorrelationTrapsDocMatchesTheImplementation:
    """`CORRELATION_TRAPS.md` must not contradict the detector it describes.

    Defect guarded (S4): the document's `c_TW ≈ 2.5`, its `Δ_TW = c_TW·σ²·N^(-1/3)`
    offset and its whole-spectrum `sigma_sq` all disagree with what ships, and a grep
    for `D-00` / `diverge` / `shipped` over the document returned ZERO hits -- nothing
    told a reader which side to believe. A `Divergences from the shipped
    implementation` section (§0) now states each difference with an executed number.
    """

    def test_the_divergence_section_exists_and_is_referenced(self):
        text = TRAPS_PATH.read_text(encoding="utf-8")
        assert "## 0. Divergences from the shipped implementation" in text, (
            "CORRELATION_TRAPS.md has no divergences section; a reader still cannot "
            "tell which of the document and the code is authoritative"
        )
        # Each contradicting site must point at it, not just the top of the file.
        assert text.count("See §0") >= 4, (
            "the contradicting formulas do not point at the divergence section; a "
            f"reader landing mid-document would still be misled (found {text.count('See §0')})"
        )

    def test_the_documented_c_tw_matches_the_constant(self):
        from dl_techniques.analyzer.constants import SPECTRAL_TW_SAFETY_FACTOR

        assert SPECTRAL_TW_SAFETY_FACTOR == 1.0, (
            f"the shipped Tracy-Widom factor moved to {SPECTRAL_TW_SAFETY_FACTOR}; the "
            "document's divergence table quotes 1.0"
        )
        text = TRAPS_PATH.read_text(encoding="utf-8")
        assert "`SPECTRAL_TW_SAFETY_FACTOR = 1.0`" in text, (
            "the divergence table no longer names the shipped c_TW value"
        )

    def test_the_two_mp_edge_spellings_agree_under_their_own_conventions(self):
        """The §0 claim that the MP-edge 'contradiction' is a Q-convention artefact.

        This is a REFUTATION of the review's S4 sub-claim, so it is asserted rather
        than repeated: the document's `σ²(1 ± √Q)²` with its own `Q = cols/rows` and
        the code's `σ²(1 ± 1/√Q)²` with `Q = larger/smaller` are the same number.
        """
        from dl_techniques.analyzer.spectral_metrics import calc_mp_edges

        evals = _wishart_spectrum()
        sigma_sq = float(np.mean(evals))
        q_doc = 50 / 200
        q_code = 200 / 50

        doc_plus = sigma_sq * (1.0 + np.sqrt(q_doc)) ** 2
        doc_minus = sigma_sq * (1.0 - np.sqrt(q_doc)) ** 2
        code_minus, code_plus = calc_mp_edges(sigma_sq, q_code)

        assert doc_plus == pytest.approx(code_plus, rel=1e-12), (
            f"doc edge {doc_plus} != code edge {code_plus}; §0's 'same formula, "
            "opposite conventions' claim is wrong"
        )
        assert doc_minus == pytest.approx(code_minus, rel=1e-12)

        # Anti-vacuity: the mistaken substitution really does inflate by Q, which is
        # the discrepancy the review reported. If this were also equal, the guard
        # above would be insensitive to the convention it exists to pin.
        mistaken = sigma_sq * (1.0 + np.sqrt(q_code)) ** 2
        assert mistaken == pytest.approx(code_plus * q_code, rel=1e-9), (
            f"substituting the code's Q into the doc's spelling gave {mistaken}, not "
            f"{code_plus * q_code}"
        )

    def test_the_documented_threshold_numbers_are_the_measured_ones(self):
        """Every number §0 quotes for the probe is recomputed here."""
        from dl_techniques.analyzer.spectral_metrics import (
            detect_correlation_trap,
            estimate_bulk_variance,
        )

        evals = _wishart_spectrum(seed=3)
        sigma_all = float(np.mean(evals))
        sigma_bulk = estimate_bulk_variance(evals, 200 / 50)
        result = detect_correlation_trap(evals, 200, 50)
        doc_threshold = (
            sigma_all * (1.0 + np.sqrt(50 / 200)) ** 2
            + 2.5 * sigma_all * (50 ** (-1.0 / 3.0))
        )
        text = TRAPS_PATH.read_text(encoding="utf-8")

        # The shipped offset, quoted in the table beside the doc's own.
        shipped_delta = result["trap_threshold"] - result["mp_lambda_plus"]
        assert f"{shipped_delta:.6f}" in text, (
            f"CORRELATION_TRAPS.md §0 no longer quotes the shipped Tracy-Widom "
            f"offset ({shipped_delta:.6f})"
        )
        for label, value in (
            ("sigma over all eigenvalues", sigma_all),
            ("bulk sigma", sigma_bulk),
            ("shipped threshold", result["trap_threshold"]),
            ("doc threshold", doc_threshold),
        ):
            assert f"{value:.6f}" in text, (
                f"CORRELATION_TRAPS.md §0 no longer quotes the measured {label} "
                f"({value:.6f}); the document's numbers have gone stale"
            )

        # The divergence must be real, not a rounding difference.
        assert doc_threshold > result["trap_threshold"] * 1.1, (
            f"the two thresholds have converged ({doc_threshold} vs "
            f"{result['trap_threshold']}); §0 calls this a real divergence"
        )

    def test_the_shipped_sigma_excludes_spikes(self):
        """The `D-017` divergence, executed rather than asserted in prose."""
        from dl_techniques.analyzer.spectral_metrics import estimate_bulk_variance

        evals = _wishart_spectrum(seed=3)
        spiked = evals.copy()
        spiked[0] = evals[0] * 20.0

        doc_estimate = float(np.mean(spiked))
        shipped = estimate_bulk_variance(spiked, 200 / 50)
        clean = estimate_bulk_variance(evals, 200 / 50)

        assert doc_estimate > 1.5 * float(np.mean(evals)), (
            "anti-vacuity: the injected spike does not move the whole-spectrum mean, "
            "so this probe cannot show the difference the document describes"
        )
        assert shipped == pytest.approx(clean, rel=0.05), (
            f"the shipped bulk variance moved with the spike ({shipped} vs {clean}); "
            "the document's divergence entry describes a fix that is no longer there"
        )

    def test_the_scale_equivariance_the_doc_now_claims_actually_holds(self):
        """§0 used to record the scale dependence as known-open; it now claims a fix.

        The claim is executed rather than trusted. This replaces the guard that
        pinned the DEFECT: that guard asserted `headroom[1.0] > 10 *
        headroom[100.0]` and a `False -> True` verdict flip, both of which were
        descriptions of the bug, so it necessarily reddened when the bug was fixed.
        """
        from dl_techniques.analyzer.spectral_metrics import detect_correlation_trap

        for seed in (3, 7, 0):
            evals = _wishart_spectrum(seed=seed)
            verdicts = {}
            headroom = {}
            for scale in (1e-4, 1.0, 100.0, 1e4):
                result = detect_correlation_trap(evals * scale, 200, 50)
                verdicts[scale] = result["has_trap"]
                headroom[scale] = (
                    result["trap_threshold"] - result["mp_lambda_plus"]
                ) / result["mp_lambda_plus"]

            assert len(set(verdicts.values())) == 1, (
                f"seed {seed}: a pure rescale still moves the verdict: {verdicts}")
            for scale, value in headroom.items():
                assert value == pytest.approx(headroom[1.0], rel=1e-12), (
                    f"seed {seed}: relative headroom {value!r} at s={scale:g} against "
                    f"{headroom[1.0]!r} at s=1"
                )

        text = TRAPS_PATH.read_text(encoding="utf-8")
        assert "The threshold is now scale-equivariant" in text, (
            "CORRELATION_TRAPS.md no longer records the scale-equivariance fix"
        )
        assert "### Known-open behaviour (recorded, not fixed)" not in text, (
            "CORRELATION_TRAPS.md still files the scale dependence under "
            "known-open behaviour; that section described the pre-fix threshold"
        )

    def test_the_divergence_from_weightwatcher_is_labelled_as_one(self):
        """The offset is a bug fix vs SETOL AND a numeric divergence from upstream.

        A previous plan established that mislabelling a divergence as parity is
        itself a defect; the in-code claim `default 1.0 = WW-exact` was exactly
        that. The document must now carry the upstream source that explains why.
        """
        text = TRAPS_PATH.read_text(encoding="utf-8")

        assert "Divergence from WeightWatcher, stated as one" in text, (
            "CORRELATION_TRAPS.md does not label the Tracy-Widom offset as a "
            "divergence from WeightWatcher"
        )
        for fragment in (
            "identify_trap_mode_indices",
            "Wscale = np.sqrt(to_plot.shape[0])/Wnorm",
            "to_plot = (Wscale*Wscale)*to_plot",
            "threshold = bulk_max_TW / (Wscale * Wscale)",
        ):
            assert fragment in text, (
                f"CORRELATION_TRAPS.md no longer quotes {fragment!r} from the "
                f"upstream source; the divergence claim becomes unverifiable prose"
            )

        from dl_techniques.analyzer.spectral_metrics import detect_correlation_trap

        # The live docstring must not repeat the refuted parity claim.
        assert "WW-exact" not in (detect_correlation_trap.__doc__ or ""), (
            "detect_correlation_trap's docstring still claims WeightWatcher-"
            "exactness for the Tracy-Widom threshold; the port dropped upstream's "
            "normalization, so the claim is false"
        )

        # The legacy anchor that carried the claim is SUPERSEDED IN PLACE, never
        # deleted: `plans/plan_2026-06-03_bc986e52/` no longer exists, so that
        # comment is the sole surviving record of the decision being overturned.
        source = (PACKAGE_ROOT / "spectral_metrics.py").read_text(encoding="utf-8")
        assert "DECISION plan_2026-06-03_bc986e52/D-003" in source, (
            "the legacy D-003 anchor was deleted rather than superseded; its plan "
            "directory is gone, so deleting the comment destroys the only record"
        )
        legacy = source.index("DECISION plan_2026-06-03_bc986e52/D-003")
        appendix = source.index("SUPERSEDED by plan-2026-09-02T041737-e85f2027/D-005")
        assert appendix > legacy, (
            "the supersession appendix does not follow the decision it supersedes"
        )
        assert "DECISION plan_2026-06-03_bc986e52/D-008" in source, (
            "D-008 (the N>=20 KS-argmin selection rule) was superseded too; only "
            "D-003 was authorized"
        )


# ---------------------------------------------------------------------
# The doc pointers themselves
# ---------------------------------------------------------------------

class TestEveryQuotedPytestSelectorMatchesSomething:
    """A doc that cites a test command must cite one that selects a test.

    This module's own first draft shipped two dead pointers -- `-k alpha_weighted` and
    `-k correlation_traps` -- because ``-k`` matches node-id substrings CASE
    SENSITIVELY and the classes are `TestAlphaWeightedIsTheCanonicalName` and
    `TestCorrelationTrapsDocMatchesTheImplementation`. A citation that selects nothing
    is exactly the "documentation pointers rot silently" failure this package has a
    recorded history of, and it is invisible to every other guard here.
    """

    DOC_SOURCES = ("README.md", "CORRELATION_TRAPS.md", "__init__.py")

    # `<something>test_<file>.py ... -k <selector>`: the selector is checked against the
    # FILE the citation names, not against the whole directory. Checking the directory
    # is what made the first spelling of this guard vacuous -- `-k alpha_weighted` does
    # select a test, just one in `test_spectral_metrics.py`, which is not the guard the
    # README was pointing the reader at.
    CITATION = re.compile(r"(test_[a-z_0-9]+\.py)[^`\n]*?-k ([A-Za-z_][A-Za-z_0-9]*)")

    @staticmethod
    def _names_in(path: Path) -> set:
        """Every class and test-function name defined in one test module."""
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        }

    @classmethod
    def audit_citations(cls, text: str, label: str) -> tuple:
        """Report every dead `test_*.py ... -k <selector>` citation in one text.

        Shared by the doc guard below and by
        `TestThePlansOwnPytestSelectorsAreAlive`, so the parse rule and the
        "matches nothing" rule exist ONCE. Writing a second copy for the plan
        would be the lockstep-invariant smell this package keeps recording.

        Args:
            text: The full text of one document to scan.
            label: How to name that document in a failure message.

        Returns:
            tuple: ``(dead, checked)`` - a list of human-readable failures and the
            number of citations whose named module existed and was therefore
            actually checked. ``checked`` is the anti-vacuity counter: a citation
            regex that stopped matching reports zero dead AND zero checked.
        """
        test_dir = Path(__file__).parent
        dead, checked = [], 0
        for module_name, selector in cls.CITATION.findall(text):
            module_path = test_dir / module_name
            if not module_path.exists():
                dead.append(f"{label}: cites missing module {module_name}")
                continue
            checked += 1
            if not any(selector in name for name in cls._names_in(module_path)):
                dead.append(f"{label}: `{module_name} -k {selector}` matches nothing")
        return dead, checked

    def test_every_documented_selector_selects_at_least_one_test(self):
        dead = []
        checked = 0
        for source in self.DOC_SOURCES:
            text = (PACKAGE_ROOT / source).read_text(encoding="utf-8")
            source_dead, source_checked = self.audit_citations(text, source)
            dead.extend(source_dead)
            checked += source_checked
        assert checked >= 3, (
            f"only {checked} documented pytest citations were parsed; the citation "
            "regex, not the docs, is what failed"
        )
        assert not dead, f"these documented pytest citations are dead: {dead}"

    def test_the_selector_check_would_notice_a_dead_pointer(self):
        """Anti-vacuity: the citation regex and the name parser both do their job.

        MEASURED while writing this guard: with the README's pointer reverted to
        `-k alpha_weighted`, the check reports
        ``['README.md: `test_analyzer_docs.py -k alpha_weighted` matches nothing']``.
        """
        names = self._names_in(Path(__file__))
        assert len(names) > 20, f"only {len(names)} names parsed out of this module"
        assert not any("no_such_selector_token" in name for name in names)
        assert self.CITATION.findall(
            "see `tests/test_analyzer/test_analyzer_docs.py -k CorrelationTraps`"
        ) == [("test_analyzer_docs.py", "CorrelationTraps")], (
            "the citation regex no longer parses the form the docs actually use"
        )


class TestThePlansOwnPytestSelectorsAreAlive:
    """A plan's Verification Strategy is a GATE; a dead `-k` runs nothing and reads green.

    MEASURED on `plan-2026-09-01T225724-e79ad4bd` before this guard existed: three of
    the seventeen per-criterion commands selected ZERO tests -
    `test_spectral_metrics.py -k identity` (126 deselected, 0 run),
    `test_analyzer_config.py -k readme` (that module was never created), and
    `test_analyzer_docs.py -k correlation_traps` (39 deselected, 0 run; `-k` is CASE
    SENSITIVE and the class is `TestCorrelationTrapsDocMatchesTheImplementation`).
    Running the gate verbatim would have reported success while executing nothing.

    Scope: only plan files that name `tests/test_analyzer` are read, and the parse /
    dead-selector rules are `TestEveryQuotedPytestSelectorMatchesSomething`'s -
    reused via `audit_citations`, not re-implemented. `plans/` is gitignored and its
    directories are deleted when a plan closes, so this guard SKIPS when no such plan
    is present: it is live exactly while the artifact it points at exists.
    """

    @staticmethod
    def _plan_files() -> list:
        repo_root = Path(__file__).resolve().parents[2]
        plans = sorted((repo_root / "plans").glob("plan-*/plan.md"))
        return [
            p for p in plans
            if "tests/test_analyzer" in p.read_text(encoding="utf-8", errors="replace")
        ]

    def test_every_plan_cited_selector_selects_at_least_one_test(self):
        plan_files = self._plan_files()
        if not plan_files:
            pytest.skip("no open plan references tests/test_analyzer")

        dead, checked = [], 0
        for plan_path in plan_files:
            text = plan_path.read_text(encoding="utf-8", errors="replace")
            plan_dead, plan_checked = (
                TestEveryQuotedPytestSelectorMatchesSomething.audit_citations(
                    text, plan_path.parent.name)
            )
            dead.extend(plan_dead)
            checked += plan_checked
        assert checked >= 3, (
            f"only {checked} pytest citations were parsed out of "
            f"{[p.parent.name for p in plan_files]}; the citation regex, not the "
            "plan, is what failed"
        )
        assert not dead, (
            f"these Verification-Strategy commands select NOTHING: {dead}"
        )

    def test_the_plan_scan_uses_the_shared_audit_and_would_see_a_dead_selector(self):
        """Anti-vacuity: one helper, and it still reports a dead pointer."""
        dead, checked = (
            TestEveryQuotedPytestSelectorMatchesSomething.audit_citations(
                "`tests/test_analyzer/test_analyzer_docs.py -k correlation_traps`",
                "probe")
        )
        assert checked == 1, f"the probe citation was not parsed at all ({checked})"
        assert dead == [
            "probe: `test_analyzer_docs.py -k correlation_traps` matches nothing"
        ], f"the shared audit no longer detects the measured dead selector: {dead}"


class TestTheBrierRedefinitionIsDocumentedWhereAReaderLooks:
    """`compute_brier_score_decomposition`'s `brier_score` key changed MEANING.

    D-014 moved every term of the Murphy decomposition into the top-1 correctness
    outcome space, which silently redefined the returned `brier_score` from the
    MULTICLASS Brier score to the BINARY top-1 one. The function has no in-library
    caller, so no other guard here can see the change, and a user diffing two runs
    reads the same key name for a different quantity.

    This guard checks that the redefinition is stated where a reader will find it
    AND that the statement is TRUE, by re-deriving both quantities.
    """

    def test_the_two_brier_scores_really_are_different_quantities(self):
        """Anti-vacuity FIRST: if they agreed, the doc note would be noise."""
        from dl_techniques.analyzer.calibration_metrics import (
            compute_brier_score,
            compute_brier_score_decomposition,
        )

        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
        y_prob = np.array([
            [0.8, 0.2], [0.3, 0.7], [0.1, 0.9], [0.9, 0.1],
            [0.4, 0.6], [0.7, 0.3], [0.2, 0.8], [0.6, 0.4],
        ])
        decomp = compute_brier_score_decomposition(y_true, y_prob, n_bins=4)
        multiclass = float(compute_brier_score(np.eye(2)[y_true], y_prob))
        raw_top1 = decomp["brier_score"] + decomp["binning_residual"]

        assert multiclass == pytest.approx(0.15, abs=1e-12), multiclass
        assert raw_top1 == pytest.approx(0.075, abs=1e-12), raw_top1
        assert abs(multiclass - raw_top1) > 0.05, (
            "the two Brier scores agree on this probe, so the documented "
            f"redefinition would be unobservable ({multiclass} vs {raw_top1})"
        )
        # And the identity the redefinition bought must actually hold.
        assert (decomp["reliability"] - decomp["resolution"]
                + decomp["uncertainty"]) == pytest.approx(
                    decomp["brier_score"], abs=1e-12)

    def test_the_readme_states_the_decomposition_brier_redefinition(self):
        text = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
        assert "compute_brier_score_decomposition" in text, (
            "the README never names the function whose key changed meaning"
        )
        block = text[text.index("compute_brier_score_decomposition"):][:1400]
        for needle in ("changed meaning", "top-1", "compute_brier_score"):
            assert needle in block, (
                f"the README's brier redefinition note does not mention {needle!r}"
            )

    def test_the_docstring_says_which_brier_score_it_returns(self):
        from dl_techniques.analyzer.calibration_metrics import (
            compute_brier_score_decomposition,
        )

        doc = compute_brier_score_decomposition.__doc__ or ""
        assert "CHANGED MEANING" in doc, (
            "the docstring does not tell a caller that the key was redefined"
        )
        assert "MULTICLASS" in doc and "compute_brier_score" in doc, (
            "the docstring does not name the quantity it used to be, nor where "
            "to get it now"
        )


class TestTheConstantWeightPcaAbortIsPinnedAndDocumented:
    """PIN, not a fix: a constant weight tensor aborts the WHOLE `analyze()` call.

    `scipy.stats.skew`/`kurtosis` return `NaN` on a constant vector
    (`analyzers/weight_analyzer.py:149`); the `NaN` reaches the concatenated PCA
    feature matrix; and `_compute_weight_pca`'s `try` catches only
    `np.linalg.LinAlgError`, so sklearn's `ValueError: Input X contains NaN`
    escapes and every other analysis is lost with it.

    PRE-EXISTING: `weight_analyzer.py` has no commit in the 2026-09-02 analyzer
    repair plan's range, so this is not a regression that plan introduced. It is
    pinned here so that (a) the behaviour is executable rather than a prose
    memory, and (b) whoever fixes it is TOLD by a red test to update the README
    caveat in the same commit.
    """

    @staticmethod
    def _model(name: str, zero_kernel: bool):
        keras.utils.set_random_seed(3)
        inputs = keras.Input(shape=(6,), name=f"{name}_in")
        x = keras.layers.Dense(8, activation="relu", name=f"{name}_d1")(inputs)
        outputs = keras.layers.Dense(
            3, activation="softmax", name=f"{name}_out")(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name=name)
        if zero_kernel:
            layer = model.get_layer(f"{name}_d1")
            weights = layer.get_weights()
            weights[0] = np.zeros_like(weights[0])
            layer.set_weights(weights)
        return model

    def _run(self, tmp_path, tag, zero_kernel):
        rng = np.random.default_rng(0)
        data = DataInput(
            x_data=rng.standard_normal((40, 6)).astype("float32"),
            y_data=rng.integers(0, 3, 40),
        )
        config = AnalysisConfig(
            analyze_weights=True, analyze_calibration=False,
            analyze_information_flow=False, analyze_training_dynamics=False,
            analyze_spectral=False, n_samples=40, save_plots=False, verbose=False,
        )
        # Two models: `_compute_weight_pca` needs >= 2 feature rows to run at all.
        analyzer = ModelAnalyzer(
            models={
                "z": self._model("z", zero_kernel),
                "z2": self._model("z2", False),
            },
            config=config,
            output_dir=str(tmp_path / tag),
        )
        return analyzer.analyze(data, analysis_types={"weights"})

    def test_a_healthy_pair_completes(self, tmp_path):
        """Anti-vacuity: the same call SUCCEEDS without the constant kernel."""
        results = self._run(tmp_path, "healthy", zero_kernel=False)
        assert results.weight_stats, "the control run produced no weight stats"

    def test_a_constant_kernel_still_aborts_the_whole_analysis(self, tmp_path):
        with pytest.raises(ValueError, match="Input X contains NaN"):
            self._run(tmp_path, "zeroed", zero_kernel=True)

    def test_the_readme_records_the_open_defect(self):
        text = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
        assert "Input X contains NaN" in text, (
            "the README no longer records the constant-weight PCA abort; if it "
            "was FIXED, delete this pin in the same commit"
        )
