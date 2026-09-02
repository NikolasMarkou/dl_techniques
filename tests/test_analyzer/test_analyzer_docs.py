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
