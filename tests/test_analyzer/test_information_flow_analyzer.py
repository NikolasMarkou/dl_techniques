"""
Tests for the InformationFlowAnalyzer.

Covers: activation capture on FUNCTIONAL and SUBCLASSED models, the seven per-layer
keys the InformationFlowVisualizer reads, depth (insertion) ordering of the
per-layer dict, and non-mutation of the analyzed model.

These are guards for a real defect: the analyzer used to capture activations with
the PyTorch-only ``layer.register_forward_hook``, which raises ``AttributeError``
on every Keras layer, leaving ``results.information_flow`` permanently empty.
"""

import keras
import numpy as np
import pytest

from dl_techniques.analyzer.config import AnalysisConfig
from dl_techniques.analyzer.data_types import AnalysisResults, DataInput
from dl_techniques.analyzer.utils import recursively_get_layers
from dl_techniques.analyzer.analyzers.information_flow_analyzer import (
    InformationFlowAnalyzer,
)

# The keys InformationFlowVisualizer indexes out of each per-layer entry.
# `information_flow_visualizer.py:161` does `analysis['mean_activation']` with no
# default, so a missing key is a KeyError at plot time, not a silent gap.
VISUALIZER_KEYS = (
    "layer_type",
    "output_shape",
    "mean_activation",
    "std_activation",
    "sparsity",
    "positive_ratio",
    "effective_rank",
)

N_SAMPLES = 32
N_FEATURES = 6


# ---------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------

def build_functional_model() -> keras.Model:
    """A minimal functional model: Input -> Dense(8) -> Dense(4) -> Dense(2)."""
    inputs = keras.Input(shape=(N_FEATURES,), name="flow_in")
    x = keras.layers.Dense(8, activation="relu", name="flow_d1")(inputs)
    x = keras.layers.Dense(4, activation="relu", name="flow_d2")(x)
    outputs = keras.layers.Dense(2, name="flow_d3")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="functional_probe")


class SubclassedProbe(keras.Model):
    """A subclassed model holding its sublayers as plain Python attributes.

    A functional sub-model slice cannot serve this case: the sublayers have no
    ``.output`` KerasTensor and the model has no ``.input``.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense_1 = keras.layers.Dense(8, name="sub_d1")
        self.relu_1 = keras.layers.ReLU(name="sub_r1")
        self.dense_2 = keras.layers.Dense(4, name="sub_d2")
        self.relu_2 = keras.layers.ReLU(name="sub_r2")
        self.dense_3 = keras.layers.Dense(2, name="sub_d3")

    def call(self, inputs, training=None):
        x = self.relu_1(self.dense_1(inputs))
        x = self.relu_2(self.dense_2(x))
        return self.dense_3(x)


def build_subclassed_model() -> keras.Model:
    """Build and materialize the subclassed probe by calling it once."""
    model = SubclassedProbe(name="subclassed_probe")
    model(make_x())
    return model


def make_x() -> np.ndarray:
    rng = np.random.default_rng(1234)
    return rng.standard_normal((N_SAMPLES, N_FEATURES)).astype("float32")


def run_analysis(model: keras.Model):
    """Drive InformationFlowAnalyzer directly, bypassing ModelAnalyzer."""
    x = make_x()
    y = np.zeros((N_SAMPLES, 2), dtype="float32")
    config = AnalysisConfig(n_samples=N_SAMPLES)
    analyzer = InformationFlowAnalyzer({model.name: model}, config)
    results = AnalysisResults()
    analyzer.analyze(results, DataInput(x_data=x, y_data=y))
    return analyzer, results


def expected_layer_order(analyzer: InformationFlowAnalyzer, model: keras.Model):
    """The analyzer's own extraction-layer order — derived, never hand-written."""
    return [
        layer.name
        for layer in analyzer._get_extraction_layers(recursively_get_layers(model))
    ]


MODEL_BUILDERS = {
    "functional": build_functional_model,
    "subclassed": build_subclassed_model,
}


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

class TestInformationFlowIsPopulated:
    """SC7 / SC8: the analyzer must actually capture activations."""

    @pytest.mark.parametrize("kind", sorted(MODEL_BUILDERS))
    def test_information_flow_is_non_empty(self, kind):
        model = MODEL_BUILDERS[kind]()
        _, results = run_analysis(model)

        assert model.name in results.information_flow, (
            f"no information_flow entry written for the {kind} model"
        )
        assert results.information_flow[model.name], (
            f"information_flow for the {kind} model is empty — no activations captured"
        )

    @pytest.mark.parametrize("kind", sorted(MODEL_BUILDERS))
    def test_every_entry_carries_all_seven_visualizer_keys(self, kind):
        model = MODEL_BUILDERS[kind]()
        _, results = run_analysis(model)
        flow = results.information_flow.get(model.name, {})

        assert flow, f"information_flow for the {kind} model is empty"
        for layer_name, analysis in flow.items():
            missing = [key for key in VISUALIZER_KEYS if key not in analysis]
            assert not missing, (
                f"{kind}/{layer_name} is missing visualizer keys {missing}; "
                f"got {sorted(analysis)}"
            )

    @pytest.mark.parametrize("kind", sorted(MODEL_BUILDERS))
    def test_activation_stats_is_populated(self, kind):
        model = MODEL_BUILDERS[kind]()
        _, results = run_analysis(model)

        assert model.name in results.activation_stats, (
            f"no activation_stats entry written for the {kind} model"
        )
        assert results.activation_stats[model.name], (
            f"activation_stats for the {kind} model is empty"
        )


class TestDepthOrdering:
    """SC9: the visualizer reads dict insertion order as network depth."""

    @pytest.mark.parametrize("kind", sorted(MODEL_BUILDERS))
    def test_insertion_order_matches_extraction_layer_order(self, kind):
        model = MODEL_BUILDERS[kind]()
        analyzer, results = run_analysis(model)

        expected = expected_layer_order(analyzer, model)
        actual = list(results.information_flow.get(model.name, {}).keys())

        assert expected, "the probe model has no extraction layers — test is vacuous"
        assert actual == expected, (
            f"{kind}: information_flow key order {actual} does not match the "
            f"analyzer's extraction-layer order {expected}"
        )


class TestModelIsNotMutated:
    """I3: any temporary `call` wrapping must be restored on every exit path."""

    @pytest.mark.parametrize("kind", sorted(MODEL_BUILDERS))
    def test_no_per_instance_call_override_lingers(self, kind):
        model = MODEL_BUILDERS[kind]()
        analyzer = InformationFlowAnalyzer(
            {model.name: model}, AnalysisConfig(n_samples=N_SAMPLES)
        )
        layers = analyzer._get_extraction_layers(recursively_get_layers(model))
        before = {
            layer.name: ("call" in layer.__dict__, type(layer).call)
            for layer in layers
        }

        _, results = run_analysis(model)
        assert results.information_flow.get(model.name), (
            "analysis captured nothing — the non-mutation check would be vacuous"
        )

        for layer in layers:
            had_override, original_call = before[layer.name]
            assert ("call" in layer.__dict__) == had_override, (
                f"{kind}/{layer.name}: a per-instance `call` override lingers "
                f"after analysis"
            )
            assert type(layer).call is original_call, (
                f"{kind}/{layer.name}: the class-level `call` was replaced"
            )

    @pytest.mark.parametrize("kind", sorted(MODEL_BUILDERS))
    def test_forward_pass_is_unchanged_after_analysis(self, kind):
        model = MODEL_BUILDERS[kind]()
        x = make_x()
        before = keras.ops.convert_to_numpy(model(x, training=False))

        _, results = run_analysis(model)
        assert results.information_flow.get(model.name), (
            "analysis captured nothing — the non-mutation check would be vacuous"
        )

        after = keras.ops.convert_to_numpy(model(x, training=False))
        np.testing.assert_array_equal(
            before, after, err_msg=f"{kind}: forward pass changed after analysis"
        )
