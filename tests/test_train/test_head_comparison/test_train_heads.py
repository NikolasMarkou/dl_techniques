"""Unit tests for the head-comparison harness (no training runs)."""

import numpy as np
import pytest
import keras

from train.head_comparison.train_heads import (
    HeadComparisonConfig,
    build_callbacks,
    build_model,
    config_from_args,
    derive_fine_to_coarse,
    expected_calibration_error,
    parse_arguments,
    tail_decile_accuracy,
    to_probabilities,
)


class TestConfig:
    def test_defaults(self):
        cfg = HeadComparisonConfig()
        assert cfg.head == "softmax"
        assert cfg.branching == (10, 10)

    @pytest.mark.parametrize("kwargs", [
        {"head": "softmax_with_extra"},
        {"harmonic_n": 0.0},
        {"hier_branching": "10,9"},
        {"hier_branching": "0,10"},
        {"reassign_every": 0},
        {"reassign_start": 0},
        {"anneal_schedule": "step"},
    ])
    def test_invalid_config_raises(self, kwargs):
        with pytest.raises(ValueError):
            HeadComparisonConfig(**kwargs)


class TestArgv:
    def test_defaults_map(self):
        cfg = config_from_args(parse_arguments([]))
        assert cfg.head == "softmax" and cfg.seed == 0 and cfg.anneal is True

    def test_flags_map(self):
        cfg = config_from_args(parse_arguments(
            ["--head", "hier_full", "--seed", "2", "--no-anneal",
             "--harmonic-n", "8.0", "--epochs", "10"]
        ))
        assert cfg.head == "hier_full" and cfg.seed == 2
        assert cfg.anneal is False and cfg.harmonic_n == 8.0 and cfg.epochs == 10

    def test_help_prints_usage(self, capsys):
        with pytest.raises(SystemExit) as exc:
            parse_arguments(["--help"])
        assert exc.value.code == 0
        assert "usage:" in capsys.readouterr().out


class TestMetricHelpers:
    def test_perfect_predictions_have_zero_ece(self):
        probs = np.eye(4)[[0, 1, 2, 3, 0]]
        assert expected_calibration_error(probs, np.array([0, 1, 2, 3, 0])) == 0.0

    def test_uncertain_wrong_predictions_raise_ece(self):
        probs = np.full((10, 2), 0.5)
        assert expected_calibration_error(probs, np.zeros(10, dtype=int)) > 0.4

    def test_fine_to_coarse_unanimity(self):
        y_fine = np.repeat(np.arange(100), 3)
        y_coarse = np.repeat(np.arange(100) // 5, 3)
        table = derive_fine_to_coarse(y_fine, y_coarse)
        np.testing.assert_array_equal(table, np.arange(100) // 5)

    def test_fine_to_coarse_split_raises(self):
        y_fine = np.repeat(np.arange(100), 4)
        y_coarse = np.tile([0, 0, 1, 1], 100)
        with pytest.raises(ValueError, match="superclasses"):
            derive_fine_to_coarse(y_fine, y_coarse)

    def test_fine_to_coarse_missing_class_raises(self):
        y_fine = np.repeat(np.arange(1, 100), 2)
        y_coarse = np.repeat(np.arange(1, 100) // 5, 2)
        with pytest.raises(ValueError, match="no samples"):
            derive_fine_to_coarse(y_fine, y_coarse)

    def test_tail_decile(self):
        # 10 classes: class 0 always wrong, rest always right.
        y_true = np.repeat(np.arange(10), 4)
        y_pred = y_true.copy()
        y_pred[y_true == 0] = 1
        assert tail_decile_accuracy(y_true, y_pred, num_classes=10) == 0.0

    def test_to_probabilities(self):
        logits = np.array([[2.0, 0.0], [0.0, 0.0]])
        probs = to_probabilities(logits, from_logits=True)
        np.testing.assert_allclose(
            probs.sum(axis=1), np.ones(2), atol=1e-12, rtol=0
        )
        assert probs[0, 0] > 0.85
        np.testing.assert_allclose(
            to_probabilities(logits, from_logits=False), logits,
            atol=0.0, rtol=0,
        )


class TestBuildModel:
    @pytest.mark.parametrize("head", [
        "softmax", "harmonic_logits", "harmonic_dist", "hier_fixed", "hier_full",
    ])
    def test_arm_output_shapes(self, head):
        cfg = HeadComparisonConfig(head=head, epochs=1)
        model, layer, _ = build_model(cfg)
        assert model.output_shape == (None, 100)
        assert layer.name == "head"

    def test_softmax_head_is_dense(self):
        _, layer, _ = build_model(HeadComparisonConfig(head="softmax"))
        assert isinstance(layer, keras.layers.Dense)

    def test_hier_head_depth_and_branching(self):
        _, layer, _ = build_model(HeadComparisonConfig(
            head="hier_full", hier_branching="10,10"
        ))
        assert layer.num_classes == 100
        assert layer.branching == (10, 10)


class TestBuildCallbacks:
    def _tiny_graph(self):
        from dl_techniques.layers.structured_linear.hierarchical_harmonic import (
            HierarchicalHarmonicHead,
        )
        inp = keras.Input((8,))
        trunk_out = keras.layers.Dense(8, name="trunk")(inp)
        head = HierarchicalHarmonicHead(100, branching=(10, 10), name="head")(trunk_out)
        return inp, trunk_out, head

    def _train_arrays(self):
        x = np.zeros((100, 8), dtype="float32")
        return x, np.arange(100)

    def test_hier_full_wires_reassign_and_anneal(self, tmp_path):
        from train.head_comparison.train_heads import build_callbacks
        inp, trunk_out, head = self._tiny_graph()
        x, y = self._train_arrays()
        callbacks = build_callbacks(
            HeadComparisonConfig(head="hier_full"), str(tmp_path),
            inp, head, trunk_out, x, y,
        )
        kinds = [type(c).__name__ for c in callbacks]
        assert "PeriodicReassignCallback" in kinds
        assert "HarmonicExponentAnnealingCallback" in kinds

    def test_softmax_wires_neither(self, tmp_path):
        from train.head_comparison.train_heads import build_callbacks
        inp, trunk_out, head = self._tiny_graph()
        x, y = self._train_arrays()
        callbacks = build_callbacks(
            HeadComparisonConfig(head="softmax"), str(tmp_path),
            inp, head, trunk_out, x, y,
        )
        kinds = [type(c).__name__ for c in callbacks]
        assert "PeriodicReassignCallback" not in kinds
        assert "HarmonicExponentAnnealingCallback" not in kinds
