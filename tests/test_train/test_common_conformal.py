"""Tests for ``train.common.conformal``: the post-training conformal step the trainers share."""

import argparse
import json

import keras
import numpy as np
import pytest
import tensorflow as tf

import train.common as common
from dl_techniques.utils.conformal_classification import (
    predict_sets,
    repeated_split_coverage,
    softmax,
)
from train.common.conformal import (
    CONFORMAL_JSON_NAME,
    CONFORMAL_MODEL_NAME,
    add_conformal_arguments,
    collect_logits,
    load_conformal_model,
    run_conformal,
)


def _identity_model(c):
    inp = keras.Input((c,))
    return keras.Model(inp, keras.layers.Identity()(inp))


def _problem(n=3000, c=10, seed=0):
    """Logits of an overconfident classifier on exchangeable samples."""
    rng = np.random.default_rng(seed)
    logits = (rng.normal(size=(n, c)) * 3.0).astype("float32")
    p = softmax(logits / 2.0)
    labels = np.array([rng.choice(c, p=row) for row in p], dtype="int32")
    return logits, labels


def _ds(logits, labels, batch=64):
    return tf.data.Dataset.from_tensor_slices((logits, labels)).batch(batch)


class TestArguments:
    def _parse(self, *argv):
        parser = argparse.ArgumentParser()
        add_conformal_arguments(parser)
        return parser.parse_args(list(argv))

    def test_defaults(self):
        args = self._parse()
        assert args.conformal_alpha == 0.1 and args.no_conformal is False

    def test_flags_are_read(self):
        args = self._parse("--conformal-alpha", "0.05", "--no-conformal")
        assert args.conformal_alpha == 0.05 and args.no_conformal is True

    @pytest.mark.parametrize("bad", ["0", "1", "-0.1", "1.5", "abc"])
    def test_an_alpha_outside_the_open_unit_interval_is_refused(self, bad, capsys):
        with pytest.raises(SystemExit) as exc:
            self._parse("--conformal-alpha", bad)
        assert exc.value.code == 2
        assert "alpha" in capsys.readouterr().err

    def test_the_note_prefixes_both_help_strings(self):
        parser = argparse.ArgumentParser()
        add_conformal_arguments(parser, note="Used only when X: ")
        helps = [a.help for a in parser._actions
                 if a.dest in ("conformal_alpha", "no_conformal")]
        assert len(helps) == 2 and all(h.startswith("Used only when X: ") for h in helps)

    def test_the_names_are_exported_from_the_hub(self):
        for name in ("add_conformal_arguments", "collect_logits",
                     "load_conformal_model", "run_conformal"):
            assert name in common.__all__ and hasattr(common, name)


class TestCollectLogits:
    def test_order_and_label_alignment(self):
        logits, labels = _problem(n=30, c=5)
        got_x, got_y = collect_logits(_identity_model(5), _ds(logits, labels, batch=7))
        np.testing.assert_allclose(got_x, logits, atol=1e-6)
        assert got_y.tolist() == labels.tolist()

    def test_a_multi_output_model_is_refused(self):
        inp = keras.Input((4,))
        d = keras.layers.Dense(3)(inp)
        ds = _ds(np.zeros((4, 4), "float32"), np.zeros(4, "int32"))
        with pytest.raises(ValueError, match="single-output"):
            collect_logits(keras.Model(inp, [d, d]), ds)

    def test_an_empty_dataset_is_refused(self):
        ds = _ds(np.zeros((0, 4), "float32"), np.zeros(0, "int32"))
        with pytest.raises(ValueError, match="empty"):
            collect_logits(_identity_model(4), ds)


class TestRunConformal:
    @pytest.fixture(scope="class")
    def run(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("conformal")
        logits, labels = _problem()
        record = run_conformal(
            _identity_model(10), _ds(logits, labels), alpha=0.1, seed=3,
            out_dir=out, input_shape=(10,), model_name="identity", weights="last epoch",
        )
        return out, record, logits, labels

    def test_the_record_on_disk_matches_the_returned_one(self, run):
        out, record, _, labels = run
        on_disk = json.loads((out / CONFORMAL_JSON_NAME).read_text())
        assert on_disk == json.loads(json.dumps(record))
        assert on_disk["alpha"] == 0.1 and on_disk["target_coverage"] == pytest.approx(0.9)
        assert on_disk["n_calibration"] == len(labels)
        assert on_disk["model"] == "identity" and on_disk["weights"] == "last epoch"
        assert on_disk["score"] == "aps_deterministic" and on_disk["tau_is_infinite"] is False
        assert 0.0 < on_disk["tau"] <= 1.0 and on_disk["caveat"]

    def test_the_random_half_splits_reach_the_target(self, run):
        splits = run[1]["repeated_splits"]
        assert splits["coverage_mean"] >= 0.9 - 0.01
        assert splits["repeats"] == 20
        assert splits["n_calibration"] + splits["n_evaluation"] == len(run[3])

    def test_the_split_statistics_describe_the_deterministic_rule_the_export_runs(self, run):
        _, record, logits, labels = run
        want = repeated_split_coverage(logits, labels, 0.1, repeats=20, seed=3,
                                       randomized=False)
        assert record["repeated_splits"] == pytest.approx(want)

    def test_the_exported_model_reproduces_the_numpy_sets(self, run):
        out, record, logits, _ = run
        model = load_conformal_model(out / CONFORMAL_MODEL_NAME)
        got_logits, _, in_set = model(logits[:200])
        np.testing.assert_allclose(keras.ops.convert_to_numpy(got_logits), logits[:200], atol=1e-6)
        want = predict_sets(logits[:200], record["tau"], randomized=False)
        assert np.array_equal(keras.ops.convert_to_numpy(in_set).astype(bool), want)

    def test_a_calibration_set_too_small_for_alpha_exports_full_sets(self, tmp_path):
        logits, labels = _problem(n=6, c=4)
        record = run_conformal(
            _identity_model(4), _ds(logits, labels), alpha=0.05, seed=0, out_dir=tmp_path,
            input_shape=(4,), model_name="identity", weights="last epoch",
        )
        assert record["tau_is_infinite"] is True and record["tau"] is None
        model = load_conformal_model(tmp_path / CONFORMAL_MODEL_NAME)
        assert keras.ops.convert_to_numpy(model(logits)[2]).all()
