"""
Guards for the SAM training path (`SAMTrainingModel`) and for the instrument
that certifies it.

Structure
---------
* ``TestDeadComponentInstrument`` (plan step 1) RED-proves
  ``dead_component_oracle.py`` itself, on a three-branch toy model whose three
  branches are *known* to be live / forward-live-but-gradient-dead / entirely
  dead. An instrument that cannot tell those three apart blinds every step
  after it, so this class is a precondition for the rest of the file rather
  than an ornament.
* ``TestSAMTrainingModel`` (plan step 2) applies the instrument to the real
  wrapper.

Measured on GPU 1 (RTX 4070), keras 3.8.0 / tf 2.18.0.
"""

from typing import Any, Dict, Tuple

import keras
import numpy as np
import pytest
from keras import ops

from .dead_component_oracle import (
    NO_GRADIENTS_MESSAGE,
    ComponentResponse,
    component_response,
    destroy_negatives,
    destroy_positives,
    fit_one_step_moved_variables,
    no_op_kill,
    outputs_stop_gradient,
    variable_labels,
    zeroed_variables,
)

# ---------------------------------------------------------------------------
# A toy model with three branches of KNOWN liveness.
# ---------------------------------------------------------------------------
PROBE_UNITS = 4
PROBE_FEATURES = 3
PROBE_BATCH = 8


class ThreeBranchProbe(keras.Model):
    """
    A model whose three branches have deliberately different liveness.

    * ``live`` -- contributes to the output AND receives gradient.
    * ``frozen`` -- contributes to the output but its gradient is severed by
      ``ops.stop_gradient``: destroying it MOVES the metric while its own
      variables never move. This is the branch that separates the instrument's
      two halves; a probe that only counts moved variables would call it dead,
      and a probe that only watches the metric would call it live.
    * ``dead`` -- multiplied by zero, so it contributes nothing to the forward
      pass and receives an all-zero gradient. Destroying it must move nothing.

    Args:
        units: Output width of each branch.
        **kwargs: Forwarded to ``keras.Model``.
    """

    def __init__(self, units: int = PROBE_UNITS, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.live = keras.layers.Dense(units, use_bias=False, name="live")
        self.frozen = keras.layers.Dense(units, use_bias=False, name="frozen")
        self.dead = keras.layers.Dense(units, use_bias=False, name="dead")

    def build(self, input_shape: Tuple[Any, ...]) -> None:
        self.live.build(input_shape)
        self.frozen.build(input_shape)
        self.dead.build(input_shape)
        super().build(input_shape)

    def call(self, inputs: Any, training: bool = None) -> Any:
        live = self.live(inputs)
        frozen = ops.stop_gradient(self.frozen(inputs))
        dead = self.dead(inputs) * 0.0
        return live + frozen + dead

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["units"] = self.units
        return config


def _probe_data(seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Fixed inputs/targets for the toy probe (deterministic across calls)."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(-1.0, 1.0, size=(PROBE_BATCH, PROBE_FEATURES)).astype("float32")
    y = rng.uniform(-1.0, 1.0, size=(PROBE_BATCH, PROBE_UNITS)).astype("float32")
    return x, y


def _built_probe(seed: int = 0) -> Tuple[ThreeBranchProbe, np.ndarray, np.ndarray]:
    """A compiled, BUILT probe with non-zero weights on every branch."""
    keras.utils.set_random_seed(seed)
    model = ThreeBranchProbe()
    x, y = _probe_data(seed)
    model(x)  # build
    # Every branch must be non-zero, or "zeroing it changed nothing" would be
    # true for a reason that has nothing to do with liveness.
    for index, variable in enumerate(model.trainable_variables):
        value = np.array(ops.convert_to_numpy(variable), copy=True)
        variable.assign(value + 0.1 * (index + 1))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-2), loss="mse")
    return model, x, y


def _mse(model: keras.Model, x: np.ndarray, y: np.ndarray) -> float:
    """Deterministic metric: inference-mode MSE computed in numpy."""
    pred = np.asarray(ops.convert_to_numpy(model(x, training=False)))
    return float(np.mean((pred - y) ** 2))


class TestDeadComponentInstrument:
    """
    RED proofs for ``dead_component_oracle.py``.

    The instrument must (1) name the variables that moved and the variables
    that did not, (2) report "did not move" for a genuinely dead component and
    "moved" for a live one, and (3) turn a live training path RED when
    ``stop_gradient`` is injected on the outputs.
    """

    def test_branch_weights_are_all_nonzero_before_any_probe(self) -> None:
        """Premise check: a zero weight would make every kill vacuous."""
        model, _, _ = _built_probe()
        for variable in model.trainable_variables:
            value = np.abs(np.asarray(ops.convert_to_numpy(variable)))
            assert float(np.min(value)) > 0.0, f"{variable.path} has a zero entry"

    def test_moved_report_names_exactly_the_live_branch(self) -> None:
        """
        (a) The instrument reports moved/unmoved BY NAME, not as a count.

        Only ``live`` receives gradient; ``frozen`` is severed by
        ``stop_gradient`` and ``dead`` gets an exactly-zero gradient, so both
        must appear in ``unmoved``.
        """
        model, x, y = _built_probe()
        report = fit_one_step_moved_variables(model, x, y, batch_size=PROBE_BATCH)

        assert report.total == 3, report.summary()
        # Labels are full Keras paths, e.g. "three_branch_probe/frozen/kernel";
        # the branch name is the second-to-last segment.
        moved_branches = {label.split("/")[-2] for label in report.moved}
        unmoved_branches = {label.split("/")[-2] for label in report.unmoved}
        assert moved_branches == {"live"}, report.summary()
        assert unmoved_branches == {"frozen", "dead"}, report.summary()
        # The magnitude is reported, not just the verdict.
        assert report.max_abs_delta[report.moved[0]] > 0.0
        for label in report.unmoved:
            assert report.max_abs_delta[label] == 0.0

    def test_instrument_reports_moved_for_a_live_component(self) -> None:
        """
        (b1) Killing a component the metric actually depends on must MOVE it.

        ``frozen`` is the discriminating case: gradient-dead but forward-live.
        """
        model, x, y = _built_probe()
        response = component_response(
            lambda: _mse(model, x, y),
            lambda: zeroed_variables(model.frozen.weights),
            name="frozen branch (forward-live)",
        )
        assert response.moved, response.summary()
        assert response.delta > 0.0, response.summary()

    def test_instrument_reports_not_moved_for_a_genuinely_dead_component(self) -> None:
        """
        (b2) Killing a component nothing depends on must report DID NOT MOVE.

        This is the half that a "the loss went down, therefore it works" test
        can never supply. ``dead`` is multiplied by zero inside ``call``, so
        zeroing its kernel is bit-identically invisible.
        """
        model, x, y = _built_probe()
        response = component_response(
            lambda: _mse(model, x, y),
            lambda: zeroed_variables(model.dead.weights),
            name="dead branch (multiplied by zero)",
        )
        assert not response.moved, response.summary()
        assert response.delta == 0.0, response.summary()

    def test_no_op_kill_is_the_instruments_own_negative_control(self) -> None:
        """A killer that destroys nothing must produce an exactly-zero delta."""
        model, x, y = _built_probe()
        response = component_response(
            lambda: _mse(model, x, y), no_op_kill, name="no-op control"
        )
        assert not response.moved and response.delta == 0.0, response.summary()

    def test_zeroed_variables_restores_the_original_values_exactly(self) -> None:
        """A killer that does not restore would poison every later measurement."""
        model, _, _ = _built_probe()
        before = [np.array(ops.convert_to_numpy(w), copy=True) for w in model.dead.weights]
        with zeroed_variables(model.dead.weights):
            during = [np.asarray(ops.convert_to_numpy(w)) for w in model.dead.weights]
        after = [np.asarray(ops.convert_to_numpy(w)) for w in model.dead.weights]
        for b, d, a in zip(before, during, after):
            assert float(np.max(np.abs(d))) == 0.0
            assert float(np.max(np.abs(a - b))) == 0.0

    def test_stop_gradient_injection_drives_a_live_training_path_red(self) -> None:
        """
        (c) The dead-component injection must make a LIVE model raise.

        The exact Keras message is asserted; ``pytest.raises(Exception)`` would
        accept any breakage, including an unrelated one.
        """
        model, x, y = _built_probe()
        with outputs_stop_gradient(model):
            with pytest.raises(ValueError, match=NO_GRADIENTS_MESSAGE):
                fit_one_step_moved_variables(model, x, y, batch_size=PROBE_BATCH)

    def test_the_same_model_trains_without_the_injection(self) -> None:
        """
        The GREEN half of the previous test: without the injection the very
        same model moves variables. Without this pairing, the raise above could
        be caused by anything.
        """
        model, x, y = _built_probe()
        report = fit_one_step_moved_variables(model, x, y, batch_size=PROBE_BATCH)
        assert report.n_moved > 0, report.summary()

    def test_stop_gradient_injection_is_removed_on_exit(self) -> None:
        """
        The injection must not survive its ``with`` block, or a later "the model
        trains" assertion would be measuring the sabotaged model.
        """
        model, x, y = _built_probe()
        with outputs_stop_gradient(model):
            pass
        assert "call" not in model.__dict__
        report = fit_one_step_moved_variables(model, x, y, batch_size=PROBE_BATCH)
        assert report.n_moved > 0, report.summary()

    def test_variable_labels_refuses_an_unbuilt_model(self) -> None:
        """
        An empty variable list makes every moved/unmoved claim vacuously true,
        so the instrument refuses rather than reporting ``0/0``.
        """
        model = ThreeBranchProbe()
        with pytest.raises(ValueError, match="ZERO trainable variables"):
            variable_labels(model)

    def test_zeroed_variables_refuses_an_empty_variable_list(self) -> None:
        """Killing nothing and seeing nothing is the probe-that-passes-both-ways."""
        with pytest.raises(ValueError, match="EMPTY variable list"):
            with zeroed_variables([]):
                pass

    def test_variable_labels_are_unique(self) -> None:
        """Labels are dict keys in the report; a collision would silently drop one."""
        model, _, _ = _built_probe()
        labels = variable_labels(model)
        assert len(labels) == len(set(labels)) == len(model.trainable_variables)

    def test_destroy_negatives_only_touches_negative_pixels(self) -> None:
        """The pixel-class killers must destroy exactly one class, not both."""
        gt = np.array([[0.0, 1.0], [1.0, 0.0]], dtype="float32")
        pred = np.array([[0.1, 0.9], [0.8, 0.2]], dtype="float32")
        out = destroy_negatives(pred, gt, wrong=0.99)
        assert out[0, 0] == pytest.approx(0.99) and out[1, 1] == pytest.approx(0.99)
        assert out[0, 1] == pytest.approx(0.9) and out[1, 0] == pytest.approx(0.8)
        # The input is not mutated.
        assert pred[0, 0] == pytest.approx(0.1)

    def test_destroy_positives_only_touches_positive_pixels(self) -> None:
        """Mirror of the previous test for the positive class."""
        gt = np.array([[0.0, 1.0], [1.0, 0.0]], dtype="float32")
        pred = np.array([[0.1, 0.9], [0.8, 0.2]], dtype="float32")
        out = destroy_positives(pred, gt, wrong=0.01)
        assert out[0, 1] == pytest.approx(0.01) and out[1, 0] == pytest.approx(0.01)
        assert out[0, 0] == pytest.approx(0.1) and out[1, 1] == pytest.approx(0.2)

    def test_pixel_killers_refuse_a_single_class_ground_truth(self) -> None:
        """
        A ground truth with no negatives (or no positives) makes the destroy
        probe a no-op that would pass against ANY loss, including a blind one.
        """
        all_positive = np.ones((2, 2), dtype="float32")
        all_negative = np.zeros((2, 2), dtype="float32")
        pred = np.full((2, 2), 0.5, dtype="float32")
        with pytest.raises(ValueError, match="NO negative pixel"):
            destroy_negatives(pred, all_positive)
        with pytest.raises(ValueError, match="NO positive pixel"):
            destroy_positives(pred, all_negative)
