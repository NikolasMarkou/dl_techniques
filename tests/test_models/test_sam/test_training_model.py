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

import os
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.models.sam import SAM, SAMTrainingModel
from dl_techniques.losses.sam_mask_loss import SAMIoULoss, SAMMaskLoss
from dl_techniques.models.sam.training_model import (
    INPUT_BOXES,
    INPUT_GT_MASK,
    INPUT_IMAGE,
    INPUT_POINT_COORDS,
    INPUT_POINT_LABELS,
    IOU_PREDICTIONS,
    IOU_SUPERVISION,
    LOW_RES_LOGITS,
    OUTPUT_KEYS,
    achieved_mask_iou,
)

from .test_correctness import (
    GRID_SIZE,
    IMG_SIZE,
    build_reduced_sam,
    seed_nonzero_weights,
)
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


# ===========================================================================
# Plan step 2 -- `SAMTrainingModel`
# ===========================================================================
#: Batch size and prompt-point count used by every wrapper guard below. Small
#: on purpose: the reduced SAM fixture is 321,862 params and the whole class
#: must stay inside the ordinary (non-slow) gate on a shared 12 GB card.
WRAPPER_BATCH = 2
WRAPPER_POINTS = 1
#: `low_res_logits` spatial size = 4x the image-embedding grid.
LOW_RES = 4 * GRID_SIZE


def _wrapper_inputs(
    labels_value: int = 1,
    with_boxes: bool = False,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Build a deterministic input dict for the wrapper.

    Args:
        labels_value: Point label written into every row. ``1`` foreground,
            ``0`` background, ``-1`` padding.
        with_boxes: Whether to add a box prompt.
        seed: Seed for the image / coordinate draws.

    Returns:
        The input dict ``SAMTrainingModel.call`` consumes.
    """
    rng = np.random.RandomState(seed)
    inputs: Dict[str, np.ndarray] = {
        INPUT_IMAGE: rng.uniform(
            0.0, 255.0, (WRAPPER_BATCH, IMG_SIZE, IMG_SIZE, 3)
        ).astype("float32"),
        INPUT_POINT_COORDS: rng.uniform(
            0.0, float(IMG_SIZE), (WRAPPER_BATCH, WRAPPER_POINTS, 2)
        ).astype("float32"),
        INPUT_POINT_LABELS: np.full(
            (WRAPPER_BATCH, WRAPPER_POINTS), labels_value, dtype="int32"
        ),
    }
    if with_boxes:
        inputs[INPUT_BOXES] = np.tile(
            np.array([[[10.0, 20.0, 100.0, 120.0]]], dtype="float32"),
            (WRAPPER_BATCH, 1, 1),
        )
    return inputs


def _gt_mask_stack(num_masks: int) -> np.ndarray:
    """A binary GT mask stack with real structure (a filled rectangle)."""
    gt = np.zeros((WRAPPER_BATCH, num_masks, LOW_RES, LOW_RES), dtype="float32")
    gt[:, :, 12:40, 20:52] = 1.0
    return gt


def _wrapper_targets(num_masks: int, seed: int = 1) -> Dict[str, np.ndarray]:
    """Dict ``y_true`` matching the wrapper's two output keys."""
    rng = np.random.RandomState(seed)
    return {
        LOW_RES_LOGITS: rng.uniform(
            0.0, 1.0, (WRAPPER_BATCH, num_masks, LOW_RES, LOW_RES)
        ).astype("float32"),
        IOU_PREDICTIONS: rng.uniform(
            0.0, 1.0, (WRAPPER_BATCH, num_masks)
        ).astype("float32"),
    }


def _built_wrapper(
    multimask_output: bool = False,
    seed: int = 7,
    inputs: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[SAMTrainingModel, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    A BUILT, seeded, compiled wrapper plus matching inputs and targets.

    The weights are seeded non-zero because many SAM weights initialize to
    exactly zero (``rel_pos_h/w``, every bias, ``not_a_point_embed``), and a
    liveness probe against an all-zero weight can be structurally unable to
    observe what it claims to measure (iteration-1 carried surprise #1).
    """
    keras.utils.set_random_seed(seed)
    model = SAMTrainingModel(build_reduced_sam(), multimask_output=multimask_output)
    x = _wrapper_inputs() if inputs is None else inputs
    y = _wrapper_targets(3 if multimask_output else 1)
    model(x)  # build
    seed_nonzero_weights(model)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss={LOW_RES_LOGITS: "mse", IOU_PREDICTIONS: "mse"},
        loss_weights={LOW_RES_LOGITS: 20.0, IOU_PREDICTIONS: 1.0},
    )
    return model, x, y


def _unmoved_variables(
    model: keras.Model, report: Any
) -> List[Any]:
    """
    Resolve a :class:`MovedVariablesReport`'s unmoved labels back to VARIABLES.

    Identity, not the name string, is what the guards compare. Keras uniquifies
    sub-layer names per process (``prompt_encoder``, ``prompt_encoder_1``, ...),
    so a name-based expectation would pass or fail depending on how many models
    an earlier test in the same session happened to construct.

    Args:
        model: The model the report was produced from.
        report: The report returned by ``fit_one_step_moved_variables``.

    Returns:
        The variables named in ``report.unmoved``, in report order.
    """
    lookup = dict(zip(variable_labels(model), model.trainable_variables))
    return [lookup[label] for label in report.unmoved]


def _ids(variables: Sequence[Any]) -> set:
    """Identity set of a variable sequence."""
    return {id(v) for v in variables}


class TestSAMTrainingModelForward:
    """The wrapper's output contract and its input validation."""

    def test_output_is_exactly_the_two_differentiable_keys(self) -> None:
        """
        ``masks`` is deliberately absent: at the default ``binarize_masks=True``
        it is a ``uint8`` tensor with zero gradient for every variable (D-011),
        and it is the full-resolution key whose resize makes ``SAM.call``
        untraceable.
        """
        model, x, _ = _built_wrapper()
        out = model(x)
        assert set(out.keys()) == set(OUTPUT_KEYS) == {LOW_RES_LOGITS, IOU_PREDICTIONS}

    @pytest.mark.parametrize("multimask,num_masks", [(False, 1), (True, 3)])
    def test_output_shapes(self, multimask: bool, num_masks: int) -> None:
        """``low_res_logits`` is (B, M, 4*grid, 4*grid); ``iou_predictions`` is (B, M)."""
        model, x, _ = _built_wrapper(multimask_output=multimask)
        out = model(x)
        assert tuple(out[LOW_RES_LOGITS].shape) == (
            WRAPPER_BATCH, num_masks, LOW_RES, LOW_RES,
        )
        assert tuple(out[IOU_PREDICTIONS].shape) == (WRAPPER_BATCH, num_masks)

    def test_low_res_logits_are_at_the_mask_prompt_resolution(self) -> None:
        """
        Step 4 feeds ``low_res_logits`` straight back as the mask prompt, and
        ``PromptEncoder`` accepts exactly ``4 * image_embedding_size`` (D-016).
        This pins that the feedback is shape-native, so a future geometry change
        breaks here rather than inside the refinement loop.
        """
        model, x, _ = _built_wrapper()
        grid = model.sam.prompt_encoder.image_embedding_size
        out = model(x)
        assert tuple(out[LOW_RES_LOGITS].shape)[2:] == (4 * grid[0], 4 * grid[1])

    def test_missing_image_is_refused(self) -> None:
        model, x, _ = _built_wrapper()
        broken = {k: v for k, v in x.items() if k != INPUT_IMAGE}
        with pytest.raises(ValueError, match=f"must contain '{INPUT_IMAGE}'"):
            model(broken)

    def test_half_a_point_prompt_is_refused(self) -> None:
        """Coords without labels would silently become an all-zero label vector."""
        model, x, _ = _built_wrapper()
        broken = {k: v for k, v in x.items() if k != INPUT_POINT_LABELS}
        with pytest.raises(ValueError, match="must be\n?\\s*supplied together"):
            model(broken)

    def test_a_prompt_less_forward_is_refused(self) -> None:
        """
        The prompt encoder happily returns a zero-length sparse embedding for no
        prompt at all, so a prompt-less batch trains SAM to ignore prompts with
        every shape assertion still green.
        """
        model, x, _ = _built_wrapper()
        with pytest.raises(ValueError, match="at least one prompt"):
            model({INPUT_IMAGE: x[INPUT_IMAGE]})


class TestSAMTrainingModelEquivalence:
    """A-2: the submodule route must reproduce ``SAM.call``'s own numbers."""

    def test_wrapper_matches_an_eager_sam_call_value_exactly(self) -> None:
        """
        The wrapper duplicates ``SAM.call``'s four-step orchestration (D-028
        names that duplication as the cost of this design), so the two are
        pinned equal here rather than assumed equal. ``SAM.call`` is used
        EAGERLY -- it cannot be traced at all (see ``TestSAMCallSpy``).
        """
        model, x, _ = _built_wrapper(multimask_output=True)
        wrapper = model(x)
        sam_out = model.sam(
            {
                "image": ops.convert_to_tensor(x[INPUT_IMAGE]),
                "points": (
                    ops.convert_to_tensor(x[INPUT_POINT_COORDS]),
                    ops.convert_to_tensor(x[INPUT_POINT_LABELS]),
                ),
                "original_size": ops.convert_to_tensor((IMG_SIZE, IMG_SIZE)),
            },
            multimask_output=True,
        )
        for key in OUTPUT_KEYS:
            diff = float(
                np.max(
                    np.abs(
                        ops.convert_to_numpy(wrapper[key])
                        - ops.convert_to_numpy(sam_out[key])
                    )
                )
            )
            assert diff == 0.0, f"{key} diverged from SAM.call by {diff}"


class TestSAMTrainingModelGradients:
    """
    SC-2/SC-3: the wrapper trains, its dead-component probe turns it RED, and
    **every** non-moving variable is named and shown to be live under the input
    that reaches it.
    """

    def test_fit_one_step_names_every_non_moving_variable(self) -> None:
        """
        The residual F-5 measured (``118/137``) and never decomposed.

        Measured here on the reduced fixture at ``multimask_output=False`` with
        a single foreground point and no box: **170 of 201** trainable variables
        move, and the 31 that do not are EXACTLY three groups, each unreachable
        by this input rather than dead:

        * ``point_embeddings[0]`` (background-point type) -- no label-0 point;
        * ``point_embeddings[2]``/``[3]`` (box corner types) -- no box prompt;
        * ``mask_downscaling`` (10 vars) -- ``masks=None`` on a single round;
        * ``output_hypernetworks_mlps[1..3]`` (18 vars) -- ``multimask_output``
          is ``False``, so ``MaskDecoder`` slices ``masks[:, 0:1]``.

        Each of the three justifications is a MEASUREMENT, not an argument: the
        three tests that follow supply the missing input and show the same
        variables move.
        """
        model, x, y = _built_wrapper()
        report = fit_one_step_moved_variables(model, x, y, batch_size=WRAPPER_BATCH)

        pe = model.sam.prompt_encoder
        md = model.sam.mask_decoder
        expected: List[Any] = []
        for index in (0, 2, 3):
            expected += list(pe.point_embeddings[index].trainable_variables)
        expected += list(pe.mask_downscaling.trainable_variables)
        for index in (1, 2, 3):
            expected += list(md.output_hypernetworks_mlps[index].trainable_variables)

        assert report.total == 201, report.summary()
        assert len(expected) == 31
        assert _ids(_unmoved_variables(model, report)) == _ids(expected), (
            report.summary()
        )
        assert report.n_moved == 170, report.summary()

    def test_background_point_swaps_which_point_type_embedding_is_dead(self) -> None:
        """
        Justification-by-measurement for ``point_embeddings[0]``.

        With a label-1 point, type embedding 0 is unreachable; with a label-0
        point, embedding 1 is. The pair must SWAP -- an embedding that stayed
        dead under both would be a real defect.
        """
        model, x, y = _built_wrapper(inputs=_wrapper_inputs(labels_value=0))
        report = fit_one_step_moved_variables(model, x, y, batch_size=WRAPPER_BATCH)
        unmoved = _ids(_unmoved_variables(model, report))
        pe = model.sam.prompt_encoder
        assert _ids(pe.point_embeddings[0].trainable_variables) & unmoved == set(), (
            "background-point embedding still dead under a background point"
        )
        assert _ids(pe.point_embeddings[1].trainable_variables) <= unmoved, (
            "foreground-point embedding moved with no foreground point"
        )

    def test_a_box_prompt_makes_the_corner_embeddings_live(self) -> None:
        """
        Justification-by-measurement for ``point_embeddings[2]``/``[3]``.

        A second, unplanned consequence is asserted alongside because it is the
        same mechanism: ``PromptEncoder.call`` sets ``pad=(boxes is None)``, so
        supplying a box removes the padding point and ``not_a_point_embed``
        becomes the unreachable one.
        """
        model, x, y = _built_wrapper(inputs=_wrapper_inputs(with_boxes=True))
        report = fit_one_step_moved_variables(model, x, y, batch_size=WRAPPER_BATCH)
        unmoved = _ids(_unmoved_variables(model, report))
        pe = model.sam.prompt_encoder
        for index in (2, 3):
            assert _ids(pe.point_embeddings[index].trainable_variables) & unmoved == set(), (
                f"box corner embedding {index} still dead under a box prompt"
            )
        assert _ids(pe.not_a_point_embed.trainable_variables) <= unmoved, (
            "not_a_point_embed moved although a box prompt suppresses padding"
        )

    def test_multimask_output_swaps_which_hypernetworks_are_dead(self) -> None:
        """
        Justification-by-measurement for ``output_hypernetworks_mlps[1..3]``.

        ``MaskDecoder`` slices ``masks[:, 0:1]`` at ``multimask_output=False``
        and ``masks[:, 1:]`` at ``True``, so the dead set must invert: exactly
        head 0 becomes unreachable and heads 1-3 become live.
        """
        model, x, y = _built_wrapper(multimask_output=True)
        report = fit_one_step_moved_variables(model, x, y, batch_size=WRAPPER_BATCH)
        unmoved = _ids(_unmoved_variables(model, report))
        md = model.sam.mask_decoder
        assert _ids(md.output_hypernetworks_mlps[0].trainable_variables) <= unmoved
        for index in (1, 2, 3):
            assert _ids(
                md.output_hypernetworks_mlps[index].trainable_variables
            ) & unmoved == set(), f"hypernetwork {index} still dead at multimask=True"

    def test_mask_downscaling_is_live_only_when_a_mask_prompt_is_supplied(self) -> None:
        """
        Justification-by-measurement for the 10 ``mask_downscaling`` variables.

        They are unreachable in step 2 because a single decoding round has no
        mask to feed back. Supplying one makes every gradient non-``None``;
        ``masks=None`` makes every gradient ``None``. Both directions are
        asserted, because "0 of 10 are None" alone would also hold for a probe
        that was measuring the wrong variables.
        """
        keras.utils.set_random_seed(11)
        sam = build_reduced_sam()
        sam.build(None)
        prompt_encoder = sam.prompt_encoder
        seed_nonzero_weights(prompt_encoder)
        variables = list(prompt_encoder.mask_downscaling.trainable_variables)
        assert len(variables) == 10

        mask = ops.convert_to_tensor(
            np.random.RandomState(3)
            .uniform(-1.0, 1.0, (WRAPPER_BATCH, 1, LOW_RES, LOW_RES))
            .astype("float32")
        )
        with tf.GradientTape() as tape:
            _, dense = prompt_encoder(points=None, boxes=None, masks=mask)
            loss = tf.reduce_sum(dense)
        with_mask_none = sum(1 for g in tape.gradient(loss, variables) if g is None)

        with tf.GradientTape() as tape:
            _, dense = prompt_encoder(points=None, boxes=None, masks=None)
            loss = tf.reduce_sum(dense)
        without_mask_none = sum(1 for g in tape.gradient(loss, variables) if g is None)

        assert with_mask_none == 0, "mask_downscaling is dead even WITH a mask prompt"
        assert without_mask_none == 10, "mask_downscaling appeared live with masks=None"

    def test_dead_component_probe_makes_the_training_path_red(self) -> None:
        """
        SC-3: with ``stop_gradient`` on both outputs the very same ``fit()``
        call must raise. Without this, the green result above is not evidence.
        """
        model, x, y = _built_wrapper()
        with outputs_stop_gradient(model):
            with pytest.raises(ValueError, match=NO_GRADIENTS_MESSAGE):
                fit_one_step_moved_variables(model, x, y, batch_size=WRAPPER_BATCH)


class TestSAMCallSpy:
    """
    SC-4 / I-3: ``SAM.call`` must be invoked ZERO times on the training path,
    pinned by a spy that is itself RED-proved by a control that deliberately
    routes through ``SAM.call``.
    """

    @staticmethod
    def _spy(monkeypatch: pytest.MonkeyPatch) -> List[int]:
        """Patch ``SAM.call`` at the CLASS level and return a mutable counter."""
        counter: List[int] = [0]
        original = SAM.call

        def counting_call(self: SAM, *args: Any, **kwargs: Any) -> Any:
            counter[0] += 1
            return original(self, *args, **kwargs)

        monkeypatch.setattr(SAM, "call", counting_call)
        return counter

    def test_sam_call_is_invoked_zero_times_during_fit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model, x, y = _built_wrapper()
        counter = self._spy(monkeypatch)
        model.fit(x, y, epochs=1, verbose=0, batch_size=WRAPPER_BATCH)
        assert counter[0] == 0, f"SAM.call was reached {counter[0]} time(s) during fit()"

    def test_the_spy_counts_when_sam_call_is_actually_reached(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        The spy's own RED proof. A spy that can only ever report 0 proves
        nothing, so the same instrument is pointed at an EAGER ``SAM.call`` --
        the one context in which that call works at all.
        """
        model, x, _ = _built_wrapper()
        counter = self._spy(monkeypatch)
        model.sam(
            {
                "image": ops.convert_to_tensor(x[INPUT_IMAGE]),
                "points": (
                    ops.convert_to_tensor(x[INPUT_POINT_COORDS]),
                    ops.convert_to_tensor(x[INPUT_POINT_LABELS]),
                ),
                "original_size": ops.convert_to_tensor((IMG_SIZE, IMG_SIZE)),
            },
            multimask_output=False,
        )
        assert counter[0] == 1

    @pytest.mark.parametrize(
        "spelling,expected_type,expected_message",
        [
            (
                "constant_tensor",
                TypeError,
                "len is not well defined for a symbolic Tensor",
            ),
            (
                "batched_from_data",
                ValueError,
                "'size' must be a 1-D Tensor of 2 elements",
            ),
            (
                "from_ops_shape",
                ValueError,
                "Only input tensors may be passed as positional arguments",
            ),
        ],
    )
    def test_routing_through_sam_call_cannot_be_traced(
        self, spelling: str, expected_type: type, expected_message: str
    ) -> None:
        """
        A-3, verified by execution for all three spellings of ``original_size``
        anyone would reach for. This is why the wrapper exists, and it is the
        control that makes the zero-count assertion above meaningful.

        Each exception TYPE was measured, never predicted -- F-5 recorded only
        the first spelling, and the other two raise a ``ValueError``, not the
        ``TypeError`` a reader would generalize from it.
        """

        class ViaSAMCall(keras.Model):
            """Deliberately-wrong control: routes the training path through SAM.call."""

            def __init__(self, sam: SAM, mode: str, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self.sam = sam
                self.mode = mode

            def build(self, input_shape: Any = None) -> None:
                self.sam.build(None)
                super().build(input_shape)

            def call(self, inputs: Dict[str, Any], training: bool = None) -> Any:
                image = inputs[INPUT_IMAGE]
                if self.mode == "constant_tensor":
                    original_size = ops.convert_to_tensor((IMG_SIZE, IMG_SIZE))
                elif self.mode == "batched_from_data":
                    original_size = inputs["original_size"]
                else:
                    original_size = ops.shape(image)[1:3]
                out = self.sam(
                    {
                        "image": image,
                        "points": (
                            inputs[INPUT_POINT_COORDS],
                            inputs[INPUT_POINT_LABELS],
                        ),
                        "original_size": original_size,
                    },
                    training=training,
                    multimask_output=False,
                )
                return {
                    LOW_RES_LOGITS: out[LOW_RES_LOGITS],
                    IOU_PREDICTIONS: out[IOU_PREDICTIONS],
                }

        keras.utils.set_random_seed(7)
        control = ViaSAMCall(build_reduced_sam(), spelling)
        x = _wrapper_inputs()
        x["original_size"] = np.tile(
            np.array([[IMG_SIZE, IMG_SIZE]], dtype="int32"), (WRAPPER_BATCH, 1)
        )
        y = _wrapper_targets(1)
        with pytest.raises(expected_type, match=expected_message):
            control(x)
            control.compile(
                optimizer="adam",
                loss={LOW_RES_LOGITS: "mse", IOU_PREDICTIONS: "mse"},
            )
            control.fit(x, y, epochs=1, verbose=0, batch_size=WRAPPER_BATCH)


class TestSAMTrainingModelBuild:
    """The two lifecycle gotchas F-5 measured, each with a discriminating control."""

    def test_build_alone_materializes_the_whole_sam(self) -> None:
        """
        ``build()`` -> ``self.sam.build(None)`` materializes every sub-model
        with **no forward pass**, which is what makes a weight restore before
        any call safe.
        """
        keras.utils.set_random_seed(7)
        model = SAMTrainingModel(build_reduced_sam())
        model.build(None)
        assert model.sam.prompt_encoder.built is True
        assert model.sam.prompt_encoder.mask_downscaling.built is True
        assert model.sam.mask_decoder.built is True

    def test_the_control_without_that_line_leaves_the_sam_unbuilt(self) -> None:
        """
        The control that makes the previous test discriminating: with the one
        line removed, ``build()`` leaves every sub-model unbuilt.
        """

        class NoSamBuild(SAMTrainingModel):
            def build(self, input_shape: Any = None) -> None:
                keras.Model.build(self, input_shape)

        keras.utils.set_random_seed(7)
        model = NoSamBuild(build_reduced_sam())
        model.build(None)
        assert model.sam.prompt_encoder.built is False
        assert model.sam.mask_decoder.built is False

    def test_f5_item_10_lazy_mask_prompt_gotcha_does_not_reproduce(self) -> None:
        """
        A plan premise, RE-MEASURED and found FALSE -- recorded as a guard so it
        cannot be silently re-asserted.

        F-5 item 10 predicted that ``PromptEncoder``'s mask-downscaling stack
        builds lazily, so a traced mask-prompt call after the model is built
        would raise "cannot add new elements of state ... to a layer that is
        already built". On this (iteration-1-repaired) code ``PromptEncoder.build``
        builds ``mask_downscaling`` explicitly, so an ordinary forward with
        ``masks=None`` already materializes it and the later traced call
        succeeds -- **with the wrapper's ``self.sam.build(None)`` line removed,
        too**. Both variants are exercised here; if either ever starts raising,
        step 4's refinement loop is affected and this test says so by name.
        """

        class NoSamBuild(SAMTrainingModel):
            def build(self, input_shape: Any = None) -> None:
                keras.Model.build(self, input_shape)

        mask = ops.convert_to_tensor(
            np.random.RandomState(3)
            .uniform(-1.0, 1.0, (WRAPPER_BATCH, 1, LOW_RES, LOW_RES))
            .astype("float32")
        )
        for cls in (SAMTrainingModel, NoSamBuild):
            keras.utils.set_random_seed(7)
            model = cls(build_reduced_sam())
            x = _wrapper_inputs()
            model(x)  # a first forward WITHOUT any mask prompt
            prompt_encoder = model.sam.prompt_encoder

            @tf.function
            def traced_mask_prompt(mask_prompt: Any) -> Any:
                _, dense = prompt_encoder(
                    points=(
                        ops.convert_to_tensor(x[INPUT_POINT_COORDS]),
                        ops.convert_to_tensor(x[INPUT_POINT_LABELS]),
                    ),
                    boxes=None,
                    masks=mask_prompt,
                )
                return dense

            dense = traced_mask_prompt(mask)
            assert tuple(dense.shape) == (
                WRAPPER_BATCH, GRID_SIZE, GRID_SIZE,
                model.sam.prompt_encoder.embed_dim,
            ), f"{cls.__name__} produced an unexpected dense embedding shape"

    def test_seed_generator_exists_on_the_instance(self) -> None:
        """
        Created in ``__init__`` so ``keras.random.*`` inside ``call()`` never
        has to add state to an already-built layer. Step 4 owns the RED proof
        (it is the first step that samples); this pins the object's existence
        and its seed so the constructor cannot quietly drop it first.
        """
        model = SAMTrainingModel(build_reduced_sam(), seed=1234)
        assert isinstance(model.seed_generator, keras.random.SeedGenerator)
        assert model.seed == 1234

    def test_a_non_sam_argument_is_refused(self) -> None:
        with pytest.raises(ValueError, match="requires a SAM instance"):
            SAMTrainingModel(keras.layers.Dense(3))


class TestSAMTrainingModelSerialization:
    """I-4: the `.keras` round-trip is re-proven, not assumed."""

    def test_get_config_round_trip_preserves_the_configuration(self) -> None:
        model = SAMTrainingModel(build_reduced_sam(), multimask_output=True, seed=99)
        restored = SAMTrainingModel.from_config(model.get_config())
        assert restored.multimask_output is True
        assert restored.seed == 99
        assert isinstance(restored.sam, SAM)

    def test_keras_round_trip_reproduces_low_res_logits_value_exactly(self) -> None:
        """
        Save -> load -> forward, on a BUILT model. Measured: 202 weights before
        and after, ``low_res_logits`` max abs diff **0.0**.
        """
        model, x, _ = _built_wrapper()
        reference = ops.convert_to_numpy(model(x)[LOW_RES_LOGITS])
        n_before = len(model.weights)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "sam_training_model.keras")
            model.save(path)
            restored = keras.models.load_model(path)
            n_after = len(restored.weights)
            got = ops.convert_to_numpy(restored(x)[LOW_RES_LOGITS])
        assert n_after == n_before == 202
        assert float(np.max(np.abs(reference - got))) == 0.0

    def test_a_wrapper_without_get_config_cannot_round_trip(self) -> None:
        """
        The control that makes the previous test discriminating: F-5 item 8
        measured that a wrapper holding a ``SAM`` fails to reload without an
        explicit ``get_config``/``from_config`` pair. The exception TYPE is
        asserted, never ``raises(Exception)``.
        """

        @keras.saving.register_keras_serializable(package="sam_test_controls")
        class NoConfigWrapper(SAMTrainingModel):
            def get_config(self) -> Dict[str, Any]:
                config = keras.Model.get_config(self)
                config.pop("sam", None)
                return config

            @classmethod
            def from_config(cls, config: Dict[str, Any]) -> "NoConfigWrapper":
                return cls(**config)

        keras.utils.set_random_seed(7)
        model = NoConfigWrapper(build_reduced_sam())
        x = _wrapper_inputs()
        model(x)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "broken.keras")
            model.save(path)
            with pytest.raises(TypeError, match="missing 1 required positional argument"):
                keras.models.load_model(path)


# ===========================================================================
# Plan step 3 -- the loss wiring, on the real wrapper
# ===========================================================================
class TestIoUSupervisionOutput:
    """
    The `iou_supervision` key: present only when `gt_mask` is supplied, and
    carrying the prediction next to its own stop-gradient target.
    """

    def test_the_key_is_absent_without_a_gt_mask(self) -> None:
        """An inference-shaped call keeps the two-key contract."""
        model, x, _ = _built_wrapper()
        assert set(model(x).keys()) == set(OUTPUT_KEYS)

    def test_the_key_appears_with_a_gt_mask_and_packs_both_values(self) -> None:
        """
        ``[..., 0]`` must be the SAME numbers as ``iou_predictions`` and
        ``[..., 1]`` the achieved IoU -- a packed pair whose halves were swapped
        or duplicated would still have the right shape.
        """
        model, x, _ = _built_wrapper()
        x = dict(x)
        x[INPUT_GT_MASK] = _gt_mask_stack(1)
        out = model(x)
        assert IOU_SUPERVISION in out
        assert tuple(out[IOU_SUPERVISION].shape) == (WRAPPER_BATCH, 1, 2)
        packed = ops.convert_to_numpy(out[IOU_SUPERVISION])
        predicted = ops.convert_to_numpy(out[IOU_PREDICTIONS])
        expected_achieved = ops.convert_to_numpy(
            achieved_mask_iou(out[LOW_RES_LOGITS], ops.convert_to_tensor(x[INPUT_GT_MASK]))
        )
        assert float(np.max(np.abs(packed[..., 0] - predicted))) == 0.0
        assert float(np.max(np.abs(packed[..., 1] - expected_achieved))) == 0.0

    def test_the_achieved_half_carries_no_gradient(self) -> None:
        """
        The target must not train the mask branch to make itself easy to
        predict.

        The assertion is on gradient VALUES, not on ``None``: ``ops.stack``
        keeps ``iou_predictions`` structurally connected to the packed tensor,
        so slicing ``[..., 1]`` yields ZERO gradients for 159 of the 201
        variables rather than ``None`` (measured -- only 42 come back ``None``).
        A ``None``-counting assertion would have failed for a reason that has
        nothing to do with the stop-gradient actually holding.
        """
        model, x, _ = _built_wrapper()
        x = dict(x)
        x[INPUT_GT_MASK] = _gt_mask_stack(1)
        with tf.GradientTape() as tape:
            achieved = model(x)[IOU_SUPERVISION][..., 1]
            loss = tf.reduce_sum(achieved)
        grads = tape.gradient(loss, model.trainable_variables)
        worst = max(
            (float(tf.reduce_max(tf.abs(tf.convert_to_tensor(g)))) for g in grads if g is not None),
            default=0.0,
        )
        assert worst == 0.0, f"the achieved-IoU half leaked gradient (max |g| = {worst})"

    def test_the_predicted_half_does_carry_gradient(self) -> None:
        """The control: without it, the previous test would also pass for a
        packed tensor that was gradient-dead on BOTH halves."""
        model, x, _ = _built_wrapper()
        x = dict(x)
        x[INPUT_GT_MASK] = _gt_mask_stack(1)
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(model(x)[IOU_SUPERVISION][..., 0])
        grads = tape.gradient(loss, model.trainable_variables)
        assert sum(1 for g in grads if g is not None) > 0


class TestEndToEndLossWiring:
    """
    SC-5 applied to the training path: the shipped losses train the real
    wrapper under stock ``fit()`` with a dict ``loss=``, and the dead-component
    probe still turns it RED.
    """

    @staticmethod
    def _compiled() -> Tuple[SAMTrainingModel, Dict[str, Any], Dict[str, Any]]:
        keras.utils.set_random_seed(7)
        model = SAMTrainingModel(build_reduced_sam(), multimask_output=False)
        gt = _gt_mask_stack(1)
        x = _wrapper_inputs()
        x[INPUT_GT_MASK] = gt
        y = {
            LOW_RES_LOGITS: gt,
            IOU_SUPERVISION: np.zeros((WRAPPER_BATCH, 1, 2), dtype="float32"),
        }
        model(x)
        seed_nonzero_weights(model)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss={LOW_RES_LOGITS: SAMMaskLoss(), IOU_SUPERVISION: SAMIoULoss()},
            loss_weights={LOW_RES_LOGITS: 1.0, IOU_SUPERVISION: 1.0},
        )
        return model, x, y

    def test_a_dict_loss_over_a_subset_of_output_keys_trains(self) -> None:
        """
        ``iou_predictions`` is deliberately left unsupervised, so ``loss=`` keys
        a strict SUBSET of the three output keys -- the configuration F-5
        measured working and ``SYSTEM.md:173`` claims is impossible.
        """
        model, x, y = self._compiled()
        report = fit_one_step_moved_variables(model, x, y, batch_size=WRAPPER_BATCH)
        assert report.n_moved > 0, report.summary()
        assert np.isfinite(report.final_loss)

    def test_the_dead_component_probe_still_turns_it_red(self) -> None:
        """The green above is not vacuous."""
        model, x, y = self._compiled()
        with outputs_stop_gradient(model):
            with pytest.raises(ValueError, match=NO_GRADIENTS_MESSAGE):
                fit_one_step_moved_variables(model, x, y, batch_size=WRAPPER_BATCH)

    def test_the_iou_head_moves_only_when_the_iou_loss_is_wired(self) -> None:
        """
        A liveness probe for the IoU term specifically: with the mask loss
        alone, the IoU head's own layers must NOT move; adding ``SAMIoULoss``
        must make them move. A loss term nobody can show is live is decorative.
        """
        def run(with_iou: bool) -> set:
            keras.utils.set_random_seed(7)
            model = SAMTrainingModel(build_reduced_sam(), multimask_output=False)
            gt = _gt_mask_stack(1)
            x = _wrapper_inputs()
            x[INPUT_GT_MASK] = gt
            y: Dict[str, Any] = {LOW_RES_LOGITS: gt}
            model(x)
            seed_nonzero_weights(model)
            losses: Dict[str, Any] = {LOW_RES_LOGITS: SAMMaskLoss()}
            if with_iou:
                losses[IOU_SUPERVISION] = SAMIoULoss()
                # MEASURED: `y_true`'s keys must match the keys `loss=` covers,
                # NOT the model's output keys. Supplying an `iou_supervision`
                # target while `loss=` omits that key raises
                # `ValueError: y_true and y_pred have different structures.`
                # This refines F-5 item 1, which recorded only that `loss=` may
                # key a subset of the OUTPUT keys.
                y[IOU_SUPERVISION] = np.zeros(
                    (WRAPPER_BATCH, 1, 2), dtype="float32"
                )
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=losses
            )
            report = fit_one_step_moved_variables(
                model, x, y, batch_size=WRAPPER_BATCH
            )
            head = model.sam.mask_decoder.iou_prediction_head
            moved = _ids(
                [
                    lookup
                    for label, lookup in zip(
                        variable_labels(model), model.trainable_variables
                    )
                    if label in set(report.moved)
                ]
            )
            return _ids(head.trainable_variables) & moved

        without_iou = run(False)
        with_iou = run(True)
        assert without_iou == set(), (
            "the IoU prediction head moved with no IoU loss wired -- the probe "
            "cannot discriminate"
        )
        assert len(with_iou) > 0, "SAMIoULoss did not move the IoU prediction head"


# ===========================================================================
# Plan step 4 -- the in-`call()` iterative refinement loop.
# ===========================================================================
def _refined(
    rounds: int,
    seed: int = 7,
    cls: Any = None,
    multimask_output: bool = False,
) -> Tuple[SAMTrainingModel, Dict[str, Any], Dict[str, Any]]:
    """
    A built, seeded, compiled refining wrapper plus inputs and a dict ``y_true``.

    ``y_true`` deliberately carries a SINGLE-instance GT stack whatever the
    round count is: the data pipeline must not have to know how many rounds the
    model runs, and ``SAMMaskLoss`` repeats the GT across the concatenated mask
    axis itself.

    The mask axis is pinned to **1 at every configuration**, including
    ``multimask_output=True``, because that is the ONLY stack the shipped
    pipeline can emit: `src/train/sam/data.py`'s `to_training_record` ends in
    `gt_stack = tf.expand_dims(mask, axis=0)`, unconditionally, and neither the
    round count nor the multimask flag reaches it. This helper used to set
    `num_masks = 3 if multimask_output else 1`, a shape no data source
    produces, and that fiction is exactly why the 208-test model gate stayed
    green while `--multimask-output` crashed at the trainer's own defaults.
    Do NOT re-introduce a GT mask axis > 1 here to "make the shapes line up" --
    lining them up in the fixture is what hides the defect.
    """
    keras.utils.set_random_seed(seed)
    model = (cls or SAMTrainingModel)(
        build_reduced_sam(),
        multimask_output=multimask_output,
        num_refinement_rounds=rounds,
    )
    num_masks = 1
    gt = _gt_mask_stack(num_masks)
    x = _wrapper_inputs()
    x[INPUT_GT_MASK] = gt
    y = {
        LOW_RES_LOGITS: gt,
        IOU_SUPERVISION: np.zeros(
            (WRAPPER_BATCH, num_masks, 2), dtype="float32"
        ),
    }
    model(x)
    seed_nonzero_weights(model)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss={LOW_RES_LOGITS: SAMMaskLoss(), IOU_SUPERVISION: SAMIoULoss()},
        loss_weights={LOW_RES_LOGITS: 1.0, IOU_SUPERVISION: 1.0},
    )
    return model, x, y


class ZeroFeedbackControl(SAMTrainingModel):
    """Control: the refinement feedback carries no information at all."""

    def _feedback_mask(self, low_res_logits: Any) -> Any:
        return ops.zeros_like(low_res_logits[:, 0:1])


class NoDetachFeedbackControl(SAMTrainingModel):
    """Control: the same feedback WITHOUT ``ops.stop_gradient``."""

    def _feedback_mask(self, low_res_logits: Any) -> Any:
        return low_res_logits[:, 0:1]


class RecordingFeedbackMixin:
    """Records the feedback tensor the last ``call`` produced."""

    def _feedback_mask(self, low_res_logits: Any) -> Any:
        recorded = super()._feedback_mask(low_res_logits)
        self.recorded_feedback = recorded
        return recorded


class RecordingDetachedControl(RecordingFeedbackMixin, SAMTrainingModel):
    pass


class RecordingAttachedControl(RecordingFeedbackMixin, NoDetachFeedbackControl):
    pass


class CapturingSamplerControl(SAMTrainingModel):
    """Records everything ``_sample_error_points`` was given and returned."""

    def _sample_error_points(
        self, low_res_logits: Any, gt_mask: Any, image_size: Any
    ) -> Any:
        coords, labels = super()._sample_error_points(
            low_res_logits, gt_mask, image_size
        )
        self.captured = (low_res_logits, gt_mask, coords, labels)
        return coords, labels


class TestRefinementShapes:
    """The round count is observable in the output, not merely stored."""

    @pytest.mark.parametrize("rounds", [1, 2, 3])
    def test_the_mask_axis_is_num_masks_times_rounds(self, rounds: int) -> None:
        model, x, _ = _refined(rounds)
        out = model(x)
        assert tuple(out[LOW_RES_LOGITS].shape) == (
            WRAPPER_BATCH, rounds, LOW_RES, LOW_RES
        )
        assert tuple(out[IOU_PREDICTIONS].shape) == (WRAPPER_BATCH, rounds)
        assert tuple(out[IOU_SUPERVISION].shape) == (WRAPPER_BATCH, rounds, 2)

    def test_multimask_multiplies_rather_than_replaces(self) -> None:
        """``M * R``, not ``max(M, R)`` -- a wrong reduction here would still
        satisfy a rank assertion."""
        model, x, _ = _refined(2, multimask_output=True)
        assert tuple(model(x)[LOW_RES_LOGITS].shape) == (
            WRAPPER_BATCH, 6, LOW_RES, LOW_RES
        )

    def test_the_class_default_runs_exactly_one_round(self) -> None:
        """Refinement is opt-in. At the default the wrapper keeps the
        inference-shaped contract every step-2 guard pins."""
        from dl_techniques.models.sam.training_model import (
            DEFAULT_REFINEMENT_ROUNDS,
            TRAINING_REFINEMENT_ROUNDS,
        )

        assert DEFAULT_REFINEMENT_ROUNDS == 1
        assert TRAINING_REFINEMENT_ROUNDS > DEFAULT_REFINEMENT_ROUNDS
        model, x, _ = _built_wrapper()
        assert model.num_refinement_rounds == DEFAULT_REFINEMENT_ROUNDS
        assert tuple(model(x)[LOW_RES_LOGITS].shape)[1] == 1

    def test_zero_rounds_is_refused(self) -> None:
        with pytest.raises(ValueError, match="num_refinement_rounds must be"):
            SAMTrainingModel(build_reduced_sam(), num_refinement_rounds=0)

    def test_the_round_count_survives_a_config_round_trip(self) -> None:
        model, _, _ = _refined(3)
        restored = SAMTrainingModel.from_config(model.get_config())
        assert restored.num_refinement_rounds == 3


class TestRefinementIsLive:
    """
    SC-7. A round that changes nothing is the dead-``iou_head_depth`` defect
    class this plan already fixed once.
    """

    def test_rounds_two_and_three_change_the_prediction(self) -> None:
        """
        Measured, not asserted as a constant: rounds 2 and 3 must differ from
        round 1 both in raw logits AND after thresholding. The binary check is
        the load-bearing half -- a feedback path that only shifted the logits by
        a constant would move ``max|delta|`` while changing no pixel's class,
        and softmax/sigmoid-style shift artefacts are exactly how a previous
        plan mistook uniform garbage for a real delta.
        """
        model, x, _ = _refined(3)
        logits = np.asarray(ops.convert_to_numpy(model(x)[LOW_RES_LOGITS]))
        first_binary = logits[:, 0] > 0.0
        for index in (1, 2):
            delta = float(np.max(np.abs(logits[:, index] - logits[:, 0])))
            agreement = float(np.mean((logits[:, index] > 0.0) == first_binary))
            assert delta > 1e-3, (
                f"round {index + 1} logits are identical to round 1 "
                f"(max|delta|={delta}) -- the refinement is decorative"
            )
            assert agreement < 1.0, (
                f"round {index + 1} thresholds to exactly the same mask as "
                f"round 1 (agreement={agreement}); a constant logit shift would "
                f"pass the max|delta| assertion above and change nothing real"
            )

    def test_zeroing_the_fed_back_logits_changes_the_measured_loss(self) -> None:
        """The feedback dead-component probe (I-5)."""
        live, x, y = _refined(3)
        dead, x_dead, y_dead = _refined(3, cls=ZeroFeedbackControl)

        def evaluate(model: keras.Model, xx: Any, yy: Any) -> float:
            result = model.evaluate(xx, yy, verbose=0, batch_size=WRAPPER_BATCH)
            return float(result[0] if isinstance(result, list) else result)

        live_loss = evaluate(live, x, y)
        dead_loss = evaluate(dead, x_dead, y_dead)
        assert abs(live_loss - dead_loss) > 1e-4, (
            f"zeroing the fed-back mask left the loss at {live_loss} vs "
            f"{dead_loss} -- the refinement feedback contributes nothing"
        )

    def test_a_refining_model_reaches_variables_a_single_round_cannot(
        self,
    ) -> None:
        """
        The strongest liveness evidence available: refinement makes the
        ``mask_downscaling`` stack (the dense mask-prompt path) reachable at
        all. At one round it is 10/10 gradient-dead by construction, because
        nothing ever supplies a mask prompt.
        """
        one, x1, y1 = _refined(1)
        report_one = fit_one_step_moved_variables(
            one, x1, y1, batch_size=WRAPPER_BATCH
        )
        downscaling_one = _ids(
            one.sam.prompt_encoder.mask_downscaling.trainable_variables
        )
        moved_one = _ids(
            [
                variable
                for label, variable in zip(
                    variable_labels(one), one.trainable_variables
                )
                if label in set(report_one.moved)
            ]
        )
        assert downscaling_one & moved_one == set(), (
            "the mask-downscaling stack moved at rounds=1, where no mask prompt "
            "is ever supplied -- the probe cannot discriminate"
        )

        three, x3, y3 = _refined(3)
        report_three = fit_one_step_moved_variables(
            three, x3, y3, batch_size=WRAPPER_BATCH
        )
        downscaling_three = _ids(
            three.sam.prompt_encoder.mask_downscaling.trainable_variables
        )
        moved_three = _ids(
            [
                variable
                for label, variable in zip(
                    variable_labels(three), three.trainable_variables
                )
                if label in set(report_three.moved)
            ]
        )
        assert downscaling_three & moved_three == downscaling_three, (
            f"refinement did not make the whole mask-downscaling stack live: "
            f"{len(downscaling_three & moved_three)} of "
            f"{len(downscaling_three)} moved"
        )
        assert report_three.n_moved > report_one.n_moved, (
            f"{report_three.n_moved} vs {report_one.n_moved}"
        )

    def test_the_dead_component_probe_still_turns_a_refining_model_red(
        self,
    ) -> None:
        model, x, y = _refined(3)
        with outputs_stop_gradient(model):
            with pytest.raises(ValueError, match=NO_GRADIENTS_MESSAGE):
                fit_one_step_moved_variables(
                    model, x, y, batch_size=WRAPPER_BATCH
                )


class TestRefinementStopGradient:
    """
    The detach really detaches -- counted as ``max |g| == 0.0``, not as a
    ``None`` count. ``ops.concatenate`` keeps the tensors structurally
    connected, so gradients come back as exact zeros rather than ``None``
    (D-036 lesson 1: a ``None``-counting assertion fails for a reason unrelated
    to the property).
    """

    @staticmethod
    def _feedback_gradient(cls: Any) -> Tuple[int, int, float]:
        keras.utils.set_random_seed(7)
        model = cls(build_reduced_sam(), num_refinement_rounds=2)
        x = _wrapper_inputs()
        x[INPUT_GT_MASK] = _gt_mask_stack(1)
        model(x)
        seed_nonzero_weights(model)
        variables = list(model.sam.image_encoder.trainable_variables)
        with tf.GradientTape() as tape:
            model(x)
            total = ops.sum(model.recorded_feedback)
        grads = tape.gradient(total, variables)
        n_none = sum(1 for g in grads if g is None)
        largest = max(
            [
                float(np.max(np.abs(ops.convert_to_numpy(g))))
                for g in grads
                if g is not None
            ],
            default=0.0,
        )
        return n_none, len(variables), largest

    def test_the_fed_back_mask_carries_no_gradient_to_the_encoder(self) -> None:
        n_none, total, largest = self._feedback_gradient(RecordingDetachedControl)
        assert largest == 0.0, (
            f"the fed-back mask still carries gradient into the image encoder: "
            f"max|g|={largest} over {total - n_none} non-None gradients"
        )

    def test_the_control_without_stop_gradient_does_carry_gradient(self) -> None:
        """
        RED proof for the assertion above. Without this control, a probe that
        measured the wrong variables, or a tape that captured nothing, would
        report ``max|g| == 0.0`` and certify a detach that was never there.
        """
        n_none, total, largest = self._feedback_gradient(RecordingAttachedControl)
        assert n_none == 0, f"{n_none}/{total} gradients were None"
        assert largest > 0.0, (
            "removing ops.stop_gradient left max|g| at 0.0 -- this control "
            "cannot RED-prove the guard above"
        )


class TestRefinementPointSampling:
    """The sampled points must land in the regions they claim to come from."""

    @staticmethod
    def _capture(seed: int = 7) -> Tuple[np.ndarray, ...]:
        keras.utils.set_random_seed(seed)
        model = CapturingSamplerControl(build_reduced_sam(), num_refinement_rounds=2)
        x = _wrapper_inputs()
        x[INPUT_GT_MASK] = _gt_mask_stack(1)
        model(x)
        seed_nonzero_weights(model)
        model(x)
        return tuple(
            np.asarray(ops.convert_to_numpy(t)) for t in model.captured
        )

    def test_the_foreground_point_lands_in_a_false_negative_pixel(
        self,
    ) -> None:
        logits, gt, coords, labels = self._capture()
        predicted = (logits[:, 0] > 0.0).astype("float32")
        truth = (gt[:, 0] > 0.5).astype("float32")
        false_negative = truth * (1.0 - predicted)
        false_positive = (1.0 - truth) * predicted
        scale = IMG_SIZE / LOW_RES

        # Premise check: the probe is worthless if either region is empty --
        # the label rule would then legitimately emit a padding point and every
        # containment assertion below would be vacuous.
        assert false_negative.sum() > 0 and false_positive.sum() > 0

        for row in range(WRAPPER_BATCH):
            for slot, region, expected_label in (
                (0, false_negative, 1),
                (1, false_positive, 0),
            ):
                x_coord, y_coord = coords[row, slot]
                col = int(round((x_coord + 0.5) / scale - 0.5))
                grid_row = int(round((y_coord + 0.5) / scale - 0.5))
                assert region[row, grid_row, col] > 0, (
                    f"batch {row} slot {slot}: sampled cell "
                    f"({grid_row}, {col}) is NOT in its error region"
                )
                assert int(labels[row, slot]) == expected_label

    def test_an_empty_error_region_is_labelled_padding_not_sampled(
        self,
    ) -> None:
        """
        The edge case the plan names explicitly. A perfect prediction leaves
        both error regions empty; the draw then degenerates to uniform, and the
        LABEL is what stops that pixel from being taught as object interior.
        """
        keras.utils.set_random_seed(7)
        model = SAMTrainingModel(build_reduced_sam(), num_refinement_rounds=2)
        model(_wrapper_inputs())
        empty = ops.zeros((WRAPPER_BATCH, LOW_RES, LOW_RES))
        _, labels = model._sample_from_region(empty, 1, (IMG_SIZE, IMG_SIZE))
        assert np.all(np.asarray(ops.convert_to_numpy(labels)) == -1)

    def test_a_non_empty_region_is_not_labelled_padding(self) -> None:
        """Control: without it, a sampler that labelled EVERYTHING ``-1``
        would pass the test above."""
        keras.utils.set_random_seed(7)
        model = SAMTrainingModel(build_reduced_sam(), num_refinement_rounds=2)
        model(_wrapper_inputs())
        region = np.zeros((WRAPPER_BATCH, LOW_RES, LOW_RES), dtype="float32")
        region[:, 10:20, 30:40] = 1.0
        coords, labels = model._sample_from_region(
            ops.convert_to_tensor(region), 1, (IMG_SIZE, IMG_SIZE)
        )
        assert np.all(np.asarray(ops.convert_to_numpy(labels)) == 1)
        scale = IMG_SIZE / LOW_RES
        drawn = np.asarray(ops.convert_to_numpy(coords))
        for row in range(WRAPPER_BATCH):
            x_coord, y_coord = drawn[row, 0]
            col = int(round((x_coord + 0.5) / scale - 0.5))
            grid_row = int(round((y_coord + 0.5) / scale - 0.5))
            assert region[row, grid_row, col] > 0

    def test_the_sampler_is_not_degenerate(self) -> None:
        """
        A sampler that always returned the same pixel would satisfy every
        containment assertion above. Draw many times from one region and
        require more than one distinct cell.
        """
        keras.utils.set_random_seed(7)
        model = SAMTrainingModel(build_reduced_sam(), num_refinement_rounds=2)
        model(_wrapper_inputs())
        region = np.zeros((WRAPPER_BATCH, LOW_RES, LOW_RES), dtype="float32")
        region[:, 10:30, 30:50] = 1.0
        tensor = ops.convert_to_tensor(region)
        seen = set()
        for _ in range(40):
            coords, _ = model._sample_from_region(tensor, 1, (IMG_SIZE, IMG_SIZE))
            drawn = np.asarray(ops.convert_to_numpy(coords))
            for row in range(WRAPPER_BATCH):
                seen.add(tuple(drawn[row, 0].tolist()))
        assert len(seen) > 5, f"only {len(seen)} distinct cells over 80 draws"


class TestSeedGeneratorGotcha:
    """
    The plan carried this as an UNPROVEN gotcha. RED-proved here -- and the
    inherited statement of it turned out to be wrong.
    """

    def test_creating_the_seed_generator_inside_call_raises(self) -> None:
        """
        The lazy spelling really does raise, and this class is exactly the shape
        that makes it raise: it defines ``build()``, so ``super().build()`` runs
        ``Layer._lock_state`` before ``call`` ever executes.
        """

        class LazySeedGeneratorControl(SAMTrainingModel):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                del self.seed_generator

            def _sample_from_region(
                self, region: Any, label: int, image_size: Any
            ) -> Any:
                if not hasattr(self, "seed_generator"):
                    self.seed_generator = keras.random.SeedGenerator(
                        seed=self.seed
                    )
                return super()._sample_from_region(region, label, image_size)

        keras.utils.set_random_seed(7)
        model = LazySeedGeneratorControl(
            build_reduced_sam(), num_refinement_rounds=2
        )
        x = _wrapper_inputs()
        x[INPUT_GT_MASK] = _gt_mask_stack(1)
        with pytest.raises(
            ValueError, match="cannot add new elements of state"
        ):
            model(x)

    def test_f5_item_10_bare_keras_random_gotcha_does_not_reproduce(
        self,
    ) -> None:
        """
        **Premise falsified.** F-5 item 10 recorded that a BARE ``keras.random.*``
        inside ``call()`` raises the same "already built" error, and the plan
        carried it as a hard constraint. Measured on keras 3.8.0 it does not:
        the bare call routes to the module-level global seed generator and adds
        no state to the layer. The explicit ``SeedGenerator`` is kept for the
        two reasons that ARE measured -- the lazy spelling above really raises,
        and an explicit generator makes the sampling reproducible from ``seed``
        and serializable. This test pins the falsification so the wrong
        justification cannot be silently restored.
        """

        class BareRandomProbe(keras.Model):
            def __init__(self, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self.dense = keras.layers.Dense(4)

            def build(self, input_shape: Any) -> None:
                self.dense.build(input_shape)
                super().build(input_shape)

            def call(self, inputs: Any, training: bool = None) -> Any:
                drawn = keras.random.categorical(
                    ops.zeros((ops.shape(inputs)[0], 8)), 1
                )
                return self.dense(inputs) + ops.cast(drawn, "float32") * 0.0

        rng = np.random.RandomState(0)
        x = rng.uniform(size=(8, 3)).astype("float32")
        y = rng.uniform(size=(8, 4)).astype("float32")
        model = BareRandomProbe()
        model(x)
        model(x)
        model.compile(optimizer="adam", loss="mse")
        history = model.fit(x, y, epochs=2, verbose=0)
        assert np.isfinite(history.history["loss"][-1])

    def test_the_seed_generator_makes_the_sampling_reproducible(self) -> None:
        """The property the explicit generator is actually kept for."""

        def draw(seed: int) -> np.ndarray:
            keras.utils.set_random_seed(11)
            model = SAMTrainingModel(
                build_reduced_sam(), num_refinement_rounds=2, seed=seed
            )
            model(_wrapper_inputs())
            region = np.zeros(
                (WRAPPER_BATCH, LOW_RES, LOW_RES), dtype="float32"
            )
            region[:, 5:60, 5:60] = 1.0
            coords, _ = model._sample_from_region(
                ops.convert_to_tensor(region), 1, (IMG_SIZE, IMG_SIZE)
            )
            return np.asarray(ops.convert_to_numpy(coords))

        assert np.array_equal(draw(123), draw(123))
        assert not np.array_equal(draw(123), draw(999))


class TestRefinementSerialization:
    """What the ``.keras`` round-trip does and does NOT preserve once the
    model samples."""

    def test_round_one_is_still_value_exact_after_a_round_trip(self) -> None:
        """
        The round-1 slice is deterministic (nothing has been sampled yet), so it
        must reproduce exactly -- that is the part of the step-2 guarantee that
        survives refinement.
        """
        model, x, _ = _refined(3)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "refining.keras")
            model.save(path)
            restored = keras.models.load_model(path)
            reference = np.asarray(
                ops.convert_to_numpy(model(x)[LOW_RES_LOGITS])
            )
            loaded = np.asarray(
                ops.convert_to_numpy(restored(x)[LOW_RES_LOGITS])
            )
        assert restored.num_refinement_rounds == 3
        assert float(np.max(np.abs(loaded[:, 0] - reference[:, 0]))) == 0.0

    def test_the_later_rounds_are_NOT_value_exact_and_that_is_declared(
        self,
    ) -> None:
        """
        Declared, not hidden. Rounds 2+ depend on a random draw whose generator
        state has advanced differently in the two processes-of-one, so the
        step-2 "max abs diff 0.0 on low_res_logits" guarantee does NOT extend to
        them. Asserting exactness here would be asserting something false; this
        test pins the honest statement instead, so a future reader does not
        "fix" a round-trip that is not broken.
        """
        model, x, _ = _refined(3)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "refining.keras")
            model.save(path)
            restored = keras.models.load_model(path)
            reference = np.asarray(
                ops.convert_to_numpy(model(x)[LOW_RES_LOGITS])
            )
            loaded = np.asarray(
                ops.convert_to_numpy(restored(x)[LOW_RES_LOGITS])
            )
        assert float(np.max(np.abs(loaded[:, 1:] - reference[:, 1:]))) > 0.0
