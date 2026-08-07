"""Tests for SAM 3's encoder query-selection head
(`models/sam3/query_selection.py`).

Four design choices in this file are load-bearing.

**The float64 oracle is written from the SPEC, not from the layer.** It
re-derives the whole expression -- both ReLU MLP stacks, the row-major anchor
grid, ``inverse_sigmoid`` and the final ``sigmoid`` -- in double precision from
the layer's WEIGHTS only. An oracle that calls the implementation is a
tautology, and this package has already shipped one green oracle certifying the
wrong quantity.

**The flatten order is PROVEN, not assumed.** ``TestFlattenOrder`` reads
``Sam3Image._flatten`` itself on a NON-SQUARE grid and checks that memory
position ``j`` is ``(row=j // W, col=j % W)``, then checks the head's anchor
grid against that same reading. On the square grids every shipped variant uses,
a transposed anchor grid is shape-compatible, finite and plausible -- the exact
silent-defect class ``necks.py``'s D-096 exists for.

**Selection image-dependence is measured at the SHIPPED step-0 init.** With the
box stack's last projection zero-initialized, every proposal box is exactly its
grid anchor, so the ONLY thing that can make ``selected_boxes`` vary across
images is WHICH positions the objectness head selects. That makes the
across-image standard deviation of ``selected_boxes`` a direct readout of the
mechanism this whole plan exists to create -- and it makes the dead-component
reading exactly ``0.0`` rather than merely small.

**The degenerate mode is named, not hoped away.** ``ops.top_k`` breaks ties by
ascending index, so a dead objectness head selects positions ``0 .. k - 1`` for
every image: right shapes, right dtypes, plausible across-QUERY spread, and
image-INDEPENDENT. ``_assert_selection_is_image_dependent`` therefore checks
three things at once (the indices differ across images, they are not the
degenerate sequence, and the across-image box spread is non-trivial), and
``TestSelectionGuardIsRedProven`` fires it with a dead-component injection.
"""

import keras
import numpy as np
import pytest
from keras import ops
from typing import Any, Dict, List

from dl_techniques.models.sam3.decoder import Sam3TransformerDecoder
from dl_techniques.models.sam3.model_misc import Sam3DotProductScoring
from dl_techniques.models.sam3.query_selection import (
    DEFAULT_ANCHOR_SIZE, Sam3EncoderQuerySelection)
from dl_techniques.models.sam3.sam3_image import Sam3Image

# ---------------------------------------------------------------------
# tiny geometry
# ---------------------------------------------------------------------

#: A deliberately NON-SQUARE grid: on a square one a transposed anchor layout is
#: shape-compatible and silently wrong.
TINY = dict(d_model=8, num_queries=3, feat_size=(2, 8), mlp_depth=3)

BATCH = 4
POSITIONS = TINY["feat_size"][0] * TINY["feat_size"][1]

#: The measured step-0 identity error of `sigmoid(inverse_sigmoid(anchor))` in
#: float32 is 5.96e-08 (max over the shipped anchor values); 1e-6 clears that by
#: more than an order while every mutation below misses by 1e-2 or more.
IDENTITY_TOL = 1e-6

#: The oracle runs in float64 against a float32 layer; GPU 1 routes float32
#: matmuls through TF32, so a tolerance derived on CPU is wrong on GPU. 1e-4
#: clears both regimes while the anchor mutation misses by ~0.1.
ORACLE_TOL = 1e-4


def _randomize(head: Sam3EncoderQuerySelection, seed: int = 0,
               skip_zero_init: bool = True) -> None:
    """Give the head non-trivial weights -- kernels AND biases.

    Args:
        head: A built head.
        seed: RNG seed.
        skip_zero_init: When True the box stack's LAST projection is left at
            its shipped zero initialization, i.e. the step-0 configuration.

    Returns:
        None.
    """
    rng = np.random.default_rng(seed)
    frozen = set()
    if skip_zero_init:
        frozen = {id(w) for w in head.box_head[-1].weights}
    for weight in head.weights:
        if id(weight) in frozen:
            continue
        weight.assign(rng.normal(0.0, 0.5, size=weight.shape).astype("float32"))


@pytest.fixture()
def head() -> Sam3EncoderQuerySelection:
    layer = Sam3EncoderQuerySelection(**TINY)
    layer.build((BATCH, POSITIONS, TINY["d_model"]))
    _randomize(layer)
    return layer


def _memory(seed: int = 3, batch: int = BATCH) -> np.ndarray:
    """Per-image-DIFFERENT memory; every image is an independent draw."""
    rng = np.random.default_rng(seed)
    return rng.normal(
        size=(batch, POSITIONS, TINY["d_model"])).astype("float32")


# ---------------------------------------------------------------------
# the float64 oracle -- written from the mechanism, never from the layer
# ---------------------------------------------------------------------


def _mlp64(stack: List[Any], x: np.ndarray) -> np.ndarray:
    """Run a flat Dense stack in float64: ReLU everywhere but the last."""
    for index, dense in enumerate(stack):
        kernel = np.asarray(dense.kernel).astype("float64")
        bias = np.asarray(dense.bias).astype("float64")
        x = x @ kernel + bias
        if index < len(stack) - 1:
            x = np.maximum(x, 0.0)
    return x


def _anchors64(feat_size, anchor_size: float) -> np.ndarray:
    """The anchor grid, re-derived from the SPEC in row-major order."""
    height, width = feat_size
    anchors = np.zeros((height * width, 4), dtype="float64")
    for row in range(height):
        for col in range(width):
            anchors[row * width + col] = (
                (col + 0.5) / width, (row + 0.5) / height,
                anchor_size, anchor_size)
    return anchors[None, ...]


def _inverse_sigmoid64(x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.log(np.maximum(x, eps) / np.maximum(1.0 - x, eps))


def _oracle(head: Sam3EncoderQuerySelection,
            memory: np.ndarray) -> Dict[str, np.ndarray]:
    """The whole head in float64, from its weights and the written spec."""
    x = memory.astype("float64")
    objectness = _mlp64(head.objectness_head, x)
    delta = _mlp64(head.box_head, x)
    anchors = _anchors64(head.feat_size, head.anchor_size)
    boxes = 1.0 / (1.0 + np.exp(-(delta + _inverse_sigmoid64(anchors))))
    order = np.argsort(-objectness[..., 0], axis=1, kind="stable")
    indices = order[:, :head.num_queries]
    return {
        "objectness": objectness,
        "boxes": boxes,
        "indices": indices,
        "selected_boxes": np.take_along_axis(
            boxes, indices[..., None], axis=1),
        "anchors": anchors,
    }


# ---------------------------------------------------------------------
# the named assertions -- each one lives in ONE place so its RED proof can
# report WHICH assertion fired
# ---------------------------------------------------------------------


def _assert_boxes_match_anchor_oracle(actual: np.ndarray,
                                      expected: np.ndarray) -> None:
    """THE anchor-arithmetic assertion.

    Args:
        actual: The layer's `boxes`.
        expected: The float64 oracle's `boxes`.

    Returns:
        None.

    Raises:
        AssertionError: If the two disagree beyond `ORACLE_TOL`.
    """
    delta = float(np.max(np.abs(actual.astype("float64") - expected)))
    assert delta <= ORACLE_TOL, (
        f"anchor arithmetic mismatch: max abs delta {delta:.3e} exceeds "
        f"{ORACLE_TOL:.1e}. `boxes` must be "
        f"sigmoid(delta + inverse_sigmoid(anchor_j)) with anchor_j the "
        f"row-major grid anchor of memory position j")


def _assert_selection_is_image_dependent(indices: np.ndarray,
                                         selected_boxes: np.ndarray) -> None:
    """THE image-dependence assertion -- the property this head exists for.

    Three checks, because the degenerate mode passes any one of them alone: a
    dead objectness head still returns the right shapes, an ordinary
    across-QUERY spread, and `ops.top_k`'s ascending-index tie-break, i.e. the
    same positions `0 .. k - 1` for every image.

    Args:
        indices: `(batch, num_queries)` selected memory positions.
        selected_boxes: `(batch, num_queries, 4)` their boxes.

    Returns:
        None.

    Raises:
        AssertionError: If the selection does not vary with the image.
    """
    batch, num_queries = indices.shape
    degenerate = np.tile(np.arange(num_queries), (batch, 1))
    assert not np.array_equal(indices, degenerate), (
        f"selection is the DEGENERATE sequence 0..{num_queries - 1} for every "
        f"image, which is what `ops.top_k`'s ascending-index tie-break returns "
        f"for an objectness field that carries no image signal")
    distinct = len({tuple(row) for row in indices.tolist()})
    assert distinct > 1, (
        f"all {batch} images selected the SAME {num_queries} memory positions "
        f"{indices[0].tolist()}: the selection is image-INDEPENDENT")
    spread = float(np.mean(np.std(selected_boxes, axis=0)))
    assert spread > 1e-4, (
        f"across-image std of selected_boxes is {spread:.3e}: the selected "
        f"anchor SET does not vary with the image, so the boxes entering the "
        f"decoder are image-independent by construction")


def _assert_step_zero_boxes_are_exactly_the_anchors(
        boxes: np.ndarray, anchors: np.ndarray) -> None:
    """THE step-0 identity assertion (the zero-init consequence).

    Args:
        boxes: The layer's `boxes` at the shipped initialization.
        anchors: The oracle's anchor grid, broadcast over the batch.

    Returns:
        None.

    Raises:
        AssertionError: If any proposal has been displaced from its anchor.
    """
    delta = float(np.max(np.abs(boxes.astype("float64") - anchors)))
    assert delta <= IDENTITY_TOL, (
        f"step-0 proposals are displaced from their anchors by {delta:.3e} "
        f"(> {IDENTITY_TOL:.1e}): the box stack's LAST projection must be "
        f"zero-initialized so that sigmoid(0 + inverse_sigmoid(anchor)) is "
        f"exactly the anchor before any gradient step (D-112's precedent)")


# ---------------------------------------------------------------------
# construction and validation
# ---------------------------------------------------------------------


class TestConstruction:

    def test_more_queries_than_positions_raises_naming_both_numbers(self):
        with pytest.raises(ValueError) as excinfo:
            Sam3EncoderQuerySelection(d_model=8, num_queries=32,
                                      feat_size=(4, 4))
        message = str(excinfo.value)
        assert "16" in message and "32" in message, message

    def test_the_small_variant_geometry_is_accepted(self):
        """256 positions and 32 queries -- the case the raise must NOT fire on."""
        head = Sam3EncoderQuerySelection(d_model=16, num_queries=32,
                                         feat_size=(16, 16))
        assert head.num_positions == 256

    def test_exactly_as_many_queries_as_positions_is_allowed(self):
        head = Sam3EncoderQuerySelection(d_model=8, num_queries=16,
                                         feat_size=(4, 4))
        assert head.num_queries == head.num_positions

    @pytest.mark.parametrize("kwargs", [
        dict(d_model=0), dict(num_queries=0), dict(feat_size=(4,)),
        dict(feat_size=(4, 0)), dict(anchor_size=0.0), dict(anchor_size=1.0),
        dict(anchor_size=-0.5), dict(mlp_depth=0),
    ])
    def test_invalid_configuration_raises(self, kwargs):
        config = dict(d_model=8, num_queries=2, feat_size=(4, 4))
        config.update(kwargs)
        with pytest.raises(ValueError):
            Sam3EncoderQuerySelection(**config)

    def test_the_shipped_anchor_size_is_the_recorded_measurement(self):
        """D-005: 0.1776, the seed-pooled mean of sqrt(w * h) over TRAIN GT."""
        assert round(DEFAULT_ANCHOR_SIZE, 4) == 0.1776
        assert round(Sam3EncoderQuerySelection(
            d_model=8, num_queries=2, feat_size=(4, 4)).anchor_size, 4) == 0.1776

    def test_the_sub_layer_stores_are_flat_not_nested(self, head):
        """D-098: nesting them silently loses weights on `.keras` round trip."""
        for stack in (head.objectness_head, head.box_head):
            assert stack
            assert all(isinstance(l, keras.layers.Layer) for l in stack)
            assert not any(isinstance(l, (list, tuple)) for l in stack)

    def test_the_stacks_come_from_the_shared_decoder_trio(self, head):
        """No fourth hand-rolled Dense-stack builder: same shape contract."""
        reference = Sam3TransformerDecoder._make_mlp(
            TINY["mlp_depth"], TINY["d_model"], 4, "probe", zero_init_last=True)
        assert [d.units for d in head.box_head] == [d.units for d in reference]
        assert [d.activation for d in head.box_head] == [
            d.activation for d in reference]

    def test_the_parameter_count_matches_the_structure(self, head):
        """Written from the STRUCTURE, not read off the layer."""
        width = TINY["d_model"]
        stack = 2 * (width * width + width)          # two ReLU projections
        expected = (stack + width * 1 + 1) + (stack + width * 4 + 4)
        assert head.count_params() == expected

    def test_build_rejects_a_memory_that_disagrees_with_feat_size(self):
        layer = Sam3EncoderQuerySelection(**TINY)
        with pytest.raises(ValueError, match="positions"):
            layer.build((BATCH, POSITIONS + 1, TINY["d_model"]))

    def test_build_rejects_a_wrong_width_and_a_wrong_rank(self):
        for shape, pattern in (((BATCH, POSITIONS, 5), "width"),
                               ((BATCH, POSITIONS), "batch, H")):
            with pytest.raises(ValueError, match=pattern):
                Sam3EncoderQuerySelection(**TINY).build(shape)


# ---------------------------------------------------------------------
# forward contract
# ---------------------------------------------------------------------


class TestForwardContract:

    def test_every_key_and_shape(self, head):
        out = head(_memory())
        assert set(out) == {"objectness", "boxes", "selected_boxes",
                            "selected_objectness", "indices"}
        assert tuple(out["objectness"].shape) == (BATCH, POSITIONS, 1)
        assert tuple(out["boxes"].shape) == (BATCH, POSITIONS, 4)
        assert tuple(out["selected_boxes"].shape) == (BATCH, 3, 4)
        assert tuple(out["selected_objectness"].shape) == (BATCH, 3, 1)
        assert tuple(out["indices"].shape) == (BATCH, 3)

    def test_compute_output_shape_agrees_with_the_forward_pass(self, head):
        declared = head.compute_output_shape((None, POSITIONS, TINY["d_model"]))
        out = head(_memory())
        assert set(declared) == set(out)
        for key, shape in declared.items():
            assert tuple(out[key].shape)[1:] == tuple(shape)[1:], key
            assert shape[0] is None, key

    def test_indices_are_int32_and_inside_the_grid(self, head):
        out = head(_memory())
        indices = np.asarray(out["indices"])
        assert indices.dtype == np.int32
        assert indices.min() >= 0 and indices.max() < POSITIONS

    def test_boxes_are_normalized_and_finite(self, head):
        boxes = np.asarray(head(_memory())["boxes"])
        assert np.all(np.isfinite(boxes))
        assert boxes.min() >= 0.0 and boxes.max() <= 1.0

    def test_selected_objectness_is_the_gathered_objectness(self, head):
        out = head(_memory())
        gathered = np.take_along_axis(
            np.asarray(out["objectness"]),
            np.asarray(out["indices"])[..., None], axis=1)
        assert float(np.max(np.abs(
            gathered - np.asarray(out["selected_objectness"])))) == 0.0

    def test_selected_boxes_are_the_gathered_boxes(self, head):
        out = head(_memory())
        gathered = np.take_along_axis(
            np.asarray(out["boxes"]),
            np.repeat(np.asarray(out["indices"])[..., None], 4, axis=-1), axis=1)
        assert float(np.max(np.abs(
            gathered - np.asarray(out["selected_boxes"])))) == 0.0

    def test_training_flag_does_not_change_the_output(self, head):
        memory = _memory()
        eager = np.asarray(head(memory, training=False)["boxes"])
        trained = np.asarray(head(memory, training=True)["boxes"])
        assert float(np.max(np.abs(eager - trained))) == 0.0

    def test_gradients_reach_every_weight(self, head):
        import tensorflow as tf
        memory = tf.constant(_memory())
        with tf.GradientTape() as tape:
            out = head(memory, training=True)
            total = (ops.sum(out["objectness"])
                     + ops.sum(out["selected_boxes"]))
        grads = tape.gradient(total, head.trainable_weights)
        assert len(grads) == len(head.trainable_weights)
        assert all(g is not None for g in grads)


# ---------------------------------------------------------------------
# the flatten order -- PROVEN against `Sam3Image._flatten`, not assumed
# ---------------------------------------------------------------------


class TestFlattenOrder:

    def test_flatten_is_row_major_on_a_non_square_grid(self):
        """The empirical proof: position j of the memory is (j // W, j % W)."""
        height, width = TINY["feat_size"]
        feature = np.arange(height * width, dtype="float32").reshape(
            1, height, width, 1)
        flat = np.asarray(Sam3Image._flatten(ops.convert_to_tensor(feature)))
        assert flat.shape == (1, height * width, 1)
        for row in range(height):
            for col in range(width):
                assert flat[0, row * width + col, 0] == feature[0, row, col, 0]

    def test_the_anchor_grid_uses_that_same_order(self, head):
        height, width = TINY["feat_size"]
        anchors = head._anchor_grid[0]
        for index in range(POSITIONS):
            row, col = divmod(index, width)
            np.testing.assert_allclose(
                anchors[index],
                [(col + 0.5) / width, (row + 0.5) / height,
                 head.anchor_size, head.anchor_size], atol=1e-7)

    def test_a_column_major_grid_would_be_a_different_tensor(self, head):
        """The control: the transposed layout is NOT accidentally equal."""
        height, width = TINY["feat_size"]
        transposed = _anchors64((width, height), head.anchor_size)[0]
        assert transposed.shape == head._anchor_grid[0].shape
        assert not np.allclose(transposed, head._anchor_grid[0])

    def test_the_centres_span_the_unit_square(self, head):
        anchors = head._anchor_grid[0]
        height, width = TINY["feat_size"]
        assert set(np.round(anchors[:, 0], 6)) == {
            round((c + 0.5) / width, 6) for c in range(width)}
        assert set(np.round(anchors[:, 1], 6)) == {
            round((r + 0.5) / height, 6) for r in range(height)}


# ---------------------------------------------------------------------
# anchor arithmetic
# ---------------------------------------------------------------------


class TestAnchorArithmetic:

    def test_boxes_match_the_float64_oracle(self, head):
        memory = _memory()
        _assert_boxes_match_anchor_oracle(
            np.asarray(head(memory)["boxes"]), _oracle(head, memory)["boxes"])

    def test_the_oracle_probe_is_unsaturated(self, head):
        """The probe must not sit where every candidate agrees."""
        memory = _memory()
        boxes = np.asarray(head(memory)["boxes"])
        assert float(np.std(boxes)) > 1e-2, float(np.std(boxes))

    def test_step_zero_proposals_are_exactly_their_anchors(self, head):
        """Zero-init consequence: measured max delta 5.96e-08 in float32."""
        memory = _memory()
        anchors = _anchors64(head.feat_size, head.anchor_size)
        _assert_step_zero_boxes_are_exactly_the_anchors(
            np.asarray(head(memory)["boxes"]),
            np.broadcast_to(anchors, (BATCH, POSITIONS, 4)))

    def test_the_step_zero_identity_is_not_vacuous(self, head):
        """A non-zero last projection displaces every anchor, by a lot."""
        last = head.box_head[-1]
        last.kernel.assign(np.full(last.kernel.shape, 0.3, dtype="float32"))
        boxes = np.asarray(head(_memory())["boxes"])
        anchors = np.broadcast_to(
            _anchors64(head.feat_size, head.anchor_size),
            (BATCH, POSITIONS, 4))
        delta = float(np.max(np.abs(boxes - anchors)))
        assert delta > 1e-2, delta
        with pytest.raises(AssertionError, match="displaced from their anchors"):
            _assert_step_zero_boxes_are_exactly_the_anchors(boxes, anchors)


class TestAnchorGuardIsRedProven:
    """SC-D proof (ii): a CONSTANT anchor grid must make the oracle FAIL."""

    def test_a_constant_anchor_grid_makes_the_oracle_assertion_fire(
            self, head, monkeypatch):
        memory = _memory()
        expected = _oracle(head, memory)["boxes"]
        constant = np.full_like(head._anchor_grid, 0.5)
        monkeypatch.setattr(head, "_anchor_grid", constant)
        mutated = np.asarray(head(memory)["boxes"])
        assert np.all(np.isfinite(mutated)), (
            "the mutation must produce a PLAUSIBLE tensor, not an obviously "
            "broken one")
        with pytest.raises(AssertionError, match="anchor arithmetic mismatch"):
            _assert_boxes_match_anchor_oracle(mutated, expected)

    def test_the_oracle_is_green_again_once_the_grid_is_restored(self, head):
        """The control -- otherwise the RED reading could be the fixture's."""
        memory = _memory()
        _assert_boxes_match_anchor_oracle(
            np.asarray(head(memory)["boxes"]), _oracle(head, memory)["boxes"])


# ---------------------------------------------------------------------
# top-k semantics
# ---------------------------------------------------------------------


class _FixedField:
    """A stand-in for a whole Dense stack: returns a designed field."""

    def __init__(self, field: np.ndarray) -> None:
        self.field = ops.convert_to_tensor(field.astype("float32"))

    def __call__(self, x: Any) -> Any:
        del x
        return self.field


class TestTopKSemantics:
    """SC-D proof (iii): exact index equality on a KNOWN argsort."""

    #: A hand-built objectness field with no ties. Row 0's descending order is
    #: 5, 1, 7, ... and row 1's is 2, 6, 0, ... -- computed by hand below.
    FIELD = np.array([
        [0.1, 0.9, 0.2, 0.3, 0.0, 1.5, 0.4, 0.8, 0.15, 0.05, 0.25, 0.35,
         0.45, 0.55, 0.65, 0.75],
        [0.7, 0.6, 2.0, 0.1, 0.2, 0.3, 1.1, 0.4, 0.05, 0.15, 0.25, 0.35,
         0.45, 0.5, 0.55, 0.65],
    ], dtype="float32")[..., None]

    @staticmethod
    def _run(field: np.ndarray) -> np.ndarray:
        layer = Sam3EncoderQuerySelection(**TINY)
        layer.build((field.shape[0], POSITIONS, TINY["d_model"]))
        layer.objectness_head = [_FixedField(field)]
        memory = _memory(batch=field.shape[0])
        return np.asarray(layer(memory)["indices"])

    def test_indices_equal_the_known_argsort_exactly(self):
        known = np.argsort(-self.FIELD[..., 0], axis=1,
                           kind="stable")[:, :TINY["num_queries"]]
        assert known[0].tolist() == [5, 1, 7]      # hand-checked
        assert known[1].tolist() == [2, 6, 0]      # hand-checked
        assert self._run(self.FIELD).tolist() == known.tolist()

    def test_a_shuffled_field_moves_the_indices_the_shuffled_way(self):
        """The control: the same values, permuted, must select the images."""
        rng = np.random.default_rng(11)
        permutation = rng.permutation(POSITIONS)
        # `shuffled[:, i] == FIELD[:, permutation[i]]`, so `permutation` maps a
        # shuffled index back to the grid cell it came from.
        shuffled = self.FIELD[:, permutation]
        actual = self._run(shuffled)
        expected = np.argsort(-shuffled[..., 0], axis=1,
                              kind="stable")[:, :TINY["num_queries"]]
        assert actual.tolist() == expected.tolist()
        assert actual.tolist() != self._run(self.FIELD).tolist()
        # And the permuted positions really are the same GRID cells.
        assert permutation[actual].tolist() == [[5, 1, 7], [2, 6, 0]]

    def test_an_all_equal_field_selects_the_degenerate_prefix(self):
        """`ops.top_k`'s tie-break, pinned: the vacuity mode this suite fears."""
        flat = np.zeros_like(self.FIELD)
        assert self._run(flat).tolist() == [[0, 1, 2], [0, 1, 2]]


# ---------------------------------------------------------------------
# image dependence -- the property the whole plan exists to create
# ---------------------------------------------------------------------


class TestSelectionIsImageDependent:

    def test_at_the_shipped_step_zero_init_the_selection_reads_the_image(
            self, head):
        out = head(_memory())
        _assert_selection_is_image_dependent(
            np.asarray(out["indices"]), np.asarray(out["selected_boxes"]))

    def test_selected_boxes_are_anchors_of_the_selected_positions(self, head):
        """Step 0: the only image-dependence is WHICH anchors were picked."""
        out = head(_memory())
        anchors = _anchors64(head.feat_size, head.anchor_size)[0]
        picked = anchors[np.asarray(out["indices"])]
        assert float(np.max(np.abs(
            np.asarray(out["selected_boxes"]) - picked))) <= IDENTITY_TOL

    def test_two_identical_images_select_identically(self, head):
        """Determinism control: the variation must come from the INPUT."""
        memory = _memory()
        memory[1] = memory[0]
        indices = np.asarray(head(memory)["indices"])
        assert indices[0].tolist() == indices[1].tolist()


class TestSelectionGuardIsRedProven:
    """SC-D proof (i): a DEAD objectness head must make the guard FAIL.

    The injection is a dead COMPONENT, not the specific bug: the objectness
    stack's last projection is zeroed (kernel AND bias), so the head emits an
    identically-zero field for every position of every image. Everything else
    stays live -- same shapes, same dtypes, finite values, no raise.
    """

    @staticmethod
    def _kill_objectness(head: Sam3EncoderQuerySelection) -> None:
        last = head.objectness_head[-1]
        last.kernel.assign(np.zeros(last.kernel.shape, dtype="float32"))
        last.bias.assign(np.zeros(last.bias.shape, dtype="float32"))

    def test_a_dead_objectness_head_makes_the_guard_assertion_fire(self, head):
        self._kill_objectness(head)
        out = head(_memory())
        indices = np.asarray(out["indices"])
        selected = np.asarray(out["selected_boxes"])

        # The mutation is PLAUSIBLE, not obviously broken: right shapes, finite
        # boxes, and an ordinary across-QUERY spread. Only the across-IMAGE
        # comparison separates it from a live selection.
        assert indices.shape == (BATCH, TINY["num_queries"])
        assert np.all(np.isfinite(selected))
        assert float(np.mean(np.std(selected, axis=1))) > 1e-2

        # `ops.top_k`'s ascending-index tie-break, as predicted.
        assert indices.tolist() == [[0, 1, 2]] * BATCH
        # ... and the across-IMAGE spread collapses to EXACTLY zero.
        assert float(np.max(np.std(selected, axis=0))) == 0.0

        with pytest.raises(AssertionError, match="DEGENERATE sequence"):
            _assert_selection_is_image_dependent(indices, selected)

    def test_a_constant_but_non_degenerate_field_also_fires_the_guard(
            self, head):
        """The harder arm: a field that is NOT all-equal but is still
        image-independent. Its selection is `[7, 3, 11]`, so it escapes the
        degenerate-sequence check and the remaining two assertions must be what
        catches it."""
        field = np.zeros((BATCH, POSITIONS, 1), dtype="float32")
        field[:, 7] = 3.0
        field[:, 3] = 2.0
        field[:, 11] = 1.0
        head.objectness_head = [_FixedField(field)]
        out = head(_memory())
        indices = np.asarray(out["indices"])
        selected = np.asarray(out["selected_boxes"])
        assert indices.tolist() == [[7, 3, 11]] * BATCH, indices.tolist()
        with pytest.raises(AssertionError, match="image-INDEPENDENT"):
            _assert_selection_is_image_dependent(indices, selected)

    def test_the_guard_is_green_again_on_a_live_head(self, head):
        """The control -- otherwise the RED reading could be the fixture's."""
        out = head(_memory())
        _assert_selection_is_image_dependent(
            np.asarray(out["indices"]), np.asarray(out["selected_boxes"]))


# ---------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------


class TestSerialization:

    def test_config_roundtrip_preserves_every_value(self):
        head = Sam3EncoderQuerySelection(
            d_model=8, num_queries=3, feat_size=(2, 8), anchor_size=0.25,
            mlp_depth=2)
        clone = Sam3EncoderQuerySelection.from_config(head.get_config())
        for key, value in head.get_config().items():
            assert clone.get_config()[key] == value, key
        for key in ("d_model", "num_queries", "feat_size", "anchor_size",
                    "mlp_depth"):
            assert key in head.get_config(), key

    def test_config_carries_the_geometry_the_anchors_were_built_from(self):
        head = Sam3EncoderQuerySelection(**TINY)
        clone = Sam3EncoderQuerySelection.from_config(head.get_config())
        np.testing.assert_array_equal(clone._anchor_grid, head._anchor_grid)

    def test_full_keras_roundtrip_is_exactly_bit_identical(self, tmp_path):
        """D-123: asserted at `training=False`; `training=None` is NOT
        inference."""
        head = Sam3EncoderQuerySelection(**TINY)
        inputs = keras.Input(shape=(POSITIONS, TINY["d_model"]))
        model = keras.Model(inputs, head(inputs))
        _randomize(head, seed=5)

        probe = _memory(seed=17)
        before = {k: np.asarray(v)
                  for k, v in model(probe, training=False).items()}

        path = tmp_path / "query_selection.keras"
        model.save(path)
        restored = keras.models.load_model(path)
        after = {k: np.asarray(v)
                 for k, v in restored(probe, training=False).items()}

        assert set(before) == set(after)
        for key in before:
            delta = float(np.max(np.abs(
                before[key].astype("float64") - after[key].astype("float64"))))
            assert delta == 0.0, f"{key} moved by {delta!r} on round trip"

    def test_the_restored_weights_are_the_saved_ones_not_fresh_ones(
            self, tmp_path):
        """D-098's failure mode reads as a count/path/param match with FRESH
        kernels, so the value comparison is the only instrument that sees it."""
        head = Sam3EncoderQuerySelection(**TINY)
        inputs = keras.Input(shape=(POSITIONS, TINY["d_model"]))
        model = keras.Model(inputs, head(inputs))
        _randomize(head, seed=9)
        path = tmp_path / "weights.keras"
        model.save(path)
        restored = keras.models.load_model(path)

        saved = [np.asarray(w) for w in model.weights]
        loaded = [np.asarray(w) for w in restored.weights]
        assert len(saved) == len(loaded)
        assert [w.shape for w in saved] == [w.shape for w in loaded]
        for a, b in zip(saved, loaded):
            assert float(np.max(np.abs(a - b))) == 0.0


# ---------------------------------------------------------------------
# the prompt-conditioned proposal head (default OFF)
#
# STEP-6 SCOPE. These guards cover construction, the flag-OFF inertness that
# keeps the on-disk checkpoints loadable, and the `get_config` round trip. The
# RED-PROVEN prompt-liveness test (dead-component injection, with the firing
# assertion named) and the model-level three-combination byte-identity gate are
# step 7's, and are deliberately NOT weakened versions of themselves here.
# ---------------------------------------------------------------------


def _prompt(seed: int = 11, batch: int = BATCH, seq: int = 5) -> np.ndarray:
    """Per-image-DIFFERENT prompt features, width `d_model`."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(batch, seq, TINY["d_model"])).astype("float32")


def _conditioned_head(seed: int = 3) -> Sam3EncoderQuerySelection:
    layer = Sam3EncoderQuerySelection(prompt_conditioned=True, **TINY)
    layer.build((BATCH, POSITIONS, TINY["d_model"]))
    _randomize(layer, seed=seed)
    return layer


# ---------------------------------------------------------------------
# prompt liveness, measured against the instrument's OWN floor
# ---------------------------------------------------------------------

#: How many DIFFERENT prompts one liveness probe sweeps. One prompt PAIR is not
#: enough to characterize this instrument: at TINY the pair-to-pair reading
#: ranges from 1/12 to 11/12 moved positions (measured, seeds 11..20), so a
#: single pair could be read as near-dead purely by which two prompts it drew.
PROMPT_PROBES = 8

#: The instrument's own noise floor, MEASURED on CPU, not assumed. TWO null
#: arms were run and BOTH read EXACTLY zero moved positions and exactly 0.0
#: box delta: the SAME prompt passed `PROMPT_PROBES` times (the head is
#: deterministic), and a flag-OFF head under all `PROMPT_PROBES` DIFFERENT
#: prompts (that path never reads the prompt at all). `_assert_the_prompt_
#: reaches_the_selection` re-measures the null arm on every call rather than
#: trusting this comment.
PROMPT_MOVED_FLOOR = 0.0

#: The margin a LIVE head must clear over that floor. Measured at TINY over
#: EIGHT independent weight seeds (0..7): mean moved fraction
#: 0.667 / 0.679 / 0.690 / 0.702 / 0.798 / 0.702 / 0.774 / 0.905, and
#: `PROMPT_PROBES` of `PROMPT_PROBES` distinct selection patterns at every one
#: of them. The bar sits at 2.7x below the smallest of those, so it is not
#: perched on the low edge of the distribution (the failure mode D-015 was
#: opened by), while both null arms read 0.0 and both dead-component
#: injections below read exactly 0.0 too.
PROMPT_MOVED_BAR = 0.25


def _sweep_prompts(head: Sam3EncoderQuerySelection, memory: np.ndarray,
                   seeds: List[int]) -> List[Dict[str, np.ndarray]]:
    """Run one head on ONE memory under a list of prompt seeds.

    Args:
        head: A built head.
        memory: `(batch, positions, d_model)` image memory, held FIXED so the
            only thing that varies across the sweep is the prompt.
        seeds: One prompt seed per probe.

    Returns:
        One output dict per seed, numpy-materialized.
    """
    mask = np.zeros((memory.shape[0], 5), dtype="bool")
    return [{k: np.asarray(v) for k, v in
             head(memory, prompt=_prompt(seed=seed), prompt_padding_mask=mask,
                  training=False).items()}
            for seed in seeds]


def _moved_fraction(sweep: List[Dict[str, np.ndarray]]) -> float:
    """Mean fraction of selected memory positions that differ from probe 0."""
    base = sweep[0]["indices"]
    return float(np.mean([np.mean(out["indices"] != base)
                          for out in sweep[1:]]))


def _box_delta(sweep: List[Dict[str, np.ndarray]]) -> float:
    """Max abs movement of `selected_boxes` away from probe 0, float64."""
    base = sweep[0]["selected_boxes"].astype("float64")
    return max(float(np.max(np.abs(out["selected_boxes"].astype("float64")
                                   - base))) for out in sweep[1:])


def _assert_the_prompt_reaches_the_selection(
        live: List[Dict[str, np.ndarray]],
        null: List[Dict[str, np.ndarray]]) -> None:
    """THE prompt-liveness assertion, stated against a MEASURED floor.

    "The output changed" is not the test. Two things make it one:

    - The comparison is against `null`, the SAME instrument run on an arm that
      cannot possibly show the effect (the same prompt repeated, or a flag-OFF
      head). If that arm is not silent, the instrument is measuring something
      other than the prompt and no reading on `live` means anything.
    - The quantity is the top-k SELECTION, not a hidden activation. A
      modulation that shifted every position equally cannot change an argsort,
      so it could condition nothing that leaves this head -- and `selected_
      boxes` at the shipped zero-init box head is a pure readout of WHICH
      positions were selected.

    Args:
        live: Sweep over `PROMPT_PROBES` DIFFERENT prompts.
        null: Sweep over an arm that cannot respond to the prompt.

    Returns:
        None.

    Raises:
        AssertionError: If the null arm is not silent, if the selection is
            prompt-invariant, or if the selected boxes did not move.
    """
    null_moved, null_box = _moved_fraction(null), _box_delta(null)
    assert null_moved == PROMPT_MOVED_FLOOR and null_box == 0.0, (
        f"the instrument is not silent on its own null arm: {null_moved:.4f} "
        f"of the selected positions moved and the selected boxes moved by "
        f"{null_box:.3e} on an arm that CANNOT respond to the prompt, so no "
        f"reading on the live arm can be attributed to the prompt")

    moved = _moved_fraction(live)
    assert moved > PROMPT_MOVED_FLOOR + PROMPT_MOVED_BAR, (
        f"prompt-INVARIANT selection: only {moved:.4f} of the selected memory "
        f"positions moved across {len(live)} different prompts, against a "
        f"measured null-arm floor of {null_moved:.4f} and a required margin "
        f"of {PROMPT_MOVED_BAR}; the head is reading the prompt but the top-k "
        f"SELECTION -- the only thing that leaves this head -- is not")

    patterns = len({tuple(out["indices"].ravel().tolist()) for out in live})
    assert patterns > 1, (
        f"all {len(live)} prompts produced the SAME selection pattern "
        f"{live[0]['indices'].tolist()}")

    box = _box_delta(live)
    assert box > null_box, (
        f"the selected boxes did not move: {box:.3e} against a null-arm floor "
        f"of {null_box:.3e}")


class TestPromptConditionedFlag:
    """The flag that lets the proposal head read the text prompt."""

    def test_the_default_is_off_and_creates_no_sub_layer(self, head):
        """A-4: the default-OFF gate is what keeps 21 checkpoints loadable."""
        assert head.prompt_conditioned is False
        assert head.prompt_film is None

    def test_the_parameter_count_at_defaults_is_unmoved(self, head):
        """The exact structure-derived oracle above must still hold at
        defaults, or the on-disk checkpoints' weight sets disagree."""
        width = TINY["d_model"]
        stack = 2 * (width * width + width)
        assert head.count_params() == (stack + width * 1 + 1) + (
            stack + width * 4 + 4)

    def test_the_flag_adds_exactly_the_structure_derived_parameters(self):
        """Written FROM THE STRUCTURE: one `d_model -> 2 * d_model` affine.

        Enumerated, not transcribed and not read off the layer: the FiLM
        projection consumes the POOLED prompt (width `d_model`) and emits a
        scale and a shift (width `d_model` each), so it is one kernel of
        `d_model x 2 * d_model` plus `2 * d_model` biases.
        """
        width = TINY["d_model"]
        expected = width * (2 * width) + 2 * width
        on = _conditioned_head()
        off = Sam3EncoderQuerySelection(**TINY)
        off.build((BATCH, POSITIONS, TINY["d_model"]))
        assert on.count_params() - off.count_params() == expected

    def test_the_film_projection_is_not_zero_initialized(self):
        """A zero init makes the modulation the EXACT identity at step 0, i.e.
        an untrained flag-on model that is bit-identical to the flag-off one
        on every prompt -- born degenerate on the one axis the flag opens."""
        fresh = Sam3EncoderQuerySelection(prompt_conditioned=True, **TINY)
        fresh.build((BATCH, POSITIONS, TINY["d_model"]))
        kernel = np.asarray(fresh.prompt_film[-1].weights[0])
        assert float(np.max(np.abs(kernel))) > 0.0

    def test_the_film_store_is_flat(self):
        """D-098: a `List[List[Layer]]` loses weights on a `.keras` round trip
        while the count, the paths and the parameter total all still match."""
        stack = _conditioned_head().prompt_film
        assert stack
        assert all(isinstance(l, keras.layers.Layer) for l in stack)
        assert not any(isinstance(l, (list, tuple)) for l in stack)

    def test_the_flag_off_head_ignores_the_prompt_bit_for_bit(self, head):
        """The prompt is passed at `Sam3Image`'s ONE call site whatever the
        flag says, so the flag-off head must be provably inert to it."""
        memory = _memory()
        blind = head(memory, training=False)
        prompted = head(memory, prompt=_prompt(),
                        prompt_padding_mask=np.zeros((BATCH, 5), dtype="bool"),
                        training=False)
        for key in blind:
            delta = float(np.max(np.abs(
                np.asarray(blind[key]).astype("float64")
                - np.asarray(prompted[key]).astype("float64"))))
            assert delta == 0.0, f"{key} moved by {delta!r} with the flag OFF"

    def test_the_flag_on_head_refuses_a_missing_prompt(self):
        """Falling back to prompt-BLIND proposals is the defect, not a
        graceful default: it has no shape, dtype or finiteness symptom."""
        with pytest.raises(ValueError, match="prompt_conditioned=True"):
            _conditioned_head()(_memory(), training=False)

    def test_the_prompt_moves_the_selection_above_the_measured_floor(self):
        """The point of the change: the top-k SELECTION is prompt-dependent.

        Goes through the SHARED guard, so this reading and the two RED-proofs
        below are the same assertion fired on different arms -- there is one
        definition of "the prompt reaches the selection" in this file, not a
        strong one for the proof and a weaker one for the green case.

        The null arm is the SAME prompt repeated: it is the head's own
        determinism, so it isolates the prompt from every other source of
        movement (weight init, memory draw, backend non-determinism).
        """
        conditioned = _conditioned_head()
        memory = _memory()
        live = _sweep_prompts(conditioned, memory,
                              list(range(100, 100 + PROMPT_PROBES)))
        null = _sweep_prompts(conditioned, memory, [100] * PROMPT_PROBES)
        _assert_the_prompt_reaches_the_selection(live, null)

    def test_a_flag_off_head_is_the_instrument_s_other_null_arm(self, head):
        """A SECOND, independent null arm: a head that structurally cannot
        read the prompt, swept over the same `PROMPT_PROBES` prompts. It must
        read exactly the floor, which is what makes the floor a property of
        the mechanism rather than of the prompt draws."""
        sweep = _sweep_prompts(head, _memory(),
                               list(range(100, 100 + PROMPT_PROBES)))
        assert _moved_fraction(sweep) == PROMPT_MOVED_FLOOR
        assert _box_delta(sweep) == 0.0

    def test_the_pool_respects_the_padding_mask(self):
        """It pools through `Sam3DotProductScoring.masked_mean_pool`, so a
        padded position must not reach the proposals. Driven by CHANGING the
        content of a masked-out position, which a mask-blind pool would see."""
        conditioned = _conditioned_head()
        memory = _memory()
        prompt = _prompt(seed=21)
        mask = np.zeros((BATCH, prompt.shape[1]), dtype="bool")
        mask[:, -2:] = True
        polluted = prompt.copy()
        polluted[:, -2:, :] = 25.0
        a = conditioned(memory, prompt=prompt, prompt_padding_mask=mask,
                        training=False)
        b = conditioned(memory, prompt=polluted, prompt_padding_mask=mask,
                        training=False)
        delta = float(np.max(np.abs(np.asarray(a["boxes"]).astype("float64")
                                    - np.asarray(b["boxes"]).astype("float64"))))
        assert delta == 0.0, (
            f"masked-out prompt positions changed the proposals by {delta!r}: "
            f"the pool is not honouring the padding mask")

    def test_config_roundtrip_carries_the_flag_at_both_values(self):
        for value in (False, True):
            head = Sam3EncoderQuerySelection(prompt_conditioned=value, **TINY)
            config = head.get_config()
            assert config["prompt_conditioned"] is value
            clone = Sam3EncoderQuerySelection.from_config(config)
            assert clone.prompt_conditioned is value
            assert (clone.prompt_film is None) is (not value)

    def test_full_keras_roundtrip_with_the_flag_on_is_bit_identical(
            self, tmp_path):
        """The flag-ON path is serialization-pinned too, from its first commit.

        Built through the functional API with TWO inputs, which is also what
        proves `call`'s new keyword is a real, traceable argument.
        """
        head = Sam3EncoderQuerySelection(prompt_conditioned=True, **TINY)
        memory_in = keras.Input(shape=(POSITIONS, TINY["d_model"]))
        prompt_in = keras.Input(shape=(5, TINY["d_model"]))
        model = keras.Model([memory_in, prompt_in],
                            head(memory_in, prompt=prompt_in))
        _randomize(head, seed=7)

        probe = [_memory(seed=19), _prompt(seed=23)]
        before = {k: np.asarray(v)
                  for k, v in model(probe, training=False).items()}
        path = tmp_path / "prompt_conditioned.keras"
        model.save(path)
        restored = keras.models.load_model(path)
        after = {k: np.asarray(v)
                 for k, v in restored(probe, training=False).items()}

        assert set(before) == set(after)
        for key in before:
            delta = float(np.max(np.abs(
                before[key].astype("float64") - after[key].astype("float64"))))
            assert delta == 0.0, f"{key} moved by {delta!r} on round trip"


# ---------------------------------------------------------------------
# SC-H: the prompt-liveness guard, proven RED by dead-component injection
# ---------------------------------------------------------------------


class _ConstantPool:
    """A stand-in for `Sam3DotProductScoring` whose pool ignores the prompt.

    Patched over the NAME `query_selection.py` imported, never over
    `Sam3DotProductScoring.masked_mean_pool` itself. That is not fastidiousness:
    the real method is a `@staticmethod`, `getattr` on the class unwraps it to a
    plain function, and `monkeypatch.undo` would put the plain function BACK --
    silently re-binding it as an instance method and breaking
    `Sam3DotProductScoring.call`'s own `self.masked_mean_pool(...)` for every
    test that runs afterwards in the same session. This package has already
    been bitten once by a test whose effect leaked into whatever collected
    next (D-016).
    """

    @staticmethod
    def masked_mean_pool(prompt, prompt_padding_mask):
        del prompt_padding_mask
        return ops.mean(ops.zeros_like(prompt), axis=1) + 0.5


def _inject_constant_pool(monkeypatch) -> None:
    """INJECTION 1: the pooled prompt becomes a constant, path intact."""
    import dl_techniques.models.sam3.query_selection as module
    assert module.Sam3DotProductScoring is Sam3DotProductScoring, (
        "the injection is patching a name the head does not actually pool "
        "through, so it would be a no-op wearing a mutation's name")
    monkeypatch.setattr(module, "Sam3DotProductScoring", _ConstantPool)


class TestPromptLivenessIsRedProven:
    """SC-H: `_assert_the_prompt_reaches_the_selection` must FAIL on a dead
    component, and the assertion that fires is named at each injection.

    Two injections, at the two ends of the one path the flag opens, because
    either alone leaves half of it unproven:

    - **The prompt's INFORMATION is killed, the code path stays alive.**
      `masked_mean_pool` is replaced by a constant pool, so the FiLM
      projection still runs, still emits a scale and a shift, and still
      modulates `memory` -- with a value that is the same for every prompt.
    - **The MODULATION is killed, the prompt keeps flowing.** The FiLM
      projection's kernel AND bias are zeroed, so `scale` and `shift` are zero
      and `memory * (1 + 0) + 0` is the exact identity. This is precisely the
      state a `zero_init_last=True` would have SHIPPED the head in (see the
      anchor in `query_selection.py`), which is why it is worth firing.

    Both injections must ALSO be provably unable to move the flag-OFF path:
    that path never constructs a FiLM projection and never pools a prompt, so
    an injection that changed it would mean the two paths are not separate.
    """

    @staticmethod
    def _sweeps(head: Sam3EncoderQuerySelection, memory: np.ndarray):
        live = _sweep_prompts(head, memory,
                              list(range(100, 100 + PROMPT_PROBES)))
        null = _sweep_prompts(head, memory, [100] * PROMPT_PROBES)
        return live, null

    @staticmethod
    def _assert_plausible(sweep) -> None:
        """The injected head must look ORDINARY, not obviously broken."""
        for out in sweep:
            assert out["indices"].shape == (BATCH, TINY["num_queries"])
            assert np.all(np.isfinite(out["selected_boxes"]))
            assert np.all((out["boxes"] > 0.0) & (out["boxes"] < 1.0))
        assert float(np.mean(np.std(sweep[0]["selected_boxes"], axis=1))) > 1e-2

    def test_a_constant_pooled_prompt_makes_the_guard_fire(self, monkeypatch):
        """INJECTION 1 -- kill the prompt's information, keep the path."""
        conditioned = _conditioned_head()
        memory = _memory()
        _inject_constant_pool(monkeypatch)
        live, null = self._sweeps(conditioned, memory)
        self._assert_plausible(live)
        assert _moved_fraction(live) == 0.0, (
            "the injection did not actually kill the prompt's information")
        with pytest.raises(AssertionError, match="prompt-INVARIANT selection"):
            _assert_the_prompt_reaches_the_selection(live, null)

    def test_a_constant_pooled_prompt_cannot_move_the_flag_off_path(
            self, head, monkeypatch):
        """The same injection, on the default path: it must be inert there."""
        memory = _memory()
        before = _sweep_prompts(head, memory, [100, 101])
        _inject_constant_pool(monkeypatch)
        after = _sweep_prompts(head, memory, [100, 101])
        for a, b in zip(before, after):
            for key in a:
                assert float(np.max(np.abs(
                    a[key].astype("float64") - b[key].astype("float64")))) == 0.0

    def test_a_zeroed_film_projection_makes_the_guard_fire(self):
        """INJECTION 2 -- kill the modulation, keep the prompt flowing.

        `scale = shift = 0` makes `memory * (1 + 0) + 0` the exact identity,
        so the head is the prompt-BLIND one wearing the flag-ON name.
        """
        conditioned = _conditioned_head()
        memory = _memory()
        last = conditioned.prompt_film[-1]
        last.kernel.assign(np.zeros(last.kernel.shape, dtype="float32"))
        last.bias.assign(np.zeros(last.bias.shape, dtype="float32"))

        live, null = self._sweeps(conditioned, memory)
        self._assert_plausible(live)
        # Not merely near-identity: EXACTLY zero moved positions and EXACTLY
        # zero box movement, because `memory * (1 + 0) + 0` is bit-exact.
        assert _moved_fraction(live) == 0.0
        assert _box_delta(live) == 0.0
        with pytest.raises(AssertionError, match="prompt-INVARIANT selection"):
            _assert_the_prompt_reaches_the_selection(live, null)

    def test_a_dead_null_arm_detector_fires_its_own_assertion(self):
        """The guard's FIRST assertion, fired on its own.

        If the null arm is not silent the live reading is uninterpretable, so
        that check has to be proven RED too -- otherwise a broken null arm
        would be waved through and every later number would rest on it.
        """
        conditioned = _conditioned_head()
        memory = _memory()
        live = _sweep_prompts(conditioned, memory,
                              list(range(100, 100 + PROMPT_PROBES)))
        with pytest.raises(AssertionError,
                           match="not silent on its own null arm"):
            _assert_the_prompt_reaches_the_selection(live, live)

    def test_the_guard_is_green_again_on_an_uninjected_head(self):
        """The control -- otherwise the RED readings could be the fixture's."""
        conditioned = _conditioned_head()
        memory = _memory()
        live, null = self._sweeps(conditioned, memory)
        _assert_the_prompt_reaches_the_selection(live, null)
