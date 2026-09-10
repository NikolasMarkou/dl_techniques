"""Behavioural guards for :class:`DocScannerRectifier`, the first runnable model.

The shared oracles under ``tests/test_models/*_oracle.py`` carry everything that
is generic -- the ``.keras`` round trip on VALUES, weights-restored-before-first-
call, build parity, gradient flow, the falsifiable smoke contract, knob
sensitivity. None of them is re-implemented here.

What IS written here is the set of claims that are specific to this assembly and
that no generic oracle can make, because each of them is **shape-preserving**: a
model with any of these defects returns ``(B, 288, 288, 2)`` finite float32,
saves, reloads and trains its loss down.

1. **The loop really runs ``iters`` times.** ``iters=1`` and ``iters=12`` must
   differ, and the training-mode stack's first slice must BE the ``iters=1``
   inference output. A loop that silently ran once, or that returned a
   constant, passes every shape test.
2. **The coordinate field is detached at the top of every iteration**
   (``model.py:86``). Dropping the ``stop_gradient`` is invisible at inference
   -- the forward VALUES are bit-identical -- and changes only the gradients.
3. **``warpfea`` is re-sampled from ``fmap1``, never chained from the previous
   ``warpfea``** (``model.py:92``). Chaining is shape-identical and only
   diverges from the third iteration on.
4. **``bm_up`` is in ABSOLUTE full-resolution pixel units** (``model.py:91``),
   not normalized and not a residual. With the flow head zeroed, the model must
   emit exactly the identity coordinate grid.

Non-square fixtures are used wherever an axis could be confused, for the reason
``test_components.py`` states at length: a square fixture is structurally blind
to an h/w swap.
"""

import inspect

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.vision.image_restoration.doc_scanner import model as model_module
from dl_techniques.models.vision.image_restoration.doc_scanner.components import (
    FLOW_CHANNELS,
    REFINE_ITERATIONS,
    SPATIAL_DIVISOR,
    DocScannerUpdateBlock,
)
from dl_techniques.models.vision.image_restoration.doc_scanner.model import (
    DocScannerRectifier,
    create_doc_scanner_rectifier,
)
from dl_techniques.models.vision.image_restoration.doc_scanner.warp import coords_grid

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    stop_all_gradients,
)
from ..knob_sensitivity_oracle import (
    assert_structural_knob_changes_weights,
    assert_value_knob_changes_output,
)
from ..lazy_build_contract_oracle import assert_lazy_build_costs_nothing
from ..roundtrip_instrument_oracle import (
    assert_build_parity,
    assert_roundtrip_output_values,
    assert_weights_restored_before_first_call,
    measure_build_parity,
    measure_roundtrip,
)
from ..smoke_contract_oracle import (
    assert_contract_rejects_a_broken_forward,
    assert_finite,
    broken_forward,
)

# ---------------------------------------------------------------------
# The subject.
#
# Every width below is a STAND-IN, deliberately small and pairwise distinct
# where a swap would otherwise hide behind two equal numbers -- the same rule
# `test_components.py` applies. The shipped widths are exercised separately, by
# the `docscanner-l` variant arm, which builds the real thing once.
#
# BATCH is 2, never 1: the smoke oracle's `slice_leading_axis` breaker slices
# each output leaf to `leaf[:1]`, which is a no-op at batch 1, and the
# meta-test would then report "the contract accepts a broken forward" about a
# breaker that broke nothing.
# ---------------------------------------------------------------------

BATCH = 2
HEIGHT, WIDTH = 32, 24          # NON-SQUARE, and both multiples of 8
ITERS = 3

_SMALL = dict(
    hidden_dim=8,
    context_dim=8,
    gru_input_dim=16,
    fnet_output_dim=16,
    encoder_stem_channels=6,
    encoder_stage_channels=(6, 10, 12),
    motion_output_dim=8,
    motion_corr_hidden=7,
    motion_corr_out=5,
    motion_flow_hidden=9,
    motion_flow_out=4,
    flow_head_hidden=11,
    mask_head_hidden=13,
    iters=ITERS,
)

INPUT_SHAPE = (None, HEIGHT, WIDTH, 3)

#: The repo convention for a value assertion: an explicit absolute tolerance
#: and no relative component.
ATOL = 1e-6


def _build(**overrides) -> DocScannerRectifier:
    """The subject, UNBUILT. Oracles that need a built model call it."""
    return DocScannerRectifier(**{**_SMALL, **overrides})


def _inputs() -> np.ndarray:
    """A deterministic input. The oracles compare across calls, so this must
    not draw fresh values."""
    return np.random.RandomState(0).randn(
        BATCH, HEIGHT, WIDTH, 3).astype("float32")


def _built(**overrides) -> DocScannerRectifier:
    model = _build(**overrides)
    model.build(INPUT_SHAPE)
    return model


@pytest.fixture(scope="module")
def built_model() -> DocScannerRectifier:
    """One built subject shared by the read-only arms."""
    keras.utils.set_random_seed(0)
    return _built()


def _as_numpy(tensor) -> np.ndarray:
    return np.asarray(keras.ops.convert_to_numpy(tensor))


# ---------------------------------------------------------------------
# 1. Shapes and the output contract
# ---------------------------------------------------------------------


class TestTheOutputContract:
    """``training`` selects the FORM of the output, not only the behaviour."""

    def test_inference_returns_one_full_resolution_backward_map(
            self, built_model):
        out = built_model(_inputs(), training=False)
        assert tuple(out.shape) == (BATCH, HEIGHT, WIDTH, FLOW_CHANNELS)
        assert_finite(out)

    def test_training_returns_the_whole_stacked_sequence(self, built_model):
        out = built_model(_inputs(), training=True)
        assert tuple(out.shape) == (
            BATCH, ITERS, HEIGHT, WIDTH, FLOW_CHANNELS)
        assert_finite(out)

    def test_the_default_training_flag_is_the_inference_form(self, built_model):
        """``training=None`` must not silently emit the 5-D training form.

        A stock ``predict()`` passes nothing; if the default resolved to the
        sequence, every downstream consumer of a backward map would receive one
        extra axis and the failure would surface far from here.
        """
        assert tuple(built_model(_inputs()).shape) == (
            BATCH, HEIGHT, WIDTH, FLOW_CHANNELS)

    def test_the_inference_map_is_the_LAST_iteration_not_the_first(
            self, built_model):
        """``predictions[-1]``, per ``model.py:95-97``'s ``test_mode`` return.

        Returning ``predictions[0]`` is shape-identical and would throw away
        every refinement the loop performs.
        """
        x = _inputs()
        stack = _as_numpy(built_model(x, training=True))
        last = _as_numpy(built_model(x, training=False))
        np.testing.assert_allclose(last, stack[:, -1], rtol=0, atol=ATOL)
        # Non-vacuity: the first and last slices must actually differ, or the
        # arm above would pass under `predictions[0]` too.
        assert np.max(np.abs(stack[:, 0] - stack[:, -1])) > 1e-4, (
            "the first and last iterations are indistinguishable on this "
            "fixture, so the arm above cannot see which one was returned")

    @pytest.mark.parametrize("height,width", [(32, 32), (40, 40), (32, 24), (24, 40)])
    def test_it_accepts_any_multiple_of_the_stride_including_non_square(
            self, height, width):
        keras.utils.set_random_seed(0)
        model = _build()
        x = np.zeros((1, height, width, 3), dtype="float32")
        assert tuple(model(x, training=False).shape) == (
            1, height, width, FLOW_CHANNELS)
        assert tuple(model(x, training=True).shape) == (
            1, ITERS, height, width, FLOW_CHANNELS)

    def test_288_is_the_papers_own_resolution_and_works(self):
        """Success criterion 1, at the paper's training resolution.

        Run at the stand-in widths and batch 1: the claim under test is the
        coordinate/upsample plumbing at 288, which is width-independent, and
        the shipped widths are covered by the variant arm below.
        """
        keras.utils.set_random_seed(0)
        model = _build()
        x = np.zeros((1, 288, 288, 3), dtype="float32")
        assert tuple(model(x, training=False).shape) == (
            1, 288, 288, FLOW_CHANNELS)
        assert tuple(model(x, training=True).shape) == (
            1, ITERS, 288, 288, FLOW_CHANNELS)


class TestTheSpatialContractIsRefusedNotPadded:
    """The model raises rather than padding internally -- see ``__init__.py``."""

    @pytest.mark.parametrize("height,width", [(30, 32), (32, 30), (33, 31)])
    def test_a_non_multiple_of_eight_raises_naming_the_size(
            self, height, width):
        model = _build()
        x = np.zeros((1, height, width, 3), dtype="float32")
        with pytest.raises(ValueError, match=r"divisible by 8"):
            model(x, training=False)

    def test_an_unknown_extent_raises_naming_the_reason(self):
        """A symbolic build at ``(None, None, None, 3)`` cannot work here.

        Unlike ``doc_res``, this model materializes an identity coordinate
        field, which is an ``arange`` per axis. The message must say so; the
        alternative is an ``arange`` error several frames down that mentions
        neither this model nor coordinates.
        """
        with pytest.raises(ValueError, match="statically-known spatial"):
            _build().build((None, None, None, 3))


# ---------------------------------------------------------------------
# 2. The iteration count is REAL
# ---------------------------------------------------------------------


class TestTheRefinementLoopReallyIterates:
    """A loop that ran once, or that returned a constant, passes every shape
    test in this file. These three arms are what convict it."""

    def test_one_iteration_and_twelve_differ(self):
        """The plan's ``iters=1`` vs ``iters=12``, weights held identical."""
        deltas = assert_value_knob_changes_output(
            {
                1: lambda: _warm(_build(iters=1)),
                REFINE_ITERATIONS: lambda: _warm(
                    _build(iters=REFINE_ITERATIONS)),
            },
            _inputs(),
            knob="iters",
            atol=1e-4,
        )
        assert all(value > 1e-4 for value in deltas.values()), deltas

    def test_the_training_stacks_first_slice_is_the_one_iteration_output(self):
        """Slice ``k`` of the stack IS the ``iters = k + 1`` inference output.

        This is the arm that a "returns a constant" or "runs once and tiles"
        implementation cannot survive: it ties the sequence axis to the loop's
        own trip count rather than merely to a reshape.
        """
        keras.utils.set_random_seed(0)
        many = _warm(_build(iters=REFINE_ITERATIONS))
        keras.utils.set_random_seed(0)
        one = _warm(_build(iters=1))
        one.set_weights(many.get_weights())

        x = _inputs()
        stack = _as_numpy(many(x, training=True))
        single = _as_numpy(one(x, training=False))
        np.testing.assert_allclose(stack[:, 0], single, rtol=0, atol=ATOL)

    def test_consecutive_iterations_are_all_distinct(self, built_model):
        """No two slices of the sequence coincide.

        A loop whose state failed to advance -- a ``net`` that was reassigned
        to the wrong variable, say -- would emit ``iters`` identical maps, and
        the stacked shape would be exactly right.
        """
        stack = _as_numpy(built_model(_inputs(), training=True))
        for k in range(ITERS - 1):
            spread = float(np.max(np.abs(stack[:, k] - stack[:, k + 1])))
            assert spread > 1e-5, (
                f"iterations {k} and {k + 1} are identical (max|delta| "
                f"{spread:.3e}); the refinement loop is not advancing")


def _warm(model: DocScannerRectifier) -> DocScannerRectifier:
    """Build by a real call, the way a subclassed model is normally built."""
    model(_inputs(), training=False)
    return model


# ---------------------------------------------------------------------
# 3. The `stop_gradient` at the top of every iteration
# ---------------------------------------------------------------------


class TestTheCoordinateFieldIsDetachedEachIteration:
    """``model.py:86``'s ``coords1 = coords1.detach()``.

    This one cannot be seen from the forward VALUES at all -- they are
    bit-identical with and without it. The guard patches the model's own
    ``_detach_coordinate_field`` seam to the identity and requires the WEIGHT
    GRADIENTS to move. If the ``stop_gradient`` were inlined into ``call``
    instead, the patch would hit dead code, nothing would move, and this test
    would go RED -- which is exactly the intent, and is why the D-017 anchor
    forbids inlining it.
    """

    @staticmethod
    def _tape_connectivity(monkeypatch, detaching: bool):
        """Run one forward under a tape that WATCHES the seam's input.

        Interface contract -- two callers, the real arm and its non-vacuity
        control:

        * Parameters: pytest's ``monkeypatch``; ``detaching``, whether the real
          seam runs (``True``) or is replaced by the identity (``False``).
        * Returns: ``(pairs, loss, tape)`` -- one ``(seam_input, seam_output)``
          pair per iteration, the scalar loss, and the persistent tape.
        * Failure mode: none of its own.

        The measurement is ``tape.gradient(seam_output, seam_input)``, which is
        EXACTLY ``None`` when the two are separated by a ``stop_gradient`` and
        exactly not-``None`` when they are the same tensor. That is a
        connectivity question, not a magnitude one, so it is deterministic and
        device-independent -- unlike a comparison of gradient MAGNITUDES, which
        this guard was first written as and which passed under a mutation that
        deleted the ``stop_gradient`` outright, because GPU reduction
        non-determinism alone moved near-zero gradients by more than any
        threshold worth setting.
        """
        keras.utils.set_random_seed(0)
        model = _built(iters=4)

        original = DocScannerRectifier._detach_coordinate_field
        tape = tf.GradientTape(persistent=True)
        pairs = []

        def seam(self, coords):
            tape.watch(coords)
            out = original(self, coords) if detaching else coords
            # Watch BOTH ends. Iteration 0's field is the constant identity
            # grid, so its detached copy would otherwise be an unwatched
            # constant and the "is it on the backward graph" control below
            # would report a vacuous probe about a perfectly healthy model.
            tape.watch(out)
            pairs.append((coords, out))
            return out

        monkeypatch.setattr(
            DocScannerRectifier, "_detach_coordinate_field", seam)

        with tape:
            outputs = model(_inputs(), training=True)
            loss = keras.ops.mean(keras.ops.square(outputs))

        return pairs, loss, tape

    def test_the_forward_values_are_identical_either_way(self, monkeypatch):
        """The premise: this is invisible to every value assertion.

        Stated as an assertion rather than as prose, so that if it ever stops
        being true the next reader learns it here instead of assuming it.
        """
        keras.utils.set_random_seed(0)
        detached = _as_numpy(_built(iters=4)(_inputs(), training=True))
        monkeypatch.setattr(
            DocScannerRectifier,
            "_detach_coordinate_field",
            lambda self, coords: coords,
        )
        keras.utils.set_random_seed(0)
        attached = _as_numpy(_built(iters=4)(_inputs(), training=True))
        np.testing.assert_allclose(detached, attached, rtol=0, atol=0.0)

    def test_the_seam_runs_once_per_iteration(self, monkeypatch):
        """If the ``stop_gradient`` were inlined into ``call``, the patched
        method would be dead code and this would collect nothing."""
        pairs, _, _ = self._tape_connectivity(monkeypatch, detaching=True)
        assert len(pairs) == 4

    def test_no_gradient_crosses_the_seam(self, monkeypatch):
        """The claim: ``d(out_k) / d(in_k)`` does not exist."""
        pairs, loss, tape = self._tape_connectivity(monkeypatch, detaching=True)
        for index, (entered, left) in enumerate(pairs):
            assert tape.gradient(left, entered) is None, (
                f"iteration {index}'s coordinate field is still connected to "
                f"the iteration before it. `model.py:86` detaches it, and "
                f"without that the network is trained through a "
                f"{len(pairs)}-deep coordinate recurrence it was never meant "
                f"to see -- with the forward VALUES unchanged. See the D-017 "
                f"anchor in model.py.")

    def test_the_seams_output_is_nonetheless_on_the_backward_graph(
            self, monkeypatch):
        """Non-vacuity, half one: ``None`` above must mean SEVERED, not DEAD.

        If the whole loop were off the tape, every gradient would be ``None``
        and the arm above would pass for the wrong reason.
        """
        pairs, loss, tape = self._tape_connectivity(monkeypatch, detaching=True)
        for index, (_, left) in enumerate(pairs):
            assert tape.gradient(loss, left) is not None, (
                f"iteration {index}'s coordinate field is not on the backward "
                f"graph at all; the connectivity probe is vacuous")

    def test_the_probe_sees_a_LIVE_connection_when_the_detach_is_removed(
            self, monkeypatch):
        """Non-vacuity, half two: the same instrument, seam replaced by the
        identity, must report a gradient. This is the arm that would go RED if
        ``stop_gradient`` were deleted from the source, because then the two
        configurations would be indistinguishable."""
        pairs, _, tape = self._tape_connectivity(monkeypatch, detaching=False)
        crossings = [
            tape.gradient(left, entered) for entered, left in pairs
        ]
        assert all(g is not None for g in crossings), (
            "with the detach replaced by the identity, the probe STILL sees "
            "no gradient crossing. Either `stop_gradient` was moved out of "
            "`_detach_coordinate_field` (so removing it here changes nothing) "
            "or the probe is broken.")


# ---------------------------------------------------------------------
# 4. `warpfea` is re-sampled from `fmap1`, never chained
# ---------------------------------------------------------------------


class TestWarpedFeaturesAreResampledFromTheEncoderOutput:
    """``model.py:92``: ``warpfea = bilinear_sampler(fmap1, coords1)``.

    The wrong version -- ``sample(warpfea, coords1)`` -- is shape-identical,
    finite, trainable and serializable, and coincides with the right one for
    the first TWO iterations (``warpfea`` starts as ``fmap1``, and the first
    sample is taken at the identity grid). So the spy runs at ``iters=4``.
    """

    @staticmethod
    def _record_first_arguments(monkeypatch, iters: int):
        # The model is BUILT before the spy is installed. `build()` traces
        # `call` symbolically, and a symbolic KerasTensor has no values to
        # record -- recording during the trace would raise, and skipping the
        # trace's calls silently would make the call COUNT depend on whether a
        # build happened to be pending.
        keras.utils.set_random_seed(0)
        model = _built(iters=iters)

        seen = []
        real = model_module.sample_at_pixel_coords

        def spy(fmap, pix_xy):
            seen.append(_as_numpy(fmap))
            return real(fmap, pix_xy)

        monkeypatch.setattr(model_module, "sample_at_pixel_coords", spy)
        model(_inputs(), training=True)
        return model, seen

    def test_the_sampler_is_called_once_per_iteration(self, monkeypatch):
        _, seen = self._record_first_arguments(monkeypatch, 4)
        assert len(seen) == 4

    def test_every_call_reads_the_same_encoder_feature_map(self, monkeypatch):
        _, seen = self._record_first_arguments(monkeypatch, 4)
        for index, array in enumerate(seen[1:], start=1):
            np.testing.assert_allclose(
                array, seen[0], rtol=0, atol=0.0,
                err_msg=(
                    f"iteration {index} sampled a DIFFERENT feature map than "
                    f"iteration 0. `warpfea` is being chained through the "
                    f"sampler instead of being re-read from `fmap1` -- see "
                    f"the D-017 anchor in model.py."))

    def test_that_map_is_the_encoders_own_output(self, monkeypatch):
        """Non-vacuity for the arm above: identical-to-each-other is not
        enough; they must all be ``fnet(x)``, not some other constant."""
        model, seen = self._record_first_arguments(monkeypatch, 4)
        expected = _as_numpy(model.fnet(_inputs(), training=True))
        np.testing.assert_allclose(seen[0], expected, rtol=0, atol=ATOL)


class TestTheEncoderOutputIsSplitStateFirstThenContext:
    """``model.py:74-76``: ``split(fmap1, [160, 160])``, ``tanh`` then ``relu``.

    At the shipped variant both halves are 160 wide, so swapping them -- or
    swapping the two activations -- is perfectly shape-legal and merely feeds
    ``tanh`` the channels ``relu`` was meant to see. The only external observer
    of either half is the update block's first argument, so the guard spies
    there.
    """

    @staticmethod
    def _first_iteration_arguments(monkeypatch):
        keras.utils.set_random_seed(0)
        model = _built()
        seen = []
        original = DocScannerUpdateBlock.call

        def spy(self, inputs, training=None):
            seen.append([_as_numpy(t) for t in inputs[:2]])
            return original(self, inputs, training=training)

        monkeypatch.setattr(DocScannerUpdateBlock, "call", spy)
        x = _inputs()
        model(x, training=True)
        return model, x, seen

    def test_the_state_is_the_TANH_of_the_first_half(self, monkeypatch):
        model, x, seen = self._first_iteration_arguments(monkeypatch)
        fmap = _as_numpy(model.fnet(x, training=True))
        expected = np.tanh(fmap[..., :_SMALL["hidden_dim"]])
        np.testing.assert_allclose(seen[0][0], expected, rtol=0, atol=ATOL)

    def test_the_context_is_the_RELU_of_the_second_half(self, monkeypatch):
        model, x, seen = self._first_iteration_arguments(monkeypatch)
        fmap = _as_numpy(model.fnet(x, training=True))
        expected = np.maximum(fmap[..., _SMALL["hidden_dim"]:], 0.0)
        np.testing.assert_allclose(seen[0][1], expected, rtol=0, atol=ATOL)

    def test_the_two_halves_are_distinguishable_on_this_fixture(
            self, monkeypatch):
        """Non-vacuity: if the encoder emitted the same values in both halves,
        or if tanh and relu happened to agree here, the two arms above would
        pass under a swap."""
        model, x, seen = self._first_iteration_arguments(monkeypatch)
        fmap = _as_numpy(model.fnet(x, training=True))
        half = _SMALL["hidden_dim"]
        swapped_state = np.tanh(fmap[..., half:])
        assert np.max(np.abs(seen[0][0] - swapped_state)) > 1e-3
        swapped_context = np.maximum(fmap[..., :half], 0.0)
        assert np.max(np.abs(seen[0][1] - swapped_context)) > 1e-3

    def test_the_context_is_constant_across_iterations(self, monkeypatch):
        """``inp`` is computed ONCE, before the loop (``model.py:74``). A
        recomputed or updated context is shape-identical and would make the
        block's second input drift."""
        _, _, seen = self._first_iteration_arguments(monkeypatch)
        for index, (_, context) in enumerate(seen[1:], start=1):
            np.testing.assert_allclose(
                context, seen[0][1], rtol=0, atol=0.0,
                err_msg=f"the context changed at iteration {index}")


# ---------------------------------------------------------------------
# 5. `bm_up` is in ABSOLUTE full-resolution pixel units
# ---------------------------------------------------------------------


class TestTheBackwardMapIsAbsolutePixelCoordinates:
    """``model.py:91``: ``bm_up = coodslar + flow_up``.

    Zero the flow head's OUTPUT convolution and the whole refinement chain
    collapses: ``delta_flow == 0``, so ``coords1 == coords0`` forever,
    ``flow_up == 0``, and the emitted map must be EXACTLY the full-resolution
    identity coordinate grid. That single equality pins the coordinate
    pipeline end to end -- the full-resolution grid, its ``(x, y)`` channel
    order, its pixel units, the absence of any normalization, and the fact
    that the convex upsample is being fed a residual rather than a coordinate.

    The fixture is NON-SQUARE, so a transposed grid cannot pass.
    """

    @staticmethod
    def _model_with_a_dead_flow_head() -> DocScannerRectifier:
        keras.utils.set_random_seed(0)
        model = _built()
        head = model.update_block.flow_head.conv2
        head.kernel.assign(keras.ops.zeros_like(head.kernel))
        head.bias.assign(keras.ops.zeros_like(head.bias))
        return model

    def test_a_zero_residual_leaves_the_identity_coordinate_grid(self):
        model = self._model_with_a_dead_flow_head()
        out = _as_numpy(model(_inputs(), training=False))
        expected = _as_numpy(coords_grid(BATCH, HEIGHT, WIDTH))
        np.testing.assert_allclose(out, expected, rtol=0, atol=0.0)

    def test_the_map_is_not_normalized(self):
        """Two-sided: the identity grid above must not be all-zeros or in
        ``[-1, 1]``, or the arm above would also pass for a normalized map."""
        expected = _as_numpy(coords_grid(BATCH, HEIGHT, WIDTH))
        assert expected[..., 0].max() == pytest.approx(WIDTH - 1)
        assert expected[..., 1].max() == pytest.approx(HEIGHT - 1)

    def test_every_iteration_of_the_sequence_is_that_same_grid(self):
        """With the residual dead, all ``iters`` predictions coincide.

        This is the arm that catches an accumulator seeded from the wrong
        field: any drift across iterations shows up here as a non-zero spread.
        """
        model = self._model_with_a_dead_flow_head()
        stack = _as_numpy(model(_inputs(), training=True))
        expected = _as_numpy(coords_grid(BATCH, HEIGHT, WIDTH))
        for k in range(ITERS):
            np.testing.assert_allclose(stack[:, k], expected, rtol=0, atol=0.0)


# ---------------------------------------------------------------------
# 6. The shared oracles
# ---------------------------------------------------------------------


class TestTheSharedOracleAdoptions:
    """One arm per oracle. Nothing here is re-implemented locally."""

    def test_the_round_trip_reproduces_the_output_values_exactly(self):
        report = measure_roundtrip(_build, _inputs, training=False)
        # The forward RESAMPLES (`sample_at_pixel_coords`), so a non-zero
        # self-spread would be a finding rather than a nuisance: nothing in the
        # loop is stochastic. Asserted before the round-trip claim, so that the
        # claim is not silently measuring the sampler.
        assert report["self_max_delta"] == 0.0, (
            f"DocScannerRectifier is not deterministic (self spread "
            f"{report['self_max_delta']:.6e})")
        assert_roundtrip_output_values(report, atol=0.0)

    def test_the_weights_are_restored_before_the_loaded_model_is_called(self):
        report = measure_roundtrip(_build, _inputs, training=False)
        assert report["call_count_before_weight_read"] == 0
        assert_weights_restored_before_first_call(report, atol=0.0)

    def test_the_lazy_build_costs_nothing(self):
        report = assert_lazy_build_costs_nothing(
            _build, _inputs, input_shape=INPUT_SHAPE, atol=0.0)
        assert report["n_weights"] == report["n_weights_reloaded"]
        assert report["perturb_liveness"] > 0.0

    def test_the_explicit_build_matches_the_lazy_build_by_weight_path(self):
        """``build(shape)`` and a first call must produce the same tree.

        This is the H-6 claim made measurable: ``build()`` calls
        ``materialize_sublayers``, and if it did not, the explicit path would
        materialize nothing and the two weight-path sets would differ.
        """
        report = measure_build_parity(_build, _inputs, input_shape=INPUT_SHAPE)
        assert_build_parity(report, autoname_stems=(), expect_path_collisions=0)

    def test_gradients_reach_every_trainable_weight(self):
        keras.utils.set_random_seed(0)
        model = _built()
        report = assert_gradients_reach_every_trainable_weight(
            model, _inputs(), training=True)
        assert len(report) == len(model.trainable_weights)

    def test_the_gradient_assertion_can_fail(self):
        """RED proof for the arm above, using the oracle's OWN injection."""
        keras.utils.set_random_seed(0)
        model = _built()
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(
                    model, _inputs(), training=True)

    def test_the_forward_satisfies_its_contract(self, built_model):
        _contract(built_model(_inputs(), training=False))

    def test_the_smoke_contract_rejects_a_broken_forward(self, built_model):
        """The oracle requires ``AssertionError`` SPECIFICALLY."""
        messages = assert_contract_rejects_a_broken_forward(
            built_model, _inputs(), _contract)
        assert messages

    def test_the_hidden_dim_is_a_STRUCTURAL_knob(self):
        """It changes the weight-SHAPE signature, so it is pinned on that.

        ``hidden_dim`` cannot move alone: it is one half of the encoder's
        output and one half of the GRU's input, and the constructor asserts
        both relationships. So the builders move the whole consistent triple.
        """
        assert_structural_knob_changes_weights(
            {
                8: lambda: _warm(_build()),
                12: lambda: _warm(_build(
                    hidden_dim=12, context_dim=12,
                    gru_input_dim=24, fnet_output_dim=24,
                    motion_output_dim=12)),
            },
            knob="hidden_dim",
        )


def _contract(output) -> None:
    """The forward contract. Shared with the meta-test so it is falsifiable."""
    assert not isinstance(output, (dict, list, tuple)), (
        f"DocScannerRectifier returns a single backward map at inference, got "
        f"{type(output)}")
    assert tuple(output.shape) == (BATCH, HEIGHT, WIDTH, FLOW_CHANNELS), (
        tuple(output.shape))
    assert_finite(output)


# ---------------------------------------------------------------------
# 7. Serialization through a REAL save/load, and the config surface
# ---------------------------------------------------------------------


class TestSerialization:

    def test_a_real_save_and_load_model_reproduces_the_values(self, tmp_path):
        """``model.save()`` / ``keras.models.load_model()``, not a config
        round trip. Writes to pytest's ``tmp_path``; NEVER into repo-root
        ``results/``."""
        keras.utils.set_random_seed(0)
        model = _built()
        x = _inputs()
        before = _as_numpy(model(x, training=False))

        path = tmp_path / "rectifier.keras"
        model.save(path)
        reloaded = keras.models.load_model(path)

        after = _as_numpy(reloaded(x, training=False))
        np.testing.assert_allclose(after, before, rtol=0, atol=0.0)

    def test_the_training_form_survives_the_same_round_trip(self, tmp_path):
        """The sequence output is what the step-12 loss consumes, so it is the
        form that must survive -- and it is NOT the form the archive traces."""
        keras.utils.set_random_seed(0)
        model = _built()
        x = _inputs()
        before = _as_numpy(model(x, training=True))

        path = tmp_path / "rectifier_seq.keras"
        model.save(path)
        after = _as_numpy(keras.models.load_model(path)(x, training=True))
        np.testing.assert_allclose(after, before, rtol=0, atol=0.0)

    def test_get_config_round_trips_every_constructor_argument(self):
        """Every named parameter of ``__init__`` appears in ``get_config``.

        Derived from the signature rather than from a hand-typed list, so a
        width added to the constructor and forgotten in ``get_config`` fails
        here instead of being restored as a default on the next reload.
        """
        parameters = [
            name for name, param
            in inspect.signature(DocScannerRectifier.__init__).parameters.items()
            if name not in ("self", "kwargs")
            and param.kind is not inspect.Parameter.VAR_KEYWORD
        ]
        config = _build().get_config()
        missing = [name for name in parameters if name not in config]
        assert not missing, f"get_config drops {missing}"

    def test_the_config_reconstructs_an_identical_configuration(self):
        config = _build().get_config()
        rebuilt = DocScannerRectifier.from_config(config)
        assert rebuilt.get_config() == config


# ---------------------------------------------------------------------
# 8. The variant surface
# ---------------------------------------------------------------------


class TestTheVariantTable:

    def test_there_is_exactly_one_row_and_it_is_docscanner_l(self):
        assert list(DocScannerRectifier.MODEL_VARIANTS) == ["docscanner-l"]

    def test_the_row_is_derived_from_the_width_table_not_restated(self):
        """A second hand-maintained copy of fifteen numbers is a lockstep
        invariant. This asserts there is only one copy."""
        from dl_techniques.models.vision.image_restoration.doc_scanner.components import (
            _VARIANT_SPEC,
        )
        row = DocScannerRectifier.MODEL_VARIANTS["docscanner-l"]
        spec = _VARIANT_SPEC["docscanner-l"]
        # LITERAL, never read from the module's own exclusion tuple: both
        # sides would then move together and this would stop measuring
        # anything. `mask_head_output_channels` is structural and derived
        # inside the update block; the two `seg_*` rows belong to the
        # SEGMENTATION stage (step 8) and have never been rectifier arguments.
        not_constructor_args = (
            "mask_head_output_channels", "seg_mid_channels", "seg_out_channels")
        for key, value in spec.items():
            if key in not_constructor_args:
                assert key not in row
                continue
            assert row[key] == (list(value) if isinstance(value, tuple)
                                else value), key
        assert row["iters"] == REFINE_ITERATIONS

    def test_the_shipped_variant_builds_and_runs(self):
        """The one arm that exercises the REAL widths, at a small resolution
        and a small trip count so it costs seconds rather than minutes."""
        model = create_doc_scanner_rectifier(iters=2)
        out = model(np.zeros((1, 32, 24, 3), dtype="float32"), training=False)
        assert tuple(out.shape) == (1, 32, 24, FLOW_CHANNELS)
        assert_finite(out)

    def test_pretrained_true_raises_naming_two_specific_divergences(self):
        with pytest.raises(NotImplementedError) as excinfo:
            DocScannerRectifier.from_variant("docscanner-l", pretrained=True)
        message = str(excinfo.value)
        assert "extractor.py:93" in message
        assert "D-012" in message
        assert "load_weights" in message

    def test_the_factory_function_raises_too(self):
        with pytest.raises(NotImplementedError):
            create_doc_scanner_rectifier(pretrained=True)

    def test_an_unknown_variant_lists_the_known_ones(self):
        with pytest.raises(ValueError, match="docscanner-l"):
            DocScannerRectifier.from_variant("docscanner-xl")

    def test_from_variant_overrides_reach_the_constructor(self):
        model = DocScannerRectifier.from_variant("docscanner-l", iters=5)
        assert model.iters == 5


# ---------------------------------------------------------------------
# 9. The registration key (H-3)
# ---------------------------------------------------------------------


class TestTheRegistrationKey:
    """A literal ``==``, never a save/load round trip.

    Save/load shares one in-process registry and cannot see a typo: the class
    is findable under whatever key it registered itself with.
    """

    def test_the_key_strips_both_the_family_and_the_subfamily(self):
        key = "dl_techniques.models.doc_scanner.model>DocScannerRectifier"
        assert keras.saving.get_registered_object(key) is DocScannerRectifier
        assert keras.saving.get_registered_name(DocScannerRectifier) == key

    def test_the_full_import_path_is_NOT_a_registration_key(self):
        assert keras.saving.get_registered_object(
            "dl_techniques.models.vision.image_restoration.doc_scanner.model"
            ">DocScannerRectifier"
        ) is None


# ---------------------------------------------------------------------
# 10. Structural facts the assembly is responsible for
# ---------------------------------------------------------------------


class TestTheAssemblyItself:

    def test_there_is_exactly_ONE_update_block_however_many_iterations(self):
        """``model.py:38`` constructs one ``BasicUpdateBlock``; the recurrence
        is in the state, not the parameters. ``iters`` separate blocks would
        train, would serialize, and would be a 12x larger architecture."""
        one = _warm(_build(iters=1))
        many = _warm(_build(iters=REFINE_ITERATIONS))
        assert len(one.weights) == len(many.weights)
        assert one.count_params() == many.count_params()

    def test_the_encoder_output_is_split_in_half_and_the_split_is_checked(self):
        with pytest.raises(ValueError, match="fnet_output_dim"):
            _build(fnet_output_dim=17)

    def test_the_feature_encoder_reduces_by_exactly_the_stride(
            self, built_model):
        fmap = built_model.fnet(_inputs(), training=False)
        assert tuple(fmap.shape) == (
            BATCH,
            HEIGHT // SPATIAL_DIVISOR,
            WIDTH // SPATIAL_DIVISOR,
            _SMALL["fnet_output_dim"],
        )

    @pytest.mark.parametrize("field", ["hidden_dim", "iters", "flow_head_hidden"])
    def test_a_non_positive_width_raises_naming_the_field(self, field):
        with pytest.raises(ValueError, match=field):
            _build(**{field: 0})
