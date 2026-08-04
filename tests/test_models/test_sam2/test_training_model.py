"""Guards for :class:`SAM2TrainingModel` -- the T-frame unrolled forward.

Every guard here was RED-proven against a wrong-VALUE mutation on GPU 1
(``CUDA_VISIBLE_DEVICES=1``), the device this package's gate runs on. Iteration
1 measured a bad gather that RAISES on CPU and SILENTLY CLAMPS TO ZEROS on GPU,
so a ``pytest.raises``-only guard is not evidence here; every guard that can
assert a value does.

THE TRAP THIS FILE EXISTS AROUND. At random initialization ``SAM2``'s
object-score head emits a NEGATIVE logit for every row, and
``_suppress_absent_object`` (D-043) then replaces every mask logit with the
uniform constant ``NO_OBJ_SCORE = -1024``. MEASURED on this wrapper at the
``tiny`` geometry: the whole ``(B, T, h, w)`` output is exactly ``-1024`` and
every frame slice is bit-identical to every other. A "the frames differ" guard
written without pinning the score is therefore not merely weak -- it is
permanently RED on correct code, and a "the frames are the same" guard is
permanently green on a dead loop. :func:`pin_object_score` is imported from
``test_model.py`` rather than re-written, and is used wherever the mask VALUES
are the thing under test.

The clip fixture here is TEMPORARY. Step 2 builds the real synthetic
moving-instance video source and step 4 replaces
:func:`clip_inputs` with it, so that every guard runs on a shape the data path
can actually emit (a 208-test SAM gate once missed a crash at shipped defaults
because a fixture hand-built a shape the pipeline cannot produce).
"""

from typing import Any, Dict, Optional
from unittest import mock

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.models.sam2.model import NO_OBJ_SCORE, SAM2, create_sam2
from dl_techniques.models.sam2.training_model import (
    OUTPUT_KEYS,
    SAM2_LOW_RES_LOGITS,
    SAM2_OBJECT_SCORE_LOGITS,
    SAM2TrainingModel,
)

from .test_model import pin_object_score

BATCH = 2
FRAMES = 3


# ---------------------------------------------------------------------
# fixtures (TEMPORARY -- replaced by `train/sam2/data.py` at step 4)
# ---------------------------------------------------------------------


def trainer(
        num_frames: int = FRAMES,
        object_score: Optional[float] = None,
        **overrides: Any,
) -> SAM2TrainingModel:
    """Build and build a ``tiny`` wrapper, optionally pinning the score.

    :param num_frames: Clip length ``T``.
    :type num_frames: int
    :param object_score: When given, pin every object-score logit to it. Use a
        POSITIVE value whenever mask VALUES are under test; see the module
        docstring.
    :type object_score: Optional[float]
    :param overrides: Forwarded to :func:`create_sam2`.
    :type overrides: Any
    :return: A built wrapper.
    :rtype: SAM2TrainingModel
    """
    overrides.setdefault("multimask_output", False)
    sam2 = create_sam2("tiny", **overrides)
    model = SAM2TrainingModel(sam2, num_frames=num_frames)
    model.build(None)
    if object_score is not None:
        pin_object_score(sam2, object_score)
    return model


def clip_inputs(
        model: SAM2TrainingModel,
        batch: int = BATCH,
        seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Build one clip batch with a frame-0 point prompt.

    :param model: The wrapper whose geometry the clip must match.
    :type model: SAM2TrainingModel
    :param batch: Batch size.
    :type batch: int
    :param seed: RNG seed.
    :type seed: int
    :return: The input dict :meth:`SAM2TrainingModel.call` accepts.
    :rtype: Dict[str, np.ndarray]
    """
    rng = np.random.default_rng(seed)
    size = model.sam2.image_size
    return {
        "image": rng.uniform(
            0.0, 255.0,
            (batch, model.num_frames, size, size, 3)).astype("float32"),
        "point_coords": rng.uniform(
            4.0, float(size) - 4.0, (batch, 1, 2)).astype("float32"),
        "point_labels": np.ones((batch, 1), dtype="int32"),
    }


def targets_for(model: SAM2TrainingModel, batch: int = BATCH
                ) -> Dict[str, np.ndarray]:
    """Zero targets matching every output key, for a stock ``fit()`` step.

    :param model: The wrapper.
    :type model: SAM2TrainingModel
    :param batch: Batch size.
    :type batch: int
    :return: A dict keyed exactly like :meth:`SAM2TrainingModel.call`'s output.
    :rtype: Dict[str, np.ndarray]
    """
    grid = model.sam2.feature_grid * 4
    return {
        SAM2_LOW_RES_LOGITS: np.zeros(
            (batch, model.num_frames, grid, grid), dtype="float32"),
        SAM2_OBJECT_SCORE_LOGITS: np.zeros(
            (batch, model.num_frames, 1), dtype="float32"),
    }


def max_abs(tensor: Any) -> float:
    """Largest absolute element of a tensor.

    :param tensor: Any tensor-like.
    :type tensor: Any
    :return: ``max(abs(tensor))``.
    :rtype: float
    """
    return float(np.max(np.abs(np.asarray(tensor))))


def encoder_gradients(model: SAM2TrainingModel, grads: Any) -> list:
    """Pick out the image encoder's gradients from a full gradient list.

    :param model: The wrapper the gradients came from.
    :type model: SAM2TrainingModel
    :param grads: The list ``tape.gradient`` returned.
    :type grads: Any
    :return: ``(path, gradient)`` pairs for image-encoder variables.
    :rtype: list
    """
    return [
        (v.path, g) for v, g in zip(model.trainable_variables, grads)
        if "image_encoder" in v.path
    ]


# ---------------------------------------------------------------------
# G1.1 -- SAM2.call is never invoked
# ---------------------------------------------------------------------


class TestSAM2CallSpy:
    """``SAM2.call`` / ``SAM2.__call__`` must be invoked ZERO times."""

    def test_neither_entry_point_is_invoked_during_fit(self) -> None:
        """A whole ``fit()`` step must not touch ``SAM2``'s own forward."""
        model = trainer(object_score=5.0)
        inputs, targets = clip_inputs(model), targets_for(model)
        model.compile(
            optimizer="adam",
            loss={SAM2_LOW_RES_LOGITS: "mse", SAM2_OBJECT_SCORE_LOGITS: "mse"},
            jit_compile=False,
        )
        with mock.patch.object(
                SAM2, "call", autospec=True) as call_spy, \
                mock.patch.object(
                    SAM2, "__call__", autospec=True) as dunder_spy:
            model.fit(inputs, targets, epochs=1, batch_size=BATCH, verbose=0)
            assert call_spy.call_count == 0, (
                f"SAM2.call was invoked {call_spy.call_count} times during "
                "fit(); the wrapper must drive the submodules directly")
            assert dunder_spy.call_count == 0, (
                f"SAM2.__call__ was invoked {dunder_spy.call_count} times "
                "during fit()")

    def test_the_spy_can_actually_count(self) -> None:
        """Fixture validity: the same patch DOES see a real ``SAM2`` call.

        Without this, a spy that patched the wrong attribute would report
        ``0`` for the same reason a correct implementation does.
        """
        model = trainer(object_score=5.0)
        image = clip_inputs(model)["image"][:, 0]
        with mock.patch.object(SAM2, "__call__", autospec=True) as spy:
            model.sam2({"image": image})
            assert spy.call_count == 1


# ---------------------------------------------------------------------
# G1.2 / G1.3 -- gradients and the fed-back boundaries
# ---------------------------------------------------------------------


def last_frame_mask_gradients(model: SAM2TrainingModel, inputs: Any,
                              frame: int = -1) -> Any:
    """Gradients of ``sum(low_res_logits[:, frame])`` w.r.t. every variable.

    :param model: The wrapper.
    :type model: SAM2TrainingModel
    :param inputs: A clip input dict.
    :type inputs: Any
    :param frame: Which frame's mask term to differentiate.
    :type frame: int
    :return: The list ``tape.gradient`` returned.
    :rtype: Any
    """
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        loss = ops.sum(outputs[SAM2_LOW_RES_LOGITS][:, frame])
    return tape.gradient(loss, model.trainable_variables)


class TestGradientReachesTheEncoder:
    """G1.2: the last frame's MASK loss must reach the image encoder."""

    def test_the_last_frames_mask_loss_reaches_the_image_encoder(self) -> None:
        """MEASURED 7.2e4 at this fixture; the assertion is only ``> 0``."""
        model = trainer(object_score=5.0)
        grads = last_frame_mask_gradients(model, clip_inputs(model))
        hits = encoder_gradients(model, grads)
        assert hits, "no image-encoder variables were found"
        assert all(g is not None for _, g in hits), (
            "some image-encoder gradients came back None; the encoder is "
            "disconnected from the last frame's mask loss")
        assert max(max_abs(g) for _, g in hits) > 0.0

    def test_an_unpinned_score_makes_the_whole_mask_output_a_constant(
            self) -> None:
        """The trap, pinned as a permanent test rather than a comment.

        At random init every object score is negative, so D-043's suppression
        makes every mask logit exactly ``NO_OBJ_SCORE``. Any guard that reads
        mask VALUES without pinning the score is measuring this constant.
        """
        model = trainer()
        outputs = model(clip_inputs(model), training=True)
        scores = np.asarray(outputs[SAM2_OBJECT_SCORE_LOGITS])
        assert np.all(scores <= 0.0), (
            "this fixture no longer lands on the negative-score branch; "
            f"scores were {scores.ravel()}")
        logits = np.asarray(outputs[SAM2_LOW_RES_LOGITS])
        np.testing.assert_allclose(logits, NO_OBJ_SCORE)


class TestStopGradientBoundaries:
    """G1.3: every fed-back state boundary is detached, proven two-sided."""

    def test_no_mem_embed_gets_exactly_zero_gradient_from_a_later_frame(
            self) -> None:
        """The two-sided probe point.

        ``no_mem_embed`` is added ONLY on frame 0 (D-027's shipped branch), so
        the last frame can reach it only through the memory bank -- which is
        detached. MEASURED: the gradient is not ``None`` but exactly ``0.0``,
        because the ``concatenate`` that builds the frame axis keeps the
        variable structurally connected. A ``None``-counting assertion would
        therefore go red for a reason unrelated to the property (D-037).
        """
        model = trainer(object_score=5.0, num_frames=FRAMES)
        grads = last_frame_mask_gradients(model, clip_inputs(model))
        pairs = {v.path: g for v, g in zip(model.trainable_variables, grads)}
        grad = next(g for path, g in pairs.items()
                    if path.endswith("no_mem_embed"))
        assert grad is not None, (
            "no_mem_embed's gradient came back None; the measured shape of "
            "this boundary is EXACT ZEROS, so a None here means the graph "
            "structure changed, not that the boundary tightened")
        assert max_abs(grad) == 0.0, (
            f"no_mem_embed moved {max_abs(grad)} under the last frame's mask "
            "loss alone; the memory boundary is leaking gradient backwards "
            "through the clip")

    def test_the_probe_point_is_live_on_its_own_frame(self) -> None:
        """Fixture validity: ``no_mem_embed`` is reachable at all.

        Without this, the zero above would be indistinguishable from a
        variable that no loss can ever move.
        """
        model = trainer(object_score=5.0, num_frames=FRAMES)
        grads = last_frame_mask_gradients(model, clip_inputs(model), frame=0)
        pairs = {v.path: g for v, g in zip(model.trainable_variables, grads)}
        grad = next(g for path, g in pairs.items()
                    if path.endswith("no_mem_embed"))
        assert grad is not None and max_abs(grad) > 0.0, (
            "frame 0's own mask loss does not move no_mem_embed, so the "
            "zero measured across frames proves nothing")

    def test_the_memory_encoder_is_fully_disconnected_from_the_loss(
            self) -> None:
        """The complementary arm, and its shape is DIFFERENT.

        The memory encoder's output reaches nothing except the bank, and the
        bank detaches on insertion, so its variables are structurally
        disconnected -- MEASURED as ``None`` for all 40 of them, not as zeros.
        The two arms of this class therefore assert two different measured
        facts, and neither was predicted correctly from reading the code.
        """
        model = trainer(object_score=5.0)
        grads = last_frame_mask_gradients(model, clip_inputs(model))
        hits = [(v.path, g) for v, g in zip(model.trainable_variables, grads)
                if "memory_encoder" in v.path]
        assert hits, "no memory-encoder variables were found"
        assert all(g is None for _, g in hits), (
            "some memory-encoder gradients are non-None: "
            f"{[p for p, g in hits if g is not None]}")


# ---------------------------------------------------------------------
# G1.4 -- the loop really ran T frames
# ---------------------------------------------------------------------


class TestTheLoopRanEveryFrame:
    """G1.4: both a count assertion and a value assertion, separately."""

    @pytest.mark.parametrize("num_frames", [1, 2, 4])
    def test_the_output_frame_axis_equals_num_frames(
            self, num_frames: int) -> None:
        """The HOW-MANY arm. Structurally blind to a frozen loop body."""
        model = trainer(num_frames=num_frames, object_score=5.0)
        outputs = model(clip_inputs(model), training=True)
        assert outputs[SAM2_LOW_RES_LOGITS].shape[1] == num_frames
        assert outputs[SAM2_OBJECT_SCORE_LOGITS].shape[1] == num_frames
        assert set(outputs) == set(OUTPUT_KEYS)

    def test_every_frame_decodes_a_different_mask(self) -> None:
        """The WHICH arm. Requires the pinned score; see the module docstring.

        MEASURED separations at this fixture: 220.6 / 220.6 / 301.3.
        """
        model = trainer(object_score=5.0)
        logits = np.asarray(
            model(clip_inputs(model), training=True)[SAM2_LOW_RES_LOGITS])
        for i in range(model.num_frames):
            for j in range(i + 1, model.num_frames):
                separation = float(np.max(np.abs(logits[:, i] - logits[:, j])))
                assert separation > 1.0, (
                    f"frames {i} and {j} decoded the same mask (max-abs "
                    f"{separation}); the loop body may not depend on t")

    def test_a_single_frame_clip_still_runs(self) -> None:
        """``T = 1`` is the control that isolates the video machinery."""
        model = trainer(num_frames=1, object_score=5.0)
        outputs = model(clip_inputs(model), training=True)
        assert tuple(outputs[SAM2_LOW_RES_LOGITS].shape)[:2] == (BATCH, 1)


# ---------------------------------------------------------------------
# G1.5 -- multimask_output is refused
# ---------------------------------------------------------------------


class TestMultimaskRefusal:
    """G1.5: a ``raises`` arm AND a value arm on the shipped path."""

    def test_multimask_output_true_is_refused_by_name(self) -> None:
        """The message must name the frame/mask interleave, not just 'no'."""
        sam2 = create_sam2("tiny", multimask_output=True)
        with pytest.raises(ValueError, match="FRAME-major"):
            SAM2TrainingModel(sam2, num_frames=FRAMES)

    def test_the_shipped_false_path_returns_a_mask_axis_of_num_frames(
            self) -> None:
        """The value arm.

        A ``raises``-only guard is exactly the shape that went permanently
        green on GPU in iteration 1, so the refusal is paired with an
        assertion that the path it protects actually produces ``M == 1`` per
        frame and therefore an unambiguous frame axis.
        """
        model = trainer(object_score=5.0)
        outputs = model(clip_inputs(model), training=True)
        assert model.sam2.multimask_output is False
        assert outputs[SAM2_LOW_RES_LOGITS].shape[1] == model.num_frames
        assert model.sam2.compute_output_shape(
            {"image": (BATCH, 64, 64, 3)})["low_res_logits"][1] == 1


# ---------------------------------------------------------------------
# G1.6 -- frame 0 and frames > 0 take different memory paths
# ---------------------------------------------------------------------


class TestFrameZeroTakesADifferentPath:
    """G1.6: the empty-memory branch, both configurations."""

    @staticmethod
    def _empty_bank_condition(model: SAM2TrainingModel) -> Any:
        """Run ``_condition`` for frame 0 against a freshly emptied bank.

        :param model: The wrapper.
        :type model: SAM2TrainingModel
        :return: ``(raw_features, positions, conditioned)``.
        :rtype: Any
        """
        from dl_techniques.models.sam2.memory_bank import SAM2MemoryBank

        size = model.sam2.image_size
        rng = np.random.default_rng(3)
        image = rng.uniform(0.0, 255.0, (BATCH, size, size, 3)).astype("f4")
        encoded = model.sam2.image_encoder(image, training=False)
        features = encoded["vision_features"]
        positions = encoded["vision_pos_enc"][-1]
        bank = SAM2MemoryBank(
            num_maskmem=model.sam2.num_maskmem,
            mem_dim=model.sam2.mem_dim,
            hidden_dim=model.sam2.hidden_dim,
        )
        conditioned = model._condition(bank, features, positions, 0, False)
        return features, positions, conditioned

    def test_frame_zero_adds_exactly_no_mem_embed(self) -> None:
        """Shipped config: the conditioned features are raw + ``no_mem_embed``.

        The value is asserted against a non-zero assigned embedding, because
        the shipped initializer is ``zeros`` -- at which "adds the embedding"
        and "does nothing at all" are the same tensor.
        """
        model = trainer()
        marker = np.full(model.sam2.no_mem_embed.shape, 0.75, dtype="float32")
        model.sam2.no_mem_embed.assign(marker)
        features, _, conditioned = self._empty_bank_condition(model)
        np.testing.assert_allclose(
            np.asarray(conditioned),
            np.asarray(features) + 0.75,
            rtol=0, atol=1e-5)

    def test_frames_after_zero_do_not_take_the_no_mem_embed_branch(
            self) -> None:
        """The other half of G1.6: frames > 0 read a NON-empty bank.

        This is the reachable form of "the two paths differ". The plan's
        named mutation -- ``directly_add_no_mem_embed=False`` -- turned out to
        be UNREACHABLE at this geometry; see
        :meth:`test_the_documented_empty_memory_fallback_is_unreachable`.
        """
        model = trainer(object_score=5.0)
        marker = np.full(model.sam2.no_mem_embed.shape, 0.75, dtype="float32")
        model.sam2.no_mem_embed.assign(marker)
        outputs = model(clip_inputs(model), training=True)
        logits = np.asarray(outputs[SAM2_LOW_RES_LOGITS])
        # If frames > 0 took frame 0's branch they would be conditioned by the
        # same constant shift and, on identical features, would agree.
        assert float(np.max(np.abs(logits[:, 0] - logits[:, 1]))) > 1.0

    def test_the_documented_empty_memory_fallback_is_unreachable(self) -> None:
        """A LATENT DEFECT in ``models/sam2/model.py``, measured here.

        D-027's ``directly_add_no_mem_embed=False`` branch builds a ONE-token
        zero memory, and memory attention's ``repeat_k=True`` rotary path
        requires the memory length to be an exact multiple of ``H * W`` (16 at
        the ``tiny`` geometry). MEASURED: the branch raises::

            ValueError: with repeat_k=True the rotated key length
            (1 = 1 - 0) must be an exact multiple of feat_shape H*W (16)

        This is NOT introduced by the training wrapper -- ``SAM2.stream_step``
        reaches the identical branch through
        ``SAM2._condition_on_memory``. It is out of scope for this step
        (``models/sam2/``'s eight pre-existing modules stay byte-unchanged), so
        it is PINNED here rather than left to be rediscovered. Both the
        reference and this port leave the branch off in every shipped config.
        """
        model = trainer(directly_add_no_mem_embed=False)
        with pytest.raises(ValueError, match="exact multiple of feat_shape"):
            self._empty_bank_condition(model)

    def test_memory_attention_is_skipped_on_frame_zero_only(self) -> None:
        """A call spy: ``T`` frames must run ``T - 1`` memory attentions."""
        model = trainer(num_frames=FRAMES, object_score=5.0)
        real = model.sam2.memory_attention.__class__.__call__
        calls = []

        def counting(self_, *args: Any, **kwargs: Any) -> Any:
            calls.append(1)
            return real(self_, *args, **kwargs)

        with mock.patch.object(
                model.sam2.memory_attention.__class__, "__call__", counting):
            model(clip_inputs(model), training=True)
        assert len(calls) == FRAMES - 1, (
            f"memory attention ran {len(calls)} times over {FRAMES} frames; "
            "frame 0 must skip it under directly_add_no_mem_embed")


# ---------------------------------------------------------------------
# tracing / serialization
# ---------------------------------------------------------------------


class TestTracing:
    """The whole unrolled loop must trace under a static input signature."""

    def test_call_traces_and_does_not_retrace_per_batch(self) -> None:
        """One concrete function; a second call with new values reuses it."""
        model = trainer(object_score=5.0)
        inputs = clip_inputs(model)
        signature = [{
            key: tf.TensorSpec(value.shape, value.dtype)
            for key, value in inputs.items()
        }]

        @tf.function(input_signature=signature)
        def traced(batch: Dict[str, Any]) -> Dict[str, Any]:
            return model(batch, training=True)

        first = traced(inputs)
        settled = len(traced._list_all_concrete_functions())
        second = traced(clip_inputs(model, seed=11))
        traced(clip_inputs(model, seed=12))
        # MEASURED: TF settles at 2 concrete functions for this signature, not
        # 1 -- so the assertion is that the count does NOT GROW per batch,
        # which is the property that matters. A `== 1` assertion here would be
        # red on correct code for a reason unrelated to the memory bank.
        assert len(traced._list_all_concrete_functions()) == settled
        assert max_abs(
            np.asarray(first[SAM2_LOW_RES_LOGITS])
            - np.asarray(second[SAM2_LOW_RES_LOGITS])) > 0.0, (
            "two different clips traced to the same output; the per-call "
            "memory bank may be holding tensors from an earlier trace")

    def test_one_fit_step_runs_under_stock_compile(self) -> None:
        """No custom ``train_step``; the loop trains as written."""
        model = trainer(object_score=5.0)
        model.compile(
            optimizer="adam",
            loss={SAM2_LOW_RES_LOGITS: "mse", SAM2_OBJECT_SCORE_LOGITS: "mse"},
            jit_compile=False,
        )
        history = model.fit(
            clip_inputs(model), targets_for(model),
            epochs=1, batch_size=BATCH, verbose=0)
        assert np.isfinite(history.history["loss"][0])


class TestXLARefusal:
    """``jit_compile=False`` is MANDATORY, and that is measured here.

    Keras 3.8's ``fit()`` defaults to ``jit_compile='auto'``, which picks XLA
    on GPU. ``Hiera``'s stem resizes its learned positional embedding
    bicubically, and ``ResizeBicubic`` has no ``XLA_GPU_JIT`` kernel. Step 5's
    trainer must therefore compile with ``jit_compile=False``; this pair of
    tests is what stops that requirement being quietly dropped as
    superstition.
    """

    def test_xla_fit_fails_on_the_bicubic_stem_resize(self) -> None:
        """The negative arm, asserted by the op name in the message."""
        model = trainer(object_score=5.0)
        model.compile(
            optimizer="adam",
            loss={SAM2_LOW_RES_LOGITS: "mse", SAM2_OBJECT_SCORE_LOGITS: "mse"},
            jit_compile=True,
        )
        with pytest.raises(Exception, match="ResizeBicubic"):
            model.fit(clip_inputs(model), targets_for(model),
                      epochs=1, batch_size=BATCH, verbose=0)

    def test_the_same_step_succeeds_without_xla(self) -> None:
        """The positive control, so the raise above is about XLA and nothing
        else."""
        model = trainer(object_score=5.0)
        model.compile(
            optimizer="adam",
            loss={SAM2_LOW_RES_LOGITS: "mse", SAM2_OBJECT_SCORE_LOGITS: "mse"},
            jit_compile=False,
        )
        history = model.fit(
            clip_inputs(model), targets_for(model),
            epochs=1, batch_size=BATCH, verbose=0)
        assert np.isfinite(history.history["loss"][0])


class TestSerialization:
    """``get_config`` / ``from_config`` must carry the wrapped ``SAM2``."""

    def test_round_trip_rebuilds_an_equivalent_wrapper(self) -> None:
        model = trainer(num_frames=2)
        restored = SAM2TrainingModel.from_config(model.get_config())
        assert restored.num_frames == 2
        assert isinstance(restored.sam2, SAM2)
        assert restored.sam2.image_size == model.sam2.image_size
        assert restored.sam2.multimask_output is False


# ---------------------------------------------------------------------
# G1.7 -- the measured dead-component partition
# ---------------------------------------------------------------------


class TestDeadComponentPartition:
    """G1.7: which guards survive a DEAD ``memory_attention``.

    Iteration 1 falsified "all guards go RED under a dead component" at five
    consecutive steps, and it is falsified again here. MEASURED with the
    ``memory_attention`` call in ``SAM2TrainingModel._condition`` replaced by
    ``conditioned = tokens``: **26 of the 30 tests in this file stayed GREEN**
    and only 4 fired.

    GREEN under the dead component -- i.e. these guards are NOT evidence about
    memory conditioning:

    * G1.1 (both spy arms), G1.2 (encoder gradient), G1.3 (all three
      stop-gradient arms), G1.5 (both), tracing (both), XLA (both),
      serialization, and all five construction refusals;
    * **G1.4's VALUE arm** -- ``test_every_frame_decodes_a_different_mask``.
      Frames differ because each decodes a DIFFERENT IMAGE, not because the
      memory is alive. That is the plan's own predicted RED arm, measured
      GREEN, and it is why the first test below exists;
    * G1.6's ``test_frame_zero_adds_exactly_no_mem_embed`` and
      ``test_frames_after_zero_do_not_take_the_no_mem_embed_branch``.

    RED under the dead component (4):
    ``test_memory_attention_is_skipped_on_frame_zero_only`` (0 calls instead
    of 2), ``test_the_documented_empty_memory_fallback_is_unreachable`` (the
    unreachable branch stops being reached), the second test below (its own
    fixture-validity arm), and -- unpredicted --
    ``test_an_unpinned_score_makes_the_whole_mask_output_a_constant``, because
    killing the memory path moves some rows' object score across zero at
    random init. That last one is an incidental sensitivity, not a memory
    guard, and it is recorded so a future reader does not count it as one.
    """

    def test_a_dead_memory_attention_still_leaves_the_frames_different(
            self) -> None:
        """Which is why the count guards are not evidence about the memory.

        Even with memory attention returning its ``tokens`` argument
        unchanged, frames still differ -- each decodes a DIFFERENT IMAGE. So
        G1.4's separation assertion is evidence that the loop body depends on
        ``t``, and NOT evidence that the memory conditioning is alive.
        """
        model = trainer(object_score=5.0)
        attention = model.sam2.memory_attention

        def dead(self_: Any, tokens: Any, memory: Any, **kwargs: Any) -> Any:
            del memory, kwargs
            return tokens

        with mock.patch.object(attention.__class__, "call", dead):
            logits = np.asarray(
                model(clip_inputs(model), training=True)[SAM2_LOW_RES_LOGITS])
        separation = float(np.max(np.abs(logits[:, 0] - logits[:, 1])))
        assert separation > 1.0, (
            "with memory attention DEAD the frames became identical; the "
            "documented partition no longer holds and G1.4's value arm has "
            "silently become a memory-conditioning guard")

    def test_a_dead_memory_attention_does_change_the_answer(self) -> None:
        """Fixture validity: the mutation is not a no-op on frames > 0."""
        model = trainer(object_score=5.0)
        inputs = clip_inputs(model)
        alive = np.asarray(
            model(inputs, training=True)[SAM2_LOW_RES_LOGITS])
        attention = model.sam2.memory_attention

        def dead(self_: Any, tokens: Any, memory: Any, **kwargs: Any) -> Any:
            del memory, kwargs
            return tokens

        with mock.patch.object(attention.__class__, "call", dead):
            killed = np.asarray(
                model(inputs, training=True)[SAM2_LOW_RES_LOGITS])
        np.testing.assert_allclose(alive[:, 0], killed[:, 0], rtol=0, atol=0)
        assert float(np.max(np.abs(alive[:, 1:] - killed[:, 1:]))) > 0.0, (
            "killing memory attention changed nothing on frames > 0; the "
            "memory path is not reaching the decode at all")


class TestConstructionRefusals:
    """The three constructor refusals, each by value."""

    def test_a_non_sam2_is_refused(self) -> None:
        with pytest.raises(ValueError, match="requires a SAM2 instance"):
            SAM2TrainingModel(keras.layers.Dense(4), num_frames=2)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_a_non_positive_num_frames_is_refused(self, bad: int) -> None:
        with pytest.raises(ValueError, match="num_frames must be >= 1"):
            SAM2TrainingModel(create_sam2("tiny"), num_frames=bad)

    def test_a_prompt_less_clip_is_refused(self) -> None:
        model = trainer()
        inputs = clip_inputs(model)
        del inputs["point_coords"], inputs["point_labels"]
        with pytest.raises(ValueError, match="requires a frame-0 prompt"):
            model(inputs, training=True)

    def test_half_a_point_prompt_is_refused(self) -> None:
        model = trainer()
        inputs = clip_inputs(model)
        del inputs["point_labels"]
        with pytest.raises(ValueError, match="must be supplied together"):
            model(inputs, training=True)
