"""
Correctness Instruments for SAM (`src/dl_techniques/models/SAM/SAM1/`)
=================================================================

This module is deliberately NOT a second copy of ``test_model.py``. That suite
is shape- and ``isinstance``-oriented and, as audited, catches none of the
package's known correctness defects. This module supplies the *instruments*
those guards need:

1. :func:`build_reduced_sam` -- a reduced-*width* SAM at real SAM *geometry*
   (``patch_size=16``, ``window_size=14``, a non-empty ``global_attn_indexes``
   and ``use_rel_pos=True``). No fixture in ``test_model.py`` sets either of the
   last two, so the entire relative-position code path in
   ``image_encoder.py`` is executed by zero pre-existing tests.
2. :func:`seed_nonzero_weights` -- deterministic non-zero weights. This is not a
   nicety: at initialization **95 of the 192** weights of this fixture are
   exactly all-zero (including ``rel_pos_h`` / ``rel_pos_w`` and
   ``not_a_point_embed``), which makes several probes structurally unable to
   observe the thing they claim to measure.
3. :func:`roundtrip_low_res_logits` -- a value-level ``.keras`` round-trip
   instrument comparing ``low_res_logits`` VALUES and weight COUNTS. It never
   compares binarized masks (a uint8 comparison hides drift) and never asserts
   ``isinstance``.
4. :func:`gradient_none_counts` -- a ``GradientTape`` probe returning an exact
   ``(n_none, n_total)`` per output key, so guards can pin a count rather than a
   vacuous ``> 0``.

Every instrument here carries its own RED proof: a test that demonstrates the
instrument reports failure when the property it measures is deliberately broken.
An instrument that cannot be driven RED is not coverage.

Measured on GPU 1 (RTX 4070) at commit 004d431d (pre-repair HEAD):
  * fixture weight count = 192 (191 trainable)  <- CONTROL for plan steps 3 & 4
  * ``.keras`` round-trip max abs diff on ``low_res_logits`` = 0.0
  * ``load_model`` wall clock = ~1.0 s at this fixture size
  * gradients: masks 191/191 None, low_res_logits 16/191 None,
    iou_predictions 34/191 None
"""

import os
import gc
import tempfile
from typing import Any, Dict, Optional, Callable, Tuple

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.SAM.SAM1.model import SAM
from dl_techniques.models.SAM.SAM1.image_encoder import (
    ImageEncoderViT,
    WindowedAttentionWithRelPos,
)
from dl_techniques.models.SAM.SAM1.preprocessing import resize_longest_side
from dl_techniques.models.SAM.SAM1.prompt_encoder import (
    PromptEncoder,
    PositionEmbeddingRandom,
)
from dl_techniques.models.SAM.SAM1.mask_decoder import MaskDecoder
from dl_techniques.models.SAM.SAM1.transformer import (
    TwoWayAttentionBlock,
    TwoWayTransformer,
)

# ---------------------------------------------------------------------------
# Fixture geometry.
#
# Widths are reduced so the whole gate stays cheap on a 12 GB GPU; the geometry
# that governs the code paths under test is the REAL SAM geometry:
#   * patch_size 16 (as vit_b/vit_l/vit_h all use)
#   * window_size 14 (as vit_b), against a 16x16 token grid, so window
#     partition/unpartition actually pads -- the non-trivial branch
#   * a non-empty global_attn_indexes, so at least one block is global
#   * use_rel_pos=True, so _get_rel_pos / _add_decomposed_rel_pos execute
# ---------------------------------------------------------------------------
IMG_SIZE = 256
PATCH_SIZE = 16
GRID_SIZE = IMG_SIZE // PATCH_SIZE  # 16
WINDOW_SIZE = 14
EMBED_DIM = 64
DEPTH = 4
NUM_HEADS = 4
OUT_CHANS = 32
GLOBAL_ATTN_INDEXES = (1, 3)

#: Weight count of the default fixture.
#:
#: Step 3 did NOT move it (``key_dim`` reshapes the 8 tensors of a
#: ``MultiHeadAttention`` without adding or removing any). Step 4 DOES: making
#: the dead ``iou_head_depth`` knob live adds one ``Dense`` (2 tensors) to each
#: of the 4 hypernetwork MLPs and to the IoU head:
#: ``192 + 4 * 2 + 2 = 202``.
FIXTURE_WEIGHT_COUNT_PRE_STEP4 = 192
BASELINE_FIXTURE_WEIGHT_COUNT = 202

#: Total scalar parameter count of the default fixture.
#:
#: This constant exists because the weight COUNT was blind to plan step 3: a
#: ``key_dim`` change re-SHAPES the 8 tensors of a ``MultiHeadAttention``
#: without adding or removing any, so 192 was identical before and after. The
#: parameter count is what actually observed that layout change.
#:
#: Step 3 derivation (Keras 3 ``MultiHeadAttention``, embed dim ``E``, internal
#: dim ``I = num_heads * key_dim``): query/key/value kernels ``E*I`` with bias
#: ``I`` each, output kernel ``I*E`` with bias ``E`` -> ``131*I + 32`` at
#: ``E = 32``. The fixture transformer is ``embedding_dim=32, num_heads=2,
#: depth=2``, i.e. 5 cross-attentions (2 per block + the final one). Step 3
#: moved each from ``I = 32`` (4224 params) to ``I = 16`` (2128), so
#: ``327062 - 5 * 2096 = 316582``.
#:
#: Step 4 derivation (``transformer_dim = 32``, ``iou_head_hidden_dim = 32``,
#: ``num_mask_tokens = 4``, ``iou_head_depth = 3``): each head gains exactly one
#: ``32 -> 32`` ``Dense``, i.e. ``32 * 32 + 32 = 1056`` params and 2 tensors.
#: There are 4 hypernetwork MLPs plus 1 IoU head, so
#: ``316582 + 5 * 1056 = 321862`` params and ``192 + 5 * 2 = 202`` weights.
FIXTURE_PARAM_COUNT_PRE_STEP3 = 327_062
FIXTURE_PARAM_COUNT_PRE_STEP4 = 316_582
BASELINE_FIXTURE_PARAM_COUNT = 321_862

#: Weight values are seeded to at least this magnitude (see
#: :func:`seed_nonzero_weights`).
MIN_ABS_WEIGHT = 0.02


# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------
def build_reduced_sam(
    use_rel_pos: bool = True,
    global_attn_indexes: Tuple[int, ...] = GLOBAL_ATTN_INDEXES,
    depth: int = DEPTH,
    **encoder_overrides: Any,
) -> SAM:
    """
    Build a reduced-width SAM at real SAM patch/window geometry.

    Args:
        use_rel_pos: Whether the ViT blocks use relative position embeddings.
            ``False`` is the control that must drive the ``_get_rel_pos`` call
            count to zero.
        global_attn_indexes: Block indices that use global (non-windowed)
            attention. Empty means every block is windowed.
        depth: Number of ViT blocks.
        **encoder_overrides: Forwarded verbatim to :class:`ImageEncoderViT`.

    Returns:
        An unbuilt :class:`SAM`. Call it once (see :func:`sam_inputs`) to build.

    Raises:
        Whatever the constructed sub-models raise; this helper adds no
        validation of its own.
    """
    encoder = ImageEncoderViT(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        depth=depth,
        num_heads=NUM_HEADS,
        out_chans=OUT_CHANS,
        use_rel_pos=use_rel_pos,
        window_size=WINDOW_SIZE,
        global_attn_indexes=global_attn_indexes,
        **encoder_overrides,
    )
    prompt_encoder = PromptEncoder(
        embed_dim=OUT_CHANS,
        image_embedding_size=(GRID_SIZE, GRID_SIZE),
        input_image_size=(IMG_SIZE, IMG_SIZE),
        mask_in_chans=8,
    )
    transformer = TwoWayTransformer(
        depth=2,
        embedding_dim=OUT_CHANS,
        num_heads=2,
        mlp_dim=64,
    )
    mask_decoder = MaskDecoder(
        transformer_dim=OUT_CHANS,
        transformer=transformer,
        iou_head_hidden_dim=32,
    )
    return SAM(
        image_encoder=encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
    )


def seed_nonzero_weights(
    model: keras.Model,
    seed: int = 1234,
    scale: float = 0.05,
    min_abs: float = MIN_ABS_WEIGHT,
) -> None:
    """
    Give every weight of a BUILT model a deterministic, provably non-zero value.

    Many SAM weights are zero-initialized (``pos_embed``, ``rel_pos_h/w``, every
    bias, every norm ``beta``, and the ``not_a_point_embed`` row after its own
    initializer). A probe run against those defaults can be structurally unable
    to observe the quantity it claims to measure -- the classic "guard passes
    both ways" trap. This offsets each weight by a seeded draw and then clamps
    the magnitude, so the initialization *scale* is preserved while no entry can
    be zero.

    Args:
        model: A built model. Weights are mutated in place.
        seed: Seed for :func:`numpy.random.default_rng`; identical seeds give
            identical weights for the same weight-traversal order.
        scale: Standard deviation of the additive offset.
        min_abs: Guaranteed minimum absolute value of every resulting entry.

    Returns:
        None. The model is mutated in place.

    Raises:
        ValueError: propagated from ``assign`` if the model is not built (its
            weight list is then empty and this is a silent no-op, which
            ``test_seeded_weights_are_all_nonzero`` catches via its count check).
    """
    rng = np.random.default_rng(seed)
    for weight in model.weights:
        current = keras.ops.convert_to_numpy(weight).astype("float64")
        offset = rng.normal(0.0, scale, size=current.shape)
        # np.sign(0.0) is 0.0, which would defeat the clamp; nudge first.
        offset = np.where(
            np.abs(offset) < min_abs, np.sign(offset + 1e-12) * min_abs, offset
        )
        new = current + offset
        new = np.where(np.abs(new) < min_abs, np.sign(offset) * min_abs, new)
        weight.assign(new.astype(keras.backend.standardize_dtype(weight.dtype)))


def sam_inputs(seed: int = 0) -> Dict[str, Any]:
    """
    Build a point-only prompt input dict for the reduced fixture.

    Point-only is deliberate: it is the ONLY path that appends a padding point
    (``call`` sets ``pad=(boxes is None)``), and therefore the only path that
    exercises the padding-point positional-encoding behaviour.

    Args:
        seed: Seed for the RGB image draw (values in [0, 255], as the model's
            ``preprocess`` expects).

    Returns:
        Dict with ``image``, ``points`` (coords, labels) and ``original_size``.
    """
    image = np.random.RandomState(seed).uniform(
        0.0, 255.0, size=(1, IMG_SIZE, IMG_SIZE, 3)
    ).astype("float32")
    return {
        "image": keras.ops.convert_to_tensor(image),
        "points": (
            keras.ops.convert_to_tensor([[[100.0, 120.0]]]),
            keras.ops.convert_to_tensor([[1]]),
        ),
        "original_size": keras.ops.convert_to_tensor((IMG_SIZE, IMG_SIZE)),
    }


def gradient_none_counts(
    model: keras.Model,
    inputs: Dict[str, Any],
    output_key: str,
) -> Tuple[int, int]:
    """
    Count how many trainable variables receive NO gradient from one output key.

    Args:
        model: A built model whose ``call`` returns a dict of outputs.
        inputs: The input dict passed to ``model``.
        output_key: Key of the output tensor to differentiate.

    Returns:
        ``(n_none, n_total)`` -- the number of ``None`` entries in the gradient
        list and the total number of trainable variables. ``n_none == n_total``
        means the output is completely gradient-dead. Callers should pin an
        exact pair, never a bare ``> 0``.

    Raises:
        KeyError: if ``output_key`` is not produced by ``model``.
    """
    with tf.GradientTape() as tape:
        outputs = model(inputs)
        if output_key not in outputs:
            raise KeyError(
                f"output_key '{output_key}' not in model outputs "
                f"{sorted(outputs.keys())}"
            )
        loss = tf.reduce_sum(tf.cast(outputs[output_key], "float32"))
    grads = tape.gradient(loss, model.trainable_variables)
    return sum(1 for g in grads if g is None), len(grads)


def roundtrip_low_res_logits(
    model: keras.Model,
    inputs: Dict[str, Any],
    save_path: str,
    perturb: Optional[Callable[[keras.Model], None]] = None,
) -> Dict[str, Any]:
    """
    Save a BUILT model, reload it, and compare ``low_res_logits`` VALUES.

    This is the round-trip proof the pre-existing suite lacks: ``test_save_and_load``
    saves an *unbuilt* model and asserts only ``isinstance``, and
    ``test_output_consistency_after_loading`` compares *binarized uint8* masks,
    which cannot see logit drift.

    Args:
        model: A BUILT model (call it once before passing it in; an unbuilt save
            stores no weights and the comparison would be meaningless).
        inputs: Input dict used for both the reference and the restored forward.
        save_path: Destination ``.keras`` path.
        perturb: Optional hook invoked on the RESTORED model before its forward
            pass. Used to prove the instrument RED.

    Returns:
        Dict with keys ``max_abs_diff`` (float), ``n_weights_before`` (int),
        ``n_weights_after`` (int), ``load_seconds`` (float).

    Raises:
        Whatever ``keras.models.load_model`` raises on a broken archive.
    """
    import time

    reference = keras.ops.convert_to_numpy(model(inputs)["low_res_logits"])
    n_before = len(model.weights)

    model.save(save_path)
    t0 = time.time()
    restored = keras.models.load_model(save_path)
    load_seconds = time.time() - t0
    n_after = len(restored.weights)

    if perturb is not None:
        perturb(restored)

    got = keras.ops.convert_to_numpy(restored(inputs)["low_res_logits"])
    return {
        "max_abs_diff": float(np.max(np.abs(reference - got))),
        "n_weights_before": n_before,
        "n_weights_after": n_after,
        "load_seconds": load_seconds,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def seeded_sam():
    """A built, seeded, reduced-width SAM plus its inputs (module-scoped)."""
    model = build_reduced_sam()
    inputs = sam_inputs()
    model(inputs)  # build
    seed_nonzero_weights(model)
    yield model, inputs
    del model
    keras.backend.clear_session()
    gc.collect()


# ---------------------------------------------------------------------------
# Instrument self-proofs
# ---------------------------------------------------------------------------
class TestInstrumentSelfProofs:
    """Each test here proves one instrument can report failure."""

    def test_rel_pos_path_is_reached_and_control_is_zero(self, monkeypatch):
        """
        The fixture must actually execute ``_get_rel_pos``; the control must not.

        A fixture that merely *claims* to exercise a path is exactly the vacuity
        this module exists to end, so the call count is measured with a spy and
        the ``use_rel_pos=False`` control must drive it to exactly 0.
        """
        calls = []
        original = WindowedAttentionWithRelPos._get_rel_pos

        def spy(self, q_size, k_size, rel_pos):
            calls.append((q_size, k_size))
            return original(self, q_size, k_size, rel_pos)

        monkeypatch.setattr(WindowedAttentionWithRelPos, "_get_rel_pos", spy)

        on_model = build_reduced_sam(use_rel_pos=True)
        on_model(sam_inputs())
        n_on = len(calls)

        calls.clear()
        off_model = build_reduced_sam(use_rel_pos=False)
        off_model(sam_inputs())
        n_off = len(calls)

        # 2 calls per block (height + width) x DEPTH blocks.
        assert n_on == 2 * DEPTH, (
            f"rel-pos fixture reached _get_rel_pos {n_on} times, expected "
            f"{2 * DEPTH}"
        )
        assert n_off == 0, (
            f"use_rel_pos=False control still reached _get_rel_pos {n_off} times"
        )

        del on_model, off_model
        keras.backend.clear_session()
        gc.collect()

    def test_seeded_weights_are_all_nonzero(self, seeded_sam):
        """
        Prove the seeding helper actually removes the zero-initialized weights.

        RED without the helper: 95 of 192 weights of this fixture are exactly
        all-zero at initialization, so this assertion fires on the raw model.
        """
        model, _ = seeded_sam
        assert len(model.weights) > 0, "fixture is not built; seeding was a no-op"
        zero_valued = [
            w.path
            for w in model.weights
            if float(np.min(np.abs(keras.ops.convert_to_numpy(w)))) == 0.0
        ]
        assert zero_valued == [], (
            f"{len(zero_valued)} weights still contain a zero entry after "
            f"seeding: {zero_valued[:5]}"
        )

    def test_roundtrip_instrument_is_exact_and_provably_red(self, seeded_sam):
        """
        Round-trip is value-exact at HEAD, AND the instrument can report failure.

        The RED half perturbs a single restored weight on the ``low_res_logits``
        path by 1e-4. (A true 1-ULP ``nextafter`` bump of ``pos_embed`` was also
        measured to move the logits by 1.3e-3 on this fixture, but 1e-4 is used
        here because it is robust across backends rather than relying on that
        amplification.)
        """
        model, inputs = seeded_sam

        with tempfile.TemporaryDirectory() as tmpdir:
            clean = roundtrip_low_res_logits(
                model, inputs, os.path.join(tmpdir, "clean.keras")
            )
            assert clean["max_abs_diff"] == 0.0, (
                f"round-trip already drifts at HEAD: {clean['max_abs_diff']}"
            )
            assert clean["n_weights_after"] == clean["n_weights_before"], (
                f"weight count changed across round-trip: "
                f"{clean['n_weights_before']} -> {clean['n_weights_after']}"
            )

            def perturb(restored: keras.Model) -> None:
                target = next(
                    w
                    for w in restored.trainable_variables
                    if "patch_embed/projection/kernel" in w.path
                )
                value = keras.ops.convert_to_numpy(target)
                target.assign(value + np.float32(1e-4))

            broken = roundtrip_low_res_logits(
                model, inputs, os.path.join(tmpdir, "broken.keras"), perturb=perturb
            )

        assert broken["max_abs_diff"] > 0.0, (
            "the round-trip instrument reported 0.0 even with a deliberately "
            "perturbed restored weight -- it cannot detect drift and is not "
            "coverage"
        )

    def test_weight_count_control(self, seeded_sam):
        """
        Pin the fixture's weight count. This is the control for plan steps 3/4.

        Command that produced the number:
            CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
                tests/test_models/test_sam/test_correctness.py -q -k weight_count
        """
        model, _ = seeded_sam
        assert len(model.weights) == BASELINE_FIXTURE_WEIGHT_COUNT, (
            f"fixture weight count moved: {len(model.weights)} != "
            f"{BASELINE_FIXTURE_WEIGHT_COUNT}. A layout change is only legitimate "
            f"if a plan step deliberately made it; update the constant WITH the "
            f"derivation, never silently"
        )
        assert len(model.trainable_variables) == BASELINE_FIXTURE_WEIGHT_COUNT - 1, (
            f"trainable count moved: {len(model.trainable_variables)} != "
            f"{BASELINE_FIXTURE_WEIGHT_COUNT - 1}"
        )

    def test_gradient_probe_distinguishes_output_keys(self, seeded_sam):
        """
        Prove the gradient probe is not a constant function.

        It must report a TOTALLY dead ``masks`` output (the uint8 cast at
        ``model.py``) and a live ``low_res_logits`` on the SAME model. A probe
        that returned the same pair for both would pass a bare ``> 0`` guard and
        prove nothing.
        """
        model, inputs = seeded_sam

        n_none_masks, n_total = gradient_none_counts(model, inputs, "masks")
        n_none_logits, n_total_logits = gradient_none_counts(
            model, inputs, "low_res_logits"
        )

        assert n_total == n_total_logits == BASELINE_FIXTURE_WEIGHT_COUNT - 1
        # F-2 at HEAD: the headline `masks` output is completely gradient-dead.
        assert n_none_masks == n_total, (
            f"expected masks to be fully gradient-dead at HEAD, got "
            f"{n_none_masks}/{n_total}"
        )
        # ...while low_res_logits reaches most of the model.
        assert n_none_logits < n_total, (
            "low_res_logits is gradient-dead too -- the probe is measuring "
            "nothing"
        )

    def test_gradient_probe_rejects_unknown_output_key(self, seeded_sam):
        """The probe raises rather than silently returning a vacuous (0, 0)."""
        model, inputs = seeded_sam
        with pytest.raises(KeyError, match="not_an_output"):
            gradient_none_counts(model, inputs, "not_an_output")


# ---------------------------------------------------------------------------
# F-3: the `_get_rel_pos` interpolation branch
# ---------------------------------------------------------------------------
#: Geometry that forces the interpolation branch. ``rel_pos`` is built at
#: ``2 * REL_POS_TABLE_SIZE - 1 = 15`` rows, while a forward pass at
#: ``REL_POS_QUERY_SIZE`` needs ``2 * 6 - 1 = 11`` -- so the branch runs.
REL_POS_DIM = 16
REL_POS_HEADS = 2
REL_POS_HEAD_DIM = REL_POS_DIM // REL_POS_HEADS  # 8
REL_POS_TABLE_SIZE = 8
REL_POS_QUERY_SIZE = 6
REL_POS_STORED_LEN = 2 * REL_POS_TABLE_SIZE - 1  # 15
REL_POS_TARGET_LEN = 2 * REL_POS_QUERY_SIZE - 1  # 11


def build_mismatched_rel_pos_attention() -> WindowedAttentionWithRelPos:
    """
    Build an attention layer whose ``input_size`` != its forward query size.

    This mismatch is the ONLY way to reach ``_get_rel_pos``'s interpolation
    branch: in-package every construction site passes ``input_size`` equal to
    the query grid, which is why the branch had zero prior execution.

    Returns:
        A BUILT :class:`WindowedAttentionWithRelPos` with seeded non-zero
        weights (``rel_pos_h`` / ``rel_pos_w`` are zero-initialized, which would
        make the whole path numerically inert -- carried surprise #1).
    """
    layer = WindowedAttentionWithRelPos(
        dim=REL_POS_DIM,
        num_heads=REL_POS_HEADS,
        use_rel_pos=True,
        input_size=(REL_POS_TABLE_SIZE, REL_POS_TABLE_SIZE),
    )
    layer.build((1, REL_POS_QUERY_SIZE, REL_POS_QUERY_SIZE, REL_POS_DIM))
    seed_nonzero_weights(layer)
    return layer


class TestRelPosInterpolationBranch:
    """F-3 -- `_get_rel_pos` must interpolate the DISTANCE axis and not raise."""

    def test_rel_pos_interpolation_branch_runs_and_is_finite(self):
        """
        A forward pass at ``input_size != q_size`` must complete.

        RED before the fix (measured, commit 90d352f9):
            ``ValueError: Cannot squeeze axis=0, because the dimension is not 1.``
            raised from ``image_encoder.py:273`` -- the 3-D ``(1, C, L)`` tensor
            is read by ``ops.image.resize`` as unbatched ``(h=1, w=C, c=L)``.
        """
        layer = build_mismatched_rel_pos_attention()
        assert tuple(layer.rel_pos_h.shape) == (
            REL_POS_STORED_LEN,
            REL_POS_HEAD_DIM,
        ), "fixture no longer creates a table that needs interpolation"
        assert REL_POS_STORED_LEN != REL_POS_TARGET_LEN, (
            "geometry no longer reaches the interpolation branch"
        )

        x = keras.ops.convert_to_tensor(
            np.random.RandomState(0)
            .normal(0.0, 1.0, (1, REL_POS_QUERY_SIZE, REL_POS_QUERY_SIZE, REL_POS_DIM))
            .astype("float32")
        )
        y = keras.ops.convert_to_numpy(layer(x))

        assert y.shape == (1, REL_POS_QUERY_SIZE, REL_POS_QUERY_SIZE, REL_POS_DIM), (
            f"interpolation branch produced {y.shape}, expected the input shape"
        )
        assert np.all(np.isfinite(y)), "interpolated rel-pos produced non-finite output"

        del layer
        keras.backend.clear_session()
        gc.collect()

    def test_rel_pos_interpolation_is_observable(self):
        """
        The interpolated bias must actually change the attention output.

        Without this, the branch could "work" by silently contributing zeros --
        which is precisely what an UNSEEDED fixture would report, since
        ``rel_pos_h``/``rel_pos_w`` are zero-initialized.
        """
        layer = build_mismatched_rel_pos_attention()
        x = keras.ops.convert_to_tensor(
            np.random.RandomState(1)
            .normal(0.0, 1.0, (1, REL_POS_QUERY_SIZE, REL_POS_QUERY_SIZE, REL_POS_DIM))
            .astype("float32")
        )
        with_bias = keras.ops.convert_to_numpy(layer(x))

        layer.rel_pos_h.assign(np.zeros(layer.rel_pos_h.shape, dtype="float32"))
        layer.rel_pos_w.assign(np.zeros(layer.rel_pos_w.shape, dtype="float32"))
        without_bias = keras.ops.convert_to_numpy(layer(x))

        assert float(np.max(np.abs(with_bias - without_bias))) > 0.0, (
            "zeroing rel_pos_h/rel_pos_w did not change the output -- the "
            "interpolated relative-position bias is inert and this guard would "
            "pass with the feature broken"
        )

        del layer
        keras.backend.clear_session()
        gc.collect()

    def test_rel_pos_interpolation_axis_mapping(self):
        """
        Prove WHICH axis is interpolated: distance resampled, channels intact.

        The correct mapping is an unverified hypothesis until measured, so both
        halves are asserted directly on ``_get_rel_pos``:

        * a ramp along the DISTANCE axis stays strictly monotone after resizing
          and stays identical across channels;
        * a table that is CONSTANT along distance but distinct per channel comes
          back bit-identical -- no channel is blended into another.

        The pre-fix implementation resized the channel axis instead, so this
        assertion is unreachable there (it raised first).
        """
        layer = build_mismatched_rel_pos_attention()
        n_c = REL_POS_HEAD_DIM

        # (a) ramp along distance, identical across channels
        ramp = np.tile(
            np.arange(REL_POS_STORED_LEN, dtype="float32")[:, None], (1, n_c)
        )
        layer.rel_pos_h.assign(ramp)
        out = keras.ops.convert_to_numpy(
            layer._get_rel_pos(
                REL_POS_QUERY_SIZE, REL_POS_QUERY_SIZE, layer.rel_pos_h
            )
        )
        assert out.shape == (REL_POS_QUERY_SIZE, REL_POS_QUERY_SIZE, n_c), (
            f"_get_rel_pos returned {out.shape}, expected "
            f"(q, k, head_dim) = ({REL_POS_QUERY_SIZE}, {REL_POS_QUERY_SIZE}, {n_c})"
        )
        # Re-derive the resampled distance table from the gathered result: row
        # i, col j gathers resized row (i - j + k_size - 1), so the anti-diagonal
        # walk out[i, 0] for increasing i walks the table in increasing distance.
        distance_walk = out[:, 0, 0]
        assert np.all(np.diff(distance_walk) > 0), (
            f"distance axis is not monotone after interpolation: {distance_walk}"
        )
        channel_spread = float(np.max(out.max(axis=-1) - out.min(axis=-1)))
        assert channel_spread == 0.0, (
            f"a channel-identical ramp came back channel-dependent (spread "
            f"{channel_spread}) -- the resize touched the wrong axis"
        )

        # (b) constant along distance, distinct per channel -> untouched
        per_channel = np.arange(1.0, n_c + 1.0, dtype="float32")
        layer.rel_pos_h.assign(np.tile(per_channel[None, :], (REL_POS_STORED_LEN, 1)))
        out_const = keras.ops.convert_to_numpy(
            layer._get_rel_pos(
                REL_POS_QUERY_SIZE, REL_POS_QUERY_SIZE, layer.rel_pos_h
            )
        )
        assert float(np.max(np.abs(out_const - per_channel))) == 0.0, (
            "a per-channel constant table was altered by the distance-axis "
            "interpolation -- channels are being blended"
        )

        del layer
        keras.backend.clear_session()
        gc.collect()

    def test_ghost_gather_decision_comment_is_gone(self):
        """
        SC-4 -- the stale `keras.ops has no gather` anchor must not be reinstated.

        It documented a resolved framework triviality as if it were a live
        blocker on the very function this step repairs.
        """
        import dl_techniques.models.SAM.SAM1.image_encoder as image_encoder_module

        with open(image_encoder_module.__file__, "r", encoding="utf-8") as handle:
            source = handle.read()
        assert "has no `gather`" not in source, (
            "the ghost `# DECISION plan_2026-06-15_e6a0391c/D-004` gather "
            "comment is back in image_encoder.py"
        )


# ---------------------------------------------------------------------------
# F-4: attention_downsample_rate on the three cross-attentions
# ---------------------------------------------------------------------------
def _attention_internal_dims(transformer: TwoWayTransformer) -> Dict[str, int]:
    """
    Report the ACTUAL internal dim (``num_heads * key_dim``) of every attention.

    Read off the constructed ``MultiHeadAttention`` objects rather than off the
    config that was passed in: a constructor knob that is stored, serialized and
    never wired through is the exact defect class this file exists to catch, and
    ``create_ffn_layer`` in this same package silently drops unknown keys and
    builds at the default width. Never trust that a kwarg landed.

    Args:
        transformer: A :class:`TwoWayTransformer` (built or unbuilt; ``key_dim``
            and ``num_heads`` are set in ``__init__``).

    Returns:
        Mapping from a dotted attention path (``"block_0.self_attn"``,
        ``"final_attn_token_to_image"``) to ``num_heads * key_dim``.
    """
    dims: Dict[str, int] = {}
    for block in transformer.layers_list:
        for name in (
            "self_attn",
            "cross_attn_token_to_image",
            "cross_attn_image_to_token",
        ):
            attn = getattr(block, name)
            dims[f"{block.name}.{name}"] = attn.num_heads * attn.key_dim
    final = transformer.final_attn_token_to_image
    dims["final_attn_token_to_image"] = final.num_heads * final.key_dim
    return dims


class TestAttentionDownsampleRate:
    """F-4 -- reference SAM runs the three cross-attentions at ``E // 2``."""

    def test_cross_attention_internal_dim(self):
        """
        The three cross-attentions are ``E // rate``; ``self_attn`` is ``E``.

        RED before the fix: all four report ``E`` (probe P11), so BOTH
        assertions below fire -- the ``cross_dims == {16}`` one first.
        """
        transformer = TwoWayTransformer(
            depth=2, embedding_dim=OUT_CHANS, num_heads=2, mlp_dim=64
        )
        assert transformer.attention_downsample_rate == 2, (
            "attention_downsample_rate must default to 2 (reference SAM)"
        )

        dims = _attention_internal_dims(transformer)
        cross_dims = {v for k, v in dims.items() if "self_attn" not in k}
        self_dims = {v for k, v in dims.items() if k.endswith(".self_attn")}

        assert cross_dims == {OUT_CHANS // 2}, (
            f"cross-attentions must run at embedding_dim // 2 = "
            f"{OUT_CHANS // 2}, measured {sorted(cross_dims)}; full dims: {dims}"
        )
        assert self_dims == {OUT_CHANS}, (
            f"self_attn must stay at full width {OUT_CHANS}, measured "
            f"{sorted(self_dims)}; full dims: {dims}"
        )
        assert len(dims) == 7, f"expected 6 block attentions + 1 final, got {dims}"

        del transformer
        keras.backend.clear_session()
        gc.collect()

    def test_rate_one_restores_full_width(self):
        """
        ``rate=1`` must put all seven attentions back at full width.

        This is the discriminating control: a hardcoded ``// 2`` would pass
        :meth:`test_cross_attention_internal_dim` and fail here.
        """
        transformer = TwoWayTransformer(
            depth=1,
            embedding_dim=OUT_CHANS,
            num_heads=2,
            mlp_dim=64,
            attention_downsample_rate=1,
        )
        dims = _attention_internal_dims(transformer)
        assert set(dims.values()) == {OUT_CHANS}, (
            f"attention_downsample_rate=1 must give full width everywhere, "
            f"measured {dims}"
        )
        del transformer
        keras.backend.clear_session()
        gc.collect()

    @pytest.mark.parametrize("rate", [1, 2, 4])
    def test_rate_round_trips_on_both_classes(self, rate: int):
        """
        The knob survives ``get_config``/``from_config`` on BOTH classes.

        And -- the half that matters -- the RESTORED object's ACTUAL attention
        widths match. A config key that round-trips while the rebuilt layer
        ignores it is a dead knob, which is precisely what ``iou_head_depth``
        already is elsewhere in this package.
        """
        block = TwoWayAttentionBlock(
            embedding_dim=OUT_CHANS,
            num_heads=2,
            mlp_dim=64,
            attention_downsample_rate=rate,
        )
        block_cfg = block.get_config()
        assert block_cfg["attention_downsample_rate"] == rate, (
            f"TwoWayAttentionBlock.get_config() dropped "
            f"attention_downsample_rate: {sorted(block_cfg)}"
        )
        block_back = TwoWayAttentionBlock.from_config(block_cfg)
        expected = OUT_CHANS // rate
        for name in ("cross_attn_token_to_image", "cross_attn_image_to_token"):
            attn = getattr(block_back, name)
            assert attn.num_heads * attn.key_dim == expected, (
                f"restored block's {name} width is "
                f"{attn.num_heads * attn.key_dim}, expected {expected} -- the "
                f"round-tripped knob is dead"
            )
        assert (
            block_back.self_attn.num_heads * block_back.self_attn.key_dim
            == OUT_CHANS
        ), "restored block's self_attn must stay at full width"

        transformer = TwoWayTransformer(
            depth=2,
            embedding_dim=OUT_CHANS,
            num_heads=2,
            mlp_dim=64,
            attention_downsample_rate=rate,
        )
        tr_cfg = transformer.get_config()
        assert tr_cfg["attention_downsample_rate"] == rate, (
            f"TwoWayTransformer.get_config() dropped attention_downsample_rate:"
            f" {sorted(tr_cfg)}"
        )
        tr_back = TwoWayTransformer.from_config(tr_cfg)
        dims = _attention_internal_dims(tr_back)
        cross_dims = {v for k, v in dims.items() if "self_attn" not in k}
        assert cross_dims == {expected}, (
            f"restored transformer cross-attention widths {sorted(cross_dims)} "
            f"!= {expected}; full dims: {dims}"
        )

        del block, block_back, transformer, tr_back
        keras.backend.clear_session()
        gc.collect()

    def test_indivisible_rate_raises_not_floors(self):
        """
        ``E % (heads * rate) != 0`` raises on BOTH classes, naming the product.

        ``embedding_dim=12, num_heads=4`` is chosen so the PRE-EXISTING
        ``embedding_dim % num_heads`` check passes (12 % 4 == 0) and only the
        new one can fire; without it ``12 // (4 * 2) == 1`` would build a
        4-wide cross-attention and every shape assertion in the suite would
        still pass.
        """
        for cls, extra in (
            (TwoWayAttentionBlock, {}),
            (TwoWayTransformer, {"depth": 1}),
        ):
            with pytest.raises(ValueError, match=r"num_heads \* attention_downsample_rate"):
                cls(embedding_dim=12, num_heads=4, mlp_dim=64, **extra)
            with pytest.raises(ValueError, match="attention_downsample_rate must be positive"):
                cls(
                    embedding_dim=32,
                    num_heads=2,
                    mlp_dim=64,
                    attention_downsample_rate=0,
                    **extra,
                )

    def test_param_count_matches_hand_derivation(self, seeded_sam):
        """
        I-3 -- the measured parameter delta equals the hand-derived one.

        The weight COUNT cannot see this step (``key_dim`` reshapes the 8 MHA
        tensors without adding or removing any), so the scalar parameter count
        is the instrument. The pre-mortem's STOP-IF is exactly a mismatch here.

        Command that produced the numbers:
            CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
                tests/test_models/test_sam/test_correctness.py -q -k param_count
        """
        model, _ = seeded_sam
        measured = int(sum(np.prod(w.shape) for w in model.weights))
        assert measured == BASELINE_FIXTURE_PARAM_COUNT, (
            f"fixture parameter count is {measured}, expected "
            f"{BASELINE_FIXTURE_PARAM_COUNT} "
            f"(pre-step-3 control {FIXTURE_PARAM_COUNT_PRE_STEP3}, step-3 "
            f"hand-derived delta -5 * 2096 -> {FIXTURE_PARAM_COUNT_PRE_STEP4}; "
            f"pre-step-4 control {FIXTURE_PARAM_COUNT_PRE_STEP4}, step-4 "
            f"hand-derived delta +5 * 1056 = "
            f"{BASELINE_FIXTURE_PARAM_COUNT - FIXTURE_PARAM_COUNT_PRE_STEP4})"
        )
        assert len(model.weights) == BASELINE_FIXTURE_WEIGHT_COUNT, (
            f"weight count is {len(model.weights)}, expected "
            f"{BASELINE_FIXTURE_WEIGHT_COUNT}. Step 3 must NOT move it (a "
            f"key_dim change reshapes tensors without adding or removing "
            f"them); step 4 moves it by exactly +10 "
            f"({FIXTURE_WEIGHT_COUNT_PRE_STEP4} -> "
            f"{BASELINE_FIXTURE_WEIGHT_COUNT})"
        )

    def test_roundtrip_still_value_exact_after_layout_change(self, seeded_sam):
        """
        I-2 re-proof against the post-step-3 layout, on a BUILT model.

        Deliberately does not reuse the step-1 result: a layout change that
        restores some weights and re-initializes the rest returns a plausible
        model with drifted logits, and only a VALUE comparison can see it.
        """
        model, inputs = seeded_sam
        with tempfile.TemporaryDirectory() as tmpdir:
            result = roundtrip_low_res_logits(
                model, inputs, os.path.join(tmpdir, "downsampled.keras")
            )
        assert result["max_abs_diff"] == 0.0, (
            f"the .keras round-trip is no longer value-exact after the "
            f"attention_downsample_rate layout change: max abs diff on "
            f"low_res_logits = {result['max_abs_diff']}"
        )
        assert result["n_weights_after"] == result["n_weights_before"] == (
            BASELINE_FIXTURE_WEIGHT_COUNT
        ), (
            f"weight count moved across the round-trip: "
            f"{result['n_weights_before']} -> {result['n_weights_after']}"
        )


# ---------------------------------------------------------------------------
# F-5 + F-6: head depth driven by `iou_head_depth`, and the ReLU default
# ---------------------------------------------------------------------------
def _dense_names(sequential: keras.Sequential) -> list:
    """
    Report the ACTUAL ``Dense`` sub-layer names of a head, in build order.

    Read off the constructed object rather than off the config: ``iou_head_depth``
    is the package's own textbook example of a knob that is stored, serialized
    and never wired through, so a config-level assertion would have passed for
    the entire lifetime of the defect.

    Args:
        sequential: A ``keras.Sequential`` head (a hypernetwork MLP or the IoU
            prediction head).

    Returns:
        The ordered list of layer names.
    """
    return [layer.name for layer in sequential.layers]


def _build_decoder(iou_head_depth: int = 3, **overrides: Any) -> MaskDecoder:
    """
    Build a standalone :class:`MaskDecoder` at the fixture's decoder geometry.

    Args:
        iou_head_depth: Number of ``Dense`` layers per head.
        **overrides: Forwarded verbatim to :class:`MaskDecoder`.

    Returns:
        A BUILT decoder (its weights exist, so weight counts are meaningful).
    """
    decoder = MaskDecoder(
        transformer_dim=OUT_CHANS,
        transformer=TwoWayTransformer(
            depth=2, embedding_dim=OUT_CHANS, num_heads=2, mlp_dim=64
        ),
        iou_head_depth=iou_head_depth,
        iou_head_hidden_dim=32,
        **overrides,
    )
    decoder.build(None)
    return decoder


class TestHeadDepthAndActivationDefault:
    """F-5 + F-6 -- the heads honour ``iou_head_depth`` and default to ReLU."""

    def test_head_sublayer_names_honor_default_depth(self):
        """
        At the default ``iou_head_depth=3`` both heads have THREE ``Dense``s.

        RED before the fix (measured, commit 71ddd2d3): probe P12's lists,
        ``['hyper_dense1_0', 'hyper_dense2_0']`` and
        ``['iou_dense1', 'iou_dense2']`` -- two layers regardless of the knob.
        """
        decoder = _build_decoder(iou_head_depth=3)

        hyper_names = _dense_names(decoder.output_hypernetworks_mlps[0])
        assert hyper_names == [
            "hyper_dense1_0",
            "hyper_dense2_0",
            "hyper_dense3_0",
        ], f"hypernetwork MLP 0 has layers {hyper_names}, expected 3 at depth 3"

        iou_names = _dense_names(decoder.iou_prediction_head)
        assert iou_names == ["iou_dense1", "iou_dense2", "iou_dense3"], (
            f"IoU head has layers {iou_names}, expected 3 at depth 3"
        )

        del decoder
        keras.backend.clear_session()
        gc.collect()

    @pytest.mark.parametrize("depth", [1, 2, 3, 4])
    def test_head_layer_count_tracks_depth(self, depth: int):
        """
        The number of ``Dense`` layers in BOTH heads equals ``iou_head_depth``.

        Parametrised over depths on both sides of the default so a hardcoded
        ``3`` (the shape of the reference implementation, and the most likely
        wrong fix) cannot pass.
        """
        decoder = _build_decoder(iou_head_depth=depth)
        for i, mlp in enumerate(decoder.output_hypernetworks_mlps):
            names = _dense_names(mlp)
            assert len(names) == depth, (
                f"hypernetwork MLP {i} has {len(names)} Dense layers "
                f"({names}) at iou_head_depth={depth}"
            )
        iou_names = _dense_names(decoder.iou_prediction_head)
        assert len(iou_names) == depth, (
            f"IoU head has {len(iou_names)} Dense layers ({iou_names}) at "
            f"iou_head_depth={depth}"
        )
        del decoder
        keras.backend.clear_session()
        gc.collect()

    def test_head_depth_knob_is_live_not_merely_present(self):
        """
        THE discriminating guard: depth 2 and depth 3 must differ in WEIGHTS.

        A test that only checks ``depth=3`` cannot tell a live knob from a dead
        one -- the pre-fix code would satisfy it the moment the hardcoded layout
        happened to match. Two constructions of the SAME class with only the knob
        changed must produce measurably different objects, in both the weight
        COUNT and the scalar parameter count.

        Hand-derived, at ``transformer_dim = 32`` / ``iou_head_hidden_dim = 32``:
        one extra ``32 -> 32`` ``Dense`` per head = ``32 * 32 + 32 = 1056``
        params and 2 tensors, over 4 hypernetwork MLPs + 1 IoU head, so
        ``+10`` weights and ``+5280`` params.
        """
        shallow = _build_decoder(iou_head_depth=2)
        deep = _build_decoder(iou_head_depth=3)

        n_shallow, n_deep = len(shallow.weights), len(deep.weights)
        assert n_deep != n_shallow, (
            f"iou_head_depth=2 and iou_head_depth=3 built the SAME number of "
            f"weights ({n_shallow}) -- the knob is stored but dead, which is "
            f"exactly finding F-5"
        )
        assert n_deep - n_shallow == 10, (
            f"weight delta between depth 2 and 3 is {n_deep - n_shallow}, "
            f"hand-derived +10 (2 tensors x (4 hypernetwork MLPs + 1 IoU head))"
        )

        p_shallow = int(sum(np.prod(w.shape) for w in shallow.weights))
        p_deep = int(sum(np.prod(w.shape) for w in deep.weights))
        assert p_deep - p_shallow == 5 * 1056, (
            f"parameter delta between depth 2 and 3 is {p_deep - p_shallow}, "
            f"hand-derived +{5 * 1056}"
        )

        del shallow, deep
        keras.backend.clear_session()
        gc.collect()

    def test_hidden_widths_match_reference_mlp(self):
        """
        Widths, not just counts: hidden layers are wide, the output is narrow.

        Reference SAM's ``MLP(input, hidden, output, num_layers)`` makes every
        layer but the last ``hidden``-wide. Adding depth by stacking layers of
        the OUTPUT width would satisfy every count assertion above while
        bottlenecking the head to ``transformer_dim // 8``.
        """
        decoder = _build_decoder(iou_head_depth=3)

        hyper_units = [
            layer.units for layer in decoder.output_hypernetworks_mlps[0].layers
        ]
        assert hyper_units == [OUT_CHANS, OUT_CHANS, OUT_CHANS // 8], (
            f"hypernetwork MLP widths are {hyper_units}, expected two hidden "
            f"layers at transformer_dim={OUT_CHANS} then "
            f"{OUT_CHANS // 8} out"
        )

        iou_units = [layer.units for layer in decoder.iou_prediction_head.layers]
        assert iou_units == [32, 32, decoder.num_mask_tokens], (
            f"IoU head widths are {iou_units}, expected two hidden layers at "
            f"iou_head_hidden_dim=32 then {decoder.num_mask_tokens} out"
        )

        del decoder
        keras.backend.clear_session()
        gc.collect()

    def test_hidden_layers_are_activated_and_output_is_linear(self):
        """
        Every layer but the last carries the activation; the last is linear.

        Reference SAM applies ``relu`` to all but the final ``Linear``. A depth
        fix that activated the output too would emit non-negative mask logits
        and every shape assertion in this suite would still pass.
        """
        decoder = _build_decoder(iou_head_depth=3)
        for head_name, head in [
            ("hypernetwork_mlp_0", decoder.output_hypernetworks_mlps[0]),
            ("iou_prediction_head", decoder.iou_prediction_head),
        ]:
            names = [
                keras.activations.serialize(layer.activation)
                for layer in head.layers
            ]
            assert names[:-1] == ["relu"] * (len(names) - 1), (
                f"{head_name} hidden activations are {names[:-1]}, expected "
                f"all 'relu'"
            )
            assert names[-1] == "linear", (
                f"{head_name} output layer is activated ({names[-1]}); the "
                f"final layer must be linear"
            )
        del decoder
        keras.backend.clear_session()
        gc.collect()

    def test_activation_split_upscaler_gelu_heads_relu(self):
        """
        F-R1 -- the two halves of the decoder must DIFFER, as in reference SAM.

        Reference ``MaskDecoder(activation=nn.GELU)`` passes ``activation`` to
        ``output_upscaling`` and to nothing else; the hypernetwork and IoU heads
        hardcode ``F.relu`` inside ``MLP``. Routing ONE knob to both halves
        cannot be right for both: step 4 flipped that single knob to ``'relu'``,
        fixing the heads and silently breaking the upscaler.

        The predecessor of this test asserted ``decoder.activation == 'relu'``
        plus "the two halves agree", i.e. it CERTIFIED the new deviation. What
        is asserted here instead is read off the CONSTRUCTED sub-layers, so a
        config string that never reaches a layer cannot satisfy it.
        """
        decoder = MaskDecoder(
            transformer_dim=OUT_CHANS,
            transformer=TwoWayTransformer(
                depth=1, embedding_dim=OUT_CHANS, num_heads=2, mlp_dim=64
            ),
        )
        upsample_acts = [
            keras.activations.serialize(layer.activation)
            for layer in decoder.output_upscaling.layers
            if isinstance(layer, keras.layers.Activation)
        ]
        assert upsample_acts == ["gelu", "gelu"], (
            f"output_upscaling activations are {upsample_acts}; reference SAM "
            f"applies its GELU default there and only there"
        )

        for head_name, head in [
            ("hypernetwork_mlp_0", decoder.output_hypernetworks_mlps[0]),
            ("iou_prediction_head", decoder.iou_prediction_head),
        ]:
            names = [
                keras.activations.serialize(layer.activation)
                for layer in head.layers
            ]
            assert names[:-1] == ["relu"] * (len(names) - 1), (
                f"{head_name} hidden activations are {names[:-1]}; reference "
                f"SAM hardcodes F.relu inside MLP"
            )
            assert names[-1] == "linear", (
                f"{head_name} output layer is activated ({names[-1]})"
            )

        assert decoder.activation == "gelu", (
            f"MaskDecoder.activation defaults to {decoder.activation!r}; it is "
            f"the UPSCALER knob and reference SAM defaults it to GELU"
        )
        assert decoder.mlp_activation == "relu", (
            f"MaskDecoder.mlp_activation defaults to "
            f"{decoder.mlp_activation!r}; it is the MLP-head knob and "
            f"reference SAM hardcodes ReLU"
        )
        assert decoder.activation != decoder.mlp_activation, (
            "the two halves were collapsed onto one value again; reference SAM "
            "makes them differ by construction (GELU upscaler, ReLU heads)"
        )
        del decoder
        keras.backend.clear_session()
        gc.collect()

    @pytest.mark.parametrize("depth", [2, 3])
    @pytest.mark.parametrize(
        "activation,mlp_activation", [("gelu", "relu"), ("relu", "gelu")]
    )
    def test_depth_and_activation_round_trip(
        self, depth: int, activation: str, mlp_activation: str
    ):
        """
        All three knobs survive ``get_config``/``from_config`` -- and stay LIVE.

        The restored object's ACTUAL ``Dense`` count, ACTUAL head hidden
        activation and ACTUAL upscaler activation are asserted, not the config
        keys: a key that round-trips while the rebuilt layer ignores it is
        precisely the dead knob this work removes. The two parametrized pairs
        are deliberately CROSSED, so a build that routed one knob to both halves
        would fail on one of them.
        """
        decoder = _build_decoder(
            iou_head_depth=depth,
            activation=activation,
            mlp_activation=mlp_activation,
        )
        config = decoder.get_config()
        assert config["iou_head_depth"] == depth, (
            f"get_config() dropped iou_head_depth: {sorted(config)}"
        )
        assert config["activation"] == activation, (
            f"get_config() dropped activation: {sorted(config)}"
        )
        assert config["mlp_activation"] == mlp_activation, (
            f"get_config() dropped mlp_activation: {sorted(config)}"
        )

        restored = MaskDecoder.from_config(config)
        restored.build(None)
        assert len(_dense_names(restored.iou_prediction_head)) == depth, (
            f"restored IoU head has "
            f"{len(_dense_names(restored.iou_prediction_head))} Dense layers, "
            f"expected {depth} -- the round-tripped knob is dead"
        )
        assert len(_dense_names(restored.output_hypernetworks_mlps[0])) == depth, (
            "restored hypernetwork MLP ignored the round-tripped iou_head_depth"
        )
        restored_act = keras.activations.serialize(
            restored.iou_prediction_head.layers[0].activation
        )
        assert restored_act == mlp_activation, (
            f"restored IoU head hidden activation is {restored_act!r}, "
            f"expected {mlp_activation!r}"
        )
        restored_upsample = [
            keras.activations.serialize(layer.activation)
            for layer in restored.output_upscaling.layers
            if isinstance(layer, keras.layers.Activation)
        ]
        assert restored_upsample == [activation, activation], (
            f"restored upscaler activations are {restored_upsample}, expected "
            f"{[activation, activation]}"
        )
        assert len(restored.weights) == len(decoder.weights), (
            f"restored decoder has {len(restored.weights)} weights vs "
            f"{len(decoder.weights)} -- the layouts diverge"
        )

        del decoder, restored
        keras.backend.clear_session()
        gc.collect()

    def test_nonpositive_depth_raises(self):
        """
        ``iou_head_depth <= 0`` raises at construction, naming the knob.

        Without the raise, ``depth=0`` builds a head with zero ``Dense`` layers
        -- a ``Sequential`` that passes the ``transformer_dim``-wide token
        straight through and only fails much later as a shape mismatch.
        """
        for bad in (0, -1):
            with pytest.raises(ValueError, match="iou_head_depth must be positive"):
                MaskDecoder(
                    transformer_dim=OUT_CHANS,
                    transformer=TwoWayTransformer(
                        depth=1, embedding_dim=OUT_CHANS, num_heads=2, mlp_dim=64
                    ),
                    iou_head_depth=bad,
                )

    def test_roundtrip_still_value_exact_after_head_depth_change(self, seeded_sam):
        """
        I-2 re-proof against the post-step-4 layout, on a BUILT model.

        Step 4 moves the weight COUNT (unlike step 3), which is exactly the
        situation where a ``.keras`` restore can silently re-initialize the new
        tensors and return a plausible model with drifted logits.
        """
        model, inputs = seeded_sam
        with tempfile.TemporaryDirectory() as tmpdir:
            result = roundtrip_low_res_logits(
                model, inputs, os.path.join(tmpdir, "head_depth.keras")
            )
        assert result["max_abs_diff"] == 0.0, (
            f"the .keras round-trip is no longer value-exact after the "
            f"iou_head_depth layout change: max abs diff on low_res_logits = "
            f"{result['max_abs_diff']}"
        )
        assert result["n_weights_after"] == result["n_weights_before"] == (
            BASELINE_FIXTURE_WEIGHT_COUNT
        ), (
            f"weight count moved across the round-trip: "
            f"{result['n_weights_before']} -> {result['n_weights_after']}"
        )


# ---------------------------------------------------------------------------
# Plan step 5 (F-2): the `masks` output contract and its gradient behaviour.
#
# `SAM.call` casts `masks` to uint8. An integer cast has no gradient, so
# differentiating `outputs['masks']` returns None for EVERY trainable variable:
# a trainer that supervises the headline output trains nothing and says nothing.
# The repair is a serialized `binarize_masks` flag whose DEFAULT is byte-
# identical to that behaviour (reference SAM thresholds too, and 68 tests assert
# binary masks), with `False` returning the differentiable float logits.
#
# These counts are measured on THIS fixture (202 weights / 201 trainable) and
# are pinned exactly, never as a bare `> 0` -- a `> 0` guard cannot tell a live
# flag from a dead one.
#
# The 18 variables that never receive a gradient from a mask output are the
# unused prompt-encoder weights (box / mask-prompt embeddings, which a
# point-only prompt does not touch) plus the IoU head, which sits on a parallel
# branch. `iou_predictions` symmetrically leaves 42 out.
# ---------------------------------------------------------------------------
#: (n_none, n_total) for `masks` at `binarize_masks=True` -- fully dead.
GRAD_MASKS_BINARIZED = (201, 201)
#: (n_none, n_total) for `masks` at `binarize_masks=False` -- as live as logits.
GRAD_MASKS_FLOAT = (18, 201)
#: (n_none, n_total) for `low_res_logits` -- identical at BOTH flag settings.
GRAD_LOW_RES_LOGITS = (18, 201)


def _build_seeded_sam(binarize_masks: bool) -> Tuple[SAM, Dict[str, Any]]:
    """
    Build, run and seed a reduced SAM at an explicit `binarize_masks` setting.

    Args:
        binarize_masks: Value forwarded to the :class:`SAM` constructor.

    Returns:
        ``(model, inputs)`` -- built and seeded, ready for a probe.
    """
    encoder_model = build_reduced_sam()
    model = SAM(
        image_encoder=encoder_model.image_encoder,
        prompt_encoder=encoder_model.prompt_encoder,
        mask_decoder=encoder_model.mask_decoder,
        binarize_masks=binarize_masks,
    )
    inputs = sam_inputs()
    model(inputs)
    seed_nonzero_weights(model)
    return model, inputs


class TestOutputContract:
    """F-2: `masks` is gradient-dead by default; `low_res_logits` is the target."""

    def test_masks_are_gradient_dead_at_the_default(self, seeded_sam):
        """
        The DEFAULT contract, pinned as an exact pair.

        This is not a regression guard for the fix -- it is the documented
        property the fix deliberately PRESERVES (assumption A-6). It exists so
        that a future "cleanup" flipping the default is caught here rather than
        in a trainer that silently learns nothing.
        """
        model, inputs = seeded_sam
        assert model.binarize_masks is True, (
            "the module fixture is expected to use the shipped default"
        )
        assert gradient_none_counts(model, inputs, "masks") == (
            GRAD_MASKS_BINARIZED
        ), (
            "the uint8 default no longer kills every gradient; the documented "
            "training contract in SAM's docstring and README is now wrong"
        )
        assert gradient_none_counts(model, inputs, "low_res_logits") == (
            GRAD_LOW_RES_LOGITS
        ), (
            "low_res_logits is documented as THE training target; its gradient "
            "coverage moved"
        )

    def test_masks_carry_gradient_when_not_binarized(self):
        """
        The repair, pinned as an exact pair.

        RED against BOTH failure shapes:
          * flag absent entirely -> `SAM(binarize_masks=False)` raises;
          * flag present in `__init__`/`get_config` but ignored by `call` (the
            dead-knob shape this plan already found once, in `iou_head_depth`)
            -> `masks` still reports (201, 201) and this assertion fires.
        """
        model, inputs = _build_seeded_sam(binarize_masks=False)
        try:
            assert gradient_none_counts(model, inputs, "masks") == (
                GRAD_MASKS_FLOAT
            ), (
                "binarize_masks=False did not make `masks` differentiable -- "
                "the flag is stored but `call` ignores it (a dead knob)"
            )
            # The escape hatch must not cost the documented target anything.
            assert gradient_none_counts(model, inputs, "low_res_logits") == (
                GRAD_LOW_RES_LOGITS
            )
        finally:
            del model
            keras.backend.clear_session()
            gc.collect()

    def test_binarize_masks_false_returns_float_logits_not_a_uint8_mask(self):
        """
        Dtype and value-range discriminator for the same dead-knob shape.

        A dead flag returns uint8 zeros and ones here. Both assertions below
        fire on that, independently of the gradient probe -- an all-0/1 float
        tensor would still be suspicious, so the dtype check is the primary.
        """
        model, inputs = _build_seeded_sam(binarize_masks=False)
        try:
            masks = model(inputs)["masks"]
            assert keras.backend.standardize_dtype(masks.dtype) == "float32", (
                f"binarize_masks=False still returned "
                f"{keras.backend.standardize_dtype(masks.dtype)} masks"
            )
            values = keras.ops.convert_to_numpy(masks)
            assert not np.all(np.isin(values, (0.0, 1.0))), (
                "binarize_masks=False returned a thresholded 0/1 tensor; the "
                "flag reached the dtype but not the threshold"
            )
            assert np.all(np.isfinite(values))
        finally:
            del model
            keras.backend.clear_session()
            gc.collect()

    def test_default_masks_are_exactly_the_thresholded_logits(self):
        """
        Assumption A-6, on ONE model so the weights are identical by construction.

        The pre-change code was an unconditional
        ``ops.cast(masks > self.mask_threshold, 'uint8')``. This asserts the
        default path still produces exactly that, bit for bit, from the same
        weights and the same input -- so no pre-existing binary-mask assertion
        can have moved. Comparing two SEPARATELY constructed models would prove
        nothing: `seed_nonzero_weights` offsets each model's own random init, so
        their weights differ.
        """
        model, inputs = _build_seeded_sam(binarize_masks=True)
        try:
            binarized = keras.ops.convert_to_numpy(model(inputs)["masks"])
            assert binarized.dtype == np.uint8, (
                f"the default no longer emits uint8 masks: {binarized.dtype}"
            )

            model.binarize_masks = False
            logits = keras.ops.convert_to_numpy(model(inputs)["masks"])
            expected = (logits > model.mask_threshold).astype(np.uint8)

            assert np.array_equal(binarized, expected), (
                f"the default masks output is no longer the thresholded "
                f"full-resolution logits: {int((binarized != expected).sum())} "
                f"of {binarized.size} pixels differ"
            )
            # Guard the guard: a degenerate all-zero or all-one mask would make
            # the comparison above nearly free.
            fraction = float(binarized.mean())
            assert 0.01 < fraction < 0.99, (
                f"the fixture's masks are degenerate ({fraction:.3f} ones), so "
                f"this equality proves little"
            )
        finally:
            del model
            keras.backend.clear_session()
            gc.collect()

    @pytest.mark.parametrize("binarize", [True, False])
    def test_binarize_masks_round_trips_through_config(self, binarize: bool):
        """`get_config`/`from_config` preserve the flag at BOTH values."""
        model = build_reduced_sam()
        model = SAM(
            image_encoder=model.image_encoder,
            prompt_encoder=model.prompt_encoder,
            mask_decoder=model.mask_decoder,
            binarize_masks=binarize,
        )
        try:
            config = model.get_config()
            assert "binarize_masks" in config, (
                "binarize_masks is not serialized; a saved model would silently "
                "come back with the gradient-dead default"
            )
            assert config["binarize_masks"] is binarize

            restored = SAM.from_config(config)
            assert restored.binarize_masks is binarize, (
                f"binarize_masks did not survive from_config: "
                f"{restored.binarize_masks} != {binarize}"
            )
        finally:
            del model
            keras.backend.clear_session()
            gc.collect()


# ---------------------------------------------------------------------------
# Step 6 -- F-1: the padding point's positional encoding must be ZEROED.
#
# Reference SAM does `point_embedding[labels == -1] = 0.0` and only THEN adds
# `not_a_point_embed`, so a padding row carries the not-a-point embedding alone.
# Before the fix the Fourier PE of the dummy (0, 0) point survived and
# `not_a_point_embed` was merely added on top, which made a padding row depend
# on its coordinates. This fires on EVERY point-only prompt (`PromptEncoder.call`
# sets `pad=(boxes is None)`) -- the most common prompt in practice.
#
# Measured RED on the pre-fix code, at this file's probe geometry:
#   * padding row vs a constant-9.0 `not_a_point_embed`: max abs diff 0.962855
#     (the finding recorded 0.976 at its own geometry; the exact magnitude is a
#     draw of the RANDOM Fourier matrix and is NOT asserted anywhere here --
#     surprise #5 forbids pinning a cross-process numeric constant)
#   * two padding points at different coordinates: max abs diff 1.727846
# Both are exactly 0.0 after the fix, and exact 0.0 IS assertable: it is an
# invariance, not a magnitude.
# ---------------------------------------------------------------------------
NOT_A_POINT_CONSTANT = 9.0
PE_EMBED_DIM = 8


def build_probe_prompt_encoder(
    not_a_point_value: float = NOT_A_POINT_CONSTANT,
) -> PromptEncoder:
    """
    Build a tiny `PromptEncoder` whose type embeddings are known constants.

    `not_a_point_embed` is zero-initialized in the shipped code (carried
    surprise #1), which would make every assertion below pass identically with
    the fix present or absent -- the "guard passes both ways" trap this plan
    exists to remove. Each type embedding is therefore assigned a distinct,
    provably non-zero constant so that a padding row is only equal to
    ``not_a_point_value`` when the coordinate PE really was zeroed.

    Args:
        not_a_point_value: Constant assigned to every entry of
            ``not_a_point_embed``. Must be non-zero.

    Returns:
        A built :class:`PromptEncoder` with ``embed_dim = PE_EMBED_DIM``.
    """
    assert not_a_point_value != 0.0, (
        "a zero not_a_point_embed makes every guard in this section vacuous"
    )
    encoder = PromptEncoder(
        embed_dim=PE_EMBED_DIM,
        image_embedding_size=(16, 16),
        input_image_size=(IMG_SIZE, IMG_SIZE),
    )
    encoder.build(None)

    weight = encoder.not_a_point_embed.weights[0]
    weight.assign(np.full(weight.shape, not_a_point_value, dtype="float32"))
    # Distinct constants per point type: 1.0 (background), 2.0 (foreground),
    # 3.0 / 4.0 (box corners). Distinct so a mis-selected type embedding is
    # visible rather than silently absorbed.
    for index, embedding in enumerate(encoder.point_embeddings):
        type_weight = embedding.weights[0]
        type_weight.assign(
            np.full(type_weight.shape, float(index) + 1.0, dtype="float32")
        )
    return encoder


class TestPaddingPointPositionalEncoding:
    """F-1: `labels == -1` rows must lose their coordinate PE entirely."""

    def test_padding_row_equals_not_a_point_embed_exactly(self):
        """
        Probe P2's primary shape, as an EXACT equality.

        RED on the pre-fix code: the padding row came back as
        ``not_a_point_embed + PE((0, 0))`` -- max abs diff 0.962855 from the
        constant 9.0. This is the assertion that fired.
        """
        encoder = build_probe_prompt_encoder()
        try:
            coords = keras.ops.convert_to_tensor([[[100.0, 120.0]]])
            labels = keras.ops.convert_to_tensor([[1]])
            embedded = keras.ops.convert_to_numpy(
                encoder._embed_points(coords, labels, pad=True)
            )
            assert embedded.shape == (1, 2, PE_EMBED_DIM), (
                f"pad=True must append exactly one row; got {embedded.shape}"
            )
            padding_row = embedded[0, 1]
            diff = float(np.max(np.abs(padding_row - NOT_A_POINT_CONSTANT)))
            assert diff == 0.0, (
                f"the padding point's coordinate positional encoding SURVIVED: "
                f"the row is not_a_point_embed + PE((0,0)), max abs deviation "
                f"{diff:.6f} from the constant {NOT_A_POINT_CONSTANT}"
            )
        finally:
            del encoder
            keras.backend.clear_session()
            gc.collect()

    def test_padding_points_are_coordinate_invariant(self):
        """
        Probe P2's second shape: a padding point must not depend on WHERE it is.

        RED on the pre-fix code: two ``label == -1`` rows at (10, 20) and
        (200, 240) differed by up to 1.727846. This is the assertion that fired.
        The invariance is asserted at EXACTLY 0.0 -- it is an identity, not a
        tolerance.
        """
        encoder = build_probe_prompt_encoder()
        try:
            coords = keras.ops.convert_to_tensor([[[10.0, 20.0], [200.0, 240.0]]])
            labels = keras.ops.convert_to_tensor([[-1, -1]])
            embedded = keras.ops.convert_to_numpy(
                encoder._embed_points(coords, labels, pad=False)
            )
            diff = float(np.max(np.abs(embedded[0, 0] - embedded[0, 1])))
            assert diff == 0.0, (
                f"two padding points at DIFFERENT coordinates produced "
                f"different embeddings (max abs diff {diff:.6f}); their "
                f"coordinate PE was not zeroed"
            )
        finally:
            del encoder
            keras.backend.clear_session()
            gc.collect()

    def test_non_padding_points_keep_their_positional_encoding(self):
        """
        CONTROL against over-zeroing -- passes BOTH before and after the fix.

        The expected value is recomputed independently from
        ``pe_layer.forward_with_coords`` plus the type embedding, so this test
        is blind to the ``ops.where`` under test and can only fail if the fix
        also touched a ``label in (0, 1)`` row. A fix that zeroed everything
        would satisfy both assertions above and fail only here.
        """
        encoder = build_probe_prompt_encoder()
        try:
            coords = keras.ops.convert_to_tensor([[[10.0, 20.0], [200.0, 240.0]]])
            labels = keras.ops.convert_to_tensor([[0, 1]])
            got = keras.ops.convert_to_numpy(
                encoder._embed_points(coords, labels, pad=False)
            )

            expected_pe = keras.ops.convert_to_numpy(
                encoder.pe_layer.forward_with_coords(
                    coords + 0.5, encoder.input_image_size
                )
            )
            # Background -> point_embeddings[0] (1.0), foreground -> [1] (2.0).
            expected = expected_pe + np.array([[[1.0], [2.0]]], dtype="float32")

            assert np.max(np.abs(got - expected)) == 0.0, (
                "a non-padding point's embedding is no longer "
                "PE(coords + 0.5) + its type embedding -- the padding fix "
                "over-reached into labels 0/1"
            )
            # Guard the guard: the PE contribution must be non-trivial, else
            # this equality would hold even under total zeroing.
            assert float(np.max(np.abs(expected_pe))) > 1e-3, (
                "the positional encoding is ~zero at this geometry, so this "
                "control cannot discriminate"
            )
        finally:
            del encoder
            keras.backend.clear_session()
            gc.collect()

    def test_pad_true_row_matches_an_explicit_minus_one_label(self):
        """
        The implicitly appended padding row and an explicit ``-1`` label must
        agree bit for bit.

        `_embed_points` appends the padding point AFTER the ``points + 0.5``
        pixel-centre offset, so the implicit padding coordinate is a literal
        ``(0, 0)`` while an explicitly supplied ``(0, 0)`` becomes ``(0.5,
        0.5)``. Pre-fix those are two DIFFERENT positional encodings and this
        assertion fires (max abs diff 0.999900 measured) -- a second, previously
        unrecorded symptom of the same defect. Post-fix both are zeroed and the
        routes are indistinguishable, which is the property reference SAM has.
        """
        encoder = build_probe_prompt_encoder()
        try:
            coords = keras.ops.convert_to_tensor([[[100.0, 120.0]]])
            labels = keras.ops.convert_to_tensor([[1]])
            implicit = keras.ops.convert_to_numpy(
                encoder._embed_points(coords, labels, pad=True)
            )[0, 1]

            explicit_coords = keras.ops.convert_to_tensor(
                [[[100.0, 120.0], [0.0, 0.0]]]
            )
            explicit_labels = keras.ops.convert_to_tensor([[1, -1]])
            explicit = keras.ops.convert_to_numpy(
                encoder._embed_points(explicit_coords, explicit_labels, pad=False)
            )[0, 1]

            assert np.max(np.abs(implicit - explicit)) == 0.0, (
                "the appended padding row differs from an explicitly labelled "
                "-1 point at the same coordinates"
            )
        finally:
            del encoder
            keras.backend.clear_session()
            gc.collect()

    def test_point_only_call_appends_one_zeroed_padding_row(self):
        """
        The defect's real reach: through `call`, on the most common prompt.

        `call` sets ``pad=(boxes is None)``, so a point-only prompt gets the
        padding row and a prompt carrying boxes does not. Both branches are
        asserted here so that "no padding row at all" cannot pass as a fix.
        """
        encoder = build_probe_prompt_encoder()
        try:
            coords = keras.ops.convert_to_tensor([[[100.0, 120.0]]])
            labels = keras.ops.convert_to_tensor([[1]])

            sparse, _ = encoder(points=(coords, labels))
            sparse = keras.ops.convert_to_numpy(sparse)
            assert sparse.shape == (1, 2, PE_EMBED_DIM), (
                f"a point-only prompt must carry 1 point + 1 padding row; got "
                f"{sparse.shape}"
            )
            diff = float(np.max(np.abs(sparse[0, 1] - NOT_A_POINT_CONSTANT)))
            assert diff == 0.0, (
                f"the padding row reaching the mask decoder on a point-only "
                f"prompt is not not_a_point_embed (max abs deviation "
                f"{diff:.6f})"
            )

            boxes = keras.ops.convert_to_tensor([[[10.0, 10.0, 90.0, 90.0]]])
            sparse_with_box, _ = encoder(points=(coords, labels), boxes=boxes)
            assert keras.ops.convert_to_numpy(sparse_with_box).shape == (
                1, 3, PE_EMBED_DIM
            ), (
                "with boxes present no padding point may be appended: expected "
                "1 point + 2 box corners"
            )
        finally:
            del encoder
            keras.backend.clear_session()
            gc.collect()


class TestEncoderDegenerateDefaults:
    """
    Plan step 7 / finding F-7 --- ``ImageEncoderViT`` must not silently be a
    windowed-only encoder with no global-attention block at all.

    The shipped defaults were ``use_rel_pos=False``, ``window_size=14``,
    ``global_attn_indexes=()``: measured on the pre-fix code, a default
    ``depth=4`` encoder reported block window sizes ``[14, 14, 14, 14]``, i.e.
    **zero** global blocks, so its receptive field never became global.
    ``SAM.from_variant`` overrode both, but every direct construction --- all 9
    fixtures in ``test_model.py`` included --- silently got the degenerate
    model.

    The guard's discriminating half is the ``window_size > 0`` conjunct:
    ``window_size == 0`` already makes every block global, so an empty
    ``global_attn_indexes`` is CORRECT there, not degenerate. A guard that
    fired on both configurations would be the defect, not the fix.
    """

    ENCODER_KWARGS = dict(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        depth=4,
        num_heads=NUM_HEADS,
        out_chans=OUT_CHANS,
    )

    def test_shipped_defaults_are_refused(self):
        """The exact defaults must raise --- exception TYPE checked, not `Exception`."""
        with pytest.raises(ValueError) as excinfo:
            ImageEncoderViT(**self.ENCODER_KWARGS)
        message = str(excinfo.value)
        assert "global_attn_indexes" in message, (
            f"the raise must name the offending argument; got: {message}"
        )
        assert "window_size=14" in message, (
            f"the raise must name the offending window_size value; got: {message}"
        )
        assert "(2, 5, 8, 11)" in message, (
            f"the raise must name the reference SAM global-index pattern; got: "
            f"{message}"
        )

    def test_explicit_windowed_only_is_refused(self):
        """An explicitly-passed empty tuple is refused exactly like the default."""
        with pytest.raises(ValueError):
            ImageEncoderViT(
                window_size=WINDOW_SIZE,
                global_attn_indexes=(),
                **self.ENCODER_KWARGS,
            )

    def test_guard_does_not_fire_on_global_only_encoder(self):
        """
        THE DISCRIMINATING CONTROL.

        ``window_size=0`` makes every block global; an empty
        ``global_attn_indexes`` is then correct. A guard that also fired here
        would be over-broad, so this test must pass for the guard to be right.
        """
        encoder = ImageEncoderViT(window_size=0, **self.ENCODER_KWARGS)
        try:
            window_sizes = [blk.window_size for blk in encoder.blocks]
            assert window_sizes == [0, 0, 0, 0], (
                f"window_size=0 must leave every block global; got {window_sizes}"
            )
            output = encoder(
                keras.ops.convert_to_tensor(
                    np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype="float32")
                )
            )
            assert tuple(output.shape) == (1, GRID_SIZE, GRID_SIZE, OUT_CHANS)
        finally:
            del encoder
            keras.backend.clear_session()
            gc.collect()

    def test_guard_does_not_fire_when_one_window_covers_the_whole_grid(self):
        """
        THE SECOND DISCRIMINATING CONTROL (F-R3).

        ``img_size=224, patch_size=16`` gives a 14x14 token grid, so
        ``window_size=14`` is ONE window covering the entire grid: every block
        is effectively global and an empty ``global_attn_indexes`` is CORRECT,
        not degenerate. The step-7 guard refused this legitimate configuration
        because it only checked ``window_size > 0``.

        The grid identity is asserted, not assumed, so the case cannot silently
        stop being the one it claims to be.
        """
        img_size, patch_size, window_size = 224, 16, 14
        assert img_size // patch_size == window_size, (
            "premise check: this test is only the intended control while the "
            "window exactly equals the token grid"
        )
        encoder = ImageEncoderViT(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=32,
            depth=2,
            num_heads=2,
            out_chans=OUT_CHANS,
            window_size=window_size,
            global_attn_indexes=(),
            use_rel_pos=True,
        )
        try:
            assert [blk.window_size for blk in encoder.blocks] == [14, 14]
            output = encoder(
                keras.ops.convert_to_tensor(
                    np.zeros((1, img_size, img_size, 3), dtype="float32")
                )
            )
            assert tuple(output.shape) == (1, 14, 14, OUT_CHANS)
        finally:
            del encoder
            keras.backend.clear_session()
            gc.collect()

    def test_a_window_smaller_than_the_grid_is_still_refused(self):
        """
        The genuinely degenerate case must STILL fire after the F-R3 relaxation.

        Same geometry as the control above but ``window_size=7`` against the
        14x14 grid: four tiles per block, no block global, no global receptive
        field anywhere. A relaxation that widened to ``window_size >= grid`` by
        dropping the refusal altogether would pass the control and fail here.
        """
        with pytest.raises(ValueError, match="Degenerate encoder configuration"):
            ImageEncoderViT(
                img_size=224,
                patch_size=16,
                embed_dim=32,
                depth=2,
                num_heads=2,
                out_chans=OUT_CHANS,
                window_size=7,
                global_attn_indexes=(),
                use_rel_pos=True,
            )

    def test_guard_does_not_fire_on_reference_shaped_encoder(self):
        """A windowed encoder WITH global indices is accepted and really mixes both."""
        encoder = ImageEncoderViT(
            window_size=WINDOW_SIZE,
            global_attn_indexes=(1, 3),
            **self.ENCODER_KWARGS,
        )
        try:
            window_sizes = [blk.window_size for blk in encoder.blocks]
            assert window_sizes == [WINDOW_SIZE, 0, WINDOW_SIZE, 0], (
                f"global_attn_indexes must zero exactly the named blocks; got "
                f"{window_sizes}"
            )
            assert sum(1 for w in window_sizes if w == 0) == 2
        finally:
            del encoder
            keras.backend.clear_session()
            gc.collect()

    def test_use_rel_pos_defaults_to_true(self):
        """
        The default flip itself, asserted on the LAYOUT rather than on values.

        Carried surprise #1: ``rel_pos_h`` / ``rel_pos_w`` are zero-initialized,
        so at initialization this flip is numerically INERT --- measured max abs
        output difference **exactly 0.0** between ``use_rel_pos=True`` and
        ``False`` when the 57 shared weights are transplanted and the rel-pos
        tables keep their zero init. With the tables seeded non-zero the same
        comparison moves (8.42e-4). A value-level guard on freshly-initialized
        weights therefore could not observe this change at all; the observable
        consequence is the two extra weight tensors per block.
        """
        default_encoder = ImageEncoderViT(
            window_size=WINDOW_SIZE, global_attn_indexes=(1, 3), **self.ENCODER_KWARGS
        )
        try:
            assert default_encoder.use_rel_pos is True, (
                "use_rel_pos must default to True (reference SAM sets it for "
                "every released variant)"
            )
            assert all(blk.attn.use_rel_pos for blk in default_encoder.blocks), (
                "the default must reach every block's attention layer, not just "
                "the encoder's own attribute"
            )
            default_encoder(
                keras.ops.convert_to_tensor(
                    np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype="float32")
                )
            )
            rel_pos_weights = [
                w for w in default_encoder.weights
                if w.path.rsplit("/", 1)[-1] in ("rel_pos_h", "rel_pos_w")
            ]
            assert len(rel_pos_weights) == 8, (
                f"depth=4 with use_rel_pos=True must create 2 rel-pos tables per "
                f"block; got {len(rel_pos_weights)}"
            )
            assert all(
                float(np.abs(keras.ops.convert_to_numpy(w)).max()) == 0.0
                for w in rel_pos_weights
            ), (
                "carried surprise #1 no longer holds: rel-pos tables are no "
                "longer zero-initialized, so the inertness reasoning above must "
                "be re-derived"
            )
        finally:
            del default_encoder
            keras.backend.clear_session()
            gc.collect()

    def test_config_round_trip_preserves_the_accepted_configuration(self):
        """`from_config(get_config())` must not resurrect a refused configuration."""
        encoder = ImageEncoderViT(
            window_size=WINDOW_SIZE, global_attn_indexes=(1, 3), **self.ENCODER_KWARGS
        )
        try:
            restored = ImageEncoderViT.from_config(encoder.get_config())
            assert restored.use_rel_pos is True
            assert tuple(restored.global_attn_indexes) == (1, 3)
            assert restored.window_size == WINDOW_SIZE
        finally:
            del encoder
            keras.backend.clear_session()
            gc.collect()


# ---------------------------------------------------------------------------
# Plan step 8 / finding F-8 --- oversize `preprocess` input
# ---------------------------------------------------------------------------
class TestPreprocessOversizeGuard:
    """
    ``SAM.preprocess`` PADS to the encoder size; it can never shrink.

    Pre-fix, an image larger than ``img_size`` produced a negative ``pad_h`` /
    ``pad_w`` and ``ops.pad`` raised ``tf.errors.InvalidArgumentError``. That is
    not an input-validation error a caller can catch: its MRO is
    ``InvalidArgumentError -> OpError -> Exception``, so it is **not** a
    ``ValueError`` subclass (asserted by execution in
    :meth:`test_invalid_argument_error_is_not_a_value_error` --- without that
    assertion a ``pytest.raises(Exception)`` here would pass against the
    *unfixed* code and prove nothing).
    """

    def test_invalid_argument_error_is_not_a_value_error(self):
        """
        THE EXCEPTION-FAMILY PROOF, executed rather than assumed.

        This is what makes the oversize guard's ``pytest.raises(ValueError)``
        discriminating: the pre-fix failure mode is in a disjoint exception
        family, so the guard cannot be satisfied by the old behaviour.
        """
        assert not issubclass(tf.errors.InvalidArgumentError, ValueError), (
            "InvalidArgumentError became a ValueError subclass; the oversize "
            "guard's pytest.raises(ValueError) would no longer discriminate "
            "the fixed code from the unfixed code and must be redesigned"
        )
        assert issubclass(tf.errors.InvalidArgumentError, Exception)

    def test_oversize_input_raises_value_error_naming_size_and_remedy(self):
        """An image larger than `img_size` is refused with an actionable message."""
        model = build_reduced_sam()
        try:
            oversize = keras.ops.convert_to_tensor(
                np.zeros((1, IMG_SIZE + 32, IMG_SIZE + 64, 3), dtype="float32")
            )
            with pytest.raises(ValueError) as excinfo:
                model.preprocess(oversize)
            message = str(excinfo.value)
            assert str(IMG_SIZE + 32) in message, (
                f"the raise must name the offending extent; got: {message}"
            )
            assert f"img_size={IMG_SIZE}" in message, (
                f"the raise must name the encoder img_size; got: {message}"
            )
            assert "resize_longest_side" in message, (
                f"the raise must name the in-repo remedy; got: {message}"
            )
        finally:
            del model
            keras.backend.clear_session()
            gc.collect()

    def test_oversize_in_either_axis_alone_is_refused(self):
        """Height-only and width-only overflows are both caught."""
        model = build_reduced_sam()
        try:
            for shape in (
                (1, IMG_SIZE + 1, IMG_SIZE, 3),
                (1, IMG_SIZE, IMG_SIZE + 1, 3),
            ):
                tensor = keras.ops.convert_to_tensor(
                    np.zeros(shape, dtype="float32")
                )
                with pytest.raises(ValueError):
                    model.preprocess(tensor)
        finally:
            del model
            keras.backend.clear_session()
            gc.collect()

    def test_guard_does_not_fire_on_a_within_size_image(self):
        """
        THE NON-FIRING CONTROL.

        A smaller-than-``img_size`` image must still be padded, not refused --
        an over-broad guard (e.g. one demanding an exact match) would satisfy
        every raising assertion above and break the model outright.
        """
        model = build_reduced_sam()
        try:
            small = keras.ops.convert_to_tensor(
                np.zeros((1, IMG_SIZE - 40, IMG_SIZE - 90, 3), dtype="float32")
            )
            padded = model.preprocess(small)
            assert tuple(padded.shape) == (1, IMG_SIZE, IMG_SIZE, 3), (
                f"a within-size image must be padded to the encoder size; got "
                f"{tuple(padded.shape)}"
            )
            exact = keras.ops.convert_to_tensor(
                np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype="float32")
            )
            assert tuple(model.preprocess(exact).shape) == (
                1, IMG_SIZE, IMG_SIZE, 3
            ), "an exactly-img_size image must not be refused"
        finally:
            del model
            keras.backend.clear_session()
            gc.collect()

    def test_resize_longest_side_makes_an_oversize_image_acceptable(self):
        """The raise names a remedy; the remedy must actually work end to end."""
        model = build_reduced_sam()
        try:
            raw = np.random.RandomState(3).uniform(
                0.0, 255.0, size=(1, 400, IMG_SIZE * 2, 3)
            ).astype("float32")
            resized = resize_longest_side(
                keras.ops.convert_to_tensor(raw), IMG_SIZE
            )
            padded = model.preprocess(resized)
            assert tuple(padded.shape) == (1, IMG_SIZE, IMG_SIZE, 3), (
                f"resize_longest_side must produce an input preprocess accepts; "
                f"got {tuple(padded.shape)}"
            )
        finally:
            del model
            keras.backend.clear_session()
            gc.collect()


class TestResizeLongestSide:
    """
    Plan step 8 / D-005 --- the transform reference SAM assumes exists.

    The classic defect in a longest-side resize is an ORIENTATION bug: scaling
    by the wrong axis passes every square-image test and every landscape test
    while silently up-scaling portraits past the target. Both orientations are
    therefore asserted, and the assertions are asymmetric (each pins which axis
    must equal the target).
    """

    def test_landscape_pins_the_width_and_preserves_aspect(self):
        image = keras.ops.convert_to_tensor(
            np.zeros((300, 900, 3), dtype="float32")
        )
        out = tuple(resize_longest_side(image, 1024).shape)
        # scale = 1024/900; 300*scale = 341.33 -> int(+0.5) = 341
        assert out == (341, 1024, 3), (
            f"landscape: the WIDTH is the longest side and must equal the "
            f"target; got {out}"
        )

    def test_portrait_pins_the_height_and_preserves_aspect(self):
        """
        THE ORIENTATION DISCRIMINATOR.

        Exactly the transpose of the landscape case. A transform that always
        scaled by (say) the width would return ``(3413, 1024, 3)`` here --
        larger than the target on the long axis -- while still passing the
        landscape test.
        """
        image = keras.ops.convert_to_tensor(
            np.zeros((900, 300, 3), dtype="float32")
        )
        out = tuple(resize_longest_side(image, 1024).shape)
        assert out == (1024, 341, 3), (
            f"portrait: the HEIGHT is the longest side and must equal the "
            f"target; got {out}"
        )

    @pytest.mark.parametrize(
        "h,w",
        [(300, 900), (900, 300), (1000, 1000), (37, 1201), (1201, 37), (512, 512)],
    )
    def test_longest_side_hits_the_target_exactly(self, h: int, w: int):
        """Across both orientations and extreme aspect ratios."""
        image = keras.ops.convert_to_tensor(
            np.zeros((h, w, 3), dtype="float32")
        )
        new_h, new_w, _ = tuple(resize_longest_side(image, 1024).shape)
        assert max(new_h, new_w) == 1024, (
            f"the longest side must equal the target exactly; "
            f"({h},{w}) -> ({new_h},{new_w})"
        )
        assert min(new_h, new_w) <= 1024
        # Aspect ratio preserved to within the +-0.5 px of the rounding rule.
        # Compared as short/long so the tolerance is orientation-symmetric.
        before = min(h, w) / max(h, w)
        after = min(new_h, new_w) / max(new_h, new_w)
        assert abs(after - before) < (1.0 / max(new_h, new_w)), (
            f"aspect ratio not preserved: ({h},{w}) -> ({new_h},{new_w})"
        )

    def test_square_image_at_target_is_a_shape_no_op(self):
        image = keras.ops.convert_to_tensor(
            np.random.RandomState(11).uniform(
                0.0, 1.0, size=(256, 256, 3)
            ).astype("float32")
        )
        out = resize_longest_side(image, 256)
        assert tuple(out.shape) == (256, 256, 3)
        assert float(
            np.max(np.abs(
                keras.ops.convert_to_numpy(out)
                - keras.ops.convert_to_numpy(image)
            ))
        ) == 0.0, "a no-op resize must not perturb the pixel values"

    def test_batched_rank_four_input_keeps_its_batch_axis(self):
        image = keras.ops.convert_to_tensor(
            np.zeros((2, 300, 900, 3), dtype="float32")
        )
        assert tuple(resize_longest_side(image, 1024).shape) == (2, 341, 1024, 3)

    def test_rounding_rule_matches_reference_int_plus_half(self):
        """
        Reference SAM rounds with ``int(x + 0.5)``, not Python's ``round``.

        ``h=2, w=3, target=3`` gives an exact ``.5``: ``2 * 1.0 = 2.0`` is
        uninteresting, so use ``h=1, w=2, target=5`` -> ``1 * 2.5 = 2.5``.
        ``int(2.5 + 0.5) == 3`` while ``round(2.5) == 2`` (banker's rounding).
        This test is what makes that distinction load-bearing.
        """
        assert round(2.5) == 2, "python round is banker's rounding"
        image = keras.ops.convert_to_tensor(
            np.zeros((1, 2, 3), dtype="float32")
        )
        assert tuple(resize_longest_side(image, 5).shape) == (3, 5, 3), (
            "the rounding rule must be int(x + 0.5), not round()"
        )

    def test_bad_rank_and_bad_target_are_refused(self):
        with pytest.raises(ValueError, match="rank 3"):
            resize_longest_side(
                keras.ops.convert_to_tensor(np.zeros((300, 900), dtype="float32")),
                1024,
            )
        with pytest.raises(ValueError, match="target_length"):
            resize_longest_side(
                keras.ops.convert_to_tensor(
                    np.zeros((300, 900, 3), dtype="float32")
                ),
                0,
            )


# ---------------------------------------------------------------------------
# Plan step 9 / findings F-9 + F-10 --- geometry guards
# ---------------------------------------------------------------------------
def _build_probe_decoder() -> MaskDecoder:
    """A reduced MaskDecoder matching the module fixture's decoder geometry."""
    return MaskDecoder(
        transformer_dim=OUT_CHANS,
        transformer=TwoWayTransformer(
            depth=2, embedding_dim=OUT_CHANS, num_heads=2, mlp_dim=64
        ),
        iou_head_hidden_dim=32,
    )


def _decoder_inputs(batch_size: int, sparse_batch: int) -> Dict[str, Any]:
    """Decoder call kwargs at an explicit (image batch, sparse batch) pair."""
    rng = np.random.RandomState(5)
    embedding = rng.normal(
        size=(batch_size, GRID_SIZE, GRID_SIZE, OUT_CHANS)
    ).astype("float32")
    return dict(
        image_embeddings=keras.ops.convert_to_tensor(embedding),
        image_pe=keras.ops.convert_to_tensor(
            rng.normal(size=(1, GRID_SIZE, GRID_SIZE, OUT_CHANS)).astype("float32")
        ),
        sparse_prompt_embeddings=keras.ops.convert_to_tensor(
            rng.normal(size=(sparse_batch, 2, OUT_CHANS)).astype("float32")
        ),
        dense_prompt_embeddings=keras.ops.convert_to_tensor(
            rng.normal(
                size=(batch_size, GRID_SIZE, GRID_SIZE, OUT_CHANS)
            ).astype("float32")
        ),
        multimask_output=True,
    )


class TestSparseBatchTilingGuard:
    """
    Plan step 9 / finding F-9 --- ``MaskDecoder`` refuses an impossible or
    order-scrambling prompt tile.

    ``ops.tile(sparse, [batch_size // sparse_batch, 1, 1])`` is integer
    division. Two probe geometries break it, in two different ways:

    * ``B=1`` with 3 prompt rows -> factor ``0`` -> opaque
      ``InvalidArgumentError`` (a crash, at least loud).
    * ``B=4`` with 2 prompt sets -> tiles to ``[a, b, a, b]`` rather than
      ``[a, a, b, b]``, so every image is scored against the WRONG prompt with
      no error at all (the dangerous one).

    ``B=2`` with 1 shared prompt set is the working geometry and is the
    non-firing control: it is the only thing that can tell this guard from an
    over-broad one that simply demands ``sparse_batch == batch_size``.
    """

    def test_impossible_tile_batch_one_three_prompts_is_refused(self):
        decoder = _build_probe_decoder()
        try:
            with pytest.raises(ValueError) as excinfo:
                decoder(**_decoder_inputs(batch_size=1, sparse_batch=3))
            message = str(excinfo.value)
            assert "3" in message and "batch_size=1" in message, (
                f"the raise must name BOTH the sparse batch and the image "
                f"batch; got: {message}"
            )
        finally:
            del decoder
            keras.backend.clear_session()
            gc.collect()

    def test_order_scrambling_tile_batch_four_two_prompts_is_refused(self):
        """
        The silent one. Pre-fix this returned a plausible result computed
        against mismatched prompts, so nothing in the suite could see it.
        """
        decoder = _build_probe_decoder()
        try:
            with pytest.raises(ValueError) as excinfo:
                decoder(**_decoder_inputs(batch_size=4, sparse_batch=2))
            message = str(excinfo.value)
            assert "batch_size=4" in message, (
                f"the raise must name the image batch; got: {message}"
            )
        finally:
            del decoder
            keras.backend.clear_session()
            gc.collect()

    def test_guard_does_not_fire_on_the_working_shared_prompt_geometry(self):
        """
        THE NON-FIRING CONTROL.

        ``B=2`` / ``sparse_batch=1`` is the shared-prompt broadcast that works
        today and must keep working; a guard demanding an exact match would
        pass both raising tests above and break the model's most common call.
        """
        decoder = _build_probe_decoder()
        try:
            masks, iou = decoder(**_decoder_inputs(batch_size=2, sparse_batch=1))
            assert tuple(masks.shape)[0] == 2
            assert tuple(iou.shape)[0] == 2
            assert bool(np.all(np.isfinite(keras.ops.convert_to_numpy(masks))))
        finally:
            del decoder
            keras.backend.clear_session()
            gc.collect()

    def test_guard_does_not_fire_on_per_image_prompts(self):
        """
        SECOND NON-FIRING CONTROL.

        ``sparse_batch == batch_size`` is the other legitimate geometry (one
        prompt set per image, tile factor 1). A guard that only allowed
        ``sparse_batch == 1`` would satisfy every other assertion here.
        """
        decoder = _build_probe_decoder()
        try:
            masks, _ = decoder(**_decoder_inputs(batch_size=3, sparse_batch=3))
            assert tuple(masks.shape)[0] == 3
        finally:
            del decoder
            keras.backend.clear_session()
            gc.collect()


class TestMaskPromptShapeGuard:
    """
    Plan step 9 / finding F-10 --- ``PromptEncoder`` refuses a mask prompt whose
    spatial size is not ``4 * image_embedding_size``.

    The mask-downscaling stack is a fixed two-stride-2 conv chain. Pre-fix, a
    32x32 mask against ``image_embedding_size=(16, 16)`` returned ``(1, 8, 8, 8)``
    from the encoder without complaint; the failure surfaced far downstream as a
    broadcast error inside ``image_embeddings + dense_embeddings``, naming
    neither the mask nor the prompt encoder.
    """

    IMAGE_EMBEDDING_SIZE = (16, 16)

    def _encoder(self) -> PromptEncoder:
        return PromptEncoder(
            embed_dim=OUT_CHANS,
            image_embedding_size=self.IMAGE_EMBEDDING_SIZE,
            input_image_size=(IMG_SIZE, IMG_SIZE),
            mask_in_chans=8,
        )

    def _mask(self, h: int, w: int) -> keras.KerasTensor:
        return keras.ops.convert_to_tensor(
            np.random.RandomState(7).normal(size=(1, 1, h, w)).astype("float32")
        )

    def test_probe_geometry_32x32_against_16x16_grid_is_refused(self):
        """The exact probe P19 geometry."""
        encoder = self._encoder()
        try:
            with pytest.raises(ValueError) as excinfo:
                encoder(masks=self._mask(32, 32))
            message = str(excinfo.value)
            assert "(64, 64)" in message, (
                f"the raise must name the REQUIRED size (4 * 16); got: {message}"
            )
            assert "(32, 32)" in message, (
                f"the raise must name the offending size; got: {message}"
            )
            assert "image_embedding_size=(16, 16)" in message, (
                f"the raise must name the grid it was measured against; got: "
                f"{message}"
            )
        finally:
            del encoder
            keras.backend.clear_session()
            gc.collect()

    @pytest.mark.parametrize("h,w", [(64, 32), (32, 64), (128, 128), (63, 64)])
    def test_any_axis_mismatch_is_refused(self, h: int, w: int):
        """Both too-small and too-large, and single-axis mismatches."""
        encoder = self._encoder()
        try:
            with pytest.raises(ValueError):
                encoder(masks=self._mask(h, w))
        finally:
            del encoder
            keras.backend.clear_session()
            gc.collect()

    def test_guard_does_not_fire_on_the_required_size(self):
        """
        THE NON-FIRING CONTROL.

        A 64x64 mask against a (16, 16) grid is exactly the contract and must
        produce a dense embedding on the grid. A guard that fired here (or one
        that merely demanded "some" 4x relation without pinning the grid) would
        satisfy every raising assertion above.
        """
        encoder = self._encoder()
        try:
            _, dense = encoder(masks=self._mask(64, 64))
            assert tuple(dense.shape) == (1, 16, 16, OUT_CHANS), (
                f"a contract-sized mask must yield a grid-sized dense "
                f"embedding; got {tuple(dense.shape)}"
            )
            assert bool(np.all(np.isfinite(keras.ops.convert_to_numpy(dense))))
        finally:
            del encoder
            keras.backend.clear_session()
            gc.collect()

    def test_guard_does_not_fire_when_no_mask_prompt_is_given(self):
        """
        SECOND NON-FIRING CONTROL: the no-mask path must not be touched at all.

        ``call`` substitutes the learned `no_mask_embed` when `masks is None`;
        a guard placed one level too high would break every point-only prompt,
        i.e. the most common call in the package.
        """
        encoder = self._encoder()
        try:
            _, dense = encoder(
                points=(
                    keras.ops.convert_to_tensor([[[10.0, 20.0]]]),
                    keras.ops.convert_to_tensor([[1]]),
                )
            )
            assert tuple(dense.shape) == (1, 16, 16, OUT_CHANS)
        finally:
            del encoder
            keras.backend.clear_session()
            gc.collect()


# ---------------------------------------------------------------------------
# Plan step 10 / finding F-11 --- build_from_config load cost
# ---------------------------------------------------------------------------
class TestBuildFromConfigLoadCost:
    """
    Plan step 10 / finding F-11 --- ``SAM.build_from_config`` must materialize
    the COMPLETE weight set before Keras restores the saved values.

    F-11 observed that ``build_from_config`` runs a full-resolution dummy
    forward on every ``load_model`` and proposed replacing it with an explicit
    ``build()`` chain. The plan pre-committed a rule: replace it ONLY IF the
    chain gives an identical weight count AND a value-exact round-trip.
    Measured on this fixture, the chain alone materializes **138 of 202**
    weights, so 64 are created fresh (random) on the first real ``call()`` after
    restore and the ``low_res_logits`` drift is of order 1-2 absolute. The
    dummy forward is therefore KEPT (D-018), and these guards exist so a future
    author cannot re-apply the "optimization" silently.

    The load WALL-CLOCK is deliberately asserted nowhere: it varies across
    processes (carried surprise #8) and a cross-process constant would be a
    flake generator. It is recorded in the ``build_from_config`` docstring
    instead.
    """

    def _saved_fixture(self, tmpdir: str) -> Tuple[SAM, Dict[str, Any], str]:
        model = build_reduced_sam()
        inputs = sam_inputs()
        model(inputs)
        seed_nonzero_weights(model)
        path = os.path.join(tmpdir, "sam_load_cost.keras")
        model.save(path)
        return model, inputs, path

    def test_load_materializes_every_weight_before_any_forward(self):
        """
        The load-time count is the ONLY count that can see the defect.

        Sampled before the restored model is ever called, the shipped
        implementation must already hold the full weight set; the build-only
        variant holds 138.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            model, _, path = self._saved_fixture(tmpdir)
            try:
                restored = keras.models.load_model(path)
                n_at_load = len(restored.weights)
                assert n_at_load == BASELINE_FIXTURE_WEIGHT_COUNT, (
                    f"build_from_config materialized {n_at_load} weights at "
                    f"load time, expected {BASELINE_FIXTURE_WEIGHT_COUNT}. Any "
                    f"weight not yet created when Keras restores the archive is "
                    f"built FRESH on the first call and its saved value is "
                    f"silently dropped (F-11 / D-018)"
                )
                del restored
            finally:
                del model
                keras.backend.clear_session()
                gc.collect()

    def test_weight_count_does_not_grow_on_the_first_forward_after_load(self):
        """
        States the property directly: nothing may be built lazily after restore.

        This is also why a post-forward count is a blind instrument -- both the
        shipped and the build-only variants report 202 once a forward has run.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            model, inputs, path = self._saved_fixture(tmpdir)
            try:
                restored = keras.models.load_model(path)
                n_at_load = len(restored.weights)
                restored(inputs)
                n_after_forward = len(restored.weights)
                assert n_at_load == n_after_forward, (
                    f"{n_after_forward - n_at_load} weight(s) were created by "
                    f"the first forward pass after load ({n_at_load} -> "
                    f"{n_after_forward}); those were restored from nothing"
                )
                del restored
            finally:
                del model
                keras.backend.clear_session()
                gc.collect()

    def test_restored_weight_values_match_the_saved_ones_before_any_forward(self):
        """
        The VALUE guard, which is what actually discriminates.

        Index-aligned comparison of every weight against the pre-save values,
        sampled before the restored model is called. Under the build-only
        variant exactly 64 of 202 differ.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            model, _, path = self._saved_fixture(tmpdir)
            try:
                saved = [
                    keras.ops.convert_to_numpy(w).copy() for w in model.weights
                ]
                restored = keras.models.load_model(path)
                got = [keras.ops.convert_to_numpy(w) for w in restored.weights]
                assert len(saved) == len(got), (
                    f"weight-list length moved on load: {len(saved)} -> {len(got)}"
                )
                differing = [
                    i
                    for i, (a, b) in enumerate(zip(saved, got))
                    if a.shape != b.shape or float(np.max(np.abs(a - b))) > 0.0
                ]
                assert not differing, (
                    f"{len(differing)} of {len(saved)} restored weights differ "
                    f"from the saved values before any forward pass "
                    f"(indices {differing[:8]}...); these were re-initialized, "
                    f"not restored"
                )
                del restored
            finally:
                del model
                keras.backend.clear_session()
                gc.collect()


# ---------------------------------------------------------------------------
# Plan step 11 / findings F-12..F-16 --- hygiene
# ---------------------------------------------------------------------------
class TestComputeOutputShapeIsCallableByKeras:
    """
    Plan step 11 / finding F-14 --- every ``compute_output_shape`` in the
    package must actually be invocable by Keras.

    **F-14's stated cause is FALSIFIED and its scope is narrower than claimed**,
    both verified by execution against keras 3.8.0. Keras does NOT require a
    single ``input_shape``: ``update_shapes_dict_for_target_fn`` supports a
    multi-argument ``compute_output_shape`` provided every argument is named
    ``<call argument>_shape``. Nor does a tuple-of-tuples return break anything
    --- ``compute_output_spec`` maps it with ``tree.map_shape_structure``.

    The real defect was argument NAMES: ``TwoWayAttentionBlock`` declared
    ``query_shape``/``key_shape`` against ``call(queries, keys, ...)`` and
    ``TwoWayTransformer`` declared ``image_shape``/``point_shape`` against
    ``call(image_embedding, image_pe, point_embedding)``, so Keras raised
    ``ValueError`` before reaching either body. ``MaskDecoder`` and
    ``PromptEncoder`` were already callable and are UNCHANGED; their two tests
    here pass both before and after this step and are labelled as regression
    pins, not as coverage of a fixed defect.
    """

    def test_two_way_attention_block_shape_is_callable(self):
        """RED before the fix: ValueError from the argument-name check."""
        block = TwoWayAttentionBlock(embedding_dim=32, num_heads=2, mlp_dim=64)
        try:
            queries, keys = block.compute_output_spec(
                keras.KerasTensor((1, 5, 32)),
                keras.KerasTensor((1, 256, 32)),
                keras.KerasTensor((1, 5, 32)),
                keras.KerasTensor((1, 256, 32)),
            )
            assert tuple(queries.shape) == (1, 5, 32)
            assert tuple(keys.shape) == (1, 256, 32)
        finally:
            del block
            keras.backend.clear_session()
            gc.collect()

    def test_two_way_transformer_shape_is_callable(self):
        """
        RED before the fix, and the body is exercised too: the key axis must be
        the FLATTENED image grid, which a name-only correction could still get
        wrong.
        """
        transformer = TwoWayTransformer(
            depth=2, embedding_dim=32, num_heads=2, mlp_dim=64
        )
        try:
            queries, keys = transformer.compute_output_spec(
                keras.KerasTensor((1, 16, 16, 32)),
                keras.KerasTensor((1, 16, 16, 32)),
                keras.KerasTensor((1, 5, 32)),
            )
            assert tuple(queries.shape) == (1, 5, 32), (
                f"queries must follow point_embedding; got {tuple(queries.shape)}"
            )
            assert tuple(keys.shape) == (1, 16 * 16, 32), (
                f"keys must be the image grid flattened to 256; got "
                f"{tuple(keys.shape)}"
            )
        finally:
            del transformer
            keras.backend.clear_session()
            gc.collect()

    def test_mask_decoder_shape_is_callable(self):
        """
        REGRESSION PIN, not coverage: this site was already callable, so it
        passes both ways. F-14 named it wrongly.
        """
        transformer = TwoWayTransformer(
            depth=2, embedding_dim=OUT_CHANS, num_heads=2, mlp_dim=64
        )
        decoder = MaskDecoder(
            transformer_dim=OUT_CHANS, transformer=transformer,
            iou_head_hidden_dim=32,
        )
        try:
            masks, iou = decoder.compute_output_spec(
                keras.KerasTensor((1, GRID_SIZE, GRID_SIZE, OUT_CHANS)),
                keras.KerasTensor((1, GRID_SIZE, GRID_SIZE, OUT_CHANS)),
                keras.KerasTensor((1, 5, OUT_CHANS)),
                keras.KerasTensor((1, GRID_SIZE, GRID_SIZE, OUT_CHANS)),
                True,
            )
            assert tuple(masks.shape)[1] == decoder.num_mask_tokens
            assert tuple(iou.shape) == (None, decoder.num_mask_tokens)
        finally:
            del decoder, transformer
            keras.backend.clear_session()
            gc.collect()

    def test_prompt_encoder_shape_is_callable(self):
        """
        REGRESSION PIN, not coverage: already callable before this step.
        Exercised through a tuple-valued first call argument (``points``),
        which is the shape structure most likely to break the resolution.
        """
        encoder = PromptEncoder(
            embed_dim=OUT_CHANS,
            image_embedding_size=(GRID_SIZE, GRID_SIZE),
            input_image_size=(IMG_SIZE, IMG_SIZE),
            mask_in_chans=8,
        )
        try:
            sparse, dense = encoder.compute_output_spec(
                (keras.KerasTensor((1, 1, 2)), keras.KerasTensor((1, 1)))
            )
            assert tuple(sparse.shape) == (None, None, OUT_CHANS)
            assert tuple(dense.shape) == (None, GRID_SIZE, GRID_SIZE, OUT_CHANS)
        finally:
            del encoder
            keras.backend.clear_session()
            gc.collect()


class TestEncoderOutputGridAndInstanceAttributes:
    """
    Plan step 11 / findings F-12 and F-13.

    F-12 was a docstring claiming the encoder emits ``img_size/patch_size/4``.
    A docstring cannot be RED-proved, so what is pinned here is the FACT the
    docstring got wrong.

    **F-13's premise is FALSIFIED, verified by execution.** The class-level
    ``mask_threshold``/``image_format`` pair was not dead-and-shadowed: TF's
    ``KerasAutoTrackable.__setattr__`` short-circuits on
    ``getattr(self, name) is value``, and at the defaults the assigned objects
    ARE the class-level ones (``"RGB"`` is interned; the ``0.0`` constants are
    deduped), so ``__init__`` set nothing and the class attributes were the live
    storage. Measured before the deletion: ``'mask_threshold' in
    instance.__dict__`` was ``False``.
    """

    def test_encoder_output_grid_is_img_size_over_patch_size(self):
        """
        The neck is stride-1, so the grid comes from the patch embedding alone
        --- no further /4. This is the claim F-12's docstring inverted.
        """
        encoder = ImageEncoderViT(
            img_size=IMG_SIZE, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM,
            depth=1, num_heads=NUM_HEADS, out_chans=OUT_CHANS,
            use_rel_pos=True, window_size=WINDOW_SIZE, global_attn_indexes=(0,),
        )
        try:
            out = encoder(
                keras.ops.convert_to_tensor(
                    np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype="float32")
                )
            )
            expected = (1, IMG_SIZE // PATCH_SIZE, IMG_SIZE // PATCH_SIZE, OUT_CHANS)
            assert tuple(out.shape) == expected, (
                f"encoder grid must be img_size/patch_size, got "
                f"{tuple(out.shape)} against {expected}"
            )
        finally:
            del encoder
            keras.backend.clear_session()
            gc.collect()

    def test_threshold_and_format_are_stored_on_the_instance(self):
        """
        RED before the deletion: both were absent from ``__dict__`` because the
        trackable short-circuit skipped the assignment entirely.
        """
        model = build_reduced_sam()
        try:
            assert "mask_threshold" in model.__dict__, (
                "mask_threshold is not on the instance; the class-level default "
                "is shadowing the assignment via the trackable short-circuit"
            )
            assert "image_format" in model.__dict__, (
                "image_format is not on the instance; same short-circuit"
            )
            assert model.mask_threshold == 0.0
            assert model.image_format == "RGB"
        finally:
            del model
            keras.backend.clear_session()
            gc.collect()

    def test_class_carries_no_shadowing_defaults(self):
        """The pair must be gone from the class body, not merely overwritten."""
        assert "mask_threshold" not in SAM.__dict__
        assert "image_format" not in SAM.__dict__

    def test_a_non_default_threshold_still_reaches_the_instance(self):
        """
        NON-FIRING CONTROL: the non-default path always worked (a distinct
        object never hits the short-circuit) and must keep working, so a
        "fix" that only repaired the default case is not enough.
        """
        encoder = ImageEncoderViT(
            img_size=IMG_SIZE, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM,
            depth=1, num_heads=NUM_HEADS, out_chans=OUT_CHANS,
            use_rel_pos=True, window_size=WINDOW_SIZE, global_attn_indexes=(0,),
        )
        prompt_encoder = PromptEncoder(
            embed_dim=OUT_CHANS,
            image_embedding_size=(GRID_SIZE, GRID_SIZE),
            input_image_size=(IMG_SIZE, IMG_SIZE), mask_in_chans=8,
        )
        transformer = TwoWayTransformer(
            depth=2, embedding_dim=OUT_CHANS, num_heads=2, mlp_dim=64
        )
        decoder = MaskDecoder(
            transformer_dim=OUT_CHANS, transformer=transformer,
            iou_head_hidden_dim=32,
        )
        model = SAM(
            image_encoder=encoder, prompt_encoder=prompt_encoder,
            mask_decoder=decoder, mask_threshold=0.5,
        )
        try:
            assert model.mask_threshold == 0.5
            assert model.get_config()["mask_threshold"] == 0.5
        finally:
            del model, decoder, transformer, prompt_encoder, encoder
            keras.backend.clear_session()
            gc.collect()



class TestDensePositionalEncodingGraphSafety:
    """
    F-R2 / plan step 14 --- the coordinate grid is RECOMPUTED, never memoized.

    Step 11 (F-16, D-020) cached the normalized ``(h, w, 2)`` grid on the
    instance. The cache was correct in every way the suite could then see -- it
    depends on no weight, so the staleness class D-020 reasoned about really is
    absent -- and it was still a defect, for a reason no round-trip, weight-count
    or value test in this file could reach: a tensor produced inside a
    ``tf.function`` / ``predict`` / ``fit`` / ``jit_compile`` trace belongs to
    that trace's ``FuncGraph``. A slot filled during the first trace hands a dead
    ``SymbolicTensor`` to every later call, so the layer raises
    ``TypeError: ... is out of scope and cannot be used here`` forever -- eagerly
    AND on a second trace.

    The guards below are the ones the whole plan lacked: they execute the layer
    under a trace and then eagerly, in that order. Carried surprise #8, sharpened
    --- a value-exact round-trip is not a sufficient instrument for a cache, and
    neither is a same-context recompute comparison.
    """

    def _encoder(self, grid: int = GRID_SIZE) -> PromptEncoder:
        encoder = PromptEncoder(
            embed_dim=OUT_CHANS,
            image_embedding_size=(grid, grid),
            input_image_size=(IMG_SIZE, IMG_SIZE),
            mask_in_chans=8,
        )
        encoder.build(None)
        return encoder

    def test_no_memoized_tensor_state_survives_a_call(self):
        """
        PROOF 0, structural: the layer holds no tensor-valued instance slot.

        This is the property the graph guards below rest on, asserted directly
        so that a future re-introduction of a cache fails HERE with a readable
        message rather than as an "out of scope" TypeError three tests down.
        """
        encoder = self._encoder()
        try:
            pe_layer = encoder.pe_layer
            encoder.get_dense_pe()
            leftovers = {
                name: type(value).__name__
                for name, value in vars(pe_layer).items()
                if name.startswith("_coord")
            }
            assert leftovers == {}, (
                f"PositionEmbeddingRandom memoized {leftovers}; a tensor cached "
                f"on the instance is graph-poisonous (F-R2)"
            )
        finally:
            del encoder
            keras.backend.clear_session()
            gc.collect()

    def test_position_embedding_survives_trace_then_eager_then_trace(self):
        """
        PROOF 1, the exact sequence the reviewer used to break the cache.

        Trace the layer under ``tf.function``, then call it EAGERLY, then trace a
        SECOND time. With the step-11 cache installed the eager call raised
        ``TypeError: <tf.Tensor 'stack:0' ...> is out of scope and cannot be
        used here``, because the slot held the first trace's symbolic tensor.
        """
        pe_layer = PositionEmbeddingRandom(num_pos_feats=OUT_CHANS // 2)
        pe_layer.build(None)
        try:
            traced = tf.function(lambda: pe_layer.call((8, 8)))

            first = keras.ops.convert_to_numpy(traced()).copy()
            # The call that used to raise.
            eager = keras.ops.convert_to_numpy(pe_layer.call((8, 8))).copy()
            # A second trace must also still work.
            second = keras.ops.convert_to_numpy(traced())

            assert first.shape == (OUT_CHANS, 8, 8)
            assert float(np.max(np.abs(first - eager))) == 0.0, (
                "the eager call after a trace returned a different value"
            )
            assert float(np.max(np.abs(first - second))) == 0.0, (
                "the second trace returned a different value"
            )
        finally:
            del pe_layer
            keras.backend.clear_session()
            gc.collect()

    def test_prompt_encoder_dense_pe_survives_trace_then_eager(self):
        """
        PROOF 2: the same sequence through the real ``get_dense_pe`` path, which
        is what ``SAM.call`` and therefore any trainer actually reaches.
        """
        encoder = self._encoder()
        try:
            traced = tf.function(lambda: encoder.get_dense_pe())
            traced_value = keras.ops.convert_to_numpy(traced()).copy()
            eager_value = keras.ops.convert_to_numpy(encoder.get_dense_pe())
            assert tuple(eager_value.shape) == (1, GRID_SIZE, GRID_SIZE, OUT_CHANS)
            assert float(np.max(np.abs(traced_value - eager_value))) == 0.0, (
                "get_dense_pe disagreed between a traced and an eager call"
            )
        finally:
            del encoder
            keras.backend.clear_session()
            gc.collect()

    def test_sam_forward_survives_trace_then_eager(self):
        """
        PROOF 3: end to end on the real reduced-SAM fixture, which is the object
        the reviewer poisoned. ``SAM.call`` is traced once and then called
        eagerly; with the cache installed the eager call raised inside
        ``PositionEmbeddingRandom.call()``.

        Only ``low_res_logits`` is compared: it is the differentiable output and
        the one every other round-trip guard in this file uses, so a drift here
        is directly comparable to those.

        Two deliberate deviations from the obvious spelling, both forced and both
        measured -- neither is a convenience:

        1. ``model.call(...)`` is traced, not ``model(...)``. Keras\' ``__call__``
           refuses a Python ``int`` inside a positional input dict
           (``ValueError: Only input tensors may be passed as positional
           arguments``), and deviation 2 needs one.
        2. ``original_size`` is a PYTHON tuple, not a tensor. With a tensor,
           ``SAM.call`` does not trace AT ALL: ``postprocess_masks`` raises
           ``OperatorNotAllowedInGraphError: Iterating over a symbolic tf.Tensor
           is not allowed`` at ``model.py:616``. That is a SEPARATE,
           PRE-EXISTING limitation of ``SAM.call`` under ``tf.function``,
           declared in ``verification.md`` § Not Verified; it long predates the
           F-16 cache and it is not what this test is about.

        A THIRD constraint, and the one that decides whether this test is
        discriminating at all: the model must be FRESH, so that the very first
        call it ever receives is the traced one. Reusing the ``seeded_sam``
        fixture makes the test vacuous --- the fixture already forward-passes
        eagerly, so a cache would already hold a valid EagerTensor and the trace
        would happily reuse it. Measured: against the cached code the fixture
        spelling PASSED, and the fresh spelling below FAILS. Trace-first is also
        the realistic order, because it is what ``fit()`` does to a new model.
        For that reason this test must NOT take the ``seeded_sam`` fixture at
        all --- an earlier spelling requested it only to ``del`` it on the first
        line, which still built and retained the module-scoped model.

        With the step-11 cache installed this test fails with the ``out of
        scope`` ``TypeError`` raised from inside
        ``PositionEmbeddingRandom.call()``.
        """
        model = build_reduced_sam()
        try:
            inputs = dict(sam_inputs())
            inputs["original_size"] = (IMG_SIZE, IMG_SIZE)
            traced = tf.function(lambda: model.call(inputs)["low_res_logits"])
            traced_value = keras.ops.convert_to_numpy(traced()).copy()
            eager_value = keras.ops.convert_to_numpy(
                model.call(inputs)["low_res_logits"]
            )
            assert traced_value.shape == eager_value.shape
            assert np.all(np.isfinite(eager_value)), (
                "the eager forward after a trace returned non-finite logits"
            )
        finally:
            del model
            keras.backend.clear_session()
            gc.collect()

    def test_grid_sizes_do_not_contaminate_each_other(self):
        """
        PROOF 4 (retained from the cache suite, now a plain correctness pin):
        each requested size gets its own grid, including a non-square one.
        """
        encoder = self._encoder()
        try:
            pe_layer = encoder.pe_layer
            g16 = keras.ops.convert_to_numpy(
                pe_layer._coord_grid(GRID_SIZE, GRID_SIZE)
            ).copy()
            g8 = keras.ops.convert_to_numpy(pe_layer._coord_grid(8, 8)).copy()
            assert g16.shape == (GRID_SIZE, GRID_SIZE, 2)
            assert g8.shape == (8, 8, 2), (
                f"a second size came back with shape {g8.shape}"
            )
            back = keras.ops.convert_to_numpy(
                pe_layer._coord_grid(GRID_SIZE, GRID_SIZE)
            )
            assert float(np.max(np.abs(back - g16))) == 0.0, (
                "returning to the first size did not reproduce its grid"
            )
            assert tuple(
                keras.ops.convert_to_numpy(pe_layer._coord_grid(8, 16)).shape
            ) == (8, 16, 2)
        finally:
            del encoder
            keras.backend.clear_session()
            gc.collect()

    def test_dense_pe_tracks_a_weight_change(self):
        """
        Retained from the cache suite --- the guard that refuses a cache of the
        ENCODED positional encoding.

        ``get_dense_pe`` must recompute from the CURRENT
        ``positional_encoding_gaussian_matrix``. A cache of the encoded PE would
        freeze the value produced by ``build_from_config``'s dummy forward,
        which runs BEFORE the saved weights are restored --- measured at up to
        1.9986 away from the correct value, silently and forever.
        """
        encoder = self._encoder()
        try:
            matrix = encoder.pe_layer.positional_encoding_gaussian_matrix
            assert not matrix.trainable, (
                "premise check: this weight is non-trainable, which is exactly "
                "why a stale cache of it would never be corrected by training"
            )
            before = keras.ops.convert_to_numpy(encoder.get_dense_pe()).copy()
            matrix.assign(
                np.random.RandomState(0)
                .normal(size=tuple(matrix.shape))
                .astype("float32")
            )
            after = keras.ops.convert_to_numpy(encoder.get_dense_pe())
            assert float(np.max(np.abs(before - after))) > 0.0, (
                "get_dense_pe did not follow its own weight -- the encoded "
                "positional encoding is being cached, which silently freezes "
                "every loaded model at its pre-restore value"
            )
        finally:
            del encoder
            keras.backend.clear_session()
            gc.collect()


class TestRealVariantForwardPass:
    """
    F-R5 --- one REAL variant is actually forward-passed, at reference geometry.

    Before this, `from_variant('vit_b'/'vit_l'/'vit_h')` were constructed and
    attribute-asserted only, so the reference geometry -- a 64x64 token grid,
    `window_size=14` (which does NOT divide 64, so window partition really pads),
    `global_attn_indexes=(2, 5, 8, 11)`, `use_rel_pos=True`, `img_size=1024` --
    was executed by ZERO tests. Everything else in this file runs at a 16x16
    grid with 2 global blocks.

    `vit_b` only is forward-passed. `vit_l` (308M) and `vit_h` (637M) are
    constructed and counted below but NOT forward-passed here, and the
    parameter-count test does not forward-pass ANY variant: a 1024x1024 forward
    through 32 blocks holds a 4096x4096x16 global-attention matrix, measured at
    a **6,754.5 MiB** peak on the 10,160 MiB GPU 1, which made the ordinary
    gate un-runnable whenever the card was already in use. That residual
    coverage gap -- no `vit_l`/`vit_h` forward anywhere -- is declared in
    `verification.md` § Not Verified.
    """

    def test_vit_b_forward_at_reference_geometry(self):
        """
        Construct `vit_b`, assert the reference geometry, then RUN it.

        Measured cost: ~0.5 s to construct and ~4 s for the forward on GPU 1 at
        batch 1, which is why it is an ordinary gate test rather than a
        `slow`-marked one.
        """
        model = SAM.from_variant("vit_b")
        try:
            encoder = model.image_encoder
            assert encoder.img_size == 1024
            assert encoder.patch_size == 16
            assert encoder.grid_size == 64
            assert encoder.use_rel_pos is True
            assert tuple(encoder.global_attn_indexes) == (2, 5, 8, 11)
            assert encoder.window_size == 14
            assert 64 % 14 != 0, (
                "premise check: window_size=14 must NOT divide the 64x64 grid, "
                "otherwise this test does not exercise window-partition padding"
            )
            window_sizes = [blk.window_size for blk in encoder.blocks]
            assert window_sizes == [
                0 if i in (2, 5, 8, 11) else 14 for i in range(12)
            ], f"block window sizes are {window_sizes}"

            inputs = {
                "image": keras.ops.convert_to_tensor(
                    np.random.RandomState(0)
                    .uniform(0.0, 255.0, size=(1, 1024, 1024, 3))
                    .astype("float32")
                ),
                "points": (
                    keras.ops.convert_to_tensor([[[500.0, 500.0]]]),
                    keras.ops.convert_to_tensor([[1]]),
                ),
                "original_size": keras.ops.convert_to_tensor((1024, 1024)),
            }
            outputs = model(inputs)

            assert tuple(outputs["masks"].shape) == (1, 3, 1024, 1024)
            assert tuple(outputs["low_res_logits"].shape) == (1, 3, 256, 256)
            assert tuple(outputs["iou_predictions"].shape) == (1, 3)
            logits = keras.ops.convert_to_numpy(outputs["low_res_logits"])
            assert np.all(np.isfinite(logits)), (
                "vit_b produced non-finite low_res_logits at reference geometry"
            )
            assert float(np.std(logits)) > 0.0, (
                "vit_b's low_res_logits are constant -- a collapsed forward "
                "pass would satisfy every shape assertion above"
            )
        finally:
            del model
            keras.backend.clear_session()
            gc.collect()

    # DECISION plan-2026-08-03T191222-1d751f81/D-027: count parameters by
    # BUILDING, never by forward-passing, and do it under keras.device("cpu").
    # Do NOT replace the explicit sub-layer builds below with the obvious
    # `model.image_encoder(zeros((1, 1024, 1024, 3)))` -- that spelling
    # measured a 6,754.5 MiB GPU peak for vit_h on a 10,160 MiB card and made
    # the ordinary gate hard-OOM (unskippable) under any concurrent GPU work.
    # And do NOT collapse the sub-layer builds into a bare
    # `encoder.build(shape)`: ImageEncoderViT.build creates only `pos_embed`,
    # so that alone counts 5,242,880 of vit_h's 637,026,048. See decisions.md
    # D-027.
    @pytest.mark.parametrize(
        "variant,encoder_params",
        [
            ("vit_b", 89_670_912),
            ("vit_l", 308_278_272),
            ("vit_h", 637_026_048),
        ],
    )
    def test_variant_parameter_counts_match_the_readme(
        self, variant: str, encoder_params: int
    ):
        """
        F-R4 --- pin the documented per-variant counts to a MEASUREMENT.

        The per-variant table's single home is the ``Model Variants:`` block of
        ``models/SAM/SAM1/model.py``'s module docstring; ``SAM1/README.md`` §7
        points at it rather than restating it. That table previously quoted
        reference-PyTorch figures that two of this iteration's own layout
        changes had falsified (the mask decoder was listed at 3,143,424 against
        a real 4,058,340). Those numbers now come from this measurement, and
        this test is what keeps them true: any further layout change fails HERE
        with the new number in the message.

        Parameter counts are exact integers and are process-independent, unlike
        the forward-pass magnitudes carried surprise #9 forbids asserting.

        NO FORWARD PASS, AND NO GPU. An earlier spelling of this test called
        `image_encoder(zeros((1, 1024, 1024, 3)))`, which for `vit_h` allocates
        the 4096x4096x16 global-attention matrix: measured peak **6,754.5 MiB**
        on the 10,160 MiB GPU 1, i.e. the ordinary gate hard-OOMed (it cannot
        skip) whenever anything else held more than ~3.3 GB of that card --
        which is this machine's normal working condition. Counting parameters
        never needed the forward. `ImageEncoderViT.build` creates only
        `pos_embed` (its sub-layers are built lazily on first call), so the
        sub-layers are built EXPLICITLY below at the geometry the forward would
        have given them, and the whole thing runs inside `keras.device("cpu")`.
        Measured after the change: all three variants exact, peak GPU **0.0
        MiB**. Do NOT "simplify" this back to a forward pass, and do NOT drop
        the explicit sub-layer builds -- `build()` alone counts 5,242,880 of
        `vit_h`'s 637,026,048. See decisions.md D-027.
        """
        with keras.device("cpu"):
            model = SAM.from_variant(variant)
            try:
                encoder = model.image_encoder
                image_shape = (1, encoder.img_size, encoder.img_size, encoder.in_chans)
                token_shape = (
                    1, encoder.grid_size, encoder.grid_size, encoder.embed_dim,
                )
                encoder.build(image_shape)
                encoder.patch_embed.build(image_shape)
                for block in encoder.blocks:
                    block.build(token_shape)
                encoder.neck.build(token_shape)
                measured = int(encoder.count_params())
                assert measured == encoder_params, (
                    f"{variant} image encoder measures {measured:,} params, "
                    f"model.py's Model Variants: table says "
                    f"{encoder_params:,} -- update that table from this "
                    f"measurement"
                )
            finally:
                del model
                keras.backend.clear_session()
                gc.collect()

    def test_prompt_encoder_and_mask_decoder_are_variant_independent(self):
        """
        The other half of the ``Model Variants:`` table in
        ``models/SAM/SAM1/model.py``'s module docstring: 6,476 and 4,058,340 for
        every one of the three variants that table lists, which is what makes
        its per-variant totals add up.

        4,058,340 is the number the reviewer measured against the stale
        3,143,424 that table used to carry; it is pinned here so it cannot rot
        again.
        """
        model = SAM.from_variant("vit_b")
        try:
            feat = keras.ops.convert_to_tensor(
                np.zeros((1, 64, 64, 256), dtype="float32")
            )
            sparse, dense = model.prompt_encoder(
                points=(
                    keras.ops.convert_to_tensor([[[500.0, 500.0]]]),
                    keras.ops.convert_to_tensor([[1]]),
                )
            )
            model.prompt_encoder(masks=keras.random.normal((1, 1, 256, 256)))
            model.mask_decoder(
                image_embeddings=feat,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=True,
            )
            assert int(model.prompt_encoder.count_params()) == 6_476, (
                f"prompt encoder measures "
                f"{int(model.prompt_encoder.count_params()):,} params, "
                f"model.py's Model Variants: table says 6,476"
            )
            assert int(model.mask_decoder.count_params()) == 4_058_340, (
                f"mask decoder measures "
                f"{int(model.mask_decoder.count_params()):,} params, "
                f"model.py's Model Variants: table says 4,058,340"
            )
        finally:
            del model
            keras.backend.clear_session()
            gc.collect()
