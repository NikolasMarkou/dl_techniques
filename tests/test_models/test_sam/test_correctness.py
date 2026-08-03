"""
Correctness Instruments for SAM (`src/dl_techniques/models/sam/`)
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

from dl_techniques.models.sam.model import SAM
from dl_techniques.models.sam.image_encoder import (
    ImageEncoderViT,
    WindowedAttentionWithRelPos,
)
from dl_techniques.models.sam.prompt_encoder import PromptEncoder
from dl_techniques.models.sam.mask_decoder import MaskDecoder
from dl_techniques.models.sam.transformer import (
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
        import dl_techniques.models.sam.image_encoder as image_encoder_module

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

    def test_mask_decoder_activation_default_is_relu(self):
        """
        F-6 -- the default is ``'relu'``, matching ``TwoWayTransformer``.

        RED before the fix: ``'gelu'`` (probe P12), so the two halves of the
        decoder disagreed with each other and with the paper.
        """
        decoder = MaskDecoder(
            transformer_dim=OUT_CHANS,
            transformer=TwoWayTransformer(
                depth=1, embedding_dim=OUT_CHANS, num_heads=2, mlp_dim=64
            ),
        )
        assert decoder.activation == "relu", (
            f"MaskDecoder.activation defaults to {decoder.activation!r}; "
            f"reference SAM's MLP hardcodes ReLU and TwoWayTransformer already "
            f"defaults to 'relu'"
        )
        assert decoder.transformer.activation == decoder.activation, (
            f"the two halves of the decoder still disagree: transformer "
            f"{decoder.transformer.activation!r} vs decoder "
            f"{decoder.activation!r}"
        )
        del decoder
        keras.backend.clear_session()
        gc.collect()

    @pytest.mark.parametrize("depth", [2, 3])
    @pytest.mark.parametrize("activation", ["relu", "gelu"])
    def test_depth_and_activation_round_trip(self, depth: int, activation: str):
        """
        Both knobs survive ``get_config``/``from_config`` -- and stay LIVE.

        The restored object's ACTUAL ``Dense`` count and ACTUAL hidden
        activation are asserted, not the config keys: a key that round-trips
        while the rebuilt layer ignores it is precisely the dead knob this step
        removes.
        """
        decoder = _build_decoder(iou_head_depth=depth, activation=activation)
        config = decoder.get_config()
        assert config["iou_head_depth"] == depth, (
            f"get_config() dropped iou_head_depth: {sorted(config)}"
        )
        assert config["activation"] == activation, (
            f"get_config() dropped activation: {sorted(config)}"
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
        assert restored_act == activation, (
            f"restored IoU head hidden activation is {restored_act!r}, "
            f"expected {activation!r}"
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
