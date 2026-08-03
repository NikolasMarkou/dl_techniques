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
from dl_techniques.models.sam.transformer import TwoWayTransformer

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

#: Weight count of the default fixture, measured at commit 004d431d.
#: Plan steps 3 and 4 change the checkpoint layout; this is their control.
BASELINE_FIXTURE_WEIGHT_COUNT = 192

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
