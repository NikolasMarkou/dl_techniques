"""The four study arms do NOT split into "one attention arm, three fixed spans".

`RESULTS.md` explained the study's settings asymmetry with "a convolution has a
fixed span". That is true of `ascii_clifford_bert` (49) and `ascii_convnext_bert`
(25) and **false of `ascii_convnext_v2_bert`**, whose
:class:`GlobalResponseNormalization` computes its per-channel score by reducing
over axes ``1 .. N-2``. `ConvNextEncoderBlock` lifts the sequence to
``(B, 1, L, D)`` before calling the block, so that reduction runs over the
**sequence axis**: GRN is a mask-unaware, sequence-global operator.

Two measured consequences, both of which contradicted a stated fact in
`RESULTS.md` when this guard was written (2026-08-30):

1. *"both conv arms measure exactly 0.000 beyond their spans."* Perturbing
   position 0 of a trained encoder and reading ``max |delta|`` at position ``d``,
   `ascii_convnext_v2_bert` gave **7.210e-02 at d=25**, 8.378e-02 at d=60 and
   3.989e-02 at d=120 -- above the transformer's own magnitudes at the same
   distances.
2. *"encoding the same text padded to 128, 256 and 512 gives identical
   embeddings."* True for the other three arms (max |delta| 1.8e-07 to 3.6e-07);
   `ascii_convnext_v2_bert` moved **1.663e-01**, cosine 0.9863.

Consequence 2 is immaterial to the study's numbers, because length-sorted
batching holds evaluation pad fractions at 1-3% where the distortion is under
0.001 in cosine. It is pinned anyway, because the reason it is harmless is a
property of the EVALUATION and not of the arm.

**This guard pins the difference in both directions on purpose.** Asserting only
that the two span-bounded arms are bounded would stay green if GRN were later
made mask-aware, silently invalidating the note in `RESULTS.md`; asserting only
that convnext_v2 is unbounded would stay green if every arm became global. If
either arm changes class, this file must fail and the prose must be revisited.
"""

from typing import Tuple

import numpy as np
import pytest

import keras

from train.embeddings_experimental.config import ExperimentConfig, build_model

SEQ_LEN = 128

#: Arms whose reach is bounded by their convolutional span, with that span.
SPAN_BOUNDED = {"ascii_clifford_bert": 49, "ascii_convnext_bert": 25}

#: The arm carrying an always-on global branch (GRN).
GLOBAL_ARM = "ascii_convnext_v2_bert"

#: Distances at which a span-bounded arm must be exactly inert. The largest
#: span above is 49, whose reach is 24 positions either side of a perturbation.
FAR = (60, 100)

#: A global operator must move the output by at least this much at FAR
#: distances. Measured on a freshly built convnext_v2 the movement is orders of
#: magnitude above it; the floor only has to exclude float noise.
MIN_GLOBAL_REACH = 1e-6


def _build(model: str) -> keras.Model:
    keras.utils.set_random_seed(0)
    encoder = build_model(
        ExperimentConfig(model=model, variant="tiny", max_seq_length=SEQ_LEN)
    )
    encoder.build((None, SEQ_LEN))
    return encoder


def _reach(encoder: keras.Model, distance: int) -> float:
    """Return ``max |delta|`` at ``distance`` when only position 0 changes."""
    rng = np.random.default_rng(0)
    ids = rng.integers(6, 101, size=(1, SEQ_LEN)).astype(np.int32)
    mask = np.ones((1, SEQ_LEN), dtype=np.int32)

    def run(x: np.ndarray) -> np.ndarray:
        out = encoder({"input_ids": x, "attention_mask": mask}, training=False)
        return np.asarray(
            keras.ops.convert_to_numpy(out["last_hidden_state"]), dtype=np.float64
        )[0]

    before = run(ids)
    moved = ids.copy()
    moved[0, 0] = (int(moved[0, 0]) - 6 + 50) % 95 + 6
    after = run(moved)
    return float(np.abs(after[distance] - before[distance]).max())


def _pad_pair(encoder: keras.Model, real: int, wide: int) -> Tuple[float, float]:
    """Embed the same real content at two pad widths; return (max |delta|, cosine)."""
    rng = np.random.default_rng(1)
    content = rng.integers(6, 101, size=real).astype(np.int32)

    def run(width: int) -> np.ndarray:
        ids = np.zeros((1, width), dtype=np.int32)
        ids[0, :real] = content
        mask = (ids != 0).astype(np.int32)
        out = encoder({"input_ids": ids, "attention_mask": mask}, training=False)
        return np.asarray(
            keras.ops.convert_to_numpy(out["pooled_output"]), dtype=np.float64
        )[0]

    a, b = run(real), run(wide)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return float(np.abs(a - b).max()), cos


@pytest.mark.parametrize("model", sorted(SPAN_BOUNDED))
@pytest.mark.parametrize("distance", FAR)
def test_the_span_bounded_arms_reach_exactly_nothing(model: str, distance: int) -> None:
    """`clifford` and `convnext` must be EXACTLY inert beyond their span."""
    reach = _reach(_build(model), distance)
    assert reach == 0.0, (
        f"{model} moved position {distance} by {reach:.3e} when only position 0 "
        f"changed, but its span is {SPAN_BOUNDED[model]} so this must be exactly "
        f"0.0. Either the span formula or the block's mixing has changed; "
        f"RESULTS.md's 'a convolution has a fixed span' reasoning depends on it."
    )


@pytest.mark.parametrize("distance", FAR)
def test_the_grn_arm_is_not_span_bounded(distance: int) -> None:
    """Anti-vacuity, and the finding: `convnext_v2` reaches everywhere.

    If this ever goes green-by-becoming-zero, GRN has been made mask- or
    sequence-aware and the notes in `RESULTS.md` and both READMEs that call this
    arm globally-mixing are stale.
    """
    reach = _reach(_build(GLOBAL_ARM), distance)
    assert reach > MIN_GLOBAL_REACH, (
        f"{GLOBAL_ARM} moved position {distance} by only {reach:.3e}. This arm "
        f"is supposed to carry an always-on global branch (GRN reduces over the "
        f"sequence axis), so it must NOT be span-bounded. If GRN became "
        f"sequence-aware, update RESULTS.md -- several notes depend on this."
    )


@pytest.mark.parametrize("model", sorted(SPAN_BOUNDED) + ["ascii_bert"])
def test_pad_width_is_inert_for_every_arm_without_a_global_branch(model: str) -> None:
    """Three arms give the same embedding at any pad width."""
    delta, cos = _pad_pair(_build(model), real=40, wide=SEQ_LEN)
    assert delta < 1e-5, (
        f"{model} embedding moved {delta:.3e} (cosine {cos:.6f}) when only the "
        f"PAD WIDTH changed. RESULTS.md's padding refutation depends on this "
        f"being inert for every arm but {GLOBAL_ARM}."
    )


def test_pad_width_is_NOT_inert_for_the_grn_arm() -> None:
    """The counterpart: `convnext_v2`'s embedding depends on the pad width.

    Harmless in this study only because `embed_texts` sorts by length, holding
    the evaluation's pad fractions at 1-3%. That is a property of the evaluation,
    not of the arm, so it is pinned here rather than assumed.
    """
    delta, cos = _pad_pair(_build(GLOBAL_ARM), real=40, wide=SEQ_LEN)
    assert delta > 1e-5, (
        f"{GLOBAL_ARM} embedding moved only {delta:.3e} (cosine {cos:.6f}) "
        f"across pad widths. If GRN became mask-aware this is good news, but "
        f"RESULTS.md's padding correction and the length-sorting rationale in "
        f"README.md both need rewriting."
    )
