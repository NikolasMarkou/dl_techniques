"""Every convolutional arm in the study must be EXACTLY span-bounded.

This is the invariant that the removal of `ascii_convnext_v2_bert` established,
and the reason the arm was removed. `RESULTS.md` explains the study's settings
asymmetry with "a convolution has a fixed span, which is why it shows no such
interaction". That was true of `ascii_clifford_bert` (span 49) and
`ascii_convnext_bert` (span 25) and **false of the V2 arm**, whose
:class:`GlobalResponseNormalization` scores each channel by its L2 magnitude
over axes ``1 .. N-2``. `ConvNextEncoderBlock` lifts the sequence to
``(B, 1, L, D)`` before calling the block, so that reduction ran over the
**sequence axis**: GRN is a mask-unaware, sequence-global operator.

Measured on trained encoders (2026-08-30) by perturbing position 0 and reading
``max |delta|`` at position ``d`` -- the measurement that decided the removal:

    arm                     span   d=25       d=60       d=120
    ascii_bert              full   4.188e-02  2.661e-02  1.696e-02
    ascii_clifford_bert     49     0.000      0.000      0.000
    ascii_convnext_bert     25     0.000      0.000      0.000
    ascii_convnext_v2_bert  25     7.210e-02  8.378e-02  3.989e-02   <- withdrawn

The V2 arm moved position 60 by MORE than the attention arm did. A study whose
arms are meant to differ only in the sequence-mixing block, with the
convolutional arms sharing a fixed span, cannot contain it. Same cause made it
the only arm whose pooled embedding depended on the PAD WIDTH (1.663e-01 across
widths 128/256/512, cosine 0.9863) rather than only on real length.

**The guard is two-sided on purpose.** Asserting only that the convolutional
arms are bounded would pass trivially if the probe stopped detecting global
mixing at all; `ascii_bert`, whose attention genuinely is global, is the
positive control that keeps the probe honest. If a future arm reaches beyond its
span, this file fails and the "fixed span" reasoning in `RESULTS.md` and both
READMEs has to be revisited rather than silently invalidated.
"""

from typing import Tuple

import numpy as np
import pytest

import keras

from train.embeddings_experimental.config import (
    MODEL_REGISTRY,
    ExperimentConfig,
    build_model,
)

SEQ_LEN = 128

#: Study arms whose reach is bounded by a convolutional span, with that span.
#: The span is the full diameter, so a perturbation reaches half of it either way.
SPAN_BOUNDED = {"ascii_clifford_bert": 49, "ascii_convnext_bert": 25}

#: The arm that is global BY DESIGN, and so acts as this probe's positive control.
GLOBAL_BY_DESIGN = "ascii_bert"

#: Distances at which a span-bounded arm must be exactly inert. The largest span
#: is 49, which reaches 24 positions either side.
FAR = (60, 100)

#: A globally-mixing arm must move the output by at least this much at FAR
#: distances. The floor only has to exclude float noise.
MIN_GLOBAL_REACH = 1e-6


def test_the_registry_holds_exactly_the_arms_this_file_classifies() -> None:
    """No study arm may go unclassified, or the coverage below is a fiction.

    Without this, adding a fourth arm would leave it untested by every
    assertion here while the file still reported green.
    """
    assert set(MODEL_REGISTRY) == set(SPAN_BOUNDED) | {GLOBAL_BY_DESIGN}, (
        f"MODEL_REGISTRY is {sorted(MODEL_REGISTRY)}, which this file does not "
        f"classify. Every arm must be declared either span-bounded (with its "
        f"span) or global by design, and then actually measured below."
    )


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
    return float(np.abs(run(moved)[distance] - before[distance]).max())


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
def test_every_convolutional_arm_reaches_exactly_nothing(
    model: str, distance: int
) -> None:
    """The invariant the V2 removal established: no global mixing anywhere."""
    reach = _reach(_build(model), distance)
    assert reach == 0.0, (
        f"{model} moved position {distance} by {reach:.3e} when only position 0 "
        f"changed, but its span is {SPAN_BOUNDED[model]} so this must be exactly "
        f"0.0. A convolutional arm that mixes globally is what got "
        f"ascii_convnext_v2_bert withdrawn from the study; see this module's "
        f"docstring before changing the threshold."
    )


@pytest.mark.parametrize("distance", FAR)
def test_the_attention_arm_does_reach_that_far(distance: int) -> None:
    """Positive control: the probe must be able to SEE global mixing.

    If this goes green-by-becoming-zero the probe is broken and the assertions
    above are vacuous, not satisfied.
    """
    reach = _reach(_build(GLOBAL_BY_DESIGN), distance)
    assert reach > MIN_GLOBAL_REACH, (
        f"{GLOBAL_BY_DESIGN} moved position {distance} by only {reach:.3e}. Its "
        f"attention is global, so the probe must detect it. Until it does, the "
        f"exactly-0.0 assertions above prove nothing."
    )


@pytest.mark.parametrize("model", sorted(MODEL_REGISTRY))
def test_pad_width_is_inert_for_every_arm(model: str) -> None:
    """No arm's embedding may depend on how far the batch was padded.

    This held for three arms and failed for `ascii_convnext_v2_bert` (1.663e-01,
    cosine 0.9863), by the same GRN reduction. Its being harmless in practice was
    a property of `embed_texts` sorting by length, not of the arm -- so it is
    pinned here rather than assumed.
    """
    delta, cos = _pad_pair(_build(model), real=40, wide=SEQ_LEN)
    assert delta < 1e-5, (
        f"{model} embedding moved {delta:.3e} (cosine {cos:.6f}) when only the "
        f"PAD WIDTH changed, with the real content held fixed. Mean pooling is "
        f"mask-aware, so this means some sub-layer reduces over the padded "
        f"sequence."
    )
