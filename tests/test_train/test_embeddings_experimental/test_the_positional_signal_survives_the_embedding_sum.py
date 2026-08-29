"""The study's encoders must carry a positional signal the model cannot ignore.

This guard exists because of a measured failure, not a style preference. With
the encoder's own ``position_embedding_type='learned'`` default, the transformer
arm converged to a **bag of characters**: on a trained model, reordering the
entire context (identical multiset) moved position 0 by 0.83% of activation
scale while replacing it moved 52.58%, a 63x ratio. It scored the
unigram-plus-copy solution almost exactly -- 2.8318 measured against 2.8022
predicted -- and stopped improving at -0.008 nats per 1000 steps. Switching this
one field to ``'sinusoidal'`` bought 0.85 nats at 256 context.

**The obvious oracle does not work here, which is why this file measures
something else.** That reorder-versus-replace probe is decisive on a *trained*
model and blind at initialization: measured on freshly built encoders it gives
0.0806 for learned against 0.0925 for sinusoidal, which discriminates nothing.
The failure is a training-dynamics failure -- the learned table starts at
essentially the word table's norm (0.1985 against 0.1987) and is then actively
abandoned, shrinking to 0.1612 over 3000 steps while the word table grows to
0.3283. No forward pass at step 0 can see that.

So the quantity pinned here is the one that actually differs at step 0: how much
of the embedding output is decided by *where* a token sits rather than *which*
token it is. Feeding a constant token id at every position makes all variation
across rows positional by construction, and swapping the id with positions held
fixed isolates the token signal. Measured at build time, ``tiny`` at 512:

===========  ==================  ==================  ======
type         positional spread   token signal        ratio
===========  ==================  ==================  ======
learned                  7.9264             11.0659    0.72
sinusoidal               9.2960              0.4152   22.39
===========  ==================  ==================  ======

A 31x separation, so the threshold below is nowhere near either value.

References:
    - Vaswani et al., 2017. Attention Is All You Need.
      (https://arxiv.org/abs/1706.03762)
"""

import numpy as np
import pytest

import keras

from train.embeddings_experimental.config import (
    MODEL_REGISTRY,
    ExperimentConfig,
    build_model,
)

# The measured values are 0.72 (learned) and 22.39 (sinusoidal). Anything in
# the wide gap between them separates the two; 5.0 is far from both.
MIN_POSITIONAL_DOMINANCE = 5.0

SEQ_LEN = 512
PROBE_TOKEN_A = 42
PROBE_TOKEN_B = 43


def positional_dominance(encoder: keras.Model, seq_len: int = SEQ_LEN) -> float:
    """Return positional signal divided by token signal for one encoder.

    Every position is fed the SAME token id, so all variation across rows of
    the output is positional by construction. The token signal is then the
    movement caused by swapping that id with the positions held fixed.

    :param encoder: A built study encoder.
    :type encoder: keras.Model
    :param seq_len: Probe sequence length.
    :type seq_len: int
    :return: Positional spread divided by token-swap movement.
    :rtype: float
    """
    def run(token_id: int) -> np.ndarray:
        ids = keras.ops.convert_to_tensor(
            np.full((1, seq_len), token_id, dtype="int32")
        )
        out = encoder({"input_ids": ids}, training=False)["last_hidden_state"]
        return keras.ops.convert_to_numpy(out)[0]

    a = run(PROBE_TOKEN_A)
    b = run(PROBE_TOKEN_B)
    positional = float(np.sqrt(((a - a.mean(0)) ** 2).sum(-1)).mean())
    token = float(np.sqrt(((b - a) ** 2).sum(-1)).mean())
    assert token > 0.0, "token swap moved nothing; the probe is vacuous"
    return positional / token


def build(model: str, position_embedding_type: str) -> keras.Model:
    """Build one study arm at ``tiny`` with an explicit position type.

    :param model: A key of :data:`MODEL_REGISTRY`.
    :type model: str
    :param position_embedding_type: ``'learned'`` or ``'sinusoidal'``.
    :type position_embedding_type: str
    :return: The built encoder.
    :rtype: keras.Model
    """
    keras.utils.set_random_seed(0)
    encoder = build_model(
        ExperimentConfig(
            model=model,
            variant="tiny",
            max_seq_length=SEQ_LEN,
            position_embedding_type=position_embedding_type,
        )
    )
    encoder.build((None, SEQ_LEN))
    return encoder


def test_the_study_default_is_sinusoidal() -> None:
    """The default must be the measured-good value, not the encoder's own."""
    assert ExperimentConfig().position_embedding_type == "sinusoidal"


@pytest.mark.parametrize("model", sorted(MODEL_REGISTRY))
def test_every_arm_carries_a_dominant_positional_signal(model: str) -> None:
    """Under the study default, position must outweigh token identity."""
    dominance = positional_dominance(build(model, "sinusoidal"))
    assert dominance > MIN_POSITIONAL_DOMINANCE, (
        f"{model}: positional signal is only {dominance:.2f}x the token "
        f"signal, under the {MIN_POSITIONAL_DOMINANCE}x floor. An encoder "
        f"whose position signal is this weak converges to a bag of characters; "
        f"see this module's docstring."
    )


@pytest.mark.parametrize("model", sorted(MODEL_REGISTRY))
def test_the_learned_table_is_what_this_guard_rejects(model: str) -> None:
    """Anti-vacuity: the defect this file exists for must fail the threshold.

    Without this, the assertion above could pass for every configuration and
    pin nothing. ``'learned'`` is the encoder's own default and the value the
    study shipped with, so this is the real failure mode, not a synthetic one.
    """
    dominance = positional_dominance(build(model, "learned"))
    assert dominance < MIN_POSITIONAL_DOMINANCE, (
        f"{model}: 'learned' now scores {dominance:.2f}x, at or above the "
        f"{MIN_POSITIONAL_DOMINANCE}x floor. Either the embedding "
        f"initialization changed or the probe stopped measuring position -- "
        f"re-derive both numbers before moving the threshold."
    )
