"""Every BLOCK must honour a dropout rate, or the study is confounded.

The `embeddings_experimental` study exists to attribute a metric difference to
the sequence-mixing block. That attribution weakens if the blocks are not
equally regularized -- and for Runs 1 through 4 they were not:

- `ascii_bert`'s block carried attention-probability dropout **and** FFN dropout,
- both convnext blocks carried `dropout_rate`,
- `ascii_clifford_bert`'s block carried **none** -- `build_clifford_block` had no
  such parameter and `CliffordNetBlock` has none of its own.

The fix lives in `CliffordEncoderBlock`, not in `CliffordNetBlock`: that layer is
shared with other packages, so the study's wrapper applies dropout to the update
before the external residual add, the same position the ConvNeXt blocks use.

**This file probes the BLOCK, never the whole model, and that distinction is the
entire point.** A first version of this guard compared two `training=True`
forward passes of the assembled encoder and was **vacuous**: `BertEmbeddings`
applies its own dropout at `hidden_dropout_rate` to every arm
(`bert_embeddings.py:646`), so every model is stochastic in training mode whether
or not its block has any dropout at all. Re-injecting the real defect left that
version at 8 passed. Verify an injection actually reddens this file before
trusting it.

For the same reason, "the clifford arm trained unregularized" -- which earlier
revisions of RESULTS.md said -- is too strong. It had embedding dropout like
everyone else. What it lacked was *block* dropout.
"""

import numpy as np
import pytest

import keras

from dl_techniques.models.embeddings_experimental.shared.blocks import (
    BLOCK_REGISTRY,
    create_encoder_block,
)

HIDDEN = 32
SEQ_LEN = 16
BATCH = 4


def block_delta(block_type: str, dropout_rate: float) -> float:
    """Return the two-pass disagreement RELATIVE to the block's own update.

    The ratio, not the absolute difference, is the measurable quantity here.
    `CliffordEncoderBlock` defaults to `layer_scale_init=1e-5`, so its entire
    update is ~1e-5 at initialization and live dropout on it perturbs two passes
    by only ~1.4e-06 -- under any absolute threshold worth setting, and
    indistinguishable from dropout being absent. Gamma is *learned*, so that
    smallness is an artifact of step 0 and not a property of the block. The same
    trap is recorded for this family's padding hazard in
    `models/embeddings_experimental/README.md`.

    Dividing by the update magnitude (`output - input`, which every block in the
    registry exposes because each one returns a residual sum) makes the probe
    scale-free and comparable across blocks.

    :param block_type: A key of :data:`BLOCK_REGISTRY`.
    :type block_type: str
    :param dropout_rate: The block's dropout rate.
    :type dropout_rate: float
    :return: Mean |pass A - pass B| divided by mean |update|.
    :rtype: float
    """
    keras.utils.set_random_seed(0)
    kwargs = {"dropout_rate": dropout_rate}
    if block_type == "transformer":
        # This builder splits the two sites, so pin both or the attention
        # branch keeps firing and the 0.0 control is not a control.
        kwargs["attention_dropout_rate"] = dropout_rate
    block = create_encoder_block(
        block_type, hidden_size=HIDDEN, name="probe", **kwargs
    )
    shape = (BATCH, SEQ_LEN, HIDDEN)
    block.build(shape)
    x = keras.ops.convert_to_tensor(
        np.random.default_rng(0).normal(size=shape).astype("float32")
    )
    a = keras.ops.convert_to_numpy(block(x, training=True))
    b = keras.ops.convert_to_numpy(block(x, training=True))
    x_np = keras.ops.convert_to_numpy(x)
    update = float(np.abs(0.5 * (a + b) - x_np).mean())
    assert update > 0.0, f"block {block_type!r} is an exact identity; probe vacuous"
    return float(np.abs(a - b).mean()) / update


@pytest.mark.parametrize("block_type", sorted(BLOCK_REGISTRY))
def test_every_block_declares_a_dropout_rate(block_type: str) -> None:
    """`create_encoder_block` rejects a keyword its builder does not declare.

    So this call raising *is* the defect: it is exactly what happened for
    `clifford` before the wrapper gained the parameter.
    """
    create_encoder_block(
        block_type, hidden_size=HIDDEN, name="probe", dropout_rate=0.1
    )


@pytest.mark.parametrize("block_type", sorted(BLOCK_REGISTRY))
def test_every_block_honours_the_dropout_rate(block_type: str) -> None:
    """Declaring it is not enough -- it has to reach a live Dropout layer."""
    delta = block_delta(block_type, 0.1)
    assert delta > 0.01, (
        f"block {block_type!r}: two training-mode passes differ by "
        f"{delta:.2e} of the update at dropout_rate=0.1, so the parameter is "
        f"declared but "
        f"inert. The arm would train less regularized than the others while "
        f"appearing configured; see this module's docstring."
    )


@pytest.mark.parametrize("block_type", sorted(BLOCK_REGISTRY))
def test_the_dropout_rate_is_what_causes_it(block_type: str) -> None:
    """Anti-vacuity: at 0.0 the same two passes must agree exactly.

    Without this, the assertion above would pass for any nondeterminism -- a
    stochastic-depth default, an unseeded initializer -- and would not pin
    dropout.
    """
    delta = block_delta(block_type, 0.0)
    assert delta == 0.0, (
        f"block {block_type!r}: two training-mode passes differ by "
        f"{delta:.2e} of the update at dropout_rate=0.0. Something other than "
        f"dropout is "
        f"stochastic, so the companion test does not pin what it claims to."
    )
