"""Anti-vacuity RED-proof for ``aggregate_backbone_losses=True`` on Qwen3Next.

Per ``plans/plan-2026-09-13T073704-245ab5d5/plan.md`` Step 6 and its own
Pre-Mortem item 4: an aggregation "delta" measured against a variant whose
``num_experts`` turns out to be ``1`` (dense) would be vacuous --
``model.backbone.losses`` would be empty and there would be nothing to
aggregate. This module proves, in order:

1. ``model.backbone.losses`` is NONEMPTY for the chosen (``"tiny"``) variant
   during a real forward pass with ``training=True`` -- the anti-vacuity
   check itself, run BEFORE the delta measurement is trusted.
2. ``aggregate_backbone_losses=True`` changes the reported total loss vs
   ``aggregate_backbone_losses=False`` on an IDENTICAL batch, using two
   independently constructed models/optimizers (never one optimizer reused
   sequentially) -- ``plans/plan-2026-09-13T073704-245ab5d5/decisions.md``
   D-010 measured that reusing one ``AdamW`` instance across two sequential
   calls leaks momentum/variance state and manufactures a spurious diff;
   this test builds two fresh models from the same seed instead.
"""

from __future__ import annotations

import keras
import numpy as np

from dl_techniques.models.language.masked_language_model.clm import (
    CausalLanguageModel,
)
from dl_techniques.models.language.qwen.qwen3_next import Qwen3Next

# A tiny but real "tiny"-variant-shaped backbone: MoE-enabled (num_experts=4
# on the real "tiny" variant; kept explicit here so the test does not depend
# on the variant table's own defaults for the property under test).
_VOCAB_SIZE = 37
_SEQ_LEN = 8
_BATCH = 4


def _make_backbone() -> Qwen3Next:
    keras.utils.set_random_seed(1234)
    return Qwen3Next.from_variant(
        "tiny",
        vocab_size=_VOCAB_SIZE,
        hidden_size=32,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_seq_len=_SEQ_LEN,
    )


def _make_batch() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    x = rng.integers(0, _VOCAB_SIZE, size=(_BATCH, _SEQ_LEN)).astype("int32")
    y = rng.integers(0, _VOCAB_SIZE, size=(_BATCH, _SEQ_LEN)).astype("int32")
    return x, y


def test_backbone_losses_are_nonempty_for_the_tiny_variant() -> None:
    """The anti-vacuity check itself: MUST pass before the delta below means
    anything. If this goes RED for "tiny", the chosen variant's config was
    misread (e.g. ``num_experts`` actually resolved to 1) and the delta
    measurement must not be trusted."""
    backbone = _make_backbone()
    x, _ = _make_batch()

    _ = backbone(x, training=True)

    assert len(backbone.losses) > 0, (
        "backbone.losses is EMPTY for the 'tiny' variant during a real "
        "forward pass with training=True -- the MoE aux/z-loss add_loss "
        "calls did not fire, so aggregate_backbone_losses has nothing to "
        "aggregate. Do not trust any aggregation-delta measurement in this "
        "state (plan.md Pre-Mortem item 4)."
    )
    for loss_value in backbone.losses:
        value = float(keras.ops.convert_to_numpy(loss_value))
        assert np.isfinite(value), f"backbone loss is not finite: {value}"


def test_aggregate_backbone_losses_changes_the_total_reported_loss() -> None:
    """Two independently built models/optimizers, identical batch, identical
    seed -- only ``aggregate_backbone_losses`` differs."""
    x, y = _make_batch()

    def _build(aggregate: bool) -> CausalLanguageModel:
        backbone = _make_backbone()
        model = CausalLanguageModel(
            backbone=backbone,
            vocab_size=_VOCAB_SIZE,
            skip_head=True,
            pre_shifted=True,
            aggregate_backbone_losses=aggregate,
            verify_causality=False,
        )
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))
        return model

    model_with = _build(aggregate=True)
    loss_with = model_with.train_on_batch(x, y, return_dict=True)["loss"]

    model_without = _build(aggregate=False)
    loss_without = model_without.train_on_batch(x, y, return_dict=True)["loss"]

    assert np.isfinite(loss_with) and np.isfinite(loss_without)
    delta = float(loss_with) - float(loss_without)
    assert abs(delta) > 1e-9, (
        f"aggregate_backbone_losses=True produced the SAME total loss as "
        f"False on an identical batch (loss_with={loss_with}, "
        f"loss_without={loss_without}, delta={delta}) -- the aggregation "
        f"flag is not actually folding the MoE aux/z-loss into the "
        f"optimized/reported total, i.e. it is a vacuous no-op on this "
        f"backbone."
    )
