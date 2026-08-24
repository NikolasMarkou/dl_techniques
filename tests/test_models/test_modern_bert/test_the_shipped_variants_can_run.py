"""Every entry in ``ModernBERT.MODEL_VARIANTS`` must survive a forward pass.

Two of the three shipped variants did not. MEASURED 2026-08-21 on a 12 GB
RTX 4070, at a sequence length of **eight**::

    ModernBERT.from_variant("base")   -> ResourceExhaustedError
    ModernBERT.from_variant("large")  -> ResourceExhaustedError
        Exception encountered when calling SingleWindowAttention.call()

``tiny`` ran (16,305,160 params, 46 weight tensors, 2 relative-position-bias
tables). ``base`` and ``large`` could not produce a single output tensor at any
admissible length, on either GPU in this machine.

The mechanism is the one ``test_shipped_window_size.py`` already pins, taken to
its conclusion. A ``window`` local layer folds ``(B, L, D)`` into a synthetic
``ceil(sqrt(L))``-square grid and pads that grid up to a multiple of
``window_size``; when ``L <= window_size**2`` the result is ONE window holding
``window_size**2`` slots. At the ``window_size=128`` those two variants shipped,
that is 16384 slots, so each of their 15 (``base``) / 19 (``large``) local
layers built a ``16384 x 16384`` score matrix per head **independent of L** --
12.9 GB per layer in float32 at 12 heads. It is not an inefficiency that shows
up at long sequences; it is a fixed cost paid at every length, including 8.

The repair is per-variant configuration, not new code: ``base`` and ``large``
now ship ``global_attention_interval = 1``, so ``is_global`` is true at every
layer and the ``'window'`` branch is never selected for them. That preserves
the all-to-all connectivity their padded single window already had, and adds
the RoPE the local layers did not receive.

Two alternatives were rejected and must not be reintroduced (see the D-019 /
D-027 / D-135 anchors in ``model.py``):

* shrinking ``local_attention_window_size`` until windowing engages -- the
  window is a SPATIAL neighbourhood over a synthetic square grid, so a smaller
  window buys a strided, non-contiguous token adjacency that is not the paper's
  1-D window: a correctness change traded for the cost fix;
* deleting the knobs -- ``tiny`` genuinely partitions above 4096 tokens and
  would lose a real 4x saving.

``test_a_local_layer_must_be_able_to_partition`` is the general rule behind the
specific failure, and is the arm that stays meaningful if the variant table
grows: a ``window`` layer whose ``window_size**2`` exceeds
``max_position_embeddings`` can never partition at ANY admissible length, so it
is dense attention wearing a window's price tag.

See decisions.md D-135 (plan-2026-08-19T163559-499b6f0e).
"""

import gc
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import keras
import numpy as np
import pytest

from dl_techniques.models.modern_bert.model import ModernBERT

SEQ_LEN = 8

VARIANTS = sorted(ModernBERT.MODEL_VARIANTS)


@pytest.fixture(autouse=True)
def release_the_variant_between_tests():
    """``large`` alone is 409,522,176 parameters (~1.6 GB in float32) and this
    module builds three variants twice over. Run as part of the whole
    ``test_modern_bert/`` directory -- 79 tests in ONE process -- that was
    enough to get the pytest process ``Killed`` (exit 137) by the OOM killer,
    with the directory otherwise green. Dropping the graph after each test is
    what keeps this file affordable in a shared session; do not remove it."""
    yield
    gc.collect()
    keras.backend.clear_session()
    gc.collect()


def _attention_schedule(variant: str) -> list:
    """Reproduce ``_build``'s own ``is_global`` expression, per layer."""
    config = ModernBERT.MODEL_VARIANTS[variant]
    interval = config["global_attention_interval"]
    return [
        "global" if (i + 1) % interval == 0 else "local"
        for i in range(config["num_layers"])
    ]


class TestEveryShippedVariantIsReachable:

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_a_forward_pass_completes_and_the_output_is_finite(self, variant):
        model = ModernBERT.from_variant(variant)
        inputs = {
            "input_ids": keras.ops.ones((1, SEQ_LEN), dtype="int32"),
            "attention_mask": keras.ops.ones((1, SEQ_LEN), dtype="int32"),
        }

        # Pre-fix, for `base` and `large`:
        #   ResourceExhaustedError from SingleWindowAttention.call
        outputs = model(inputs)

        hidden = keras.ops.convert_to_numpy(outputs["last_hidden_state"])
        expected = ModernBERT.MODEL_VARIANTS[variant]["hidden_size"]
        assert hidden.shape == (1, SEQ_LEN, expected), hidden.shape
        # A model that emits NaN has "run" without being usable, which is the
        # same practical outcome as not running at all.
        assert np.all(np.isfinite(hidden)), (
            f"{variant} produced non-finite activations at L={SEQ_LEN}"
        )


class TestNoVariantShipsAWindowThatCannotPartition:

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_a_local_layer_must_be_able_to_partition(self, variant):
        config = ModernBERT.MODEL_VARIANTS[variant]
        if "local" not in _attention_schedule(variant):
            pytest.skip(f"{variant} emits no local layers")

        window_slots = config["local_attention_window_size"] ** 2
        max_len = ModernBERT.DEFAULT_MAX_POSITION_EMBEDDINGS
        assert window_slots <= max_len, (
            f"{variant} ships local layers at window_size="
            f"{config['local_attention_window_size']}, so every window is "
            f"padded to {window_slots} slots while the longest admissible "
            f"sequence is {max_len}. The window can never partition: each "
            f"local layer is a {window_slots}x{window_slots} score matrix per "
            f"head at EVERY length, which is what made base/large unrunnable."
        )

    def test_tiny_still_partitions_and_keeps_its_local_layers(self):
        """The control on the repair's scope. ``base``/``large`` were changed;
        ``tiny`` must not have been swept along, because its windowing is real
        (4 windows for 4097 <= L <= 8192) and deleting it would remove a
        capability rather than a no-op."""
        schedule = _attention_schedule("tiny")
        assert schedule.count("local") > 0, schedule
        assert ModernBERT.MODEL_VARIANTS["tiny"][
                   "local_attention_window_size"] == 64
        assert ModernBERT.MODEL_VARIANTS["tiny"][
                   "global_attention_interval"] == 2

    @pytest.mark.parametrize("variant", ["base", "large"])
    def test_the_hybrid_schedule_is_still_reachable_by_override(self, variant):
        """``local_attention_window_size`` is retained rather than deleted, so
        it must not become a dead knob: the hybrid schedule stays one keyword
        away. Configuration only -- no forward pass, because that
        configuration is precisely the one that cannot run here."""
        model = ModernBERT.from_variant(variant, global_attention_interval=3)
        assert model.global_attention_interval == 3
        assert model.local_attention_window_size == 128
