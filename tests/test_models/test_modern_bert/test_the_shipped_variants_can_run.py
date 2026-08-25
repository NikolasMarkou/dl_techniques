"""Every entry in ``ModernBERT.MODEL_VARIANTS`` must survive a forward pass.

Two of the three shipped variants did not. MEASURED 2026-08-21 on a 12 GB
RTX 4070, at a sequence length of **eight**::

    ModernBERT.from_variant("base")   -> ResourceExhaustedError
    ModernBERT.from_variant("large")  -> ResourceExhaustedError
        Exception encountered when calling SingleWindowAttention.call()

``tiny`` ran. ``base`` and ``large`` could not produce a single output tensor at
any admissible length, on either GPU in this machine.

The mechanism: a ``window`` local layer folded ``(B, L, D)`` into a synthetic
``ceil(sqrt(L))``-square grid and padded that grid up to a multiple of
``window_size``; when ``L <= window_size**2`` the result was ONE window holding
``window_size**2`` slots. At the ``window_size=128`` those two variants ship,
that is 16384 slots, so each of their local layers built a ``16384 x 16384``
score matrix per head **independent of L** -- 12.9 GB per layer in float32 at
12 heads. Not an inefficiency that shows up at long sequences; a fixed cost paid
at every length, including 8.

**The 2026-08-21 repair has been REPLACED, 2026-08-25**
(plan-2026-08-25T053412-0f1fa04f). That repair was per-variant configuration:
``base``/``large`` shipped ``global_attention_interval = 1``, so no local layer
was ever built. D-135 recorded it as conditional and named its own release
condition -- *"the real fix is a 1-D sliding-window layer in
``layers/attention/``, which does not exist"*. It exists now
(``partition_mode='band'``, registry key ``'window_band'``), D-012 wired
ModernBERT's local layers to it, and the paper's ``global_attention_interval=3``
is restored on both variants. Re-measured on the same 12 GB RTX 4070, all
constructing AND forwarding: ``base`` L=8 0.692 GB / L=2048 1.640 GB GPU peak,
``large`` L=8 1.707 GB / L=2048 2.929 GB.

One alternative was rejected in 2026-08-21 and is STILL rejected: shrinking
``local_attention_window_size`` until windowing engages. Note that D-012 is not
that alternative even though the diff resembles it -- the stored value is
unchanged at 128 and only its UNIT moved (2-D edge length -> 1-D full span), so
the ``// 2`` at the call site is a span-to-half-width conversion. See the D-012
anchor in ``model.py``.

``test_a_local_layer_must_be_able_to_partition`` was the general rule behind the
specific failure -- a ``window`` layer whose ``window_size**2`` exceeds
``max_position_embeddings`` can never partition at any admissible length, so it
is dense attention wearing a window's price tag. That rule has no referent once
the local layers are 1-D bands: a band has no windows to partition into and pads
nothing. It is replaced below by the rule that DOES still bite.

See decisions.md D-135 (plan-2026-08-19T163559-499b6f0e) and D-012
(plan-2026-08-25T053412-0f1fa04f).
"""

import gc
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import keras
import numpy as np
import pytest

from dl_techniques.models.language.modern_bert.model import ModernBERT

SEQ_LEN = 8

VARIANTS = sorted(ModernBERT.MODEL_VARIANTS)


@pytest.fixture(autouse=True)
def release_the_variant_between_tests():
    """``large`` alone is 399,560,704 parameters (~1.5 GB in float32) and this
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


class TestNoVariantShipsALocalBandItCannotUse:
    """REPLACES ``TestNoVariantShipsAWindowThatCannotPartition``.

    Its rule was ``window_size**2 <= max_position_embeddings`` -- can this
    layer ever partition into more than one window? A 1-D band has no windows,
    so that question has no answer. The rule that still bites is about SPAN:
    a local band whose full span covers every admissible sequence is global
    attention wearing a local layer's name.
    """

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_a_local_band_must_be_narrower_than_the_longest_sequence(self, variant):
        config = ModernBERT.MODEL_VARIANTS[variant]
        if "local" not in _attention_schedule(variant):
            pytest.skip(f"{variant} emits no local layers")

        span = config["local_attention_window_size"]
        max_len = ModernBERT.DEFAULT_MAX_POSITION_EMBEDDINGS
        assert span < max_len, (
            f"{variant} ships local layers at a band span of {span} tokens "
            f"while the longest admissible sequence is {max_len}. Every query "
            f"would see every key, so the local layers are global attention "
            f"under another name and the hybrid schedule is decorative."
        )

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_the_layer_receives_half_the_configured_span(self, variant):
        """The unit conversion D-012 turns on, pinned at the layer itself
        rather than inferred from the config. If someone "restores"
        ``window_size=self.local_attention_window_size`` at the call site the
        span silently DOUBLES, and nothing else in the suite would notice."""
        config = ModernBERT.MODEL_VARIANTS[variant]
        schedule = _attention_schedule(variant)
        if "local" not in schedule:
            pytest.skip(f"{variant} emits no local layers")

        model = ModernBERT.from_variant(variant)
        local_index = schedule.index("local")
        layer = model.encoder_layers[local_index]
        assert layer.attention_type == "window_band"
        assert layer.attention.partition_mode == "band"
        assert layer.attention.window_size == (
            config["local_attention_window_size"] // 2
        ), (
            f"{variant}: the band half-width is {layer.attention.window_size} "
            f"but local_attention_window_size is "
            f"{config['local_attention_window_size']}, a FULL span. Upstream's "
            f"rule is sliding_window = local_attention // 2."
        )

    def test_tiny_still_partitions_and_keeps_its_local_layers(self):
        """The control on the repair's scope. ``tiny`` was never changed by
        either repair -- it kept its hybrid schedule and its window size
        throughout -- so it is the arm that shows the fixes were targeted."""
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
