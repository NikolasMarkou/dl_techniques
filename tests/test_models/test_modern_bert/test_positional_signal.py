"""Where ModernBERT's positional signal comes from, and where it does not.

These tests exist because the model shipped with a docstring promising Rotary
Position Embeddings and no RoPE anywhere in the package: every global layer was
built as ``attention_type='multi_head'``, ``MultiHeadAttention`` declares no
RoPE parameter, and the embedding stage adds no positional term. A stack of
global layers was therefore exactly permutation-equivariant — a bag of words
wearing an encoder's interface.

Two regimes are pinned here:

* Global layers MUST be position-sensitive (RoPE), and their RoPE table must be
  materialized rather than left at its zero initializer.
* Local layers are ``window`` attention, which is a **spatial** layer. It folds
  the token sequence into a synthetic ``ceil(sqrt(L))`` square grid. The
  resulting adjacency is a documented deviation from the paper, not a bug, and
  the last test in this module pins it as such so a future reader does not
  "fix" it by accident.

Every probe reads a **per-position** output. A pooled read would be useless:
permuting tokens ahead of a sum/mean reduction cannot change the reduction, so
a pooled probe passes whether or not the model has any positional signal.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.language.modern_bert.model import ModernBERT

VOCAB = 64
PERM = np.array([3, 0, 7, 2, 5, 1, 6, 4])
IDS = np.array([[3, 1, 4, 1, 5, 9, 2, 6]], dtype="int32")


def _forward(model, ids):
    out = model({"input_ids": ops.convert_to_tensor(ids)}, training=False)
    return np.asarray(ops.convert_to_numpy(out["last_hidden_state"]))


@pytest.fixture()
def all_global_model():
    """A stack whose every layer is global — the regime C-1 was about.

    ``global_attention_interval=1`` is load-bearing: at the shipped interval of
    2 or 3 the interleaved ``window`` layers contribute their own (synthetic,
    grid-shaped) order signal, so a whole-model probe could pass with the global
    layers still position-blind.

    ``initializer_range=0.2`` is an INSTRUMENT setting, not the shipped 0.02, for
    the reason spelled out in ``test_oracle_adoption.py``'s rope-theta probe.
    The only positional signal in this stack is RoPE inside global attention --
    MEASURED, the embedding stage alone is EXACTLY permutation-equivariant
    (max|delta| = 0.0) -- so the whole-model defect scales with the attention
    branch's share of the residual stream. Since D-600 delivered ModernBERT's
    ``TruncatedNormal(0.02)`` to those projections (they previously drew at
    ``glorot_uniform``, ten times wider), the same probe MEASURES
    ``initializer_range`` -> defect: 0.02 -> 7.27e-05, 0.1 -> 4.63e-02,
    0.2 -> **4.46e-01**. At the shipped 0.02 the assertion below would read a
    live positional signal as absent -- an untrained network's branch being
    quiet, not a defect.
    """
    keras.utils.set_random_seed(1234)
    return ModernBERT(
        vocab_size=VOCAB,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        intermediate_size=48,
        global_attention_interval=1,
        max_position_embeddings=128,
        hidden_dropout_rate=0.0,
        attention_probs_dropout_rate=0.0,
        initializer_range=0.2,
    )


class TestGlobalLayersArePositionAware:

    def test_global_stack_is_not_permutation_equivariant(self, all_global_model):
        reference = _forward(all_global_model, IDS)
        permuted = _forward(all_global_model, IDS[:, PERM])
        # f is permutation-equivariant iff f(P x) == P f(x) position by position.
        defect = float(np.max(np.abs(reference[:, PERM, :] - permuted)))
        assert defect > 1e-3, (
            "the global layers are permutation-equivariant: permuting the input "
            f"tokens moved the per-position output by only {defect:g}. They "
            "carry no positional signal, and neither does the embedding stage."
        )

    def test_global_attention_layer_holds_a_live_rope_table(self, all_global_model):
        _forward(all_global_model, IDS)
        attention = all_global_model.encoder_layers[0].attention
        rope = getattr(attention, "rope", None)
        assert rope is not None, (
            f"{type(attention).__name__} has no RoPE machinery at all; passing "
            "an unknown key such as use_rope=True through "
            "create_attention_layer was silently dropped, not honoured (that "
            "factory raises on such a key since 2026-08-17, "
            "plan-2026-08-17T183311-79c63e38/D-011 — but a wrong attention_type "
            "with no rope kwarg at all, which is what this asserts against, "
            "still builds happily)."
        )
        cos = np.asarray(ops.convert_to_numpy(rope.cos_cached))
        # cos(0 * w) == 1 for every frequency; the zero-initializer value is 0.
        # This separates "table built" from "table created and never assigned",
        # which is what a StatelessScope does to an `.assign()` inside build().
        np.testing.assert_allclose(cos[0], np.ones_like(cos[0]), atol=1e-6)
        assert float(np.std(cos)) > 1e-3, (
            "the RoPE cos table is constant across positions, so every position "
            "rotates identically and RoPE contributes nothing."
        )

    def test_max_position_embeddings_reaches_the_rope_table(self):
        keras.utils.set_random_seed(3)
        model = ModernBERT(
            vocab_size=VOCAB, hidden_size=32, num_layers=1, num_heads=4,
            intermediate_size=48, global_attention_interval=1,
            max_position_embeddings=97, global_rope_theta=12345.0,
        )
        _forward(model, IDS)
        rope = model.encoder_layers[0].attention.rope
        assert rope.max_seq_len == 97
        assert rope.rope_theta == pytest.approx(12345.0)

    @pytest.mark.parametrize("field,value", [
        ("max_position_embeddings", 0),
        ("global_rope_theta", 0.0),
    ])
    def test_rope_configuration_is_validated(self, field, value):
        with pytest.raises(ValueError, match=field):
            ModernBERT(vocab_size=VOCAB, hidden_size=32, num_layers=1,
                       num_heads=4, intermediate_size=48, **{field: value})

    def test_config_round_trip_carries_the_rope_fields(self):
        # `local_attention_window_size=2` is HISTORICAL and now merely tidy.
        # With the default `global_attention_interval=3` and `num_layers=1`,
        # layer 0 is a LOCAL layer. It used to build `window` attention at the
        # shipped default of 128, whose relative-position index is
        # O(window_size**4) and was built in `__init__` -- ~5 GB at CONSTRUCTION
        # time, and this test builds two models (the original and the
        # `from_config` clone), so it peaked at 6.7 GB before this argument was
        # added. Two changes in plan-2026-08-25T053412-0f1fa04f removed that
        # cost: D-006 moved the index into `build()`, and D-012 routed local
        # layers to `'window_band'`, which has no index at all. Construction of
        # a full `base` is 0.69 GB now. Keep the argument anyway -- it costs
        # nothing and nothing here asserts anything about the band.
        model = ModernBERT(vocab_size=VOCAB, hidden_size=32, num_layers=1,
                           num_heads=4, intermediate_size=48,
                           local_attention_window_size=2,
                           max_position_embeddings=256, global_rope_theta=5000.0)
        clone = ModernBERT.from_config(model.get_config())
        assert clone.max_position_embeddings == 256
        assert clone.global_rope_theta == pytest.approx(5000.0)
        assert clone.local_attention_window_size == 2


class TestLocalNeighbourhoodIsAContiguousOneDimensionalBand:
    """REPLACES ``TestLocalWindowAdjacencyIsSynthetic`` (deleted 2026-08-25).

    That class pinned a documented DEFECT: ``window`` attention is a spatial
    layer, so given a rank-3 text sequence it reshaped ``(B, L, D)`` into a
    synthetic ``ceil(sqrt(L))``-square grid and attended inside
    ``window_size``-square blocks. At ``L=16, window_size=2`` the grid was 4x4
    and token 0's neighbourhood was tokens 0, 1, 4, 5 -- while tokens 2 and 3,
    which a genuine 1-D window of any width >= 2 includes BEFORE token 4, were
    invisible to it. It asserted exactly that: ``influence(4) > 1e-3`` and
    ``influence(2) == 0.0``.

    Its own docstring set its termination condition: *"If a 1-D sliding-window
    attention layer is ever added to ``layers/attention/`` and wired in here,
    this test SHOULD fail, and the correct response is to delete it -- not to
    reinstate the grid."* ``partition_mode='band'`` landed
    (plan-2026-08-25T053412-0f1fa04f, D-003/D-010) and D-012 wired it in, so the
    class is gone and this one states the property that replaced it.

    ``local_attention_window_size`` is now a 1-D FULL SPAN, so at 2 the layer
    receives half-width 1: token ``i`` attends exactly ``i-1, i, i+1``. The
    claim is measured the only way it can be honestly measured -- by
    PERTURBATION, one input token at a time, never by reading the mask back.
    """

    def test_the_local_neighbourhood_is_contiguous_and_the_grid_is_gone(self):
        keras.utils.set_random_seed(7)
        model = ModernBERT(
            vocab_size=VOCAB, hidden_size=32, num_layers=1, num_heads=4,
            intermediate_size=48,
            global_attention_interval=999,  # no layer is global
            local_attention_window_size=2,  # full span 2 -> half-width 1
            hidden_dropout_rate=0.0, attention_probs_dropout_rate=0.0,
        )
        assert model.encoder_layers[0].attention_type == "window_band"

        base = (np.arange(16, dtype="int32") + 1).reshape(1, 16)
        reference = _forward(model, base)[0, 0, :]

        def influence_on_token_0(position):
            perturbed = base.copy()
            perturbed[0, position] = VOCAB - 1
            return float(np.max(np.abs(_forward(model, perturbed)[0, 0, :] - reference)))

        in_band = influence_on_token_0(1)
        just_outside = influence_on_token_0(2)
        old_grid_partner = influence_on_token_0(4)

        assert in_band > 1e-3, (
            "token 1 is inside token 0's half-width-1 band and must influence it; "
            "if this is 0.0 the band is dead, not narrow"
        )
        assert just_outside == 0.0, (
            f"token 2 is outside a half-width-1 band and moved token 0 by "
            f"{just_outside}. The band is wider than local_attention_window_size // 2, "
            f"or the mask is not being applied."
        )
        assert old_grid_partner == 0.0, (
            f"token 4 moved token 0 by {old_grid_partner}. Token 4 was token 0's "
            f"partner in the OLD 2x2 synthetic grid block and is 4 positions away "
            f"in 1-D -- a nonzero here means the grid adjacency is back."
        )
