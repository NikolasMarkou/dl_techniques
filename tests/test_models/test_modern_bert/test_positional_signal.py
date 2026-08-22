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

from dl_techniques.models.modern_bert.model import ModernBERT

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
        model = ModernBERT(vocab_size=VOCAB, hidden_size=32, num_layers=1,
                           num_heads=4, intermediate_size=48,
                           max_position_embeddings=256, global_rope_theta=5000.0)
        clone = ModernBERT.from_config(model.get_config())
        assert clone.max_position_embeddings == 256
        assert clone.global_rope_theta == pytest.approx(5000.0)


class TestLocalWindowAdjacencyIsSynthetic:
    """PINS A KNOWN, DOCUMENTED PROPERTY — this is not a bug report.

    ``window`` attention is a spatial layer. Given a rank-3 text sequence it
    does **not** raise; it reshapes ``(B, L, D)`` into a synthetic
    ``ceil(sqrt(L))``-square grid and attends inside ``window_size``-square
    blocks of that grid. With ``L=16`` and ``window_size=2`` the grid is 4x4 and
    the first block is grid cells (0,0), (0,1), (1,0), (1,1) — i.e. tokens
    0, 1, 4 and 5. Tokens 2 and 3, which a genuine 1-D window of any width >= 2
    would include before token 4, are invisible to token 0.

    If a 1-D sliding-window attention layer is ever added to
    ``layers/attention/`` and wired in here, this test SHOULD fail, and the
    correct response is to delete it — not to reinstate the grid.
    """

    def test_local_neighbourhood_follows_the_synthetic_grid(self):
        keras.utils.set_random_seed(7)
        model = ModernBERT(
            vocab_size=VOCAB, hidden_size=32, num_layers=1, num_heads=4,
            intermediate_size=48,
            global_attention_interval=999,  # no layer is global
            local_attention_window_size=2,
            hidden_dropout_rate=0.0, attention_probs_dropout_rate=0.0,
        )
        assert model.encoder_layers[0].attention_type == "window"

        base = (np.arange(16, dtype="int32") + 1).reshape(1, 16)
        reference = _forward(model, base)[0, 0, :]

        def influence_on_token_0(position):
            perturbed = base.copy()
            perturbed[0, position] = VOCAB - 1
            return float(np.max(np.abs(_forward(model, perturbed)[0, 0, :] - reference)))

        same_grid_block = influence_on_token_0(4)
        adjacent_in_1d = influence_on_token_0(2)

        assert same_grid_block > 1e-3, (
            "token 4 shares a 2x2 grid block with token 0 and must influence it"
        )
        assert adjacent_in_1d == 0.0, (
            "token 2 influenced token 0, so the window is no longer the "
            "synthetic 2-D grid this test documents"
        )
