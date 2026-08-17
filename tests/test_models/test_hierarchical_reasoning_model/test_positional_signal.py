"""Where HRM's positional signal comes from, and where it did not.

``HierarchicalReasoningCore`` defaults ``pos_encodings="rope"``, constructs a
``RotaryPositionEmbedding``, builds it in ``build()`` — and never hands it a Q or
a K tensor. All attention is delegated to ``HierarchicalReasoningModule``, which
built its ``TransformerLayer`` stack with ``attention_type='multi_head'``, a
registry entry that carries no RoPE parameter at all. Unlike TRM (whose sibling
defect dropped ``rope_theta``/``max_seq_len`` through the attention factory), HRM
never populated ``attention_args`` in the first place, so the factory's strict
mode (plan-2026-08-17T183311-79c63e38/D-011) could not surface it either. Every
HRM built the documented way was therefore exactly permutation-equivariant: a bag
of tokens wearing a reasoner's interface.

RoPE is a per-Q/K rotation applied INSIDE attention. It is not an additive term
on the ``(batch, total_seq_len, embed_dim)`` input embedding, which is why the
fix is architectural — route attention through a RoPE-capable type — rather than
"call ``self.rope`` in ``_input_embeddings``".

The probe reads a **per-position** output (``logits``). A pooled read would be
useless: permuting tokens ahead of a sum/mean reduction cannot change the
reduction, so a pooled probe passes whether or not the model has any positional
signal.

``PERM`` is deliberately neither a reversal nor a cyclic shift. A model whose
token mixing is a DFT (or any rotation-equivariant operator) is invariant to a
cyclic shift up to a phase, so a shift-based probe can be vacuous against a model
that genuinely has no positional signal.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.hierarchical_reasoning_model.model import (
    HierarchicalReasoningModel,
)

VOCAB = 64
SEQ_LEN = 8
PERM = np.array([3, 0, 7, 2, 5, 1, 6, 4])
TOKEN_IDS = np.array([[3, 1, 4, 1, 5, 9, 2, 6]], dtype="int32")
PUZZLE_IDS = np.array([0], dtype="int32")


def _build(**overrides):
    kwargs = dict(
        vocab_size=VOCAB,
        seq_len=SEQ_LEN,
        embed_dim=32,
        num_puzzle_identifiers=4,
        puzzle_emb_dim=0,
        batch_size=1,
        h_layers=1,
        l_layers=1,
        h_cycles=1,
        l_cycles=1,
        num_heads=4,
        ffn_expansion_factor=2,
        halt_max_steps=1,
        halt_exploration_prob=0.0,
        dropout_rate=0.0,
    )
    kwargs.update(overrides)
    return HierarchicalReasoningModel(**kwargs)


def _forward(model, token_ids):
    batch = {
        "token_ids": ops.convert_to_tensor(token_ids),
        "puzzle_ids": ops.convert_to_tensor(PUZZLE_IDS),
    }
    outputs = model(batch, training=False)
    return np.asarray(ops.convert_to_numpy(outputs["logits"]))


@pytest.fixture()
def model():
    keras.utils.set_random_seed(1234)
    return _build()


class TestHRMIsPositionAware:

    def test_output_varies_across_positions(self, model):
        """Anti-vacuity arm for the equivariance probe below.

        If the per-position logits were constant along the token axis, the
        equivariance comparison would be trivially satisfiable and would prove
        nothing about positional signal either way.
        """
        reference = _forward(model, TOKEN_IDS)
        spread = float(np.max(np.std(reference, axis=1)))
        assert spread > 1e-3, (
            "the per-position logits are constant along the token axis "
            f"(max per-channel std {spread:g}), so the equivariance probe "
            "below cannot distinguish a position-aware model from a "
            "position-blind one."
        )

    def test_reasoning_stack_is_not_permutation_equivariant(self, model):
        reference = _forward(model, TOKEN_IDS)
        permuted = _forward(model, TOKEN_IDS[:, PERM])
        # f is permutation-equivariant iff f(P x) == P f(x) position by position.
        defect = float(np.max(np.abs(reference[:, PERM, :] - permuted)))
        assert defect > 1e-3, (
            "HRM is permutation-equivariant: permuting the input tokens moved "
            f"the per-position logits by only {defect:g}. Its attention carries "
            "no positional signal, and `pos_encodings='rope'` builds a "
            "RotaryPositionEmbedding on the core that is never handed a Q or K "
            "tensor — so `rope_theta` reaches nothing."
        )

    def test_attention_holds_a_live_rope_table(self, model):
        _forward(model, TOKEN_IDS)
        attention = model.core.l_reasoning.blocks[0].attention
        rope = getattr(attention, "rope", None)
        assert rope is not None, (
            f"{type(attention).__name__} has no RoPE machinery at all; HRM's "
            "reasoning stack runs a rope-free attention type while the core "
            "advertises `pos_encodings='rope'`."
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

    def test_rope_configuration_reaches_the_table(self):
        keras.utils.set_random_seed(3)
        model = _build(rope_theta=12345.0, puzzle_emb_dim=16)
        _forward(model, TOKEN_IDS)
        rope = model.core.h_reasoning.blocks[0].attention.rope
        assert rope.max_seq_len == model.core.total_seq_len
        assert rope.rope_theta == pytest.approx(12345.0)

    def test_core_does_not_own_an_unused_rope_layer(self, model):
        """The core must not carry a RoPE instance it never calls.

        `HierarchicalReasoningCore` owned a built `RotaryPositionEmbedding` that
        no code path ever handed a Q or K tensor. Owning it is what made the
        position-blindness invisible to every reader: `pos_encodings='rope'`
        constructed a real, built, correctly-configured layer, and the model
        still had no positional signal.
        """
        assert not hasattr(model.core, "rope"), (
            "HierarchicalReasoningCore still owns a `self.rope` attribute. RoPE "
            "belongs inside attention (a per-Q/K rotation); a core-level "
            "instance is dead weight that reads as evidence of a feature the "
            "model does not have."
        )
