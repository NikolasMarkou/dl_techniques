"""Where TRM's positional signal comes from, and where it did not.

TRM's package docstring and every one of its constructors advertise a
``rope_theta``, and ``TRMReasoningModule`` duly forwarded
``{'max_seq_len': ..., 'rope_theta': ...}`` into ``TransformerLayer``. But the
default ``attention_type`` was ``'multi_head'``, and ``MultiHeadAttention``
declares no RoPE parameter at all — ``create_attention_layer`` filtered kwargs
against the target type's registry allowlist and dropped the rest silently, so
both keys evaporated. (That silent drop is HISTORICAL as of 2026-08-17,
plan-2026-08-17T183311-79c63e38/D-011: the factory now RAISES on an undeclared
key, so the pre-fix configuration would fail at construction rather than ship a
position-blind model. This file's probes are unchanged and still valid — they
assert positional SIGNAL, not the factory's error behaviour.) The whole reasoning stack was therefore exactly
permutation-equivariant: a bag of tokens wearing a reasoner's interface.

The probe reads a **per-position** output (``logits``). A pooled read would be
useless: permuting tokens ahead of a sum/mean reduction cannot change the
reduction, so a pooled probe passes whether or not the model has any positional
signal.

``PERM`` is deliberately neither a reversal nor a cyclic shift. A model whose
token mixing is a DFT (or any rotation-equivariant operator) is invariant to a
cyclic shift up to a phase, so a shift-based probe can be vacuous against a
model that genuinely has no positional signal.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.tiny_recursive_model import TRM

VOCAB = 64
SEQ_LEN = 8
PUZZLE_EMB_LEN = 2
PERM = np.array([3, 0, 7, 2, 5, 1, 6, 4])
IDS = np.array([[3, 1, 4, 1, 5, 9, 2, 6]], dtype="int32")


def _forward(model, ids):
    batch = {"inputs": ops.convert_to_tensor(ids)}
    carry = model.initial_carry(batch)
    _, outputs = model(carry, batch, training=False)
    return np.asarray(ops.convert_to_numpy(outputs["logits"]))


@pytest.fixture()
def model():
    keras.utils.set_random_seed(1234)
    return TRM(
        vocab_size=VOCAB,
        hidden_size=32,
        num_heads=4,
        expansion=2.0,
        seq_len=SEQ_LEN,
        puzzle_emb_len=PUZZLE_EMB_LEN,
        h_layers=1,
        l_layers=1,
        halt_max_steps=1,
        halt_exploration_prob=0.0,
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
    )


class TestTRMIsPositionAware:

    def test_output_varies_across_positions(self, model):
        """Anti-vacuity arm for the equivariance probe below.

        If the per-position logits were constant along the token axis, the
        equivariance comparison would be trivially satisfiable and would prove
        nothing about positional signal either way.
        """
        reference = _forward(model, IDS)
        spread = float(np.max(np.std(reference, axis=1)))
        assert spread > 1e-3, (
            "the per-position logits are constant along the token axis "
            f"(max per-channel std {spread:g}), so the equivariance probe "
            "below cannot distinguish a position-aware model from a "
            "position-blind one."
        )

    def test_reasoning_stack_is_not_permutation_equivariant(self, model):
        reference = _forward(model, IDS)
        permuted = _forward(model, IDS[:, PERM])
        # f is permutation-equivariant iff f(P x) == P f(x) position by position.
        defect = float(np.max(np.abs(reference[:, PERM, :] - permuted)))
        assert defect > 1e-3, (
            "TRM is permutation-equivariant: permuting the input tokens moved "
            f"the per-position logits by only {defect:g}. Its attention carries "
            "no positional signal, and nothing in the embedding stage adds one "
            "— so `rope_theta` reaches nothing."
        )

    def test_attention_holds_a_live_rope_table(self, model):
        _forward(model, IDS)
        attention = model.inner.L_level.layers_list[0].attention
        rope = getattr(attention, "rope", None)
        assert rope is not None, (
            f"{type(attention).__name__} has no RoPE machinery at all; the "
            "`max_seq_len`/`rope_theta` handed to create_attention_layer were "
            "silently dropped, not honoured (that factory raises on them since "
            "2026-08-17, plan-2026-08-17T183311-79c63e38/D-011, so this assertion "
            "now guards against a rope-free attention_type rather than a silent "
            "kwarg drop)."
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
        model = TRM(
            vocab_size=VOCAB, hidden_size=32, num_heads=4, expansion=2.0,
            seq_len=SEQ_LEN, puzzle_emb_len=PUZZLE_EMB_LEN,
            h_layers=1, l_layers=1, halt_max_steps=1,
            rope_theta=12345.0,
        )
        _forward(model, IDS)
        rope = model.inner.H_level.layers_list[0].attention.rope
        assert rope.max_seq_len == SEQ_LEN + PUZZLE_EMB_LEN
        assert rope.rope_theta == pytest.approx(12345.0)
