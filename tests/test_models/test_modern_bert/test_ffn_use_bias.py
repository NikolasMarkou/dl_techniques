"""ModernBERT's `use_bias=False` must reach its GeGLU FFN weight set.

ModernBERT is the ONE shipped model whose weight set moves under
plan-2026-08-19-a616f581/D-006 (measured, whole-`src/` AST enumeration of
every `TransformerLayer(...)`/`TransformerDecoderLayer(...)` construction):

* it constructs its blocks with ``ffn_type='geglu'`` and ``use_bias=False``
  (all three ``MODEL_VARIANTS`` set ``use_bias: False``), and
* unlike Qwen3 it passes no ``use_bias`` through ``ffn_args`` (its
  ``ffn_args`` carries only ``activation``), so before D-006 the `geglu`
  class default ``use_bias=True`` won and every encoder layer carried
  ``ffn/input_proj/bias`` and ``ffn/output_proj/bias``.

Every other `use_bias=False` consumer of `TransformerLayer` reaches the FFN
with ``ffn_type='swiglu'`` (Qwen3, `HierarchicalReasoningModule`, BLT's
`blt_core`), and D-006 deliberately withholds ``use_bias`` for `swiglu`, so
their weight sets are byte-for-byte unchanged.

The ``/bias`` SUFFIX predicate is load-bearing: it excludes the norm
sublayers' ``beta``/``gamma`` and any positional bias TABLE.
"""

import pytest

from dl_techniques.models.modern_bert.model import ModernBERT


def _tiny_modern_bert(*, use_bias: bool) -> ModernBERT:
    """A 1-layer, local-attention ModernBERT built and called once."""
    import keras

    model = ModernBERT(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        intermediate_size=24,
        use_bias=use_bias,
        global_attention_interval=999,  # no layer is global
        local_attention_window_size=16,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    _ = model(
        {
            "input_ids": keras.ops.zeros((1, 16), dtype="int32"),
            "attention_mask": keras.ops.ones((1, 16), dtype="int32"),
        },
        training=False,
    )
    return model


def _encoder_ffn_bias_paths(model: ModernBERT) -> list:
    return [
        w.path for w in model.encoder_layers[0].weights
        if "/ffn/" in w.path and w.path.endswith("/bias")
    ]


class TestModernBertFFNHonoursUseBias:
    """The fix arm and its liveness twin, on the real model."""

    def test_use_bias_false_leaves_no_ffn_bias_weights(self) -> None:
        """Pre-fix this returned the two GeGLU projections' biases."""
        leaked = _encoder_ffn_bias_paths(_tiny_modern_bert(use_bias=False))
        assert leaked == [], (
            f"ModernBERT(use_bias=False) leaked FFN bias weights: {leaked}"
        )

    def test_use_bias_true_still_creates_both_ffn_biases(self) -> None:
        """LIVENESS arm — without it the arm above passes just as well against
        a model whose GeGLU has no biases at ANY setting."""
        present = _encoder_ffn_bias_paths(_tiny_modern_bert(use_bias=True))
        assert len(present) == 2, (
            f"expected input_proj/bias and output_proj/bias at "
            f"use_bias=True, got {present}"
        )
        assert any(p.endswith("/input_proj/bias") for p in present), present
        assert any(p.endswith("/output_proj/bias") for p in present), present

    def test_ffn_args_carries_no_use_bias(self) -> None:
        """The premise of this module: ModernBERT has no `ffn_args` escape
        hatch, so the block-level `use_bias` is the ONLY channel. If a future
        edit adds one, this module stops measuring what it claims to.
        """
        layer = _tiny_modern_bert(use_bias=False).encoder_layers[0]
        assert "use_bias" not in layer.ffn_args, (
            f"ModernBERT now passes use_bias through ffn_args ({layer.ffn_args}); "
            f"this module's premise, and D-006's blast-radius claim, need re-measuring"
        )
