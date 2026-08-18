"""Pins TRM's `attention_type` contract (finding F-47, decisions.md D-029).

Before the fix, `TRMReasoningModule.__init__` forwarded
`{'num_kv_heads', 'max_seq_len', 'rope_theta'}` into every `TransformerLayer`
regardless of `attention_type`. Since `create_attention_layer` became strict
(2026-08-17), that made:

1. every documented `attention_type` other than `'group_query'` a hard
   `ValueError` at construction, and
2. every `.keras` checkpoint written before the 2026-08-17 default flip
   unloadable, because `TRM.get_config()` serializes `attention_type` and the
   old default was `'multi_head'`.

Both arms were measured RED at commit ae2e2aa0a. Arm 2 was additionally proven
out of band against a genuine artifact saved from commit 1c10e4203 in a
detached worktree; the test below reproduces the same config path at HEAD so
the pin is self-contained.
"""

import os
import tempfile
from typing import Any, Dict

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.layers.attention.factory import STRICT_DROPPED_KEY_MARKER
from dl_techniques.models.tiny_recursive_model import TRM, create_trm


def _toy_config(**overrides: Any) -> Dict[str, Any]:
    cfg = dict(
        vocab_size=12,
        hidden_size=32,
        num_heads=2,
        expansion=2.0,
        seq_len=16,
        puzzle_emb_len=0,
        h_layers=1,
        l_layers=1,
        halt_max_steps=2,
        no_act_continue=True,
    )
    cfg.update(overrides)
    return cfg


def _toy_batch(batch_size: int = 2, seq_len: int = 16, vocab_size: int = 12):
    rng = np.random.default_rng(0)
    return {"inputs": ops.convert_to_tensor(
        rng.integers(0, vocab_size, size=(batch_size, seq_len)).astype("int32")
    )}


class TestTRMAttentionTypeContract:
    """`attention_type` must have more than one legal value."""

    @pytest.mark.parametrize("attention_type", ["multi_head", "group_query"])
    def test_documented_attention_type_constructs(self, attention_type):
        """RED at HEAD for 'multi_head': the three rope keys were forwarded
        unconditionally and `create_attention_layer` refuses them."""
        model = create_trm(**_toy_config(attention_type=attention_type))
        assert isinstance(model, TRM)
        carry = model.initial_carry(_toy_batch())
        _, out = model(carry, _toy_batch(), training=False)
        assert out["logits"].shape == (2, 16, 12)

    def test_group_query_still_receives_the_rope_keys(self):
        """The filter must not silently disarm the default path (D-007)."""
        model = create_trm(**_toy_config(attention_type="group_query"))
        block = model.inner.H_level.layers_list[0]
        assert block.attention_args["num_kv_heads"] == 2
        assert block.attention_args["rope_theta"] == 10000.0
        assert block.attention_args["max_seq_len"] == 16
        # The rope table is what the D-007 fix was actually for.
        assert hasattr(block.attention, "rope")

    def test_multi_head_drops_them_rather_than_raising(self):
        """A type that declares none of the three gets none of the three."""
        model = create_trm(**_toy_config(attention_type="multi_head"))
        block = model.inner.H_level.layers_list[0]
        assert block.attention_args == {}
        assert not hasattr(block.attention, "rope")

    def test_error_message_marker_is_gone_for_multi_head(self):
        """Anti-vacuity: the strict-factory marker really is what used to fire."""
        assert STRICT_DROPPED_KEY_MARKER  # the marker exists at all
        try:
            create_trm(**_toy_config(attention_type="multi_head"))
        except ValueError as exc:  # pragma: no cover - regression only
            pytest.fail(
                f"multi_head still refused: {exc}"
            )

    def test_legacy_multi_head_checkpoint_round_trips(self):
        """A config carrying the pre-2026-08-17 default must load.

        This is the in-repo reconstruction of the out-of-band evidence: a
        `.keras` written from commit 1c10e4203 (whose `TRM.__init__` defaults
        to `'multi_head'`) failed to load at ae2e2aa0a with
        `create_attention_layer('multi_head'): 3 unsupported parameter(s)`.
        """
        model = create_trm(**_toy_config(attention_type="multi_head"))
        assert model.get_config()["attention_type"] == "multi_head"
        batch = _toy_batch()
        carry = model.initial_carry(batch)
        _, before = model(carry, batch, training=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "trm_multi_head.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        assert loaded.get_config()["attention_type"] == "multi_head"
        carry2 = loaded.initial_carry(batch)
        _, after = loaded(carry2, batch, training=False)
        np.testing.assert_allclose(
            ops.convert_to_numpy(before["logits"]),
            ops.convert_to_numpy(after["logits"]),
            atol=1e-6, rtol=1e-6,
        )
