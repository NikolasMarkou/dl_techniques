"""RED proof for F-83 -- ``Gemma3TransformerBlock`` drops two attention kwargs.

``Gemma3TransformerBlock.__init__`` accepts ``kernel_initializer`` and
``use_bias``, stores both, and ``get_config()`` serializes both -- so a round
trip advertises knobs that reached a weight. The GeGLU FFN receives both. The
``create_attention_layer("group_query", ...)`` call received NEITHER, although
``ATTENTION_REGISTRY['group_query']`` declares both.

Two knobs, two instruments, deliberately:

* ``kernel_initializer`` is a SCOPED VALUE knob. A whole-model output diff is
  unsound here because the FFN *does* honour the knob at HEAD, so two arms
  already differ. The discriminating fact is the ``/attention/`` subtree's own
  weight VALUES -- see
  ``tests/test_models/knob_sensitivity_oracle.py``
  ``assert_scoped_value_knob_changes_weights``.
* ``use_bias=True`` changes the weight SET, not weight values. An output diff
  is unsound for it in the opposite direction (the arms cannot share a
  signature), so it is asserted by NAME: the attention subtree must contain
  bias tensors when asked for them.

Measured at commit d3ba16af2 with ``use_bias=True``, the block's 14 weights were
``.../attention/{w_q,w_k,w_v,w_o}/kernel`` + two rope caches and
``.../ffn/{input_proj,output_proj}/{kernel,bias}``: four attention projections,
zero attention biases.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.gemma.components import Gemma3TransformerBlock

from ..knob_sensitivity_oracle import (
    assert_scoped_value_knob_changes_weights,
    build_seeded,
    weights_in_scope,
)

BLOCK_CONFIG = dict(
    hidden_size=32,
    num_attention_heads=4,
    num_key_value_heads=2,
    ffn_hidden_size=64,
    max_seq_len=16,
    dropout_rate=0.0,
)

X = np.random.default_rng(0).random((1, 8, 32)).astype("float32")

#: Scoped with the slashes so ``post_attention_layernorm`` -- whose name also
#: contains "attention", and whose ones-init scale is identical in every arm --
#: cannot dilute the comparison.
ATTENTION_SCOPE = "/attention/"


class TestGemmaBlockForwardsAttentionKnobs:
    """Both attention kwargs must reach the ``group_query`` layer."""

    def test_kernel_initializer_reaches_the_attention_projections(self):
        builders = {
            "he_normal": lambda: Gemma3TransformerBlock(
                kernel_initializer="he_normal", **BLOCK_CONFIG
            ),
            "wide_normal": lambda: Gemma3TransformerBlock(
                kernel_initializer=keras.initializers.RandomNormal(stddev=0.5),
                **BLOCK_CONFIG,
            ),
        }
        assert_scoped_value_knob_changes_weights(
            builders, X, knob="kernel_initializer", scope=ATTENTION_SCOPE
        )

    def test_use_bias_true_creates_attention_bias_tensors(self):
        """``use_bias=True`` must add bias tensors to the attention subtree.

        Asserted by NAME rather than by an output diff: this knob changes the
        weight SET, so the two arms do not share a weight-shape signature and no
        value comparison between them is attributable to the knob.
        """
        block = build_seeded(
            lambda: Gemma3TransformerBlock(use_bias=True, **BLOCK_CONFIG)
        )
        block(X)
        attention_paths = [
            w.path for w in weights_in_scope(block, ATTENTION_SCOPE)
        ]
        biases = [p for p in attention_paths if p.endswith("/bias")]
        assert biases, (
            "use_bias=True created no bias tensor anywhere under the attention "
            f"subtree; its weights are {attention_paths}. The kwarg is not "
            "reaching create_attention_layer('group_query', ...)."
        )

    def test_use_bias_false_creates_no_attention_bias_tensors(self):
        """The other half of the claim: the default must stay bias-free.

        Without this, a fix that hard-coded ``use_bias=True`` would satisfy the
        test above while breaking every shipped Gemma 3 checkpoint's weight tree.
        """
        block = build_seeded(
            lambda: Gemma3TransformerBlock(use_bias=False, **BLOCK_CONFIG)
        )
        block(X)
        biases = [
            w.path
            for w in weights_in_scope(block, ATTENTION_SCOPE)
            if w.path.endswith("/bias")
        ]
        assert not biases, f"use_bias=False still produced {biases}"
