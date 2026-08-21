"""
GPT-2's FFN nonlinearity is ``gelu_new`` -- step 27, D-130.

The reference implementation uses the tanh approximation of GELU. Keras' own
``'gelu'`` string resolves to ``gelu(x, approximate=False)``, the exact-erf
form, and ``GPT2`` passed no ``activation`` to ``TextDecoder`` at all, so every
FFN in the stack silently got the erf form.

WHY THIS TEST PINS THE INSTALLED FUNCTION AND NOT THE LOGITS
------------------------------------------------------------
The model-level consequence at random init is too small to be a usable oracle.
Measured 2026-08-21 (GPU:1 / RTX 4070, seed 3, dropout pinned 0.0,
vocab 64 / embed 128 / depth 4):

    erf-vs-tanh, raw activation, 1e5 draws of N(0, 3) : max|d| 4.74e-04
    erf-vs-tanh, GPT2 logits (|logits|max = 1.7925)   : max|d| 2.12e-06

2.12e-06 is not distinguishable from ordinary float32 accumulation noise by a
reader, and a threshold placed there would be a coin flip. At
``TruncatedNormal(0.02)`` init the FFN pre-activations sit in the narrow band
around zero where the two GELUs almost coincide; the gap widens on a trained
model. So the assertion is an IDENTITY pin on the function actually installed
on ``MLPBlock.activation_fn``, checked at both ends of the plumbing:
``TextDecoder(activation=...)`` -> ``TransformerLayer`` -> ``MLPBlock``.

Proven RED against the real source by deleting ``activation=gpt2_gelu`` from
``GPT2._build_architecture``: ``test_every_ffn_got_it`` and
``test_survives_round_trip`` both fail. ``test_helper_is_the_tanh_form`` stays
GREEN and is SUPPOSED to -- it is a definitional check on ``gpt2_gelu`` itself,
and it exists so that a later "simplification" of the helper to ``'gelu'`` is
caught by something. Two of three is the honest RED count.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.gpt2.gpt2 import GPT2, gpt2_gelu

_PROBE = np.linspace(-6.0, 6.0, 25).astype("float32")


def _mlp_blocks(model: GPT2):
    out = []
    for layer in model.decoder.decoder_layers:
        out += [s for s in layer._layers if type(s).__name__ == "MLPBlock"]
    return out


@pytest.fixture(scope="module")
def model() -> GPT2:
    keras.utils.set_random_seed(3)
    m = GPT2(
        vocab_size=64,
        embed_dim=128,
        depth=4,
        num_heads=4,
        max_seq_len=16,
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
    )
    m(np.arange(16).reshape(2, 8) % 64, training=False)
    return m


class TestGPT2UsesTanhGelu:
    def test_helper_is_the_tanh_form(self):
        assert np.max(
            np.abs(np.array(gpt2_gelu(_PROBE)) - np.array(keras.activations.gelu(_PROBE, approximate=True)))
        ) == 0.0
        erf_gap = float(
            np.max(np.abs(np.array(gpt2_gelu(_PROBE)) - np.array(keras.activations.gelu(_PROBE, approximate=False))))
        )
        assert erf_gap > 1e-4, f"tanh and erf GELU are indistinguishable here ({erf_gap:.3e})"

    def test_every_ffn_got_it(self, model):
        blocks = _mlp_blocks(model)
        assert len(blocks) == 4, f"expected one MLPBlock per layer, found {len(blocks)}"
        for i, b in enumerate(blocks):
            gap = float(
                np.max(np.abs(np.array(b.activation_fn(_PROBE)) - np.array(keras.activations.gelu(_PROBE, approximate=True))))
            )
            assert gap == 0.0, f"block {i} is not on gelu_new (gap {gap:.3e})"

    def test_survives_round_trip(self, model, tmp_path):
        import os

        path = os.path.join(str(tmp_path), "g.keras")
        model.save(path)
        reloaded = keras.models.load_model(path)
        for b in _mlp_blocks(reloaded):
            assert np.max(
                np.abs(np.array(b.activation_fn(_PROBE)) - np.array(keras.activations.gelu(_PROBE, approximate=True)))
            ) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
