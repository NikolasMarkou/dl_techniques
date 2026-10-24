"""Every norm of a TopoLM model runs at the model's ``layer_norm_eps``, the final one included.

plan-2026-10-23T190557-972a9ac5 step 13.3 (review pass 6, concern 9). The block's
docstring calls two norms at different epsilons inside one forward pass a silent
defect (a 1000x spread of the denominator), yet nothing pinned the model's FINAL
norm: replacing ``{"epsilon": self.layer_norm_eps}`` by ``{}`` in ``TopoLM.build`` left
the whole model suite green (the factory then falls back to its own default). The
tests here read the epsilon back from every norm and also compute the final norm's
output with numpy at a large epsilon, where the denominator is dominated by it.

Run on CPU (``CUDA_VISIBLE_DEVICES=``).
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.language.topolm import TopoLM

#: Not 1e-5 (the model default), 1e-6 or 1e-3 (the two factory defaults), so a norm
#: that fell back to any default is told apart.
EPS = 0.25

CONFIG = dict(vocab_size=50, embed_dim=64, depth=2, num_heads=4,
              ffn_intermediate_size=128, max_seq_len=16, radius=1)


def make(**kwargs) -> TopoLM:
    keras.utils.set_random_seed(0)
    model = TopoLM(layer_norm_eps=EPS, name="topolm", **{**CONFIG, **kwargs})
    model.build((None, 8))
    return model


def norms(model: TopoLM):
    found = [("final_norm", model.final_norm)]
    for block in model.blocks:
        found += [(f"{block.name}.attention_norm", block.attention_norm),
                  (f"{block.name}.ffn_norm", block.ffn_norm)]
    return found


class TestEveryNormUsesTheModelsEpsilon:
    @pytest.mark.parametrize("normalization_type", ["layer_norm", "rms_norm"])
    def test_the_epsilon_of_every_norm(self, normalization_type):
        model = make(normalization_type=normalization_type)
        assert len(norms(model)) == 1 + 2 * CONFIG["depth"]
        for name, norm in norms(model):
            assert norm.epsilon == EPS, (name, norm.epsilon)

    def test_a_norms_own_args_still_win_for_that_norm_only(self):
        model = make(ffn_norm_args={"epsilon": 1e-3})
        assert {n: x.epsilon for n, x in norms(model)} == {
            "final_norm": EPS,
            "block_0.attention_norm": EPS, "block_0.ffn_norm": 1e-3,
            "block_1.attention_norm": EPS, "block_1.ffn_norm": 1e-3,
        }

    def test_the_final_norm_output_is_the_numpy_layer_norm_at_that_epsilon(self):
        """At eps 0.25 and a unit-scale input the output is far from the 1e-6 one."""
        model = make()
        x = np.random.default_rng(0).standard_normal((2, 5, 64)).astype("float32")
        gamma = keras.ops.convert_to_numpy(model.final_norm.gamma)
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        expected = (x - mu) / np.sqrt(var + EPS) * gamma
        got = keras.ops.convert_to_numpy(model.final_norm(x))
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-5)
        wrong = (x - mu) / np.sqrt(var + 1e-6) * gamma
        assert np.abs(wrong - expected).max() > 0.05  # the instrument can tell them apart

    def test_the_epsilon_survives_a_config_round_trip(self):
        model = make()
        clone = TopoLM.from_config(model.get_config())
        clone.build((None, 8))
        assert [x.epsilon for _, x in norms(clone)] == [EPS] * (1 + 2 * CONFIG["depth"])
