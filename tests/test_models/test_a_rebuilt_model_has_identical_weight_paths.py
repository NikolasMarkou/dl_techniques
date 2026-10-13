"""Two same-config builds in one process have identical weight paths.

plan-2026-10-11T170958-dc4d2202 D-014. A sub-layer left without ``name=`` takes
Keras's per-process counter (``group_attention_4``, ``mamba_layer_6``,
``swi_gluffn_16``, ``linear_gating_8``), so the second model built in a process
disagrees with the first on its weight paths and any by-path comparison misses.
A DECOY model is built first so the counters are already advanced; the two
models that are compared are built after it.

Archives are keyed by the attribute tree, not by layer name, so naming the
layers moves no value: the old-archive/new-code and new-archive/old-code loads
were probed bit-equal for every subject here (``load_model``,
``load_weights(.weights.h5)``, and the ``pretrained=`` route where one exists)
before this file was committed.
"""

import keras
import numpy as np
import pytest

_VOCAB = 64


def _ids(batch=2, length=8, vocab=_VOCAB):
    return np.random.default_rng(0).integers(1, vocab, (batch, length)).astype("int32")


def _tree(name):
    from dl_techniques.models.language.tree_transformer.model import TreeTransformer
    return TreeTransformer(
        vocab_size=_VOCAB, hidden_size=32, num_layers=2, num_heads=2,
        intermediate_size=64, max_len=16, name=name)


def _mamba(name):
    from dl_techniques.models.language.mamba.mamba_v1 import Mamba
    return Mamba(vocab_size=_VOCAB, d_model=32, num_layers=2, d_state=8, name=name)


def _mamba2(name):
    from dl_techniques.models.language.mamba.mamba_v2 import Mamba2
    return Mamba2(vocab_size=_VOCAB, d_model=32, num_layers=2, d_state=8,
                  headdim=16, name=name)


def _qwen3_next(name):
    from dl_techniques.models.language.qwen.qwen3_next import Qwen3Next
    return Qwen3Next(
        vocab_size=_VOCAB, hidden_size=32, num_layers=1, num_attention_heads=2,
        num_key_value_heads=1, max_seq_len=32, num_experts=2,
        num_experts_per_tok=1, moe_intermediate_size=32, name=name)


def _qwen3_moe(name):
    from dl_techniques.models.language.qwen.qwen3 import Qwen3
    return Qwen3(
        vocab_size=_VOCAB, hidden_size=32, num_layers=2, num_attention_heads=2,
        num_key_value_heads=1, max_seq_len=32, moe_layers=[1], num_experts=2,
        num_experts_per_tok=1, moe_intermediate_size=32, dropout_rate=0.0,
        name=name)


def _zamba2(name):
    from dl_techniques.models.language.zamba2.model import Zamba2Model
    return Zamba2Model(
        vocab_size=37, hidden_size=32,
        layer_mapping=["m", "m", "g", "m", "m", "g"], num_mem_blocks=2,
        num_heads=4, max_seq_len=16, d_state=16, headdim=8, name=name)


def _nam(name):
    from dl_techniques.models.neural_computer.nam.model import NAM, NAMConfig
    return NAM(config=NAMConfig(
        hidden_size=32, num_heads=4, num_tree_layers=2, intermediate_size=64,
        memory_size=8, num_read_heads=2, max_expression_len=16,
        halt_max_steps=2, hidden_dropout_rate=0.0, attention_dropout_rate=0.0),
        name=name)


def _forward(subject, model):
    if subject == "nam":
        batch = {"input_ids": _ids(2, 16, 16)}
        model(model.initial_carry(batch), batch, training=False)
    elif subject == "zamba2":
        model(_ids(2, 8, 37), training=False)
    elif subject == "tree":
        model({"input_ids": _ids()}, training=False)
    else:
        model(_ids(), training=False)


_SUBJECTS = {
    "tree": _tree, "nam": _nam, "mamba": _mamba, "mamba2": _mamba2,
    "qwen3_next": _qwen3_next, "qwen3_moe": _qwen3_moe, "zamba2": _zamba2,
}


@pytest.mark.parametrize("subject", sorted(_SUBJECTS))
def test_two_builds_after_a_decoy_have_identical_weight_paths(subject):
    build = _SUBJECTS[subject]
    keras.utils.set_random_seed(0)
    _forward(subject, build(f"decoy_{subject}"))  # advances the name counters
    first, second = build(f"p_{subject}"), build(f"p_{subject}")
    _forward(subject, first)
    _forward(subject, second)

    paths_a = [w.path for w in first.weights]
    paths_b = [w.path for w in second.weights]
    assert len(paths_a) > 5, "too few weights: the arm proves nothing"
    assert paths_a == paths_b, [(a, b) for a, b in zip(paths_a, paths_b) if a != b][:4]
