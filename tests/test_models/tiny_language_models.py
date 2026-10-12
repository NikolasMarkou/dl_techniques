"""Tiny, fast-to-build configurations of the language models that own a task head.

One row per class, shared by the contract tests that need "a small instance of
each" (``test_a_decoder_assumes_no_padding_token``,
``test_a_language_model_reports_its_output_shape``). The vocabulary is
``VOCAB`` for every row, a value that is no class's default.

Interface contract: ``TINY[name]`` is a :class:`Tiny` (all fifteen classes that own a head); ``tiny.cls()`` imports
and returns the class; ``tiny.kwargs`` are constructor keyword arguments that
build a one-layer model in a fraction of a second (no ``include_top``, no
``task_config``: the caller adds those). Adding a class is one row.
"""

import importlib
from typing import Any, Dict, NamedTuple, Optional, Sequence
from unittest import mock

from dl_techniques.models.language.hnet.config import (
    MODEL_VARIANTS as HNET_VARIANTS,
    AttnSpec,
    HNetArchConfig,
    SSMSpec,
)

VOCAB = 64


def tiny_hnet_arch(
        vocab_size: int = VOCAB,
        d_model: int = 16,
        layout: Optional[Sequence[Any]] = None,
) -> HNetArchConfig:
    """A two-stage H-Net layout a CPU builds in a fraction of a second.

    The vocabulary lives in the architecture, not in a constructor argument, so
    a vocabulary-64 HNet is a vocabulary-64 ``HNetArchConfig``. The default
    layout has attention only in the inner stage (no input-length limit);
    ``layout`` swaps it (``["T1", ["T1"], "T1"]`` puts attention at stage 0).
    """
    return HNetArchConfig(
        arch_layout=list(layout) if layout is not None else ["m1", ["T1"], "m1"],
        d_model=[d_model, d_model],
        d_intermediate=[0, 0],
        vocab_size=vocab_size,
        ssm_cfg=SSMSpec(d_conv=4, expand=2, d_state=8),
        attn_cfg=AttnSpec(num_heads=(2, 2), rotary_emb_dim=(4, 4),
                          window_size=(-1, -1)),
    )


def create_tiny_hnet(pretrained: Any = False, vocab_size: int = VOCAB,
                     d_model: int = 16, layout: Optional[Sequence[Any]] = None,
                     **model_kwargs: Any):
    """``create_hnet`` on a tiny variant registered for the call only.

    Every shipped variant is 1024 wide or more, so the ``pretrained=<path>``
    route of ``create_hnet`` / ``HNet.from_variant`` can only be exercised on
    a CPU through a variant of the test's own. ``model_kwargs`` go to the
    constructor (``max_seq_len`` defaults to 16 and ``headdim`` to 8).
    """
    from dl_techniques.models.language.hnet.model import create_hnet

    arch = tiny_hnet_arch(vocab_size, d_model, layout)
    model_kwargs.setdefault("max_seq_len", 16)
    model_kwargs.setdefault("headdim", 8)
    with mock.patch.dict(HNET_VARIANTS, {"hnet_tiny_test": arch}):
        return create_hnet("hnet_tiny_test", pretrained=pretrained, **model_kwargs)


class Tiny(NamedTuple):
    module: str
    name: str
    kwargs: Dict[str, Any]

    def cls(self):
        return getattr(importlib.import_module(self.module), self.name)


TINY = {
    "bert": Tiny(
        "dl_techniques.models.language.bert.model", "BERT",
        dict(vocab_size=VOCAB, hidden_size=32, num_layers=1, num_heads=2,
             intermediate_size=64, max_position_embeddings=32)),
    "tree_transformer": Tiny(
        "dl_techniques.models.language.tree_transformer.model", "TreeTransformer",
        dict(vocab_size=VOCAB, hidden_size=32, num_layers=1, num_heads=2,
             intermediate_size=64, max_len=16)),
    "distilbert": Tiny(
        "dl_techniques.models.language.distilbert.model", "DistilBERT",
        dict(vocab_size=VOCAB, hidden_size=32, num_layers=1, num_heads=2,
             intermediate_size=64, max_position_embeddings=32)),
    "modern_bert": Tiny(
        "dl_techniques.models.language.modern_bert.model", "ModernBERT",
        dict(vocab_size=VOCAB, hidden_size=32, num_layers=1, num_heads=2,
             intermediate_size=64, max_position_embeddings=32,
             local_attention_window_size=8)),
    "fnet": Tiny(
        "dl_techniques.models.language.fnet.model", "FNet",
        dict(vocab_size=VOCAB, hidden_size=32, num_layers=1,
             intermediate_size=64, max_position_embeddings=32)),
    "gpt2": Tiny(
        "dl_techniques.models.language.gpt2.gpt2", "GPT2",
        dict(vocab_size=VOCAB, embed_dim=32, depth=1, num_heads=2,
             max_seq_len=32)),
    "topolm": Tiny(
        "dl_techniques.models.language.topolm.model", "TopoLM",
        dict(vocab_size=VOCAB, embed_dim=64, depth=1, num_heads=2,
             max_seq_len=32, radius=2)),
    "wave_field": Tiny(
        "dl_techniques.models.language.wave_field.model", "WaveFieldLLM",
        dict(vocab_size=VOCAB, embed_dim=32, depth=1, num_heads=2,
             max_seq_len=32, field_size=32, dropout_rate=0.0,
             attention_dropout_rate=0.0)),
    "qwen3": Tiny(
        "dl_techniques.models.language.qwen.qwen3", "Qwen3",
        dict(vocab_size=VOCAB, hidden_size=32, num_layers=1,
             num_attention_heads=2, num_key_value_heads=1, max_seq_len=32,
             moe_layers=[], num_experts=1, num_experts_per_tok=1,
             dropout_rate=0.0)),
    "qwen3_next": Tiny(
        "dl_techniques.models.language.qwen.qwen3_next", "Qwen3Next",
        dict(vocab_size=VOCAB, hidden_size=32, num_layers=1,
             num_attention_heads=2, num_key_value_heads=1, max_seq_len=32,
             num_experts=2, num_experts_per_tok=1, moe_intermediate_size=32)),
    "gemma3": Tiny(
        "dl_techniques.models.language.gemma.gemma3", "Gemma3",
        dict(vocab_size=VOCAB, hidden_size=32, num_layers=1,
             num_attention_heads=2, num_key_value_heads=1, ffn_hidden_size=64,
             max_seq_len=32, sliding_window_size=8)),
    "mamba": Tiny(
        "dl_techniques.models.language.mamba.mamba_v1", "Mamba",
        dict(vocab_size=VOCAB, d_model=32, num_layers=1, d_state=8, d_conv=4,
             expand=2)),
    "mamba2": Tiny(
        "dl_techniques.models.language.mamba.mamba_v2", "Mamba2",
        dict(vocab_size=VOCAB, d_model=32, num_layers=1, d_state=8, d_conv=4,
             expand=2, headdim=16)),
    "zamba2": Tiny(
        "dl_techniques.models.language.zamba2.model", "Zamba2Model",
        dict(vocab_size=VOCAB, hidden_size=32, layer_mapping=["m", "g"],
             num_mem_blocks=1, num_heads=2, max_seq_len=32, d_state=8,
             headdim=16)),
    "hnet": Tiny(
        "dl_techniques.models.language.hnet.model", "HNet",
        dict(arch_config=tiny_hnet_arch(), max_seq_len=32, headdim=8)),
}
