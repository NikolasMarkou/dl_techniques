"""Zamba2 owns one task head when ``include_top=True``.

Headless, :class:`Zamba2Model` still returns the tied-logits tensor (every
existing test and the ``skip_head=True`` trainer depend on that). With a
``task_config`` it returns the finalized output of a ``layers/heads/nlp`` head
named ``task_head`` instead, fed the final-norm output and a padding mask, and
the logits are not computed (plan-2026-10-11T170958-dc4d2202 D-005). The claims
shared with HNet live in ``headed_logit_decoder_cases``; the ones below are
Zamba2's own: the weight-path identity of the backbone across tops, the
``zamba2_*`` variant route, and the layer-mapping variants that change which
blocks exist (no ``'g'`` position, so no RoPE limit).
"""

from typing import Any

import keras
import numpy as np
import pytest

from dl_techniques.models.language.zamba2 import model as zamba2_module
from dl_techniques.models.language.zamba2.model import Zamba2Model, create_zamba2

from ..headed_logit_decoder_cases import (
    CLF,
    NER,
    HeadedLogitDecoderCases,
    head_batch,
    relative_paths,
    to_np,
)

#: Two 'g' positions on ONE physical block (round-robin reuse) between mamba
#: blocks: every kind of sub-layer the model has is present.
KW = dict(
    hidden_size=32, layer_mapping=["m", "g", "m", "g"], num_mem_blocks=1,
    num_heads=4, max_seq_len=16, d_state=8, headdim=8,
)


class TestZamba2IncludeTop(HeadedLogitDecoderCases):
    CLS = Zamba2Model
    MODULE = zamba2_module
    #: embedding, one attention + one MLP mem-block (RoPE caches included), two mamba blocks, final norm.
    N_BACKBONE = 32
    MISFIT = dict(hidden_size=64)

    def model(self, **kw: Any) -> keras.Model:
        return Zamba2Model(**dict(KW, vocab_size=64, **kw))

    def factory(self, **kw: Any) -> keras.Model:
        return create_zamba2("zamba2_mini", **dict(KW, vocab_size=64, **kw))

    def logits_from_hidden(self, twin: keras.Model, hidden: np.ndarray):
        return np.einsum("bld,vd->blv", hidden, to_np(twin.embedding.weights[0]))

    def test_the_backbone_paths_are_the_same_in_every_top(self):
        """Headless paths == headed paths minus ``task_head/``, per task."""
        def paths(**kw):
            model = self.model(**kw)
            model(head_batch(), training=False)
            return relative_paths(model)

        headless = paths()
        for task in (CLF, NER):
            headed = paths(include_top=True, task_config=task)
            assert [p for p in headed if not p.startswith("task_head/")] == headless

    def test_the_shared_blocks_still_run_once_per_position_under_a_head(self):
        """The head changes nothing about the stack: the shared attention
        block is applied at both 'g' positions, so a change to its weights
        moves the head output."""
        model = self.model(include_top=True, task_config=CLF)
        batch = head_batch()
        before = to_np(model(batch, training=False))
        shared = model.mem_attention_blocks[0]
        for w in shared.weights:
            w.assign(np.asarray(w) * 0.0 + 0.05)
        assert not np.allclose(to_np(model(batch, training=False)), before)

    def test_a_stack_without_a_g_position_takes_a_head(self):
        """No shared attention, so no RoPE limit: a longer input than
        ``max_seq_len`` is legal headless and headed alike."""
        kw = dict(KW, layer_mapping=["m", "m"])
        ids = np.random.default_rng(5).integers(1, 64, (2, 24)).astype("int32")
        headed = Zamba2Model(vocab_size=64, **kw, include_top=True, task_config=CLF)
        assert tuple(headed(ids, training=False).shape) == (2, 3)

    def test_over_long_input_is_still_refused_under_a_head(self):
        """The step-4c guard holds for a headed model: ``build`` refuses a
        static length above ``max_seq_len`` before any variable exists."""
        model = self.model(include_top=True, task_config=CLF)
        with pytest.raises(ValueError, match="exceeds max_seq_len"):
            model.build((2, 17))
        assert not model.weights
        legal = np.random.default_rng(6).integers(1, 64, (2, 12)).astype("int32")
        assert tuple(model(legal, training=False).shape) == (2, 3)
