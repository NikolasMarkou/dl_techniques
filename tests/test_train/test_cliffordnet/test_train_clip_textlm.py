"""Guard for the fourth caller-side rank-wrap site (D-003).

``_TextLMWrapper.call`` in ``src/train/cliffordnet/train_clip.py`` hand-copies
:meth:`CliffordCLIP.encode_text` for the CLM pretraining phase, so it carried
its own copy of the caller-side ``expand_dims(axis=1)`` / ``squeeze(axis=1)``
wrap around the text-block loop. That copy was deleted; this module is the only
durable evidence that it stays deleted, because reintroducing it is numerically
INERT in eager execution — no output-value test can catch it. We observe what
actually ARRIVES at the block instead.

This is the first test module under ``tests/test_train/test_cliffordnet/``; the
site previously had zero coverage of any kind.
"""

import numpy as np

from dl_techniques.layers.geometric.clifford_block import CliffordNetBlock
from dl_techniques.models.clip.clifford_clip import CliffordCLIP
from train.cliffordnet.train_clip import _TextLMWrapper

VOCAB = 64
CONTEXT = 8


def _tiny_clip() -> CliffordCLIP:
    """Smallest CliffordCLIP that still has a real text-block ladder."""
    return CliffordCLIP(
        image_size=32,
        vision_patch_size=4,
        vision_stage_channels=[8, 8],
        vision_stage_depths=[1, 1],
        vision_stochastic_depth_rate=0.0,
        vocab_size=VOCAB,
        context_length=CONTEXT,
        text_channels=16,
        text_depth=2,
        text_stochastic_depth_rate=0.0,
        embed_dim=16,
        dropout_rate=0.0,
    )


def test_text_lm_wrapper_feeds_blocks_rank_3(monkeypatch):
    """``_TextLMWrapper.call`` must hand the text blocks native rank-3."""
    m = _tiny_clip()
    m.build({"image": (None, 32, 32, 3), "text": (None, CONTEXT)})
    wrapper = _TextLMWrapper(m, vocab_size=VOCAB, context_length=CONTEXT)

    ids = np.random.default_rng(0).integers(
        0, VOCAB, size=(2, CONTEXT)
    ).astype("int32")
    # Warm-up OUTSIDE the recorder: the first ``wrapper(...)`` runs ``call``
    # twice (Keras builds, then invokes), which would make the expected block
    # count an artefact of Keras' build protocol rather than of the wrapper.
    wrapper(ids, training=False)

    seen = []
    original_call = CliffordNetBlock.call

    def recording_call(self, inputs, training=None):
        seen.append((self.name, len(inputs.shape)))
        return original_call(self, inputs, training=training)

    # ``CausalCliffordNetBlock`` does not override ``call``, so patching the
    # base class intercepts the text blocks too.
    monkeypatch.setattr(CliffordNetBlock, "call", recording_call)

    out = wrapper(ids, training=False)
    assert tuple(out["logits"].shape) == (2, CONTEXT, VOCAB)

    assert len(seen) == len(m.text_blocks) == 2, (
        f"recorder saw {len(seen)} block call(s), expected one per text block "
        f"({len(m.text_blocks)}): {seen}. An empty/short recorder means the "
        f"patch did not intercept — the guard would be vacuous."
    )
    bad = [(name, rank) for name, rank in seen if rank != 3]
    assert not bad, (
        f"_TextLMWrapper fed the CliffordCLIP text blocks non-rank-3 input: "
        f"{bad}. That tower is sequence mode — do not reintroduce the "
        f"caller-side expand_dims(axis=1)/squeeze(axis=1) in "
        f"_TextLMWrapper.call (D-003)."
    )
