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


def _tiny_clip(text_stochastic_depth_rate: float = 0.0) -> CliffordCLIP:
    """Smallest CliffordCLIP that still has a real text-block ladder.

    ``dropout_rate=0.0`` everywhere, so when
    ``text_stochastic_depth_rate > 0`` the text drop-paths are the ONLY
    stochastic source in the wrapper — which is what lets
    :func:`test_text_lm_wrapper_applies_drop_path` attribute run-to-run
    variance to them and nothing else.
    """
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
        text_stochastic_depth_rate=text_stochastic_depth_rate,
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


def test_text_lm_wrapper_applies_external_residual():
    """The wrapper's block ladder must match ``encode_text``'s exactly.

    The text blocks are TRANSFORM-ONLY — the residual add is external and each
    block is near-zero at init (LayerScale gamma=1e-5). Applying them as
    ``x = block(x)`` replaces the signal with ~1e-5 of itself per block rather
    than augmenting it, annihilating it (measured 2.76 -> 3.05e-12 over four
    blocks, underflowing to 0.0 at realistic depths) and making the CLM
    pretraining phase optimize a different function from the contrastive phase
    that later reuses these weights.
    """
    from keras import ops

    m = _tiny_clip()
    m.build({"image": (None, 32, 32, 3), "text": (None, CONTEXT)})
    wrapper = _TextLMWrapper(m, vocab_size=VOCAB, context_length=CONTEXT)

    ids = np.random.default_rng(0).integers(
        0, VOCAB, size=(2, CONTEXT)
    ).astype("int32")
    got = wrapper(ids, training=False)["logits"]

    # Recompute encode_text's ladder locally and push it through the SAME
    # lm_head, so the ONLY thing under test is the block loop itself.
    positions = ops.arange(CONTEXT)
    x = m.token_embedding(ids) + m.position_embedding(positions)
    x = m.text_embed_norm(x)
    x = m.text_embed_dropout(x, training=False)
    for block, drop_path in zip(m.text_blocks, m.text_drop_paths):
        x = x + drop_path(block(x, training=False), training=False)
    pre_head = m.text_head_norm(x)
    if m.text_head_dropout is not None:
        pre_head = m.text_head_dropout(pre_head, training=False)
    want = wrapper.lm_head(pre_head)

    delta = float(ops.max(ops.abs(got - want)))
    assert delta == 0.0, (
        f"_TextLMWrapper's block ladder diverged from CliffordCLIP.encode_text: "
        f"max|delta| = {delta:.6e}, expected exactly 0.0. Keep the loop as "
        f"``x = x + drop_path(block(x, training=training), training=training)`` "
        f"over zip(m.text_blocks, m.text_drop_paths)."
    )

    # Independent of the comparison above: the signal must survive the ladder.
    # A residual-free loop collapses it by ~1e-5 per block, so this catches the
    # regression even if the reference were computed the same wrong way.
    embed_absmax = float(ops.max(ops.abs(
        m.text_embed_norm(
            m.token_embedding(ids) + m.position_embedding(positions)
        )
    )))
    pre_head_absmax = float(ops.max(ops.abs(pre_head)))
    assert pre_head_absmax > 0.01 * embed_absmax, (
        f"text-block ladder annihilated the signal: pre-head absmax "
        f"{pre_head_absmax:.6e} vs embedding absmax {embed_absmax:.6e}. The "
        f"blocks are transform-only; the external residual add is required."
    )


def test_text_lm_wrapper_applies_drop_path():
    """The wrapper must route the blocks through the model's own drop-paths.

    With ``dropout_rate=0.0`` the text drop-paths are the only stochastic
    source, so a wrapper that skipped them would be deterministic under
    ``training=True``.
    """
    from keras import ops

    m = _tiny_clip(text_stochastic_depth_rate=0.5)
    m.build({"image": (None, 32, 32, 3), "text": (None, CONTEXT)})
    wrapper = _TextLMWrapper(m, vocab_size=VOCAB, context_length=CONTEXT)

    rates = [dp.drop_path_rate for dp in m.text_drop_paths]
    assert max(rates) > 0.0, (
        f"fixture did not produce a live drop-path rate: {rates}. Without one "
        f"this test cannot distinguish a present from an absent drop_path."
    )

    ids = np.random.default_rng(0).integers(
        0, VOCAB, size=(2, CONTEXT)
    ).astype("int32")
    draws = [
        float(ops.max(ops.abs(wrapper(ids, training=True)["logits"])))
        for _ in range(8)
    ]
    assert len(set(draws)) > 1, (
        f"_TextLMWrapper produced identical output across 8 training-mode "
        f"draws ({draws[0]:.6e}) with drop_path rates {rates}. The block loop "
        f"is bypassing m.text_drop_paths."
    )
