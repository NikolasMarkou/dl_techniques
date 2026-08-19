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
from dl_techniques.layers.stochastic_depth import StochasticDepth
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


def test_text_lm_wrapper_applies_drop_path(monkeypatch):
    """The wrapper must route the blocks through the model's OWN drop-paths.

    Observed at the ground truth: every ``StochasticDepth`` invocation is
    recorded, checked to be one of ``m.text_drop_paths`` BY IDENTITY, and
    reduced to the per-sample keep/drop realization its Bernoulli mask
    produced. A wrapper that bypasses ``m.text_drop_paths`` records nothing.

    Do NOT reduce this to ``ops.max(ops.abs(logits))`` over the batch again
    (D-061): ``StochasticDepth`` draws ONE independent mask per sample, so a
    surviving sample carrying the batch extremum makes that scalar insensitive
    to the drop. Measured 5 failures / 74 fresh-process runs (~6.8%) on the
    scalar-max form, against a ~5e-10 collapse probability for this one.
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
    # Warm-up OUTSIDE the recorder: the first ``wrapper(...)`` runs ``call``
    # twice (Keras builds, then invokes), so the record count would otherwise
    # be an artefact of Keras' build protocol.
    wrapper(ids, training=False)

    own = {id(dp) for dp in m.text_drop_paths}
    observed = []
    foreign = []
    original_call = StochasticDepth.call

    def recording_call(self, inputs, training=None):
        out = original_call(self, inputs, training=training)
        if id(self) not in own:
            foreign.append(self.name)
            return out
        # Recorded for EVERY invocation, not only training-mode ones: a
        # wrapper that forgets to forward ``training=`` must be distinguished
        # from one that bypasses the drop-paths altogether.
        #
        # A dropped sample is EXACTLY zero (``inputs * 0 / keep_prob``); a
        # kept one is ``inputs / keep_prob``. Read the realization off the
        # output rather than re-deriving the mask, so this oracle does not
        # restate the implementation it is meant to observe.
        flat_in = ops.reshape(inputs, (ops.shape(inputs)[0], -1))
        flat_out = ops.reshape(out, (ops.shape(out)[0], -1))
        live_in = np.asarray(ops.max(ops.abs(flat_in), axis=-1))
        live_out = np.asarray(ops.max(ops.abs(flat_out), axis=-1))
        observed.append((bool(training), self.name, self.drop_path_rate,
                         tuple(float(v) for v in live_in),
                         tuple(bool(v > 0.0) for v in live_out)))
        return out

    monkeypatch.setattr(StochasticDepth, "call", recording_call)

    draws = 16
    for _ in range(draws):
        wrapper(ids, training=True)

    assert not foreign, (
        f"the wrapper routed the text blocks through StochasticDepth layers "
        f"that are NOT m.text_drop_paths: {sorted(set(foreign))}. It must "
        f"reuse the model's own drop-paths so the CLM and contrastive phases "
        f"train one function."
    )
    expected = draws * len(m.text_drop_paths)
    assert len(observed) == expected, (
        f"recorder saw {len(observed)} StochasticDepth call(s) on the model's "
        f"own drop-paths, expected {expected} ({draws} draws x "
        f"{len(m.text_drop_paths)} drop-paths). The block loop is bypassing "
        f"m.text_drop_paths."
    )

    inference_mode = [i for i, rec in enumerate(observed) if not rec[0]]
    assert not inference_mode, (
        f"{len(inference_mode)} of {expected} drop-path invocations ran with "
        f"a falsy training= flag (draw indices {inference_mode[:5]}). The "
        f"wrapper must forward its own training= into drop_path(...), or the "
        f"drop-paths are inert identities during CLM pretraining."
    )

    # Anti-vacuity: "dropped" is only readable as an all-zero output while
    # every sample ARRIVES nonzero. If a block emitted an all-zero sample the
    # keep/drop reading below would be undefined, not merely noisy.
    dead_inputs = [rec for rec in observed if min(rec[3]) <= 0.0]
    assert not dead_inputs, (
        f"{len(dead_inputs)} drop-path input sample(s) arrived all-zero, so "
        f"keep/drop cannot be read off the output: {dead_inputs[:3]}"
    )

    # Pooled PER LAYER, and only over the layers whose rate is live: the
    # linear schedule gives the first drop-path rate 0.0, and a rate-0.0 layer
    # is an identity by design, so pooling all layers together would let one
    # live layer's variation mask another's deadness.
    live = {rec[1] for rec in observed if rec[2] > 0.0}
    assert live, f"no live-rate drop-path was invoked; rates {rates}"
    for name in sorted(live):
        masks = {rec[4] for rec in observed if rec[1] == name}
        assert len(masks) > 1, (
            f"drop-path {name!r} (rate {dict((r[1], r[2]) for r in observed)[name]}) "
            f"realized the SAME per-sample keep/drop mask {masks} on all "
            f"{draws} draws. Either it is running in inference mode "
            f"(training= not forwarded) or the mask is not drawn per call."
        )
