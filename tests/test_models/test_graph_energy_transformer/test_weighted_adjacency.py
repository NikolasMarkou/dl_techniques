"""Proven-RED fp16/XLA BACKWARD-pass guard for the weighted-adjacency projector.

**This file's load-bearing test is the fp16/XLA training guard for the trainable
:class:`WeightedAdjacencyProjector`** (paper eq. 25 / App D.1, the Branch-A weighted
adjacency wired in steps 1-2 of plan ``plan-2026-07-15T053724-78001af1``). It is the direct
analog of the sibling ``test_model.py::TestFp16XlaTrainingGuard``, but its DEAD COMPONENT is
the projector, not the norm: a projector whose ``Ŵ`` carries no gradient trains NOTHING while
the rest of the model happily trains, produces finite loss, and looks alive — the exact
"a feature that silently does nothing" trap (LESSONS: a forward-only fp16 test certifies
nothing; run the BACKWARD pass and assert weights MOVE; prove the guard RED with a dead
component; never test a reduction at a toy N).

**Why a MINIMAL model, built directly on** :class:`EnergyTransformer`. Step 4 wires the flag
into ``GraphClassifier``; this guard must NOT depend on that wiring, so it builds the smallest
trainable graph model by hand: ``Dense`` token embed -> one ``EnergyTransformer`` block
(``use_weighted_adjacency=True``) fed a rank-3 ``(B, N, N)`` binary adjacency as its
``attention_mask`` -> mean-pool -> a small classification head.

**The block runs in its fp32 variable dtype** (``dtype=variable_dtype`` under the
``mixed_float16`` policy — the same D-002 protection the shipped graph backbone applies). That
is the INTENDED step-4 wiring, and it keeps ``EnergyLayerNorm``'s ``(var+eps)^-1.5`` backward
out of fp16 (which would otherwise overflow on the all-zero PAD tokens and mask the projector
signal). The projector's ``Conv2D`` therefore also runs in fp32, so its ``X⊗X`` never
overflows fp16 here — a genuine projector fp16 overflow would be a NEW finding, distinct from
the known ``EnergyLayerNorm`` one.

**Proven-RED.** Two runs from the SAME minimal config:

* GREEN — the shipped projector. It must TRAIN: over ``FIT_STEPS`` real fp16/XLA ``fit``
  steps the projector's ``Conv2D`` (and token-proj) weights MOVE by ``> 0`` AND the dynamic
  loss scale does NOT collapse.
* RED — a DEAD-COMPONENT injection (:class:`_DeadProjector`) that ``stop_gradient``s ``Ŵ`` so
  the projector receives NO gradient. It must FREEZE: the projector weights move by EXACTLY
  ``0.0``. If the dead projector still trained, the RED assertion FAILS LOUDLY — a guard that
  cannot go red is itself a bug.

Unlike the sibling norm-overflow guard, the RED here is a ``stop_gradient`` (a
device-independent dead path), so it bites on CPU and GPU alike and needs no GPU gate. The
GREEN test is still meaningful only where ``mixed_float16`` actually computes in fp16 + XLA
(a GPU); on a CPU-only box it runs in fp32 (still a valid, if weaker, backward-pass check).
The CI harness runs with ``CUDA_VISIBLE_DEVICES=1``.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.transformers.energy_transformer import (
    EnergyTransformer,
    WeightedAdjacencyProjector,
)
from dl_techniques.models.graph_energy_transformer.model import (
    GraphAnomalyDetector,
    GraphEnergyTransformerBackbone,
    create_graph_anomaly_detector,
    GRAPH_BACKBONE_NAME,
)
from dl_techniques.datasets.graphs.fraud import build_fraud_subgraph_dataset

# --- geometry. N >= 64 WITH PAD rows. DO NOT SHRINK N. ----------------------
N_REAL = 64
N_PAD = 16
N = N_REAL + N_PAD                    # 80 tokens, a fifth of them PAD (the norm-overflow hazard)
F = 8                                 # node-feature dim
EMBED_DIM = 32
NUM_HEADS = 4
HEAD_DIM = 8
HOPFIELD_DIM = 64
NUM_STEPS = 3                         # T; a small T keeps the fp16/XLA fit fast on GPU1
ADJ_PROJ_DIM = 8                      # the OOM escape hatch: conv sees 8^2 = 64 channels, not 32^2
NUM_CLASSES = 3
BATCH = 8

# 20 fp16 fit steps: a dead run halves the loss scale on all 20 -> 2**15 / 2**20 = 0.03, a
# healthy run never halves it and stays at 2**15 = 32768 -> a ~1e6x separation. The threshold
# below sits squarely between them (mirrors the sibling guard).
FIT_STEPS = 20
INITIAL_LOSS_SCALE = 2.0 ** 15
COLLAPSE_THRESHOLD = INITIAL_LOSS_SCALE / 32.0    # 1024: below = collapsed, above = healthy

_PROJ_NEEDLE = "weighted_adjacency_projector"     # matches both the conv and the token-proj


# ---------------------------------------------------------------------------
# The DEAD-COMPONENT injection (test-only): a projector whose Ŵ carries NO gradient.
# ---------------------------------------------------------------------------


class _DeadProjector(WeightedAdjacencyProjector):
    """RED control: ``stop_gradient`` on ``Ŵ`` so the projector receives no gradient.

    The forward VALUES are unchanged (the attention still sees the exact same ``Ŵ``), so the
    model trains, the loss is finite, and everything LOOKS alive — but ``d(loss)/d(conv)`` is
    zero, so the projector's own ``Conv2D``/``Dense`` weights never move. This isolates
    exactly the "the projector trains" property the GREEN guard asserts: a single-variable
    dead path, device-independent (it bites on CPU and GPU alike).
    """

    def call(self, inputs):
        return keras.ops.stop_gradient(super().call(inputs))


# ---------------------------------------------------------------------------
# The minimal trainable model wrapping ONE EnergyTransformer block, flag ON.
# ---------------------------------------------------------------------------


class _WeightedAdjacencyModel(keras.Model):
    """Dense embed -> EnergyTransformer(use_weighted_adjacency=True) -> mean-pool -> head.

    The rank-3 ``(B, N, N)`` adjacency is fed as the block's ``attention_mask`` (D-006: a
    pair-level key x query keep), which is exactly what activates the Branch-A weighted path
    (``_binary_adjacency`` returns it as-is, the projector turns it into ``Ŵ``).
    """

    def __init__(self, dead_projector: bool = False, **kwargs):
        # A leading-underscore auto name ('_weighted_adjacency_model') is an invalid XLA root
        # scope; give the model an explicit valid name.
        kwargs.setdefault("name", "weighted_adjacency_model")
        super().__init__(**kwargs)
        # The block runs in its fp32 VARIABLE dtype even under the mixed_float16 policy — the
        # D-002 protection the shipped graph backbone applies (keeps the norm's -1.5 backward
        # and the projector's X⊗X out of fp16). The embed/head stay on the fp16 policy so the
        # LossScaleOptimizer is genuinely exercised.
        block_dtype = keras.mixed_precision.global_policy().variable_dtype

        self.embed = keras.layers.Dense(EMBED_DIM, name="token_embed")
        self.block = EnergyTransformer(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            hopfield_dim=HOPFIELD_DIM,
            num_steps=NUM_STEPS,
            step_size=0.05,
            attn_self=True,               # real nodes may attend themselves -> no fully-masked query
            noise_std=0.0,
            use_weighted_adjacency=True,
            adjacency_proj_dim=ADJ_PROJ_DIM,
            name="et_block",
            dtype=block_dtype,
        )
        if dead_projector:
            # Swap in the dead projector BEFORE the block is built (lazy build happens on the
            # first forward), so the block builds and tracks the frozen one.
            self.block.adjacency_projector = _DeadProjector(
                num_heads=NUM_HEADS,
                embed_dim=EMBED_DIM,
                proj_dim=ADJ_PROJ_DIM,
                name=_PROJ_NEEDLE,
                dtype=block_dtype,
            )
        # fp32 head so the logits (and the loss) are stable regardless of the fp16 policy.
        self.head = keras.layers.Dense(NUM_CLASSES, name="head", dtype="float32")

    def call(self, inputs):
        x = self.embed(inputs["node_features"])
        x = self.block(x, attention_mask=inputs["adjacency"])
        x = keras.ops.mean(x, axis=1)                  # mean-pool over tokens (no CLS)
        x = keras.ops.cast(x, "float32")
        return self.head(x)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _gpu_available() -> bool:
    return len(tf.config.list_physical_devices("GPU")) > 0


def _padded_graph_batch(seed: int) -> dict:
    """``BATCH`` graphs, ``N_REAL`` real + ``N_PAD`` PAD nodes each.

    PAD nodes carry ZERO features -> after the embed they are near-constant tokens (the
    ``EnergyLayerNorm`` fp16-backward hazard). The adjacency connects only real nodes and adds
    a self-loop on every real node (so no real query column is fully masked).
    """
    rng = np.random.default_rng(seed)
    node_features = np.zeros((BATCH, N, F), dtype="float32")
    node_features[:, :N_REAL, :] = rng.normal(size=(BATCH, N_REAL, F)).astype("float32")

    adjacency = np.zeros((BATCH, N, N), dtype="float32")
    sub = (rng.random((BATCH, N_REAL, N_REAL)) < 0.15).astype("float32")
    adjacency[:, :N_REAL, :N_REAL] = sub
    idx = np.arange(N_REAL)
    adjacency[:, idx, idx] = 1.0                       # a self-loop on every real node

    return {"node_features": node_features, "adjacency": adjacency}


def _labels(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, NUM_CLASSES, size=(BATCH,)).astype("int32")


def _projector_snapshot(model) -> dict:
    return {
        v.path: keras.ops.convert_to_numpy(v).astype("float64")
        for v in model.trainable_weights
        if _PROJ_NEEDLE in v.path
    }


def _run_fit(model, inputs, y):
    """Stock ``compile(jit_compile=True)`` + a real ``fit`` over ``FIT_STEPS`` steps.

    Returns ``(loss_scale, projector_deltas, loss_finite)`` where
    ``projector_deltas[path] = max|dW|`` for each projector weight. NO ``train_step`` override
    — the fp16/XLA backward pass is the whole point.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        jit_compile=True,               # XLA is half the hazard; exercise it explicitly
    )
    assert "LossScale" in type(model.optimizer).__name__, "mixed_float16 policy is not active"

    ds = tf.data.Dataset.from_tensor_slices((inputs, y)).batch(BATCH).repeat()
    before = _projector_snapshot(model)
    assert before, "no projector weights found — the weighted path is not wired"
    history = model.fit(ds, steps_per_epoch=FIT_STEPS, epochs=1, verbose=0)
    after = _projector_snapshot(model)

    loss_finite = bool(np.all(np.isfinite(history.history["loss"])))
    scale = float(keras.ops.convert_to_numpy(model.optimizer.dynamic_scale))
    deltas = {p: float(np.max(np.abs(after[p] - before[p]))) for p in before}
    return scale, deltas, loss_finite


# ---------------------------------------------------------------------------


@pytest.fixture
def mixed_f16():
    """Set the GLOBAL mixed_float16 policy for one test, then restore it."""
    previous = keras.mixed_precision.global_policy()
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        yield
    finally:
        keras.mixed_precision.set_global_policy(previous)


class TestWeightedAdjacencyFp16XlaGuard:
    """The load-bearing guard: does the projector actually TRAIN under fp16/XLA?

    GREEN (shipped projector) trains; RED (stop-gradient dead projector) freezes. Both at
    ``N >= 64`` with PAD rows, the weighted path ACTIVE.
    """

    def test_projector_trains_under_mixed_float16(self, mixed_f16):
        """GREEN: projector weights MOVE AND the loss scale holds."""
        keras.utils.set_random_seed(13)
        model = _WeightedAdjacencyModel(dead_projector=False)
        model(_padded_graph_batch(seed=0))            # build

        assert model.embed.compute_dtype == "float16", "mixed_float16 policy is not active"
        # The D-002 protection: the block (and thus its projector) runs in fp32 variable dtype.
        assert model.block.compute_dtype == "float32"
        assert model.block.adjacency_projector.conv.compute_dtype == "float32"

        inputs = _padded_graph_batch(seed=1)
        scale, deltas, loss_finite = _run_fit(model, inputs, _labels(2))
        print(
            f"\n[fp16 GREEN] loss_scale={scale:.1f} loss_finite={loss_finite} "
            f"conv_kernel|dW|={max(d for p, d in deltas.items() if 'pair_conv' in p and 'kernel' in p):.3e} "
            f"max|dW|(all projector)={max(deltas.values()):.3e}"
        )

        assert loss_finite, (
            "loss went non-finite — a projector fp16 OVERFLOW or a DIFFERENT bug from the "
            "silent freeze this guards. STOP and report (do not paper over)."
        )
        assert scale >= COLLAPSE_THRESHOLD, (
            f"loss scale COLLAPSED to {scale:.3e} (start {INITIAL_LOSS_SCALE:.0f}) — the "
            "weighted projector froze the model under fp16/XLA"
        )
        moved = {p: d for p, d in deltas.items() if d > 0.0}
        assert moved == deltas and deltas, (
            f"a projector weight did NOT move over {FIT_STEPS} fp16 fit steps: {deltas}. "
            "The trainable Ŵ projector is training on nothing (silent no-op)."
        )
        # named-needle anti-vacuity: the Conv2D kernel is the core projector weight.
        conv_moves = [d for p, d in deltas.items() if "pair_conv" in p and "kernel" in p]
        assert conv_moves and max(conv_moves) > 0.0, (
            f"the projector Conv2D kernel did not move (moves={conv_moves}) — Ŵ carries no "
            "gradient, the weighted-adjacency feature trains nothing"
        )

    def test_dead_projector_freezes(self, mixed_f16):
        """RED: a stop-gradient projector must FREEZE (proves the GREEN guard bites)."""
        keras.utils.set_random_seed(13)
        model = _WeightedAdjacencyModel(dead_projector=True)
        model(_padded_graph_batch(seed=0))            # build

        assert isinstance(model.block.adjacency_projector, _DeadProjector)
        assert model.block.compute_dtype == "float32"

        inputs = _padded_graph_batch(seed=1)
        scale, deltas, loss_finite = _run_fit(model, inputs, _labels(2))
        print(
            f"\n[fp16 RED dead] loss_scale={scale:.1f} loss_finite={loss_finite} "
            f"max|dW|(all projector)={max(deltas.values()):.3e} "
            f"frozen={sum(1 for d in deltas.values() if d == 0.0)}/{len(deltas)}"
        )

        # The dead path does NOT collapse the loss scale (the rest of the model still trains);
        # the signal is that the PROJECTOR weights are pinned.
        assert loss_finite, "dead-projector run went non-finite — a different failure"
        frozen = sorted(p for p, d in deltas.items() if d == 0.0)
        assert len(frozen) == len(deltas), (
            f"dead-projector RED did NOT bite: only {len(frozen)}/{len(deltas)} projector "
            f"weights froze (deltas={deltas}). A stop_gradient projector still trained, so "
            "the GREEN 'weights move' guard cannot go red — the guard is unverifiable."
        )


# ===========================================================================
# Variant-B (GraphAnomalyDetector / descend_capture) dead-projector guard.
#
# The guard above exercises the CLASSIFIER path (`block.__call__`), which was always
# correct. Variant B does NOT use `block.__call__` — it runs the block's descent MANUALLY
# through `GraphEnergyTransformerBackbone.descend_capture`. Before plan step 6 that hand-rolled
# loop called `block.attention.update(...)` WITHOUT `adjacency_weight` and never computed `Ŵ`,
# so a `GraphAnomalyDetector(use_weighted_adjacency=True)` silently ran the BINARY path: its
# projector `Conv2D` was built (via `backbone.build -> block.build`) and SERIALIZED, yet
# received no gradient and trained NOTHING — the exact "feature that does nothing" trap.
#
# Proven-RED: the fixed `descend_capture` (forwards the hoisted `Ŵ`) MOVES the projector; a
# faithful reproduction of the PRE-FIX loop (`_LegacyDescendBackbone`, no `Ŵ` forward) FREEZES
# it. This RED is not a stop_gradient trick — it runs the literal old code path, so it fails
# LOUDLY against the pre-fix source and is what makes the fix load-bearing.
# ===========================================================================

# --- variant-B geometry: synthetic fraud subgraphs, small + fast on GPU1. ---
B_MAX_NODES = 32
B_FEAT = 8
B_EMBED = 32
B_HEADS = 4
B_HEAD_DIM = 8
B_HOPFIELD = 32
B_STEPS = 4                           # T; the head captures g_1 and g_T
B_PROJ_DIM = 8                        # OOM escape hatch: conv sees 8^2 = 64 channels
B_MLP_HIDDEN = 16
B_BATCH = 8
B_FIT_STEPS = 15


def _legacy_descend_capture(backbone, tokens, adjacency_mask, node_mask_2d, capture_steps):
    """The PRE-FIX ``descend_capture`` body — the binary path with NO ``Ŵ`` forward.

    A verbatim reproduction of the loop step 6 replaced: ``block.attention.update`` is called
    without ``adjacency_weight`` and ``_weighted_adjacency`` is never invoked. Shared by the
    RED backbone below and the trajectory-diff test so the "old behaviour" lives in ONE place.
    """
    block = backbone.blocks[0]
    capture = set(int(t) for t in capture_steps)
    x = keras.ops.cast(tokens, block.compute_dtype)
    captured = {}
    for t in range(1, backbone.num_steps + 1):
        g = block.norm(x)
        update = (
            block.attention.update(g, attention_mask=adjacency_mask, mask=node_mask_2d)
            + block.hopfield.update(g, mask=node_mask_2d)
        )
        x = x + block.step_size * update
        if t in capture:
            captured[t] = keras.ops.cast(block.norm(x), backbone.compute_dtype)
    return captured


class _LegacyDescendBackbone(GraphEnergyTransformerBackbone):
    """RED control: the backbone with the PRE-FIX ``descend_capture`` (never forwards ``Ŵ``).

    Every other line matches the shipped backbone, so the projector is still created and BUILT
    (the silent-serialization half of the bug); the only difference is that ``Ŵ`` never reaches
    ``attention.update`` — so the projector gets no gradient and must FREEZE under ``fit``.
    """

    def descend_capture(self, tokens, adjacency_mask, node_mask_2d, capture_steps,
                        training=None):
        return _legacy_descend_capture(
            self, tokens, adjacency_mask, node_mask_2d, capture_steps
        )


def _make_variant_b(backbone_cls=GraphEnergyTransformerBackbone):
    """A variant-B :class:`GraphAnomalyDetector` (flag ON) on ``backbone_cls``.

    Mirrors :func:`create_graph_anomaly_detector`'s fixed topology (num_blocks=1, no CLS/PE,
    noise_std=0) but lets a test swap in the RED backbone.
    """
    backbone = backbone_cls(
        node_feature_dim=B_FEAT,
        embed_dim=B_EMBED,
        num_heads=B_HEADS,
        head_dim=B_HEAD_DIM,
        hopfield_dim=B_HOPFIELD,
        num_blocks=1,
        num_steps=B_STEPS,
        use_pe=False,
        use_cls=False,
        noise_std=0.0,
        attn_self=True,
        use_weighted_adjacency=True,
        adjacency_proj_dim=B_PROJ_DIM,
        name=GRAPH_BACKBONE_NAME,
    )
    return GraphAnomalyDetector(backbone=backbone, mlp_hidden_dim=B_MLP_HIDDEN)


def _variant_b_dataset(seed: int, repeat: bool = True):
    """A synthetic variant-B fraud-subgraph dataset with ``(B, 1)`` float labels for BCE."""
    ds, meta = build_fraud_subgraph_dataset(
        synthetic=True,
        split="train",
        batch_size=B_BATCH,
        max_nodes=B_MAX_NODES,
        n_synth_nodes=400,
        synth_num_features=B_FEAT,
        synth_anomaly_ratio=0.3,
        seed=seed,
        shuffle=False,
    )
    assert meta["num_features"] == B_FEAT
    ds = ds.map(lambda x, y: (x, tf.cast(tf.reshape(y, (-1, 1)), tf.float32)))
    return ds.repeat() if repeat else ds


def _b_projector_snapshot(model) -> dict:
    return {
        v.path: keras.ops.convert_to_numpy(v).astype("float64")
        for v in model.trainable_weights
        if _PROJ_NEEDLE in v.path
    }


def _fit_variant_b(model, seed: int):
    """Stock ``compile`` + a short ``fit``; returns ``(projector_deltas, loss_finite)``."""
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
    )
    ds = _variant_b_dataset(seed)
    # One forward builds the backbone (and thus the projector, via block.build) before snapshot.
    for xb, _ in ds.take(1):
        model(xb)
    before = _b_projector_snapshot(model)
    assert before, (
        "no projector weights on the variant-B model — block.build did not build the "
        "weighted-adjacency projector (the silent-serialization half of the bug)"
    )
    history = model.fit(ds, steps_per_epoch=B_FIT_STEPS, epochs=1, verbose=0)
    after = _b_projector_snapshot(model)
    loss_finite = bool(np.all(np.isfinite(history.history["loss"])))
    deltas = {p: float(np.max(np.abs(after[p] - before[p]))) for p in before}
    return deltas, loss_finite


class TestVariantBDescendCaptureWeightedAdjacency:
    """Does the projector actually TRAIN on the variant-B ``descend_capture`` path?

    GREEN (fixed backbone, forwards ``Ŵ``) trains; RED (``_LegacyDescendBackbone``, the literal
    pre-fix loop) freezes. Both a real ``GraphAnomalyDetector`` fit on synthetic fraud subgraphs.
    """

    def test_projector_trains_on_variant_b(self):
        """GREEN: the fixed descend_capture forwards Ŵ -> the projector Conv2D MOVES."""
        keras.utils.set_random_seed(7)
        model = _make_variant_b()
        deltas, loss_finite = _fit_variant_b(model, seed=0)
        conv_moves = [d for p, d in deltas.items() if "pair_conv" in p and "kernel" in p]
        print(
            f"\n[variant-B GREEN] loss_finite={loss_finite} "
            f"conv_kernel|dW|={max(conv_moves):.3e} max|dW|(all projector)={max(deltas.values()):.3e}"
        )
        assert loss_finite, "variant-B fit went non-finite — a different failure, STOP and report"
        assert conv_moves and max(conv_moves) > 0.0, (
            f"the projector Conv2D kernel did NOT move on variant B (moves={conv_moves}) — "
            "descend_capture is running the binary path, the weighted-adjacency projector is a "
            "silent dead feature (the exact defect step 6 fixes)"
        )
        assert all(d > 0.0 for d in deltas.values()), (
            f"a projector weight stayed frozen on variant B: {deltas}"
        )

    def test_legacy_descend_freezes_projector(self):
        """RED: the PRE-FIX descend loop (no Ŵ forward) FREEZES the projector.

        Proves the GREEN guard bites: run against the literal old code path and the projector
        weights are pinned at EXACTLY 0.0 despite being built and serialized.
        """
        keras.utils.set_random_seed(7)
        model = _make_variant_b(backbone_cls=_LegacyDescendBackbone)
        deltas, loss_finite = _fit_variant_b(model, seed=0)
        frozen = sorted(p for p, d in deltas.items() if d == 0.0)
        print(
            f"\n[variant-B RED legacy] loss_finite={loss_finite} "
            f"frozen={len(frozen)}/{len(deltas)} max|dW|={max(deltas.values()):.3e}"
        )
        assert loss_finite, "legacy variant-B fit went non-finite — a different failure"
        assert len(frozen) == len(deltas), (
            f"the pre-fix descend_capture RED did NOT bite: only {len(frozen)}/{len(deltas)} "
            f"projector weights froze (deltas={deltas}). If the projector still trained without "
            "the Ŵ forward, the GREEN 'weights move' guard is unverifiable."
        )

    def test_weighted_adjacency_reaches_the_descent(self):
        """The flag actually changes the trajectory: fixed descend_capture (forwards Ŵ) differs
        from the pre-fix loop (no Ŵ) on the SAME model weights -> Ŵ genuinely reaches the descent.
        """
        keras.utils.set_random_seed(7)
        model = create_graph_anomaly_detector(
            node_feature_dim=B_FEAT,
            embed_dim=B_EMBED,
            num_heads=B_HEADS,
            head_dim=B_HEAD_DIM,
            hopfield_dim=B_HOPFIELD,
            mlp_hidden_dim=B_MLP_HIDDEN,
            num_steps=B_STEPS,
            attn_self=True,
            use_weighted_adjacency=True,
            adjacency_proj_dim=B_PROJ_DIM,
        )
        ds = _variant_b_dataset(seed=1, repeat=False)
        inputs, _ = next(iter(ds))
        model(inputs)                                   # build all weights

        backbone = model.backbone
        x0 = backbone.embed(inputs)
        adj = inputs["adjacency"]
        node_mask = inputs.get("node_mask", None)
        steps = {1, backbone.num_steps}

        caps_fixed = backbone.descend_capture(x0, adj, node_mask, steps)     # forwards Ŵ
        caps_legacy = _legacy_descend_capture(backbone, x0, adj, node_mask, steps)  # no Ŵ

        diff_T = float(
            np.max(np.abs(
                keras.ops.convert_to_numpy(caps_fixed[backbone.num_steps])
                - keras.ops.convert_to_numpy(caps_legacy[backbone.num_steps])
            ))
        )
        print(f"\n[variant-B Ŵ-reaches-descent] max|g_T diff|={diff_T:.3e}")
        assert diff_T > 1e-6, (
            f"the weighted adjacency did NOT change the descent trajectory (max|g_T diff|="
            f"{diff_T:.3e}) — Ŵ is not reaching attention.update in descend_capture"
        )
