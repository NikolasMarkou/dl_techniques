"""Test suite for the Graph Energy Transformer models (variant B + variant C-lite).

**This file's load-bearing test is the fp16/XLA BACKWARD-pass training guard**
(:class:`TestFp16XlaTrainingGuard`). It is the direct graph-domain analog of the image
suite's ``TestMixedPrecisionBackwardPass.test_mim_actually_trains_under_mixed_float16``,
and it exists because the exact same defect surface bit the image Energy Transformer once:
``EnergyLayerNorm``'s backward forms ``(var + eps)^(-3/2)``, which OVERFLOWS fp16's 65504
under XLA when a token is near-constant (``var -> 0`` with ``eps < 6.1e-4``). The symptom is
SILENT: the loss stays finite, ``TerminateOnNaN`` never fires, and the optimizer rejects
EVERY step so the model trains on NOTHING.

The graph trunk's near-constant-token hazard is a **PAD row**: a padded node has zero
features, so after ``node_proj`` (zero-init bias) it is the all-zero token, whose variance
across the embedding axis is 0. ``EnergyLayerNorm`` runs on ALL tokens (PAD included) BEFORE
the node-mask excludes them from attention, so a PAD token drives the overflow on the very
first descent step. This is why every numerical test here uses ``N >= 64`` **with PAD rows
present** — a toy ``N`` with no PAD would hide the failure exactly as it did in the image
block once (LESSONS: never test a reduction at a toy size; inject a real dead component).

**Proven-RED.** For BOTH heads the guard is run twice from the SAME config:

* GREEN — the SHIPPED model (blocks in the fp32 ``variable_dtype``, the D-002 fix). It must
  train: the dynamic loss scale does NOT collapse AND named trunk weights MOVE.
* RED — a DEAD-COMPONENT injection (:class:`_Fp16UnsafeBackbone`) that overrides the
  backbone's ``_make_block`` seam to build the blocks at the fp16 COMPUTE policy (no fp32
  variable-dtype protection). It must FREEZE: the loss scale collapses and the trunk weights
  move by exactly 0.0. If the broken model still trains, the RED assertion FAILS LOUDLY — a
  guard that cannot go red is itself a bug (LESSONS: guards must be proven RED).

**GPU.** The overflow is a GPU+XLA phenomenon: on a CPU-only box ``mixed_float16`` silently
computes in fp32, so neither the GREEN XLA path nor the RED overflow is exercised. Mirroring
the (ungated) image suite, these tests always RUN; but each RED test SKIPS with a clear reason
if the injection did not bite AND no GPU is visible (that is the CPU-fp32 case, not a real
regression). On a GPU box (the CI harness runs with ``CUDA_VISIBLE_DEVICES=1``) the RED must
bite or it fails.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.transformers.energy_transformer import EnergyTransformer
from dl_techniques.models.graph_energy_transformer.model import (
    GRAPH_BACKBONE_NAME,
    GraphAnomalyDetector,
    GraphClassifier,
    GraphEnergyTransformerBackbone,
)

# --- geometry. N >= 64 WITH PAD rows. DO NOT SHRINK N. ----------------------
N_REAL = 64
N_PAD = 32
N = N_REAL + N_PAD                    # 96 tokens, a third of them PAD (the overflow hazard)
F = 8                                 # node-feature dim
EMBED_DIM = 64
NUM_HEADS = 4
HEAD_DIM = 16
HOPFIELD_DIM = 128
NUM_STEPS = 4                         # T; the overflow bites on step 1, so a small T is decisive
PE_DIM = 8
NUM_CLASSES = 3
MLP_HIDDEN = 64
BATCH = 8

# 20 fp16 fit steps: a dead run halves the loss scale on all 20 -> 2**15 / 2**20 = 0.03, a
# healthy run never halves it and stays at 2**15 = 32768 -> a ~1e6x separation. The threshold
# below sits squarely between them.
FIT_STEPS = 20
INITIAL_LOSS_SCALE = 2.0 ** 15       # keras LossScaleOptimizer default initial scale
COLLAPSE_THRESHOLD = INITIAL_LOSS_SCALE / 32.0    # 1024: below = collapsed, above = healthy


# ---------------------------------------------------------------------------
# The DEAD-COMPONENT injection (test-only). Overrides the ONE production seam.
# ---------------------------------------------------------------------------


class _Fp16UnsafeBackbone(GraphEnergyTransformerBackbone):
    """RED control: build ET blocks at the fp16 COMPUTE policy, NOT the fp32 variable dtype.

    This overrides exactly the ``_make_block`` seam the production backbone exposes, swapping
    ``dtype=self.dtype_policy.variable_dtype`` (fp32 under mixed_float16 — the D-002 fix) for
    ``dtype=self.dtype_policy`` (fp16 compute). ``call()``/``descend_capture()`` then cast the
    tokens to the block's fp16 compute dtype, so ``EnergyLayerNorm``'s backward runs in fp16 and
    ``(var + eps)^(-3/2)`` overflows on the all-zero PAD tokens. Nothing else about the model
    changes — this is a single-variable control isolating the dtype fix.
    """

    def _make_block(self, index: int) -> EnergyTransformer:
        return EnergyTransformer(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            hopfield_dim=self.hopfield_dim,
            num_steps=self.num_steps,
            step_size=self.step_size,
            beta=self.beta,
            attn_self=self.attn_self,
            hopfield_activation=self.hopfield_activation,
            hopfield_beta=self.hopfield_beta,
            noise_std=self.noise_std,
            return_energy=False,
            norm_epsilon=self.norm_epsilon,
            seed=self.seed,
            name=f"et_block_{index}",
            dtype=self.dtype_policy,          # <-- fp16 compute: the (var+eps)^-1.5 overflow
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _gpu_available() -> bool:
    return len(tf.config.list_physical_devices("GPU")) > 0


def _b_backbone_kwargs() -> dict:
    """Variant-B trunk config (num_blocks=1, no CLS/PE, noiseless), matching the B factory."""
    return dict(
        node_feature_dim=F, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, head_dim=HEAD_DIM,
        hopfield_dim=HOPFIELD_DIM, num_blocks=1, num_steps=NUM_STEPS,
        use_pe=False, use_cls=False, noise_std=0.0, name=GRAPH_BACKBONE_NAME,
    )


def _c_backbone_kwargs() -> dict:
    """Variant-C-lite trunk config (num_blocks=4, CLS + Laplacian PE + saddle noise)."""
    return dict(
        node_feature_dim=F, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, head_dim=HEAD_DIM,
        hopfield_dim=HOPFIELD_DIM, num_blocks=4, num_steps=NUM_STEPS, step_size=0.01,
        noise_std=0.02, use_pe=True, pe_dim=PE_DIM, use_cls=True, name=GRAPH_BACKBONE_NAME,
    )


def _b_model(unsafe: bool = False) -> GraphAnomalyDetector:
    backbone_cls = _Fp16UnsafeBackbone if unsafe else GraphEnergyTransformerBackbone
    return GraphAnomalyDetector(
        backbone=backbone_cls(**_b_backbone_kwargs()), mlp_hidden_dim=MLP_HIDDEN
    )


def _c_model(unsafe: bool = False) -> GraphClassifier:
    backbone_cls = _Fp16UnsafeBackbone if unsafe else GraphEnergyTransformerBackbone
    return GraphClassifier(
        backbone=backbone_cls(**_c_backbone_kwargs()), num_classes=NUM_CLASSES
    )


def _padded_graph_batch(seed: int, with_pe: bool):
    """A dense-batched graph batch of ``BATCH`` graphs, each ``N_REAL`` real + ``N_PAD`` PAD nodes.

    PAD nodes carry ZERO features (and zero PE), so after the zero-init-bias node projection they
    become all-zero tokens — the near-constant-token hazard that overflows ``EnergyLayerNorm``'s
    fp16 backward. ``node_mask`` marks them invalid (0). The adjacency connects only real nodes.
    """
    rng = np.random.default_rng(seed)
    node_features = np.zeros((BATCH, N, F), dtype="float32")
    node_features[:, :N_REAL, :] = rng.normal(size=(BATCH, N_REAL, F)).astype("float32")

    adjacency = np.zeros((BATCH, N, N), dtype="float32")
    sub = (rng.random((BATCH, N_REAL, N_REAL)) < 0.15).astype("float32")
    adjacency[:, :N_REAL, :N_REAL] = sub

    node_mask = np.zeros((BATCH, N), dtype="float32")
    node_mask[:, :N_REAL] = 1.0

    inputs = {
        "node_features": node_features,
        "adjacency": adjacency,
        "node_mask": node_mask,
    }
    if with_pe:
        pe = np.zeros((BATCH, N, PE_DIM), dtype="float32")
        pe[:, :N_REAL, :] = rng.normal(size=(BATCH, N_REAL, PE_DIM)).astype("float32")
        inputs["pe"] = pe
    else:
        inputs["target_index"] = np.zeros((BATCH,), dtype="int32")   # sampler puts target at 0
    return inputs


def _trunk_weights_snapshot(model) -> dict:
    return {
        v.path: keras.ops.convert_to_numpy(v).astype("float64")
        for v in model.trainable_weights
    }


def _run_fit(model, inputs, y, loss):
    """Stock ``compile(jit_compile=True)`` + a real ``fit`` over ``FIT_STEPS`` steps.

    Returns ``(loss_scale, deltas)`` where ``deltas[path] = max|dW|`` per trainable weight.
    NO ``train_step`` override — the fp16/XLA bug only appears through the real backward pass.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=loss,
        jit_compile=True,          # XLA is half of the bug; exercise it explicitly
    )
    assert "LossScale" in type(model.optimizer).__name__, "mixed_float16 policy is not active"

    ds = tf.data.Dataset.from_tensor_slices((inputs, y)).batch(BATCH).repeat()
    before = _trunk_weights_snapshot(model)
    history = model.fit(ds, steps_per_epoch=FIT_STEPS, epochs=1, verbose=0)
    after = _trunk_weights_snapshot(model)

    assert np.all(np.isfinite(history.history["loss"])), (
        "loss went non-finite — a DIFFERENT bug from the silent freeze this guards"
    )
    scale = float(keras.ops.convert_to_numpy(model.optimizer.dynamic_scale))
    deltas = {p: float(np.max(np.abs(after[p] - before[p]))) for p in before}
    return scale, deltas


def _b_labels(seed: int) -> np.ndarray:
    return (np.random.default_rng(seed).random((BATCH, 1)) < 0.5).astype("float32")


def _c_labels(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, NUM_CLASSES, size=(BATCH,)).astype("int32")


_BCE = keras.losses.BinaryCrossentropy(from_logits=True)


def _sparse_ce():
    return keras.losses.SparseCategoricalCrossentropy(from_logits=True)


# ---------------------------------------------------------------------------


@pytest.fixture
def mixed_f16():
    """Set the GLOBAL mixed_float16 policy for one test, then restore it.

    Must be global (not a ``dtype=`` kwarg) — that is how it bites in production: every layer
    built while it is active picks it up.
    """
    previous = keras.mixed_precision.global_policy()
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        yield
    finally:
        keras.mixed_precision.set_global_policy(previous)


class TestFp16XlaTrainingGuard:
    """The load-bearing guard: real ``fit`` backward pass under ``mixed_float16`` + XLA.

    GREEN (shipped model) trains; RED (fp16-unsafe injection) freezes. Both at ``N >= 64`` with
    PAD rows, for BOTH graph heads.
    """

    # ---- GREEN: the shipped models must TRAIN -----------------------------

    def test_b_trains_under_mixed_float16(self, mixed_f16):
        """Variant B (GraphAnomalyDetector): loss scale holds AND trunk weights move."""
        keras.utils.set_random_seed(13)
        model = _b_model(unsafe=False)
        model.build(None)
        assert model.backbone.compute_dtype == "float16"          # policy IS active
        # The FIX: blocks run in their fp32 variable dtype even under the fp16 policy.
        assert model.backbone.blocks[0].compute_dtype == "float32"

        inputs = _padded_graph_batch(seed=1, with_pe=False)
        scale, deltas = _run_fit(model, inputs, _b_labels(2), _BCE)

        assert scale >= COLLAPSE_THRESHOLD, (
            f"loss scale COLLAPSED to {scale:.3e} (start {INITIAL_LOSS_SCALE:.0f}) — the shipped "
            "variant-B model froze under fp16/XLA, which is exactly the bug the D-002 fix prevents"
        )
        moved = {p: d for p, d in deltas.items() if "et_block" in p and d > 0.0}
        assert moved, (
            f"no trunk (et_block) weight moved over {FIT_STEPS} fp16 fit steps: {deltas}"
        )
        # named-needle anti-vacuity: the specific weights whose grad flows through the norm.
        for needle in ("energy_attention/w_key", "energy_layer_norm/gamma", "hopfield_network/xi"):
            hit = [d for p, d in deltas.items() if needle in p]
            assert hit, f"no trunk weight matched '{needle}' — assertion is looking at nothing"
            assert max(hit) > 0.0, f"trunk weight '{needle}' did not move (max|dW|={max(hit):.3e})"

    def test_c_trains_under_mixed_float16(self, mixed_f16):
        """Variant C-lite (GraphClassifier): loss scale holds AND trunk weights move."""
        keras.utils.set_random_seed(17)
        model = _c_model(unsafe=False)
        model.build(None)
        assert model.backbone.compute_dtype == "float16"
        assert model.backbone.blocks[0].compute_dtype == "float32"

        inputs = _padded_graph_batch(seed=3, with_pe=True)
        scale, deltas = _run_fit(model, inputs, _c_labels(4), _sparse_ce())

        assert scale >= COLLAPSE_THRESHOLD, (
            f"loss scale COLLAPSED to {scale:.3e} (start {INITIAL_LOSS_SCALE:.0f}) — the shipped "
            "variant-C model froze under fp16/XLA"
        )
        moved = {p: d for p, d in deltas.items() if "et_block" in p and d > 0.0}
        assert moved, f"no trunk (et_block) weight moved over {FIT_STEPS} fp16 fit steps"
        for needle in ("energy_attention/w_key", "energy_layer_norm/gamma", "hopfield_network/xi"):
            hit = [d for p, d in deltas.items() if needle in p]
            assert hit, f"no trunk weight matched '{needle}' — assertion is looking at nothing"
            assert max(hit) > 0.0, f"trunk weight '{needle}' did not move (max|dW|={max(hit):.3e})"

    # ---- RED: the fp16-unsafe injection must FREEZE -----------------------

    def _assert_red_bites(self, scale, deltas, variant):
        """The broken model must have frozen: loss scale collapsed AND trunk weights pinned.

        If it did NOT freeze, the guard cannot bite. On a GPU that is a real regression (fail);
        on a CPU-only box mixed_float16 runs in fp32 so the overflow never happens (skip).
        """
        trunk = {p: d for p, d in deltas.items() if "et_block" in p}
        frozen = sorted(p for p, d in trunk.items() if d == 0.0)
        collapsed = scale < COLLAPSE_THRESHOLD

        if not collapsed:
            if not _gpu_available():
                pytest.skip(
                    "fp16/XLA (var+eps)^-1.5 overflow needs a GPU; mixed_float16 computes in "
                    "fp32 on CPU so the RED injection cannot bite here"
                )
            pytest.fail(
                f"variant-{variant} RED did NOT bite on GPU: loss scale stayed {scale:.3e} "
                f">= {COLLAPSE_THRESHOLD:.0f}. The fp16-unsafe backbone still trained, so the "
                "guard cannot go red — the fix is unverifiable."
            )

        assert len(frozen) == len(trunk), (
            f"variant-{variant} RED: loss scale collapsed to {scale:.3e} (good) but only "
            f"{len(frozen)}/{len(trunk)} trunk weights froze — expected ALL trunk weights pinned "
            "when every step is rejected"
        )

    def test_b_fp16_unsafe_backbone_freezes(self, mixed_f16):
        """RED: variant-B with fp16 blocks must freeze (proves the GREEN guard bites)."""
        keras.utils.set_random_seed(13)
        model = _b_model(unsafe=True)
        model.build(None)
        # the injection: blocks now run in fp16 (the overflow), not fp32.
        assert model.backbone.blocks[0].compute_dtype == "float16"

        inputs = _padded_graph_batch(seed=1, with_pe=False)
        scale, deltas = _run_fit(model, inputs, _b_labels(2), _BCE)
        self._assert_red_bites(scale, deltas, variant="B")

    def test_c_fp16_unsafe_backbone_freezes(self, mixed_f16):
        """RED: variant-C-lite with fp16 blocks must freeze (proves the GREEN guard bites)."""
        keras.utils.set_random_seed(17)
        model = _c_model(unsafe=True)
        model.build(None)
        assert model.backbone.blocks[0].compute_dtype == "float16"

        inputs = _padded_graph_batch(seed=3, with_pe=True)
        scale, deltas = _run_fit(model, inputs, _c_labels(4), _sparse_ce())
        self._assert_red_bites(scale, deltas, variant="C")


# ===========================================================================
# Step 10 — forward / shape, `.keras` round-trip, serialization, dead-path guard.
#
# These run under the DEFAULT float32 policy (NO ``mixed_f16`` fixture): the plain
# forward + serialization surface, the direct graph analog of the image suite's
# ``TestForwardShapes`` / ``TestSerialization``. The load-bearing fp16/XLA guard
# above is untouched and NOT duplicated here. Every synthetic graph comes from the
# same network-free ``_padded_graph_batch`` builder the guard uses (``N`` = 96, a
# third of it PAD), so these tests never touch the network.
# ===========================================================================


# atol: 1e-6..1e-7 is the CPU float32 convention; the CI harness runs on GPU
# (CUDA_VISIBLE_DEVICES=1), where 1e-4 is the house tolerance (mirrors the image
# suite's round-trip). Use the looser GPU bound so the gate is not flaky on GPU.
RT_ATOL = 1e-4


def _backbone(kwargs: dict) -> GraphEnergyTransformerBackbone:
    return GraphEnergyTransformerBackbone(**kwargs)


def _predict(model, inputs) -> np.ndarray:
    """Deterministic forward (``training=False`` -> no eq.-27 noise, no dropout)."""
    return keras.ops.convert_to_numpy(model(inputs, training=False))


class TestGraphForwardShapes:
    """Forward pass: output SHAPE + FINITENESS for the bare backbone and both heads."""

    def test_backbone_b_config_tokens(self):
        """B-config backbone -> (B, N, D): N' == N (no CLS token)."""
        keras.utils.set_random_seed(0)
        out = _predict(_backbone(_b_backbone_kwargs()),
                       _padded_graph_batch(seed=1, with_pe=False))
        assert tuple(out.shape) == (BATCH, N, EMBED_DIM)
        assert np.all(np.isfinite(out))

    def test_backbone_c_config_tokens(self):
        """C-config backbone -> (B, N+1, D): N' == N + 1 (CLS prepended)."""
        keras.utils.set_random_seed(0)
        out = _predict(_backbone(_c_backbone_kwargs()),
                       _padded_graph_batch(seed=2, with_pe=True))
        assert tuple(out.shape) == (BATCH, N + 1, EMBED_DIM)
        assert np.all(np.isfinite(out))

    def test_anomaly_detector_forward(self):
        """Variant B on its contract {node_features,adjacency,node_mask,target_index} -> (B, 1)."""
        keras.utils.set_random_seed(0)
        out = _predict(_b_model(unsafe=False),
                       _padded_graph_batch(seed=1, with_pe=False))
        assert tuple(out.shape) == (BATCH, 1)
        assert np.all(np.isfinite(out))

    def test_classifier_forward(self):
        """Variant C on its contract {node_features,adjacency,pe,node_mask} -> (B, num_classes)."""
        keras.utils.set_random_seed(0)
        out = _predict(_c_model(unsafe=False),
                       _padded_graph_batch(seed=2, with_pe=True))
        assert tuple(out.shape) == (BATCH, NUM_CLASSES)
        assert np.all(np.isfinite(out))


class TestComputeOutputShapeUnbuilt:
    """``compute_output_shape`` returns the right shape WITHOUT materializing weights."""

    @staticmethod
    def _nf(n: int) -> dict:
        return {"node_features": (None, n, F)}

    def test_backbone_b_unbuilt(self):
        bb = _backbone(_b_backbone_kwargs())
        assert not bb.built and len(bb.weights) == 0
        assert bb.compute_output_shape(self._nf(N)) == (None, N, EMBED_DIM)
        assert not bb.built and len(bb.weights) == 0

    def test_backbone_c_unbuilt(self):
        bb = _backbone(_c_backbone_kwargs())
        assert not bb.built and len(bb.weights) == 0
        assert bb.compute_output_shape(self._nf(N)) == (None, N + 1, EMBED_DIM)
        assert not bb.built and len(bb.weights) == 0

    def test_anomaly_detector_unbuilt(self):
        model = _b_model(unsafe=False)
        assert not model.built and len(model.weights) == 0
        assert model.compute_output_shape(self._nf(N)) == (None, 1)
        assert not model.built and len(model.weights) == 0

    def test_classifier_unbuilt(self):
        model = _c_model(unsafe=False)
        assert not model.built and len(model.weights) == 0
        assert model.compute_output_shape(self._nf(N)) == (None, NUM_CLASSES)
        assert not model.built and len(model.weights) == 0


class TestGraphSerialization:
    """The acceptance gate (SC1): ``.keras`` round-trip -> deterministic output equal
    AND equal weight COUNT (an output-only compare can miss a weight dropped on a dead
    path — the lazy-build trap — so the COUNT is asserted too, mirroring the image test).
    """

    def _round_trip(self, model, inputs, tmp_path, name, expect_cls) -> int:
        before = _predict(model, inputs)                 # this builds the model
        n_train = len(model.trainable_weights)
        n_all = len(model.weights)

        path = str(tmp_path / f"{name}.keras")
        model.save(path)
        loaded = keras.models.load_model(path)

        after = _predict(loaded, inputs)
        assert isinstance(loaded, expect_cls), (
            f"reloaded object is a {type(loaded).__name__}, expected {expect_cls.__name__}"
        )
        assert len(loaded.trainable_weights) == n_train, (
            f"{name}: trainable weight COUNT changed on round-trip "
            f"({len(loaded.trainable_weights)} vs {n_train}) — a weight was dropped"
        )
        assert len(loaded.weights) == n_all
        np.testing.assert_allclose(before, after, atol=RT_ATOL)
        return n_train

    def test_backbone_b_round_trip(self, tmp_path):
        keras.utils.set_random_seed(0)
        self._round_trip(_backbone(_b_backbone_kwargs()),
                         _padded_graph_batch(seed=1, with_pe=False),
                         tmp_path, "gb_backbone_b", GraphEnergyTransformerBackbone)

    def test_backbone_c_round_trip(self, tmp_path):
        keras.utils.set_random_seed(0)
        self._round_trip(_backbone(_c_backbone_kwargs()),
                         _padded_graph_batch(seed=2, with_pe=True),
                         tmp_path, "gb_backbone_c", GraphEnergyTransformerBackbone)

    def test_anomaly_detector_round_trip(self, tmp_path):
        keras.utils.set_random_seed(0)
        self._round_trip(_b_model(unsafe=False),
                         _padded_graph_batch(seed=1, with_pe=False),
                         tmp_path, "gb_b", GraphAnomalyDetector)

    def test_classifier_round_trip(self, tmp_path):
        keras.utils.set_random_seed(0)
        self._round_trip(_c_model(unsafe=False),
                         _padded_graph_batch(seed=2, with_pe=True),
                         tmp_path, "gb_c", GraphClassifier)


class TestRoundTripIsAlive:
    """Anti-vacuity dead-path guard: perturbing ONE POST-NORM weight of the RELOADED
    model must move the output well past ``RT_ATOL``, so the round-trip equality above
    cannot be vacuously satisfied by a dead model (image test: ``max_delta > 1e-2``).
    """

    DEAD_GUARD = 1e-2                                     # two orders over RT_ATOL

    def _perturb_and_check(self, model, inputs, tmp_path, name, layer_name):
        before = _predict(model, inputs)
        path = str(tmp_path / f"{name}.keras")
        model.save(path)
        loaded = keras.models.load_model(path)

        target = next(
            w for w in loaded.weights if layer_name in w.path and w.path.endswith("bias")
        )
        target.assign(keras.ops.add(target.value, 1.0))

        after = _predict(loaded, inputs)
        max_delta = float(np.max(np.abs(before - after)))
        assert max_delta > self.DEAD_GUARD, (
            f"round-trip guard is dead for {name}: perturbing '{layer_name}/bias' moved the "
            f"output by only {max_delta:.3e} (<= {self.DEAD_GUARD:.0e}) — the round-trip "
            "equality would be vacuously satisfiable by a dead model"
        )
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(before, after, atol=RT_ATOL)

    def test_anomaly_detector_round_trip_alive(self, tmp_path):
        # `head_mlp_out` is the FINAL Dense — POST every LayerNorm — so a +1.0 bias bump
        # moves the logit by O(1), unlike a norm-headed trunk weight (image-test lesson).
        keras.utils.set_random_seed(0)
        self._perturb_and_check(_b_model(unsafe=False),
                                _padded_graph_batch(seed=1, with_pe=False),
                                tmp_path, "gb_b_perturb", "head_mlp_out")

    def test_classifier_round_trip_alive(self, tmp_path):
        # `head_dense` is the FINAL Dense (post `head_norm`); +1.0 shifts every class logit.
        keras.utils.set_random_seed(0)
        self._perturb_and_check(_c_model(unsafe=False),
                                _padded_graph_batch(seed=2, with_pe=True),
                                tmp_path, "gb_c_perturb", "head_dense")

    def test_backbone_round_trip_alive(self, tmp_path):
        # `node_proj/bias` shifts x0 UNIFORMLY; `EnergyLayerNorm` removes uniform shifts from
        # the per-step UPDATE (`g = norm(x)`), but the block RETURNS `x` (never end-normed),
        # so the shift survives to the backbone output.
        keras.utils.set_random_seed(0)
        self._perturb_and_check(_backbone(_b_backbone_kwargs()),
                                _padded_graph_batch(seed=1, with_pe=False),
                                tmp_path, "gb_backbone_perturb", "node_proj")


class TestGraphConfigCycle:
    """``get_config`` -> ``from_config`` reconstructs without error and yields the same
    config; the nested backbone is DESERIALIZED to a model instance (not left as a dict).
    """

    def test_backbone_b_cycle(self):
        bb = _backbone(_b_backbone_kwargs())
        bb2 = GraphEnergyTransformerBackbone.from_config(bb.get_config())
        assert bb2.get_config() == bb.get_config()

    def test_backbone_c_cycle(self):
        bb = _backbone(_c_backbone_kwargs())
        bb2 = GraphEnergyTransformerBackbone.from_config(bb.get_config())
        assert bb2.get_config() == bb.get_config()

    def test_anomaly_detector_cycle(self):
        model = _b_model(unsafe=False)
        model2 = GraphAnomalyDetector.from_config(model.get_config())
        assert isinstance(model2.backbone, GraphEnergyTransformerBackbone), (
            "nested backbone was left as a serialized dict, not deserialized to a model"
        )
        assert model2.backbone.get_config() == model.backbone.get_config()
        assert model2.mlp_hidden_dim == model.mlp_hidden_dim
        assert model2.mlp_dropout == model.mlp_dropout

    def test_classifier_cycle(self):
        model = _c_model(unsafe=False)
        model2 = GraphClassifier.from_config(model.get_config())
        assert isinstance(model2.backbone, GraphEnergyTransformerBackbone), (
            "nested backbone was left as a serialized dict, not deserialized to a model"
        )
        assert model2.backbone.get_config() == model.backbone.get_config()
        assert model2.num_classes == model.num_classes
        assert model2.head_dropout_rate == model.head_dropout_rate
