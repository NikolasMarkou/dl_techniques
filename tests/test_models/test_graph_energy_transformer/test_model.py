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
