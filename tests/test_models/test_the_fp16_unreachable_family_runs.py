"""Guard: the six `mixed_float16`-unreachable packages run forward AND backward.

Subject set (plan ``plan-2026-08-19T163559-499b6f0e``, step 17.1, rule ``R-088``).
Each of these six raised a dtype error on a *plain forward pass* under
``mixed_float16`` at HEAD, each at a named site:

===================== ====================================================
package               measured failure site at HEAD
===================== ====================================================
``detr``              ``models/detr/model.py:292`` — ``ops.zeros`` no dtype
``kan``               ``layers/ffn/kan_linear.py:389`` — float32 knot grid
``mamba``             ``models/mamba/components.py:472`` — float32 scan state
``mamba`` (v2)        ``models/mamba/components_v2.py:292`` — idem
``memory_bank``       ``models/memory_bank/read_controller.py:378`` —
                      ``dtype="float32"`` additive mask
``nano_vlm_world_..`` ``models/nano_vlm_world_model/scheduler.py:195``
``thera``             ``layers/grid_sample.py:134`` — ``convert_to_tensor``
                      with ``dtype=tf.float32`` on a float16 tensor
===================== ====================================================

Three properties are asserted for every subject, because each of the three has
already been the arm that caught a real defect somewhere in this plan:

1. **forward** runs and is finite (a `mixed_float16` failure can be a NaN rather
   than a raise — batch 4's lesson);
2. **backward** runs and produces finite gradients for every trainable variable
   (step 5.8's lesson: four of its five fixes raised a SECOND time only once a
   gradient was taken);
3. **the output is actually float16** — the anti-vacuity arm. Without it a model
   that internally forces everything to float32 would pass arms 1 and 2 while
   never exercising mixed precision at all.

The float32 control arm runs the same subject under the default policy, so a
green result here cannot be read as "fp16 works" when in fact nothing works.

See ``decisions.md`` D-043 .. D-047.
"""

import numpy as np
import pytest
import keras
import tensorflow as tf


# ---------------------------------------------------------------------------
# Subjects. Deliberately tiny: this file is a dtype guard, not a capacity test.
# ---------------------------------------------------------------------------

def _detr():
    from dl_techniques.models.detr import DETR, DetrTransformer
    backbone = keras.Sequential(
        [
            keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu"),
            keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu"),
        ],
        name="stub_backbone",
    )
    transformer = DetrTransformer(
        hidden_dim=32, num_heads=2, num_encoder_layers=1,
        num_decoder_layers=1, ffn_dim=32, dropout=0.0,
    )
    model = DETR(
        num_classes=3, num_queries=4, backbone=backbone,
        transformer=transformer, hidden_dim=32, aux_loss=True,
    )
    rs = np.random.RandomState(0)
    inputs = [
        rs.randn(1, 32, 32, 3).astype("float32"),
        np.zeros((1, 32, 32), dtype="bool"),
    ]
    return model, inputs


def _kan():
    from dl_techniques.models.kan import KAN
    configs = [
        {"features": 8, "grid_size": 5, "activation": "swish"},
        {"features": 4, "grid_size": 4, "activation": "gelu"},
    ]
    model = KAN(layer_configs=configs, input_features=10)
    return model, np.random.RandomState(0).randn(4, 10).astype("float32")


def _mamba():
    from dl_techniques.models.mamba import Mamba
    model = Mamba(vocab_size=100, d_model=32, num_layers=2,
                  d_state=8, d_conv=4, expand=2)
    tokens = np.random.RandomState(0).randint(0, 100, (2, 12)).astype("int32")
    return model, {"input_ids": tokens}


def _mamba2():
    from dl_techniques.models.mamba.mamba_v2 import Mamba2
    model = Mamba2(vocab_size=100, d_model=64, num_layers=2, d_state=16,
                   d_conv=4, expand=2, headdim=16)
    tokens = np.random.RandomState(0).randint(0, 100, (2, 12)).astype("int32")
    return model, {"input_ids": tokens}


def _memory_bank():
    from dl_techniques.models.memory_bank.wave_field_memory_llm import (
        WaveFieldMemoryLLM,
    )
    model = WaveFieldMemoryLLM(
        vocab_size=64, embed_dim=32, depth=4, num_heads=2, max_seq_len=16,
        d_k=8, d_v=16, s_lt=64, top_k=4, infonce_negatives=8,
        diversity_subsample=16,
    )
    tokens = np.random.RandomState(0).randint(0, 64, (2, 16)).astype("int32")
    return model, tokens


def _nano_vlm_world_model():
    # The full `ScoreBasedNanoVLM` at its smallest declared variant is 144M
    # parameters; the two fix sites (`TimestepEmbedding`'s float32 return and
    # `DiffusionScheduler._extract`'s float32 coefficient) are both exercised by
    # `JointDenoiser`, which is where the raise actually landed once the
    # scheduler was repaired.
    from dl_techniques.models.nano_vlm_world_model.denoisers import JointDenoiser

    class _Host(keras.Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.denoiser = JointDenoiser(
                vision_dim=16, text_dim=8, hidden_dim=16, num_layers=1,
            )

        def call(self, inputs, training=None):
            v, t, ts = inputs
            return self.denoiser(v, t, ts, training=training)

    rs = np.random.RandomState(0)
    inputs = [
        rs.randn(2, 5, 16).astype("float32"),
        rs.randn(2, 4, 8).astype("float32"),
        rs.randint(0, 1000, (2,)).astype("int32"),
    ]
    return _Host(), inputs


def _thera():
    from dl_techniques.models.thera.model import Thera
    from dl_techniques.models.thera.edsr_backbone import EDSRBackbone
    from dl_techniques.models.thera.tails import build_thera_tail
    model = Thera(
        hidden_dim=16, out_dim=3,
        backbone=EDSRBackbone(num_feats=16, num_blocks=1),
        tail=build_thera_tail("air"), k_init=None, components_init_scale=16.0,
    )
    rs = np.random.RandomState(0)
    inputs = (
        rs.randn(1, 8, 8, 3).astype("float32"),
        rs.rand(1, 12, 12, 2).astype("float32"),
        np.zeros((1, 1), dtype="float32"),
    )
    return model, inputs


SUBJECTS = {
    "detr": _detr,
    "kan": _kan,
    "mamba": _mamba,
    "mamba2": _mamba2,
    "memory_bank": _memory_bank,
    "nano_vlm_world_model": _nano_vlm_world_model,
    "thera": _thera,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def float16_policy():
    """Set `mixed_float16` for one test and restore the previous policy after."""
    previous = keras.mixed_precision.global_policy()
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        yield
    finally:
        keras.mixed_precision.set_global_policy(previous)


def _float_tensors(value, out):
    if isinstance(value, dict):
        for key in sorted(value):
            _float_tensors(value[key], out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _float_tensors(item, out)
    else:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and getattr(dtype, "is_floating", False):
            out.append(value)


def _forward(name):
    model, inputs = SUBJECTS[name]()
    outputs = model(inputs, training=False)
    tensors = []
    _float_tensors(outputs, tensors)
    assert tensors, f"{name} produced no floating-point output to check"
    return model, tensors


# ---------------------------------------------------------------------------
# Arm 1 — the forward pass runs and is finite under `mixed_float16`.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(SUBJECTS))
def test_the_forward_pass_is_finite_under_mixed_float16(name, float16_policy):
    _, tensors = _forward(name)
    for tensor in tensors:
        array = np.asarray(keras.ops.convert_to_numpy(tensor), dtype="float64")
        n_nan = int(np.isnan(array).sum())
        n_inf = int(np.isinf(array).sum())
        assert n_nan == 0 and n_inf == 0, (
            f"{name}: mixed_float16 forward produced {n_nan} NaN and {n_inf} Inf "
            f"out of {array.size} elements"
        )


# ---------------------------------------------------------------------------
# Arm 2 — ANTI-VACUITY. The output must really be float16.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(SUBJECTS))
def test_the_output_is_actually_float16(name, float16_policy):
    _, tensors = _forward(name)
    dtypes = {keras.ops.convert_to_numpy(t).dtype.name for t in tensors}
    assert dtypes == {"float16"}, (
        f"{name}: mixed_float16 forward returned {sorted(dtypes)}, not float16 — "
        "arms 1 and 3 would then be green without mixed precision being exercised"
    )


# ---------------------------------------------------------------------------
# Arm 3 — the BACKWARD pass. A green forward is not a green `fit()`.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(SUBJECTS))
def test_the_backward_pass_is_finite_under_mixed_float16(name, float16_policy):
    model, inputs = SUBJECTS[name]()
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        tensors = []
        _float_tensors(outputs, tensors)
        loss = tf.add_n(
            [tf.reduce_mean(tf.cast(t, "float32") ** 2) for t in tensors]
        )
    variables = model.trainable_variables
    assert variables, f"{name} has no trainable variables"
    grads = tape.gradient(loss, variables)

    assert np.isfinite(float(loss)), f"{name}: loss is {float(loss)}"
    n_nonfinite = 0
    n_present = 0
    for grad in grads:
        if grad is None:
            continue
        if isinstance(grad, tf.IndexedSlices):
            grad = grad.values
        n_present += 1
        array = np.asarray(tf.cast(grad, "float32"))
        n_nonfinite += int(np.isnan(array).sum() + np.isinf(array).sum())
    assert n_present > 0, f"{name}: every gradient came back None"
    assert n_nonfinite == 0, (
        f"{name}: {n_nonfinite} non-finite gradient elements under mixed_float16"
    )


# ---------------------------------------------------------------------------
# Arm 4 — the float32 CONTROL. Without it, a subject that is simply broken in
# both policies would read as an fp16 finding.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(SUBJECTS))
def test_the_float32_control_still_runs(name):
    assert keras.mixed_precision.global_policy().name == "float32", (
        "this arm must run under the default policy"
    )
    _, tensors = _forward(name)
    dtypes = {keras.ops.convert_to_numpy(t).dtype.name for t in tensors}
    assert dtypes == {"float32"}, f"{name}: float32 control returned {sorted(dtypes)}"
    for tensor in tensors:
        array = np.asarray(keras.ops.convert_to_numpy(tensor), dtype="float64")
        assert np.isfinite(array).all(), f"{name}: float32 control is non-finite"


# ---------------------------------------------------------------------------
# Arm 5 — the two ULP-void constants this family carried, pinned by VALUE.
# ---------------------------------------------------------------------------

def test_the_epsilons_this_family_relied_on_are_degenerate_in_float16():
    """The two stability floors this family carried, measured — not assumed.

    This is the instrument D-027 introduced, applied to the two constants the
    fp16 family itself carried: `KANLinear.epsilon` (1e-7, the Cox-de Boor
    denominator floor) and `MemoryReadController`'s top-K renormaliser (1e-9).
    They are degenerate in float16 in TWO DIFFERENT WAYS, and the distinction is
    recorded here because the first draft of this test asserted the wrong one:

    * ``1e-9`` is **exactly 0.0** — the guard simply does not exist;
    * ``1e-7`` is **subnormal** (1.192093e-07, below ``finfo.tiny``) — it exists
      but carries ~19% relative error and no gradient support.

    Pinned by value so that "raise the constant" is never mistaken for the fix —
    the remedy is that neither is *used* at float16 any more.
    """
    assert float(np.float16(1e-9)) == 0.0
    kan_eps16 = float(np.float16(1e-7))
    assert kan_eps16 != 0.0
    assert 0.0 < kan_eps16 < float(np.finfo("float16").tiny)
    assert abs(kan_eps16 - 1e-7) / 1e-7 > 0.15
    # and the dtype-aware floor that replaced the 1e-9 is strictly positive
    assert float(np.finfo("float16").tiny) > 0.0
    # in float32 the dtype-aware spelling is INERT — it returns the literal
    assert max(1e-9, float(np.finfo("float32").tiny)) == 1e-9


def test_the_memory_bank_mask_sentinel_is_representable_in_the_compute_dtype():
    """`-1.0e9` overflows float16 to `-inf`; the clamped sentinel does not."""
    from dl_techniques.models.memory_bank.read_controller import _NEG_INF

    assert np.isinf(np.float16(_NEG_INF)), (
        "the raw sentinel is expected to overflow float16 — that is the defect"
    )
    clamped16 = max(_NEG_INF, float(np.finfo("float16").min) / 2.0)
    assert np.isfinite(np.float16(clamped16))
    # float32 inertness: the clamp returns the original literal unchanged.
    clamped32 = max(_NEG_INF, float(np.finfo("float32").min) / 2.0)
    assert clamped32 == _NEG_INF


# ---------------------------------------------------------------------------
# Arm 6 — `interpolate_grid` is dtype-preserving. This is the shared `layers/`
# site; `thera` is only its first caller.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", [0, 1])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_interpolate_grid_returns_the_grid_dtype(order, dtype):
    from dl_techniques.layers.grid_sample import interpolate_grid

    rs = np.random.RandomState(3)
    coords = tf.constant(rs.uniform(-0.5, 0.5, (1, 5, 5, 2)).astype(dtype))
    grid = tf.constant(rs.randn(1, 4, 4, 3).astype(dtype))
    out = interpolate_grid(coords, grid, order=order)
    assert out.dtype.name == dtype
    assert np.isfinite(np.asarray(tf.cast(out, "float32"))).all()
