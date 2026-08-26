"""The MoE auxiliary loss must survive a real ``model.fit()`` step under mixed precision.

**Why this module exists, and why it is a ``fit()`` and not a forward pass.**

At ``c38d5f17b``, ``compute_auxiliary_loss`` (``layers/moe/gating.py``) returned its
scalar in the ambient COMPUTE dtype. Under ``mixed_float16`` that is ``float16``, and
``layer.py:278`` hands the value straight to ``add_loss``. Keras'
``_aggregate_additional_loss`` (``keras/src/trainers/trainer.py:389-400``) casts only
NON-float losses to ``floatx()``, so the ``float16`` scalar passed through untouched
into the list reduced at ``trainer.py:365`` (``total_loss = ops.sum(losses)``)
alongside the ``float32`` compiled loss. MEASURED, one ``fit()`` step, 4 experts,
linear gating, ``top_k=2``, the shipped default ``aux_loss_weight=0.01``::

    TypeError: Cannot convert a list containing a tensor of dtype <dtype: 'float16'>
    to <dtype: 'float32'> (Tensor is: <tf.Tensor 'Sum:0' shape=() dtype=float16>)

``model.fit()`` was therefore broken for every mixed-precision MoE consumer --
``models/language/qwen/qwen3.py`` (``top_k=8``) and ``qwen3_next.py`` (``top_k=10``).

The package already had a mixed-precision test
(``test_layer.py::test_mixed_precision_forward_is_finite_for_every_gating_type``), and it
is exactly the shape that let this ship: it runs a ``training=False`` forward pass, which
never calls ``add_loss`` and never reaches Keras' loss aggregation. **The defect lives in
the aggregation, not in the layer's numerics, so only a real optimizer step can see it.**
Do not "simplify" the cells below into forward passes.

See ``decisions.md`` D-005 of ``plan-2026-08-26T155709-fb07cf4e``, and the anchor at
``compute_auxiliary_loss``.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.moe.config import ExpertConfig, GatingConfig, MoEConfig
from dl_techniques.layers.moe.layer import MixtureOfExperts

SEQ_LEN, MODEL_DIM, NUM_EXPERTS = 6, 16, 4

# Both half-precision policies. `mixed_bfloat16` is included because the failing
# mechanism is dtype heterogeneity in Keras' loss list, not fp16's narrow range --
# bfloat16 is just as far from `floatx()` and just as uncast.
HALF_POLICIES = ("mixed_float16", "mixed_bfloat16")


@pytest.fixture(params=HALF_POLICIES)
def half_policy(request):
    """Set a half-precision global policy for one test, then ALWAYS restore it.

    A local sibling of ``tests/test_layers/conftest.py``'s ``mixed_float16_policy``;
    that fixture does not cover ``mixed_bfloat16``. The restore lives in a ``finally``
    for the reason the shared conftest documents: a leaked global policy corrupts every
    later test in the process.

    :param request: pytest request carrying the parametrized policy name.

    :yield: the policy name in force (``'mixed_float16'`` or ``'mixed_bfloat16'``).
    :rtype: str
    """
    previous = keras.mixed_precision.global_policy().name
    keras.mixed_precision.set_global_policy(request.param)
    try:
        yield request.param
    finally:
        keras.mixed_precision.set_global_policy(previous)


def _model(gating_type: str, z_loss_weight: float) -> keras.Model:
    """Build a small compiled functional model wrapping one ``MixtureOfExperts``.

    :param gating_type: a HARD-routed gating type (``'linear'`` or ``'cosine'``).
        ``'softmoe'`` is deliberately not accepted: ``layer.py`` skips the aux/z-loss
        branch entirely for it, so a softmoe cell could not fail this module's claim.
    :type gating_type: str
    :param z_loss_weight: ``0.0`` isolates the auxiliary loss; anything positive puts a
        float32 z-loss in the same list, which is the mixed-dtype configuration.
    :type z_loss_weight: float

    :return: a model compiled with a float32 MSE loss and a stock optimizer.
    :rtype: keras.Model
    """
    assert gating_type in ("linear", "cosine")
    config = MoEConfig(
        num_experts=NUM_EXPERTS,
        expert_config=ExpertConfig(
            ffn_config={
                "type": "mlp",
                "hidden_dim": 2 * MODEL_DIM,
                "output_dim": MODEL_DIM,
            }
        ),
        gating_config=GatingConfig(
            gating_type=gating_type,
            top_k=2,
            embedding_dim=MODEL_DIM,
            aux_loss_weight=0.01,  # the SHIPPED default, not a value chosen to fail
            z_loss_weight=z_loss_weight,
        ),
    )
    inputs = keras.Input(shape=(SEQ_LEN, MODEL_DIM))
    outputs = MixtureOfExperts(config=config)(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def _batch():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4, SEQ_LEN, MODEL_DIM)).astype("float32")
    y = rng.standard_normal((4, SEQ_LEN, MODEL_DIM)).astype("float32")
    return x, y


@pytest.mark.parametrize("gating_type", ("linear", "cosine"))
@pytest.mark.parametrize(
    "z_loss_weight",
    (0.0, 1e-3),
    ids=("aux_only", "aux_and_z"),
)
def test_one_fit_step_completes(half_policy, gating_type, z_loss_weight):
    """One real optimizer step, the cell that was RED at ``c38d5f17b``.

    ``aux_only`` (``z_loss_weight=0.0``) is the harder case: with the z-loss switched
    off, the auxiliary loss is the ONLY ``add_loss`` value, so nothing else can mask
    its dtype. ``aux_and_z`` is the shipped default pairing.
    """
    model = _model(gating_type, z_loss_weight)
    x, y = _batch()

    history = model.fit(x, y, epochs=1, batch_size=2, verbose=0)

    loss = history.history["loss"][-1]
    assert np.isfinite(loss), (
        f"{half_policy}/{gating_type}: fit() completed but the loss is {loss}"
    )


def test_the_layer_really_emits_half_precision_under_the_policy(half_policy):
    """Anti-vacuity: the guard above is empty unless the gate computes in half.

    If ``MixtureOfExperts`` ever stopped honouring the global policy internally, every
    cell in this module would pass for the wrong reason -- there would be no fp16 value
    to mis-aggregate. Pin the premise, not just the conclusion.
    """
    model = _model("linear", z_loss_weight=0.0)
    moe = next(layer for layer in model.layers if isinstance(layer, MixtureOfExperts))
    assert moe.compute_dtype == half_policy.removeprefix("mixed_"), (
        f"MoE layer computes in {moe.compute_dtype} under {half_policy}; "
        "this module's premise no longer holds"
    )


def test_the_aggregated_loss_list_is_uniformly_float32(half_policy):
    """The mechanism itself: every ``add_loss`` value must be float32.

    ``model.fit()`` succeeding is the CONSEQUENCE; the invariant is that the list Keras
    reduces at ``trainer.py:365`` is dtype-uniform. Asserting it directly means a future
    Keras that starts casting float losses (making ``fit()`` pass regardless) cannot
    silently retire this guard.
    """
    model = _model("linear", z_loss_weight=1e-3)
    moe = next(layer for layer in model.layers if isinstance(layer, MixtureOfExperts))
    x, _ = _batch()

    moe(keras.ops.convert_to_tensor(x), training=True)

    assert len(moe.losses) == 2, (
        f"expected an aux loss and a z-loss, got {len(moe.losses)}"
    )
    for loss in moe.losses:
        assert "float32" in str(loss.dtype), (
            f"add_loss value has dtype {loss.dtype} under {half_policy}; "
            "Keras casts only NON-float losses, so this crashes ops.sum(self.losses)"
        )
