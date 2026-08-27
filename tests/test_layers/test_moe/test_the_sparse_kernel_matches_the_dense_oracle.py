"""Equivalence gate for ``MixtureOfExperts``' two hard-routing kernels.

``MixtureOfExperts`` carries two implementations of the same function:

``_process_hard_routing_dense``
    Runs every expert on every token and masks the result. O(num_experts) in
    expert FLOPs. Measured against an independent dense numpy reference at max
    abs diff ~2e-7 across five configurations (plan
    ``plan-2026-08-26T100331-f3744602``, ``findings/layer-experts-review.md`` #1),
    which is what makes it usable as an oracle.

``_process_hard_routing_sparse``
    Gathers the token rows routed to each expert, runs the FFN on that gather
    only, and scatter-adds the weighted result back. O(top_k) in expert FLOPs.

This module is the gate that keeps the second numerically equal to the first.
It is deliberately written *before* the sparse kernel and its tolerance
(``atol=1e-5``, ``rtol=0``) is pre-committed: relaxing it after seeing the
numbers is the stop condition, not a fix.

**Regime.** Every equivalence assertion here runs on CPU with TF32 disabled
(``tf32_disabled``, ``tests/test_layers/conftest.py``). TF32 has been measured to
swing a float32 comparison in this repo by ~1500x, which would make a
1e-5 bound meaningless.

**Distinct dimensions.** Every grid entry keeps ``batch``, ``seq``, ``d_model``,
``num_experts``, ``top_k``, ``hidden`` and ``output_dim`` mutually distinct.
Equal dimensions hide transpose and scatter bugs; this repo has shipped two.
"""

import os

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.moe.config import ExpertConfig, GatingConfig, MoEConfig
from dl_techniques.layers.moe.layer import MixtureOfExperts

# CPU + TF32 off for the whole module. The GPU is not pinned off here (the repo
# runs its suites under `CUDA_VISIBLE_DEVICES=1`); TF32 is what actually moves the
# numbers, and the fixture below is the tree's single restore-safe toggle.
pytestmark = pytest.mark.usefixtures("tf32_disabled")

# Pre-committed. Do not widen.
ATOL = 1e-5
RTOL = 0.0


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def _build_moe(
        *,
        num_experts: int,
        top_k: int,
        d_model: int,
        hidden: int,
        output_dim: int,
        gating_type: str,
        add_noise: bool = False,
) -> MixtureOfExperts:
    """Construct a deterministic hard-routed MoE layer.

    ``add_noise`` and ``jitter_noise`` default to off so two forward passes over
    the same input are bit-identical; otherwise a dense-vs-sparse diff would be
    measuring the RNG. ``add_noise=True`` is only usable together with
    :func:`_pin_the_noise_draw` -- see
    ``test_the_kernels_agree_with_the_router_noise_on``.

    :param num_experts: Number of experts.
    :type num_experts: int
    :param top_k: Experts activated per token.
    :type top_k: int
    :param d_model: Input feature dimension.
    :type d_model: int
    :param hidden: Expert FFN hidden dimension.
    :type hidden: int
    :param output_dim: Expert FFN output dimension.
    :type output_dim: int
    :param gating_type: ``'linear'`` or ``'cosine'``.
    :type gating_type: str
    :param add_noise: Whether the router adds training-time noise to its logits.
        Only ``LinearGating`` has a noise path.
    :type add_noise: bool
    :return: An unbuilt MoE layer.
    :rtype: MixtureOfExperts
    """
    gating_kwargs = dict(gating_type=gating_type, top_k=top_k, add_noise=add_noise)
    if gating_type == 'cosine':
        # 7 keeps embedding_dim distinct from every other dimension in the grid.
        gating_kwargs['embedding_dim'] = 7
    config = MoEConfig(
        num_experts=num_experts,
        expert_config=ExpertConfig(
            ffn_config={
                'type': 'mlp',
                'hidden_dim': hidden,
                'output_dim': output_dim,
                'activation': 'gelu',
            }
        ),
        gating_config=GatingConfig(**gating_kwargs),
        jitter_noise=0.0,
    )
    return MixtureOfExperts(config)


def _dense_output(layer, x):
    """Run ``layer`` forcing the dense kernel, whatever the dispatch would choose.

    Swaps the sparse method for the dense one on the *instance*, so the layer's
    own dispatch logic is exercised end to end and only the kernel changes.

    :param layer: A built or unbuilt :class:`MixtureOfExperts`.
    :param x: Input tensor.
    :return: The layer output computed by the dense kernel.
    """
    original = layer._process_hard_routing_sparse
    layer._process_hard_routing_sparse = layer._process_hard_routing_dense
    try:
        return layer(x, training=False)
    finally:
        layer._process_hard_routing_sparse = original


_KERNEL_NAMES = ('_process_hard_routing_dense', '_process_hard_routing_sparse')


def _make_spy(name, original, seen):
    """Wrap an unbound kernel method so every invocation appends its name to ``seen``.

    Used by the two code-path assertions in this module. The wrapper delegates to
    ``original`` unchanged, so the layer's numerics are untouched -- only the call
    record is added.

    :param name: The attribute name being wrapped, recorded on each call.
    :type name: str
    :param original: The unbound method to delegate to.
    :param seen: Mutable list the call record is appended to.
    :type seen: list
    :return: A replacement unbound method.
    """
    def _spy(self, *args, **kwargs):
        seen.append(name)
        return original(self, *args, **kwargs)
    return _spy


def _max_abs_diff(a, b) -> float:
    """Max absolute elementwise difference of two tensors, as a Python float."""
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64)
                               - np.asarray(b, dtype=np.float64))))


# The grid. Token counts: 12 (divisible by num_experts=6), 15 (not divisible by
# num_experts=4) and 17 (prime, rank-2 input so tokens == batch).
_SHAPES = {
    # name: (input_shape, num_experts, d_model, hidden, output_dim)
    'divides':     ((3, 4, 9), 6, 9, 13, 8),
    'not_divides': ((3, 5, 9), 4, 9, 13, 8),
    'prime_rank2': ((17, 9), 4, 9, 13, 8),
}

_TOP_K = {
    'divides': (1, 2, 6),
    'not_divides': (1, 2, 4),
    'prime_rank2': (1, 2, 4),
}

_GRID = [
    (shape_name, top_k, gating_type)
    for shape_name in _SHAPES
    for top_k in _TOP_K[shape_name]
    for gating_type in ('linear', 'cosine')
]


# ---------------------------------------------------------------------
# 3b: the equivalence grid
# ---------------------------------------------------------------------

@pytest.mark.parametrize("shape_name,top_k,gating_type", _GRID)
def test_the_sparse_kernel_matches_the_dense_oracle(shape_name, top_k, gating_type):
    """Sparse and dense hard routing agree to ``atol=1e-5, rtol=0``.

    One row of the pre-committed grid: token counts that do and do not divide
    ``num_experts`` plus a prime token count, ``top_k`` in
    ``{1, 2, num_experts}``, both gating types that hard-route.
    """
    input_shape, num_experts, d_model, hidden, output_dim = _SHAPES[shape_name]
    keras.utils.set_random_seed(1234)
    layer = _build_moe(
        num_experts=num_experts,
        top_k=top_k,
        d_model=d_model,
        hidden=hidden,
        output_dim=output_dim,
        gating_type=gating_type,
    )
    x = keras.ops.convert_to_tensor(
        np.random.default_rng(7).normal(size=input_shape).astype('float32'))

    sparse = layer(x, training=False)
    dense = _dense_output(layer, x)

    assert tuple(sparse.shape) == tuple(dense.shape), (
        f"shape disagreement: sparse {sparse.shape} vs dense {dense.shape}")
    diff = _max_abs_diff(dense, sparse)
    assert diff <= ATOL, (
        f"{shape_name}/top_k={top_k}/{gating_type}: "
        f"max|dense - sparse| = {diff:.3e} exceeds the pre-committed {ATOL:.0e}")
    np.testing.assert_allclose(
        np.asarray(sparse), np.asarray(dense), atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------
# 4b: the same equivalence with the router's NOISE path live
# ---------------------------------------------------------------------

def _pin_the_noise_draw(monkeypatch, calls):
    """Replace ``keras.random.normal`` with a deterministic shape-keyed draw.

    Resetting a global seed between the two arms does NOT work here: Keras'
    ``SeedGenerator`` is stateful and is not re-pinned by
    ``keras.utils.set_random_seed`` mid-process (finding G-6; the reviewer's
    seed-reset attempt produced a phantom ``1.504e+00`` diff that was the
    instrument, not the kernels). Pinning the draw itself is the only sound
    instrument: a fresh ``default_rng(4242)`` per call means two calls with the
    same shape return bit-identical values, in either arm, in any order.

    :param monkeypatch: pytest's monkeypatch fixture.
    :param calls: Mutable list appended to on every intercepted draw, so the
        test can assert the noise path was actually reached.
    :type calls: list
    """
    def _pinned(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
        dims = tuple(int(d) for d in np.asarray(keras.ops.convert_to_numpy(shape)).reshape(-1))
        calls.append(dims)
        arr = np.random.default_rng(4242).standard_normal(dims) * stddev + mean
        return keras.ops.convert_to_tensor(arr.astype('float32'), dtype=dtype)

    monkeypatch.setattr(keras.random, 'normal', _pinned)


@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_the_kernels_agree_with_the_router_noise_on(monkeypatch, top_k):
    """Dense and sparse agree at ``atol=1e-5, rtol=0`` with ``add_noise=True``.

    This is the first of the two areas the iteration-2 sparse-kernel attack
    named as UNVERIFIED: every other equivalence cell in this module runs with
    the noise path off, so the noisy router's ``top_k`` selection -- which can
    differ from the noiseless one, and is what the sparse gather/scatter is
    indexed by -- was never compared across the two kernels.

    Three anti-vacuity assertions, because the obvious way to write this test is
    one that cannot fail: the intercepted draw must actually have happened, the
    noisy output must differ from the noiseless one, and ``training=True`` must
    be passed (the noise is training-gated).
    """
    calls = []
    _pin_the_noise_draw(monkeypatch, calls)

    keras.utils.set_random_seed(1234)
    layer = _build_moe(num_experts=4, top_k=top_k, d_model=9, hidden=13,
                       output_dim=8, gating_type='linear', add_noise=True)
    x = keras.ops.convert_to_tensor(
        np.random.default_rng(7).normal(size=(3, 5, 9)).astype('float32'))

    sparse = layer(x, training=True)
    assert calls, "the pinned noise draw was never reached: the cell is vacuous"
    n_after_sparse = len(calls)
    dense = _dense_output_training(layer, x)
    assert len(calls) > n_after_sparse, "the dense arm did not draw noise"

    diff = _max_abs_diff(dense, sparse)
    assert diff <= ATOL, (
        f"add_noise=True/top_k={top_k}: max|dense - sparse| = {diff:.3e} "
        f"exceeds the pre-committed {ATOL:.0e}")
    np.testing.assert_allclose(
        np.asarray(sparse), np.asarray(dense), atol=ATOL, rtol=RTOL)

    # The noise is not a no-op: without it the output is a different tensor.
    noiseless = layer(x, training=False)
    assert _max_abs_diff(noiseless, sparse) > 0.0, (
        "noisy and noiseless outputs are identical -- the noise path is inert, "
        "so this cell would pass with the noise code deleted")


def _dense_output_training(layer, x):
    """``_dense_output``'s ``training=True`` twin, for the noise path.

    Kept separate rather than adding a flag to ``_dense_output``: every other
    caller in this module is deliberately ``training=False``, and a defaulted
    flag is how a training-gated path silently stops being exercised.
    """
    original = layer._process_hard_routing_sparse
    layer._process_hard_routing_sparse = layer._process_hard_routing_dense
    try:
        return layer(x, training=True)
    finally:
        layer._process_hard_routing_sparse = original


def test_softmoe_never_reaches_the_hard_routing_kernels():
    """SoftMoE is out of scope for this gate, by code path and not by assumption.

    ``MixtureOfExperts.call`` sends ``gating_type == 'softmoe'`` to
    ``_process_softmoe``; it computes a weighted combination over *all* experts
    and never produces top-k indices, so neither hard-routing kernel runs and
    there is nothing for this module to compare. Asserted rather than stated,
    so a future dispatch change cannot silently pull SoftMoE into an untested
    kernel.
    """
    seen = []
    patched = {name: getattr(MixtureOfExperts, name) for name in _KERNEL_NAMES}
    for name, original in patched.items():
        setattr(MixtureOfExperts, name, _make_spy(name, original, seen))

    try:
        keras.utils.set_random_seed(11)
        config = MoEConfig(
            num_experts=4,
            expert_config=ExpertConfig(
                ffn_config={'type': 'mlp', 'hidden_dim': 13,
                            'output_dim': 9, 'activation': 'gelu'}
            ),
            gating_config=GatingConfig(gating_type='softmoe', num_slots=2),
            jitter_noise=0.0,
        )
        layer = MixtureOfExperts(config)
        x = keras.ops.convert_to_tensor(
            np.random.default_rng(3).normal(size=(3, 5, 9)).astype('float32'))
        out = layer(x, training=False)
        assert out is not None
    finally:
        for name, original in patched.items():
            setattr(MixtureOfExperts, name, original)

    assert seen == [], f"SoftMoE reached a hard-routing kernel: {seen}"


# ---------------------------------------------------------------------
# 3d: top_k == num_experts takes the dense path, by code path
# ---------------------------------------------------------------------

@pytest.mark.parametrize("gating_type", ('linear', 'cosine'))
def test_top_k_equal_num_experts_dispatches_to_the_dense_path(gating_type):
    """``top_k == num_experts`` runs the dense kernel *itself*, not an equal-valued twin.

    Asserted by code path (a spy on both kernels), because "the outputs match"
    cannot distinguish "dispatched to dense" from "the sparse kernel happened to
    agree" — and the whole point of the branch is that there is no sparsity to
    exploit when every expert is selected.
    """
    calls = []
    patched = {name: getattr(MixtureOfExperts, name) for name in _KERNEL_NAMES}
    for name, original in patched.items():
        setattr(MixtureOfExperts, name, _make_spy(name, original, calls))
    try:
        keras.utils.set_random_seed(21)
        layer = _build_moe(num_experts=5, top_k=5, d_model=9, hidden=13,
                           output_dim=8, gating_type=gating_type)
        x = keras.ops.convert_to_tensor(
            np.random.default_rng(5).normal(size=(3, 4, 9)).astype('float32'))
        layer(x, training=False)
    finally:
        for name, original in patched.items():
            setattr(MixtureOfExperts, name, original)

    assert calls == ['_process_hard_routing_dense'], (
        f"expected exactly one dense-kernel call, got {calls}")


# ---------------------------------------------------------------------
# 3e: a starved expert
# ---------------------------------------------------------------------

def _starve(layer, x, num_experts):
    """Zero the gate kernel so every token routes to the lowest expert indices.

    With all gate logits equal, ``keras.ops.top_k`` breaks ties toward the lowest
    index, so with ``top_k=2`` experts ``2..num_experts-1`` receive **zero**
    tokens. Deterministic starvation, rather than hoping a random gate produces it.

    :return: the list of starved expert ids.
    :rtype: list[int]
    """
    layer(x, training=False)  # build
    gate_dense = layer.gating_network.gate_dense
    kernel = gate_dense.weights[0]
    kernel.assign(keras.ops.zeros_like(kernel))
    return list(range(2, num_experts))


def test_a_starved_expert_does_not_raise_and_contributes_exactly_zero():
    """A zero-length gather must run, and the starved experts must be inert.

    Two independent assertions, because either alone is weak:

    1. the sparse output still equals the dense oracle (the gather of length 0
       neither raised nor corrupted the scatter), and
    2. perturbing a starved expert's weights changes the sparse output by
       **exactly** 0.0 -- the direct statement that it contributed nothing.
    """
    num_experts, top_k = 6, 2
    keras.utils.set_random_seed(31)
    layer = _build_moe(num_experts=num_experts, top_k=top_k, d_model=9,
                       hidden=13, output_dim=8, gating_type='linear')
    x = keras.ops.convert_to_tensor(
        np.random.default_rng(13).normal(size=(3, 5, 9)).astype('float32'))

    starved = _starve(layer, x, num_experts)
    assert starved, "the starvation helper selected no experts"

    sparse = np.asarray(layer(x, training=False))
    dense = np.asarray(_dense_output(layer, x))
    diff = _max_abs_diff(dense, sparse)
    assert diff <= ATOL, f"starved-expert grid cell: max|dense - sparse| = {diff:.3e}"

    # Perturb every starved expert. Nothing routes to them, so nothing may move.
    for expert_id in starved:
        for w in layer.experts[expert_id].weights:
            w.assign(keras.ops.ones_like(w) * 37.0)
    perturbed = np.asarray(layer(x, training=False))
    assert _max_abs_diff(sparse, perturbed) == 0.0, (
        "perturbing a starved expert changed the sparse output; it is not inert")


# ---------------------------------------------------------------------
# 3f: serialization, mixed precision, gradient flow
# ---------------------------------------------------------------------

def test_the_sparse_kernel_survives_a_full_keras_round_trip(tmp_path):
    """``model.save()`` / ``load_model()`` reproduces the sparse output by VALUE."""
    keras.utils.set_random_seed(41)
    layer = _build_moe(num_experts=5, top_k=2, d_model=9, hidden=13,
                       output_dim=8, gating_type='linear')
    inputs = keras.Input(shape=(4, 9))
    model = keras.Model(inputs, layer(inputs))
    x = keras.ops.convert_to_tensor(
        np.random.default_rng(17).normal(size=(3, 4, 9)).astype('float32'))
    before = np.asarray(model(x, training=False))

    path = os.path.join(str(tmp_path), 'moe_sparse.keras')
    model.save(path)
    reloaded = keras.saving.load_model(path)
    after = np.asarray(reloaded(x, training=False))

    assert _max_abs_diff(before, after) == 0.0, "round-trip changed the output"


@pytest.mark.parametrize("policy", ('mixed_float16', 'mixed_bfloat16'))
@pytest.mark.parametrize("gating_type", ('linear', 'cosine', 'softmoe'))
def test_mixed_precision_forward_is_finite_for_every_gating_type(policy, gating_type):
    """Every gating type runs finite under both mixed policies with the sparse kernel.

    D-064's invariant is what is really under test: the routing gate is cast to
    the expert output's dtype, never the reverse. A violation raises
    ``InvalidArgumentError`` on the multiply rather than producing a wrong number,
    so a successful finite forward is the observable.
    """
    previous = keras.mixed_precision.global_policy().name
    keras.mixed_precision.set_global_policy(policy)
    try:
        keras.utils.set_random_seed(51)
        if gating_type == 'softmoe':
            config = MoEConfig(
                num_experts=4,
                expert_config=ExpertConfig(
                    ffn_config={'type': 'mlp', 'hidden_dim': 13,
                                'output_dim': 8, 'activation': 'gelu'}),
                gating_config=GatingConfig(gating_type='softmoe', num_slots=2),
                jitter_noise=0.0,
            )
            layer = MixtureOfExperts(config)
        else:
            layer = _build_moe(num_experts=5, top_k=2, d_model=9, hidden=13,
                               output_dim=8, gating_type=gating_type)
        x = keras.ops.convert_to_tensor(
            np.random.default_rng(19).normal(size=(3, 4, 9)).astype('float32'))
        out = layer(x, training=False)
        arr = np.asarray(keras.ops.cast(out, 'float32'))
        assert np.isfinite(arr).all(), f"{gating_type}/{policy}: non-finite output"
        assert keras.backend.standardize_dtype(out.dtype) == layer.compute_dtype
    finally:
        keras.mixed_precision.set_global_policy(previous)


def test_every_routed_expert_receives_gradient_under_sgd():
    """Per-weight gradient flow, measured under **SGD** and not Adam.

    Adam normalizes by the gradient's own running scale, which hides weighting
    effects from a total-``|dW|`` probe (measured elsewhere in this repo at ~0.9x
    under Adam vs 26x under SGD). SGD's step is proportional to the gradient, so
    ``total|dW| > 0`` is a direct statement that the weight was reached.

    With the sparse kernel an expert is reached only if a token routed to it, so
    the assertion is scoped to the experts the router actually selected -- which
    is the behavioural difference from the dense kernel, where every expert sees
    every token.
    """
    keras.utils.set_random_seed(61)
    num_experts, top_k = 5, 2
    layer = _build_moe(num_experts=num_experts, top_k=top_k, d_model=9,
                       hidden=13, output_dim=8, gating_type='linear')
    x = keras.ops.convert_to_tensor(
        np.random.default_rng(23).normal(size=(4, 6, 9)).astype('float32'))

    with tf.GradientTape() as tape:
        out = layer(x, training=True)
        loss = keras.ops.sum(keras.ops.square(out))
    variables = layer.trainable_variables
    grads = tape.gradient(loss, variables)

    # Which experts did the router actually select?
    _, indices, _ = layer.gating_network(
        keras.ops.reshape(x, (-1, 9)), training=False)
    routed = set(np.unique(np.asarray(indices)).tolist())
    assert len(routed) >= 2, f"degenerate routing, only {routed} selected"

    before = [np.asarray(v) for v in variables]
    keras.optimizers.SGD(learning_rate=0.1).apply_gradients(zip(grads, variables))
    moved = {v.path: float(np.sum(np.abs(np.asarray(v) - b)))
             for v, b in zip(variables, before)}

    for expert_id in sorted(routed):
        expert = layer.experts[expert_id]
        assert expert.trainable_variables, f"expert_{expert_id} has no weights"
        for v in expert.trainable_variables:
            assert moved[v.path] > 0.0, (
                f"routed expert_{expert_id} weight {v.path} did not move under SGD")

    gate_kernel = layer.gating_network.gate_dense.weights[0]
    assert moved[gate_kernel.path] > 0.0, "the gate kernel received no gradient"
