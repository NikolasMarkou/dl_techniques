"""Tests for ``RoutingModule`` -- H-Net's boundary-detection layer.

The grading oracle is ``hnet_reference_numpy.routing_reference``: the float64
NumPy transcription of ``_reference/dc.py:69-138`` written *before* this layer
existed and RED-proven by 14 injected mutations (plan decision D-011). No second
reference is written here; where a value is pinned to arithmetic rather than to
the oracle, the arithmetic is carried out in the docstring.

Four guards are owed by plan step 4, and two of them are owed in a form the plan
text got wrong:

(i)   parity against ``routing_reference`` at ``rtol=0`` and a DERIVED ``atol``;
(ii)  identity init means the layer computes RAW adjacent cosine similarity,
      pinned to a hand-computed cosine with a non-identity anti-vacuity twin;
(iii) the shift direction -- ``p_{t+1}`` comes from ``cos(h_t, h_{t+1})``. This
      guard **must** carry a non-identity-projection arm. Under the reference's
      identity init the routing pair is symmetric, so a MIRRORED implementation
      is bit-identical, not merely close, and survives an identity-init parity
      suite completely (D-011, measured: mutation M1 initially SURVIVED a
      25-test suite);
(iv)  the hard threshold is strictly ``p > 0.5``. ``dc.py:104-106`` takes
      ``argmax([1 - p, p])`` and ``argmax`` resolves a tie to the FIRST index,
      so exactly ``0.5`` is NOT a boundary. The plan text's ``p >= 0.5`` is a
      paraphrase defect (findings.md, [CORRECTED iter-1]).

Every "these agree" assertion below carries a "these differ" twin, because an
instrument that cannot report a difference has not been shown to measure
anything (House Rule 3).

Masks are built from ``keras.ops.arange`` broadcasts and never from
``keras.ops.tril``/``triu``: those raise ``TypeError: ('pred must not be a
Python bool', True)`` under plain ``tf.function`` as well as under XLA
(measured, D-010(b)), and eager ``tril`` is bitwise equal to the ``arange``
form, so a wrong call looks correct until it is traced.
"""

import numpy as np
import pytest
import keras
import tensorflow as tf

from dl_techniques.layers.dynamic_chunking.routing_module import RoutingModule
from tests.numerics import matmul_precision_atol, matmul_unit_roundoff

from .hnet_reference_numpy import routing_reference

# ---------------------------------------------------------------------
# Derived tolerance
# ---------------------------------------------------------------------

_F32_U = float(np.finfo(np.float32).eps) / 2.0   # 5.96e-08, unit roundoff
_TAIL_FACTOR = 8.0                               # 8-sigma tail on the random walk


def routing_parity_atol(d_model: int, scale: float) -> float:
    """Bound on ``|float32 RoutingModule output - float64 oracle output|``.

    **Two arithmetic regimes, one expression.** The derivation below assumes the device
    performs TRUE float32 matmuls. On a tensor-core GPU with TF32 enabled -- the DEFAULT
    on this box -- ``q_proj``/``k_proj`` are computed with a 10-bit mantissa instead, and
    that is a different regime, not a defect: see the module-level note under
    :func:`~tests.numerics.matmul_unit_roundoff`. The returned bound is therefore the
    MAXIMUM of the float32 random-walk term derived below and
    :func:`~tests.numerics.matmul_precision_atol`, which is 4 unit roundoffs of whatever
    precision the active device's matmul was MEASURED to use. On CPU and on a
    TF32-disabled GPU the second term is ~2.4e-07 and the first one wins at every
    ``d_model``, so the CPU bound -- and with it every CPU-measured RED proof in this
    file -- is bit-for-bit what it was before the regime term was added.

    Interface contract: pure function, no state, never raises for
    ``d_model >= 0``; returns a strictly positive float. Callers MUST pass
    ``rtol=0`` -- ``assert_allclose``'s default ``rtol=1e-7`` is of the same
    order as this whole bound at ``d_model = 8`` and would make it decorative.

    Derivation (so this is a bound, not a pasted magic number). The compared
    quantity is a boundary probability of order 1, computed once per position
    along this dependency chain, per side of the pair:

    ==========================  ==========================  ===========
    step                        rounded float32 operations  count
    ==========================  ==========================  ===========
    ``h @ kernel``              D multiplies + D-1 adds     ``2D - 1``
    ``sum(x*x)``                D multiplies + D-1 adds     ``2D - 1``
    ``sqrt`` and ``max``        1 + 0                       ``1``
    divide by the norm          1 per component             ``1``
    ==========================  ==========================  ===========

    i.e. ``4D`` per side, ``8D`` for the ``q``/``k`` pair, plus ``2D - 1`` for
    the cosine contraction and ``2`` for ``(1 - cos) / 2`` (the halving is exact,
    counted anyway), giving::

        M = 10 * D + 1

    Rounding errors are not adversarially aligned, so over ``M`` rounded
    operations they accumulate as a random walk of relative size
    ``sqrt(M) * u``. Take an 8-sigma tail and scale by the output magnitude::

        atol = 8 * sqrt(10 * D + 1) * u_f32 * max(1, |output|)

    Only ONE side is charged. The other side is the float64 oracle, whose own
    unit roundoff is ``1.11e-16`` -- nine orders of magnitude below this -- so it
    contributes nothing at this precision.

    **Why this is not** ``tests/numerics.reassociation_atol``, which plan step 4
    named. That helper answers a different question and its op count does not
    describe this path. It is written for two REASSOCIATED float32 evaluations of
    one formula, so it charges BOTH sides (``M = 2 * num_steps * sum(L)``); here
    the second side is float64 and charging it is simply wrong. More
    importantly, its ``reduction_lengths`` signature counts contraction lengths
    only: applied to this path as ``reassociation_atol([D], 1, scale)`` it yields
    ``M = 2D`` and misses the entire L2-normalize chain -- at ``D = 8`` that is
    ``1.91e-06`` against the ``4.29e-06`` derived here, i.e. it would
    UNDER-count, not over-loosen. A bound below a correct implementation's own
    noise floor can never pass, which is the exact defect the D-024 comment at
    the top of `tests/numerics.py` records. The divergence from the plan text is
    recorded in decisions.md D-012.

    :param d_model: Hidden width ``D``.
    :type d_model: int
    :param scale: Magnitude of the compared output, ``max|expected|``.
    :type scale: float
    :return: Absolute tolerance, valid in whichever matmul regime is currently active.
    :rtype: float
    """
    ops_count = 10.0 * float(d_model) + 1.0
    float32_term = _TAIL_FACTOR * np.sqrt(ops_count) * _F32_U * max(1.0, float(scale))
    return max(float32_term, matmul_precision_atol(scale))


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

#: The 4-token, 2-channel example D-011 pinned on paper. ``(3,4)``, ``(4,3)`` and
#: ``(-3,4)`` all have norm exactly 5, so every cosine is an exact small rational.
PIN_HIDDEN = np.array(
    [[[1.0, 0.0], [3.0, 4.0], [4.0, 3.0], [-3.0, 4.0]]], dtype=np.float32
)

#: ``nn.Linear``-layout 90-degree rotation used by D-011's orientation pin.
#: ``keras.layers.Dense`` computes ``x @ kernel`` while ``nn.Linear`` computes
#: ``x @ W.T``, so the kernel is the TRANSPOSE of the oracle's ``q_weight``.
ROT90_LINEAR_WEIGHT = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32)


def _padding_mask(batch_size: int, seq_len: int, lengths) -> np.ndarray:
    """Boolean ``(B, L)`` validity mask built from an ``arange`` broadcast.

    Never ``tril``/``triu`` -- see this module's docstring.

    :param batch_size: Number of rows.
    :type batch_size: int
    :param seq_len: Sequence length.
    :type seq_len: int
    :param lengths: Per-row count of valid leading positions.
    :type lengths: Sequence[int]
    :return: ``(B, L)`` boolean array, ``True`` = valid.
    :rtype: numpy.ndarray
    """
    positions = keras.ops.arange(seq_len)[None, :]
    limits = keras.ops.convert_to_tensor(np.asarray(lengths, dtype="int32"))[:, None]
    mask = keras.ops.convert_to_numpy(keras.ops.less(positions, limits))
    assert mask.shape == (batch_size, seq_len)
    return mask.astype(bool)


def _run(layer, hidden, mask=None):
    """Call ``layer`` and return the three outputs as NumPy arrays.

    :param layer: The :class:`RoutingModule` under test.
    :type layer: RoutingModule
    :param hidden: ``(B, L, D)`` float32 array.
    :type hidden: numpy.ndarray
    :param mask: ``(B, L)`` boolean array or ``None``.
    :type mask: numpy.ndarray or None
    :return: ``(boundary_prob, boundary_mask, selected_probs)`` as NumPy.
    :rtype: tuple
    """
    kwargs = {}
    if mask is not None:
        kwargs["mask"] = keras.ops.convert_to_tensor(mask)
    outputs = layer(keras.ops.convert_to_tensor(hidden), **kwargs)
    return tuple(keras.ops.convert_to_numpy(o) for o in outputs)


def _set_linear_weights(layer, q_linear_weight=None, k_linear_weight=None):
    """Overwrite the projections with ``nn.Linear``-layout weights.

    The layer stores ``keras.layers.Dense`` kernels (``x @ kernel``); the oracle
    takes ``nn.Linear`` weights (``x @ W.T``). This helper transposes, so a test
    and the oracle can be handed the SAME matrix.

    :param layer: A BUILT :class:`RoutingModule`.
    :type layer: RoutingModule
    :param q_linear_weight: ``(D, D)`` weight for ``q_proj``, or ``None``.
    :type q_linear_weight: numpy.ndarray or None
    :param k_linear_weight: ``(D, D)`` weight for ``k_proj``, or ``None``.
    :type k_linear_weight: numpy.ndarray or None
    """
    if q_linear_weight is not None:
        layer.q_proj.kernel.assign(np.asarray(q_linear_weight, "float32").T)
    if k_linear_weight is not None:
        layer.k_proj.kernel.assign(np.asarray(k_linear_weight, "float32").T)


def _mirrored_boundary_prob(hidden, q_linear_weight=None, k_linear_weight=None):
    """The MIRRORED routing probability: ``q`` reads ``h_{t+1}``, ``k`` reads ``h_t``.

    This is mutation M1 of D-011 -- the ``ops.roll``-class direction defect -- and
    it is expressed here as a float64 NumPy computation so a test can state
    executably WHY the identity init cannot see it. It is not a second oracle:
    it is the defect, and it is only ever asserted DIFFERENT from the layer.

    :param hidden: ``(B, L, D)`` array.
    :type hidden: numpy.ndarray
    :param q_linear_weight: ``(D, D)`` ``nn.Linear``-layout weight, or ``None``
        for the identity.
    :type q_linear_weight: numpy.ndarray or None
    :param k_linear_weight: same, for ``k``.
    :type k_linear_weight: numpy.ndarray or None
    :return: ``(B, L)`` float64 boundary probability, position 0 forced to 1.0.
    :rtype: numpy.ndarray
    """
    hidden = np.asarray(hidden, dtype=np.float64)
    d_model = hidden.shape[-1]
    q_w = np.eye(d_model) if q_linear_weight is None else np.asarray(
        q_linear_weight, dtype=np.float64
    )
    k_w = np.eye(d_model) if k_linear_weight is None else np.asarray(
        k_linear_weight, dtype=np.float64
    )

    def _norm(x):
        n = np.sqrt(np.sum(x * x, axis=-1, keepdims=True))
        return x / np.maximum(n, 1e-12)

    # The ONLY change from the oracle: the slices are swapped.
    q = _norm(hidden[:, 1:] @ q_w.T)
    k = _norm(hidden[:, :-1] @ k_w.T)
    cos_sim = np.einsum("bld,bld->bl", q, k)
    prob = np.clip((1.0 - cos_sim) / 2.0, 0.0, 1.0)
    return np.pad(prob, ((0, 0), (1, 0)), mode="constant", constant_values=1.0)


# =====================================================================
# Construction, configuration and validation
# =====================================================================


class TestInitAndConfig:
    """Constructor, stored configuration and the serialization round trip."""

    def test_init_stores_config_and_creates_both_projections_unbuilt(self):
        layer = RoutingModule(d_model=16)
        assert layer.d_model == 16
        assert layer.q_proj is not None and layer.k_proj is not None
        # Sub-layers are CREATED in __init__ and must NOT yet own weights.
        assert not layer.built
        assert not layer.q_proj.built
        assert not layer.k_proj.built

    def test_build_creates_exactly_two_identity_kernels_and_no_bias(self):
        layer = RoutingModule(d_model=4)
        layer.build((None, 7, 4))
        weights = layer.weights
        assert len(weights) == 2, [w.path for w in weights]
        for kernel in (layer.q_proj.kernel, layer.k_proj.kernel):
            value = keras.ops.convert_to_numpy(kernel)
            assert value.shape == (4, 4)
            np.testing.assert_array_equal(value, np.eye(4, dtype="float32"))
        assert layer.q_proj.bias is None and layer.k_proj.bias is None

    def test_get_config_round_trip_reproduces_behaviour(self):
        layer = RoutingModule(d_model=6, name="router")
        config = layer.get_config()
        assert config["d_model"] == 6
        assert config["name"] == "router"

        clone = RoutingModule.from_config(config)
        assert clone.d_model == 6

        rng = np.random.default_rng(11)
        hidden = rng.standard_normal((2, 5, 6)).astype("float32")
        original = _run(layer, hidden)
        rebuilt = _run(clone, hidden)
        # AGREE: same config, same identity init, same numbers.
        np.testing.assert_array_equal(original[0], rebuilt[0])
        np.testing.assert_array_equal(original[1], rebuilt[1])

        # DIFFER twin: the comparison above is not comparing constants.
        other = rng.standard_normal((2, 5, 6)).astype("float32")
        assert np.max(np.abs(_run(clone, other)[0] - rebuilt[0])) > 1e-2

    def test_layer_is_registered_under_the_package_qualified_key(self):
        key = keras.saving.get_registered_name(RoutingModule)
        assert key == "dl_techniques.layers.dynamic_chunking.routing_module>RoutingModule"
        assert keras.saving.get_registered_object(key) is RoutingModule


class TestValidation:
    """Every raise, and the input shapes that must reach one."""

    @pytest.mark.parametrize("bad", [0, -1, -16])
    def test_non_positive_d_model_raises(self, bad):
        with pytest.raises(ValueError, match="must be positive"):
            RoutingModule(d_model=bad)

    @pytest.mark.parametrize("bad", [8.0, "8", None, True])
    def test_non_int_d_model_raises(self, bad):
        with pytest.raises(ValueError, match="must be an int"):
            RoutingModule(d_model=bad)

    def test_rank_two_input_raises(self):
        layer = RoutingModule(d_model=4)
        with pytest.raises(ValueError, match="rank-3"):
            layer.build((None, 4))

    def test_last_axis_mismatch_raises(self):
        layer = RoutingModule(d_model=4)
        with pytest.raises(ValueError, match="must equal d_model"):
            layer.build((None, 7, 5))

    def test_a_correct_shape_does_not_raise(self):
        """Anti-vacuity twin for the two raises above."""
        RoutingModule(d_model=4).build((None, 7, 4))


# =====================================================================
# Forward pass
# =====================================================================


class TestForwardPass:
    """Shapes, dtypes, invariants that hold for every input."""

    @pytest.mark.parametrize(
        "batch_size,seq_len,d_model", [(1, 2, 2), (3, 7, 8), (2, 33, 16)]
    )
    def test_output_shapes_and_dtypes(self, batch_size, seq_len, d_model):
        rng = np.random.default_rng(3)
        hidden = rng.standard_normal((batch_size, seq_len, d_model)).astype("float32")
        layer = RoutingModule(d_model=d_model)
        prob, mask, selected = _run(layer, hidden)

        assert prob.shape == (batch_size, seq_len, 2)
        assert mask.shape == (batch_size, seq_len)
        assert selected.shape == (batch_size, seq_len, 1)
        assert prob.dtype == np.float32 and selected.dtype == np.float32
        assert mask.dtype == np.bool_

    def test_compute_output_shape_matches_the_real_call_on_an_unbuilt_layer(self):
        layer = RoutingModule(d_model=8)
        declared = layer.compute_output_shape((None, 7, 8))
        assert not layer.built  # must be answerable from stored config alone

        rng = np.random.default_rng(4)
        actual = _run(layer, rng.standard_normal((3, 7, 8)).astype("float32"))
        for declared_shape, array in zip(declared, actual):
            assert declared_shape[1:] == array.shape[1:]

    def test_probabilities_are_a_finite_two_class_distribution(self):
        rng = np.random.default_rng(5)
        hidden = (rng.standard_normal((4, 12, 8)) * 30.0).astype("float32")
        prob, _, selected = _run(RoutingModule(d_model=8), hidden)
        assert np.all(np.isfinite(prob)) and np.all(np.isfinite(selected))
        assert np.all(prob >= 0.0) and np.all(prob <= 1.0)
        # Derived: `[1 - p, p]` is two float32 values whose exact sum is 1, so the
        # rounded sum is off by at most 2 unit roundoffs of 1.0, and `p` itself carries
        # the routing chain's own error -- `routing_parity_atol` bounds both.
        np.testing.assert_allclose(
            prob.sum(axis=-1), 1.0, rtol=0, atol=routing_parity_atol(8, 1.0)
        )

    def test_position_zero_is_always_a_boundary_and_always_has_probability_one(self):
        rng = np.random.default_rng(6)
        hidden = rng.standard_normal((5, 9, 8)).astype("float32")
        prob, mask, selected = _run(RoutingModule(d_model=8), hidden)
        np.testing.assert_array_equal(prob[:, 0, 1], np.ones(5, dtype="float32"))
        assert np.all(mask[:, 0])
        np.testing.assert_array_equal(selected[:, 0, 0], np.ones(5, dtype="float32"))

        # DIFFER twin: the forcing is specific to position 0, not a constant
        # applied to the whole row.
        assert not np.all(prob[:, 1:, 1] == 1.0)

    def test_padded_positions_are_never_boundaries(self):
        rng = np.random.default_rng(7)
        hidden = rng.standard_normal((3, 10, 8)).astype("float32")
        mask = _padding_mask(3, 10, [10, 6, 1])

        _, unmasked_boundary, _ = _run(RoutingModule(d_model=8), hidden)
        _, masked_boundary, _ = _run(RoutingModule(d_model=8), hidden, mask=mask)

        assert not np.any(masked_boundary[~mask])
        # DIFFER twin: the mask actually removed boundaries; if the unmasked run
        # had no boundary in the padded region there would be nothing to prove.
        assert np.any(unmasked_boundary[~mask])
        # AGREE: valid positions are untouched by masking.
        np.testing.assert_array_equal(masked_boundary[mask], unmasked_boundary[mask])

    def test_selected_probs_reports_the_winning_class_even_at_a_padded_position(self):
        """`dc.py:130-132` gathers from the UNMASKED decision."""
        rng = np.random.default_rng(8)
        hidden = rng.standard_normal((2, 12, 8)).astype("float32")
        mask = _padding_mask(2, 12, [12, 4])
        prob, _, selected = _run(RoutingModule(d_model=8), hidden, mask=mask)
        winner = np.max(prob, axis=-1)
        np.testing.assert_array_equal(selected[..., 0], winner)

    def test_training_flag_does_not_change_the_output(self):
        rng = np.random.default_rng(9)
        hidden = keras.ops.convert_to_tensor(
            rng.standard_normal((3, 11, 8)).astype("float32")
        )
        layer = RoutingModule(d_model=8)
        train = keras.ops.convert_to_numpy(layer(hidden, training=True)[0])
        infer = keras.ops.convert_to_numpy(layer(hidden, training=False)[0])
        # AGREE: no dropout, no normalization statistics, nothing mode-dependent.
        np.testing.assert_array_equal(train, infer)

        # DIFFER twin: the comparison is on live numbers, not on a constant.
        other = keras.ops.convert_to_tensor(
            rng.standard_normal((3, 11, 8)).astype("float32")
        )
        assert np.max(
            np.abs(keras.ops.convert_to_numpy(layer(other, training=False)[0]) - infer)
        ) > 1e-2

    def test_the_forward_pass_traces_under_tf_function(self):
        """The selection path must survive graph tracing, not only eager."""
        layer = RoutingModule(d_model=8)
        rng = np.random.default_rng(10)
        hidden = keras.ops.convert_to_tensor(
            rng.standard_normal((2, 9, 8)).astype("float32")
        )
        mask = keras.ops.convert_to_tensor(_padding_mask(2, 9, [9, 5]))

        @tf.function
        def traced(h, m):
            prob, boundary, selected = layer(h, mask=m)
            return prob, tf.cast(boundary, tf.float32), selected

        eager = _run(layer, keras.ops.convert_to_numpy(hidden),
                     keras.ops.convert_to_numpy(mask))
        graph = [np.asarray(t) for t in traced(hidden, mask)]
        np.testing.assert_array_equal(graph[0], eager[0])
        np.testing.assert_array_equal(graph[1], eager[1].astype("float32"))
        np.testing.assert_array_equal(graph[2], eager[2])


# =====================================================================
# Guard (i): parity against the float64 oracle
# =====================================================================


class TestGuardOneOracleParity:
    """The layer reproduces ``routing_reference`` at ``rtol=0``, derived ``atol``."""

    @pytest.mark.parametrize("seed,batch_size,seq_len,d_model", [
        (101, 1, 2, 2),
        (102, 3, 7, 8),
        (103, 2, 16, 4),
        (104, 4, 33, 16),
    ])
    def test_parity_without_a_mask(self, seed, batch_size, seq_len, d_model):
        rng = np.random.default_rng(seed)
        hidden = rng.standard_normal((batch_size, seq_len, d_model)).astype("float32")
        prob, boundary, selected = _run(RoutingModule(d_model=d_model), hidden)
        ref_prob, ref_boundary, ref_selected = routing_reference(hidden)

        atol = routing_parity_atol(d_model, float(np.max(np.abs(ref_prob))))
        np.testing.assert_allclose(prob, ref_prob, rtol=0, atol=atol)
        np.testing.assert_allclose(selected, ref_selected, rtol=0, atol=atol)
        np.testing.assert_array_equal(boundary, ref_boundary)

    @pytest.mark.parametrize("seed,lengths", [
        (201, [10, 6, 1]),
        (202, [10, 10, 9]),
        (203, [2, 5, 7]),
    ])
    def test_parity_with_a_padding_mask(self, seed, lengths):
        rng = np.random.default_rng(seed)
        hidden = rng.standard_normal((3, 10, 8)).astype("float32")
        mask = _padding_mask(3, 10, lengths)
        prob, boundary, selected = _run(RoutingModule(d_model=8), hidden, mask=mask)
        ref_prob, ref_boundary, ref_selected = routing_reference(hidden, mask=mask)

        atol = routing_parity_atol(8, float(np.max(np.abs(ref_prob))))
        np.testing.assert_allclose(prob, ref_prob, rtol=0, atol=atol)
        np.testing.assert_allclose(selected, ref_selected, rtol=0, atol=atol)
        np.testing.assert_array_equal(boundary, ref_boundary)

    def test_parity_survives_a_non_identity_weight_pair(self):
        """Parity must not be an artifact of the identity init."""
        rng = np.random.default_rng(301)
        q_w = rng.standard_normal((8, 8)).astype("float32")
        k_w = rng.standard_normal((8, 8)).astype("float32")
        hidden = rng.standard_normal((3, 12, 8)).astype("float32")

        layer = RoutingModule(d_model=8)
        layer.build((None, 12, 8))
        _set_linear_weights(layer, q_w, k_w)

        prob, boundary, selected = _run(layer, hidden)
        ref_prob, ref_boundary, ref_selected = routing_reference(
            hidden, q_weight=q_w, k_weight=k_w
        )
        atol = routing_parity_atol(8, float(np.max(np.abs(ref_prob))))
        np.testing.assert_allclose(prob, ref_prob, rtol=0, atol=atol)
        np.testing.assert_allclose(selected, ref_selected, rtol=0, atol=atol)
        np.testing.assert_array_equal(boundary, ref_boundary)

    def test_the_parity_instrument_can_report_a_mismatch(self):
        """DIFFER twin for the whole class: the bound is not unfailable.

        The oracle is fed the SAME hidden states rolled by one position -- an
        O(1) perturbation of the boundary probabilities -- and the parity
        assertion must fail. Without this, an ``atol`` accidentally set to
        infinity would look like four passing tests.
        """
        rng = np.random.default_rng(302)
        hidden = rng.standard_normal((3, 12, 8)).astype("float32")
        prob, _, _ = _run(RoutingModule(d_model=8), hidden)
        wrong_prob, _, _ = routing_reference(np.roll(hidden, 1, axis=1))

        atol = routing_parity_atol(8, float(np.max(np.abs(wrong_prob))))
        observed = float(np.max(np.abs(prob - wrong_prob)))
        assert observed > atol, (
            f"instrument is blind: max|delta| = {observed:.3e} <= atol = {atol:.3e}"
        )

    def test_the_derived_bound_sits_above_the_active_devices_measured_noise(self):
        """The bound must be ATTAINABLE, not merely non-infinite.

        A bound below a correct implementation's own noise floor can never pass
        and therefore measures nothing (`tests/numerics.py`, D-024). This records
        the ratio so a future change that shrinks the bound is caught here rather
        than by a mysterious red elsewhere.

        **This runs in whichever matmul regime is active, and it is the arm that
        proved the old bound unattainable on a GPU.** MEASURED 2026-09-09, worst of
        16 draws at ``d_model = 8``, and the bound each regime derives:

        ==============================  ============  ==========  =====
        regime                          measured      bound       ratio
        ==============================  ============  ==========  =====
        CPU (true float32)              8.65e-08      4.29e-06    0.02
        GPU RTX 4090, TF32 DISABLED     8.65e-08      4.29e-06    0.02
        GPU RTX 4090, TF32 ON (default) 1.51e-04      1.95e-03    0.08
        ==============================  ============  ==========  =====

        The middle row is the load-bearing one: with TF32 off, the GPU attains the
        float32 bound exactly, so that bound is NOT unattainable hardware -- it is
        unattainable *arithmetic*, and the regime term is what names the difference.
        Before the regime term existed this test read 1.51e-04 against 4.29e-06 on
        GPU 0 and was RED.
        """
        rng = np.random.default_rng(303)
        worst = 0.0
        for _ in range(16):
            hidden = rng.standard_normal((4, 24, 8)).astype("float32")
            prob, _, _ = _run(RoutingModule(d_model=8), hidden)
            ref_prob, _, _ = routing_reference(hidden)
            worst = max(worst, float(np.max(np.abs(prob - ref_prob))))
        atol = routing_parity_atol(8, 1.0)
        regime = (
            "reduced-precision (TF32)"
            if matmul_unit_roundoff() > _F32_U
            else "true float32"
        )
        assert 0.0 < worst < atol, (
            f"{regime} matmul: measured {worst:.3e} vs bound {atol:.3e}"
        )

    def test_the_bound_cannot_go_vacuous_in_either_regime(self):
        """A regime-aware bound must not become a licence to pass anything.

        The signal these guards exist to catch is O(0.1): the DIFFER twin above
        perturbs the oracle by rolling the sequence one position, and a boundary
        probability is O(1). MEASURED, the loosest bound this file can produce is
        the TF32 one, ``4 * 2**-11 = 1.95e-03`` per unit of output scale -- two
        orders below the signal. This test pins that gap so a future "just widen
        the allowance" edit is caught here and not by a guard silently going
        blind. It is deliberately expressed against the LOOSEST regime, so it is
        equally meaningful on CPU, where the returned bound is 450x tighter still.
        """
        for d_model in (2, 4, 8, 16, 32):
            assert routing_parity_atol(d_model, 1.0) < 1.0e-2, d_model
        # And the regime term itself is one of exactly two known values.
        assert matmul_unit_roundoff() in (_F32_U, 2.0 ** -11)


# =====================================================================
# Guard (ii): identity init computes RAW adjacent cosine similarity
# =====================================================================


class TestGuardTwoIdentityInitIsRawCosine:
    """At step 0 the layer is exactly adjacent cosine dissimilarity."""

    def test_hand_computed_cosine_on_the_four_token_pin(self):
        """D-011's pin 1, re-derived here rather than imported.

        ``h = [[1,0], [3,4], [4,3], [-3,4]]``; ``(3,4)``, ``(4,3)`` and
        ``(-3,4)`` all have norm exactly 5, and ``[1,0]`` has norm 1::

            cos(h0, h1) = (1*3 + 0*4) / (1 * 5)   = 3/5   = 0.60
            cos(h1, h2) = (3*4 + 4*3) / (5 * 5)   = 24/25 = 0.96
            cos(h2, h3) = (4*-3 + 3*4) / (5 * 5)  = 0/25  = 0.00

            p = (1 - cos)/2, left-padded with 1.0
              = [1.00, 0.20, 0.02, 0.50]

        Position 3 lands on the tie exactly; guard (iv) owns its consequence.
        Every value above is a dyadic rational except 0.20 and 0.02, so the
        comparison uses the derived bound rather than ``atol=0``.
        """
        prob, _, selected = _run(RoutingModule(d_model=2), PIN_HIDDEN)
        expected_p = np.array([[1.00, 0.20, 0.02, 0.50]])
        expected_selected = np.array([[1.00, 0.80, 0.98, 0.50]])

        atol = routing_parity_atol(2, 1.0)
        np.testing.assert_allclose(prob[..., 1], expected_p, rtol=0, atol=atol)
        np.testing.assert_allclose(
            selected[..., 0], expected_selected, rtol=0, atol=atol
        )

    def test_a_non_identity_init_moves_the_result(self):
        """Anti-vacuity twin: the pin above is a property of the IDENTITY init.

        With ``q_weight`` a 90-degree rotation the same tokens give
        ``p = [1.00, 0.10, 0.64, 0.00]`` (D-011 pin 2), so the identity claim is
        falsifiable and the layer's weights are genuinely load-bearing.
        """
        layer = RoutingModule(d_model=2)
        layer.build((None, 4, 2))
        _set_linear_weights(layer, q_linear_weight=ROT90_LINEAR_WEIGHT)

        prob, _, _ = _run(layer, PIN_HIDDEN)
        atol = routing_parity_atol(2, 1.0)
        np.testing.assert_allclose(
            prob[..., 1], np.array([[1.00, 0.10, 0.64, 0.00]]), rtol=0, atol=atol
        )
        assert np.max(
            np.abs(prob[..., 1] - np.array([[1.00, 0.20, 0.02, 0.50]]))
        ) > 0.1

    def test_a_tiny_but_nonzero_row_still_normalizes_to_UNIT_length(self):
        """The denominator is ``max(norm, 1e-12)``, never ``norm + 1e-12``.

        Both spellings agree to within float32 noise on ordinary rows and both
        give a finite answer on an all-zero row, so neither of the two tests
        around this one can tell them apart. The spellings separate only when the
        norm is within an order of magnitude of ``eps``. With
        ``h_0 = h_1 = (5e-12, 0)`` the two adjacent states are IDENTICAL, so the
        correct layer must report ``cos = 1`` and ``p = 0``::

            max(5e-12, 1e-12) = 5e-12  ->  unit vector  ->  cos = 1     -> p = 0
            5e-12 + 1e-12     = 6e-12  ->  length 5/6   ->  cos = 25/36 -> p ~ 0.153

        i.e. the defect is 0.153 away, four orders above the parity bound.
        """
        tiny = np.float32(5e-12)
        hidden = np.array([[[tiny, 0.0], [tiny, 0.0]]], dtype="float32")
        prob, _, _ = _run(RoutingModule(d_model=2), hidden)
        ref_prob, _, _ = routing_reference(hidden)

        atol = routing_parity_atol(2, 1.0)
        np.testing.assert_allclose(prob, ref_prob, rtol=0, atol=atol)
        np.testing.assert_allclose(prob[0, 1, 1], 0.0, rtol=0, atol=atol)
        # DIFFER twin: the `norm + eps` spelling is what this excludes.
        assert abs(float(prob[0, 1, 1]) - (1.0 - (25.0 / 36.0)) / 2.0) > 0.1

    def test_the_probability_is_clipped_into_the_unit_interval(self):
        """``dc.py:92``'s clamp is a no-op only when float32 behaves.

        It does not always. Searched over 28000 random float32 vectors, the
        largest self-cosine ``sum(normalize(v) * normalize(v))`` observed on this
        stack is ``1 + 2.38e-07`` at ``v = (-0.9199936, 0.6750645)``, which makes
        ``(1 - cos)/2 = -1.19e-07`` -- NEGATIVE. That is far below the parity
        bound, so no parity test can see the clamp being deleted; only a strict
        sign assertion can.
        """
        v = np.array([-0.9199936, 0.6750645], dtype="float32")
        hidden = np.stack([v, v])[None, ...]
        prob, _, selected = _run(RoutingModule(d_model=2), hidden)

        assert float(prob[0, 1, 1]) >= 0.0, float(prob[0, 1, 1])
        assert float(prob.min()) >= 0.0 and float(prob.max()) <= 1.0
        assert float(selected.min()) >= 0.0 and float(selected.max()) <= 1.0
        # DIFFER twin: the construction really does sit on the clamp -- an
        # unclamped evaluation of the same expression is strictly negative.
        unclamped = (1.0 - float(np.float32(np.dot(v / np.linalg.norm(v).astype("float32"),
                                                   v / np.linalg.norm(v).astype("float32"))))) / 2.0
        assert unclamped < 0.0 or float(prob[0, 1, 1]) == 0.0

    def test_an_all_zero_row_normalizes_to_zero_rather_than_nan(self):
        """``F.normalize`` clamps its denominator; ``norm + eps`` would not.

        With ``h_t = 0`` the normalized vector is exactly ``0``, so
        ``cos = 0`` and ``p = 0.5`` -- finite. The ``norm + eps`` spelling gives
        the same finite answer here but a different one for a tiny non-zero row,
        which the twin below drives.
        """
        hidden = np.zeros((1, 3, 4), dtype="float32")
        hidden[0, 1] = [1.0, 0.0, 0.0, 0.0]
        prob, _, _ = _run(RoutingModule(d_model=4), hidden)
        assert np.all(np.isfinite(prob))
        np.testing.assert_allclose(
            prob[0, 1:, 1], [0.5, 0.5], rtol=0, atol=routing_parity_atol(4, 1.0)
        )

        # DIFFER twin: a NON-zero row does not sit at 0.5.
        hidden[0, 2] = [-1.0, 0.0, 0.0, 0.0]
        prob, _, _ = _run(RoutingModule(d_model=4), hidden)
        assert abs(float(prob[0, 2, 1]) - 0.5) > 0.4


# =====================================================================
# Guard (iii): the shift direction, WITH a non-identity arm
# =====================================================================


class TestGuardThreeShiftDirection:
    """``p_{t+1}`` comes from ``cos(h_t, h_{t+1})``, and ``q`` reads the EARLIER token."""

    def test_a_delta_impulse_moves_exactly_two_boundary_probabilities(self):
        """Perturbing ``h_t`` may move only ``p_t`` and ``p_{t+1}``.

        An ``ops.roll``-class off-by-one shows up here as the moved pair
        landing at the wrong indices.
        """
        rng = np.random.default_rng(401)
        hidden = rng.standard_normal((1, 9, 8)).astype("float32")
        layer = RoutingModule(d_model=8)
        base, _, _ = _run(layer, hidden)

        perturbed = hidden.copy()
        perturbed[0, 4] += 3.7
        moved, _, _ = _run(layer, perturbed)

        delta = np.abs(moved[0, :, 1] - base[0, :, 1])
        touched = set(np.flatnonzero(delta > 1e-4).tolist())
        assert touched == {4, 5}, f"moved positions {sorted(touched)}, expected {{4, 5}}"
        # DIFFER twin: the probe can report movement at all.
        assert float(np.max(delta)) > 1e-2

    def test_routing_applies_q_to_the_EARLIER_token_and_k_to_the_LATER_one(self):
        """The non-identity arm guard (iii) is blind without (D-011, M1).

        With ``q_weight`` a 90-degree rotation and ``k_weight`` the identity, the
        four-token pin gives ``p = [1.00, 0.10, 0.64, 0.00]``. A MIRRORED layer
        -- ``q`` reading ``h_{t+1}``, ``k`` reading ``h_t`` -- gives
        ``p = [1.00, 0.90, 0.36, 1.00]``, which is the complement, i.e. as wrong
        as it is possible to be. Derivation of the correct arm, with
        ``R = [[0,-1],[1,0]]`` applied as ``nn.Linear`` does (``x @ R.T``)::

            q(h0) = R (1, 0)  = (0, 1)      k(h1) = (3, 4)/5
            q(h1) = R (3, 4)  = (-4, 3)     k(h2) = (4, 3)/5
            q(h2) = R (4, 3)  = (-3, 4)     k(h3) = (-3, 4)/5

            cos_0 = (0*3 + 1*4)/(1*5)        =  4/5   =  0.80 -> p = 0.10
            cos_1 = (-4*4 + 3*3)/(5*5)       = -7/25  = -0.28 -> p = 0.64
            cos_2 = (-3*-3 + 4*4)/(5*5)      = 25/25  =  1.00 -> p = 0.00
        """
        layer = RoutingModule(d_model=2)
        layer.build((None, 4, 2))
        _set_linear_weights(layer, q_linear_weight=ROT90_LINEAR_WEIGHT)

        prob, boundary, _ = _run(layer, PIN_HIDDEN)
        atol = routing_parity_atol(2, 1.0)

        correct = np.array([[1.00, 0.10, 0.64, 0.00]])
        mirrored = np.array([[1.00, 0.90, 0.36, 1.00]])
        np.testing.assert_allclose(prob[..., 1], correct, rtol=0, atol=atol)
        # DIFFER: the mirrored defect is not merely outside the bound, it is
        # 0.8 away -- this arm is what makes the guard able to see M1 at all.
        assert np.max(np.abs(prob[..., 1] - mirrored)) > 0.5
        np.testing.assert_array_equal(boundary, np.array([[True, False, True, False]]))

    def test_identity_init_makes_the_orientation_defect_INVISIBLE(self):
        """Why the arm above needs a non-identity weight, stated executably.

        Under the identity init ``cos(h_t, h_{t+1}) == cos(h_{t+1}, h_t)``, so
        the mirrored implementation is BIT-IDENTICAL -- not close, identical --
        and no identity-init test of any strength can distinguish it.
        """
        rng = np.random.default_rng(402)
        hidden = rng.standard_normal((3, 10, 8)).astype("float32")
        prob, _, _ = _run(RoutingModule(d_model=8), hidden)
        mirrored = _mirrored_boundary_prob(hidden)

        # The mirrored defect is within float32 noise of the correct layer.
        assert np.max(np.abs(prob[..., 1] - mirrored)) < routing_parity_atol(8, 1.0)

        # DIFFER twin: the SAME comparison separates them once the projection is
        # no longer symmetric, so the agreement above is a property of the init
        # and not of the comparison.
        layer = RoutingModule(d_model=2)
        layer.build((None, 4, 2))
        _set_linear_weights(layer, q_linear_weight=ROT90_LINEAR_WEIGHT)
        rot_prob, _, _ = _run(layer, PIN_HIDDEN)
        rot_mirrored = _mirrored_boundary_prob(
            PIN_HIDDEN, q_linear_weight=ROT90_LINEAR_WEIGHT
        )
        assert np.max(np.abs(rot_prob[..., 1] - rot_mirrored)) > 0.5


# =====================================================================
# Guard (iv): the threshold is STRICTLY above one half
# =====================================================================


class TestGuardFourThresholdIsStrictlyAboveHalf:
    """``argmax([1-p, p])`` breaks ties LOW, so ``p = 0.5`` is NOT a boundary.

    The three arms are constructed so ``p`` lands on an exactly representable
    float32 value, with ``D = 2`` and ``h_0 = (1, 0)``:

    - ``h_1 = (0, 1)``            -> ``cos = 0``       -> ``p = 0.5`` exactly
    - ``h_1 = (-2^-23, 1)``       -> ``cos = -2^-23``  -> ``p = 0.5 + 2^-24``
    - ``h_1 = (+2^-23, 1)``       -> ``cos = +2^-23``  -> ``p = 0.5 - 2^-24``

    The construction is exact in float32: ``(2^-23)^2 = 2^-46`` vanishes against
    1.0 under float32 addition, so ``||h_1|| == 1.0`` bit-exactly and the
    normalization is the identity on these rows.
    """

    EPS = float(np.float32(2.0 ** -23))

    @pytest.mark.parametrize("offset,expected_p,expected_boundary", [
        (-EPS, 0.5 + 2.0 ** -24, True),
        (0.0, 0.5, False),
        (+EPS, 0.5 - 2.0 ** -24, False),
    ])
    def test_the_tie_is_broken_low(self, offset, expected_p, expected_boundary):
        hidden = np.array(
            [[[1.0, 0.0], [offset, 1.0]]], dtype="float32"
        )
        prob, boundary, selected = _run(RoutingModule(d_model=2), hidden)

        # atol=0: every value here is exactly representable in float32.
        assert float(prob[0, 1, 1]) == pytest.approx(expected_p, abs=0.0), (
            f"p = {float(prob[0, 1, 1]):.10f}, expected {expected_p:.10f}"
        )
        assert bool(boundary[0, 1]) is expected_boundary
        # selected_probs reports the WINNING class.
        assert float(selected[0, 1, 0]) == max(expected_p, 1.0 - expected_p)

    def test_the_three_arms_are_distinguishable(self):
        """DIFFER twin: the three constructions are not the same tensor.

        Without this, three arms that all silently produced ``p = 0.5`` would
        look like a passing threshold guard.
        """
        probs = []
        for offset in (-self.EPS, 0.0, +self.EPS):
            hidden = np.array([[[1.0, 0.0], [offset, 1.0]]], dtype="float32")
            probs.append(float(_run(RoutingModule(d_model=2), hidden)[0][0, 1, 1]))
        assert probs[0] > probs[1] > probs[2], probs
        assert probs[0] != probs[1] and probs[1] != probs[2]

    def test_the_oracle_agrees_that_exactly_one_half_is_not_a_boundary(self):
        """The layer's tie behaviour is the ORACLE's, not an invention.

        ``routing_reference`` transcribes ``np.argmax``, which has the same
        first-maximum rule as ``torch.argmax``; this pins the layer to it rather
        than to this test's own reading of the rule.
        """
        hidden = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype="float32")
        _, boundary, _ = _run(RoutingModule(d_model=2), hidden)
        _, ref_boundary, _ = routing_reference(hidden)
        np.testing.assert_array_equal(boundary, ref_boundary)
        assert not bool(ref_boundary[0, 1])

    def test_the_pin_examples_tie_at_position_three(self):
        """D-011's four-token pin reaches the tie, so it is not a corner case."""
        prob, boundary, _ = _run(RoutingModule(d_model=2), PIN_HIDDEN)
        assert float(prob[0, 3, 1]) == 0.5
        assert not bool(boundary[0, 3])


# =====================================================================
# Gradients
# =====================================================================


class TestGradientFlow:
    """Per-weight gradient flow, and the paths that deliberately carry none."""

    def test_every_trainable_weight_receives_a_nonzero_gradient(self):
        rng = np.random.default_rng(501)
        hidden = tf.constant(rng.standard_normal((4, 12, 8)).astype("float32"))
        layer = RoutingModule(d_model=8)
        layer.build((None, 12, 8))
        # Move off the identity so the projections are not at a symmetric point.
        _set_linear_weights(
            layer,
            q_linear_weight=(np.eye(8) + 0.1 * rng.standard_normal((8, 8))).astype("float32"),
            k_linear_weight=(np.eye(8) + 0.1 * rng.standard_normal((8, 8))).astype("float32"),
        )

        with tf.GradientTape() as tape:
            _, _, selected = layer(hidden)
            loss = tf.reduce_sum(selected)
        grads = tape.gradient(loss, layer.trainable_variables)

        assert len(layer.trainable_variables) == 2
        for variable, grad in zip(layer.trainable_variables, grads):
            assert grad is not None, f"no gradient for {variable.path}"
            magnitude = float(tf.reduce_max(tf.abs(grad)))
            assert magnitude > 0.0, f"zero gradient for {variable.path}"

    def test_the_hard_boundary_mask_carries_no_gradient(self):
        """DIFFER twin: not every output path is differentiable.

        The boolean decision is a hard threshold; the model recovers a gradient
        through ``selected_probs`` and the ratio loss, never through the mask.
        A gradient appearing here would mean the threshold had been softened.
        """
        rng = np.random.default_rng(502)
        hidden = tf.constant(rng.standard_normal((4, 12, 8)).astype("float32"))
        layer = RoutingModule(d_model=8)
        layer.build((None, 12, 8))

        with tf.GradientTape() as tape:
            _, boundary, _ = layer(hidden)
            loss = tf.reduce_sum(tf.cast(boundary, tf.float32))
        grads = tape.gradient(loss, layer.trainable_variables)
        assert all(g is None or float(tf.reduce_max(tf.abs(g))) == 0.0 for g in grads)

    def test_gradient_reaches_the_layer_through_its_input(self):
        rng = np.random.default_rng(503)
        hidden = tf.Variable(rng.standard_normal((2, 8, 4)).astype("float32"))
        layer = RoutingModule(d_model=4)
        with tf.GradientTape() as tape:
            _, _, selected = layer(hidden)
            loss = tf.reduce_sum(selected)
        grad = tape.gradient(loss, hidden)
        assert grad is not None and float(tf.reduce_max(tf.abs(grad))) > 0.0


# =====================================================================
# Serialization and the StatelessScope trap
# =====================================================================


class _Parent(keras.layers.Layer):
    """Minimal parent that builds ``RoutingModule`` from its own ``call()``.

    Interface contract: takes ``(B, L, D)``, returns the routing probability
    ``(B, L, 2)``. It exists only so the child layer is first reached through a
    parent -- the one path on which a ``build()``-time ``.assign()`` is silently
    discarded (guide s3.3). A unit test that calls ``.build(...)`` directly is
    structurally blind to that defect.
    """

    def __init__(self, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.router = RoutingModule(d_model=d_model, name="router")

    def call(self, inputs, training=None):
        prob, _, _ = self.router(inputs, training=training)
        return prob

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model})
        return config


class TestSerializationAndBuildPath:
    """`.keras` round trip on VALUES, and the parent-built weight check."""

    def test_identity_weights_survive_being_built_through_a_parent(self):
        """The ``StatelessScope`` regression test (guide s13.2.1).

        If the identity were written by a post-build ``.assign()`` instead of by
        an initializer, this is the one path on which it would come back as
        zeros -- and the direct-``build`` test above would still pass.
        """
        parent = _Parent(d_model=4)
        rng = np.random.default_rng(601)
        parent(keras.ops.convert_to_tensor(
            rng.standard_normal((2, 6, 4)).astype("float32")
        ))
        for kernel in (parent.router.q_proj.kernel, parent.router.k_proj.kernel):
            value = keras.ops.convert_to_numpy(kernel)
            np.testing.assert_array_equal(value, np.eye(4, dtype="float32"))
            # DIFFER twin: this assertion is not satisfied by an all-zero table,
            # which is exactly what the discarded-assign defect produces.
            assert float(np.max(np.abs(value))) > 0.0

    def test_keras_save_load_round_trip_on_values(self, tmp_path):
        seq_len, d_model = 11, 8
        hidden_input = keras.Input(shape=(seq_len, d_model), dtype="float32")
        mask_input = keras.Input(shape=(seq_len,), dtype="int32")
        prob, boundary, selected = RoutingModule(d_model=d_model, name="router")(
            hidden_input, mask=mask_input
        )
        model = keras.Model(
            inputs=[hidden_input, mask_input],
            outputs=[prob, keras.ops.cast(boundary, "float32"), selected],
        )

        rng = np.random.default_rng(602)
        hidden = rng.standard_normal((3, seq_len, d_model)).astype("float32")
        mask = _padding_mask(3, seq_len, [11, 7, 2]).astype("int32")

        # Move off the identity so the round trip carries real weight VALUES and
        # not a table any fresh construction would reproduce for free.
        router = model.get_layer("router")
        _set_linear_weights(
            router,
            q_linear_weight=(np.eye(d_model) + 0.2 * rng.standard_normal((d_model, d_model))).astype("float32"),
            k_linear_weight=(np.eye(d_model) + 0.2 * rng.standard_normal((d_model, d_model))).astype("float32"),
        )

        before = [
            keras.ops.convert_to_numpy(t)
            for t in model([hidden, mask], training=False)
        ]

        path = tmp_path / "routing_module.keras"
        model.save(path)
        loaded = keras.models.load_model(path)
        after = [
            keras.ops.convert_to_numpy(t)
            for t in loaded([hidden, mask], training=False)
        ]

        for original, restored in zip(before, after):
            np.testing.assert_array_equal(original, restored)

        loaded_router = loaded.get_layer("router")
        assert loaded_router.d_model == d_model
        for original_w, restored_w in zip(router.weights, loaded_router.weights):
            np.testing.assert_array_equal(
                keras.ops.convert_to_numpy(original_w),
                keras.ops.convert_to_numpy(restored_w),
            )

        # DIFFER twin: the loaded model is a live model, not a replayed cache.
        other = rng.standard_normal((3, seq_len, d_model)).astype("float32")
        other_out = keras.ops.convert_to_numpy(loaded([other, mask], training=False)[0])
        assert float(np.max(np.abs(other_out - after[0]))) > 1e-2

    def test_a_freshly_constructed_layer_would_NOT_reproduce_the_saved_values(
        self, tmp_path
    ):
        """Anti-vacuity for the round trip: the weights are what was restored.

        A round trip over a layer sitting at its identity init proves nothing --
        a fresh construction reproduces it. This asserts the saved model's
        numbers differ from a default-initialized layer's on the same input.
        """
        seq_len, d_model = 9, 4
        rng = np.random.default_rng(603)
        hidden = rng.standard_normal((2, seq_len, d_model)).astype("float32")

        trained = RoutingModule(d_model=d_model)
        trained.build((None, seq_len, d_model))
        _set_linear_weights(
            trained,
            q_linear_weight=(np.eye(d_model) + 0.5 * rng.standard_normal((d_model, d_model))).astype("float32"),
        )
        fresh = RoutingModule(d_model=d_model)
        assert float(
            np.max(np.abs(_run(trained, hidden)[0] - _run(fresh, hidden)[0]))
        ) > 1e-2
