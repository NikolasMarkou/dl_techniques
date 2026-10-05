"""Test suite for `TopographicProduct` (Keller & Welling 2022, Eq. 6-9).

The layer is the model's mathematical core, and its whole value is a *closed form*:
`t = sqrt(2)(z - mu) / sqrt(nu * energy + eps)` with `energy = sum_delta W R_delta u_{l+delta}^2`.
So the suite is built around comparing the layer against a NumPy transcription of
that formula, mode by mode, plus the two properties that are easy to get wrong and
invisible in a shape test:

- **The window stack must survive the `StatelessScope` build pass.** `G` is a
  constant table; a `.assign()` in `build()` is discarded when the layer is first
  reached from a parent's `call()`, leaving all zeros in every real model while a
  direct `layer.build(...)` test still passes. `test_the_window_stack_survives_the
  _stateless_build_pass` is the only probe that sees it, and
  `test_the_stateless_probe_can_fail` proves it by inlining the defect.
- **Capsules must be independent.** `W` is block-diagonal, so perturbing `u` in one
  capsule must leave every other capsule's energy bit-identical. A block-ONES `W`
  would correlate everything and still produce plausible-looking activations.

The NumPy oracle in `windowed_stack_energy` is written as an explicit Python loop
over the window offsets on purpose: a matching mistake in the layer's vectorized
gather could not cancel out against the same mistake transcribed twice.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from tests.numerics import matmul_precision_atol, reassociation_atol

from dl_techniques.layers.generative.topographic_product import (
    ENERGY_EPSILON,
    TEMPORAL_COHERENCE_TYPES,
    TOPOGRAPHY_TYPES,
    TopographicProduct,
    build_roll_matrix_1d,
    build_window_matrix_1d,
    build_window_matrix_2d,
    build_window_stack,
    windowed_stack_energy,
)

B, S, C, D = 2, 6, 3, 4
LATENT = C * D


def _inputs(seed=0, sequence=S, latent=LATENT):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(B, sequence, latent)).astype("float32")
    u = rng.normal(size=(B, sequence, latent)).astype("float32")
    return ops.convert_to_tensor(z), ops.convert_to_tensor(u)


def _float32_bound(expected, contraction=D, steps=None):
    """Absolute tolerance for a float32 layer against its float64 oracle.

    Both terms are DERIVED from the shape of the computation rather than pasted,
    and the maximum is taken so neither regime loosens the other:

    - :func:`reassociation_atol` bounds the difference from REORDERING true
      float32 arithmetic across the window contraction.
    - :func:`matmul_precision_atol` bounds the difference from doing that
      arithmetic in a NARROWER format, which is what a tensor-core GPU does with
      TF32 (10-bit mantissa, ~4100x float32's roundoff). It is INERT on CPU and
      DOMINANT on GPU, and it is measured rather than assumed.

    The reason this matters here specifically: ``t`` is O(1e+02) at these inputs,
    because Eq. 6 divides by ``sqrt(nu * energy)``. One float32 ulp at that
    magnitude is already 1.0e-05, so a flat ``atol=1e-4`` sits within a few ulps
    of the arithmetic's own resolution -- a bound that measures nothing on a CPU
    and fails on a GPU. MEASURED before this: 2.1e-02 against that 1.0e-04.

    :param expected: The float64 oracle output, whose magnitude sets the scale.
    :param contraction: The reduction length on the compared path.
    :param steps: Overrides the number of window slices applied.
    :return: An absolute tolerance.
    :rtype: float
    """
    scale = float(np.abs(expected).max()) if np.size(expected) else 1.0
    return max(
        reassociation_atol([contraction], num_steps=steps or 1, scale=scale),
        matmul_precision_atol(scale),
    )


def _oracle(layer, z, u):
    """NumPy transcription of the layer's own formula, from its config."""
    window = layer._build_window_matrix()
    stack = build_window_stack(
        window, layer.latent_dim, layer.coherence_window, layer.temporal_coherence
    )
    energy = windowed_stack_energy(stack, np.asarray(u, dtype=np.float64) ** 2)
    numerator = np.sqrt(2.0) * (
        np.asarray(z, dtype=np.float64) - layer.prior_mean
    )
    return numerator / np.sqrt(layer.degrees_of_freedom * energy + layer.epsilon)


def _relative(layer):
    return sorted(w.path.split("/", 1)[-1] for w in layer.weights)


class _Parent(keras.layers.Layer):
    """Minimal parent whose ``call()`` is the only path that builds the child."""

    def __init__(self, child, **kwargs):
        super().__init__(**kwargs)
        self.child = child

    def call(self, inputs, training=None):
        return self.child(inputs, training=training)


class TestTopographicProductConstructor:
    """Stored configuration and validation."""

    def test_stores_its_configuration(self):
        layer = TopographicProduct(
            num_capsules=5,
            capsule_dim=7,
            coherence_window=2,
            neighborhood_size=3,
            temporal_coherence="stationary",
            degrees_of_freedom=4.0,
            prior_mean=11.0,
        )
        assert layer.num_capsules == 5
        assert layer.capsule_dim == 7
        assert layer.latent_dim == 35
        assert layer.coherence_window == 2
        assert layer.neighborhood_size == 3
        assert layer.temporal_coherence == "stationary"
        assert layer.degrees_of_freedom == 4.0
        assert layer.prior_mean == 11.0
        assert not layer.built

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"num_capsules": 0}, "num_capsules must be positive"),
            ({"capsule_dim": 0}, "capsule_dim must be positive"),
            ({"coherence_window": -1}, "coherence_window must be non-negative"),
            ({"neighborhood_size": 0}, r"neighborhood_size must be in"),
            ({"neighborhood_size": 99}, r"neighborhood_size must be in"),
            ({"temporal_coherence": "sideways"}, "temporal_coherence must be one of"),
            ({"topography": "spiral"}, "topography must be one of"),
            ({"degrees_of_freedom": 0.0}, "degrees_of_freedom must be positive"),
            ({"epsilon": 0.0}, "epsilon must be positive"),
        ],
    )
    def test_rejects_invalid_arguments(self, kwargs, match):
        base = {"num_capsules": 3, "capsule_dim": 4}
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            TopographicProduct(**base)

    def test_a_2d_torus_refuses_to_be_shifted(self):
        """A 2-D lattice has no single cyclic axis; that is a construction error.

        The alternative -- silently rolling along the row axis -- would make the
        model's capability depend on an arbitrary axis ordering.
        """
        with pytest.raises(ValueError, match="torus_2d.*supports only"):
            TopographicProduct(
                num_capsules=1,
                capsule_dim=16,
                topography="torus_2d",
                grid_shape=(4, 4),
                temporal_coherence="shifting",
            )

    def test_torus_2d_defaults_to_no_temporal_coherence(self):
        """Leaving temporal_coherence unspecified on a torus resolves to "none".

        The signature default is "shifting", which the torus cannot honour, so
        resolving the sentinel here is what lets `topography` alone decide.
        """
        layer = TopographicProduct(
            num_capsules=1,
            capsule_dim=16,
            topography="torus_2d",
            grid_shape=(4, 4),
        )
        assert layer.temporal_coherence == "none"

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"grid_shape": (4, 4)}, "grid_shape is only meaningful"),
            ({"kernel_shape": (3, 3)}, "kernel_shape is only meaningful"),
        ],
    )
    def test_capsule_1d_rejects_lattice_arguments(self, kwargs, match):
        base = {"num_capsules": 2, "capsule_dim": 4}
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            TopographicProduct(**base)

    def test_torus_2d_requires_a_matching_grid(self):
        with pytest.raises(ValueError, match="requires grid_shape"):
            TopographicProduct(
                num_capsules=1, capsule_dim=16, topography="torus_2d"
            )
        with pytest.raises(ValueError, match="does not equal"):
            TopographicProduct(
                num_capsules=1,
                capsule_dim=16,
                topography="torus_2d",
                grid_shape=(3, 3),
            )

    def test_torus_2d_rejects_a_kernel_that_contradicts_the_neighborhood(self):
        """A 2-D KxK neighbourhood has area K^2; a mismatch is an error, not a
        silently-rescaled window."""
        with pytest.raises(ValueError, match="does not match"):
            TopographicProduct(
                num_capsules=1,
                capsule_dim=16,
                neighborhood_size=3,
                topography="torus_2d",
                grid_shape=(4, 4),
                kernel_shape=(2, 2),
            )

    def test_registers_under_a_package_qualified_key(self):
        import keras

        key = keras.saving.get_registered_name(TopographicProduct)
        assert (
            key
            == "dl_techniques.layers.generative.topographic_product"
            ">TopographicProduct"
        ), key


class TestWindowMatrixClosedForms:
    """The three pure helpers, pinned at closed forms."""

    def test_circulant_window_sums_the_next_k_slots_with_wraparound(self):
        window = build_window_matrix_1d(5, 3)
        np.testing.assert_array_equal(
            window,
            np.array(
                [
                    [1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1],
                    [1, 0, 0, 1, 1],
                    [1, 1, 0, 0, 1],
                ],
                dtype=np.float64,
            ),
        )

    def test_neighborhood_size_one_is_the_identity(self):
        np.testing.assert_array_equal(
            build_window_matrix_1d(6, 1), np.eye(6, dtype=np.float64)
        )

    def test_neighborhood_size_equal_to_the_width_is_all_ones(self):
        """The paper's degenerate ISA case: everything shares with everything."""
        window = build_window_matrix_1d(4, 4)
        np.testing.assert_array_equal(window, np.ones((4, 4), dtype=np.float64))

    def test_roll_matrix_matches_the_layer_convention(self):
        # out[i] = v[(i - 1 + shift) mod D], so shift=1 sends source D-1 to slot 0.
        roll = build_roll_matrix_1d(4, 1)
        values = np.arange(4.0)
        np.testing.assert_array_equal(roll @ values, [3.0, 0.0, 1.0, 2.0])

    def test_roll_matrix_is_a_permutation(self):
        roll = build_roll_matrix_1d(5, 3)
        np.testing.assert_array_equal(np.sort(roll.sum(axis=0)), np.ones(5))
        np.testing.assert_array_equal(np.sort(roll.sum(axis=1)), np.ones(5))

    @pytest.mark.parametrize("shift", [0, 1, 2, 5, -3])
    def test_the_roll_matrix_is_the_same_operator_as_capsule_roll(self, shift):
        """The layer and the traversal operator must agree, or the model learns one
        transformation and is *measured* on another.

        MEASURED on the reference instance (C=2, D=4): ``R_delta @ v`` equals
        ``CapsuleRoll(v, shift=delta)`` exactly for every delta, including the
        negative ones. A sign slip in `build_roll_matrix_1d` makes the temporal
        coherence roll one way while the inference-time traversal rolls the other,
        and NOTHING in the forward pass notices.
        """
        from dl_techniques.layers.capsules import CapsuleRoll

        width = 4
        # float32 for BOTH sides: the matrix product runs in float64 on a float64
        # input and the layer runs in float32, so comparing them at atol=0.0
        # would measure the dtype, not the permutation. One capsule of width D,
        # which is exactly the block `build_window_stack` embeds.
        values = np.random.default_rng(shift + 5).normal(size=(width,)).astype(
            "float32"
        )
        matrix = build_roll_matrix_1d(width, shift)
        from_matrix = (matrix @ values.astype(np.float64)).astype("float32")

        roll = CapsuleRoll(num_capsules=1, capsule_dim=width)
        from_layer = ops.convert_to_numpy(
            roll(ops.convert_to_tensor(values[None, :]), shift=shift)
        )[0]
        np.testing.assert_allclose(from_matrix, from_layer, atol=0.0, rtol=0)

    def test_torus_window_wraps_in_both_axes(self):
        window = build_window_matrix_2d(3, 3, 3, 3)
        np.testing.assert_array_equal(window, np.ones((9, 9), dtype=np.float64))

    def test_torus_window_of_the_full_kernel_shares_everything(self):
        """A 3x3 neighbourhood on a 3x3 lattice: every cell sees all nine.

        Only true because the axes wrap; a 'valid' padding would zero the
        border and make the corners structurally different from the centre.
        """
        window = build_window_matrix_2d(3, 3, 2, 2)
        # Interior cells see 4, edge/corner cells would see fewer WITHOUT wrap.
        np.testing.assert_array_equal(window.sum(axis=1), np.full(9, 4.0))

    def test_window_stack_slices_are_w_times_roll(self):
        window = build_window_matrix_1d(4, 2)
        stack = build_window_stack(window, 4, 2, "shifting")
        assert stack.shape == (5, 4, 4)
        for index, delta in enumerate(range(-2, 3)):
            expected = window @ build_roll_matrix_1d(4, delta)
            np.testing.assert_allclose(stack[index], expected, atol=0.0, rtol=0)

    def test_stationary_stack_repeats_w_at_every_offset(self):
        window = build_window_matrix_1d(4, 2)
        stack = build_window_stack(window, 4, 2, "stationary")
        for slice_ in stack:
            np.testing.assert_allclose(slice_, window, atol=0.0, rtol=0)

    def test_shifting_and_stationary_genuinely_differ(self):
        """The anti-vacuity arm: if these agreed, the flag would be inert.

        MEASURED sum|shifting - stationary| = 288.0 for D=4, K=2, L=2 -- three
        orders above the float32 noise floor.
        """
        window = np.kron(np.ones((3, 3)), build_window_matrix_1d(4, 2))
        shifting = build_window_stack(window, LATENT, 2, "shifting")
        stationary = build_window_stack(window, LATENT, 2, "stationary")
        difference = float(np.abs(shifting - stationary).sum())
        assert difference > 1.0, f"the two coherence modes agree to {difference}"


class TestTopographicProductForward:
    """The layer equals its closed form, in every mode."""

    @pytest.mark.parametrize("temporal_coherence", TEMPORAL_COHERENCE_TYPES)
    def test_matches_the_num_oracle(self, temporal_coherence):
        window = 0 if temporal_coherence == "none" else 2
        layer = TopographicProduct(
            num_capsules=C,
            capsule_dim=D,
            coherence_window=window,
            neighborhood_size=2,
            temporal_coherence=temporal_coherence,
        )
        z, u = _inputs(seed=1)
        got = ops.convert_to_numpy(layer([z, u], training=False))
        expected = _oracle(layer, z, u)
        # TOLERANCE DERIVED, not pasted. `t` is O(1e+02) here -- Eq. 6 divides by
        # sqrt(nu * energy), and the energy of unit-variance `u` under K=2 is
        # order 1 per coordinate but the oracle's float64 and the layer's float32
        # disagree about the last bits of it. A flat `atol=1e-4` is BELOW the
        # float32 resolution at that magnitude (1.19e-07 * 86 = 1.0e-05 per ulp,
        # and the sum runs over 2L+1 slices), so on a tensor-core GPU the
        # comparison failed by 2.1e-02 against a 1.0e-04 bound while being a
        # pure reassociation. `matmul_precision_atol` is the shared helper for
        # exactly this: it measures the ACTIVE device's matmul roundoff rather
        # than assuming one, so the same assertion holds on CPU and on GPU.
        atol = _float32_bound(expected, steps=2 * window + 1)
        np.testing.assert_allclose(got, expected, atol=atol, rtol=0)
        assert np.all(np.isfinite(got))

    @pytest.mark.parametrize("degrees_of_freedom", [0.5, 1.0, 8.0])
    def test_degrees_of_freedom_is_the_denominator(self, degrees_of_freedom):
        """`nu` enters only under the square root, so doubling it divides t by sqrt(2).

        Derived exactly from Eq. 6, not measured: the numerator is untouched, so
        `t(nu) / t(1) == 1 / sqrt(nu)` for every element.
        """
        z, u = _inputs(seed=2)
        base = TopographicProduct(
            num_capsules=C, capsule_dim=D, coherence_window=1,
            neighborhood_size=2, degrees_of_freedom=1.0,
        )
        scaled = TopographicProduct(
            num_capsules=C, capsule_dim=D, coherence_window=1,
            neighborhood_size=2, degrees_of_freedom=degrees_of_freedom,
        )
        unit = ops.convert_to_numpy(base([z, u]))
        other = ops.convert_to_numpy(scaled([z, u]))
        expected = unit / np.sqrt(degrees_of_freedom)
        np.testing.assert_allclose(
            other, expected, rtol=0, atol=_float32_bound(expected)
        )

    def test_neighborhood_size_one_reduces_to_the_elementwise_form(self):
        """K=1 means each variable sees only itself, so `W` is the identity."""
        z, u = _inputs(seed=3)
        layer = TopographicProduct(
            num_capsules=C, capsule_dim=D, coherence_window=0, neighborhood_size=1
        )
        got = ops.convert_to_numpy(layer([z, u]))
        expected = np.sqrt(2.0) * (np.asarray(z, dtype=np.float64) - 30.0) / np.sqrt(
            np.asarray(u, dtype=np.float64) ** 2 + layer.epsilon
        )
        np.testing.assert_allclose(
            got, expected, rtol=0, atol=_float32_bound(expected)
        )

    def test_capsules_are_statistically_independent(self):
        """The other half of the pair: perturbing capsule 0 must not move capsule 1.

        `W` is block-diagonal, so this is EXACT (atol=0.0), not approximate. A
        block-ONES `W` would correlate everything and still produce activations
        that look entirely reasonable.
        """
        z_np = np.random.default_rng(4).normal(size=(B, S, LATENT))
        u_np = np.random.default_rng(5).normal(size=(B, S, LATENT))
        layer = TopographicProduct(
            num_capsules=C, capsule_dim=D, coherence_window=0, neighborhood_size=D
        )
        window = layer._build_window_matrix()
        stack = build_window_stack(window, layer.latent_dim, 0, "none")

        perturbed = u_np.copy()
        perturbed[:, :, 0] += 10.0
        base_energy = windowed_stack_energy(stack, u_np**2)
        moved_energy = windowed_stack_energy(stack, perturbed**2)

        assert np.abs(base_energy[:, :, :D] - moved_energy[:, :, :D]).max() > 1.0, (
            "capsule 0's energy should have moved"
        )
        np.testing.assert_allclose(
            base_energy[:, :, D:], moved_energy[:, :, D:], atol=0.0, rtol=0
        )

    def test_the_window_edges_are_edge_replicated_not_zero_padded(self):
        """The first timestep's energy must use real `u` values, not zeros.

        Zero padding would make the boundary energy strictly smaller than the
        interior's for the same `u`, which is precisely the artificial absence of
        variance the topographic prior is about.
        """
        layer = TopographicProduct(
            num_capsules=C, capsule_dim=D, coherence_window=2, neighborhood_size=1
        )
        z, u = _inputs(seed=6)
        base = ops.convert_to_numpy(layer([z, u]))

        # Changing ONLY the last timestep's u must leave the FIRST timestep's
        # output untouched (its window is [-2, 0, +2] -> clamped to [0, 0, 1]).
        bumped = np.array(u.numpy())
        bumped[:, -1] += 5.0
        moved = ops.convert_to_numpy(
            layer([z, ops.convert_to_tensor(bumped.astype("float32"))])
        )
        np.testing.assert_allclose(
            moved[:, 0], base[:, 0], atol=0.0, rtol=0,
            err_msg="timestep 0 must only see timesteps 0 and 1 (edge replication)",
        )
        assert np.abs(moved[:, -1] - base[:, -1]).max() > 1e-3, (
            "the last timestep must see the changed one"
        )

    def test_an_all_zero_energy_stays_finite(self):
        """The degenerate case: `u == 0` everywhere makes `energy` zero.

        Without the epsilon floor the 1/sqrt(.) is a division by zero and every
        output is inf or NaN, on a correctly-shaped tensor.
        """
        z, _ = _inputs(seed=7)
        zeros = ops.convert_to_tensor(np.zeros((B, S, LATENT), "float32"))
        layer = TopographicProduct(
            num_capsules=C, capsule_dim=D, coherence_window=1, neighborhood_size=2
        )
        out = ops.convert_to_numpy(layer([z, zeros]))
        assert np.all(np.isfinite(out)), "an all-zero energy must not produce NaN/inf"

    def test_a_NEGATIVE_energy_stays_finite(self):
        """The case the epsilon floor does NOT cover, and the reason for the clamp.

        `energy` is non-negative in exact arithmetic -- `G`'s entries are 0 or 1
        -- but it is computed as a signed sum of products, so a `G` carrying
        negative entries produces a negative energy. Adding `epsilon` to a
        negative number leaves it negative, so the floor alone takes the `sqrt`
        to NaN. MEASURED before the clamp: every `t` was NaN.

        The layer does not own `G` -- it is a non-trainable constant this layer
        builds -- so the defect is only reachable by handing it one, which is
        exactly what the round-trip oracle's weight perturbation does.
        """
        z, u = _inputs(seed=7)
        layer = TopographicProduct(
            num_capsules=C, capsule_dim=D, coherence_window=1, neighborhood_size=2
        )
        layer([z, u])  # build, so the stack exists
        stack = ops.convert_to_numpy(layer.window_stack)
        assert np.abs(stack).min() >= 0.0, "the built stack is already signed"

        layer.window_stack.assign(
            ops.convert_to_tensor(stack - 2.0, "float32")
        )
        out = ops.convert_to_numpy(layer([z, u]))
        assert np.all(np.isfinite(out)), (
            "a negative energy reached the sqrt; the clamp at zero is gone"
        )

    def test_the_clamp_does_not_rescale_a_healthy_energy(self):
        """The clamp is at ZERO, not at a positive floor.

        A clamp at `epsilon` would leave the healthy path unchanged too -- which
        is why the previous arm alone cannot tell the two apart, and why this one
        compares against the un-clamped reference on unmodified weights.
        """
        z, u = _inputs(seed=7)
        layer = TopographicProduct(
            num_capsules=C, capsule_dim=D, coherence_window=1, neighborhood_size=2
        )
        got = ops.convert_to_numpy(layer([z, u]))

        squared = np.square(
            ops.convert_to_numpy(
                ops.pad(
                    ops.convert_to_tensor(u, "float32"),
                    [[0, 0], [1, 0], [0, 0]],
                    constant_values=0.0,
                )
            )[:, :S]
        )
        energies = [
            float(np.einsum("de,bse->bsd", ops.convert_to_numpy(layer.window_stack)[i],
                            squared[:, max(0, i - 1): i - 1 + S]).sum(axis=1).min())
            for i in range(3)
        ]
        assert min(energies) > 0.0, f"a healthy energy is already clamped: {energies}"
        # So the clamp is a no-op here, and the two paths are bit-identical.
        reference = TopographicProduct(
            num_capsules=C, capsule_dim=D, coherence_window=1, neighborhood_size=2
        )
        np.testing.assert_array_equal(
            got, ops.convert_to_numpy(reference([z, u]))
        )

    @pytest.mark.parametrize("sequence", [1, 2, 9])
    def test_accepts_any_sequence_length(self, sequence):
        """`S` need not equal `2L+1`; the window clamps at the edges."""
        z, u = _inputs(seed=8, sequence=sequence)
        layer = TopographicProduct(
            num_capsules=C, capsule_dim=D, coherence_window=3, neighborhood_size=2
        )
        out = ops.convert_to_numpy(layer([z, u]))
        assert out.shape == (B, sequence, LATENT)
        assert np.all(np.isfinite(out))

    def test_the_torus_layout_works(self):
        layer = TopographicProduct(
            num_capsules=1, capsule_dim=16, neighborhood_size=3,
            topography="torus_2d", grid_shape=(4, 4),
        )
        rng = np.random.default_rng(9)
        z = ops.convert_to_tensor(rng.normal(size=(B, S, 16)).astype("float32"))
        u = ops.convert_to_tensor(rng.normal(size=(B, S, 16)).astype("float32"))
        got = ops.convert_to_numpy(layer([z, u]))
        expected = _oracle(layer, z, u)
        np.testing.assert_allclose(
            got, expected, rtol=0, atol=_float32_bound(expected)
        )

    def test_rejects_a_wrong_rank_input(self):
        import keras

        layer = TopographicProduct(num_capsules=C, capsule_dim=D)
        with pytest.raises(ValueError, match="must be rank 3"):
            layer.build(((None, LATENT), (None, LATENT)))

    def test_rejects_mismatched_latent_widths(self):
        import keras

        layer = TopographicProduct(num_capsules=C, capsule_dim=D)
        with pytest.raises(ValueError, match="must share latent_dim"):
            layer.build(((None, S, LATENT), (None, S, LATENT + 1)))

    def test_rejects_a_latent_width_that_contradicts_the_capsules(self):
        import keras

        layer = TopographicProduct(num_capsules=C, capsule_dim=D)
        with pytest.raises(ValueError, match="expected latent_dim"):
            layer.build(((None, S, LATENT + 1), (None, S, LATENT + 1)))


class TestTopographicProductStatelessBuild:
    """The `StatelessScope` trap, and a proof the probe can fail."""

    def test_the_window_stack_survives_the_stateless_build_pass(self):
        """Build the layer THROUGH a parent's `call()` — the only path that sees it.

        MEASURED: a `G` table assigned in `build()` reads back as **exactly 0.0**
        here, while a direct `layer.build(...)` gives the correct value.
        """
        layer = TopographicProduct(
            num_capsules=C, capsule_dim=D, coherence_window=2, neighborhood_size=2
        )
        parent = _Parent(layer)
        z_input = keras.Input(shape=(S, LATENT))
        u_input = keras.Input(shape=(S, LATENT))
        keras.Model([z_input, u_input], parent([z_input, u_input]))

        stack = ops.convert_to_numpy(layer.window_stack)
        expected = build_window_stack(
            layer._build_window_matrix(), layer.latent_dim, 2, "shifting"
        )
        # atol=0.0: the initializer either ran or it did not.
        np.testing.assert_allclose(stack, expected, atol=0.0, rtol=0)
        assert np.abs(stack).sum() > 0.0, "the window stack is all zeros"

    def test_the_stateless_probe_can_fail(self):
        """RED proof: the SAME probe rejects a constant table that reads back zero.

        Reproduces the failure the `.assign()`-in-`build()` anti-pattern produces,
        in a throwaway subclass. A literal `.assign()` cannot be used for the
        injection: inside the StatelessScope pass it RAISES ("cannot convert a
        symbolic tf.Tensor to a numpy array") rather than being silently dropped,
        so the test would fail for the wrong reason. Initializing the table to
        zeros reproduces the exact observable the probe must catch.

        If this test failed to raise, the probe above would be measuring nothing.
        """
        class ZeroInitialized(TopographicProduct):
            def build(self, input_shape):
                # Same tree, but the table is created with the "zeros"
                # initializer and never filled in -- the state the anti-pattern
                # leaves behind.
                latent_dim = self.latent_dim
                shape = (
                    2 * self.coherence_window + 1,
                    latent_dim,
                    latent_dim,
                )
                self.window_stack = self.add_weight(
                    name="window_stack",
                    shape=shape,
                    initializer="zeros",
                    trainable=False,
                )
                keras.layers.Layer.build(self, input_shape)

        layer = ZeroInitialized(
            num_capsules=C, capsule_dim=D, coherence_window=2, neighborhood_size=2
        )
        parent = _Parent(layer)
        z_input = keras.Input(shape=(S, LATENT))
        u_input = keras.Input(shape=(S, LATENT))
        keras.Model([z_input, u_input], parent([z_input, u_input]))

        stack = ops.convert_to_numpy(layer.window_stack)
        expected = build_window_stack(
            layer._build_window_matrix(), layer.latent_dim, 2, "shifting"
        )
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(stack, expected, atol=0.0, rtol=0)
        assert np.abs(stack).sum() == 0.0, "the injected defect must be visible"

    def test_a_direct_build_also_produces_the_table(self):
        """The twin: the table is right on the direct path too, so the difference
        above is the build ROUTE and not the table."""
        layer = TopographicProduct(
            num_capsules=C, capsule_dim=D, coherence_window=2, neighborhood_size=2
        )
        layer.build(((None, S, LATENT), (None, S, LATENT)))
        stack = ops.convert_to_numpy(layer.window_stack)
        expected = build_window_stack(
            layer._build_window_matrix(), layer.latent_dim, 2, "shifting"
        )
        np.testing.assert_allclose(stack, expected, atol=0.0, rtol=0)

    def test_build_parity_between_the_two_paths(self):
        explicit = TopographicProduct(
            num_capsules=C, capsule_dim=D, coherence_window=2, neighborhood_size=2
        )
        explicit.build(((None, S, LATENT), (None, S, LATENT)))
        lazy = TopographicProduct(
            num_capsules=C, capsule_dim=D, coherence_window=2, neighborhood_size=2
        )
        lazy([_inputs()[0], _inputs()[1]])
        assert _relative(explicit) == _relative(lazy)

    def test_the_window_stack_is_not_trainable(self):
        """A learned `W` would no longer be the a-priori topographic prior."""
        layer = TopographicProduct(num_capsules=C, capsule_dim=D)
        layer([_inputs()[0], _inputs()[1]])
        assert layer.trainable_weights == [], "the neighbourhood must stay fixed"
        assert len(layer.non_trainable_weights) == 1


class TestTopographicProductGraphSafety:
    """`call()` must be symbolic-only and survive tracing and XLA."""

    def test_matches_under_tf_function(self):
        layer = TopographicProduct(
            num_capsules=C, capsule_dim=D, coherence_window=2, neighborhood_size=2
        )
        z, u = _inputs(seed=10)
        eager = ops.convert_to_numpy(layer([z, u], training=False))

        @tf.function
        def traced(z_arg, u_arg):
            return layer([z_arg, u_arg], training=False)

        np.testing.assert_allclose(
            ops.convert_to_numpy(traced(z, u)), eager, atol=1e-5, rtol=1e-5
        )

    def test_matches_under_xla(self):
        layer = TopographicProduct(
            num_capsules=C, capsule_dim=D, coherence_window=2, neighborhood_size=2
        )
        z, u = _inputs(seed=11)

        @tf.function(jit_compile=True)
        def compiled(z_arg, u_arg):
            return layer([z_arg, u_arg], training=False)

        eager = ops.convert_to_numpy(layer([z, u], training=False))
        np.testing.assert_allclose(
            ops.convert_to_numpy(compiled(z, u)), eager, atol=1e-5, rtol=1e-5
        )

    def test_a_dynamic_sequence_length_traces(self):
        """The window gather must not branch on the sequence length."""
        layer = TopographicProduct(
            num_capsules=C, capsule_dim=D, coherence_window=3, neighborhood_size=2
        )

        @tf.function(input_signature=[
            tf.TensorSpec([None, None, LATENT], tf.float32),
            tf.TensorSpec([None, None, LATENT], tf.float32),
        ])
        def traced(z_arg, u_arg):
            return layer([z_arg, u_arg], training=False)

        for sequence in (2, 7):
            z, u = _inputs(seed=12, sequence=sequence)
            out = ops.convert_to_numpy(traced(z, u))
            assert out.shape == (B, sequence, LATENT)
            assert np.all(np.isfinite(out))

    @pytest.mark.parametrize("policy", ["mixed_float16", "float64"])
    def test_dtype_arms(self, policy):
        """MEASURED float32 reference for the same input, as the fp16 control."""
        import keras

        z_np, u_np = _inputs(seed=13)
        reference = TopographicProduct(
            num_capsules=C, capsule_dim=D, coherence_window=1, neighborhood_size=2
        )
        expected = ops.convert_to_numpy(reference([z_np, u_np]))

        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy(policy)
        try:
            layer = TopographicProduct(
                num_capsules=C, capsule_dim=D, coherence_window=1,
                neighborhood_size=2,
            )
            out = ops.convert_to_numpy(layer([z_np, u_np]))
            assert np.all(np.isfinite(out)), f"{policy} produced NaN/inf"
            np.testing.assert_allclose(
                out.astype("float64"), expected.astype("float64"),
                rtol=2e-2, atol=2e-2,
                err_msg=f"{policy} disagrees with the float32 control",
            )
        finally:
            keras.mixed_precision.set_global_policy(previous)

    def test_the_epsilon_is_far_below_a_healthy_energy(self):
        """The floor must never materially resize a real energy.

        Derived: the smallest energy a K-wide window of standard normals produces
        is ~K (the sum of K squared Gaussians concentrates at K), so a floor of
        1e-12 is at least nine orders below it.
        """
        assert ENERGY_EPSILON < 1e-9
        rng = np.random.default_rng(14)
        smallest = (rng.normal(size=(200000, 3)) ** 2).sum(axis=1).min()
        assert ENERGY_EPSILON < smallest, (
            f"the floor {ENERGY_EPSILON} is not below the smallest observed "
            f"energy {smallest}"
        )


class TestTopographicProductSerialization:
    """Config completeness and a value round trip."""

    def test_config_is_complete(self):
        layer = TopographicProduct(
            num_capsules=3, capsule_dim=5, coherence_window=2,
            neighborhood_size=2, temporal_coherence="stationary",
            degrees_of_freedom=3.0, prior_mean=7.0,
        )
        config = layer.get_config()
        for key in (
            "num_capsules", "capsule_dim", "coherence_window", "neighborhood_size",
            "temporal_coherence", "topography", "grid_shape", "kernel_shape",
            "degrees_of_freedom", "prior_mean", "epsilon",
        ):
            assert key in config, key
        assert config["temporal_coherence"] == "stationary"
        assert config["degrees_of_freedom"] == 3.0

    @pytest.mark.parametrize("temporal_coherence", TEMPORAL_COHERENCE_TYPES)
    def test_from_config_round_trips(self, temporal_coherence):
        window = 0 if temporal_coherence == "none" else 2
        layer = TopographicProduct(
            num_capsules=3, capsule_dim=4, coherence_window=window,
            neighborhood_size=2, temporal_coherence=temporal_coherence,
        )
        restored = TopographicProduct.from_config(layer.get_config())
        for key in ("num_capsules", "capsule_dim", "coherence_window",
                    "neighborhood_size", "temporal_coherence", "degrees_of_freedom"):
            assert getattr(restored, key) == getattr(layer, key), key

    def test_from_config_round_trips_the_lattice_arguments(self):
        layer = TopographicProduct(
            num_capsules=1, capsule_dim=16, neighborhood_size=3,
            topography="torus_2d", grid_shape=(4, 4),
        )
        restored = TopographicProduct.from_config(layer.get_config())
        assert restored.grid_shape == (4, 4)
        assert restored.kernel_shape == (3, 3)
        assert restored.temporal_coherence == "none"

    @pytest.mark.parametrize("temporal_coherence", TEMPORAL_COHERENCE_TYPES)
    def test_value_round_trip(self, temporal_coherence):
        import os
        import tempfile

        import keras

        window = 0 if temporal_coherence == "none" else 2
        layer = TopographicProduct(
            num_capsules=C, capsule_dim=D, coherence_window=window,
            neighborhood_size=2, temporal_coherence=temporal_coherence,
        )
        z_input = keras.Input(shape=(S, LATENT))
        u_input = keras.Input(shape=(S, LATENT))
        model = keras.Model([z_input, u_input], layer([z_input, u_input]))
        z, u = _inputs(seed=15)

        before = ops.convert_to_numpy(model([z, u], training=False))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "topo.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
            after = ops.convert_to_numpy(loaded([z, u], training=False))

        # atol=0.0: the forward pass has no stochastic element, so a reload that
        # restored the table and the config must reproduce it bit-for-bit.
        np.testing.assert_allclose(before, after, atol=0.0, rtol=0)

    def test_compute_output_shape_works_unbuilt(self):
        layer = TopographicProduct(num_capsules=C, capsule_dim=D)
        assert not layer.built
        assert layer.compute_output_shape(
            ((None, S, LATENT), (None, S, LATENT))
        ) == (None, S, LATENT)
        assert not layer.built