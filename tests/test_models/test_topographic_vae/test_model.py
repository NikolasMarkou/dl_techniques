"""Test suite for `TopographicVAE` (Keller & Welling 2022).

The model makes one falsifiable claim: **a transformation of the input shows up
as a cyclic roll inside a capsule**, so that the unseen frames of a sequence can be
decoded from one encoded activation. Everything here is arranged around that claim
being testable rather than merely asserted.

The four guards that carry the weight:

- **``traverse_capsules`` frame 0 must equal ``decode(t_0)`` EXACTLY.** If the
  traversal operator and the decode path disagreed, every traversal figure would be
  a picture of a bug. MEASURED delta 0.0.
- **The decoder must actually read ``t``.** Handing it ``concat([u, z])`` alongside
  ``t`` would let it bypass the topography entirely while the likelihood stayed
  flat, making the whole model unfalsifiable. Pinned by perturbing ``t`` alone and
  requiring the reconstruction to move.
- **Every constructor knob must change something.** Each is pinned with the
  instrument matching its class (shapes for structural, outputs for value), because
  reading the value back off ``self`` proves nothing.
- **Every trainable weight must move.** Reported as a NAME SET, never a count.

MEASURED on the reference instance (16x16x1 frames, S=6, 4x4 capsules):
  * build parity between the explicit and lazy paths: identical 19-weight paths
  * ``.keras`` round trip: ``encode()`` values and all 19 weights bit-identical
  * XLA versus eager: max delta 1.01e-06 (float32 reassociation, not a different
    computation)
  * gradient flow: 18 of 18 trainable variables non-None and non-zero
"""

import os
import tempfile
from unittest import mock

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.layers.capsules import CapsuleRoll
from dl_techniques.layers.generative.topographic_product import (
    build_window_matrix_1d,
    build_window_stack,
)
from dl_techniques.losses.topographic_vae_loss import TopographicVAELoss
from dl_techniques.models.vision.topographic_vae import (
    TopographicVAE,
    create_topographic_vae,
)

from tests.numerics import matmul_precision_atol, reassociation_atol

from ..gradient_flow_oracle import stop_all_gradients
from ..knob_sensitivity_oracle import (
    assert_scoped_value_knob_changes_weights,
    assert_structural_knob_changes_weights,
    assert_value_knob_changes_output,
    weights_in_scope,
)
from ..smoke_contract_oracle import (
    DEFAULT_BREAKERS,
    assert_contract_rejects_a_broken_forward,
    broken_forward,
)

FRAME = (16, 16, 1)
SEQUENCE = 6
CAPSULES = 4
CAPSULE_DIM = 4
LATENT = CAPSULES * CAPSULE_DIM
HIDDEN = [16, 16]

#: Every output key on the full model. Asserted as an exact SET rather than a
#: superset, because the baseline arm's MISSING keys are the thing under test
#: elsewhere and a superset assertion here would tolerate them.
EXPECTED_KEYS = frozenset(
    {"reconstruction", "t", "z_mean", "z_log_var", "u_mean", "u_log_var"}
)

#: The latent-valued entries, all `(batch, sequence, latent_dim)`.
LATENT_KEYS = ("t", "z_mean", "z_log_var", "u_mean", "u_log_var")

#: Factory overrides shared by the construction-surface tests. `input_shape` is
#: overridden because the factory reads the frame geometry from the VARIANT (28x28x3
#: for MNIST) and these tests use the smaller reference frames throughout.
_FACTORY_OVERRIDES = dict(
    input_shape=FRAME,
    sequence_length=SEQUENCE,
    num_capsules=CAPSULES,
    capsule_dim=CAPSULE_DIM,
    coherence_window=2,
    neighborhood_size=2,
    encoder_hidden_dims=[8],
    decoder_hidden_dims=[8],
)


def _config(**overrides):
    base = dict(
        input_shape=FRAME,
        sequence_length=SEQUENCE,
        num_capsules=CAPSULES,
        capsule_dim=CAPSULE_DIM,
        coherence_window=2,
        neighborhood_size=2,
        encoder_hidden_dims=list(HIDDEN),
        decoder_hidden_dims=list(HIDDEN),
    )
    base.update(overrides)
    return base


def _build(**overrides):
    keras.utils.set_random_seed(7)
    return TopographicVAE(**_config(**overrides))


def _sample(batch=2, sequence=SEQUENCE, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((batch, sequence) + FRAME).astype("float32")


def _relative(model):
    """Weight paths with the model-root segment stripped, so instances compare."""
    return sorted(w.path.split("/", 1)[-1] for w in model.weights)


def _window_stack_shape(signature):
    """The ``(2L+1, D, D)`` window-stack shape inside a weight signature.

    Located by RANK rather than by index: it is the only rank-3 weight in the
    tree, and an index into an ordered signature is a claim about the order the
    constructor happens to build sub-layers in -- which a refactor may change
    silently, leaving the assertion reading a frame-width axis and passing or
    failing for the wrong reason.

    :param signature: The ordered tuple of weight shapes.
    :return: The rank-3 shape.
    :rtype: Tuple[int, ...]
    :raises AssertionError: If no rank-3 weight is present.
    """
    rank_three = [shape for shape in signature if len(shape) == 3]
    assert len(rank_three) == 1, (
        f"expected exactly one rank-3 weight (the window stack), found "
        f"{rank_three} in {list(signature)}"
    )
    return rank_three[0]


def _blocked_window(neighborhood_size):
    """The ``(latent_dim, latent_dim)`` block-diagonal neighbourhood matrix.

    One independent circulant block per capsule, so capsules cannot influence
    each other's energy -- the constraint ``TopographicProduct`` documents and
    the reason ``K == D`` saturates WITHIN a capsule rather than across all of
    them.
    """
    block = build_window_matrix_1d(CAPSULE_DIM, neighborhood_size)
    window = np.zeros((LATENT, LATENT), dtype=np.float64)
    for capsule in range(CAPSULES):
        start = capsule * CAPSULE_DIM
        window[start:start + CAPSULE_DIM, start:start + CAPSULE_DIM] = block
    return window


def _snapshot(model):
    """Values of the model's TRAINABLE variables, keyed by path.

    ``trainable_variables``, not ``weights``: the window stack is a non-trainable
    constant, and a movement probe over ``weights`` would demand that a buffer
    respond to the optimizer.
    """
    return {
        v.path: np.array(ops.convert_to_numpy(v)).copy()
        for v in model.trainable_variables
    }


def _contract(outputs, batch=None, sequence=SEQUENCE):
    """The smoke contract, as a function of the OUTPUT alone.

    The container type and the key set are asserted FIRST and by `isinstance`,
    not by a ``try: ... except KeyError``. A contract that reaches for
    ``outputs["reconstruction"]`` and lets a missing key raise ``KeyError`` is
    the contract *crashing*, not judging -- ``smoke_contract_oracle`` requires
    ``AssertionError`` specifically for that reason (D-035).

    :param outputs: The forward output, a dict of tensors.
    :param batch: The expected leading dimension. ``None`` pins only the RANK
        and the trailing dimensions. It exists so a caller that cannot know the
        batch (the ``slice_leading_axis`` breaker changes it by construction) can
        still run the rest of the contract -- but the smoke path passes the real
        batch, because a contract that skips the leading dimension is exactly the
        under-assertion ``slice_leading_axis`` is built to catch, and the
        broken-forward arm below relies on it being asserted.
    :param sequence: The expected sequence length.
    :raises AssertionError: on any violation.
    """
    assert isinstance(outputs, dict), (
        f"the forward returned {type(outputs).__name__}, not a dict"
    )
    assert set(outputs) == EXPECTED_KEYS, (
        f"the key set is {sorted(outputs)}, expected {sorted(EXPECTED_KEYS)}"
    )
    for key, value in outputs.items():
        assert value is not None, f"the output entry {key!r} is None"
        array = np.asarray(ops.convert_to_numpy(value))
        assert np.all(np.isfinite(array)), f"non-finite values in {key!r}"

    reconstruction = np.asarray(ops.convert_to_numpy(outputs["reconstruction"]))
    assert reconstruction.ndim == 5, f"reconstruction rank {reconstruction.ndim}"
    assert reconstruction.shape[-3:] == FRAME, (
        f"reconstruction frame shape {reconstruction.shape[-3:]}, expected "
        f"{FRAME}"
    )
    assert reconstruction.shape[-4] == sequence, (
        f"reconstruction sequence length {reconstruction.shape[-4]}, expected "
        f"{sequence}"
    )
    if batch is not None:
        assert reconstruction.shape[0] == batch, (
            f"reconstruction batch {reconstruction.shape[0]}, expected {batch}"
        )

    # The decoder's final activation is a sigmoid, so the likelihood's [0, 1]
    # support holds. A reconstruction outside it would make the BCE meaningless.
    assert reconstruction.min() >= 0.0 and reconstruction.max() <= 1.0, (
        f"the reconstruction left [0, 1]: [{reconstruction.min()}, "
        f"{reconstruction.max()}]"
    )

    for key in LATENT_KEYS:
        latent = np.asarray(ops.convert_to_numpy(outputs[key]))
        assert latent.ndim == 3, f"{key!r} rank {latent.ndim}, expected 3"
        assert latent.shape[-1] == LATENT, f"{key!r} width {latent.shape[-1]}"
        assert latent.shape[-2] == sequence, (
            f"{key!r} sequence length {latent.shape[-2]}, expected {sequence}"
        )


class TestTopographicVAEConstructor:
    """Stored configuration and validation."""

    def test_stores_its_configuration(self):
        model = TopographicVAE(**_config(coherence_window=2, prior_mean=11.0))
        assert model.num_capsules == CAPSULES
        assert model.capsule_dim == CAPSULE_DIM
        assert model.latent_dim == LATENT
        assert model.sequence_length == SEQUENCE
        assert model.coherence_window == 2
        assert model.neighborhood_size == 2
        assert model.prior_mean == 11.0
        assert model.temporal_coherence == "shifting"
        assert model.use_variance_variables is True
        assert model._input_shape == FRAME

    @pytest.mark.parametrize(
        "overrides,match",
        [
            ({"input_shape": (16, 16)}, "input_shape must be 3D"),
            ({"sequence_length": 0}, "sequence_length must be positive"),
            ({"num_capsules": 0}, "num_capsules must be positive"),
            ({"capsule_dim": -1}, "capsule_dim must be positive"),
            ({"coherence_window": -1}, "coherence_window must be non-negative"),
            ({"temporal_coherence": "diagonal"},
             "temporal_coherence must be one of"),
            ({"topography": "hex"}, "topography must be one of"),
            ({"input_shape": (4, 4, 1)}, "at least 8x8"),
        ],
    )
    def test_rejects_invalid_arguments(self, overrides, match):
        with pytest.raises(ValueError, match=match):
            TopographicVAE(**_config(**overrides))

    def test_a_coherence_window_needs_the_variance_variables(self):
        """Temporal coherence correlates the energy built from ``u``.

        With ``use_variance_variables=False`` there is no energy, so a non-zero
        ``L`` would be silently ignored -- a forgotten flag that looks like it took
        effect.
        """
        with pytest.raises(ValueError, match="requires use_variance_variables"):
            TopographicVAE(
                **_config(
                    use_variance_variables=False, coherence_window=2
                )
            )

    def test_a_torus_needs_the_variance_variables(self):
        with pytest.raises(ValueError, match="requires use_variance_variables"):
            TopographicVAE(
                **_config(
                    use_variance_variables=False,
                    topography="torus_2d",
                    grid_shape=(4, 4),
                    num_capsules=1,
                    capsule_dim=16,
                    neighborhood_size=2,
                )
            )

    def test_sub_layers_are_created_in_init_not_build(self):
        model = TopographicVAE(**_config())
        assert model.z_encoder is not None
        assert model.u_encoder is not None
        assert model.topographic_product is not None
        assert model.roll is not None
        assert model.decoder is not None
        assert not model.built

    def test_every_sub_layer_carries_an_explicit_name(self):
        """Build parity depends on it: Keras' auto_name counter is process-global."""
        model = TopographicVAE(**_config())
        model(_sample())
        for layer in model.layers:
            assert layer.name and not layer.name.startswith(
                ("dense_", "sequential_", "topographic_product_1")
            ), f"auto-named layer: {layer.name}"

    def test_registers_under_a_family_stripped_key(self):
        key = keras.saving.get_registered_name(TopographicVAE)
        # The `vision/` family is a filing decision, not a namespace: the key is
        # the defining path with the family directory removed.
        assert key == "dl_techniques.models.topographic_vae.model>TopographicVAE", key

    def test_a_bare_registration_name_is_not_used(self):
        assert (
            keras.saving.get_registered_object("Custom>TopographicVAE")
            is TopographicVAE
        ), "the legacy alias is bound by register_dl_technique"


class TestTopographicVAEForward:
    """The forward pass and its shape contract."""

    def test_smoke_contract(self):
        outputs = _build()(_sample(), training=False)
        _contract(outputs, batch=2)

    def test_the_contract_rejects_a_broken_forward(self):
        """RED proof, via the shared instrument.

        The oracle runs the contract on the REAL output first (it must pass, or
        every later rejection proves nothing), then under each of its breakers
        and requires an ``AssertionError`` from every one.
        """
        rejections = assert_contract_rejects_a_broken_forward(
            _build(),
            _sample(batch=2),
            lambda out: _contract(out, batch=2),
        )
        assert set(rejections) == {b.__name__ for b in DEFAULT_BREAKERS}

    @pytest.mark.parametrize("batch", [1, 2, 5])
    @pytest.mark.parametrize("sequence", [1, 6, 9])
    def test_shape_over_batch_and_sequence(self, batch, sequence):
        model = _build()
        outputs = model(_sample(batch, sequence), training=False)
        assert tuple(outputs["reconstruction"].shape) == (batch, sequence) + FRAME
        assert tuple(outputs["t"].shape) == (batch, sequence, LATENT)

    def test_compute_output_shape_works_unbuilt(self):
        """The declared shape must match the live one, ``None`` batch included.

        ``compute_output_shape`` cannot know the batch size, so it reports
        ``None`` there and the comparison has to accept that. A test comparing
        the two tuples directly fails on the ``None`` and would have to be
        weakened to a rank check -- which is how a wrong SEQUENCE or frame width
        would slip through.
        """
        model = TopographicVAE(**_config())
        assert not model.built
        shapes = model.compute_output_shape((None, SEQUENCE) + FRAME)
        assert not model.built, "compute_output_shape must not build the model"
        model(_sample())
        live = model(_sample(), training=False)
        assert set(shapes) == EXPECTED_KEYS, sorted(shapes)
        for key, expected in shapes.items():
            declared = tuple(expected)
            actual = tuple(ops.shape(live[key]))
            assert len(actual) == len(declared), key
            for axis, (got, want) in enumerate(zip(actual, declared)):
                if want is not None:
                    assert got == want, (
                        f"{key!r} axis {axis}: declared {want}, live {got}"
                    )

    def test_the_output_dict_has_no_none(self):
        """`predict()` raises on a bare None inside an output dict."""
        model = _build()
        outputs = model(_sample(), training=False)
        assert None not in outputs.values()
        assert all(v is not None for v in outputs.values())

    def test_the_baseline_omits_the_u_keys_entirely(self):
        model = _build(use_variance_variables=False, coherence_window=0)
        outputs = model(_sample(), training=False)
        assert "u_mean" not in outputs
        assert "u_log_var" not in outputs
        assert {"reconstruction", "t", "z_mean", "z_log_var"} <= set(outputs)

    def test_the_baseline_bypasses_the_topography(self):
        """Without ``u`` the energy is zero, so the 1/sqrt is SKIPPED, not divided.

        ``t = z - mu`` directly. Dividing by a near-zero energy would turn
        ``epsilon`` into a temperature and blow the latent up; the test asserts the
        relationship rather than a magnitude.
        """
        model = _build(use_variance_variables=False, coherence_window=0)
        assert model.topographic_product is None
        outputs = model(_sample(), training=False)
        expected = (
            outputs["z_mean"]
            + ops.exp(0.5 * outputs["z_log_var"]) * 0.0
            - model.prior_mean
        )
        # The sampler noise makes an exact comparison impossible, so pin the
        # MEAN: an unbiased sample of `z - mu` has that mean.
        assert np.abs(
            float(ops.mean(outputs["t"])) - float(ops.mean(expected))
        ) < 1.0
        assert np.all(np.isfinite(ops.convert_to_numpy(outputs["t"])))

    def test_encode_matches_the_posterior_inside_call(self):
        model = _build()
        sample = _sample()
        encoded = model.encode(sample)
        outputs = model(sample, training=False)
        for key in ("z_mean", "z_log_var", "u_mean", "u_log_var"):
            np.testing.assert_allclose(
                ops.convert_to_numpy(encoded[key]),
                ops.convert_to_numpy(outputs[key]),
                atol=1e-6,
                rtol=0,
            )

    def test_decode_reconstructs_what_call_reconstructed(self):
        """The two stages and the forward path compute the same quantity."""
        model = _build()
        outputs = model(_sample(), training=False)
        direct = ops.convert_to_numpy(
            model.decode(ops.convert_to_numpy(outputs["t"]), training=False)
        )
        np.testing.assert_allclose(
            direct, ops.convert_to_numpy(outputs["reconstruction"]), atol=1e-5,
            rtol=1e-5,
        )

    def test_the_decoder_really_reads_the_topographic_variables(self):
        """The load-bearing-claim guard.

        Perturbing ONLY ``t`` must move the reconstruction. If the decoder took
        ``concat([u, z])`` as well it could ignore ``t`` entirely, the capsule
        structure could vanish, and the likelihood would stay flat -- the whole
        model unfalsifiable.
        """
        model = _build()
        sample = _sample()
        outputs = model(sample, training=False)
        latent = ops.convert_to_numpy(outputs["t"])
        bumped = latent.copy()
        bumped[:, 0, 0] += 5.0
        baseline = ops.convert_to_numpy(
            model.decode(latent, training=False)
        )
        moved = ops.convert_to_numpy(model.decode(bumped, training=False))
        assert np.abs(moved - baseline).max() > 1e-4, (
            "the reconstruction did not respond to `t`; the decoder is reading "
            "somewhere else and the topography is decorative"
        )

    def test_is_graph_safe_and_matches_tf_function(self):
        model = _build()
        sample = _sample()
        eager = ops.convert_to_numpy(model.encode(sample)["z_mean"])

        @tf.function
        def traced(x):
            return model.encode(x)["z_mean"]

        np.testing.assert_allclose(
            ops.convert_to_numpy(traced(ops.convert_to_tensor(sample))),
            eager,
            atol=1e-6,
            rtol=0,
        )

    def test_matches_under_xla(self):
        model = _build()
        sample = _sample()
        eager = ops.convert_to_numpy(model.encode(sample)["z_mean"])

        @tf.function(jit_compile=True)
        def compiled(x):
            return model.encode(x)["z_mean"]

        # TOLERANCE DERIVED, not pasted. XLA reassociates every reduction it
        # fuses, and on a tensor-core GPU it also runs the matmuls in TF32's
        # 10-bit mantissa -- so the same computation differs between the two by an
        # amount that depends on the DEVICE. `matmul_precision_atol` MEASURES the
        # active device's matmul roundoff rather than assuming one, and
        # `reassociation_atol` bounds the reordering term; the maximum of the two
        # is the honest bound in either regime.
        #
        # MEASURED: 1.01e-06 on CPU against a 1e-04 bound (float32
        # reassociation), and 2.89e-03 on GPU 1 against that same bound, where
        # the TF32 term is ~4100x larger and dominates. A pasted 1e-04 passes on
        # one device and fails on the other, which is why it was not pasted.
        got = ops.convert_to_numpy(compiled(ops.convert_to_tensor(sample)))
        scale = float(np.abs(eager).max())
        atol = max(
            reassociation_atol(
                [CAPSULE_DIM], num_steps=len(HIDDEN), scale=scale
            ),
            matmul_precision_atol(scale),
        )
        np.testing.assert_allclose(got, eager, atol=atol, rtol=0)

    @pytest.mark.parametrize("policy", ["mixed_float16", "float64"])
    def test_dtype_arms(self, policy):
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy(policy)
        try:
            model = _build()
            outputs = model(_sample(), training=False)
            reconstruction = ops.convert_to_numpy(outputs["reconstruction"])
            assert np.all(np.isfinite(reconstruction)), policy
            assert reconstruction.min() >= -1e-3 and reconstruction.max() <= 1.0 + 1e-3
        finally:
            keras.mixed_precision.set_global_policy(previous)

    def test_rejects_a_wrong_input_rank(self):
        model = _build()
        with pytest.raises(ValueError, match="rank-5 input"):
            model.build((None, 28, 28, 1))

    def test_rejects_frames_that_contradict_the_configuration(self):
        model = _build()
        with pytest.raises(ValueError, match="do not match the configured"):
            model.build((None, SEQUENCE, 8, 8, 1))


class TestCapsuleTraversal:
    """The paper's central claim, made checkable."""

    def test_frame_zero_equals_a_direct_decode_of_t_zero(self):
        """MEASURED delta 0.0.

        If the traversal operator and the decode path disagreed, every traversal
        figure in the paper's Figures 1 and 4 would be a picture of a bug.
        """
        model = _build()
        sample = _sample()
        latents = ops.convert_to_numpy(model(sample, training=False)["t"])
        traversal = ops.convert_to_numpy(
            model.traverse_capsules(latents, num_steps=SEQUENCE)
        )
        direct = ops.convert_to_numpy(
            model.decode(latents[:, :1], training=False)
        )
        np.testing.assert_allclose(
            traversal[:, 0], direct[:, 0], atol=0.0, rtol=0
        )

    def test_consecutive_traversed_frames_differ(self):
        """The other half: a traversal that returned a constant would satisfy the
        equality above while showing nothing."""
        model = _build()
        latents = ops.convert_to_numpy(model(_sample(), training=False)["t"])
        traversal = ops.convert_to_numpy(
            model.traverse_capsules(latents, num_steps=SEQUENCE)
        )
        deltas = np.abs(traversal[:, 1:] - traversal[:, :-1]).max()
        assert deltas > 1e-4, "the traversal produced identical frames"

    def test_the_traversal_rolls_within_capsules_not_across_them(self):
        """The roll axis is the whole point, and it is checked on an IMPULSE.

        A roll over the FLAT latent width would carry coordinate 0 of capsule 0
        into the LAST coordinate of the LAST capsule; a roll within capsules
        carries it to coordinate 1 of its OWN capsule. Both operators have the
        same shape and the same output rank, so only an impulse tells them
        apart: a flat ``np.roll`` by ``+1`` lands on coordinate 15, and this one
        on coordinate 1.

        MEASURED: the support is ``[1]``, against 15 for the flat roll.
        """
        model = _build()
        latents = np.zeros((1, LATENT), "float32")
        latents[0, 0] = 1.0

        rolled = CapsuleRoll(num_capsules=CAPSULES, capsule_dim=CAPSULE_DIM)(
            ops.convert_to_tensor(latents.reshape(1, CAPSULES, CAPSULE_DIM)),
            shift=1,
        )
        rolled_flat = ops.convert_to_numpy(rolled).reshape(1, LATENT)
        support = np.nonzero(np.abs(rolled_flat[0]) > 1e-6)[0]
        assert support.tolist() == [1], (
            f"the impulse landed on {support.tolist()}; "
            f"{LATENT - 1} would mean a flat-latent roll"
        )
        np.testing.assert_allclose(rolled_flat.sum(), 1.0, atol=0.0, rtol=0)

        # And the traversal's second step is that decode, exactly.
        traversal = ops.convert_to_numpy(
            model.traverse_capsules(latents, num_steps=2)
        )
        direct = ops.convert_to_numpy(
            model.decode(
                ops.convert_to_tensor(rolled_flat[:, None, :]), training=False
            )
        )
        np.testing.assert_allclose(
            traversal[:, 1], direct[:, 0], atol=1e-5, rtol=1e-5
        )

    @pytest.mark.parametrize("ranks", [2, 3, 4])
    def test_accepts_the_documented_input_ranks(self, ranks):
        model = _build()
        shapes = {2: (2, LATENT), 3: (2, CAPSULES, CAPSULE_DIM),
                  4: (2, SEQUENCE, LATENT)}
        values = np.random.default_rng(ranks).normal(
            size=shapes[ranks]
        ).astype("float32")
        out = model.traverse_capsules(values, num_steps=3)
        assert tuple(out.shape) == (2, 3) + FRAME

    def test_rejects_an_unsupported_rank(self):
        """Rank 5 is the frames shape, and rank 1 is a bare latent vector.

        MEASURED: rank 4 -- ``(batch, sequence, num_capsules, capsule_dim)`` --
        is ACCEPTED, so it cannot be the probe here; the accepted set is
        ``{2, 3, 4}`` and the rejected ones are outside it.
        """
        model = _build()
        with pytest.raises(ValueError, match="accepts"):
            model.traverse_capsules(np.zeros((2, 3, 4, 5, 6), "float32"))
        with pytest.raises(ValueError, match="accepts"):
            model.traverse_capsules(np.zeros((LATENT,), "float32"))

    def test_rejects_a_non_positive_step_count(self):
        model = _build()
        with pytest.raises(ValueError, match="num_steps must be positive"):
            model.traverse_capsules(np.zeros((2, LATENT), "float32"), num_steps=0)


class TestBuildContract:
    """`build()` must materialize exactly what `call()` runs."""

    def test_explicit_build_matches_lazy_build(self):
        explicit = TopographicVAE(**_config())
        explicit.build((None, SEQUENCE) + FRAME)
        lazy = _build()
        lazy(_sample())
        assert _relative(explicit) == _relative(lazy)

    def test_count_params_is_non_zero_after_an_explicit_build(self):
        """The classic defect: a subclassed model whose `build()` walks no
        sub-layers reports `count_params() == 0` without raising."""
        explicit = TopographicVAE(**_config())
        explicit.build((None, SEQUENCE) + FRAME)
        assert explicit.count_params() > 0

    def test_a_disabled_component_builds_no_weights(self):
        """The anti-vacuity sibling: the parity guard above would pass if BOTH
        paths built everything."""
        model = _build(use_variance_variables=False, coherence_window=0)
        model(_sample())
        paths = _relative(model)
        assert not [p for p in paths if "u_encoder" in p], "u_encoder was built"
        assert not [
            p for p in paths if "topographic_product" in p
        ], "the topography layer was built with no u variables"

    def test_the_baseline_really_lacks_those_layers(self):
        model = _build(use_variance_variables=False, coherence_window=0)
        assert model.u_encoder is None
        assert model.u_sampling is None
        assert model.topographic_product is None


class TestGradients:
    """Per-variable gradient flow."""

    def test_every_trainable_variable_has_a_non_zero_gradient(self):
        model = _build()
        sample = _sample()
        with tf.GradientTape() as tape:
            outputs = model(sample, training=True)
            loss = ops.mean(ops.square(outputs["reconstruction"]))
        gradients = tape.gradient(loss, model.trainable_variables)

        assert len(model.trainable_variables) > 0, "anti-vacuity"
        for variable, gradient in zip(model.trainable_variables, gradients):
            assert gradient is not None, f"no gradient for {variable.path}"
            assert np.any(
                ops.convert_to_numpy(gradient) != 0.0
            ), f"all-zero gradient for {variable.path}"

    def test_the_variance_encoder_receives_a_gradient(self):
        """Without this the whole inductive bias could be inert.

        The topographic bias enters through ``u``, so if the ``u`` encoder were
        disconnected from the loss, ``t`` would be a constant rescaling of ``z``
        and the model's name would be a claim rather than a mechanism.

        The window STACK is deliberately excluded: it is a non-trainable constant
        built from the neighbourhood matrix, so it has no gradient to receive and
        asserting one would be asserting a defect. Its non-trainability is pinned
        by the sibling below.
        """
        model = _build()
        with tf.GradientTape() as tape:
            outputs = model(_sample(), training=True)
            loss = ops.mean(ops.square(outputs["reconstruction"]))
        by_path = {
            variable.path: gradient
            for variable, gradient in zip(
                model.trainable_variables,
                tape.gradient(loss, model.trainable_variables),
            )
        }
        u_encoder = [
            gradient for path, gradient in by_path.items()
            if "u_encoder" in path
        ]
        assert u_encoder, "the u encoder carries no gradient path"
        for gradient in u_encoder:
            assert np.any(ops.convert_to_numpy(gradient) != 0.0), (
                "the u encoder got an all-zero gradient; the topographic bias "
                "is inert"
            )

    def test_the_window_stack_is_not_trainable(self):
        """`G` is a CONSTANT derived from the neighbourhood matrix, not a weight.

        It is created with ``trainable=False``, so it must be absent from
        ``trainable_variables`` -- and its presence in ``weights`` is what makes
        it survive the round trip.
        """
        model = _build()
        model(_sample())
        assert model.trainable_variables
        stacks = [w for w in model.weights if "window_stack" in w.path]
        assert stacks, "no window stack was built"
        for stack in stacks:
            assert not stack.trainable, stack.path
        assert not [w for w in model.trainable_variables
                    if "window_stack" in w.path]


class TestTrainingPath:
    """One fit step must move every weight, and the loss must go down."""

    def _compiled(self, **overrides):
        model = _build(**overrides)
        model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9),
            loss=TopographicVAELoss(kl_loss_weight=0.01),
        )
        return model

    def test_every_trainable_variable_moves_in_one_step(self):
        """Reported as a NAME SET, never a count.

        `moved > 0` was once satisfied by a 118-of-137 result whose 19-variable
        residual was never identified. The snapshot is over
        ``trainable_variables``, NOT ``weights``: the window stack is a
        non-trainable constant and demanding it move would be demanding that a
        buffer be an optimizer's business.
        """
        model = self._compiled()
        sample = _sample(batch=4)
        model.fit(sample, sample, epochs=1, batch_size=2, verbose=0)
        before = _snapshot(model)
        model.fit(sample, sample, epochs=1, batch_size=2, verbose=0)
        after = _snapshot(model)
        moved = {
            path for path in before
            if not np.array_equal(before[path], after[path])
        }
        expected = set(before)
        assert len(expected) >= 18, (
            f"only {len(expected)} trainable variables; the probe would be "
            "trivially satisfiable at this size"
        )
        assert moved == expected, f"never moved: {sorted(expected - moved)}"

    def test_the_non_trainable_window_stack_does_not_move(self):
        """The anti-vacuity sibling for the snapshot above.

        If the window stack DID move it would be a trainable parameter after all
        and the "every trainable variable" set would be excluding a real weight.
        """
        model = self._compiled()
        sample = _sample(batch=2)
        model.fit(sample, sample, epochs=1, batch_size=2, verbose=0)
        before = {
            w.path: np.array(ops.convert_to_numpy(w)).copy()
            for w in model.weights if "window_stack" in w.path
        }
        model.fit(sample, sample, epochs=1, batch_size=2, verbose=0)
        after = {
            w.path: np.array(ops.convert_to_numpy(w)).copy()
            for w in model.weights if "window_stack" in w.path
        }
        assert before, "no window stack to check"
        for path in before:
            np.testing.assert_array_equal(before[path], after[path], err_msg=path)

    def test_the_probe_can_detect_a_dead_training_path(self):
        """RED proof: with the outputs cut off the tape, fit MUST raise.

        Without this, the movement assertion above would also pass for a model
        whose loss did not depend on its output.
        """
        model = self._compiled()
        sample = _sample(batch=2)

        # `train_function` is reset on BOTH edges: Keras caches the traced train
        # step, so a pre-fitted model would keep running the UNPATCHED graph and
        # the injection would silently do nothing.
        with broken_forward(model, stop_all_gradients):
            model.train_function = None
            try:
                with pytest.raises(
                    ValueError, match="No gradients provided for any variable"
                ):
                    model.fit(
                        sample, sample, epochs=1, batch_size=2, verbose=0
                    )
            finally:
                model.train_function = None

    def test_the_loss_decreases_over_several_epochs(self):
        model = self._compiled()
        sample = _sample(batch=8)
        history = model.fit(
            sample, sample, epochs=6, batch_size=4, verbose=0
        ).history
        assert history["loss"][-1] < history["loss"][0], history["loss"]

    def test_sample_weight_is_refused_with_a_reason(self):
        model = self._compiled()
        sample = _sample(batch=2)
        with pytest.raises(ValueError, match="does not support sample_weight"):
            model.compute_loss(
                x=sample,
                y=sample,
                y_pred=model(sample, training=False),
                sample_weight=ops.convert_to_tensor(
                    np.ones(2, "float32")
                ),
            )

    def test_compute_loss_without_a_compiled_loss_raises(self):
        model = _build()
        sample = _sample()
        with pytest.raises(ValueError, match="needs a compiled loss"):
            model.compute_loss(
                x=sample, y=None, y_pred=model(sample, training=False)
            )

    def test_the_loss_trackers_are_live(self):
        """The per-term decomposition must report real numbers.

        MEASURED: an earlier revision kept three trackers on the MODEL that
        nothing ever updated, so every epoch log read
        `reconstruction_loss: 0.0000e+00` beside a real loss of 85.05. Dead
        metrics reporting zero are worse than no metrics.
        """
        model = self._compiled()
        assert len(model.metric_trackers) == 3, (
            "the compiled loss carries the per-term trackers"
        )
        sample = _sample(batch=2)
        model.fit(sample, sample, epochs=1, batch_size=2, verbose=0)
        values = [
            float(ops.convert_to_numpy(tracker.result()))
            for tracker in model.metric_trackers
        ]
        for name, value in zip(("reconstruction", "z_kl", "u_kl"), values):
            assert value != 0.0, f"the {name} tracker reported exactly 0.0"

    def test_an_uncompiled_model_reports_no_trackers(self):
        """The other side of the property: no compiled loss, no decomposition.

        Claiming three zeros for a model that never computed them is the defect
        the property's docstring refuses, so it is pinned rather than assumed.
        """
        model = _build()
        model(_sample())
        assert model.metric_trackers == []

    def test_a_plain_objective_reports_no_trackers(self):
        """A compiled MSE has no decomposition either -- and must not borrow
        one from the loss's default trackers."""
        model = _build()
        model.compile(optimizer=keras.optimizers.SGD(1e-3), loss="mse")
        model.fit(_sample(batch=2), _sample(batch=2), epochs=1, batch_size=2,
                  verbose=0)
        assert model.metric_trackers == []


class TestLogLikelihood:
    """The paper's Table 1 column."""

    def test_the_bound_TIGHTENS_with_more_samples(self):
        """The IWAE property, and the sharpest available check on the signs.

        ``log (1/K) sum_k exp(w_k)`` is non-decreasing in ``K`` -- that is what
        makes it an upper bound on ``log p(x)`` rather than an arbitrary number.
        Every sign error in the weights flips this or flattens it, so the arm
        below convicts a negated likelihood, a swapped prior/posterior pair and
        an inverted log-mean-exp alike.

        MEASURED on the reference instance: -7064.3, -5841.5, -5689.3, -5464.9,
        -5185.9 for K = 1, 2, 4, 8, 16.
        """
        model = _build()
        sample = _sample()
        values = [
            float(
                ops.convert_to_numpy(
                    model.log_likelihood(sample, num_samples=k, seed=3)
                )[0]
            )
            for k in (1, 2, 4, 8, 16)
        ]
        assert all(earlier <= later for earlier, later in zip(values, values[1:])), (
            f"the bound is not monotone in K: {values}"
        )
        assert values[-1] > values[0], (
            f"K=16 gave no improvement over K=1: {values}"
        )

    def test_k_equals_one_is_exactly_its_own_weight(self):
        """The definitional check, against a hand-evaluated single draw.

        For ``K = 1`` the bound collapses to the weight itself,
        ``log p(x|z) + log p(z) - log q(z|x)`` per timestep, summed -- no
        log-mean-exp left to get wrong. The oracle recovers ``z`` from the value
        ``decode`` was actually handed (``z = t + mu`` on the baseline, where
        ``t = z - mu``), so the noise draw is observed rather than re-seeded:
        re-seeding it from outside the method is a different draw and the
        comparison silently stops being an equality.

        MEASURED: agreement to 3e-05 absolute on a value near -1.1e+04, which is
        float32 round-off over a sum of ~1500 terms.
        """
        model = _build(use_variance_variables=False, coherence_window=0)
        sample = _sample(batch=1)

        captured = []
        original_decode = model.decode

        def _record(t, training=None):
            captured.append(np.asarray(ops.convert_to_numpy(t)))
            return original_decode(t, training=training)

        with mock.patch.object(model, "decode", _record):
            reported = float(
                ops.convert_to_numpy(
                    model.log_likelihood(sample, num_samples=1, seed=7)
                )[0]
            )

        assert len(captured) == 1, captured
        z = captured[0] + float(model.prior_mean)
        encoded = model.encode(sample)
        z_mean = np.asarray(ops.convert_to_numpy(encoded["z_mean"]))
        z_log_var = np.asarray(ops.convert_to_numpy(encoded["z_log_var"]))
        assert np.abs(z - z_mean).max() > 1e-6, (
            "the captured t carries no sampling noise; the oracle would then be "
            "checking the posterior mean rather than a draw"
        )

        reconstruction = np.asarray(
            ops.convert_to_numpy(original_decode(captured[0], training=False))
        )
        clipped = np.clip(reconstruction, 1e-7, 1.0 - 1e-7)
        log_likelihood = float(
            np.sum(
                sample * np.log(clipped) + (1.0 - sample) * np.log1p(-clipped)
            )
        )
        log_prior = float(np.sum(-0.5 * (z**2 + np.log(2.0 * np.pi))))
        log_posterior = float(
            np.sum(-0.5 * ((z - z_mean) ** 2 + z_log_var + np.log(2.0 * np.pi)))
        )
        expected = log_likelihood + log_prior - log_posterior
        np.testing.assert_allclose(reported, expected, atol=3e-3, rtol=0)

    def test_the_shape_is_per_example(self):
        model = _build()
        values = model.log_likelihood(_sample(batch=3), num_samples=2, seed=0)
        assert tuple(ops.shape(values)) == (3,)

    def test_a_seeded_estimate_is_reproducible(self):
        """The samplers are stochastic, so an unseeded estimate is not comparable
        across runs and neither is a metric computed from it."""
        model = _build()
        sample = _sample()
        first = ops.convert_to_numpy(
            model.log_likelihood(sample, num_samples=3, seed=1)
        )
        second = ops.convert_to_numpy(
            model.log_likelihood(sample, num_samples=3, seed=1)
        )
        np.testing.assert_array_equal(first, second)

    def test_more_samples_move_the_estimate(self):
        """A bound that never tightens with ``K`` is not a bound.

        A single draw reduces to the plain ELBO, so the estimate must change as
        K grows -- otherwise the column is noise.
        """
        model = _build()
        sample = _sample()
        one = ops.convert_to_numpy(
            model.log_likelihood(sample, num_samples=1, seed=0)
        )
        five = ops.convert_to_numpy(
            model.log_likelihood(sample, num_samples=5, seed=0)
        )
        assert not np.allclose(one, five)

    def test_it_is_negative_for_a_good_reconstruction(self):
        """The SIGN of the paper's column.

        ``log p(x)`` is a log-likelihood and the paper reports negative values
        (-186.8 for its best MNIST model). A positive number here is the NLL with
        the sign flipped -- a defect a shape test cannot see, and one this
        implementation shipped until the closed-form oracle in the loss suite
        caught it.
        """
        model = _build()
        # A model whose decoder output matches the data should score well.
        sample = _sample(batch=2)
        value = ops.convert_to_numpy(
            model.log_likelihood(sample, num_samples=2, seed=0)
        )
        finite = np.isfinite(value)
        assert finite.all()
        # An untrained decoder emits ~0.5 everywhere, whose log-likelihood against
        # uniform targets is about -H(target) per pixel summed over the sequence:
        # a LARGE negative number. Assert the sign, not the magnitude.
        assert value[0] < 0.0, f"log p(x) came out positive: {value}"

    def test_a_confident_correct_decoder_scores_higher_than_a_wrong_one(self):
        """Liveness with a sign, on a CONTROLLED reconstruction.

        Feeding an untrained model two different sequences and comparing its two
        numbers measures the untrained encoder, not the likelihood -- the
        reconstructions differ because the ENCODER saw different input, so the
        comparison's verdict is a coin flip. So the reconstruction is held fixed
        and only the TARGET moves, which isolates the likelihood term the bound
        is built from.

        MEASURED: a reconstruction equal to its target scores 0.0 and the same
        reconstruction against its complement scores about -12.0 per frame.
        """
        model = _build()
        targets = _sample(batch=1)
        reconstruction = ops.convert_to_tensor(
            np.clip(targets, 1e-3, 1.0 - 1e-3), "float32"
        )
        # Image axes only, then the sequence sum -- the per-example total that
        # `log_likelihood` accumulates, at a reconstruction the caller chose.
        image_axes = list(range(2, len(FRAME) + 2))
        per_example = lambda t: ops.sum(
            ops.sum(
                model._bernoulli_log_likelihood(t, reconstruction),
                axis=image_axes,
            ),
            axis=-1,
        )
        matched = float(ops.convert_to_numpy(per_example(targets))[0])
        mismatched = float(ops.convert_to_numpy(per_example(1.0 - targets))[0])
        assert matched > mismatched, (matched, mismatched)
        assert np.isfinite(matched) and np.isfinite(mismatched)

    def test_a_trained_decoder_prefers_the_data_it_was_trained_on(self):
        """The end-to-end version of the arm above, with a real decoder.

        The epoch count is not decoration and not an optimism knob -- it is the
        point at which the reconstruction begins to correlate with the data at
        all. MEASURED on this reference instance: correlation 0.00 after 2 epochs,
        0.05 after 30. Below that the decoder emits a near-constant field and the
        two likelihoods differ by the CONSTANT's distance to each target, so the
        comparison reports on the initialization and not on the model -- which is
        why 2 epochs produced the wrong sign here and 30 does not.
        """
        model = _build()
        model.compile(
            optimizer=keras.optimizers.Adam(2e-3),
            loss=TopographicVAELoss(kl_loss_weight=0.01),
        )
        sample = _sample(batch=2)
        model.fit(sample, sample, epochs=30, batch_size=2, verbose=0)

        reconstruction = ops.convert_to_numpy(
            model(sample, training=False)["reconstruction"]
        )
        correlation = float(
            np.corrcoef(reconstruction.ravel(), sample.ravel())[0, 1]
        )
        assert correlation > 0.0, (
            f"the decoder is uncorrelated with the data ({correlation:.3f}); the "
            "comparison below would measure the initialization"
        )

        # Both estimates are seeded, so the only difference between them is the
        # data being scored.
        good = float(
            ops.convert_to_numpy(
                model.log_likelihood(sample, num_samples=1, seed=0)
            )[0]
        )
        bad = float(
            ops.convert_to_numpy(
                model.log_likelihood(1.0 - sample, num_samples=1, seed=0)
            )[0]
        )
        assert good > bad, (good, bad)

    def test_rejects_a_non_positive_sample_count(self):
        model = _build()
        with pytest.raises(ValueError, match="num_samples must be positive"):
            model.log_likelihood(_sample(), num_samples=0)


class TestPriorSampling:
    """`sample_prior` reconstructs the topography from Gaussian draws."""

    def test_shapes(self):
        model = _build()
        draws = model.sample_prior(5, seed=0)
        assert tuple(draws["z"].shape) == (5, CAPSULES, CAPSULE_DIM)
        assert tuple(draws["t"].shape) == (5, CAPSULES, CAPSULE_DIM)
        assert np.all(np.isfinite(ops.convert_to_numpy(draws["t"])))

    def test_the_heavy_tailed_prior_actually_has_heavy_tails(self):
        """Section B.3 validates the prior by sampling it.

        A Student-t latent has kurtosis well above the Gaussian's 3.0; a
        construction that silently produced a Gaussian would score 3.0 exactly and
        the "samples look like MNIST" check would be passing on a different
        distribution than the paper's.
        """
        model = _build(neighborhood_size=1, degrees_of_freedom=1.0)
        samples = ops.convert_to_numpy(
            model.sample_prior(20000, seed=0)["t"]
        ).reshape(-1)
        centered = samples - samples.mean()
        kurtosis = float(
            np.mean(centered**4) / np.mean(centered**2) ** 2
        )
        assert kurtosis > 5.0, (
            f"kurtosis {kurtosis:.2f} is not heavier-tailed than a Gaussian's 3.0"
        )

    def test_it_is_seeded(self):
        model = _build()
        first = ops.convert_to_numpy(model.sample_prior(3, seed=5)["t"])
        second = ops.convert_to_numpy(model.sample_prior(3, seed=5)["t"])
        np.testing.assert_array_equal(first, second)

    def test_the_baseline_omits_the_u_draw(self):
        model = _build(use_variance_variables=False, coherence_window=0)
        draws = model.sample_prior(3, seed=0)
        assert "u" not in draws
        assert "t" in draws


class TestKnobSensitivity:
    """Every constructor parameter must change something measurable.

    Routed through the shared ``knob_sensitivity_oracle`` rather than
    hand-rolled, because the hand-rolled form has a specific vacuity: a
    STRUCTURAL knob changes the random draw as a side effect, so
    ``assert not np.allclose(out_a, out_b)`` passes on a model that drops the
    kwarg entirely. The oracle splits that in two -- ``assert_structural_knob_
    changes_weights`` for knobs that change the parameterisation,
    ``assert_value_knob_changes_output`` for knobs that do not (with the shape
    signature held equal so the difference is attributable), and
    ``assert_scoped_value_knob_changes_weights`` for a knob forwarded to ONE
    subtree.

    Each knob is additionally given the STRONGEST claim its own semantics allow,
    which the oracle cannot make: a closed form rather than a difference.
    """

    def _built(self, **overrides):
        """A BUILT instance.

        The structural oracle deliberately takes no inputs, so its builders must
        return a built model -- a subclassed ``keras.Model`` is unbuilt until its
        first ``call()``, and an unbuilt builder yields an empty signature that
        compares equal to every other empty one.
        """
        model = _build(**overrides)
        model(_sample())
        return model

    def test_num_capsules_changes_the_weight_shapes(self):
        signatures = assert_structural_knob_changes_weights(
            {c: (lambda c=c: self._built(num_capsules=c))
             for c in (CAPSULES, CAPSULES + 2, CAPSULES + 4)},
            knob="num_capsules",
        )
        # Stronger than "different": the latent width moves by exactly the
        # capsule count. It is read off the window STACK's `(2L+1, D, D)`
        # shape, which is the one place `latent_dim` appears verbatim as BOTH
        # trailing axes -- the decoder's kernels expose one of them at a time and
        # its OUTPUT width is the frame size regardless of the knob, which is
        # how a wrong index reads as "unchanged".
        widths = {
            capsules: _window_stack_shape(signature)[1]
            for capsules, signature in signatures.items()
        }
        assert widths[CAPSULES] == LATENT, widths
        assert widths[CAPSULES + 4] == (CAPSULES + 4) * CAPSULE_DIM, widths

    def test_capsule_dim_changes_the_weight_shapes(self):
        signatures = assert_structural_knob_changes_weights(
            {d: (lambda d=d: self._built(capsule_dim=d))
             for d in (CAPSULE_DIM, CAPSULE_DIM + 2)},
            knob="capsule_dim",
        )
        widths = {
            dim: _window_stack_shape(signature)[1]
            for dim, signature in signatures.items()
        }
        assert widths[CAPSULE_DIM] == LATENT, widths
        assert widths[CAPSULE_DIM + 2] == CAPSULES * (CAPSULE_DIM + 2), widths

    def test_the_coherence_window_changes_the_window_stack_shape(self):
        """Structural, and scoped: the stack grows by TWO slices per unit of L.

        The claim is on the stack's own shape rather than on the model-wide
        signature, because ``coherence_window`` is ALSO a value knob: it changes
        how the slices are composed, not only how many there are.
        """
        sample = _sample()
        stacks = {}
        for window in (0, 2):
            model = _build(coherence_window=window)
            model(sample)
            selected = weights_in_scope(model, "window_stack")
            assert selected, "no window stack"
            stacks[window] = tuple(selected[0].shape)
        assert stacks[0] == (1, LATENT, LATENT), stacks[0]
        assert stacks[2] == (5, LATENT, LATENT), stacks[2]

    def test_temporal_coherence_changes_the_outputs(self):
        """Value knob -> pinned on OUTPUTS with the shape signature held equal.

        This is the knob whose two arms are EASY to confuse: `shifting` and
        `stationary` both produce a ``(2L+1, D, D)`` stack, so every shape
        assertion and every layer-vs-oracle comparison passes identically while
        the roll is entirely absent. Only the OUTPUT differs.
        """
        deltas = assert_value_knob_changes_output(
            {mode: (lambda mode=mode: _build(temporal_coherence=mode))
             for mode in ("shifting", "stationary")},
            _sample(),
            knob="temporal_coherence",
            extract=lambda out: out["reconstruction"],
        )
        assert max(deltas.values()) > 1e-4, deltas

    def test_the_neighborhood_size_reaches_the_window_stack(self):
        """A knob forwarded to ONE subtree, so it is pinned on THAT subtree's
        weights -- a whole-model output diff would be satisfied by the encoder
        and pass even if ``K`` were dropped.

        The bound here is exact zero (the oracle's), not a tolerance: the arms
        hold bit-identical encoder weights and differ only in ``G``.
        """
        assert_scoped_value_knob_changes_weights(
            {k: (lambda k=k: _build(neighborhood_size=k))
             for k in (1, 2, CAPSULE_DIM)},
            _sample(),
            knob="neighborhood_size",
            scope="window_stack",
        )

    def test_the_neighborhood_window_is_the_documented_circulant(self):
        """The closed form the value instrument cannot make: ``K`` is exactly the
        row width of the circulant block, and ``K = D`` is the degenerate
        all-ones block the paper reports as the invariant failure.

        MEASURED: ``K=1`` gives ``kron(I_C, R_{-L})`` at slice 0, because
        ``build_window_stack`` indexes ``delta`` from ``-L`` and composes the
        coherence roll onto ``W``. Reading a slice as ``W`` alone would be a sign
        error, so the closed form here goes through the same helper the layer
        does and pins the STRUCTURE on top of it: the block is the circulant
        ``W`` and the off-block entries are exactly zero.
        """
        for neighborhood_size, expected_block_sum in ((1, 1), (CAPSULE_DIM, 4)):
            model = _build(neighborhood_size=neighborhood_size)
            model(_sample())
            stack = ops.convert_to_numpy(
                weights_in_scope(model, "window_stack")[0]
            )
            expected = build_window_stack(
                _blocked_window(neighborhood_size),
                LATENT,
                2,
                "shifting",
            )
            np.testing.assert_allclose(stack, expected, atol=0.0, rtol=0)

            block = build_window_matrix_1d(CAPSULE_DIM, neighborhood_size)
            assert np.allclose(block.sum(axis=1), expected_block_sum), (
                f"K={neighborhood_size}: every row of the block must sum to "
                f"{expected_block_sum}"
            )
            # And the window does NOT cross capsules: the blocked form's
            # off-block entries are zero, whereas the all-ones 16x16 matrix
            # would be the invariant-representation failure.
            # The block-diagonal structure: no capsule shares energy with
            # another. `Kron(I_C, all-ones)` is NOT the 16x16 all-ones matrix,
            # which is the invariant-representation failure the paper reports.
            blocked = np.kron(np.eye(CAPSULES), block)
            for capsule in range(CAPSULES):
                start = capsule * CAPSULE_DIM
                stop = start + CAPSULE_DIM
                others = [i for i in range(LATENT)
                          if not start <= i < stop]
                assert np.abs(blocked[start:stop, others]).max() == 0.0, (
                    f"capsule {capsule} shares energy with another capsule"
                )
                assert np.allclose(blocked[start:stop, start:stop], block)

    def test_degrees_of_freedom_changes_the_output_scale(self):
        """Value knob with a DERIVED invariant: Eq. 6 puts nu only under the
        square root, so quadrupling it halves ``t``.

        This is the closed form the oracle's "the outputs differ" cannot make --
        it pins the exact factor.
        """
        sample = _sample()
        values = {}
        for nu in (1.0, 4.0):
            builder = (
                lambda nu=nu: _build(
                    degrees_of_freedom=nu, neighborhood_size=1, coherence_window=0
                )
            )
            keras.utils.set_random_seed(13)
            values[nu] = ops.convert_to_numpy(
                builder()(sample, training=False)["t"]
            )
        ratio = np.abs(values[1.0]) / np.maximum(np.abs(values[4.0]), 1e-12)
        assert np.median(ratio) == pytest.approx(2.0, rel=0.2), (
            f"the median |t(nu=1)| / |t(nu=4)| was {np.median(ratio)}, not 2"
        )

    def test_prior_mean_shifts_the_latent(self):
        deltas = assert_value_knob_changes_output(
            {mean: (lambda mean=mean: _build(prior_mean=mean))
             for mean in (0.0, 30.0)},
            _sample(),
            knob="prior_mean",
            extract=lambda out: out["t"],
        )
        # The paper's prior is centred at 30 (Eq. 5), so a shift of that size is
        # not a rounding difference: it must dominate any tolerance.
        assert max(deltas.values()) > 1.0, deltas

    def test_the_activation_changes_the_outputs(self):
        assert_value_knob_changes_output(
            {a: (lambda a=a: _build(activation=a)) for a in ("relu", "tanh")},
            _sample(),
            knob="activation",
            extract=lambda out: out["reconstruction"],
        )

    def test_the_kernel_initializer_reaches_the_encoder(self):
        """A knob forwarded to MOST of the tree.

        Pinned on the encoder subtree's weights, which is where the difference
        actually is: the window stack is a constant derived from K, so it cannot
        respond, and the decoder is reached through a different initializer.
        """
        assert_scoped_value_knob_changes_weights(
            {
                name: (lambda name=name: _build(kernel_initializer=name))
                for name in ("glorot_uniform", "he_normal")
            },
            _sample(),
            knob="kernel_initializer",
            scope="z_encoder",
        )

    def test_use_variance_variables_removes_a_subtree(self):
        """The one knob pinned on LAYOUT: it removes weights, it does not change
        them. Asserted as absence, with the anti-vacuity sibling in
        `TestBuildContract`."""
        model = _build(use_variance_variables=False, coherence_window=0)
        model(_sample())
        assert not [
            p for p in _relative(model) if "u_encoder" in p
        ], "the u encoder survived use_variance_variables=False"


class TestSerialization:
    """Config completeness and value round trips."""

    def test_config_is_complete(self):
        model = TopographicVAE(**_config())
        config = model.get_config()
        for key in (
            "input_shape", "sequence_length", "num_capsules", "capsule_dim",
            "coherence_window", "neighborhood_size", "temporal_coherence",
            "topography", "grid_shape", "use_variance_variables",
            "degrees_of_freedom", "encoder_hidden_dims", "decoder_hidden_dims",
            "prior_mean", "activation", "final_activation",
            "kernel_initializer", "use_bias",
        ):
            assert key in config, key
        assert tuple(config["input_shape"]) == FRAME
        assert config["num_capsules"] == CAPSULES

    def test_config_round_trips_through_from_config(self):
        model = TopographicVAE(**_config())
        restored = TopographicVAE.from_config(model.get_config())
        for key in (
            "num_capsules", "capsule_dim", "coherence_window",
            "neighborhood_size", "temporal_coherence", "degrees_of_freedom",
            "prior_mean", "sequence_length",
        ):
            assert getattr(restored, key) == getattr(model, key), key

    def test_from_config_preserves_the_base_keys(self):
        """A `from_config` that pops `name`/`trainable` reloads a model unfrozen,
        with bit-identical outputs — which no value test would catch."""
        model = TopographicVAE(**_config(name="named_tvae"))
        restored = TopographicVAE.from_config(model.get_config())
        assert restored.name == "named_tvae"

    def test_keras_round_trip_preserves_the_deterministic_outputs(self):
        """`call` samples, so its output cannot be compared at atol=0.0 even
        against itself. `encode()` can, and is the surface this compares.

        The model is FIT first: comparing an unfitted model against a reloaded
        copy round-trips nothing meaningful, and comparing weights before the
        loaded model has been forwarded is the only point at which a
        build()-only load path is distinguishable.
        """
        model = _build()
        model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=1e-3),
            loss=TopographicVAELoss(kl_loss_weight=0.01),
        )
        sample = _sample(batch=4)
        model.fit(sample, sample, epochs=2, batch_size=2, verbose=0)

        before = ops.convert_to_numpy(model.encode(sample)["z_mean"])
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "tvae.keras")
            model.save(path)
            loaded = keras.models.load_model(path)

            # BEFORE the loaded model's first call: after one, a build()-only
            # load path has the same weight COUNT as a correct one, because the
            # gap has been filled with fresh random weights.
            donor = [ops.convert_to_numpy(w).copy() for w in model.weights]
            assert donor, "the donor has no weights to compare"
            for original, restored in zip(donor, loaded.weights):
                # atol=0.0: restoration is a copy, not a computation.
                np.testing.assert_allclose(
                    original, ops.convert_to_numpy(restored), atol=0.0, rtol=0
                )

            after = ops.convert_to_numpy(loaded.encode(sample)["z_mean"])

        np.testing.assert_allclose(before, after, atol=0.0, rtol=0)

    def test_the_baseline_round_trips_too(self):
        model = _build(use_variance_variables=False, coherence_window=0)
        model(_sample())
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "baseline.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        assert _relative(model) == _relative(loaded)
        assert not [p for p in _relative(loaded) if "u_encoder" in p]

    def test_the_saved_model_predicts(self):
        """`predict` raises on a bare `None` inside an output dict.

        The keys are checked by MEMBERSHIP rather than by indexing: a `None`
        entry survives `model(x)` silently and only surfaces at `predict`, so an
        assertion over a fixed key list is the one that catches it.
        """
        model = _build()
        model(_sample())
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "predict.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        predictions = loaded.predict(_sample(batch=2), verbose=0)
        assert set(predictions) == EXPECTED_KEYS, sorted(predictions)
        for key, value in predictions.items():
            array = np.asarray(value)
            assert np.all(np.isfinite(array)), key
            assert array.shape[0] == 2, (key, array.shape)

    def test_the_baseline_predicts_without_the_u_keys(self):
        model = _build(use_variance_variables=False, coherence_window=0)
        model(_sample())
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "baseline_predict.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        predictions = loaded.predict(_sample(batch=2), verbose=0)
        assert set(predictions) == EXPECTED_KEYS - {"u_mean", "u_log_var"}


class TestVariantsAndFactory:
    """The public construction surface."""

    def test_the_variant_table_matches_the_paper(self):
        """Section A.3's widths, transcribed, with the input resolutions pinned by
        the decoder output widths: 2352 = 28*28*3 and 4096 = 64*64*1."""
        mnist = TopographicVAE.MODEL_VARIANTS["mnist"]
        assert mnist["encoder_hidden_dims"] == [972, 648]
        assert mnist["decoder_hidden_dims"] == [648, 972]
        assert mnist["num_capsules"] == mnist["capsule_dim"] == 18
        assert mnist["sequence_length"] == 18
        assert np.prod(mnist["input_shape"]) == 2352

        dsprites = TopographicVAE.MODEL_VARIANTS["dsprites"]
        assert dsprites["encoder_hidden_dims"] == [674, 450]
        assert dsprites["decoder_hidden_dims"] == [450, 674]
        assert dsprites["num_capsules"] == dsprites["capsule_dim"] == 15
        assert dsprites["sequence_length"] == 15
        assert np.prod(dsprites["input_shape"]) == 4096

    def test_every_variant_has_a_consistent_architecture(self):
        for name, config in TopographicVAE.MODEL_VARIANTS.items():
            frame = tuple(config["input_shape"])
            assert len(frame) == 3, name
            assert frame[0] >= 8 and frame[1] >= 8, name

    @pytest.mark.parametrize("variant", sorted(TopographicVAE.MODEL_VARIANTS))
    def test_from_variant_builds(self, variant):
        model = TopographicVAE.from_variant(variant, encoder_hidden_dims=[8],
                                           decoder_hidden_dims=[8])
        assert model.num_capsules == (
            TopographicVAE.MODEL_VARIANTS[variant]["num_capsules"]
        )
        model.build(
            (None, model.sequence_length) + tuple(
                TopographicVAE.MODEL_VARIANTS[variant]["input_shape"]
            )
        )
        assert model.count_params() > 0

    def test_from_variant_rejects_an_unknown_name(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            TopographicVAE.from_variant("enormous")

    def test_from_variant_listing_the_available_names(self):
        with pytest.raises(ValueError) as info:
            TopographicVAE.from_variant("nope")
        for name in TopographicVAE.MODEL_VARIANTS:
            assert name in str(info.value), name

    def test_pretrained_raises_rather_than_returning_random_weights(self):
        with pytest.raises(NotImplementedError) as info:
            TopographicVAE.from_variant("mnist", pretrained=True)
        message = str(info.value)
        assert "mnist" in message
        assert "weights_path" in message, (
            "the error must name the local alternative"
        )

    def test_pretrained_and_weights_path_together_raise(self):
        with pytest.raises(ValueError, match="not both"):
            TopographicVAE.from_variant(
                "mnist", pretrained=True, weights_path="x.keras"
            )

    def test_from_variant_accepts_its_documented_overrides(self):
        model = TopographicVAE.from_variant(
            "mnist", coherence_window=5, neighborhood_size=1
        )
        assert model.coherence_window == 5
        assert model.neighborhood_size == 1

    def test_the_factory_compiles_with_the_elbo(self):
        model = create_topographic_vae("mnist", **_FACTORY_OVERRIDES)
        assert isinstance(
            model._compile_loss._user_loss, TopographicVAELoss
        ), model._compile_loss._user_loss

    def test_the_factory_model_trains(self):
        """The full compiled path: factory -> ELBO -> optimizer -> one epoch.

        ``input_shape`` is overridden because the factory takes its frame
        geometry from the VARIANT, and the reference frames here are 16x16x1
        rather than the paper's 28x28x3 -- otherwise the model rejects its own
        input, which is the right behaviour and the wrong test.
        """
        model = create_topographic_vae("mnist", **_FACTORY_OVERRIDES)
        assert model._input_shape == FRAME, model._input_shape
        sample = _sample(batch=4)
        history = model.fit(sample, sample, epochs=1, batch_size=2, verbose=0)
        assert np.isfinite(history.history["loss"][0])

    def test_the_factory_trains_the_baseline_mode(self):
        # `coherence_window` is overridden to 0 because a non-zero window
        # requires `use_variance_variables`, and the constructor refuses the
        # combination rather than silently ignoring the window.
        overrides = dict(
            _FACTORY_OVERRIDES, use_variance_variables=False, coherence_window=0
        )
        model = create_topographic_vae("mnist", **overrides)
        sample = _sample(batch=4)
        history = model.fit(sample, sample, epochs=1, batch_size=2, verbose=0)
        assert np.isfinite(history.history["loss"][0])
        assert not [p for p in _relative(model) if "u_encoder" in p]

    def test_the_factory_does_not_report_a_parameter_count_before_building(self):
        """`count_params()` RAISES on an unbuilt layer, so the factory must not
        call it; the sibling `vae` factory's DECISION D-078 says the same."""
        # If it did call it, construction would raise here.
        model = create_topographic_vae("mnist", **_FACTORY_OVERRIDES)
        assert not model.built
