"""``SharedTokenDecoder``: the exact-erf GELU, the scale invariance, the bijection tie.

Three claims here have no shape, dtype, finiteness or round-trip symptom, so
nothing else in this directory can see them:

1. **The activation is the EXACT erf GELU, not the tanh approximation.** This
   package uses both -- ``blocks.py`` selects the ``gelu_tanh`` FFN factory key
   because upstream's ``DiTBlock`` writes ``nn.GELU(approximate="tanh")``
   (``reference/dit.py:117``), while the decoder's ``torch.nn.GELU()`` takes the
   default ``approximate='none'`` (``reference/token_decoder.py:19,21``). A
   "consistency" refactor unifying them changes every logit by a small amount
   and raises nothing. The arms below recompute the whole forward pass in NumPy
   from the model's OWN weights, once with each activation, and show that only
   the exact-erf reconstruction matches.

2. **The decoder is scale-invariant.** ``F.normalize`` on the last axis is what
   "undo the dataest [sic] scaling" means upstream: the bridge carries
   embeddings multiplied by ``BridgeConfig.token_scale``, and after the
   normalize the decoder cannot tell ``x`` from ``k * x``. Deleting the
   normalize leaves a model that trains, saves, loads and predicts.

3. **The decoder sits downstream of step 2's bijection.** Packing a
   ``token_flat`` into a bridge tensor and unpacking it must leave the logits
   untouched; that is the only reason the decoder may ignore the bridge layout.

**The step-7 fresh-model-outputs-zero hazard does NOT apply here.** ``DiTXA``
predicts the exact zero tensor at initialisation because its adaLN gates and its
final projection are both zero-initialised, which silently vacuates any
output-level assertion. This decoder has no zero-initialised kernel: all three
``Dense`` layers use Glorot-uniform kernels (only the biases are zero), so a
fresh model already produces a non-constant output. ``test_a_fresh_decoder_is_
not_degenerate`` asserts that rather than assuming it, and every value arm below
is therefore run on a freshly built model with no ``activate()`` helper.
"""

import math


import keras
import numpy as np
import pytest

from dl_techniques.models.vision_language.bit_diffusion.config import (
    BRIDGE_PRESETS,
)
from dl_techniques.models.vision_language.bit_diffusion.token_bridge import (
    bridge_to_token_flat,
    token_flat_to_bridge,
)
from dl_techniques.models.vision_language.bit_diffusion.token_decoder import (
    DEFAULT_NORMALIZE_EPSILON,
    GELU_APPROXIMATE,
    NUM_MLP_LAYERS,
    SharedTokenDecoder,
    create_shared_token_decoder,
)

from ._ditxa_helpers import np_

#: The test geometry: the ``tiny`` bridge preset (8 tokens x 32 dims = 256).
PRESET = BRIDGE_PRESETS["tiny"]
VOCAB_SIZE = 37
HIDDEN_DIM = 24


def build_decoder(**overrides) -> SharedTokenDecoder:
    """A built decoder over the ``tiny`` preset geometry."""
    kwargs = dict(
        vocab_size=VOCAB_SIZE,
        hidden_dim=HIDDEN_DIM,
        token_seq_len=PRESET.token_seq_len,
        token_emb_dim=PRESET.token_emb_dim,
    )
    kwargs.update(overrides)
    decoder = SharedTokenDecoder(**kwargs)
    decoder.build((None, kwargs["token_seq_len"] * kwargs["token_emb_dim"]))
    return decoder


def token_flat(batch_size: int = 3, seed: int = 7, scale: float = 1.0) -> np.ndarray:
    """A ``(B, token_flat_dim)`` float32 draw at the given absolute scale."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(batch_size, PRESET.token_flat_dim)).astype("float32")
    return (x * np.float32(scale)).astype("float32")


# ---------------------------------------------------------------------
# NumPy reference forward pass -- the instrument arms 1 uses
# ---------------------------------------------------------------------


def gelu_exact(x: np.ndarray) -> np.ndarray:
    """``0.5 * x * (1 + erf(x / sqrt(2)))`` -- ``torch.nn.GELU()``'s default."""
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    """The tanh approximation -- ``nn.GELU(approximate="tanh")``."""
    inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def numpy_forward(
    decoder: SharedTokenDecoder, x: np.ndarray, activation
) -> np.ndarray:
    """Recompute the decoder's forward pass in float64 with ``activation``.

    Reads the model's own weights, so the only free variable is which GELU is
    used. Returns ``(B, token_seq_len, vocab_size)``.
    """
    tokens = x.astype("float64").reshape(
        x.shape[0], decoder.token_seq_len, decoder.token_emb_dim
    )
    norm = np.sqrt(np.sum(tokens**2, axis=-1, keepdims=True))
    tokens = tokens / np.maximum(norm, decoder.normalize_epsilon)

    def dense(t, layer):
        out = t @ np_(layer.kernel).astype("float64")
        if layer.use_bias:
            out = out + np_(layer.bias).astype("float64")
        return out

    h = activation(dense(tokens, decoder.mlp_in))
    h = activation(dense(h, decoder.mlp_hidden))
    return dense(h, decoder.mlp_out)


def test_the_numpy_reference_can_tell_the_two_gelus_apart():
    """Anti-vacuity for arm 1: the two activations must actually differ here.

    Measured 2026-09-02 over ``linspace(-6, 6, 240001)``: the two formulas
    differ by at most **4.732e-04**, attained at ``x = -2.699`` -- and by only
    1.7e-05 at ``x = 0.5``. The separation is small and it is entirely in the
    tails, which is why arm 1 uses a NumPy rebuild at ``atol=2e-5`` rather than
    a coarse "the outputs differ" comparison, and why its inputs are unit-norm
    token rows fed through a Glorot-uniform layer (which reaches |x| ~ 2-3)
    rather than a toy grid clustered at zero.
    """
    grid = np.linspace(-6.0, 6.0, 24001)
    delta = np.abs(gelu_exact(grid) - gelu_tanh(grid))
    assert 4.0e-4 < delta.max() < 6.0e-4, (
        "the measured max separation of the two GELU formulas moved; arm 1's "
        f"2e-5 tolerance was chosen against it: max|delta| = {delta.max():.3e}"
    )
    assert abs(grid[int(delta.argmax())]) > 2.0, (
        "the separation is no longer in the tails; re-derive arm 1's inputs"
    )
    # And the SciPy-free erf reference must be the real thing.
    assert np.allclose(gelu_exact(np.array([0.0])), 0.0, atol=0, rtol=0)
    assert abs(float(gelu_exact(np.array([1.0]))[0]) - 0.8413447460685429) < 1e-12


def amplified_decoder() -> SharedTokenDecoder:
    """A decoder whose weights push the hidden pre-activations into the TAILS.

    The two GELU formulas separate by at most 4.732e-04, at ``x = -2.699``; at
    ``x = 0.5`` they differ by 1.7e-05. A Glorot-uniform decoder over unit-norm
    token rows only reaches ``|pre-activation| ~ 0.58`` (measured), i.e. the
    flattest part of the difference, and the final projection's mixed signs then
    cancel most of what is left. So this fixture does two things on purpose:
    it widens the first kernel (pre-activations span roughly ``[-8, 8]``) and
    makes the two later kernels ALL-POSITIVE, so the per-unit differences
    accumulate coherently instead of cancelling.

    Measured with these weights: ``max|float32 forward - exact-erf float64
    rebuild| = 2.2e-04`` against ``max|float32 forward - tanh float64 rebuild| =
    4.9e-02`` -- a 227x separation on logits of magnitude ~1050.
    """
    decoder = build_decoder()
    rng = np.random.default_rng(0)
    decoder.mlp_in.kernel.assign(
        rng.normal(scale=3.0, size=(PRESET.token_emb_dim, HIDDEN_DIM)).astype("float32")
    )
    decoder.mlp_hidden.kernel.assign(
        np.abs(rng.normal(size=(HIDDEN_DIM, HIDDEN_DIM))).astype("float32")
    )
    decoder.mlp_out.kernel.assign(
        np.abs(rng.normal(size=(HIDDEN_DIM, VOCAB_SIZE))).astype("float32")
    )
    return decoder


class TestTheActivationIsTheExactErfGelu:
    """The decoder's GELU is ``approximate='none'``; the block MLP's is not."""

    def test_the_module_constant_is_the_exact_erf_flag(self):
        """The single knob, pinned by name.

        ``GELU_APPROXIMATE is False`` is what ``call()`` passes to
        ``keras.ops.gelu``. This arm is the cheap tripwire; the value arms below
        are the ones that survive someone inlining the constant.
        """
        assert GELU_APPROXIMATE is False, (
            "the decoder uses torch's default approximate='none' (exact erf); "
            "the tanh approximation belongs to the DiTXA block MLP only"
        )

    def test_the_tail_reaching_forward_matches_only_the_exact_erf_rebuild(self):
        """The PRIMARY value arm, on weights that reach where the two differ.

        RED-proved by flipping ``GELU_APPROXIMATE`` to ``True`` in the real
        ``token_decoder.py``: this arm fails with
        ``max|got - exact| = 8.285e-02`` against a 1e-3 bound (the natural-init
        arm below reads 8.145e-06 against its 1e-6 bound in the same run).

        A natural Glorot init is NOT enough on its own -- see
        :func:`amplified_decoder` for the measured reason, and
        ``test_a_natural_init_forward_also_discriminates_but_only_barely``
        below for how thin that margin is.
        """
        decoder = amplified_decoder()
        x = token_flat()
        got = np_(decoder(keras.ops.convert_to_tensor(x), training=False)).astype(
            "float64"
        )
        exact = numpy_forward(decoder, x, gelu_exact)
        tanh = numpy_forward(decoder, x, gelu_tanh)

        exact_delta = float(np.max(np.abs(got - exact)))
        tanh_delta = float(np.max(np.abs(got - tanh)))
        assert exact_delta < 1e-3, (
            "the forward pass does not reproduce the exact-erf rebuild: "
            f"max|delta| = {exact_delta:.3e}"
        )
        assert tanh_delta > 100.0 * max(exact_delta, 1e-12), (
            "the tanh rebuild fits about as well as the exact one, so this arm "
            f"cannot discriminate: exact={exact_delta:.3e}, tanh={tanh_delta:.3e}"
        )

    def test_a_natural_init_forward_also_discriminates_but_only_barely(self):
        """Same claim at the shipped initialisation, with the margin recorded.

        Measured 2026-09-02 at Glorot init over unit-norm rows:
        ``exact = 3.39e-08``, ``tanh = 7.96e-06`` -- a 234x ratio, but both
        numbers are tiny, so the ``atol`` is 1e-6: thirty times above the
        float32 rebuild noise and eight times below the tanh signal. An
        ``atol`` of 2e-5 was tried first and was measured GREEN under the tanh
        injection -- a value arm that could not fail.
        """
        decoder = build_decoder()
        x = token_flat()
        got = np_(decoder(keras.ops.convert_to_tensor(x), training=False)).astype(
            "float64"
        )
        exact_delta = float(np.max(np.abs(got - numpy_forward(decoder, x, gelu_exact))))
        tanh_delta = float(np.max(np.abs(got - numpy_forward(decoder, x, gelu_tanh))))
        assert exact_delta < 1e-6, f"max|delta| = {exact_delta:.3e}"
        assert tanh_delta > 20.0 * max(exact_delta, 1e-12), (
            f"exact={exact_delta:.3e}, tanh={tanh_delta:.3e}"
        )


class TestScaleInvariance:
    """Per-token L2 normalization makes the absolute input scale irrelevant."""

    @pytest.mark.parametrize("k", [2.0, 0.125, 97.5])
    def test_logits_are_invariant_to_a_positive_rescaling(self, k: float):
        """``decode(k * x) == decode(x)`` for ``k > 0``.

        This is the operational meaning of upstream's "undo the dataest
        scaling" comment: the bridge carries embeddings multiplied by
        ``token_scale``, and the decoder must not care.
        """
        decoder = build_decoder()
        x = token_flat()
        base = np_(decoder(keras.ops.convert_to_tensor(x), training=False))
        scaled = np_(
            decoder(keras.ops.convert_to_tensor(x * np.float32(k)), training=False)
        )
        # Measured 2026-09-02: EXACTLY 0.0 for k = 2 and k = 0.125 (powers of
        # two rescale the norm exactly); 3.7e-08 for k = 97.5. The bound is
        # 1e-6, which is ~25x the largest measured deviation and ~4,000,000x
        # below the 2.6e-01 a sign flip produces.
        np.testing.assert_allclose(scaled, base, atol=1e-6, rtol=0)

    def test_the_arm_above_is_not_vacuous_because_the_logits_are_flat(self):
        """A constant output would satisfy every invariance arm for free."""
        decoder = build_decoder()
        logits = np_(decoder(keras.ops.convert_to_tensor(token_flat()), training=False))
        assert float(np.std(logits)) > 1e-3, (
            f"logits are near-constant (std={np.std(logits):.3e}); the "
            "invariance arms would be vacuously satisfied"
        )

    def test_a_sign_flip_is_not_invariant(self):
        """``k < 0`` is NOT covered: the claim is about POSITIVE rescalings.

        ``-x`` has the same norm, so the normalize passes the sign through and
        the MLP sees a genuinely different input. Pinning this stops the
        invariance claim being over-read as "the decoder ignores its input".
        """
        decoder = build_decoder()
        x = token_flat()
        base = np_(decoder(keras.ops.convert_to_tensor(x), training=False))
        flipped = np_(decoder(keras.ops.convert_to_tensor(-x), training=False))
        assert float(np.max(np.abs(flipped - base))) > 1e-3

    def test_a_zero_token_row_decodes_to_the_bias_path_and_never_to_nan(self):
        """The ``k = 0`` / padding case, stated and measured.

        ``keras.ops.normalize`` floors the norm at ``epsilon`` exactly the way
        ``torch.nn.functional.normalize`` does (``x / max(||x||, eps)``), so an
        all-zero token row stays an all-zero token row -- it has no direction to
        keep. The row therefore decodes to whatever the MLP maps the zero vector
        to (the bias path), finitely, and identically for every padding
        position. RED-proved by replacing the normalize with a plain
        ``x / ||x||``: every padding logit becomes ``nan``.

        The biases are deliberately randomised first. At initialisation they are
        ZERO, so the bias path is the zero vector and the "every padding token
        decodes alike" assertion would be comparing zeros with zeros -- true for
        a decoder that returned a constant, true for one that crashed the
        gradient, true for anything.
        """
        decoder = build_decoder()
        rng = np.random.default_rng(5)
        for layer in (decoder.mlp_in, decoder.mlp_hidden, decoder.mlp_out):
            layer.bias.assign(
                rng.normal(size=np_(layer.bias).shape).astype("float32")
            )

        x = token_flat()
        x[:, : PRESET.token_emb_dim] = 0.0  # token 0 of every row is padding
        x[1, :] = 0.0  # row 1 is entirely padding

        normalized = np_(decoder.normalize_tokens(keras.ops.convert_to_tensor(x)))
        assert np.all(np.isfinite(normalized))
        np.testing.assert_array_equal(normalized[:, 0, :], 0.0)
        assert np.any(normalized[0, 1, :] != 0.0), "the non-padding rows vanished too"

        logits = np_(decoder(keras.ops.convert_to_tensor(x), training=False))
        assert np.all(np.isfinite(logits)), "a padding token produced non-finite logits"
        assert np.any(logits[1] != 0.0), (
            "the bias path is the zero vector, so the equality below would be "
            "vacuous; the bias randomisation above did not take effect"
        )
        # Every all-zero token decodes to the same vector, whichever row it is in.
        np.testing.assert_allclose(
            logits[1], np.repeat(logits[0:1, 0], PRESET.token_seq_len, axis=0),
            atol=1e-6, rtol=0,
        )
        # ...and a NON-padding token decodes to something else entirely.
        assert float(np.max(np.abs(logits[0, 1] - logits[0, 0]))) > 1e-3

    def test_the_epsilon_default_is_the_torch_one(self):
        """``F.normalize``'s ``eps`` default, ported rather than invented."""
        assert DEFAULT_NORMALIZE_EPSILON == 1e-12
        assert build_decoder().normalize_epsilon == 1e-12


class TestShapeAndConstruction:
    """``(B, token_flat_dim)`` in -> ``(B, token_seq_len, vocab_size)`` out."""

    def test_the_output_shape(self):
        decoder = build_decoder()
        out = decoder(keras.ops.convert_to_tensor(token_flat(batch_size=5)))
        assert tuple(out.shape) == (5, PRESET.token_seq_len, VOCAB_SIZE)

    def test_compute_output_shape_agrees_without_building(self):
        """The static answer must match the traced one, on an UNBUILT model."""
        decoder = SharedTokenDecoder(
            vocab_size=VOCAB_SIZE,
            hidden_dim=HIDDEN_DIM,
            token_seq_len=PRESET.token_seq_len,
            token_emb_dim=PRESET.token_emb_dim,
        )
        assert not decoder.built
        assert decoder.compute_output_shape((None, PRESET.token_flat_dim)) == (
            None,
            PRESET.token_seq_len,
            VOCAB_SIZE,
        )

    def test_a_wrong_input_width_raises(self):
        decoder = SharedTokenDecoder(
            vocab_size=VOCAB_SIZE, token_seq_len=8, token_emb_dim=32
        )
        with pytest.raises(ValueError, match="does not match token_seq_len"):
            decoder.build((None, 255))

    def test_a_rank_three_input_raises(self):
        decoder = SharedTokenDecoder(
            vocab_size=VOCAB_SIZE, token_seq_len=8, token_emb_dim=32
        )
        with pytest.raises(ValueError, match="rank-2"):
            decoder.build((None, 8, 32))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"vocab_size": 0},
            {"vocab_size": 10, "hidden_dim": -1},
            {"vocab_size": 10, "token_seq_len": 0},
            {"vocab_size": 10, "token_emb_dim": 0},
        ],
    )
    def test_a_non_positive_size_raises(self, kwargs):
        with pytest.raises(ValueError, match="must be positive"):
            SharedTokenDecoder(**kwargs)

    def test_a_non_positive_epsilon_raises(self):
        with pytest.raises(ValueError, match="normalize_epsilon must be positive"):
            SharedTokenDecoder(vocab_size=10, normalize_epsilon=0.0)

    def test_the_factory_builds_the_same_thing(self):
        decoder = create_shared_token_decoder(
            vocab_size=VOCAB_SIZE,
            hidden_dim=HIDDEN_DIM,
            token_seq_len=PRESET.token_seq_len,
            token_emb_dim=PRESET.token_emb_dim,
        )
        assert isinstance(decoder, SharedTokenDecoder)
        assert decoder.token_flat_dim == PRESET.token_flat_dim

    def test_a_fresh_decoder_is_not_degenerate(self):
        """The step-7 hazard, checked rather than assumed.

        A fresh ``DiTXA`` predicts the EXACT zero tensor (zero-init adaLN gates
        x zero-init final projection), which vacuates output-level arms. This
        model's three kernels are Glorot-uniform, so it must not. If this arm
        ever fails, every value arm in this file needs an ``activate()`` helper.
        """
        decoder = build_decoder()
        logits = np_(decoder(keras.ops.convert_to_tensor(token_flat()), training=False))
        assert np.any(logits != 0.0), "a fresh decoder output the exact zero tensor"
        assert float(np.std(logits)) > 1e-3
        assert len(np.unique(logits.round(6))) > 10

    def test_the_head_is_the_three_layer_mlp_upstream_ships(self):
        """Depth and widths, pinned against ``reference/token_decoder.py:17-23``.

        ``Linear(D, hidden) -> Linear(hidden, hidden) -> Linear(hidden, vocab)``.
        A fourth layer, or a hidden layer of the wrong width, changes the
        parameter count and nothing else a value test would notice.
        """
        decoder = build_decoder()
        dense = [
            layer
            for layer in decoder._flatten_layers(include_self=False)
            if isinstance(layer, keras.layers.Dense)
        ]
        assert len(dense) == NUM_MLP_LAYERS == 3, [layer.name for layer in dense]
        assert np_(decoder.mlp_in.kernel).shape == (PRESET.token_emb_dim, HIDDEN_DIM)
        assert np_(decoder.mlp_hidden.kernel).shape == (HIDDEN_DIM, HIDDEN_DIM)
        assert np_(decoder.mlp_out.kernel).shape == (HIDDEN_DIM, VOCAB_SIZE)

    def test_no_two_dense_layers_share_an_initializer_instance(self):
        """Object identity, the only instrument that can see this.

        A shared ``Initializer`` instance draws bit-identically for every layer
        it is handed to whenever the shapes agree. Nothing else -- not a shape
        check, not a config round trip, not a seeded value test -- can see it.
        """
        decoder = build_decoder()
        layers = [decoder.mlp_in, decoder.mlp_hidden, decoder.mlp_out]
        inits = [layer.kernel_initializer for layer in layers] + [
            layer.bias_initializer for layer in layers
        ]
        for i in range(len(inits)):
            for j in range(i + 1, len(inits)):
                assert inits[i] is not inits[j], (
                    f"initializer instance shared between slots {i} and {j}"
                )


class TestSerialization:
    """Full ``.keras`` round trip, compared on VALUES."""

    def test_get_config_carries_every_constructor_argument(self):
        decoder = build_decoder(normalize_epsilon=1e-9, use_bias=False)
        config = decoder.get_config()
        for key in (
            "vocab_size",
            "hidden_dim",
            "token_seq_len",
            "token_emb_dim",
            "normalize_epsilon",
            "use_bias",
        ):
            assert key in config, f"get_config() dropped {key!r}"
        rebuilt = SharedTokenDecoder.from_config(config)
        assert rebuilt.get_config() == config

    def test_the_keras_round_trip_preserves_the_logits(self, tmp_path):
        """``atol=1e-6, rtol=0``, ``training=False`` explicit."""
        decoder = build_decoder()
        x = keras.ops.convert_to_tensor(token_flat(batch_size=2, seed=11))
        before = np_(decoder(x, training=False))

        path = tmp_path / "shared_token_decoder.keras"
        decoder.save(path)
        loaded = keras.models.load_model(path)
        after = np_(loaded(x, training=False))

        assert np.any(before != 0.0), "round trip compared two zero tensors"
        np.testing.assert_allclose(after, before, atol=1e-6, rtol=0)


class TestTheBridgeBijectionTie:
    """The decoder reads the reverse direction's OUTPUT, via step 2's packing."""

    def test_packing_and_unpacking_leaves_the_logits_untouched(self):
        """``decode(unpack(pack(x))) == decode(x)``.

        Ties this model to ``token_flat_to_bridge`` / ``bridge_to_token_flat``:
        the decoder may ignore the bridge layout only because that pair is an
        exact bijection. A layout change that broke the bijection would show up
        here as changed logits, not as a shape error.
        """
        decoder = build_decoder()
        x = keras.ops.convert_to_tensor(token_flat(batch_size=4, seed=3))

        direct = np_(decoder(x, training=False))
        bridge = token_flat_to_bridge(x, PRESET)
        assert tuple(bridge.shape)[1:] == PRESET.bridge_shape
        recovered = bridge_to_token_flat(bridge, PRESET)
        via_bridge = np_(decoder(recovered, training=False))

        np.testing.assert_allclose(np_(recovered), np_(x), atol=0, rtol=0)
        np.testing.assert_allclose(via_bridge, direct, atol=0, rtol=0)

    def test_the_tie_would_notice_a_broken_round_trip(self):
        """Anti-vacuity: a PERTURBED bridge must change the logits.

        Without this, the arm above would pass for a decoder that ignored its
        input entirely.
        """
        decoder = build_decoder()
        x = keras.ops.convert_to_tensor(token_flat(batch_size=4, seed=3))
        direct = np_(decoder(x, training=False))

        bridge = np_(token_flat_to_bridge(x, PRESET)).copy()
        bridge[:, 0, 0, 0] += 5.0
        perturbed = bridge_to_token_flat(keras.ops.convert_to_tensor(bridge), PRESET)
        moved = np_(decoder(perturbed, training=False))
        assert float(np.max(np.abs(moved - direct))) > 1e-4
