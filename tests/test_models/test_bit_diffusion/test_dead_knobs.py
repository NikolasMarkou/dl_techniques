"""Every constructor knob in this package is proven to reach something.

A ``test_different_X`` sweep that asserts only "the output shape is unchanged"
is the exact inversion of the claim it exists to make: the shape is identical
whether or not the kwarg ever reached the graph. So this file uses the shared
``tests/test_models/knob_sensitivity_oracle.py`` rather than hand-rolling a
sweep, and it classifies every knob first:

STRUCTURAL
    The knob changes the parameterisation, so the WEIGHT-SHAPE SIGNATURE must
    change. An output-difference assertion would be satisfied by the different
    random draw alone and would pass on a model that dropped the kwarg.

VALUE
    Same signature, different arithmetic. Under one seed the two models hold
    bit-identical weights, so an output difference is attributable to the knob.

SHAPE-ONLY
    ``grid_height`` / ``token_seq_len`` reach neither the weights nor the
    arithmetic -- they choose how a flat payload is folded. Their liveness claim
    is the OUTPUT SHAPE, asserted against an independently computed expectation.

TRAINING-MODE
    ``dropout_rate``, ``drop_path_rate``, ``class_dropout_rate``, ``seed``.
    These are inert at inference **by design** -- that is what dropout means --
    so their liveness is proven under ``training=True``, and their inference-time
    inertness is asserted as a positive claim rather than left as a hole.

**One measured trap, recorded because it defeats the obvious instrument.**
Constructing a ``Dropout`` at a non-zero rate CONSUMES process-global RNG, so
two ``DiTXA`` built at the same seed with ``dropout_rate=0.0`` and
``dropout_rate=0.5`` hold the SAME weight-shape signature and **12 of 54
bit-different weight tensors**. ``assert_value_knob_changes_output`` would pass
on that pair for a reason that has nothing to do with dropout -- its signature
pre-check cannot see a value difference -- and it would keep passing if dropout
were deleted from ``call()`` entirely. Every training-mode knob below is
therefore measured as ``training=True`` vs ``training=False`` **on one model**,
where the weights are not merely identical but the same objects.

**Findings, stated rather than waived** (see the report and decisions.md):

* ``FlowMatchingODE.force_unconditional`` is **INERT**. It is stored, it is
  serialized, and nothing reads it: upstream honours it inside a
  ``FlowMatchingODE.dX_t`` override that this port does not have. Pinned below
  by an ``xfail(strict=True)``, so wiring it turns the arm red and forces the
  marker off.
* ``drop_path_rate`` measures ``9.5e-07`` at inference rather than exactly 0.
  That is not the knob: it is float32 re-association between ``block(x)`` and
  ``x + (block(x) - x)``. The training-mode arm measures ``3.47``.
"""

import ast
import inspect
from pathlib import Path

import keras
import numpy as np
import pytest

from dl_techniques.models.vision_language import bit_diffusion as dl_bit_diffusion
from dl_techniques.layers.embedding.class_label_embedding import (
    ClassLabelEmbedding,
)
from dl_techniques.models.vision_language.bit_diffusion.blocks import (
    DiTXABlock,
    DiTXATimestepEmbedder,
)
from dl_techniques.models.vision_language.bit_diffusion.model import (
    DiTXA,
    DiTXAFinalLayer,
)
from dl_techniques.models.vision_language.bit_diffusion.sde import (
    BridgeSDE,
    CosineDecayingVolatilitySDE,
    FlowMatchingODE,
    PeriodicVolatilitySDE,
    UniformVolatilitySDE,
)
from dl_techniques.models.vision_language.bit_diffusion.token_decoder import (
    SharedTokenDecoder,
)

from ..knob_sensitivity_oracle import (
    assert_structural_knob_changes_weights,
    assert_value_knob_changes_output,
    weight_signature,
)
from ._ditxa_helpers import activate, np_

BATCH = 3
TOKENS = 9
HIDDEN = 32
HEADS = 4

#: A small but non-degenerate DiTXA. Every knob sweep starts from this.
BASE = dict(
    input_size=8,
    patch_size=2,
    in_channels=4,
    hidden_size=64,
    depth=2,
    num_heads=4,
    num_classes=4,
    frequency_embedding_size=16,
)


def ditxa_inputs(config, seed=7):
    """A deterministic input dict matching ``config``'s geometry."""
    rng = np.random.default_rng(seed)
    n, c = config["input_size"], config["in_channels"]
    return {
        "x_t": rng.normal(size=(BATCH, n, n, c)).astype("float32"),
        "t": rng.uniform(0.1, 0.9, size=(BATCH,)).astype("float32"),
        "y": rng.integers(0, config["num_classes"], size=(BATCH,)).astype("int32"),
        "x_cond": rng.normal(size=(BATCH, n, n, c)).astype("float32"),
        "direction": np.array([0.0, 1.0, 0.0], dtype="float32"),
    }


def make_ditxa(**overrides):
    """A BUILT, ACTIVATED ``DiTXA``.

    ``activate`` is not optional here. A fresh ``DiTXA`` emits the EXACT zero
    tensor (zero-init adaLN gates times a zero-init final projection), so every
    value-knob comparison would be ``0 - 0`` and every one of them would report
    the knob dead. The replacement draws come from a fixed NumPy generator, so
    two configurations with the same signature stay bit-identical after it.
    """
    config = dict(BASE)
    config.update(overrides)
    # Seeded HERE, not left to the caller. Without it two configurations draw
    # different random numbers and every "the knob changed the output" reading
    # below would be attributable to the draw instead of to the knob. The
    # oracle seeds too; a builder that also seeds is idempotent, a builder that
    # does not is at the mercy of collection order.
    keras.utils.set_random_seed(1234)
    model = DiTXA(**config)
    model(ditxa_inputs(config))
    return activate(model, seed=5)


def ditxa_builder(**overrides):
    """A zero-argument builder for the oracle."""
    return lambda: make_ditxa(**overrides)


FIXED_INPUTS = ditxa_inputs(BASE)


def assert_not_degenerate(array, label):
    """A knob comparison over the zero tensor proves nothing."""
    assert np.any(np.asarray(array) != 0.0), (
        f"{label}: the reference output is identically zero, so any 'the knob "
        "changed the output' claim below is vacuous"
    )


# ---------------------------------------------------------------------
# DiTXA
# ---------------------------------------------------------------------


class TestDiTXAStructuralKnobs:
    """Pinned on the weight-shape signature, which RNG luck cannot fake."""

    @pytest.mark.parametrize(
        "knob, values",
        [
            ("input_size", [8, 16]),
            ("patch_size", [2, 4]),
            ("in_channels", [4, 6]),
            ("hidden_size", [64, 128]),
            ("depth", [2, 3]),
            ("mlp_ratio", [4.0, 2.0]),
            ("num_classes", [4, 9]),
            ("use_bias", [True, False]),
            ("frequency_embedding_size", [16, 32]),
            # 0.0 removes the extra unconditional row that CFG needs, so this
            # knob is structural at the 0/non-0 boundary and training-mode
            # everywhere else. Both halves are covered.
            ("class_dropout_rate", [0.0, 0.2]),
        ],
    )
    def test_the_knob_changes_the_weight_shape_signature(self, knob, values):
        assert_structural_knob_changes_weights(
            {v: ditxa_builder(**{knob: v}) for v in values}, knob=knob
        )

    def test_depth_grows_the_parameter_count_monotonically(self):
        """A stronger claim than "the signature changed"."""
        totals = []
        for depth in (1, 2, 3):
            model = make_ditxa(depth=depth)
            totals.append(model.count_params())
            assert len(model.blocks) == depth
        assert totals[0] < totals[1] < totals[2], totals


class TestDiTXAValueKnobs:
    """Same signature, same seed, therefore bit-identical weights."""

    @pytest.mark.parametrize(
        "knob, values, atol",
        [
            ("num_heads", [4, 8], 1e-3),
            ("forward_cond_scale", [1.0, 3.0], 1e-3),
            ("time_scale", [1000.0, 250.0], 1e-3),
            ("norm_epsilon", [1e-6, 0.5], 1e-3),
            ("qk_norm_epsilon", [1e-6, 0.5], 1e-3),
        ],
    )
    def test_the_knob_changes_the_output(self, knob, values, atol):
        assert_not_degenerate(
            np_(make_ditxa()(FIXED_INPUTS, training=False)), knob
        )
        deltas = assert_value_knob_changes_output(
            {v: ditxa_builder(**{knob: v}) for v in values},
            FIXED_INPUTS,
            knob=knob,
            atol=atol,
            extract=lambda out: out,
        )
        # The measured deltas are 4 to 6 orders of magnitude above the bound;
        # if one ever lands near it, that is a signal, not a reason to widen.
        assert min(deltas.values()) > 10 * atol, deltas

    def test_norm_epsilon_reaches_the_sub_layers_and_not_just_the_attribute(self):
        """``assert model.norm_epsilon == x`` is a knob ECHO, not a test."""
        model = make_ditxa(norm_epsilon=0.25)
        reached = [
            layer.epsilon
            for layer in model._flatten_layers(include_self=False)
            if type(layer).__name__ == "LayerNormalization"
        ]
        assert reached, "no LayerNormalization found; the walk is broken"
        assert set(reached) == {0.25}, sorted(set(reached))


class TestTheTrainingModeKnobs:
    """Inert at inference BY DESIGN; live under ``training=True``.

    Measured on ONE model, ``training=True`` against ``training=False``, so the
    two readings share not merely equal weights but the same weight objects.
    Two separately constructed models would not do: building a ``Dropout`` at a
    non-zero rate consumes process-global RNG and leaves 12 of 54 weight
    tensors bit-different at the same seed.
    """

    def _both_modes(self, **overrides):
        model = make_ditxa(**overrides)
        keras.utils.set_random_seed(99)
        train = np_(model(FIXED_INPUTS, training=True))
        infer = np_(model(FIXED_INPUTS, training=False))
        return train, infer

    @pytest.mark.parametrize(
        "overrides, floor",
        [
            ({"dropout_rate": 0.5}, 1e-2),
            ({"drop_path_rate": 0.5}, 1e-2),
            ({"class_dropout_rate": 0.9}, 1e-2),
        ],
    )
    def test_the_knob_changes_the_output_under_training(self, overrides, floor):
        train, infer = self._both_modes(**overrides)
        assert_not_degenerate(infer, str(overrides))
        delta = float(np.max(np.abs(train - infer)))
        assert delta > floor, (
            f"{overrides} left training=True and training=False agreeing to "
            f"max|delta| = {delta:.3e}; the knob does not reach call()"
        )

    def test_at_rate_zero_training_changes_nothing(self):
        """The control. Without it the arms above could be measuring anything.

        With every stochastic rate at zero, ``training=True`` and
        ``training=False`` must produce the SAME numbers -- so the deltas above
        are attributable to the rates and not to some other training-mode
        branch in the graph.
        """
        train, infer = self._both_modes(
            dropout_rate=0.0, drop_path_rate=0.0, class_dropout_rate=0.0
        )
        np.testing.assert_allclose(train, infer, atol=0, rtol=0)

    def test_label_seed_changes_which_labels_are_dropped(self):
        """Two seeds, bit-identical weights (measured), different CFG draws."""
        a = make_ditxa(class_dropout_rate=0.5, label_seed=1)
        b = make_ditxa(class_dropout_rate=0.5, label_seed=999)
        assert weight_signature(a) == weight_signature(b)
        same = [
            np.array_equal(np_(x), np_(y)) for x, y in zip(a.weights, b.weights)
        ]
        assert all(same), (
            "label_seed changed the initial WEIGHTS, so the output difference "
            "below would not be attributable to the dropout draw"
        )
        out_a = np_(a(FIXED_INPUTS, training=True))
        out_b = np_(b(FIXED_INPUTS, training=True))
        assert float(np.max(np.abs(out_a - out_b))) > 1e-3

    def test_drop_path_rate_is_inert_at_inference_to_float_noise(self):
        """The 9.5e-07 reading, recorded so it is not mistaken for liveness.

        ``StochasticDepth`` is the identity at inference, but the model computes
        ``x + drop_path(block(x) - x)`` when a rate is set and ``block(x)``
        when it is not. Those are the same value in exact arithmetic and differ
        by float32 re-association. The bound is set from the measurement
        (9.5e-07), not chosen for comfort.
        """
        off = np_(make_ditxa(drop_path_rate=0.0)(FIXED_INPUTS, training=False))
        on = np_(make_ditxa(drop_path_rate=0.5)(FIXED_INPUTS, training=False))
        assert_not_degenerate(off, "drop_path_rate")
        delta = float(np.max(np.abs(on - off)))
        assert delta < 1e-5, (
            f"drop_path_rate moved the INFERENCE output by {delta:.3e}; "
            "stochastic depth must be the identity when not training"
        )
        assert float(np.max(np.abs(off))) > 0.1, "the comparison is near zero"


# ---------------------------------------------------------------------
# The sub-layers
# ---------------------------------------------------------------------


def block_inputs(hidden=HIDDEN, tokens=TOKENS, seed=1):
    rng = np.random.default_rng(seed)
    return [
        rng.normal(size=(BATCH, tokens, hidden)).astype("float32"),
        rng.normal(size=(BATCH, hidden)).astype("float32"),
        rng.normal(size=(BATCH, tokens, hidden)).astype("float32"),
    ]


def make_block(**overrides):
    config = dict(hidden_size=HIDDEN, num_heads=HEADS)
    config.update(overrides)
    keras.utils.set_random_seed(1234)
    layer = DiTXABlock(**config)
    tensors = block_inputs(hidden=config["hidden_size"])
    layer(tensors)
    return activate(layer, seed=5)


class TestDiTXABlockKnobs:
    """The block is where the 12-way modulation lives; every knob must reach it."""

    @pytest.mark.parametrize(
        "knob, values",
        [
            ("hidden_size", [HIDDEN, 2 * HIDDEN]),
            ("mlp_ratio", [4.0, 2.0]),
            ("use_bias", [True, False]),
        ],
    )
    def test_structural(self, knob, values):
        assert_structural_knob_changes_weights(
            {v: (lambda v=v: make_block(**{knob: v})) for v in values}, knob=knob
        )

    @pytest.mark.parametrize(
        "knob, values",
        [
            ("num_heads", [4, 8]),
            ("norm_epsilon", [1e-6, 0.5]),
            ("qk_norm_epsilon", [1e-6, 0.5]),
        ],
    )
    def test_value(self, knob, values):
        assert_value_knob_changes_output(
            {v: (lambda v=v: make_block(**{knob: v})) for v in values},
            block_inputs(),
            knob=knob,
            atol=1e-4,
        )

    def test_dropout_rate_is_live_under_training(self):
        layer = make_block(dropout_rate=0.5)
        tensors = block_inputs()
        keras.utils.set_random_seed(4)
        train = np_(layer(tensors, training=True))
        infer = np_(layer(tensors, training=False))
        assert float(np.max(np.abs(train - infer))) > 1e-3

    def test_dropout_rate_zero_is_inert_across_training(self):
        layer = make_block(dropout_rate=0.0)
        tensors = block_inputs()
        train = np_(layer(tensors, training=True))
        infer = np_(layer(tensors, training=False))
        np.testing.assert_allclose(train, infer, atol=0, rtol=0)


def make_final_layer(**overrides):
    config = dict(
        hidden_size=HIDDEN,
        patch_size=2,
        out_channels=4,
        grid_height=3,
        grid_width=3,
    )
    config.update(overrides)
    keras.utils.set_random_seed(1234)
    layer = DiTXAFinalLayer(**config)
    tokens = config["grid_height"] * config["grid_width"]
    rng = np.random.default_rng(2)
    layer(
        [
            rng.normal(size=(BATCH, tokens, config["hidden_size"])).astype("float32"),
            rng.normal(size=(BATCH, config["hidden_size"])).astype("float32"),
        ]
    )
    return activate(layer, seed=5)


def final_layer_inputs(hidden=HIDDEN, tokens=9, seed=2):
    rng = np.random.default_rng(seed)
    return [
        rng.normal(size=(BATCH, tokens, hidden)).astype("float32"),
        rng.normal(size=(BATCH, hidden)).astype("float32"),
    ]


class TestDiTXAFinalLayerKnobs:
    @pytest.mark.parametrize(
        "knob, values",
        [
            ("hidden_size", [HIDDEN, 2 * HIDDEN]),
            ("patch_size", [2, 3]),
            ("out_channels", [4, 6]),
        ],
    )
    def test_structural(self, knob, values):
        assert_structural_knob_changes_weights(
            {v: (lambda v=v: make_final_layer(**{knob: v})) for v in values},
            knob=knob,
        )

    def test_norm_epsilon_is_a_live_value_knob(self):
        assert_value_knob_changes_output(
            {
                eps: (lambda eps=eps: make_final_layer(norm_epsilon=eps))
                for eps in (1e-6, 0.5)
            },
            final_layer_inputs(),
            knob="norm_epsilon",
            atol=1e-4,
        )

    @pytest.mark.parametrize("knob", ["grid_height", "grid_width"])
    def test_the_grid_knobs_choose_the_output_geometry(self, knob):
        """SHAPE-ONLY: they reach neither the weights nor the arithmetic.

        ``unpatchify`` folds ``(B, h*w, p*p*C)`` into ``(B, h*p, w*p, C)``. The
        expectation is computed here from the knob, not read back from the
        layer, so a layer that ignored the knob would be convicted.
        """
        shapes = {}
        for value in (2, 3):
            layer = make_final_layer(**{knob: value})
            tokens = layer.grid_height * layer.grid_width
            out = layer(final_layer_inputs(tokens=tokens))
            shapes[value] = tuple(out.shape)
            expected_h = layer.grid_height * layer.patch_size
            expected_w = layer.grid_width * layer.patch_size
            assert shapes[value] == (BATCH, expected_h, expected_w, 4)
        assert shapes[2] != shapes[3], shapes


def make_embedder(**overrides):
    config = dict(hidden_size=HIDDEN, frequency_embedding_size=16)
    config.update(overrides)
    keras.utils.set_random_seed(1234)
    layer = DiTXATimestepEmbedder(**config)
    layer(np.linspace(0.0, 900.0, BATCH).astype("float32"))
    return layer


EMBEDDER_INPUT = np.linspace(0.0, 900.0, BATCH).astype("float32")


class TestTimestepEmbedderKnobs:
    @pytest.mark.parametrize(
        "knob, values",
        [
            ("hidden_size", [HIDDEN, 2 * HIDDEN]),
            ("frequency_embedding_size", [16, 32]),
        ],
    )
    def test_structural(self, knob, values):
        assert_structural_knob_changes_weights(
            {v: (lambda v=v: make_embedder(**{knob: v})) for v in values},
            knob=knob,
        )

    @pytest.mark.parametrize(
        "knob, values",
        [
            ("max_period", [10000.0, 100.0]),
            # A different init STDDEV gives different weight VALUES at the same
            # shapes; that IS the knob's effect, and the oracle's signature
            # pre-check still holds.
            ("kernel_stddev", [0.02, 0.5]),
        ],
    )
    def test_value(self, knob, values):
        assert_value_knob_changes_output(
            {v: (lambda v=v: make_embedder(**{knob: v})) for v in values},
            EMBEDDER_INPUT,
            knob=knob,
            atol=1e-5,
        )


def make_decoder(**overrides):
    config = dict(
        vocab_size=17, hidden_dim=24, token_seq_len=4, token_emb_dim=8
    )
    config.update(overrides)
    keras.utils.set_random_seed(1234)
    decoder = SharedTokenDecoder(**config)
    width = config["token_seq_len"] * config["token_emb_dim"]
    decoder(np.random.default_rng(6).normal(size=(BATCH, width)).astype("float32"))
    return decoder


DECODER_INPUT = np.random.default_rng(6).normal(size=(BATCH, 32)).astype("float32")


class TestSharedTokenDecoderKnobs:
    @pytest.mark.parametrize(
        "knob, values",
        [
            ("vocab_size", [17, 29]),
            ("hidden_dim", [24, 48]),
            ("token_emb_dim", [8, 16]),
            ("use_bias", [True, False]),
        ],
    )
    def test_structural(self, knob, values):
        assert_structural_knob_changes_weights(
            {v: (lambda v=v: make_decoder(**{knob: v})) for v in values},
            knob=knob,
        )

    def test_normalize_epsilon_is_a_live_value_knob(self):
        """Measured against the op's own algebra, not guessed.

        ``keras.ops.normalize(order=2)`` computes
        ``x * minimum(rsqrt(sum_sq), 1 / epsilon)``, so the epsilon only bites
        once ``1 / epsilon`` drops below the row's inverse norm. ``10.0`` is
        chosen to be firmly inside that regime; ``1e-12`` (the shipped default)
        is firmly outside it.
        """
        assert_value_knob_changes_output(
            {
                eps: (lambda eps=eps: make_decoder(normalize_epsilon=eps))
                for eps in (1e-12, 10.0)
            },
            DECODER_INPUT,
            knob="normalize_epsilon",
            atol=1e-4,
        )

    def test_token_seq_len_chooses_the_output_geometry(self):
        """SHAPE-ONLY, with ``token_emb_dim`` held so the input width is fixed."""
        shapes = {}
        for seq in (4, 8):
            decoder = make_decoder(token_seq_len=seq, token_emb_dim=8)
            width = seq * 8
            out = decoder(
                np.random.default_rng(6).normal(size=(BATCH, width)).astype("float32")
            )
            shapes[seq] = tuple(out.shape)
            assert shapes[seq] == (BATCH, seq, 17)
        assert shapes[4] != shapes[8], shapes


def make_label_embedding(**overrides):
    config = dict(num_classes=5, hidden_size=HIDDEN, dropout_rate=0.0)
    config.update(overrides)
    keras.utils.set_random_seed(1234)
    layer = ClassLabelEmbedding(**config)
    layer(np.zeros((BATCH,), dtype="int32"))
    return layer


LABELS = np.array([0, 1, 2], dtype="int32")


class TestClassLabelEmbeddingKnobs:
    @pytest.mark.parametrize(
        "knob, values",
        [
            ("num_classes", [5, 9]),
            ("hidden_size", [HIDDEN, 2 * HIDDEN]),
            # 0 -> non-zero adds the extra unconditional row.
            ("dropout_rate", [0.0, 0.3]),
        ],
    )
    def test_structural(self, knob, values):
        assert_structural_knob_changes_weights(
            {v: (lambda v=v: make_label_embedding(**{knob: v})) for v in values},
            knob=knob,
        )

    def test_embeddings_initializer_reaches_the_table(self):
        assert_value_knob_changes_output(
            {
                name: (
                    lambda name=name: make_label_embedding(
                        embeddings_initializer=name
                    )
                )
                for name in ("zeros", "ones")
            },
            LABELS,
            knob="embeddings_initializer",
            atol=1e-6,
        )

    def test_seed_changes_which_labels_are_dropped(self):
        a = make_label_embedding(dropout_rate=0.5, seed=1)
        b = make_label_embedding(dropout_rate=0.5, seed=999)
        labels = np.arange(5, dtype="int32")
        keras.utils.set_random_seed(0)
        out_a = np_(a(labels, training=True))
        keras.utils.set_random_seed(0)
        out_b = np_(b(labels, training=True))
        assert float(np.max(np.abs(out_a - out_b))) > 0.0, (
            "two different dropout seeds dropped exactly the same labels; the "
            "seed does not reach the RNG"
        )


# ---------------------------------------------------------------------
# The SDE classes -- pure math objects, so the oracle does not apply
# ---------------------------------------------------------------------

TIMES = np.array([0.0, 0.25, 0.6, 1.0], dtype="float64")


def spread(values):
    """``max|a - b|`` over an adjacent pair of arrays."""
    return float(np.max(np.abs(np.asarray(values[0]) - np.asarray(values[1]))))


class TestTheSDEKnobs:
    """``sigma`` / ``phi`` / ``C`` are the observable surface here."""

    def test_uniform_K_reaches_sigma_and_C(self):
        a, b = UniformVolatilitySDE(K=1.0), UniformVolatilitySDE(K=2.5)
        assert spread([a.sigma(TIMES), b.sigma(TIMES)]) > 1.0
        assert spread([a.C(0.0, TIMES, TIMES), b.C(0.0, TIMES, TIMES)]) > 1.0

    def test_uniform_A_reaches_phi_and_C(self):
        """The OU branch. Three of the four variants have ``A == 0``, so a
        parameter list without a non-zero ``A`` member is structurally blind
        to the entire Ornstein-Uhlenbeck code path."""
        a, b = UniformVolatilitySDE(A=0.0), UniformVolatilitySDE(A=1.5)
        assert spread([a.phi(0.0, TIMES), b.phi(0.0, TIMES)]) > 0.5
        assert spread([a.C(0.0, TIMES, TIMES), b.C(0.0, TIMES, TIMES)]) > 0.1
        # And the base class stores it.
        assert BridgeSDE(A=1.5).A == 1.5

    @pytest.mark.parametrize(
        "knob, values",
        [("alpha", [0.95, 0.2]), ("k", [1.0, 3.0]), ("eps", [0.05, 0.4])],
    )
    def test_periodic_knobs_reach_sigma_and_C(self, knob, values):
        made = [PeriodicVolatilitySDE(**{knob: v}) for v in values]
        assert spread([m.sigma(TIMES) for m in made]) > 1e-3, knob
        assert spread([m.C(0.0, TIMES, TIMES) for m in made]) > 1e-3, knob

    @pytest.mark.parametrize("knob, values", [("alpha", [0.95, 0.2]), ("eps", [0.05, 0.4])])
    def test_cosine_decay_knobs_reach_sigma_and_C(self, knob, values):
        made = [CosineDecayingVolatilitySDE(**{knob: v}) for v in values]
        assert spread([m.sigma(TIMES) for m in made]) > 1e-3, knob
        assert spread([m.C(0.0, TIMES, TIMES) for m in made]) > 1e-3, knob

    def test_cosine_decay_does_not_expose_k_as_a_knob(self):
        """``k = 0.5`` defines the variant; it is deliberately not a knob."""
        with pytest.raises(TypeError):
            CosineDecayingVolatilitySDE(k=2.0)
        assert "k" not in CosineDecayingVolatilitySDE().get_config()

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "MEASURED DEAD KNOB. FlowMatchingODE.force_unconditional is stored "
            "and serialized and read by nothing: upstream honours it inside a "
            "FlowMatchingODE.dX_t override (reference/sde_utils_sde.py:71-76) "
            "that passes an all-zero cond_mask and rejects cfg_scale != 0, and "
            "this port has no such override -- the inherited BridgeSDE.dX_t "
            "calls self.sigma(t), which FlowMatchingODE raises on, so the "
            "variant cannot be SAMPLED at all. strict=True on purpose: the "
            "moment someone wires the knob, this arm XPASSes and turns the "
            "suite red, which is the prompt to delete this marker and write a "
            "real guard."
        ),
    )
    def test_force_unconditional_changes_sampling_behaviour(self):
        """Is the knob READ anywhere in the package outside store/serialize?

        Scanned over the whole package directory rather than over
        ``FlowMatchingODE``'s own methods: the honouring code could legitimately
        live in ``BridgeSDE.dX_t``, in ``simulate``, or in ``bridge_process``,
        and a predicate that looked only at the class would stay red after a
        correct fix -- an arm that cannot go green is as useless as one that
        cannot go red.
        """
        package = (
            Path(dl_bit_diffusion.__file__).parent
            if hasattr(dl_bit_diffusion, "__file__")
            else None
        )
        readers = []
        for module in sorted(package.glob("*.py")):
            tree = ast.parse(module.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue
                if node.name in ("__init__", "get_config"):
                    continue
                for inner in ast.walk(node):
                    if (
                        isinstance(inner, ast.Attribute)
                        and inner.attr == "force_unconditional"
                    ):
                        readers.append(f"{module.name}:{node.name}")
        assert readers, (
            "nothing in the bit_diffusion package reads force_unconditional "
            "outside __init__/get_config, so the knob changes no behaviour"
        )

    def test_the_dead_knob_predicate_can_find_a_reader(self):
        """Control for the arm above: the AST scan must not be blind.

        The predicate is run against upstream's shape -- a ``dX_t`` override
        that reads ``self.force_unconditional`` (reference/sde_utils_sde.py:71)
        -- and must report it. Without this, "no readers found" would be
        indistinguishable from "the scan looks for the wrong node type".
        """
        source = (
            "class FlowMatchingODE:\n"
            "    def __init__(self, force_unconditional=False):\n"
            "        self.force_unconditional = force_unconditional\n"
            "    def dX_t(self, x_t):\n"
            "        if self.force_unconditional:\n"
            "            return 0\n"
            "        return 1\n"
        )
        readers = [
            node.name
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef)
            and node.name not in ("__init__", "get_config")
            and any(
                isinstance(inner, ast.Attribute)
                and inner.attr == "force_unconditional"
                for inner in ast.walk(node)
            )
        ]
        assert readers == ["dX_t"], readers

    def test_force_unconditional_is_at_least_carried_through_the_config(self):
        """What the knob DOES do today, pinned so the gap is documented."""
        assert FlowMatchingODE(force_unconditional=True).get_config()[
            "force_unconditional"
        ] is True
        assert FlowMatchingODE().get_config()["force_unconditional"] is False


# ---------------------------------------------------------------------
# The census -- a knob added later must be covered, not silently skipped
# ---------------------------------------------------------------------

#: Every constructor knob this file makes a liveness claim about, per class.
COVERED = {
    DiTXA: {
        "input_size", "patch_size", "in_channels", "hidden_size", "depth",
        "num_heads", "mlp_ratio", "num_classes", "class_dropout_rate",
        "forward_cond_scale", "time_scale", "drop_path_rate", "dropout_rate",
        "norm_epsilon", "qk_norm_epsilon", "use_bias",
        "frequency_embedding_size", "label_seed",
    },
    DiTXABlock: {
        "hidden_size", "num_heads", "mlp_ratio", "norm_epsilon",
        "qk_norm_epsilon", "dropout_rate", "use_bias",
    },
    DiTXAFinalLayer: {
        "hidden_size", "patch_size", "out_channels", "grid_height",
        "grid_width", "norm_epsilon",
    },
    DiTXATimestepEmbedder: {
        "hidden_size", "frequency_embedding_size", "max_period", "kernel_stddev",
    },
    SharedTokenDecoder: {
        "vocab_size", "hidden_dim", "token_seq_len", "token_emb_dim",
        "normalize_epsilon", "use_bias",
    },
    ClassLabelEmbedding: {
        "num_classes", "hidden_size", "dropout_rate", "embeddings_initializer",
        "seed",
    },
    UniformVolatilitySDE: {"A", "K"},
    PeriodicVolatilitySDE: {"alpha", "k", "eps"},
    CosineDecayingVolatilitySDE: {"alpha", "eps"},
    FlowMatchingODE: {"force_unconditional"},
}


@pytest.mark.parametrize("cls", list(COVERED), ids=lambda c: c.__name__)
def test_every_constructor_knob_of_every_class_is_covered(cls):
    """Executable census. A knob added tomorrow reddens this, not nothing.

    Derived from ``inspect.signature`` rather than from a hand-written list, so
    it cannot drift the way a prose follow-up does.
    """
    declared = {
        name
        for name, parameter in inspect.signature(cls.__init__).parameters.items()
        if name not in ("self", "kwargs")
        and parameter.kind is not inspect.Parameter.VAR_KEYWORD
    }
    uncovered = sorted(declared - COVERED[cls])
    assert not uncovered, (
        f"{cls.__name__} declares constructor knob(s) {uncovered} that no arm "
        "in this file makes a liveness claim about. A knob that changes "
        "nothing is a defect; add the arm, do not extend COVERED."
    )
    stale = sorted(COVERED[cls] - declared)
    assert not stale, (
        f"COVERED[{cls.__name__}] names {stale}, which the constructor no "
        "longer declares"
    )
