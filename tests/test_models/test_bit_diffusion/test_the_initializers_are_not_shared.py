"""No ``Initializer`` INSTANCE is shared, and every norm epsilon is ``1e-6``.

Invariants 9 and 10 of the plan. Two structural censuses that no value-level
test can perform, plus the one initializer in this model whose Keras default is
quietly the wrong distribution.

**Why sharing an instance matters, and why nothing else sees it.** A Keras
``Initializer`` is a stateless callable in appearance only: a single instance
handed to N layers draws the SAME numbers for every one of them whenever the
shapes agree. The XL variant stacks 28 blocks, each with a zero-init
``Dense(12 * hidden)``; sharing there is invisible today because zeros are zeros,
and becomes a silent rank collapse the day someone changes that initializer to
anything else. No shape assertion, no config round trip, no gradient check and no
seeded value test can see it -- a seeded test passes both ways. Only object
identity can, which is what the first class asserts.

**Why the epsilon census is a census and not a spot check.** Bare Keras
``LayerNormalization`` defaults to ``epsilon=1e-3``; upstream is ``1e-6``. That
is a 1000x error with no shape symptom, no NaN and no failing gradient -- it just
trains slightly differently forever. A spot check on one norm would miss the
other sixteen, so the arm below enumerates every normalization sub-layer in the
built model and pins the population size as well as the values.
"""

import math

import keras
import numpy as np
import pytest

from dl_techniques.models.vision_language.bit_diffusion.model import (
    DiTXA,
    flattened_linear_xavier,
)

from ._ditxa_helpers import batch, np_

#: The one epsilon this package uses, everywhere, on purpose.
EPSILON = 1e-6


@pytest.fixture(scope="module")
def model() -> DiTXA:
    """A built ``tiny`` model. No activation needed: both arms are structural."""
    m = DiTXA.from_variant("tiny")
    m(batch(m, batch_size=2))
    return m


def _walk(layer, seen=None):
    """Yield ``layer`` and every sub-layer reachable from it, once each."""
    seen = set() if seen is None else seen
    if id(layer) in seen:
        return
    seen.add(id(layer))
    yield layer
    for child in getattr(layer, "_layers", []) or []:
        yield from _walk(child, seen)


def _initializer_slots(model: DiTXA):
    """``(label, initializer)`` for every slot this port owns.

    Deliberately NOT a tree-wide sweep. Reused layers such as
    ``MultiHeadCrossAttention`` legitimately pass ONE ``kernel_initializer``
    instance to their own several ``Dense`` sub-layers -- a within-layer decision
    that is out of this port's scope and is pinned by that layer's own suite. The
    population here is the slots ``model.py`` and ``blocks.py`` construct.
    """
    slots = []
    for i, block in enumerate(model.blocks):
        slots.append((f"block_{i}.adaln.kernel", block.adaln_dense.kernel_initializer))
        slots.append((f"block_{i}.adaln.bias", block.adaln_dense.bias_initializer))
    slots += [
        ("final.adaln.kernel", model.final_layer.adaln_dense.kernel_initializer),
        ("final.adaln.bias", model.final_layer.adaln_dense.bias_initializer),
        ("final.linear.kernel", model.final_layer.linear.kernel_initializer),
        ("final.linear.bias", model.final_layer.linear.bias_initializer),
    ]
    for name in ("x_embedder", "cond_embedder_forward", "cond_embedder_reverse"):
        embedder = getattr(model, name)
        slots.append((f"{name}.kernel", embedder.kernel_initializer))
        slots.append((f"{name}.bias", embedder.bias_initializer))
    for name in ("t_embedder", "cond_t_embedder"):
        embedder = getattr(model, name)
        slots.append((f"{name}.mlp_in.kernel", embedder.mlp_in.kernel_initializer))
        slots.append((f"{name}.mlp_out.kernel", embedder.mlp_out.kernel_initializer))
    slots.append(("y_embedder.table", model.y_embedder.embeddings_initializer))
    return slots


class TestNoInitializerInstanceIsShared:
    """Object identity, pair by pair."""

    def test_every_slot_holds_its_own_instance(self, model):
        """``is not`` over every pair of the port's own initializer slots."""
        slots = _initializer_slots(model)
        by_id = {}
        collisions = []
        for label, initializer in slots:
            key = id(initializer)
            if key in by_id:
                collisions.append(f"{by_id[key]} is {label} ({initializer})")
            else:
                by_id[key] = label
        assert not collisions, (
            "these initializer slots hold the SAME Initializer instance, so "
            "they will draw bit-identical values forever. Construct a fresh one "
            f"per slot. Found: {collisions}"
        )

    def test_the_census_covers_the_population_it_claims_to(self, model):
        """Anti-vacuity: the sweep must actually reach every block.

        A collector that silently returned an empty list -- or that stopped at
        block 0 -- would report zero collisions forever.
        """
        slots = _initializer_slots(model)
        expected = 2 * model.depth + 4 + 6 + 4 + 1
        assert len(slots) == expected, (
            f"the initializer census collected {len(slots)} slots, expected "
            f"{expected} for depth={model.depth}"
        )
        labels = {label for label, _ in slots}
        for i in range(model.depth):
            assert f"block_{i}.adaln.kernel" in labels

    def test_the_predicate_fires_on_an_injected_shared_instance(self, model):
        """Dead-component probe for the collision detector itself.

        Two slots pointed at ONE object must be reported. Without this arm a
        detector that compared, say, ``type(a) is type(b)`` -- true for every
        pair of ``Zeros()`` -- would look identical and catch nothing.
        """
        hoisted = keras.initializers.Zeros()
        slots = [("a", hoisted), ("b", hoisted)]
        by_id = {}
        collisions = []
        for label, initializer in slots:
            if id(initializer) in by_id:
                collisions.append(label)
            else:
                by_id[id(initializer)] = label
        assert collisions == ["b"]
        assert type(keras.initializers.Zeros()) is type(hoisted)


class TestEveryNormEpsilonIsOneEMinusSix:
    """The census, not a spot check."""

    def _norm_layers(self, model):
        return [
            layer
            for layer in _walk(model)
            if hasattr(layer, "epsilon")
            and ("Norm" in type(layer).__name__)
        ]

    def test_every_normalization_sublayer_reports_the_same_epsilon(self, model):
        """Exact equality; ``1e-3`` is the Keras default this exists to reject."""
        offenders = [
            f"{type(l).__name__}('{l.name}') epsilon={l.epsilon}"
            for l in self._norm_layers(model)
            if float(l.epsilon) != EPSILON
        ]
        assert not offenders, (
            "every normalization layer in this package must carry an EXPLICIT "
            f"epsilon of {EPSILON}. Bare keras.layers.LayerNormalization defaults "
            f"to 1e-3, a 1000x error with no shape symptom. Found: {offenders}"
        )

    def test_the_census_population_is_the_expected_one(self, model):
        """Pinned by formula, so a lost sub-layer reddens instead of shrinking.

        Per block: four non-affine ``LayerNormalization`` (``norm1``,
        ``norm_cross``, ``norm_cond``, ``norm2``) and four per-head ``RMSNorm``
        (Q and K in each of the two attention sub-layers). Plus the final
        layer's ``norm_final``.
        """
        norms = self._norm_layers(model)
        expected = 8 * model.depth + 1
        assert len(norms) == expected, (
            f"expected {expected} normalization sub-layers for depth="
            f"{model.depth}, the walk found {len(norms)}: "
            f"{[(type(l).__name__, l.name) for l in norms]}"
        )
        kinds = {type(l).__name__ for l in norms}
        assert kinds == {"LayerNormalization", "RMSNorm"}, kinds

    def test_the_qk_norms_are_non_affine(self, model):
        """Upstream's ``elementwise_affine=False``.

        A learnable gain on Q/K is a different architecture that trains and
        never raises, so it needs its own assertion rather than riding on the
        epsilon one.
        """
        rms = [l for l in _walk(model) if type(l).__name__ == "RMSNorm"]
        assert rms, "no RMSNorm found; the qk_norm wiring changed"
        for layer in rms:
            assert layer.use_scale is False, (
                f"RMSNorm('{layer.name}') carries a learnable gain; upstream is "
                "elementwise_affine=False"
            )


class TestThePatchEmbedInitIsTheFlattenedLinearOne:
    """D-016: a conv kernel initialised as the flattened ``Linear`` it replaces."""

    def test_the_limit_matches_the_flattened_fans(self, model):
        """``limit = sqrt(6 / (p*p*C_in + D))`` -- ``fan_out`` has no ``p*p``."""
        fan_in = model.patch_size * model.patch_size * model.in_channels
        expected = math.sqrt(6.0 / (fan_in + model.hidden_size))
        for name in ("x_embedder", "cond_embedder_forward", "cond_embedder_reverse"):
            initializer = getattr(model, name).kernel_initializer
            assert isinstance(initializer, keras.initializers.RandomUniform), (
                f"{name} uses {type(initializer).__name__}; D-016 requires the "
                "explicit flattened-Linear xavier"
            )
            assert initializer.maxval == pytest.approx(expected, rel=1e-12)
            assert initializer.minval == pytest.approx(-expected, rel=1e-12)

    def test_it_is_measurably_different_from_the_keras_conv_default(self, model):
        """The whole point: ``"glorot_uniform"`` is NOT this.

        Keras computes a convolution's fans over the full kernel shape, so its
        ``fan_out`` carries the ``p * p`` receptive field that upstream's
        reshape removes. The two limits are pinned against each other here so
        that "just use glorot_uniform" reddens rather than passing quietly.
        """
        p, c, d = model.patch_size, model.in_channels, model.hidden_size
        ours = math.sqrt(6.0 / (p * p * c + d))
        keras_conv = math.sqrt(6.0 / (p * p * c + p * p * d))
        assert abs(ours - keras_conv) / keras_conv > 0.5, (
            f"the two limits are indistinguishable (ours={ours}, "
            f"keras_conv={keras_conv}), so this guard would not see the "
            "substitution it exists to reject"
        )

    def test_each_call_returns_a_fresh_instance(self):
        """The helper must never be turned into a cached singleton."""
        first = flattened_linear_xavier(16, 64)
        second = flattened_linear_xavier(16, 64)
        assert first is not second
        assert first.maxval == second.maxval

    @pytest.mark.parametrize("fan_in,fan_out", [(0, 4), (4, 0), (-1, 4)])
    def test_it_rejects_a_non_positive_fan(self, fan_in, fan_out):
        with pytest.raises(ValueError, match="positive"):
            flattened_linear_xavier(fan_in, fan_out)
