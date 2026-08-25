"""Mask propagation across ``layers/norms/`` (finding B13).

A shape-preserving normalization layer that sits directly after an
``Embedding(mask_zero=True)`` must not destroy the Keras mask. At HEAD before
this module landed, 14 of the 16 registered classes did exactly that, and Keras
itself emitted ``UserWarning: Layer '...' does not support masking and will
therefore destroy the mask information``.

``supports_masking = True`` is a PROMISE, not a formality: it says the layer
treats each (sample, token) position independently, so a mask that was valid on
the input is still valid on the output. This module therefore pins BOTH sides of
the promise:

* the **inclusion set** carries the flag AND is measured to be token-independent;
* the **exclusion set** does NOT carry the flag, and the token-coupled members of
  it are measured to genuinely leak across tokens.

The exclusion assertions exist so that a future blanket "add ``supports_masking``
to every norm" sweep fails loudly instead of shipping a false promise.
"""

from typing import Any, Callable, Dict, List, Tuple

import keras
import numpy as np
import pytest

from dl_techniques.layers.norms.adaptive_band_rms import AdaptiveBandRMS
from dl_techniques.layers.norms.band_logit_norm import BandLogitNorm
from dl_techniques.layers.norms.band_rms import BandRMS
from dl_techniques.layers.norms.bias_free_batch_norm import BiasFreeBatchNorm
from dl_techniques.layers.norms.dynamic_tanh import DynamicTanh
from dl_techniques.layers.norms.energy_layer_norm import EnergyLayerNorm
from dl_techniques.layers.norms.global_response_norm import (
    GlobalResponseNormalization,
)
from dl_techniques.layers.norms.logit_norm import LogitNorm
from dl_techniques.layers.norms.max_logit_norm import (
    DMLPlus,
    DecoupledMaxLogit,
    MaxLogitNorm,
)
from dl_techniques.layers.norms.polar_weight_norm import PolarWeightNorm
from dl_techniques.layers.norms.rms_norm import RMSNorm
from dl_techniques.layers.norms.zero_centered_adaptive_band_rms_norm import (
    ZeroCenteredAdaptiveBandRMS,
)
from dl_techniques.layers.norms.zero_centered_band_rms_norm import (
    ZeroCenteredBandRMSNorm,
)
from dl_techniques.layers.norms.zero_centered_rms_norm import ZeroCenteredRMSNorm

# ---------------------------------------------------------------------------
# Populations
# ---------------------------------------------------------------------------

LayerFactory = Callable[[], keras.layers.Layer]

#: Shape-preserving AND token-independent: every output position depends only on
#: the input at that same (sample, token). These may honestly claim masking.
#: ``DynamicTanh`` and ``EnergyLayerNorm`` already carried the flag before B13.
MASK_PROPAGATING: List[Tuple[str, LayerFactory]] = [
    ("RMSNorm", RMSNorm),
    ("ZeroCenteredRMSNorm", ZeroCenteredRMSNorm),
    ("BandRMS", BandRMS),
    ("ZeroCenteredBandRMSNorm", ZeroCenteredBandRMSNorm),
    ("LogitNorm", LogitNorm),
    ("MaxLogitNorm", MaxLogitNorm),
    ("BandLogitNorm", BandLogitNorm),
    ("DynamicTanh", DynamicTanh),
    ("EnergyLayerNorm", EnergyLayerNorm),
]

#: Shape-preserving but TOKEN-COUPLED: the output at one token depends on OTHER
#: tokens (and, for ``BiasFreeBatchNorm`` in training mode, on other samples), so
#: a masked position still contributes to what unmasked positions become.
#: ``supports_masking = True`` would be a false promise here - the mask would
#: pass through while the statistics behind it had already been contaminated.
#: The per-layer reason is asserted by ``test_the_token_coupled_exclusions_really_leak``.
TOKEN_COUPLED_EXCLUSIONS: List[Tuple[str, LayerFactory]] = [
    # Aggregates RMS over EVERY non-batch axis (incl. the token axis) to drive an
    # internal Dense whose output rescales all tokens of the sample.
    ("AdaptiveBandRMS", AdaptiveBandRMS),
    ("ZeroCenteredAdaptiveBandRMS", ZeroCenteredAdaptiveBandRMS),
    # Reduces over axes (1, ..., rank-2) - i.e. the token/spatial axes - by design.
    ("GlobalResponseNormalization", GlobalResponseNormalization),
    # Training mode computes the batch variance over all non-channel axes.
    ("BiasFreeBatchNorm", BiasFreeBatchNorm),
]

#: Not shape-preserving at all: the first two reduce ``axis`` and return tuples,
#: the third is a Dense weight reparameterization that changes the feature dim.
#: A propagated mask would not even have a valid shape against the output.
SHAPE_CHANGING_EXCLUSIONS: List[Tuple[str, LayerFactory]] = [
    ("DecoupledMaxLogit", DecoupledMaxLogit),
    ("DMLPlus", lambda: DMLPlus(model_type="focal")),
    ("PolarWeightNorm", lambda: PolarWeightNorm(units=4)),
]

BATCH, TOKENS, FEATURES = 3, 5, 8
VOCAB = 11


def _sample_inputs() -> Tuple[np.ndarray, np.ndarray]:
    """Build a base tensor and a copy perturbed at one (sample, token) only.

    The perturbation is deliberately NON-UNIFORM across the feature axis: a
    uniform offset lies in the null space of zero-centering, which silently made
    an earlier version of this probe blind to the zero-centered layers.

    :return: ``(x, x_perturbed)``, differing only at ``[0, TOKENS - 1, :]``.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    rng = np.random.default_rng(0)
    x = rng.normal(size=(BATCH, TOKENS, FEATURES)).astype("float32")
    perturbed = x.copy()
    perturbed[0, TOKENS - 1, :] += 100.0 * rng.normal(size=(FEATURES,))
    return x, perturbed


def _make_nontrivial(layer: keras.layers.Layer) -> None:
    """Give an internal band ``Dense`` a non-zero kernel, if the layer has one.

    ``AdaptiveBandRMS`` and its zero-centered sibling default to
    ``band_initializer="zeros"``, which pins the Dense output to a constant and
    makes any token-coupling probe read exactly ``0.0`` regardless of the input.
    That is a probe that cannot fail; a trained model has no such kernel.

    :param layer: A built layer, possibly owning a ``dense_layer`` sublayer.
    :type layer: keras.layers.Layer
    """
    dense = getattr(layer, "dense_layer", None)
    if dense is None:
        return
    rng = np.random.default_rng(1)
    kernel = dense.kernel
    dense.kernel.assign(
        keras.ops.convert_to_tensor(
            rng.normal(size=kernel.shape).astype("float32")
        )
    )


def _cross_token_leak(
    factory: LayerFactory,
    training: bool,
) -> float:
    """Measure how much a change at ONE token moves the OTHER tokens' outputs.

    :param factory: Zero-argument callable returning a fresh layer.
    :type factory: LayerFactory
    :param training: Value forwarded as the ``training`` argument.
    :type training: bool
    :return: ``max|delta|`` over every output position except the perturbed one.
    :rtype: float
    """
    x, perturbed = _sample_inputs()
    layer = factory()
    layer.build(x.shape)
    _make_nontrivial(layer)

    base = keras.ops.convert_to_numpy(
        layer(keras.ops.convert_to_tensor(x), training=training)
    )
    moved = keras.ops.convert_to_numpy(
        layer(keras.ops.convert_to_tensor(perturbed), training=training)
    )

    other_tokens = float(
        np.max(np.abs(base[0, : TOKENS - 1] - moved[0, : TOKENS - 1]))
    )
    other_samples = float(np.max(np.abs(base[1:] - moved[1:])))
    return max(other_tokens, other_samples)


def _embedding_model(layer: keras.layers.Layer) -> Dict[str, Any]:
    """Run a mask-producing ``Embedding`` into ``layer`` and report both masks.

    :param layer: The normalization layer under test.
    :type layer: keras.layers.Layer
    :return: ``{"embedding_mask": ..., "output_mask": ..., "output": ...}``.
    :rtype: Dict[str, Any]
    """
    inputs = keras.Input(shape=(TOKENS,), dtype="int32")
    embedded = keras.layers.Embedding(
        input_dim=VOCAB, output_dim=FEATURES, mask_zero=True
    )(inputs)
    outputs = layer(embedded)
    return {
        "embedding_mask": getattr(embedded, "_keras_mask", None),
        "output_mask": getattr(outputs, "_keras_mask", None),
        "output": outputs,
    }


# ---------------------------------------------------------------------------
# Inclusion set
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,factory", MASK_PROPAGATING, ids=lambda v: v)
def test_the_mask_propagating_norms_preserve_shape(
    name: str, factory: LayerFactory
) -> None:
    """Shape preservation is a precondition for claiming mask support."""
    x, _ = _sample_inputs()
    layer = factory()
    output = layer(keras.ops.convert_to_tensor(x))
    assert tuple(output.shape) == x.shape, (
        f"{name} is not shape-preserving; it must not claim supports_masking"
    )


@pytest.mark.parametrize("name,factory", MASK_PROPAGATING, ids=lambda v: v)
def test_the_mask_propagating_norms_declare_supports_masking(
    name: str, factory: LayerFactory
) -> None:
    """The flag itself. RED at HEAD for the seven B13 layers."""
    assert factory().supports_masking is True, (
        f"{name}.supports_masking is not True, so Keras drops the mask "
        f"(and warns) whenever it follows an Embedding(mask_zero=True)"
    )


@pytest.mark.parametrize("name,factory", MASK_PROPAGATING, ids=lambda v: v)
def test_the_mask_survives_an_embedding(
    name: str, factory: LayerFactory
) -> None:
    """The behaviour the flag exists for: the mask reaches the next layer."""
    result = _embedding_model(factory())
    assert result["embedding_mask"] is not None, (
        "precondition failed: Embedding(mask_zero=True) produced no mask"
    )
    assert result["output_mask"] is not None, (
        f"{name} destroyed the Keras mask; downstream layers lose it silently"
    )


@pytest.mark.parametrize("name,factory", MASK_PROPAGATING, ids=lambda v: v)
def test_the_mask_propagating_norms_are_token_independent(
    name: str, factory: LayerFactory
) -> None:
    """The flag is only honest if no other position moves. Measured, not assumed."""
    for training in (False, True):
        leak = _cross_token_leak(factory, training=training)
        assert leak == 0.0, (
            f"{name} (training={training}) moved another position by {leak:.3e}; "
            f"it is token-coupled and must NOT claim supports_masking"
        )


# ---------------------------------------------------------------------------
# Exclusion set - a future blanket sweep must fail here
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,factory",
    TOKEN_COUPLED_EXCLUSIONS + SHAPE_CHANGING_EXCLUSIONS,
    ids=lambda v: v,
)
def test_the_excluded_norms_do_not_claim_masking(
    name: str, factory: LayerFactory
) -> None:
    """These layers must NOT carry the flag; setting it would be a false promise."""
    assert factory().supports_masking is False, (
        f"{name} claims supports_masking, but it is either token-coupled or not "
        f"shape-preserving. Do not add the flag here - see the module docstring "
        f"and decisions.md D-007 of plan-2026-08-25T195813-d5a035ab"
    )


@pytest.mark.parametrize(
    "name,factory", TOKEN_COUPLED_EXCLUSIONS, ids=lambda v: v
)
def test_the_token_coupled_exclusions_really_leak(
    name: str, factory: LayerFactory
) -> None:
    """Document WHY each token-coupled exclusion is excluded, by measurement."""
    leak = max(
        _cross_token_leak(factory, training=False),
        _cross_token_leak(factory, training=True),
    )
    assert leak > 1e-3, (
        f"{name} was excluded from supports_masking on the grounds that it mixes "
        f"tokens, but the measured cross-position leak is only {leak:.3e}. Either "
        f"the layer changed or this probe went blind - re-adjudicate before "
        f"moving it into MASK_PROPAGATING"
    )


def test_the_shape_changing_exclusions_are_not_shape_preserving() -> None:
    """Pin the structural reason the last three are excluded."""
    x = keras.ops.convert_to_tensor(_sample_inputs()[0])

    combined, max_cosine, max_norm = DecoupledMaxLogit()(x)
    assert tuple(combined.shape) == (BATCH, TOKENS)
    assert tuple(max_cosine.shape) == (BATCH, TOKENS)
    assert tuple(max_norm.shape) == (BATCH, TOKENS)

    # DMLPlus goes further still: on a rank-3 input it collapses the BATCH axis
    # too (measured ``(TOKENS,)``, not ``(BATCH, TOKENS)``), because its focal
    # branch reduces over axis 0. Pinned as observed - it is out of B13's scope,
    # but it makes the "never propagate a mask through this" case overwhelming.
    combined, _, _ = DMLPlus(model_type="focal")(x)
    assert tuple(combined.shape) == (TOKENS,)

    projected = PolarWeightNorm(units=4)(x)
    assert tuple(projected.shape) == (BATCH, TOKENS, 4)
