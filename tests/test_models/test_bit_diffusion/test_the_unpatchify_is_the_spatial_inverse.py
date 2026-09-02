r"""``DiTXAFinalLayer.unpatchify`` must be the spatial inverse of ``x_embedder``.

The defect this exists to catch
-------------------------------
``model.py`` unpatchifies with ``reshape -> transpose(0, 1, 3, 2, 4, 5) ->
reshape``. Swap that permutation to ``(0, 2, 1, 3, 4, 5)`` and **every one of the
525 tests in this directory stays green**: ``input_size`` is square in all five
variants, so ``grid_height == grid_width``, the output shape is unchanged, and
the round trip, the gradient-flow oracle, the precision arm and the trainer smoke
are all indifferent to WHERE a payload lands. The model would emit a spatially
scrambled bridge tensor and nothing would say so.

This is the same defect class the plan already closed on the input side --
``test_the_packing_agrees_with_the_conv_patch_grid.py``, raised by
``probes/orchestrator_transpose_verification.md`` -- applied to the output side.
Reviewer finding W-4.

How the positions are pinned
----------------------------
NEVER by re-invoking the code's own ``reshape``/``transpose``. Every expected
position here is computed from the DEFINITION of the layout:

    token ``n`` of a ``(h, w)`` grid is the patch at ``(i, j) = (n // w, n % w)``
    and its payload index ``k`` decomposes as ``k = ((a * p) + b) * c + ch``,
    so element ``k`` of token ``n`` belongs at spatial position
    ``(i * p + a, j * p + b)``, channel ``ch``.

Two independent instruments:

* an EXHAUSTIVE census -- every one of the ``N * p * p * C`` payload elements is
  given a unique id and its landing site is compared against the formula above,
  so a permutation that merely preserves the multiset of positions is convicted;
* a JOINT arm that measures, behaviourally, which token index ``x_embedder``
  assigns to each spatial block (all-ones conv kernel, single-block input) and
  requires ``unpatchify`` to send that same token back to that same block. That
  is the "spatial inverse" claim as a measured fact rather than two separately
  plausible conventions.

Traps designed out
------------------
THE SQUARE-GRID BLINDNESS IS ITSELF TESTED. The rectangular arm builds a
``DiTXAFinalLayer`` at ``grid_height != grid_width``, where a transposed
interleave is not even shape-compatible. It exists to prove the formula is
row-major rather than accidentally symmetric -- but it is deliberately NOT the
only arm, because no shipped variant is rectangular and a guard that only fires
off-configuration would not have caught the real defect.

ANTI-VACUITY. ``p``, ``h``, ``w`` and ``C`` must all exceed 1. At ``p = 1`` the
interleave is the identity and the whole file is vacuous; at ``C = 1`` a
channel/position confusion is invisible.

DEAD-COMPONENT PROBE. ``test_the_census_convicts_a_transposed_interleave``
reproduces both injections in NumPy -- the swapped permutation and the transpose
deleted -- so the census's power is asserted in every run, not only in the run
where someone edited the source.
"""

import itertools

import keras
import numpy as np
import pytest

from dl_techniques.models.vision_language.bit_diffusion.config import (
    BRIDGE_PRESETS,
)
from dl_techniques.models.vision_language.bit_diffusion.model import (
    DiTXA,
    DiTXAFinalLayer,
)

from ._ditxa_helpers import np_

#: The `tiny` preset: bridge 8x8x4, patch 2, so a 4x4 grid of 16 tokens.
CONFIG = BRIDGE_PRESETS["tiny"]


@pytest.fixture(scope="module")
def model() -> DiTXA:
    """A ``tiny`` model whose ``x_embedder`` and ``final_layer`` both exist."""
    m = DiTXA.from_variant("tiny")
    m.x_embedder.build((None, CONFIG.height, CONFIG.width, CONFIG.channels))
    return m


def _expected_position(n, k, h, w, p, c):
    """Where payload element ``k`` of token ``n`` belongs, from the DEFINITION.

    Interface contract: pure arithmetic over the layout convention; it calls
    nothing from the module under test. Returns ``(row, col, channel)``.
    """
    i, j = divmod(n, w)
    within, ch = divmod(k, c)
    a, b = divmod(within, p)
    return i * p + a, j * p + b, ch


def _identity_tokens(num_tokens, payload):
    """``(1, N, payload)`` where every element is a distinct positive integer."""
    return np.arange(
        1, num_tokens * payload + 1, dtype="float32"
    ).reshape(1, num_tokens, payload)


def _census(layer, num_tokens, payload, h, w, p, c):
    """Map every unique id to (observed position, expected position)."""
    out = np_(layer.unpatchify(keras.ops.convert_to_tensor(
        _identity_tokens(num_tokens, payload)
    )))
    assert out.shape == (1, h * p, w * p, c), (
        f"unpatchify returned {out.shape}, geometry says {(1, h * p, w * p, c)}"
    )
    observed = {}
    for row, col, ch in itertools.product(range(h * p), range(w * p), range(c)):
        observed[int(out[0, row, col, ch])] = (row, col, ch)
    return out, observed


# ---------------------------------------------------------------------
# 0. Anti-vacuity
# ---------------------------------------------------------------------


def test_the_geometry_makes_this_file_non_vacuous(model):
    """At ``p = 1`` or ``C = 1`` the layout question does not exist."""
    layer = model.final_layer
    assert layer.patch_size > 1, "a 1x1 patch makes the interleave the identity"
    assert layer.out_channels > 1, (
        "a single channel hides every position/channel confusion"
    )
    assert layer.grid_height > 1 and layer.grid_width > 1
    assert (layer.grid_height, layer.grid_width) == (
        CONFIG.patch_h, CONFIG.patch_w
    ), "the final layer's grid disagrees with the bridge geometry"
    # The square symmetry the reviewer named, asserted rather than assumed:
    # this IS the configuration in which a transposed interleave is invisible
    # to every shape-based check, which is why the census below is exhaustive.
    assert layer.grid_height == layer.grid_width


# ---------------------------------------------------------------------
# 1. The exhaustive positional census on the REAL shipped layer
# ---------------------------------------------------------------------


def test_unpatchify_places_every_payload_element_where_the_layout_says(model):
    """All ``N * p * p * C`` elements, each against an independent formula."""
    layer = model.final_layer
    p, c = layer.patch_size, layer.out_channels
    h, w = layer.grid_height, layer.grid_width
    num_tokens, payload = h * w, p * p * c

    out, observed = _census(layer, num_tokens, payload, h, w, p, c)
    assert len(observed) == num_tokens * payload, (
        "unpatchify lost or duplicated elements: "
        f"{len(observed)} distinct ids for {num_tokens * payload} inputs"
    )

    wrong = []
    for n in range(num_tokens):
        for k in range(payload):
            uid = n * payload + k + 1
            expected = _expected_position(n, k, h, w, p, c)
            if observed[uid] != expected:
                wrong.append((n, k, observed[uid], expected))
    assert not wrong, (
        f"{len(wrong)} of {num_tokens * payload} payload elements landed at the "
        f"wrong spatial position. First four (token, payload_index, got, want): "
        f"{wrong[:4]}. The interleave in DiTXAFinalLayer.unpatchify is not the "
        "row-major inverse of the patch grid."
    )


def test_the_census_convicts_a_transposed_interleave(model):
    """DEAD-COMPONENT PROBE, reproduced in NumPy, run every time.

    Both injections the reviewer named. If either of these ever agrees with the
    formula, the census above has stopped discriminating and must be rewritten
    before it is trusted again.
    """
    layer = model.final_layer
    p, c = layer.patch_size, layer.out_channels
    h, w = layer.grid_height, layer.grid_width
    num_tokens, payload = h * w, p * p * c
    tokens = _identity_tokens(num_tokens, payload)

    swapped = np.transpose(
        tokens.reshape(1, h, w, p, p, c), (0, 2, 1, 3, 4, 5)
    ).reshape(1, h * p, w * p, c)
    deleted = tokens.reshape(1, h, w, p, p, c).reshape(1, h * p, w * p, c)
    correct = np_(layer.unpatchify(keras.ops.convert_to_tensor(tokens)))

    assert swapped.shape == correct.shape, (
        "the swapped permutation is shape-INCOMPATIBLE here, so this arm is not "
        "measuring the invisible defect; the grid must be square"
    )
    mismatched = int((swapped != correct).sum())
    assert mismatched > 0, (
        "transpose(0,2,1,3,4,5) produced the IDENTICAL tensor; the census "
        "cannot see the defect it exists to catch"
    )
    assert int((deleted != correct).sum()) > 0, (
        "deleting the transpose produced the IDENTICAL tensor"
    )
    # Both injections preserve the SET of values -- which is exactly why a
    # sum, a norm, a shape or a round trip is blind to them.
    assert sorted(swapped.ravel()) == sorted(correct.ravel())
    assert sorted(deleted.ravel()) == sorted(correct.ravel())


# ---------------------------------------------------------------------
# 2. The joint arm: unpatchify undoes x_embedder's own patch grid
# ---------------------------------------------------------------------


def _embedder_token_of_block(model, i, j):
    """Which token index ``x_embedder`` assigns to spatial block ``(i, j)``.

    MEASURED, not assumed: the conv kernel is set to ones and the bias to zeros,
    so a single nonzero spatial block produces a nonzero embedding at exactly
    the token whose window covers it.
    """
    conv = model.x_embedder.proj
    conv.kernel.assign(np.ones(np_(conv.kernel).shape, dtype="float32"))
    conv.bias.assign(np.zeros(np_(conv.bias).shape, dtype="float32"))
    p = CONFIG.patch_size
    bridge = np.zeros(
        (1, CONFIG.height, CONFIG.width, CONFIG.channels), dtype="float32"
    )
    bridge[0, i * p : (i + 1) * p, j * p : (j + 1) * p, :] = 1.0
    out = np_(model.x_embedder(keras.ops.convert_to_tensor(bridge)))
    hits = np.nonzero(np.abs(out[0]).sum(axis=-1))[0].tolist()
    assert len(hits) == 1, (
        f"spatial block {(i, j)} activated {len(hits)} tokens: {hits}"
    )
    return int(hits[0])


def test_unpatchify_returns_each_token_to_the_block_x_embedder_read_it_from(model):
    """THE INVERSE CLAIM, both halves measured on the real objects."""
    layer = model.final_layer
    p, c = layer.patch_size, layer.out_channels
    h, w = layer.grid_height, layer.grid_width
    num_tokens, payload = h * w, p * p * c

    out, _ = _census(layer, num_tokens, payload, h, w, p, c)
    ids_per_token = {
        n: set(range(n * payload + 1, (n + 1) * payload + 1))
        for n in range(num_tokens)
    }

    for i, j in itertools.product(range(h), range(w)):
        token = _embedder_token_of_block(model, i, j)
        block = out[0, i * p : (i + 1) * p, j * p : (j + 1) * p, :]
        landed = set(int(v) for v in block.ravel())
        assert landed == ids_per_token[token], (
            f"x_embedder reads spatial block {(i, j)} as token {token}, but "
            f"unpatchify fills that block with payload from token(s) "
            f"{sorted({(v - 1) // payload for v in landed})}. The final layer "
            "is not the spatial inverse of the patch embedder."
        )


# ---------------------------------------------------------------------
# 3. The rectangular arm -- row-major, not accidentally symmetric
# ---------------------------------------------------------------------


def test_a_rectangular_grid_is_row_major_too():
    """``grid_height != grid_width``, where the square symmetry cannot hide.

    Not a substitute for the census above: no shipped variant is rectangular,
    so a guard that only fired here would not have caught the real defect.
    """
    h, w, p, c = 3, 5, 2, 3
    layer = DiTXAFinalLayer(
        hidden_size=8, patch_size=p, out_channels=c,
        grid_height=h, grid_width=w, name="rect",
    )
    num_tokens, payload = h * w, p * p * c
    _, observed = _census(layer, num_tokens, payload, h, w, p, c)
    for n in range(num_tokens):
        for k in range(payload):
            uid = n * payload + k + 1
            assert observed[uid] == _expected_position(n, k, h, w, p, c), (
                f"token {n} payload {k} landed at {observed[uid]}, the "
                f"row-major layout says {_expected_position(n, k, h, w, p, c)}"
            )
