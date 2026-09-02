"""The token<->bridge packing is a lossless bijection with a pinned layout.

A layout error in a reshape/transpose chain passes every shape assertion, every
round-trip assertion and every finiteness assertion. The only instrument that
sees it is one that says WHERE a specific flat column must land, computed from a
hand-derived index expression rather than by re-running the implementation's own
arithmetic.

Every expected ``(row, col, channel)`` in this file comes from
``_expected_bridge_position`` below, which is written out in closed form and
never calls ``keras.ops.reshape``/``keras.ops.transpose`` and never calls
``token_flat_to_bridge``. Deriving the expectation from the code under test
would produce a guard that agrees with any implementation, including a wrong
one.

Success criterion: SC-1.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision_language.bit_diffusion.config import (
    BRIDGE_PRESETS,
    BridgeConfig,
)
from dl_techniques.models.vision_language.bit_diffusion.token_bridge import (
    bridge_to_token_flat,
    token_flat_to_bridge,
)

PRESET_NAMES = sorted(BRIDGE_PRESETS.keys())


# ---------------------------------------------------------------------
# Independent index arithmetic -- the oracle.
# ---------------------------------------------------------------------
# Derived by hand from the packing definition, NOT from the implementation:
#
#   flat column      = token_index * token_emb_dim
#                      + patch_within_token * patch_payload_dim
#                      + payload_index
#   patch_index      = token_index * patches_per_token + patch_within_token
#                      (row_major layout == identity permutation)
#   grid_row/grid_col= divmod(patch_index, patch_w)
#   within a patch the payload is ordered (row offset, col offset, channel)
#                      with the channel varying FASTEST
#   row              = grid_row * patch_size + row offset
#   col              = grid_col * patch_size + col offset
#   channel          = channel offset
# ---------------------------------------------------------------------


def _flat_column(config: BridgeConfig, token_index, patch_within_token, payload_index):
    """Flat index into ``token_flat`` of one (token, sub-patch, payload) triple."""
    return (
        token_index * config.token_emb_dim
        + patch_within_token * config.patch_payload_dim
        + payload_index
    )


def _expected_bridge_position(
    config: BridgeConfig, token_index, patch_within_token, payload_index
):
    """Return the ``(row, col, channel)`` that flat column must occupy."""
    p = config.patch_size
    c = config.channels
    patch_index = token_index * config.patches_per_token + patch_within_token
    grid_row = patch_index // config.patch_w
    grid_col = patch_index % config.patch_w
    row_offset = payload_index // (p * c)
    col_offset = (payload_index // c) % p
    channel = payload_index % c
    return grid_row * p + row_offset, grid_col * p + col_offset, channel


def _triple_for_patch(config: BridgeConfig, patch_index, payload_index):
    """Split a patch index back into ``(token_index, patch_within_token)``."""
    ppt = config.patches_per_token
    return patch_index // ppt, patch_index % ppt, payload_index


def _one_hot_batch(config: BridgeConfig, columns):
    """``(len(columns), token_flat_dim)`` float32 one-hot rows."""
    out = np.zeros((len(columns), config.token_flat_dim), dtype="float32")
    out[np.arange(len(columns)), np.asarray(columns, dtype="int64")] = 1.0
    return out


def _probe_triples(config: BridgeConfig):
    """Boundary + interior patches crossed with several intra-patch payloads.

    The payload sweep is what makes an intra-patch ``p``/``q`` permutation
    visible: a probe that only ever sets ``payload_index=0`` sits at the fixed
    point of that permutation and cannot see it.
    """
    p = config.patch_size
    c = config.channels
    assert config.patch_h >= 3 and config.patch_w >= 3, (
        "the interior probe below needs a grid of at least 3x3 patches"
    )
    patches = [
        0,                                             # h=0, w=0 corner
        config.num_patches - 1,                        # h=patch_h-1, w=patch_w-1
        1 * config.patch_w + 2,                        # interior, grid_row != grid_col
        2 * config.patch_w + 1,                        # interior, transposed partner
    ]
    payloads = sorted(
        {
            0,                       # row offset 0, col offset 0, channel 0
            min(1, c - 1),           # channel varies fastest
            c % config.patch_payload_dim,          # col offset 1 (if p > 1)
            (p * c) % config.patch_payload_dim,    # row offset 1 (if p > 1)
            config.patch_payload_dim - 1,          # last element of the patch
        }
    )
    return [
        _triple_for_patch(config, patch_index, payload_index)
        for patch_index in patches
        for payload_index in payloads
    ]


# ---------------------------------------------------------------------
# Arm 1 -- one-hot probes land at the independently computed position.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", PRESET_NAMES)
def test_one_hot_columns_land_at_the_independently_computed_position(preset_name):
    config = BRIDGE_PRESETS[preset_name]
    triples = _probe_triples(config)
    columns = [_flat_column(config, *triple) for triple in triples]

    bridge = keras.ops.convert_to_numpy(
        token_flat_to_bridge(
            keras.ops.convert_to_tensor(_one_hot_batch(config, columns)),
            config=config,
        )
    )

    assert bridge.shape == (
        len(columns),
        config.height,
        config.width,
        config.channels,
    )

    for i, triple in enumerate(triples):
        expected = _expected_bridge_position(config, *triple)
        nonzero = np.argwhere(bridge[i] != 0.0)
        assert nonzero.shape[0] == 1, (
            f"{preset_name}: triple {triple} produced {nonzero.shape[0]} nonzero "
            f"entries, expected exactly 1 (the packing must be a permutation)"
        )
        actual = tuple(int(v) for v in nonzero[0])
        assert actual == expected, (
            f"{preset_name}: flat column {columns[i]} "
            f"(token={triple[0]}, patch_within_token={triple[1]}, "
            f"payload={triple[2]}) landed at (row, col, channel)={actual}, "
            f"expected {expected}"
        )
        assert bridge[i][expected] == 1.0


@pytest.mark.parametrize("preset_name", PRESET_NAMES)
def test_every_flat_column_lands_at_its_independently_computed_position(preset_name):
    """Exhaustive census: all ``token_flat_dim`` columns at once, one call.

    A spot check can be defeated by a permutation that happens to fix the
    probed columns; a full census cannot.
    """
    config = BRIDGE_PRESETS[preset_name]
    if config.token_flat_dim > 4096:
        pytest.skip("census restricted to presets small enough for a dense identity")

    n = config.token_flat_dim
    bridge = keras.ops.convert_to_numpy(
        token_flat_to_bridge(
            keras.ops.convert_to_tensor(np.eye(n, dtype="float32")), config=config
        )
    )
    flat = bridge.reshape(n, -1)

    assert np.count_nonzero(flat) == n, (
        f"{preset_name}: expected exactly one nonzero per column, got "
        f"{np.count_nonzero(flat)} over {n} columns"
    )

    expected_flat = np.empty(n, dtype="int64")
    for column in range(n):
        token_index, rest = divmod(column, config.token_emb_dim)
        patch_within_token, payload_index = divmod(rest, config.patch_payload_dim)
        row, col, channel = _expected_bridge_position(
            config, token_index, patch_within_token, payload_index
        )
        expected_flat[column] = (
            row * config.width * config.channels + col * config.channels + channel
        )

    actual_flat = np.argmax(flat != 0.0, axis=1)
    mismatches = np.argwhere(actual_flat != expected_flat).ravel()
    assert mismatches.size == 0, (
        f"{preset_name}: {mismatches.size}/{n} columns landed at the wrong "
        f"(row, col, channel); first offenders "
        f"{[(int(k), int(actual_flat[k]), int(expected_flat[k])) for k in mismatches[:8]]}"
    )
    assert len(set(expected_flat.tolist())) == n, "the oracle itself must be a bijection"


# ---------------------------------------------------------------------
# Arm 2 -- exact round trip.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", PRESET_NAMES)
def test_the_round_trip_is_exact(preset_name):
    config = BRIDGE_PRESETS[preset_name]
    rng = np.random.default_rng(20260902)
    token_flat = rng.standard_normal((3, config.token_flat_dim)).astype("float32")

    bridge = token_flat_to_bridge(
        keras.ops.convert_to_tensor(token_flat), config=config
    )
    recovered = keras.ops.convert_to_numpy(
        bridge_to_token_flat(bridge, config=config)
    )

    assert recovered.shape == token_flat.shape
    assert np.array_equal(recovered, token_flat), (
        f"{preset_name}: round trip is not exact at atol=0; "
        f"max|delta| = {np.abs(recovered - token_flat).max()}"
    )


@pytest.mark.parametrize("preset_name", PRESET_NAMES)
def test_the_round_trip_is_exact_from_the_bridge_side(preset_name):
    config = BRIDGE_PRESETS[preset_name]
    rng = np.random.default_rng(7556)
    bridge = rng.standard_normal(
        (2, config.height, config.width, config.channels)
    ).astype("float32")

    recovered = keras.ops.convert_to_numpy(
        token_flat_to_bridge(
            bridge_to_token_flat(keras.ops.convert_to_tensor(bridge), config=config),
            config=config,
        )
    )
    assert np.array_equal(recovered, bridge), (
        f"{preset_name}: bridge -> tokens -> bridge is not exact at atol=0; "
        f"max|delta| = {np.abs(recovered - bridge).max()}"
    )


# ---------------------------------------------------------------------
# Arm 3 -- config validation.
# ---------------------------------------------------------------------


def test_every_shipped_preset_validates():
    for name, config in BRIDGE_PRESETS.items():
        config.validate()  # must not raise
        assert config.token_flat_dim == config.bridge_flat_dim, name
        assert config.token_seq_len * config.patches_per_token == config.num_patches


def test_validate_rejects_a_token_flat_dim_that_does_not_match_the_bridge():
    bad = BridgeConfig(
        token_seq_len=64,
        token_emb_dim=32,          # 64*32 = 2048 != 4*32*32 = 4096
        bridge_shape=(32, 32, 4),
        patch_size=2,
    )
    assert bad.token_flat_dim != bad.bridge_flat_dim
    with pytest.raises(ValueError, match="token_flat_dim"):
        bad.validate()


def test_validate_rejects_a_patch_count_that_does_not_match_the_token_grid():
    # 16 * 24 = 384 == 12 * 8 * 4, so the flat-dim check passes; but
    # patches_per_token floors to 24 // 16 = 1 and 16 * 1 != 24 patches.
    bad = BridgeConfig(
        token_seq_len=16,
        token_emb_dim=24,
        bridge_shape=(12, 8, 4),
        patch_size=2,
    )
    assert bad.token_flat_dim == bad.bridge_flat_dim
    assert bad.token_seq_len * bad.patches_per_token != bad.num_patches
    with pytest.raises(ValueError, match="patches_per_token"):
        bad.validate()


def test_validate_rejects_a_bridge_not_divisible_by_the_patch_size():
    bad = BridgeConfig(
        token_seq_len=16,
        token_emb_dim=45,
        bridge_shape=(15, 12, 4),
        patch_size=2,
    )
    with pytest.raises(ValueError, match="divisible"):
        bad.validate()
