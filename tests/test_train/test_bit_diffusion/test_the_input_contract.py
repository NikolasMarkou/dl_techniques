r"""Guard: the pre-encoded input contract, and the ``.npz`` branch that reads it.

The defect class this exists to catch
-------------------------------------
Two, and they are the same defect seen from opposite ends.

1. AN ADVERTISED BRANCH THAT NOTHING EVER CONSTRUCTS. ``--train-npz`` /
   ``--val-npz`` route the trainer at real pre-encoded data instead of the
   synthetic generator. ``tests/test_train/test_config_fields_are_live.py``
   proves ``main()`` READS those fields; it cannot prove the branch RUNS. This
   repo has shipped a CLI branch that crashed on step 1 under a green suite
   because no test ever constructed it. ``test_the_npz_branch_actually_loads``
   constructs it.

2. A CONTRACT STATED ONLY IN PROSE. ``synthetic_data``'s module docstring makes
   three claims that no shape check can see: latents arrive pre-scaled, token
   rows are unit-norm times ``token_scale`` with exactly-zero padding, and
   labels index the prompt kind. The second is the one the port DEPENDS on --
   ``norm_based_token_stops`` recovers the sequence length from it -- so the
   generator's compliance is MEASURED here against
   ``compute_token_norms``/``norm_based_token_stops`` themselves rather than
   against a re-derivation of their arithmetic.

Trap designed out: A NORM ASSERTION ALONE IS VACUOUS ON A FULL SEQUENCE. If the
generator always filled every row, "padding rows are exactly zero" would be
trivially true (there are none) and the stop-detection path would never run.
``test_the_generator_produces_a_spread_of_stop_positions`` asserts the drawn
stops actually vary and include at least one short sequence before the
zero-padding arm is allowed to mean anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from dl_techniques.models.vision_language.bit_diffusion.config import (
    PROMPT_NUM_CLASSES,
    get_bridge_config,
)
from dl_techniques.models.vision_language.bit_diffusion.token_bridge import (
    compute_token_norms,
    norm_based_token_stops,
    token_flat_to_bridge,
)
from train.bit_diffusion.synthetic_data import (
    CONTRACT_KEYS,
    load_records_npz,
    save_records_npz,
    synthetic_records,
    validate_records,
)
from train.bit_diffusion.train_bit_diffusion import (
    TrainingConfig,
    load_or_draw_records,
)

SEED = 11
COUNT = 24


@pytest.fixture()
def bridge():
    return get_bridge_config("tiny")


@pytest.fixture()
def records(bridge):
    return synthetic_records(COUNT, bridge, seed=SEED)


# ---------------------------------------------------------------------
# the contract
# ---------------------------------------------------------------------


def test_the_generator_satisfies_its_own_contract(records, bridge):
    assert set(records) == set(CONTRACT_KEYS)
    assert validate_records(records, bridge) == COUNT
    assert records["latent"].dtype == np.float32
    assert records["text_token_emb"].dtype == np.float32
    assert records["prompt_kind_label"].dtype == np.int32
    assert records["prompt_kind_label"].max() < PROMPT_NUM_CLASSES


@pytest.mark.parametrize(
    "key,bad",
    [
        ("latent", np.zeros((COUNT, 4, 4, 4), dtype="float32")),
        ("text_token_emb", np.zeros((COUNT, 3), dtype="float32")),
        ("prompt_kind_label", np.zeros((COUNT, 1), dtype="int32")),
    ],
)
def test_validate_records_rejects_a_wrong_shape(records, bridge, key, bad):
    broken = dict(records)
    broken[key] = bad
    with pytest.raises(ValueError):
        validate_records(broken, bridge)


def test_validate_records_rejects_a_missing_key(records, bridge):
    broken = {k: v for k, v in records.items() if k != "text_token_emb"}
    with pytest.raises(KeyError):
        validate_records(broken, bridge)


def test_validate_records_rejects_a_ragged_batch(records, bridge):
    broken = dict(records)
    broken["latent"] = broken["latent"][:-1]
    with pytest.raises(ValueError, match="ragged"):
        validate_records(broken, bridge)


# ---------------------------------------------------------------------
# the text side is structurally honest
# ---------------------------------------------------------------------


def test_the_generator_produces_a_spread_of_stop_positions(records, bridge):
    """ANTI-VACUITY for the two arms below."""
    tokens = records["text_token_emb"].reshape(
        COUNT, bridge.token_seq_len, bridge.token_emb_dim
    )
    stops = (np.linalg.norm(tokens, axis=-1) > 1e-6).sum(axis=1)
    assert stops.min() < bridge.token_seq_len, (
        f"every sample fills all {bridge.token_seq_len} rows ({stops}); the "
        "zero-padding arm would then be trivially true and the stop-detection "
        "path would never run"
    )
    assert stops.max() >= 1
    assert len(set(stops.tolist())) > 1, f"stop positions do not vary: {stops}"


def test_real_rows_carry_the_token_scale_and_padding_rows_are_exactly_zero(
    records, bridge
):
    tokens = records["text_token_emb"].reshape(
        COUNT, bridge.token_seq_len, bridge.token_emb_dim
    )
    norms = np.linalg.norm(tokens, axis=-1)
    real = norms > 1e-6
    np.testing.assert_allclose(
        norms[real], bridge.token_scale, rtol=1e-5,
        err_msg="a non-padding token row is not unit-norm * token_scale",
    )
    assert np.all(tokens[~real] == 0.0), (
        "a padding row is not EXACTLY zero; norm_based_token_stops reads that "
        "exact zero, and 'almost zero' silently moves the stop"
    )


def test_the_port_s_own_stop_detector_recovers_the_generator_s_stops(
    records, bridge
):
    """Measured through the REAL detector, not a re-derivation of its math."""
    tokens = records["text_token_emb"].reshape(
        COUNT, bridge.token_seq_len, bridge.token_emb_dim
    )
    expected = (np.linalg.norm(tokens, axis=-1) > 1e-6).sum(axis=1)

    x_bridge = token_flat_to_bridge(records["text_token_emb"], bridge)
    norms = np.asarray(compute_token_norms(x_bridge, bridge))
    # `norm_based_token_stops` returns `(stops, norms)`, not a bare `stops`.
    recovered, detector_norms = norm_based_token_stops(x_bridge, bridge)
    recovered = np.asarray(recovered)
    np.testing.assert_allclose(np.asarray(detector_norms), norms, rtol=0, atol=0)

    np.testing.assert_allclose(norms[norms > 1e-3], 1.0, rtol=1e-5)
    np.testing.assert_array_equal(recovered, expected)


# ---------------------------------------------------------------------
# the .npz branch
# ---------------------------------------------------------------------


def test_npz_round_trips_exactly(records, bridge, tmp_path):
    path = save_records_npz(records, tmp_path / "shard-00000.npz")
    assert path.is_file()
    reloaded = load_records_npz(path)
    assert validate_records(reloaded, bridge) == COUNT
    for key in CONTRACT_KEYS:
        np.testing.assert_array_equal(reloaded[key], records[key])


def test_load_records_npz_rejects_a_file_missing_a_contract_key(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, latent=np.zeros((2, 8, 8, 4), dtype="float32"))
    with pytest.raises(KeyError):
        load_records_npz(path)


def test_the_npz_branch_actually_loads(records, bridge, tmp_path):
    """The advertised ``--train-npz`` path, CONSTRUCTED rather than assumed live.

    Both arms are driven: with the flag the records come off disk BYTE-FOR-BYTE,
    without it they come from the generator. The second arm is what proves the
    first is measuring the branch rather than a coincidence -- the two must
    differ, since ``num_train_samples`` is deliberately not the shard's size.
    """
    path = save_records_npz(records, tmp_path / "train.npz")
    config = TrainingConfig(
        bridge_preset="tiny",
        variant="tiny",
        train_npz=str(path),
        num_train_samples=COUNT + 5,
        seed=SEED,
    )
    from_disk = load_or_draw_records(config, "train")
    assert from_disk["latent"].shape[0] == COUNT
    np.testing.assert_array_equal(from_disk["latent"], records["latent"])

    synthetic = load_or_draw_records(
        TrainingConfig(
            bridge_preset="tiny", variant="tiny",
            num_train_samples=COUNT + 5, seed=SEED,
        ),
        "train",
    )
    assert synthetic["latent"].shape[0] == COUNT + 5


def test_the_two_splits_do_not_share_a_seed(bridge):
    """A val split identical to train reports a meaningless val_loss."""
    config = TrainingConfig(
        bridge_preset="tiny", variant="tiny",
        num_train_samples=COUNT, num_val_samples=COUNT, seed=SEED,
    )
    train = load_or_draw_records(config, "train")
    val = load_or_draw_records(config, "val")
    assert not np.array_equal(train["latent"], val["latent"])


def test_load_or_draw_records_rejects_an_unknown_split():
    config = TrainingConfig(bridge_preset="tiny", variant="tiny")
    with pytest.raises(ValueError, match="split"):
        load_or_draw_records(config, "test")
