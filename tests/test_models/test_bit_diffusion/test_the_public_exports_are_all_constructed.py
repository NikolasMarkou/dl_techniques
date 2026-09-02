r"""Every name in the package ``__all__`` is CONSTRUCTED by something.

Reviewer finding W-6. Four exported names had no test that ever called them:

======================== ================================================
name                     disposition (step 9.1)
======================== ================================================
``pad_id_token_stops``   KEPT, tested here
``TOKEN_LAYOUTS``        KEPT, tested here
``PROMPT_KIND_TO_LABEL`` KEPT, tested here
``prepare_bridge_batch`` KEPT, tested here
======================== ================================================

Why all four were KEPT rather than dropped from ``__all__``
-----------------------------------------------------------
The rule this plan applies (D-008) is that a guard no input can reach is not a
guard, and the sibling rule is that code no caller can reach is not code. None
of these four is unreachable; each has a named consumer, which is exactly what
distinguishes "untested" from "dead":

* ``pad_id_token_stops`` is the id-based twin of ``norm_based_token_stops``, and
  it is upstream's real evaluation-loop function -- ``FULL_INGEST.py:5572``
  defines it and ``:3738`` calls it on ``pred_ids`` to find each row's stop.
  This port ships ``SharedTokenDecoder``, whose logits ``argmax`` to exactly the
  ``pred_ids`` it takes, so the surface it completes is present. It is also the
  only caller of ``_first_true_index``'s "no match anywhere -> T" branch that
  reaches it through a hard EQUALITY rather than a threshold.
* ``TOKEN_LAYOUTS`` is the registry ``_patch_positions`` validates ``layout=``
  against, and every packing function takes that argument. Dropping the constant
  from ``__all__`` would leave a public keyword whose legal values are private.
* ``PROMPT_KIND_TO_LABEL`` is the input contract for ``prompt_kind_label``: a
  caller building records has to know which integer means which caption kind,
  and its length must equal the class-embedding table's row count.
* ``prepare_bridge_batch`` is called by the trainer
  (``train/bit_diffusion/synthetic_data.py``) and is where the two ``*_as_noise``
  ablations actually happen. Untested, not unused.

So the disposition is uniform -- keep and test -- and the reason is uniform: all
four are reachable public surface, and the defect was the missing test, not the
export. The recorded failure shape this closes is "an advertised branch was DEAD
under a green suite".

Traps designed out
------------------
NO ARM ASSERTS ONLY A SHAPE OR A TYPE. ``TOKEN_LAYOUTS`` is checked by feeding
its members to the function that validates against it AND by feeding a
non-member and requiring a raise; ``PROMPT_KIND_TO_LABEL`` is checked against
``PROMPT_NUM_CLASSES`` and against the label range a real embedding table
accepts, not merely for having three entries.

THE ABLATION ARMS PIN WHICH TENSOR MOVED. ``text_as_noise`` must change the
process start point and leave the conditioning endpoint alone; asserting only
"the output changed" would pass if both moved, which is the bug that flag exists
to avoid.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision_language.bit_diffusion import (
    PROMPT_KIND_TO_LABEL,
    PROMPT_NUM_CLASSES,
    TOKEN_LAYOUTS,
    BRIDGE_PRESETS,
    BridgeConfig,
    pad_id_token_stops,
    prepare_bridge_batch,
    token_flat_to_bridge,
)

from ._ditxa_helpers import np_

CONFIG = BRIDGE_PRESETS["tiny"]
SEED = 3


def _records(count: int = 4, config: BridgeConfig = CONFIG) -> dict:
    """A contract-shaped record batch, built here rather than imported.

    Deliberately NOT ``train.bit_diffusion.synthetic_data.synthetic_records``:
    the package under test must not need the training package to be testable,
    and a hand-built batch makes the endpoint values known constants that the
    ablation arms below can identify by value.
    """
    rng = np.random.default_rng(SEED)
    return {
        "latent": rng.normal(
            size=(count, config.height, config.width, config.channels)
        ).astype("float32"),
        "text_token_emb": rng.normal(
            size=(count, config.token_flat_dim)
        ).astype("float32"),
        "prompt_kind_label": rng.integers(
            0, PROMPT_NUM_CLASSES, size=(count,)
        ).astype("int32"),
    }


# ---------------------------------------------------------------------
# pad_id_token_stops
# ---------------------------------------------------------------------


class TestPadIdTokenStops:
    """The id-based stop detector, on hand-built rows with known answers."""

    def test_it_returns_the_first_pad_position_of_every_row(self):
        """Every branch in one table: early pad, late pad, none, index 0.

        The expected values are typed out, not computed from the input by the
        same rule the function uses -- a self-referential oracle would pass
        against any monotone re-implementation.
        """
        pad = 7
        ids = np.array(
            [
                [1, 2, 7, 3, 4],   # first pad at 2, a LATER pad at nothing else
                [7, 7, 7, 7, 7],   # every position is pad -> 0
                [1, 2, 3, 4, 5],   # no pad at all -> T == 5
                [1, 2, 3, 4, 7],   # pad only in the last column -> 4
                [1, 7, 3, 7, 5],   # TWO pads; the FIRST one wins
            ],
            dtype="int32",
        )
        expected = [2, 0, 5, 4, 1]
        got = np_(pad_id_token_stops(keras.ops.convert_to_tensor(ids), pad)).tolist()
        assert got == expected, f"got {got}, expected {expected}"

    def test_a_row_with_no_padding_returns_the_full_length(self):
        """The branch a threshold-based detector reaches differently.

        ``norm_based_token_stops`` finds "no padding" via a comparison that can
        be nudged by a tolerance; this one needs an exact id match, so the
        all-real-tokens row is a genuinely different code path through
        ``_first_true_index``.
        """
        ids = np.arange(12, dtype="int32").reshape(3, 4)
        stops = np_(pad_id_token_stops(keras.ops.convert_to_tensor(ids), 99))
        assert stops.tolist() == [4, 4, 4]

    def test_the_pad_id_is_the_thing_that_selects(self):
        """ANTI-VACUITY: changing only ``pad_id`` must change the answer.

        Without this, a function that always returned "first column" or always
        returned ``T`` would pass one of the arms above.
        """
        ids = keras.ops.convert_to_tensor(
            np.array([[5, 6, 7, 8]], dtype="int32")
        )
        seen = {
            pad: int(np_(pad_id_token_stops(ids, pad))[0])
            for pad in (5, 6, 7, 8, 9)
        }
        assert seen == {5: 0, 6: 1, 7: 2, 8: 3, 9: 4}, seen

    def test_it_accepts_the_argmax_of_decoder_logits(self):
        """The real consumer shape: ``(B, T)`` ids from an ``argmax``."""
        logits = keras.random.normal((2, 6, 11), seed=SEED)
        ids = keras.ops.argmax(logits, axis=-1)
        stops = np_(pad_id_token_stops(ids, 0))
        assert stops.shape == (2,)
        assert np.all((stops >= 0) & (stops <= 6))


# ---------------------------------------------------------------------
# TOKEN_LAYOUTS
# ---------------------------------------------------------------------


class TestTokenLayouts:
    """The registry, exercised through the function that validates against it."""

    def test_every_registered_layout_is_accepted_by_the_packing(self):
        flat = keras.random.normal((2, CONFIG.token_flat_dim), seed=SEED)
        for layout in TOKEN_LAYOUTS:
            bridge = token_flat_to_bridge(flat, CONFIG, layout=layout)
            assert tuple(bridge.shape) == (
                2, CONFIG.height, CONFIG.width, CONFIG.channels
            ), f"layout {layout!r} produced {bridge.shape}"

    def test_an_unregistered_layout_raises_and_names_the_registry(self):
        flat = keras.random.normal((2, CONFIG.token_flat_dim), seed=SEED)
        with pytest.raises(ValueError, match="Unknown token layout"):
            token_flat_to_bridge(flat, CONFIG, layout="column_major")

    def test_the_default_argument_is_a_registered_layout(self):
        """A default outside the registry would make every call raise.

        Cheap, and it is the failure a rename of the single member would cause.
        """
        assert "row_major" in TOKEN_LAYOUTS
        assert len(TOKEN_LAYOUTS) == len(set(TOKEN_LAYOUTS))


# ---------------------------------------------------------------------
# PROMPT_KIND_TO_LABEL
# ---------------------------------------------------------------------


class TestPromptKindToLabel:
    """The label contract, checked against the count the model is built with."""

    def test_the_labels_are_a_contiguous_zero_based_range(self):
        """Non-contiguous labels would index a class table out of bounds."""
        labels = sorted(PROMPT_KIND_TO_LABEL.values())
        assert labels == list(range(len(PROMPT_KIND_TO_LABEL))), labels

    def test_the_count_agrees_with_the_class_embedding_row_count(self):
        """``PROMPT_NUM_CLASSES`` is what ``DiTXA``'s label table is sized by.

        The port deliberately follows the number the MODEL was built with (3),
        not the four caption kinds the upstream encoder declares; the two
        disagree upstream and the discrepancy is recorded, not reconciled. If
        someone adds a fourth entry here without resizing the table, this arm is
        what says so.
        """
        assert len(PROMPT_KIND_TO_LABEL) == PROMPT_NUM_CLASSES == 3

    def test_every_label_is_a_legal_index_into_that_table(self):
        table = keras.layers.Embedding(PROMPT_NUM_CLASSES, 4)
        table.build((None,))
        ids = np.array(sorted(PROMPT_KIND_TO_LABEL.values()), dtype="int32")
        out = np_(table(keras.ops.convert_to_tensor(ids)))
        assert out.shape == (PROMPT_NUM_CLASSES, 4)
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------
# prepare_bridge_batch
# ---------------------------------------------------------------------


class TestPrepareBridgeBatch:
    """The five-tensor seam, including both ablation flags."""

    def test_it_returns_the_real_endpoints_when_no_ablation_is_set(self):
        """The process pair must BE the real pair, bit for bit."""
        records = _records()
        x_0_p, x_1_p, y, x_0, x_1 = prepare_bridge_batch(
            records, CONFIG, seed=SEED
        )
        assert np.array_equal(np_(x_0_p), np_(x_0))
        assert np.array_equal(np_(x_1_p), np_(x_1))
        # x_1 is the latent, verbatim.
        assert np.allclose(np_(x_1), records["latent"], atol=0, rtol=0)
        # x_0 is the PACKED text, which is not the raw flat array; it must equal
        # the packing function's own output rather than merely "have the shape".
        assert np.array_equal(
            np_(x_0),
            np_(token_flat_to_bridge(
                keras.ops.convert_to_tensor(records["text_token_emb"]), CONFIG
            )),
        )
        assert np_(y).tolist() == records["prompt_kind_label"].tolist()
        assert np_(y).dtype == np.int32

    def test_text_as_noise_moves_the_start_point_and_ONLY_the_start_point(self):
        """Pins WHICH tensor moved, not merely that something did.

        The trailing pair is documented to stay REAL so a caller can still
        condition on genuine text while the process runs on noise; an ablation
        that also replaced ``x_0`` would silently make the conditioning stream
        noise too, and a "the output changed" assertion would not notice.
        """
        records = _records()
        config = BridgeConfig(
            token_seq_len=CONFIG.token_seq_len,
            token_emb_dim=CONFIG.token_emb_dim,
            bridge_shape=CONFIG.bridge_shape,
            patch_size=CONFIG.patch_size,
            text_as_noise=True,
        ).validate()
        x_0_p, x_1_p, _, x_0, x_1 = prepare_bridge_batch(
            records, config, seed=SEED
        )
        assert not np.allclose(np_(x_0_p), np_(x_0)), (
            "text_as_noise left the process start point equal to the real text"
        )
        assert np.array_equal(np_(x_1_p), np_(x_1)), (
            "text_as_noise disturbed the IMAGE endpoint"
        )
        assert np.array_equal(np_(x_1), records["latent"])
        assert np.array_equal(
            np_(x_0),
            np_(token_flat_to_bridge(
                keras.ops.convert_to_tensor(records["text_token_emb"]), config
            )),
        ), "text_as_noise corrupted the REAL text endpoint the caller conditions on"

    def test_image_as_noise_moves_the_endpoint_and_ONLY_the_endpoint(self):
        records = _records()
        config = BridgeConfig(
            token_seq_len=CONFIG.token_seq_len,
            token_emb_dim=CONFIG.token_emb_dim,
            bridge_shape=CONFIG.bridge_shape,
            patch_size=CONFIG.patch_size,
            image_as_noise=True,
        ).validate()
        x_0_p, x_1_p, _, x_0, x_1 = prepare_bridge_batch(
            records, config, seed=SEED
        )
        assert not np.allclose(np_(x_1_p), np_(x_1))
        assert np.array_equal(np_(x_0_p), np_(x_0))
        assert np.array_equal(np_(x_1), records["latent"]), (
            "image_as_noise corrupted the REAL latent endpoint"
        )

    def test_the_seed_reproduces_the_ablation_draw(self):
        """Same seed, same noise; different seed, different noise.

        Both halves matter: reproducibility alone is satisfied by a constant.
        """
        records = _records()
        config = BridgeConfig(
            token_seq_len=CONFIG.token_seq_len,
            token_emb_dim=CONFIG.token_emb_dim,
            bridge_shape=CONFIG.bridge_shape,
            patch_size=CONFIG.patch_size,
            text_as_noise=True,
        ).validate()
        a = np_(prepare_bridge_batch(records, config, seed=11)[0])
        b = np_(prepare_bridge_batch(records, config, seed=11)[0])
        c = np_(prepare_bridge_batch(records, config, seed=12)[0])
        assert np.array_equal(a, b), "the same seed gave a different draw"
        assert not np.allclose(a, c), "two seeds gave the identical draw"
