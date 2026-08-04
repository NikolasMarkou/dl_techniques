"""Tests for ``SAM2MemoryBank`` -- plan step 6, guards G6.1 through G6.4.

The bank is a plain-Python state container: no weights, no Keras registration,
no ``get_config``. Everything here is therefore about POLICY and LAYOUT, both of
which are silent when ported wrong -- every mutation guarded below leaves shapes
and dtypes untouched.

The hand-derived selection tables in :class:`TestSelectionPolicy` are written out
literally rather than computed by calling the implementation. Re-deriving them
with the code under test would be a tautology.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.models.sam2.memory_bank import SAM2MemoryBank

# ---------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------

BATCH = 2
GRID = 2
TOKENS_PER_FRAME = GRID * GRID
MEM_DIM = 8
HIDDEN_DIM = 32
# The shipped SAM 2 ratio: 256 // 64 == 4 pseudo-spatial tokens per pointer.
TOKENS_PER_POINTER = HIDDEN_DIM // MEM_DIM
NUM_MASKMEM = 7


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def make_features(frame_idx: int, seed_offset: int = 0):
    """Build a per-frame feature grid whose values identify the frame.

    :param frame_idx: Frame index, folded into the values.
    :type frame_idx: int
    :param seed_offset: Extra offset so features and encodings differ.
    :type seed_offset: int
    :return: ``(BATCH, GRID, GRID, MEM_DIM)`` float32 tensor.
    :rtype: Any
    """
    rng = np.random.default_rng(1000 + frame_idx * 7 + seed_offset)
    base = rng.standard_normal((BATCH, GRID, GRID, MEM_DIM)).astype("float32")
    return ops.convert_to_tensor(base + float(frame_idx))


def make_pointer(frame_idx: int):
    """Build an object pointer whose values identify the frame.

    :param frame_idx: Frame index, folded into the values.
    :type frame_idx: int
    :return: ``(BATCH, HIDDEN_DIM)`` float32 tensor.
    :rtype: Any
    """
    rng = np.random.default_rng(5000 + frame_idx * 13)
    base = rng.standard_normal((BATCH, HIDDEN_DIM)).astype("float32")
    return ops.convert_to_tensor(base + 100.0 * float(frame_idx))


def populate(bank, frames, cond_frames=(), with_pointers=True):
    """Fill ``bank`` with a scripted frame history.

    :param bank: The bank to fill.
    :type bank: SAM2MemoryBank
    :param frames: Frame indices to store as non-conditioning.
    :type frames: Any
    :param cond_frames: Frame indices to store as conditioning.
    :type cond_frames: Any
    :param with_pointers: Attach an object pointer to every frame.
    :type with_pointers: bool
    :return: The same bank, for chaining.
    :rtype: SAM2MemoryBank
    """
    for frame_idx in list(cond_frames) + list(frames):
        bank.add_frame(
            frame_idx,
            make_features(frame_idx),
            make_features(frame_idx, seed_offset=1),
            obj_ptr=make_pointer(frame_idx) if with_pointers else None,
            is_conditioning=frame_idx in cond_frames,
        )
    return bank


@pytest.fixture
def bank():
    """A bank wide enough to hold the whole 20-frame script.

    ``fifo_capacity`` is pinned here so the selection-policy tables are not
    entangled with eviction; eviction has its own tests.

    :return: An empty configured bank.
    :rtype: SAM2MemoryBank
    """
    return SAM2MemoryBank(
        num_maskmem=NUM_MASKMEM,
        mem_dim=MEM_DIM,
        hidden_dim=HIDDEN_DIM,
        fifo_capacity=64,
    )


# ---------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------


class TestConstruction:
    """Configuration validation and derived sizes."""

    def test_is_not_a_keras_layer_and_is_not_registered(self):
        """The bank must stay a plain object: no weights, no registry key.

        Making it a ``keras.layers.Layer`` would give it a serialized config
        that duplicates state ``SAM2`` already owns, and would mint a registry
        key the model never reads.
        """
        instance = SAM2MemoryBank()
        assert not isinstance(instance, keras.layers.Layer)
        assert not hasattr(instance, "get_config")
        assert not hasattr(instance, "weights")
        registered = keras.saving.get_registered_object("SAM2MemoryBank")
        assert registered is None

    def test_tokens_per_pointer_is_derived(self):
        """``hidden_dim // mem_dim`` is 4 at the shipped 256/64."""
        assert SAM2MemoryBank(
            mem_dim=64, hidden_dim=256).tokens_per_pointer == 4
        assert SAM2MemoryBank(
            mem_dim=MEM_DIM, hidden_dim=HIDDEN_DIM
        ).tokens_per_pointer == TOKENS_PER_POINTER

    @pytest.mark.parametrize("kwargs", [
        {"num_maskmem": 0},
        {"mem_dim": 0},
        {"hidden_dim": 0},
        {"hidden_dim": 100, "mem_dim": 64},
        {"memory_temporal_stride_for_eval": 0},
        {"max_obj_ptrs_in_encoder": -1},
        {"max_cond_frames": -1},
        {"fifo_capacity": 0},
    ])
    def test_invalid_configuration_raises(self, kwargs):
        """Bad sizes raise at construction, never silently at read time."""
        with pytest.raises(ValueError):
            SAM2MemoryBank(**kwargs)

    def test_derived_fifo_capacity_covers_the_selection_reach(self):
        """The default FIFO must never evict a frame the policy can name.

        Derived capacity scales with the STRIDE, not just ``num_maskmem``: at
        stride 2 the policy reaches roughly twice as far back, so a capacity of
        ``num_maskmem`` would silently drop the oldest slots.
        """
        for stride in (1, 2, 3):
            probe = SAM2MemoryBank(
                num_maskmem=NUM_MASKMEM, mem_dim=MEM_DIM,
                hidden_dim=HIDDEN_DIM,
                memory_temporal_stride_for_eval=stride,
            )
            populate(probe, range(0, 40))
            named = [
                probe._previous_frame_index(40, t_pos, stride, False)
                for t_pos in range(1, NUM_MASKMEM)
            ]
            retained = set(probe.non_cond_frames)
            assert set(named) <= retained, (
                f"stride={stride}: policy names {sorted(named)} but the FIFO "
                f"retained only {sorted(retained)}"
            )


# ---------------------------------------------------------------------
# G6.1 -- selection policy
# ---------------------------------------------------------------------


class TestSelectionPolicy:
    """G6.1: the exact selected frame list, against a hand-derived table.

    Every expectation below was derived by hand from
    ``t_rel = num_maskmem - t_pos``, ``t_rel == 1 -> f -+ 1``, and
    ``t_rel >= 2 -> ((f - 2) // s) * s - (t_rel - 2) * s`` (sign-flipped under
    ``track_in_reverse``). None of it is computed by the implementation.
    """

    # (frame_idx, stride, track_in_reverse) -> expected frame list, in
    # memory-sequence order (t_pos ascending: oldest first, previous frame last)
    HAND_TABLE = {
        # f=20, s=1: anchor = ((20-2)//1)*1 = 18; t_rel 6..2 -> 14,15,16,17,18;
        # t_rel 1 -> 19.
        (20, 1, False): [14, 15, 16, 17, 18, 19],
        # f=20, s=2: anchor = ((18)//2)*2 = 18; t_rel 6..2 -> 10,12,14,16,18;
        # t_rel 1 -> 19.
        (20, 2, False): [10, 12, 14, 16, 18, 19],
        # f=5 reverse, s=1: anchor = -((-5-2)//1)*1 = 7; t_rel 6..2 ->
        # 11,10,9,8,7; t_rel 1 -> 6.
        (5, 1, True): [11, 10, 9, 8, 7, 6],
        # f=5 reverse, s=2: anchor = -((-7)//2)*2 = 8; t_rel 6..2 ->
        # 16,14,12,10,8; t_rel 1 -> 6.
        (5, 2, True): [16, 14, 12, 10, 8, 6],
        # f=6 reverse, s=2: anchor = -((-8)//2)*2 = 8; t_rel 6..2 ->
        # 16,14,12,10,8; t_rel 1 -> 7.  (The only row where the t_rel == 1
        # special case is OBSERVABLE under reverse tracking -- see
        # TestSpecialCaseVisibility.)
        (6, 2, True): [16, 14, 12, 10, 8, 7],
    }

    @pytest.mark.parametrize("key", sorted(HAND_TABLE))
    def test_selected_frames_match_the_hand_derived_table(self, bank, key):
        """The selected frame list equals the hand-derived one exactly."""
        frame_idx, stride, reverse = key
        populate(bank, range(0, 20))
        selections = bank.select_frames(
            frame_idx, track_in_reverse=reverse, stride=stride)
        got = [s.frame_idx for s in selections]
        assert got == self.HAND_TABLE[key], (
            f"frame_idx={frame_idx} stride={stride} reverse={reverse}: "
            f"selected {got}, hand-derived table says {self.HAND_TABLE[key]}"
        )

    def test_t_pos_runs_one_through_num_maskmem_minus_one(self, bank):
        """Non-conditioning slots occupy ``t_pos`` 1..num_maskmem-1 in order."""
        populate(bank, range(0, 20))
        selections = bank.select_frames(20)
        assert [s.t_pos for s in selections] == list(range(1, NUM_MASKMEM))

    def test_slot_index_is_num_maskmem_minus_t_pos_minus_one(self, bank):
        """The returned ``maskmem_tpos_enc`` row is ``N - t_pos - 1``."""
        populate(bank, range(0, 20))
        selections = bank.select_frames(20)
        for selection in selections:
            assert selection.tpos_slot == NUM_MASKMEM - selection.t_pos - 1
        # The most recent frame lands in slot 0, the furthest in slot 5.
        assert selections[-1].tpos_slot == 0
        assert selections[0].tpos_slot == NUM_MASKMEM - 2

    def test_missing_frames_are_skipped_not_padded(self, bank):
        """Early in a video the sequence is SHORTER, never zero-padded."""
        populate(bank, [0, 1])
        selections = bank.select_frames(2)
        assert [s.frame_idx for s in selections] == [0, 1]

    def test_stride_below_one_raises(self, bank):
        """A zero stride would divide by zero inside the policy."""
        with pytest.raises(ValueError):
            bank.select_frames(20, stride=0)


class TestSpecialCaseVisibility:
    """G6.1's mutation, and WHERE it is observable.

    ``t_rel == 1`` is a special case that always takes the immediately preceding
    frame. Folding it into the general formula gives
    ``((f - 2) // s) * s + s``. These tests pin exactly which configurations can
    tell the two apart, so a future author cannot "simplify" the branch away and
    point at a green stride-1 suite.
    """

    @staticmethod
    def _folded(frame_idx, stride, reverse):
        """The general formula evaluated at ``t_rel == 1``.

        :param frame_idx: Query frame.
        :type frame_idx: int
        :param stride: Temporal stride.
        :type stride: int
        :param reverse: Track backwards in time.
        :type reverse: bool
        :return: The frame the folded formula would name.
        :rtype: int
        """
        if not reverse:
            return ((frame_idx - 2) // stride) * stride + stride
        return -(((-frame_idx) - 2) // stride) * stride - stride

    def test_the_special_case_is_INVISIBLE_at_stride_one(self):
        """Measured: at stride 1 the folded formula names the SAME frame.

        The plan predicted the mutation fires at stride 2; this test records the
        other half of that prediction, that a stride-1 test is BLIND to it. Both
        directions of tracking are blind.
        """
        for frame_idx in range(3, 25):
            assert self._folded(frame_idx, 1, False) == frame_idx - 1
            assert self._folded(frame_idx, 1, True) == frame_idx + 1

    def test_the_special_case_is_visible_at_stride_two(self):
        """At stride 2 the folded formula names a different frame."""
        # Forward: f=20 -> folded names 20 (the CURRENT frame), not 19.
        assert self._folded(20, 2, False) == 20
        # Reverse: f=6 -> folded names 6 (the current frame), not 7.
        assert self._folded(6, 2, True) == 6

    def test_reverse_stride_two_is_blind_at_ODD_query_frames(self):
        """Measured, and NOT predicted by the plan: parity decides visibility.

        Under reverse tracking at stride 2 the folded formula agrees with the
        special case whenever ``frame_idx`` is odd, so the reverse arm of G6.1
        must query an EVEN frame to have any discriminating power. The
        hand-derived table's ``(5, 2, True)`` row is deliberately one of the
        blind ones and ``(6, 2, True)`` is the row that fires.
        """
        assert self._folded(5, 2, True) == 6  # blind: equals 5 + 1
        assert self._folded(7, 2, True) == 8  # blind: equals 7 + 1
        assert self._folded(6, 2, True) == 6  # visible: 6 != 6 + 1
        assert self._folded(8, 2, True) == 8  # visible: 8 != 8 + 1


# ---------------------------------------------------------------------
# G6.3 -- conditioning frames
# ---------------------------------------------------------------------


class TestConditioningFrames:
    """G6.3: conditioning frames sit in ``t_pos = 0``, however distant."""

    @pytest.mark.parametrize("cond_idx", [0, 1, 5])
    def test_conditioning_t_pos_is_zero_regardless_of_distance(
            self, bank, cond_idx):
        """A prompt 20 frames ago still occupies bucket 0.

        A distance-derived ``t_pos`` would produce a plausible model that
        quietly forgets the prompt, with no shape error anywhere.
        """
        populate(bank, range(cond_idx + 1, 20), cond_frames=[cond_idx])
        selections = bank.select_frames(20)
        cond = [s for s in selections if s.is_conditioning]
        assert len(cond) == 1
        assert cond[0].frame_idx == cond_idx
        assert cond[0].t_pos == 0, (
            f"conditioning frame {cond_idx} is {20 - cond_idx} frames away and "
            f"still must have t_pos == 0, got {cond[0].t_pos}"
        )
        assert cond[0].tpos_slot == NUM_MASKMEM - 1

    def test_all_conditioning_frames_share_slot_and_come_first(self, bank):
        """Every conditioning frame gets the same slot, ahead of the FIFO."""
        populate(bank, range(6, 20), cond_frames=[0, 3])
        selections = bank.select_frames(20)
        assert [s.is_conditioning for s in selections][:2] == [True, True]
        assert not any(s.is_conditioning for s in selections[2:])
        assert {s.tpos_slot for s in selections[:2]} == {NUM_MASKMEM - 1}
        assert [s.frame_idx for s in selections[:2]] == [0, 3]

    def test_max_cond_frames_keeps_the_nearest(self, bank):
        """Capping conditioning frames retains the temporally nearest ones."""
        capped = SAM2MemoryBank(
            num_maskmem=NUM_MASKMEM, mem_dim=MEM_DIM, hidden_dim=HIDDEN_DIM,
            fifo_capacity=64, max_cond_frames=2,
        )
        populate(capped, range(10, 20), cond_frames=[0, 3, 9])
        cond = [s.frame_idx
                for s in capped.select_frames(20) if s.is_conditioning]
        assert cond == [3, 9]

    def test_a_frame_promoted_to_conditioning_leaves_the_fifo(self, bank):
        """Re-adding a frame as conditioning must not double-count it."""
        populate(bank, [4])
        assert 4 in bank.non_cond_frames
        bank.add_frame(4, make_features(4), make_features(4, 1),
                       obj_ptr=make_pointer(4), is_conditioning=True)
        assert 4 in bank.cond_frames
        assert 4 not in bank.non_cond_frames
        assert bank.num_frames == 1


# ---------------------------------------------------------------------
# G6.2 -- object pointers at the tail
# ---------------------------------------------------------------------


class TestObjectPointerLayout:
    """G6.2: pointers land at the TAIL, ``4 * n_pointers`` tokens wide.

    This is what makes step 2's ``num_k_exclude`` correct end to end. Memory
    attention excludes exactly ``num_obj_ptr_tokens`` TRAILING key rows from
    rotary embedding; a prepend would exclude the wrong rows -- rotating the
    pointers (which must stay unrotated) and un-rotating the oldest spatial
    frame -- with identical shapes and no error.
    """

    def test_token_count_is_four_times_the_pointer_count(self, bank):
        """``num_obj_ptr_tokens == tokens_per_pointer * n_pointers``."""
        populate(bank, range(0, 20), cond_frames=[0])
        readout = bank.read(20)
        n_pointers = len(readout.obj_ptr_frames)
        assert n_pointers > 0
        assert readout.num_obj_ptr_tokens == (
            TOKENS_PER_POINTER * n_pointers)

    def test_pointer_tokens_occupy_the_tail_slice(self, bank):
        """The last ``num_obj_ptr_tokens`` rows are exactly the split pointers.

        Compares by VALUE against an independently built reshape of the same
        pointers, so a prepend fails the tail slice AND the leading spatial
        slice.
        """
        populate(bank, range(0, 20), cond_frames=[0])
        readout = bank.read(20)
        memory = ops.convert_to_numpy(readout.memory)

        expected = np.concatenate(
            [ops.convert_to_numpy(make_pointer(idx)).reshape(
                BATCH, TOKENS_PER_POINTER, MEM_DIM)
             for idx in readout.obj_ptr_frames],
            axis=1,
        )
        tail = memory[:, -readout.num_obj_ptr_tokens:, :]
        assert np.max(np.abs(tail - expected)) == 0.0, (
            "object-pointer tokens are not the TAIL of the memory sequence"
        )

        # ...and the head is spatial, not pointer data.
        num_spatial = sum(readout.frame_token_counts)
        assert num_spatial + readout.num_obj_ptr_tokens == memory.shape[1]
        head = memory[:, :num_spatial, :]
        first = ops.convert_to_numpy(
            bank._entry(readout.selections[0]).features)
        assert np.max(np.abs(head[:, :TOKENS_PER_FRAME, :] - first)) == 0.0

    def test_pointer_split_is_contiguous_chunks_of_mem_dim(self, bank):
        """One ``hidden_dim`` vector becomes ``tokens_per_pointer`` chunks.

        Token ``p * tokens_per_pointer + s`` must be pointer ``p``'s ``s``-th
        CONTIGUOUS ``mem_dim``-wide slice -- an interleaved split would give the
        same shape and different values.
        """
        populate(bank, [], cond_frames=[0, 1])
        readout = bank.read(2)
        memory = ops.convert_to_numpy(readout.memory)
        # The pointer block starts AFTER every spatial frame -- including the
        # conditioning ones, which are spatial memory too.
        offset = sum(readout.frame_token_counts)
        for p, frame in enumerate(readout.obj_ptr_frames):
            raw = ops.convert_to_numpy(make_pointer(frame))
            for s in range(TOKENS_PER_POINTER):
                token = memory[:, offset + p * TOKENS_PER_POINTER + s, :]
                chunk = raw[:, s * MEM_DIM:(s + 1) * MEM_DIM]
                assert np.max(np.abs(token - chunk)) == 0.0

    def test_pointer_pos_enc_rows_are_zero(self, bank):
        """Pointer rows carry no SPATIAL encoding (H-13/H-14).

        Their temporal encoding is the caller's to build from
        ``obj_ptr_tpos``; the bank must not smuggle a spatial one in.
        """
        populate(bank, range(0, 20), cond_frames=[0])
        readout = bank.read(20)
        pos = ops.convert_to_numpy(readout.memory_pos)
        tail = pos[:, -readout.num_obj_ptr_tokens:, :]
        assert np.max(np.abs(tail)) == 0.0

    def test_temporal_positions_are_repeat_interleaved(self, bank):
        """Each pointer's 4 sub-tokens share ONE temporal difference.

        ``repeat_interleave``, not ``tile``: the sequence must be
        ``[d0, d0, d0, d0, d1, d1, ...]``, never ``[d0, d1, ..., d0, d1, ...]``.
        """
        populate(bank, range(0, 20), cond_frames=[0])
        readout = bank.read(20)
        assert len(readout.obj_ptr_tpos) == readout.num_obj_ptr_tokens
        for p, frame in enumerate(readout.obj_ptr_frames):
            block = readout.obj_ptr_tpos[
                p * TOKENS_PER_POINTER:(p + 1) * TOKENS_PER_POINTER]
            assert set(block) == {float(abs(20 - frame))}

    def test_signed_temporal_positions_when_configured(self):
        """``use_signed_tpos_enc_to_obj_ptrs`` reports a signed difference."""
        signed = SAM2MemoryBank(
            num_maskmem=NUM_MASKMEM, mem_dim=MEM_DIM, hidden_dim=HIDDEN_DIM,
            fifo_capacity=64, use_signed_tpos_enc_to_obj_ptrs=True)
        populate(signed, [], cond_frames=[8])
        forward = signed.read(4).obj_ptr_tpos[0]
        reverse = signed.read(4, track_in_reverse=True).obj_ptr_tpos[0]
        assert forward == -4.0
        assert reverse == 4.0

    def test_pointer_cap_is_honoured(self, bank):
        """``max_obj_ptrs`` bounds the tail block."""
        populate(bank, range(0, 20), cond_frames=[0])
        readout = bank.read(20, max_obj_ptrs=3)
        assert len(readout.obj_ptr_frames) == 3
        assert readout.num_obj_ptr_tokens == 3 * TOKENS_PER_POINTER

    def test_no_pointers_gives_a_pure_spatial_sequence(self, bank):
        """Without pointers the exclusion count is 0 and the tail is spatial."""
        populate(bank, range(0, 20), with_pointers=False)
        readout = bank.read(20)
        assert readout.num_obj_ptr_tokens == 0
        assert readout.obj_ptr_tpos == ()
        assert readout.memory.shape[1] == sum(readout.frame_token_counts)


# ---------------------------------------------------------------------
# readout assembly
# ---------------------------------------------------------------------


class TestReadout:
    """Sequence assembly, widths, and the caller-facing contract."""

    def test_memory_width_is_mem_dim_and_length_is_the_token_sum(self, bank):
        """Spatial tokens plus pointer tokens, all ``mem_dim`` wide."""
        populate(bank, range(0, 20), cond_frames=[0])
        readout = bank.read(20)
        expected = sum(readout.frame_token_counts) + readout.num_obj_ptr_tokens
        assert readout.memory.shape == (BATCH, expected, MEM_DIM)
        assert readout.memory_pos.shape == readout.memory.shape

    def test_slot_indices_align_with_the_spatial_frames(self, bank):
        """One slot index and one token count per spatial frame, in order."""
        populate(bank, range(0, 20), cond_frames=[0])
        readout = bank.read(20)
        assert len(readout.tpos_slots) == len(readout.selections)
        assert len(readout.frame_token_counts) == len(readout.selections)
        assert readout.tpos_slots == tuple(
            s.tpos_slot for s in readout.selections)
        assert set(readout.frame_token_counts) == {TOKENS_PER_FRAME}
        assert all(0 <= slot < NUM_MASKMEM for slot in readout.tpos_slots)

    def test_empty_bank_returns_no_memory(self, bank):
        """The first prompted frame has nothing to read."""
        readout = bank.read(0)
        assert bank.is_empty
        assert readout.memory is None
        assert readout.memory_pos is None
        assert readout.num_obj_ptr_tokens == 0
        assert readout.selections == ()

    def test_flattened_and_gridded_inputs_agree(self):
        """``(B, H, W, C)`` and ``(B, H * W, C)`` inputs store identically."""
        grid_bank = SAM2MemoryBank(mem_dim=MEM_DIM, hidden_dim=HIDDEN_DIM,
                                   fifo_capacity=8)
        flat_bank = SAM2MemoryBank(mem_dim=MEM_DIM, hidden_dim=HIDDEN_DIM,
                                   fifo_capacity=8)
        features = make_features(3)
        flat = ops.reshape(features, (BATCH, TOKENS_PER_FRAME, MEM_DIM))
        grid_bank.add_frame(3, features, features)
        flat_bank.add_frame(3, flat, flat)
        a = ops.convert_to_numpy(grid_bank.read(4).memory)
        b = ops.convert_to_numpy(flat_bank.read(4).memory)
        assert np.max(np.abs(a - b)) == 0.0

    @pytest.mark.parametrize("bad", ["rank", "width", "mismatch", "pointer"])
    def test_malformed_inputs_raise(self, bank, bad):
        """Wrong ranks and widths raise at insertion, not at read time."""
        good = make_features(0)
        if bad == "rank":
            with pytest.raises(ValueError):
                bank.add_frame(0, ops.reshape(good, (-1,)), good)
        elif bad == "width":
            wide = ops.concatenate([good, good], axis=-1)
            with pytest.raises(ValueError):
                bank.add_frame(0, wide, wide)
        elif bad == "mismatch":
            other = ops.reshape(good, (BATCH, TOKENS_PER_FRAME, MEM_DIM))
            with pytest.raises(ValueError):
                bank.add_frame(
                    0, other, ops.concatenate([other, other], axis=1))
        else:
            with pytest.raises(ValueError):
                bank.add_frame(0, good, good,
                               obj_ptr=ops.zeros((BATCH, HIDDEN_DIM + 1)))


# ---------------------------------------------------------------------
# purity, determinism, gradient boundary
# ---------------------------------------------------------------------


class TestPurityAndState:
    """The bank is weightless and pure: querying must not mutate it."""

    def test_read_is_deterministic(self, bank):
        """Two identical reads over identical state return identical results."""
        populate(bank, range(0, 20), cond_frames=[0])
        first = bank.read(20)
        second = bank.read(20)
        assert first.selections == second.selections
        assert first.tpos_slots == second.tpos_slots
        assert first.obj_ptr_tpos == second.obj_ptr_tpos
        assert np.max(np.abs(
            ops.convert_to_numpy(first.memory)
            - ops.convert_to_numpy(second.memory))) == 0.0

    def test_read_does_not_mutate_the_fifo(self, bank):
        """Querying is read-only over both stores."""
        populate(bank, range(0, 20), cond_frames=[0])
        before_cond = sorted(bank.cond_frames)
        before_non_cond = sorted(bank.non_cond_frames)
        for frame_idx in (5, 12, 20):
            bank.read(frame_idx)
            bank.read(frame_idx, track_in_reverse=True)
            bank.select_frames(frame_idx, stride=3)
        assert sorted(bank.cond_frames) == before_cond
        assert sorted(bank.non_cond_frames) == before_non_cond

    def test_fifo_evicts_the_oldest_first(self):
        """The FIFO drops the lowest frame index once past capacity."""
        small = SAM2MemoryBank(mem_dim=MEM_DIM, hidden_dim=HIDDEN_DIM,
                               fifo_capacity=3)
        populate(small, range(0, 6))
        assert sorted(small.non_cond_frames) == [3, 4, 5]

    def test_conditioning_frames_are_never_evicted(self):
        """Prompts outlive the FIFO however long the video runs."""
        small = SAM2MemoryBank(mem_dim=MEM_DIM, hidden_dim=HIDDEN_DIM,
                               fifo_capacity=2)
        populate(small, range(1, 30), cond_frames=[0])
        assert sorted(small.cond_frames) == [0]
        assert small.num_frames == 3

    def test_reset_clears_every_store(self, bank):
        """One bank per video: ``reset`` must leave nothing behind."""
        populate(bank, range(0, 5), cond_frames=[0])
        bank.reset()
        assert bank.is_empty
        assert bank.num_frames == 0
        assert bank.read(6).memory is None

    def test_stored_state_is_detached(self, bank):
        """H-4: memory crossing a frame boundary carries no gradient.

        Without this, N frames of propagation build one N-deep recurrent graph
        instead of N independent decodes.
        """
        variable = keras.Variable(
            np.ones((BATCH, GRID, GRID, MEM_DIM), dtype="float32"))
        with tf.GradientTape() as tape:
            live = variable * 3.0
            bank.add_frame(0, live, live, obj_ptr=None)
            out = ops.sum(bank.read(1).memory)
        assert tape.gradient(out, variable) is None

    def test_stop_gradient_can_be_disabled(self):
        """The boundary is a deliberate choice, so it is observable."""
        leaky = SAM2MemoryBank(mem_dim=MEM_DIM, hidden_dim=HIDDEN_DIM,
                               fifo_capacity=4, stop_gradient=False)
        variable = keras.Variable(
            np.ones((BATCH, GRID, GRID, MEM_DIM), dtype="float32"))
        with tf.GradientTape() as tape:
            live = variable * 3.0
            leaky.add_frame(0, live, live, obj_ptr=None)
            out = ops.sum(leaky.read(1).memory)
        assert tape.gradient(out, variable) is not None


# ---------------------------------------------------------------------
# G6.4 -- dead-component probes
# ---------------------------------------------------------------------


class TestDeadComponentProbes:
    """G6.4: measure what each guard can and cannot see.

    A guard that stays green when the thing it guards is DEAD proves nothing.
    Each probe below removes one component and names the guard that goes red,
    and -- just as importantly -- the guards that do NOT.
    """

    def test_an_empty_fifo_kills_the_selection_guard_only(self, bank):
        """Dead FIFO: G6.1 goes red, G6.3 stays green.

        The conditioning guard cannot see a dead non-conditioning store, which
        is why the two are separate tests.
        """
        populate(bank, range(0, 20), cond_frames=[0])
        bank.non_cond_frames = {}
        selections = bank.select_frames(20)
        assert [s.frame_idx for s in selections] != [14, 15, 16, 17, 18, 19]
        # G6.3's observation survives.
        assert [s.t_pos for s in selections if s.is_conditioning] == [0]

    def test_an_empty_cond_store_kills_the_conditioning_guard_only(self, bank):
        """Dead conditioning store: G6.3 goes red, G6.1 stays green."""
        populate(bank, range(0, 20), cond_frames=[0])
        bank.cond_frames = {}
        selections = bank.select_frames(20)
        assert not any(s.is_conditioning for s in selections)
        # G6.1's observation survives untouched.
        assert [s.frame_idx for s in selections] == [14, 15, 16, 17, 18, 19]

    def test_dropping_pointers_kills_the_tail_guard_only(self, bank):
        """Dead pointer store: G6.2 goes red, G6.1 and G6.3 stay green."""
        populate(bank, range(0, 20), cond_frames=[0], with_pointers=False)
        readout = bank.read(20)
        assert readout.num_obj_ptr_tokens == 0
        # G6.1 and G6.3 are structurally blind to a dead pointer store.
        assert [s.frame_idx for s in readout.selections
                if not s.is_conditioning] == [14, 15, 16, 17, 18, 19]
        assert readout.selections[0].t_pos == 0

    def test_zeroed_features_are_invisible_to_all_three_policy_guards(
            self, bank):
        """Measured partition: G6.1-G6.3 are POLICY guards, not value guards.

        Replacing every stored feature with zeros leaves the selection list, the
        slot indices, the pointer count and the tail position all unchanged.
        Any future value-level claim about the memory contents needs its own
        assertion; these three cannot carry it.
        """
        populate(bank, range(0, 20), cond_frames=[0])
        live = bank.read(20)
        for store in (bank.cond_frames, bank.non_cond_frames):
            for key, entry in list(store.items()):
                store[key] = entry._replace(
                    features=ops.zeros_like(entry.features),
                    pos_enc=ops.zeros_like(entry.pos_enc),
                )
        dead = bank.read(20)
        assert dead.selections == live.selections
        assert dead.tpos_slots == live.tpos_slots
        assert dead.num_obj_ptr_tokens == live.num_obj_ptr_tokens
        assert np.max(np.abs(ops.convert_to_numpy(
            dead.memory[:, -dead.num_obj_ptr_tokens:, :]
            - live.memory[:, -live.num_obj_ptr_tokens:, :]))) == 0.0
        # ...and the spatial head IS dead, which nothing above noticed.
        num_spatial = sum(dead.frame_token_counts)
        assert np.max(np.abs(ops.convert_to_numpy(
            dead.memory[:, :num_spatial, :]))) == 0.0


# ---------------------------------------------------------------------
# integration with step 2
# ---------------------------------------------------------------------


class TestMemoryAttentionContract:
    """The readout is consumable by ``SAM2MemoryAttention`` unchanged."""

    def test_readout_feeds_memory_attention(self):
        """End-to-end: bank -> ``num_obj_ptr_tokens`` -> RoPE exclusion.

        ``num_k_exclude`` counts TRAILING key rows, so this only stays correct
        while the bank keeps pointers at the tail (G6.2).
        """
        from dl_techniques.models.sam2.memory_attention import (
            SAM2MemoryAttention,
        )

        d_model, mem_dim, feat = 16, 4, (2, 2)
        live_bank = SAM2MemoryBank(
            num_maskmem=4, mem_dim=mem_dim, hidden_dim=4 * mem_dim,
            fifo_capacity=8)
        for frame_idx in range(4):
            grid = ops.convert_to_tensor(
                np.random.default_rng(frame_idx).standard_normal(
                    (1, feat[0], feat[1], mem_dim)).astype("float32"))
            live_bank.add_frame(
                frame_idx, grid, grid,
                obj_ptr=ops.convert_to_tensor(
                    np.random.default_rng(100 + frame_idx).standard_normal(
                        (1, 4 * mem_dim)).astype("float32")),
                is_conditioning=frame_idx == 0,
            )
        readout = live_bank.read(4)
        assert readout.num_obj_ptr_tokens == 4 * len(readout.obj_ptr_frames)

        attention = SAM2MemoryAttention(
            d_model=d_model, num_layers=1, num_heads=1, kv_in_dim=mem_dim,
            dim_feedforward=32, dropout=0.0, feat_sizes=feat,
        )
        features = ops.convert_to_tensor(
            np.random.default_rng(0).standard_normal(
                (1, feat[0] * feat[1], d_model)).astype("float32"))
        out = attention(
            features, readout.memory,
            features_pos=ops.zeros_like(features),
            memory_pos=readout.memory_pos,
            num_obj_ptr_tokens=readout.num_obj_ptr_tokens,
        )
        assert out.shape == features.shape
        assert np.all(np.isfinite(ops.convert_to_numpy(out)))
