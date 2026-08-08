"""
SAM 2 memory bank: the per-video streaming state container.
===========================================================

:class:`SAM2MemoryBank` holds the state SAM 2's online tracker carries from
frame to frame -- the conditioning-frame store, the non-conditioning frame
FIFO, and the object-pointer store. It implements the temporal frame-selection
policy and assembles the memory sequence ``SAM2MemoryAttention`` reads as keys
and values.

Based on:
---------
- Ravi, N. et al. (2024). "SAM 2: Segment Anything in Images and Videos."

Key Features:
------------
- **This is NOT a Keras layer.** It is a plain-Python object: no weights, no
  ``@keras.saving.register_keras_serializable`` decorator, no ``get_config``.
- Everything learned lives on the top-level ``SAM2`` model:
  ``maskmem_tpos_enc``, the ``(num_maskmem, 1, 1, mem_dim)`` learned per-slot
  temporal embedding, and the ``no_mem_embed`` / ``no_obj_ptr`` no-object
  mechanisms.
- The bank returns the SLOT INDICES ``num_maskmem - t_pos - 1`` that select
  rows of that weight; it never adds the embedding itself.

Architecture Overview:
---------------------
1. ``add_conditioning_frame`` / ``add_frame`` -- the two stores plus the FIFO.
2. ``select_frames`` / ``select_object_pointer_frames`` -- the temporal policy.
3. ``assemble`` -- the memory sequence, with object pointers appended.

Usage Examples:
--------------
```python
from dl_techniques.models.SAM.SAM2 import SAM2MemoryBank
bank = SAM2MemoryBank(num_maskmem=7, mem_dim=64)
memory, memory_pos, num_obj_ptr_tokens = bank.assemble(frame_index=3)
```

Measured caveats:
----------------
- The weightless/stateful split is deliberate, and it is what makes the spatial
  / temporal separation testable at all. RoPE inside memory attention carries
  SPATIAL position only, broadcast identically across every memory frame; the
  temporal distinction is carried EXCLUSIVELY by the additive per-slot
  embedding whose indices this bank hands back. If the bank added the embedding
  itself, no test could tell the two mechanisms apart.
- **Not to be confused with** ``src/dl_techniques/models/memory_bank/``. That
  package is ``WaveFieldMemoryLLM``'s keyed read/write store for language
  modelling -- a different data structure with a colliding name. It was
  reviewed and REJECTED as a reuse target here; nothing in this file derives
  from it.
- Two mechanisms are SILENT when ported wrong -- shapes are identical either
  way -- and both are guarded behaviourally in
  ``tests/test_models/test_sam2/test_memory_bank.py``:

  1. **Object-pointer tokens sit at the TAIL of the memory sequence.** Memory
     attention excludes exactly ``num_obj_ptr_tokens`` TRAILING key rows from
     rotary embedding. Prepending them instead would exclude the wrong rows:
     the pointers would get spatial rotation they must not have, and the oldest
     spatial frame would lose the rotation it needs. No shape changes.
  2. **Conditioning frames always occupy temporal slot ``t_pos = 0``**, however
     far away in time they are -- they are always maximally relevant, unlike
     the recency-decayed non-conditioning slots. Deriving ``t_pos`` from the
     temporal distance yields a plausible model that quietly forgets the
     prompt.
"""

from keras import ops
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


class _FrameMemory(NamedTuple):
    """One stored frame's memory entry.

    :param features: Spatial memory features, ``(batch, num_tokens, mem_dim)``.
    :type features: Any
    :param pos_enc: Spatial positional encoding, same shape as ``features``.
    :type pos_enc: Any
    :param obj_ptr: Object pointer, ``(batch, hidden_dim)``, or ``None``.
    :type obj_ptr: Optional[Any]
    """

    features: Any
    pos_enc: Any
    obj_ptr: Optional[Any]


class _Selection(NamedTuple):
    """One selected spatial memory frame, in memory-sequence order.

    :param frame_idx: Index of the selected frame.
    :type frame_idx: int
    :param t_pos: Temporal bucket, ``0`` for conditioning frames and
        ``1 .. num_maskmem - 1`` for non-conditioning ones.
    :type t_pos: int
    :param tpos_slot: Row of ``maskmem_tpos_enc`` to add, defined as
        ``num_maskmem - t_pos - 1``.
    :type tpos_slot: int
    :param is_conditioning: Whether this frame came from the conditioning store.
    :type is_conditioning: bool
    """

    frame_idx: int
    t_pos: int
    tpos_slot: int
    is_conditioning: bool


class _MemoryReadout(NamedTuple):
    """The assembled memory sequence for one frame.

    Returned by :meth:`SAM2MemoryBank.read`. Deliberately a private NamedTuple
    rather than a public class: it is a return shape, not an abstraction, and
    the plan's public-class budget is spent on :class:`SAM2MemoryBank` alone.

    :param memory: ``(batch, num_tokens, mem_dim)`` -- spatial frame tokens
        followed by object-pointer tokens -- or ``None`` when the bank is empty.
    :type memory: Optional[Any]
    :param memory_pos: Positional encoding aligned with ``memory``. The
        object-pointer rows are ZEROS: their temporal encoding is the caller's
        to build from ``obj_ptr_tpos``.
    :type memory_pos: Optional[Any]
    :param selections: One :class:`_Selection` per spatial memory frame, in
        memory-sequence order.
    :type selections: Tuple[_Selection, ...]
    :param tpos_slots: ``selection.tpos_slot`` for each spatial frame, in the
        same order -- the rows of ``maskmem_tpos_enc`` the caller must add.
    :type tpos_slots: Tuple[int, ...]
    :param frame_token_counts: Number of memory tokens contributed by each
        spatial frame, so a caller can expand ``tpos_slots`` per token.
    :type frame_token_counts: Tuple[int, ...]
    :param num_obj_ptr_tokens: Count of trailing object-pointer tokens, equal to
        ``tokens_per_pointer * len(obj_ptr_frames)``. Feed this straight to
        ``SAM2MemoryAttention(..., num_obj_ptr_tokens=...)``.
    :type num_obj_ptr_tokens: int
    :param obj_ptr_frames: Frame index of each contributing object pointer.
    :type obj_ptr_frames: Tuple[int, ...]
    :param obj_ptr_tpos: PER-TOKEN temporal difference for the object-pointer
        tokens -- already ``repeat_interleave``\\ d ``tokens_per_pointer`` times,
        so its length equals ``num_obj_ptr_tokens``.
    :type obj_ptr_tpos: Tuple[float, ...]
    """

    memory: Optional[Any]
    memory_pos: Optional[Any]
    selections: Tuple[_Selection, ...]
    tpos_slots: Tuple[int, ...]
    frame_token_counts: Tuple[int, ...]
    num_obj_ptr_tokens: int
    obj_ptr_frames: Tuple[int, ...]
    obj_ptr_tpos: Tuple[float, ...]


class SAM2MemoryBank:
    """Streaming memory state for one tracked object in one video.

    Holds three stores and implements the temporal selection policy over them:

    * **conditioning frames** -- frames that received a prompt. Kept
      indefinitely and always assigned ``t_pos = 0``.
    * **non-conditioning frames** -- a bounded FIFO of previously tracked
      frames. Its capacity is DERIVED from ``num_maskmem`` and the temporal
      stride so that no frame the selection policy can reach is ever evicted.
    * **object pointers** -- one ``hidden_dim``-wide vector per frame, split
      into ``hidden_dim // mem_dim`` tokens of width ``mem_dim`` and appended at
      the TAIL of the memory sequence.

    **Selection policy.** For a query frame ``f``, the ``num_maskmem - 1``
    non-conditioning slots take ``t_pos in 1 .. num_maskmem - 1`` with
    ``t_rel = num_maskmem - t_pos`` ("how many frames back"). ``t_rel == 1`` is a
    SPECIAL CASE that always takes the immediately preceding frame ``f - 1``
    regardless of stride; ``t_rel >= 2`` takes
    ``((f - 2) // stride) * stride - (t_rel - 2) * stride``. ``track_in_reverse``
    flips the sign of both arms. Do not fold the special case into the general
    formula: at ``stride = 1`` the two agree, so a stride-1 test cannot see the
    difference, and at ``stride >= 2`` the general formula silently drops the
    most recent frame.

    Frames the policy names but the stores do not hold are skipped, not padded.

    **Gradient boundary.** Stored tensors pass through ``ops.stop_gradient`` by
    default (H-4). Without it, N frames of memory propagation build one
    N-deep recurrent graph instead of N cheap decodes.

    :param num_maskmem: Number of spatial memory slots, including the
        conditioning bucket. Shipped value ``7``.
    :type num_maskmem: int
    :param mem_dim: Width of a memory token. Shipped value ``64``.
    :type mem_dim: int
    :param hidden_dim: Width of an object pointer before splitting. Shipped
        value ``256``; must be a positive multiple of ``mem_dim``.
    :type hidden_dim: int
    :param memory_temporal_stride_for_eval: Default temporal subsampling stride
        for non-conditioning selection. ``1`` during training.
    :type memory_temporal_stride_for_eval: int
    :param max_obj_ptrs_in_encoder: Cap on the number of object pointers fed to
        memory attention.
    :type max_obj_ptrs_in_encoder: int
    :param max_cond_frames: Cap on conditioning frames used per read; ``None``
        uses all of them.
    :type max_cond_frames: Optional[int]
    :param use_signed_tpos_enc_to_obj_ptrs: Report signed rather than absolute
        temporal differences for object pointers.
    :type use_signed_tpos_enc_to_obj_ptrs: bool
    :param only_obj_ptrs_in_the_past_for_eval: Restrict CONDITIONING-frame
        object pointers to frames at or before the queried frame (at or after
        it when tracking in reverse). Shipped value ``True``. Without it a
        conditioning frame from the FUTURE contributes its pointer to the
        memory of an earlier frame, which is a silent information leak: the
        shapes, the token count and the loss are all unchanged.
    :type only_obj_ptrs_in_the_past_for_eval: bool
    :param fifo_capacity: Explicit non-conditioning FIFO capacity. ``None``
        derives it from ``num_maskmem`` and the evaluation stride so that no
        frame the selection policy can reach is ever evicted.
    :type fifo_capacity: Optional[int]
    :param stop_gradient: Detach stored tensors on insertion (H-4).
    :type stop_gradient: bool
    :raises ValueError: If any size is non-positive or ``hidden_dim`` is not a
        multiple of ``mem_dim``.

    Example:

    .. code-block:: python

        bank = SAM2MemoryBank(num_maskmem=7, mem_dim=64, hidden_dim=256)
        bank.add_frame(0, feats, pos, obj_ptr=ptr, is_conditioning=True)
        bank.add_frame(1, feats, pos, obj_ptr=ptr)
        readout = bank.read(frame_idx=2)
        conditioned = memory_attention(
            features, readout.memory,
            features_pos=features_pos, memory_pos=readout.memory_pos,
            num_obj_ptr_tokens=readout.num_obj_ptr_tokens,
        )
    """

    def __init__(
            self,
            num_maskmem: int = 7,
            mem_dim: int = 64,
            hidden_dim: int = 256,
            memory_temporal_stride_for_eval: int = 1,
            max_obj_ptrs_in_encoder: int = 16,
            max_cond_frames: Optional[int] = None,
            use_signed_tpos_enc_to_obj_ptrs: bool = False,
            only_obj_ptrs_in_the_past_for_eval: bool = True,
            fifo_capacity: Optional[int] = None,
            stop_gradient: bool = True,
    ) -> None:
        if num_maskmem < 1:
            raise ValueError(f"num_maskmem must be >= 1, got {num_maskmem}")
        if mem_dim < 1:
            raise ValueError(f"mem_dim must be >= 1, got {mem_dim}")
        if hidden_dim < 1 or hidden_dim % mem_dim != 0:
            raise ValueError(
                "hidden_dim must be a positive multiple of mem_dim, got "
                f"hidden_dim={hidden_dim}, mem_dim={mem_dim}"
            )
        if memory_temporal_stride_for_eval < 1:
            raise ValueError(
                "memory_temporal_stride_for_eval must be >= 1, got "
                f"{memory_temporal_stride_for_eval}"
            )
        if max_obj_ptrs_in_encoder < 0:
            raise ValueError(
                f"max_obj_ptrs_in_encoder must be >= 0, got "
                f"{max_obj_ptrs_in_encoder}"
            )
        if max_cond_frames is not None and max_cond_frames < 0:
            raise ValueError(
                f"max_cond_frames must be >= 0 or None, got {max_cond_frames}"
            )
        if fifo_capacity is not None and fifo_capacity < 1:
            raise ValueError(
                f"fifo_capacity must be >= 1 or None, got {fifo_capacity}"
            )

        self.num_maskmem = int(num_maskmem)
        self.mem_dim = int(mem_dim)
        self.hidden_dim = int(hidden_dim)
        self.memory_temporal_stride_for_eval = int(
            memory_temporal_stride_for_eval)
        self.max_obj_ptrs_in_encoder = int(max_obj_ptrs_in_encoder)
        self.max_cond_frames = max_cond_frames
        self.use_signed_tpos_enc_to_obj_ptrs = bool(
            use_signed_tpos_enc_to_obj_ptrs)
        self.only_obj_ptrs_in_the_past_for_eval = bool(
            only_obj_ptrs_in_the_past_for_eval)
        self._fifo_capacity_override = fifo_capacity
        self.stop_gradient = bool(stop_gradient)

        # One 256-wide object pointer becomes this many 64-wide pseudo-spatial
        # tokens, so it is concatenable with the mem_dim-wide frame tokens.
        self.tokens_per_pointer = self.hidden_dim // self.mem_dim

        self.cond_frames: Dict[int, _FrameMemory] = {}
        self.non_cond_frames: Dict[int, _FrameMemory] = {}

        logger.debug(
            "SAM2MemoryBank created: num_maskmem=%d mem_dim=%d hidden_dim=%d "
            "tokens_per_pointer=%d fifo_capacity=%d",
            self.num_maskmem, self.mem_dim, self.hidden_dim,
            self.tokens_per_pointer, self.fifo_capacity,
        )

    # -----------------------------------------------------------------
    # derived sizes
    # -----------------------------------------------------------------

    @property
    def fifo_capacity(self) -> int:
        """Non-conditioning FIFO capacity, derived from the selection reach.

        The furthest frame the policy can name is ``f - 1 -
        (num_maskmem - 2) * stride`` (the ``t_rel = num_maskmem - 1`` arm, worst
        case over the floor division), so keeping that many frames plus the
        immediately preceding one is sufficient and nothing selectable is ever
        evicted. Deriving it -- rather than hardcoding ``num_maskmem`` -- is what
        keeps the FIFO correct at ``stride > 1``, where the reach is
        ``stride`` times longer than the slot count.

        :return: Number of non-conditioning frames retained.
        :rtype: int
        """
        if self._fifo_capacity_override is not None:
            return self._fifo_capacity_override
        stride = self.memory_temporal_stride_for_eval
        return max(1, (self.num_maskmem - 2) * stride + 2)

    @property
    def num_frames(self) -> int:
        """Total stored frames across both stores.

        :return: Frame count.
        :rtype: int
        """
        return len(self.cond_frames) + len(self.non_cond_frames)

    @property
    def is_empty(self) -> bool:
        """Whether the bank holds no frames at all.

        :return: ``True`` when both stores are empty.
        :rtype: bool
        """
        return self.num_frames == 0

    # -----------------------------------------------------------------
    # mutation
    # -----------------------------------------------------------------

    def reset(self) -> None:
        """Clear every store. Call once per video, before frame 0."""
        self.cond_frames = {}
        self.non_cond_frames = {}
        logger.debug("SAM2MemoryBank reset")

    def add_frame(
            self,
            frame_idx: int,
            maskmem_features: Any,
            maskmem_pos_enc: Any,
            obj_ptr: Optional[Any] = None,
            is_conditioning: bool = False,
    ) -> None:
        """Store one frame's memory.

        ``maskmem_features`` / ``maskmem_pos_enc`` may be given either as the
        memory encoder's ``(batch, height, width, mem_dim)`` grid or already
        flattened to ``(batch, num_tokens, mem_dim)``; both are stored flattened.

        :param frame_idx: Index of the frame within the video.
        :type frame_idx: int
        :param maskmem_features: Spatial memory features.
        :type maskmem_features: Any
        :param maskmem_pos_enc: Spatial positional encoding, same shape.
        :type maskmem_pos_enc: Any
        :param obj_ptr: Object pointer, ``(batch, hidden_dim)``, or ``None``.
        :type obj_ptr: Optional[Any]
        :param is_conditioning: Store in the conditioning bucket (prompted
            frame) rather than the FIFO.
        :type is_conditioning: bool
        :raises ValueError: If the feature/encoding widths are not ``mem_dim``,
            their shapes disagree, or ``obj_ptr`` is not ``hidden_dim`` wide.
        """
        features = self._flatten_spatial(maskmem_features, "maskmem_features")
        pos_enc = self._flatten_spatial(maskmem_pos_enc, "maskmem_pos_enc")
        if tuple(features.shape) != tuple(pos_enc.shape):
            raise ValueError(
                "maskmem_features and maskmem_pos_enc must have the same "
                f"shape, got {tuple(features.shape)} and {tuple(pos_enc.shape)}"
            )
        if obj_ptr is not None:
            if len(obj_ptr.shape) != 2 or obj_ptr.shape[-1] != self.hidden_dim:
                raise ValueError(
                    "obj_ptr must have shape (batch, hidden_dim="
                    f"{self.hidden_dim}), got {tuple(obj_ptr.shape)}"
                )
            if self.stop_gradient:
                obj_ptr = ops.stop_gradient(obj_ptr)

        if self.stop_gradient:
            # H-4: without this boundary, N frames of memory propagation build
            # one N-deep recurrent graph instead of N independent decodes.
            features = ops.stop_gradient(features)
            pos_enc = ops.stop_gradient(pos_enc)

        entry = _FrameMemory(features=features, pos_enc=pos_enc,
                             obj_ptr=obj_ptr)
        frame_idx = int(frame_idx)
        if is_conditioning:
            self.cond_frames[frame_idx] = entry
            self.non_cond_frames.pop(frame_idx, None)
        else:
            self.non_cond_frames[frame_idx] = entry
            self._evict()

    def _evict(self) -> None:
        """Drop the oldest non-conditioning frames past :attr:`fifo_capacity`."""
        while len(self.non_cond_frames) > self.fifo_capacity:
            oldest = min(self.non_cond_frames)
            del self.non_cond_frames[oldest]

    def _flatten_spatial(self, tensor: Any, name: str) -> Any:
        """Flatten a ``(B, H, W, C)`` grid to ``(B, H * W, C)``, or pass through.

        :param tensor: Spatial tensor of rank 3 or 4.
        :type tensor: Any
        :param name: Argument name, used in error messages.
        :type name: str
        :return: ``(batch, num_tokens, mem_dim)``.
        :rtype: Any
        :raises ValueError: On a wrong rank or a wrong channel width.
        """
        rank = len(tensor.shape)
        if rank not in (3, 4):
            raise ValueError(
                f"{name} must have rank 3 (batch, tokens, mem_dim) or rank 4 "
                f"(batch, height, width, mem_dim), got rank {rank}"
            )
        if tensor.shape[-1] != self.mem_dim:
            raise ValueError(
                f"{name} must be mem_dim={self.mem_dim} wide, got "
                f"{tensor.shape[-1]}"
            )
        if rank == 3:
            return tensor
        shape = ops.shape(tensor)
        return ops.reshape(
            tensor, (shape[0], shape[1] * shape[2], shape[3]))

    # -----------------------------------------------------------------
    # selection policy (pure -- never mutates any store)
    # -----------------------------------------------------------------

    def select_frames(
            self,
            frame_idx: int,
            track_in_reverse: bool = False,
            stride: Optional[int] = None,
    ) -> List[_Selection]:
        """Select the spatial memory frames for ``frame_idx``.

        Pure: reads the stores and returns, never mutating them. Conditioning
        frames come first (ascending frame index, ``t_pos = 0``), then the
        non-conditioning slots in ascending ``t_pos`` -- i.e. oldest first, the
        immediately preceding frame last.

        :param frame_idx: The frame being tracked.
        :type frame_idx: int
        :param track_in_reverse: Track backwards in time; flips the sign of all
            frame-index arithmetic.
        :type track_in_reverse: bool
        :param stride: Temporal subsampling stride; ``None`` uses
            ``memory_temporal_stride_for_eval``.
        :type stride: Optional[int]
        :return: Selected frames in memory-sequence order.
        :rtype: List[_Selection]
        :raises ValueError: If ``stride`` is given and is below 1.
        """
        if stride is None:
            stride = self.memory_temporal_stride_for_eval
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        frame_idx = int(frame_idx)

        selections: List[_Selection] = []

        # Conditioning frames: t_pos is 0 no matter how distant they are.
        # DECISION plan-2026-08-04T044628-4c240b4c/D-019
        cond_indices = sorted(self.cond_frames)
        if self.max_cond_frames is not None:
            cond_indices = self._nearest(
                cond_indices, frame_idx, self.max_cond_frames)
        for cond_idx in cond_indices:
            selections.append(_Selection(
                frame_idx=cond_idx,
                t_pos=0,
                tpos_slot=self.num_maskmem - 1,
                is_conditioning=True,
            ))

        for t_pos in range(1, self.num_maskmem):
            prev_idx = self._previous_frame_index(
                frame_idx, t_pos, stride, track_in_reverse)
            if prev_idx not in self.non_cond_frames:
                # A frame the policy names but the FIFO does not hold (early in
                # a video, or a frame stored as conditioning) is SKIPPED, not
                # padded -- the memory sequence simply gets shorter.
                continue
            selections.append(_Selection(
                frame_idx=prev_idx,
                t_pos=t_pos,
                tpos_slot=self.num_maskmem - t_pos - 1,
                is_conditioning=False,
            ))
        return selections

    def _previous_frame_index(
            self,
            frame_idx: int,
            t_pos: int,
            stride: int,
            track_in_reverse: bool,
    ) -> int:
        """Frame index for non-conditioning slot ``t_pos``.

        :param frame_idx: The frame being tracked.
        :type frame_idx: int
        :param t_pos: Temporal bucket in ``1 .. num_maskmem - 1``.
        :type t_pos: int
        :param stride: Temporal subsampling stride.
        :type stride: int
        :param track_in_reverse: Flip the direction of time.
        :type track_in_reverse: bool
        :return: The named frame index (which may not exist in any store).
        :rtype: int
        """
        t_rel = self.num_maskmem - t_pos
        if t_rel == 1:
            # DECISION plan-2026-08-04T044628-4c240b4c/D-018
            # SPECIAL CASE, not an instance of the general formula: the most
            # recent slot ALWAYS takes the immediately preceding frame,
            # regardless of stride. Do NOT "simplify" this away -- at stride 1
            # the general formula happens to agree, so a stride-1 test proves
            # nothing; at stride >= 2 it silently returns an older frame and the
            # tracker loses its most informative memory with no shape error.
            return frame_idx + t_rel if track_in_reverse else frame_idx - t_rel
        if not track_in_reverse:
            anchor = ((frame_idx - 2) // stride) * stride
            return anchor - (t_rel - 2) * stride
        anchor = -(((-frame_idx) - 2) // stride) * stride
        return anchor + (t_rel - 2) * stride

    def select_object_pointer_frames(
            self,
            frame_idx: int,
            track_in_reverse: bool = False,
            max_obj_ptrs: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Select the frames whose object pointers enter the memory sequence.

        Conditioning frames first (ascending index), then non-conditioning
        frames walked backward from ``frame_idx`` until the cap is reached.

        :param frame_idx: The frame being tracked.
        :type frame_idx: int
        :param track_in_reverse: Walk forward in index instead of backward, and
            flip the sign of a signed temporal difference.
        :type track_in_reverse: bool
        :param max_obj_ptrs: Cap on the number of pointers; ``None`` uses
            ``max_obj_ptrs_in_encoder``.
        :type max_obj_ptrs: Optional[int]
        :return: ``(frame_index, temporal_difference)`` pairs.
        :rtype: List[Tuple[int, float]]
        """
        if max_obj_ptrs is None:
            max_obj_ptrs = self.max_obj_ptrs_in_encoder
        frame_idx = int(frame_idx)
        sign = -1 if track_in_reverse else 1

        def t_diff(other: int) -> float:
            if self.use_signed_tpos_enc_to_obj_ptrs:
                return float((frame_idx - other) * sign)
            return float(abs(frame_idx - other))

        # DECISION plan-2026-08-04T044628-4c240b4c/D-041
        # Conditioning pointers are filtered to the PAST (to the future when
        # tracking in reverse). Do NOT drop this filter: a conditioning frame
        # is kept indefinitely and may sit anywhere in the video, so without it
        # a prompt given at frame 50 contributes its object pointer to the
        # memory assembled for frame 10. Nothing about that is observable from
        # outside -- same token count, same shapes, same loss -- it simply
        # leaks future information into an earlier frame's prediction. See
        # decisions.md D-041.
        def in_the_past(other: int) -> bool:
            if not self.only_obj_ptrs_in_the_past_for_eval:
                return True
            return other >= frame_idx if track_in_reverse else other <= frame_idx

        chosen: List[Tuple[int, float]] = []
        for cond_idx in sorted(self.cond_frames):
            if self.cond_frames[cond_idx].obj_ptr is None:
                continue
            if not in_the_past(cond_idx):
                continue
            if len(chosen) >= max_obj_ptrs:
                return chosen
            chosen.append((cond_idx, t_diff(cond_idx)))

        step = 1 if track_in_reverse else -1
        for offset in range(1, max_obj_ptrs + 1):
            if len(chosen) >= max_obj_ptrs:
                break
            other = frame_idx + step * offset
            entry = self.non_cond_frames.get(other)
            if entry is None or entry.obj_ptr is None:
                continue
            chosen.append((other, t_diff(other)))
        return chosen

    @staticmethod
    def _nearest(indices: List[int], frame_idx: int, keep: int) -> List[int]:
        """Keep the ``keep`` indices closest to ``frame_idx``, still sorted.

        :param indices: Candidate frame indices.
        :type indices: List[int]
        :param frame_idx: Reference frame.
        :type frame_idx: int
        :param keep: How many to retain.
        :type keep: int
        :return: A sorted subset.
        :rtype: List[int]
        """
        if keep >= len(indices):
            return indices
        return sorted(sorted(indices, key=lambda i: abs(i - frame_idx))[:keep])

    # -----------------------------------------------------------------
    # assembly
    # -----------------------------------------------------------------

    def read(
            self,
            frame_idx: int,
            track_in_reverse: bool = False,
            stride: Optional[int] = None,
            max_obj_ptrs: Optional[int] = None,
    ) -> _MemoryReadout:
        """Assemble the memory sequence for ``frame_idx``.

        Pure: never mutates any store, so repeated calls with the same state
        return identical results.

        The returned sequence is ``[spatial frame tokens ...][object-pointer
        tokens ...]``. The object-pointer block is at the TAIL because memory
        attention excludes exactly ``num_obj_ptr_tokens`` TRAILING key rows from
        rotary embedding (H-14) -- prepending would exclude the wrong rows with
        no shape error anywhere.

        :param frame_idx: The frame being tracked.
        :type frame_idx: int
        :param track_in_reverse: Track backwards in time.
        :type track_in_reverse: bool
        :param stride: Temporal subsampling stride; ``None`` uses the configured
            evaluation stride.
        :type stride: Optional[int]
        :param max_obj_ptrs: Cap on object pointers; ``None`` uses
            ``max_obj_ptrs_in_encoder``.
        :type max_obj_ptrs: Optional[int]
        :return: The assembled readout.
        :rtype: _MemoryReadout
        :raises ValueError: If two selected frames disagree on token count.
        """
        selections = self.select_frames(
            frame_idx, track_in_reverse=track_in_reverse, stride=stride)
        ptr_frames = self.select_object_pointer_frames(
            frame_idx, track_in_reverse=track_in_reverse,
            max_obj_ptrs=max_obj_ptrs)

        parts: List[Any] = []
        pos_parts: List[Any] = []
        token_counts: List[int] = []
        for selection in selections:
            entry = self._entry(selection)
            parts.append(entry.features)
            pos_parts.append(entry.pos_enc)
            token_counts.append(int(entry.features.shape[1]))

        num_ptr_tokens = 0
        ptr_tpos: List[float] = []
        if ptr_frames:
            pointers = [self._entry_for_index(idx).obj_ptr
                        for idx, _ in ptr_frames]
            # (batch, num_pointers, hidden_dim) -> (batch, num_pointers *
            # tokens_per_pointer, mem_dim): each hidden_dim-wide pointer splits
            # into tokens_per_pointer CONTIGUOUS mem_dim-wide chunks, so token
            # (p * tokens_per_pointer + s) is pointer p's chunk s.
            stacked = ops.stack(pointers, axis=1)
            batch = ops.shape(stacked)[0]
            num_ptr_tokens = len(pointers) * self.tokens_per_pointer
            ptr_tokens = ops.reshape(
                stacked, (batch, num_ptr_tokens, self.mem_dim))
            # DECISION plan-2026-08-04T044628-4c240b4c/D-020
            # TAIL, not head. Memory attention's num_k_exclude counts TRAILING
            # rows.
            parts.append(ptr_tokens)
            pos_parts.append(ops.zeros_like(ptr_tokens))
            # Each of a pointer's sub-tokens shares that pointer's temporal
            # difference -- a repeat_interleave, not a tile.
            for _, diff in ptr_frames:
                ptr_tpos.extend([diff] * self.tokens_per_pointer)

        if not parts:
            memory = None
            memory_pos = None
        else:
            memory = ops.concatenate(parts, axis=1)
            memory_pos = ops.concatenate(pos_parts, axis=1)

        return _MemoryReadout(
            memory=memory,
            memory_pos=memory_pos,
            selections=tuple(selections),
            tpos_slots=tuple(s.tpos_slot for s in selections),
            frame_token_counts=tuple(token_counts),
            num_obj_ptr_tokens=num_ptr_tokens,
            obj_ptr_frames=tuple(idx for idx, _ in ptr_frames),
            obj_ptr_tpos=tuple(ptr_tpos),
        )

    def _entry(self, selection: _Selection) -> _FrameMemory:
        """Fetch the stored entry a selection refers to.

        :param selection: A selection produced by :meth:`select_frames`.
        :type selection: _Selection
        :return: The stored memory entry.
        :rtype: _FrameMemory
        """
        store = self.cond_frames if selection.is_conditioning \
            else self.non_cond_frames
        return store[selection.frame_idx]

    def _entry_for_index(self, frame_idx: int) -> _FrameMemory:
        """Fetch a stored entry by frame index, either store.

        :param frame_idx: Frame index.
        :type frame_idx: int
        :return: The stored memory entry.
        :rtype: _FrameMemory
        :raises KeyError: If neither store holds the frame.
        """
        if frame_idx in self.cond_frames:
            return self.cond_frames[frame_idx]
        return self.non_cond_frames[frame_idx]

    def __repr__(self) -> str:
        """Readable summary of the bank's configuration and occupancy.

        :return: Debug representation.
        :rtype: str
        """
        return (
            f"SAM2MemoryBank(num_maskmem={self.num_maskmem}, "
            f"mem_dim={self.mem_dim}, hidden_dim={self.hidden_dim}, "
            f"cond_frames={sorted(self.cond_frames)}, "
            f"non_cond_frames={sorted(self.non_cond_frames)})"
        )


# ---------------------------------------------------------------------
