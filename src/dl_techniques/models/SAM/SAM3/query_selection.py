"""
SAM 3 Encoder Query Selection: DINO-style *mixed* proposal generation.
======================================================================

:class:`Sam3EncoderQuerySelection` reads the decoder's image ``memory`` -- the
flattened finest neck level -- and emits, per position, an objectness logit and
a ``cxcywh`` box refined from that position's grid anchor. The top
``num_queries`` boxes replace the decoder's learned, image-INDEPENDENT
``reference_points`` table as its initial ``reference_boxes``. This head is NOT
a reference component: it is this package's own addition, reached through
``Sam3Image(..., query_selection=True)``, OFF by default.

Based on:
---------
- Zhang, H. et al. (2022). DINO -- the *mixed* selection implemented here: a
  query's POSITIONAL part comes from the encoder, its CONTENT part stays a
  learned table.
- Ravi, N. et al. (2025). "SAM 3: Segment Anything with Concepts."

Key Features:
------------
- Per-position objectness and box MLPs over the image memory.
- Row-major grid anchors at pixel centres.
- Optional, default-OFF ``prompt_conditioned`` FiLM modulation that makes the
  top-k SELECTION itself prompt-dependent.

Architecture Overview:
---------------------
1. ``memory (batch, H * W, d_model)`` -> objectness MLP ``-> 1`` and box MLP
   ``-> 4`` (a delta).
2. ``boxes = sigmoid(delta + inverse_sigmoid(anchor_j))``, with
   ``anchor_j = ((col + 0.5) / W, (row + 0.5) / H, anchor_size, anchor_size)``.
3. ``top_k(objectness[..., 0], k=num_queries)``, then ``gather(boxes, indices)``.
4. Behind ``prompt_conditioned``, before either MLP reads it: ``scale, shift =
   split(Dense(2 * d_model)(masked_mean_pool(prompt)))`` then
   ``memory = memory * (1 + scale[:, None, :]) + shift[:, None, :]``.

Usage Examples:
--------------
```python
from dl_techniques.models.SAM.SAM3.query_selection import (
    Sam3EncoderQuerySelection)
head = Sam3EncoderQuerySelection(d_model=256, num_queries=200,
                                 feat_size=(72, 72))
```

Measured caveats:
----------------
- **Why this head exists, MEASURED not guessed**: SAM 3's box output was
  image-independent BY CONSTRUCTION, not by a training-time collapse.
  ``val_box_std_across_images`` read ``6.9e-06`` against an across-*query*
  spread of ``0.13``, and is already that low at epoch 0 -- the decoder's box
  chain is ``sigmoid(delta + inverse_sigmoid(reference))`` with a zero-init last
  projection over a learned table broadcast across the batch, so at step 0 the
  boxes cannot depend on the image at all.
- With the flag off nothing changes and no weight is created, so the on-disk
  checkpoints and the exact parameter-count oracle are untouched.
- The flatten order is row-major and not a matter of taste: ``anchor_j`` must be
  laid out exactly as the memory it annotates, and a transposed grid is a
  silent, plausible-looking defect with no shape symptom.
- The box stack's last projection is zero-initialized (D-112), so at step 0
  every proposal is EXACTLY its grid anchor.
- A degenerate objectness field selects positions ``0 .. k - 1``, because
  ``top_k`` breaks ties by ascending index -- an image-INDEPENDENT selection
  with the right shapes, dtypes and a plausible spread. That vacuity mode is
  what this layer's guards exist to exclude.
"""

import keras
import numpy as np
from keras import ops
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .decoder import Sam3TransformerDecoder
from .model_misc import Sam3DotProductScoring
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

#: Step-0 side length of every proposal anchor, in normalized image units.
#:
#: MEASURED, not invented: the seed-pooled mean of ``sqrt(w * h)`` over the
#: 4547 valid ground-truth boxes of the TRAIN split at seeds 1/2/3, read through
#: ``train.sam3.baselines.pool_train_gt`` (the single home for "read the train
#: split's GT"). Per-seed values 0.1771 / 0.1767 / 0.1791; per-seed spread
#: 0.00236, i.e. 1.3% of the mean and 6.2% of one standard deviation (0.0379).
#: The scoring split (``seed + 10_000``) was never read, so no scoring-split
#: statistic enters this model's initialization.
#:
#: This is a DIFFERENT constant with a DIFFERENT provenance from
#: ``train.sam3.baselines.GRID_BOX_SIZE = 0.2``, which is a hand-written
#: comparator geometry fitted to nothing. The two are numerically close and must
#: never be "reconciled" into one shared constant: one is a fit, the other is
#: deliberately not. See decisions.md D-005.
#:
#: The same measurement shows the data does NOT support a square anchor
#: (``mean(w) = 0.2004`` vs ``mean(h) = 0.1632``, ``mean(w/h) = 1.374``); a
#: square anchor is a stated modelling choice, and separate ``anchor_w`` /
#: ``anchor_h`` is the named follow-up lever, deliberately not taken here.
DEFAULT_ANCHOR_SIZE: float = 0.1776

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class Sam3EncoderQuerySelection(keras.layers.Layer):
    """Per-memory-position objectness and box proposals, plus the top-k pick.

    :param d_model: Width of the image memory this head reads. Default:
        ``256``.
    :type d_model: int
    :param num_queries: Number of proposals to select, i.e. ``k`` of the
        ``top_k``. Must not exceed the number of memory positions. Default:
        ``200``.
    :type num_queries: int
    :param feat_size: Image-memory grid ``(height, width)``; its product is the
        number of memory positions. Default: ``(72, 72)``.
    :type feat_size: Tuple[int, int]
    :param anchor_size: Side length of every grid anchor, in normalized image
        units. Default: :data:`DEFAULT_ANCHOR_SIZE`.
    :type anchor_size: float
    :param mlp_depth: Number of ``Dense`` layers in each of the two heads; all
        but the last carry a ReLU. Default: ``3``.
    :type mlp_depth: int
    :param prompt_conditioned: Whether the head reads the text prompt. When
        ``False`` (the default) NO extra sub-layer is created and this layer
        owns exactly the weights it owned before this flag existed, so the
        on-disk checkpoints and the exact parameter-count oracle are unmoved.
        When ``True`` the pooled prompt drives a FiLM-style per-channel affine
        on ``memory`` before both MLPs -- see :meth:`call`. Default: ``False``.
    :type prompt_conditioned: bool
    :raises ValueError: If any width is non-positive, if ``feat_size`` is not a
        pair of positive integers, if ``anchor_size`` is outside ``(0, 1)``, if
        ``mlp_depth`` is below one, or if the grid holds fewer positions than
        ``num_queries``.

    Example:
        >>> import numpy as np
        >>> head = Sam3EncoderQuerySelection(d_model=8, num_queries=3,
        ...                                  feat_size=(4, 4))
        >>> memory = np.zeros((2, 16, 8), dtype="float32")
        >>> {k: tuple(v.shape) for k, v in sorted(head(memory).items())}
        ... # doctest: +NORMALIZE_WHITESPACE
        {'boxes': (2, 16, 4), 'indices': (2, 3), 'objectness': (2, 16, 1),
         'selected_boxes': (2, 3, 4), 'selected_objectness': (2, 3, 1)}
    """

    def __init__(
            self, d_model: int = 256, num_queries: int = 200,
            feat_size: Tuple[int, int] = (72, 72),
            anchor_size: float = DEFAULT_ANCHOR_SIZE, mlp_depth: int = 3,
            prompt_conditioned: bool = False, **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        for name, value in (("d_model", d_model),
                            ("num_queries", num_queries)):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if len(feat_size) != 2 or min(feat_size) <= 0:
            raise ValueError(f"feat_size must be a pair of positive ints, got "
                             f"{feat_size}")
        if not 0.0 < anchor_size < 1.0:
            raise ValueError(f"anchor_size must be in (0, 1), got "
                             f"{anchor_size}")
        if mlp_depth < 1:
            raise ValueError(f"mlp_depth must be at least one, got "
                             f"{mlp_depth}")

        self.d_model = int(d_model)
        self.num_queries = int(num_queries)
        self.feat_size = (int(feat_size[0]), int(feat_size[1]))
        self.anchor_size = float(anchor_size)
        self.mlp_depth = int(mlp_depth)
        self.prompt_conditioned = bool(prompt_conditioned)
        self.num_positions = self.feat_size[0] * self.feat_size[1]

        # There are only `H * W` distinct proposals to choose from, so asking
        # for more than that is not a degraded selection -- it is an impossible
        # one, and `ops.top_k` would raise deep inside the forward pass with a
        # message naming neither this layer nor its configuration.
        if self.num_positions < self.num_queries:
            raise ValueError(
                f"feat_size {self.feat_size} holds {self.num_positions} memory "
                f"positions but num_queries is {self.num_queries}: query "
                f"selection picks the top {self.num_queries} of "
                f"{self.num_positions} positions and cannot pick more "
                f"proposals than the grid has positions")

        # Every sub-layer store here is FLAT. A `List[List[Layer]]` restores
        # freshly initialized kernels on a `.keras` round trip while the weight
        # count, every weight path and the parameter total all match -- measured
        # in this package, see decisions.md D-098.
        #
        # Both stacks come from the decoder's own sanctioned MLP trio
        # (`_make_mlp` / `_build_mlp` / `_run_mlp`) rather than from a fourth
        # hand-rolled Dense-stack builder.
        self.objectness_head = Sam3TransformerDecoder._make_mlp(
            self.mlp_depth, self.d_model, 1, "objectness_head")

        # DECISION plan-2026-08-06T185813-fd80240f/D-005
        # The LAST projection of the box stack is ZERO-initialized, exactly as
        # the decoder's `bbox_embed` is (D-112 of
        # plan-2026-08-04T044628-4c240b4c). Do NOT "fix" this to a standard
        # initializer: the proposal chain is
        # `sigmoid(delta + inverse_sigmoid(anchor))`, so a zero delta is what
        # makes every proposal EXACTLY its grid anchor at step 0. With any
        # non-zero init the head displaces every anchor before a single
        # gradient step, the decoder's initial references are displaced with
        # it, and boxRPB's bias -- which is built FROM those references -- is
        # displaced too. There is no shape, dtype or finiteness symptom.
        # See decisions.md D-005.
        self.box_head = Sam3TransformerDecoder._make_mlp(
            self.mlp_depth, self.d_model, 4, "box_head", zero_init_last=True)

        # DECISION plan-2026-08-07T065516-6add49a9/D-014
        # Created ONLY when the flag is on, and NOT zero-initialized. Two
        # things are pinned here and neither is style.
        #   * Creating it unconditionally is the one change that flips
        #     `test_query_selection.py`'s exact parameter-count oracle RED at
        #     defaults and stops the 21 on-disk checkpoints loading. The flag
        #     is what buys byte-identity-when-off, so do NOT hoist this out of
        #     the `if`.
        #   * Do NOT copy `box_head`'s `zero_init_last=True` here. A zero
        #     initializer makes the modulation the EXACT identity at step 0, so
        #     an untrained flag-on model is bit-identical to the flag-off one
        #     and every prompt-liveness measurement reads exactly 0.0 -- the
        #     head would be born degenerate on precisely the axis this flag
        #     exists to open. The box head's zero init is correct for a
        #     DISPLACEMENT of an anchor; this is a GATE, and the two want
        #     opposite initializations.
        # The stack is the decoder's sanctioned `_make_mlp` trio at depth 1
        # (one linear projection, no activation), not a fourth Dense-stack
        # builder, and it is stored FLAT like the two above it.
        # See decisions.md D-014.
        self.prompt_film = None
        if self.prompt_conditioned:
            self.prompt_film = Sam3TransformerDecoder._make_mlp(
                1, self.d_model, 2 * self.d_model, "prompt_film")

        self._anchor_grid = self._anchors()

        logger.info(
            f"Sam3EncoderQuerySelection: d_model={self.d_model}, "
            f"queries={self.num_queries} of {self.num_positions} positions "
            f"(grid {self.feat_size}), anchor_size={self.anchor_size}, "
            f"mlp_depth={self.mlp_depth}, "
            f"prompt_conditioned={self.prompt_conditioned}"
        )

    # -----------------------------------------------------------------
    # anchors
    # -----------------------------------------------------------------

    def _anchors(self) -> np.ndarray:
        """Build the constant per-position anchor grid.

        :return: ``(1, H * W, 4)`` normalized ``cxcywh`` anchors, float32. The
            leading singleton axis broadcasts over the batch.
        :rtype: np.ndarray
        """
        # DECISION plan-2026-08-06T185813-fd80240f/D-005
        # The layout is ROW-MAJOR: position `j` is `(row, col)` with
        # `row = j // W` and `col = j % W`. This is NOT a free choice and NOT
        # an assumption -- it is what the two producers of this index do:
        #   * `Sam3Image._flatten` reshapes a channels-last `(B, H, W, C)` map
        #     to `(B, H * W, C)`, so the WIDTH axis varies fastest;
        #   * `decoder._box_rpb_bias` builds its key axis by outer-summing a
        #     `height`-indexed and a `width`-indexed embedding and reshaping
        #     `(..., height, width) -> (..., height * width)`, i.e. the same
        #     order, on the same tensor this head annotates.
        # Do NOT swap the two, and do NOT "simplify" this to one `arange` over
        # `H * W`: on the square grids every shipped variant uses, a transposed
        # anchor grid is shape-compatible, finite, plausible and silently wrong
        # -- the exact failure class `necks.py`'s D-096 exists for. It is
        # pinned empirically by `test_query_selection.py`'s flatten-order proof,
        # which reads `Sam3Image._flatten` itself rather than restating it.
        height, width = self.feat_size
        rows, cols = np.divmod(np.arange(self.num_positions), width)
        anchors = np.stack([
            (cols.astype("float32") + 0.5) / float(width),
            (rows.astype("float32") + 0.5) / float(height),
            np.full(self.num_positions, self.anchor_size, dtype="float32"),
            np.full(self.num_positions, self.anchor_size, dtype="float32"),
        ], axis=-1)
        return anchors[None, ...].astype("float32")

    # -----------------------------------------------------------------
    # build
    # -----------------------------------------------------------------

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build every MLP stack from the memory shape.

        MEASURED constraint, not a style choice: this stays a ONE-argument
        ``build``. Keras refuses a multi-argument ``build`` whose argument
        names do not match ``call``'s (``ValueError: ... received build()
        argument 'input_shape', but call() does not have argument 'input'``),
        and renaming it to ``memory_shape`` would change the key Keras records
        in the build config -- which every on-disk checkpoint carries as
        ``input_shape``. Nothing is lost: the FiLM projection reads the POOLED
        prompt, whose width is ``d_model``, so no prompt shape is needed to
        build it.

        :param input_shape: Memory shape ``(batch, H * W, d_model)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: On a rank other than three, a width other than
            ``d_model``, or a key count other than ``feat_size[0] *
            feat_size[1]``.
        """
        # Re-entry guard, matching the other `build` methods in this package:
        # on `.keras` LOAD Keras rebuilds a component from its recorded build
        # config BEFORE `build_from_config` runs, and a second `build()` without
        # this guard raises `ValueError: You cannot add new elements of state
        # ... to a layer that is already built` (D-136).
        if self.built:
            return
        if len(input_shape) != 3:
            raise ValueError(f"memory must have shape (batch, H * W, d_model), "
                             f"got {input_shape}")
        if input_shape[-1] is not None and input_shape[-1] != self.d_model:
            raise ValueError(f"memory width {input_shape[-1]} != d_model "
                             f"{self.d_model}")
        if (input_shape[1] is not None
                and input_shape[1] != self.num_positions):
            raise ValueError(
                f"memory has {input_shape[1]} positions but feat_size "
                f"{self.feat_size} implies {self.num_positions}; the anchor "
                f"grid is built on that geometry, so a mismatch is a silently "
                f"wrong anchor per position rather than a shape error")

        Sam3TransformerDecoder._build_mlp(self.objectness_head, input_shape)
        Sam3TransformerDecoder._build_mlp(self.box_head, input_shape)
        if self.prompt_film is not None:
            Sam3TransformerDecoder._build_mlp(
                self.prompt_film, (input_shape[0], self.d_model))
        super().build(input_shape)

    # -----------------------------------------------------------------
    # call
    # -----------------------------------------------------------------

    def call(
            self, memory: keras.KerasTensor,
            prompt: Optional[keras.KerasTensor] = None,
            prompt_padding_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """Score every memory position and select the top ``num_queries``.

        :param memory: Image memory ``(batch, H * W, d_model)`` -- the
            flattened finest neck level.
        :type memory: keras.KerasTensor
        :param prompt: Text-prompt features ``(batch, seq, d_model)``. Read
            only when ``prompt_conditioned`` is set; ignored otherwise, and
            defaulted to ``None`` so a single-input functional model over this
            layer still builds.
        :type prompt: Optional[keras.KerasTensor]
        :param prompt_padding_mask: ``(batch, seq)``, ``True`` at PADDING
            positions -- the polarity :meth:`Sam3DotProductScoring.
            masked_mean_pool` documents. ``None`` pools every position.
        :type prompt_padding_mask: Optional[keras.KerasTensor]
        :param training: Training-mode flag; accepted for uniformity, unused
            (this head holds no dropout and no normalization).
        :type training: Optional[bool]
        :raises ValueError: If ``prompt_conditioned`` is set and ``prompt`` is
            ``None`` -- a silently prompt-blind proposal head is the exact
            defect this flag exists to remove.
        :return: ``objectness`` ``(batch, H * W, 1)``, ``boxes``
            ``(batch, H * W, 4)`` in normalized ``cxcywh``, ``selected_boxes``
            ``(batch, num_queries, 4)``, ``selected_objectness``
            ``(batch, num_queries, 1)`` and ``indices``
            ``(batch, num_queries)`` (``int32``, the selected memory
            positions).
        :rtype: Dict[str, keras.KerasTensor]
        """
        del training

        if self.prompt_conditioned:
            if prompt is None:
                raise ValueError(
                    "Sam3EncoderQuerySelection was configured with "
                    "prompt_conditioned=True but call() got prompt=None: the "
                    "head would silently fall back to the prompt-BLIND "
                    "proposals this flag exists to replace, with no shape, "
                    "dtype or finiteness symptom")
            # DECISION plan-2026-08-07T065516-6add49a9/D-014
            # The modulation is a per-CHANNEL affine on `memory`, applied
            # BEFORE both MLPs, and it is placed here rather than on the
            # objectness LOGITS on purpose: a term added to the logits that is
            # constant across positions cannot change an argsort, so a
            # `top_k` fed by it selects the same positions for every prompt --
            # a prompt-conditioning that provably cannot condition the
            # SELECTION, which is the one thing this flag is for. A per-channel
            # SCALE reweights each position's own features differently, so the
            # objectness ordering can (and must be shown to) move. Do NOT
            # "simplify" this to a bias on `objectness`, and do NOT drop the
            # scale and keep only the shift.
            # See decisions.md D-014.
            pooled = Sam3DotProductScoring.masked_mean_pool(
                ops.cast(prompt, memory.dtype), prompt_padding_mask)
            film = Sam3TransformerDecoder._run_mlp(self.prompt_film, pooled)
            scale, shift = ops.split(film, 2, axis=-1)
            memory = (memory * (1.0 + ops.expand_dims(scale, axis=1))
                      + ops.expand_dims(shift, axis=1))

        objectness = Sam3TransformerDecoder._run_mlp(
            self.objectness_head, memory)
        delta = Sam3TransformerDecoder._run_mlp(self.box_head, memory)

        anchors = ops.cast(
            ops.convert_to_tensor(self._anchor_grid), delta.dtype)
        boxes = ops.sigmoid(
            delta + Sam3TransformerDecoder._inverse_sigmoid(anchors))

        # `ops.top_k` breaks ties by ASCENDING index, so an objectness field
        # that carries no image signal selects positions `0 .. k - 1` for every
        # image. That reads as a perfectly ordinary output; only an
        # across-image comparison separates it from a live selection.
        values, indices = ops.top_k(objectness[..., 0], k=self.num_queries)
        selected_boxes = ops.take_along_axis(
            boxes, ops.expand_dims(indices, axis=-1), axis=1)

        return {
            "objectness": objectness,
            "boxes": boxes,
            "selected_boxes": selected_boxes,
            "selected_objectness": ops.expand_dims(values, axis=-1),
            "indices": indices,
        }

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...],
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """Return one shape per output key.

        :param input_shape: Memory shape ``(batch, H * W, d_model)``. No output
            shape depends on the prompt, so this stays a one-argument method
            for the same Keras naming reason :meth:`build` does.
        :type input_shape: Tuple[Optional[int], ...]
        :return: One shape per key of :meth:`call`'s output dict.
        :rtype: Dict[str, Tuple[Optional[int], ...]]
        """
        batch = input_shape[0]
        return {
            "objectness": (batch, self.num_positions, 1),
            "boxes": (batch, self.num_positions, 4),
            "selected_boxes": (batch, self.num_queries, 4),
            "selected_objectness": (batch, self.num_queries, 1),
            "indices": (batch, self.num_queries),
        }

    def get_config(self) -> Dict[str, Any]:
        """Return every ``__init__`` parameter.

        :return: Serializable configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_queries": self.num_queries,
            "feat_size": self.feat_size,
            "anchor_size": self.anchor_size,
            "mlp_depth": self.mlp_depth,
            "prompt_conditioned": self.prompt_conditioned,
        })
        return config
