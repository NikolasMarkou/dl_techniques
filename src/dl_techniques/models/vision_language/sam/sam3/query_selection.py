"""Sam3EncoderQuerySelection, DINO-style mixed proposal generation for SAM 3.

Reads the decoder's flattened image memory and emits, per grid position, an
objectness logit and a box refined from that position's fixed anchor. The
top ``num_queries`` boxes replace the decoder's learned, image-independent
reference-point table as its initial reference boxes -- following DINO's
mixed selection, where a query's position comes from the encoder and its
content stays a learned table. This head is not part of the SAM 3
reference; it is reached only through ``Sam3Image(..., query_selection=True)``
and is off by default, so nothing changes when the flag is off.

An optional ``prompt_conditioned`` mode applies a FiLM-style per-channel
affine to the memory, built from the pooled text prompt, before both MLPs,
making the top-k selection itself prompt-dependent.

Grid anchors are laid out row-major to match the memory's own flatten
order; a transposed grid is shape-compatible and silently wrong. The box
head's last projection is zero-initialized, so at step 0 every proposal is
exactly its grid anchor.

References:
    - Zhang et al., 2022. DINO: DETR with Improved DeNoising Anchor Boxes
      for End-to-End Object Detection.
    - Ravi et al., 2025. SAM 3: Segment Anything with Concepts.
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
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

# DECISION plan-2026-08-06T185813-fd80240f/D-005: step-0 anchor side length,
# measured as the mean sqrt(w*h) over 4547 train-split ground-truth boxes.
# A different constant from train.sam3.baselines.GRID_BOX_SIZE=0.2 (an unrelated hand-fit comparator) -- never merge them. See decisions.md.
DEFAULT_ANCHOR_SIZE: float = 0.1776

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.sam3.query_selection")
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

        # Only H * W proposals exist; asking for more is impossible, not degraded.
        if self.num_positions < self.num_queries:
            raise ValueError(
                f"feat_size {self.feat_size} holds {self.num_positions} memory "
                f"positions but num_queries is {self.num_queries}: query "
                f"selection picks the top {self.num_queries} of "
                f"{self.num_positions} positions and cannot pick more "
                f"proposals than the grid has positions")

        # Sub-layers are stored flat, not List[List[Layer]] -- that shape
        # restores freshly initialized kernels on a .keras round trip. See decisions.md D-098.
        self.objectness_head = Sam3TransformerDecoder._make_mlp(
            self.mlp_depth, self.d_model, 1, "objectness_head")

        # DECISION plan-2026-08-06T185813-fd80240f/D-005: zero-init the box
        # stack's last projection, matching the decoder's bbox_embed (D-112).
        # A non-zero init displaces every anchor before the first gradient step, with no shape/dtype symptom. See decisions.md.
        self.box_head = Sam3TransformerDecoder._make_mlp(
            self.mlp_depth, self.d_model, 4, "box_head", zero_init_last=True)

        # DECISION plan-2026-08-07T065516-6add49a9/D-014: create prompt_film
        # only when the flag is on, and never zero-initialize it.
        # Unconditional creation breaks byte-identity-when-off and 21 on-disk checkpoints; zero-init here (unlike box_head) would make the gate born degenerate. See decisions.md.
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
        # DECISION plan-2026-08-06T185813-fd80240f/D-005: row-major layout,
        # position j = (row, col) with row = j // W, col = j % W.
        # Must match Sam3Image._flatten and decoder._box_rpb_bias's index order exactly; a transposed grid is silently wrong (necks.py D-096's failure class). See decisions.md.
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

        Stays a one-argument ``build`` (Keras requires its argument names to
        match ``call``'s, and every on-disk checkpoint's build config carries
        the key ``input_shape``). No prompt shape is needed: the FiLM
        projection reads the pooled prompt, whose width is ``d_model``.

        :param input_shape: Memory shape ``(batch, H * W, d_model)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: On a rank other than three, a width other than
            ``d_model``, or a key count other than ``feat_size[0] *
            feat_size[1]``.
        """
        # Re-entry guard, matching the other build() methods in this package (D-136).
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
            # DECISION plan-2026-08-07T065516-6add49a9/D-014: modulate memory
            # before both MLPs, never add a bias to the objectness logits.
            # A per-position-constant logit bias cannot change a top_k argsort, so the selection would stay prompt-blind. See decisions.md.
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

        # ops.top_k breaks ties by ascending index, so a signal-free
        # objectness field selects positions 0..k-1 for every image.
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
