"""Block-wise masking for BEiT masked-image-modelling (MIM).

This module supplies the DATA-SIDE half of BEiT's MIM objective, in the same
spirit as its sibling :mod:`masked_patches`: the occlusion mask reaches the loss
through Keras' sanctioned channel -- ``sample_weight`` -- so the model stays a
stock ``keras.Model`` and training is a stock
``model.compile(loss=SparseCategoricalCrossentropy(from_logits=True))`` +
``model.fit(ds)``. No ``train_step`` / ``test_step`` / ``compute_loss`` override
is needed anywhere.

It differs from ``masked_patches`` in the two ways that make BEiT BEiT:

* the mask is **block-wise**, not i.i.d. per patch -- rectangular blocks with a
  log-uniform aspect ratio are stamped into the grid until a budget is reached;
* the targets are **discrete code ids** from a frozen visual tokenizer, not raw
  pixel patches, so the loss is a cross-entropy over a codebook rather than an
  MSE over pixels.

Data contract
-------------
``make_beit_mim_map_fn`` returns a function mappable over a ``tf.data`` dataset
of raw images, producing the 3-tuple element::

    ((image, bool_mask), target_ids, sample_weight)

    image        (H, W, C)  float32   the UNMODIFIED image
    bool_mask    (N,)       bool      True  -> the model substitutes its
                                      learnable MASK token for that token
    target_ids   (N,)       int32     per-patch visual-token code id
    sample_weight(N,)       float32   1.0 at masked positions, 0.0 elsewhere

``sample_weight`` is exactly ``cast(bool_mask, float32)`` -- no rescaling. BEiT's
objective is the *sum* of the log-likelihoods at masked positions
(arXiv:2106.08254 eq. 2); Keras' ``sum_over_batch_size`` reduction then turns the
weighted per-token losses into a mean over ``B * N`` with the unmasked positions
contributing exactly zero. (This is deliberately unlike ``masked_patches``, which
scales its weights by ``N / n_loss`` to obtain a mean over the loss SET. Either
convention is defensible; only the effective learning rate differs, and the
un-scaled form is the one that matches the reference implementation's
``nn.CrossEntropyLoss()(logits[bool_masked_pos], labels[bool_masked_pos])`` up to
a constant factor.)

Reference fidelity
------------------
:class:`BeitMaskingGenerator` is a transcription of the official
``microsoft/unilm/beit/masking_generator.py``. Its quirks are reproduced ON
PURPOSE and are load-bearing -- see the class docstring for the itemised list.
Two deliberate departures, both stated at their site:

1. ``dtype=np.int`` in the reference is not merely deprecated but REMOVED in
   modern numpy; ``np.int64`` is used instead.
2. An optional ``rng`` may be injected so tests can be deterministic. The
   reference draws from the global ``random`` module, and that remains the
   default (``rng=None``), so reference behaviour is what you get unless you ask
   for otherwise. **No seeding contract is claimed for the default path.**

Raw TensorFlow ops are used deliberately here: this is a ``tf.data`` transform,
not a Keras layer. The model package stays ``keras.ops``-clean.
"""

import math
import random
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# BEiT v1's ACTUAL command-line defaults, from
# `microsoft/unilm/beit/run_beit_pretraining.py`:
#   --num_mask_patches            default 75
#   --min_mask_patches_per_block  default 16
#   --max_mask_patches_per_block  default None (-> falls back to num_masking_patches)
# On the paper's 14x14 = 196 patch grid, 75 / 196 = 38.3%, which the paper rounds
# to "roughly 40% of the image patches". Note that 16 is NOT the constructor
# default of the reference class (that is 4) -- it is what the pre-training script
# passes, so it is exported here rather than baked into the class.
# ---------------------------------------------------------------------

BEIT_NUM_MASK_PATCHES: int = 75
BEIT_MIN_MASK_PATCHES_PER_BLOCK: int = 16

# ---------------------------------------------------------------------


class BeitMaskingGenerator:
    """Block-wise mask generator for BEiT masked-image-modelling.

    Stamps rectangular blocks of masked patches into an ``(H, W)`` grid until a
    budget of ``num_masking_patches`` cells is reached, or until a placement
    round fails outright. Each candidate block draws a target area uniformly from
    ``[min_num_patches, max_mask_patches]`` and an aspect ratio LOG-uniformly
    from ``[min_aspect, max_aspect]``.

    This is a transcription of ``microsoft/unilm/beit/masking_generator.py``.
    The following behaviours are reproduced deliberately and must not be
    "improved" -- each one changes the mask distribution the reference produces:

    * **Strict rejection.** A candidate is rejected unless ``w < width`` and
      ``h < height`` (strict ``<``), so a block can never span a full grid
      dimension. On small grids this rejects most candidates.
    * **Net-new-area acceptance window.** A candidate is stamped only when
      ``0 < h * w - num_already_masked <= max_mask_patches``. Cells already
      masked inside the rectangle are skipped, not double-counted.
    * **Ten attempts, break on first success.** ``_mask`` tries at most 10
      candidates and breaks out as soon as one of them sets at least one cell.
    * **Early termination / under-fill.** If a ``_mask`` round sets nothing
      (10 consecutive rejections), ``__call__`` gives up and RETURNS a mask with
      FEWER than ``num_masking_patches`` cells set. It does not raise, does not
      retry, and does not fall back to i.i.d. masking. A caller that needs an
      exact count must enforce it itself.

    Two departures from the reference, both deliberate:

    * ``dtype=np.int64`` -- the reference's ``np.int`` was removed in numpy 1.24
      and would raise here.
    * an optional injected ``rng``. The reference samples from the global
      ``random`` module; passing ``rng=None`` (the default) preserves exactly
      that, including its lack of any internal seeding. Pass a
      ``random.Random(seed)`` to obtain reproducible masks -- that is an addition
      for testability, not a property of the reference.

    Args:
        input_size: Patch-grid size. An ``int`` means a square ``(n, n)`` grid;
            a 2-sequence is taken as ``(height, width)``.
        num_masking_patches: Target number of masked cells. Treated as a budget,
            not a guarantee (see the under-fill note above).
        min_num_patches: Lower bound of the per-block target area. Defaults to
            the reference's ``4``; BEiT's pre-training script passes
            :data:`BEIT_MIN_MASK_PATCHES_PER_BLOCK` (16).
        max_num_patches: Upper bound of the per-block target area. ``None``
            defaults to ``num_masking_patches``, i.e. a single block may in
            principle cover the whole budget.
        min_aspect: Lower bound of the block aspect ratio (``height / width``).
        max_aspect: Upper bound of the block aspect ratio. ``None`` defaults to
            ``1 / min_aspect``.
        rng: Optional ``random.Random`` instance to draw from. ``None`` uses the
            global ``random`` module, which is the reference's behaviour.

    Raises:
        ValueError: If the grid dimensions are not positive, if
            ``num_masking_patches`` is not positive or exceeds the grid size, if
            ``min_num_patches`` is not positive or exceeds ``max_num_patches``,
            or if the aspect bounds are not positive with ``min_aspect <=
            max_aspect``.

    Example:
        >>> gen = BeitMaskingGenerator(14, BEIT_NUM_MASK_PATCHES,
        ...                            min_num_patches=BEIT_MIN_MASK_PATCHES_PER_BLOCK,
        ...                            rng=random.Random(0))
        >>> mask = gen()
        >>> mask.shape, mask.dtype
        ((14, 14), dtype('int64'))
    """

    def __init__(
        self,
        input_size: Union[int, Sequence[int]],
        num_masking_patches: int,
        min_num_patches: int = 4,
        max_num_patches: Optional[int] = None,
        min_aspect: float = 0.3,
        max_aspect: Optional[float] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        if isinstance(input_size, int):
            input_size = (input_size,) * 2
        else:
            input_size = tuple(int(v) for v in input_size)
            if len(input_size) != 2:
                raise ValueError(
                    f"input_size must be an int or a 2-sequence (H, W), got {input_size}"
                )
        self.height, self.width = input_size

        if self.height <= 0 or self.width <= 0:
            raise ValueError(
                f"grid dimensions must be positive, got ({self.height}, {self.width})"
            )

        self.num_patches = self.height * self.width

        if num_masking_patches <= 0:
            raise ValueError(
                f"num_masking_patches must be positive, got {num_masking_patches}"
            )
        if num_masking_patches > self.num_patches:
            raise ValueError(
                f"num_masking_patches ({num_masking_patches}) exceeds the grid size "
                f"({self.height} x {self.width} = {self.num_patches})"
            )
        self.num_masking_patches = num_masking_patches

        if min_num_patches <= 0:
            raise ValueError(
                f"min_num_patches must be positive, got {min_num_patches}"
            )
        self.min_num_patches = min_num_patches
        self.max_num_patches = (
            num_masking_patches if max_num_patches is None else max_num_patches
        )
        if self.min_num_patches > self.max_num_patches:
            raise ValueError(
                f"min_num_patches ({self.min_num_patches}) must not exceed "
                f"max_num_patches ({self.max_num_patches})"
            )

        if min_aspect <= 0.0:
            raise ValueError(f"min_aspect must be positive, got {min_aspect}")
        max_aspect = max_aspect or 1.0 / min_aspect
        if max_aspect < min_aspect:
            raise ValueError(
                f"max_aspect ({max_aspect}) must not be below min_aspect ({min_aspect})"
            )
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

        # `random` (the module) exposes the same `uniform` / `randint` surface as a
        # `random.Random` instance, so the two paths are interchangeable here.
        self._rng: Any = random if rng is None else rng

    def get_shape(self) -> Tuple[int, int]:
        """Return the ``(height, width)`` of the patch grid."""
        return self.height, self.width

    def _mask(self, mask: np.ndarray, max_mask_patches: int) -> int:
        """Attempt to stamp one block into ``mask`` in place.

        Args:
            mask: The ``(H, W)`` int64 grid being filled, modified in place.
            max_mask_patches: Remaining budget for this round, already clipped to
                ``self.max_num_patches`` by the caller.

        Returns:
            The number of cells newly set by this round -- ``0`` if all 10
            attempts were rejected, which is the caller's signal to stop.
        """
        delta = 0
        for _attempt in range(10):
            target_area = self._rng.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(self._rng.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            # STRICT `<`: a block may never span a full grid dimension.
            if w < self.width and h < self.height:
                top = self._rng.randint(0, self.height - h)
                left = self._rng.randint(0, self.width - w)
                num_masked = mask[top : top + h, left : left + w].sum()
                # Accept only if the block contributes NET NEW area inside the
                # remaining budget.
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1
                if delta > 0:
                    break
        return delta

    def __call__(self) -> np.ndarray:
        """Draw one block-wise mask.

        Returns:
            An ``(H, W)`` ``int64`` array of 0/1. The number of ones is at most
            ``num_masking_patches`` and MAY BE FEWER -- including zero on a grid
            too small to admit any block. Under-fill is the reference's
            behaviour and is not an error condition.
        """
        # np.int64, not the reference's `np.int` (removed in numpy >= 1.24).
        mask = np.zeros(shape=self.get_shape(), dtype=np.int64)
        mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)
            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                # Under-fill: give up rather than loop forever. Returning a
                # short mask is the documented contract.
                break
            mask_count += delta
        return mask


# ---------------------------------------------------------------------


def make_beit_mim_map_fn(
    tokenizer_fn: Callable[[tf.Tensor], tf.Tensor],
    grid_size: Union[int, Sequence[int]],
    num_masking_patches: int = BEIT_NUM_MASK_PATCHES,
    min_num_patches: int = BEIT_MIN_MASK_PATCHES_PER_BLOCK,
    max_num_patches: Optional[int] = None,
    min_aspect: float = 0.3,
    max_aspect: Optional[float] = None,
    rng: Optional[random.Random] = None,
) -> Callable[..., Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor, tf.Tensor]]:
    """Build the ``tf.data`` map function for BEiT MIM pre-training.

    The returned function maps a single raw image to the element
    ``((image, bool_mask), target_ids, sample_weight)`` described in the module
    docstring.

    All configuration is validated EAGERLY, at construction time (the
    :class:`BeitMaskingGenerator` is instantiated here), so a bad budget fails
    when the pipeline is built rather than a thousand steps into training.

    The tokenizer is supplied by the CALLER; this module imports no tokenizer and
    knows nothing about how the code ids are produced. ``tokenizer_fn`` receives
    the UNBATCHED image tensor and must return integer code ids whose total size
    is the number of patches -- an ``(gh, gw)`` grid and a flat ``(N,)`` vector
    are both accepted, and the result is reshaped to ``(N,)`` and cast to
    ``int32``. It runs inside a ``tf.data`` graph, so it must be built from
    TensorFlow ops. A tokenizer that only accepts batches is wrapped by the
    caller, e.g.::

        tokenizer_fn = lambda img: tokenizer.encode_to_indices(img[None])[0]

    Note that the mask is drawn on the PYTHON side (the generator is numpy +
    ``random``), so it is wrapped in ``tf.numpy_function``. That wrapper returns
    a tensor of statically UNKNOWN shape, which silently breaks downstream
    batching; the shape is therefore re-pinned with ``tf.ensure_shape``
    immediately after the call, and the test suite asserts the emitted element
    spec is fully defined.

    Args:
        tokenizer_fn: Callable mapping an unbatched image to per-patch code ids.
        grid_size: Patch grid, ``int`` for square or ``(gh, gw)``. Must match the
            encoder's patch grid AND the tokenizer's code grid.
        num_masking_patches: Mask budget. Defaults to BEiT's
            :data:`BEIT_NUM_MASK_PATCHES`.
        min_num_patches: Minimum per-block target area. Defaults to BEiT's
            :data:`BEIT_MIN_MASK_PATCHES_PER_BLOCK`, which is the pre-training
            script's value rather than the reference class's own default of 4.
        max_num_patches: Maximum per-block target area, ``None`` -> the budget.
        min_aspect: Minimum block aspect ratio.
        max_aspect: Maximum block aspect ratio, ``None`` -> ``1 / min_aspect``.
        rng: Optional ``random.Random`` for reproducible masks. ``None`` uses the
            global ``random`` module (reference behaviour, no seeding contract).

    Returns:
        A ``tf.data``-mappable callable
        ``image -> ((image, bool_mask), target_ids, sample_weight)``.

    Raises:
        ValueError: Propagated from :class:`BeitMaskingGenerator` for any invalid
            grid / budget / aspect configuration.
    """
    generator = BeitMaskingGenerator(
        input_size=grid_size,
        num_masking_patches=num_masking_patches,
        min_num_patches=min_num_patches,
        max_num_patches=max_num_patches,
        min_aspect=min_aspect,
        max_aspect=max_aspect,
        rng=rng,
    )
    num_patches = generator.num_patches

    logger.info(
        f"BEiT MIM map fn: grid={generator.get_shape()}, N={num_patches}, "
        f"budget={generator.num_masking_patches} "
        f"({generator.num_masking_patches / num_patches:.1%}), "
        f"block_area=[{generator.min_num_patches}, {generator.max_num_patches}], "
        f"aspect=[{generator.min_aspect:.3f}, {generator.max_aspect:.3f}], "
        f"rng={'injected' if rng is not None else 'global random module'}"
    )

    def _draw_mask() -> np.ndarray:
        """Draw one flattened boolean mask on the python side."""
        return generator().reshape(-1).astype(np.bool_)

    def map_fn(
        image: tf.Tensor, *_unused: Any
    ) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor, tf.Tensor]:
        """Map one raw image to the BEiT MIM training element."""
        image = tf.cast(image, tf.float32)

        # `stateful=True` (the default) is required: a stateless py_function may
        # be constant-folded, which would freeze ONE mask for the whole dataset.
        bool_mask = tf.numpy_function(
            func=_draw_mask, inp=[], Tout=tf.bool, stateful=True
        )
        # A numpy_function result has statically UNKNOWN shape. Without this the
        # element spec carries `(None,)` and downstream `.batch()` silently
        # produces ragged-looking specs / late failures.
        bool_mask = tf.ensure_shape(bool_mask, (num_patches,))

        target_ids = tf.cast(tokenizer_fn(image), tf.int32)
        target_ids = tf.reshape(target_ids, (num_patches,))
        target_ids = tf.ensure_shape(target_ids, (num_patches,))

        # Exactly the mask, unscaled: 1.0 at masked positions, 0.0 elsewhere.
        sample_weight = tf.cast(bool_mask, tf.float32)

        return (image, bool_mask), target_ids, sample_weight

    return map_fn

# ---------------------------------------------------------------------
