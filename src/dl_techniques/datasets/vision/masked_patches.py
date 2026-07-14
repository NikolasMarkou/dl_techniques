"""Masked-patch ``tf.data`` transform for masked-image-modelling (MIM).

This module supplies the DATA-SIDE half of the Energy Transformer's masked
image completion objective. It exists so that the occlusion mask can reach the
LOSS through Keras' sanctioned channel — ``sample_weight`` — which means the
model itself stays a stock ``keras.Model`` and training is a stock
``model.compile(loss="mse")`` + ``model.fit(ds)``. No ``train_step``,
``test_step`` or ``compute_loss`` override is needed anywhere.

Data contract
-------------
``make_masked_patch_map_fn`` returns a function mappable over a ``tf.data``
dataset of raw images, producing the 3-tuple element::

    ((image, input_mask), target_patches, loss_weight)

    image          (H, W, C)  float32   the UNMODIFIED image
    input_mask     (N,)       bool      True  -> the model substitutes its
                                        learnable MASK token for that token
    target_patches (N, P)     float32   P = patch_size * patch_size * C
    loss_weight    (N,)       float32   1{i in S} * (N / n_loss)

Two distinct sets are involved and they are NOT the same set (this is the
paper's 90/10 rule, arXiv:2302.07253 §3):

* the LOSS SET ``S`` — ``n_loss = round(mask_ratio * N)`` tokens drawn per
  sample. These are the tokens the reconstruction loss is computed on.
* the INPUT MASK — a random ``n_input = round(mask_token_frac * n_loss)``
  subset of ``S``. Only these tokens are actually occluded in the model input.

The remaining ``~10%`` of ``S`` keep their TRUE patch embedding but still count
in the loss. The paper reports that the Hopfield network only learns meaningful
filters when un-occluded patches are present in the loss, so ``input_mask`` is
a STRICT subset of the loss set by construction.

The ``N / n_loss`` scale on ``loss_weight`` is what turns Keras'
``sum_over_batch_size`` reduction into exactly ``mean_{i in S} MSE``:
``keras.losses.mse`` reduces the ``P`` axis first, leaving a ``(B, N)``
per-token loss which the reduction divides by ``B * N``. Since
``sum_i loss_weight_i == N`` per sample, the weighted sum divided by ``B * N``
is the per-sample mean over ``S``, averaged over the batch. Verified
empirically at ``N=196`` (delta 0.0 vs a numpy reference).

Raw TensorFlow ops are used deliberately here: this is a ``tf.data`` transform,
not a Keras layer. The model package stays ``keras.ops``-clean.
"""

from typing import Any, Callable, Optional, Tuple

import tensorflow as tf

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


def patchify_targets(image: tf.Tensor, patch_size: int) -> tf.Tensor:
    """Split an image into a row-major sequence of flattened patches.

    The token ORDER and the intra-patch flatten order are load-bearing: token
    ``i`` here must be the same patch that ``PatchEmbedding2D`` embeds at token
    ``i``, otherwise the decoder is trained against spatially scrambled targets
    and the loss still descends while the reconstructions are garbage.

    ``PatchEmbedding2D`` is ``Conv2D(kernel=patch, stride=patch, padding="valid")``
    followed by ``reshape(B, -1, D)``. Its Conv2D kernel has shape
    ``(patch, patch, C, D)`` and it contracts over ``(h, w, c)``; the reshape
    then walks the ``(grid_h, grid_w)`` output grid row-major. Both facts are
    matched exactly by ``tf.image.extract_patches``, whose last axis is a
    ``(row, col, channel)`` row-major flatten of the patch and whose output grid
    is ``(grid_h, grid_w)``. This agreement is proven, not assumed, by the
    identity-kernel test in ``tests/test_datasets/test_masked_patches.py``.

    Args:
        image: Image tensor of shape ``(H, W, C)`` or a batch ``(B, H, W, C)``.
            Cast to ``float32`` internally.
        patch_size: Side length of a square patch. ``H`` and ``W`` must both be
            divisible by it.

    Returns:
        A ``float32`` tensor of shape ``(N, P)`` for a rank-3 input, or
        ``(B, N, P)`` for a rank-4 input, where ``N = (H / patch_size) *
        (W / patch_size)`` and ``P = patch_size * patch_size * C``.

    Raises:
        ValueError: If ``patch_size`` is not positive, if ``image`` is neither
            rank 3 nor rank 4, or if the (statically known) spatial dims are not
            divisible by ``patch_size``.
    """
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")

    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.float32)

    rank = image.shape.rank
    if rank not in (3, 4):
        raise ValueError(
            f"image must be rank 3 (H, W, C) or rank 4 (B, H, W, C), got rank {rank}"
        )

    unbatched = rank == 3
    if unbatched:
        image = tf.expand_dims(image, axis=0)

    height, width, channels = image.shape[1], image.shape[2], image.shape[3]
    for name, dim in (("height", height), ("width", width)):
        if dim is not None and dim % patch_size != 0:
            raise ValueError(
                f"image {name} ({dim}) must be divisible by patch_size ({patch_size})"
            )

    # (B, grid_h, grid_w, patch*patch*C). The final axis is a row-major
    # (row, col, channel) flatten of the patch -- i.e. HWC, matching Conv2D.
    patches = tf.image.extract_patches(
        images=image,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )

    patch_dim = patch_size * patch_size * channels if channels is not None else -1
    batch_size = tf.shape(patches)[0]
    # Row-major flatten of the (grid_h, grid_w) grid -- matches PatchEmbedding2D's
    # reshape(B, -1, D) over the Conv2D output grid.
    patches = tf.reshape(patches, (batch_size, -1, patch_dim))

    if unbatched:
        patches = tf.squeeze(patches, axis=0)

    return patches


def make_masked_patch_map_fn(
    patch_size: int,
    image_size: int,
    mask_ratio: float = 0.5,
    mask_token_frac: float = 0.9,
    seed: Optional[int] = None,
) -> Callable[[tf.Tensor], Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor, tf.Tensor]]:
    """Build the ``tf.data`` map function for masked-patch pretraining.

    The returned function maps a single raw image to the element
    ``((image, input_mask), target_patches, loss_weight)`` described in the
    module docstring. Selection is per-sample and graph-safe (the standard
    argsort-of-uniform-noise random-permutation trick), so it composes with
    ``dataset.map(..., num_parallel_calls=tf.data.AUTOTUNE)``.

    All configuration is validated EAGERLY, at construction time — a bad
    ``mask_ratio`` must fail when the pipeline is built, not silently produce a
    degenerate mask a thousand steps into training.

    Args:
        patch_size: Side length of a square patch.
        image_size: Side length of the (square) image the pipeline feeds. Must
            be divisible by ``patch_size``.
        mask_ratio: Fraction of the ``N`` tokens that enter the LOSS set ``S``.
            The paper's image default is ``0.5``.
        mask_token_frac: Fraction of ``S`` that is additionally OCCLUDED in the
            model input. The paper's value is ``0.9``; the remaining ``~10%``
            of ``S`` stay visible to the model but still count in the loss.
        seed: Optional seed for a deterministic ``tf.random.Generator``. Leave
            ``None`` for a non-deterministic stream in real training.

    Returns:
        A ``tf.data``-mappable callable ``image -> ((image, input_mask),
        target_patches, loss_weight)``.

    Raises:
        ValueError: If ``image_size`` is not divisible by ``patch_size``; if
            ``mask_ratio`` / ``mask_token_frac`` are outside ``(0, 1]``; if the
            resulting ``n_loss`` is 0; if ``n_input`` is 0; or if
            ``n_input == n_loss`` (which would silently delete the ~10%
            un-occluded signal the paper's Hopfield network needs).
    """
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    if image_size % patch_size != 0:
        raise ValueError(
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
        )
    if not 0.0 < mask_ratio <= 1.0:
        raise ValueError(f"mask_ratio must be in (0, 1], got {mask_ratio}")
    if not 0.0 < mask_token_frac <= 1.0:
        raise ValueError(f"mask_token_frac must be in (0, 1], got {mask_token_frac}")

    grid = image_size // patch_size
    num_patches = grid * grid
    n_loss = int(round(mask_ratio * num_patches))
    n_input = int(round(mask_token_frac * n_loss))

    if n_loss < 1:
        raise ValueError(
            f"mask_ratio ({mask_ratio}) gives n_loss=0 at num_patches={num_patches}: "
            f"the loss would be computed on zero tokens"
        )
    if n_input < 1:
        raise ValueError(
            f"mask_token_frac ({mask_token_frac}) gives n_input=0 at n_loss={n_loss}: "
            f"no token would ever be occluded and the task would be an identity map"
        )
    if n_input == n_loss:
        # Invariant I2 (plan-2026-07-14T163315-29a4fef4): reject, do NOT clamp.
        # n_input == n_loss means EVERY loss token is occluded, which deletes the
        # ~10% un-occluded-but-in-the-loss patches the paper (arXiv:2302.07253 §3)
        # says the Hopfield network needs to learn meaningful filters. Silently
        # clamping n_input to n_loss-1 would train a subtly different (and worse)
        # objective than the config asks for, and nothing downstream would notice.
        raise ValueError(
            f"mask_token_frac ({mask_token_frac}) gives n_input == n_loss == {n_loss} "
            f"at num_patches={num_patches}: every loss token would be occluded, deleting "
            f"the un-occluded patches the loss set must retain (the paper's 90/10 rule). "
            f"Lower mask_token_frac or raise num_patches."
        )

    rng = (
        tf.random.Generator.from_seed(seed)
        if seed is not None
        else tf.random.Generator.from_non_deterministic_state()
    )

    logger.info(
        f"masked-patch map fn: image_size={image_size}, patch_size={patch_size}, "
        f"N={num_patches}, n_loss={n_loss}, n_input={n_input}, "
        f"loss_weight_scale={num_patches / n_loss:.6f}, seed={seed}"
    )

    weight_scale = tf.constant(num_patches / n_loss, dtype=tf.float32)

    def map_fn(
        image: tf.Tensor, *_unused: Any
    ) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor, tf.Tensor]:
        """Map one raw image to the masked-patch training element."""
        image = tf.cast(image, tf.float32)
        target_patches = patchify_targets(image, patch_size)  # (N, P)

        # Random permutation of [0, N): rank[i] is the position of token i in a
        # uniformly random shuffle. rank < k therefore selects a uniformly random
        # k-subset, and rank < n_input is a random subset of rank < n_loss --
        # which is exactly what makes input_mask a STRICT subset of the loss set.
        noise = rng.uniform(shape=(num_patches,), dtype=tf.float32)
        rank = tf.argsort(tf.argsort(noise, axis=-1), axis=-1)

        loss_set = rank < n_loss  # (N,) bool
        input_mask = rank < n_input  # (N,) bool, strict subset of loss_set

        loss_weight = tf.cast(loss_set, tf.float32) * weight_scale  # (N,) float32

        # Static shapes: tf.data needs them for a well-defined batch spec.
        input_mask = tf.ensure_shape(input_mask, (num_patches,))
        loss_weight = tf.ensure_shape(loss_weight, (num_patches,))

        return (image, input_mask), target_patches, loss_weight

    return map_fn

# ---------------------------------------------------------------------
