"""Shared plumbing for the three BEiT trainers.

Deliberately SMALL. It holds exactly three things, and two of them are not written here
at all:

1. :func:`build_raw_image_dataset` — RE-EXPORTED from ``train.common.datasets``.
   The raw-image ``tf.data`` pipeline (imagenette via tfds, cifar10 in-memory) with the
   ``element_map_fn`` hook the MIM trainer needs.
2. :func:`build_optimizer` — RE-EXPORTED from ``train.common.optimizer``. The
   ``learning_rate_schedule_builder`` / ``optimizer_builder`` block.
3. :func:`load_frozen_tokenizer` — the only genuinely BEiT-specific helper: load a saved
   VQ-VAE, freeze it, VERIFY its code grid, and hand back a callable that produces the
   MIM targets.

There is NO shared config dataclass and NO shared ``train()`` orchestrator: each trainer
owns its own ``TrainingConfig``, ``parse_arguments()`` and ``config_from_args()`` so that
an argparse flag which never reaches the config is a LOCAL, greppable, testable defect
rather than an inherited one.
"""

import os
from typing import Any, Callable, Optional, Sequence

import keras
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

# DECISION plan-2026-08-11T012340-f63796dc/D-008
# `build_raw_image_dataset` and `build_optimizer` are IMPORTED, not re-implemented.
# WHAT NOT TO DO: do NOT copy either function's body into this file "so the BEiT package
# is self-contained". Both are already dataset/optimizer-generic and already support
# exactly the two datasets BEiT needs (imagenette + cifar10) and exactly the
# `element_map_fn` hook the MIM trainer needs; a copy would be a ~230-line fork of a
# pipeline that carries THREE load-bearing in-code decision anchors (D-007/D-008/D-040 of
# two other plans) about branch ordering that a fork would silently drift away from.
# The cross-package import is the house convention, not an exception: `train/dino/
# train_dino.py:237` imports these same two names, and
# `train/graph_energy_transformer/common.py:24` re-exports `build_optimizer` the same way.
# BOTH functions' HOME is now under `train.common`, not `train.energy_transformer.common`
# where they were originally written (each is consumed by four packages, so both were
# promoted: the pipeline to `train.common.datasets`, the optimizer adapter to
# `train.common.optimizer`). `train.energy_transformer.common` re-exports both and those
# paths still resolve to the SAME OBJECTS.
# See decisions.md D-008.
from train.common.datasets import (  # noqa: F401  (re-exported)
    DATASET_NUM_CLASSES,
    SUPPORTED_DATASETS,
    build_raw_image_dataset,
)
from train.common.optimizer import build_optimizer  # noqa: F401  (re-exported)
from dl_techniques.utils.logger import logger
from dl_techniques.models.vq_vae_rotation.model import VQVAERotationTrick

__all__ = [
    "DATASET_NUM_CLASSES",
    "SUPPORTED_DATASETS",
    "build_optimizer",
    "build_raw_image_dataset",
    "load_frozen_tokenizer",
]


# ---------------------------------------------------------------------
# frozen discrete visual tokenizer (stage 0 -> stage 1)
# ---------------------------------------------------------------------

def load_frozen_tokenizer(
        path: str,
        expected_grid: Sequence[int],
        image_shape: Sequence[int],
        custom_objects: Optional[dict] = None,
) -> Callable[[Any], Any]:
    """Load a stage-0 VQ-VAE tokenizer, freeze it, and verify its code grid.

    The returned callable maps a BATCHED image tensor ``(B, H, W, C)`` to integer code
    ids ``(B, gh, gw)`` via :meth:`VQVAERotationTrick.encode_to_indices`. The BEiT MIM
    map function wants an UNBATCHED image, so wrap it at the call site::

        tokenizer_fn = load_frozen_tokenizer(path, (14, 14), (224, 224, 3))
        map_fn = make_beit_mim_map_fn(
            tokenizer_fn=lambda img: tokenizer_fn(img[None])[0],
            grid_size=(14, 14),
        )

    Every failure here is a LOUD raise, never a warning. A tokenizer whose code grid does
    not line up with the encoder's patch grid produces a finite, plausible, completely
    wrong MIM loss: every target would be read from the wrong spatial position with no
    error anywhere in the pipeline.

    The grid is verified by an ACTUAL forward pass at the caller's ``image_shape``, not
    derived from ``expected_grid`` and the tokenizer's own ``downsample_factor``. Deriving
    it would make the check self-referential — it would agree with the implementation by
    construction and could never fail.

    Args:
        path: Path to a ``.keras`` file saved from a :class:`VQVAERotationTrick`.
        expected_grid: The ``(gh, gw)`` code grid the caller requires — i.e. the BEiT
            encoder's patch grid ``(H // patch_h, W // patch_w)``.
        image_shape: The ``(H, W, C)`` image shape the trainer will actually feed. The
            probe runs at this shape; a tokenizer that only produces ``expected_grid`` at
            some OTHER resolution is rejected.
        custom_objects: Forwarded to ``keras.models.load_model``.

    Returns:
        A callable ``(B, H, W, C) -> (B, gh, gw)`` int code ids. The frozen model itself
        is attached as the ``.model`` attribute of the returned callable, so a caller (or
        a test) can inspect ``fn.model.trainable`` without reaching into a closure.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If ``path`` is not a ``.keras`` file, or ``expected_grid`` /
            ``image_shape`` are malformed, or the tokenizer's code grid at
            ``image_shape`` differs from ``expected_grid``.
        TypeError: If the loaded object is not a :class:`VQVAERotationTrick`.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tokenizer checkpoint not found: {path}")
    if not str(path).endswith(".keras"):
        raise ValueError(
            f"load_frozen_tokenizer only supports .keras files; got {path!r}. "
            "Re-save the stage-0 tokenizer via model.save('path.keras')."
        )

    expected_grid = tuple(int(v) for v in expected_grid)
    if len(expected_grid) != 2 or expected_grid[0] <= 0 or expected_grid[1] <= 0:
        raise ValueError(
            f"expected_grid must be a positive 2-sequence (gh, gw), got {expected_grid}"
        )
    image_shape = tuple(int(v) for v in image_shape)
    if len(image_shape) != 3 or any(v <= 0 for v in image_shape):
        raise ValueError(
            f"image_shape must be a positive 3-sequence (H, W, C), got {image_shape}"
        )

    tokenizer = keras.models.load_model(path, custom_objects=custom_objects)

    # A plain `.keras` load happily returns whatever was saved. A Sequential, a
    # classifier, or a plain VQ-VAE with a different API would fail LATER, inside the
    # tf.data graph, with an AttributeError nobody can trace back to here.
    if not isinstance(tokenizer, VQVAERotationTrick):
        raise TypeError(
            f"{path!r} holds a {type(tokenizer).__name__}, not a VQVAERotationTrick. "
            "The BEiT MIM target is the code id of a vector-quantized tokenizer; only a "
            "VQVAERotationTrick checkpoint (stage 0, `train_tokenizer.py`) is accepted."
        )

    tokenizer.trainable = False

    # The probe both VERIFIES the grid and builds the model eagerly, which matters: the
    # returned callable runs inside a tf.data graph, and a lazily-built sub-layer would
    # create its variables there.
    probe = np.zeros((1,) + image_shape, dtype="float32")
    try:
        code_ids = tokenizer.encode_to_indices(probe)
    except ValueError as exc:
        # MEASURED: an auto-built VQVAERotationTrick pins its encoder's `InputSpec` to
        # the `input_shape` it was constructed with, so a different resolution does not
        # merely produce a different grid -- it REFUSES the call with `Input 0 of layer
        # "vqvae_rotation_encoder" is incompatible ...`, a message naming neither this
        # checkpoint nor what to do about it.
        raise ValueError(
            f"Tokenizer {path!r} cannot encode an image of shape {image_shape}: {exc}. "
            "The stage-0 tokenizer was auto-built at a FIXED input shape; retrain it at "
            "the trainer's image size, or run the trainer at the tokenizer's."
        ) from exc
    actual_grid = tuple(int(v) for v in keras.ops.shape(code_ids)[1:])

    if actual_grid != expected_grid:
        raise ValueError(
            f"Tokenizer code-grid mismatch: {path!r} produces a {actual_grid} code grid "
            f"at image_shape={image_shape} (downsample_factor="
            f"{tokenizer.downsample_factor}), but the BEiT encoder's patch grid is "
            f"{expected_grid}. Training would read every MIM target from the wrong "
            "spatial position and still produce a finite, plausible loss. Retrain the "
            "tokenizer at a downsample_factor that lands on the patch grid, or change "
            "the encoder's patch_size / image_size to match."
        )

    logger.info(
        f"frozen tokenizer: {path} -> grid {actual_grid} at image_shape={image_shape}, "
        f"codebook={tokenizer.num_embeddings}, embedding_dim={tokenizer.embedding_dim}, "
        f"downsample_factor={tokenizer.downsample_factor}, "
        f"trainable={tokenizer.trainable}"
    )

    def tokenize(images: Any) -> Any:
        """Map a BATCHED image tensor to ``(B, gh, gw)`` int code ids."""
        return tf.cast(tokenizer.encode_to_indices(images), tf.int32)

    # Exposed so callers/tests can assert on the frozen model without a closure hack.
    tokenize.model = tokenizer  # type: ignore[attr-defined]
    tokenize.grid_size = actual_grid  # type: ignore[attr-defined]
    return tokenize

# ---------------------------------------------------------------------
