"""
Segmentation Wrapper Loss — a serializable name-dispatched Keras Loss.

This module promotes the previously inner-class `WrappedLoss` (defined inside
`create_loss_function` in `segmentation_loss.py`) into a first-class library
loss that follows dl_techniques sibling-loss conventions: one-loss-per-module,
bare `@keras.saving.register_keras_serializable()`, snake_case file with the
`_loss.py` suffix, and a PascalCase class with the `Loss` suffix.

The `SegmentationWrapperLoss` selects one of the nine method-based loss
implementations on `SegmentationLosses` (the underlying calculator class
defined in `segmentation_loss.py`) by **name**, and exposes them as a Keras
`Loss` object that round-trips losslessly through `model.save(...)` /
`keras.models.load_model(...)` without `custom_objects` and without
`compile=False` workarounds.

Components
----------
- `SegmentationWrapperLoss`: the public class. Construct directly or via
  `create_segmentation_wrapper_loss` (or via the legacy
  `create_loss_function` factory in `segmentation_loss.py`, which now
  delegates to this class).
- `_LOSS_METHOD_MAP`: module-level mapping from public loss names to the
  corresponding `SegmentationLosses` method names. Used both for dispatch
  and for input validation.

Mathematical Notes
------------------
This module performs no math of its own: it forwards `(y_true, y_pred)` to
the selected method on `SegmentationLosses`. See `segmentation_loss.py` for
the underlying mathematical formulations of each individual loss.

Why a separate module?
----------------------
See plan_2026-05-10_17633038 / D-002. The previous in-file inner-class
implementation could not survive `model.save(...)` round-trips because it
captured a bound method via closure and omitted both `loss_fn` and
`reduction` from `get_config`. Hoisting the class to module scope fixes
those issues; placing it in its own module aligns it with the rest of the
losses package (one loss per module, importable from
`dl_techniques.losses` directly).

Example
-------
    >>> import keras
    >>> from dl_techniques.losses import SegmentationWrapperLoss
    >>> from dl_techniques.losses.segmentation_loss import LossConfig
    >>> loss = SegmentationWrapperLoss(
    ...     'focal_tversky',
    ...     LossConfig(num_classes=3, focal_gamma=2.0, tversky_alpha=0.3),
    ... )
    >>> # Compile and save normally.
    >>> # model.compile(optimizer='adam', loss=loss)
    >>> # model.save('seg.keras')
    >>> # reloaded = keras.models.load_model('seg.keras')   # no custom_objects
"""

from typing import Optional, Dict, Any

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.losses.segmentation_loss import LossConfig, SegmentationLosses

# ---------------------------------------------------------------------
# Module-level dispatch table — single source of truth for loss names.
# Used for validation (in `__init__`) and for `getattr` resolution against
# `SegmentationLosses`. Adding a new loss = add one entry here.
# ---------------------------------------------------------------------

_LOSS_METHOD_MAP: Dict[str, str] = {
    "cross_entropy": "cross_entropy_loss",
    "dice": "dice_loss",
    "focal": "focal_loss",
    "tversky": "tversky_loss",
    "focal_tversky": "focal_tversky_loss",
    "lovasz": "lovasz_softmax_loss",
    "combo": "combo_loss",
    "boundary": "boundary_loss",
    "hausdorff": "hausdorff_distance_loss",
}

# ---------------------------------------------------------------------


# DECISION plan_2026-05-10_17633038/D-002 — promoted from an inner class
# inside `create_loss_function` to its own module so that it can be
# `@register_keras_serializable`-discoverable at load time without
# `custom_objects` and so that it conforms to F-004 sibling-loss conventions.
@keras.saving.register_keras_serializable()
class SegmentationWrapperLoss(keras.losses.Loss):
    """Name-dispatched Keras `Loss` wrapping `SegmentationLosses` methods.

    Selects one of nine segmentation loss implementations on
    `SegmentationLosses` by name, exposing it as a fully serializable
    `keras.losses.Loss` subclass that survives `model.save(...)` /
    `keras.models.load_model(...)` round-trips without `custom_objects`.

    Args:
        loss_name: One of `cross_entropy`, `dice`, `focal`, `tversky`,
            `focal_tversky`, `lovasz`, `combo`, `boundary`, `hausdorff`.
        config: `LossConfig` controlling per-loss parameters. Defaults to
            `LossConfig(num_classes=1)` when not provided.
        name: Optional Keras loss name. Defaults to `loss_name`.
        reduction: Keras reduction strategy. Defaults to
            `'sum_over_batch_size'` (Keras 3 default).
        **kwargs: Forwarded to `keras.losses.Loss`.

    Raises:
        ValueError: If `loss_name` is not one of the nine supported names.

    Example:
        >>> from dl_techniques.losses.segmentation_loss import LossConfig
        >>> loss = SegmentationWrapperLoss(
        ...     'dice', LossConfig(num_classes=3))
        >>> # loss is a keras.losses.Loss; can be passed to model.compile.
    """

    def __init__(
        self,
        loss_name: str,
        config: Optional[LossConfig] = None,
        name: Optional[str] = None,
        reduction: str = "sum_over_batch_size",
        **kwargs,
    ) -> None:
        if loss_name not in _LOSS_METHOD_MAP:
            available = list(_LOSS_METHOD_MAP)
            logger.error(
                f"Unknown loss function: {loss_name}. Available: {available}"
            )
            raise ValueError(
                f"Unknown loss function: {loss_name}. "
                f"Available losses: {available}"
            )
        super().__init__(name=name or loss_name, reduction=reduction, **kwargs)
        self.loss_name = loss_name
        self.config = config if config is not None else LossConfig(num_classes=1)
        self._losses = SegmentationLosses(self.config)
        self.loss_fn = getattr(self._losses, _LOSS_METHOD_MAP[loss_name])

    def call(self, y_true: Any, y_pred: Any) -> Any:
        """Forward `(y_true, y_pred)` to the selected `SegmentationLosses` method.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted probabilities/logits as expected by the
                underlying loss method.

        Returns:
            Scalar loss tensor.
        """
        return self.loss_fn(y_true, y_pred)

    def get_config(self) -> Dict[str, Any]:
        """Return a fully serializable config dict.

        Includes the parent `keras.losses.Loss` config (`name`, `reduction`)
        plus `loss_name` and a Keras-serialized `LossConfig` payload.

        Returns:
            Configuration dictionary suitable for `from_config`.
        """
        config = super().get_config()
        config.update(
            {
                "loss_name": self.loss_name,
                "config": keras.saving.serialize_keras_object(self.config),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SegmentationWrapperLoss":
        """Reconstruct a `SegmentationWrapperLoss` from its config dict.

        Args:
            config: Dict produced by `get_config`.

        Returns:
            A new `SegmentationWrapperLoss` equivalent to the original.
        """
        config = dict(config)  # do not mutate caller's dict
        loss_config_payload = config.pop("config")
        loss_config = keras.saving.deserialize_keras_object(loss_config_payload)
        return cls(config=loss_config, **config)


# ---------------------------------------------------------------------


def create_segmentation_wrapper_loss(
    loss_name: str,
    config: Optional[LossConfig] = None,
) -> SegmentationWrapperLoss:
    """Convenience factory mirroring `create_hrm_loss` / `create_siglip_loss`.

    Args:
        loss_name: See `SegmentationWrapperLoss`.
        config: See `SegmentationWrapperLoss`.

    Returns:
        A configured `SegmentationWrapperLoss` instance.
    """
    logger.info(f"Created SegmentationWrapperLoss: {loss_name}")
    return SegmentationWrapperLoss(loss_name, config)


# ---------------------------------------------------------------------
