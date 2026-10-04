"""
PeriodicReassignCallback — re-estimate a hierarchical head's assignment.

Calls ``head.reassign(class_means)`` every ``every_n_epochs`` epochs, where
the class means are trunk-feature means over the provided training arrays.
Keeps the learned taxonomy in step with the trunk as it trains, without a
custom training loop.

The head is resolved from the trained model at epoch end: pass it
directly, or by (top-level) layer name. The feature extractor is either
passed directly as a model (required when the trunk is a nested submodel,
whose inner layers ``get_layer`` cannot reach) or resolved by layer name.
Only names ride in ``get_config``; data arrays and live objects do not. A
head without a ``reassign`` method, or a class absent from ``y``, raises
rather than silently skipping.
"""

from typing import List, Optional, Union

import numpy as np

import keras

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.callbacks.periodic_reassign")
class PeriodicReassignCallback(keras.callbacks.Callback):
    """Refresh a hierarchical head's class-to-slot assignment on schedule.

    :param head: Head layer exposing ``reassign(representatives)`` (e.g.
        ``HierarchicalHarmonicHead``), or its top-level layer name.
    :param x: Training inputs, ``(N, ...)`` numpy array. Held by
        reference, not copied.
    :param y: Integer training labels, ``(N,)``. Every class in
        ``range(num_classes)`` must appear at least once.
    :param num_classes: Number of classes.
    :param feature_model: ``keras.Model`` mapping model inputs to the
        head's input space. Required when the trunk is nested; otherwise
        ``feature_layer_name`` is resolved from the trained model.
    :param feature_layer_name: Top-level layer name whose output is the
        head's input space. Used only when ``feature_model`` is None.
    :param every_n_epochs: Reassign when ``(epoch + 1) % every_n_epochs == 0``.
        Must be >= 1.
    :param start_epoch: 1-based epoch of the first reassignment. Must be >= 1.
    :param batch_size: Batch size for the feature-extraction predict pass.
    """

    def __init__(
        self,
        head: Union[keras.layers.Layer, str],
        x: np.ndarray,
        y: np.ndarray,
        num_classes: int,
        feature_model: Optional[keras.Model] = None,
        feature_layer_name: Optional[str] = None,
        every_n_epochs: int = 2,
        start_epoch: int = 1,
        batch_size: int = 512,
    ) -> None:
        super().__init__()
        if every_n_epochs < 1:
            raise ValueError(f"every_n_epochs must be >= 1, got {every_n_epochs}")
        if start_epoch < 1:
            raise ValueError(f"start_epoch must be >= 1, got {start_epoch}")
        x = np.asarray(x)
        y = np.asarray(y)
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"x and y disagree on sample count: {x.shape[0]} vs {y.shape[0]}"
            )
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) \
                or num_classes <= 0:
            raise ValueError(
                f"num_classes must be a positive integer, got {num_classes}"
            )
        missing = sorted(set(range(num_classes)) - set(np.unique(y).tolist()))
        if missing:
            raise ValueError(
                f"Classes {missing} have no samples in y; class means would "
                f"be undefined. Pass the full training labels."
            )
        self.head_ref = head
        self.feature_ref = feature_model
        self.feature_layer_name = feature_layer_name
        self.x = x
        self.y = y
        self.num_classes = num_classes
        self.every_n_epochs = int(every_n_epochs)
        self.start_epoch = int(start_epoch)
        self.batch_size = int(batch_size)
        self._extractor: Optional[keras.Model] = None

    # ------------------------------------------------------------------
    def _due(self, epoch: int) -> bool:
        """Return whether epoch (0-based) triggers a reassignment."""
        return (epoch + 1) >= self.start_epoch and (epoch + 1) % self.every_n_epochs == 0

    def _resolve(self):
        """Resolve (head, extractor), building the extractor once."""
        if self.model is None:
            raise ValueError("Callback is not attached to a model.")
        head = self.head_ref
        if isinstance(head, str):
            try:
                head = self.model.get_layer(head)
            except ValueError as e:
                raise ValueError(
                    f"Head layer '{self.head_ref}' not found among top-level "
                    f"model layers {[l.name for l in self.model.layers]}."
                ) from e
        if not callable(getattr(head, "reassign", None)):
            raise ValueError(
                f"Head '{getattr(head, 'name', head)}' "
                f"({type(head).__name__}) has no reassign() method."
            )
        if self._extractor is None:
            if self.feature_ref is not None:
                self._extractor = self.feature_ref
            elif self.feature_layer_name is not None:
                try:
                    feature_output = self.model.get_layer(
                        self.feature_layer_name
                    ).output
                except ValueError as e:
                    raise ValueError(
                        f"Feature layer '{self.feature_layer_name}' not found "
                        f"among top-level model layers."
                    ) from e
                self._extractor = keras.Model(self.model.input, feature_output)
            else:
                raise ValueError(
                    "Neither feature_model nor feature_layer_name was given."
                )
        return head

    # ------------------------------------------------------------------
    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        if not self._due(epoch):
            return
        head = self._resolve()
        assert self._extractor is not None
        features = self._extractor.predict(self.x, batch_size=self.batch_size, verbose=0)
        means = np.stack([
            np.asarray(features)[np.asarray(self.y) == c].mean(axis=0)
            for c in range(self.num_classes)
        ])
        stats = head.reassign(means)
        logger.info(
            f"PeriodicReassignCallback: epoch {epoch + 1}: "
            f"reassigned {stats.get('moved', '?')} of {self.num_classes} classes."
        )

    def get_config(self) -> dict:
        # Data arrays (x, y) and live objects (head layer, feature model)
        # are deliberately excluded: they are large or caller-owned.
        # Reattach them when rebuilding by hand.
        head_name = self.head_ref if isinstance(self.head_ref, str) else None
        return {
            "head_layer_name": head_name,
            "feature_layer_name": self.feature_layer_name,
            "num_classes": self.num_classes,
            "every_n_epochs": self.every_n_epochs,
            "start_epoch": self.start_epoch,
            "batch_size": self.batch_size,
        }
