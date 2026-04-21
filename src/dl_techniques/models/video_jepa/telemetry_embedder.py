"""Telemetry embedder (D-004 + D-006).

Maps a raw per-frame telemetry vector ``(B, T, k)`` to a conditioning
embedding ``(B, T, cond_dim)`` via continuous sinusoidal encoding followed
by a small LayerNorm+Dense refinement. The output is consumed by
:class:`AdaLNZeroConditionalBlock` inside the predictor.

Forward path:

.. code-block:: text

    telemetry : (B, T, k)
            │
            ▼  ContinuousSinCosEmbed(dim=cond_dim, ndim=k, assert_positive=False)
    pe        : (B, T, cond_dim)
            │
            ▼  LayerNorm
            ▼  Dense(cond_dim)
    c         : (B, T, cond_dim)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import keras

from dl_techniques.layers.embedding.continuous_sin_cos_embedding import (
    ContinuousSinCosEmbed,
)


@keras.saving.register_keras_serializable()
class TelemetryEmbedder(keras.layers.Layer):
    """Continuous sin/cos telemetry encoder with LayerNorm+Dense refinement.

    :param cond_dim: Output conditioning dimension (must equal
        ``embed_dim`` for downstream AdaLN modulation to broadcast).
    :param telemetry_dim: Number of raw telemetry channels ``k``.
    :param max_wavelength: Sin/cos maximum wavelength (default 10_000).
    :param kwargs: passthrough to :class:`keras.layers.Layer`.
    """

    def __init__(
        self,
        cond_dim: int,
        telemetry_dim: int,
        max_wavelength: float = 10000.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if cond_dim <= 0:
            raise ValueError(f"cond_dim must be positive, got {cond_dim}")
        if telemetry_dim <= 0:
            raise ValueError(
                f"telemetry_dim must be positive, got {telemetry_dim}"
            )

        self.cond_dim = cond_dim
        self.telemetry_dim = telemetry_dim
        self.max_wavelength = max_wavelength

        # assert_positive=False — IMU deltas / velocities can be signed.
        self.sincos = ContinuousSinCosEmbed(
            dim=cond_dim,
            ndim=telemetry_dim,
            max_wavelength=max_wavelength,
            assert_positive=False,
            name="sincos",
        )
        self.norm = keras.layers.LayerNormalization(
            epsilon=1e-6, name="ln",
        )
        self.proj = keras.layers.Dense(
            cond_dim, name="proj",
        )

    # ------------------------------------------------------------------
    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        if len(input_shape) != 3:
            raise ValueError(
                f"TelemetryEmbedder expects 3D input (B, T, k); got rank "
                f"{len(input_shape)} shape {input_shape}"
            )
        if input_shape[-1] != self.telemetry_dim:
            raise ValueError(
                f"Last dim of input ({input_shape[-1]}) must equal "
                f"telemetry_dim ({self.telemetry_dim})."
            )
        self.sincos.build(input_shape)
        emb_shape = (input_shape[0], input_shape[1], self.cond_dim)
        self.norm.build(emb_shape)
        self.proj.build(emb_shape)
        super().build(input_shape)

    # ------------------------------------------------------------------
    def call(
        self,
        telemetry: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Embed per-frame telemetry ``(B, T, k) → (B, T, cond_dim)``."""
        x = self.sincos(telemetry, training=training)
        x = self.norm(x)
        x = self.proj(x)
        return x

    # ------------------------------------------------------------------
    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        return (input_shape[0], input_shape[1], self.cond_dim)

    # ------------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "cond_dim": self.cond_dim,
            "telemetry_dim": self.telemetry_dim,
            "max_wavelength": self.max_wavelength,
        })
        return config
