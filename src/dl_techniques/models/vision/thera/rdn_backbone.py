"""THERA's RDN (Residual Dense Network) feature backbone as a Keras layer.

`RDNBackbone` is the RDN feature extractor without its upsampling tail: it
maps an input image `(B, H, W, n_colors)` to a dense feature map
`(B, H, W, growth_rate_0)` at the input resolution, for downstream THERA
components to resample. `RDBConv` and `RDB` are the building blocks: dense
per-unit channel concatenation inside each block, collapsed back to
`growth_rate_0` by a 1x1 local-fusion convolution, plus a block-level residual.

Channel growth through the dense concatenation is closed-form rather than
data-dependent, so every inner `Conv2D` is built with an explicitly computed
input-channel shape in `build()` instead of an eager dummy forward.

References:
    - Becker et al. Thera: Aliasing-Free Arbitrary-Scale Super-Resolution with
      Neural Heat Fields.
    - Zhang et al., 2018. Residual Dense Network for Image Super-Resolution.
      CVPR 2018.
"""

import keras
from keras import ops
from typing import Any, Dict, List, Optional, Tuple

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------------
# config name -> (num_rdb_D, num_conv_layers_C, growth_rate_G)
# ---------------------------------------------------------------------------
_RDN_CONFIGS: Dict[str, Tuple[int, int, int]] = {
    "A": (20, 6, 32),
    "B": (16, 8, 64),
}


@register_dl_technique("dl_techniques.models.thera.rdn_backbone")
class RDBConv(keras.layers.Layer):
    """Single dense-connected conv unit of a Residual Dense Block.

    Applies a `(kernel_size, kernel_size)` convolution with ReLU, then
    concatenates the result onto the input along the channel axis, so the
    output has `C_in + growth_rate` channels.

    Architecture:

    .. code-block:: text

        x [B, H, W, C_in]
        │
        ├──────────────────────────┐
        ▼                          │
        ┌────────────────┐          │
        │ conv kxk + relu │          │
        └────────┬────────┘          │
                  ▼                  │
                 out                 │
                  │                  │
                  └────► concat ◄────┘
                           │
                           ▼
              [B, H, W, C_in + growth_rate]

    :param growth_rate: Number of feature maps produced by the inner
        convolution, the channel growth per unit. Must be positive.
    :type growth_rate: int
    :param kernel_size: Spatial size of the square convolution kernel.
    :type kernel_size: int
    :param kwargs: Forwarded to :class:`keras.layers.Layer`.

    Input shape:
        4D tensor ``(batch, height, width, C_in)`` (NHWC).

    Output shape:
        4D tensor ``(batch, height, width, C_in + growth_rate)``.
    """

    def __init__(
        self,
        growth_rate: int,
        kernel_size: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if growth_rate <= 0:
            raise ValueError(f"growth_rate must be positive, got {growth_rate}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")

        self.growth_rate = growth_rate
        self.kernel_size = kernel_size

        self.conv = keras.layers.Conv2D(
            filters=growth_rate,
            kernel_size=kernel_size,
            padding="same",
            name="conv",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the inner conv before `super().build()`, for reload-safe weights."""
        self.conv.build(input_shape)
        super().build(input_shape)

    def call(self, x: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        out = ops.relu(self.conv(x, training=training))
        return ops.concatenate([x, out], axis=-1)

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        shape = list(input_shape)
        c_in = shape[-1]
        shape[-1] = None if c_in is None else c_in + self.growth_rate
        return tuple(shape)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "growth_rate": self.growth_rate,
                "kernel_size": self.kernel_size,
            }
        )
        return config


@register_dl_technique("dl_techniques.models.thera.rdn_backbone")
class RDB(keras.layers.Layer):
    """Residual Dense Block: `C` dense conv units, local fusion, residual.

    Runs `num_conv_layers` :class:`RDBConv` units, each growing the channel
    count by `growth_rate`, fuses the densely concatenated stack back to
    `growth_rate_0` channels with a 1x1 convolution, and adds the block input
    as a residual. Input and output both have `growth_rate_0` channels.

    Architecture:

    .. code-block:: text

        x [B, H, W, G0]
        │
        ├────────────────────────────────┐ (residual)
        ▼                                │
        ┌────────────────────┐            │
        │ RDBConv x C          │            │  dense growth: G0 -> G0+C*G
        └──────────┬──────────┘            │
                    ▼                      │
        ┌────────────────────┐            │
        │ conv 1x1, same       │            │  local feature fusion -> G0
        └──────────┬──────────┘            │
                    ▼                      │
                   add ◄────────────────────┘
                    │
                    ▼
            output [B, H, W, G0]

    :param growth_rate_0: The block's input/output channel count `G0`; the
        local-fusion conv collapses the dense stack back to this.
    :type growth_rate_0: int
    :param growth_rate: Per-unit channel growth `G`.
    :type growth_rate: int
    :param num_conv_layers: Number of :class:`RDBConv` units `C`.
    :type num_conv_layers: int
    :param kernel_size: Conv kernel size for the dense units.
    :type kernel_size: int
    :param kwargs: Forwarded to :class:`keras.layers.Layer`.

    Input shape:
        4D tensor ``(batch, height, width, growth_rate_0)``.

    Output shape:
        4D tensor ``(batch, height, width, growth_rate_0)``.
    """

    def __init__(
        self,
        growth_rate_0: int,
        growth_rate: int,
        num_conv_layers: int,
        kernel_size: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if growth_rate_0 <= 0:
            raise ValueError(f"growth_rate_0 must be positive, got {growth_rate_0}")
        if growth_rate <= 0:
            raise ValueError(f"growth_rate must be positive, got {growth_rate}")
        if num_conv_layers <= 0:
            raise ValueError(
                f"num_conv_layers must be positive, got {num_conv_layers}"
            )

        self.growth_rate_0 = growth_rate_0
        self.growth_rate = growth_rate
        self.num_conv_layers = num_conv_layers
        self.kernel_size = kernel_size

        self.conv_units: List[RDBConv] = [
            RDBConv(growth_rate=growth_rate, kernel_size=kernel_size, name=f"rdb_conv_{c}")
            for c in range(num_conv_layers)
        ]
        # Local feature fusion: 1x1 conv collapsing the dense stack -> G0.
        self.local_fusion = keras.layers.Conv2D(
            filters=growth_rate_0,
            kernel_size=1,
            padding="same",
            name="local_fusion",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build each sublayer with its exact propagated shape, for reload-safe weights.

        Entering the block the tensor has `G0` channels; after the `c`-th
        `RDBConv` it has `G0 + c*G` channels.
        """
        shape = list(input_shape)
        running_channels = shape[-1]
        for unit in self.conv_units:
            unit.build(tuple(shape))
            if running_channels is not None:
                running_channels = running_channels + self.growth_rate
            shape[-1] = running_channels
        # After all C units, `shape` carries G0 + C*G channels -> local fusion.
        self.local_fusion.build(tuple(shape))
        super().build(input_shape)

    def call(self, x: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        res = x
        for unit in self.conv_units:
            x = unit(x, training=training)
        x = self.local_fusion(x, training=training)
        return x + res

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        shape = list(input_shape)
        shape[-1] = self.growth_rate_0
        return tuple(shape)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "growth_rate_0": self.growth_rate_0,
                "growth_rate": self.growth_rate,
                "num_conv_layers": self.num_conv_layers,
                "kernel_size": self.kernel_size,
            }
        )
        return config


@register_dl_technique("dl_techniques.models.thera.rdn_backbone")
class RDNBackbone(keras.layers.Layer):
    """THERA's RDN feature backbone, with no upsampling.

    Two shallow convs produce `f_1` and the first RDB input; `D` stacked
    :class:`RDB` blocks each emit a `G0`-channel feature map; all `D` outputs
    are concatenated and fused back to `G0` by a 1x1 then a kxk convolution;
    the shallow feature `f_1` is added back. The result is a feature map at
    the input resolution, for a downstream arbitrary-scale component to resample.

    Architecture:

    .. code-block:: text

        x [B, H, W, n_colors]
        │
        ┌─────▼─────┐
        │ conv kxk    │  = f_1
        └─────┬─────┘
              ├─────────────────────────────┐ (residual)
              ▼                              │
        ┌─────────────┐                      │
        │ conv kxk      │                      │
        └─────┬───────┘                      │
              ▼                              │
        ┌─────────────┐                      │
        │ RDB x D       │  collect each output │
        └─────┬───────┘                      │
              ▼                              │
        concat(D outputs)  [D*G0 channels]     │
              ▼                              │
        ┌─────────────┐                      │
        │ conv 1x1      │                      │
        └─────┬───────┘                      │
              ▼                              │
        ┌─────────────┐                      │
        │ conv kxk      │                      │
        └─────┬───────┘                      │
              ▼                              │
             add  ◄──────────────────────────┘
              ▼
        features [B, H, W, G0]

    :param growth_rate_0: Base channel width `G0`, also the output channel count.
    :type growth_rate_0: int
    :param kernel_size: kxk conv kernel size for the shallow/global convs and
        the RDB dense units.
    :type kernel_size: int
    :param config: One of `"A"`, `"B"`, selecting the RDN depth/width preset:
        `"A"` is `(D=20, C=6, G=32)`, `"B"` is `(D=16, C=8, G=64)`.
    :type config: str
    :param kwargs: Forwarded to :class:`keras.layers.Layer`.

    Input shape:
        4D tensor ``(batch, height, width, n_colors)`` (NHWC).

    Output shape:
        4D tensor ``(batch, height, width, growth_rate_0)``.

    :raises ValueError: If ``config`` is not one of ``{"A", "B"}``, or if
        ``growth_rate_0`` / ``kernel_size`` are non-positive.
    """

    def __init__(
        self,
        growth_rate_0: int = 64,
        kernel_size: int = 3,
        config: str = "B",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if config not in _RDN_CONFIGS:
            raise ValueError(
                f"config must be one of {sorted(_RDN_CONFIGS)}, got {config!r}"
            )
        if growth_rate_0 <= 0:
            raise ValueError(f"growth_rate_0 must be positive, got {growth_rate_0}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")

        self.growth_rate_0 = growth_rate_0
        self.kernel_size = kernel_size
        self.config = config

        num_rdb_d, num_conv_c, growth_rate_g = _RDN_CONFIGS[config]
        self.num_rdb = num_rdb_d
        self.num_conv_layers = num_conv_c
        self.growth_rate = growth_rate_g

        logger.info(
            f"RDNBackbone config={config!r}: D={num_rdb_d}, C={num_conv_c}, "
            f"G={growth_rate_g}, G0={growth_rate_0}, k={kernel_size}"
        )

        # Shallow feature extraction (two kxk convs -> G0).
        self.conv_a = keras.layers.Conv2D(
            filters=growth_rate_0, kernel_size=kernel_size, padding="same", name="sfe_a"
        )
        self.conv_b = keras.layers.Conv2D(
            filters=growth_rate_0, kernel_size=kernel_size, padding="same", name="sfe_b"
        )

        # D residual dense blocks (each G0 -> G0).
        self.rdbs: List[RDB] = [
            RDB(
                growth_rate_0=growth_rate_0,
                growth_rate=growth_rate_g,
                num_conv_layers=num_conv_c,
                kernel_size=kernel_size,
                name=f"rdb_{i}",
            )
            for i in range(num_rdb_d)
        ]

        # Global feature fusion: 1x1 (D*G0 -> G0) then kxk (G0 -> G0).
        self.gff_1x1 = keras.layers.Conv2D(
            filters=growth_rate_0, kernel_size=1, padding="same", name="gff_1x1"
        )
        self.gff_kxk = keras.layers.Conv2D(
            filters=growth_rate_0, kernel_size=kernel_size, padding="same", name="gff_kxk"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        # Build every sublayer with explicitly-propagated channel shapes BEFORE
        # super().build() so `.keras` reload restores all weights.
        self.conv_a.build(input_shape)

        # conv_a -> f_1 with G0 channels; conv_b consumes f_1.
        g0_shape = list(input_shape)
        g0_shape[-1] = self.growth_rate_0
        g0_shape = tuple(g0_shape)
        self.conv_b.build(g0_shape)

        # Each RDB consumes and emits a G0-channel tensor.
        for rdb in self.rdbs:
            rdb.build(g0_shape)

        # Global dense feature fusion concatenates D RDB outputs -> D*G0 channels.
        concat_shape = list(input_shape)
        concat_shape[-1] = self.num_rdb * self.growth_rate_0
        concat_shape = tuple(concat_shape)
        self.gff_1x1.build(concat_shape)
        # gff_1x1 -> G0 channels; gff_kxk consumes that.
        self.gff_kxk.build(g0_shape)

        super().build(input_shape)

    def call(self, x: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        f_1 = self.conv_a(x, training=training)
        x = self.conv_b(f_1, training=training)

        rdb_outputs: List[keras.KerasTensor] = []
        for rdb in self.rdbs:
            x = rdb(x, training=training)
            rdb_outputs.append(x)

        x = ops.concatenate(rdb_outputs, axis=-1)
        x = self.gff_1x1(x, training=training)
        x = self.gff_kxk(x, training=training)
        return x + f_1

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        shape = list(input_shape)
        shape[-1] = self.growth_rate_0
        return tuple(shape)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "growth_rate_0": self.growth_rate_0,
                "kernel_size": self.kernel_size,
                "config": self.config,
            }
        )
        return config
