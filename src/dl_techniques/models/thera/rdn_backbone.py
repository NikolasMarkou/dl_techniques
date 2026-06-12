"""THERA's RDN (Residual Dense Network) feature backbone as a Keras layer.

This ports the **feature extractor** of the Residual Dense Network used by THERA
(Aliasing-Free Arbitrary-Scale Super-Resolution with Neural Heat Fields). It is
the RDN body *without* the upsampling tail: it maps an input image
``(B, H, W, n_colors)`` to a dense feature map ``(B, H, W, growth_rate_0)`` that
downstream THERA components (tails / hypernetwork) consume. There is **no**
spatial resolution change here.

The reference JAX/Flax implementation (THERA ``model/rdn.py``, instantiated as
``RDN()`` with defaults ``G0=64, RDNconfig='B'``) is::

    class RDB_Conv(nn.Module):           # growRate, kSize=3
        def __call__(self, x):
            out = Sequential([Conv(growRate, (kSize, kSize),
                                   padding=(kSize - 1) // 2), relu])(x)
            return concatenate((x, out), -1)        # channel-growing dense concat

    class RDB(nn.Module):                # growRate0, growRate, nConvLayers
        def __call__(self, x):
            res = x
            for c in range(nConvLayers):
                x = RDB_Conv(growRate)(x)
            x = Conv(growRate0, (1, 1))(x)          # local feature fusion (1x1)
            return x + res

    class RDN(nn.Module):                # G0=64, RDNkSize=3, RDNconfig='B'
        def __call__(self, x, _=None):
            D, C, G = {'A': (20, 6, 32),
                       'B': (16, 8, 64)}[RDNconfig]
            f_1 = Conv(G0, (k, k))(x)
            x   = Conv(G0, (k, k))(f_1)
            RDBs_out = []
            for i in range(D):
                x = RDB(G0, G, C)(x); RDBs_out.append(x)
            x = concatenate(RDBs_out, -1)           # global dense feature fusion
            x = Sequential([Conv(G0, (1, 1)),
                            Conv(G0, (k, k))])(x)    # global feature fusion
            x = x + f_1
            return x                                 # features, NO upsampling

Channel bookkeeping (the load-bearing detail for explicit Keras builds)
-----------------------------------------------------------------------
The dense concat inside each RDB grows the channel count deterministically:
entering an RDB the tensor has ``G0`` channels; after the ``c``-th
:class:`RDBConv` (1-indexed) it has ``G0 + c * G`` channels. The local-fusion
1x1 conv collapses ``G0 + C * G`` channels back down to ``G0`` so the residual
``x + res`` is shape-consistent. Because this growth is closed-form (not
data-dependent), every inner ``Conv2D`` is built with an explicitly computed
input-channel shape in ``build()`` -- no eager dummy forward is required.

THERA's ``padding=(kSize - 1) // 2`` for odd ``kSize`` is exactly Keras
``padding="same"``; the 1x1 fusion convs also use ``"same"`` (a no-op pad).

Reference:
    Becker et al., "Thera: Aliasing-Free Arbitrary-Scale Super-Resolution with
    Neural Heat Fields" (original JAX/Flax ``model/rdn.py``); Zhang et al.,
    "Residual Dense Network for Image Super-Resolution" (CVPR 2018).
"""

import keras
from keras import ops
from typing import Any, Dict, List, Optional, Tuple

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------------
# config name -> (num_rdb_D, num_conv_layers_C, growth_rate_G)
# ---------------------------------------------------------------------------
_RDN_CONFIGS: Dict[str, Tuple[int, int, int]] = {
    "A": (20, 6, 32),
    "B": (16, 8, 64),
}


@keras.saving.register_keras_serializable()
class RDBConv(keras.layers.Layer):
    """Single dense-connected conv unit of a Residual Dense Block.

    Applies a ``(kernel_size, kernel_size)`` convolution (``padding="same"``)
    followed by ReLU, then **concatenates** the result onto the input along the
    channel axis. The output therefore has ``C_in + growth_rate`` channels,
    where ``C_in`` is the number of input channels.

    **Intent**: Provide the atomic dense-growth building block of an RDB: a
    conv-ReLU whose output is concatenated back onto its input so that each
    successive unit sees an ever-wider feature stack (densely-connected growth),
    while keeping channel bookkeeping closed-form for explicit, reload-safe
    ``build()`` of the inner convolution.

    **Architecture**:
    ```
    x  -->  Conv(growRate, kxk, same)  -->  relu  -->  out
    │                                                    │
    └──────────────────  concat([x, out], axis=-1)  ─────┘
                                 ↓
            (channels grow by growRate: C_in -> C_in + growRate)
    ```

    Args:
        growth_rate: Integer, number of feature maps produced by the inner
            convolution (the channel growth per unit). Must be positive.
        kernel_size: Integer, spatial size of the (square) convolution kernel.
            Defaults to ``3``.
        **kwargs: Forwarded to :class:`keras.layers.Layer`.

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
        # Build the inner conv with the propagated NHWC input shape BEFORE
        # super().build() so that a `.keras` reload restores its weights.
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


@keras.saving.register_keras_serializable()
class RDB(keras.layers.Layer):
    """Residual Dense Block: ``C`` dense conv units + local fusion + residual.

    Runs ``num_conv_layers`` :class:`RDBConv` units (each growing the channel
    count by ``growth_rate``), fuses the densely-concatenated stack back to
    ``growth_rate_0`` channels with a 1x1 convolution (local feature fusion),
    and adds the block input as a residual. Input and output both have
    ``growth_rate_0`` channels.

    **Intent**: Assemble the RDB of the Residual Dense Network -- a contiguous
    memory mechanism where ``C`` dense conv units accumulate features, a 1x1
    local-fusion conv adaptively collapses the wide stack back to ``G0``, and a
    local residual connection stabilises training and preserves the block's
    input/output channel contract.

    **Architecture**:
    ```
    x ──────────────────────────────────────────────┐ (local residual)
    │                                                 │
    └─> [ RDBConv x C ]  (dense growth: G0 -> G0+C*G) │
              ↓                                        │
        Conv(G0, 1x1, same)  (local feature fusion)    │
              ↓                                        │
              + x  <───────────────────────────────────┘
              ↓
            output (G0 channels)
    ```

    Args:
        growth_rate_0: Integer, the block's input/output channel count ``G0``
            (the local-fusion conv collapses the dense stack back to this).
        growth_rate: Integer, per-unit channel growth ``G``.
        num_conv_layers: Integer, number of :class:`RDBConv` units ``C``.
        kernel_size: Integer, conv kernel size for the dense units. Defaults to
            ``3``.
        **kwargs: Forwarded to :class:`keras.layers.Layer`.

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
        # Track channel growth through the dense concat explicitly. Entering the
        # block the tensor has G0 channels (== input_shape[-1]); after the c-th
        # RDBConv it has G0 + c*G channels. Build each sublayer with the exact
        # input shape it will see, BEFORE super().build(), for reload-safe weights.
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


@keras.saving.register_keras_serializable()
class RDNBackbone(keras.layers.Layer):
    """THERA's RDN feature backbone (no upsampling).

    Two shallow convs produce ``f_1`` and the first RDB input; ``D`` stacked
    :class:`RDB` blocks each emit a ``G0``-channel feature map; all ``D`` outputs
    are concatenated (global dense feature fusion) and fused back to ``G0`` via a
    1x1 then a kxk conv (global feature fusion); finally the shallow feature
    ``f_1`` is added back. The result is a ``(B, H, W, growth_rate_0)`` feature
    map at the **input resolution** (no upsampling tail).

    **Intent**: Provide THERA's deep RDN feature extractor as a single reusable
    Keras layer -- shallow feature extraction, ``D`` residual dense blocks with
    global hierarchical feature fusion, and a global residual to the shallow
    features -- yielding a resolution-preserving feature map for downstream
    THERA tails/hypernetwork, deliberately omitting the upsampling tail so
    arbitrary-scale components own the resampling.

    **Architecture**:
    ```
    x
    ↓
    Conv(G0, kxk, same) = f_1 ───────────────────────┐ (global residual)
    ↓                                                 │
    Conv(G0, kxk, same)                               │
    ↓                                                 │
    [ RDB x D ]  (collect each block output)          │
    ↓                                                 │
    concat(RDB_out[0..D-1], axis=-1)  (D*G0 channels) │
    ↓                                                 │
    Conv(G0, 1x1, same)  ─┐                           │
    ↓                     ├─ global feature fusion     │
    Conv(G0, kxk, same)  ─┘                           │
    ↓                                                 │
    + f_1  <──────────────────────────────────────────┘
    ↓
    features (B, H, W, G0)   # input resolution, no upsampling
    ```

    Args:
        growth_rate_0: Integer, base channel width ``G0`` (also the output
            channel count). Defaults to ``64``.
        kernel_size: Integer, kxk conv kernel size for the shallow/global convs
            and the RDB dense units. Defaults to ``3``.
        config: String in ``{"A", "B"}`` selecting the RDN depth/width preset.
            ``"A" -> (D=20, C=6, G=32)``, ``"B" -> (D=16, C=8, G=64)``. THERA's
            default is ``"B"``. Defaults to ``"B"``.
        **kwargs: Forwarded to :class:`keras.layers.Layer`.

    Input shape:
        4D tensor ``(batch, height, width, n_colors)`` (NHWC).

    Output shape:
        4D tensor ``(batch, height, width, growth_rate_0)``.

    Raises:
        ValueError: If ``config`` is not one of ``{"A", "B"}``, or if
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
