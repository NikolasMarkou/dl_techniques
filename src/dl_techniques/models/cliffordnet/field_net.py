"""
CliffordFieldNet — CliffordNet augmented with gauge field theory layers.

Integrates six field-theoretic improvements from the ``fields/`` subpackage
into the CliffordNet vision backbone:

1. **Curvature-aware normalization** — ``FieldNormalization`` replaces plain
   ``LayerNorm``; a learned curvature estimator modulates normalisation
   strength per position.
2. **Connection-guided context** — ``ConnectionLayer`` computes a gauge
   connection from (detail, curvature); context features are transported
   through it before the sparse geometric product.
3. **Holonomy global context** — ``HolonomyLayer`` extracts gauge-invariant
   Wilson-loop features as global context, replacing (or augmenting) the
   simple GAP branch.
4. **Parallel-transport residual** — ``ParallelTransportLayer`` maps the
   skip connection into the correct tangent frame before addition.
5. **Manifold stress (anomaly detection)** — ``ManifoldStressLayer`` at the
   model head produces per-spatial-location stress/anomaly tensors.
6. **Gauge-invariant attention (optional)** — ``GaugeInvariantAttention``
   adds long-range geometric attention alongside the local geometric
   product. Off by default to preserve the attention-free philosophy.

Architecture::

    Patch Embedding → L × CliffordFieldBlock → GAP → LayerNorm → Dense
                                                  ↘ (ManifoldStress)

Pre-defined variants:
    - ``CliffordFieldNet.nano``  — channels=128, depth=12, shifts=[1,2]
    - ``CliffordFieldNet.lite``  — channels=128, depth=12, shifts=[1,2,4,8,16]
    - ``CliffordFieldNet.base``  — channels=192, depth=16, shifts=[1,2,4,8,16]
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import keras
from keras import initializers, regularizers

from dl_techniques.layers.geometric.clifford_block import (
    CliMode,
    CtxMode,
    SparseRollingGeometricProduct,
    GatedGeometricResidual,
)
from dl_techniques.layers.geometric.fields.holonomic_transformer import (
    FieldNormalization,
)
from dl_techniques.layers.geometric.fields.connection_layer import ConnectionLayer
from dl_techniques.layers.geometric.fields.parallel_transport import (
    ParallelTransportLayer,
)
from dl_techniques.layers.geometric.fields.holonomy_layer import HolonomyLayer
from dl_techniques.layers.geometric.fields.gauge_invariant_attention import (
    GaugeInvariantAttention,
)
from dl_techniques.layers.geometric.fields.manifold_stress import ManifoldStressLayer
from dl_techniques.utils.logger import logger

# Match the reference: trunc_normal_(std=0.02) for all Conv2d and Linear.
_DEFAULT_KERNEL_INIT = initializers.TruncatedNormal(stddev=0.02)

# Global-branch constants matching the original implementation
_GLOBAL_SHIFTS: List[int] = [1, 2]
_GLOBAL_CLI_MODE: CliMode = "full"


# ---------------------------------------------------------------------------
# Helper: stochastic-depth rate schedule
# ---------------------------------------------------------------------------


def _linear_drop_path_rates(num_blocks: int, max_rate: float) -> List[float]:
    """Return linearly spaced drop-path rates from 0 to *max_rate*.

    :param num_blocks: Total number of blocks.
    :param max_rate: Maximum (last-block) drop probability.
    :return: List of per-block drop-path rates.
    """
    if num_blocks <= 1:
        return [0.0] * num_blocks
    step = max_rate / (num_blocks - 1)
    return [round(i * step, 6) for i in range(num_blocks)]


# ===========================================================================
# CliffordFieldBlock
# ===========================================================================


@keras.saving.register_keras_serializable()
class CliffordFieldBlock(keras.layers.Layer):
    """CliffordNet block augmented with gauge field theory layers.

    Extends :class:`CliffordNetBlock` with six field-theoretic improvements:

    1. ``FieldNormalization`` with a learned curvature estimator replaces
       plain ``LayerNormalization``.
    2. A ``ConnectionLayer`` computes a gauge connection from (detail,
       curvature); the context stream is transported through it.
    3. ``HolonomyLayer`` produces gauge-invariant global context features,
       replacing the simple GAP branch.
    4. ``ParallelTransportLayer`` transports the skip connection into the
       correct tangent frame.
    5. ``ManifoldStressLayer`` (optional) produces per-position stress.
    6. ``GaugeInvariantAttention`` (optional) adds long-range geometric
       attention parallel to the local geometric product.

    :param channels: Feature dimensionality D (constant throughout).
    :param shifts: Channel-shift offsets for the sparse rolling product.
    :param cli_mode: ``"inner"`` | ``"wedge"`` | ``"full"`` (default).
    :param ctx_mode: ``"diff"`` (default) | ``"abs"``.
    :param use_holonomy_context: Use holonomy global context. Defaults
        to ``True``.
    :param use_parallel_transport_residual: Transport the skip connection
        via ``ParallelTransportLayer``. Defaults to ``True``.
    :param use_gauge_attention: Add gauge-invariant attention module.
        Defaults to ``False``.
    :param num_attention_heads: Number of heads for gauge-invariant
        attention. Defaults to 4.
    :param connection_type: Connection type for ``ConnectionLayer``.
        Defaults to ``"yang_mills"``.
    :param num_generators: Lie algebra generators for Yang-Mills
        connection. Defaults to 4.
    :param holonomy_loop_sizes: Loop sizes for ``HolonomyLayer``.
        Defaults to ``[2, 4, 8]``.
    :param layer_scale_init: Initial LayerScale value. Defaults to 1e-5.
    :param drop_path_rate: DropPath probability. Defaults to 0.0.
    :param use_bias: Whether Dense layers use bias. Defaults to ``True``.
    :param kernel_initializer: Kernel initializer for Dense layers.
    :param bias_initializer: Bias initializer for Dense layers.
    :param kernel_regularizer: Kernel regularizer for Dense layers.
    :param bias_regularizer: Bias regularizer for Dense layers.
    """

    def __init__(
        self,
        channels: int,
        shifts: List[int],
        cli_mode: CliMode = "full",
        ctx_mode: CtxMode = "diff",
        use_holonomy_context: bool = True,
        use_parallel_transport_residual: bool = True,
        use_gauge_attention: bool = False,
        num_attention_heads: int = 4,
        connection_type: str = "yang_mills",
        num_generators: int = 4,
        holonomy_loop_sizes: Optional[List[int]] = None,
        layer_scale_init: float = 1e-5,
        drop_path_rate: float = 0.0,
        use_bias: bool = True,
        kernel_initializer: Any = "glorot_uniform",
        bias_initializer: Any = "zeros",
        kernel_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if ctx_mode not in ("diff", "abs"):
            raise ValueError(f"ctx_mode must be 'diff' or 'abs', got {ctx_mode!r}")

        # Store configuration
        self.channels = channels
        self.shifts = list(shifts)
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.use_holonomy_context = use_holonomy_context
        self.use_parallel_transport_residual = use_parallel_transport_residual
        self.use_gauge_attention = use_gauge_attention
        self.num_attention_heads = num_attention_heads
        self.connection_type = connection_type
        self.num_generators = num_generators
        self.holonomy_loop_sizes = holonomy_loop_sizes or [2, 4, 8]
        self.layer_scale_init = layer_scale_init
        self.drop_path_rate = drop_path_rate
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        _dense_kwargs: Dict[str, Any] = dict(
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )

        # ---- Improvement 1: Curvature-aware normalization ----
        self.input_norm = FieldNormalization(
            epsilon=1e-6,
            use_curvature_scaling=True,
            name="field_norm",
        )

        # Learned curvature estimator: Dense(D -> D, tanh) * 0.1
        self.curvature_dense = keras.layers.Dense(
            channels,
            activation="tanh",
            name="curvature_dense",
            **_dense_kwargs,
        )

        # ---- Detail stream (1x1 pointwise) ----
        self.linear_det = keras.layers.Dense(
            channels, name="linear_det", **_dense_kwargs
        )

        # ---- Context stream (two stacked 3x3 DWConvs) ----
        self.dw_conv = keras.layers.DepthwiseConv2D(
            kernel_size=3,
            padding="same",
            use_bias=False,
            name="dw_conv",
        )
        self.dw_conv2 = keras.layers.DepthwiseConv2D(
            kernel_size=3,
            padding="same",
            use_bias=False,
            name="dw_conv2",
        )
        self.ctx_bn = keras.layers.BatchNormalization(name="ctx_bn")

        # ---- Improvement 2: Connection-guided context ----
        self.connection_layer = ConnectionLayer(
            hidden_dim=channels,
            connection_dim=channels,
            connection_type=connection_type,
            num_generators=num_generators,
            use_metric=False,
            antisymmetric=True,
            connection_regularization=0.001,
            name="connection",
        )

        # ---- Local sparse rolling geometric product ----
        self.local_geo_prod = SparseRollingGeometricProduct(
            channels=channels,
            shifts=shifts,
            cli_mode=cli_mode,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="local_geo_prod",
        )

        # ---- Improvement 3: Holonomy global context ----
        if use_holonomy_context:
            self.holonomy_layer = HolonomyLayer(
                hidden_dim=channels,
                loop_sizes=self.holonomy_loop_sizes,
                loop_type="rectangular",
                num_loops=4,
                use_trace=True,
                holonomy_regularization=0.001,
                name="holonomy",
            )
            self.holonomy_proj = keras.layers.Dense(
                channels, name="holonomy_proj", **_dense_kwargs
            )
        else:
            self.holonomy_layer = None
            self.holonomy_proj = None

        # ---- Improvement 4: Parallel-transport residual ----
        if use_parallel_transport_residual:
            self.transport_layer = ParallelTransportLayer(
                transport_dim=channels,
                num_steps=3,
                transport_method="direct",
                step_size=0.1,
                name="residual_transport",
            )
        else:
            self.transport_layer = None

        # ---- Improvement 6: Gauge-invariant attention (optional) ----
        if use_gauge_attention:
            self.gauge_attention = GaugeInvariantAttention(
                hidden_dim=channels,
                num_heads=num_attention_heads,
                attention_metric="hybrid",
                use_curvature_gating=True,
                use_parallel_transport=True,
                dropout_rate=0.0,
                name="gauge_attention",
            )
            self.attention_gate = keras.layers.Dense(
                channels,
                activation="sigmoid",
                name="attention_gate",
                **_dense_kwargs,
            )
        else:
            self.gauge_attention = None
            self.attention_gate = None

        # ---- GGR ----
        self.ggr = GatedGeometricResidual(
            channels=channels,
            layer_scale_init=layer_scale_init,
            drop_path_rate=drop_path_rate,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="ggr",
        )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, input_shape: Tuple) -> None:
        """Build all sub-layers in dependency order.

        :param input_shape: ``(B, H, W, D)``
        """
        spatial_shape = input_shape  # (B, H, W, D)
        D = spatial_shape[-1]

        # Curvature estimator
        self.curvature_dense.build(spatial_shape)

        # FieldNormalization — build with list [emb_shape, curv_shape]
        curv_shape = spatial_shape  # same shape: (B, H, W, D)
        self.input_norm.build([spatial_shape, curv_shape])

        # Detail stream
        self.linear_det.build(spatial_shape)
        stream_shape = self.linear_det.compute_output_shape(spatial_shape)

        # Context stream
        self.dw_conv.build(spatial_shape)
        dw1_out = self.dw_conv.compute_output_shape(spatial_shape)
        self.dw_conv2.build(dw1_out)
        dw2_out = self.dw_conv2.compute_output_shape(dw1_out)
        self.ctx_bn.build(dw2_out)

        # Connection — expects [embeddings(B,S,D), curvature(B,S,D)]
        # We'll flatten spatial dims: (B, H*W, D)
        B = spatial_shape[0]
        seq_shape = (B, None, D)
        self.connection_layer.build([seq_shape, seq_shape])

        # Local geometric product
        self.local_geo_prod.build(stream_shape)

        # Holonomy
        if self.holonomy_layer is not None:
            conn_shape = (B, None, D, D)
            self.holonomy_layer.build([seq_shape, conn_shape])
            hol_out = self.holonomy_layer.compute_output_shape(
                [seq_shape, conn_shape]
            )
            self.holonomy_proj.build(hol_out)

        # Parallel transport
        if self.transport_layer is not None:
            conn_shape = (B, None, D, D)
            self.transport_layer.build([seq_shape, conn_shape])

        # Gauge-invariant attention
        if self.gauge_attention is not None:
            conn_shape = (B, None, D, D)
            self.gauge_attention.build([seq_shape, seq_shape, conn_shape])
            self.attention_gate.build(spatial_shape)

        # GGR
        self.ggr.build(stream_shape)

        super().build(input_shape)

    # ------------------------------------------------------------------
    # Helpers: 4D <-> 3D reshaping
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_spatial(
        x: keras.KerasTensor,
    ) -> Tuple[keras.KerasTensor, int, int]:
        """Reshape ``(B, H, W, D)`` → ``(B, H*W, D)``; return H, W."""
        shape = keras.ops.shape(x)
        B, H, W, D = shape[0], shape[1], shape[2], shape[3]
        return keras.ops.reshape(x, (B, H * W, D)), H, W

    @staticmethod
    def _unflatten_spatial(
        x: keras.KerasTensor, H: int, W: int
    ) -> keras.KerasTensor:
        """Reshape ``(B, H*W, D)`` → ``(B, H, W, D)``."""
        shape = keras.ops.shape(x)
        B, D = shape[0], shape[2]
        return keras.ops.reshape(x, (B, H, W, D))

    # ------------------------------------------------------------------
    # Call
    # ------------------------------------------------------------------

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass.

        :param inputs: Feature tensor ``(B, H, W, D)``.
        :param training: Whether in training mode.
        :return: Updated feature tensor ``(B, H, W, D)``.
        """
        x_prev = inputs  # (B, H, W, D)

        # ---- Curvature estimation ----
        curvature = self.curvature_dense(x_prev) * 0.1  # (B, H, W, D)

        # ---- Improvement 1: Curvature-aware normalization ----
        # FieldNormalization expects 3D (B, S, D); flatten spatial dims
        x_seq_norm, H_n, W_n = self._flatten_spatial(x_prev)
        curv_seq_norm, _, _ = self._flatten_spatial(curvature)
        x_norm = self.input_norm([x_seq_norm, curv_seq_norm])  # (B, S, D)
        x_norm = self._unflatten_spatial(x_norm, H_n, W_n)  # (B, H, W, D)

        # ---- Dual-stream generation ----
        z_det = self.linear_det(x_norm)  # (B, H, W, D)

        z_ctx = self.dw_conv(x_norm)
        z_ctx = self.dw_conv2(z_ctx)
        z_ctx = keras.activations.silu(
            self.ctx_bn(z_ctx, training=training)
        )  # (B, H, W, D)

        if self.ctx_mode == "diff":
            z_ctx = z_ctx - z_det  # discrete Laplacian

        # ---- Improvement 2: Connection-guided context ----
        # Flatten to 3D for fields layers
        z_det_seq, H, W = self._flatten_spatial(z_det)  # (B, S, D)
        curv_seq, _, _ = self._flatten_spatial(curvature)  # (B, S, D)

        connection = self.connection_layer(
            [z_det_seq, curv_seq], training=training
        )  # (B, S, D, D)

        # Transport context through the connection
        z_ctx_seq, _, _ = self._flatten_spatial(z_ctx)  # (B, S, D)
        z_ctx_transported = keras.ops.einsum(
            "bsij,bsj->bsi", connection, z_ctx_seq
        )  # (B, S, D)
        # Blend transported context with original (stability)
        z_ctx_transported = z_ctx_seq + 0.1 * z_ctx_transported
        z_ctx = self._unflatten_spatial(z_ctx_transported, H, W)  # (B, H, W, D)

        # ---- Local sparse geometric product ----
        g_feat = self.local_geo_prod(z_det, z_ctx)  # (B, H, W, D)

        # ---- Improvement 3: Holonomy global context ----
        if self.holonomy_layer is not None:
            x_norm_seq, _, _ = self._flatten_spatial(x_norm)  # (B, S, D)
            holonomy_feat = self.holonomy_layer(
                [x_norm_seq, connection]
            )  # (B, S, hidden_dim)
            holonomy_feat = self.holonomy_proj(holonomy_feat)  # (B, S, D)
            holonomy_feat_4d = self._unflatten_spatial(
                holonomy_feat, H, W
            )  # (B, H, W, D)
            g_feat = g_feat + 0.1 * holonomy_feat_4d

        # ---- Improvement 6: Gauge-invariant attention (optional) ----
        if self.gauge_attention is not None:
            x_norm_seq, _, _ = self._flatten_spatial(x_norm)  # (B, S, D)
            attn_out = self.gauge_attention(
                [x_norm_seq, curv_seq, connection]
            )  # (B, S, D)
            attn_out_4d = self._unflatten_spatial(attn_out, H, W)  # (B, H, W, D)
            # Gated addition — sigmoid gate controls attention contribution
            gate = self.attention_gate(x_norm)  # (B, H, W, D)
            g_feat = g_feat + gate * attn_out_4d

        # ---- GGR update ----
        h_mix = self.ggr(x_norm, g_feat, training=training)  # (B, H, W, D)

        # ---- Improvement 4: Parallel-transport residual ----
        if self.transport_layer is not None:
            x_prev_seq, _, _ = self._flatten_spatial(x_prev)  # (B, S, D)
            x_transported = self.transport_layer(
                [x_prev_seq, connection]
            )  # (B, S, D)
            x_prev = self._unflatten_spatial(x_transported, H, W)  # (B, H, W, D)

        return x_prev + h_mix

    # ------------------------------------------------------------------
    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape (same as input).

        :param input_shape: ``(B, H, W, D)``
        :return: Same as input shape.
        """
        return input_shape

    # ------------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        """Return serialisable configuration."""
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "shifts": self.shifts,
                "cli_mode": self.cli_mode,
                "ctx_mode": self.ctx_mode,
                "use_holonomy_context": self.use_holonomy_context,
                "use_parallel_transport_residual": self.use_parallel_transport_residual,
                "use_gauge_attention": self.use_gauge_attention,
                "num_attention_heads": self.num_attention_heads,
                "connection_type": self.connection_type,
                "num_generators": self.num_generators,
                "holonomy_loop_sizes": self.holonomy_loop_sizes,
                "layer_scale_init": self.layer_scale_init,
                "drop_path_rate": self.drop_path_rate,
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": regularizers.serialize(
                    self.bias_regularizer
                ),
            }
        )
        return config


# ===========================================================================
# CliffordFieldNet
# ===========================================================================


@keras.saving.register_keras_serializable()
class CliffordFieldNet(keras.Model):
    """CliffordNet vision backbone augmented with gauge field theory.

    Follows the same columnar design as :class:`CliffordNet` — patch
    embedding to a fixed ``channels`` feature map, then *L* identical
    :class:`CliffordFieldBlock` layers, global average pooling, layer
    normalisation, and a Dense classifier — but each block integrates
    six field-theoretic improvements (curvature-aware norm, connection-
    guided context, holonomy global context, parallel-transport residual,
    optional gauge-invariant attention, and optional manifold stress at
    the model head).

    :param num_classes: Number of output classes.
    :param channels: Feature dimensionality D (constant throughout).
    :param depth: Number of CliffordFieldBlock layers.
    :param patch_size: Stride of the patch-embedding convolution.
    :param shifts: Channel-shift offsets for the sparse rolling product.
    :param cli_mode: ``"inner"`` | ``"wedge"`` | ``"full"`` (default).
    :param ctx_mode: ``"diff"`` (default) | ``"abs"``.
    :param use_holonomy_context: Use holonomy global context in blocks.
    :param use_parallel_transport_residual: Transport skip connections.
    :param use_gauge_attention: Add gauge-invariant attention in blocks.
    :param num_attention_heads: Heads for gauge-invariant attention.
    :param connection_type: Connection type for ``ConnectionLayer``.
    :param num_generators: Lie algebra generators count.
    :param holonomy_loop_sizes: Loop sizes for ``HolonomyLayer``.
    :param use_anomaly_detection: Attach ``ManifoldStressLayer`` at the
        model head. When enabled, ``call()`` returns a dict with keys
        ``"logits"`` and ``"stress"``.
    :param stress_types: Stress types for ``ManifoldStressLayer``.
    :param layer_scale_init: Initial LayerScale value.
    :param stochastic_depth_rate: Maximum DropPath rate (linear schedule).
    :param dropout_rate: Pre-classifier head dropout.
    :param use_bias: Whether Dense / projection layers use bias.
    :param kernel_initializer: Kernel initializer.
    :param bias_initializer: Bias initializer.
    :param kernel_regularizer: Kernel regularizer.
    :param bias_regularizer: Bias regularizer.

    **Call arguments:**

    :param inputs: Image tensor ``(B, H, W, C_in)``.
    :param training: Python bool or ``None``.

    :returns: Logit tensor ``(B, num_classes)`` when
        ``use_anomaly_detection=False``, or dict
        ``{"logits": (B, num_classes), "stress": (B, 1),
        "anomaly_mask": (B, 1)}`` when enabled.
    """

    LAYERNORM_EPSILON: float = 1e-6

    def __init__(
        self,
        num_classes: int,
        channels: int = 128,
        depth: int = 12,
        patch_size: int = 2,
        shifts: Optional[List[int]] = None,
        cli_mode: CliMode = "full",
        ctx_mode: CtxMode = "diff",
        use_holonomy_context: bool = True,
        use_parallel_transport_residual: bool = True,
        use_gauge_attention: bool = False,
        num_attention_heads: int = 4,
        connection_type: str = "yang_mills",
        num_generators: int = 4,
        holonomy_loop_sizes: Optional[List[int]] = None,
        use_anomaly_detection: bool = False,
        stress_types: Optional[List[str]] = None,
        layer_scale_init: float = 1e-5,
        stochastic_depth_rate: float = 0.0,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        kernel_initializer: Any = "glorot_uniform",
        bias_initializer: Any = "zeros",
        kernel_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")

        # Store configuration
        self.num_classes = num_classes
        self.channels = channels
        self.depth = depth
        self.patch_size = patch_size
        self.shifts = shifts if shifts is not None else [1, 2]
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.use_holonomy_context = use_holonomy_context
        self.use_parallel_transport_residual = use_parallel_transport_residual
        self.use_gauge_attention = use_gauge_attention
        self.num_attention_heads = num_attention_heads
        self.connection_type = connection_type
        self.num_generators = num_generators
        self.holonomy_loop_sizes = holonomy_loop_sizes or [2, 4, 8]
        self.use_anomaly_detection = use_anomaly_detection
        self.stress_types = stress_types or ["curvature", "connection", "combined"]
        self.layer_scale_init = layer_scale_init
        self.stochastic_depth_rate = stochastic_depth_rate
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # Build sub-component groups
        self._build_stem()
        self._build_blocks()
        self._build_head()

        logger.info(
            f"Created CliffordFieldNet (channels={channels}, depth={depth}, "
            f"patch_size={patch_size}, shifts={self.shifts}, "
            f"cli_mode={cli_mode}, ctx_mode={ctx_mode}, "
            f"holonomy={use_holonomy_context}, transport={use_parallel_transport_residual}, "
            f"gauge_attn={use_gauge_attention}, anomaly={use_anomaly_detection})"
        )

    # ------------------------------------------------------------------
    # Private builder helpers
    # ------------------------------------------------------------------

    def _build_stem(self) -> None:
        """Build patch-embedding (GeometricStem) layers."""
        _conv_kw: Dict[str, Any] = dict(
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
        )

        if self.patch_size == 1:
            self.stem_conv1 = keras.layers.Conv2D(
                filters=self.channels // 2,
                kernel_size=3,
                strides=1,
                padding="same",
                use_bias=False,
                name="stem_conv1",
                **_conv_kw,
            )
            self.stem_bn1 = keras.layers.BatchNormalization(name="stem_bn1")
            self.stem_conv2 = keras.layers.Conv2D(
                filters=self.channels,
                kernel_size=3,
                strides=1,
                padding="same",
                use_bias=False,
                name="stem_conv2",
                **_conv_kw,
            )
        elif self.patch_size == 2:
            self.stem_conv = keras.layers.Conv2D(
                filters=self.channels,
                kernel_size=3,
                strides=2,
                padding="same",
                use_bias=self.use_bias,
                name="stem_conv",
                **_conv_kw,
            )
        elif self.patch_size == 4:
            self.stem_conv1 = keras.layers.Conv2D(
                filters=self.channels // 2,
                kernel_size=3,
                strides=2,
                padding="same",
                use_bias=False,
                name="stem_conv1",
                **_conv_kw,
            )
            self.stem_bn1 = keras.layers.BatchNormalization(name="stem_bn1")
            self.stem_conv2 = keras.layers.Conv2D(
                filters=self.channels,
                kernel_size=3,
                strides=2,
                padding="same",
                use_bias=False,
                name="stem_conv2",
                **_conv_kw,
            )
        else:
            self.stem_conv = keras.layers.Conv2D(
                filters=self.channels,
                kernel_size=self.patch_size,
                strides=self.patch_size,
                padding="same",
                use_bias=self.use_bias,
                name="stem_conv",
                **_conv_kw,
            )

        self.stem_norm = keras.layers.BatchNormalization(name="stem_norm")

    def _build_blocks(self) -> None:
        """Build the CliffordFieldBlock list with linear drop-path schedule."""
        drop_rates = _linear_drop_path_rates(
            self.depth, self.stochastic_depth_rate
        )

        _block_kw: Dict[str, Any] = dict(
            channels=self.channels,
            shifts=self.shifts,
            cli_mode=self.cli_mode,
            ctx_mode=self.ctx_mode,
            use_holonomy_context=self.use_holonomy_context,
            use_parallel_transport_residual=self.use_parallel_transport_residual,
            use_gauge_attention=self.use_gauge_attention,
            num_attention_heads=self.num_attention_heads,
            connection_type=self.connection_type,
            num_generators=self.num_generators,
            holonomy_loop_sizes=self.holonomy_loop_sizes,
            layer_scale_init=self.layer_scale_init,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
        )

        self.blocks_list: List[CliffordFieldBlock] = []
        for i in range(self.depth):
            block = CliffordFieldBlock(
                drop_path_rate=drop_rates[i],
                name=f"clifford_field_block_{i}",
                **_block_kw,
            )
            self.blocks_list.append(block)

    def _build_head(self) -> None:
        """Build classifier head and optional anomaly detection."""
        self.global_pool = keras.layers.GlobalAveragePooling2D(
            name="global_pool"
        )
        self.head_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON, name="head_norm"
        )
        self.head_dropout = (
            keras.layers.Dropout(self.dropout_rate, name="head_dropout")
            if self.dropout_rate > 0.0
            else None
        )
        self.classifier = keras.layers.Dense(
            self.num_classes,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="classifier",
        )

        # ---- Improvement 5: Manifold stress anomaly detection ----
        if self.use_anomaly_detection:
            self.stress_curvature_dense = keras.layers.Dense(
                self.channels,
                activation="tanh",
                name="stress_curvature_dense",
            )
            self.stress_connection = ConnectionLayer(
                hidden_dim=self.channels,
                connection_dim=self.channels,
                connection_type=self.connection_type,
                num_generators=self.num_generators,
                use_metric=False,
                antisymmetric=True,
                connection_regularization=0.001,
                name="stress_connection",
            )
            self.stress_layer = ManifoldStressLayer(
                hidden_dim=self.channels,
                stress_types=self.stress_types,
                stress_threshold=0.5,
                use_learnable_baseline=True,
                name="stress",
            )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the model via a symbolic forward pass.

        :param input_shape: Input tensor shape ``(B, H, W, C_in)``.
        """
        super().build(input_shape)
        if len(input_shape) == 3:
            build_shape = (None,) + tuple(input_shape)
        else:
            build_shape = tuple(input_shape)
        dummy = keras.KerasTensor(build_shape)
        _ = self.call(dummy)

    # ------------------------------------------------------------------
    # Forward pass helpers
    # ------------------------------------------------------------------

    def _apply_stem(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool],
    ) -> keras.KerasTensor:
        """Apply the patch embedding stem.

        :param inputs: Raw image batch ``(B, H, W, C_in)``.
        :param training: Whether in training mode.
        :return: Embedded feature map ``(B, h, w, channels)``.
        """
        if self.patch_size in (1, 4):
            x = keras.activations.silu(
                self.stem_bn1(self.stem_conv1(inputs), training=training)
            )
            x = self.stem_conv2(x)
        else:
            x = self.stem_conv(inputs)

        return self.stem_norm(x, training=training)

    # ------------------------------------------------------------------
    # Call
    # ------------------------------------------------------------------

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]:
        """Forward pass.

        :param inputs: Image batch ``(B, H, W, C_in)``.
        :param training: Whether in training mode.
        :returns: Logit tensor ``(B, num_classes)`` or dict with
            ``"logits"``, ``"stress"``, ``"anomaly_mask"`` keys.
        """
        x = self._apply_stem(inputs, training=training)

        for block in self.blocks_list:
            x = block(x, training=training)

        # ---- Improvement 5: Manifold stress (before pooling) ----
        stress_out = None
        anomaly_out = None
        if self.use_anomaly_detection:
            shape = keras.ops.shape(x)
            B, H, W, D = shape[0], shape[1], shape[2], shape[3]
            x_seq = keras.ops.reshape(x, (B, H * W, D))

            curv = self.stress_curvature_dense(x_seq) * 0.1
            conn = self.stress_connection([x_seq, curv], training=training)
            stress, anomaly_mask = self.stress_layer(
                [x_seq, curv, conn], training=training
            )
            # Pool stress to per-image scalar
            stress_out = keras.ops.mean(stress, axis=1)  # (B, 1)
            anomaly_out = keras.ops.max(anomaly_mask, axis=1)  # (B, 1)

        # Head: GAP -> LayerNorm -> (Dropout) -> Dense
        x = self.global_pool(x)  # (B, channels)
        x = self.head_norm(x)

        if self.head_dropout is not None:
            x = self.head_dropout(x, training=training)

        logits = self.classifier(x)  # (B, num_classes)

        if self.use_anomaly_detection:
            return {
                "logits": logits,
                "stress": stress_out,
                "anomaly_mask": anomaly_out,
            }
        return logits

    # ------------------------------------------------------------------
    # Shape inference
    # ------------------------------------------------------------------

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Union[Tuple[Optional[int], ...], Dict[str, Tuple]]:
        """Compute output shape.

        :param input_shape: ``(B, H, W, C_in)``
        :return: ``(B, num_classes)`` or dict of shapes.
        """
        if self.use_anomaly_detection:
            return {
                "logits": (input_shape[0], self.num_classes),
                "stress": (input_shape[0], 1),
                "anomaly_mask": (input_shape[0], 1),
            }
        return (input_shape[0], self.num_classes)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return serialisable configuration."""
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "channels": self.channels,
                "depth": self.depth,
                "patch_size": self.patch_size,
                "shifts": self.shifts,
                "cli_mode": self.cli_mode,
                "ctx_mode": self.ctx_mode,
                "use_holonomy_context": self.use_holonomy_context,
                "use_parallel_transport_residual": self.use_parallel_transport_residual,
                "use_gauge_attention": self.use_gauge_attention,
                "num_attention_heads": self.num_attention_heads,
                "connection_type": self.connection_type,
                "num_generators": self.num_generators,
                "holonomy_loop_sizes": self.holonomy_loop_sizes,
                "use_anomaly_detection": self.use_anomaly_detection,
                "stress_types": self.stress_types,
                "layer_scale_init": self.layer_scale_init,
                "stochastic_depth_rate": self.stochastic_depth_rate,
                "dropout_rate": self.dropout_rate,
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": regularizers.serialize(
                    self.bias_regularizer
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CliffordFieldNet":
        """Reconstruct model from configuration dict.

        :param config: Dictionary produced by :meth:`get_config`.
        :return: New :class:`CliffordFieldNet` instance.
        """
        for key in ("kernel_regularizer", "bias_regularizer"):
            if config.get(key) and isinstance(config[key], dict):
                config[key] = regularizers.deserialize(config[key])
        return cls(**config)

    # ------------------------------------------------------------------
    # Factory class methods
    # ------------------------------------------------------------------

    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "nano": dict(
            channels=128,
            depth=12,
            patch_size=2,
            shifts=[1, 2],
            cli_mode="full",
            ctx_mode="diff",
            use_holonomy_context=True,
            use_parallel_transport_residual=True,
            use_gauge_attention=False,
            num_attention_heads=4,
            connection_type="yang_mills",
            num_generators=4,
            holonomy_loop_sizes=[2, 4, 8],
            layer_scale_init=1e-5,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
        "lite": dict(
            channels=128,
            depth=12,
            patch_size=2,
            shifts=[1, 2, 4, 8, 16],
            cli_mode="full",
            ctx_mode="diff",
            use_holonomy_context=True,
            use_parallel_transport_residual=True,
            use_gauge_attention=False,
            num_attention_heads=4,
            connection_type="yang_mills",
            num_generators=4,
            holonomy_loop_sizes=[2, 4, 8],
            layer_scale_init=1e-5,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
        "base": dict(
            channels=192,
            depth=16,
            patch_size=2,
            shifts=[1, 2, 4, 8, 16],
            cli_mode="full",
            ctx_mode="diff",
            use_holonomy_context=True,
            use_parallel_transport_residual=True,
            use_gauge_attention=True,
            num_attention_heads=8,
            connection_type="yang_mills",
            num_generators=8,
            holonomy_loop_sizes=[2, 4, 8],
            layer_scale_init=1e-5,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
    }

    @classmethod
    def from_variant(
        cls,
        variant: str,
        num_classes: int,
        **kwargs: Any,
    ) -> "CliffordFieldNet":
        """Create a :class:`CliffordFieldNet` from a predefined variant.

        :param variant: One of ``"nano"``, ``"lite"``, ``"base"``.
        :param num_classes: Number of output classes.
        :param kwargs: Override any default hyperparameter.
        :return: Configured :class:`CliffordFieldNet` instance.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Available: {list(cls.MODEL_VARIANTS.keys())}"
            )

        defaults = dict(cls.MODEL_VARIANTS[variant])
        defaults.update(kwargs)

        logger.info(f"Creating CliffordFieldNet-{variant.upper()}")
        return cls(num_classes=num_classes, **defaults)

    @classmethod
    def nano(cls, num_classes: int, **kwargs: Any) -> "CliffordFieldNet":
        """CliffordFieldNet-Nano: channels=128, depth=12, shifts=[1,2].

        All field improvements enabled except gauge attention.

        :param num_classes: Number of output classes.
        :param kwargs: Override any default hyperparameter.
        :return: Configured :class:`CliffordFieldNet` instance.
        """
        return cls.from_variant("nano", num_classes=num_classes, **kwargs)

    @classmethod
    def lite(cls, num_classes: int, **kwargs: Any) -> "CliffordFieldNet":
        """CliffordFieldNet-Lite: channels=128, depth=12, shifts=[1,2,4,8,16].

        All field improvements enabled except gauge attention.

        :param num_classes: Number of output classes.
        :param kwargs: Override any default hyperparameter.
        :return: Configured :class:`CliffordFieldNet` instance.
        """
        return cls.from_variant("lite", num_classes=num_classes, **kwargs)

    @classmethod
    def base(cls, num_classes: int, **kwargs: Any) -> "CliffordFieldNet":
        """CliffordFieldNet-Base: channels=192, depth=16, shifts=[1,2,4,8,16].

        All field improvements enabled including gauge-invariant attention.

        :param num_classes: Number of output classes.
        :param kwargs: Override any default hyperparameter.
        :return: Configured :class:`CliffordFieldNet` instance.
        """
        return cls.from_variant("base", num_classes=num_classes, **kwargs)
