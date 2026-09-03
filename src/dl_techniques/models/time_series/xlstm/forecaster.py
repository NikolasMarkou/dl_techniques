"""Continuous-input forecaster built on the xLSTM block stack: reversible
per-instance normalization, a Dense projection in place of a token
embedding, the same mLSTM/sLSTM residual blocks as :class:`model.xLSTM`,
and a pooled head that emits either non-crossing quantiles or a point
forecast.

xLSTM's recurrence keeps a fixed-size state regardless of context length,
so a longer lookback costs linear time and constant memory rather than
quadratic attention cost, and its exponential gating lets a regime change
overwrite old memory instead of only decaying it. Scale handling reuses
the reversible instance normalization from the TiRex models: statistics
are computed per series and per feature over the time axis, NaNs are
zeroed before the statistics are taken so a gap neither skews them nor
propagates forward, and the prediction is mapped back with
`y = out * std + mean`. The time axis is pooled to a single vector before
the head, trading the ability to see where in the window a pattern
occurred for one forward pass with no compounding autoregressive error.
The quantile head enforces `Q_i = Q_{i-1} + softplus(r_i)` so quantiles
are non-decreasing by construction; the point head returns
`quantiles=None` rather than fabricate an interval it never estimated.

A caller with `num_features > 1` should note that only the quantile head's
inversion uses the last feature's normalization statistics — the point
head inverts every channel with its own per-feature statistics, since
sharing one channel's statistics across all of them would silently put
the other channels in the wrong units.

References:
    - Beck et al., 2024. xLSTM: Extended Long Short-Term Memory.
      (https://arxiv.org/abs/2405.04517)
    - Kim et al., 2022. Reversible Instance Normalization for Accurate Time-Series
      Forecasting against Distribution Shift. ICLR 2022.
    - Wen et al., 2017. A Multi-Horizon Quantile Recurrent Forecaster.
      (https://arxiv.org/abs/1711.11053)
    - Salinas et al., 2020. DeepAR: Probabilistic forecasting with autoregressive
      recurrent networks. International Journal of Forecasting 36(3).
      (https://arxiv.org/abs/1704.04110)
"""

import keras
import numpy as np
from keras import ops, initializers
from typing import Optional, Union, List, Any, Dict, Sequence, Tuple, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.models.time_series.forecast import Forecast, ForecastMixin
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.time_series.xlstm_blocks import mLSTMBlock, sLSTMBlock
from dl_techniques.layers.time_series.quantile_head_fixed_io import QuantileHead
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------

# Default quantile levels; canonical home is xLSTMForecaster.DEFAULT_QUANTILES,
# this module constant is an alias for the constructor/factory defaults.
# DECISION plan-2026-08-19T163559-499b6f0e/D-079: keep this a tuple, not a list.
# A list let a caller mutate the shared default in place. See decisions.md.
DEFAULT_QUANTILES: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.xlstm.forecaster")
class xLSTMForecaster(keras.Model, ForecastMixin):
    """xLSTM forecaster: a continuous context window in, an H-step forecast out.

    Emits ``[B, H, Q]`` in quantile mode or ``[B, H, num_features]`` in point
    mode. Mixes in :class:`ForecastMixin` for a uniform
    ``predict_forecast(x) -> Forecast`` entry point.

    Architecture:

    .. code-block:: text

        context [B, input_length, F]
           |
        reversible instance-norm      (optional)
           |
        Dense(embed_dim)
           |
        mLSTM / sLSTM block  x num_layers
           |
        final_norm
           |
        mean-pool over time            -> [B, 1, embed_dim]
           |
        head: QuantileHead -> [B, H, Q]  or  Dense -> [B, H, F]
           |
        reversible denormalization    (optional)

    The first ``int(num_layers * mlstm_ratio)`` blocks are mLSTM; the rest are sLSTM.

    :param input_length: Length of the input context window.
    :type input_length: int
    :param prediction_length: Length of the forecast horizon `H`.
    :type prediction_length: int
    :param num_features: Number of input/output features `F`. Defaults to 1.
    :type num_features: int
    :param embed_dim: Dimensionality of the latent representation. Must be divisible by `mlstm_num_heads`.
    :type embed_dim: int
    :param num_layers: Total number of xLSTM blocks.
    :type num_layers: int
    :param mlstm_ratio: Fraction of layers that are mLSTM, in [0, 1]. Defaults to 0.5.
    :type mlstm_ratio: float
    :param mlstm_num_heads: Number of heads for mLSTM blocks. Must divide `embed_dim`. Defaults to 4.
    :type mlstm_num_heads: int
    :param mlstm_expansion_factor: Expansion factor for mLSTM blocks. Defaults to 2.
    :type mlstm_expansion_factor: int
    :param slstm_forget_gate: sLSTM forget-gate activation, ``'sigmoid'`` or ``'exp'``. Defaults to ``'sigmoid'``.
    :type slstm_forget_gate: str
    :param ffn_type: FFN type for sLSTM blocks. Defaults to ``'swiglu'``.
    :type ffn_type: str
    :param ffn_expansion_factor: FFN expansion factor for sLSTM. Defaults to 2.
    :type ffn_expansion_factor: int
    :param normalization_type: Normalization layer type. Defaults to ``'layer_norm'``.
    :type normalization_type: str
    :param normalization_kwargs: Extra keyword arguments for the normalization layer.
    :type normalization_kwargs: dict, optional
    :param dropout_rate: Dropout rate. Defaults to 0.0.
    :type dropout_rate: float
    :param use_quantile_head: Use a :class:`QuantileHead` (quantile mode) instead of a Dense point head. Defaults to True.
    :type use_quantile_head: bool
    :param quantile_levels: Quantile levels to predict, quantile mode only. Defaults to `DEFAULT_QUANTILES`.
    :type quantile_levels: Sequence[float]
    :param enforce_monotonicity: Enforce non-crossing quantiles in the `QuantileHead`. Defaults to True.
    :type enforce_monotonicity: bool
    :param use_normalization: Enable reversible per-instance z-score normalization. Defaults to True.
    :type use_normalization: bool
    :param kernel_initializer: Initializer for kernel weights.
    :param recurrent_initializer: Initializer for recurrent weights.
    :param bias_initializer: Initializer for bias weights.
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :param recurrent_regularizer: Optional regularizer for recurrent weights.
    :param bias_regularizer: Optional regularizer for bias weights.
    :param name: Model name. Defaults to ``'xLSTMForecaster'``.
    :type name: str
    :param kwargs: Additional arguments for the Keras `Model` base class.

    Input shape:
        3D tensor with shape ``(batch_size, input_length, num_features)``.
        A 2D tensor ``(batch_size, input_length)`` is expanded to 3D.

    Output shape:
        Quantile mode: 3D tensor ``(batch_size, prediction_length, num_quantiles)``.
        Point mode: 3D tensor ``(batch_size, prediction_length, num_features)``.

    Example:
        ```python
        model = xLSTMForecaster(
            input_length=64,
            prediction_length=24,
            num_features=1,
            embed_dim=128,
            num_layers=4,
            mlstm_num_heads=8,
            use_quantile_head=True,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        context = keras.random.normal((8, 64, 1))
        preds = model(context)        # (8, 24, 3)
        fc = model.predict_forecast(context)  # Forecast(point, quantiles, levels)
        ```
    """

    # The module-level DEFAULT_QUANTILES is an alias of this class attribute.
    DEFAULT_QUANTILES: Tuple[float, ...] = DEFAULT_QUANTILES

    MODEL_VARIANTS = {
        "tiny": {
            "embed_dim": 64,
            "num_layers": 2,
            "mlstm_ratio": 0.5,
            "mlstm_num_heads": 4,
            "dropout_rate": 0.1,
        },
        "small": {
            "embed_dim": 128,
            "num_layers": 4,
            "mlstm_ratio": 0.5,
            "mlstm_num_heads": 8,
            "dropout_rate": 0.1,
        },
    }

    def __init__(
        self,
        input_length: int,
        prediction_length: int,
        num_features: int = 1,
        embed_dim: int = 128,
        num_layers: int = 4,
        mlstm_ratio: float = 0.5,
        mlstm_num_heads: int = 4,
        mlstm_expansion_factor: int = 2,
        slstm_forget_gate: Literal['sigmoid', 'exp'] = 'sigmoid',
        ffn_type: str = 'swiglu',
        ffn_expansion_factor: int = 2,
        normalization_type: str = 'layer_norm',
        normalization_kwargs: Optional[Dict[str, Any]] = None,
        dropout_rate: float = 0.0,
        use_quantile_head: bool = True,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
        enforce_monotonicity: bool = True,
        use_normalization: bool = True,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        recurrent_initializer: Union[str, initializers.Initializer] = 'orthogonal',
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        recurrent_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        name: str = "xLSTMForecaster",
        **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)

        if input_length <= 0:
            raise ValueError(f"input_length must be positive, got {input_length}")
        if prediction_length <= 0:
            raise ValueError(f"prediction_length must be positive, got {prediction_length}")
        if num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if not 0 <= mlstm_ratio <= 1:
            raise ValueError(f"mlstm_ratio must be in [0, 1], got {mlstm_ratio}")
        if embed_dim % mlstm_num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by mlstm_num_heads "
                f"({mlstm_num_heads})"
            )
        if use_quantile_head and len(quantile_levels) == 0:
            raise ValueError("quantile_levels must be non-empty when use_quantile_head=True")

        self.input_length = input_length
        self.prediction_length = prediction_length
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.mlstm_ratio = mlstm_ratio
        self.mlstm_num_heads = mlstm_num_heads
        self.mlstm_expansion_factor = mlstm_expansion_factor
        self.slstm_forget_gate = slstm_forget_gate
        self.ffn_type = ffn_type
        self.ffn_expansion_factor = ffn_expansion_factor
        self.normalization_type = normalization_type
        # Keeps the None sentinel for lossless round-trip; `or {}` applies only
        # at the create_normalization_layer call site below.
        self.normalization_kwargs = normalization_kwargs
        self.dropout_rate = dropout_rate
        self.use_quantile_head = use_quantile_head
        self.quantile_levels = list(quantile_levels)
        self.enforce_monotonicity = enforce_monotonicity
        self.use_normalization = use_normalization
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.bias_regularizer = bias_regularizer

        # Replaces the LM's token Embedding for continuous input.
        self.input_projection = keras.layers.Dense(
            embed_dim,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name='input_projection',
        )

        self.blocks = []
        num_mlstm = int(num_layers * mlstm_ratio)
        for i in range(num_layers):
            if i < num_mlstm:
                block = mLSTMBlock(
                    units=embed_dim,
                    expansion_factor=mlstm_expansion_factor,
                    num_heads=mlstm_num_heads,
                    normalization_type=normalization_type,
                    normalization_kwargs=normalization_kwargs,
                    kernel_initializer=kernel_initializer,
                    recurrent_initializer=recurrent_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    recurrent_regularizer=recurrent_regularizer,
                    bias_regularizer=bias_regularizer,
                    name=f'mlstm_block_{i}',
                )
            else:
                block = sLSTMBlock(
                    units=embed_dim,
                    ffn_type=ffn_type,
                    ffn_expansion_factor=ffn_expansion_factor,
                    normalization_type=normalization_type,
                    normalization_kwargs=normalization_kwargs,
                    forget_gate_activation=slstm_forget_gate,
                    dropout_rate=dropout_rate,
                    kernel_initializer=kernel_initializer,
                    recurrent_initializer=recurrent_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    recurrent_regularizer=recurrent_regularizer,
                    bias_regularizer=bias_regularizer,
                    name=f'slstm_block_{i}',
                )
            self.blocks.append(block)

        self.final_norm = create_normalization_layer(
            normalization_type=normalization_type,
            name='final_norm',
            **(self.normalization_kwargs or {})
        )

        if self.use_quantile_head:
            self.head = QuantileHead(
                num_quantiles=len(self.quantile_levels),
                output_length=self.prediction_length,
                dropout_rate=min(self.dropout_rate, 0.1),
                enforce_monotonicity=self.enforce_monotonicity,
                use_bias=True,
                flatten_input=True,
                name='quantile_head',
            )
        else:
            self.head = keras.layers.Dense(
                self.prediction_length * self.num_features,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name='point_head',
            )

        logger.info(
            f"xLSTMForecaster initialized: {num_layers} blocks "
            f"({num_mlstm} mLSTM / {num_layers - num_mlstm} sLSTM), "
            f"embed_dim={embed_dim}, input_length={input_length}, "
            f"prediction_length={prediction_length}, "
            f"head={'quantile' if use_quantile_head else 'point'}"
        )

    def build(self, input_shape) -> None:
        """Build every sublayer explicitly, before ``super().build()``, so weights restore on `.keras` load.

        The head consumes the mean-pooled `[B, 1, embed_dim]` shape, since
        `QuantileHead(flatten_input=True)` needs a statically known input length.

        :param input_shape: Shape of the context input, `(batch_size, input_length[, num_features])`.
        """
        input_shape = tuple(input_shape)
        # call() expands a 2D [B, T] context to [B, T, 1]; build for 3D.
        if len(input_shape) == 2:
            input_shape = input_shape + (1,)

        seq_len = input_shape[1]
        projected_shape = (input_shape[0], seq_len, self.embed_dim)
        pooled_shape = (input_shape[0], 1, self.embed_dim)

        self.input_projection.build(input_shape)
        for block in self.blocks:
            block.build(projected_shape)
        self.final_norm.build(projected_shape)
        self.head.build(pooled_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: Union[keras.KerasTensor, np.ndarray],
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Run the forward pass through the xLSTM forecaster.

        :param inputs: Context window of shape `[B, input_length, num_features]` (a 2D `[B, input_length]` tensor is expanded to 3D).
        :param training: Whether the call runs in training mode.
        :type training: bool, optional
        :return: Quantile mode: `[B, prediction_length, num_quantiles]`. Point mode: `[B, prediction_length, num_features]`.
        :rtype: keras.KerasTensor
        """
        if len(inputs.shape) == 2:
            inputs = ops.expand_dims(inputs, axis=-1)

        # NaNs are zeroed before normalization so they cannot propagate.
        nan_mask = ops.logical_not(ops.isnan(inputs))
        nan_mask = ops.cast(nan_mask, dtype=inputs.dtype)
        clean_inputs = ops.where(ops.isnan(inputs), ops.zeros_like(inputs), inputs)

        if self.use_normalization:
            valid_count = ops.maximum(ops.sum(nan_mask, axis=1, keepdims=True), 1e-7)
            mean = ops.sum(clean_inputs * nan_mask, axis=1, keepdims=True) / valid_count
            sq_diff = ((clean_inputs - mean) * nan_mask) ** 2
            variance = ops.sum(sq_diff, axis=1, keepdims=True) / valid_count
            std = ops.sqrt(variance)
            std = ops.maximum(std, 1e-7)
            x = (clean_inputs - mean) / std
        else:
            x = clean_inputs
            mean = None
            std = None

        x = self.input_projection(x, training=training)

        for block in self.blocks:
            x = block(x, training=training, mask=None)

        x = self.final_norm(x, training=training)

        # keepdims=True keeps a static seq_len of 1, required by
        # QuantileHead(flatten_input=True).
        pooled = ops.mean(x, axis=1, keepdims=True)

        if self.use_quantile_head:
            outputs = self.head(pooled, training=training)
        else:
            flat = self.head(pooled, training=training)
            batch_size = ops.shape(flat)[0]
            outputs = ops.reshape(
                flat,
                (batch_size, self.prediction_length, self.num_features),
            )

        if self.use_normalization:
            # DECISION plan-2026-08-14T233721-d4f9beb2/D-036: quantile and point
            # heads must invert with different statistics; do not route the point
            # head through _get_target_stats. See decisions.md.
            if self.use_quantile_head:
                norm_mean, norm_std = self._get_target_stats(mean, std)
            else:
                norm_mean, norm_std = mean, std
            outputs = (outputs * norm_std) + norm_mean

        return outputs

    @staticmethod
    def _get_target_stats(
        mean: keras.KerasTensor,
        std: keras.KerasTensor,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Extract normalization statistics for the target, last, feature.

        Used by the quantile head only, whose `(B, H, Q)` output is Q
        quantiles of a single series (the last feature for multivariate
        inputs). The point head emits one value per feature and is inverted
        with the full `(B, 1, F)` statistics directly in :meth:`call` instead.

        :param mean: Mean tensor of shape `(batch, 1, features)`.
        :param std: Std tensor of shape `(batch, 1, features)`.
        :return: Tuple `(norm_mean, norm_std)`, each shaped `(batch, 1, 1)`.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        if mean.shape[-1] is not None and mean.shape[-1] > 1:
            norm_mean = mean[:, :, -1:]
            norm_std = std[:, :, -1:]
        else:
            norm_mean = mean
            norm_std = std
        return norm_mean, norm_std

    def predict_quantiles(
        self,
        context: Union[np.ndarray, keras.utils.PyDataset],
        quantile_levels: Optional[List[float]] = None,
        batch_size: int = 32,
        **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate specific quantile and point, median, forecasts.

        Wraps ``model.predict()``, mapping requested quantile levels to the
        model's trained output indices and extracting the median as the point
        forecast.

        :param context: Input data, a numpy array of shape `(batch_size, input_length, features)` or a dataset.
        :param quantile_levels: Levels to extract; ``None`` returns all trained quantiles. A level not present is mapped to the closest trained level, with a warning.
        :type quantile_levels: List[float], optional
        :param batch_size: Inference batch size. Defaults to 32.
        :type batch_size: int
        :param kwargs: Forwarded to `model.predict()`, e.g. `verbose`.
        :return: Tuple `(quantile_preds, point_preds)`: quantile_preds is `(batch_size, prediction_length, num_requested_quantiles)`, point_preds is `(batch_size, prediction_length)` (the median).
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        if quantile_levels is None:
            quantile_levels = self.quantile_levels

        # [batch, prediction_length, num_trained_quantiles].
        raw_predictions = self.predict(context, batch_size=batch_size, **kwargs)

        quantile_indices = []
        trained_quantiles_arr = np.array(self.quantile_levels)
        for q in quantile_levels:
            if q in self.quantile_levels:
                quantile_indices.append(self.quantile_levels.index(q))
            else:
                closest_idx = int(np.argmin(np.abs(trained_quantiles_arr - q)))
                quantile_indices.append(closest_idx)
                logger.warning(
                    f"Requested quantile {q} not found in trained model "
                    f"{self.quantile_levels}. Using closest match: "
                    f"{self.quantile_levels[closest_idx]}"
                )

        quantile_preds = raw_predictions[:, :, quantile_indices]

        if 0.5 in self.quantile_levels:
            median_idx = self.quantile_levels.index(0.5)
        else:
            median_idx = len(self.quantile_levels) // 2
            logger.debug(
                f"Median (0.5) not found in quantiles. Using index {median_idx} "
                f"({self.quantile_levels[median_idx]}) as point forecast."
            )

        mean_preds = raw_predictions[:, :, median_idx]

        return quantile_preds, mean_preds

    def _forecast(
        self,
        x: Union[np.ndarray, keras.utils.PyDataset],
        quantile_levels: Optional[List[float]] = None,
        **kwargs: Any
    ) -> Forecast:
        """Produce a unified :class:`Forecast`, the ``ForecastMixin`` hook.

        In quantile mode this delegates to :meth:`predict_quantiles`. In point
        mode it calls :meth:`predict` directly and returns `quantiles=None`.

        :param x: Context window of shape `[B, input_length, F]`, or a dataset.
        :param quantile_levels: Levels to extract, quantile mode only; defaults to `self.quantile_levels`.
        :type quantile_levels: List[float], optional
        :param kwargs: Forwarded to `predict_quantiles` / `predict`.
        :return: A :class:`Forecast`. Quantile mode: `point` `[B, H]` and `quantiles` `[B, H, Q]`. Point mode: `point` `[B, H, F]` and `quantiles=None`.
        :rtype: Forecast
        """
        if self.use_quantile_head:
            levels = quantile_levels if quantile_levels is not None else self.quantile_levels
            quantile_preds, point_preds = self.predict_quantiles(x, levels, **kwargs)
            return Forecast(
                point=np.asarray(point_preds),
                quantiles=np.asarray(quantile_preds),
                quantile_levels=list(levels),
            )

        point_preds = self.predict(x, **kwargs)
        return Forecast(
            point=np.asarray(point_preds),
            quantiles=None,
            quantile_levels=None,
        )

    @classmethod
    def from_variant(
        cls,
        variant: str,
        pretrained: bool = False,
        **overrides: Any
    ) -> "xLSTMForecaster":
        """Create an :class:`xLSTMForecaster` from a predefined variant.

        :param variant: One of ``"tiny"``, ``"small"``.
        :type variant: str
        :param pretrained: Must be False; pretrained weights are not provided.
        :type pretrained: bool
        :param overrides: Constructor arguments, e.g. `input_length`, `prediction_length`, taking precedence over the variant defaults.
        :return: An :class:`xLSTMForecaster` instance.
        :raises ValueError: If `variant` is not recognized.
        :raises NotImplementedError: If `pretrained=True` — no checkpoints are shipped.

        Example:
            >>> model = xLSTMForecaster.from_variant(
            ...     "tiny", input_length=64, prediction_length=24)
        """
        if pretrained:
            raise NotImplementedError(
                "Pretrained xLSTMForecaster weights are not provided. "
                "Use pretrained=False and train from scratch."
            )
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()
        config.update(overrides)

        logger.info(f"Creating xLSTMForecaster-{variant.upper()} model")

        return cls(**config)

    def get_config(self) -> Dict[str, Any]:
        """Return the full configuration of the model for serialization."""
        config = super().get_config()
        config.update({
            'input_length': self.input_length,
            'prediction_length': self.prediction_length,
            'num_features': self.num_features,
            'embed_dim': self.embed_dim,
            'num_layers': self.num_layers,
            'mlstm_ratio': self.mlstm_ratio,
            'mlstm_num_heads': self.mlstm_num_heads,
            'mlstm_expansion_factor': self.mlstm_expansion_factor,
            'slstm_forget_gate': self.slstm_forget_gate,
            'ffn_type': self.ffn_type,
            'ffn_expansion_factor': self.ffn_expansion_factor,
            'normalization_type': self.normalization_type,
            'normalization_kwargs': self.normalization_kwargs,
            'dropout_rate': self.dropout_rate,
            'use_quantile_head': self.use_quantile_head,
            'quantile_levels': self.quantile_levels,
            'enforce_monotonicity': self.enforce_monotonicity,
            'use_normalization': self.use_normalization,
            'kernel_initializer': keras.initializers.serialize(
                initializers.get(self.kernel_initializer)
            ),
            'recurrent_initializer': keras.initializers.serialize(
                initializers.get(self.recurrent_initializer)
            ),
            'bias_initializer': keras.initializers.serialize(
                initializers.get(self.bias_initializer)
            ),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer': keras.regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'xLSTMForecaster':
        """Create a model from a configuration, deserializing initializers and regularizers first.

        :param config: Configuration dict as returned by :meth:`get_config`.
        :return: A reconstructed :class:`xLSTMForecaster` instance.
        """
        config = dict(config)
        for key in ("kernel_initializer", "recurrent_initializer",
                    "bias_initializer"):
            if config.get(key) is not None:
                config[key] = keras.initializers.deserialize(config[key])
        for key in ("kernel_regularizer", "recurrent_regularizer",
                    "bias_regularizer"):
            if config.get(key) is not None:
                config[key] = keras.regularizers.deserialize(config[key])
        return cls(**config)


# ---------------------------------------------------------------------


def create_xlstm_forecaster(
    input_length: int,
    prediction_length: int,
    num_features: int = 1,
    embed_dim: int = 128,
    num_layers: int = 4,
    mlstm_ratio: float = 0.5,
    mlstm_num_heads: int = 4,
    use_quantile_head: bool = True,
    quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
    **kwargs: Any
) -> xLSTMForecaster:
    """Factory for :class:`xLSTMForecaster`.

    :param input_length: Length of the input context window.
    :type input_length: int
    :param prediction_length: Forecast horizon `H`.
    :type prediction_length: int
    :param num_features: Number of input/output features. Defaults to 1.
    :type num_features: int
    :param embed_dim: Latent dimensionality. Defaults to 128.
    :type embed_dim: int
    :param num_layers: Number of xLSTM blocks. Defaults to 4.
    :type num_layers: int
    :param mlstm_ratio: Fraction of layers that are mLSTM. Defaults to 0.5.
    :type mlstm_ratio: float
    :param mlstm_num_heads: Number of mLSTM heads. Defaults to 4.
    :type mlstm_num_heads: int
    :param use_quantile_head: Whether to use the quantile head. Defaults to True.
    :type use_quantile_head: bool
    :param quantile_levels: Quantile levels, quantile mode only. Defaults to `DEFAULT_QUANTILES`.
    :type quantile_levels: Sequence[float]
    :param kwargs: Forwarded to the :class:`xLSTMForecaster` constructor.
    :return: A configured :class:`xLSTMForecaster` instance.
    :rtype: xLSTMForecaster
    """
    return xLSTMForecaster(
        input_length=input_length,
        prediction_length=prediction_length,
        num_features=num_features,
        embed_dim=embed_dim,
        num_layers=num_layers,
        mlstm_ratio=mlstm_ratio,
        mlstm_num_heads=mlstm_num_heads,
        use_quantile_head=use_quantile_head,
        quantile_levels=quantile_levels,
        **kwargs
    )

# ---------------------------------------------------------------------
