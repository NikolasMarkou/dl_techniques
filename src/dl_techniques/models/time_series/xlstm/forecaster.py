"""
xLSTM-based forecaster for continuous time-series probabilistic forecasting.

This module provides :class:`xLSTMForecaster`, a forecasting sibling of the
language-model :class:`dl_techniques.models.time_series.xlstm.model.xLSTM`. It
reuses the same mLSTM/sLSTM residual block stack but replaces the LM-specific
token ``Embedding`` + ``vocab`` head with a continuous-input projection and a
forecasting head (quantile or point), and adds the unified ``Forecast``
contract via :class:`ForecastMixin`.

Architecture Overview:
    ```
    Context [B, input_length, F]
        |
    (optional) reversible instance-norm (per-instance z-score over time)
        |
    Dense input projection -> [B, T, embed_dim]
        |
    [mLSTM / sLSTM blocks] x num_layers   (mlstm_ratio picks the split)
        |
    final normalization
        |
    global mean-pool over time -> [B, 1, embed_dim]
        |
    head:  QuantileHead  -> [B, H, Q]      (quantile mode)
           Dense+reshape -> [B, H, F]      (point mode)
        |
    (optional) reversible denormalization (undo the instance-norm)
    ```

The global mean-pool collapses the sequence axis to a static length of 1, which
satisfies the ``QuantileHead(flatten_input=True)`` requirement for a defined
(non-``None``) sequence length. This mirrors the proven TiRex pooled-head
pattern.

References:
    - Beck, M., et al. (2024). xLSTM: Extended Long Short-Term Memory.
      arXiv:2405.04517v2
    - Kim, T., et al. (2022). Reversible Instance Normalization for Accurate
      Time-Series Forecasting against Distribution Shift. ICLR.
"""

import keras
import numpy as np
from keras import ops, initializers
from typing import Optional, Union, List, Any, Dict, Tuple, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.models.time_series.forecast import Forecast, ForecastMixin
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.time_series.xlstm_blocks import mLSTMBlock, sLSTMBlock
from dl_techniques.layers.time_series.quantile_head_fixed_io import QuantileHead

# ---------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------

# Default quantile levels for probabilistic forecasting
DEFAULT_QUANTILES: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class xLSTMForecaster(keras.Model, ForecastMixin):
    """
    xLSTM forecaster for continuous time-series forecasting.

    Consumes a continuous context window ``[B, input_length, num_features]`` and
    emits an ``H``-step forecast. In quantile mode it emits ``[B, H, Q]``; in
    point mode ``[B, H, num_features]``. Mixes in :class:`ForecastMixin` so
    callers get a uniform ``predict_forecast(x) -> Forecast`` entry point.

    **Intent**: Provide a continuous-input forecasting counterpart to the LM
    :class:`xLSTM`, reusing its mLSTM/sLSTM block stack while plugging into the
    shared ``Forecast`` contract (the TiRex template: revin + global mean-pool +
    ``QuantileHead`` + ``ForecastMixin``).

    **Architecture**:
    ```
    Context [B, input_length, F]
       -> (optional) reversible instance-norm
       -> Dense(embed_dim)
       -> [mLSTM / sLSTM blocks] x num_layers
       -> final_norm
       -> global mean-pool over time -> [B, 1, embed_dim]
       -> head (QuantileHead -> [B, H, Q] | Dense -> [B, H, F])
       -> (optional) reversible denormalization
    ```

    The blocks are distributed according to ``mlstm_ratio``: the first
    ``int(num_layers * mlstm_ratio)`` blocks are mLSTM, the remaining are sLSTM.

    Args:
        input_length: Integer, length of the input context window.
        prediction_length: Integer, length of the forecast horizon ``H``.
        num_features: Integer, number of input/output features ``F``. Defaults
            to 1 (the safe default for ``QuantileLoss``).
        embed_dim: Integer, dimensionality of the latent representation. Must be
            divisible by ``mlstm_num_heads``.
        num_layers: Integer, total number of xLSTM blocks.
        mlstm_ratio: Float in ``[0, 1]``, fraction of layers that are mLSTM.
            Defaults to 0.5.
        mlstm_num_heads: Integer, number of heads for mLSTM blocks. Defaults
            to 4. Must divide ``embed_dim``.
        mlstm_expansion_factor: Integer, expansion factor for mLSTM blocks.
            Defaults to 2.
        slstm_forget_gate: ``'sigmoid'`` or ``'exp'``, sLSTM forget-gate
            activation. Defaults to ``'sigmoid'``.
        ffn_type: String, FFN type for sLSTM blocks. Defaults to ``'swiglu'``.
        ffn_expansion_factor: Integer, FFN expansion for sLSTM. Defaults to 2.
        normalization_type: String, normalization layer type. Defaults to
            ``'layer_norm'``.
        normalization_kwargs: Optional dict of normalization kwargs.
        dropout_rate: Float, dropout rate. Defaults to 0.0.
        use_quantile_head: Boolean, if True use a :class:`QuantileHead`
            (quantile mode); otherwise a ``Dense`` point head. Defaults to True.
        quantile_levels: List of floats, the quantile levels to predict (quantile
            mode only). Defaults to ``DEFAULT_QUANTILES``.
        enforce_monotonicity: Boolean, enforce non-crossing quantiles in the
            ``QuantileHead``. Defaults to True.
        use_normalization: Boolean, enable reversible per-instance z-score
            normalization (RevIN). Defaults to True.
        kernel_initializer: Initializer for kernel weights.
        recurrent_initializer: Initializer for recurrent weights.
        bias_initializer: Initializer for bias weights.
        kernel_regularizer: Optional regularizer for kernel weights.
        recurrent_regularizer: Optional regularizer for recurrent weights.
        bias_regularizer: Optional regularizer for bias weights.
        name: String, model name. Defaults to ``'xLSTMForecaster'``.
        **kwargs: Additional arguments for the Model base class.

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

    # Model variant configurations (small / tiny for quick experiments).
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
        quantile_levels: List[float] = DEFAULT_QUANTILES,
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

        # Validate inputs
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

        # Store configuration
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
        self.normalization_kwargs = normalization_kwargs or {}
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

        # Continuous input projection (replaces the LM token Embedding).
        self.input_projection = keras.layers.Dense(
            embed_dim,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name='input_projection',
        )

        # Create xLSTM blocks (block-stacking loop copied from the LM xLSTM).
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

        # Final normalization (wiring copied from the LM xLSTM).
        self.final_norm = create_normalization_layer(
            normalization_type=normalization_type,
            name='final_norm',
            **self.normalization_kwargs
        )

        # Forecasting head (quantile or point), gated by use_quantile_head.
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

    def call(
        self,
        inputs: Union[keras.KerasTensor, np.ndarray],
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """
        Forward pass through the xLSTM forecaster.

        Args:
            inputs: Context window of shape ``[B, input_length, num_features]``
                (a 2D ``[B, input_length]`` tensor is expanded to 3D).
            training: Boolean, whether in training mode.

        Returns:
            Quantile mode: ``[B, prediction_length, num_quantiles]``.
            Point mode: ``[B, prediction_length, num_features]``.
        """
        # Ensure 3D input.
        if len(inputs.shape) == 2:
            inputs = ops.expand_dims(inputs, axis=-1)

        # 1. HANDLE MASKING (before normalization, to avoid NaN propagation).
        nan_mask = ops.logical_not(ops.isnan(inputs))
        nan_mask = ops.cast(nan_mask, dtype=inputs.dtype)
        clean_inputs = ops.where(ops.isnan(inputs), ops.zeros_like(inputs), inputs)

        # 2. REVERSIBLE INSTANCE-NORM (NaN-safe z-score over the time axis).
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

        # 3. INPUT PROJECTION -> [B, T, embed_dim].
        x = self.input_projection(x, training=training)

        # 4. xLSTM BLOCK STACK.
        for block in self.blocks:
            x = block(x, training=training, mask=None)

        # 5. FINAL NORMALIZATION.
        x = self.final_norm(x, training=training)

        # 6. GLOBAL MEAN-POOL over time -> [B, 1, embed_dim].
        #    keepdims=True keeps a STATIC seq_len of 1, required by
        #    QuantileHead(flatten_input=True).
        pooled = ops.mean(x, axis=1, keepdims=True)

        # 7. HEAD.
        if self.use_quantile_head:
            # QuantileHead -> [B, prediction_length, num_quantiles].
            outputs = self.head(pooled, training=training)
        else:
            # Dense -> [B, 1, H*F] -> reshape [B, H, F].
            flat = self.head(pooled, training=training)
            batch_size = ops.shape(flat)[0]
            outputs = ops.reshape(
                flat,
                (batch_size, self.prediction_length, self.num_features),
            )

        # 8. DENORMALIZE OUTPUT (undo the reversible instance-norm).
        if self.use_normalization:
            norm_mean, norm_std = self._get_target_stats(mean, std)
            outputs = (outputs * norm_std) + norm_mean

        return outputs

    @staticmethod
    def _get_target_stats(
        mean: keras.KerasTensor,
        std: keras.KerasTensor,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Extract normalization stats for the target (last) feature.

        For multivariate inputs the target is assumed to be the last feature.
        Returns stats shaped ``(Batch, 1, 1)`` so they broadcast against both
        quantile predictions ``(B, H, Q)`` and point predictions ``(B, H, F)``.

        Args:
            mean: Mean tensor of shape ``(Batch, 1, Features)``.
            std: Std tensor of shape ``(Batch, 1, Features)``.

        Returns:
            Tuple ``(norm_mean, norm_std)``, each shaped ``(Batch, 1, 1)``.
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
        """
        Generate specific quantile and point (median) forecasts.

        High-level wrapper around ``model.predict()`` that maps user-requested
        quantile levels to the model's trained output indices, and extracts the
        median (0.5) as the point forecast.

        Args:
            context: Input data, either a numpy array of shape
                ``(batch_size, input_length, features)`` or a dataset.
            quantile_levels: Levels to extract. If None, returns all trained
                quantiles. Levels not present are mapped to the closest trained
                level (with a warning).
            batch_size: Integer, inference batch size. Defaults to 32.
            **kwargs: Forwarded to ``model.predict()`` (e.g. ``verbose``).

        Returns:
            Tuple ``(quantile_preds, point_preds)``:
            - ``quantile_preds``: ``(batch_size, prediction_length, num_requested_quantiles)``.
            - ``point_preds``: ``(batch_size, prediction_length)`` (the median).
        """
        if quantile_levels is None:
            quantile_levels = self.quantile_levels

        # Forward pass: [batch, prediction_length, num_trained_quantiles].
        raw_predictions = self.predict(context, batch_size=batch_size, **kwargs)

        # Map requested levels to trained output indices.
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

        # Extract the median (0.5) as the point forecast.
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
        """Produce a unified :class:`Forecast` (the ``ForecastMixin`` hook).

        In quantile mode this delegates to :meth:`predict_quantiles` and packs
        the median + quantiles. In point mode it calls :meth:`predict` directly
        and returns ``quantiles=None`` (a point model must not fabricate
        intervals).

        Args:
            x: Context window of shape ``[B, input_length, F]`` (or a dataset).
            quantile_levels: Levels to extract (quantile mode only); defaults to
                the model's configured ``self.quantile_levels``.
            **kwargs: Forwarded to ``predict_quantiles`` / ``predict``.

        Returns:
            A :class:`Forecast`. Quantile mode: ``point`` ``[B, H]`` and
            ``quantiles`` ``[B, H, Q]``. Point mode: ``point`` ``[B, H, F]`` and
            ``quantiles=None``.
        """
        if self.use_quantile_head:
            levels = quantile_levels if quantile_levels is not None else self.quantile_levels
            quantile_preds, point_preds = self.predict_quantiles(x, levels, **kwargs)
            return Forecast(
                point=np.asarray(point_preds),
                quantiles=np.asarray(quantile_preds),
                quantile_levels=list(levels),
            )

        # Point mode: [B, H, F], no quantiles.
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
        """
        Create an :class:`xLSTMForecaster` from a predefined variant.

        Args:
            variant: One of ``"tiny"``, ``"small"``.
            pretrained: Must be False; pretrained weights are not provided.
            **overrides: Override / supply constructor arguments (e.g.
                ``input_length``, ``prediction_length``). These take precedence
                over the variant defaults.

        Returns:
            An :class:`xLSTMForecaster` instance.

        Raises:
            ValueError: If ``variant`` is not recognized.
            NotImplementedError: If ``pretrained=True`` (no checkpoints shipped).

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
        """Create a model from its configuration."""
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
    quantile_levels: List[float] = DEFAULT_QUANTILES,
    **kwargs: Any
) -> xLSTMForecaster:
    """
    Factory for :class:`xLSTMForecaster`.

    Thin config-driven constructor wrapper following the repo factory
    convention. All additional constructor arguments are forwarded via
    ``**kwargs``.

    Args:
        input_length: Length of the input context window.
        prediction_length: Forecast horizon ``H``.
        num_features: Number of input/output features. Defaults to 1.
        embed_dim: Latent dimensionality. Defaults to 128.
        num_layers: Number of xLSTM blocks. Defaults to 4.
        mlstm_ratio: Fraction of layers that are mLSTM. Defaults to 0.5.
        mlstm_num_heads: Number of mLSTM heads. Defaults to 4.
        use_quantile_head: Whether to use the quantile head. Defaults to True.
        quantile_levels: Quantile levels (quantile mode). Defaults to
            ``DEFAULT_QUANTILES``.
        **kwargs: Forwarded to the :class:`xLSTMForecaster` constructor.

    Returns:
        A configured :class:`xLSTMForecaster` instance.
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
