"""
Haar Wavelet Decomposition Layer supporting multi-dimensional inputs.

This module provides a Keras layer for Haar Discrete Wavelet Transform (DWT)
decomposition that works with timeseries (1D), images (2D), and voxels (3D).
"""

import keras
from keras import ops
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class HaarWaveletDecomposition(keras.layers.Layer):
    """
    Performs Haar Discrete Wavelet Transform (DWT) decomposition.

    Decomposes an input signal into multiple frequency bands using the Haar
    wavelet basis. The Haar wavelet is the simplest wavelet and computes
    averages (approximation) and differences (detail) coefficients.

    Supports:
        - 1D signals (timeseries): Input shape [batch, seq_len, channels]
        - 2D signals (images): Input shape [batch, height, width, channels]
        - 3D signals (voxels): Input shape [batch, depth, height, width, channels]

    **Architecture (1D example)**::

        Input(shape=[batch, seq_len, channels])
               ↓
        For each decomposition level:
               ↓
          +----+----+
          ↓         ↓
        Low-pass  High-pass
        (approx)  (detail)
          ↓         ↓
          +----+----+
               ↓
        Output: List of [approx, detail_1, ..., detail_K]

    **Architecture (2D example)**::

        Input(shape=[batch, H, W, channels])
               ↓
        For each decomposition level:
               ↓
          +----+----+----+----+
          ↓    ↓    ↓    ↓
         LL   LH   HL   HH
          ↓    ↓    ↓    ↓
          +----+----+----+----+
               ↓
        Output: List of [approx, details_1, ..., details_K]
                where each details_i is a tuple of (LH, HL, HH)

    :param num_levels: Number of decomposition levels. Each level halves
        the spatial resolution along each dimension. Defaults to 3.
    :type num_levels: int
    :param kwargs: Additional arguments for the Layer base class.

    :raises ValueError: If num_levels < 1.

    Example::

        # 1D timeseries
        layer = HaarWaveletDecomposition(num_levels=3)
        x = keras.random.normal((2, 128, 16))
        coeffs = layer(x)  # [approx, detail_3, detail_2, detail_1]

        # 2D images
        layer = HaarWaveletDecomposition(num_levels=2)
        x = keras.random.normal((2, 64, 64, 3))
        coeffs = layer(x)  # [approx, (LH2, HL2, HH2), (LH1, HL1, HH1)]
    """

    def __init__(
        self,
        num_levels: int = 3,
        **kwargs: Any
    ) -> None:
        """
        Initialize the Haar Wavelet Decomposition layer.

        :param num_levels: Number of decomposition levels.
        :type num_levels: int
        :param kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)

        if num_levels < 1:
            raise ValueError(
                f"num_levels must be >= 1, got {num_levels}"
            )

        self.num_levels = num_levels
        self._ndim: Optional[int] = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer by determining input dimensionality.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If input rank is not 3, 4, or 5.
        """
        rank = len(input_shape)

        if rank == 3:
            # [batch, seq_len, channels] -> 1D signal
            self._ndim = 1
        elif rank == 4:
            # [batch, height, width, channels] -> 2D signal
            self._ndim = 2
        elif rank == 5:
            # [batch, depth, height, width, channels] -> 3D signal
            self._ndim = 3
        else:
            raise ValueError(
                f"Input must have rank 3 (1D), 4 (2D), or 5 (3D), "
                f"got rank {rank} with shape {input_shape}"
            )

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> List[Union[keras.KerasTensor, Tuple[keras.KerasTensor, ...]]]:
        """
        Apply Haar DWT decomposition.

        :param inputs: Input tensor.
        :type inputs: keras.KerasTensor
        :param training: Training mode flag (unused).
        :type training: Optional[bool]
        :return: List of frequency bands. For 1D: [approx, detail_1, ..., detail_K].
            For 2D/3D: [approx, (details_1), ..., (details_K)] where each
            details tuple contains the detail coefficients for that level.
        :rtype: List[Union[keras.KerasTensor, Tuple[keras.KerasTensor, ...]]]
        """
        if self._ndim == 1:
            return self._decompose_1d(inputs)
        elif self._ndim == 2:
            return self._decompose_2d(inputs)
        else:
            return self._decompose_3d(inputs)

    def _decompose_1d(
        self,
        inputs: keras.KerasTensor
    ) -> List[keras.KerasTensor]:
        """
        Perform 1D Haar wavelet decomposition.

        :param inputs: Input tensor of shape [batch, seq_len, channels].
        :type inputs: keras.KerasTensor
        :return: List [approx, detail_L, detail_L-1, ..., detail_1].
        :rtype: List[keras.KerasTensor]
        """
        details = []
        approximation = inputs
        sqrt2 = ops.sqrt(ops.cast(2.0, inputs.dtype))

        for _ in range(self.num_levels):
            seq_len = ops.shape(approximation)[1]
            even_len = (seq_len // 2) * 2

            x = approximation[:, :even_len, :]
            batch_size = ops.shape(x)[0]
            channels = ops.shape(x)[-1]

            x_reshaped = ops.reshape(
                x,
                (batch_size, even_len // 2, 2, channels)
            )

            low_pass = (x_reshaped[:, :, 0, :] + x_reshaped[:, :, 1, :]) / sqrt2
            high_pass = (x_reshaped[:, :, 0, :] - x_reshaped[:, :, 1, :]) / sqrt2

            details.append(high_pass)
            approximation = low_pass

        return [approximation] + details[::-1]

    def _decompose_2d(
        self,
        inputs: keras.KerasTensor
    ) -> List[Union[keras.KerasTensor, Tuple[keras.KerasTensor, ...]]]:
        """
        Perform 2D Haar wavelet decomposition.

        :param inputs: Input tensor of shape [batch, height, width, channels].
        :type inputs: keras.KerasTensor
        :return: List [approx, (LH_L, HL_L, HH_L), ..., (LH_1, HL_1, HH_1)].
        :rtype: List[Union[keras.KerasTensor, Tuple[keras.KerasTensor, ...]]]
        """
        details = []
        approximation = inputs
        sqrt2 = ops.sqrt(ops.cast(2.0, inputs.dtype))

        for _ in range(self.num_levels):
            shape = ops.shape(approximation)
            height = shape[1]
            width = shape[2]

            even_h = (height // 2) * 2
            even_w = (width // 2) * 2

            x = approximation[:, :even_h, :even_w, :]
            batch_size = ops.shape(x)[0]
            channels = ops.shape(x)[-1]

            # Reshape for row-wise transform: [batch, h//2, 2, w, c]
            x_rows = ops.reshape(
                x,
                (batch_size, even_h // 2, 2, even_w, channels)
            )
            low_rows = (x_rows[:, :, 0, :, :] + x_rows[:, :, 1, :, :]) / sqrt2
            high_rows = (x_rows[:, :, 0, :, :] - x_rows[:, :, 1, :, :]) / sqrt2

            # Reshape for column-wise transform on low_rows: [batch, h//2, w//2, 2, c]
            low_cols = ops.reshape(
                low_rows,
                (batch_size, even_h // 2, even_w // 2, 2, channels)
            )
            ll = (low_cols[:, :, :, 0, :] + low_cols[:, :, :, 1, :]) / sqrt2
            lh = (low_cols[:, :, :, 0, :] - low_cols[:, :, :, 1, :]) / sqrt2

            # Reshape for column-wise transform on high_rows: [batch, h//2, w//2, 2, c]
            high_cols = ops.reshape(
                high_rows,
                (batch_size, even_h // 2, even_w // 2, 2, channels)
            )
            hl = (high_cols[:, :, :, 0, :] + high_cols[:, :, :, 1, :]) / sqrt2
            hh = (high_cols[:, :, :, 0, :] - high_cols[:, :, :, 1, :]) / sqrt2

            details.append((lh, hl, hh))
            approximation = ll

        return [approximation] + details[::-1]

    def _decompose_3d(
        self,
        inputs: keras.KerasTensor
    ) -> List[Union[keras.KerasTensor, Tuple[keras.KerasTensor, ...]]]:
        """
        Perform 3D Haar wavelet decomposition.

        :param inputs: Input tensor of shape [batch, depth, height, width, channels].
        :type inputs: keras.KerasTensor
        :return: List [approx, (details_L), ..., (details_1)] where each
            details tuple contains 7 subbands (LLH, LHL, LHH, HLL, HLH, HHL, HHH).
        :rtype: List[Union[keras.KerasTensor, Tuple[keras.KerasTensor, ...]]]
        """
        details = []
        approximation = inputs
        sqrt2 = ops.sqrt(ops.cast(2.0, inputs.dtype))

        for _ in range(self.num_levels):
            shape = ops.shape(approximation)
            depth = shape[1]
            height = shape[2]
            width = shape[3]

            even_d = (depth // 2) * 2
            even_h = (height // 2) * 2
            even_w = (width // 2) * 2

            x = approximation[:, :even_d, :even_h, :even_w, :]
            batch_size = ops.shape(x)[0]
            channels = ops.shape(x)[-1]

            # Reshape for depth-wise: [batch, d//2, 2, h, w, c]
            x_depth = ops.reshape(
                x,
                (batch_size, even_d // 2, 2, even_h, even_w, channels)
            )
            low_d = (x_depth[:, :, 0, :, :, :] + x_depth[:, :, 1, :, :, :]) / sqrt2
            high_d = (x_depth[:, :, 0, :, :, :] - x_depth[:, :, 1, :, :, :]) / sqrt2

            # Process low_d for height: [batch, d//2, h//2, 2, w, c]
            low_d_h = ops.reshape(
                low_d,
                (batch_size, even_d // 2, even_h // 2, 2, even_w, channels)
            )
            ll_d = (low_d_h[:, :, :, 0, :, :] + low_d_h[:, :, :, 1, :, :]) / sqrt2
            lh_d = (low_d_h[:, :, :, 0, :, :] - low_d_h[:, :, :, 1, :, :]) / sqrt2

            # Process high_d for height
            high_d_h = ops.reshape(
                high_d,
                (batch_size, even_d // 2, even_h // 2, 2, even_w, channels)
            )
            hl_d = (high_d_h[:, :, :, 0, :, :] + high_d_h[:, :, :, 1, :, :]) / sqrt2
            hh_d = (high_d_h[:, :, :, 0, :, :] - high_d_h[:, :, :, 1, :, :]) / sqrt2

            # Process for width dimension
            # ll_d -> lll, llh
            ll_d_w = ops.reshape(
                ll_d,
                (batch_size, even_d // 2, even_h // 2, even_w // 2, 2, channels)
            )
            lll = (ll_d_w[:, :, :, :, 0, :] + ll_d_w[:, :, :, :, 1, :]) / sqrt2
            llh = (ll_d_w[:, :, :, :, 0, :] - ll_d_w[:, :, :, :, 1, :]) / sqrt2

            # lh_d -> lhl, lhh
            lh_d_w = ops.reshape(
                lh_d,
                (batch_size, even_d // 2, even_h // 2, even_w // 2, 2, channels)
            )
            lhl = (lh_d_w[:, :, :, :, 0, :] + lh_d_w[:, :, :, :, 1, :]) / sqrt2
            lhh = (lh_d_w[:, :, :, :, 0, :] - lh_d_w[:, :, :, :, 1, :]) / sqrt2

            # hl_d -> hll, hlh
            hl_d_w = ops.reshape(
                hl_d,
                (batch_size, even_d // 2, even_h // 2, even_w // 2, 2, channels)
            )
            hll = (hl_d_w[:, :, :, :, 0, :] + hl_d_w[:, :, :, :, 1, :]) / sqrt2
            hlh = (hl_d_w[:, :, :, :, 0, :] - hl_d_w[:, :, :, :, 1, :]) / sqrt2

            # hh_d -> hhl, hhh
            hh_d_w = ops.reshape(
                hh_d,
                (batch_size, even_d // 2, even_h // 2, even_w // 2, 2, channels)
            )
            hhl = (hh_d_w[:, :, :, :, 0, :] + hh_d_w[:, :, :, :, 1, :]) / sqrt2
            hhh = (hh_d_w[:, :, :, :, 0, :] - hh_d_w[:, :, :, :, 1, :]) / sqrt2

            # 7 detail subbands (excluding LLL which is the approximation)
            details.append((llh, lhl, lhh, hll, hlh, hhl, hhh))
            approximation = lll

        return [approximation] + details[::-1]

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> List[Union[Tuple[Optional[int], ...], Tuple[Tuple[Optional[int], ...], ...]]]:
        """
        Compute output shapes for all frequency bands.

        :param input_shape: Input shape tuple.
        :type input_shape: Tuple[Optional[int], ...]
        :return: List of output shapes for each band/level.
        :rtype: List[Union[Tuple[Optional[int], ...], Tuple[Tuple[Optional[int], ...], ...]]]
        """
        rank = len(input_shape)

        if rank == 3:
            return self._compute_output_shape_1d(input_shape)
        elif rank == 4:
            return self._compute_output_shape_2d(input_shape)
        elif rank == 5:
            return self._compute_output_shape_3d(input_shape)
        else:
            raise ValueError(
                f"Input must have rank 3 (1D), 4 (2D), or 5 (3D), "
                f"got rank {rank}"
            )

    def _compute_output_shape_1d(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> List[Tuple[Optional[int], ...]]:
        """Compute output shapes for 1D decomposition."""
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        channels = input_shape[2]

        shapes: List[Tuple[Optional[int], ...]] = []
        current_len = seq_len

        # Compute final approximation length
        for _ in range(self.num_levels):
            if current_len is not None:
                current_len = (current_len // 2) * 2 // 2

        shapes.append((batch_size, current_len, channels))

        # Detail shapes from coarsest to finest
        detail_len = current_len
        for _ in range(self.num_levels):
            shapes.append((batch_size, detail_len, channels))
            if detail_len is not None:
                detail_len = detail_len * 2

        return shapes

    def _compute_output_shape_2d(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> List[Union[Tuple[Optional[int], ...], Tuple[Tuple[Optional[int], ...], ...]]]:
        """Compute output shapes for 2D decomposition."""
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]

        shapes: List[Union[Tuple[Optional[int], ...], Tuple[Tuple[Optional[int], ...], ...]]] = []
        current_h = height
        current_w = width

        # Compute final approximation dimensions
        for _ in range(self.num_levels):
            if current_h is not None:
                current_h = (current_h // 2) * 2 // 2
            if current_w is not None:
                current_w = (current_w // 2) * 2 // 2

        shapes.append((batch_size, current_h, current_w, channels))

        # Detail shapes from coarsest to finest (each is a tuple of 3)
        detail_h = current_h
        detail_w = current_w
        for _ in range(self.num_levels):
            detail_shape = (batch_size, detail_h, detail_w, channels)
            shapes.append((detail_shape, detail_shape, detail_shape))
            if detail_h is not None:
                detail_h = detail_h * 2
            if detail_w is not None:
                detail_w = detail_w * 2

        return shapes

    def _compute_output_shape_3d(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> List[Union[Tuple[Optional[int], ...], Tuple[Tuple[Optional[int], ...], ...]]]:
        """Compute output shapes for 3D decomposition."""
        batch_size = input_shape[0]
        depth = input_shape[1]
        height = input_shape[2]
        width = input_shape[3]
        channels = input_shape[4]

        shapes: List[Union[Tuple[Optional[int], ...], Tuple[Tuple[Optional[int], ...], ...]]] = []
        current_d = depth
        current_h = height
        current_w = width

        # Compute final approximation dimensions
        for _ in range(self.num_levels):
            if current_d is not None:
                current_d = (current_d // 2) * 2 // 2
            if current_h is not None:
                current_h = (current_h // 2) * 2 // 2
            if current_w is not None:
                current_w = (current_w // 2) * 2 // 2

        shapes.append((batch_size, current_d, current_h, current_w, channels))

        # Detail shapes from coarsest to finest (each is a tuple of 7)
        detail_d = current_d
        detail_h = current_h
        detail_w = current_w
        for _ in range(self.num_levels):
            detail_shape = (batch_size, detail_d, detail_h, detail_w, channels)
            shapes.append(tuple(detail_shape for _ in range(7)))
            if detail_d is not None:
                detail_d = detail_d * 2
            if detail_h is not None:
                detail_h = detail_h * 2
            if detail_w is not None:
                detail_w = detail_w * 2

        return shapes

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "num_levels": self.num_levels,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HaarWaveletDecomposition":
        """
        Create layer from configuration dictionary.

        :param config: Configuration dictionary.
        :type config: Dict[str, Any]
        :return: New layer instance.
        :rtype: HaarWaveletDecomposition
        """
        return cls(**config)

# ---------------------------------------------------------------------
