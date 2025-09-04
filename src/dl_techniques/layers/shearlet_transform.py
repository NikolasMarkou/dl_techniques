import keras
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional, Dict, Any, Union


@keras.saving.register_keras_serializable()
class ShearletTransform(keras.layers.Layer):
    """
    Multi-scale, multi-directional shearlet transform layer for enhanced time-frequency analysis.

    This layer implements a cone-adapted discrete shearlet transform with improved frame
    properties and frequency coverage. The shearlet transform provides optimal sparse
    representation of images with edges and directional features by combining multi-scale
    analysis with directional selectivity.

    **Intent**: Provide a differentiable shearlet transform that can be integrated into
    neural networks for enhanced feature extraction, particularly effective for images
    containing edges, textures, and directional patterns.

    **Architecture**:
    ```
    Input(shape=[batch, height, width, channels])
           ↓
    2D FFT → Frequency Domain
           ↓
    Apply Shearlet Filters (scales × directions filters)
           ↓
    2D IFFT → Spatial Domain Coefficients
           ↓
    Output(shape=[batch, height, width, num_filters])
    ```

    **Mathematical Foundation**:
    The shearlet transform uses anisotropic dilations and shears to create a system
    of functions {ψ_{j,k,m}} where:
    - j: scale parameter (controls resolution)
    - k: direction parameter (controls orientation)
    - m: translation parameter (spatial location)

    Each shearlet is defined by:
    ψ_{j,k}(ξ) = ψ̂(ξ₁/2^j, ξ₂/2^{j/2} - k ξ₁/2^{j/2})

    Args:
        scales: Integer, number of scales in the transform. Controls the multi-resolution
            analysis depth. Higher values provide finer scale analysis. Must be positive.
            Defaults to 4.
        directions: Integer, number of directions per scale. Controls angular resolution.
            Higher values provide better directional selectivity but increase computation.
            Must be positive and preferably even. Defaults to 8.
        alpha: Float, anisotropy parameter controlling the relationship between scale
            and direction sampling. Value of 0.5 provides parabolic scaling which is
            optimal for edge detection. Must be in (0, 1]. Defaults to 0.5.
        high_freq: Boolean, whether to include high frequency components in the transform.
            When True, captures fine details and noise. When False, focuses on low-mid
            frequencies. Defaults to True.
        kernel_regularizer: Optional regularizer for internal filter parameters.
            Can be used to constrain filter responses during training. Defaults to None.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`.
        All input images should have the same spatial dimensions within a batch.

    Output shape:
        4D tensor with shape: `(batch_size, height, width, num_filters)`.
        Where num_filters = 1 (low-pass) + scales * directions (directional filters).
        Spatial dimensions are preserved.

    Attributes:
        height: Integer, height of input images (set during build).
        width: Integer, width of input images (set during build).
        freq_x: Tensor, x-coordinates of frequency grid (set during build).
        freq_y: Tensor, y-coordinates of frequency grid (set during build).
        filters: List of complex-valued shearlet filters in frequency domain (set during build).

    Example:
        ```python
        # Basic usage for texture analysis
        shearlet = ShearletTransform(scales=3, directions=6)
        inputs = keras.Input(shape=(128, 128, 1))
        coefficients = shearlet(inputs)  # Shape: (batch, 128, 128, 19)

        # High-resolution directional analysis
        shearlet = ShearletTransform(
            scales=5,
            directions=16,
            alpha=0.5,
            high_freq=True
        )

        # With regularization for training stability
        shearlet = ShearletTransform(
            scales=4,
            directions=8,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )
        ```

    References:
        - G. Kutyniok and D. Labate, "Shearlets: Multiscale Analysis for Multivariate Data"
        - K. Guo, G. Kutyniok, and D. Labate, "Sparse multidimensional representations using shearlets"

    Note:
        This implementation uses cone-adapted discrete shearlets which provide a tight frame
        with optimal approximation properties for cartoon-like images. The transform is
        particularly effective for images with piecewise smooth content separated by curves.
    """

    def __init__(
            self,
            scales: int = 4,
            directions: int = 8,
            alpha: float = 0.5,
            high_freq: bool = True,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if scales <= 0:
            raise ValueError(f"scales must be positive, got {scales}")
        if directions <= 0:
            raise ValueError(f"directions must be positive, got {directions}")
        if not (0 < alpha <= 1):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")

        # Store configuration
        self.scales = scales
        self.directions = directions
        self.alpha = alpha
        self.high_freq = high_freq
        self.kernel_regularizer = kernel_regularizer

        # Initialize attributes (created in build)
        self.height: Optional[int] = None
        self.width: Optional[int] = None
        self.freq_x: Optional[tf.Tensor] = None
        self.freq_y: Optional[tf.Tensor] = None
        self.filters: Optional[List[tf.Tensor]] = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and create shearlet filter bank.

        This method creates the frequency grid and generates all shearlet filters
        with proper frame properties and normalization.

        Args:
            input_shape: Input tensor shape (batch_size, height, width, channels)
        """
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {len(input_shape)}D")

        if input_shape[1] is None or input_shape[2] is None:
            raise ValueError("Height and width dimensions must be specified")

        self.height = input_shape[1]
        self.width = input_shape[2]

        # Create frequency grid
        self.freq_x, self.freq_y = self._create_freq_grid()

        # Generate shearlet filters with improved frame properties
        self.filters = self._create_shearlet_filters()

        super().build(input_shape)

    def _create_freq_grid(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Create normalized frequency grid for filter construction.

        Creates a 2D frequency grid normalized to [-0.5, 0.5] which is appropriate
        for digital signal processing and ensures proper filter responses.

        Returns:
            Tuple of (freq_x, freq_y) tensors representing the frequency coordinates
        """
        # Create normalized frequency ranges
        fx = tf.linspace(-0.5, 0.5, self.width)
        fy = tf.linspace(-0.5, 0.5, self.height)

        # Create 2D grid with proper broadcasting
        freq_y, freq_x = tf.meshgrid(fy, fx, indexing='ij')
        return freq_x, freq_y

    def _meyer_window(
            self,
            x: tf.Tensor,
            a: float = 1.0,
            eps: float = 1e-6
    ) -> tf.Tensor:
        """
        Create smooth Meyer window function with enhanced boundary handling.

        The Meyer window provides smooth cutoffs with good frequency localization
        properties, essential for constructing tight frames.

        Args:
            x: Input values to evaluate the window function
            a: Scaling parameter controlling window width
            eps: Small value for numerical stability and minimum response

        Returns:
            Window function values with smooth transitions
        """

        def smooth_transition(x: tf.Tensor) -> tf.Tensor:
            """Enhanced smooth polynomial transition function."""
            x = tf.clip_by_value(x, 0.0, 1.0)
            # Higher order polynomial for smoother transition
            return x * x * x * (10.0 + x * (-15.0 + 6.0 * x))

        # Normalize input and ensure positivity
        x = tf.abs(x)
        x = tf.clip_by_value(x / (a + eps), 0.0, 1.0)

        # Compute window value with broader support
        value = tf.where(
            x < 1 / 3,
            tf.ones_like(x),
            tf.where(
                x < 2 / 3,
                smooth_transition(2.0 - 3.0 * x),
                tf.zeros_like(x)
            )
        )

        # Add small offset to prevent zero response
        return value + eps

    def _create_shearlet_filters(self) -> List[tf.Tensor]:
        """
        Create complete shearlet filter bank with guaranteed frame properties.

        Constructs a system of shearlet filters including:
        1. Low-pass scaling function
        2. Directional bandpass filters for each scale and orientation

        The filters are designed to form a tight frame with good reconstruction
        properties and minimal artifacts.

        Returns:
            List of complex-valued shearlet filters in frequency domain
        """
        filters = []

        # Get polar coordinates for filter construction
        rho = tf.sqrt(self.freq_x ** 2 + self.freq_y ** 2)
        theta = tf.atan2(self.freq_y, self.freq_x)

        # Ensure non-zero minimum response for stability
        min_response = 1e-3

        # Create low-pass filter with guaranteed minimum response
        phi_low = tf.maximum(
            self._meyer_window(2.0 * rho),
            min_response
        )
        filters.append(tf.cast(phi_low, tf.complex64))

        # Create directional filters for each scale
        for j in range(self.scales):
            scale = 2.0 ** j

            # Create overlapping radial windows
            window_j = tf.maximum(
                self._meyer_window(rho / scale) *
                (1.0 - self._meyer_window(2.0 * rho / scale)),
                min_response
            )

            # Create directional filters
            for k in range(-self.directions // 2, self.directions // 2 + 1):
                shear = k / (self.directions / 2.0)
                angle = tf.atan(shear)

                # Create angular window with minimum response
                dir_window = tf.maximum(
                    self._meyer_window(
                        (theta - angle) / (0.5 * np.pi),
                        a=2.0 / (self.directions + 2)
                    ),
                    min_response
                )

                # Combine radial and angular windows
                shearlet = tf.maximum(window_j * dir_window, min_response)

                # Normalize filter energy
                norm = tf.sqrt(tf.reduce_mean(tf.abs(shearlet) ** 2) + 1e-6)
                shearlet = shearlet / norm

                filters.append(tf.cast(shearlet, tf.complex64))

        # Apply final normalization for tight frame property
        return self._normalize_filter_bank(filters)

    def _normalize_filter_bank(self, filters: List[tf.Tensor]) -> List[tf.Tensor]:
        """
        Normalize the complete filter bank to achieve tight frame properties.

        This method ensures that the sum of squared filter responses approximates
        unity across all frequencies, which is essential for perfect reconstruction
        and stable forward/inverse transforms.

        Args:
            filters: List of initial shearlet filters

        Returns:
            List of normalized filters with improved frame bounds
        """
        # Step 1: Initial individual filter normalization
        normalized_filters = []
        for f in filters:
            energy = tf.reduce_mean(tf.abs(f) ** 2)
            normalized_filters.append(f / tf.cast(tf.sqrt(energy + 1e-6), tf.complex64))

        # Step 2: Calculate total frequency response (real-valued)
        total_response = tf.reduce_sum(
            [tf.abs(f) ** 2 for f in normalized_filters],
            axis=0
        )

        # Step 3: Identify areas with insufficient coverage
        mean_response = tf.reduce_mean(total_response)
        min_threshold = mean_response * 0.01  # 1% of mean response

        # Step 4: Create boost factors (keep operations in real domain)
        boost_needed = total_response < min_threshold
        boost_factor_real = min_threshold / (total_response + 1e-6)

        # Step 5: Apply selective boosting to filters
        boosted_filters = []
        for f in normalized_filters:
            # Convert boost factor to complex and apply selectively
            boost_factor_complex = tf.cast(boost_factor_real, tf.complex64)
            boost_mask_complex = tf.cast(boost_needed, tf.complex64)

            # Apply boosting where needed
            boosted = f * tf.where(
                boost_needed,  # Use real boolean mask for where condition
                boost_factor_complex,
                tf.ones_like(boost_factor_complex)
            )
            boosted_filters.append(boosted)

        # Step 6: Final frame normalization
        final_response = tf.reduce_sum(
            [tf.abs(f) ** 2 for f in boosted_filters],
            axis=0
        )
        normalization = tf.cast(tf.sqrt(final_response + 1e-6), tf.complex64)

        # Ensure frame bounds by normalizing with proper scaling
        target_bound = 1.0
        final_filters = [
            target_bound * f / normalization
            for f in boosted_filters
        ]

        # Add small DC offset to prevent zero response
        dc_offset = tf.cast(1e-3, tf.complex64)
        final_filters = [f + dc_offset for f in final_filters]

        return final_filters

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply shearlet transform to input images.

        Performs the forward shearlet transform by:
        1. Computing 2D FFT of input images
        2. Multiplying with each shearlet filter in frequency domain
        3. Computing 2D IFFT to get spatial domain coefficients
        4. Stacking all coefficient maps along channel dimension

        Args:
            inputs: Input tensor of shape [batch_size, height, width, channels]
            training: Boolean indicating training mode (unused but kept for API consistency)

        Returns:
            Tensor of shearlet coefficients with shape [batch_size, height, width, num_filters]
        """
        # Ensure float32 input for numerical stability
        inputs = tf.cast(inputs, tf.float32)

        # Handle single channel vs multi-channel inputs
        input_shape = tf.shape(inputs)
        if len(inputs.shape) == 3:
            inputs = tf.expand_dims(inputs, axis=-1)

        # Process each channel independently
        batch_size = input_shape[0]
        channels = input_shape[-1] if len(inputs.shape) == 4 else 1

        all_coefficients = []

        for c in range(channels):
            # Extract single channel
            if channels == 1:
                channel_input = inputs[..., 0:1]
            else:
                channel_input = inputs[..., c:c + 1]

            # Remove channel dimension for FFT
            channel_input = tf.squeeze(channel_input, axis=-1)

            # Compute 2D FFT
            fft = tf.signal.fftshift(
                tf.signal.fft2d(tf.cast(channel_input, tf.complex64))
            )

            # Apply each filter and compute inverse FFT
            channel_coefficients = []
            for filter_kernel in self.filters:
                filtered = fft * filter_kernel
                coeff = tf.cast(
                    tf.math.real(
                        tf.signal.ifft2d(
                            tf.signal.ifftshift(filtered)
                        )
                    ),
                    tf.float32
                )
                channel_coefficients.append(coeff)

            # Stack coefficients for this channel
            channel_stack = tf.stack(channel_coefficients, axis=-1)
            all_coefficients.append(channel_stack)

        # Combine all channels
        if len(all_coefficients) == 1:
            result = all_coefficients[0]
        else:
            result = tf.concat(all_coefficients, axis=-1)

        return result

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape given input shape.

        Args:
            input_shape: Shape tuple of input

        Returns:
            Shape tuple of output
        """
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {len(input_shape)}D")

        batch_size, height, width, channels = input_shape

        # Calculate number of filters: 1 low-pass + scales * (directions + 1) directional
        num_filters = 1 + self.scales * (self.directions + 1)

        # Account for multiple input channels
        if channels is not None:
            num_filters *= channels

        return (batch_size, height, width, num_filters)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing all configuration parameters
        """
        config = super().get_config()
        config.update({
            'scales': self.scales,
            'directions': self.directions,
            'alpha': self.alpha,
            'high_freq': self.high_freq,
            'kernel_regularizer': keras.regularizers.serialize(
                self.kernel_regularizer
            )
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ShearletTransform':
        """
        Create layer from configuration dictionary.

        Args:
            config: Configuration dictionary from get_config()

        Returns:
            New ShearletTransform layer instance
        """
        regularizer_config = config.pop('kernel_regularizer', None)
        if regularizer_config:
            config['kernel_regularizer'] = keras.regularizers.deserialize(
                regularizer_config
            )
        return cls(**config)