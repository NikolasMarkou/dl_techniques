"""
Inject explicit spatial coordinate information into feature maps.

This layer addresses a fundamental property of standard convolutional neural
networks: translation equivariance. While this property is a powerful
inductive bias for many tasks, it makes CNNs agnostic to the absolute
spatial location of features. This layer provides a simple, non-trainable
mechanism to explicitly encode coordinate information, making the network
"aware" of where features are located within the input grid. This technique
is a core component of architectures like CoordConv.

Architecturally, the layer functions as a deterministic coordinate generator.
It does not learn any parameters. Instead, its operation is two-fold:
1.  **Build Phase:** A low-resolution "prototype" coordinate grid is created
    and stored internally. This grid represents the normalized `x` and `y`
    coordinates for a small feature map.
2.  **Call Phase:** During the forward pass, this prototype grid is
    dynamically resized to match the spatial dimensions of the input
    feature map using interpolation. The resized grid is then tiled to match
    the batch size.

The final output is a tensor of shape `(batch, height, width, 2)` that can
be concatenated with the input feature map, effectively adding two new
channels that explicitly state the `(x, y)` position of each feature vector.

The mathematical process involves two key steps. First, a base grid is
formed by creating two matrices for the normalized `x` and `y` coordinates,
typically spanning the interval `[-0.5, 0.5]`. The second, and more critical,
step is the standardization of these coordinate grids. Each grid is
independently normalized to have a mean of zero and a standard deviation of
one:

`z = (x - μ) / σ`

This standardization is crucial for training stability. It ensures that the
coordinate features have a similar statistical distribution to typical
learned feature activations. Without it, the large, unshuffled values of the
coordinates could dominate the initial learning process when concatenated
with activations that are often centered around zero.

References:
    - Liu et al., 2018. An intriguing failing of convolutional neural
      networks and the CoordConv solution.
      (https://arxiv.org/abs/1807.03247)

"""

import keras
from keras import ops
from typing import Tuple, Optional, Any, Literal

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SpatialLayer(keras.layers.Layer):
    """
    Spatial coordinate grid generator for injecting positional information into models.

    This layer creates normalized coordinate grids (x, y) that provide explicit spatial
    information to neural networks. It generates standardized coordinate features that
    can be concatenated with existing feature maps to enhance spatial reasoning capabilities
    in computer vision_heads tasks.

    **Intent**: Enable explicit spatial awareness in neural networks by providing
    normalized coordinate information for each spatial location, improving performance
    in tasks requiring precise spatial reasoning such as object detection, segmentation,
    and generative modeling.

    **Architecture**:
    ```
    Input(shape=[batch, height, width, channels])
           ↓
    Internal Grid(resolution) → Standardize → Resize(height, width)
           ↓
    Batch Expansion → Output(shape=[batch, height, width, 2])
    ```

    **Grid Processing Steps**:
    1. **Base Grid Creation**: Generate normalized coordinate meshgrid at specified resolution
    2. **Standardization**: Normalize coordinates to zero mean and unit variance
    3. **Dynamic Resizing**: Resize grid to match input spatial dimensions
    4. **Batch Tiling**: Expand to match input batch size

    **Mathematical Operations**:
    - **Grid Generation**: x, y ∈ [-0.5, 0.5] over specified resolution
    - **Standardization**: coord_norm = (coord - μ) / (σ + ε)
    - **Bilinear/Nearest Interpolation**: Resize to target spatial dimensions

    The output provides explicit (x, y) coordinates for each spatial location,
    enabling the network to directly access positional information.

    Args:
        resolution: Tuple of integers (height, width) specifying the initial grid resolution.
            Controls the base resolution before dynamic resizing. Both values must be positive.
            Higher resolutions provide finer initial coordinate granularity. Defaults to (4, 4).
        resize_method: Interpolation method for dynamic resizing to match input dimensions.
            Available options:
            - 'nearest': Nearest neighbor interpolation (faster, less smooth)
            - 'bilinear': Bilinear interpolation (slower, smoother gradients)
            Defaults to 'nearest'.
        **kwargs: Additional keyword arguments for Layer base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`.
        The height and width dimensions determine the output coordinate grid size.

    Output shape:
        4D tensor with shape: `(batch_size, height, width, 2)`.
        The last dimension contains normalized (x, y) coordinates for each spatial location.

    Attributes:
        xy_grid: Internal coordinate grid tensor created during build phase.
            Shape: (1, resolution[0], resolution[1], 2)

    Example:
        ```python
        # Basic usage with default resolution
        spatial_layer = SpatialLayer()
        inputs = keras.Input(shape=(64, 64, 3))
        coords = spatial_layer(inputs)  # Shape: (batch, 64, 64, 2)

        # High-resolution base grid with bilinear resizing
        spatial_layer = SpatialLayer(
            resolution=(16, 16),
            resize_method='bilinear'
        )

        # Concatenate with original features
        inputs = keras.Input(shape=(32, 32, 128))
        coords = SpatialLayer()(inputs)
        combined = keras.layers.Concatenate(axis=-1)([inputs, coords])
        # combined.shape: (batch, 32, 32, 130)

        # In a complete model
        inputs = keras.Input(shape=(224, 224, 3))
        x = keras.layers.Conv2D(64, 3, activation='relu')(inputs)
        coords = SpatialLayer(resolution=(8, 8))(x)
        features_with_coords = keras.layers.Concatenate()([x, coords])
        outputs = keras.layers.Conv2D(32, 1)(features_with_coords)
        model = keras.Model(inputs, outputs)
        ```

    Note:
        The layer is non-trainable by design as it provides deterministic coordinate
        information. The coordinate grid is standardized to match the scale of typical
        neural network activations, promoting training stability when concatenated with
        learned features.

    References:
        - CoordConv: An intriguing failing of convolutional neural networks and the CoordConv solution
        - Used in various generative models and spatial attention mechanisms
    """

    def __init__(
        self,
        resolution: Tuple[int, int] = (4, 4),
        resize_method: Literal['nearest', 'bilinear'] = 'nearest',
        **kwargs: Any
    ) -> None:
        # Force non-trainable since this is a deterministic coordinate generator
        kwargs['trainable'] = False
        super().__init__(**kwargs)

        # Validate inputs
        if len(resolution) != 2:
            raise ValueError(f"resolution must be a tuple of 2 integers, got {resolution}")
        if resolution[0] <= 0 or resolution[1] <= 0:
            raise ValueError(f"resolution values must be positive, got {resolution}")

        valid_methods = ['nearest', 'bilinear']
        if resize_method not in valid_methods:
            raise ValueError(f"resize_method must be one of {valid_methods}, got '{resize_method}'")

        # Store configuration
        self.resolution = resolution
        self.resize_method = resize_method

        # Coordinate grid attribute - created in build()
        self.xy_grid = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Creates the normalized coordinate grid during layer building.

        This method generates the base coordinate grid at the specified resolution,
        applies standardization, and prepares it for dynamic resizing during forward passes.

        Args:
            input_shape: Shape tuple of the input tensor. Must be 4D.

        Raises:
            ValueError: If input_shape is not 4D.
        """
        if len(input_shape) != 4:
            raise ValueError(f"SpatialLayer expects 4D input, got shape {input_shape}")

        # Create coordinate grids using keras.ops
        x_coords = ops.linspace(
            start=-0.5,
            stop=0.5,
            num=self.resolution[1]  # width
        )
        y_coords = ops.linspace(
            start=-0.5,
            stop=0.5,
            num=self.resolution[0]  # height
        )

        # Create meshgrid - note: meshgrid returns (Y, X) by default in TensorFlow style
        yy_grid, xx_grid = ops.meshgrid(y_coords, x_coords, indexing='ij')

        # Normalize the grids to have zero mean and unit standard deviation
        # This ensures compatibility with typical neural network activation scales
        epsilon = 1e-7  # Numerical stability

        xx_normalized = (xx_grid - ops.mean(xx_grid)) / (ops.std(xx_grid) + epsilon)
        yy_normalized = (yy_grid - ops.mean(yy_grid)) / (ops.std(yy_grid) + epsilon)

        # Stack x and y coordinates along last dimension
        # Shape: (resolution[0], resolution[1], 2)
        coordinate_grid = ops.stack([xx_normalized, yy_normalized], axis=-1)

        # Add batch dimension for later broadcasting
        # Shape: (1, resolution[0], resolution[1], 2)
        self.xy_grid = ops.expand_dims(coordinate_grid, axis=0)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        **kwargs: Any
    ) -> keras.KerasTensor:
        """
        Forward pass that dynamically resizes coordinate grid to match input dimensions.

        Args:
            inputs: Input tensor with shape (batch_size, height, width, channels).
            training: Boolean indicating training mode (unused, kept for interface consistency).
            **kwargs: Additional keyword arguments.

        Returns:
            Coordinate grid tensor with shape (batch_size, height, width, 2).
            Each location contains the (x, y) coordinates for that spatial position.
        """
        # Get input spatial dimensions
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        target_height = input_shape[1]
        target_width = input_shape[2]

        # Resize the coordinate grid to match input spatial dimensions
        # Use keras.ops.image.resize for backend-agnostic resizing
        xy_grid_resized = ops.image.resize(
            image=self.xy_grid,
            size=(target_height, target_width),
            interpolation=self.resize_method,
            data_format='channels_last'
        )

        # Tile the grid to match the batch size
        # ops.repeat repeats along specified axis
        xy_grid_batched = ops.repeat(
            xy_grid_resized,
            repeats=batch_size,
            axis=0
        )

        return xy_grid_batched

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Computes the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            Output shape tuple: (batch_size, height, width, 2)
        """
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {input_shape}")

        # Output preserves batch and spatial dimensions, but has 2 channels for (x, y)
        return (input_shape[0], input_shape[1], input_shape[2], 2)

    def get_config(self) -> dict:
        """
        Returns the configuration dictionary for serialization.

        Returns:
            Dictionary containing all constructor parameters needed for reconstruction.
        """
        config = super().get_config()
        config.update({
            'resolution': self.resolution,
            'resize_method': self.resize_method,
        })
        return config

# ---------------------------------------------------------------------
