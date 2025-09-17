"""Implement the recursive fractal block from the FractalNet architecture.

This layer constructs a deep, self-similar network structure by recursively
applying a simple expansion rule, providing an alternative to residual
connections for training ultra-deep networks. The core principle is to create
a rich ensemble of diverse computational paths within a single, unified
architecture.

Architectural and Conceptual Underpinnings:

The FractalNet architecture is built upon a recursive design pattern. A
`FractalBlock` of depth `k` is defined as the composition of two parallel
`FractalBlock` sub-modules, each of depth `k-1`. The outputs of these two
parallel branches, which have shared architectural motifs but independent
parameters, are averaged to produce the final output. The base case for this
recursion, a block of depth `1`, is a standard computational unit, such as a
simple convolutional block.

This expansion rule results in a computational structure that resembles a
binary tree. A block of depth `k` contains `2^(k-1)` leaf nodes (base blocks)
and an exponential number of distinct paths from input to output. This design
implicitly trains an ensemble of sub-networks of varying depths, as any path
from the root to a leaf constitutes a valid, shallower network.

Foundational Mathematics and Regularization:

The recursive expansion is formally defined as:
    `F_k(x) = 0.5 * (path_1 + path_2)`

where each path is a regularized application of the block of the previous
depth, `F_{k-1}`. The key to training such a deep, redundant structure is the
regularization strategy known as "drop-path."

Drop-path is a form of stochastic depth where entire branches of the fractal
are randomly dropped during training. This forces the network to learn
meaningful representations without relying on any single computational path.
By randomly sampling sub-networks during each training step, drop-path ensures
that all paths, from the shallowest to the deepest, are trained to contribute
to the final task.

At inference time, all paths are active, and their outputs are averaged. This
process is analogous to averaging the predictions of an exponential ensemble
of networks that were trained jointly, which provides the robustness and strong
performance characteristic of the FractalNet architecture.

References:
    - Larsson, G., et al. (2017). FractalNet: Ultra-Deep Neural Networks
      without Residuals. *ICLR*.
"""

import keras
from typing import Tuple, Optional, Any, Dict, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from .convblock import ConvBlock
from .stochastic_depth import StochasticDepth

# ---------------------------------------------------------------------

BlockClass = Literal["ConvBlock"]

@keras.saving.register_keras_serializable()
class FractalBlock(keras.layers.Layer):
    """
    Recursive fractal block implementing the fractal expansion rule for FractalNet.

    This layer implements the recursive fractal expansion where each level creates
    two parallel paths through the same computational structure with different
    parameter instances. The fractal rule is: F_{k+1}(x) = 0.5 * (DP(F_k(x)) + DP(F_k(x)))
    where DP represents drop-path (stochastic depth) regularization.

    **Intent**: Create fractal network structures that provide multiple computational
    paths with shared architectural patterns but independent parameters. This design
    promotes robustness through path diversity and enables effective regularization
    via stochastic depth during training.

    **Architecture** (example with depth=3):
    ```
    Input(shape=[batch, height, width, channels])
           ↓
    ┌─────────────┬─────────────┐
    │   Branch₁   │   Branch₂   │ (Both are FractalBlocks of depth=2)
    │      ↓      │      ↓      │
    │  DropPath₁  │  DropPath₂  │ (Stochastic depth regularization)
    └─────────────┴─────────────┘
           ↓
    Mean Join: 0.5 * (Branch₁ + Branch₂)
           ↓
    Output(shape=[batch, new_height, new_width, new_channels])
    ```

    **Fractal Expansion Levels**:
    - **depth=1**: F₁(x) = BaseBlock(x) - Single block execution
    - **depth=2**: F₂(x) = 0.5 * (DP(BaseBlock(x)) + DP(BaseBlock(x))) - Two parallel base blocks
    - **depth=3**: F₃(x) = 0.5 * (DP(F₂(x)) + DP(F₂(x))) - Two parallel depth-2 fractals
    - **depth=k**: Fₖ(x) = 0.5 * (DP(Fₖ₋₁(x)) + DP(Fₖ₋₁(x))) - Recursive expansion

    **Drop-Path Regularization**: During training, each branch path can be randomly
    dropped with probability `drop_path_rate`, forcing the network to not rely on
    any single computational path and improving generalization.

    **Configuration-Based Design**: Uses serializable configuration dictionaries
    rather than callable factories, ensuring full model save/load capability while
    maintaining architectural flexibility.

    Args:
        block_config: Dictionary containing the configuration for the base block.
            This should be the output of `get_config()` from a Keras layer
            (typically a ConvBlock). Must contain all parameters needed to
            reconstruct the base block layer.
        block_class: String name of the block class to instantiate. Currently
            supports "ConvBlock". Used with block_config to create base blocks.
            Defaults to "ConvBlock".
        depth: Integer, depth of fractal expansion. Must be >= 1. Controls the
            number of recursive levels in the fractal structure:
            - 1: Single base block (no fractal expansion)
            - 2: Two parallel base blocks with mean join
            - k: Recursive structure with 2^(k-1) base blocks at the leaves
            Defaults to 1.
        drop_path_rate: Float between 0.0 and 1.0, probability of dropping each
            path during training for stochastic depth regularization. Higher values
            increase regularization strength but may hurt performance if too high.
            Set to 0.0 to disable stochastic depth. Defaults to 0.15.
        **kwargs: Additional keyword arguments passed to the base Layer class,
            such as name, trainable, dtype.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`.
        The specific input requirements depend on the base block configuration.

    Output shape:
        4D tensor with shape determined by the base block's output transformation.
        The fractal structure preserves the shape transformations of the underlying
        base blocks while applying the fractal expansion rule.

    Attributes:
        block: Base block layer instance (only for depth=1).
        branch1: First recursive FractalBlock branch (for depth>1).
        branch2: Second recursive FractalBlock branch (for depth>1).
        drop_path1: StochasticDepth layer for first branch (for depth>1).
        drop_path2: StochasticDepth layer for second branch (for depth>1).

    Raises:
        ValueError: If depth is not a positive integer.
        ValueError: If drop_path_rate is not between 0.0 and 1.0.
        ValueError: If block_config is not a dictionary.
        ValueError: If block_class is not a supported block type.

    Example:
        ```python
        # Create base block configuration
        base_block = ConvBlock(filters=64, kernel_size=3)
        block_config = base_block.get_config()

        # Simple fractal (depth=1, just the base block)
        fractal1 = FractalBlock(
            block_config=block_config,
            depth=1
        )

        # Two-level fractal (depth=2, two parallel base blocks)
        fractal2 = FractalBlock(
            block_config=block_config,
            depth=2,
            drop_path_rate=0.1
        )

        # Deep fractal (depth=3, recursive structure)
        fractal3 = FractalBlock(
            block_config=block_config,
            depth=3,
            drop_path_rate=0.2
        )

        # Apply to input
        inputs = keras.Input(shape=(32, 32, 32))
        outputs = fractal3(inputs)
        print(f"Output shape: {outputs.shape}")  # Depends on base block config

        # In a complete model
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        ```

    Note:
        The fractal structure creates an exponential number of paths (2^(depth-1)
        base blocks at depth k), but uses parameter sharing within each level.
        This provides architectural diversity without proportional parameter growth.
        For very deep fractals, consider the computational and memory implications.

    References:
        - FractalNet: Ultra-Deep Neural Networks without Residuals
          (Larsson et al., 2017): https://arxiv.org/abs/1605.07648
    """

    def __init__(
        self,
        block_config: Dict[str, Any],
        block_class: BlockClass = "ConvBlock",
        depth: int = 1,
        drop_path_rate: float = 0.15,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if not isinstance(depth, int) or depth < 1:
            raise ValueError(f"depth must be a positive integer, got {depth}")

        if not 0.0 <= drop_path_rate <= 1.0:
            raise ValueError(f"drop_path_rate must be between 0.0 and 1.0, got {drop_path_rate}")

        if not isinstance(block_config, dict):
            raise ValueError(f"block_config must be a dictionary, got {type(block_config)}")

        if block_class not in ["ConvBlock"]:
            raise ValueError(f"Unsupported block_class: {block_class}")

        # Store configuration
        self.block_config = block_config
        self.block_class = block_class
        self.depth = depth
        self.drop_path_rate = drop_path_rate

        # CREATE all sub-layers in __init__ following modern Keras 3 pattern
        if self.depth == 1:
            # Base case: create single base block
            self.block = self._create_block_from_config()
            self.branch1 = None
            self.branch2 = None
            self.drop_path1 = None
            self.drop_path2 = None
            logger.debug(f"Created FractalBlock base case with depth=1")
        else:
            # Recursive case: create two branches with drop-path layers
            self.block = None
            self.branch1 = FractalBlock(
                block_config=self.block_config,
                block_class=self.block_class,
                depth=self.depth - 1,
                drop_path_rate=self.drop_path_rate,
                name="branch1"
            )
            self.branch2 = FractalBlock(
                block_config=self.block_config,
                block_class=self.block_class,
                depth=self.depth - 1,
                drop_path_rate=self.drop_path_rate,
                name="branch2"
            )

            # Create stochastic depth layers
            self.drop_path1 = StochasticDepth(
                drop_path_rate=self.drop_path_rate,
                name="drop_path1"
            )
            self.drop_path2 = StochasticDepth(
                drop_path_rate=self.drop_path_rate,
                name="drop_path2"
            )
            logger.debug(f"Created FractalBlock recursive case with depth={self.depth}")

    def _create_block_from_config(self) -> keras.layers.Layer:
        """
        Create a block instance from the stored configuration.

        Returns:
            A new block instance configured according to block_config.

        Raises:
            ValueError: If block_class is not supported.
        """
        if self.block_class == "ConvBlock":
            return ConvBlock.from_config(self.block_config)
        else:
            raise ValueError(f"Unsupported block_class: {self.block_class}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the FractalBlock and all its sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization.
        This ensures all weight variables exist before weight restoration during
        model loading.

        Args:
            input_shape: Shape tuple of the input tensor, including batch dimension.
        """
        if self.depth == 1:
            # Base case: build single block
            if self.block is not None:
                self.block.build(input_shape)
        else:
            # Recursive case: build all sub-layers
            if self.branch1 is not None:
                self.branch1.build(input_shape)
            if self.branch2 is not None:
                self.branch2.build(input_shape)

            # Determine output shape for drop-path layers
            # Both branches should have the same output shape
            if self.branch1 is not None:
                branch_output_shape = self.branch1.compute_output_shape(input_shape)
            else:
                branch_output_shape = input_shape

            if self.drop_path1 is not None:
                self.drop_path1.build(branch_output_shape)
            if self.drop_path2 is not None:
                self.drop_path2.build(branch_output_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the FractalBlock.

        Implements the fractal expansion rule recursively. For the base case
        (depth=1), applies the block function directly. For recursive cases,
        combines two branches using mean join after applying stochastic depth.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels).
            training: Boolean indicating whether the layer should behave in
                training mode (applying stochastic depth) or inference mode.

        Returns:
            Output tensor after fractal processing. Shape depends on the
            base block configuration's transformations.
        """
        if self.depth == 1:
            # Base case: apply base block
            return self.block(inputs, training=training)
        else:
            # Recursive case: combine two branches with drop-path
            y1 = self.branch1(inputs, training=training)
            y2 = self.branch2(inputs, training=training)

            # Apply stochastic depth (drop-path)
            y1 = self.drop_path1(y1, training=training)
            y2 = self.drop_path2(y2, training=training)

            # Mean join: average the paths
            # This encourages both paths to contribute meaningfully
            return keras.ops.add(keras.ops.multiply(y1, 0.5), keras.ops.multiply(y2, 0.5))

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape of the FractalBlock.

        The output shape is determined by the base block's transformation,
        as the fractal structure preserves the shape transformations.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple after fractal processing.
        """
        if self.depth == 1:
            if self.block is not None:
                return self.block.compute_output_shape(input_shape)
        else:
            if self.branch1 is not None:
                return self.branch1.compute_output_shape(input_shape)

        # Fallback: assume shape is preserved
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration. Fully serializable
            for configuration-based approach.
        """
        config = super().get_config()
        config.update({
            "block_config": self.block_config,
            "block_class": self.block_class,
            "depth": self.depth,
            "drop_path_rate": self.drop_path_rate,
        })
        return config

# ---------------------------------------------------------------------
