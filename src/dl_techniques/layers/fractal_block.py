"""
A recursive fractal block from the FractalNet architecture.

This layer constructs a deep, self-similar network structure by recursively
applying a simple expansion rule, providing an alternative to residual
connections for training ultra-deep networks. A FractalBlock of depth k is
defined as the composition of two parallel FractalBlock sub-modules of depth
k-1, averaged together: F_k(x) = 0.5 * (DP(F_{k-1}(x)) + DP(F_{k-1}(x))),
where DP is drop-path stochastic depth regularization. The base case F_1(x)
is a standard computational unit such as a ConvBlock.

References:
    - Larsson, G., et al. (2017). FractalNet: Ultra-Deep Neural Networks
      without Residuals. *ICLR*.
"""

import keras
from typing import Tuple, Optional, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from .standard_blocks import ConvBlock
from .stochastic_depth import StochasticDepth

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class FractalBlock(keras.layers.Layer):
    """
    Recursive fractal block implementing the fractal expansion rule for FractalNet.

    Implements the recursive fractal expansion where each level creates two
    parallel paths through the same computational structure with different
    parameter instances. The fractal rule is F_{k+1}(x) = 0.5 * (DP(F_k(x)) +
    DP(F_k(x))) where DP represents drop-path (stochastic depth) regularization.
    At depth 1 the block is a single base block; at depth k it creates 2^(k-1)
    leaf nodes and an exponential number of distinct input-to-output paths,
    forming an implicit ensemble of sub-networks. Uses configuration-based
    design with serializable dictionaries for full model save/load capability.

    **Architecture Overview:**

    .. code-block:: text

        ┌────────────────────────────────────────────────┐
        │  Input [batch, height, width, channels]        │
        └──────────────────┬─────────────────────────────┘
                           ▼
             ┌─────────────┴─────────────┐
             ▼                           ▼
        ┌──────────────┐         ┌──────────────┐
        │  Branch 1    │         │  Branch 2    │
        │  FractalBlock│         │  FractalBlock│
        │  depth = k-1 │         │  depth = k-1 │
        └──────┬───────┘         └──────┬───────┘
               ▼                        ▼
        ┌──────────────┐         ┌──────────────┐
        │  DropPath 1  │         │  DropPath 2  │
        │  (stochastic)│         │  (stochastic)│
        └──────┬───────┘         └──────┬───────┘
               └─────────┬──────────────┘
                         ▼
        ┌────────────────────────────────────────────────┐
        │  Mean Join: 0.5 * (Branch_1 + Branch_2)        │
        └──────────────────┬─────────────────────────────┘
                           ▼
        ┌────────────────────────────────────────────────┐
        │  Output                                        │
        └────────────────────────────────────────────────┘

    :param block_config: Dictionary containing the configuration for the base
        block. Should be the output of ``get_config()`` from a Keras layer
        (typically a ConvBlock).
    :type block_config: Dict[str, Any]
    :param depth: Depth of fractal expansion. Must be >= 1. At depth 1 a single
        base block is used; at depth k the structure has 2^(k-1) leaf nodes.
        Defaults to 1.
    :type depth: int
    :param drop_path_rate: Probability of dropping each path during training
        for stochastic depth regularization. Defaults to 0.15.
    :type drop_path_rate: float
    :param kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
        self,
        block_config: Dict[str, Any],
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


        # Store configuration
        self.block_config = block_config
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
                depth=self.depth - 1,
                drop_path_rate=self.drop_path_rate,
                name="branch1"
            )
            self.branch2 = FractalBlock(
                block_config=self.block_config,
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
        """Create a block instance from the stored configuration.

        :return: A new block instance configured according to block_config.
        :rtype: keras.layers.Layer
        """
        return ConvBlock.from_config(self.block_config)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the FractalBlock and all its sub-layers.

        Explicitly builds each sub-layer for robust serialization, ensuring
        all weight variables exist before weight restoration during model loading.

        :param input_shape: Shape tuple of the input tensor, including batch
            dimension.
        :type input_shape: Tuple[Optional[int], ...]
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
        logger.debug(f"Built FractalBlock with input_shape={input_shape}, depth={self.depth}")

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the FractalBlock.

        Implements the fractal expansion rule recursively. For the base case
        (depth=1), applies the block function directly. For recursive cases,
        combines two branches using mean join after applying stochastic depth.

        :param inputs: Input tensor.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training mode (controls stochastic
            depth).
        :type training: Optional[bool]
        :return: Output tensor after fractal processing.
        :rtype: keras.KerasTensor
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
        """Compute output shape of the FractalBlock.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple after fractal processing.
        :rtype: Tuple[Optional[int], ...]
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
        """Get layer configuration for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "depth": self.depth,
            "block_config": self.block_config,
            "drop_path_rate": self.drop_path_rate,
        })
        return config

# ---------------------------------------------------------------------
