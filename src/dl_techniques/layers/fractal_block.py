"""
FractalBlock layer implementation for FractalNet architecture.

This module provides the recursive fractal block that implements the fractal
expansion rule: F_{k+1}(x) = 0.5 * (DP(F_k(x)) + DP(F_k(x)))
where DP is drop-path (stochastic depth) and F_1(x) = B(x) is the base block.
"""

import keras
from typing import Union, Tuple, Optional, Any, Dict, Callable

from dl_techniques.utils.logger import logger

from .convblock import ConvBlock
from .stochastic_depth import StochasticDepth

@keras.saving.register_keras_serializable()
class FractalBlock(keras.layers.Layer):
    """Recursive fractal block implementing the fractal expansion rule.

    Implements the recursive expansion: F_{k+1}(x) = 0.5 * (DP(F_k(x)) + DP(F_k(x)))
    where DP is drop-path (stochastic depth) and F_1(x) = B(x) is the base block.

    The fractal structure creates multiple paths through the network, with each path
    having the same computational structure but different parameter instances. During
    training, stochastic depth randomly drops some paths, forcing the network to
    learn robust representations that don't depend on any single path.

    This implementation uses a configuration-based approach for full serialization
    support, allowing models to be saved and loaded seamlessly.

    Args:
        block_config: Dictionary containing the configuration for the base block.
            This should be the output of `get_config()` from a Keras layer
            (typically a ConvBlock). Alternatively, can be a callable that returns
            a layer instance for backward compatibility.
        block_class: String name or class reference for the block type. Defaults to
            "ConvBlock". Only used when block_config is a dictionary.
        depth: Integer, depth of fractal expansion. Must be >= 1.
            - depth=1: Just the base block (F_1(x) = B(x))
            - depth=2: F_2(x) = 0.5 * (DP(B(x)) + DP(B(x)))
            - depth=3: F_3(x) = 0.5 * (DP(F_2(x)) + DP(F_2(x)))
            - etc.
        drop_path_rate: Float between 0 and 1, probability of dropping paths during
            training. Higher values increase regularization but may hurt performance.
            Defaults to 0.15.
        **kwargs: Additional keyword arguments passed to the base Layer class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`

    Output shape:
        4D tensor with shape determined by the base block configuration. The fractal
        structure preserves the shape transformations of the underlying blocks.

    Example:
        >>> # Configuration-based approach (fully serializable)
        >>> from dl_techniques.layers.convblock import ConvBlock
        >>> base_block = ConvBlock(filters=64, kernel_size=3)
        >>> block_config = base_block.get_config()
        >>> fractal = FractalBlock(
        ...     block_config=block_config,
        ...     block_class="ConvBlock",
        ...     depth=3,
        ...     drop_path_rate=0.2
        ... )
        >>> x = keras.random.normal((2, 32, 32, 32))
        >>> y = fractal(x)
        >>> print(y.shape)
        (2, 32, 32, 64)

        >>> # Callable approach (backward compatibility, not serializable)
        >>> def make_block():
        ...     return ConvBlock(filters=64, kernel_size=3)
        >>> fractal = FractalBlock(block_config=make_block, depth=3)
        >>> # This works but model.save() will have limitations
    """

    def __init__(
        self,
        block_config: Union[Dict[str, Any], Callable[[], keras.layers.Layer]],
        block_class: Union[str, type] = "ConvBlock",
        depth: int = 1,
        drop_path_rate: float = 0.15,
        **kwargs: Any
    ) -> None:
        """Initialize the FractalBlock.

        Validates inputs and stores configuration. The fractal structure is
        built recursively in the build() method.

        Raises:
            ValueError: If depth is not a positive integer.
            ValueError: If drop_path_rate is not between 0 and 1.
            TypeError: If block_config is neither dict nor callable.
        """
        super().__init__(**kwargs)

        # Validate inputs
        if not isinstance(depth, int) or depth < 1:
            raise ValueError(f"depth must be a positive integer, got {depth}")

        if not 0.0 <= drop_path_rate <= 1.0:
            raise ValueError(f"drop_path_rate must be between 0 and 1, got {drop_path_rate}")

        if not (isinstance(block_config, dict) or callable(block_config)):
            raise TypeError(
                f"block_config must be a dictionary or callable, got {type(block_config)}"
            )

        # Store configuration
        self.block_config = block_config
        self.block_class = block_class
        self.depth = depth
        self.drop_path_rate = drop_path_rate

        # Determine if using legacy callable approach
        self._is_callable_config = callable(block_config)
        if self._is_callable_config:
            logger.warning(
                "Using callable block_config. This provides flexibility but "
                "prevents full model serialization. Consider using configuration "
                "dictionaries for production deployment."
            )

        # Initialize layer attributes - will be set in build()
        self.block = None
        self.branch1 = None
        self.branch2 = None
        self.drop_path1 = None
        self.drop_path2 = None

        # Store build input shape for serialization
        self._build_input_shape = None

        logger.debug(f"Initialized FractalBlock with depth={depth}, "
                    f"drop_path_rate={drop_path_rate}, "
                    f"callable_config={self._is_callable_config}")

    def _create_block_from_config(self) -> keras.layers.Layer:
        """Create a block instance from the stored configuration.

        Returns:
            A new block instance configured according to block_config.
        """
        if self._is_callable_config:
            # Legacy callable approach
            return self.block_config()
        else:
            # Configuration-based approach
            if isinstance(self.block_class, str):
                # Import the class dynamically
                if self.block_class == "ConvBlock":
                    from dl_techniques.layers.convblock import ConvBlock
                    block_cls = ConvBlock
                else:
                    # For other classes, try to get from keras.layers first
                    block_cls = getattr(keras.layers, self.block_class, None)
                    if block_cls is None:
                        raise ValueError(f"Unknown block class: {self.block_class}")
            else:
                block_cls = self.block_class

            return block_cls.from_config(self.block_config)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the FractalBlock structure recursively.

        For depth=1, creates a single base block. For depth>1, creates two
        recursive branches with stochastic depth layers for regularization.

        Args:
            input_shape: Shape tuple of the input tensor, including batch dimension.
        """
        # Store input shape for serialization
        self._build_input_shape = input_shape

        if self.depth == 1:
            # Base case: just the configured block
            self.block = self._create_block_from_config()
            self.block.build(input_shape)
            logger.debug(f"Built FractalBlock base case with depth=1")
        else:
            # Recursive case: two branches with drop-path
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

            # Build branches
            self.branch1.build(input_shape)
            self.branch2.build(input_shape)

            # Drop-path layers for stochastic depth
            self.drop_path1 = StochasticDepth(
                drop_rate=self.drop_path_rate,
                name="drop_path1"
            )
            self.drop_path2 = StochasticDepth(
                drop_rate=self.drop_path_rate,
                name="drop_path2"
            )

            logger.debug(f"Built FractalBlock recursive case with depth={self.depth}")

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the FractalBlock.

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
            # Base case: apply configured block
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
            return 0.5 * (y1 + y2)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration. Fully serializable
            when using configuration-based approach.
        """
        config = super().get_config()

        if self._is_callable_config:
            logger.warning(
                "FractalBlock.get_config() cannot serialize callable block_config. "
                "Model save/load will fail. Use configuration-based approach instead."
            )
            # Store placeholder that will cause clear error on deserialization
            serializable_config = None
        else:
            serializable_config = self.block_config

        config.update({
            "block_config": serializable_config,
            "block_class": self.block_class if isinstance(self.block_class, str) else self.block_class.__name__,
            "depth": self.depth,
            "drop_path_rate": self.drop_path_rate,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build layer from configuration.

        Args:
            config: Build configuration dictionary containing input_shape.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FractalBlock":
        """Create layer from configuration.

        Args:
            config: Configuration dictionary from get_config().

        Returns:
            FractalBlock instance.

        Raises:
            ValueError: If block_config is None (indicating callable was used).
        """
        if config.get("block_config") is None:
            raise ValueError(
                "Cannot deserialize FractalBlock that was created with callable "
                "block_config. Use configuration-based approach for full "
                "serialization support."
            )

        return cls(**config)

    @staticmethod
    def create_from_block_instance(
        block: keras.layers.Layer,
        depth: int,
        drop_path_rate: float = 0.15,
        **kwargs: Any
    ) -> "FractalBlock":
        """Convenience method to create FractalBlock from a block instance.

        This automatically extracts the configuration and class information
        from an existing layer instance, ensuring full serialization support.

        Args:
            block: A Keras layer instance to use as the base block.
            depth: Fractal expansion depth.
            drop_path_rate: Drop-path probability.
            **kwargs: Additional arguments for FractalBlock.

        Returns:
            FractalBlock instance with configuration extracted from the block.

        Example:
            >>> base_block = ConvBlock(filters=64, kernel_size=3)
            >>> fractal = FractalBlock.create_from_block_instance(
            ...     block=base_block,
            ...     depth=3,
            ...     drop_path_rate=0.2
            ... )
        """
        block_config = block.get_config()
        block_class = block.__class__.__name__

        return cls(
            block_config=block_config,
            block_class=block_class,
            depth=depth,
            drop_path_rate=drop_path_rate,
            **kwargs
        )