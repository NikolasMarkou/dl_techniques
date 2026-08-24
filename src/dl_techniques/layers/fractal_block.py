"""
A recursive fractal block from the FractalNet architecture.

This layer constructs a deep, self-similar network structure by recursively
applying a simple expansion rule, providing an alternative to residual
connections for training ultra-deep networks. A FractalBlock of depth k
joins a DEEP branch that COMPOSES two depth-(k-1) FractalBlocks with a SHALLOW
branch that is a single base block applied to the same input --
F_1(x) = block(x), F_k(x) = join(F_{k-1}(F_{k-1}(x)), block(x)) -- under local
drop-path. Composition, not parallelism, is what makes the longest path
2^(k-1) base blocks long while the shortest stays 1. The base case F_1(x) is a
standard computational unit such as a ConvBlock.

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

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class FractalBlock(keras.layers.Layer):
    """
    Recursive fractal block implementing the fractal expansion rule for FractalNet.

    Implements the paper's expansion rule f_{C+1}(z) = [f_C(f_C(z))] join
    [conv(z)]: at depth k the DEEP branch COMPOSES two depth-(k-1)
    ``FractalBlock``s (the second consumes the first's OUTPUT) and the SHALLOW
    branch is a single base block applied to the same input; the two are joined
    under local drop-path. Do NOT "restore consistency" by rewriting this as two
    PARALLEL depth-(k-1) branches over the same input,
    ``F_k(x) = 0.5 * (DP(F_{k-1}(x)) + DP(F_{k-1}(x)))`` -- that was this
    layer's shipped defect until 2026-08-14, and because the recursion then
    terminates with every leaf receiving the block's own input, every path
    traversed exactly ONE convolution at any ``depth``. Parameter count, layer
    count and output shape are IDENTICAL under both rules, so only the
    RECEPTIVE FIELD detects it (pinned by
    ``tests/test_layers/test_fractal_block.py::TestFractalExpansionRule``).

    At depth 1 the block is a single base block. At depth k the structure holds
    ``2^k - 1`` leaf base blocks -- the recurrence is ``L(1) = 1``,
    ``L(k) = 2 * L(k-1) + 1``, counting both halves of the deep branch plus the
    shallow one, NOT ``2^(k-1)`` -- with a longest path of ``2^(k-1)`` blocks, a
    shortest path of 1, and an exponential number of distinct input-to-output
    paths forming an implicit ensemble of sub-networks. Uses configuration-based
    design with serializable dictionaries for full model save/load capability.

    **Architecture Overview:**

    .. code-block:: text

        ┌────────────────────────────────────────────────┐
        │  Input [batch, height, width, channels]        │
        └──────────────────┬─────────────────────────────┘
             ┌─────────────┴─────────────┐
             │ DEEP branch               │ SHALLOW branch
             ▼                           ▼
        ┌──────────────┐         ┌──────────────┐
        │  deep_first  │         │  shallow     │
        │  FractalBlock│         │  ONE base    │
        │  depth = k-1 │         │  block       │
        └──────┬───────┘         └──────┬───────┘
               ▼                        │
        ┌──────────────┐                │
        │  deep_second │  COMPOSED:     │
        │  FractalBlock│  consumes      │
        │  depth = k-1 │  deep_first's  │
        └──────┬───────┘  OUTPUT        │
               └─────────┬──────────────┘
                         ▼
        ┌────────────────────────────────────────────────┐
        │  Local drop-path join: mean over the SURVIVING │
        │  branches. If both draws drop, one is revived  │
        │  by a fair coin, so the join is never zero.    │
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
        base block is used; at depth k the structure has ``2^k - 1`` leaf base
        blocks (``L(1)=1``, ``L(k)=2*L(k-1)+1``) and a longest path of
        ``2^(k-1)`` blocks. Defaults to 1.
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

        # A composed deep path applies its base block 2^(depth-1) times, so a
        # stride > 1 inside the block would downsample the deep and shallow
        # branches by different factors and the join would receive mismatched
        # shapes. FractalNet downsamples BETWEEN blocks, not inside them; the
        # caller is responsible for that (see FractalNet._build_fractal_stage).
        block_stride = self.block_config.get("strides", 1)
        if block_stride not in (1, (1, 1), [1, 1]):
            raise ValueError(
                f"block_config['strides'] must be 1 inside a FractalBlock, got "
                f"{block_stride!r}. A fractal runs at constant resolution: the "
                f"deep branch applies the base block 2^(depth-1) times, so any "
                f"stride > 1 would downsample it 2^(depth-1) times against the "
                f"shallow branch's once. Downsample between blocks instead."
            )

        # CREATE all sub-layers in __init__ following modern Keras 3 pattern
        if self.depth == 1:
            # Base case: a single base block; the fractal bottoms out here.
            self.block = self._create_block_from_config()
            self.deep_first = None
            self.deep_second = None
            self.shallow = None
            logger.debug("Created FractalBlock base case with depth=1")
        else:
            # DECISION plan-2026-08-18T140459-7991552f/D-057
            # Paper's expansion rule, f_{C+1}(z) = [f_C(f_C(z))] join [conv(z)]:
            # the DEEP branch COMPOSES two depth-(C) fractals, and the SHALLOW
            # branch is a single base block applied to the same input. That
            # composition is what makes the longest path 2^(depth-1) blocks long
            # while the shortest stays 1, which is the entire point of the
            # architecture -- the short path is what trains, the long path is
            # what the short path teaches.
            #
            # Do NOT rewrite this as two PARALLEL depth-(C) branches over the
            # same input, F_k(x) = 0.5*(DP(F_{k-1}(x)) + DP(F_{k-1}(x))). That
            # form shipped until 2026-08-14 and made every path traverse exactly
            # ONE convolution at any depth, because the recursion then terminates
            # with every leaf receiving the block's own input. Parameter count,
            # layer count and output shape are IDENTICAL under both rules, so the
            # suite stayed green for months; only the RECEPTIVE FIELD sees it
            # (test_fractal_block.py::TestFractalExpansionRule). The docstrings
            # above stated the parallel rule until 2026-08-19, which is why this
            # note exists: a maintainer restoring doc/code consistency reverts
            # the repair.
            self.block = None
            self.deep_first = FractalBlock(
                block_config=self.block_config,
                depth=self.depth - 1,
                drop_path_rate=self.drop_path_rate,
                name="deep_first"
            )
            self.deep_second = FractalBlock(
                block_config=self.block_config,
                depth=self.depth - 1,
                drop_path_rate=self.drop_path_rate,
                name="deep_second"
            )
            self.shallow = self._create_block_from_config()
            logger.debug(f"Created FractalBlock recursive case with depth={self.depth}")

        # One generator per block so the join's draws are reproducible under a
        # seeded run and independent between blocks.
        self._seed_generator = keras.random.SeedGenerator()

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
            self.block.build(input_shape)
        else:
            # The deep branch is COMPOSED, so the second half is built on the
            # first half's OUTPUT shape, not on the block's input shape.
            self.deep_first.build(input_shape)
            intermediate_shape = self.deep_first.compute_output_shape(input_shape)
            self.deep_second.build(intermediate_shape)
            self.shallow.build(input_shape)

        logger.debug(f"Built FractalBlock with input_shape={input_shape}, depth={self.depth}")

        # Always call parent build at the end (MUST be last)
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the FractalBlock.

        Implements the fractal expansion rule recursively. For the base case
        (depth=1), applies the block function directly. For recursive cases,
        runs the COMPOSED deep branch (``deep_second(deep_first(x))``) and the
        single-block shallow branch on the same input, then mean-joins the
        surviving branches under local drop-path (see :meth:`_join`).

        :param inputs: Input tensor.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training mode (controls stochastic
            depth).
        :type training: Optional[bool]
        :return: Output tensor after fractal processing.
        :rtype: keras.KerasTensor
        """
        if self.depth == 1:
            return self.block(inputs, training=training)

        deep = self.deep_second(
            self.deep_first(inputs, training=training), training=training
        )
        shallow = self.shallow(inputs, training=training)
        return self._join(deep, shallow, training=training)

    def _join(
        self,
        deep: keras.KerasTensor,
        shallow: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Mean-join the two branches under local drop-path.

        At inference, or when ``drop_path_rate`` is zero, this is the plain mean
        of the two branches.

        During training each branch is dropped by its own per-sample Bernoulli
        draw and the join averages only the SURVIVORS, which is what makes the
        join a mean over a varying number of paths rather than a fixed one. The
        previous implementation instead scaled each branch by a fixed ``0.5``
        after an independent :class:`StochasticDepth`, so when both draws
        dropped -- which happens at rate ``drop_path_rate ** 2``, about 2.3% at
        the 0.15 default -- the block emitted EXACTLY ZERO and destroyed the
        signal for that sample. Here the both-dropped case is explicitly
        rescued: one branch is revived by a fair coin, so at least one path is
        always live and the block is never a zero map.

        :param deep: Output of the composed deep branch.
        :type deep: keras.KerasTensor
        :param shallow: Output of the single-block shallow branch.
        :type shallow: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: The joined tensor.
        :rtype: keras.KerasTensor
        """
        if training is False or self.drop_path_rate == 0.0:
            return keras.ops.multiply(keras.ops.add(deep, shallow), 0.5)

        batch_size = keras.ops.shape(deep)[0]
        draw_shape = [batch_size] + [1] * (len(deep.shape) - 1)
        keep_prob = 1.0 - self.drop_path_rate

        def _bernoulli(threshold: float) -> keras.KerasTensor:
            u = keras.random.uniform(
                draw_shape, dtype=deep.dtype, seed=self._seed_generator
            )
            return keras.ops.cast(u < threshold, deep.dtype)

        keep_deep = _bernoulli(keep_prob)
        keep_shallow = _bernoulli(keep_prob)

        # Rescue the both-dropped case with a fair coin rather than emitting 0.
        both_dropped = keras.ops.cast(
            keras.ops.add(keep_deep, keep_shallow) < 0.5, deep.dtype
        )
        coin = _bernoulli(0.5)
        keep_deep = keras.ops.add(keep_deep, keras.ops.multiply(both_dropped, coin))
        keep_shallow = keras.ops.add(
            keep_shallow, keras.ops.multiply(both_dropped, 1.0 - coin)
        )

        survivors = keras.ops.add(keep_deep, keep_shallow)
        summed = keras.ops.add(
            keras.ops.multiply(deep, keep_deep),
            keras.ops.multiply(shallow, keep_shallow),
        )
        return keras.ops.divide(summed, survivors)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape of the FractalBlock.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple after fractal processing.
        :rtype: Tuple[Optional[int], ...]
        """
        if self.depth == 1:
            return self.block.compute_output_shape(input_shape)
        else:
            return self.deep_second.compute_output_shape(
                self.deep_first.compute_output_shape(input_shape)
            )

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
