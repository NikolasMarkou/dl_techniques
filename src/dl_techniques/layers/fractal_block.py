"""
Recursive fractal block from the FractalNet architecture.

``FractalBlock`` builds a deep, self-similar structure by recursive expansion,
as an alternative to residual connections for training very deep networks:

    F_1(x) = block(x);  F_k(x) = join(F_{k-1}(F_{k-1}(x)), block(x))

The deep branch composes two depth-(k-1) blocks, feeding the second the first's
output, while the shallow branch applies one base block to the same input.
Composition, rather than two parallel branches, is what makes the longest path
``2^(k-1)`` base blocks long while the shortest stays 1. The join averages the
branches that survive a per-sample drop-path draw.

The base block is built with ``ConvBlock.from_config``, so ``block_config`` must
be a ``ConvBlock`` configuration. Its ``strides`` entry must be 1: a fractal
runs at constant resolution, and downsampling belongs between blocks rather than
inside one.

References:
    - Larsson et al., 2017. FractalNet: Ultra-Deep Neural Networks without
      Residuals. (ICLR)
"""

import keras
from typing import Tuple, Optional, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .conv_blocks.conv_block import ConvBlock
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.fractal_block")
class FractalBlock(keras.layers.Layer):
    """Expand a base block into a fractal of composed and shallow paths.

    At depth 1 the block is a single base block. At depth k it holds two
    depth-(k-1) ``FractalBlock``s in series plus one base block in parallel with
    them, joined under local drop-path. Every sub-block runs at the input
    resolution, so the two branches always meet at the same shape.

    Architecture (depth k > 1):

    .. code-block:: text

                           input [B, H, W, C]
                                    │
                      ┌─────────────┴─────────────┐
                      ▼                           ▼
            ┌───────────────────┐       ┌───────────────────┐
            │ deep_first        │       │ shallow           │
            │  FractalBlock k-1 │       │  one base block   │
            └─────────┬─────────┘       └─────────┬─────────┘
                      ▼                           │
            ┌───────────────────┐                 │
            │ deep_second       │                 │
            │  FractalBlock k-1 │                 │
            └─────────┬─────────┘                 │
                      └─────────────┬─────────────┘
                                    ▼
                      ┌───────────────────────────┐
                      │ drop-path join            │
                      │  mean over survivors      │
                      └─────────────┬─────────────┘
                                    ▼
                                 output

    At depth 1 the whole picture collapses to one base block, with no join.

    Drop-path join:

    .. code-block:: text

                     deep        shallow
                       │             │
                       ▼             ▼
                ┌──────────────────────────┐
                │ per-sample Bernoulli(p)  │
                │  on each branch          │
                └────────────┬─────────────┘
                             ▼
                ┌──────────────────────────┐
                │ if both dropped, revive  │
                │  one by a fair coin      │
                └────────────┬─────────────┘
                             ▼
                ┌──────────────────────────┐
                │ sum kept / number kept   │
                └────────────┬─────────────┘
                             ▼
                          joined

    Join outcomes, per sample:

    .. code-block:: text

        keep_deep   keep_shallow   result
        ─────────   ────────────   ────────────────────
        1           1              (deep + shallow) / 2
        1           0              deep
        0           1              shallow
        0           0              a fair coin picks one

    Size by depth:

    .. code-block:: text

        depth   leaf base blocks   longest path   shortest path
        ─────   ────────────────   ────────────   ─────────────
        1       1                  1              1
        2       3                  2              1
        3       7                  4              1
        4       15                 8              1
        k       2^k - 1            2^(k-1)        1

    :param block_config: Configuration for the base block, as returned by
        ``ConvBlock.get_config()``. It is rebuilt with
        ``ConvBlock.from_config``, so it must describe a ``ConvBlock``, and its
        ``strides`` entry must be 1.
    :type block_config: Dict[str, Any]
    :param depth: Depth of fractal expansion. Must be an integer of at least 1.
        Defaults to 1.
    :type depth: int
    :param drop_path_rate: Probability of dropping each branch at the join
        during training. Defaults to 0.15.
    :type drop_path_rate: float
    :param kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        Whatever the base block accepts, typically
        ``(batch, height, width, channels)``.

    Output shape:
        The base block's output shape for that input; unchanged spatially,
        since every stride is 1.

    :raises ValueError: If ``depth`` is not an integer of at least 1, if
        ``drop_path_rate`` falls outside ``[0, 1]``, if ``block_config`` is not
        a dict, or if its ``strides`` entry is anything other than 1.
    """

    def __init__(
        self,
        block_config: Dict[str, Any],
        depth: int = 1,
        drop_path_rate: float = 0.15,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(depth, int) or depth < 1:
            raise ValueError(f"depth must be a positive integer, got {depth}")

        if not 0.0 <= drop_path_rate <= 1.0:
            raise ValueError(f"drop_path_rate must be between 0.0 and 1.0, got {drop_path_rate}")

        if not isinstance(block_config, dict):
            raise ValueError(f"block_config must be a dictionary, got {type(block_config)}")


        self.block_config = block_config
        self.depth = depth
        self.drop_path_rate = drop_path_rate

        # The deep branch applies its base block 2^(depth-1) times, so a stride
        # above 1 would downsample the two branches by different factors.
        block_stride = self.block_config.get("strides", 1)
        if block_stride not in (1, (1, 1), [1, 1]):
            raise ValueError(
                f"block_config['strides'] must be 1 inside a FractalBlock, got "
                f"{block_stride!r}. A fractal runs at constant resolution: the "
                f"deep branch applies the base block 2^(depth-1) times, so any "
                f"stride > 1 would downsample it 2^(depth-1) times against the "
                f"shallow branch's once. Downsample between blocks instead."
            )

        if self.depth == 1:
            self.block = self._create_block_from_config()
            self.deep_first = None
            self.deep_second = None
            self.shallow = None
            logger.debug("Created FractalBlock base case with depth=1")
        else:
            # DECISION D-057: the deep branch composes two depth-(k-1) fractals;
            # two parallel branches collapse every path to length 1. decisions.md.
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

        # One generator per block, so the join's draws are reproducible under a
        # seeded run and independent between blocks.
        self._seed_generator = keras.random.SeedGenerator()

    def _create_block_from_config(self) -> keras.layers.Layer:
        """Create a block instance from the stored configuration.

        :return: A new block instance configured according to block_config.
        :rtype: keras.layers.Layer
        """
        return ConvBlock.from_config(self.block_config)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build every sub-layer, so weights exist before a checkpoint loads.

        :param input_shape: Shape tuple of the input tensor, including batch
            dimension.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.depth == 1:
            self.block.build(input_shape)
        else:
            # The deep branch is composed, so the second half builds on the
            # first half's output shape rather than on the block's input shape.
            self.deep_first.build(input_shape)
            intermediate_shape = self.deep_first.compute_output_shape(input_shape)
            self.deep_second.build(intermediate_shape)
            self.shallow.build(input_shape)

        logger.debug(f"Built FractalBlock with input_shape={input_shape}, depth={self.depth}")

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply the fractal expansion to the input.

        At depth 1 this is the base block. At greater depth it runs the composed
        deep branch ``deep_second(deep_first(x))`` and the single-block shallow
        branch on the same input, then joins them with :meth:`_join`.

        :param inputs: Input tensor.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training mode, which controls the
            drop-path draw at the join.
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

        The plain mean of both branches is returned when ``training`` is
        exactly ``False`` or ``drop_path_rate`` is zero. Any other value of
        ``training``, including the default ``None``, takes the stochastic
        path: each branch gets its own per-sample Bernoulli draw and the mean
        covers only the survivors. Both branches drop together at rate
        ``drop_path_rate ** 2``, about 2.3% at the 0.15 default, and that case
        revives one branch with a fair coin, so the block never emits a zero
        map.

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
        """Compute the output shape by threading it through the deep branch.

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

        # Unreachable: both arms above return.
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