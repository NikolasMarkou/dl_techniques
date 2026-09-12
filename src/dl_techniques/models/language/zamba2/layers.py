"""Zamba2-specific building blocks.

This module holds the layers that are genuinely new for Zamba2 -- the ones
that have no reusable precedent elsewhere in the repository -- plus the thin
per-block wrappers that compose Zamba2's decoder stack out of existing
primitives (``Mamba2ResidualBlock``, the attention factory, ``RMSNorm``,
``RotaryPositionEmbedding``). Classes are added incrementally; see
``plans/plan-2026-09-12T075714-035fd488/plan.md`` Steps 1-4 for the build
order. Only :class:`LoRAAdapter` exists so far.

References:
    - Glorioso, P. et al., 2024. Zamba2: A Compact and Fast Hybrid Model.
      (https://arxiv.org/abs/2411.15242)
    - Hu, E. J. et al., 2021. LoRA: Low-Rank Adaptation of Large Language
      Models. (https://arxiv.org/abs/2106.09685)
"""

from typing import Any, Dict, Optional, Tuple, Union

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.zamba2.lora_adapter")
class LoRAAdapter(keras.layers.Layer):
    """
    Additive low-rank adapter with one independent ``A``/``B`` pair per occurrence.

    Computes a pure additive delta ``(x @ A[i]) @ B[i] * (alpha / rank)`` for a
    caller-selected occurrence index ``i``. It does **not** own or wrap the
    base projection it augments -- the caller (a shared block invoked at
    several depths) is responsible for adding this delta onto its own base
    ``Dense`` output. This is the mechanism Zamba2 uses to let ``num_mem_blocks``
    physical shared blocks, reused across many more depth positions, specialize
    per depth without multiplying the shared block's own parameter count: every
    depth position gets its own LoRA pair even though several depth positions
    route through the same physical block.

    Architecture:

    .. code-block:: text

        x  [..., input_dim]
              │
              ▼  select occurrence i (a plain Python int, fixed per call site)
        A[i]: (input_dim, rank)          -- unique per occurrence
              │
              ▼  [..., rank]
        B[i]: (rank, output_dim)         -- unique per occurrence, zero-init
              │
              ▼  [..., output_dim]
        * (alpha / rank)
              │
              ▼
        delta  [..., output_dim]   (added by the CALLER onto its base output)

        A is one weight tensor of shape (num_occurrences, input_dim, rank);
        B is one weight tensor of shape (num_occurrences, rank, output_dim).
        Each occurrence's (input_dim, rank) / (rank, output_dim) slice is
        initialized independently -- see the Note on initialization below.

    :param output_dim: Width of the delta this adapter produces. Must match
        the base projection's output width the caller will add this onto.
        Must be positive.
    :type output_dim: int
    :param rank: Bottleneck width shared by every occurrence's ``A``/``B``
        pair. Must be positive.
    :type rank: int
    :param alpha: LoRA scaling numerator; the applied scale is
        ``alpha / rank``. Must be positive.
    :type alpha: float
    :param num_occurrences: Number of independent ``A``/``B`` pairs to
        allocate -- one per depth position that will select into this
        adapter via ``call(..., occurrence_idx=...)``, NOT one per physical
        shared block. Must be positive.
    :type num_occurrences: int
    :param kernel_initializer: Initializer applied independently to each
        occurrence's ``A`` slice. Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kwargs: Extra arguments for ``keras.layers.Layer`` (``name``,
        ``dtype``, and so on).
    :type kwargs: Any

    :ivar output_dim: The stored output width.
    :vartype output_dim: int
    :ivar rank: The stored bottleneck width.
    :vartype rank: int
    :ivar alpha: The stored scaling numerator.
    :vartype alpha: float
    :ivar num_occurrences: The stored occurrence count.
    :vartype num_occurrences: int
    :ivar scale: The resolved ``alpha / rank`` scale applied to every delta.
    :vartype scale: float
    :ivar kernel_initializer: The resolved initializer for ``A``.
    :vartype kernel_initializer: keras.initializers.Initializer
    :ivar a: Weight of shape ``(num_occurrences, input_dim, rank)``.
    :vartype a: keras.Variable
    :ivar b: Weight of shape ``(num_occurrences, rank, output_dim)``.
    :vartype b: keras.Variable

    :raises ValueError: If ``output_dim``, ``rank``, ``alpha``, or
        ``num_occurrences`` is not positive.
    :raises ValueError: If ``call()`` is given an ``occurrence_idx`` outside
        ``[0, num_occurrences)``.

    Input shape:
        Tensor of rank >= 2, shape ``(..., input_dim)``.

    Output shape:
        Same rank and leading axes as the input, with the last axis set to
        ``output_dim``.

    Example:
        .. code-block:: python

            adapter = LoRAAdapter(output_dim=256, rank=8, alpha=16.0, num_occurrences=4)
            x = keras.random.normal((2, 10, 256))
            delta_0 = adapter(x, occurrence_idx=0)
            delta_1 = adapter(x, occurrence_idx=1)
            # delta_0 != delta_1: each occurrence has its own A/B pair.

    Note:
        ``B`` is initialized to zeros, the standard LoRA convention: every
        occurrence's delta is exactly zero at construction, so attaching this
        adapter to an already-trained shared block does not perturb its
        output until training updates ``B`` away from zero. ``A`` is
        initialized by applying ``kernel_initializer`` independently to each
        occurrence's ``(input_dim, rank)`` slice -- NOT by initializing the
        full stacked ``(num_occurrences, input_dim, rank)`` tensor in one
        call. Keras' built-in initializers compute fan-in/fan-out from a
        3-D shape as if it were a convolution kernel's
        ``(receptive_field, fan_in, fan_out)``, which would read
        ``num_occurrences`` as a spatial extent and scale every slice
        incorrectly. Initializing slice-by-slice instead makes the stacked
        tensor equivalent to ``num_occurrences`` independent 2-D weights.
    """

    def __init__(
        self,
        output_dim: int,
        rank: int,
        alpha: float,
        num_occurrences: int,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        **kwargs: Any,
    ) -> None:
        """Validate the configuration and resolve the LoRA scale.

        No weights are created here -- ``A``/``B`` need the input width,
        which is only known in ``build()``.

        :raises ValueError: If ``output_dim``, ``rank``, ``alpha``, or
            ``num_occurrences`` is not positive.
        """
        super().__init__(**kwargs)

        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if num_occurrences <= 0:
            raise ValueError(f"num_occurrences must be positive, got {num_occurrences}")

        self.output_dim = output_dim
        self.rank = rank
        self.alpha = alpha
        self.num_occurrences = num_occurrences
        self.scale = alpha / rank
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

        self.a: Optional[keras.Variable] = None
        self.b: Optional[keras.Variable] = None

        logger.info(
            f"Initialized LoRAAdapter with output_dim={output_dim}, rank={rank}, "
            f"alpha={alpha}, num_occurrences={num_occurrences}, scale={self.scale}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the per-occurrence ``A``/``B`` weight tensors.

        :param input_shape: Shape tuple of the input tensor; only the last
            axis (``input_dim``) is used.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input's last axis is not statically known.
        """
        if self.built:
            return

        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError(
                "LoRAAdapter requires a statically-known input feature "
                f"dimension; got input_shape={input_shape}"
            )

        num_occurrences = self.num_occurrences
        rank = self.rank
        output_dim = self.output_dim
        a_initializer_fn = self.kernel_initializer

        def _a_initializer(shape: Tuple[int, int, int], dtype: Any = None) -> Any:
            # Apply the 2-D initializer independently to every occurrence's
            # own (input_dim, rank) slice -- see the class Note on why the
            # stacked 3-D shape must not be handed to the initializer as-is.
            slices = [
                a_initializer_fn(shape=(shape[1], shape[2]), dtype=dtype)
                for _ in range(shape[0])
            ]
            return keras.ops.stack(slices, axis=0)

        self.a = self.add_weight(
            name="lora_a",
            shape=(num_occurrences, input_dim, rank),
            initializer=_a_initializer,
            trainable=True,
        )
        self.b = self.add_weight(
            name="lora_b",
            shape=(num_occurrences, rank, output_dim),
            initializer="zeros",
            trainable=True,
        )

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        occurrence_idx: int,
    ) -> keras.KerasTensor:
        """
        Compute the additive LoRA delta for one occurrence.

        :param inputs: Input tensor of shape ``(..., input_dim)``.
        :type inputs: keras.KerasTensor
        :param occurrence_idx: Which occurrence's ``A``/``B`` pair to use. A
            plain Python ``int`` -- the depth position an occurrence
            corresponds to is static (fixed by ``layer_mapping`` at model
            construction time), never a data-dependent tensor value.
        :type occurrence_idx: int
        :return: Delta tensor of shape ``(..., output_dim)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If ``occurrence_idx`` is outside
            ``[0, num_occurrences)``.
        """
        if not (0 <= occurrence_idx < self.num_occurrences):
            raise ValueError(
                f"occurrence_idx must be in [0, {self.num_occurrences}), "
                f"got {occurrence_idx}"
            )

        a_i = self.a[occurrence_idx]
        b_i = self.b[occurrence_idx]

        delta = keras.ops.matmul(inputs, a_i)
        delta = keras.ops.matmul(delta, b_i)
        return delta * self.scale

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple with the last dimension set to
            ``output_dim``.
        :rtype: Tuple[Optional[int], ...]
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        :return: Dictionary containing every constructor argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "rank": self.rank,
                "alpha": self.alpha,
                "num_occurrences": self.num_occurrences,
                "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            }
        )
        return config

# ---------------------------------------------------------------------
