"""Low-rank adapters: an additive delta applied on top of a caller's own projection.

:class:`LoRAAdapter` was extracted verbatim from
``models/language/zamba2/layers.py`` where it was the one adapter primitive
Zamba2 needed. It is not Zamba2-specific -- the adapter axis means "several
independent low-rank slots at one call site", and the two current consumers
want that for unrelated reasons:

* **Zamba2** (``models/language/zamba2/``) uses the slot as a *depth position*.
  ``num_mem_blocks`` physical shared mem-blocks are reused round-robin across
  far more depth positions than there are blocks, and each position gets its own
  ``A``/``B`` pair so it can specialize without multiplying the shared block's
  parameter count.
* **Local Support Learning** (``layers/statistics/local_support_gate.py`` and
  the gated composition in :mod:`dl_techniques.layers.adapters.gated_adapter`)
  uses the slot as a *learning phase*. One adapter per phase, each gated by its
  own support estimator, so an update only reaches inputs drawn from the
  distribution that produced it.

Because the slot index is the only thing that differs between those, it is a
plain Python ``int`` on the ``call()`` signature and never a tensor: the slot is
fixed at model-construction time, so it must stay outside the traced graph.

The layer computes a **pure additive delta**. It does not own or wrap the base
projection it augments -- the caller adds the returned delta onto its own base
output. That is the seam that lets one adapter serve a ``Dense`` up-projection
inside a shared MLP block and a gated phase slot inside an LLM without either
call site knowing about the other.

References:
    - Hu, E. J. et al., 2021. LoRA: Low-Rank Adaptation of Large Language
      Models. (https://arxiv.org/abs/2106.09685)
    - Ben-Kish, A. et al., 2026. Local Support Learning.
      (https://arxiv.org/abs/2610.02126)
"""

from typing import Any, Dict, Optional, Tuple, Union

import keras

from dl_techniques.initializers import clone_initializer
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.logger import logger

#: Constructor key the ``num_adapters`` argument was renamed FROM. A config dict
#: written by a pre-extraction archive carries this instead, and
#: :meth:`LoRAAdapter.from_config` substitutes it (see that method for why the
#: substitution cannot live in ``__init__``).
LEGACY_NUM_OCCURRENCES_KEY: str = "num_occurrences"


@register_dl_technique(
    "dl_techniques.layers.adapters.lora",
    legacy_packages=("dl_techniques.models.zamba2.lora_adapter",),
)
class LoRAAdapter(keras.layers.Layer):
    """
    Additive low-rank adapter with one independent ``A``/``B`` pair per adapter slot.

    Computes a pure additive delta ``(x @ A[i]) @ B[i] * (alpha / rank)`` for a
    caller-selected slot index ``i``. It does **not** own or wrap the base
    projection it augments -- the caller is responsible for adding this delta
    onto its own base output. This is the mechanism that lets a shared block
    invoked at several sites (or one block serving several learning phases)
    specialize per site without multiplying the shared block's own parameter
    count: every site gets its own LoRA pair even though several sites route
    through the same physical layer.

    Architecture:

    .. code-block:: text

        x  [..., input_dim]
              │
              ▼  select slot i (a plain Python int, fixed per call site)
        A[i]: (input_dim, rank)          -- unique per slot
              │
              ▼  [..., rank]
        B[i]: (rank, output_dim)         -- unique per slot, zero-init
              │
              ▼  [..., output_dim]
        * (alpha / rank)
              │
              ▼
        delta  [..., output_dim]   (added by the CALLER onto its base output)

        A is one weight tensor of shape (num_adapters, input_dim, rank);
        B is one weight tensor of shape (num_adapters, rank, output_dim).
        Each slot's (input_dim, rank) / (rank, output_dim) slice is
        initialized independently -- see the Note on initialization below.

    :param output_dim: Width of the delta this adapter produces. Must match
        the base projection's output width the caller will add this onto.
        Must be positive.
    :type output_dim: int
    :param rank: Bottleneck width shared by every slot's ``A``/``B``
        pair. Must be positive.
    :type rank: int
    :param alpha: LoRA scaling numerator; the applied scale is
        ``alpha / rank``. Must be positive.
    :type alpha: float
    :param num_adapters: Number of independent ``A``/``B`` pairs to
        allocate -- one per call site that will select into this adapter via
        ``call(..., adapter_idx=...)``, NOT one per physical layer instance.
        Zamba2 reads this as the number of depth positions sharing a mem-block;
        Local Support Learning reads it as the number of learning phases.
        Must be positive.
    :type num_adapters: int
    :param kernel_initializer: Initializer applied independently to each
        slot's ``A`` slice. Defaults to 'glorot_uniform'.
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
    :ivar num_adapters: The stored slot count.
    :vartype num_adapters: int
    :ivar scale: The resolved ``alpha / rank`` scale applied to every delta.
    :vartype scale: float
    :ivar kernel_initializer: The resolved initializer for ``A``.
    :vartype kernel_initializer: keras.initializers.Initializer
    :ivar a: Weight of shape ``(num_adapters, input_dim, rank)``.
    :vartype a: keras.Variable
    :ivar b: Weight of shape ``(num_adapters, rank, output_dim)``.
    :vartype b: keras.Variable

    :raises ValueError: If ``output_dim``, ``rank``, ``alpha``, or
        ``num_adapters`` is not positive.
    :raises ValueError: If ``call()`` is given an ``adapter_idx`` outside
        ``[0, num_adapters)``.

    Input shape:
        Tensor of rank >= 2, shape ``(..., input_dim)``.

    Output shape:
        Same rank and leading axes as the input, with the last axis set to
        ``output_dim``.

    Example:
        .. code-block:: python

            adapter = LoRAAdapter(output_dim=256, rank=8, alpha=16.0, num_adapters=4)
            x = keras.random.normal((2, 10, 256))
            delta_0 = adapter(x, adapter_idx=0)
            delta_1 = adapter(x, adapter_idx=1)
            # delta_0 != delta_1: each slot has its own A/B pair.

    Note:
        ``B`` is initialized to zeros, the standard LoRA convention: every
        slot's delta is exactly zero at construction, so attaching this
        adapter to an already-trained shared block does not perturb its
        output until training updates ``B`` away from zero. ``A`` is
        initialized by applying ``kernel_initializer`` independently to each
        slot's ``(input_dim, rank)`` slice -- NOT by initializing the
        full stacked ``(num_adapters, input_dim, rank)`` tensor in one
        call. Keras' built-in initializers compute fan-in/fan-out from a
        3-D shape as if it were a convolution kernel's
        ``(receptive_field, fan_in, fan_out)``, which would read
        ``num_adapters`` as a spatial extent and scale every slice
        incorrectly. Each slice's initializer is also a fresh
        :func:`dl_techniques.initializers.clone.clone_initializer` clone of
        ``kernel_initializer``, not the same resolved instance reused across
        the loop -- a shared seedless Keras 3 initializer instance is
        stateless-deterministic and replays the identical sample at every
        call of the same shape, which previously made every slot's
        ``A`` slice bit-identical (measured: ``A[i] == A[j]`` for all
        ``i != j``; see ``plan-2026-09-12T075714-035fd488/D-007``).
        Initializing slice-by-slice with independent clones makes the stacked
        tensor equivalent to ``num_adapters`` genuinely independent 2-D
        weights.
    """

    def __init__(
        self,
        output_dim: int,
        rank: int,
        alpha: float,
        num_adapters: int,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        **kwargs: Any,
    ) -> None:
        """Validate the configuration and resolve the LoRA scale.

        No weights are created here -- ``A``/``B`` need the input width,
        which is only known in ``build()``.

        ``num_adapters`` is a required keyword. It deliberately has **no
        ``None`` default that means "accept the old spelling"**: a config
        carrying :data:`LEGACY_NUM_OCCURRENCES_KEY` is remapped by
        :meth:`from_config`, which is the only path that sees an archive's raw
        dict. Accepting the alias here as well would put two names for one
        value in ``get_config()``'s round trip.

        :raises ValueError: If ``output_dim``, ``rank``, ``alpha``, or
            ``num_adapters`` is not positive.
        """
        super().__init__(**kwargs)

        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if num_adapters <= 0:
            raise ValueError(f"num_adapters must be positive, got {num_adapters}")

        self.output_dim = output_dim
        self.rank = rank
        self.alpha = alpha
        self.num_adapters = num_adapters
        self.scale = alpha / rank
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

        self.a: Optional[keras.Variable] = None
        self.b: Optional[keras.Variable] = None

        logger.info(
            f"Initialized LoRAAdapter with output_dim={output_dim}, rank={rank}, "
            f"alpha={alpha}, num_adapters={num_adapters}, scale={self.scale}"
        )

    @property
    def num_occurrences(self) -> int:
        """Deprecated alias of :attr:`num_adapters`, kept for pre-extraction callers.

        Reads only. Writing to it is not supported -- ``num_adapters`` is the
        single stored value, and an alias setter would create two sources of
        truth for the slot count that ``build()`` and ``call()`` both read.

        :return: The stored slot count.
        :rtype: int
        """
        return self.num_adapters

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the per-slot ``A``/``B`` weight tensors.

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

        num_adapters = self.num_adapters
        rank = self.rank
        output_dim = self.output_dim
        a_initializer_fn = self.kernel_initializer

        def _a_initializer(shape: Tuple[int, int, int], dtype: Any = None) -> Any:
            # Apply the 2-D initializer independently to every slot's own
            # (input_dim, rank) slice -- see the class Note on why the
            # stacked 3-D shape must not be handed to the initializer as-is.
            #
            # DECISION plan-2026-09-12T075714-035fd488/D-007
            # A single initializer INSTANCE is stateless-deterministic: every
            # call with the same shape replays the same underlying sample
            # (dl_techniques.initializers.clone.clone_initializer module
            # docstring, measured). Calling `a_initializer_fn` directly in
            # this loop (the original implementation) therefore produced
            # `num_adapters` BIT-IDENTICAL slices, not independent ones --
            # every slot read the same rank-`r` subspace of `x` at init
            # (review-iter-1.md concern 3). Do NOT call `a_initializer_fn`
            # directly here again -- clone a fresh initializer per slice so
            # each slot draws its own sample.
            slices = [
                clone_initializer(a_initializer_fn)(
                    shape=(shape[1], shape[2]), dtype=dtype
                )
                for _ in range(shape[0])
            ]
            return keras.ops.stack(slices, axis=0)

        self.a = self.add_weight(
            name="lora_a",
            shape=(num_adapters, input_dim, rank),
            initializer=_a_initializer,
            trainable=True,
        )
        self.b = self.add_weight(
            name="lora_b",
            shape=(num_adapters, rank, output_dim),
            initializer="zeros",
            trainable=True,
        )

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        adapter_idx: Optional[int] = None,
        occurrence_idx: Optional[int] = None,
    ) -> keras.KerasTensor:
        """
        Compute the additive LoRA delta for one slot.

        :param inputs: Input tensor of shape ``(..., input_dim)``.
        :type inputs: keras.KerasTensor
        :param adapter_idx: Which slot's ``A``/``B`` pair to use. A plain
            Python ``int`` -- the site a slot corresponds to is static (fixed at
            model-construction time), never a data-dependent tensor value.
            Defaults to `0`, which is what makes the layer usable where Keras
            supplies only the input: inside ``keras.Sequential``, a functional
            model, or a ``fit()``-driven call.
        :type adapter_idx: Optional[int]
        :param occurrence_idx: Deprecated alias of ``adapter_idx``, kept so
            pre-extraction call sites keep working. Supplying both, with
            different values, raises rather than silently preferring one.
        :type occurrence_idx: Optional[int]
        :return: Delta tensor of shape ``(..., output_dim)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If the two spellings disagree, or if the resolved
            index is outside ``[0, num_adapters)``.
        """
        idx = self._resolve_adapter_idx(adapter_idx, occurrence_idx)

        a_i = self.a[idx]
        b_i = self.b[idx]

        delta = keras.ops.matmul(inputs, a_i)
        delta = keras.ops.matmul(delta, b_i)
        return delta * self.scale

    def _resolve_adapter_idx(
        self,
        adapter_idx: Optional[int],
        occurrence_idx: Optional[int],
    ) -> int:
        """Reconcile the current and deprecated slot-index spellings into one int.

        Split out of :meth:`call` so the validation order is stated once: the
        conflict and the missing-argument cases are checked before the range
        check, because a caller who supplied a bad index under BOTH names needs
        the conflict reported, not a range error against whichever name was read
        first.

        :param adapter_idx: The current spelling, or ``None``.
        :type adapter_idx: Optional[int]
        :param occurrence_idx: The deprecated spelling, or ``None``.
        :type occurrence_idx: Optional[int]
        :return: The resolved slot index.
        :rtype: int
        :raises ValueError: If neither is supplied, if both are supplied with
            different values, or if the resolved index is out of range.
        """
        if adapter_idx is not None and occurrence_idx is not None:
            if adapter_idx != occurrence_idx:
                raise ValueError(
                    f"adapter_idx={adapter_idx} and its deprecated alias "
                    f"occurrence_idx={occurrence_idx} disagree. Pass only "
                    f"adapter_idx; 'occurrence_idx' is a pre-extraction "
                    f"spelling of the same argument."
                )
        idx = adapter_idx if adapter_idx is not None else occurrence_idx

        if idx is None:
            # Defaults to slot 0 rather than raising, so the layer is usable
            # where Keras supplies only the input: inside `keras.Sequential`, a
            # functional model, or a `fit()`-driven call, all of which invoke
            # `layer(x)`. A required keyword would make the adapter unusable in
            # every one of those, and slot 0 is the only sensible reading of
            # "the one slot" for a single-adapter layer.
            return 0

        if not (0 <= idx < self.num_adapters):
            raise ValueError(
                f"adapter_idx must be in [0, {self.num_adapters}), got {idx}"
            )
        return idx

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Reads :attr:`output_dim` from stored config rather than off a weight, so
        it works unbuilt.

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

        Emits ``num_adapters`` only. :meth:`from_config` is the sole reader of
        the old spelling, so a round trip through this method never produces a
        dict carrying both names.

        :return: Dictionary containing every constructor argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "rank": self.rank,
                "alpha": self.alpha,
                "num_adapters": self.num_adapters,
                "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LoRAAdapter":
        """Build from a config dict, substituting the pre-rename slot-count key.

        ``legacy_packages`` in the :func:`~dl_techniques.utils.keras_registration.register_dl_technique`
        decorator keeps the pre-extraction *registry key*
        (``dl_techniques.models.zamba2.lora_adapter>LoRAAdapter``) resolvable,
        but a serialized archive also stores the constructor argument NAME that
        was current when it was written. Such a dict carries
        ``num_occurrences``, which ``__init__`` does not declare, so without
        this substitution every pre-extraction checkpoint raises
        ``ValueError`` on load while every test written after the extraction
        passes.

        The substitution lives here and not in ``__init__`` for the reason
        guide §6.2 gives: a key read out of ``**kwargs`` in the constructor
        while also being forwarded to ``super().__init__()`` is dead on arrival.
        A constructor that accepted the alias would also have to choose which
        spelling to write back, and a dict carrying both is not a valid config.

        :param config: The serialized configuration dict.
        :type config: Dict[str, Any]
        :return: A freshly constructed adapter.
        :rtype: LoRAAdapter
        :raises ValueError: If the dict declares the slot count under both
            names with different values, since there is no correct resolution.
        """
        config = dict(config)
        legacy_value = config.pop(LEGACY_NUM_OCCURRENCES_KEY, None)

        if legacy_value is not None:
            current = config.get("num_adapters")
            if current is not None and current != legacy_value:
                raise ValueError(
                    f"LoRAAdapter config declares both "
                    f"{LEGACY_NUM_OCCURRENCES_KEY}={legacy_value} and "
                    f"num_adapters={current} with different values. "
                    f"'{LEGACY_NUM_OCCURRENCES_KEY}' is the pre-extraction "
                    f"spelling of 'num_adapters'; declare it once."
                )
            logger.warning(
                f"LoRAAdapter.from_config: config key "
                f"'{LEGACY_NUM_OCCURRENCES_KEY}'={legacy_value} is a "
                f"pre-extraction spelling of 'num_adapters'; substituting. "
                f"Re-save the model to write the current key."
            )
            config["num_adapters"] = legacy_value

        return cls(**config)
