"""``BaseVLMHead``: shared construction for the fusion stack two VLM heads use."""

import keras
from typing import Dict, List, Optional, Union, Tuple, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ...fusion.multimodal_fusion import FusionStrategy, MultiModalFusion
from ....utils.tensors import is_single_shape
from .task_types import VLMTaskConfig
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.layers.heads.vlm.base")
class BaseVLMHead(keras.layers.Layer):
    """
    Shared construction for the fusion layer two VLM heads use.

    This class creates the ``MultiModalFusion`` layer and stores the
    configuration it needs. It has no ``call``, so it is not a head:
    constructing it directly raises ``TypeError``.
    :class:`VisualGroundingHead` and :class:`ImageTextMatchingHead` inherit
    from it and supply their own forward pass. :class:`ImageCaptioningHead`,
    :class:`VQAHead` and :class:`MultiTaskVLMHead` do not inherit from it at
    all.

    It owns exactly one sub-layer, ``fusion``, because that is the one layer
    both subclasses run. Each subclass calls :meth:`_build_fusion_stack` from
    its own ``build`` with the shapes it will really pass.

    **Architecture Overview:**

    .. code-block:: text

        vision_features        text_features
        (B, N, D_vis)          (B, N, D_txt)
                  │                 │
                  └────────┬────────┘
                           ▼
                  ┌─────────────────────┐
                  │ fusion              │
                  │ MultiModalFusion    │
                  └──────────┬──────────┘
                             ▼ (B, N, hidden_dim)
                       squeeze_axis (optional)
                             ▼
                    the subclass task layers

    Only :class:`ImageTextMatchingHead` runs a post-fusion norm and FFN, so
    that head owns them and takes their arguments (``normalization_type``,
    ``use_post_fusion_ffn``, ``ffn_type``, ``ffn_expansion_factor``).
    :class:`VisualGroundingHead` goes from ``fusion`` straight to its scorer
    and refuses those arguments, as an unrecognized keyword.

    Every feature width comes from the shapes handed to ``build``. The head
    takes no declared ``vision_dim``/``text_dim``; ``task_config`` carries
    ``hidden_size``, the working width.

    Two fusion strategies are rejected up front. ``cross_attention`` returns
    one tensor per modality and ``attention_pooling`` returns a rank-2 tensor,
    and both subclasses consume a single rank-preserving tensor. Six of the
    eight strategies pass. See :meth:`_build_fusion_stack`.

    :param task_config: Task configuration. ``hidden_size`` becomes
        ``hidden_dim``, the working width of the fusion.
    :type task_config: VLMTaskConfig
    :param fusion_strategy: Which ``MultiModalFusion`` strategy to build. It
        has no default here; each subclass supplies its own.
    :type fusion_strategy: FusionStrategy
    :param fusion_config: Extra keyword arguments for ``MultiModalFusion``.
    :type fusion_config: Optional[Dict[str, Any]]
    :param kwargs: Additional arguments for the base Layer.

    :ivar hidden_dim: Working width, taken from ``task_config.hidden_size``.
    :vartype hidden_dim: int
    :ivar fusion: The ``MultiModalFusion`` layer.
    :vartype fusion: MultiModalFusion
    """

    def __init__(
        self,
        task_config: VLMTaskConfig,
        fusion_strategy: FusionStrategy,
        fusion_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Store the configuration and create the fusion layer.

        The layer name is derived from ``task_config.name``, so a caller must
        not pass ``name``.

        :param task_config: Task configuration for this head.
        :type task_config: VLMTaskConfig
        :param fusion_strategy: Which ``MultiModalFusion`` strategy to build.
        :type fusion_strategy: FusionStrategy
        :param fusion_config: Extra keyword arguments for ``MultiModalFusion``.
        :type fusion_config: Optional[Dict[str, Any]]
        :param kwargs: Additional arguments for the base Layer class.
        :return: None.
        :rtype: None
        :raises TypeError: If this class is constructed directly rather than
            through a subclass.
        """
        # DECISION plan-2026-10-14T062616-81938e81/D-059
        # Refuse direct construction. Do NOT hand back a bare BaseVLMHead: it
        # has no call(), so it constructed fine and raised
        # NotImplementedError, naming the base class, on first use. See
        # decisions.md D-059.
        if type(self) is BaseVLMHead:
            raise TypeError(
                "BaseVLMHead is the shared base of VisualGroundingHead and "
                "ImageTextMatchingHead and has no call(); construct one of "
                "those, or create_vlm_head() for a task type."
            )
        kwargs.setdefault("name", f"{task_config.name}_head")
        super().__init__(**kwargs)

        # Store configuration
        self.task_config = task_config
        self.fusion_strategy = fusion_strategy
        self.fusion_config = fusion_config or {}

        # `VLMTaskConfig.__post_init__` in task_types.py refuses any
        # `hidden_size` that is not a positive integer, so a head never sees
        # None here. Three sites had drifted to two different fallback
        # expressions for a value that was never missing.
        #
        # DECISION plan-2026-07-30T081929-1645aa52/D-014
        # Read `hidden_size` straight. Do NOT re-add an `or <fallback>` in any
        # spelling: the fallback was dead code, proved by constructing with
        # `hidden_size=None` and dims 64/96, which yielded 768, NOT 96.
        # See decisions.md D-014.
        self.hidden_dim = task_config.hidden_size

        # DECISION plan-2026-10-14T062616-81938e81/D-032
        # The base owns the fusion and nothing else. Do NOT move the
        # post-fusion norm, dropout or FFN back here: VisualGroundingHead
        # never runs them (it created all three and built none), the dropout
        # ran in no call at all, and use_post_fusion_ffn was silently ignored
        # by the grounding head. Only ImageTextMatchingHead runs the stack,
        # so it owns it. See decisions.md D-032.
        self.fusion = MultiModalFusion(
            dim=self.hidden_dim,
            fusion_strategy=self.fusion_strategy,
            dropout_rate=self.task_config.dropout_rate,
            name=f"{self.name}_fusion",
            **self.fusion_config,
        )

    def build(self, input_shape: Union[Tuple, Dict]) -> None:
        """
        Mark the layer built. It builds no sub-layer.

        It cannot build ``fusion``, because its shapes depend on how each
        subclass drives it. :class:`VisualGroundingHead` fuses per-region
        ``(B, N, D)`` tensors. :class:`ImageTextMatchingHead` fuses pooled
        ``(B, 1, D)`` tensors and then squeezes. Each subclass calls
        :meth:`_build_fusion_stack` from its own ``build`` with its real
        shapes.

        :param input_shape: Input shape(s), passed straight to the base class.
        :type input_shape: Union[Tuple, Dict]
        :return: None.
        :rtype: None
        """
        super().build(input_shape)

    def _build_fusion_stack(
        self,
        fusion_input_shapes: List[Tuple[Optional[int], ...]],
        squeeze_axis: Optional[int] = None,
    ) -> Tuple[Optional[int], ...]:
        """Build ``fusion`` and return the shape the subclass consumes.

        Shapes are DERIVED via the fusion's ``compute_output_shape`` rather
        than recomputed from ``hidden_dim``, so this stays correct if a fusion
        strategy changes its output width.

        :param fusion_input_shapes: ``[vision_shape, text_shape]`` exactly as the
            subclass passes them to ``self.fusion``.
        :type fusion_input_shapes: List[Tuple[Optional[int], ...]]
        :param squeeze_axis: Axis the subclass squeezes off the fusion output,
            or ``None`` if it does not squeeze.
        :type squeeze_axis: Optional[int]
        :return: The fusion output shape, after the squeeze.
        :rtype: Tuple[Optional[int], ...]
        :raises ValueError: If the fusion layer refuses the input shapes (the
            message names this head and the shapes), the fusion strategy
            returns one shape per modality, or it does not preserve rank.
        """
        # DECISION plan-2026-10-14T062616-81938e81/D-207
        # Re-raise the fusion layer's own refusal with the head's name. Do
        # NOT pre-check the widths against hidden_dim here: which widths a
        # fusion accepts is the strategy's rule, and a copy of it in the head
        # would drift. See decisions.md D-207.
        try:
            self.fusion.build(fusion_input_shapes)
        except ValueError as error:
            raise ValueError(
                f"{type(self).__name__} ({self.name!r}): "
                f"fusion_strategy={self.fusion_strategy!r} at hidden_size="
                f"{self.hidden_dim} cannot fuse vision/text inputs of shapes "
                f"{[tuple(s) for s in fusion_input_shapes]}: {error}"
            ) from error
        fused_shape = self.fusion.compute_output_shape(fusion_input_shapes)

        # Two strategies are rejected here, for two different reasons.
        # `cross_attention` returns one tensor per modality. `attention_pooling`
        # returns a single rank-2 tensor, having already pooled away the axis the
        # caller means to squeeze. Both used to reach Keras as a shape and die
        # with an error naming neither the strategy nor the reason. The two fail
        # identically on the lazy and the explicit-build path, so this is a
        # capability gap, not a `build()` defect. The check reads the fusion
        # layer's declared output contract, not a hardcoded strategy list, so a
        # future strategy that returns a tuple is rejected on its own terms.
        #
        # Do NOT instead teach the subclasses to fan in a per-modality tuple.
        # Which tensor feeds the task head is a modelling decision nobody has
        # asked for.
        #
        # Measured over all 8 strategies at input shapes [(2,4,32),(2,4,32)]:
        # 7 return a single tensor, and 6 of those also preserve rank 3.
        #
        # DECISION plan-2026-07-30T081929-1645aa52/D-011
        # Raise twice. Once on a strategy whose output is not one tensor
        # (cross_attention). Once on one that drops rank (attention_pooling,
        # checked below and stated in this method's `:raises` field). 6 of the
        # 8 strategies pass both checks. See decisions.md D-011.
        if not is_single_shape(fused_shape):
            raise ValueError(
                f"fusion_strategy={self.fusion_strategy!r} produces one output "
                f"per modality (compute_output_shape returned "
                f"{fused_shape!r}), but {type(self).__name__} consumes a "
                f"single fused tensor. Use one of the "
                f"single-tensor strategies (concatenation, addition, "
                f"multiplication, gated, bilinear, tensor_fusion)."
            )

        # Rank must be PRESERVED. Both heads hand `self.fusion` rank-3
        # `(batch, seq_or_region, dim)` tensors and consume a rank-3 result --
        # `ImageTextMatchingHead` squeezes its length-1 axis back off,
        # `VisualGroundingHead` keeps the region axis and scores per region. A
        # strategy that pools an axis away (`attention_pooling` -> rank 2) breaks
        # both, but at different places and with different errors: ITM died in the
        # squeeze, VG only later inside an ArgMax ("Expected dimension in the
        # range [-1, 1)"). One rank check upstream covers both heads and any
        # future pooling strategy, and needs no `squeeze_axis` special-casing.
        if len(fused_shape) != len(tuple(fusion_input_shapes[0])):
            raise ValueError(
                f"fusion_strategy={self.fusion_strategy!r} returns a "
                f"rank-{len(fused_shape)} output {tuple(fused_shape)!r} for "
                f"rank-{len(tuple(fusion_input_shapes[0]))} inputs, so it pools "
                f"away an axis that {type(self).__name__} still needs. Use a "
                f"rank-preserving strategy (concatenation, addition, "
                f"multiplication, gated, bilinear, tensor_fusion)."
            )

        if squeeze_axis is not None:
            fused_shape = tuple(
                d for i, d in enumerate(fused_shape) if i != squeeze_axis
            )
        return tuple(fused_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments needed to rebuild this head.

        ``task_config`` is converted to a JSON-safe dict, because its
        ``task_type`` field holds an enum.

        :return: Serializable layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "task_config": self.task_config.to_dict(),
                "fusion_strategy": self.fusion_strategy,
                "fusion_config": self.fusion_config,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseVLMHead":
        """Reconstruct the head, deserializing the ``task_config`` dataclass.

        The stored ``name`` is passed on, so a custom name survives; the
        default is ``<task_config.name>_head``.

        :param config: Serialized layer configuration.
        :type config: Dict[str, Any]
        :return: A reconstructed head instance.
        :rtype: BaseVLMHead
        """
        config = dict(config)
        config["task_config"] = VLMTaskConfig.coerce(config["task_config"])
        return cls(**config)
