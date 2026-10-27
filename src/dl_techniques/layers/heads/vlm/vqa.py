"""``VQAHead``: pools each modality, concatenates, classifies."""

import keras
from typing import Dict, List, Optional, Union, Tuple, Any, Sequence

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ...attention.factory import create_attention_layer
from ...sequence_pooling import SequencePooling
from ..validation import positive_int
from .task_types import VLMTaskConfig
from dl_techniques.utils.keras_registration import register_dl_technique
from .utils import _check_task_config_fields, _pool_sequence


@register_dl_technique("dl_techniques.layers.heads.vlm.vqa")
class VQAHead(keras.layers.Layer):
    """
    Pools each modality, concatenates them, and classifies the answer.

    This head picks one answer from a fixed vocabulary. It pools the vision
    features and the question features to one vector each, joins them end to
    end, and runs the result through an MLP.

    It does NOT inherit from :class:`BaseVLMHead` and has no fusion layer. It
    derives from ``keras.layers.Layer`` directly.

    It is also the one head in this module that reads ``question_features``
    rather than ``text_features``. :class:`MultiTaskVLMHead` fans one input
    dict to every head, so mixing this head with any other needs both keys.

    **Architecture Overview:**

    .. code-block:: text

        vision_features           question_features
        (B, N_vis, D_vis)         (B, S, D_txt)
                │                          │
                ▼                          ▼
        ┌────────────────┐        ┌────────────────┐
        │ pool vision    │        │ pool question  │
        │ mean/max/attn  │        │ mean/max/attn  │
        └───────┬────────┘        └───────┬────────┘
                ▼ (B, D_v)                ▼ (B, D_q)
                └────────────┬────────────┘
                             ▼
                  ┌─────────────────────┐
                  │ concatenate axis -1 │
                  └──────────┬──────────┘
                             ▼ (B, pooled_width)
                  ┌─────────────────────┐
                  │ hidden_layers[i]    │
                  │ Dense gelu          │
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │ dropout_layers[i]   │
                  └──────────┬──────────┘
                             ▼
                   repeat x len(hidden_dims)
                             ▼
                  ┌─────────────────────┐
                  │ output_layer Dense  │
                  └──────────┬──────────┘
                             ▼
                   'answer_logits' (B, num_classes)

    The ``"attention"`` strategy is not two pooling layers. One
    cross-attention layer runs in both directions, vision attending to the
    question and the question attending to vision, and each attended result is
    then mean-pooled. Running it in reverse needs the two feature widths to
    match, which is a standing contract of this head.

    ``pooled_width`` depends on the strategy. ``mean`` and ``max`` keep each
    modality's own channel count, so it is ``D_vis + D_txt``. ``attention``
    maps both to ``embed_dim``, so it is ``2 * embed_dim``.

    Every pooling honours the optional keep masks ``'vision_mask'`` and
    ``'question_mask'`` (``(batch, length)``, 1 = real token). ``mean`` and
    ``max`` reduce over the real positions only. ``attention`` passes each
    mask as the key mask of the direction that attends to that modality, then
    mean-pools each attended result over its own real positions.

    Input shape:
        ``{'vision_features': (batch, num_vision, vision_width),
        'question_features': (batch, seq_len, text_width)}``, plus the
        optional masks. ``mean`` and ``max`` also take rank-2 (already
        pooled) features, and refuse a mask for them. ``attention`` needs
        both inputs rank 3 and ``hidden_size`` wide, checked in ``build``.

    Output shape:
        ``{'answer_logits': (batch, num_classes)}``.

    :param task_config: Configuration object for the task. ``num_classes``
        is required and ``vocab_size`` must be unset. ``hidden_size`` becomes
        ``embed_dim``; only the ``"attention"`` strategy reads it.
    :type task_config: VLMTaskConfig
    :param hidden_dims: Widths of the classifier MLP's hidden layers.
    :type hidden_dims: Sequence[int]
    :param pooling_strategy: One of ``"mean"``, ``"max"`` or ``"attention"``.
    :type pooling_strategy: str
    :param num_heads: Head count of the ``"attention"`` strategy's
        cross-attention. ``None`` means 8. It must divide ``hidden_size``.
        The other two strategies run no attention and refuse a value.
    :type num_heads: Optional[int]
    :param kwargs: Additional arguments for the base Layer.

    :ivar embed_dim: Attention width, taken from ``task_config.hidden_size``.
    :vartype embed_dim: int
    :ivar num_heads: The resolved head count, or ``None`` without attention.
    :vartype num_heads: Optional[int]
    :ivar attention_pooling: Cross-attention layer, or ``None`` when the
        strategy is not ``"attention"``.
    :vartype attention_pooling: Optional[keras.layers.Layer]
    :ivar vision_pooling: Weightless ``SequencePooling`` over the vision
        sequence (``'max'`` for the max strategy, ``'mean'`` otherwise).
    :vartype vision_pooling: SequencePooling
    :ivar question_pooling: The same, over the question sequence.
    :vartype question_pooling: SequencePooling
    :ivar hidden_layers: The classifier MLP's ``Dense`` layers.
    :vartype hidden_layers: List[keras.layers.Dense]
    :ivar dropout_layers: One ``Dropout`` per hidden layer.
    :vartype dropout_layers: List[keras.layers.Dropout]
    :ivar output_layer: Final ``Dense`` to ``num_classes``.
    :vartype output_layer: keras.layers.Dense
    """

    def __init__(
        self,
        task_config: VLMTaskConfig,
        hidden_dims: Sequence[int] = (512, 256),
        pooling_strategy: str = "attention",
        num_heads: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Store the configuration and create the pooling and MLP sub-layers.

        ``attention_pooling`` is created only for the ``"attention"``
        strategy. The two weightless ``SequencePooling`` layers are created
        for every strategy. One ``Dense`` and one ``Dropout`` are created per
        entry in ``hidden_dims``, plus the output ``Dense``.

        The layer name is derived from ``task_config.name``, so a caller must
        not pass ``name``.

        :param task_config: Configuration object for the task.
        :type task_config: VLMTaskConfig
        :param hidden_dims: Widths of the classifier MLP's hidden layers.
        :type hidden_dims: Sequence[int]
        :param pooling_strategy: One of ``"mean"``, ``"max"``, ``"attention"``.
        :type pooling_strategy: str
        :param num_heads: Attention head count; ``None`` means 8. Only the
            ``"attention"`` strategy takes one.
        :type num_heads: Optional[int]
        :param kwargs: Additional arguments for the base Layer class.
        :raises ValueError: If ``task_config.num_classes`` is unset or
            ``task_config.vocab_size`` is set, if ``pooling_strategy`` is not
            one of the three, if ``num_heads`` is not a positive integer or
            does not divide ``hidden_size``, or if ``num_heads`` is given
            without the ``"attention"`` strategy.
        :return: None.
        :rtype: None
        """
        _check_task_config_fields(task_config, "VQAHead", required=("num_classes",))
        if pooling_strategy not in ["mean", "max", "attention"]:
            raise ValueError(f"Unsupported pooling_strategy: {pooling_strategy}")

        # DECISION plan-2026-10-14T062616-81938e81/D-056
        # The head count is an argument, resolved to 8 only where it is read.
        # Do NOT hardcode it again (hidden_size=36 died inside the attention
        # factory naming neither VQAHead nor a knob the caller could change),
        # and do NOT accept it for mean/max, which run no attention (a
        # default-valued dead knob is the D-030 defect). See decisions.md
        # D-056.
        if pooling_strategy == "attention":
            num_heads = positive_int(
                8 if num_heads is None else num_heads, "num_heads", "VQAHead")
            if task_config.hidden_size % num_heads != 0:
                raise ValueError(
                    f"VQAHead: task_config.hidden_size={task_config.hidden_size} "
                    f"must be divisible by num_heads={num_heads} for the "
                    f"attention pooling; pass num_heads=... to match."
                )
        elif num_heads is not None:
            raise ValueError(
                f"VQAHead: num_heads={num_heads!r} is read only by "
                f"pooling_strategy='attention'; pooling_strategy="
                f"{pooling_strategy!r} runs no attention. Leave it unset."
            )
        kwargs.setdefault("name", f"{task_config.name}_head")
        super().__init__(**kwargs)

        # Store configuration
        self.task_config = task_config
        # DECISION plan-2026-08-19T163559-499b6f0e/D-085: the DEFAULT is a
        # tuple (R-009 S1) and the STORED attribute is a list. Keeping the
        # store as `list(...)` is what makes the conversion invisible: it is
        # the type `get_config` has always emitted, so a saved config's JSON
        # shape and every `== [..]` assertion in the suites are unchanged.
        self.hidden_dims = list(hidden_dims)
        self.pooling_strategy = pooling_strategy
        self.num_heads = num_heads
        # D-014: dead fallback removed (see BaseVLMHead.__init__).
        self.embed_dim = self.task_config.hidden_size

        # CREATE sub-layers
        if self.pooling_strategy == "attention":
            self.attention_pooling = create_attention_layer(
                "multi_head_cross",
                dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout_rate=self.task_config.dropout_rate,
                name="attention_pooling",
            )
        else:
            self.attention_pooling = None

        # Weightless; `max` reduces with `max`, the other two with `mean`.
        reduction = "max" if self.pooling_strategy == "max" else "mean"
        self.vision_pooling = SequencePooling(strategy=reduction, name="vision_pooling")
        self.question_pooling = SequencePooling(strategy=reduction, name="question_pooling")

        self.hidden_layers = []
        self.dropout_layers = []
        for i, hidden_dim in enumerate(self.hidden_dims):
            self.hidden_layers.append(
                keras.layers.Dense(hidden_dim, activation="gelu", name=f"hidden_{i}")
            )
            self.dropout_layers.append(
                keras.layers.Dropout(self.task_config.dropout_rate, name=f"dropout_{i}")
            )

        self.output_layer = keras.layers.Dense(
            self.task_config.num_classes, name="output_layer"
        )

    def build(self, input_shape: Union[Dict, Tuple, List]) -> None:
        """Explicitly build every sub-layer ``call()`` uses, in order.

        Only the sub-layers this head's ``pooling_strategy`` actually exercises
        are built. ``attention_pooling`` exists only for ``"attention"``.
        Building it otherwise would create weights the lazy path never created.

        The classifier's input width is derived from the ACTUAL input shapes,
        not from any declared width. The two pooling branches produce
        different widths. ``mean``/``max`` keep each modality's own channel
        count, while ``attention`` maps both to ``embed_dim``.

        :param input_shape: Shape dict with ``vision_features`` and
            ``question_features``.
        :type input_shape: Union[Dict, Tuple, List]
        """
        if self.built:
            return

        vision_shape = tuple(input_shape["vision_features"])
        question_shape = tuple(input_shape["question_features"])
        batch = vision_shape[0]

        if self.pooling_strategy == "attention":
            # One cross-attention layer is reused in BOTH directions
            # (vision<-question and question<-vision), so each modality is a
            # query once and must be `embed_dim` wide. The attention layer
            # checks only the query of the direction it is built with, so
            # the question width used to pass build and fail in call; both
            # are checked here (D-056).
            for key, shape in (("vision_features", vision_shape),
                               ("question_features", question_shape)):
                if shape[-1] is not None and shape[-1] != self.embed_dim:
                    raise ValueError(
                        f"VQAHead: {key} is {shape[-1]} wide, but "
                        f"pooling_strategy='attention' runs one cross-attention "
                        f"in both directions, so both inputs must be "
                        f"task_config.hidden_size={self.embed_dim} wide. Project "
                        f"them first, or use pooling_strategy='mean' or 'max', "
                        f"which take any widths."
                    )
            self.attention_pooling.build([vision_shape, question_shape])
            pooled_width = self.embed_dim * 2
        else:
            pooled_width = vision_shape[-1] + question_shape[-1]

        # The poolers are weightless; building them only marks them built.
        attended = self.pooling_strategy == "attention"
        self.vision_pooling.build(
            vision_shape[:-1] + (self.embed_dim,) if attended else vision_shape)
        self.question_pooling.build(
            question_shape[:-1] + (self.embed_dim,) if attended else question_shape)

        x_shape = (batch, pooled_width)
        for hidden_layer, dropout_layer in zip(self.hidden_layers, self.dropout_layers):
            hidden_layer.build(x_shape)
            x_shape = hidden_layer.compute_output_shape(x_shape)
            dropout_layer.build(x_shape)

        self.output_layer.build(x_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: Dict[str, keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """
        Pool both modalities, concatenate them, and classify.

        ``"mean"`` and ``"max"`` reduce over axis 1, so they need rank-3
        inputs. ``"attention"`` runs one cross-attention layer in both
        directions and mean-pools each attended result.

        :param inputs: Dict with ``vision_features`` and ``question_features``,
            and optionally ``vision_mask`` and ``question_mask``.
        :type inputs: Dict[str, keras.KerasTensor]
        :param training: Keras training flag, forwarded to every sub-layer.
        :type training: Optional[bool]
        :return: Dict with a single ``answer_logits`` entry.
        :rtype: Dict[str, keras.KerasTensor]
        :raises ValueError: If ``pooling_strategy`` is not one of the three,
            or a mask is given for a rank-2 input.
        """
        if self.pooling_strategy in ("mean", "max"):
            vision_pooled = _pool_sequence(
                self.vision_pooling, inputs, "vision_features", "VQAHead")
            text_pooled = _pool_sequence(
                self.question_pooling, inputs, "question_features", "VQAHead")
        elif self.pooling_strategy == "attention":
            vision_features = inputs["vision_features"]
            question_features = inputs["question_features"]
            vision_mask = inputs.get("vision_mask")
            question_mask = inputs.get("question_mask")
            # Each direction masks the modality it attends TO, then the
            # attended sequence is mean-pooled over its own real positions.
            vision_attended = self.attention_pooling(
                vision_features, kv_input=question_features,
                attention_mask=question_mask, training=training
            )
            text_attended = self.attention_pooling(
                question_features, kv_input=vision_features,
                attention_mask=vision_mask, training=training
            )
            vision_pooled = self.vision_pooling(vision_attended, mask=vision_mask)
            text_pooled = self.question_pooling(text_attended, mask=question_mask)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        x = keras.ops.concatenate([vision_pooled, text_pooled], axis=-1)

        for hidden_layer, dropout_layer in zip(self.hidden_layers, self.dropout_layers):
            x = hidden_layer(x)
            x = dropout_layer(x, training=training)

        logits = self.output_layer(x)
        return {"answer_logits": logits}

    def compute_output_shape(self, input_shape):
        """
        Return the output shape, mirroring the ``call`` return value.

        ``call`` concatenates the two pooled vectors, so the batch axis is
        whichever of the two inputs states it (VLM-4 review item: this read
        the vision batch alone and lost a batch only the question stated).

        :param input_shape: Shape dict with ``vision_features`` and
            ``question_features``.
        :type input_shape: Dict
        :return: Shape for ``answer_logits``.
        :rtype: Dict[str, Tuple[Optional[int], ...]]
        """
        batch = input_shape["vision_features"][0]
        if batch is None:
            batch = input_shape["question_features"][0]
        return {"answer_logits": (batch, self.task_config.num_classes)}

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments needed to rebuild this head.

        :return: Serializable layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "task_config": self.task_config.to_dict(),
                "hidden_dims": self.hidden_dims,
                "pooling_strategy": self.pooling_strategy,
                "num_heads": self.num_heads,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "VQAHead":
        """
        Reconstruct the head, deserializing the ``task_config`` dataclass.

        The stored ``name`` is passed on, so a custom name survives; the
        default is ``<task_config.name>_head``.

        :param config: Serialized layer configuration.
        :type config: Dict[str, Any]
        :return: A reconstructed head instance.
        :rtype: VQAHead
        """
        config = dict(config)
        config["task_config"] = VLMTaskConfig.coerce(config["task_config"])
        return cls(**config)
