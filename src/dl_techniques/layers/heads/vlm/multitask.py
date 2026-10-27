"""``MultiTaskVLMHead``: several VLM heads behind one layer."""

import keras
from typing import Dict, Optional, Union, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ....utils.logger import logger
from .task_types import VLMTaskConfig
from dl_techniques.utils.keras_registration import register_dl_technique
from .utils import (
    _KERAS_BASE_LAYER_KWARGS,
    _WRAPPER_OWNED_HEAD_KWARGS,
    _accepted_constructor_kwargs,
)
from .dispatch import get_head_class


@register_dl_technique("dl_techniques.layers.heads.vlm.multitask")
class MultiTaskVLMHead(keras.layers.Layer):
    """
    Several VLM heads behind one layer, fed from one shared input dict.

    This wrapper builds one head per entry in ``task_configs`` and hands every
    head the SAME ``inputs`` dict. There is no per-task projection stage, so
    the heads see identical tensors and differ only in what they compute.

    It does not inherit from :class:`BaseVLMHead`. It derives from
    ``keras.layers.Layer`` directly.

    **Architecture Overview:**

    .. code-block:: text

        ┌────────────────────────────────────────┐
        │ inputs, one shared dict                │
        │  {'vision_features': ...,              │
        │   'text_features': ...,                │
        │   'question_features': ...}            │
        └─────────────────────┬──────────────────┘
                              ▼
        ┌──────────────────────────────────────────────┐
        │ call(inputs, training=None, *, task_name)    │
        └──────────┬───────────────────────┬───────────┘
                   │ yes                   │ no
                   ▼                       ▼
        ┌─────────────────────┐ ┌─────────────────────┐
        │ task_heads[name]    │ │ every task head     │
        │ that head's output  │ │ {name: output, ...} │
        └─────────────────────┘ └─────────────────────┘

    ``task_name`` is keyword-only, so the second positional slot is
    ``training``, as on every other head. A ``task_name`` that is not in
    ``task_heads`` raises ``ValueError``, on the eager and on the symbolic
    path, and the symbolic output and ``compute_output_shape`` follow
    ``task_name`` as the eager call does.

    Keyword routing has three tiers. Wrapper-owned arguments
    (``task_configs``, ``task_specific_kwargs``) are consumed here. ``task_specific_kwargs``
    names one head, so it is checked strictly. Everything else is shared and
    routed to whichever heads accept it, because the head classes do not
    share a signature. A shared keyword no head accepts raises; partial
    application is reported through ``logger.info``.

    :class:`VQAHead` reads ``question_features`` while the other four heads
    read ``text_features``. One dict goes to all of them, so a wrapper mixing
    VQA with any other task raises ``KeyError`` unless the caller supplies
    both keys. Duplicating the same tensor under both keys is fine.

    Input shape:
        One shape dict, keyed by feature name, holding every key any of the
        configured heads reads.

    Output shape:
        With ``task_name``, that head's own output. Without it, a dict mapping
        each task name to that head's output.

    :param task_configs: Mapping of task name to its ``VLMTaskConfig`` (or a
        dict of its fields). One head is built per entry. At least one entry.
    :type task_configs: Dict[str, Union[VLMTaskConfig, Dict[str, Any]]]
    :param task_specific_kwargs: Per-task constructor overrides, keyed by the
        same task names as ``task_configs``. Merged over the shared keywords,
        so a per-task value wins.
    :type task_specific_kwargs: Optional[Dict[str, Dict[str, Any]]]
    :param kwargs: Shared per-head constructor settings plus this layer's own
        Keras base arguments. The two are separated in ``__init__``.

    :ivar task_heads: Task name to head instance. Each head is also set as an
        attribute named ``head_<task_name>``.
    :vartype task_heads: Dict[str, keras.layers.Layer]
    :ivar shared_head_kwargs: The shared keywords, with the Keras base
        arguments removed.
    :vartype shared_head_kwargs: Dict[str, Any]
    """

    def __init__(
        self,
        task_configs: Dict[str, VLMTaskConfig],
        task_specific_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """Build one head per task, from shared settings plus per-task overrides.

        :param task_configs: Mapping of task name to its ``VLMTaskConfig``, or
            to a dict of its fields.
        :type task_configs: Dict[str, Union[VLMTaskConfig, Dict[str, Any]]]
        :param task_specific_kwargs: Per-task constructor overrides, keyed by the
            same task names as ``task_configs`` -- e.g.
            ``{"cap": {"num_heads": 4, "num_layers": 1}}``. Merged over the
            shared kwargs, so a task-specific value wins.

            This is an EXPLICIT parameter rather than something read back out of
            ``**kwargs``. It used to be the latter, which made it unusable. The
            dict stayed in ``kwargs`` and was forwarded to ``Layer.__init__``.
            Keras rejected it with ``ValueError: Unrecognized keyword
            arguments``. So the documented feature always raised, and every head
            was forced onto the shared defaults.
        :type task_specific_kwargs: Optional[Dict[str, Dict[str, Any]]]
        :param kwargs: Shared per-head constructor settings plus this layer's own
            Keras base arguments. The two are SEPARATED below. Base arguments
            go only to ``Layer.__init__``, never into a head constructor. There
            ``name`` would collide with the head's own auto-generated name.

            A shared kwarg goes to the heads whose class ACCEPTS it. The rest
            skip it. The five head classes do not share one signature.
            ``ImageCaptioningHead`` and ``VQAHead`` are plain ``Layer``
            subclasses, so a ``BaseVLMHead`` argument such as
            ``fusion_strategy`` is a hard error on them. Requiring every head to
            accept every shared argument would make this wrapper unusable for
            any mixed set of tasks.

            Two guards keep that from being silent. A shared kwarg accepted by
            NO head raises. Partial application is reported via
            ``logger.info``. Use ``task_specific_kwargs`` to target one head
            exactly. Those ARE validated strictly, since they name a head.
        :raises ValueError: If ``task_configs`` is not a dict or is empty, or
            ``task_specific_kwargs`` names a task absent from ``task_configs``.
        :raises TypeError: If a value is neither a ``VLMTaskConfig`` nor a dict.
        """
        if not isinstance(task_configs, dict):
            raise ValueError(
                "MultiTaskVLMHead: task_configs must be a dict of task name to "
                f"VLMTaskConfig, got {type(task_configs).__name__}; "
                "create_multi_task_vlm_head also takes a list or tuple."
            )
        if not task_configs:
            raise ValueError(
                "MultiTaskVLMHead needs at least one task in task_configs; an "
                "empty wrapper returns {} for any input."
            )
        # Keras base arguments belong to THIS layer; everything else is intended
        # for the per-task heads. Forwarding `name` into a head would give every
        # head this wrapper's name (each head defaults to
        # `f"{task_config.name}_head"`), and forwarding `trainable`/`dtype`
        # silently reconfigures the children.
        #
        # DECISION plan-2026-07-30T081929-1645aa52/D-012
        # Keep and forward the Keras base kwargs. Do NOT re-add a ``config.pop``
        # loop for them: it discarded the values, so a head saved with
        # ``name='mt'``/``trainable=False`` reloaded as
        # ``name='multi_task_vlm_head'``/``trainable=True``. See decisions.md D-012.
        base_kwargs = {
            k: v for k, v in kwargs.items() if k in _KERAS_BASE_LAYER_KWARGS
        }
        shared_head_kwargs = {
            k: v for k, v in kwargs.items() if k not in _KERAS_BASE_LAYER_KWARGS
        }
        super().__init__(**base_kwargs)

        # A dict of config fields is accepted as create_vlm_head accepts it;
        # this is also the deserialization from_config relies on.
        task_configs = {
            name: VLMTaskConfig.coerce(config)
            for name, config in task_configs.items()
        }
        self.task_configs = task_configs
        self.shared_head_kwargs = shared_head_kwargs
        self.task_specific_kwargs = dict(task_specific_kwargs or {})

        # A typo'd task name would otherwise apply to nothing at all, which is
        # exactly the silent-no-op class this package's conventions forbid.
        unknown_tasks = set(self.task_specific_kwargs) - set(task_configs)
        if unknown_tasks:
            raise ValueError(
                f"task_specific_kwargs names task(s) absent from task_configs: "
                f"{sorted(unknown_tasks)}. Known tasks: {sorted(task_configs)}."
            )

        head_classes = {
            task_name: get_head_class(task_config.task_type)
            for task_name, task_config in task_configs.items()
        }
        accepted = {
            task_name: _accepted_constructor_kwargs(head_class)
            for task_name, head_class in head_classes.items()
        }

        # The wrapper supplies these itself; a caller passing one through would
        # collide with that and raise an opaque duplicate-argument TypeError.
        for source, mapping in (
            ("shared kwargs", self.shared_head_kwargs),
            *(
                (f"task_specific_kwargs[{t!r}]", o)
                for t, o in self.task_specific_kwargs.items()
            ),
        ):
            reserved = set(mapping) & _WRAPPER_OWNED_HEAD_KWARGS
            if reserved:
                raise ValueError(
                    f"{source} may not set {sorted(reserved)} — "
                    f"MultiTaskVLMHead supplies these to every head itself, "
                    f"from task_configs."
                )

        # A per-task override names ONE head explicitly, so an argument that head
        # cannot accept is a caller error and is rejected outright.
        for task_name, overrides in self.task_specific_kwargs.items():
            unusable = set(overrides) - accepted[task_name]
            if unusable:
                raise ValueError(
                    f"task_specific_kwargs[{task_name!r}] sets {sorted(unusable)}, "
                    f"which {head_classes[task_name].__name__} does not accept. "
                    f"It accepts: {sorted(accepted[task_name] - _WRAPPER_OWNED_HEAD_KWARGS)}."
                )

        # A SHARED kwarg is best-effort by design: the head classes do not share
        # a signature, so requiring every head to accept every shared argument
        # would make this wrapper unusable for any mixed set of tasks (which is
        # its entire purpose). It is routed to the heads that accept it and
        # skipped for the rest.
        #
        # The guard that keeps that from becoming a silent no-op: a shared kwarg
        # accepted by NO head is a typo or a mistake, and raises. Partial
        # application is reported through the logger so it is discoverable.
        for key in sorted(self.shared_head_kwargs):
            takers = [t for t, acc in accepted.items() if key in acc]
            if not takers:
                every = sorted(
                    set().union(*accepted.values()) - _WRAPPER_OWNED_HEAD_KWARGS
                )
                raise ValueError(
                    f"shared kwarg {key!r} is not accepted by ANY head in this "
                    f"multi-task wrapper "
                    f"({ {t: c.__name__ for t, c in head_classes.items()} }). "
                    f"Accepted by at least one head: {every}."
                )
            skipped = sorted(set(accepted) - set(takers))
            if skipped:
                logger.info(
                    f"MultiTaskVLMHead: shared kwarg '{key}' applied to "
                    f"{sorted(takers)}; skipped for {skipped} "
                    f"(their head classes do not accept it). Use "
                    f"task_specific_kwargs to configure those explicitly."
                )

        self.task_heads = {}
        for task_name, task_config in task_configs.items():
            head_class = head_classes[task_name]

            # Shared settings, filtered to what THIS head accepts, then this
            # task's overrides on top (already validated as acceptable above).
            head_kwargs = {
                k: v
                for k, v in self.shared_head_kwargs.items()
                if k in accepted[task_name]
            }
            head_kwargs.update(self.task_specific_kwargs.get(task_name, {}))

            head = head_class(task_config=task_config, **head_kwargs)
            self.task_heads[task_name] = head
            setattr(self, f"head_{task_name}", head)

    def build(self, input_shape: Dict) -> None:
        """Build every task head against the shared input shapes.

        ``call`` hands the same ``inputs`` dict to every head, so each head is
        built from the same shapes. A head that needs a key the caller did not
        supply raises here rather than at first use. That is the same failure
        ``call`` would produce, only earlier, and it is not swallowed.

        :param input_shape: Shared input shapes, keyed by feature name.
        :type input_shape: Dict
        :return: None.
        :rtype: None
        :raises KeyError: If a head needs an input key the caller omitted.
        """
        if self.built:
            return

        for head in self.task_heads.values():
            head.build(input_shape)

        super().build(input_shape)

    def _selected_heads(self, task_name: Optional[str]) -> Dict[str, keras.layers.Layer]:
        """Return the heads a call runs: one for ``task_name``, else all of them.

        :param task_name: A configured task name, or ``None``.
        :type task_name: Optional[str]
        :return: Task name to head.
        :rtype: Dict[str, keras.layers.Layer]
        :raises ValueError: If ``task_name`` names no configured task.
        """
        if task_name is None:
            return dict(self.task_heads)
        if task_name not in self.task_heads:
            raise ValueError(
                f"Unknown task: {task_name!r}. Configured tasks: "
                f"{sorted(self.task_heads)}."
            )
        return {task_name: self.task_heads[task_name]}

    # DECISION plan-2026-10-14T062616-81938e81/D-060
    # `task_name` is keyword-only and `training` keeps the second positional
    # slot. Do NOT move `task_name` back in front of `training`: a positional
    # flag then landed in `task_name` (`.call(inputs, True)` raised "Unknown
    # task: True", a tensor flag "Tensor is unhashable"). See decisions.md
    # D-060.
    def call(
        self,
        inputs: Dict[str, keras.KerasTensor],
        training: Optional[bool] = None,
        *,
        task_name: Optional[str] = None,
    ) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Run one task head, or every one of them.

        :param inputs: One shared input dict, handed to each head unchanged.
        :type inputs: Dict[str, keras.KerasTensor]
        :param training: Keras training flag, forwarded to each head.
        :type training: Optional[bool]
        :param task_name: Keyword only. Which head to run; ``None`` runs them
            all.
        :type task_name: Optional[str]
        :return: That head's output, or a dict of task name to output.
        :rtype: Union[Dict[str, Any], Dict[str, Dict[str, Any]]]
        :raises ValueError: If ``task_name`` names no configured task.
        """
        outputs = {
            name: head(inputs, training=training)
            for name, head in self._selected_heads(task_name).items()
        }
        return outputs[task_name] if task_name is not None else outputs

    def compute_output_shape(
        self, input_shape: Dict, *, task_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return the output shapes ``call`` returns for the same ``task_name``.

        :param input_shape: Shared input shapes (dict keyed by feature name).
        :type input_shape: Dict
        :param task_name: Keyword only, as in ``call``.
        :type task_name: Optional[str]
        :return: That head's output shapes, or task name to output shapes.
        :rtype: Dict[str, Any]
        :raises ValueError: If ``task_name`` names no configured task.
        """
        shapes = {
            name: head.compute_output_shape(input_shape)
            for name, head in self._selected_heads(task_name).items()
        }
        return shapes[task_name] if task_name is not None else shapes

    def compute_output_spec(
        self,
        inputs: Dict[str, Any],
        training: Optional[bool] = None,
        *,
        task_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return the symbolic outputs ``call`` returns for the same ``task_name``.

        Keras' default routes a symbolic call through
        ``compute_output_shape`` with the input shapes only, so ``task_name``
        never reached it and a symbolic ``head(inputs, task_name='itm')``
        returned every task's outputs. Each selected head reports its own
        spec instead.

        :param inputs: Symbolic shared inputs.
        :type inputs: Dict[str, Any]
        :param training: Unused; accepted because ``call`` takes it.
        :type training: Optional[bool]
        :param task_name: Keyword only, as in ``call``.
        :type task_name: Optional[str]
        :return: That head's output spec, or task name to output spec.
        :rtype: Dict[str, Any]
        :raises ValueError: If ``task_name`` names no configured task.
        """
        specs = {
            name: head.compute_output_spec(inputs)
            for name, head in self._selected_heads(task_name).items()
        }
        return specs[task_name] if task_name is not None else specs

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments needed to rebuild this wrapper.

        The shared per-head keywords are written back out flat, alongside the
        wrapper's own arguments, which is the shape ``__init__`` expects.

        :return: Serializable layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "task_configs": {
                    name: tc.to_dict()
                    for name, tc in self.task_configs.items()
                },
                "task_specific_kwargs": self.task_specific_kwargs,
            }
        )
        # `shared_head_kwargs` no longer contains this layer's Keras base
        # arguments, so this cannot clobber the `name`/`dtype` that
        # `super().get_config()` already wrote.
        config.update(self.shared_head_kwargs)
        return config
