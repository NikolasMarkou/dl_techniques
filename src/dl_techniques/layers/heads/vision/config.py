"""Head configuration presets and the model-side head_config_overrides merge.

`HeadConfiguration` supplies two keyword-argument presets for every vision
task type that has a head (default and efficient). `merge_head_config_overrides`
and `build_model_vision_head` are the model-side helpers: preset -> model's
own keys -> caller's overrides -> `create_vision_head`.
"""

from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .base import BaseVisionHead
from .dispatch import (
    VISION_HEAD_DISPATCH,
    _as_vision_task_type,
    _dispatched_task_list,
    create_vision_head,
)
from .task_types import VisionTaskType
from .utils import _drop_off_stage_knobs


#: Head-config keys every :class:`HeadConfiguration` preset forces, per task
#: type, because that task's head never applies the layer the key would
#: build and refuses ``True`` (:func:`_refuse_common_layers_call_never_runs`).
#: Applied last by both presets, so a preset never builds an unusable
#: config.
PRESET_FIXED_KEYS: Dict[VisionTaskType, Dict[str, Any]] = {
    VisionTaskType.CLASSIFICATION: {'use_ffn': False},
    VisionTaskType.DENOISING: {
        'use_attention': False, 'use_ffn': False, 'dropout_rate': 0.0},
    VisionTaskType.SUPER_RESOLUTION: {
        'use_attention': False, 'use_ffn': False, 'dropout_rate': 0.0},
}


class HeadConfiguration:
    """
    Two keyword-argument presets for the vision heads.

    This class builds no layer. Each static method returns a dict you pass to
    a head constructor or to :func:`create_vision_head`. The presets differ in
    width, dropout, attention and FFN choice. Each of the ten task types in
    :data:`VISION_HEAD_DISPATCH` has both presets, and every head either
    preset configures builds and runs (pinned by the head tests). A task type
    with no head has no preset: both methods raise ``ValueError``.

    A third, ``get_high_performance_config``, was deleted on 2026-10-14: its
    ``attention_type='differential'`` needs ``num_heads`` and ``head_dim``,
    which the heads' attention builder never passes, so it raised
    ``ValueError`` for every task type and had built no head.

    **Architecture Overview:**

    .. code-block:: text

        get_default_config(task_type)
                  │
                  ▼
        no head for task_type? ──► ValueError
                  │
                  ▼
        base_config, 7 keys shared by every task
                  │
                  ▼
        + the task_specific entry, for 8 of the 10 task types
                  │
                  ▼
        get_efficient_config
        overrides 6 keys
                  │
                  ▼
        both presets: the PRESET_FIXED_KEYS entry for the task
                  │
                  ▼
        both presets, last: drop the knobs of every stage left off

    ``get_efficient_config`` calls ``get_default_config`` first, so it
    inherits the task-specific keys and then overrides the shared ones.

    Both presets then apply :data:`PRESET_FIXED_KEYS` for their task type,
    last, so no preset asks a head for a layer its ``call`` never runs:
    classification gets ``use_ffn=False`` and the two enhancement tasks get
    ``use_attention=False``, ``use_ffn=False`` and ``dropout_rate=0.0``.
    Those heads refuse the other values. Each preset then drops the knobs
    of every stage it leaves off (``attention_type`` without attention,
    ``ffn_type`` and ``ffn_expansion_factor`` without the FFN), because a
    head refuses a knob for a stage that is off (D-210).

    Example:
        >>> cfg = HeadConfiguration.get_default_config(
        ...     VisionTaskType.CLASSIFICATION)
        >>> cfg['num_classes']
        1000
    """

    @staticmethod
    def get_default_config(task_type: VisionTaskType) -> Dict[str, Any]:
        """
        Get the default preset for a task type.

        Seven shared keys are the starting point. Eight task types add their
        own keys on top: every dispatched type but the two enhancement
        tasks. Then the task's :data:`PRESET_FIXED_KEYS` entry is applied,
        which turns off the common layers classification and the two
        enhancement tasks never run, and last the knobs of every stage left
        off are dropped (a head refuses them, D-210).

        :param task_type: Task the preset is for, as an enum member or its
            string value.
        :type task_type: VisionTaskType
        :return: Keyword arguments for the matching head class.
        :rtype: Dict[str, Any]
        :raises ValueError: If ``task_type`` has no head.
        """
        task_type = _as_vision_task_type(task_type)
        if task_type not in VISION_HEAD_DISPATCH:
            raise ValueError(
                f"HeadConfiguration has no preset for {task_type.value!r}: "
                f"it has no vision head. {_dispatched_task_list()}"
            )

        base_config = {
            'hidden_dim': 256,
            'normalization_type': 'layer_norm',
            'activation_type': 'gelu',
            'dropout_rate': 0.1,
            'use_ffn': True,
            'ffn_type': 'mlp',
            'ffn_expansion_factor': 4
        }

        task_specific = {
            VisionTaskType.DETECTION: {
                # 80 classes is the COCO default.
                'num_classes': 80,
                'num_anchors': 9,
                'bbox_dims': 4,
                'use_attention': False
            },
            VisionTaskType.SEGMENTATION: {
                # 21 classes is the VOC default.
                'num_classes': 21,
                'upsampling_factor': 4,
                'use_skip_connections': True,
                'use_attention': True,
                'attention_type': 'cbam'
            },
            VisionTaskType.DEPTH_ESTIMATION: {
                'output_channels': 1,
                'min_depth': 0.1,
                'max_depth': 100.0,
                'use_log_depth': True,
                'use_attention': False
            },
            VisionTaskType.CLASSIFICATION: {
                # 1000 classes is the ImageNet default. No attention: the
                # 'multi_head' this preset once named takes a (B, N, D)
                # sequence and raised ValueError on the 4D map that 'avg'
                # pooling expects (measured 2026-10-14).
                'num_classes': 1000,
                'use_global_pooling': True,
                'pooling_type': 'avg',
                'use_attention': False,
            },
            VisionTaskType.INSTANCE_SEGMENTATION: {
                'num_classes': 80,
                'num_instances': 100,
                'mask_size': (28, 28),
                'use_attention': True,
                'attention_type': 'cbam'
            },
            VisionTaskType.SURFACE_NORMALS: {'use_attention': False},
            VisionTaskType.OPTICAL_FLOW: {'use_attention': False},
            VisionTaskType.KEYPOINT_DETECTION: {
                # 65 is SuperPoint's detector: 64 positions in an 8x8 cell
                # plus a dustbin, one score vector per cell.
                'num_classes': 65,
                'num_anchors': 1,
                'use_regression': False,
                'use_attention': False,
            },
        }

        config = base_config.copy()
        if task_type in task_specific:
            config.update(task_specific[task_type])
        config.update(PRESET_FIXED_KEYS.get(task_type, {}))

        return _drop_off_stage_knobs(config)

    @staticmethod
    def get_efficient_config(task_type: VisionTaskType) -> Dict[str, Any]:
        """
        Get the lightweight preset for a task type.

        Starts from :meth:`get_default_config` and overrides six keys: a
        narrower ``hidden_dim``, no dropout, no attention, and a GLU FFN with
        a smaller expansion. :data:`PRESET_FIXED_KEYS` is applied last.

        :param task_type: Task the preset is for, as an enum member or its
            string value.
        :type task_type: VisionTaskType
        :return: Keyword arguments for the matching head class.
        :rtype: Dict[str, Any]
        :raises ValueError: If ``task_type`` has no head.
        """

        task_type = _as_vision_task_type(task_type)
        config = HeadConfiguration.get_default_config(task_type)
        config.update({
            'hidden_dim': 128,
            'dropout_rate': 0.0,
            'use_attention': False,
            'use_ffn': True,
            # 'glu' is the more efficient registered FFN type.
            'ffn_type': 'glu',
            'ffn_expansion_factor': 2
        })
        config.update(PRESET_FIXED_KEYS.get(task_type, {}))
        return _drop_off_stage_knobs(config)


# ---------------------------------------------------------------------
# Model-side head_config_overrides merge
# ---------------------------------------------------------------------

#: Head-config keys a model owns and a ``head_config_overrides`` dict may not
#: replace. ``name`` fixes the head's weight paths (by-name weight transfer
#: reads them), ``num_classes`` must agree with the model's own
#: ``num_classes`` and its ``get_config``, and ``task_type`` picks the head
#: class.
RESERVED_HEAD_OVERRIDE_KEYS: Tuple[str, ...] = ('name', 'num_classes', 'task_type')


def merge_head_config_overrides(
        head_config: Dict[str, Any],
        overrides: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Merge a model's ``head_config_overrides`` into its head config.

    Interface contract, shared by every model that builds a vision head from
    a config dict plus caller overrides:

    * ``head_config`` is not mutated. The return value is a new dict.
    * ``overrides`` of ``None`` or ``{}`` returns a plain copy.
    * An override naming any of :data:`RESERVED_HEAD_OVERRIDE_KEYS` raises
      ``ValueError`` naming every offending key. Nothing is merged then.
    * Every other override key replaces or adds the matching entry. Unknown
      keys are not checked here: the head constructor rejects them.

    A plain ``dict.update`` let ``num_classes`` silently desynchronise the
    head from the model's config, and let ``name`` collide with the model's
    own ``name=`` argument as a ``TypeError``.

    :param head_config: The model's head configuration.
    :type head_config: Dict[str, Any]
    :param overrides: Caller overrides, or ``None``.
    :type overrides: Optional[Dict[str, Any]]
    :return: The merged configuration.
    :rtype: Dict[str, Any]
    :raises ValueError: If ``overrides`` names a reserved key.
    """
    merged = dict(head_config)
    if not overrides:
        return merged

    reserved = sorted(set(overrides) & set(RESERVED_HEAD_OVERRIDE_KEYS))
    if reserved:
        raise ValueError(
            f"head_config_overrides may not set {reserved}: the model owns "
            f"{list(RESERVED_HEAD_OVERRIDE_KEYS)}. Pass num_classes to the "
            f"model itself; the head name and task type are fixed."
        )

    merged.update(overrides)
    return merged


def build_model_vision_head(
        task_type: Union[VisionTaskType, str],
        head_config: Dict[str, Any],
        head_config_overrides: Optional[Dict[str, Any]] = None
) -> BaseVisionHead:
    """
    Build a model's vision head in one call: preset, then the model's own
    keys, then the caller's overrides.

    Interface contract, shared by every vision model that owns a task head
    (22 construction sites in ``models/vision`` and FastVLM; FFTNet's three,
    one per task, since D-229):

    * Starts from :meth:`HeadConfiguration.get_default_config` for
      ``task_type`` (so a task type with no head raises ``ValueError`` here).
    * ``head_config`` holds the keys the model owns (``num_classes``,
      ``name``, its reference head shape, its initializers); it replaces or
      adds the matching preset entries. It is not mutated.
    * ``head_config_overrides`` (the model's ``head_config_overrides``
      argument) is merged last through :func:`merge_head_config_overrides`,
      which refuses ``name``, ``num_classes`` and ``task_type``. ``None``
      merges nothing.
    * A preset knob of a stage the merged config turns off
      (``attention_type``; ``ffn_type``, ``ffn_expansion_factor``) is
      dropped, unless ``head_config`` or the overrides set it: then it
      reaches the head, which refuses it (D-210).
    * Returns ``create_vision_head(task_type, **merged)``; an unknown key is
      refused by the head constructor.

    It replaces the four-step ``get_default_config`` -> ``update`` ->
    ``merge_head_config_overrides`` -> ``create_vision_head`` sequence each
    model used to spell out, so the order of the three sources cannot drift
    between models.

    :param task_type: Task the head is for, as an enum member or its string
        value.
    :type task_type: Union[VisionTaskType, str]
    :param head_config: The model's own head keys.
    :type head_config: Dict[str, Any]
    :param head_config_overrides: The caller's overrides, or ``None``.
    :type head_config_overrides: Optional[Dict[str, Any]]
    :return: The configured head.
    :rtype: BaseVisionHead
    :raises ValueError: If ``task_type`` has no head, or the overrides name a
        reserved key.
    """
    # DECISION plan-2026-10-14T062616-81938e81/D-053
    # Preset, then the model's keys, then the caller's overrides: do NOT
    # reorder (an override must win over the model's reference head shape,
    # and the model's num_classes/name over the preset), and do NOT copy these
    # three lines back into a model. See decisions.md D-053.
    config = HeadConfiguration.get_default_config(task_type)
    config.update(head_config)
    config = merge_head_config_overrides(config, head_config_overrides)
    # A preset knob for a stage the model or the caller turned off is dropped
    # with the stage; one they set themselves reaches the head and is refused.
    config = _drop_off_stage_knobs(
        config, keep=(head_config, head_config_overrides or {}))
    return create_vision_head(task_type, **config)
