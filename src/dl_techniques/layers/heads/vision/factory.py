"""Vision task heads, and the factory that picks one.

This module is a thin re-export layer over the vision head sub-modules:

* ``base``            — :class:`BaseVisionHead`
* ``detection``       — :class:`DetectionHead`
* ``segmentation``    — :class:`SegmentationHead`
* ``classification``  — :class:`ClassificationHead`
* ``instance_segmentation`` — :class:`InstanceSegmentationHead`
* ``depth``           — :class:`DepthEstimationHead`
* ``dense_vector``    — :class:`SurfaceNormalsHead`, :class:`OpticalFlowHead`
* ``enhancement``     — :class:`EnhancementHead`
* ``multitask``       — :class:`MultiTaskHead`, :func:`create_multi_task_head`
* ``dispatch``        — :data:`VISION_HEAD_DISPATCH`,
                        :func:`create_vision_head`, :func:`create_enhancement_head`
* ``config``          — :class:`HeadConfiguration`,
                        :func:`merge_head_config_overrides`,
                        :func:`build_model_vision_head`

All class names, registration keys and export names are unchanged from the
single-file ``factory.py`` — only the file layout differs. Import from here
(or from the ``vision`` package) exactly as before.
"""

# ---------------------------------------------------------------------
# Base head
# ---------------------------------------------------------------------

from .base import BaseVisionHead

# ---------------------------------------------------------------------
# Task heads
# ---------------------------------------------------------------------

from .detection import DetectionHead
from .segmentation import SegmentationHead
from .classification import ClassificationHead
from .instance_segmentation import InstanceSegmentationHead
from .depth import DepthEstimationHead
from .dense_vector import SurfaceNormalsHead, OpticalFlowHead
from .enhancement import EnhancementHead
from .multitask import MultiTaskHead, create_multi_task_head

# ---------------------------------------------------------------------
# Dispatch table and factory functions
# ---------------------------------------------------------------------

from .dispatch import (
    VISION_HEAD_DISPATCH,
    ENHANCEMENT_TASK_TYPES,
    create_vision_head,
    create_enhancement_head,
)

# ---------------------------------------------------------------------
# Configuration presets and model-side merge
# ---------------------------------------------------------------------

from .task_types import VisionTaskType

from .config import (
    PRESET_FIXED_KEYS,
    HeadConfiguration,
    RESERVED_HEAD_OVERRIDE_KEYS,
    merge_head_config_overrides,
    build_model_vision_head,
)

# ---------------------------------------------------------------------
# Local helpers used across sub-modules
# ---------------------------------------------------------------------

from .utils import (
    STAGE_KNOB_DEFAULTS,
    _drop_off_stage_knobs,
    _refuse_common_layers_call_never_runs,
    _validated_block_dims,
    _UNRUNNABLE_ATTENTION_TYPES,
    _DEFAULT_BBOX_DIMS,
    _SPATIAL_ATTENTION_TYPES,
    _require_hidden_dim_at_least,
    _refuse_dropout_no_layer_applies,
)

# ---------------------------------------------------------------------
# Progressive decoder
# ---------------------------------------------------------------------

from .decoder import (
    _PROGRESSIVE_DOUBLINGS,
    _create_progressive_decoder,
    _build_progressive_decoder,
    _run_progressive_decoder,
)
