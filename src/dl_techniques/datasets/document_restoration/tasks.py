"""The single task-specification table for DocRes.

DocRes's network is not task-conditioned in any way: ``Restormer.forward``
accepts a ``task`` string and never reads it. All task specificity lives in
three places -- which prompt is concatenated onto the RGB input, which output
channels are supervised and by what loss, and what post-processing turns the
prediction into a picture.

Upstream spreads those three decisions across ``inference.py``, ``eval.py``,
``train.py`` and ``loaders/docres_loader.py``, with the prompt recipes
duplicated verbatim three times. **This table is the one place they live in
this port.** Nothing else in the tree may branch on a task string: the staging
script, the training pipeline and the inference shim all resolve a name through
:func:`get_task` and read fields off the returned :class:`TaskSpec`.

The table is deliberately free of Keras: ``loss`` and ``postprocess`` are
*names*, not objects. That keeps this module importable by the offline sidecar
precompute (which needs the prompt generators and no deep-learning framework)
and keeps the table a data structure rather than a hidden dependency edge. The
consumer maps the name to a ``keras.losses`` object or a post-processing
function through a dispatch dict keyed by the constants exported here -- so a
typo is an import error, not a silent fallthrough.

Public surface:
    * :class:`TaskSpec` — one task's binding.
    * :data:`TASKS` — the immutable name -> spec mapping.
    * :func:`get_task` — resolve a name, raising with the legal keys.
    * :func:`task_names` — the legal names, in table order.
    * The ``LOSS_*`` / ``POSTPROCESS_*`` name constants and their frozensets.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Tuple

from dl_techniques.utils.logger import logger

from .dtsprompt import (
    appearance_prompt,
    binarization_prompt,
    deblur_prompt,
    deshadow_prompt,
    dewarp_prompt,
)

# ---------------------------------------------------------------------------
# Loss names. Sourced from DocRes `train.py:144-159`.
# ---------------------------------------------------------------------------

# DECISION plan-2026-09-08T111844-de235227/D-022
# These are NAMES, not `keras.losses` objects, and `postprocess` below is a
# name rather than a callable. Do NOT "improve" them into live objects: this
# module is imported by the offline prompt-sidecar precompute, which needs the
# generators and must not drag in a deep-learning framework, and by the
# inference shim, which needs the post-processing and not the loss. Binding
# either eagerly makes the table a dependency edge instead of a data
# structure. Consumers map name -> object through a dispatch dict keyed by
# these constants, so a typo is an import error. See D-022 in decisions.md.
LOSS_L1: str = "l1"
"""Mean absolute error over the supervised channel slice."""

LOSS_CATEGORICAL_CROSSENTROPY: str = "categorical_crossentropy"
"""Two-class cross-entropy over the supervised channel slice, treated as logits."""

LOSSES: frozenset = frozenset({LOSS_L1, LOSS_CATEGORICAL_CROSSENTROPY})

# ---------------------------------------------------------------------------
# Post-processing mode names. Sourced from DocRes `inference.py:118-251`.
# ---------------------------------------------------------------------------

POSTPROCESS_CLAMP_IMAGE: str = "clamp_image"
"""``clip(pred, 0, 1) * 255 -> uint8``. Above 1600 px the caller additionally
applies the homomorphic re-composition (``inference.py:163-166``)."""

POSTPROCESS_ARGMAX_BINARY: str = "argmax_binary"
"""``argmax`` over the two supervised channels, scaled to ``{0, 255}``."""

POSTPROCESS_FLOW_REMAP: str = "flow_remap"
"""Add the base-coordinate grid to the two predicted flow channels, box-blur,
rescale to the source resolution and backward-warp the original page."""

POSTPROCESS_MODES: frozenset = frozenset(
    {POSTPROCESS_CLAMP_IMAGE, POSTPROCESS_ARGMAX_BINARY, POSTPROCESS_FLOW_REMAP}
)

# ---------------------------------------------------------------------------

N_OUTPUT_CHANNELS: int = 3
"""The network head is 3 channels for **every** task (``restormer_arch.py``'s
``out_channels=3``), even where only 2 are supervised. The unsupervised third
channel is architecturally present and never trained."""

N_PROMPT_CHANNELS: int = 3
"""Every prompt is 3 channels, so the network input is always ``3 + 3 = 6``."""


@dataclass(frozen=True)
class TaskSpec:
    """Everything that distinguishes one DocRes task from another.

    Attributes:
        name: The task's canonical name; equals its key in :data:`TASKS`.
        prompt_fn: The DTSPrompt generator. Called as ``prompt_fn(img)`` unless
            ``requires_mask`` is set, in which case ``prompt_fn(img, mask)``.
        requires_mask: Whether ``prompt_fn`` needs a caller-supplied document
            mask. True only for dewarping, whose mask comes from the MBD
            network that this port does not include.
        prompt_dtype: ``"uint8"`` or ``"float32"`` -- the dtype and hence the
            range (``[0, 255]`` or ``[0, 1]``) ``prompt_fn`` returns. The
            pipeline must know this before it normalises.
        n_supervised_channels: How many of the model's 3 output channels the
            loss sees, counted from channel 0. The remainder are unsupervised.
        loss: One of :data:`LOSSES`.
        postprocess: One of :data:`POSTPROCESS_MODES`.
    """

    name: str
    prompt_fn: Callable[..., Any]
    requires_mask: bool
    prompt_dtype: str
    n_supervised_channels: int
    loss: str
    postprocess: str

    @property
    def supervised_slice(self) -> slice:
        """The channel slice the loss is computed over.

        Returns:
            ``slice(0, n_supervised_channels)``.
        """
        return slice(0, self.n_supervised_channels)


_TASK_LIST: Tuple[TaskSpec, ...] = (
    # `train.py:149` -> L1 on channels [:2] only; the loaded mask ground truth
    # is never used in the loss, so the 3rd channel is unsupervised.
    TaskSpec(
        name="dewarping",
        prompt_fn=dewarp_prompt,
        requires_mask=True,
        prompt_dtype="float32",
        n_supervised_channels=2,
        loss=LOSS_L1,
        postprocess=POSTPROCESS_FLOW_REMAP,
    ),
    TaskSpec(
        name="deshadowing",
        prompt_fn=deshadow_prompt,
        requires_mask=False,
        prompt_dtype="uint8",
        n_supervised_channels=3,
        loss=LOSS_L1,
        postprocess=POSTPROCESS_CLAMP_IMAGE,
    ),
    TaskSpec(
        name="appearance",
        prompt_fn=appearance_prompt,
        requires_mask=False,
        prompt_dtype="uint8",
        n_supervised_channels=3,
        loss=LOSS_L1,
        postprocess=POSTPROCESS_CLAMP_IMAGE,
    ),
    TaskSpec(
        name="deblurring",
        prompt_fn=deblur_prompt,
        requires_mask=False,
        prompt_dtype="uint8",
        n_supervised_channels=3,
        loss=LOSS_L1,
        postprocess=POSTPROCESS_CLAMP_IMAGE,
    ),
    # `train.py:146` -> CrossEntropy on channels [:2] as a 2-class logit pair,
    # `inference.py:249-251` -> softmax + argmax over the same two.
    TaskSpec(
        name="binarization",
        prompt_fn=binarization_prompt,
        requires_mask=False,
        prompt_dtype="uint8",
        n_supervised_channels=2,
        loss=LOSS_CATEGORICAL_CROSSENTROPY,
        postprocess=POSTPROCESS_ARGMAX_BINARY,
    ),
)


def _validate(specs: Tuple[TaskSpec, ...]) -> None:
    """Fail at import time on a malformed table.

    A table this small is read far more often than it is written, and a bad
    field would otherwise surface as a confusing failure inside a training run.

    Args:
        specs: The candidate task specs.

    Raises:
        ValueError: On a duplicate name, an unknown loss or post-processing
            mode, a non-string dtype tag, or a supervised-channel count outside
            ``1..N_OUTPUT_CHANNELS``.
    """
    seen = set()
    for spec in specs:
        if spec.name in seen:
            raise ValueError(f"duplicate task name in TASKS: {spec.name!r}")
        seen.add(spec.name)
        if spec.loss not in LOSSES:
            raise ValueError(
                f"task {spec.name!r} declares unknown loss {spec.loss!r}; "
                f"legal: {sorted(LOSSES)}"
            )
        if spec.postprocess not in POSTPROCESS_MODES:
            raise ValueError(
                f"task {spec.name!r} declares unknown postprocess "
                f"{spec.postprocess!r}; legal: {sorted(POSTPROCESS_MODES)}"
            )
        if spec.prompt_dtype not in ("uint8", "float32"):
            raise ValueError(
                f"task {spec.name!r} declares unknown prompt_dtype "
                f"{spec.prompt_dtype!r}"
            )
        if not 1 <= spec.n_supervised_channels <= N_OUTPUT_CHANNELS:
            raise ValueError(
                f"task {spec.name!r} supervises {spec.n_supervised_channels} of "
                f"{N_OUTPUT_CHANNELS} output channels, which is out of range"
            )


_validate(_TASK_LIST)

TASKS: Mapping[str, TaskSpec] = MappingProxyType(
    {spec.name: spec for spec in _TASK_LIST}
)
"""Immutable task name -> :class:`TaskSpec`. The single home of DocRes task
specificity; a ``MappingProxyType`` so a consumer cannot mutate it in place."""


def task_names() -> Tuple[str, ...]:
    """The legal task names, in table order.

    Returns:
        Tuple of task names.
    """
    return tuple(TASKS.keys())


def get_task(name: str) -> TaskSpec:
    """Resolve a task name to its spec.

    Args:
        name: A DocRes task name.

    Returns:
        The matching :class:`TaskSpec`.

    Raises:
        ValueError: If ``name`` is not a key of :data:`TASKS`. The message
            lists the legal names, so a CLI typo is self-correcting.
    """
    try:
        return TASKS[name]
    except KeyError:
        raise ValueError(
            f"unknown DocRes task {name!r}; legal tasks: {list(task_names())}"
        ) from None


logger.debug("DocRes TASKS table loaded with %d tasks: %s", len(TASKS), task_names())
