"""The single task-specification table for DocRes.

DocRes's network is not task-conditioned in any way: ``Restormer.forward``
accepts a ``task`` string and never reads it. All task specificity lives in
three places -- which prompt is concatenated onto the RGB input, which output
channels are supervised and by what loss, and what post-processing turns the
prediction into a picture.

Upstream spreads those three decisions across ``inference.py``, ``eval.py``,
``train.py`` and ``loaders/docres_loader.py``, with the prompt recipes
duplicated verbatim three times. **This table is the one place they live in
this port.** No consumer may *branch* on a task string -- no ``== "name"``
comparison and no dict keyed by task names: the staging script, the training
pipeline and the inference shim all resolve a name through :func:`get_task` and
read fields off the returned :class:`TaskSpec`. That is the invariant, and it is
exactly what ``test_pipeline.py::test_no_task_string_is_compared_against_in_
the_shipped_train_doc_res_modules`` checks, over every module in
``src/train/doc_res/``.

It is deliberately narrower than "no task name appears outside this table". One
task-name literal legitimately lives elsewhere: ``infer_doc_res.END2END_STAGES``
names the three stages of the composite ``end2end`` mode, in order, because that
ORDER is a property of the composite pipeline (``inference.py:305-312``) and not
of any single task. It selects nothing and branches on nothing -- every element
is still resolved through :func:`get_task`.

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
    * :data:`PROMPT_SIDECAR_SUFFIXES` — the one dtype -> file-suffix mapping,
      shared by the sidecar writer and the sidecar reader.
    * :data:`CLAMP_MAX_INPUT_SIZE` — the value behind
      :attr:`TaskSpec.max_input_size` for the two tasks that carry it.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Tuple

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
"""``clip(pred, 0, 1) * 255 -> uint8``. A task that also declares a
:attr:`TaskSpec.max_input_size` gets the homomorphic re-composition above it
(``inference.py:163-166``); the re-composition is bound to that FIELD, not to
this mode, because it models a multiplicative degradation only. See D-033."""

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

PROMPT_SIDECAR_SUFFIXES: Mapping[str, str] = MappingProxyType(
    {"uint8": ".png", "float32": ".npy"}
)
"""``prompt_dtype`` -> the file suffix its precomputed sidecar is written with.

The ONE home of that mapping: the staging script that WRITES sidecars and the
training pipeline that READS them both key off this dict, so a suffix known in
two places cannot drift into a writer/reader mismatch. It is also the authority
:func:`_validate` checks ``prompt_dtype`` against -- a dtype with no suffix has
no on-disk representation and is therefore not a legal declaration."""

CLAMP_MAX_INPUT_SIZE: int = 1600
"""Longest edge above which upstream runs a ``clamp_image`` task at
``1600x1600`` and rebuilds full resolution homomorphically
(``inference.py:150-166``). The value of :attr:`TaskSpec.max_input_size` for
the two tasks upstream applies it to."""


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
        max_input_size: Longest edge at or above which inference runs the page
            downscaled and rebuilds full resolution by homomorphic
            re-composition, or ``None`` for a task that is always padded to a
            multiple of 8 and run at full resolution. ``None`` is the default
            because the re-composition is a multiplicative-field model, which
            is right only for a multiplicative degradation. See the D-033
            anchor below.
    """

    name: str
    prompt_fn: Callable[..., Any]
    requires_mask: bool
    prompt_dtype: str
    n_supervised_channels: int
    loss: str
    postprocess: str
    max_input_size: Optional[int] = None

    @property
    def supervised_slice(self) -> slice:
        """The channel slice the loss is computed over.

        Returns:
            ``slice(0, n_supervised_channels)``.
        """
        return slice(0, self.n_supervised_channels)


# DECISION plan-2026-09-08T111844-de235227/D-033
# `max_input_size` is per TASK, not per post-processing mode, and deblurring's
# `None` is LOAD-BEARING. Do NOT "tidy" the three clamp_image tasks into one
# shared cap. At or above the cap the inference shim does not merely downscale:
# it treats the low-resolution prediction as a multiplicative illumination
# FIELD and rebuilds the page as `original / resize(resize(original)/pred)`
# (`infer_doc_res._postprocess_clamp_image`). That model is right for a
# multiplicative degradation (a shadow, an appearance cast) and physically
# wrong for a convolutional one (a blur) -- which is why upstream's
# `deblurring()` (`inference.py:200-227`) has no MAX_SIZE branch at all and
# always pads. Most document photographs exceed 1600 px, so this is the common
# path for deblurring, not an edge case. See D-033 in decisions.md.
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
        max_input_size=CLAMP_MAX_INPUT_SIZE,
    ),
    TaskSpec(
        name="appearance",
        prompt_fn=appearance_prompt,
        requires_mask=False,
        prompt_dtype="uint8",
        n_supervised_channels=3,
        loss=LOSS_L1,
        postprocess=POSTPROCESS_CLAMP_IMAGE,
        max_input_size=CLAMP_MAX_INPUT_SIZE,
    ),
    TaskSpec(
        name="deblurring",
        prompt_fn=deblur_prompt,
        requires_mask=False,
        prompt_dtype="uint8",
        n_supervised_channels=3,
        loss=LOSS_L1,
        postprocess=POSTPROCESS_CLAMP_IMAGE,
        # `inference.py:200-227`: upstream's `deblurring()` has NO MAX_SIZE
        # branch. A blur is a convolution, not a multiplicative field, so the
        # homomorphic re-composition the cap triggers would be the wrong model.
        max_input_size=None,
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
            mode, a dtype tag with no entry in
            :data:`PROMPT_SIDECAR_SUFFIXES`, a supervised-channel count outside
            ``1..N_OUTPUT_CHANNELS``, or a non-positive ``max_input_size``.
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
        if spec.prompt_dtype not in PROMPT_SIDECAR_SUFFIXES:
            raise ValueError(
                f"task {spec.name!r} declares unknown prompt_dtype "
                f"{spec.prompt_dtype!r}"
            )
        if not 1 <= spec.n_supervised_channels <= N_OUTPUT_CHANNELS:
            raise ValueError(
                f"task {spec.name!r} supervises {spec.n_supervised_channels} of "
                f"{N_OUTPUT_CHANNELS} output channels, which is out of range"
            )
        if spec.max_input_size is not None and spec.max_input_size <= 0:
            raise ValueError(
                f"task {spec.name!r} declares max_input_size "
                f"{spec.max_input_size!r}; it must be a positive pixel count "
                "or None"
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
