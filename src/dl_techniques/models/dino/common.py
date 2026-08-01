"""Shared internals for the three DINO factory functions.

This module exists for exactly one reason: `create_dino_v1`, `create_dino_v2` and
`create_dino_v3` converged on a single parameter scheme (see
`src/dl_techniques/models/dino/README.md` § "Factory signatures"), and one piece of
that scheme — refusing the removed `input_shape` spelling — is identical in all three.
It is imported by `src/dl_techniques/models/dino/dino_v1.py`,
`src/dl_techniques/models/dino/dino_v2.py` and
`src/dl_techniques/models/dino/dino_v3.py`.

Do NOT grow this into a shared ViT trunk or a shared `MODEL_VARIANTS` table — that
unification is a deliberate, recorded non-goal (plan decision D-003: the three model
files have no test suite dense enough to prove a behaviour-preserving merge).
"""

from typing import Any, Dict

# ---------------------------------------------------------------------

__all__ = ["reject_input_shape"]

# ---------------------------------------------------------------------


def reject_input_shape(kwargs: Dict[str, Any], factory_name: str) -> None:
    """Raise if a caller passed the removed ``input_shape`` factory argument.

    Interface contract:
        Parameters:
            kwargs: The factory's own ``**kwargs`` dict. Inspected, never mutated.
            factory_name: The calling factory's name, quoted back in the message so
                the error names the call site the user actually wrote.
        Returns:
            ``None`` when ``input_shape`` is absent.
        Failure mode:
            ``TypeError`` when ``input_shape`` is present. It is a ``TypeError`` and
            not a ``ValueError`` because, from the caller's point of view, this is an
            unexpected keyword argument — the same class of failure Python raises for
            a name a function does not accept.

    Why this refusal exists rather than a silent pass-through: the ``DINOv1`` and
    ``DINOv2`` CONSTRUCTORS still accept ``input_shape`` as a lower-level escape
    hatch, so an ``input_shape`` left in a factory call would flow through
    ``**kwargs`` and reach the constructor, where it can DISAGREE with
    ``image_size``. That disagreement is the measured silent defect recorded as
    plan decision D-013: construction succeeds with the wrong patch count and the
    model only fails (or worse, does not fail) much later.
    """
    if "input_shape" in kwargs:
        raise TypeError(
            f"{factory_name}() no longer accepts 'input_shape'; use 'image_size' "
            f"instead. The input shape is derived as (*image_size, in_channels). "
            f"Passing both spellings allowed them to disagree, which built a model "
            f"with a patch grid that did not match its input."
        )

# ---------------------------------------------------------------------
