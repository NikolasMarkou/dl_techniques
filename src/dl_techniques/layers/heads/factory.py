"""Top-level ``create_head`` dispatch for the ``heads`` package.

One function lives here. :func:`create_head` picks a domain factory and calls
it. The three domains are ``nlp``, ``vision`` and ``vlm``, and each has its own
factory inside its own sub-package.

The three domain factories take different arguments. This module does not
change that. It forwards every positional and keyword argument to the chosen
factory unchanged.

Scope
-----
Only the single-head ``create_*_head`` factories are reachable through this
module. Multi-task heads take domain-specific ``task_configs`` arguments, so
callers build them by calling ``create_multi_task_nlp_head``,
``create_multi_task_head`` or ``create_multi_task_vlm_head`` directly.

Design note
-----------
The plan that added this facade (``plan_2026-06-08_8b32ca51``, D-004) accepted
it as a near-single-use abstraction. It charged the facade 1/2 of that plan's
complexity budget. It stays thin. The three domain factories keep different
calling conventions. ``nlp`` takes ``task_config`` and ``input_dim``.
``vision`` takes a ``task_type`` plus head kwargs. ``vlm`` takes
``task_config`` plus ``vision_dim`` and ``text_dim``. They stay independent.
Do not add a unified parameter set here, and do not merge the three
``Base*Head`` classes. That plan's directory has been deleted, so this
paragraph and the anchor above :func:`create_head` are the only surviving
record of the decision.

Example
-------
>>> from dl_techniques.layers.heads import create_head
>>> # nlp: task_config + input_dim
>>> nlp = create_head('nlp', task_config=cfg, input_dim=768)
>>> # vision: task_type (+ head kwargs)
>>> vis = create_head('vision', 'classification', num_classes=10)
>>> # vlm: task_config (+ vision_dim/text_dim/...)
>>> vlm = create_head('vlm', task_config=vlm_cfg, vision_dim=768, text_dim=768)
"""

from typing import Any, Literal

from dl_techniques.utils.logger import logger

from .nlp import create_nlp_head
from .vision import create_vision_head
from .vlm import create_vlm_head

# Supported head domains for the dispatch facade.
HeadDomain = Literal['nlp', 'vision', 'vlm']

_VALID_DOMAINS = ('nlp', 'vision', 'vlm')


# DECISION plan_2026-06-08_8b32ca51/D-004: this facade stays a thin dispatcher.
# Do NOT add signature unification, a unified parameter set, or a Base*Head
# merge here. The three domain factories keep divergent calling conventions and
# stay independent. The owning plan directory no longer exists, so the module
# docstring above carries the rest; there is no decisions.md left to consult.
def create_head(domain: HeadDomain, *args: Any, **kwargs: Any) -> Any:
    """Create a task head by domain, forwarding to the per-domain factory.

    The function checks ``domain`` against the three supported values, then
    calls that domain's own factory. Every remaining positional and keyword
    argument is forwarded unchanged. The three domain signatures are not
    unified here.

    **Architecture Overview:**

    .. code-block:: text

        create_head(domain, *args, **kwargs)
                │
                ▼
        ┌─────────────────┐            ┌────────────┐
        │ validate domain │─ unknown ─►│ ValueError │
        └───────┬─────────┘            └────────────┘
                │ ok
                ▼
            ┌───────────────────────────────────────────────┐
            │ select the domain factory                     │
            │ forward *args / **kwargs verbatim             │
            └───┬───────────────┬───────────────────┬───────┘
                │               │                   │
              'nlp'         'vision'              'vlm'
                ▼               ▼                   ▼
        ┌───────────────┐ ┌──────────────────┐ ┌───────────────┐
        │create_nlp_head│ │create_vision_head│ │create_vlm_head│
        └───────────────┘ └──────────────────┘ └───────────────┘

    :param domain: One of ``'nlp'``, ``'vision'``, ``'vlm'``.
    :type domain: HeadDomain
    :param args: Positional arguments forwarded verbatim to the domain factory.
    :param kwargs: Keyword arguments forwarded verbatim to the domain factory.
    :return: The configured head layer produced by the selected domain factory.
    :rtype: Any
    :raises ValueError: If ``domain`` is not one of the supported domains.

    Per-domain calling conventions (args/kwargs are forwarded to these):

    - ``nlp``   -> :func:`~dl_techniques.layers.heads.nlp.create_nlp_head`::

          create_head('nlp', task_config=cfg, input_dim=768)

    - ``vision`` -> :func:`~dl_techniques.layers.heads.vision.create_vision_head`::

          create_head('vision', 'classification', num_classes=10)

    - ``vlm``   -> :func:`~dl_techniques.layers.heads.vlm.create_vlm_head`::

          create_head('vlm', task_config=vlm_cfg, vision_dim=768, text_dim=768)

    .. note::
       Multi-task heads are not dispatched here. Build them by calling
       ``create_multi_task_nlp_head``, ``create_multi_task_head`` or
       ``create_multi_task_vlm_head`` directly.
    """
    if domain not in _VALID_DOMAINS:
        raise ValueError(
            f"Unknown head domain '{domain}'. "
            f"Available domains: {list(_VALID_DOMAINS)}"
        )

    logger.debug(f"create_head dispatching to domain '{domain}'")

    if domain == 'nlp':
        return create_nlp_head(*args, **kwargs)
    elif domain == 'vision':
        return create_vision_head(*args, **kwargs)
    # The only remaining valid domain is 'vlm'.
    else:
        return create_vlm_head(*args, **kwargs)
