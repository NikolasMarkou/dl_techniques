"""Top-level ``create_head`` dispatch for the ``heads`` package.

One function lives here. :func:`create_head` picks a domain factory and calls
it. The four domains are ``nlp``, ``vision``, ``vlm`` and ``time_series``, and
each has its own factory inside its own sub-package.

The four domain factories take different arguments. This module does not
change that. It forwards every positional and keyword argument to the chosen
factory unchanged.

Scope
-----
Only the single-head ``create_*_head`` factories are reachable through this
module. Multi-task heads take domain-specific ``task_configs`` arguments, so
callers build them by calling ``create_multi_task_nlp_head``,
``create_multi_task_head`` or ``create_multi_task_vlm_head`` directly. The
``time_series`` domain has no multi-task head.

Design note
-----------
The plan that added this facade (``plan_2026-06-08_8b32ca51``, D-004) accepted
it as a near-single-use abstraction. It charged the facade 1/2 of that plan's
complexity budget. It stays thin. The domain factories keep different
calling conventions. ``nlp`` takes ``task_config`` and ``input_dim``.
``vision`` takes a ``task_type`` plus head kwargs. ``vlm`` takes
``task_config`` plus head kwargs. ``time_series``, added on 2026-10-31, takes a
``task_type`` plus head kwargs. They stay independent.
Do not add a unified parameter set here, and do not merge the three
``Base*Head`` classes. That plan's directory has been deleted, so this
paragraph and the anchor above :func:`create_head` are the only surviving
record of the decision.

Consumers
---------
No model calls :func:`create_head`. The language models build their heads
with ``nlp.model_head.build_model_task_head`` and the vision models with
``vision.factory.build_model_vision_head``; each adds a domain rule this
facade cannot know (a model's vocabulary, a preset merge). The forecasting
models call ``time_series.create_time_series_head``. The facade is kept
as the library's one documented entry point for code outside this
repository's models, and ``tests/test_layers/test_heads/test_factory.py``
pins that it forwards every argument (plan-2026-10-14T062616-81938e81 D-067,
which weighed deleting it).

Example
-------
>>> from dl_techniques.layers.heads import create_head
>>> # nlp: task_config + input_dim
>>> nlp = create_head('nlp', task_config=cfg, input_dim=768)
>>> # vision: task_type (+ head kwargs)
>>> vis = create_head('vision', 'classification', num_classes=10)
>>> # vlm: task_config (+ head kwargs)
>>> vlm = create_head('vlm', task_config=vlm_cfg, num_layers=2)
>>> # time_series: task_type (+ head kwargs)
>>> ts = create_head('time_series', 'quantile_sequence', num_quantiles=3)
"""

from typing import Any, Literal

from dl_techniques.utils.logger import logger

from .nlp import create_nlp_head
from .vision import create_vision_head
from .vision_language import create_vlm_head
from .time_series import create_time_series_head

# Supported head domains for the dispatch facade.
HeadDomain = Literal['nlp', 'vision', 'vlm', 'time_series']

_VALID_DOMAINS = ('nlp', 'vision', 'vlm', 'time_series')


# DECISION plan_2026-06-08_8b32ca51/D-004: this facade stays a thin dispatcher.
# Do NOT add signature unification, a unified parameter set, or a Base*Head
# merge here. The three domain factories keep divergent calling conventions and
# stay independent. The owning plan directory no longer exists, so the module
# docstring above carries the rest; there is no decisions.md left to consult.
# DECISION plan-2026-10-14T062616-81938e81/D-067: kept, with no production
# caller, as the documented library entry point. Do NOT route a model's head
# through it (models need build_model_task_head / build_model_vision_head), and
# do NOT delete it without the docs that name it. See decisions.md D-067.
# NOTE 2026-10-31: 'time_series' joined as a fourth domain. D-004 and D-067 hold
# for it too: forward verbatim, and the forecasting models call
# create_time_series_head, not this facade.
def create_head(domain: HeadDomain, *args: Any, **kwargs: Any) -> Any:
    """Create a task head by domain, forwarding to the per-domain factory.

    The function checks ``domain`` against the four supported values, then
    calls that domain's own factory. Every remaining positional and keyword
    argument is forwarded unchanged. The four domain signatures are not
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
            └───┬───────────────┬───────────────┬───────┬───┘
                │               │               │       │
              'nlp'         'vision'          'vlm'  'time_series'
                ▼               ▼               ▼       ▼
          create_nlp_head  create_vision_head   │  create_time_series_head
                                                ▼
                                          create_vlm_head

    :param domain: One of ``'nlp'``, ``'vision'``, ``'vlm'``, ``'time_series'``.
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

    - ``vlm``   -> :func:`~dl_techniques.layers.heads.vision_language.create_vlm_head`::

          create_head('vlm', task_config=vlm_cfg, num_layers=2)

    - ``time_series`` -> :func:`~dl_techniques.layers.heads.time_series.create_time_series_head`::

          create_head('time_series', 'quantile_sequence', num_quantiles=3)

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
    elif domain == 'vlm':
        return create_vlm_head(*args, **kwargs)
    elif domain == 'time_series':
        return create_time_series_head(*args, **kwargs)
    else:
        raise ValueError(f"Unknown domain '{domain}'")
