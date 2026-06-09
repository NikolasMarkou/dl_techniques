"""Top-level ``create_head`` dispatch facade for the merged ``heads`` package.

This module is a thin *domain dispatcher* over the three per-domain head
factories (``nlp``, ``vision``, ``vlm``). It does **not** unify their
signatures: each domain keeps its own native calling convention, and the
remaining positional/keyword arguments are forwarded verbatim to the selected
domain factory (see decisions.md D-004).

Scope
-----
The facade covers the *single-head* ``create_*_head`` factories only. Multi-task
heads have domain-specific argument shapes (``task_configs`` lists/dicts) and are
created directly via the domain functions
(``create_multi_task_nlp_head`` / ``create_multi_task_head`` /
``create_multi_task_vlm_head``) — they are intentionally not routed here.

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


# DECISION plan_2026-06-08_8b32ca51/D-004: This facade is a deliberate
# near-single-use abstraction (charged 1/2 of the Complexity Budget). It is a
# THIN dispatcher only — it must NOT grow signature-unification logic across
# domains. The three domain factories have intentionally divergent calling
# conventions (nlp: task_config+input_dim; vision: task_type+kwargs; vlm:
# task_config+kwargs) and stay independent per D-001. Do NOT add a "unified"
# parameter set or a Base*Head merge here; forward args verbatim. See D-004.
def create_head(domain: HeadDomain, *args: Any, **kwargs: Any) -> Any:
    """Create a task head by domain, forwarding to the per-domain factory.

    This is a thin dispatch shim. It selects the domain factory and forwards all
    remaining positional and keyword arguments verbatim — each domain keeps its
    own native calling convention (no signature unification; see D-004).

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
       Multi-task heads are not dispatched here; build them via the domain
       functions (``create_multi_task_nlp_head`` / ``create_multi_task_head`` /
       ``create_multi_task_vlm_head``) directly.
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
    else:  # domain == 'vlm'
        return create_vlm_head(*args, **kwargs)
