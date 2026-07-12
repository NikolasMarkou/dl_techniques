"""Load-time provenance gate for bias-free denoiser checkpoints.

Why this module exists
----------------------
A bias-free denoiser is degree-1 homogeneous with ``f(0) = 0``: it has NO mechanism
to subtract a DC offset. Feeding ``[0,1]`` data to a network trained on the legacy
``[-0.5,+0.5]`` pixel domain (or vice versa) therefore produces **silent garbage** —
a complete, finite, plausible-looking WRONG image, with no exception and no NaN.
There is no in-band signal that can detect this after the fact, so the only defense
is a *load-time* refusal driven by the checkpoint's recorded provenance.

Every ``[0,1]``-trained run stamps ``data_range: "[0,1]"`` into its ``config.json``
(``BFUnetTrainingConfig.data_range``, ``src/train/bfunet/common.py``). Any checkpoint
without that stamp predates the unit-domain migration and is refused.

Why it lives in ``dl_techniques.utils`` and not in ``src/train/bfunet/common.py``
---------------------------------------------------------------------------------
The gate has FOUR callers on two sides of a deliberate dependency boundary:

* ``src/train/bfunet/eval_psnr_vs_noise.py``   (``load_denoiser``)
* ``src/train/bfunet/eval_per_pixel_uncertainty.py`` (``_load_denoiser``)
* ``src/train/bfunet/common.py``               (the ``--init-from`` warm-start)
* ``src/applications/bias_free_denoiser/denoiser_prior.py`` (``from_pretrained``)

``src/applications/`` must not import ``src/train/`` (an established boundary — see
``denoiser_prior.py``'s ``_build_dynamic``, which deliberately replicates the trainer's
kwargs mapping rather than import it; ``common.py`` also drags in TensorFlow, matplotlib,
``train.common`` and ``train.superpoint``). ``dl_techniques.utils`` is the neutral
package BOTH sides already import from (``utils.logger``, ``utils.weight_transfer``),
so it is the only home that gives all four call sites ONE implementation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

from dl_techniques.utils.logger import logger

# The exact stamp a checkpoint's config.json must carry to be loadable. Written by
# BFUnetTrainingConfig.data_range (src/train/bfunet/common.py) via save_config_json.
UNIT_DOMAIN_STAMP = "[0,1]"

_CONFIG_JSON_NAME = "config.json"

# Sentinel distinguishing "key absent" from "key present but null" in the refusal
# message. Both REFUSE — only the wording differs.
_MISSING = object()


def resolve_config_path(checkpoint_path: Union[str, Path]) -> Path:
    """Resolve the sibling ``config.json`` of a checkpoint path.

    Args:
        checkpoint_path: Either a ``.keras`` file, or a training results directory
            containing ``best_model.keras`` + ``config.json``.

    Returns:
        The path where the checkpoint's ``config.json`` is expected to live. The
        file is NOT required to exist (an absent config is itself a refusal, not a
        lookup failure — see :func:`require_unit_domain_checkpoint`).
    """
    p = Path(checkpoint_path)
    parent = p if p.is_dir() else p.parent
    return parent / _CONFIG_JSON_NAME


def _read_config(config_path: Path) -> Optional[dict]:
    """Parse a checkpoint's ``config.json``, or return ``None`` if it is unusable.

    Fails CLOSED on every degenerate case: absent file, unreadable file, malformed
    JSON, and valid-but-non-object JSON (e.g. ``[1, 2]``, whose ``.get`` would raise
    ``AttributeError`` instead of the intended loud ``ValueError``) all collapse to
    ``None``.
    """
    try:
        raw = json.loads(Path(config_path).read_text())
    except (OSError, ValueError):
        return None
    return raw if isinstance(raw, dict) else None


def require_unit_domain_checkpoint(checkpoint_path: Union[str, Path]) -> None:
    """Refuse any checkpoint not stamped ``data_range == "[0,1]"``.

    # DECISION plan_2026-07-12_e56909cd/D-005: ONE shared gate, called from ALL FOUR
    # checkpoint-load paths (the two eval tools, the trainer's --init-from warm-start,
    # and DenoiserPrior.from_pretrained). Do NOT re-implement this check at a call site:
    # the whole hazard class here is a *partially* migrated data path, and a gate that
    # covers 3 of 4 loaders is a gate that reports plausible wrong dB numbers from the
    # 4th. Do NOT add an `allow_legacy` / `--force` escape hatch either: a switch whose
    # only purpose is to re-enable a knowingly-broken path is a compat shim by another
    # name, and the migration mandate forbids one (INV-4). This gate never changes any
    # math — it only refuses. An ABSENT data_range key must FAIL CLOSED (legacy =>
    # refuse), because a bias-free net is degree-1 homogeneous with f(0) = 0 and cannot
    # subtract a DC offset: a legacy checkpoint fed [0,1] data emits a plausible WRONG
    # image, never an error (INV-1). The fix for a refused checkpoint is to RETRAIN.
    # See decisions.md D-005.

    Args:
        checkpoint_path: A ``.keras`` file, or the results directory holding it.

    Raises:
        ValueError: If the sibling ``config.json`` is missing, unreadable, malformed,
            carries no ``data_range`` key, or records anything other than ``"[0,1]"``.
    """
    config_path = resolve_config_path(checkpoint_path)
    raw = _read_config(config_path)
    found = _MISSING if raw is None else raw.get("data_range", _MISSING)
    if found == UNIT_DOMAIN_STAMP:
        return

    if raw is None and not config_path.is_file():
        why = f"its config.json is missing or unreadable ({config_path})"
    elif raw is None:
        why = f"its config.json is unreadable or malformed ({config_path})"
    elif found is _MISSING:
        why = f"its config.json ({config_path}) records NO data_range key at all"
    else:
        why = f"its config.json records data_range={found!r}"

    logger.error("provenance gate REFUSED checkpoint %s: %s", checkpoint_path, why)
    raise ValueError(
        f"REFUSING to load denoiser checkpoint {config_path.parent}: {why}, but this "
        f"code path requires data_range={UNIT_DOMAIN_STAMP!r}.\n"
        f"An absent data_range key means the checkpoint predates the unit-domain "
        f"migration, i.e. it was trained on the LEGACY [-0.5,+0.5] pixel domain. A "
        f"bias-free denoiser is degree-1 homogeneous and cannot subtract a DC offset, "
        f"so running it on [0,1] data yields SILENT garbage rather than an error. "
        f"A RETRAIN on the [0,1] domain is required; there is no compatibility shim."
    )
