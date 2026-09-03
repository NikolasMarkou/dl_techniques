"""Variant registry and diffusion-side configuration for the DiT port.

This module holds the two things every other module of the package reads and
neither of them owns: the twelve named DiT sizes published with the paper, and
the diffusion-side knobs (`DiffusionConfig`) that the model, the loss and the
sampler must all agree on. It defines no layer and imports no Keras.

The variant table is a *transcription*. Every number in :data:`DIT_VARIANTS`
was copied from the named reference below; none of it is computed, inferred or
recalled. The paper's naming is ``DiT-<scale>/<patch>``, and the patch size is
part of the name because it is the single knob that changes the token count
without changing the parameter count of a block:

.. code-block:: text

    latent input                token grid                 transformer
    [B, H, W, C]                [B, T, D]                  depth x blocks
    ┌──────────────┐            ┌──────────────┐           ┌──────────────┐
    │  input_size  │  patchify  │  T = (H/p)^2 │           │ hidden_size  │
    │  x           │ ─────────▶ │  D = hidden  │ ────────▶ │ x depth      │
    │  input_size  │  patch p   │              │           │ num_heads    │
    └──────────────┘            └──────────────┘           └──────────────┘
           │                           │                          │
           ▼                           ▼                          ▼
      H = W = input_size        T shrinks by p^2         D, depth, heads
      C = in_channels           as p grows               fixed per scale

    scale   depth   hidden_size   num_heads      patch p in {2, 4, 8}
    ─────   ─────   ───────────   ─────────
    S          12           384           6      T at input_size=32:
    B          12           768          12        p=2 -> 256 tokens
    L          24          1024          16        p=4 ->  64 tokens
    XL         28          1152          16        p=8 ->  16 tokens

Halving the patch size quadruples ``T`` and therefore roughly quadruples the
attention cost while leaving the weight count essentially unchanged, which is
why ``DiT-XL/2`` is the expensive configuration and ``DiT-S/8`` the cheap one.

The canonical key form is the upstream public name -- ``"DiT-XL/2"`` -- because
that is the string checkpoints, papers and command lines use, and a port that
renames it forces every reader to translate. Filesystem-safe spellings such as
``"dit_xl_2"`` are accepted by :func:`normalize_variant_name`, which maps them
onto the canonical key. There is only one table: a second table keyed by a
sanitized name would be the same twelve rows under a second spelling and
would have to be kept in lockstep by hand.

References:
    - Peebles, W. and Xie, S. "Scalable Diffusion Models with Transformers."
      arXiv:2212.09748, 2022. https://arxiv.org/abs/2212.09748
    - Upstream ``fast-DiT`` reference copy staged under the plan's
      ``reference/models.py`` (the ``DiT Configs`` section and the
      ``DiT_models`` dict), the arbiter for all twelve rows and for the
      ``DiT.__init__`` / ``create_diffusion`` defaults reproduced here.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.ddpm_schedule import (
    DDPMSchedule,
    VALID_BETA_SCHEDULES,
    get_named_beta_schedule,
)

# ---------------------------------------------------------------------
# The variant registry
# ---------------------------------------------------------------------

#: Field names of one :data:`DIT_VARIANTS` row, in the order the paper lists
#: them. Named so tests and ``from_variant`` agree on the row schema instead of
#: each spelling the four keys out.
VARIANT_FIELDS: Tuple[str, ...] = ("depth", "hidden_size", "patch_size", "num_heads")

#: The twelve named DiT configurations, keyed by their upstream public names.
#:
#: Transcribed verbatim from the ``DiT Configs`` section of the reference copy of
#: upstream ``models.py`` (``DiT_XL_2`` ... ``DiT_S_8`` and the ``DiT_models``
#: dict); see this module's ``References``. Nothing here is derived: the four
#: scales do not follow a formula (``S`` and ``B`` share ``depth=12`` while
#: their widths differ by 2x, and ``L``/``XL`` share ``num_heads=16`` while
#: their depths differ), so any "obvious" pattern used to regenerate a row
#: would be wrong.
DIT_VARIANTS: Dict[str, Dict[str, int]] = {
    "DiT-XL/2": {"depth": 28, "hidden_size": 1152, "patch_size": 2, "num_heads": 16},
    "DiT-XL/4": {"depth": 28, "hidden_size": 1152, "patch_size": 4, "num_heads": 16},
    "DiT-XL/8": {"depth": 28, "hidden_size": 1152, "patch_size": 8, "num_heads": 16},
    "DiT-L/2": {"depth": 24, "hidden_size": 1024, "patch_size": 2, "num_heads": 16},
    "DiT-L/4": {"depth": 24, "hidden_size": 1024, "patch_size": 4, "num_heads": 16},
    "DiT-L/8": {"depth": 24, "hidden_size": 1024, "patch_size": 8, "num_heads": 16},
    "DiT-B/2": {"depth": 12, "hidden_size": 768, "patch_size": 2, "num_heads": 12},
    "DiT-B/4": {"depth": 12, "hidden_size": 768, "patch_size": 4, "num_heads": 12},
    "DiT-B/8": {"depth": 12, "hidden_size": 768, "patch_size": 8, "num_heads": 12},
    "DiT-S/2": {"depth": 12, "hidden_size": 384, "patch_size": 2, "num_heads": 6},
    "DiT-S/4": {"depth": 12, "hidden_size": 384, "patch_size": 4, "num_heads": 6},
    "DiT-S/8": {"depth": 12, "hidden_size": 384, "patch_size": 8, "num_heads": 6},
}


def normalize_variant_name(name: str) -> str:
    """Map a loosely-spelled variant name onto its canonical :data:`DIT_VARIANTS` key.

    Accepts the canonical public name (``"DiT-XL/2"``), the same name without its
    ``DiT`` prefix (``"XL/2"``), and filesystem- or CLI-safe spellings that use
    ``-`` or ``_`` in place of ``/`` and any casing (``"dit_xl_2"``,
    ``"DIT-XL-2"``). The result is always the canonical key, so callers can key
    the one table with whatever spelling reached them.

    :param name: A variant name in any accepted spelling.
    :type name: str
    :return: The canonical key, e.g. ``"DiT-XL/2"``.
    :rtype: str
    :raises ValueError: If ``name`` is not two ``<scale>``/``<patch>`` parts, or
        if the normalized result is not a registered variant.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"variant name must be a non-empty string, got {name!r}")

    token = name.strip().upper().replace("_", "-").replace("/", "-")
    if token.startswith("DIT-"):
        token = token[len("DIT-"):]

    parts = [p for p in token.split("-") if p]
    if len(parts) != 2:
        raise ValueError(
            f"variant name {name!r} does not parse as '<scale>/<patch>'; "
            f"expected one of {sorted(DIT_VARIANTS)}"
        )

    canonical = f"DiT-{parts[0]}/{parts[1]}"
    if canonical not in DIT_VARIANTS:
        raise ValueError(
            f"unknown DiT variant {name!r} (normalized to {canonical!r}). "
            f"Available: {sorted(DIT_VARIANTS)}"
        )
    return canonical


def get_variant_config(name: str) -> Dict[str, int]:
    """Look a variant's ``(depth, hidden_size, patch_size, num_heads)`` row up by name.

    :param name: Any spelling accepted by :func:`normalize_variant_name`.
    :type name: str
    :return: A fresh copy of the row, so a caller cannot mutate the registry.
    :rtype: Dict[str, int]
    :raises ValueError: If ``name`` is not a registered variant.
    """
    return dict(DIT_VARIANTS[normalize_variant_name(name)])


# ---------------------------------------------------------------------
# The diffusion-side configuration
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class DiffusionConfig:
    """Diffusion-side knobs shared by the model, the loss and the sampler.

    Every default reproduces upstream: the geometry and conditioning fields come
    from ``DiT.__init__`` (``input_size=32``, ``in_channels=4``,
    ``mlp_ratio=4.0``, ``class_dropout_prob=0.1``, ``num_classes=1000``,
    ``learn_sigma=True``) and the chain fields from ``create_diffusion``
    (``diffusion_steps=1000``, ``noise_schedule="linear"``). The latent geometry
    is square by construction -- upstream requires ``h == w == sqrt(T)`` when it
    unpatchifies -- so one ``input_size`` describes both axes.

    .. code-block:: text

        DiffusionConfig                       consumers
        ┌────────────────────────┐
        │ input_size, in_channels│──────────▶ DiT patchify  [B, H, W, C]
        │ num_classes            │──────────▶ label table   [num_classes + 1, D]
        │ class_dropout_rate     │──────────▶ null row / CFG availability
        │ learn_sigma            │──────────▶ out_channels = 2 * C
        │ mlp_ratio              │──────────▶ block FFN width
        │ num_timesteps          │──────────▶ ┐
        │ schedule_name          │──────────▶ ┴ DDPMSchedule tables [T]
        └────────────────────────┘
                    │
                    ▼
             build_schedule()  ->  the SAME tables in loss and sampler

    :param input_size: Side of the square latent grid (``H == W``). This is the
        latent resolution, not the pixel resolution.
    :type input_size: int
    :param in_channels: Latent channel count ``C`` of the model input.
    :type in_channels: int
    :param num_classes: Number of real class labels. The label table carries one
        extra row when ``class_dropout_rate > 0`` (the null row at index
        ``num_classes``).
    :type num_classes: int
    :param class_dropout_rate: Probability of replacing a label with the null
        row during training. ``0.0`` means no null row exists and
        classifier-free guidance is unavailable.
    :type class_dropout_rate: float
    :param learn_sigma: If ``True`` the model emits ``2 * in_channels`` channels,
        the second half being a variance-interpolation logit rather than a
        second epsilon prediction.
    :type learn_sigma: bool
    :param mlp_ratio: Block FFN hidden width as a multiple of ``hidden_size``.
    :type mlp_ratio: float
    :param num_timesteps: Length of the diffusion chain ``T``.
    :type num_timesteps: int
    :param schedule_name: Beta schedule name; one of
        :data:`~dl_techniques.utils.ddpm_schedule.VALID_BETA_SCHEDULES`.
    :type schedule_name: str
    :raises ValueError: From ``__post_init__``, naming the offending field, for a
        non-positive dimension, an out-of-range dropout rate, an unknown
        schedule name, or a ``num_timesteps`` the named schedule cannot produce.
    """

    input_size: int = 32
    in_channels: int = 4
    num_classes: int = 1000
    class_dropout_rate: float = 0.1
    learn_sigma: bool = True
    mlp_ratio: float = 4.0
    num_timesteps: int = 1000
    schedule_name: str = "linear"

    def __post_init__(self) -> None:
        """Validate every field, raising :class:`ValueError` naming the offender."""
        for field_name in ("input_size", "in_channels", "num_classes", "num_timesteps"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(
                    f"{field_name} must be a positive int, got {value!r}"
                )

        if not 0.0 <= self.class_dropout_rate < 1.0:
            raise ValueError(
                f"class_dropout_rate must lie in [0.0, 1.0), got "
                f"{self.class_dropout_rate!r}"
            )

        if not self.mlp_ratio > 0.0:
            raise ValueError(f"mlp_ratio must be positive, got {self.mlp_ratio!r}")

        if self.schedule_name not in VALID_BETA_SCHEDULES:
            raise ValueError(
                f"schedule_name must be one of {list(VALID_BETA_SCHEDULES)}, got "
                f"{self.schedule_name!r}"
            )

        # DECISION plan-2026-09-02T170923-1285ed83/D-010: do not replace this with a
        # literal minimum-step threshold — the accepted set is {1} U [20, inf), which no
        # single threshold expresses. Build the schedule and let it decide. See decisions.md.
        try:
            self.build_schedule()
        except ValueError as exc:
            raise ValueError(
                f"num_timesteps={self.num_timesteps} is not valid for "
                f"schedule_name={self.schedule_name!r}: {exc}"
            ) from exc

    # -- derived geometry -----------------------------------------------

    @property
    def out_channels(self) -> int:
        """Model output channel count: ``2 * in_channels`` when ``learn_sigma``."""
        return self.in_channels * 2 if self.learn_sigma else self.in_channels

    def num_patches(self, patch_size: int) -> int:
        """Token count ``T = (input_size / patch_size) ** 2`` for a patch size.

        :param patch_size: Side of the square patch, from the variant row.
        :type patch_size: int
        :return: Number of tokens the patchified latent produces.
        :rtype: int
        :raises ValueError: If ``patch_size`` does not divide ``input_size``.
        """
        self.validate_patch_size(patch_size)
        return (self.input_size // patch_size) ** 2

    def validate_patch_size(self, patch_size: int) -> None:
        """Raise unless ``patch_size`` tiles the latent grid exactly.

        Lives here rather than in ``__post_init__`` because the patch size is a
        property of the *variant*, not of the diffusion config; the pairing is
        only known once a variant is chosen.

        :param patch_size: Side of the square patch.
        :type patch_size: int
        :raises ValueError: If ``patch_size`` is not positive or does not divide
            ``input_size``.
        """
        if not isinstance(patch_size, int) or isinstance(patch_size, bool) or patch_size <= 0:
            raise ValueError(f"patch_size must be a positive int, got {patch_size!r}")
        if self.input_size % patch_size != 0:
            raise ValueError(
                f"input_size ({self.input_size}) must be divisible by patch_size "
                f"({patch_size}); a non-square or ragged token grid is out of scope"
            )

    # -- the shared schedule ---------------------------------------------

    def build_schedule(self) -> DDPMSchedule:
        """Construct the DDPM constant tables this config describes.

        The loss, the sampler and the data pipeline each call this rather than
        passing tables around, so all three provably read the same numbers.

        :return: The fully derived schedule.
        :rtype: ~dl_techniques.utils.ddpm_schedule.DDPMSchedule
        :raises ValueError: If the named schedule cannot produce a valid beta
            array at this ``num_timesteps``.
        """
        return DDPMSchedule.from_betas(
            get_named_beta_schedule(self.schedule_name, self.num_timesteps)
        )


logger.debug(
    "dit config loaded: %d variants, schedules %s",
    len(DIT_VARIANTS),
    ", ".join(VALID_BETA_SCHEDULES),
)
