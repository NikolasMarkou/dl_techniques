"""Geometry and constants of the BiT/BiB text<->image bridge.

The bridge is the single tensor both modalities are forced to share. A caption
is a sequence of token embeddings; an image is a VAE latent map. Neither can be
diffused into the other until they are the *same object*, so the port fixes one
shape -- a latent-sized tensor ``(H, W, C)`` -- and requires the token sequence
to be a lossless repacking of exactly that many numbers. `BridgeConfig` is the
arithmetic that makes the two views commensurable, and `validate` is what stops
a preset whose two views merely look plausible.

The invariant that carries the whole design is `token_flat_dim ==
bridge_flat_dim`. Once it holds, "how many patches does one token occupy" is not
a free parameter: it is forced, and so is the patch grid. That is why the
presets below look over-determined -- they are.

This module is channels-last. Upstream's ``TokenBridgeConfig`` stores
``bridge_shape`` as ``(C, H, W)`` because PyTorch convolutions are channels-first;
here the same field is ``(H, W, C)``. The derived quantities are unchanged, but
the field order is not, and reading an upstream preset into this class without
permuting it would produce a config that validates and is wrong.

References:
    - Upstream ``token_bridge.py`` (staged verbatim under the plan's
      ``reference/`` directory), from which every constant here is ported.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------

#: Endpoint clamp for every time sampler. ``C(0, t, t)`` and ``C(t, 1, 1)`` are
#: exactly zero at ``t = 0`` / ``t = 1``, so the score targets divide by zero
#: there. Named, never inlined as a literal.
TIME_EPS: float = 1e-4

#: Prompt-kind conditioning classes. Upstream's ``PROMPT_KIND_TO_LABEL`` has
#: three entries and its ``DiT`` is built with ``num_classes=3``, while the GPIC
#: encoder that produced the captions declares four caption types
#: ``("tag", "short", "medium", "long")``. The two disagree upstream; this port
#: follows the number the *model* was built with (3), because that is what the
#: class-embedding table's row count has to match. The discrepancy is recorded
#: rather than silently reconciled.
PROMPT_KIND_TO_LABEL: Dict[str, int] = {"original": 0, "short": 1, "medium": 2}
PROMPT_NUM_CLASSES: int = 3

#: The only registered token->patch layout. ``row_major`` is the identity
#: permutation; the indirection exists because upstream registers it as an
#: extension point and the packing functions take a ``layout`` argument.
TOKEN_LAYOUTS: Tuple[str, ...] = ("row_major",)

#: VAE latent normalization scalars, defined locally on purpose: they are
#: properties of the encoders the upstream presets name, and importing them from
#: another model package would couple this package to that one's lifecycle.
SD_LATENT_SCALE: float = 0.18215
SD_LATENT_SHIFT: float = 0.0
FLUX_LATENT_SCALE: float = 0.3611
FLUX_LATENT_SHIFT: float = 0.1159


@dataclass(frozen=True)
class BridgeConfig:
    """Geometry of one token<->bridge packing.

    :param token_seq_len: Number of text tokens carried by one bridge tensor.
    :type token_seq_len: int
    :param token_emb_dim: Width of a single token embedding.
    :type token_emb_dim: int
    :param bridge_shape: Bridge tensor shape as ``(height, width, channels)``.
        **Channels-last**, unlike upstream's ``(C, H, W)``.
    :type bridge_shape: Tuple[int, int, int]
    :param patch_size: Side of the square patch the bridge is tiled into.
    :type patch_size: int
    :param text_as_noise: Ablation flag -- replace the text endpoint with noise.
    :type text_as_noise: bool
    :param image_as_noise: Ablation flag -- replace the image endpoint with noise.
    :type image_as_noise: bool
    :param latent_scale: Multiplier applied to raw VAE latents.
    :type latent_scale: float
    :param latent_shift: Offset applied to raw VAE latents.
    :type latent_shift: float
    """

    token_seq_len: int = 64
    token_emb_dim: int = 64
    bridge_shape: Tuple[int, int, int] = (32, 32, 4)  # (H, W, C) -- channels-last
    patch_size: int = 2
    text_as_noise: bool = False
    image_as_noise: bool = False
    latent_scale: float = SD_LATENT_SCALE
    latent_shift: float = SD_LATENT_SHIFT

    # -- bridge geometry ------------------------------------------------

    @property
    def height(self) -> int:
        return int(self.bridge_shape[0])

    @property
    def width(self) -> int:
        return int(self.bridge_shape[1])

    @property
    def channels(self) -> int:
        return int(self.bridge_shape[2])

    @property
    def patch_h(self) -> int:
        """Patch-grid rows."""
        return self.height // self.patch_size

    @property
    def patch_w(self) -> int:
        """Patch-grid columns."""
        return self.width // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.patch_h * self.patch_w

    @property
    def patch_payload_dim(self) -> int:
        """Numbers carried by one patch: ``C * p * p``."""
        return self.channels * self.patch_size * self.patch_size

    # -- token geometry -------------------------------------------------

    @property
    def patches_per_token(self) -> int:
        """How many patches one token embedding fills."""
        return self.token_emb_dim // self.patch_payload_dim

    @property
    def token_flat_dim(self) -> int:
        return self.token_seq_len * self.token_emb_dim

    @property
    def bridge_flat_dim(self) -> int:
        return self.channels * self.height * self.width

    @property
    def token_scale(self) -> float:
        """``sqrt(token_emb_dim)``; real token embeddings are unit-norm once divided by it."""
        return float(self.token_emb_dim) ** 0.5

    # -- validation -----------------------------------------------------

    def validate(self) -> "BridgeConfig":
        """Raise :class:`ValueError` unless the two views describe the same numbers.

        :return: ``self``, so the call can be chained.
        :rtype: BridgeConfig
        :raises ValueError: On a non-divisible bridge, a token/bridge element-count
            mismatch, or a token grid that does not tile the patch grid.
        """
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")
        if self.height % self.patch_size or self.width % self.patch_size:
            raise ValueError(
                f"bridge height and width must be divisible by patch_size; got "
                f"height={self.height}, width={self.width}, patch_size={self.patch_size}"
            )
        if self.token_flat_dim != self.bridge_flat_dim:
            raise ValueError(
                f"token_flat_dim ({self.token_seq_len} * {self.token_emb_dim} = "
                f"{self.token_flat_dim}) must equal bridge_flat_dim "
                f"({self.height} * {self.width} * {self.channels} = "
                f"{self.bridge_flat_dim}); the packing cannot be a bijection otherwise"
            )
        # DECISION plan-2026-09-02T094601-77d4a04e/D-008
        # Do NOT add a fourth `token_emb_dim % patch_payload_dim == 0` check here.
        # Given the divisibility check above, any two of {flat-dim equality, this
        # patches_per_token check, exact payload divisibility} imply the third, so a
        # fourth check is one that can never fire -- and a check that cannot fail is
        # not a guard, it is a claim no test can falsify. See decisions.md D-008.
        if self.token_seq_len * self.patches_per_token != self.num_patches:
            raise ValueError(
                f"token_seq_len * patches_per_token "
                f"({self.token_seq_len} * {self.patches_per_token} = "
                f"{self.token_seq_len * self.patches_per_token}) must equal num_patches "
                f"({self.patch_h} * {self.patch_w} = {self.num_patches}); "
                f"token_emb_dim ({self.token_emb_dim}) is most likely not a whole "
                f"multiple of patch_payload_dim ({self.patch_payload_dim})"
            )
        return self


# ---------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------
#
#   sd   : 64 tokens x 64 dims  = 4096  ==  32 * 32 * 4  = 4096
#   flux : 128 tokens x 128 dims = 16384 == 32 * 32 * 16 = 16384
#   tiny : 8 tokens x 32 dims   = 256   ==  8 *  8 * 4   = 256   (tests only)

BRIDGE_PRESETS: Dict[str, BridgeConfig] = {
    "sd": BridgeConfig(
        token_seq_len=64,
        token_emb_dim=64,
        bridge_shape=(32, 32, 4),
        patch_size=2,
        latent_scale=SD_LATENT_SCALE,
        latent_shift=SD_LATENT_SHIFT,
    ),
    "flux": BridgeConfig(
        token_seq_len=128,
        token_emb_dim=128,
        bridge_shape=(32, 32, 16),
        patch_size=2,
        latent_scale=FLUX_LATENT_SCALE,
        latent_shift=FLUX_LATENT_SHIFT,
    ),
    "tiny": BridgeConfig(
        token_seq_len=8,
        token_emb_dim=32,
        bridge_shape=(8, 8, 4),
        patch_size=2,
        latent_scale=SD_LATENT_SCALE,
        latent_shift=SD_LATENT_SHIFT,
    ),
}

for _name, _config in BRIDGE_PRESETS.items():
    _config.validate()
logger.debug(
    "bit_diffusion bridge presets validated: %s", ", ".join(sorted(BRIDGE_PRESETS))
)


def get_bridge_config(name: str) -> BridgeConfig:
    """Look a preset up by name.

    :param name: One of the keys of :data:`BRIDGE_PRESETS`.
    :type name: str
    :return: The validated preset.
    :rtype: BridgeConfig
    :raises ValueError: If ``name`` is not a registered preset.
    """
    if name not in BRIDGE_PRESETS:
        raise ValueError(
            f"Unknown bridge preset '{name}'. Available: {sorted(BRIDGE_PRESETS)}"
        )
    return BRIDGE_PRESETS[name]
