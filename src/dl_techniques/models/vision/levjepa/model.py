"""LeVJEPA scale table, variant registry, and factory functions.

Defines :data:`SCALE_CONFIGS`, :func:`from_variant`, and
:func:`create_levjepa`, reproducing the LeVJEPA PyTorch reference's
``vit_tiny`` through ``vit_gigantic`` factory functions as a data table,
in the same shape as ``models/vision/vit/model.py``'s ``SCALE_CONFIGS`` /
``MODEL_VARIANTS`` / ``from_variant`` / ``create_*``.

References:
    - LeVJEPA PyTorch reference, ``module.py::vit_tiny`` .. ``vit_gigantic``
      (pasted transcript; no public arXiv id in this plan's context).
"""

import keras
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.models.vision.levjepa.encoder import LeVJEPAEncoder, AttnMode

# ---------------------------------------------------------------------

#: ``scale -> (embed_dim, depth, num_heads, mlp_ratio, patch_size)``. Ported
#: verbatim from the reference's ``vit_tiny`` .. ``vit_gigantic`` factory
#: functions. ``qkv_bias=True`` and ``LayerNorm(eps=1e-6)`` are constant
#: across every scale and are not part of this table -- they are
#: ``LeVJEPABlock``'s / ``LeVJEPAEncoder``'s own defaults.
SCALE_CONFIGS: Dict[str, Tuple[int, int, int, float, int]] = {
    "vit_tiny": (192, 12, 3, 4.0, 16),
    "vit_small": (384, 12, 6, 4.0, 16),
    "vit_base": (768, 12, 12, 4.0, 16),
    "vit_large": (1024, 24, 16, 4.0, 16),
    "vit_huge": (1280, 32, 16, 4.0, 16),
    "vit_giant": (1408, 40, 16, 48.0 / 11.0, 16),
    "vit_gigantic": (1664, 48, 16, 64.0 / 13.0, 14),
}

#: Thin wrapper over :data:`SCALE_CONFIGS`, resnet-template shape: a variant
#: key maps to the kwargs :meth:`LeVJEPAEncoder.from_variant`-equivalent
#: construction needs beyond ``SCALE_CONFIGS`` itself. Kept as an explicit
#: dict (rather than deriving it inline in ``from_variant``) so a caller can
#: introspect ``MODEL_VARIANTS.keys()`` the same way ``ViT.MODEL_VARIANTS``
#: supports.
MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {name: {"scale": name} for name in SCALE_CONFIGS}


def from_variant(
    variant: str,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_frames: int = 1,
    tubelet_size: int = 2,
    use_rope: bool = False,
    rope_theta: float = 10000.0,
    attn_mode: AttnMode = "full",
    token_drop_rate: float = 0.0,
    dropout_rate: float = 0.0,
    attention_dropout_rate: float = 0.0,
    init_std: float = 0.02,
    uniform_power: bool = False,
    name: Optional[str] = None,
    **kwargs: Any,
) -> LeVJEPAEncoder:
    """Construct a :class:`LeVJEPAEncoder` from a named scale variant.

    :param variant: One of :data:`SCALE_CONFIGS`'s keys (``'vit_tiny'`` ..
        ``'vit_gigantic'``).
    :type variant: str
    :param input_shape: Spatial input shape ``(height, width, channels)``.
        Defaults to ``(224, 224, 3)``.
    :type input_shape: Tuple[int, int, int]
    :param num_frames: Number of frames per clip; ``1`` (default) is the
        still-image path.
    :type num_frames: int
    :param tubelet_size: Temporal patch size for the video path. Defaults to
        ``2``.
    :type tubelet_size: int
    :param use_rope: Whether to use :class:`VideoRoPE3D` instead of the
        frozen sincos table. Defaults to ``False``.
    :type use_rope: bool
    :param rope_theta: Rotary base frequency. Defaults to ``10000.0``.
    :type rope_theta: float
    :param attn_mode: ``'full'`` or ``'block_causal'``. Defaults to
        ``'full'``.
    :type attn_mode: AttnMode
    :param token_drop_rate: Train-time patch-token drop fraction. Defaults to
        ``0.0``.
    :type token_drop_rate: float
    :param dropout_rate: General dropout rate. Defaults to ``0.0``.
    :type dropout_rate: float
    :param attention_dropout_rate: Attention-weight dropout rate. Defaults to
        ``0.0``.
    :type attention_dropout_rate: float
    :param init_std: Base truncated-normal std. Defaults to ``0.02``.
    :type init_std: float
    :param uniform_power: Forwarded to the 3D sincos table builder. Defaults
        to ``False``.
    :type uniform_power: bool
    :param name: Model name; auto-generated when ``None``.
    :type name: Optional[str]
    :param kwargs: Additional keyword arguments forwarded to
        :class:`LeVJEPAEncoder`.
    :type kwargs: Any
    :return: Configured :class:`LeVJEPAEncoder` instance.
    :rtype: LeVJEPAEncoder
    :raises ValueError: If ``variant`` is not recognized, propagated from
        :class:`LeVJEPAEncoder` for any other invalid configuration.
    """
    if variant not in SCALE_CONFIGS:
        raise ValueError(
            f"Unknown variant '{variant}'. Available variants: {list(SCALE_CONFIGS.keys())}"
        )

    embed_dim, depth, num_heads, mlp_ratio, patch_size = SCALE_CONFIGS[variant]

    if name is None:
        name = f"levjepa_{variant}"

    logger.info(f"Creating LeVJEPAEncoder variant '{variant}'")

    return LeVJEPAEncoder(
        input_shape=input_shape,
        num_frames=num_frames,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        use_rope=use_rope,
        rope_theta=rope_theta,
        attn_mode=attn_mode,
        token_drop_rate=token_drop_rate,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        init_std=init_std,
        uniform_power=uniform_power,
        name=name,
        **kwargs,
    )


def create_levjepa(
    variant: str = "vit_base",
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_frames: int = 1,
    tubelet_size: int = 2,
    use_rope: bool = False,
    rope_theta: float = 10000.0,
    attn_mode: AttnMode = "full",
    token_drop_rate: float = 0.0,
    dropout_rate: float = 0.0,
    attention_dropout_rate: float = 0.0,
    init_std: float = 0.02,
    uniform_power: bool = False,
    **kwargs: Any,
) -> LeVJEPAEncoder:
    """Create a :class:`LeVJEPAEncoder` with the specified configuration.

    Thin wrapper over :func:`from_variant`, matching the house
    ``create_<name>()`` factory-function convention
    (``models/vision/vit/model.py::create_vit``).

    :param variant: Scale variant key. Defaults to ``'vit_base'``.
    :type variant: str
    :param input_shape: Spatial input shape ``(height, width, channels)``.
    :type input_shape: Tuple[int, int, int]
    :param num_frames: Number of frames per clip.
    :type num_frames: int
    :param tubelet_size: Temporal patch size for the video path.
    :type tubelet_size: int
    :param use_rope: Whether to use rotary position embedding.
    :type use_rope: bool
    :param rope_theta: Rotary base frequency.
    :type rope_theta: float
    :param attn_mode: ``'full'`` or ``'block_causal'``.
    :type attn_mode: AttnMode
    :param token_drop_rate: Train-time patch-token drop fraction.
    :type token_drop_rate: float
    :param dropout_rate: General dropout rate.
    :type dropout_rate: float
    :param attention_dropout_rate: Attention-weight dropout rate.
    :type attention_dropout_rate: float
    :param init_std: Base truncated-normal std.
    :type init_std: float
    :param uniform_power: Forwarded to the 3D sincos table builder.
    :type uniform_power: bool
    :param kwargs: Additional keyword arguments forwarded to
        :class:`LeVJEPAEncoder`.
    :type kwargs: Any
    :return: Configured :class:`LeVJEPAEncoder` instance.
    :rtype: LeVJEPAEncoder
    :raises ValueError: If ``variant`` is not recognized.

    Example:

    .. code-block:: python

        from dl_techniques.models.vision.levjepa.model import create_levjepa

        encoder = create_levjepa(variant="vit_tiny", input_shape=(64, 64, 3))
    """
    return from_variant(
        variant=variant,
        input_shape=input_shape,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        use_rope=use_rope,
        rope_theta=rope_theta,
        attn_mode=attn_mode,
        token_drop_rate=token_drop_rate,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        init_std=init_std,
        uniform_power=uniform_power,
        **kwargs,
    )


# ---------------------------------------------------------------------
