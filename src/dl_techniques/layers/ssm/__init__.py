"""State space layers: selective SSM primitives and temporal context mixer.

This package exports the SSM layers plus the factory interface that builds
them from a config dict. Import a class directly, or call
``create_ssm_layer(type=...)`` when the type comes from configuration.

``factory.py`` registers 4 keys: ``selective_ssm`` (generic S6 scan),
``context_mamba`` (MambaLCT temporal wrapper), ``mamba`` (paper-named
alias of the S6 scan) and ``mamba2`` (multi-head SSD scan). The residual
wrappers (``MambaResidualBlock``, ``Mamba2ResidualBlock``) are
direct-import classes, not factory keys.
"""

from .factory import (
    SSM_REGISTRY,
    SsmType,
    create_ssm_from_config,
    create_ssm_layer,
    validate_ssm_config,
    assemble_ssm_config,
    get_ssm_info,
    list_ssm_types,
    get_ssm_requirements,
)
from .selective_ssm import SelectiveSSMLayer, MambaLayer, MambaResidualBlock
from .context_mamba import ContextMambaLayer
from .mamba2 import Mamba2Layer, Mamba2ResidualBlock

__all__ = [
    "SSM_REGISTRY",
    "SsmType",
    "create_ssm_from_config",
    "create_ssm_layer",
    "validate_ssm_config",
    "assemble_ssm_config",
    "get_ssm_info",
    "list_ssm_types",
    "get_ssm_requirements",
    "SelectiveSSMLayer",
    "MambaLayer",
    "MambaResidualBlock",
    "ContextMambaLayer",
    "Mamba2Layer",
    "Mamba2ResidualBlock",
]
