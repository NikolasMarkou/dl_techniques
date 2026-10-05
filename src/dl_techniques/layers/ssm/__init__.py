"""State space layers: selective SSM primitives and temporal context mixer.

This package exports the SSM layers plus the factory interface that builds
them from a config dict. Import a class directly, or call
``create_ssm_layer(type=...)`` when the type comes from configuration.

``factory.py`` registers 2 keys: ``selective_ssm`` (generic S6 scan) and
``context_mamba`` (MambaLCT temporal wrapper over it).
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
from .selective_ssm import SelectiveSSMLayer
from .context_mamba import ContextMambaLayer

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
    "ContextMambaLayer",
]
