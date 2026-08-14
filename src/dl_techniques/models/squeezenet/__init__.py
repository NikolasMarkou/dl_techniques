"""SqueezeNet — public API re-exports.

`squeezenet_v1` is the original SqueezeNet (Fire modules, optional bypass);
`squeezenet_v2` is SqueezeNodule-Net, a 3x3-only simplified-Fire variant with
2D and 3D configurations. They are separate architectures, not a version
upgrade — neither deprecates the other.
"""
from .squeezenet_v1 import FireModule, SqueezeNetV1, create_squeezenet_v1
from .squeezenet_v2 import (
    SimplifiedFireModule,
    SqueezeNoduleNetV2,
    create_squeezenodule_net_v2,
)

__all__ = [
    "FireModule",
    "SqueezeNetV1",
    "create_squeezenet_v1",
    "SimplifiedFireModule",
    "SqueezeNoduleNetV2",
    "create_squeezenodule_net_v2",
]
