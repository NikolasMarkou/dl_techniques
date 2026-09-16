"""RBF prototype-network public API.

Re-exports both model classes and their factories. ``RBFProtoNet`` composes a
CIFAR-style CNN backbone with an ``RBFLayer`` prototype-classification head
(``output_mode='normalized'``); ``create_rbf_protonet()`` is the delegating
factory. ``CliffordRBFProtoNet`` swaps in an isotropic Clifford-algebra
backbone while reusing the same RBF head design; ``create_clifford_rbf_protonet()``
is its delegating factory.
"""

from .model import (
    RBFProtoNet,
    create_rbf_protonet,
    CliffordRBFProtoNet,
    create_clifford_rbf_protonet,
)

__all__ = [
    'RBFProtoNet',
    'create_rbf_protonet',
    'CliffordRBFProtoNet',
    'create_clifford_rbf_protonet',
]
