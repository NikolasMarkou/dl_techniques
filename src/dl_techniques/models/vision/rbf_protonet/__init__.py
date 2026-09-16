"""RBF prototype-network public API.

Re-exports the model class and its factory. ``RBFProtoNet`` composes a
CIFAR-style CNN backbone with an ``RBFLayer`` prototype-classification head
(``output_mode='normalized'``); ``create_rbf_protonet()`` is the delegating
factory.
"""

from .model import RBFProtoNet, create_rbf_protonet

__all__ = [
    'RBFProtoNet',
    'create_rbf_protonet',
]
