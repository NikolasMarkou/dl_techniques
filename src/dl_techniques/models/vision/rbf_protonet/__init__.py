"""RBF prototype-network public API.

Re-exports the backbone class. As of this revision the class implements only
the CNN backbone (no RBF head yet, no ``create_rbf_protonet`` factory) -- both
are added in a later revision of ``model.py``, at which point this file's
exports and ``__all__`` are extended to match.
"""

from .model import RBFProtoNet

__all__ = [
    'RBFProtoNet',
]
