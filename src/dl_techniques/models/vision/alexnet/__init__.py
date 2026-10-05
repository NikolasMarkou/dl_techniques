"""AlexNet (Krizhevsky, Sutskever & Hinton, NeurIPS 2012) public API.

Re-exports the model class and the factory function. Internal callers may still
import from ``dl_techniques.models.vision.alexnet.model`` directly.

This package is deliberately **not** a wrapper around the AlexNet-*variant* backbones
in :mod:`dl_techniques.models.vision.siamfc` and
:mod:`dl_techniques.models.vision.dasiamrpn`. Those two inline a five-stage
``valid``-padded trunk for tracking, with BatchNorm and no local response
normalization -- the tracking papers ablated LRN out and changed the padding. They
are separate implementations and were left alone; sharing a package would have
meant editing two published tracking models to serve a third consumer.
"""

from .model import AlexNet, create_alexnet

__all__ = [
    'AlexNet',
    'create_alexnet'
]