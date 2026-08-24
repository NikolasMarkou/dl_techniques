"""Self-Organizing Map — public API re-exports.

There is no ``MODEL_VARIANTS`` table and none was invented: a SOM is specified
entirely by its grid extent and input dimensionality, both continuous and
problem-specific, and Kohonen defines no named scale family. ``create_som``
therefore constructs the class directly rather than delegating to a
``from_variant``.
"""
from dl_techniques.models.som.model import SOMModel, create_som

__all__ = [
    "SOMModel",
    "create_som",
]
