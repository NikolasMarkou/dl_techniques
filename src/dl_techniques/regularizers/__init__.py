"""Advanced Keras Regularizers for Deep Learning.

This package provides a collection of sophisticated regularization techniques
implemented as Keras regularizers. These tools are designed to go beyond
standard L1/L2 penalties, offering advanced control over the learning dynamics
and representational structure of neural networks.

Available Regularizers:
-----------------------
-   `BinaryPreferenceRegularizer`: Encourages weights to adopt binary values (0, 1).
-   `TriStatePreferenceRegularizer`: Encourages weights towards ternary values (-1, 0, 1).
-   `EntropyRegularizer`: Controls the information distribution in weight
    matrices by targeting a specific Shannon entropy level.
-   `SoftOrthogonalConstraintRegularizer`: Encourages weight matrix columns to be
    mutually orthogonal, improving gradient flow.
-   `SoftOrthonormalConstraintRegularizer`: A stricter version that encourages both
    orthogonality and unit-norm columns.
-   `SRIPRegularizer`: Enforces near-orthonormality using a spectral norm
    penalty (Spectral Restricted Isometry Property).

Each regularizer is designed to be easily integrated into existing Keras
models and is fully serializable.
"""

from .binary_preference import (
    BinaryPreferenceRegularizer,
    create_binary_preference_regularizer,
)

from .entropy_regularizer import (
    EntropyRegularizer,
    create_entropy_regularizer,
)

from .soft_orthogonal import (
    SoftOrthogonalConstraintRegularizer,
    SoftOrthonormalConstraintRegularizer,
)

from .srip import (
    SRIPRegularizer,
    create_srip_regularizer,
)

from .tri_state_preference import (
    TriStatePreferenceRegularizer
)


# Define the public API for the package
__all__ = [
    # Classes
    "BinaryPreferenceRegularizer",
    "EntropyRegularizer",
    "SRIPRegularizer",
    "SoftOrthogonalConstraintRegularizer",
    "SoftOrthonormalConstraintRegularizer",
    "TriStatePreferenceRegularizer",
    # Factory Functions
    "create_binary_preference_regularizer",
    "create_entropy_regularizer",
    "create_srip_regularizer",
]