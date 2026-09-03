"""Custom Keras regularizers.

These go beyond L1/L2 to control the structure of a layer's weights: their
value distribution, their information content, or the geometry of the linear
map they define. Each is fully serializable.

Available regularizers:

-   ``BinaryPreferenceRegularizer``: double-well penalty that pulls weights
    toward two targets, {0, 1} by default. Use ``for_gates`` for gates and
    masks, ``for_bipolar_weights`` for layer kernels.
-   ``TriStatePreferenceRegularizer``: triple-well penalty that pulls weights
    toward {-target, 0, +target}. Match ``target`` to the scale the weights
    actually occupy; ``from_weight_scale`` does that from ``fan_in``.
-   ``EntropyRegularizer``: penalizes the distance from a target normalized
    Shannon entropy, controlling whether a layer develops concentrated or
    distributed weights.
-   ``SoftOrthogonalConstraintRegularizer``: penalizes the off-diagonal entries
    of the kernel's Gram matrix, so the compared directions decorrelate while
    magnitudes are left alone.
-   ``SoftOrthonormalConstraintRegularizer``: penalizes the full deviation from
    the identity, so the directions decorrelate and reach unit norm.
-   ``SRIPRegularizer``: enforces near-orthonormality through the spectral norm
    of ``W^T W - I``, approximated by power iteration.

Both preference regularizers pair with a callback that ramps their multiplier
during training, ``BinaryPressureScheduler`` and ``TriStatePressureScheduler``.
Annealing is not optional for either: at full strength from step zero the
barriers freeze each weight into whichever well its initializer placed it in.

The ``create_*`` functions are thin constructor forwarders kept for interface
consistency; all validation lives in the constructors.

Two more live in this package but are not exported here, since neither is a
weight regularizer in the Keras sense. Import them from their own modules:
``L2_custom`` (``l2_custom.py``), an L2 penalty whose factor may be negative,
and ``SIGRegLayer`` (``sigreg.py``), an activation-based sliced Gaussian
regularizer built as a Layer.
"""

from .binary_preference import (
    BinaryPreferenceRegularizer,
    BinaryPressureScheduler,
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
    TriStatePreferenceRegularizer,
    TriStatePressureScheduler,
    create_tri_state_preference_regularizer,
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
    # Callbacks
    "BinaryPressureScheduler",
    "TriStatePressureScheduler",
    # Factory Functions
    "create_binary_preference_regularizer",
    "create_entropy_regularizer",
    "create_srip_regularizer",
    "create_tri_state_preference_regularizer",
]
