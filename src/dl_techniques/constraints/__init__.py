"""Custom Keras weight constraints.

A constraint is a projection Keras applies to a weight after every optimizer
update. Two live here, both bounding a weight to an interval:

-   ``ValueRangeConstraint`` (``value_range_constraint.py``) clips exactly, with
    ``w' = max(lo, min(w, hi))``. It is flat outside the interval, so how far a
    weight went past a bound is discarded.
-   ``SoftValueRangeConstraint`` (``soft_value_range_constraint.py``) uses a
    monotone softplus composition that keeps rising outside the interval, so
    weights beyond a bound stay ordered instead of piling onto it. Use it for a
    WGAN critic, or for any bounded parameter that stalls when it saturates.

This package exports no public API; import each class from its own module.
"""
