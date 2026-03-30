# Constraints Package

Custom Keras weight constraints applied during training to enforce value bounds on model parameters.

## Modules

- `value_range_constraint.py` — `ValueRangeConstraint` that clips weights to a configurable `[min_value, max_value]` range

## Conventions

- Constraints inherit from `keras.constraints.Constraint`
- Must implement `__call__(self, w)` and `get_config()` for serialization
- `__init__.py` is empty — import directly: `from dl_techniques.constraints.value_range_constraint import ValueRangeConstraint`

## Testing

Tests in `tests/test_constraints/`.
