# Regularizers

Regularization techniques

**7 modules in this category**

## Binary_Preference

### regularizers.binary_preference
Encourage network weights to adopt binary values (0 or 1).

**Classes:**
- `BinaryPreferenceRegularizer`

**Functions:** `create_binary_preference_regularizer`, `get_config`

*📁 File: `src/dl_techniques/regularizers/binary_preference.py`*

## Core

### regularizers
Advanced Keras Regularizers for Deep Learning.

*📁 File: `src/dl_techniques/regularizers/__init__.py`*

## Entropy_Regularizer

### regularizers.entropy_regularizer
Encourage a target entropy level in network weight distributions.

**Classes:**
- `EntropyRegularizer`

**Functions:** `create_entropy_regularizer`, `get_config`

*📁 File: `src/dl_techniques/regularizers/entropy_regularizer.py`*

## L2_Custom

### regularizers.l2_custom
A generalized L2 regularization penalty supporting weight decay or growth.

**Classes:**
- `L2_custom`

**Functions:** `validate_float_arg`, `get_config`

*📁 File: `src/dl_techniques/regularizers/l2_custom.py`*

## Soft_Orthogonal

### regularizers.soft_orthogonal
Theory and Implementation of Soft Orthogonality and Orthonormality Constraints

**Classes:**
- `SoftOrthogonalConstraintRegularizer`
- `SoftOrthonormalConstraintRegularizer`

**Functions:** `get_config`, `get_config`

*📁 File: `src/dl_techniques/regularizers/soft_orthogonal.py`*

## Srip

### regularizers.srip
Enforce near-orthonormality in weight matrices via spectral norm penalty.

**Classes:**
- `SRIPRegularizer`

**Functions:** `create_srip_regularizer`, `current_lambda`, `update_lambda`, `get_config`, `from_config`

*📁 File: `src/dl_techniques/regularizers/srip.py`*

## Tri_State_Preference

### regularizers.tri_state_preference
Encourage network weights to adopt ternary values (-1, 0, or 1).

**Classes:**
- `TriStatePreferenceRegularizer`

**Functions:** `get_config`, `from_config`

*📁 File: `src/dl_techniques/regularizers/tri_state_preference.py`*