"""
A projection layer based on Tversky's contrast model of similarity.

This layer serves as a psychologically plausible alternative to the standard
fully-connected (Dense) layer. Whereas a Dense layer computes similarity
based on a geometric dot product, which is inherently symmetric, this layer
implements a differentiable version of Amos Tversky's feature-based model.
This allows it to capture the asymmetric nature of human similarity judgments
(e.g., a son resembles his father more than a father resembles his son) and
learn complex, non-linear decision boundaries within a single layer.

Architecture:
    The layer's core operation is to compute a similarity score between an
    input vector and a set of learned 'prototypes'. These prototypes are
    analogous to the weight matrix in a standard Dense layer. The final output
    is a vector of these similarity scores.

    This computation relies on a third component: a learnable 'feature bank'.
    This bank defines a universe of abstract features. An object (either an
    input or a prototype) is considered to "possess" a feature if its dot
    product with that feature's vector is positive. This mechanism provides a
    differentiable bridge from continuous vector representations to the discrete,
    set-based logic of Tversky's model.

    The conceptual data flow is as follows:
    1. An input vector is received.
    2. For each prototype, the layer determines the sets of common and
       distinctive features between the input and the prototype.
    3. The salience of these feature sets is measured and combined according
       to Tversky's formula.
    4. The resulting similarity scores for all prototypes form the output vector.

Foundational Mathematics:
    The similarity `S` of an object `a` to an object `b` is defined by
    Tversky's contrast model:

        S(a, b) = θ * f(A ∩ B) - α * f(A - B) - β * f(B - A)

    Where:
    - `A` and `B` are the sets of features possessed by `a` and `b`, respectively.
    - `f(A ∩ B)` is a measure of the salience of their common features.
    - `f(A - B)` measures the salience of features distinctive to `a`.
    - `f(B - A)` measures the salience of features distinctive to `b`.
    - `θ`, `α`, and `β` are learnable scalar parameters that weight the
      importance of commonality versus distinctiveness.

    This implementation makes the model differentiable by defining feature
    presence and salience in terms of vector operations. The "salience" of a
    feature for a given object is its positive dot product value. Consequently,
    the set-based measures are calculated as follows:
    - `f(A ∩ B)`: An aggregation (e.g., product, min) of the salience scores
      for all features present in *both* the input and the prototype.
    - `f(A - B)`: An aggregation of salience scores for features present in
      the input but *not* the prototype.

    By learning the prototypes, the feature bank, and the contrast parameters
    (θ, α, β) simultaneously via gradient descent, the network can discover a
    task-specific, psychologically-grounded similarity metric.

References:
    - The core theory of similarity:
      Tversky, A. (1977). Features of similarity. Psychological Review.
    - The neural network implementation:
      Doumbouya, M. K. B., et al. (2025). Tversky Neural Networks. arXiv.
"""

import keras
from keras import ops, layers, initializers
from typing import Optional, Union, Tuple, Dict, Any, Literal

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class TverskyProjectionLayer(layers.Layer):
    """
    Projection layer based on a differentiable Tversky similarity model.

    This layer replaces the standard dot-product of a Dense layer with a
    learnable, differentiable version of Tversky's contrast model. It computes
    similarity between an input vector and a set of learned prototypes based on
    their common and distinctive features via a learned feature bank. The
    similarity formula is ``S(a, b) = theta * f(A ∩ B) - alpha * f(A - B) - beta * f(B - A)``,
    where feature presence is determined by positive dot products with the
    feature bank vectors and all parameters are learned end-to-end.

    **Architecture Overview:**

    .. code-block:: text

        ┌────────────────────────────────────────┐
        │  Input [..., input_dim]                │
        └──────────────┬─────────────────────────┘
                       ▼
        ┌────────────────────────────────────────┐
        │  Compute feature presence scores       │
        │  input_dots = Input @ feature_bank^T  │
        │  proto_dots = prototypes @ feat_bank^T │
        └──────────────┬─────────────────────────┘
                       ▼
        ┌────────────────────────────────────────┐
        │  Set operations:                       │
        │  f(A∩B), f(A-B), f(B-A)              │
        │  via masks + aggregation               │
        └──────────────┬─────────────────────────┘
                       ▼
        ┌────────────────────────────────────────┐
        │  Tversky contrast:                     │
        │  S = θ*f(A∩B) - α*f(A-B) - β*f(B-A) │
        └──────────────┬─────────────────────────┘
                       ▼
        ┌────────────────────────────────────────┐
        │  Output [..., units]                   │
        └────────────────────────────────────────┘

    :param units: Dimensionality of the output space (number of prototypes). Must be positive.
    :type units: int
    :param num_features: Size of the learnable feature universe. Must be positive.
    :type num_features: int
    :param intersection_reduction: Aggregation for common features. One of
        ``'product'``, ``'min'``, ``'mean'``. Defaults to ``'product'``.
    :type intersection_reduction: str
    :param difference_reduction: Method for distinctive features. One of
        ``'ignorematch'``, ``'subtractmatch'``. Defaults to ``'subtractmatch'``.
    :type difference_reduction: str
    :param prototype_initializer: Initializer for prototype vectors.
    :type prototype_initializer: str or keras.initializers.Initializer
    :param feature_initializer: Initializer for feature bank vectors.
    :type feature_initializer: str or keras.initializers.Initializer
    :param contrast_initializer: Initializer for Tversky contrast parameters.
    :type contrast_initializer: str or keras.initializers.Initializer
    :param kwargs: Additional arguments for Layer base class.
    :type kwargs: Any
    """

    def __init__(
        self,
        units: int,
        num_features: int,
        intersection_reduction: Literal['product', 'min', 'mean'] = 'product',
        difference_reduction: Literal['ignorematch', 'subtractmatch'] = 'subtractmatch',
        prototype_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        feature_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        contrast_initializer: Union[str, initializers.Initializer] = 'ones',
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if units <= 0:
            raise ValueError(f"`units` must be positive, got {units}")
        if num_features <= 0:
            raise ValueError(f"`num_features` must be positive, got {num_features}")

        # Store ALL configuration for serialization
        self.units = units
        self.num_features = num_features
        self.intersection_reduction = intersection_reduction
        self.difference_reduction = difference_reduction
        self.prototype_initializer = initializers.get(prototype_initializer)
        self.feature_initializer = initializers.get(feature_initializer)
        self.contrast_initializer = initializers.get(contrast_initializer)

        # Initialize weight attributes - they will be created in build()
        self.prototypes = None
        self.feature_bank = None
        self.theta = None
        self.alpha = None
        self.beta = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the layer's weights: prototypes, features, and contrast params."""
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError(
                "The last dimension of the input to `TverskyProjectionLayer` "
                "must be defined. Found `None`."
            )

        # Create the learnable prototype bank
        self.prototypes = self.add_weight(
            name='prototypes',
            shape=(self.units, input_dim),
            initializer=self.prototype_initializer,
            trainable=True,
        )

        # Create the learnable feature universe
        self.feature_bank = self.add_weight(
            name='feature_bank',
            shape=(self.num_features, input_dim),
            initializer=self.feature_initializer,
            trainable=True,
        )

        # Create the Tversky contrast model scalar parameters
        self.theta = self.add_weight(
            name='theta', shape=(), initializer=self.contrast_initializer, trainable=True
        )
        self.alpha = self.add_weight(
            name='alpha', shape=(), initializer=self.contrast_initializer, trainable=True
        )
        self.beta = self.add_weight(
            name='beta', shape=(), initializer=self.contrast_initializer, trainable=True
        )

        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Forward pass computation of Tversky similarity."""
        # Compute dot products to get feature presence scores.
        # inputs shape: (batch_size, input_dim)
        # feature_bank shape: (num_features, input_dim)
        # -> input_dots shape: (batch_size, num_features)
        input_dots = ops.matmul(inputs, ops.transpose(self.feature_bank))

        # prototypes shape: (units, input_dim)
        # -> proto_dots shape: (units, num_features)
        proto_dots = ops.matmul(self.prototypes, ops.transpose(self.feature_bank))

        # Create boolean masks for set operations (feature is present if dot > 0).
        input_mask = input_dots > 0
        proto_mask = proto_dots > 0

        # Reshape for broadcasting:
        # (batch, 1, num_features) vs (1, units, num_features)
        # -> results will have shape: (batch, units, num_features)
        input_dots_b = ops.expand_dims(input_dots, axis=1)
        input_mask_b = ops.expand_dims(input_mask, axis=1)
        proto_dots_b = ops.expand_dims(proto_dots, axis=0)
        proto_mask_b = ops.expand_dims(proto_mask, axis=0)

        # Calculate f(A ∩ B): common features measure.
        common_mask = ops.logical_and(input_mask_b, proto_mask_b)

        if self.intersection_reduction == 'product':
            intersection_scores = input_dots_b * proto_dots_b
        elif self.intersection_reduction == 'min':
            intersection_scores = ops.minimum(input_dots_b, proto_dots_b)
        elif self.intersection_reduction == 'mean':
            intersection_scores = (input_dots_b + proto_dots_b) / 2.0
        else:
            raise NotImplementedError(
                f"Intersection reduction '{self.intersection_reduction}' not implemented."
            )
        f_intersection = ops.sum(
            ops.where(common_mask, intersection_scores, 0.0), axis=-1
        )

        # Calculate f(A - B) and f(B - A): distinctive features measures.
        if self.difference_reduction == 'ignorematch':
            input_distinct_mask = ops.logical_and(input_mask_b, ops.logical_not(proto_mask_b))
            f_input_distinctive = ops.sum(
                ops.where(input_distinct_mask, input_dots_b, 0.0), axis=-1
            )
            proto_distinct_mask = ops.logical_and(proto_mask_b, ops.logical_not(input_mask_b))
            f_proto_distinctive = ops.sum(
                ops.where(proto_distinct_mask, proto_dots_b, 0.0), axis=-1
            )
        elif self.difference_reduction == 'subtractmatch':
            subtract_mask_A = ops.logical_and(common_mask, input_dots_b > proto_dots_b)
            f_input_distinctive = ops.sum(
                ops.where(subtract_mask_A, input_dots_b - proto_dots_b, 0.0), axis=-1
            )
            subtract_mask_B = ops.logical_and(common_mask, proto_dots_b > input_dots_b)
            f_proto_distinctive = ops.sum(
                ops.where(subtract_mask_B, proto_dots_b - input_dots_b, 0.0), axis=-1
            )
        else:
            raise NotImplementedError(
                f"Difference reduction '{self.difference_reduction}' not implemented."
            )

        # Apply Tversky's contrast model formula.
        similarity = (
            self.theta * f_intersection
            - self.alpha * f_input_distinctive
            - self.beta * f_proto_distinctive
        )
        return similarity

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer."""
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of the layer for serialization."""
        config = super().get_config()
        config.update({
            'units': self.units,
            'num_features': self.num_features,
            'intersection_reduction': self.intersection_reduction,
            'difference_reduction': self.difference_reduction,
            'prototype_initializer': initializers.serialize(self.prototype_initializer),
            'feature_initializer': initializers.serialize(self.feature_initializer),
            'contrast_initializer': initializers.serialize(self.contrast_initializer),
        })
        return config

# ---------------------------------------------------------------------
