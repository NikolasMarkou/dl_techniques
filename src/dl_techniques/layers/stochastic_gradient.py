"""
Stochastic gradient blocking as a regularization mechanism.

This layer embodies the principle of asymmetric forward and backward
computation, a design paradigm that leaves the network's activations entirely
untouched while injecting stochasticity into the gradient signal alone. The
core idea is that regularization does not require perturbing what the network
computes; it is sufficient to perturb what the network learns from. By randomly
severing gradient paths, the optimizer sees a different subgraph on each step,
which discourages downstream layers from developing rigid co-dependencies on
any single upstream path.

The distinction from Stochastic Depth is the essential point. Stochastic Depth
removes a residual branch from the forward pass, so activations, batch
statistics, and the effective network depth all change from step to step, and
an expectation correction is needed at inference time to compensate. This layer
changes none of them. The forward map is the identity under all conditions, in
both training and inference, so activation magnitudes, normalization statistics,
and the deterministic input-output behaviour of the model are preserved exactly.
Only the Jacobian seen by backpropagation is stochastic.

Mechanically, with keep probability `p = 1 - drop_path_rate`, a single scalar
`u ~ U(0, 1)` is drawn per call and the layer resolves to one of two branches:

`u < p   ->  y = x`                    (gradient flows unchanged)
`u >= p  ->  y = stop_gradient(x)`     (gradient is severed)

Both branches emit the same value, so the choice is invisible in the forward
direction. In the first case the local Jacobian is the identity and upstream
parameters receive their gradient normally. In the second, `stop_gradient`
makes the output a constant with respect to the input, so the local Jacobian is
zero and every parameter reachable only through this path receives no update on
that step. The decision is a single scalar rather than a per-example or
per-element mask, so an entire path is either live or dead for the whole batch,
which mirrors the path-level granularity of drop-path methods.

Because gradient computation does not occur outside of training, the layer is a
strict no-op at inference: no scaling correction, no branch selection, and no
runtime cost beyond passing the tensor through. The selection is expressed with
`keras.ops.cond` rather than Python control flow so that the branch remains a
traceable graph operation under compiled and graph-mode execution rather than
being frozen at trace time.

References:
    - Huang et al., 2016. Deep Networks with Stochastic Depth.
      (https://arxiv.org/abs/1603.09382)
    - Srivastava et al., 2014. Dropout: A Simple Way to Prevent Neural Networks
      from Overfitting. JMLR 15(56).
    - Larsson et al., 2017. FractalNet: Ultra-Deep Neural Networks without
      Residuals. (https://arxiv.org/abs/1605.07648)

"""

import keras
from typing import Optional, Dict, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class StochasticGradient(keras.layers.Layer):
    """
    Stochastic Gradient dropping regularization for deep networks.

    This layer stochastically stops gradient flow during backpropagation with
    probability ``drop_path_rate``. The forward pass is always an identity
    function -- unlike Stochastic Depth, only the backward pass is affected.
    During inference the layer has no effect.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────┐
        │  Input [any shape]              │
        └──────────────┬──────────────────┘
                       ▼
        ┌─────────────────────────────────┐
        │  Forward: identity (always)     │
        │  Backward (training):           │
        │    p < keep_prob → pass grads   │
        │    p ≥ keep_prob → stop_gradient│
        └──────────────┬──────────────────┘
                       ▼
        ┌─────────────────────────────────┐
        │  Output [same shape as input]   │
        └─────────────────────────────────┘

    :param drop_path_rate: Probability of stopping the gradient. Must be in
        ``[0, 1)``. Defaults to 0.5.
    :type drop_path_rate: float
    :param kwargs: Additional keyword arguments for the parent Layer class.
    :type kwargs: Any
    """

    def __init__(
            self,
            drop_path_rate: float = 0.5,
            **kwargs: Any
    ) -> None:
        """
        Initialize the StochasticGradient layer.

        :param drop_path_rate: Probability of dropping the gradient. Must be in ``[0, 1)``.
        :type drop_path_rate: float
        :param kwargs: Additional keyword arguments for the parent Layer class.
        :type kwargs: Any
        """
        super().__init__(**kwargs)

        # Validate drop_path_rate
        if not isinstance(drop_path_rate, (int, float)):
            raise TypeError("drop_path_rate must be a number")
        if not 0.0 <= drop_path_rate < 1.0:
            raise ValueError(
                f"drop_path_rate must be in [0, 1), got {drop_path_rate}"
            )

        self.drop_path_rate = float(drop_path_rate)

        logger.info(
            f"Created StochasticGradient layer '{self.name}' with "
            f"drop_path_rate={self.drop_path_rate}"
        )

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of the layer.

        :param inputs: Input tensor.
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: bool or None
        :return: Output tensor (same as input; gradient may be stopped during training).
        :rtype: keras.KerasTensor
        """
        if training is False or self.drop_path_rate == 0.0:
            return inputs

        # Determine whether to drop the gradient for this call
        keep_prob =1.0 - self.drop_path_rate

        random_tensor = keras.random.uniform(shape=[])

        # Use a conditional to apply stop_gradient.
        # This ensures that the graph is traceable by frameworks like TF.
        return keras.ops.cond(
            random_tensor < keep_prob,
            lambda: inputs,
            lambda: keras.ops.stop_gradient(inputs)
        )

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        :return: Output shape tuple (same as input shape).
        :rtype: tuple
        """
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the config dictionary for layer serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            "drop_path_rate": self.drop_path_rate,
        })
        return config

# ---------------------------------------------------------------------
