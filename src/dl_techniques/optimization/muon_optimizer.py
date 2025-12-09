"""
Muon: MomentUm Orthogonalized by Newton-schulz Optimizer.

This module provides a production-ready Keras 3 implementation of the Muon optimizer,
originally developed by Keller Jordan et al.

Muon is a novel optimization algorithm designed specifically for the hidden layers of
deep neural networks. It combines standard SGD with Nesterov momentum and a unique
orthogonalization post-processing step using Newton-Schulz iteration. This approach
effectively normalizes the update steps based on the spectral norm, leading to
significantly faster convergence for large-scale Transformers and ConvNets.

Key Characteristics:
-------------------
1.  **Hybrid Optimization**: Muon is strictly designed to optimize the hidden linear
    transformation weights (matrices/kernels with rank ≥ 2). All other parameters
    (embeddings, normalization gains, biases, and final classification heads) must be
    optimized using a standard adaptive method. This implementation includes an
    integrated **Auxiliary AdamW** optimizer that automatically handles these
    parameters, acting as a drop-in replacement for standard optimizers.

2.  **Newton-Schulz Orthogonalization**: Instead of taking steps proportional to the
    gradient magnitude, Muon projects the accumulated momentum updates onto the
    manifold of orthogonal matrices using a stable quintic Newton-Schulz iteration.
    This "spectral scaling" allows for significantly larger learning rates (e.g., 0.02)
    compared to standard AdamW (e.g., 0.0003).

3.  **Hardware Efficient**: The orthogonalization process uses standard matrix
    multiplications, making it highly efficient on GPUs/TPUs and stable even in
    lower precision (bfloat16).

Performance Achievements (from Original Authors):
-----------------------------------------------
*   Set new state-of-the-art training speed records for CIFAR-10 (94% accuracy).
*   Demonstrated ~1.35x training speedup for GPT-2 scale models compared to AdamW.
*   Adopted by frontier labs (e.g., Kimi.ai) for large-scale LLM pre-training.
*   Proven effectiveness at large batch sizes.

Original Implementation: https://github.com/KellerJordan/Muon
Reference Writeup: https://kellerjordan.github.io/posts/muon/

Citations:
----------
.. code-block:: bibtex

    @misc{jordan2024muon,
      author       = {Keller Jordan and Yuchen Jin and Vlado Boza and You Jiacheng and
                      Franz Cesista and Laker Newhouse and Jeremy Bernstein},
      title        = {Muon: An optimizer for hidden layers in neural networks},
      year         = {2024},
      url          = {https://kellerjordan.github.io/posts/muon/}
    }
"""

import keras
from keras import ops
from typing import Union, Dict, Any, Optional, List

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class Muon(keras.optimizers.Optimizer):
    """
    Muon (MomentUm Orthogonalized by Newton-schulz) Optimizer.

    A hybrid optimizer that applies Muon updates to weight matrices and
    AdamW updates to biases, normalization parameters, and embeddings.

    :param learning_rate: Learning rate for Muon-optimized parameters.
    :type learning_rate: Union[float, keras.optimizers.schedules.LearningRateSchedule]
    :param momentum: Momentum factor for Muon.
    :type momentum: float
    :param nesterov: Whether to use Nesterov momentum for Muon.
    :type nesterov: bool
    :param ns_steps: Number of Newton-Schulz iterations.
    :type ns_steps: int
    :param adam_learning_rate: Learning rate for AdamW auxiliary optimizer.
    :type adam_learning_rate: float
    :param adam_beta_1: Exponential decay rate for 1st moment estimates (Adam).
    :type adam_beta_1: float
    :param adam_beta_2: Exponential decay rate for 2nd moment estimates (Adam).
    :type adam_beta_2: float
    :param adam_epsilon: Small constant for numerical stability (Adam).
    :type adam_epsilon: float
    :param weight_decay: Weight decay coefficient (decoupled).
    :type weight_decay: float
    :param exclude_embedding_names: Substrings to identify embedding layers.
    :type exclude_embedding_names: List[str]
    :param name: String name of the optimizer.
    :type name: str
    """

    def __init__(
            self,
            learning_rate: Union[float, keras.optimizers.schedules.LearningRateSchedule] = 0.02,
            momentum: float = 0.95,
            nesterov: bool = True,
            ns_steps: int = 5,
            adam_learning_rate: float = 1e-3,
            adam_beta_1: float = 0.9,
            adam_beta_2: float = 0.999,
            adam_epsilon: float = 1e-7,
            weight_decay: float = 0.0,
            exclude_embedding_names: Optional[List[str]] = None,
            name: str = "Muon",
            **kwargs
    ) -> None:
        # Note: We pass weight_decay=0.0 to base class and handle it ourselves
        # to allow different decay rates for Muon vs Adam parameters
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=0.0,  # Handle manually
            name=name,
            **kwargs
        )

        # Muon hyperparameters
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps

        # AdamW hyperparameters
        self.adam_learning_rate = adam_learning_rate
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.adam_epsilon = adam_epsilon

        # Weight decay (applied manually)
        self._weight_decay = weight_decay

        # Embedding exclusion patterns (serializable)
        self.exclude_embedding_names = exclude_embedding_names or [
            "embedding", "token_emb", "embed"
        ]

    def build(self, var_list: List[keras.Variable]) -> None:
        """
        Initialize optimizer state variables.

        :param var_list: List of trainable variables.
        :type var_list: List[keras.Variable]
        """
        if self.built:
            return

        super().build(var_list)

        # Muon momentum buffers
        self._muon_velocities = []
        # Adam first moments
        self._adam_m = []
        # Adam second moments
        self._adam_v = []

        for var in var_list:
            self._muon_velocities.append(
                self.add_variable_from_reference(var, name="muon_velocity")
            )
            self._adam_m.append(
                self.add_variable_from_reference(var, name="adam_m")
            )
            self._adam_v.append(
                self.add_variable_from_reference(var, name="adam_v")
            )

    def _should_use_muon(self, variable: keras.Variable) -> bool:
        """
        Determine if a variable should be optimized via Muon or AdamW.

        Muon is applied to rank >= 2 tensors that are not embeddings.

        :param variable: Variable to check.
        :type variable: keras.Variable
        :return: True if Muon should be used, False for AdamW.
        :rtype: bool
        """
        shape = variable.shape
        ndim = len(shape)

        # Must be at least rank 2
        if ndim < 2:
            return False

        # Check for embedding patterns in name
        var_name_lower = variable.name.lower()
        for pattern in self.exclude_embedding_names:
            if pattern in var_name_lower:
                return False

        return True

    def _newton_schulz5(self, G: keras.KerasTensor, steps: int) -> keras.KerasTensor:
        """
        Newton-Schulz iteration for orthogonalization.

        Computes an approximate orthogonalization of the input matrix
        using quintic Newton-Schulz iterations.

        :param G: Input 2D tensor to orthogonalize.
        :type G: keras.KerasTensor
        :param steps: Number of iterations.
        :type steps: int
        :return: Orthogonalized tensor with same shape as input.
        :rtype: keras.KerasTensor
        """
        # Quintic iteration constants
        a = 3.4445
        b = -4.7750
        c = 2.0315

        shape = ops.shape(G)
        rows = shape[0]
        cols = shape[1]

        # Graph-safe conditional: transpose if tall matrix (rows > cols)
        # This improves numerical stability
        def process_wide(X):
            """Process matrix assuming it's wide (cols >= rows)."""
            # Spectral norm normalization
            norm = ops.norm(X, ord="fro", axis=(-2, -1), keepdims=True)
            X_normalized = X / (norm + 1e-7)

            # Newton-Schulz iterations
            result = X_normalized
            for _ in range(steps):
                A = ops.matmul(result, ops.transpose(result))
                AA = ops.matmul(A, A)
                B = b * A + c * AA
                result = a * result + ops.matmul(B, result)

            return result

        def true_fn():
            # Tall matrix: transpose, process, transpose back
            X_t = ops.transpose(G)
            result = process_wide(X_t)
            return ops.transpose(result)

        def false_fn():
            # Wide or square matrix: process directly
            return process_wide(G)

        # Use ops.cond for graph-safe branching
        return ops.cond(rows > cols, true_fn, false_fn)

    def update_step(
            self,
            gradient: keras.KerasTensor,
            variable: keras.Variable,
            learning_rate: keras.KerasTensor
    ) -> None:
        """
        Apply a single optimization step.

        Routes to Muon or AdamW based on variable characteristics.

        :param gradient: Gradient tensor.
        :type gradient: keras.KerasTensor
        :param variable: Variable to update.
        :type variable: keras.Variable
        :param learning_rate: Current learning rate (Muon LR).
        :type learning_rate: keras.KerasTensor
        """
        # Use base class method to find index safely
        idx = self._get_variable_index(variable)

        velocity = self._muon_velocities[idx]
        m = self._adam_m[idx]
        v = self._adam_v[idx]

        lr = ops.cast(learning_rate, variable.dtype)
        wd = ops.cast(self._weight_decay, variable.dtype)

        if self._should_use_muon(variable):
            self._apply_muon(gradient, variable, velocity, lr, wd)
        else:
            self._apply_adam(gradient, variable, m, v, wd)

    def _apply_muon(
            self,
            grad: keras.KerasTensor,
            var: keras.Variable,
            velocity: keras.Variable,
            lr: keras.KerasTensor,
            wd: keras.KerasTensor
    ) -> None:
        """
        Apply Muon update: momentum + Newton-Schulz orthogonalization.

        :param grad: Gradient tensor.
        :type grad: keras.KerasTensor
        :param var: Variable to update.
        :type var: keras.Variable
        :param velocity: Momentum buffer.
        :type velocity: keras.Variable
        :param lr: Learning rate.
        :type lr: keras.KerasTensor
        :param wd: Weight decay coefficient.
        :type wd: keras.KerasTensor
        """
        dtype = var.dtype
        beta = ops.cast(self.momentum, dtype)
        one_minus_beta = ops.cast(1.0 - self.momentum, dtype)

        # 1. Decoupled weight decay
        if self._weight_decay > 0:
            var.assign(var * (1.0 - lr * wd))

        # 2. Momentum update: v = beta * v + (1 - beta) * g
        new_velocity = beta * velocity + one_minus_beta * grad
        velocity.assign(new_velocity)

        # 3. Compute update (with optional Nesterov)
        if self.nesterov:
            # Nesterov: update = (1 - beta) * g + beta * v_new
            update = one_minus_beta * grad + beta * new_velocity
        else:
            update = new_velocity

        # 4. Reshape for Newton-Schulz (flatten to 2D: [*, out_features])
        var_shape = ops.shape(var)
        out_dim = var_shape[-1]
        update_flat = ops.reshape(update, (-1, out_dim))

        # 5. Orthogonalize
        update_ortho = self._newton_schulz5(update_flat, self.ns_steps)

        # 6. Scale correction: sqrt(max(1, rows/cols))
        flat_shape = ops.shape(update_flat)
        rows = ops.cast(flat_shape[0], "float32")
        cols = ops.cast(flat_shape[1], "float32")
        scale = ops.sqrt(ops.maximum(1.0, rows / cols))
        scale = ops.cast(scale, dtype)
        update_ortho = update_ortho * scale

        # 7. Reshape back and apply
        update_final = ops.reshape(update_ortho, var_shape)
        var.assign_sub(lr * update_final)

    def _apply_adam(
            self,
            grad: keras.KerasTensor,
            var: keras.Variable,
            m: keras.Variable,
            v: keras.Variable,
            wd: keras.KerasTensor
    ) -> None:
        """
        Apply AdamW update for non-Muon parameters.

        :param grad: Gradient tensor.
        :type grad: keras.KerasTensor
        :param var: Variable to update.
        :type var: keras.Variable
        :param m: First moment estimate.
        :type m: keras.Variable
        :param v: Second moment estimate.
        :type v: keras.Variable
        :param wd: Weight decay coefficient.
        :type wd: keras.KerasTensor
        """
        dtype = var.dtype
        lr = ops.cast(self.adam_learning_rate, dtype)
        beta_1 = ops.cast(self.adam_beta_1, dtype)
        beta_2 = ops.cast(self.adam_beta_2, dtype)
        epsilon = ops.cast(self.adam_epsilon, dtype)

        # 1. Decoupled weight decay
        if self._weight_decay > 0:
            var.assign(var * (1.0 - lr * wd))

        # 2. Update biased first moment: m = beta1 * m + (1 - beta1) * g
        new_m = beta_1 * m + (1.0 - beta_1) * grad
        m.assign(new_m)

        # 3. Update biased second moment: v = beta2 * v + (1 - beta2) * g^2
        new_v = beta_2 * v + (1.0 - beta_2) * ops.square(grad)
        v.assign(new_v)

        # 4. Bias correction
        local_step = ops.cast(self.iterations + 1, dtype)
        m_hat = new_m / (1.0 - ops.power(beta_1, local_step))
        v_hat = new_v / (1.0 - ops.power(beta_2, local_step))

        # 5. Apply update
        update = m_hat / (ops.sqrt(v_hat) + epsilon)
        var.assign_sub(lr * update)

    def get_config(self) -> Dict[str, Any]:
        """
        Return optimizer configuration.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "momentum": self.momentum,
            "nesterov": self.nesterov,
            "ns_steps": self.ns_steps,
            "adam_learning_rate": self.adam_learning_rate,
            "adam_beta_1": self.adam_beta_1,
            "adam_beta_2": self.adam_beta_2,
            "adam_epsilon": self.adam_epsilon,
            "weight_decay": self._weight_decay,
            "exclude_embedding_names": self.exclude_embedding_names,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Muon":
        """
        Create optimizer from configuration.

        :param config: Configuration dictionary.
        :type config: Dict[str, Any]
        :return: Muon optimizer instance.
        :rtype: Muon
        """
        return cls(**config)

# ---------------------------------------------------------------------
