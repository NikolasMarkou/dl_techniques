"""GPT-2 Pre-training with Soft Orthonormal Regularization.

Extends the standard GPT-2 CLM pre-training with soft orthonormal
constraints on weight matrices. Penalizes ``||W^T·W - I||_F^2`` on
all Dense/attention kernel weights, encouraging:

1. **Gradient preservation** — orthonormal weights maintain gradient
   magnitudes during backpropagation, reducing vanishing/exploding.
2. **Better conditioning** — the loss landscape is smoother when
   weight matrices are well-conditioned (singular values near 1).
3. **Improved generalization** — constraining the capacity of each
   layer reduces memorization and improves out-of-distribution behavior.

The regularization strength (``--so-lambda``) controls the trade-off
between language modeling loss and orthonormality. Too high → underfits
(weights forced to identity-like). Too low → no effect.

Usage::

    # Standard SO pretraining
    python -m train.gpt2.pretrain_so --gpu 0 --variant small --epochs 3

    # Custom regularization strength
    python -m train.gpt2.pretrain_so --so-lambda 1e-4 --so-l2 0.0

    # Resume from checkpoint
    python -m train.gpt2.pretrain_so --resume path/to/step_0450000.keras

References:
    Bansal et al. (2018). "Can We Gain More from Orthogonality
    Regularizations in Training Deep CNNs?" NeurIPS.
"""

import keras
import argparse
from dataclasses import dataclass
from typing import Optional

from train.common import setup_gpu
from train.gpt2.pretrain import (
    TrainingConfig,
    _build_parser,
    _config_from_args,
)
from dl_techniques.utils.logger import logger
from dl_techniques.regularizers import SoftOrthonormalConstraintRegularizer


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


@dataclass
class SOTrainingConfig(TrainingConfig):
    """Training config extended with soft orthonormal regularization.

    :param so_lambda: Strength of the orthonormality penalty
        ``||W^T·W - I||_F^2``. Default ``1e-3``.
    :param so_l1: Optional L1 penalty on weights. Default ``0.0``.
    :param so_l2: Optional L2 penalty on weights. Default ``1e-4``.
    :param so_matrix_scaling: Scale penalty by matrix size for
        consistent effect across layers. Default ``True``.
    :param so_skip_embeddings: Skip embedding layers (large, sparse,
        orthonormality less meaningful). Default ``True``.
    """

    so_lambda: float = 1e-3
    so_l1: float = 0.0
    so_l2: float = 1e-4
    so_matrix_scaling: bool = True
    so_skip_embeddings: bool = True


# ---------------------------------------------------------------------
# Regularization Application
# ---------------------------------------------------------------------


def collect_kernel_weights(
    model: keras.Model,
    skip_embeddings: bool = True,
) -> list:
    """Collect all kernel (dense/attention) weight tensors from the model.

    Recursively walks layers, collecting 2D+ weight tensors that
    represent linear projections (Q, K, V, O, FFN up/down).

    :param model: Built Keras model.
    :param skip_embeddings: Skip embedding layer weights.
    :return: List of weight tensors.
    """
    kernels = []

    def _collect(layer):
        if skip_embeddings and isinstance(layer, keras.layers.Embedding):
            return
        for w in layer.weights:
            # Kernel weights are typically 2D (in, out) or higher (conv)
            # Skip biases (1D), norms (1D), and embedding matrices
            if (
                len(w.shape) >= 2
                and "bias" not in w.name
                and "norm" not in w.name
                and (not skip_embeddings or "embedding" not in w.name)
            ):
                kernels.append(w)
        if hasattr(layer, "layers"):
            for sub in layer.layers:
                _collect(sub)

    for layer in model.layers:
        _collect(layer)

    return kernels


class SoftOrthonormalCallback(keras.callbacks.Callback):
    """Add soft orthonormal regularization loss during training.

    Computes ``||W^T·W - I||_F^2`` for all collected kernel weights
    and adds the penalty to the model's loss via ``add_loss()``.
    This approach works reliably with pre-built models where
    post-hoc ``kernel_regularizer`` assignment doesn't register.

    :param regularizer: The SO regularizer instance.
    :param skip_embeddings: Skip embedding layers.
    """

    def __init__(
        self,
        regularizer: SoftOrthonormalConstraintRegularizer,
        skip_embeddings: bool = True,
    ):
        super().__init__()
        self._regularizer = regularizer
        self._skip_embeddings = skip_embeddings
        self._kernels = []

    def set_model(self, model):
        super().set_model(model)
        self._kernels = collect_kernel_weights(
            model, self._skip_embeddings,
        )
        logger.info(
            f"SoftOrthonormalCallback: tracking {len(self._kernels)} "
            f"kernel weights"
        )

    def on_train_batch_begin(self, batch, logs=None):
        """Add SO penalty before each training step."""
        for w in self._kernels:
            self.model.add_loss(lambda w=w: self._regularizer(w))


# ---------------------------------------------------------------------
# Override model creation to inject regularization
# ---------------------------------------------------------------------


import train.gpt2.pretrain as _pretrain_module

_original_train = _pretrain_module.train_gpt2


def train_gpt2_so(config: SOTrainingConfig):
    """Wrap train_gpt2 to inject the SO callback."""

    # Monkey-patch create_nlp_callbacks to inject our callback
    from train.common.nlp import create_nlp_callbacks as _orig_callbacks

    regularizer = SoftOrthonormalConstraintRegularizer(
        lambda_coefficient=config.so_lambda,
        l1_coefficient=config.so_l1,
        l2_coefficient=config.so_l2,
        use_matrix_scaling=config.so_matrix_scaling,
    )
    so_callback = SoftOrthonormalCallback(
        regularizer=regularizer,
        skip_embeddings=config.so_skip_embeddings,
    )

    # Inject SO callback by patching the callback creation
    _orig_create_callbacks = _pretrain_module.create_nlp_callbacks

    def _patched_callbacks(*args, **kwargs):
        callbacks, results_dir = _orig_create_callbacks(*args, **kwargs)
        callbacks.append(so_callback)
        return callbacks, results_dir

    _pretrain_module.create_nlp_callbacks = _patched_callbacks
    try:
        return _original_train(config)
    finally:
        _pretrain_module.create_nlp_callbacks = _orig_create_callbacks


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def _build_so_parser() -> argparse.ArgumentParser:
    """Extend the base parser with SO regularization args."""
    p = _build_parser()
    p.description = "GPT-2 Pre-training with Soft Orthonormal Regularization"

    so_group = p.add_argument_group("Soft Orthonormal Regularization")
    so_group.add_argument(
        "--so-lambda", type=float, default=1e-3,
        help="Orthonormality penalty strength ||W^T·W - I||_F^2",
    )
    so_group.add_argument(
        "--so-l1", type=float, default=0.0,
        help="L1 weight penalty (on top of orthonormality)",
    )
    so_group.add_argument(
        "--so-l2", type=float, default=1e-4,
        help="L2 weight penalty (on top of orthonormality)",
    )
    so_group.add_argument(
        "--no-so-matrix-scaling", action="store_true",
        help="Disable matrix-size scaling of the penalty",
    )
    so_group.add_argument(
        "--so-include-embeddings", action="store_true",
        help="Also regularize embedding layers (not recommended)",
    )

    return p


def _so_config_from_args(args: argparse.Namespace) -> SOTrainingConfig:
    """Map parsed CLI args to SOTrainingConfig."""
    base = _config_from_args(args)
    return SOTrainingConfig(
        **{k: v for k, v in base.__dict__.items()},
        so_lambda=args.so_lambda,
        so_l1=args.so_l1,
        so_l2=args.so_l2,
        so_matrix_scaling=not args.no_so_matrix_scaling,
        so_skip_embeddings=not args.so_include_embeddings,
    )


def main() -> None:
    """Main entry point for GPT-2 SO pre-training."""
    args = _build_so_parser().parse_args()
    setup_gpu(gpu_id=args.gpu)

    config = _so_config_from_args(args)
    logger.info(
        f"Config: variant={config.gpt2_variant}, "
        f"epochs={config.num_epochs}, batch={config.batch_size}, "
        f"lr={config.learning_rate}, loss={config.loss_type}, "
        f"SO(λ={config.so_lambda}, l2={config.so_l2})"
    )

    train_gpt2_so(config)


if __name__ == "__main__":
    main()
