"""What a denoising run compiles with, and the scores of the model that does nothing.

The ``image_denoising`` task of ``train.api`` puts the loss and three metrics into ``model.compile`` and
compares every score with the same score of the IDENTITY (the noisy input returned as the denoised image): a
denoiser that does not beat its own input has learned nothing, however low its loss. This module is the one
home of those pieces; the pure image-metric helpers of the legacy evaluation (``train.bfunet``) are not here.

Interface contract
------------------
``DENOISING_METRIC_NAMES``
    ``("mae", "psnr", "ssim")``: the history keys (and, with ``val_``, the validation columns) of the metrics
    :func:`denoising_metrics` builds, known without building one (the task's ``metric_names`` hook needs them
    before the GPU exists, and a Keras metric allocates variables).

``LOSS_NAMES``
    ``("mse", "mae", "charbonnier")``, the values of ``loss_args.name``; ``train.api.denoising_args`` holds the same
    tuple and a test holds the two equal.

``CHARBONNIER_EPSILON``
    ``1e-3``, the constant of the Charbonnier loss ``sqrt(d^2 + eps^2)`` (the library's default).

``denoising_loss(name)``
    :param name: one of :data:`LOSS_NAMES`.
    :return: a fresh ``keras.losses.Loss``: the mean squared error, the mean absolute error, or
        :class:`~dl_techniques.losses.image_restoration_loss.CharbonnierLoss`. Each is the mean over every pixel of
        the batch, so ``val_loss`` is the mean over the validation patches whatever the batch size.
    :raises ValueError: for an unknown name.

``denoising_metrics()``
    :return: ``[mae, psnr, ssim]`` fresh: ``MeanAbsoluteError(name="mae")``,
        ``PsnrMetric(max_val=1.0, name="psnr")`` and ``SsimMetric(max_val=1.0, name="ssim")``. The images are in
        ``[0, 1]``, so ``max_val`` is 1. PSNR is the MEAN OVER IMAGES of each image's PSNR (the library's metric),
        which is not the PSNR of the mean squared error and is capped at 70 dB per image (its MSE floor is
        ``keras.config.epsilon()``).

``identity_scores(dataset, loss_name)``
    :param dataset: a batched dataset of ``(noisy, clean)`` (the task's validation pipeline).
    :param loss_name: the loss of the run, so the identity loss is the loss of the same quantity.
    :return: ``{"loss", "mae", "psnr", "ssim"}`` of the model that returns its input, through the SAME loss and
        metric objects the run compiles and the same ``evaluate`` that reads ``val_psnr``, so ``psnr`` (the INPUT
        PSNR) and the run's ``val_psnr`` are the same statistic. The identity layer is float32 whatever the
        global dtype policy (a half-precision copy of a ``[0, 1]`` image would cost about 0.5e-3 of error).
    :raises ValueError: for an unknown ``loss_name`` or an empty dataset.
"""

from typing import Dict, List

import keras
import tensorflow as tf

from dl_techniques.losses.image_restoration_loss import CharbonnierLoss
from dl_techniques.metrics.psnr_metric import PsnrMetric
from dl_techniques.metrics.ssim_metric import SsimMetric

# ---------------------------------------------------------------------

DENOISING_METRIC_NAMES = ("mae", "psnr", "ssim")

LOSS_NAMES = ("mse", "mae", "charbonnier")

CHARBONNIER_EPSILON = 1e-3


def denoising_loss(name: str) -> keras.losses.Loss:
    """Return a fresh loss of ``name``; see the module contract."""
    if name == "mse":
        return keras.losses.MeanSquaredError()
    if name == "mae":
        return keras.losses.MeanAbsoluteError()
    if name == "charbonnier":
        return CharbonnierLoss(epsilon=CHARBONNIER_EPSILON)
    raise ValueError(f"Unknown denoising loss {name!r}; known: {list(LOSS_NAMES)}")


def denoising_metrics() -> List[keras.metrics.Metric]:
    """Return fresh ``[mae, psnr, ssim]``; see the module contract."""
    return [
        keras.metrics.MeanAbsoluteError(name="mae"),
        PsnrMetric(max_val=1.0, name="psnr"),
        SsimMetric(max_val=1.0, name="ssim"),
    ]


def identity_scores(dataset: "tf.data.Dataset", loss_name: str) -> Dict[str, float]:
    """Return the scores of the identity "denoiser" on ``dataset``; see the module contract."""
    spec = dataset.element_spec[0]
    inputs = keras.Input(shape=tuple(spec.shape[1:]), dtype="float32")
    identity = keras.Model(inputs, keras.layers.Identity(dtype="float32")(inputs), name="identity")
    # `jit_compile=False`: this model exists to be evaluated once; an XLA compile for it is pure cost.
    identity.compile(loss=denoising_loss(loss_name), metrics=denoising_metrics(), jit_compile=False)
    scores = identity.evaluate(dataset, verbose=0, return_dict=True)
    if not scores or "loss" not in scores:
        raise ValueError("the dataset holds no batch, so the identity has no score")
    return {key: float(value) for key, value in scores.items()}
