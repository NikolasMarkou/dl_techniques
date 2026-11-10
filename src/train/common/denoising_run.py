"""What a denoising run compiles with, scores and draws: the loss and metrics, the identity's scores, the held-out
cells and the figures.

The ``image_denoising`` task of ``train.api`` puts the loss and three metrics into ``model.compile`` and
compares every score with the same score of the IDENTITY (the noisy input returned as the denoised image): a
denoiser that does not beat its own input has learned nothing, however low its loss. This module is the one
home of those pieces and of the pure image-metric helpers the legacy evaluation (``train.bfunet``) shares with
the task (``expected_noise_variance``, ``expected_input_psnr``, ``add_awgn``, ``psnr_per_image``,
``predict_denoised``, ``denoise_whole_image``): ``train.bfunet.common`` and ``train.bfunet.eval_psnr_vs_noise``
import them from here, so each has one implementation (D-017).

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

``MAX_VAL`` / ``DATA_MIN`` / ``DATA_MAX``
    ``1.0`` and ``0.0``, ``1.0``: the peak-to-peak width and the pixel domain of every denoiser here. The width is
    domain-invariant (it was 1.0 on the legacy ``[-0.5, +0.5]`` domain too), so do NOT "fix" it. ``train.bfunet.common``
    keeps its own ``DATA_MIN``/``DATA_MAX`` pair (the legacy trainer's) and a test holds the two equal.

``expected_noise_variance(sigma_min, sigma_max)`` / ``expected_input_psnr(sigma_min, sigma_max)``
    Moved verbatim from ``train.bfunet.common``: ``E[sigma^2]`` and the mean per-image PSNR of the noisy input for a
    sigma uniform on the range (additive noise only).

``add_awgn(clean, sigma_norm, clip, rng)`` / ``psnr_per_image(denoised, clean, max_val)``
    Moved verbatim from ``train.bfunet.eval_psnr_vs_noise``: numpy additive noise from a ``RandomState`` and the
    per-image PSNR of a batch ``(N, H, W, C)`` (the MSE floored at ``1e-12``).

``predict_denoised(model, noisy, batch_size)`` / ``denoise_whole_image(model, noisy, size_multiple, eager=False)``
    ``predict_denoised`` is the old ``_predict_denoised`` (padded to one trace, multi-output reduced to output 0);
    ``denoise_whole_image`` is the old ``_denoise_full`` (reflect-pad H and W to a multiple of ``size_multiple``, one
    forward pass, crop back). ``eager=True`` (the held-out cells) runs ``model(x, training=False)`` instead of
    ``model.predict``: no per-size XLA compile and no per-size trace (an XLA compile is 3 to 44 s per new image size,
    measured in step 2). The default is what the legacy script always did, so its numbers do not change.

``HELD_OUT_SEED``, ``HELD_OUT_GB_PER_MEGAPIXEL``, ``HELD_OUT_MEMORY_BUDGET_GB``
    The fixed noise seed of the held-out cells (a constant, NOT the run's ``seed``: two runs are compared on the same
    noise), and the whole-image memory rule: peak GPU memory of one forward pass is estimated as
    ``HELD_OUT_GB_PER_MEGAPIXEL`` (1.6, measured for a base-size U-Net without XLA: 6.4 GB at 4.19 Mpx) times the padded
    megapixels; an image above ``HELD_OUT_MEMORY_BUDGET_GB`` (8.0) is NAMED and SKIPPED, not tiled or guessed at.

``HeldOutSpec``
    ``sigmas_255``, ``noise_kind``, ``composite_ratio``, ``clip`` (the run's noise), ``channels``, ``size_multiple``,
    ``seed`` and ``budget_gb``.

``score_held_out(denoise, datasets, spec)``
    :param denoise: ``(H, W, C) float32 -> (H, W, C)``; the caller binds the model (and its padding).
    :param datasets: ``{name: [image paths]}``, whole images.
    :return: :class:`HeldOutScores`: for every dataset and sigma the per-image PSNR of the noisy input and of the
        denoised image, and the images skipped by the memory rule (path, size, estimate). The noise of image ``i`` of
        dataset ``d`` at sigma ``s`` is ``add_noise`` with the seed ``(spec.seed, f(d, i, s))``: a pure function of the
        image's place, so every model scored with the same ``spec`` sees the same noise, in any order.

``pair_held_out(best, final)``
    :return: ``{dataset: {sigma: {"input_psnr", "best_psnr", "final_psnr", "n"}}}``: each value the mean over the
        dataset's images of the per-image PSNR (a cell is per dataset and sigma; nothing here averages datasets).
    :raises ValueError: when the two scorings did not see the same noisy inputs or skipped different images.

``DenoisingTriptychCallback(noisy, clean, out_dir, title)``
    Writes ``epoch_NNN_triptych.png`` (noisy | denoised | clean | noisy minus denoised, one row per fixed patch) after
    epoch 1, every ``max(1, epochs // 20)`` epochs and at the end. Predictions use ``training=False``; a figure that
    raises is a WARNING and ``failed``, never a stopped run.

``residual_autocorrelation(residual, max_lag)`` / ``write_end_figures(vis_dir, ...)``
    The normalised autocorrelation of the residual along width and height (white noise reads about 0 at every lag),
    and the end-of-run files ``psnr_vs_sigma.png`` (one panel per dataset: input, best, final over sigma) and
    ``residual_whiteness.png``; each independent, a failure listed in ``failed``.

``first_pairs(dataset, count)`` / ``AnalysisSample(x_fig, y_fig)``
    The first ``count`` ``(noisy, clean)`` patches of a batched dataset as numpy arrays, and the record
    ``report_best_weights`` takes as its ``data`` (it reads only those two attributes).
"""

import dataclasses
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import keras
import numpy as np
import tensorflow as tf
from matplotlib.figure import Figure
from PIL import Image

from dl_techniques.losses.image_restoration_loss import CharbonnierLoss
from dl_techniques.metrics.psnr_metric import PsnrMetric
from dl_techniques.metrics.ssim_metric import SsimMetric
from dl_techniques.utils.logger import logger
from train.common.classification_data import decode_image_file
from train.common.denoising_data import add_noise

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


# ---------------------------------------------------------------------
# The pure helpers the legacy evaluation shares (moved here, D-017)
# ---------------------------------------------------------------------

# Images live in [0, 1] -> peak-to-peak range = 1.0. This value is domain-INVARIANT (it was also 1.0 on the legacy
# [-0.5, +0.5] domain, whose peak-to-peak width is likewise 1.0). Do NOT "fix" it to match the new domain -- see
# `train.bfunet.common` D-001 / plan INV-2. `train.bfunet.common` keeps its own DATA_MIN / DATA_MAX (the legacy
# trainer's pair) because this module must not import `train.bfunet`; a test holds the two pairs equal.
MAX_VAL: float = 1.0
DATA_MIN: float = 0.0
DATA_MAX: float = 1.0


def expected_noise_variance(sigma_min: float, sigma_max: float) -> float:
    """``E[sigma^2]`` for ``sigma ~ U(sigma_min, sigma_max)`` = ``(a^2 + ab + b^2)/3``.

    This is the per-epoch NOISE FLOOR: the MSE a do-nothing identity denoiser would
    score against the clean target. It is the natural difficulty normalizer for the
    curriculum (the training noise fn draws a fresh ``sigma`` per image from exactly
    this uniform range -- see ``make_curriculum_noise_fn``).

    Slight over-estimate in practice: the pipeline clips the noisy patch to
    ``[DATA_MIN, DATA_MAX]``, which removes a little noise energy near the boundaries.
    So the reported denoising gain is a mild UNDER-estimate. Additive noise only.
    """
    a, b = float(sigma_min), float(sigma_max)
    if b < a:
        a, b = b, a
    return (a * a + a * b + b * b) / 3.0


def expected_input_psnr(sigma_min: float, sigma_max: float) -> float:
    """Mean per-image PSNR (dB) of the NOISY INPUT for ``sigma ~ U(sigma_min, sigma_max)``.

    Per-image PSNR at ``max_val=1`` is ``-20*log10(sigma)``, and ``PsnrMetric`` averages
    per-image PSNR, so the reference is ``E[-20*log10(sigma)]``, NOT ``-20*log10(E[sigma])``
    (Jensen -- the two differ). With ``E[ln s] = (b(ln b - 1) - a(ln a - 1)) / (b - a)``.

    The ``a = 0`` case is finite even though a single ``sigma -> 0`` image has infinite
    PSNR: the limit of ``a*ln(a)`` is 0, giving ``E[ln s] -> ln(b) - 1``.
    """
    a, b = float(sigma_min), float(sigma_max)
    if b <= 0.0:
        return float("nan")
    if a <= 0.0:
        mean_ln = float(np.log(b)) - 1.0
    elif abs(b - a) < 1e-12:
        mean_ln = float(np.log(a))
    else:
        mean_ln = (b * (np.log(b) - 1.0) - a * (np.log(a) - 1.0)) / (b - a)
    return float(-20.0 / np.log(10.0) * mean_ln)


def add_awgn(clean: np.ndarray, sigma_norm: float, clip: bool, rng: np.random.RandomState) -> np.ndarray:
    """Additive white Gaussian noise ``y = x + N(0, sigma^2)`` in normalized space."""
    noisy = clean + rng.standard_normal(clean.shape).astype(np.float32) * np.float32(sigma_norm)
    return np.clip(noisy, DATA_MIN, DATA_MAX) if clip else noisy


def psnr_per_image(denoised: np.ndarray, clean: np.ndarray, max_val: float = MAX_VAL) -> np.ndarray:
    """Per-image PSNR (dB): ``10 * log10(max_val^2 / MSE)`` -- matches PsnrMetric."""
    mse = np.mean((denoised - clean) ** 2, axis=(1, 2, 3))
    mse = np.maximum(mse, 1e-12)  # guard against perfect reconstruction / log(0)
    return 10.0 * np.log10((max_val ** 2) / mse)


def predict_denoised(model: keras.Model, noisy: np.ndarray, batch_size: int) -> np.ndarray:
    """Run the denoiser; reduce multi-output models to output 0 (final denoised image)."""
    # DECISION plan-2026-09-19T224205-49c8bf80/D-057: pad to a multiple of ``batch_size``, crop back. A
    # short last batch is a second trace per model and TF counts traces across all models: the
    # ``retracing`` warning. Do NOT drop the padding; guard: ``TestPredictTracesOncePerModel``.
    n = len(noisy)
    padded = noisy[np.arange(-(-n // batch_size) * batch_size) % n]
    pred = model.predict(padded, batch_size=batch_size, verbose=0)
    if isinstance(pred, (list, tuple)):
        pred = pred[0]  # deep supervision / exposed bottleneck -> final denoised output
    return np.asarray(pred)[:n]


def denoise_whole_image(
        model: keras.Model, noisy: np.ndarray, size_multiple: int, eager: bool = False) -> np.ndarray:
    """Denoise one whole image: reflect-pad H,W to a multiple of ``size_multiple`` (the
    U-Net downsample factor), run a single forward pass, then crop back to the original size.

    ``eager`` runs ``model(x, training=False)`` instead of ``model.predict`` (see the module contract)."""
    h, w = noisy.shape[:2]
    ph = (size_multiple - h % size_multiple) % size_multiple
    pw = (size_multiple - w % size_multiple) % size_multiple
    padded = np.pad(noisy, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    if eager:
        out = model(padded[None, ...], training=False)
        if isinstance(out, (list, tuple)):
            out = out[0]
        out = np.asarray(keras.ops.convert_to_numpy(out), dtype=np.float32)[0]
    else:
        out = predict_denoised(model, padded[None, ...], batch_size=1)[0]
    return out[:h, :w, :]


# ---------------------------------------------------------------------
# Held-out cells: whole images, the run's noise, the same noise for every model
# ---------------------------------------------------------------------

# DECISION plan-2026-11-10T095307-d483ffab/D-026
# The held-out cells are scored on WHOLE images with the run's noise, the same noise for the best and the final model
# (the noise is a pure function of image place and sigma, so each model is scored in its own pass and the two
# passes are checked equal in `pair_held_out`), and the seed is a CONSTANT, not the run's `seed`: a run of seed 43 and a
# run of seed 42 must meet the same noisy images, or their difference is partly noise. Do NOT key the noise on `seed`
# or on the model, do NOT average cells over datasets (a cell is one dataset at one sigma), and do NOT run the models
# through `model.predict` (an XLA compile per image size, 3 to 44 s each). The stream is not the legacy
# `RandomState` one, so the numbers agree with `eval_psnr_vs_noise` in distribution, not byte for byte.
HELD_OUT_SEED = 1009

#: Peak GPU memory of one whole-image forward pass without XLA, per padded megapixel, for a base-size U-Net (step 2
#: item h: 6.4 GB at 4.19 Mpx; SCUNet tiny needed 1.0 GB per Mpx, a smaller number, so this one is the safe side).
HELD_OUT_GB_PER_MEGAPIXEL = 1.6

#: The budget of one forward pass: about what is free on the 24 GB card with an 11 GB resident neighbour, and
#: two thirds of the 12 GB card. An image whose estimate is above it is named and skipped.
HELD_OUT_MEMORY_BUDGET_GB = 8.0

#: The keys of a dataset's seed: ``(dataset index * 10**6 + image index) * 10**6 + sigma_255 * 1000``.
_SEED_STRIDE = 10 ** 6


@dataclasses.dataclass(frozen=True)
class HeldOutSpec:
    """What every model of a held-out evaluation shares; see the module contract."""

    sigmas_255: Tuple[float, ...]
    noise_kind: str
    composite_ratio: float
    clip: bool
    channels: int
    size_multiple: int
    seed: int = HELD_OUT_SEED
    budget_gb: float = HELD_OUT_MEMORY_BUDGET_GB


@dataclasses.dataclass
class HeldOutScores:
    """The per-image PSNR of one model on every cell: ``{dataset: {sigma key: float64 array}}`` twice, and the
    images the memory rule skipped (``{dataset: [{path, height, width, estimate_gb, budget_gb}]}``)."""

    input_psnr: Dict[str, Dict[str, np.ndarray]]
    output_psnr: Dict[str, Dict[str, np.ndarray]]
    skipped: Dict[str, List[Dict[str, Any]]]


def sigma_key(sigma_255: float) -> str:
    """The key of a sigma in the summary: ``15.0`` -> ``"15"``, ``12.5`` -> ``"12.5"``."""
    return format(float(sigma_255), "g")


def whole_image_size(path: str) -> Tuple[int, int]:
    """``(height, width)`` of an image file from its header; no pixel is decoded."""
    with Image.open(path) as image:
        width, height = image.size
    return height, width


def whole_image_peak_gb(height: int, width: int, size_multiple: int) -> float:
    """The estimated peak memory (GB) of one forward pass over an image padded to ``size_multiple``."""
    padded_h = -(-height // size_multiple) * size_multiple
    padded_w = -(-width // size_multiple) * size_multiple
    return HELD_OUT_GB_PER_MEGAPIXEL * padded_h * padded_w / 1e6


def held_out_noise_seed(seed: int, dataset_index: int, image_index: int, sigma_255: float) -> tf.Tensor:
    """The int64 seed pair of one image at one sigma: a pure function of its place, never of a model or an order."""
    if not (0 <= image_index < _SEED_STRIDE and 0 <= dataset_index < 1000 and 0 < sigma_255 <= 255):
        raise ValueError(
            f"held-out seed out of range: dataset {dataset_index}, image {image_index}, sigma_255 {sigma_255}")
    key = (dataset_index * _SEED_STRIDE + image_index) * _SEED_STRIDE + int(round(sigma_255 * 1000))
    return tf.constant([seed, key], dtype=tf.int64)


def score_held_out(
        denoise: Callable[[np.ndarray], np.ndarray], datasets: Mapping[str, Sequence[str]], spec: HeldOutSpec,
) -> HeldOutScores:
    """Score ``denoise`` on every whole image of ``datasets`` at every sigma; see the module contract."""
    keys = [sigma_key(sigma) for sigma in spec.sigmas_255]
    inputs: Dict[str, Dict[str, np.ndarray]] = {}
    outputs: Dict[str, Dict[str, np.ndarray]] = {}
    skipped: Dict[str, List[Dict[str, Any]]] = {}
    for dataset_index, (name, files) in enumerate(datasets.items()):
        in_cols: Dict[str, List[float]] = {key: [] for key in keys}
        out_cols: Dict[str, List[float]] = {key: [] for key in keys}
        skipped[name] = []
        for image_index, path in enumerate(files):
            height, width = whole_image_size(path)
            estimate = whole_image_peak_gb(height, width, spec.size_multiple)
            if estimate > spec.budget_gb:
                skipped[name].append({
                    "path": str(path), "height": height, "width": width,
                    "estimate_gb": round(estimate, 2), "budget_gb": spec.budget_gb,
                })
                logger.warning(
                    f"held-out {name}: {path} ({height}x{width}) skipped: a whole-image forward pass is estimated at "
                    f"{estimate:.1f} GB, above the {spec.budget_gb:g} GB budget")
                continue
            clean = decode_image_file(str(path), spec.channels).numpy()
            for sigma_255, key in zip(spec.sigmas_255, keys):
                noisy = add_noise(
                    tf.constant(clean), tf.constant(sigma_255 / 255.0, tf.float32),
                    held_out_noise_seed(spec.seed, dataset_index, image_index, sigma_255),
                    kind=spec.noise_kind, composite_ratio=spec.composite_ratio, clip=spec.clip,
                ).numpy()
                reference = clean[None].astype(np.float64)
                in_cols[key].append(float(psnr_per_image(noisy[None].astype(np.float64), reference)[0]))
                restored = np.asarray(denoise(noisy))
                out_cols[key].append(float(psnr_per_image(restored[None].astype(np.float64), reference)[0]))
        inputs[name] = {key: np.asarray(values, dtype=np.float64) for key, values in in_cols.items()}
        outputs[name] = {key: np.asarray(values, dtype=np.float64) for key, values in out_cols.items()}
    return HeldOutScores(inputs, outputs, skipped)


def _mean_or_none(values: np.ndarray) -> Optional[float]:
    return float(np.mean(values)) if len(values) else None


def pair_held_out(best: HeldOutScores, final: HeldOutScores) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Combine two scorings of one held-out set into the summary's cells; see the module contract."""
    if best.skipped != final.skipped or list(best.input_psnr) != list(final.input_psnr):
        raise ValueError("the best and the final model were not scored on the same images")
    cells: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for name, per_sigma in best.input_psnr.items():
        cells[name] = {}
        for key, best_in in per_sigma.items():
            final_in = final.input_psnr[name][key]
            if not np.array_equal(best_in, final_in):
                raise ValueError(
                    f"held-out {name} at sigma_255 {key}: the best and the final model were scored on different "
                    "noisy inputs, so their difference would be partly noise")
            cells[name][key] = {
                "input_psnr": _mean_or_none(best_in),
                "best_psnr": _mean_or_none(best.output_psnr[name][key]),
                "final_psnr": _mean_or_none(final.output_psnr[name][key]),
                "n": int(len(best_in)),
            }
    return cells


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

#: Rows of the per-epoch triptych and patches of the whiteness panel (first patches of the fixed validation set).
TRIPTYCH_PATCHES = 4
WHITENESS_PATCHES = 16

#: Lags of the whiteness panel, in pixels.
WHITENESS_MAX_LAG = 16

PathLike = Union[str, Path]


@dataclasses.dataclass(frozen=True)
class AnalysisSample:
    """The two attributes ``classification_run.report_best_weights`` reads from its ``data`` argument: the noisy patches
    (``x_fig``) and their clean targets (``y_fig``), handed to the analyzer as ``(x, y)``."""

    x_fig: np.ndarray
    y_fig: np.ndarray


def first_pairs(dataset: Any, count: int) -> Tuple[np.ndarray, np.ndarray]:
    """The first ``count`` ``(noisy, clean)`` patches of a batched dataset, as float32 numpy arrays."""
    if count < 1:
        raise ValueError(f"count must be a positive int, got {count!r}")
    noisy_rows: List[np.ndarray] = []
    clean_rows: List[np.ndarray] = []
    seen = 0
    for noisy, clean in dataset:
        noisy_rows.append(np.asarray(noisy, dtype=np.float32))
        clean_rows.append(np.asarray(clean, dtype=np.float32))
        seen += len(noisy_rows[-1])
        if seen >= count:
            break
    if not seen:
        raise ValueError("the dataset yielded no batch")
    return np.concatenate(noisy_rows)[:count], np.concatenate(clean_rows)[:count]


def _shown(image: np.ndarray) -> np.ndarray:
    """An image as ``imshow`` takes it: clipped to [0, 1], a grey one as 2-D."""
    image = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    return image[..., 0] if image.shape[-1] == 1 else image


def plot_triptych(
        noisy: np.ndarray, denoised: np.ndarray, clean: np.ndarray, out_path: PathLike, title: str = "",
) -> List[float]:
    """Save ``noisy | denoised | clean | noisy minus denoised`` per row; return each row's PSNR of the denoised image.

    The residual column is the part the model removed, on a symmetric scale; the other three are clipped to [0, 1]
    for display only (the arrays are not clipped).
    """
    noisy, denoised, clean = (np.asarray(a, dtype=np.float32) for a in (noisy, denoised, clean))
    if not noisy.shape == denoised.shape == clean.shape or noisy.ndim != 4 or len(noisy) < 1:
        raise ValueError(
            f"expected three equal (N, H, W, C) arrays, got {noisy.shape}, {denoised.shape}, {clean.shape}")
    rows = len(noisy)
    residual = noisy - denoised
    limit = float(np.max(np.abs(residual))) or 1.0
    scores = [float(v) for v in psnr_per_image(denoised.astype(np.float64), clean.astype(np.float64))]
    scores_in = [float(v) for v in psnr_per_image(noisy.astype(np.float64), clean.astype(np.float64))]
    fig = Figure(figsize=(10.4, 2.6 * rows + 0.6), layout="constrained")
    axes = fig.subplots(rows, 4, squeeze=False)
    heads = ("noisy", "denoised", "clean", "noisy minus denoised")
    for r in range(rows):
        for c, image in enumerate((noisy[r], denoised[r], clean[r])):
            axes[r][c].imshow(_shown(image), cmap="gray", vmin=0.0, vmax=1.0)
        axes[r][3].imshow(residual[r][..., 0] if residual.shape[-1] == 1 else residual[r].mean(axis=-1),
                          cmap="coolwarm", vmin=-limit, vmax=limit)
        axes[r][0].set_ylabel(f"in {scores_in[r]:.2f} dB\nout {scores[r]:.2f} dB", fontsize=8)
        for c in range(4):
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])
            if r == 0:
                axes[r][c].set_title(heads[c], fontsize=10)
    if title:
        fig.suptitle(title, fontsize=11)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    return scores


class DenoisingTriptychCallback(keras.callbacks.Callback):
    """Writes ``epoch_NNN_triptych.png`` for one fixed batch, on a cadence; see the module contract.

    Files are numbered by completed epochs: ``epoch_001`` after the first, then every ``max(1, epochs // 20)``-th, and
    the last completed epoch is always drawn (``on_train_end``). Place it AFTER the callbacks that edit ``logs`` and
    BEFORE the dashboard. ``written`` lists the files, ``failed`` maps a file to its error.
    """

    def __init__(self, noisy: np.ndarray, clean: np.ndarray, out_dir: PathLike, title: str = "") -> None:
        super().__init__()
        if len(noisy) != len(clean) or len(noisy) < 1:
            raise ValueError(f"noisy has {len(noisy)} rows and clean {len(clean)}: need equal, non-zero lengths")
        self.noisy = np.asarray(noisy, dtype=np.float32)
        self.clean = np.asarray(clean, dtype=np.float32)
        self.out_dir = Path(out_dir)
        self.title = title
        self.written: List[str] = []
        self.failed: Dict[str, str] = {}
        self._completed = 0
        self._drawn_at = -1
        self._last_psnr: Optional[float] = None

    def _interval(self) -> int:
        planned = (self.params or {}).get("epochs")
        try:
            return max(1, int(planned) // 20)
        except (TypeError, ValueError):
            return 1

    def _draw(self, completed: int) -> None:
        name = f"epoch_{completed:03d}_triptych.png"
        suffix = f"after epoch {completed}"
        if self._last_psnr is not None:
            suffix += f", val PSNR {self._last_psnr:.2f} dB"
        try:
            denoised = self.model(self.noisy, training=False)
            if isinstance(denoised, (list, tuple)):
                denoised = denoised[0]
            plot_triptych(
                self.noisy, np.asarray(keras.ops.convert_to_numpy(denoised), dtype=np.float32), self.clean,
                self.out_dir / name, title=f"{self.title} {suffix}".strip())
            self.written.append(name)
            self.failed.pop(name, None)
        except Exception as exc:  # noqa: BLE001 - a figure must never stop training
            logger.warning(f"Triptych {name} failed: {type(exc).__name__}: {exc}")
            self.failed[name] = f"{type(exc).__name__}: {exc}"
        self._drawn_at = completed

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        self._completed = epoch + 1
        value = (logs or {}).get("val_psnr")
        self._last_psnr = None if value is None else float(value)
        if self._completed == 1 or self._completed % self._interval() == 0:
            self._draw(self._completed)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        if self._completed and self._drawn_at != self._completed:
            self._draw(self._completed)


def residual_autocorrelation(residual: np.ndarray, max_lag: int = WHITENESS_MAX_LAG) -> Dict[str, np.ndarray]:
    """The normalised autocorrelation of ``residual`` ``(N, H, W, C)`` at lags ``1..max_lag`` along width and height.

    Each (patch, channel) plane has its mean removed; the lag-``k`` value is the sum of products of pixels ``k`` apart
    over the sum of squares, averaged over planes. White noise reads about 0 at every lag; a residual that carries image
    structure (a model that leaves edges behind) reads well above it at small lags.

    :return: ``{"lags", "horizontal", "vertical", "planes"}``.
    :raises ValueError: for a wrong rank, or a lag that does not fit the patch.
    """
    residual = np.asarray(residual, dtype=np.float64)
    if residual.ndim != 4 or len(residual) < 1:
        raise ValueError(f"expected an (N, H, W, C) array, got shape {residual.shape}")
    if not 1 <= max_lag < min(residual.shape[1], residual.shape[2]):
        raise ValueError(f"max_lag must be in [1, {min(residual.shape[1], residual.shape[2]) - 1}], got {max_lag}")
    planes = np.moveaxis(residual, -1, 1).reshape(-1, residual.shape[1], residual.shape[2])
    planes = planes - planes.mean(axis=(1, 2), keepdims=True)
    energy = np.maximum((planes ** 2).sum(axis=(1, 2)), 1e-30)
    lags = np.arange(1, max_lag + 1)
    horizontal = np.stack([(planes[:, :, :-k] * planes[:, :, k:]).sum(axis=(1, 2)) / energy for k in lags], axis=1)
    vertical = np.stack([(planes[:, :-k, :] * planes[:, k:, :]).sum(axis=(1, 2)) / energy for k in lags], axis=1)
    return {"lags": lags, "horizontal": horizontal.mean(axis=0), "vertical": vertical.mean(axis=0),
            "planes": np.asarray(len(planes))}


def plot_residual_whiteness(
        autocorrelation: Dict[str, np.ndarray], pixels: int, out_path: PathLike, title: str = "") -> None:
    """Save the autocorrelation of the residual over the lag; the band is what white noise stays inside (95 percent)."""
    lags = autocorrelation["lags"]
    band = 1.96 / np.sqrt(float(pixels) * float(autocorrelation["planes"]))
    fig = Figure(figsize=(6.4, 4.0), layout="constrained")
    ax = fig.subplots()
    ax.axhspan(-band, band, color="0.85", label="white noise, 95 percent band")
    ax.plot(lags, autocorrelation["horizontal"], "-o", ms=3, label="along width")
    ax.plot(lags, autocorrelation["vertical"], "-s", ms=3, label="along height")
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xlabel("lag (pixels)")
    ax.set_ylabel("autocorrelation of noisy minus denoised")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)


def plot_psnr_vs_sigma(
        cells: Mapping[str, Mapping[str, Mapping[str, Any]]], out_path: PathLike, title: str = "") -> None:
    """Save PSNR over sigma, ONE PANEL PER DATASET: the noisy input (dashed), the best and the final model."""
    names = list(cells)
    if not names:
        raise ValueError("no dataset to draw")
    columns = min(4, len(names))
    rows = -(-len(names) // columns)
    fig = Figure(figsize=(4.6 * columns, 3.6 * rows), layout="constrained")
    axes = fig.subplots(rows, columns, squeeze=False)
    for index, name in enumerate(names):
        ax = axes[index // columns][index % columns]
        points = sorted((float(key), cell) for key, cell in cells[name].items())
        for field, label, style in (("input_psnr", "noisy input", "--"), ("best_psnr", "best", "-o"),
                                    ("final_psnr", "final", "-s")):
            xs = [x for x, cell in points if cell[field] is not None]
            if xs:
                ax.plot(xs, [cell[field] for x, cell in points if cell[field] is not None], style, ms=4, label=label)
        ax.set_title(f"{name} (n={points[0][1]['n'] if points else 0})", fontsize=10)
        ax.set_xlabel("sigma (0-255 scale)")
        ax.set_ylabel("PSNR (dB)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for index in range(len(names), rows * columns):
        axes[index // columns][index % columns].set_axis_off()
    if title:
        fig.suptitle(title, fontsize=11)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)


def write_end_figures(
        vis_dir: PathLike, *, cells: Optional[Mapping[str, Mapping[str, Mapping[str, Any]]]],
        residual: Optional[np.ndarray], title: str = "",
) -> Dict[str, Any]:
    """Write ``psnr_vs_sigma.png`` and ``residual_whiteness.png``; return ``{files, failed, skipped, not_drawn}``.

    :param cells: :func:`pair_held_out` output, or ``None`` when no held-out set was scored (the PSNR figure is then
        listed in ``not_drawn`` with its reason, not drawn from nothing).
    :param residual: ``(N, H, W, C)`` noisy minus denoised of the BEST weights on fixed validation patches, or ``None``.
    """
    vis_dir = Path(vis_dir)
    record: Dict[str, Any] = {"files": [], "failed": [], "skipped": False, "not_drawn": {}}

    def attempt(name: str, draw: Callable[[], Any]) -> None:
        try:
            draw()
            if (vis_dir / name).is_file():
                record["files"].append(name)
        except Exception as exc:  # noqa: BLE001 - a figure must not fail the run
            logger.warning(f"Visualization {name} failed: {type(exc).__name__}: {exc}")
            record["failed"].append(name)

    if cells:
        attempt("psnr_vs_sigma.png", lambda: plot_psnr_vs_sigma(cells, vis_dir / "psnr_vs_sigma.png", title))
    else:
        record["not_drawn"]["psnr_vs_sigma.png"] = "no held-out set was scored (data.test_dirs is not set)"
    if residual is not None:
        residual = np.asarray(residual)
        lag = min(WHITENESS_MAX_LAG, min(residual.shape[1], residual.shape[2]) - 1)
        attempt("residual_whiteness.png", lambda: plot_residual_whiteness(
            residual_autocorrelation(residual, lag), residual.shape[1] * residual.shape[2],
            vis_dir / "residual_whiteness.png", title))
    else:
        record["not_drawn"]["residual_whiteness.png"] = "no validation patches"
    return record
