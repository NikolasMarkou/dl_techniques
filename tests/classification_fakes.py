"""A tiny stand-in for the raw MNIST/CIFAR arrays, for trainer tests.

``train.common.classification_data._load_raw`` is the only place that reads
``keras.datasets``; a trainer test that must not touch the Keras cache replaces it with
:func:`tiny_raw` (through :func:`use_tiny_raw`). The shapes and class counts follow the
dataset spec, the pixels are noise, and the default sizes leave a train split of at least
one batch of 4 after the 10% selection cut.
"""

import numpy as np
import tensorflow as tf

import dl_techniques.datasets.vision.classification as classification
import dl_techniques.datasets.vision.imagenette as imagenette
import train.common.classification_data as classification_data


def tiny_raw(n_train: int = 40, n_test: int = 8, seed: int = 0):
    """A ``_load_raw`` replacement: ``name -> ((x_train, y_train), (x_test, y_test))``."""

    def load(name):
        spec = classification.get_spec(name)
        shape = spec.input_shape[:2] if name == "mnist" else spec.input_shape
        rng = np.random.default_rng(seed)

        def split(n):
            x = rng.integers(0, 256, size=(n, *shape), dtype=np.uint8)
            return x, rng.integers(0, spec.num_classes, size=n).astype(np.int64)

        return split(n_train), split(n_test)

    return load


def use_tiny_raw(monkeypatch, **kwargs) -> None:
    """Replace the raw-array seam with :func:`tiny_raw` for the test."""
    monkeypatch.setattr(classification_data, "_load_raw", tiny_raw(**kwargs))


def raw_from_unit(load, min_train: int = 40):
    """Adapt a fake that returns ``((x, y), (x, y), shape, classes)`` with float images in
    [0, 1] (the retired ``load_dataset`` form, MNIST repeated to three channels) into a
    ``_load_raw`` replacement: the pixels are rounded to uint8 and MNIST goes back to one
    channel. A train split smaller than ``min_train`` rows is repeated cyclically up to it,
    so the 10% selection cut leaves at least one row out and a batch of a few rows in."""

    def load_raw(name):
        (x_train, y_train), (x_test, y_test), _, _ = load(name)
        if len(x_train) < min_train:
            reps = -(-min_train // len(x_train))
            x_train = np.concatenate([x_train] * reps)[:min_train]
            y_train = np.concatenate([np.asarray(y_train).reshape(-1)] * reps)[:min_train]

        def raw(x):
            x = np.rint(np.asarray(x) * 255.0).astype(np.uint8)
            return x[..., 0] if name == "mnist" else x

        return ((raw(x_train), np.asarray(y_train).reshape(-1)),
                (raw(x_test), np.asarray(y_test).reshape(-1)))

    return load_raw


def use_raw_from_unit(monkeypatch, load, **kwargs) -> None:
    """Install :func:`raw_from_unit` of ``load`` on the raw-array seam (``min_train=0`` keeps
    the fake's rows exactly, for a test that audits every pixel)."""
    monkeypatch.setattr(classification_data, "_load_raw", raw_from_unit(load, **kwargs))


# ---------------------------------------------------------------------
# Imagenette records
# ---------------------------------------------------------------------

IMAGENETTE_FAKE_TRAIN = 60
IMAGENETTE_FAKE_TEST = 20
_IMAGE_SIZES = [(40, 50), (48, 36), (44, 44)]


def _jpeg(value: int, size) -> bytes:
    """Mean colour ``value`` with a horizontal ramp, so a crop or a flip changes the pixels."""
    ramp = (np.arange(size[1]) - size[1] / 2.0) * 0.1
    image = np.clip(value + ramp[None, :, None] + np.zeros((*size, 3)), 0, 255).astype(np.uint8)
    return tf.io.encode_jpeg(tf.constant(image), quality=100).numpy()


def imagenette_fake_records(n: int, offset: int):
    """``n`` encoded jpegs of mean colour ``4 * i + offset`` and labels cycling the 10 classes."""
    images = np.empty(n, dtype=object)
    for i in range(n):
        images[i] = _jpeg(4 * i + offset, _IMAGE_SIZES[i % len(_IMAGE_SIZES)])
    return images, (np.arange(n) % 10).astype(np.int32)


def use_fake_imagenette(monkeypatch, n_train: int = IMAGENETTE_FAKE_TRAIN,
                        n_test: int = IMAGENETTE_FAKE_TEST):
    """Replace the TFDS record reader with in-memory jpegs; returns the list of splits read."""
    calls = []

    def read(split, data_dir=None):
        calls.append(split)
        return (imagenette_fake_records(n_train, 0) if split == "train"
                else imagenette_fake_records(n_test, 2))

    monkeypatch.setattr(imagenette, "read_imagenette_records", read)
    return calls
