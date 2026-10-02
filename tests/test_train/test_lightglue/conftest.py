"""Shared tiny fixtures for the LightGlue trainer tests (random tiny SuperPoint, jpg folder)."""

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.vision.keypoints.lightglue.model import LightGlue
from dl_techniques.models.vision.keypoints.superpoint.model import SuperPoint

SIZE = 64
DESCRIPTOR_DIM = 32


def make_superpoint() -> SuperPoint:
    """A randomly initialised float32 SuperPoint small enough for a CPU-speed test."""
    model = SuperPoint(depths=(1, 1, 1), dims=(8, 16, 32), input_shape=(SIZE, SIZE, 1),
                       descriptor_dim=DESCRIPTOR_DIM, dtype="float32")
    model.build((None, SIZE, SIZE, 1))
    return model


def make_lightglue() -> LightGlue:
    return LightGlue(input_dim=DESCRIPTOR_DIM, descriptor_dim=32, num_layers=2, num_heads=2)


def write_images(folder: Path, count: int) -> Path:
    """Write ``count`` random-texture jpgs (90x120) into ``folder``."""
    rng = np.random.RandomState(0)
    for i in range(count):
        arr = rng.randint(30, 200, size=(90, 120, 3)).astype(np.uint8)
        (folder / f"img{i:03d}.jpg").write_bytes(tf.io.encode_jpeg(tf.constant(arr)).numpy())
    return folder


@pytest.fixture(scope="module")
def superpoint_path(tmp_path_factory) -> str:
    """A saved tiny SuperPoint checkpoint (what ``--superpoint-checkpoint`` points at)."""
    path = tmp_path_factory.mktemp("superpoint") / "final_model.keras"
    make_superpoint().save(str(path))
    return str(path)
