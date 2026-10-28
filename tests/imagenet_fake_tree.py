"""A tiny fake ILSVRC2012 tree (train, flat val, devkit) for tests.

Shared by ``tests/test_datasets/test_imagenet2012.py`` and the trainer tests so
none of them copies the builder. Sorted wnid order is a < b < c; the ILSVRC ids
are deliberately NOT in that order, so a loader that used the id as the class
index would be wrong.
"""

import os

import numpy as np
import tensorflow as tf
from scipy.io import savemat

WNIDS = ["n0000000a", "n0000000b", "n0000000c"]
ILSVRC_ID = {"n0000000a": 3, "n0000000b": 1, "n0000000c": 2}
PER_CLASS = 4
VAL_CLASSES = [2, 0, 1, 1, 0, 2]  # sorted-wnid index of val image #1..#6


def jpeg(path, h, w, value):
    # Seeded noise, so a crop of a different region gives different pixels.
    rng = np.random.default_rng(int(value))
    img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    tf.io.write_file(str(path), tf.io.encode_jpeg(tf.constant(img)))


def make_devkit(devkit_dir):
    os.makedirs(devkit_dir, exist_ok=True)
    dtype = [("ILSVRC2012_ID", "O"), ("WNID", "O"), ("num_children", "O")]
    rows = [(ILSVRC_ID[w], w, 0) for w in WNIDS]
    rows += [(1001, "n9999999x", 2), (1002, "n9999999y", 3)]  # non-leaf synsets
    synsets = np.empty((len(rows), 1), dtype=dtype)
    for i, r in enumerate(rows):
        synsets[i, 0] = r
    savemat(os.path.join(devkit_dir, "meta.mat"), {"synsets": synsets})
    # ground truth holds ILSVRC ids, one per val image in file-name order
    id_of_class = {i: ILSVRC_ID[w] for i, w in enumerate(WNIDS)}
    with open(os.path.join(devkit_dir, "ILSVRC2012_validation_ground_truth.txt"), "w") as f:
        f.write("\n".join(str(id_of_class[c]) for c in VAL_CLASSES) + "\n")


def build_fake_imagenet(root):
    """Write the tree under ``root`` (a ``pathlib.Path``) and return its path as ``str``."""
    for ci, w in enumerate(WNIDS):
        (root / "train" / w).mkdir(parents=True)
        for j in range(PER_CLASS):
            jpeg(root / "train" / w / f"{w}_{j}.JPEG", 40 + 5 * j, 60, 50 * ci + 10 * j)
    (root / "val").mkdir()
    for k in range(len(VAL_CLASSES)):
        jpeg(root / "val" / f"ILSVRC2012_val_{k + 1:08d}.JPEG", 48, 64, 20 * k)
    make_devkit(str(root / "devkit"))
    return str(root)
