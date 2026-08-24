"""F-61: `DPTDecoder.build()` built none of its sublayers.

Three instruments were run against HEAD-before-fix, standalone, on CPU, with
every weight perturbed by +0.137 before the save. They do NOT agree, and that
disagreement is the finding:

===================================  ==========  =========
instrument                           before      after
===================================  ==========  =========
weight count, sampled BEFORE a        **0**       12
forward pass
weight count, sampled AFTER a         12          12   <- BLIND
forward pass
archive HDF5 dataset count            12          12   <- BLIND
perturb-save-reload ``max|dOut|``     **70.98**   0.0
weights back at class defaults        **12/12**   0/12
===================================  ==========  =========

A save-side instrument cannot see a load-side loss: the archive was complete
both times. A post-forward count cannot see it either, because ``__call__``
materialises the sublayers. Only the pre-forward count and the perturb arm
discriminate, and only the perturb arm proves the values were lost rather than
merely late. ``keras.models.load_model`` builds from the SAVED ``input_shape``
and restores IMMEDIATELY, so "sublayers are built lazily on the first call" is
true of ``__call__`` and false of the load path.

Inside ``DepthAnything`` this was masked by that model's ``load_own_variables``
force-build. ``DPTDecoder`` is public API and has no such parent here.

CPU only.
"""

import os
import zipfile

import h5py
import numpy as np
import pytest
import keras

from dl_techniques.models.depth_anything.components import DPTDecoder

SHAPE = (None, 8, 8, 16)
X = np.random.RandomState(0).randn(2, 8, 8, 16).astype("float32")


@keras.saving.register_keras_serializable(package="test_dpt_decoder_standalone")
class _ExplicitBuildWrapper(keras.Model):
    """A parent that builds its child EXPLICITLY -- the `load_model` shape."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dec = DPTDecoder(dims=[8, 4], output_channels=1)

    def build(self, input_shape):
        self.dec.build(input_shape)
        super().build(input_shape)

    def call(self, x, training=None):
        return self.dec(x, training=training)


def _archive_dataset_count(path, workdir):
    with zipfile.ZipFile(path) as z:
        z.extract("model.weights.h5", workdir)
    n = [0]
    with h5py.File(os.path.join(workdir, "model.weights.h5")) as f:
        f.visititems(lambda _n, o: n.__setitem__(0, n[0] + 1) if isinstance(o, h5py.Dataset) else None)
    return n[0]


class TestDPTDecoderStandaloneRoundTrip:
    def test_build_alone_creates_every_weight(self):
        """INSTRUMENT A. Sampled BEFORE any forward pass; 0 -> 12."""
        dec = DPTDecoder(dims=[8, 4], output_channels=1)
        dec.build(SHAPE)
        assert len(dec.weights) > 0, "build() created no weights at all"
        lazy = DPTDecoder(dims=[8, 4], output_channels=1)
        lazy(X)
        assert len(dec.weights) == len(lazy.weights)

    def test_perturbed_values_survive_a_reload(self, tmp_path):
        """INSTRUMENT C, which subsumes A and B. `max|dOut|` 70.98 -> 0.0."""
        keras.utils.set_random_seed(0)
        model = _ExplicitBuildWrapper()
        model.build(SHAPE)
        _ = model(X)
        for w in model.weights:
            w.assign(np.asarray(w) + 0.137)
        before = np.asarray(model(X))

        path = str(tmp_path / "m.keras")
        model.save(path)
        # INSTRUMENT B, recorded to pin that it is BLIND: complete both ways.
        assert _archive_dataset_count(path, str(tmp_path)) == len(model.weights)

        reloaded = keras.models.load_model(path)
        after = np.asarray(reloaded(X))
        assert float(np.max(np.abs(before - after))) < 1e-6
        mismatched = sum(
            1 for a, b in zip(model.weights, reloaded.weights)
            if float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) > 1e-6
        )
        assert mismatched == 0, f"{mismatched} of {len(model.weights)} weights lost"
