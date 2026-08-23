"""``ConvDecoder`` on its own, including the flat-list contract at LAYER scope.

Why this file exists
--------------------
``test_the_decoder_weights_reach_the_archive.py`` pins the same weight-survival
property, but through a whole ``MaskedAutoencoder`` -- so it can only observe the
decoder as a sublayer of a ``keras.Model``. The defect it guards against
(D-026: Keras 3.8 does not write a layer container nested >=2 deep whose OWNER is
a ``keras.layers.Layer``) is a property of ``ConvDecoder`` itself, and
``ConvDecoder`` is reusable outside the MAE. This file therefore takes the same
measurement with the decoder as the ONLY custom layer in the archive, where a
loss cannot be attributed to the encoder or the masking path.

MEASURED on this class before D-026: the list-of-dicts form put **11 of 51
tensors and 98,403 of 329,827 parameters** into ``model.weights.h5``. So the
arms here are not shape checks.

Two round trips, not one
------------------------
Every weight is perturbed off its initializer BEFORE the first save (a
zero-initialized BatchNorm beta matches by coincidence otherwise, which is how
43 of 51 tensors hid the original defect), and the reloaded model is perturbed
AGAIN before the second save. A save path that is correct only for a
freshly-constructed layer -- e.g. one that re-derives sublayers from
``get_config`` rather than restoring them -- survives one trip and fails the
second. Both trips assert ``max|delta| == 0.0`` on the forward output, at
``atol`` exactly zero.

The upsampling arithmetic is asserted as a RELATIONSHIP
(``H_out == H_in * 2 ** len(decoder_dims)``) over several ``decoder_dims``
lengths, not as a pasted literal, and is cross-checked against the layer's own
``compute_output_shape`` so a drift between the two is visible.
"""

import io
import zipfile

import h5py
import keras
import numpy as np
import pytest

from dl_techniques.models.masked_autoencoder import ConvDecoder

LATENT = (4, 4, 32)
SEED = 20260823


def _wrap(decoder: ConvDecoder, latent=LATENT) -> keras.Model:
    """The decoder as the single custom layer of a functional model."""
    inp = keras.Input(shape=latent)
    return keras.Model(inp, decoder(inp), name="decoder_only")


def _perturb_every_weight(model: keras.Model, seed: int) -> None:
    """Move every non-moving-statistic weight off its current value."""
    rng = np.random.RandomState(seed)
    for weight in model.weights:
        if "moving_" in weight.path:
            continue
        value = np.array(weight)
        sigma = max(0.25 * float(np.std(value)), 1e-3)
        weight.assign(value + rng.randn(*value.shape).astype(value.dtype) * sigma)


def _archive_content(path) -> tuple:
    """(dataset count, element count) of the archive's ``model.weights.h5``."""
    with zipfile.ZipFile(path) as archive:
        payload = archive.read("model.weights.h5")

    datasets = 0
    elements = 0

    def _visit(_name, obj):
        nonlocal datasets, elements
        if isinstance(obj, h5py.Dataset):
            datasets += 1
            elements += int(np.prod(obj.shape)) if obj.shape else 1

    with h5py.File(io.BytesIO(payload), "r") as handle:
        handle.visititems(_visit)
    return datasets, elements


class TestTheConstructorRefusesAnImpossibleDecoder:
    def test_an_empty_decoder_dims_is_refused(self):
        with pytest.raises(ValueError, match="decoder_dims cannot be empty"):
            ConvDecoder(decoder_dims=[])

    @pytest.mark.parametrize("dims", [[16, 0], [-8], [32, -1, 8]])
    def test_a_non_positive_width_is_refused(self, dims):
        with pytest.raises(ValueError, match="must be positive"):
            ConvDecoder(decoder_dims=dims)

    @pytest.mark.parametrize("channels", [0, -3])
    def test_a_non_positive_output_channel_count_is_refused(self, channels):
        with pytest.raises(ValueError, match="output_channels must be positive"):
            ConvDecoder(decoder_dims=[8], output_channels=channels)

    def test_a_legal_decoder_constructs(self):
        """The control: the checks above must not be rejecting everything."""
        assert ConvDecoder(decoder_dims=[8, 4], output_channels=1).num_blocks == 2


class TestTheUpsamplingArithmetic:
    """Each entry in ``decoder_dims`` doubles both spatial axes. Exactly."""

    @pytest.mark.parametrize("dims", [(8,), (8, 4), (16, 8, 4), (16, 8, 8, 4)])
    def test_the_output_is_the_input_scaled_by_two_to_the_number_of_blocks(self, dims):
        decoder = ConvDecoder(decoder_dims=dims, output_channels=3, use_batch_norm=False)
        model = _wrap(decoder)
        out = model(np.zeros((2,) + LATENT, dtype="float32"), training=False)

        factor = 2 ** len(dims)
        expected = (2, LATENT[0] * factor, LATENT[1] * factor, 3)
        assert tuple(out.shape) == expected, (
            f"{len(dims)} decoder blocks over a {LATENT[0]}x{LATENT[1]} latent "
            f"must give {expected[1]}x{expected[2]}, got {tuple(out.shape)[1:3]}"
        )

    @pytest.mark.parametrize("dims", [(8,), (8, 4), (16, 8, 4)])
    def test_compute_output_shape_agrees_with_the_forward_pass(self, dims):
        """A drift between the declared and the actual shape is the bug class."""
        decoder = ConvDecoder(decoder_dims=dims, output_channels=5, use_batch_norm=False)
        declared = decoder.compute_output_shape((None,) + LATENT)
        actual = _wrap(decoder).output_shape
        assert tuple(declared) == tuple(actual), (declared, actual)

    def test_output_channels_reaches_the_final_projection(self):
        for channels in (1, 3, 7):
            decoder = ConvDecoder(
                decoder_dims=(8,), output_channels=channels, use_batch_norm=False
            )
            out = _wrap(decoder)(np.zeros((1,) + LATENT, dtype="float32"))
            assert tuple(out.shape)[-1] == channels


class TestTheFlatSublayerLists:
    """D-026's storage shape, asserted directly -- six flat lists, never nested."""

    def test_the_norm_lists_stay_empty_without_batch_norm(self):
        decoder = ConvDecoder(decoder_dims=(8, 4), use_batch_norm=False)
        assert decoder.norm_upsamples == []
        assert decoder.norm_refines == []
        assert decoder.num_blocks == 2
        block = decoder.decoder_block(0)
        assert block["norm_upsample"] is None and block["norm_refine"] is None

    def test_batch_norm_populates_both_norm_lists(self):
        """The control for the arm above."""
        decoder = ConvDecoder(decoder_dims=(8, 4), use_batch_norm=True)
        assert len(decoder.norm_upsamples) == 2
        assert len(decoder.norm_refines) == 2
        assert decoder.decoder_block(1)["norm_refine"] is decoder.norm_refines[1]

    def test_no_tracked_attribute_holds_a_container_of_containers(self):
        """The exact shape Keras 3.8 fails to write, under a Layer owner."""
        decoder = ConvDecoder(decoder_dims=(8, 4))
        for name in (
            "upsample_convs", "norm_upsamples", "act_upsamples",
            "refine_convs", "norm_refines", "act_refines",
        ):
            for item in getattr(decoder, name):
                assert isinstance(item, keras.layers.Layer), (name, type(item))
        assert not hasattr(decoder, "decoder_blocks"), (
            "`decoder_blocks` (a list of dicts of Layers) is the container D-026 "
            "removed; its return would silently drop decoder weights on save."
        )


class TestAPerturbationIsVisibleInTheOutput:
    """Anti-vacuity for everything below: the weights must actually matter."""

    def test_moving_one_kernel_moves_the_output(self):
        keras.utils.set_random_seed(SEED)
        decoder = ConvDecoder(
            decoder_dims=(8, 4), output_channels=3, use_batch_norm=False
        )
        model = _wrap(decoder)
        x = np.random.RandomState(SEED).randn(2, *LATENT).astype("float32")
        before = np.array(model(x, training=False))

        kernel = decoder.refine_convs[0].kernel
        kernel.assign(np.array(kernel) + 0.5)
        after = np.array(model(x, training=False))

        assert float(np.max(np.abs(after - before))) > 0.0, (
            "perturbing `refine_convs[0].kernel` did not move the output; the "
            "sublayer is not on the forward path and every weight-survival arm "
            "in this file would be vacuous."
        )


@pytest.fixture(scope="module")
def two_round_trips(tmp_path_factory):
    """Perturb / save / load, twice, on the decoder alone."""
    keras.utils.set_random_seed(SEED)
    decoder = ConvDecoder(
        decoder_dims=(16, 8), output_channels=3, use_batch_norm=True
    )
    model = _wrap(decoder)
    model(np.zeros((1,) + LATENT, dtype="float32"), training=False)

    x = np.random.RandomState(SEED).randn(2, *LATENT).astype("float32")
    directory = tmp_path_factory.mktemp("conv_decoder_archive")
    trips = []
    current = model

    for index, seed in enumerate((SEED, SEED + 1)):
        _perturb_every_weight(current, seed)
        before = np.array(current(x, training=False))
        weights_before = [np.array(w) for w in current.weights]

        path = directory / f"decoder_trip_{index}.keras"
        current.save(path)
        datasets, elements = _archive_content(path)

        restored = keras.models.load_model(path)
        after = np.array(restored(x, training=False))

        trips.append({
            "expected_tensors": len(current.weights),
            "expected_params": current.count_params(),
            "archive_datasets": datasets,
            "archive_elements": elements,
            "weights_before": weights_before,
            "weights_after": [np.array(w) for w in restored.weights],
            "delta": float(np.max(np.abs(after - before))),
            "magnitude": float(np.max(np.abs(before))),
        })
        current = restored

    return trips


@pytest.mark.parametrize("trip", [0, 1])
def test_the_archive_holds_every_decoder_tensor(two_round_trips, trip):
    data = two_round_trips[trip]
    assert data["archive_datasets"] == data["expected_tensors"], (
        f"trip {trip}: `model.weights.h5` holds {data['archive_datasets']} "
        f"datasets against {data['expected_tensors']} weights. A layer container "
        f"nested >=2 deep owned by a `keras.layers.Layer` is not written -- keep "
        f"`ConvDecoder`'s sublayers in FLAT per-role lists (D-026)."
    )


@pytest.mark.parametrize("trip", [0, 1])
def test_the_archive_holds_every_decoder_parameter(two_round_trips, trip):
    data = two_round_trips[trip]
    assert data["archive_elements"] == data["expected_params"], (
        f"trip {trip}: archive holds {data['archive_elements']} elements against "
        f"count_params() = {data['expected_params']}"
    )


@pytest.mark.parametrize("trip", [0, 1])
def test_every_decoder_weight_survives_the_round_trip(two_round_trips, trip):
    data = two_round_trips[trip]
    before, after = data["weights_before"], data["weights_after"]
    assert len(before) == len(after)
    mismatched = [
        i for i, (a, b) in enumerate(zip(before, after))
        if a.shape != b.shape or not np.array_equal(a, b)
    ]
    assert not mismatched, (
        f"trip {trip}: {len(mismatched)} of {len(before)} tensors changed across "
        f"the round trip (ordinal indices {mismatched[:8]}). Compared by ORDINAL "
        f"-- a reloaded model reports different weight paths."
    )


@pytest.mark.parametrize("trip", [0, 1])
def test_the_decoder_forward_is_bit_identical_after_reload(two_round_trips, trip):
    data = two_round_trips[trip]
    assert data["delta"] == 0.0, (
        f"trip {trip}: decoder output moved by {data['delta']:.6e} across the "
        f"round trip, against its own dynamic range {data['magnitude']:.6e}. "
        f"Trip 1 failing while trip 0 passes means the save path is correct only "
        f"for a freshly-constructed layer."
    )


class TestSerialization:
    def test_the_config_round_trips(self):
        decoder = ConvDecoder(
            decoder_dims=(16, 8),
            output_channels=2,
            kernel_size=5,
            use_batch_norm=False,
            final_activation="sigmoid",
        )
        config = decoder.get_config()
        assert config["decoder_dims"] == [16, 8], "the stored form is a list (D-085)"
        clone = ConvDecoder.from_config(config)
        assert clone.decoder_dims == [16, 8]
        assert clone.output_channels == 2
        assert clone.kernel_size == 5
        assert clone.use_batch_norm is False
        assert clone.num_blocks == decoder.num_blocks

    def test_a_final_activation_is_applied(self):
        """`final_activation='sigmoid'` must bound the output, not merely be stored."""
        keras.utils.set_random_seed(SEED)
        decoder = ConvDecoder(
            decoder_dims=(8,), output_channels=3,
            use_batch_norm=False, final_activation="sigmoid",
        )
        model = _wrap(decoder)
        _perturb_every_weight(model, SEED)
        out = np.array(model(
            np.random.RandomState(SEED).randn(2, *LATENT).astype("float32") * 10.0,
            training=False,
        ))
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0

    def test_no_final_activation_leaves_the_output_unbounded(self):
        """The control: without it the arm above passes on a small-magnitude output."""
        keras.utils.set_random_seed(SEED)
        decoder = ConvDecoder(
            decoder_dims=(8,), output_channels=3, use_batch_norm=False
        )
        model = _wrap(decoder)
        _perturb_every_weight(model, SEED)
        out = np.array(model(
            np.random.RandomState(SEED).randn(2, *LATENT).astype("float32") * 10.0,
            training=False,
        ))
        assert float(out.min()) < 0.0, out.min()
