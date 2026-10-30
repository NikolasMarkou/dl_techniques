"""``OmniPoint`` holds the two ``StateIndicatorEmbedding`` layers itself.

There is no wrapper layer between the model and ``intrinsics_state`` / ``depth_state``, so
their weights sit directly under the model: ``<model>/intrinsics_state/present_embedding``.
The trainer's gradient probes (``tests/test_train/test_omnipoint``) find them by that
substring. The chain order matters too: the intrinsics flag selects the intrinsics layer's
vector and the sparse-depth flag selects the depth layer's.
"""

import keras
import numpy as np

from dl_techniques.layers.embedding.state_indicator_embedding import StateIndicatorEmbedding
from dl_techniques.models.vision.omnipoint import OmniPoint

_IMAGE_SHAPE = (28, 28, 3)
_STATES = ("intrinsics_state", "depth_state")


def _model(enable_conditioning: bool = True) -> OmniPoint:
    return OmniPoint(
        image_shape=_IMAGE_SHAPE, vit_scale="pico", decoder_dims=[16, 8, 8, 4],
        metric_hidden_dim=8, enable_conditioning=enable_conditioning,
        intrinsics_channels=2, depth_channels=2, conditioning_hidden_channels=4,
    )


def _batch():
    rng = np.random.default_rng(0)
    image = rng.uniform(0.0, 1.0, (2,) + _IMAGE_SHAPE).astype("float32")
    ray_map = rng.normal(size=(2,) + _IMAGE_SHAPE[:2] + (3,)).astype("float32")
    return image, ray_map


def _relative(path: str, model_name: str) -> str:
    parts = path.split("/")
    return "/".join(parts[1:] if parts[0] == model_name else parts)


def test_the_two_embeddings_are_direct_children_of_the_model():
    model = _model()
    model(_batch()[0])
    for name in _STATES:
        layer = getattr(model, name)
        assert isinstance(layer, StateIndicatorEmbedding)
        assert layer.name == name
    paths = {_relative(w.path, model.name) for w in model.weights}
    for name in _STATES:
        for weight in ("present_embedding", "absent_embedding"):
            assert f"{name}/{weight}" in paths


def test_without_conditioning_the_model_holds_no_embedding():
    model = _model(enable_conditioning=False)
    model(_batch()[0])
    assert model.intrinsics_state is None and model.depth_state is None
    assert not any("_state/" in w.path for w in model.weights)


def _set_embedding(layer: StateIndicatorEmbedding, present: float, absent: float) -> None:
    shape = tuple(layer.present_embedding.shape)
    layer.present_embedding.assign(np.full(shape, present, dtype="float32"))
    layer.absent_embedding.assign(np.full(shape, absent, dtype="float32"))


def _mask_logit(model: OmniPoint, image, ray_map) -> np.ndarray:
    # Intrinsics supplied for the whole batch, sparse depth absent: the intrinsics flag is True
    # and the sparse-depth flag is False for every sample.
    return keras.ops.convert_to_numpy(model(image, intrinsics_ray_map=ray_map, training=False)[3])


def test_each_flag_selects_the_vector_of_its_own_layer():
    model = _model()
    image, ray_map = _batch()
    model(image, intrinsics_ray_map=ray_map)

    _set_embedding(model.intrinsics_state, 0.0, 0.0)
    _set_embedding(model.depth_state, 0.0, 0.0)
    baseline = _mask_logit(model, image, ray_map)

    # Only the intrinsics layer's PRESENT vector is nonzero: reached iff the intrinsics flag
    # (True) drives intrinsics_state.
    _set_embedding(model.intrinsics_state, 5.0, 0.0)
    assert np.max(np.abs(_mask_logit(model, image, ray_map) - baseline)) > 0.0

    # Only the depth layer's PRESENT vector is nonzero: the sparse-depth flag is False, so it
    # must select the (zero) absent vector and leave the output unchanged.
    _set_embedding(model.intrinsics_state, 0.0, 0.0)
    _set_embedding(model.depth_state, 5.0, 0.0)
    np.testing.assert_array_equal(_mask_logit(model, image, ray_map), baseline)

    # And the depth layer's ABSENT vector is what the False flag does select.
    _set_embedding(model.depth_state, 0.0, 5.0)
    assert np.max(np.abs(_mask_logit(model, image, ray_map) - baseline)) > 0.0
