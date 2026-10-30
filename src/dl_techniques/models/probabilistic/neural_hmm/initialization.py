"""Start a Gaussian :class:`~.model.NeuralHMM`'s emissions from k-means on the data.

The default start puts every emission mean near 0 with a scale near 1, so data far from
that has to be reached by Adam, one small step at a time, before any state can separate.
:func:`initialize_emissions_from_kmeans` fits the emission head's last layer in closed form,
so that each state starts at one k-means cluster, with that cluster's standard deviation.
The head and the state embedding stay ordinary weights and train from there.
"""

import warnings
from typing import Any, Mapping

import keras
import numpy as np
from scipy.cluster.vq import kmeans2

from .diagnostics import validate_inputs
from .model import NeuralHMM


def initialize_emissions_from_kmeans(model: NeuralHMM, inputs: Mapping[str, Any], seed: int = 0) -> None:
    """Set the emission head's last layer so that each state starts at a k-means cluster.

    With ``hidden_dim >= n_states - 1`` the head reproduces the clusters, as far as the
    state features are linearly independent; below that it is their least-squares fit.
    Builds the model if needed. Call it before ``fit``, on the training split.

    :param model: A generative Gaussian model whose emission reads no context.
    :param inputs: ``{"obs", "mask"}``, plus ``"inputs"`` with ``input_dim`` set.
    :param seed: Seed of k-means.
    :raises ValueError: If the model is not of that kind, or the inputs fail
        :func:`~.diagnostics.validate_inputs`.
    """
    if model.mode != "generative" or model.obs_type != "gaussian" or model.emission_context != "none":
        raise ValueError(
            f"initialize_emissions_from_kmeans needs mode='generative', obs_type='gaussian' and "
            f"emission_context='none', got {model.mode!r}, {model.obs_type!r} and {model.emission_context!r}."
        )
    validate_inputs(model, inputs)
    obs = np.asarray(inputs["obs"])
    valid = np.ones(obs.shape[:2], dtype=bool) if inputs.get("mask") is None else np.asarray(inputs["mask"]) != 0
    x = obs[valid].astype(np.float64)
    if len(x) < model.n_states:
        raise ValueError(f"k-means over {model.n_states} states needs as many valid steps, got {len(x)}.")
    if not model.built:
        model({key: value[:1] for key, value in inputs.items()}, training=False)
    with warnings.catch_warnings():
        # An emptied cluster keeps a random point, which is what a start needs.
        warnings.simplefilter("ignore", UserWarning)
        centers, labels = kmeans2(x, model.n_states, iter=20, minit="++", missing="warn", seed=seed)

    emission = model.emission
    floor = 2.0 * emission.min_scale
    scales = np.stack([x[labels == k].std(axis=0) if (labels == k).sum() > 1 else x.std(axis=0)
                       for k in range(model.n_states)])
    raw_scales = np.log(np.expm1(np.maximum(scales, floor) - emission.min_scale))
    targets = np.concatenate([centers, raw_scales], axis=-1)

    head = emission.head
    features = keras.ops.convert_to_numpy(
        head.activation_fn(head.fc1(keras.ops.convert_to_tensor(emission.state_embedding)))
    ).astype(np.float64)
    feature_mean, target_mean = features.mean(axis=0), targets.mean(axis=0)
    kernel = np.linalg.lstsq(features - feature_mean, targets - target_mean, rcond=None)[0]
    head.fc2.kernel.assign(kernel)
    head.fc2.bias.assign(target_mean - feature_mean @ kernel)
