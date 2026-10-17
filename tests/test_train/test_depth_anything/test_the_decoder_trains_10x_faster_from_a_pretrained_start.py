"""From a pretrained start the decoder trains at 10x the encoder's rate; from random weights, 1x.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 1.5 (D10, D-127,
D-133). The paper (arXiv 2401.10891, section 4.1): "the base learning rate of
the pre-trained encoder is set as 5e-6, while the randomly initialized decoder
uses a 10x larger learning rate". Until this step the trainer had one rate for
every variable. ``train_depth_anything`` now compiles an ``LRMultiplierAdamW``
whose table gives every trainable decoder variable ``DECODER_LR_MULTIPLIER``
when ``--init-from``, ``--pretrained-encoder-weights`` or
``--pretrained-encoder`` (the official DINOv2 weights) is given; a random
start gets a stock ``AdamW`` (every variable at the base rate, as the Step 1
measurements were made). The data the trainer feeds is ImageNet-standardized,
the input the official encoder was trained on.

The optimizer and the datasets are read off the real ``train_depth_anything``
(data and weight loading stubbed), so the test sees what a run trains with,
not a helper the trainer might not call.
"""

import keras
import numpy as np
import pytest

from dl_techniques.optimization import LRMultiplierAdamW
from train.depth_anything import train_depth_anything as module


class _Stop(Exception):
    pass


class _OneBatch:
    def __len__(self):
        return 1


STARTS = {
    "random": {},
    "init_from": {"init_from": "previous_run/model_inference.keras"},
    "pretrained_encoder_weights": {"pretrained_encoder_weights": "vit_s_encoder.keras"},
    "pretrained_encoder": {"pretrained_encoder": True},
}


def _run_until_compile(tmp_path, start, encoder_kind="real"):
    config = module.DepthAnythingTrainingConfig(
        output_dir=str(tmp_path), experiment_name="run", epochs=1, steps_per_epoch=1,
        encoder_type="vit_s", encoder_kind=encoder_kind, patch_size=28,
        dpt_features=8, dpt_out_channels=(8, 8, 8, 8), **STARTS[start])
    model = module.create_model(config)
    seen = {"datasets": [], "loaded": []}

    def compile_(**kwargs):
        seen.update(kwargs)
        raise _Stop()

    def dataset(*args, **kwargs):
        seen["datasets"].append(kwargs)
        return _OneBatch()

    paths = [str(tmp_path / f"p{i}") for i in range(20)]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(module, "split_megadepth_pairs",
                   lambda *a, **k: (paths[:18], paths[:18], paths[18:], paths[18:]))
        mp.setattr(module, "MegaDepthDataset", dataset)
        mp.setattr(module, "create_model", lambda c: model)
        mp.setattr(module, "init_from_checkpoint", lambda m, p: seen["loaded"].append("init_from"))
        mp.setattr(model, "from_pretrained_encoder",
                   lambda p: seen["loaded"].append("encoder_keras") or model, raising=False)
        mp.setattr(model, "load_official_encoder_weights",
                   lambda: seen["loaded"].append("official") or model, raising=False)
        mp.setattr(model, "compile", compile_, raising=False)
        with pytest.raises(_Stop):
            module.train_depth_anything(config)
    return model, seen


def test_the_multiplier_is_the_papers():
    assert module.DECODER_LR_MULTIPLIER == 10.0


@pytest.mark.parametrize("start", ["init_from", "pretrained_encoder_weights", "pretrained_encoder"])
def test_a_pretrained_start_trains_only_the_decoder_10x_faster(tmp_path, start):
    model, seen = _run_until_compile(tmp_path, start)
    optimizer = seen["optimizer"]
    assert type(optimizer) is LRMultiplierAdamW
    decoder = {id(v) for v in model.decoder.trainable_variables}
    assert decoder
    variables = model.trainable_variables
    optimizer.build(variables)
    for variable in variables:
        expected = 10.0 if id(variable) in decoder else 1.0
        assert optimizer.multiplier_for(variable.path) == expected, variable.path
    assert len(variables) > len(decoder)  # the encoder is in the run, at 1x


def test_a_random_start_trains_every_variable_at_the_base_rate(tmp_path):
    _model, seen = _run_until_compile(tmp_path, "random")
    assert type(seen["optimizer"]) is keras.optimizers.AdamW
    assert seen["loaded"] == []


@pytest.mark.parametrize("start", list(STARTS))
def test_the_rest_of_the_optimizer_is_unchanged(tmp_path, start):
    _model, seen = _run_until_compile(tmp_path, start)
    config = module.DepthAnythingTrainingConfig(experiment_name="run", **STARTS[start])
    optimizer = seen["optimizer"]
    assert optimizer.global_clipnorm == config.gradient_clipping
    assert optimizer.weight_decay == config.weight_decay


@pytest.mark.parametrize("start,loads", [
    ("init_from", ["init_from"]), ("pretrained_encoder_weights", ["encoder_keras"]),
    ("pretrained_encoder", ["official"]), ("random", [])])
def test_each_start_loads_what_it_names(tmp_path, start, loads):
    _model, seen = _run_until_compile(tmp_path, start)
    assert seen["loaded"] == loads


def test_a_pretrained_start_defaults_to_the_papers_encoder_rate():
    for start in ("init_from", "pretrained_encoder_weights", "pretrained_encoder"):
        assert module.DepthAnythingTrainingConfig(
            experiment_name="run", **STARTS[start]).learning_rate == 5e-6, start
    assert module.DepthAnythingTrainingConfig(experiment_name="run").learning_rate == \
        module.FROM_SCRATCH_LEARNING_RATE


def test_two_encoder_starts_or_a_placeholder_are_refused():
    with pytest.raises(ValueError, match="pretrained_encoder"):
        module.DepthAnythingTrainingConfig(experiment_name="run", pretrained_encoder=True,
                                           init_from="x.keras")
    with pytest.raises(ValueError, match="pretrained_encoder"):
        module.DepthAnythingTrainingConfig(experiment_name="run", pretrained_encoder=True,
                                           encoder_kind="placeholder")


def test_the_pretrained_encoder_flag_reaches_the_config(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train", "--pretrained-encoder", "--encoder-type", "vit_s"])
    assert module.parse_arguments().pretrained_encoder is True
    monkeypatch.setattr("sys.argv", ["train"])
    assert module.parse_arguments().pretrained_encoder is False


@pytest.mark.parametrize("start", ["random", "pretrained_encoder"])
def test_the_datasets_feed_imagenet_standardized_rgb(tmp_path, start):
    _model, seen = _run_until_compile(tmp_path, start)
    assert len(seen["datasets"]) == 2
    assert all(kwargs.get("imagenet_standardize") is True for kwargs in seen["datasets"])
