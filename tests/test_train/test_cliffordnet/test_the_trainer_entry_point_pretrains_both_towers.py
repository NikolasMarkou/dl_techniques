"""`train.cliffordnet.train_clip.main` runs its DEFAULT path: both pretraining stages, then CLIP.

Rationale
---------
Vision pretraining is on by default (`--pretrain-vision-steps 50000`), and its
wrapper read `vision_global_pool`, a CliffordCLIP attribute that no longer
exists (the model pools through `vision_det_pool`/`vision_ctx_pool`), so every
default run died with `AttributeError` before the first contrastive step. The
Step 11 entry-point test passed `--skip-pretrain` and never reached it
(plan-2026-10-14T062616-81938e81 D-054). Under `--synthetic` the LM stage was
switched off with a warning, so no test could reach it either.

This file drives `main(argv)` WITHOUT `--skip-pretrain`, one step per stage,
on the in-memory synthetic data every stage now has under `--synthetic`, with
every output under `tmp_path`. `keras.Model.fit` is spied on (not replaced):
each call records which model it trained, how many batches, and every
CliffordCLIP weight before and after, so the freeze map is checked by what
actually moved, not by the trainable flags the trainer set.
"""

import math

import keras
import numpy as np
import pytest

from dl_techniques.models.vision_language.clifford_clip import CliffordCLIP
from train.clip.train_clip import StackedCLIPLogits
from train.cliffordnet.train_clip import main

from .test_the_trainer_entry_point_trains_under_stock_fit import _TINY_ROW

_VARIANT = "pretrain_test_tiny"

#: The default path (no `--skip-pretrain`), one step per pretraining stage.
#: Contrastive training takes two: its warmup is one step, which runs at
#: `warmup_start_lr` 1e-8, below one ulp of `logit_scale` (2.66, ulp 2.4e-7),
#: so a one-step run cannot show the temperature training. Probes off: the
#: Step 11 test covers them; here the subject is the pretraining.
_ARGV = (
    "--synthetic", "--variant", _VARIANT,
    "--image-size", "64", "--context-length", "16", "--vision-patch-size", "4",
    "--batch-size", "8", "--max-train-samples", "16", "--max-val-samples", "8",
    "--epochs", "1", "--steps-per-epoch", "2",
    "--pretrain-vision-steps", "1", "--pretrain-vision-batch-size", "8",
    "--pretrain-lm-steps", "1", "--pretrain-lm-batch-size", "2",
    "--save-every-steps", "1", "--log-every-steps", "1",
    "--probe-every-steps", "0", "--gamma-probe-every-steps", "0", "--seed", "5",
)

#: Direct CliffordCLIP sub-layers each pretraining stage may move. The
#: projection, the Clifford head (geometric product, context pool, LayerScale)
#: and the temperature have no objective before contrastive training.
_VISION_BACKBONE = ("vision_stem", "vision_clifford_block_", "vision_patch_merge_",
                    "vision_merge_proj_", "vision_head_norm")
_TEXT_BACKBONE = ("token_embedding", "position_embedding", "text_embed_norm",
                  "text_clifford_block_", "text_head_norm")


class _BatchCounter(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.batches = 0

    def on_train_batch_end(self, batch, logs=None):
        self.batches += 1


def _snapshot(clip_model):
    return {w.path: np.array(w) for w in clip_model.weights}


def _component(path):
    """The direct CliffordCLIP sub-layer a weight path belongs to."""
    parts = path.split("/")
    return parts[1] if len(parts) > 2 else parts[-1]


def _moved(record):
    return sorted(p for p, v in record["before"].items()
                  if not np.array_equal(v, record["after"][p]))


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """One `main()` run shared by the assertions below."""
    monkeypatch = pytest.MonkeyPatch()
    calls = []
    stock_fit = keras.Model.fit

    def spy(self, *args, **kwargs):
        clip_model = getattr(self, "clip_model", self)
        counter = _BatchCounter()
        kwargs["callbacks"] = [*(kwargs.get("callbacks") or []), counter]
        before = _snapshot(clip_model)
        history = stock_fit(self, *args, **kwargs)
        calls.append(dict(model=self, clip=clip_model, batches=counter.batches,
                          history=history, before=before, after=_snapshot(clip_model)))
        return history

    try:
        monkeypatch.setitem(CliffordCLIP.MODEL_VARIANTS, _VARIANT, _TINY_ROW)
        monkeypatch.setattr(keras.Model, "fit", spy)
        out = tmp_path_factory.mktemp("cliffordclip_pretrain") / "run"
        run_dir = main([*_ARGV, "--output-dir", str(out)])
    finally:
        monkeypatch.undo()
    return out, run_dir, calls


def test_the_default_path_trains_three_stages_in_order(run):
    """Vision pretraining, LM pretraining, then contrastive, each one stock `fit` call."""
    out, run_dir, calls = run
    assert run_dir == out
    assert [c["model"].name for c in calls[:2]] == [
        "pretrain_vision_wrapper", "pretrain_lm_wrapper"], calls
    assert len(calls) == 3 and isinstance(calls[2]["model"], StackedCLIPLogits)
    assert len({id(c["clip"]) for c in calls}) == 1, "one CliffordCLIP throughout"
    assert all(type(c["clip"]) is CliffordCLIP for c in calls)
    assert [c["batches"] for c in calls] == [1, 1, 2]
    for c in calls:
        loss = c["history"].history["loss"]
        assert len(loss) == 1 and math.isfinite(loss[0]), (c["model"].name, loss)
    assert (out / "tensorboard" / "pretrain_vision").is_dir()
    assert (out / "tensorboard" / "pretrain_lm").is_dir()


def test_vision_pretraining_moves_the_vision_backbone_only(run):
    _, _, calls = run
    moved = _moved(calls[0])
    components = {_component(p) for p in moved}
    assert any(c.startswith("vision_clifford_block_") for c in components), moved
    assert any(c.startswith("vision_stem") for c in components), moved
    stray = sorted(c for c in components if not c.startswith(_VISION_BACKBONE))
    assert not stray, f"vision pretraining moved non-backbone sub-layers {stray}"


def test_lm_pretraining_moves_the_text_backbone_only(run):
    _, _, calls = run
    moved = _moved(calls[1])
    components = {_component(p) for p in moved}
    assert "token_embedding" in components, moved
    assert any(c.startswith("text_clifford_block_") for c in components), moved
    stray = sorted(c for c in components if not c.startswith(_TEXT_BACKBONE))
    assert not stray, f"LM pretraining moved non-backbone sub-layers {stray}"


def test_contrastive_training_trains_what_pretraining_froze(run):
    """Every freeze is undone: the temperature and both projections move."""
    _, _, calls = run
    components = {_component(p) for p in _moved(calls[2])}
    for name in ("logit_scale", "vision_projection", "text_projection"):
        assert name in components, (name, sorted(components))
    clip_model = calls[2]["clip"]
    assert clip_model.logit_scale.trainable
    assert all(layer.trainable for layer in clip_model.layers)
