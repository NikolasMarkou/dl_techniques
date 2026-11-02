"""``init_scale_warning`` is only raised, or denied, on a measurement that means it (D-045).

The epoch-0 evaluation runs BatchNorm in inference mode on moving statistics that are still the
initial mean 0 and variance 1, i.e. with no normalization at all. The first training step uses
batch statistics. On ``resnet18`` the two differ by a factor of 15 (ratio to ``ln C`` of 16.3
against 1.5 at CIFAR shapes, 17.4 against 1.04 at ImageNet shapes), so every ResNet run was told
its initialization was wrong ("15.3x ln(1000)") when it was not.

The repair gives no verdict where the measurement cannot give one: a model with unset BatchNorm
statistics gets ``init_scale_warning: null`` and a log line and a summary note that say what was
measured. It runs no training-mode pass, because that would move the moving statistics and every
dropout generator (a diagnostic must not perturb the run), so a badly scaled BatchNorm model is
not caught at epoch 0; a badly scaled model WITHOUT unset statistics still warns.

What is pinned:

* per family (all six): which models are judged and which are not, and that the unjudged ones are
  exactly those that hold unset BatchNorm layers;
* the control that makes the verdict mean something: resnet's inference-mode ratio IS above the
  warning threshold and its training-mode ratio is not, so the old ``True`` was false;
* a truly mis-scaled model without BatchNorm warns, and so does a mis-scaled BatchNorm model whose
  statistics have moved;
* the check reads and assigns nothing (weights, moving statistics and seed generators bit-equal);
* ``unset_batch_statistics`` on small models: fresh, moved mean, moved variance, frozen, the
  variance-only ``BiasFreeBatchNorm``;
* a real API run: the summary, its note, the log and the sidecar, for ``resnet18`` and, as the
  control whose output must not change, ``vit_pico``.

Tests write only under ``tmp_path``.
"""

import json

import keras
import numpy as np
import pytest

from dl_techniques.layers.norms.bias_free_batch_norm import BiasFreeBatchNorm
from train.api import recipes, registry
from train.api.args import AutoContext, OptimArgs
from train.api.tasks.image_classification import ImageClassificationTrainer
from train.common import classification_run
from train.common.classification_run import INITIAL_LOSS_WARN_FACTOR
from train.common.resume_state import FIT_STATE_NAME

from tests.classification_fakes import use_tiny_raw

from .test_task_trainer import make_config

CLASSES = 10
SIZE = 32

#: ``family/variant`` -> the verdict ``check_initial_loss`` must give on an untrained model with a
#: sound initialization: ``False`` (judged, no warning) or ``None`` (not judged: it holds BatchNorm
#: layers with unset statistics). The three ``None`` rows are the three families with BatchNorm.
VERDICT = {
    "vit/vit_pico": False,
    "resnet/resnet18": None,
    "convnext_v1/cifar10": False,
    "convnext_v2/atto": False,
    "cliffordnet/nano": None,
    "cliffordnet_hier/hier_345": None,
}


@pytest.fixture(scope="module")
def builtin():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        yield registry.MODELS


def build(builtin, name):
    family, variant = name.split("/")
    spec = builtin[family]
    context = AutoContext(
        family=family, variant=variant, dataset="cifar10", num_classes=CLASSES,
        image_size=SIZE, defaults=spec.recipe("cifar10", variant),
    )
    kwargs = spec.load_resolver()(context, OptimArgs().resolve_auto(context), {})
    keras.utils.set_random_seed(0)
    model = recipes.build_model(spec, variant, CLASSES, (SIZE, SIZE, 3), kwargs)
    model.compile(
        optimizer="sgd",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    return model


def batch():
    rng = np.random.default_rng(0)
    return (
        rng.standard_normal((64, SIZE, SIZE, 3)).astype("float32"),
        rng.integers(0, CLASSES, 64),
    )


def inference_loss(model, x, y):
    """The loss of ``model(x, training=False)`` as the ``Evaluate`` seam returns it."""
    logits = np.asarray(model(x, training=False), dtype="float64")
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return {"loss": float(-log_probs[np.arange(len(y)), y].mean())}


def check(model):
    x, y = batch()
    return classification_run.check_initial_loss(
        model, None, len(x), CLASSES,
        evaluate=lambda m, _dataset: inference_loss(m, x, y), hint="hint",
    )


def seed_generator_states(model):
    return [
        np.asarray(generator.state.numpy())
        for layer in model._flatten_layers()
        for generator in getattr(layer, "_seed_generators", [])
    ]


# ---------------------------------------------------------------------
# The six families
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(VERDICT))
def test_each_family_is_judged_or_not_by_what_its_pass_measured(name, builtin):
    model = build(builtin, name)
    _metrics, ratio, warned = check(model)
    assert warned is VERDICT[name], (name, ratio)
    layers = [
        layer for layer in model._flatten_layers(include_self=False)
        if hasattr(layer, "moving_variance") or hasattr(layer, "running_var")
    ]
    unset = classification_run.unset_batch_statistics(model)
    # Not judged exactly when there is a BatchNorm layer with unset statistics, and then all of
    # them are unset (nothing has trained).
    assert (warned is None) == bool(layers) == bool(unset), name
    assert len(unset) == len(layers), name


def test_resnets_old_true_was_false_the_first_step_is_near_chance(builtin):
    """The control: the inference-mode ratio is above the threshold, the training-mode one is not."""
    model = build(builtin, "resnet/resnet18")
    x, y = batch()
    inference = inference_loss(model, x, y)["loss"] / np.log(CLASSES)
    assert inference > INITIAL_LOSS_WARN_FACTOR, "the fixture no longer reproduces the false warning"
    logits = np.asarray(model(x, training=True), dtype="float64")  # moves the statistics: last
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    training = float(-log_probs[np.arange(len(y)), y].mean()) / np.log(CLASSES)
    assert training < INITIAL_LOSS_WARN_FACTOR / 2, training


# ---------------------------------------------------------------------
# A model that is wrong still warns
# ---------------------------------------------------------------------


def blow_up_head(model, factor=1e3):
    """Scale the last layer holding a ``kernel`` by ``factor``: a truly mis-scaled output."""
    head = [layer for layer in model._flatten_layers() if hasattr(layer, "kernel")][-1]
    head.kernel.assign(head.kernel.numpy() * factor)


def test_a_mis_scaled_model_without_batchnorm_warns(builtin):
    """Control and arm on the same model: sound, then its head scaled by 1000."""
    model = build(builtin, "vit/vit_pico")
    assert check(model)[2] is False
    blow_up_head(model)
    _metrics, ratio, warned = check(model)
    assert warned is True and ratio > INITIAL_LOSS_WARN_FACTOR


def test_a_mis_scaled_batchnorm_model_with_moved_statistics_warns(builtin):
    model = build(builtin, "resnet/resnet18")
    x, _ = batch()
    for layer in model._flatten_layers():
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.momentum = 0.0  # one pass sets the statistics to the batch's own
    model(x, training=True)  # the statistics move: the evaluation is a measurement again
    assert classification_run.unset_batch_statistics(model) == []
    assert check(model)[2] is False
    blow_up_head(model)
    assert check(model)[2] is True


# ---------------------------------------------------------------------
# The check perturbs nothing
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", ["resnet/resnet18", "vit/vit_pico", "cliffordnet/nano"])
def test_the_check_leaves_weights_statistics_and_generators_bit_equal(name, builtin):
    model = build(builtin, name)
    model(batch()[0][:1], training=False)  # built lazily; an inference call moves nothing
    before = [w.copy() for w in model.get_weights()]
    assert before, "the model holds no weights before the check: nothing would be compared"
    generators = [s.copy() for s in seed_generator_states(model)]
    check(model)
    classification_run.unset_batch_statistics(model)
    after = model.get_weights()
    assert len(after) == len(before)
    assert all(np.array_equal(a, b) for a, b in zip(after, before))
    assert all(
        np.array_equal(a, b) for a, b in zip(seed_generator_states(model), generators)
    )


# ---------------------------------------------------------------------
# unset_batch_statistics on small models
# ---------------------------------------------------------------------


def small(layer):
    model = keras.Sequential([keras.layers.Input((4,)), layer])
    return model, layer


def test_fresh_statistics_are_listed_and_moved_ones_are_not():
    model, layer = small(keras.layers.BatchNormalization(name="bn"))
    assert classification_run.unset_batch_statistics(model) == ["bn"]
    layer.moving_mean.assign(np.full((4,), 0.25, "float32"))
    assert classification_run.unset_batch_statistics(model) == []
    layer.moving_mean.assign(np.zeros((4,), "float32"))
    assert classification_run.unset_batch_statistics(model) == ["bn"]
    layer.moving_variance.assign(np.full((4,), 1.5, "float32"))
    assert classification_run.unset_batch_statistics(model) == []


def test_statistics_that_are_only_partly_at_their_initial_value_are_not_unset():
    """Mutant ``ub_any_not_all`` (review-iter-2 concern 19): ``any`` for ``all`` listed a layer whose
    variance is 1 in three channels and 2 in the fourth, or whose mean is 0 in three and 0.1 in
    the fourth. Those statistics have moved; only ALL channels at 1 and 0 are the initial ones."""
    model, layer = small(keras.layers.BatchNormalization(name="bn"))
    assert classification_run.unset_batch_statistics(model) == ["bn"]
    layer.moving_variance.assign(np.array([1.0, 1.0, 1.0, 2.0], "float32"))
    assert classification_run.unset_batch_statistics(model) == []
    layer.moving_variance.assign(np.ones((4,), "float32"))
    layer.moving_mean.assign(np.array([0.0, 0.0, 0.0, 0.1], "float32"))
    assert classification_run.unset_batch_statistics(model) == []
    model, layer = small(BiasFreeBatchNorm(name="bf"))
    layer.running_var.assign(np.array([1.0, 3.0, 1.0, 1.0], "float32"))
    assert classification_run.unset_batch_statistics(model) == []


def test_a_frozen_layer_is_not_listed_because_inference_is_its_training_behaviour():
    model, layer = small(keras.layers.BatchNormalization(name="bn"))
    layer.trainable = False
    assert classification_run.unset_batch_statistics(model) == []


def test_the_variance_only_layer_is_listed_by_its_running_variance():
    model, layer = small(BiasFreeBatchNorm(name="bf"))
    assert not hasattr(layer, "moving_variance")
    assert classification_run.unset_batch_statistics(model) == ["bf"]
    layer.running_var.assign(np.full((4,), 2.0, "float32"))
    assert classification_run.unset_batch_statistics(model) == []


def test_a_model_without_batch_normalization_lists_nothing():
    model = keras.Sequential(
        [keras.layers.Input((4,)), keras.layers.Dense(3), keras.layers.LayerNormalization()]
    )
    assert classification_run.unset_batch_statistics(model) == []


# ---------------------------------------------------------------------
# A real run
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def runs(builtin, tmp_path_factory):
    from train.api.tasks import register_builtin_tasks

    with pytest.MonkeyPatch.context() as patch:
        use_tiny_raw(patch, n_train=40, n_test=8)
        register_builtin_tasks()
        out = {}
        for name in ("resnet/resnet18", "vit/vit_pico"):
            run_dir = tmp_path_factory.mktemp(name.split("/")[0]) / "run"
            config = make_config(
                run_dir, model=name, epochs=1,
                **{
                    "metric_args.success_threshold": 0.0,
                    "analysis_args.figures": False,
                    "logger_args.tensorboard": None,
                },
            )
            ImageClassificationTrainer(config).run()
            out[name] = run_dir
        yield out


def summary_of(run_dir):
    return json.loads((run_dir / "results_summary.json").read_text())


def test_a_resnet_run_says_not_judged_and_says_what_was_measured(runs):
    run_dir = runs["resnet/resnet18"]
    summary = summary_of(run_dir)
    assert summary["init_scale_warning"] is None
    assert summary["initial_loss_ratio"] > INITIAL_LOSS_WARN_FACTOR  # the baseline is kept
    notes = [n for n in summary["notes"] if n.startswith("`init_scale_warning` is null")]
    assert len(notes) == 1 and "inference mode" in notes[0] and "batch statistics" in notes[0]
    # Review-iter-2 concern 12: the note says what the withheld verdict costs.
    assert "a really mis-scaled BatchNorm model is NOT caught here" in notes[0]
    log = (run_dir / "run.log").read_text()
    assert "far too large" not in log
    assert "No verdict on the initialization is given" in log
    sidecar = json.loads((run_dir / "checkpoints" / FIT_STATE_NAME).read_text())
    assert sidecar["initial"]["warned"] is None
    assert sidecar["initial"]["ratio"] == summary["initial_loss_ratio"]


def test_a_vit_run_is_as_it_was(runs):
    """The control: no BatchNorm, so ``false`` is still written and no note is added."""
    run_dir = runs["vit/vit_pico"]
    summary = summary_of(run_dir)
    assert summary["init_scale_warning"] is False
    assert not [n for n in summary["notes"] if "init_scale_warning" in n]
    assert "No verdict on the initialization" not in (run_dir / "run.log").read_text()
    sidecar = json.loads((run_dir / "checkpoints" / FIT_STATE_NAME).read_text())
    assert sidecar["initial"]["warned"] is False
