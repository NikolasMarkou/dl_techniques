"""The six families have a recipe on both ImageNet and Imagenette, and the recipes are what they say.

Step 16 of the plan. Before it, a run of any family on either dataset that had no entry in the
family's ``per_dataset`` table logged ``No recipe overlay for <family> on <dataset>``: ten of the
twelve pairs had none. Now every pair has an entry (some are equal to the base recipe: the entry is
the statement that the base IS the recipe there, and it is what the notice looks for).

Five things are pinned.

* ``RECIPES``: the RESOLVED recipe of each of the twelve (family, dataset) pairs, every number the
  README's table lists (epochs, batch, the optimizer's 14 fields, augment, erasing, patience, watch
  metric, loader, model keywords), as literals. A change of one value is a changed test.
* Where a legacy trainer exists, the recipe EQUALS the legacy trainer's own default, read from the
  legacy module (``TrainingConfig`` of ``train_vit`` / ``train_resnet`` / ``convnext.common``), not
  retyped here. The Imagenette pairs of vit, resnet and convnext have NO legacy source; a test pins
  that fact (convnext refuses Imagenette) so that a legacy trainer that starts to support one turns
  it red and the table's justification is reopened.
* The notice: no pair warns, and a dataset with no entry still does (the control).
* Every pair builds its smallest variant at 224 px and takes ONE optimizer step on fake data on CPU
  with the optimizer the recipe resolves to; the loss is finite and the step count moves by one.
* The loss has no label smoothing (the task builds a plain sparse cross-entropy, every legacy value
  was 0 or dead).

Everything runs on the CPU and writes nothing.
"""

import dataclasses
import importlib

import numpy as np
import pytest

from train.api import recipes, registry
from train.api.args import AutoContext, OptimArgs, TransformArgs
from train.api.config import validate
from train.api.data_args import DataFacts
from train.api.optim import build_optimizer_and_schedule
from train.api.tasks.image_classification import ImageClassificationConfig, ImageClassificationTrainer

IMAGE_SIZE = 224
CLASSES = {"imagenet": 1000, "imagenette": 10}
#: Steps per epoch of the fit split at the recipe's batch: ImageNet 1,271,167 images, Imagenette
#: 8,523 (findings/imagenet-imagenette.md). Used only for the optimizer's schedule.
TRAIN_IMAGES = {"imagenet": 1_271_167, "imagenette": 8_523}

#: The smallest variant of each family that is built at 224 px.
VARIANT = {
    "vit": "vit_pico", "resnet": "resnet18", "convnext_v1": "tiny", "convnext_v2": "atto",
    "cliffordnet": "nano", "cliffordnet_hier": "hier_345",
}

ADAMW_FIELDS = dict(
    optimizer="adamw", momentum=None, beta_1=None, beta_2=None, epsilon=None, schedule="cosine_decay",
    warmup_start_lr=1e-8, min_lr_ratio=0.01, clipnorm=None, global_clipnorm=None,
)


def row(*, epochs, batch_size, patience, watch, optim, loader=None, model_args, l2=None, augment="none"):
    """One pinned row; ``optim`` holds the fields that differ from ``ADAMW_FIELDS``."""
    return dict(
        epochs=epochs, batch_size=batch_size, patience=patience, watch=watch, augment=augment,
        erasing=0.0, loader=loader or {}, model_args=model_args, l2=l2,
        optim={**ADAMW_FIELDS, **optim},
    )


VIT_MODEL = {"dropout_rate": 0.1, "attention_dropout_rate": 0.1, "patch_size": 16}
VIT_OPTIM = dict(lr=3e-4, weight_decay=0.05, decay_over="total", global_clipnorm=1.0)
#: The penalty strength the recipe hands the model: Keras `L2(l)` has the gradient `2 * l * w`, SGD weight decay `wd` has the
#: gradient `wd * w`, so the model gets `wd / 2` (D-042).
RESNET_L2 = 1e-4 / 2
# No clip: He et al. 2015 and torchvision's reference train.py (`--clip-grad-norm` default None) have none (D-042).
RESNET_OPTIM = dict(
    optimizer="sgd", lr=0.1, weight_decay=1e-4, decay_over="total", momentum=0.9,
)
CONVNEXT_OPTIM = dict(lr=1e-3, weight_decay=1e-4, warmup_epochs=0, decay_over="post_warmup", clipnorm=1.0)
CONVNEXT_MODEL = {"strides": 4, "drop_path_rate": 0.1, "dropout_rate": 0.0}
CLIFFORD_OPTIM = dict(
    lr=1e-3, weight_decay=0.1, warmup_epochs=0, decay_over="post_warmup", beta_1=0.9, beta_2=0.999,
    epsilon=1e-8,
)

#: (family, dataset) -> the resolved recipe. The sources are in the README's table and in D-035.
RECIPES = {
    # vit/imagenet: train_vit.py TrainingConfig (epochs 50, batch 128, 3e-4, wd 0.05, warmup 5, clip 1.0, patience 15).
    ("vit", "imagenet"): row(
        epochs=50, batch_size=128, patience=15, watch="val_accuracy",
        optim={**VIT_OPTIM, "warmup_epochs": 5}, model_args=VIT_MODEL),
    # vit/imagenette: NEW. 200 epochs of 66 steps is the step budget of the cliffordnet Imagenette
    # recipe; the warmup keeps the legacy 10% of the schedule.
    ("vit", "imagenette"): row(
        epochs=200, batch_size=128, patience=15, watch="val_accuracy",
        optim={**VIT_OPTIM, "warmup_epochs": 20}, model_args=VIT_MODEL),
    # resnet/imagenet: train_resnet.py TrainingConfig; the loader's colour jitter is the legacy augmentation.
    ("resnet", "imagenet"): row(
        epochs=100, batch_size=256, patience=15, watch="val_loss",
        optim={**RESNET_OPTIM, "warmup_epochs": 5}, loader={"imagenet_augment_color": True},
        model_args={"stem_type": "imagenet"}, l2=RESNET_L2),
    # resnet/imagenette: NEW. Goyal et al.'s linear rule at half the batch (0.1 * 128 / 256), warmup 5% of the schedule;
    # the loader refuses imagenet_augment_color for Imagenette.
    ("resnet", "imagenette"): row(
        epochs=200, batch_size=128, patience=15, watch="val_loss",
        optim={**RESNET_OPTIM, "lr": 0.05, "warmup_epochs": 10},
        model_args={"stem_type": "imagenet"}, l2=RESNET_L2),
    # convnext_v1 and v2 /imagenet: convnext/common.py (100, 64, 1e-3, 1e-4, 0, patience 50, strides 4, 0.1, 0.0).
    ("convnext_v1", "imagenet"): row(
        epochs=100, batch_size=64, patience=50, watch="val_loss", optim=CONVNEXT_OPTIM, model_args=CONVNEXT_MODEL),
    ("convnext_v2", "imagenet"): row(
        epochs=100, batch_size=64, patience=50, watch="val_loss", optim=CONVNEXT_OPTIM, model_args=CONVNEXT_MODEL),
    # convnext /imagenette: NEW (the legacy trainer refuses it): the ImageNet row, 100 epochs of 133 steps at batch 64.
    ("convnext_v1", "imagenette"): row(
        epochs=100, batch_size=64, patience=50, watch="val_loss", optim=CONVNEXT_OPTIM, model_args=CONVNEXT_MODEL),
    ("convnext_v2", "imagenette"): row(
        epochs=100, batch_size=64, patience=50, watch="val_loss", optim=CONVNEXT_OPTIM, model_args=CONVNEXT_MODEL),
    # cliffordnet and hier /imagenet and /imagenette: train_cliffordnet.py (200, 1e-3, wd 0.1, patience 30, augment none)
    # with the BATCH from the step 17 memory measurement at 224 px (memory_fit_evidence.json): the legacy 128 asks
    # 147.6 GiB of an isotropic nano, so cliffordnet takes 8 and cliffordnet_hier 16. The learning rate is NOT scaled:
    # the legacy trainer has no batch rule (lr 1e-3 at batch 128 in the paper recipe and at batch 32 in its README).
    ("cliffordnet", "imagenet"): row(
        epochs=200, batch_size=8, patience=30, watch="val_accuracy", optim=CLIFFORD_OPTIM,
        model_args={"stochastic_depth_rate": 0.3}),
    ("cliffordnet", "imagenette"): row(
        epochs=200, batch_size=8, patience=30, watch="val_accuracy", optim=CLIFFORD_OPTIM,
        model_args={"stochastic_depth_rate": 0.3}),
    ("cliffordnet_hier", "imagenet"): row(
        epochs=200, batch_size=16, patience=30, watch="val_accuracy", optim=CLIFFORD_OPTIM,
        model_args={"stochastic_depth_rate": 0.1}),
    ("cliffordnet_hier", "imagenette"): row(
        epochs=200, batch_size=16, patience=30, watch="val_accuracy", optim=CLIFFORD_OPTIM,
        model_args={"stochastic_depth_rate": 0.1}),
}
PAIRS = sorted(RECIPES)
IDS = [f"{family}-{dataset}" for family, dataset in PAIRS]


@pytest.fixture(scope="module")
def builtin():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        yield registry.MODELS


def resolve(builtin, family, dataset):
    """Return ``(context, optim_args, transform_args, factory kwargs)`` as the trainer resolves them."""
    context = AutoContext(
        family=family, variant=VARIANT[family], dataset=dataset, num_classes=CLASSES[dataset],
        image_size=IMAGE_SIZE, defaults=builtin[family].recipe(dataset, VARIANT[family]),
    )
    optim = OptimArgs().resolve_auto(context)
    transform = TransformArgs().resolve_auto(context)
    kwargs = builtin[family].load_resolver()(context, optim, {})
    return context, optim, transform, kwargs


def test_the_table_has_the_twelve_pairs():
    assert len(PAIRS) == 12
    assert {family for family, _ in PAIRS} == set(VARIANT)
    assert {dataset for _, dataset in PAIRS} == set(CLASSES)


@pytest.mark.parametrize("family, dataset", PAIRS, ids=IDS)
def test_the_resolved_recipe_is_the_pinned_one(builtin, family, dataset):
    want = RECIPES[(family, dataset)]
    context, optim, transform, kwargs = resolve(builtin, family, dataset)
    recipe = context.defaults
    assert recipe["epochs"] == want["epochs"] and recipe["batch_size"] == want["batch_size"]
    assert recipe["checkpoint_args"] == {"early_stopping_patience": want["patience"]}
    assert recipe["metric_args"] == {"watch_metric": want["watch"]}
    assert recipe.get("loader", {}) == want["loader"]
    assert dataclasses.asdict(optim) == want["optim"]
    assert (transform.augment, transform.random_erasing_prob) == (want["augment"], want["erasing"])
    assert transform.image_size == IMAGE_SIZE
    regularizer = kwargs.pop("kernel_regularizer", None)
    assert kwargs == want["model_args"]
    assert (None if regularizer is None else float(regularizer.l2)) == want["l2"]


@pytest.mark.parametrize("family, dataset", PAIRS, ids=IDS)
def test_no_pair_logs_the_no_overlay_warning(builtin, family, dataset, tmp_path):
    notices = trainer(tmp_path).recipe_notices(
        DataFacts(dataset=dataset, num_classes=CLASSES[dataset], image_size=IMAGE_SIZE), family
    )
    assert notices == []


def trainer(tmp_path):
    config = validate(ImageClassificationConfig, {
        "out": str(tmp_path / "run"), "model": "vit/vit_pico", "data": {"dataset": "cifar10"},
    })
    return ImageClassificationTrainer(config)


@pytest.mark.parametrize("dataset", sorted(CLASSES))
@pytest.mark.parametrize("family", sorted(VARIANT))
def test_the_control_a_pair_whose_entry_is_deleted_warns_again(builtin, family, dataset, tmp_path, monkeypatch):
    """This is the RED of the step in miniature: delete one overlay, the notice comes back."""
    spec = builtin[family]
    kept = {name: row for name, row in spec.defaults["per_dataset"].items() if name != dataset}
    monkeypatch.setitem(
        registry.MODELS, family, dataclasses.replace(spec, defaults={**spec.defaults, "per_dataset": kept})
    )
    notices = trainer(tmp_path).recipe_notices(
        DataFacts(dataset=dataset, num_classes=CLASSES[dataset], image_size=IMAGE_SIZE), family
    )
    assert [level for level, _ in notices] == ["warning"]
    assert f"No recipe overlay for {family} on {dataset}" in notices[0][1]


# ---------------------------------------------------------------------
# The sgd weight decay is the paper's, and nothing clips (D-042)
# ---------------------------------------------------------------------

#: SGD weight decays: the ResNet recipe's, AlexNet's 5e-4 (Krizhevsky et al.), and a large one.
WEIGHT_DECAYS = [1e-4, 5e-4, 0.05]


def penalty_gradient(regularizer, weights):
    """The gradient of ``regularizer(w)`` with respect to ``w``, by ``GradientTape``."""
    import tensorflow as tf

    variable = tf.Variable(weights)
    with tf.GradientTape() as tape:
        penalty = regularizer(variable)
    return tape.gradient(penalty, variable).numpy()


@pytest.mark.parametrize("decay", WEIGHT_DECAYS)
def test_the_sgd_penalty_has_the_gradient_of_the_sgd_weight_decay(decay):
    """Keras ``L2(l)`` is ``l * sum(w^2)``, whose gradient is ``2 * l * w``; SGD weight decay ``wd`` adds ``wd * w`` to the
    gradient (He et al. 2015 section 3.4; torch.optim.SGD; torchvision's reference). With ``w = 1`` the gradient of the
    recipe's penalty is therefore ``wd`` exactly, and ``2 * wd`` was the doubled decay."""
    regularizer = recipes._weight_decay_regularizer(OptimArgs(optimizer="sgd", weight_decay=decay))
    ones = np.ones((4, 3), "float32")
    assert (penalty_gradient(regularizer, ones) == np.float32(decay)).all()
    weights = np.random.default_rng(0).normal(size=(4, 3)).astype("float32")
    assert (penalty_gradient(regularizer, weights) == (np.float32(decay) * weights)).all()


@pytest.mark.parametrize("decay", WEIGHT_DECAYS)
def test_one_sgd_step_on_the_penalty_alone_shrinks_the_weights_by_one_minus_lr_wd(decay):
    """The recipe's regularizer under a real Keras ``SGD`` step, with a data loss that is exactly 0: the weights end at
    ``w * (1 - lr * wd)``, which is what ``torch.optim.SGD(weight_decay=wd)`` does to a parameter with a zero gradient."""
    import keras

    lr = 0.1
    regularizer = recipes._weight_decay_regularizer(OptimArgs(optimizer="sgd", weight_decay=decay))
    layer = keras.layers.Dense(3, use_bias=False, kernel_regularizer=regularizer)
    inputs = keras.Input((4,))
    model = keras.Model(inputs, keras.layers.Lambda(lambda t: t * 0.0)(layer(inputs)))
    start = np.random.default_rng(1).normal(size=(4, 3)).astype("float32")
    layer.kernel.assign(start)
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr), loss="mse")
    model.train_on_batch(np.ones((2, 4), "float32"), np.zeros((2, 3), "float32"))
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(layer.kernel), start * (1.0 - lr * decay), rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("family, dataset", [pair for pair in PAIRS if pair[0] == "resnet"], ids=lambda v: str(v))
def test_the_resnet_recipe_hands_the_model_the_penalty_of_its_weight_decay_and_builds_an_unclipped_optimizer(
    builtin, family, dataset
):
    context, optim, _, kwargs = resolve(builtin, family, dataset)
    assert (penalty_gradient(kwargs["kernel_regularizer"], np.ones((2, 2), "float32")) == np.float32(optim.weight_decay)).all()
    optimizer, _ = build_optimizer_and_schedule(optim, TRAIN_IMAGES[dataset] // context.defaults["batch_size"], 2)
    assert (optimizer.clipnorm, optimizer.global_clipnorm, optimizer.clipvalue) == (None, None, None)


# ---------------------------------------------------------------------
# Legacy sources
# ---------------------------------------------------------------------


def legacy_defaults(module, name="TrainingConfig"):
    """The default of every field of a legacy ``TrainingConfig`` dataclass, read from the module."""
    cls = getattr(importlib.import_module(module), name)
    return {field.name: field.default for field in dataclasses.fields(cls)}


def test_vit_on_imagenet_is_the_legacy_trainers_default(builtin):
    legacy = legacy_defaults("train.vit.train_vit")
    context, optim, _, kwargs = resolve(builtin, "vit", "imagenet")
    assert (context.defaults["epochs"], context.defaults["batch_size"]) == (legacy["epochs"], legacy["batch_size"])
    assert (optim.lr, optim.weight_decay, optim.warmup_epochs, optim.global_clipnorm) == (
        legacy["learning_rate"], legacy["weight_decay"], legacy["warmup_epochs"], legacy["gradient_clipping"])
    assert (optim.optimizer, optim.schedule) == (legacy["optimizer_type"], legacy["lr_schedule_type"])
    assert context.defaults["checkpoint_args"]["early_stopping_patience"] == legacy["early_stopping_patience"]
    assert (kwargs["dropout_rate"], kwargs["attention_dropout_rate"]) == (
        legacy["dropout_rate"], legacy["attention_dropout_rate"])
    assert kwargs["patch_size"] == 16 and legacy["patch_size"] == 4  # the ImageNet branch of main(), :426-428
    # Its loss is a plain sparse cross-entropy; the `label_smoothing` field, which only refused a nonzero value, is gone.
    assert "label_smoothing" not in legacy


def test_resnet_on_imagenet_is_the_legacy_trainers_default(builtin):
    legacy = legacy_defaults("train.resnet.train_resnet")
    context, optim, _, kwargs = resolve(builtin, "resnet", "imagenet")
    assert (context.defaults["epochs"], context.defaults["batch_size"]) == (legacy["epochs"], legacy["batch_size"])
    assert (optim.optimizer, optim.lr, optim.weight_decay, optim.warmup_epochs, optim.momentum) == (
        legacy["optimizer_type"], legacy["learning_rate"], legacy["weight_decay"], legacy["warmup_epochs"],
        legacy["momentum"])
    assert context.defaults["checkpoint_args"]["early_stopping_patience"] == legacy["early_stopping_patience"]
    # Neither arm clips (the paper and torchvision's reference do not) and neither has a smoothing field: the legacy 0.1
    # never reached the loss (README, D-003) and the field is gone (D-042). The recipe carries no clip.
    assert optim.clipnorm is None and optim.global_clipnorm is None
    assert "gradient_clipping" not in legacy and "label_smoothing" not in legacy
    # Both arms give the model the same penalty: wd / 2, whose gradient is the SGD weight decay (D-042).
    assert float(kwargs["kernel_regularizer"].l2) == legacy["weight_decay"] / 2


@pytest.mark.parametrize("family", ["convnext_v1", "convnext_v2"])
def test_convnext_on_imagenet_is_the_legacy_trainers_default_and_imagenette_is_not_legacy(builtin, family):
    common = importlib.import_module("train.convnext.common")
    legacy = legacy_defaults("train.convnext.common")
    context, optim, _, kwargs = resolve(builtin, family, "imagenet")
    assert (context.defaults["epochs"], context.defaults["batch_size"]) == (legacy["epochs"], legacy["batch_size"])
    assert (optim.lr, optim.weight_decay, optim.warmup_epochs) == (
        legacy["learning_rate"], legacy["weight_decay"], legacy["warmup_epochs"])
    assert context.defaults["checkpoint_args"]["early_stopping_patience"] == legacy["patience"]
    assert kwargs["strides"] == common.DEFAULT_STRIDES["imagenet"]
    assert {key: kwargs[key] for key in ("drop_path_rate", "dropout_rate")} == common.REGULARIZATION_DEFAULTS["imagenet"]
    assert legacy["label_smoothing"] == 0.0
    # The fact the Imagenette row's "no legacy source" rests on.
    assert "imagenette" in common.UNSUPPORTED_DATASETS and "imagenette" not in common.DATASETS


# ---------------------------------------------------------------------
# The model builds at its input size and takes a step
# ---------------------------------------------------------------------


@pytest.mark.parametrize("family, dataset", PAIRS, ids=IDS)
def test_the_model_builds_at_224_and_takes_one_optimizer_step(builtin, family, dataset):
    import keras

    keras.utils.set_random_seed(0)
    context, optim, _, kwargs = resolve(builtin, family, dataset)
    classes = CLASSES[dataset]
    model = recipes.build_model(builtin[family], VARIANT[family], classes, (IMAGE_SIZE, IMAGE_SIZE, 3), kwargs)
    images = np.random.default_rng(0).normal(size=(2, IMAGE_SIZE, IMAGE_SIZE, 3)).astype("float32")
    assert tuple(model(images, training=False).shape) == (2, classes)
    steps = TRAIN_IMAGES[dataset] // context.defaults["batch_size"]
    optimizer, _ = build_optimizer_and_schedule(optim, steps, context.defaults["epochs"])
    model.compile(optimizer=optimizer, loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    loss = model.train_on_batch(images, np.array([0, classes - 1], "int32"))
    assert np.isfinite(float(loss)), loss
    assert int(optimizer.iterations) == 1
    assert all(np.isfinite(np.asarray(w)).all() for w in model.get_weights())


def test_the_loss_the_task_builds_has_no_smoothing(tmp_path):
    from types import SimpleNamespace

    loss, _ = trainer(tmp_path).build_loss_and_metrics(
        trainer(tmp_path).config, SimpleNamespace(num_classes=10))
    assert type(loss).__name__ == "SparseCategoricalCrossentropy"
    assert loss.get_config().get("from_logits") is True and "label_smoothing" not in loss.get_config()
