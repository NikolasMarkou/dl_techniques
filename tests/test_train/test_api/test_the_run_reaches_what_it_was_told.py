"""The run reaches what the request told it to: one behaviour test per mutant that survived.

Review iter 2, concern 2. Twelve one-line mutants of the trainer, the task and
the data loader left all 1037 tests of ``test_api`` green, so the suite could
not see a wrong loss, a selection on the test set, lost augmentation, an
ignored seed or a dropped knob. Each test below is the observation that kills
one of them, and states the mutant it kills:

========================  ===================================================
``logits_false``          the compiled loss is cross entropy ON LOGITS, for
                          each of the six families
``select_on_test``        ``fit`` validates on validation rows
``init_loss_on_test``     the initial-loss check sees validation rows
``augment_dropped``       the training pixels of a recipe with a preset differ
                          from the raw ones
``seed_data_fixed``       ``config.seed`` changes the split and the
                          augmentation, and the same seed reproduces them
``top_k_const``           ``metric_args.top_k`` reaches the metric
``best_load_err_off``     an unloadable best checkpoint fails the run
``mode_dropped``          ``watch_mode`` reaches the callbacks, so the
                          opposite epoch is the one kept
``reduce_lr_always_off``  ``schedule=constant`` installs
                          ``ReduceLROnPlateau``, trains, warmup 0
``analyzer_freq``         ``epoch_analysis_every`` is the analyzer's cadence
``conformal_seed``        ``config.seed`` is the conformal splits' seed
========================  ===================================================

The fake is :func:`tests.classification_fakes.use_identity_cifar10`: row ``i``
of the train pool has the mean colour ``4 * i`` and row ``i`` of the test set
``4 * i + 2``, so which split an image came from is read off its pixels
(:func:`kinds`), not off the name of a dataset attribute. Most arms stop when
``fit`` begins (:class:`Spy`), after the real claim, resolution, data, model,
compile and initial-loss check, so they cost seconds. Everything is CPU and
writes under ``tmp_path``.
"""

import json

import keras
import numpy as np
import pytest

import train.api.recipes as recipes
import train.common.conformal as conformal_module
from train.api import cli, registry
from train.api.tasks import register_builtin_tasks
from train.api.tasks.image_classification import ImageClassificationTrainer

from tests.classification_fakes import row_identities, use_identity_cifar10

from .test_task_trainer import make_config, read_json, resolve

#: One cheap variant per family, as ``test_every_family_builds_from_its_defaults`` lists them.
FAMILIES = [
    ("vit", "vit_pico"),
    ("resnet", "resnet18"),
    ("convnext_v1", "cifar10"),
    ("convnext_v2", "atto"),
    ("cliffordnet", "nano"),
    ("cliffordnet_hier", "hier_345"),
]


@pytest.fixture(scope="module", autouse=True)
def world():
    with pytest.MonkeyPatch.context() as patch:
        use_identity_cifar10(patch)
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        register_builtin_tasks()
        yield


# ---------------------------------------------------------------------
# Reading the fake
# ---------------------------------------------------------------------


def images_of(dataset) -> np.ndarray:
    """Every image of a batched ``(image, label)`` dataset, one pass, in order."""
    return np.concatenate([np.asarray(batch[0]) for batch in dataset])


def kinds(dataset) -> set:
    """The set of split markers of the images: ``{0}`` for train-pool rows, ``{2}`` for test rows.

    The fake gives pool row ``i`` the mean colour ``4 * i + 1.5`` (the ramp adds 1.5) and
    test row ``i`` the colour ``4 * i + 3.5``, so the colour modulo the pitch of 4 names the
    split. Valid for an unaugmented dataset.
    """
    from dl_techniques.datasets.vision.classification import get_spec

    spec = get_spec("cifar10")
    level = (images_of(dataset) * np.asarray(spec.std) + np.asarray(spec.mean)).mean(
        axis=(1, 2, 3)
    ) * 255.0
    return {int(k) for k in np.rint(level - 1.5).astype(int) % 4}


def identities(dataset) -> list:
    """The pool-row identity of every image of an unaugmented dataset of pool rows."""
    return sorted(int(i) for i in row_identities(images_of(dataset), "cifar10"))


def build_data(seed=42, **changes):
    """Return ``(data, resolved)`` of the vit recipe on the fake, as the run builds them."""
    config = make_config("/never/opened", seed=seed, **changes)
    resolved, context = resolve(config)
    return ImageClassificationTrainer(resolved).build_data(resolved, context), resolved


# ---------------------------------------------------------------------
# The spy: the real sequence up to the start of ``fit``
# ---------------------------------------------------------------------


class _Stop(Exception):
    """Raised when ``fit`` begins."""


class Spy(ImageClassificationTrainer):
    """The real trainer, recording what its model is handed, and stopping when ``fit`` begins.

    :cvar seen: the record of the last run: ``model``, ``data``, ``compile`` (the keywords
        of ``model.compile``), ``evaluate`` (``(dataset, result)`` per call), ``fit`` (the
        keywords of ``model.fit``) and ``callbacks``.
    :cvar stop: whether ``fit`` ends at its first line (``Run`` turns it off).
    """

    seen = {}
    stop = True

    def build_model(self, config, context, data):
        model = super().build_model(config, context, data)
        seen = type(self).seen = {
            "model": model, "data": data, "compile": None, "evaluate": [], "fit": None,
            "callbacks": None,
        }
        compile_, evaluate_, fit_ = model.compile, model.evaluate, model.fit

        def compile(*args, **kwargs):
            seen["compile"] = kwargs
            return compile_(*args, **kwargs)

        def evaluate(x=None, *args, **kwargs):
            result = evaluate_(x, *args, **kwargs)
            seen["evaluate"].append((x, result))
            return result

        def fit(*args, **kwargs):
            seen["fit"] = kwargs
            return fit_(*args, **kwargs)

        model.compile, model.evaluate, model.fit = compile, evaluate, fit
        return model

    def _callbacks(self, *arguments):
        epoch_line, callbacks = super()._callbacks(*arguments)
        type(self).seen["callbacks"] = list(callbacks)
        return epoch_line, callbacks

    def extra_callbacks(self, config, data):
        if not self.stop:
            return []

        class Stop(keras.callbacks.Callback):
            def on_train_begin(self, logs=None):
                raise _Stop

        return [Stop()]


class Run(Spy):
    """:class:`Spy` that lets ``fit`` run."""

    stop = False


def spy(tmp_path, trainer=Spy, **changes) -> dict:
    """Run ``trainer`` on the fake until ``fit`` begins; return what it recorded."""
    trainer.seen = {}
    config = make_config(tmp_path / "run", epochs=2, **changes)
    with pytest.raises(_Stop):
        trainer(config).run()
    return trainer.seen


# ---------------------------------------------------------------------
# logits_false
# ---------------------------------------------------------------------


def cross_entropy_of_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return float(-log_probabilities[np.arange(len(labels)), labels].mean())


def test_the_families_listed_here_are_the_classification_families_registered():
    # Step 28 (D-077, and plan 98b866fb step 2): the registry also holds the four segmentation families and the five
    # denoising ones (plan d483ffab step 7); this file spies on the
    # CLASSIFICATION run (cross-entropy on logits, top-k), and the segmentation run's compile is
    # checked in test_image_segmentation_task.py and test_segmentation_metrics.py.
    classifiers = sorted(f for f, spec in registry.MODELS.items() if spec.tasks == ("image_classification",))
    assert sorted(family for family, _ in FAMILIES) == classifiers
    assert sorted(set(registry.MODELS) - set(classifiers)) == [
        "accunet", "bfcnn", "bfconvunext", "bfunet", "convnext_seg", "convunext", "convunext_biased", "resnet_seg",
        "scunet",
    ]


@pytest.mark.parametrize("family, variant", FAMILIES, ids=[f"{f}/{v}" for f, v in FAMILIES])
def test_the_compiled_loss_is_cross_entropy_on_the_logits_the_head_emits(family, variant, tmp_path):
    seen = spy(tmp_path, model=f"{family}/{variant}")
    loss = seen["compile"]["loss"]
    x, y = next(iter(seen["data"].select_ds))
    logits = np.asarray(seen["model"](x, training=False))
    labels = np.asarray(y)
    # The head emits logits: its rows are not distributions already (a softmax head with
    # `from_logits=True` would soften the loss; logits with `from_logits=False` would
    # renormalize raw scores).
    assert not np.allclose(logits.sum(axis=-1), 1.0, atol=1e-3)
    assert getattr(loss, "from_logits", None) is True
    assert float(loss(labels, logits)) == pytest.approx(
        cross_entropy_of_logits(logits, labels), rel=1e-4
    )


# ---------------------------------------------------------------------
# select_on_test, init_loss_on_test
# ---------------------------------------------------------------------


def test_the_fake_tells_the_validation_rows_from_the_test_rows():
    data, _ = build_data()
    assert kinds(data.select_ds) == {0}
    assert kinds(data.test_ds) == {2}
    # And the validation rows are not the training rows (no augmentation, so identities read).
    plain, _ = build_data(**{"transform_args.augment": "none"})
    assert not set(identities(plain.select_ds)) & set(identities(plain.train_ds))


def test_fit_validates_on_the_validation_rows_and_never_on_the_test_rows(tmp_path):
    seen = spy(tmp_path)
    validation = seen["fit"]["validation_data"]
    assert kinds(validation) == {0}, "fit validated on test rows"
    assert identities(validation) == identities(seen["data"].select_ds)


def test_the_initial_loss_check_sees_the_validation_rows_and_never_the_test_rows(tmp_path):
    seen = spy(tmp_path)
    first, _ = seen["evaluate"][0]
    assert kinds(first) == {0}, "the initial loss was measured on test rows"
    assert identities(first) == identities(seen["data"].select_ds)


# ---------------------------------------------------------------------
# augment_dropped, seed_data_fixed
# ---------------------------------------------------------------------


def first_pass(seed=42, **changes):
    """Return ``(train pixels of ONE pass, validation identities)`` of a freshly built pipeline.

    One pass only: the training dataset reshuffles on every iteration
    (``reshuffle_each_iteration=True``), so a second pass over the same dataset is another
    order, and comparing a first pass with a second one measures the shuffle, not the
    augmentation.
    """
    data, _ = build_data(seed, **changes)
    return images_of(data.train_ds), identities(data.select_ds)


def test_a_recipe_with_a_preset_trains_on_pixels_that_differ_from_the_raw_ones():
    _, resolved = build_data()
    assert resolved.transform_args.augment == "cifar_light", "the arm needs a recipe with a preset"
    plain, _ = first_pass(**{"transform_args.augment": "none"})
    again, _ = first_pass(**{"transform_args.augment": "none"})
    preset, _ = first_pass()
    # Control: the unaugmented pipeline is reproducible and, at one seed, in one order, so
    # a difference between the arms is the preset.
    assert np.array_equal(plain, again)
    assert preset.shape == plain.shape
    assert not np.array_equal(preset, plain)


def test_the_seed_changes_the_split_and_the_augmentation_and_the_same_seed_reproduces_them():
    one, one_split = first_pass(1)
    one_again, one_again_split = first_pass(1)
    two, two_split = first_pass(2)
    assert one_split == one_again_split
    assert one_split != two_split, "the split ignored the seed"
    assert np.array_equal(one, one_again)
    assert not np.array_equal(one, two)
    # The augmentation alone: the same seed gives the same order with and without the
    # preset, so the pixels differ by what the seeded preset did.
    plain, _ = first_pass(1, **{"transform_args.augment": "none"})
    assert not np.array_equal(one, plain)


# ---------------------------------------------------------------------
# top_k_const
# ---------------------------------------------------------------------


@pytest.mark.parametrize("top_k", [2, 3])
def test_top_k_reaches_the_metric_the_model_reports(top_k, tmp_path):
    seen = spy(tmp_path, **{"metric_args.top_k": top_k})
    reported = seen["evaluate"][0][1]
    assert f"top{top_k}_accuracy" in reported
    assert not [name for name in reported if name.startswith("top") and name != f"top{top_k}_accuracy"]
    stock = [m for m in seen["compile"]["metrics"] if hasattr(m, "k")]
    assert [m.k for m in stock] == [top_k]


# ---------------------------------------------------------------------
# best_load_err_off
# ---------------------------------------------------------------------


def test_an_unloadable_best_checkpoint_fails_the_run_after_the_summary_says_why(
    tmp_path, monkeypatch
):
    out = tmp_path / "run"

    def corrupting(self, config, data):
        class Corrupt(keras.callbacks.Callback):
            def on_train_end(self, logs=None):
                # After ModelCheckpoint has written the best model: the file stays, unreadable.
                (out / "best_model.keras").write_bytes(b"this is not a keras archive")

        return [Corrupt()]

    monkeypatch.setattr(ImageClassificationTrainer, "extra_callbacks", corrupting)
    tokens = [
        f"out={out}", "model=vit/vit_pico", "data.dataset=cifar10", "epochs=1",
        "batch_size=4", "analysis_args.model_analysis=false",
        "metric_args.success_threshold=0.0",
    ]
    # `cli.main` catches `DLTrainError` and nothing else, so a RuntimeError is the process
    # ending non-zero (`test_a_failed_api_run_exits_nonzero` runs that in a child process).
    with pytest.raises(RuntimeError, match="did not load or evaluate"):
        cli.main(["image_classification", *tokens])
    summary = read_json(out, "results_summary.json")
    assert summary["best_checkpoint_load_error"], "the corruption did not reach the loader"


# ---------------------------------------------------------------------
# mode_dropped
# ---------------------------------------------------------------------


class WeightsRecorder(keras.callbacks.Callback):
    """Records the weights at the end of each epoch."""

    def __init__(self):
        super().__init__()
        self.weights = []

    def on_epoch_end(self, epoch, logs=None):
        self.weights.append([np.array(w) for w in self.model.get_weights()])


class Recording(Run):
    recorder = None

    def extra_callbacks(self, config, data):
        type(self).recorder = WeightsRecorder()
        return [type(self).recorder]


def same_weights(left, right) -> bool:
    return len(left) == len(right) and all(np.array_equal(a, b) for a, b in zip(left, right))


def test_watch_mode_reaches_the_callbacks_so_the_opposite_epoch_is_kept(tmp_path):
    out = tmp_path / "run"
    Recording.recorder = None
    Recording(
        make_config(
            out,
            epochs=3,
            **{
                "metric_args.watch_metric": "val_loss",
                "metric_args.watch_mode": "max",
                "analysis_args.figures": False,
                "logger_args.tensorboard": None,
                "checkpoint_args.save_last": False,
            },
        )
    ).run()
    losses = read_json(out, "training_history.json")["val_loss"]
    highest, lowest = int(np.argmax(losses)), int(np.argmin(losses))
    assert highest != lowest and len(set(losses)) == 3
    recorded = Recording.recorder.weights
    assert not same_weights(recorded[highest], recorded[lowest]), "the epochs must differ"
    kept = keras.models.load_model(out / "best_model.keras").get_weights()
    # `max` keeps the epoch with the HIGHEST val_loss. The name-derived direction for a loss
    # is `min`, which is what a dropped `watch_mode` would keep.
    assert same_weights(kept, recorded[highest])
    assert not same_weights(kept, recorded[lowest])


# ---------------------------------------------------------------------
# reduce_lr_always_off
# ---------------------------------------------------------------------


def names_of(callbacks) -> list:
    return [type(callback).__name__ for callback in callbacks]


def test_a_cosine_schedule_installs_no_plateau_callback(tmp_path):
    seen = spy(tmp_path)
    assert "ReduceLROnPlateau" not in names_of(seen["callbacks"])


def test_a_constant_schedule_installs_the_plateau_callback_trains_and_has_no_warmup(tmp_path):
    out = tmp_path / "run"
    Run.seen = {}
    config = make_config(
        out,
        epochs=2,
        **{
            "optim_args.schedule": "constant",
            "analysis_args.figures": False,
            "logger_args.tensorboard": None,
        },
    )
    Run(config).run()
    assert "ReduceLROnPlateau" in names_of(Run.seen["callbacks"])
    resolved = read_json(out, "config.json")
    assert resolved["optim_args"]["schedule"] == "constant"
    assert resolved["optim_args"]["warmup_epochs"] == 0
    history = read_json(out, "training_history.json")
    assert len(history["val_accuracy"]) == 2
    # No warmup and no decay: the rate is the configured one at the start of every epoch.
    assert history["lr"] == [pytest.approx(resolved["optim_args"]["lr"])] * 2


# ---------------------------------------------------------------------
# analyzer_freq
# ---------------------------------------------------------------------


def test_epoch_analysis_every_is_the_cadence_of_the_analyzer(tmp_path):
    from dl_techniques.callbacks.analyzer_callback import EpochAnalyzerCallback

    seen = spy(tmp_path, **{"analysis_args.epoch_analysis_every": 3})
    [analyzer] = [c for c in seen["callbacks"] if isinstance(c, EpochAnalyzerCallback)]
    analyzed = []
    analyzer._analyze = analyzed.append
    # The standard schedule after the pre-training snapshot: epoch 1, then every 3rd from 3.
    for epoch in range(7):
        analyzer.on_epoch_end(epoch)
    assert analyzed == [1, 3, 6]


def test_no_epoch_analysis_means_no_analyzer_callback(tmp_path):
    from dl_techniques.callbacks.analyzer_callback import EpochAnalyzerCallback

    seen = spy(tmp_path, **{"analysis_args.epoch_analysis_every": 0})
    assert not [c for c in seen["callbacks"] if isinstance(c, EpochAnalyzerCallback)]


# ---------------------------------------------------------------------
# conformal_seed
# ---------------------------------------------------------------------


def test_the_conformal_splits_use_the_run_seed(tmp_path, monkeypatch):
    calls = []
    original = conformal_module.run_conformal

    def recording(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(conformal_module, "run_conformal", recording)
    Run.seen = {}
    Run(
        make_config(
            tmp_path / "run",
            epochs=1,
            seed=7,
            **{
                "analysis_args.conformal_alpha": 0.2,
                "analysis_args.figures": False,
                "logger_args.tensorboard": None,
                "checkpoint_args.save_last": False,
            },
        )
    ).run()
    assert [call["seed"] for call in calls] == [7]
    record = json.loads((tmp_path / "run" / "conformal.json").read_text())
    assert record["alpha"] == 0.2
