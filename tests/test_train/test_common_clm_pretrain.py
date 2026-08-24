"""Contract tests for ``train.common.clm_pretrain`` (the shared CLM wrapper layer).

These are the PERMANENT guards. The one-shot equivalence proof that compared this
module against the four PRE-change trainer copies read out of ``git show HEAD:`` lives
in the plan scratchpad (its method, N and control readings are recorded in the plan's
verification.md); it cannot be a permanent test because its inputs stop existing the
moment steps 6-8 delete those copies.

What is pinned here is what a future edit could silently break:

* the checkpoint step parser's VALUES, including the no-match / multi-match /
  non-digit / empty cases (a regex "tidy-up" is the realistic threat);
* the steps-per-epoch TFDS short-circuit AND the ``max(1, ...)`` floor -- the floor is
  only observable when ``max_samples // batch_size == 0``, so a grid of large values
  measures it zero times;
* the loss branch: object TYPE, ``get_config()`` and the two ``logger.info`` lines
  (D-008 -- the log lines are the 3-of-4 majority behaviour a merge nearly dropped);
* the exact KWARGS both dataset loaders forward, with the underlying loaders
  monkeypatched -- no network, no TFDS download;
* the dict-output ``{"logits": y}`` wrap, asserted by EXECUTING the map fn (a bytecode
  fingerprint was measured BLIND to a ``"logits"`` -> ``"logit"`` mutation);
* ``data_seed`` having NO default (D-009 -- cliffordnet's dead ``= 42``).

CPU-only, no GPU, no dataset download.
"""

import logging
import dataclasses

import pytest

import keras

from train.common import clm_pretrain as cp


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


class Cfg:
    """Minimal stand-in for the four trainers' ``TrainingConfig`` dataclasses."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class FakeDS:
    """Stands in for a ``tf.data.Dataset`` so ``.map()`` is observable."""

    def __init__(self, tag):
        self.tag = tag
        self.mapped = None

    def map(self, fn, num_parallel_calls=None):
        out = FakeDS(f"map({self.tag})")
        # EXECUTE the map fn rather than fingerprinting it: co_code is identical
        # for `{"logits": y}` and `{"logit": y}`.
        out.mapped = fn("X", "Y")
        return out


class Recorder:
    def __init__(self, ret):
        self.ret = ret
        self.calls = []

    def __call__(self, *a, **kw):
        self.calls.append((a, kw))
        return self.ret


@pytest.fixture
def patched_loaders(monkeypatch):
    """Monkeypatch the three leaf loaders the module closes over."""
    recs = {
        "load_text_dataset": Recorder(FakeDS("raw")),
        "preprocess_clm_dataset": Recorder(FakeDS("pp")),
        "load_wikipedia_train_val": Recorder(
            (FakeDS("hf_train"), FakeDS("hf_val"), 4242, 99)
        ),
    }
    for name, rec in recs.items():
        monkeypatch.setattr(cp, name, rec)
    return recs


def hf_cfg(**over):
    base = dict(
        dataset_source="huggingface", dataset_name="imdb_reviews", max_samples=None,
        max_seq_length=1024, batch_size=8, hf_cache_dir="/cache/x",
        hf_wikipedia_config="20231101.en", min_article_length=0, val_fraction=0.01,
        max_train_samples=1000, max_val_samples=50, shuffle_shards=4,
        steps_per_epoch=None,
    )
    base.update(over)
    return Cfg(**base)


# ---------------------------------------------------------------------
# extract_step_from_checkpoint
# ---------------------------------------------------------------------


class TestExtractStepFromCheckpoint:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ("step_0025000.keras", 25000),
            ("final.keras", 0),
            ("/abs/path/to/step_42.keras", 42),
            ("step_0.keras", 0),
            ("step_007.keras", 7),
            # multi-match: the FIRST group wins, not the last and not the max
            ("step_1_step_2.keras", 1),
            ("step_2_step_1.keras", 2),
            # a directory component is NOT consulted -- only the basename
            ("dir_step_5/final.keras", 0),
            ("dir_step_5/step_9.keras", 9),
            # non-digit / near-miss spellings all fall through to 0
            ("step_abc.keras", 0),
            ("step_.keras", 0),
            ("step-9.keras", 0),
            ("STEP_12.keras", 0),
            ("step12.keras", 0),
            ("", 0),
            (".", 0),
            ("/", 0),
            # unicode digits are NOT matched by \d under `re` default flags... they
            # ARE; this pins whichever behaviour the pre-change copies had.
            ("step_١٢.keras", 12),
            # a leading sign is not part of the group
            ("step_+12.keras", 0),
            ("step_1e5.keras", 1),
        ],
    )
    def test_values(self, path, expected):
        assert cp.extract_step_from_checkpoint(path) == expected

    def test_returns_int_not_str(self):
        assert isinstance(cp.extract_step_from_checkpoint("step_12.keras"), int)


# ---------------------------------------------------------------------
# make_clm_steps_per_epoch
# ---------------------------------------------------------------------


class TestMakeClmStepsPerEpoch:
    def test_tfds_short_circuit_divides_samples_by_batch(self):
        cfg = hf_cfg(dataset_source="tfds", max_samples=100, batch_size=8)
        assert cp.make_clm_steps_per_epoch(cfg, None) == 12

    def test_tfds_short_circuit_floors_at_one(self):
        # LOAD-BEARING: 16 // 32 == 0, so this is the ONLY regime in which the
        # `max(1, ...)` floor is observable at all.
        cfg = hf_cfg(dataset_source="tfds", max_samples=16, batch_size=32)
        assert cp.make_clm_steps_per_epoch(cfg, None) == 1

    def test_tfds_short_circuit_is_skipped_when_override_is_set(self):
        cfg = hf_cfg(dataset_source="tfds", max_samples=100, batch_size=8,
                     steps_per_epoch=77)
        assert cp.make_clm_steps_per_epoch(cfg, None) == 77

    def test_tfds_short_circuit_is_skipped_when_max_samples_is_falsy(self):
        cfg_none = hf_cfg(dataset_source="tfds", max_samples=None)
        cfg_zero = hf_cfg(dataset_source="tfds", max_samples=0)
        # falls through to the canonical estimator, which is article-count driven
        assert cp.make_clm_steps_per_epoch(cfg_none, 1000) == \
            cp.make_clm_steps_per_epoch(cfg_zero, 1000)
        assert cp.make_clm_steps_per_epoch(cfg_none, 1000) != 0

    def test_override_wins_on_the_hf_path(self):
        assert cp.make_clm_steps_per_epoch(hf_cfg(steps_per_epoch=123), 5000) == 123

    def test_falls_back_to_max_train_samples_when_article_count_is_none_or_zero(self):
        cfg = hf_cfg(max_train_samples=1000)
        assert cp.make_clm_steps_per_epoch(cfg, None) == \
            cp.make_clm_steps_per_epoch(cfg, 0) == \
            cp.make_clm_steps_per_epoch(cfg, 1000)

    def test_more_articles_means_more_steps(self):
        cfg = hf_cfg()
        assert cp.make_clm_steps_per_epoch(cfg, 100_000) > \
            cp.make_clm_steps_per_epoch(cfg, 1_000)


# ---------------------------------------------------------------------
# create_clm_loss_fn
# ---------------------------------------------------------------------


class TestCreateClmLossFn:
    def test_focal_branch_type_and_config(self):
        loss = cp.create_clm_loss_fn(
            Cfg(loss_type="focal", focal_gamma=2.5, label_smoothing=0.1)
        )
        assert type(loss).__name__ == "FocalCausalLMLoss"
        cfgd = loss.get_config()
        assert cfgd["gamma"] == pytest.approx(2.5)
        assert cfgd["label_smoothing"] == pytest.approx(0.1)

    @pytest.mark.parametrize("loss_type", ["masked", "", None, "FOCAL", "Focal"])
    def test_every_non_focal_spelling_takes_the_masked_branch(self, loss_type):
        # the branch is an EXACT `== "focal"`, so "FOCAL" is masked, not focal
        loss = cp.create_clm_loss_fn(
            Cfg(loss_type=loss_type, focal_gamma=2.0, label_smoothing=0.0)
        )
        assert type(loss).__name__ == "MaskedCausalLMLoss"

    def test_masked_branch_receives_label_smoothing(self):
        loss = cp.create_clm_loss_fn(
            Cfg(loss_type="masked", focal_gamma=2.0, label_smoothing=0.3)
        )
        assert loss.get_config()["label_smoothing"] == pytest.approx(0.3)

    def test_focal_gamma_is_not_hardcoded(self):
        a = cp.create_clm_loss_fn(Cfg(loss_type="focal", focal_gamma=1.0,
                                      label_smoothing=0.0)).get_config()["gamma"]
        b = cp.create_clm_loss_fn(Cfg(loss_type="focal", focal_gamma=4.0,
                                      label_smoothing=0.0)).get_config()["gamma"]
        assert (a, b) == (pytest.approx(1.0), pytest.approx(4.0))

    def test_returns_a_keras_loss(self):
        loss = cp.create_clm_loss_fn(
            Cfg(loss_type="masked", focal_gamma=2.0, label_smoothing=0.0)
        )
        assert isinstance(loss, keras.losses.Loss)

    # D-008: the two logger.info lines are the 3-of-4 majority behaviour and are
    # what one copy -- train_memory.py, since deleted (user instruction 2026-08-13,
    # last present at 9f3208319) -- had silently dropped. Nothing else can see them
    # -- the returned loss object is identical with or without them.
    def test_focal_branch_logs_the_loss_provenance_line_with_ASCII_gamma(self, caplog):
        with caplog.at_level(logging.INFO):
            cp.create_clm_loss_fn(
                Cfg(loss_type="focal", focal_gamma=2.0, label_smoothing=0.0)
            )
        assert "Loss: FocalCausalLMLoss(gamma=2.0)" in caplog.text
        assert "γ" not in caplog.text  # unicode gamma is NOT the canonical form

    def test_masked_branch_logs_the_loss_provenance_line(self, caplog):
        with caplog.at_level(logging.INFO):
            cp.create_clm_loss_fn(
                Cfg(loss_type="masked", focal_gamma=2.0, label_smoothing=0.0)
            )
        assert "Loss: MaskedCausalLMLoss" in caplog.text


# ---------------------------------------------------------------------
# load_tfds_clm_datasets / load_hf_clm_datasets / load_train_val_datasets
# ---------------------------------------------------------------------


class TestLoadTfdsClmDatasets:
    def test_forwards_train_and_test_splits_in_order(self, patched_loaders):
        cfg = hf_cfg(dataset_source="tfds", dataset_name="imdb_reviews",
                     max_samples=64, max_seq_length=128, batch_size=8)
        cp.load_tfds_clm_datasets(cfg, "PREP")
        assert [c[0] for c in patched_loaders["load_text_dataset"].calls] == [
            ("imdb_reviews", "train", 64),
            ("imdb_reviews", "test", 64),
        ]

    def test_forwards_preprocessor_seq_len_and_batch_positionally(self, patched_loaders):
        cfg = hf_cfg(dataset_source="tfds", max_samples=64, max_seq_length=128,
                     batch_size=8)
        cp.load_tfds_clm_datasets(cfg, "PREP")
        for args, kwargs in patched_loaders["preprocess_clm_dataset"].calls:
            assert args[1:] == ("PREP", 128, 8)
            assert kwargs == {}


class TestLoadHfClmDatasets:
    def test_forwards_every_wikipedia_kwarg_by_name(self, patched_loaders):
        cfg = hf_cfg(min_article_length=500, val_fraction=0.02, max_train_samples=7,
                     max_val_samples=3, shuffle_shards=9)
        cp.load_hf_clm_datasets(cfg, "PREP", 1234)
        (args, kwargs), = patched_loaders["load_wikipedia_train_val"].calls
        assert args == ()
        assert kwargs == {
            "cache_dir": "/cache/x",
            "config_name": "20231101.en",
            "min_article_length": 500,
            "val_fraction": 0.02,
            "max_train_samples": 7,
            "max_val_samples": 3,
            "seed": 1234,
            "num_shards": 9,
            "return_counts": True,
        }

    def test_data_seed_reaches_the_split_as_seed(self, patched_loaders):
        cp.load_hf_clm_datasets(hf_cfg(), "PREP", 99)
        assert patched_loaders["load_wikipedia_train_val"].calls[0][1]["seed"] == 99

    def test_returns_the_post_filter_train_article_count(self, patched_loaders):
        _, _, n_train = cp.load_hf_clm_datasets(hf_cfg(), "PREP", 0)
        assert n_train == 4242  # the 3rd element of the loader's tuple, not the 4th


class TestLoadTrainValDatasets:
    def test_tfds_source_routes_to_the_tfds_loader_and_reports_no_article_count(
        self, patched_loaders
    ):
        cfg = hf_cfg(dataset_source="tfds", max_samples=64)
        _, _, n = cp.load_train_val_datasets(cfg, "PREP", 7)
        assert n is None
        assert patched_loaders["load_wikipedia_train_val"].calls == []
        assert len(patched_loaders["load_text_dataset"].calls) == 2

    def test_hf_source_routes_to_the_hf_loader_and_reports_the_article_count(
        self, patched_loaders
    ):
        _, _, n = cp.load_train_val_datasets(hf_cfg(), "PREP", 7)
        assert n == 4242
        assert patched_loaders["load_text_dataset"].calls == []
        assert len(patched_loaders["load_wikipedia_train_val"].calls) == 1

    def test_wraps_labels_into_the_logits_dict(self, patched_loaders):
        train, val, _ = cp.load_train_val_datasets(hf_cfg(), "PREP", 7)
        # asserted by EXECUTING the map fn, not by inspecting it
        assert train.mapped == ("X", {"logits": "Y"})
        assert val.mapped == ("X", {"logits": "Y"})

    def test_unknown_dataset_source_raises_ValueError_naming_both_legal_values(self):
        with pytest.raises(ValueError) as e:
            cp.load_train_val_datasets(hf_cfg(dataset_source="bogus"), "PREP", 7)
        msg = str(e.value)
        assert "bogus" in msg and "tfds" in msg and "huggingface" in msg

    def test_dataset_source_is_checked_before_any_loader_runs(self, patched_loaders):
        with pytest.raises(ValueError):
            cp.load_train_val_datasets(hf_cfg(dataset_source="bogus"), "PREP", 7)
        assert patched_loaders["load_text_dataset"].calls == []
        assert patched_loaders["load_wikipedia_train_val"].calls == []

    # D-009: cliffordnet's `data_seed: int = 42` default was dead and is NOT carried
    # over. A default here would turn a resume-seeding wiring mistake from a loud
    # TypeError into a silent replay of the first N chunks.
    def test_data_seed_has_no_default(self):
        import inspect

        sig = inspect.signature(cp.load_train_val_datasets)
        assert sig.parameters["data_seed"].default is inspect.Parameter.empty


# ---------------------------------------------------------------------
# module surface
# ---------------------------------------------------------------------


class TestModuleSurface:
    # This set is EXACT on purpose -- it is what makes an accidental widening of
    # this module's surface visible. `ClmPretrainConfig` was added deliberately
    # (decisions.md D-002/D-010): the shared CLM config is the config half of
    # the same concern the six functions own, and it lives here rather than in a
    # new `clm_config.py`. Do not relax this to a subset check when adding a
    # name; add the name and say why.
    def test_the_canonical_names_are_exported(self):
        assert set(cp.__all__) == {
            "ClmPretrainConfig",
            "extract_step_from_checkpoint",
            "create_clm_loss_fn",
            "load_train_val_datasets",
            "load_tfds_clm_datasets",
            "load_hf_clm_datasets",
            "make_clm_steps_per_epoch",
        }
        for name in cp.__all__:
            assert callable(getattr(cp, name))
        # A class is callable, so the loop above cannot tell the config apart
        # from the functions. Pin what it actually is.
        assert dataclasses.is_dataclass(cp.ClmPretrainConfig)

    def test_the_package_hub_re_exports_the_same_objects_not_copies(self):
        import train.common as hub

        for name in cp.__all__:
            assert getattr(hub, name) is getattr(cp, name)
            assert name in hub.__all__

    def test_it_delegates_to_the_canonical_steps_per_epoch_helper(self):
        # src/train/CLAUDE.md D-001: never roll a local estimator.
        from train.common import nlp

        assert cp.estimate_clm_steps_per_epoch is nlp.estimate_clm_steps_per_epoch
