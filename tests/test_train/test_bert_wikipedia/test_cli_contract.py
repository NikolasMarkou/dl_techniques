"""CLI contract guards for ``src/train/bert/wikipedia/``.

This package had ZERO tests of any kind, and all three of its scripts were
non-runnable at ``011c56409``:

* both pretrain scripts raised ``ValueError: Unrecognized keyword arguments
  passed to BERT: {'dropout_rate': 0.1}`` at model construction,
* ``finetune.py`` raised ``ValueError: as_supervised=True but glue does not
  support a supervised structure`` at data load,
* and none of the three defined an ``ArgumentParser``, so ``--help`` was
  ignored and ``main()`` ran a real training attempt (all three exited 1 with
  zero bytes of stdout).

What each guard pins, and why it is not satisfied-by-construction:

``test_help_exits_zero_without_reaching_anything_expensive``
    The real invariant is NOT "the parser exists" and NOT "``--help`` exits 0".
    It is that ``--help`` exits having called NOTHING expensive. Sentinels are
    installed over ``tf.distribute.MirroredStrategy``, ``datasets.load_dataset``,
    ``tfds.load``, ``BERT.from_variant`` and ``train.common.setup_gpu``; each
    raises on contact and each is asserted to have been called ZERO times.
    Moving ``parse_arguments`` back below the strategy build fails this by the
    named ``--help reached ...`` message, not by the exit code -- so the guard
    distinguishes "parsed first" from "parsed at all".

``test_argv_maps_onto_the_config``
    The repo's documented silent-no-op bug class: a flag that parses and is
    then never forwarded onto the config. Every pinned value DIFFERS from the
    class default, and the test additionally asserts one untouched field still
    holds its default, so it cannot pass by observing a fresh default config.
    It also asserts WHICH sentinel stopped ``main()``, so a probe that returned
    early without reaching the config assignments fails instead of passing.

``test_from_variant_is_called_with_the_kwargs_BERT_actually_has`` /
``test_from_variant_source_kwargs_are_accepted_by_BERT``
    The first is a name assertion on the AST of the real call site. The second
    is the non-textual half: it EXECUTES ``BERT.from_variant`` with the kwarg
    names taken from the source, so restoring ``dropout_rate=0.1`` fails
    against the real constructor rather than against a string match.

``test_preprocess_glue_consumes_dict_elements`` /
``test_finetune_does_not_request_as_supervised``
    TFDS' GLUE builder declares no supervised structure. The fixture's keys are
    the builder's own, MEASURED on this machine against
    ``/media/arxwn/data0_4tb/datasets/tensorflow_datasets/glue/sst2/2.0.0``:
    ``{'idx': int32, 'label': int64, 'sentence': string}``. The fixture's
    ``idx`` values are deliberately far from its ``label`` values so a wrapper
    that reads the wrong key produces different labels rather than plausible
    ones.

No network and no training run: these scripts need HF Hub access and
MirroredStrategy, both of which are sentinelled off here.
"""

import ast
from pathlib import Path
from typing import Any, Dict, List

import pytest
import tensorflow as tf

import datasets
import tensorflow_datasets as tfds

import train.common
import train.bert.wikipedia.finetune as finetune
import train.bert.wikipedia.pretrain as pretrain
import train.bert.wikipedia.pretrain_english as pretrain_english

from dl_techniques.models.bert import BERT

SRC = Path(__file__).resolve().parents[3] / "src"
WIKIPEDIA = SRC / "train" / "bert" / "wikipedia"

MODULES = {
    "pretrain": pretrain,
    "pretrain_english": pretrain_english,
    "finetune": finetune,
}

PRETRAIN_SCRIPTS = ("pretrain", "pretrain_english")


class _Abort(BaseException):
    """Raised by a sentinel on contact.

    Derives from ``BaseException``, NOT ``Exception``, on purpose: both pretrain
    scripts wrap the ``MirroredStrategy`` build in ``except Exception``, so an
    ``Exception``-derived sentinel would be swallowed and ``main()`` would carry
    on into the dataset build. Measured necessity, not caution.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


class _Sentinel:
    """Callable that records contact, and aborts unless it is passive.

    Contract: ``calls`` counts invocations. An active sentinel raises
    :class:`_Abort` carrying ``name`` and never returns; a passive one records
    and returns ``None``, which is what lets a probe run PAST a cheap call it
    only wants to neutralise (``setup_gpu``) to reach the code it measures.
    """

    def __init__(self, name: str, active: bool = True) -> None:
        self.name = name
        self.active = active
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.active:
            raise _Abort(self.name)
        return None


def _install_sentinels(
    monkeypatch: pytest.MonkeyPatch, passive: tuple = ()
) -> Dict[str, _Sentinel]:
    """Install a recording sentinel over every expensive thing ``main()`` reaches.

    Contract:
      - Returns ``{name: sentinel}``; every sentinel counts its calls, and every
        sentinel NOT named in ``passive`` raises :class:`_Abort` on contact.
      - Covers the four side effects the plan names (distributed strategy, HF
        Hub load, TFDS load, model construction) plus ``setup_gpu``, which the
        wikipedia scripts import INSIDE ``main()`` (see decisions.md D-006) and
        which is therefore patched on ``train.common``, not on the script module.
      - ``monkeypatch`` restores every attribute at teardown.
    """
    sentinels = {
        name: _Sentinel(name, active=name not in passive)
        for name in (
            "tf.distribute.MirroredStrategy",
            "datasets.load_dataset",
            "tfds.load",
            "BERT.from_variant",
            "train.common.setup_gpu",
        )
    }
    monkeypatch.setattr(
        tf.distribute, "MirroredStrategy", sentinels["tf.distribute.MirroredStrategy"]
    )
    monkeypatch.setattr(datasets, "load_dataset", sentinels["datasets.load_dataset"])
    monkeypatch.setattr(tfds, "load", sentinels["tfds.load"])
    monkeypatch.setattr(BERT, "from_variant", sentinels["BERT.from_variant"])
    monkeypatch.setattr(train.common, "setup_gpu", sentinels["train.common.setup_gpu"])
    return sentinels


def _record_config(
    monkeypatch: pytest.MonkeyPatch, module: Any, class_name: str, **overrides: Any
) -> List[Any]:
    """Replace ``module.<class_name>`` with a subclass that registers instances.

    Contract:
      - Returns a list that receives every instance ``main()`` constructs, in
        construction order.
      - ``overrides`` are applied to each instance right after construction --
        used to point ``save_dir`` at a tmp path so ``main()``'s ``os.makedirs``
        cannot write into the repo.
      - Subclassing (rather than patching the instance afterwards) is what makes
        the config observable at all: these configs are plain classes that
        ``main()`` builds and mutates as a LOCAL, with no ``build_config(args)``
        seam to intercept.
    """
    real = getattr(module, class_name)
    made: List[Any] = []

    class _Recorded(real):  # type: ignore[valid-type,misc]
        def __init__(self) -> None:
            super().__init__()
            for key, value in overrides.items():
                setattr(self, key, value)
            made.append(self)

    monkeypatch.setattr(module, class_name, _Recorded)
    return made


def _from_variant_keywords(script: str) -> List[str]:
    """Keyword names at the single ``BERT.from_variant`` call site of ``script``."""
    tree = ast.parse((WIKIPEDIA / f"{script}.py").read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_variant"
    ]
    assert len(calls) == 1, (
        f"{script}.py has {len(calls)} BERT.from_variant call sites; this guard "
        "assumes exactly one and would otherwise check the wrong one"
    )
    return [keyword.arg for keyword in calls[0].keywords]


def _tfds_load_keywords() -> List[str]:
    """Keyword names at ``finetune.py``'s single ``tfds.load`` call site."""
    tree = ast.parse((WIKIPEDIA / "finetune.py").read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tfds"
    ]
    assert len(calls) == 1, (
        f"finetune.py has {len(calls)} tfds.load call sites; this guard assumes one"
    )
    return [keyword.arg for keyword in calls[0].keywords]


# ---------------------------------------------------------------------
# (b) --help is cheap -- the footgun this package existed to demonstrate
# ---------------------------------------------------------------------


@pytest.mark.parametrize("script", sorted(MODULES))
def test_help_exits_zero_without_reaching_anything_expensive(
    script: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    module = MODULES[script]
    sentinels = _install_sentinels(monkeypatch)

    try:
        module.main(["--help"])
    except SystemExit as exit_exc:
        code = exit_exc.code
    except _Abort as abort:
        pytest.fail(
            f"{script}: --help reached {abort.name!r} before argparse could "
            f"exit. `args = parse_arguments(argv)` must be the FIRST statement "
            f"of main(), above the strategy build, the dataset build and model "
            f"construction."
        )
    else:
        pytest.fail(f"{script}: main(['--help']) returned without raising SystemExit")

    assert code == 0, f"{script}: --help exited {code!r}, expected 0"

    captured = capsys.readouterr()
    assert "usage:" in captured.out, (
        f"{script}: --help printed no usage line; stdout was {captured.out[:200]!r}"
    )

    reached = {name: s.calls for name, s in sentinels.items() if s.calls}
    assert reached == {}, (
        f"{script}: --help performed side effects: {reached}. A --help that "
        f"builds a strategy, loads a dataset, allocates a GPU or constructs a "
        f"model is the original defect."
    )


# ---------------------------------------------------------------------
# (a) argv -> config: the silent-no-op bug class
# ---------------------------------------------------------------------

# Each expected value differs from the class default on purpose, so a dropped
# assignment cannot pass. `stops_at` pins WHICH sentinel halted main(), so a
# probe that returned early fails rather than reporting a pristine config.
ARGV_CASES = {
    "pretrain": {
        "argv": ["--variant", "tiny", "--batch-size", "8",
                 "--total-steps", "77", "--learning-rate", "0.005"],
        "expected": {"bert_variant": "tiny", "global_batch_size": 8,
                     "total_steps": 77, "learning_rate": 0.005},
        "untouched": {"vocab_size": 100277, "max_seq_length": 512},
        "config_class": "PretrainConfig",
        "stops_at": "tf.distribute.MirroredStrategy",
    },
    "pretrain_english": {
        "argv": ["--variant", "tiny", "--batch-size", "8",
                 "--total-steps", "77", "--learning-rate", "0.005",
                 "--max-non-ascii-ratio", "0.42"],
        "expected": {"bert_variant": "tiny", "global_batch_size": 8,
                     "total_steps": 77, "learning_rate": 0.005,
                     "max_non_ascii_ratio": 0.42},
        "untouched": {"vocab_size": 100277, "wikipedia_id": "20220301.en"},
        "config_class": "PretrainConfig",
        "stops_at": "tf.distribute.MirroredStrategy",
    },
    "finetune": {
        "argv": ["--batch-size", "4", "--epochs", "9", "--learning-rate", "0.001"],
        "expected": {"batch_size": 4, "epochs": 9, "learning_rate": 0.001},
        "untouched": {"task_name": "sst2", "max_seq_length": 128},
        "config_class": "FinetuneConfig",
        "stops_at": "tfds.load",
    },
}


@pytest.mark.parametrize("script", sorted(ARGV_CASES))
def test_argv_maps_onto_the_config(
    script: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case = ARGV_CASES[script]
    module = MODULES[script]
    # `setup_gpu` is passive here: it is called BEFORE the config assignments,
    # so an aborting sentinel would stop main() short of the thing under test.
    # It is still counted, and asserted to have been called exactly once.
    sentinels = _install_sentinels(monkeypatch, passive=("train.common.setup_gpu",))
    made = _record_config(
        monkeypatch, module, case["config_class"], save_dir=str(tmp_path / "run")
    )

    with pytest.raises(_Abort) as excinfo:
        module.main(case["argv"])

    assert excinfo.value.name == case["stops_at"], (
        f"{script}: main() stopped at {excinfo.value.name!r}, expected "
        f"{case['stops_at']!r} -- the probe did not reach the point it claims to measure"
    )
    assert len(made) == 1, f"{script}: main() built {len(made)} configs, expected 1"
    config = made[0]

    assert sentinels["train.common.setup_gpu"].calls == 1, (
        f"{script}: main() called setup_gpu "
        f"{sentinels['train.common.setup_gpu'].calls} times, expected exactly 1 "
        f"-- --gpu must still be wired through to the shared helper"
    )

    for field, want in case["expected"].items():
        got = getattr(config, field)
        assert got == pytest.approx(want) if isinstance(want, float) else got == want, (
            f"{script}: CLI value for {field} never reached the config "
            f"(got {got!r}, argv asked for {want!r}). A flag that parses and is "
            f"then not forwarded is a knob that silently does nothing."
        )

    for field, want in case["untouched"].items():
        assert getattr(config, field) == want, (
            f"{script}: untouched field {field} is not at its default {want!r} -- "
            f"this test is not looking at the config it thinks it is"
        )


@pytest.mark.parametrize("script", sorted(ARGV_CASES))
def test_parse_arguments_returns_defaults_for_empty_argv(script: str) -> None:
    """Bare argv parses, and every flag defaults to the config's own value."""
    module = MODULES[script]
    args = module.parse_arguments([])
    assert args.gpu is None
    config_class = getattr(module, ARGV_CASES[script]["config_class"])
    for field in ARGV_CASES[script]["expected"]:
        flag = {
            "bert_variant": "variant",
            "global_batch_size": "batch_size",
        }.get(field, field)
        assert getattr(args, flag) == getattr(config_class, field), (
            f"{script}: --{flag.replace('_', '-')} defaults to "
            f"{getattr(args, flag)!r}, but {config_class.__name__}.{field} is "
            f"{getattr(config_class, field)!r}; a bare run would silently change "
            f"behaviour relative to running the script before it had a CLI"
        )


# ---------------------------------------------------------------------
# (c) BERT.from_variant call shape -- the dropout_rate crash
# ---------------------------------------------------------------------


@pytest.mark.parametrize("script", PRETRAIN_SCRIPTS)
def test_from_variant_is_called_with_the_kwargs_BERT_actually_has(script: str) -> None:
    keywords = _from_variant_keywords(script)
    assert "dropout_rate" not in keywords, (
        f"{script}.py passes dropout_rate to BERT.from_variant. BERT.__init__ has "
        f"no such parameter and Keras raises "
        f"'Unrecognized keyword arguments passed to BERT'. Use "
        f"hidden_dropout_rate / attention_probs_dropout_rate (both default 0.1)."
    )
    assert {"hidden_dropout_rate", "attention_probs_dropout_rate"} <= set(keywords), (
        f"{script}.py no longer passes both dropout probabilities explicitly; "
        f"got {keywords}. The working sibling train/bert/pretrain.py:85-86 passes both."
    )


@pytest.mark.parametrize("script", PRETRAIN_SCRIPTS)
def test_from_variant_source_kwargs_are_accepted_by_BERT(script: str) -> None:
    """Execute the source's own kwarg NAMES against the real constructor.

    The non-textual half of the guard above: a kwarg name that BERT does not
    accept fails here against Keras, not against a string comparison.
    """
    small = {"variant": "tiny", "vocab_size": 64, "max_position_embeddings": 32}
    call = {name: small.get(name, 0.1) for name in _from_variant_keywords(script)}
    try:
        encoder = BERT.from_variant(**call)
    except ValueError as exc:  # pragma: no cover - only on a regression
        pytest.fail(
            f"{script}.py's BERT.from_variant call shape is rejected by BERT: "
            f"{exc} (kwargs used: {sorted(call)})"
        )
    assert encoder.hidden_dropout_rate == pytest.approx(0.1)
    assert encoder.attention_probs_dropout_rate == pytest.approx(0.1)


# ---------------------------------------------------------------------
# (d) GLUE is dict-shaped, not a supervised pair
# ---------------------------------------------------------------------

# Measured against the local TFDS build of glue/sst2 2.0.0 -- not guessed:
#   {'idx': int32, 'label': int64, 'sentence': string}
GLUE_SST2_FEATURES = {"idx", "label", "sentence"}


def _glue_like_dataset() -> tf.data.Dataset:
    """A synthetic stand-in with GLUE/SST-2's exact element structure.

    ``idx`` values are far away from ``label`` values on purpose: a wrapper that
    reads the wrong key yields 100/101/... instead of 0/1, so the labels
    assertion distinguishes them.
    """
    return tf.data.Dataset.from_tensor_slices(
        {
            "idx": tf.constant([100, 101, 102, 103], tf.int32),
            "label": tf.constant([0, 1, 1, 0], tf.int64),
            "sentence": tf.constant(
                ["a great film", "utterly dull", "wonderful and warm", "a waste of time"]
            ),
        }
    )


def test_the_glue_fixture_matches_the_real_feature_names() -> None:
    """Anti-vacuity: the fixture is only evidence if its keys are GLUE's."""
    element = _glue_like_dataset().element_spec
    assert set(element) == GLUE_SST2_FEATURES


def test_preprocess_glue_consumes_dict_elements() -> None:
    config = finetune.FinetuneConfig()
    config.max_seq_length = 32
    config.batch_size = 2
    tokenizer = finetune.create_tokenizer(config)

    processed = finetune.preprocess_glue(_glue_like_dataset(), tokenizer, config)

    labels: List[int] = []
    batches = 0
    for inputs, batch_labels in processed:
        batches += 1
        assert set(inputs) == {"input_ids", "attention_mask", "token_type_ids"}
        for key in inputs:
            assert tuple(inputs[key].shape) == (2, 32), (
                f"{key} has shape {tuple(inputs[key].shape)}, expected (2, 32)"
            )
            assert inputs[key].dtype == tf.int32
        labels.extend(int(v) for v in batch_labels.numpy())

    assert batches == 2
    assert sorted(labels) == [0, 0, 1, 1], (
        f"labels came back as {sorted(labels)}; GLUE's labels are 0/1 while this "
        f"fixture's idx values are 100-103, so this means the wrapper read the "
        f"wrong feature key"
    )


def test_finetune_does_not_request_as_supervised() -> None:
    keywords = _tfds_load_keywords()
    assert "as_supervised" not in keywords, (
        "finetune.py passes as_supervised to tfds.load. GLUE declares no "
        "supervised structure, so TFDS raises 'as_supervised=True but glue does "
        "not support a supervised structure' before a single element is read."
    )
