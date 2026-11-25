"""The shared pieces of the text tasks' tests; no ``test_`` prefix, so pytest does not collect it.

``language_model_pretraining`` and ``text_classification`` are tested on the offline ``synthetic`` bank with the character tokenizer
``ascii95`` (101 ids, no file): a tiny GPT-2 or BERT (64 wide, 2 blocks, 2 heads) over windows or rows of 32 tokens trains in seconds
on a CPU. This module holds what the task, resume, families and predictor-and-export tests of the two tasks, their init-pair tests and
the init tests that read a text run have in common: the request of a tiny run of either task, the world (an empty registry holding
every built-in family and task), what a resolved trainer looks like before its data is built, a directory that looks like the pinned
tokenizer directory but holds other bytes, and the bound an exported file's answer must meet. The children of the real-kill resume
tests are ``_language_child.py``'s.

Interface contract
------------------
``request(out, **changes)``
    :return: the mapping of a tiny run (a dotted key, ``"data.seq_len"``, sets a field of a group). ``analysis_args`` is off.

``world`` (fixture, module scope)
    An empty registry holding every built-in family and task, restored afterwards.

``prepared(out, **changes)``
    :return: ``(trainer, resolved config, context)``: what the sequence hands the hooks, resolved the way ``run`` resolves,
        nothing claimed and no data loaded.

``wrong_tokenizer_dir(parent)``
    :return: a directory holding the two file names of the pinned ``gpt2`` cache with other bytes (a refusal names their sha256).

``CLASSIFIER_ARGS``
    The ``model_args`` that shrink each classifier family (``bert``, ``gpt2``) to a tiny CPU model.

``classifier_request(out, family="gpt2", **changes)`` / ``classifier_prepared(out, family="gpt2", **changes)``
    The ``text_classification`` twins of ``request`` and ``prepared``: a tiny run of ``<family>/tiny`` on the labelled synthetic bank
    (``CLASSIFIER_DOCUMENTS`` documents, 60 percent of them class 0, a quarter selecting), ``CLASSIFIER_EPOCHS`` epochs.

``within_the_shared_check(got, expected)``
    The bound of the export check (``TFLITE_CHECK_RTOL`` times the larger of 1 and the largest absolute value of ``expected``) as an
    assertion: ``got`` has the shape of ``expected`` and is within it of it. :return: the largest absolute difference.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest

from train.api import recipes, registry
from train.api import text_tokenizer as tt
from train.api.config import validate
from train.api.exporting import TFLITE_CHECK_RTOL
from train.api.tasks import register_builtin_tasks
from train.api.tasks.language_model_pretraining import LanguageModelPretrainingConfig, LanguageModelPretrainingTrainer

from ._resume_protocol import merge_changes

MODEL = "gpt2_lm/tiny"
#: The ``tiny`` variant is 256 wide with 4 blocks; the tests shrink it (``model_args`` win over the variant's table).
TINY_MODEL_ARGS = {"embed_dim": 64, "depth": 2, "num_heads": 2}
#: The same shrink for the ``bert`` classifier (its table names the width ``hidden_size`` and the depth ``num_layers``).
BERT_MODEL_ARGS = {"hidden_size": 64, "num_layers": 2, "num_heads": 2, "intermediate_size": 128}
CLASSIFIER_ARGS = {"bert": BERT_MODEL_ARGS, "gpt2": TINY_MODEL_ARGS}
SEQ_LEN, BATCH, EPOCHS = 32, 16, 3
#: A classifier needs more steps than a language model to leave the majority share on a bank this small, and a selection split of
#: 68 documents (a quarter of 300 by the hash split: 42 and 26) keeps its bar, 0.7944, under what a learner reaches (D-076).
CLASSIFIER_EPOCHS = 8
CLASSIFIER_DOCUMENTS = 300
CLASSIFIER_SPLIT = 0.25
#: Documents of the synthetic bank a tiny run reads (the first ``max_samples`` of each split).
DOCUMENTS = 200
ASCII_VOCAB = 101


def request(out: Any, **changes: Any) -> Dict[str, Any]:
    """The mapping of a tiny run of ``gpt2_lm/tiny`` on the synthetic bank; see the module contract."""
    asked: Dict[str, Any] = {
        "out": str(out),
        "model": MODEL,
        "data": {"dataset": "synthetic", "tokenizer": "ascii95", "seq_len": SEQ_LEN, "max_samples": DOCUMENTS},
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "model_args": dict(TINY_MODEL_ARGS),
        # a tiny model over a hundred steps wants a rate a real run would not take
        "optim_args": {"lr": 3e-3},
        "analysis_args": {"model_analysis": False, "figures": False},
    }
    return merge_changes(asked, changes)


@pytest.fixture(scope="module")
def world():
    """An empty registry holding every built-in family and task."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        register_builtin_tasks()
        yield


def classifier_request(out: Any, family: str = "gpt2", **changes: Any) -> Dict[str, Any]:
    """The mapping of a tiny ``text_classification`` run of ``<family>/tiny`` on the labelled synthetic bank."""
    asked: Dict[str, Any] = {
        "out": str(out),
        "model": f"{family}/tiny",
        "data": {"dataset": "synthetic", "tokenizer": "ascii95", "seq_len": SEQ_LEN, "max_samples": CLASSIFIER_DOCUMENTS,
                 "validation_split": CLASSIFIER_SPLIT},
        "epochs": CLASSIFIER_EPOCHS,
        "batch_size": BATCH,
        "model_args": dict(CLASSIFIER_ARGS[family]),
        "optim_args": {"lr": 3e-3},
        "analysis_args": {"model_analysis": False, "figures": False},
    }
    return merge_changes(asked, changes)


def prepared(out: Any, **changes: Any) -> Tuple[LanguageModelPretrainingTrainer, Any, Any]:
    """``(trainer, resolved config, context)`` of a tiny request; see the module contract."""
    config = validate(LanguageModelPretrainingConfig, request(out, **changes))
    trainer = LanguageModelPretrainingTrainer(config)
    family, variant = registry.parse_model_name(config.model, task=trainer.task)
    resolved, context = trainer._resolve(trainer.describe_data(config), family, variant)
    return trainer, resolved, context


def wrong_tokenizer_dir(parent: Path) -> Path:
    """A directory with the cache file names of ``gpt2`` and other bytes in them; see the module contract."""
    directory = Path(parent) / "wrong_tokenizer_cache"
    directory.mkdir()
    for name, _ in tt.tokenizer_spec("gpt2").files:
        (directory / name).write_bytes(b"not the bpe file")
    return directory


def classifier_prepared(out: Any, family: str = "gpt2", **changes: Any) -> Tuple[Any, Any, Any]:
    """``(trainer, resolved config, context)`` of a tiny classifier request; see :func:`prepared`."""
    from train.api.tasks.text_classification import TextClassificationConfig, TextClassificationTrainer

    config = validate(TextClassificationConfig, classifier_request(out, family, **changes))
    trainer = TextClassificationTrainer(config)
    parsed, variant = registry.parse_model_name(config.model, task=trainer.task)
    resolved, context = trainer._resolve(trainer.describe_data(config), parsed, variant)
    return trainer, resolved, context


def within_the_shared_check(got: np.ndarray, expected: np.ndarray) -> float:
    """Assert ``got`` is ``expected`` within the export check's bound; see the module contract."""
    assert got.shape == expected.shape
    scale = max(1.0, float(np.abs(expected).max()))
    worst = float(np.abs(got.astype(np.float64) - expected).max())
    assert worst <= TFLITE_CHECK_RTOL * scale, f"{worst} against an allowance of {TFLITE_CHECK_RTOL * scale}"
    return worst
