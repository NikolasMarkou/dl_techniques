"""A model with a position limit refuses an over-long sequence CLEANLY, on every path.

Subject: ``dl_techniques.utils.sequence_length_guard`` and every language model
wired onto it (``SUBJECTS`` below; plan ``plan-2026-10-11T170958-dc4d2202``,
D-006). "Cleanly" has four parts, and each is an arm because each failed
somewhere on its own:

``test_a_static_over_long_first_call_is_refused_before_any_variable_exists``
    ``model(ids)`` with a statically over-long length raises ``ValueError``
    from ``build``, and the model holds NO variable afterwards. A refusal that
    fires inside ``call`` (after ``build`` made the variables) is the
    precondition of the unusable-model defect below.
``test_a_refused_first_predict_leaves_the_model_usable``
    A ``ValueError`` raised during the first trace of Keras's cached
    ``predict_function`` leaves that function broken for every later call
    (measured in 9 of 13 classes before the guard). The refusal must drop the
    cached step functions, so the next legal ``predict`` works and is finite.
``test_a_refusal_caught_inside_fit_does_not_kill_the_fit``
    The other side of that drop: a BUILT model must keep its step functions.
    ``fit`` makes ``train_function`` once, before its epoch loop, so a refusal
    caught by a callback mid-fit that nulled it left the loop calling ``None``
    (``TypeError``) on the next epoch. Run on BERT, GPT2 and Qwen3 (a learned
    table, a ``TextDecoder`` table, a RoPE decoder), headed so ``fit`` has a
    loss; the fit must finish all three epochs and the model still predict.
``test_the_declared_limit_itself_is_legal``
    Anti-vacuity: a ``<`` where ``<=`` belongs passes both raise arms while
    breaking every model run at exactly its declared maximum.
``test_a_relaxed_retrace_refuses_an_over_long_batch_and_stays_usable``
    After ``predict`` has seen two lengths, TensorFlow retraces with the
    sequence axis relaxed to ``None`` and never re-enters Python for a new
    length, so the static check is skipped. The run-time gate must fire there:
    ``InvalidArgumentError`` (a Python ``ValueError`` cannot cross a graph
    boundary). Run three ways: Keras's default on a CPU-only box, Keras's
    default with a GPU visible (XLA), and ``jit_compile=True``. Under XLA the
    assertion is dropped, and only the zero-sum term, which fails the XLA
    compile naming ``seq_len_exceeds_<limit_name>_<limit>``, refuses; on a
    gather-based position table the alternative is FINITE GARBAGE. A legal call
    afterwards must work again.
``test_the_model_reaches_the_shared_guard``
    The guard is only worth its name if a class cannot silently bypass it with
    a private copy. The model's module is patched to record the shared
    functions' calls: both must be reached, by the module-level names.

Why XLA is forced: a CPU-only suite never sees XLA, and Keras 3.8 resolves
``jit_compile="auto"`` through ``Trainer._resolve_auto_jit_compile``, which on
a GPU box returns ``model_supports_jit(model)``. The resolver is replaced so
every arm runs the same configuration on any machine;
``monkeypatch.setattr`` raises if Keras renames it, and each arm asserts the
resolved value, so neither can go vacuous.

Adding a subject is one ``Subject(...)`` row in ``SUBJECTS``.

Two more groups sit below the subject arms:

``NO_LIMIT`` (``test_a_model_without_a_position_limit_serves_any_length``)
    Models with no position limit must NOT be guarded: Mamba, Mamba2, Zamba2
    with no ``g`` position (the shared attention, the only RoPE table, is never
    run), ModernBERT with fewer layers than ``global_attention_interval`` (no
    global layer, so no RoPE table) and HNet whose attention is only in an
    inner stage or absent (inner stages see at most ``max_chunks`` positions;
    the cap truncates silently, which is out of scope). Each serves a length
    2.5x or more over the constructor's ``max_seq_len``, finite, eagerly and
    under ``jit_compile=True``.
    This is also the arm that goes RED if a guard is wired unconditionally.
``test_an_archive_with_an_over_long_recorded_shape_loads_every_weight``
    Every subject rebuilds from the shape recorded in its ``.keras`` archive.
    Keras 3.8 runs that rebuild inside a bare ``try/except`` and then marks the
    model built anyway, so a refusal there loaded an old archive whose recorded
    shape is over-long with ZERO weights and only a ``UserWarning`` (12 of 14
    subjects before ``RecordedBuildShapeMixin``; WaveFieldLLM, whose variables
    exist from ``__init__``, kept them but still raised the ``UserWarning``,
    an error under this suite). The load must log a warning, restore every
    variable, and leave the model refusing an over-long call.
``test_blt_warns_and_loads_an_archive_with_an_over_long_recorded_shape``
    BLT's own override of the same load (it predates the mixin), with its
    forward pinned too (precedent: ``PositionalEmbedding.build_from_config``).
"""

import importlib
import json
import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType
from dl_techniques.models.language.hnet.config import (
    AttnSpec,
    HNetArchConfig,
    SSMSpec,
)
from dl_techniques.utils import sequence_length_guard as guard
from dl_techniques.models.language.bert import create_bert

from .headed_logit_decoder_cases import CLF
from .tiny_language_models import TINY

# ---------------------------------------------------------------------

BATCH = 2
LEGAL_LENGTHS = (5, 12)
OVER_BY = 1

# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Subject:
    """One model under test.

    :param name: Test id.
    :param module: Importable module whose namespace holds the model's
        references to the two shared guard functions (patched by the
        reachability arm).
    :param limit_name: Name of the limit attribute on the model; also the
        stem of the XLA guard node.
    :param factory: Builds a fresh, unbuilt, tiny model with a small limit.
    :param feed: ``(model, seq_len) -> model input`` of that sequence length.
    :param message_regex: Pattern (with ``{limit}``) the graph-path refusal
        must match.
    :param single_length: True when the model can serve only the ONE length it
        was first built at (FNet: its DFT matrix is ``(S, S)`` for the first
        ``S`` seen, and any other length raises ``Incompatible shapes``; BLT
        under XLA: a second length retraces with a relaxed axis and
        ``LocalDecoder`` fails ``BroadcastArgs must be a compile-time
        constant``, with or without the guard, measured at HEAD). Its only
        legal length is then the limit itself, and the relaxed retrace is
        reached by one legal call followed by the over-long one.
    :param variables_at_init: True when the model creates every variable in
        ``__init__`` (WaveFieldLLM builds at ``max_seq_len`` there), so a
        refusal cannot precede variable creation; the static arm then asserts
        the variable set is unchanged by the refusal instead of empty.
    """

    name: str
    module: str
    limit_name: str
    factory: Callable[[], keras.Model]
    feed: Callable[[keras.Model, int], Any]
    message_regex: str
    single_length: bool = False
    variables_at_init: bool = False


def _bert_factory() -> keras.Model:
    return create_bert(
        "tiny",
        include_top=True,
        task_config=NLPTaskConfig(
            name="sentiment",
            task_type=NLPTaskType.SENTIMENT_ANALYSIS,
            num_classes=3,
        ),
        max_position_embeddings=16,
    )


def _ids(seq_len: int) -> np.ndarray:
    return np.random.RandomState(7).randint(0, 60, (BATCH, seq_len)).astype("int32")


def _bert_feed(model: keras.Model, seq_len: int) -> Dict[str, np.ndarray]:
    ids = _ids(seq_len)
    return {
        "input_ids": ids,
        "attention_mask": np.ones_like(ids),
        "token_type_ids": np.zeros_like(ids),
    }


def _mask_feed(model: keras.Model, seq_len: int) -> Dict[str, np.ndarray]:
    ids = _ids(seq_len)
    return {"input_ids": ids, "attention_mask": np.ones_like(ids)}


def _ids_only_feed(model: keras.Model, seq_len: int) -> Dict[str, np.ndarray]:
    return {"input_ids": _ids(seq_len)}


def _array_feed(model: keras.Model, seq_len: int) -> np.ndarray:
    """For a model whose ``call`` takes the id tensor itself, not a dict."""
    return _ids(seq_len)


def _tiny(module: str, cls: str, **kwargs: Any) -> Callable[[], keras.Model]:
    """A factory for a tiny, unbuilt ``cls`` from ``module`` (vocab 64)."""

    def factory() -> keras.Model:
        klass = getattr(importlib.import_module(module), cls)
        return klass(vocab_size=64, **kwargs)

    return factory


def _hnet(layout: list, max_seq_len: int = 16) -> Callable[[], keras.Model]:
    """A factory for a tiny two-stage HNet with the given ``arch_layout``.

    ``max_chunks`` is pinned to 8, so inner stages never see more than 8
    positions whatever the input length (the cap truncates silently).
    """

    def factory() -> keras.Model:
        from dl_techniques.models.language.hnet.model import HNet

        config = HNetArchConfig(
            arch_layout=layout,
            d_model=[16, 16],
            d_intermediate=[0, 0],
            vocab_size=64,
            ssm_cfg=SSMSpec(d_conv=4, expand=2, d_state=8),
            attn_cfg=AttnSpec(
                num_heads=(2, 2), rotary_emb_dim=(4, 4), window_size=(-1, -1)
            ),
        )
        return HNet(config, max_chunks=(8,), max_seq_len=max_seq_len, headdim=8)

    return factory


_LANG = "dl_techniques.models.language."
_POSITION_TABLE = r"exceeds max_position_embeddings={limit}(?![0-9])"
_SEQ_LEN = r"exceeds max_seq_len={limit}(?![0-9])"

SUBJECTS = [
    Subject(
        name="bert",
        module=_LANG + "bert.model",
        limit_name="max_position_embeddings",
        factory=_bert_factory,
        feed=_bert_feed,
        message_regex=_POSITION_TABLE,
    ),
    Subject(
        name="distilbert",
        module=_LANG + "distilbert.model",
        limit_name="max_position_embeddings",
        factory=_tiny(
            _LANG + "distilbert.model", "DistilBERT", hidden_size=32,
            num_layers=1, num_heads=2, intermediate_size=64,
            max_position_embeddings=16,
        ),
        feed=_mask_feed,
        message_regex=_POSITION_TABLE,
    ),
    Subject(
        name="fnet",
        module=_LANG + "fnet.model",
        limit_name="max_position_embeddings",
        factory=_tiny(
            _LANG + "fnet.model", "FNet", hidden_size=32, num_layers=1,
            intermediate_size=64, max_position_embeddings=16,
        ),
        feed=_mask_feed,
        message_regex=_POSITION_TABLE,
        single_length=True,
    ),
    Subject(
        name="modern_bert",
        module=_LANG + "modern_bert.model",
        limit_name="max_position_embeddings",
        # Three layers, global attention every 3rd: only a global layer has a
        # RoPE table, and a model without one has no position limit.
        factory=_tiny(
            _LANG + "modern_bert.model", "ModernBERT", hidden_size=32,
            num_layers=3, num_heads=2, intermediate_size=64,
            max_position_embeddings=16, local_attention_window_size=8,
        ),
        feed=_mask_feed,
        message_regex=_POSITION_TABLE,
    ),
    Subject(
        name="tree_transformer",
        module=_LANG + "tree_transformer.model",
        limit_name="max_len",
        factory=_tiny(
            _LANG + "tree_transformer.model", "TreeTransformer",
            hidden_size=32, num_layers=1, num_heads=2, intermediate_size=64,
            max_len=16,
        ),
        feed=_mask_feed,
        message_regex=r"exceeds max_len={limit}(?![0-9])",
    ),
    Subject(
        name="gpt2",
        module=_LANG + "gpt2.gpt2",
        limit_name="max_seq_len",
        factory=_tiny(
            _LANG + "gpt2.gpt2", "GPT2", embed_dim=32, depth=1, num_heads=2,
            max_seq_len=16,
        ),
        feed=_mask_feed,
        message_regex=_SEQ_LEN,
    ),
    Subject(
        name="topolm",
        module=_LANG + "topolm.model",
        limit_name="max_seq_len",
        factory=_tiny(
            _LANG + "topolm.model", "TopoLM", embed_dim=64, depth=1,
            num_heads=2, max_seq_len=16, radius=2,
        ),
        feed=_mask_feed,
        message_regex=_SEQ_LEN,
    ),
    Subject(
        name="wave_field",
        module=_LANG + "wave_field.model",
        limit_name="max_seq_len",
        factory=_tiny(
            _LANG + "wave_field.model", "WaveFieldLLM", embed_dim=32, depth=1,
            num_heads=2, max_seq_len=16, field_size=32, dropout_rate=0.0,
            attention_dropout_rate=0.0,
        ),
        feed=_mask_feed,
        message_regex=_SEQ_LEN,
        variables_at_init=True,
    ),
    Subject(
        name="qwen3",
        module=_LANG + "qwen.qwen3",
        limit_name="max_seq_len",
        factory=_tiny(
            _LANG + "qwen.qwen3", "Qwen3", hidden_size=32, num_layers=1,
            num_attention_heads=2, num_key_value_heads=1, max_seq_len=16,
            moe_layers=[], num_experts=1, num_experts_per_tok=1,
            dropout_rate=0.0,
        ),
        feed=_mask_feed,
        message_regex=_SEQ_LEN,
    ),
    Subject(
        name="qwen3_next",
        module=_LANG + "qwen.qwen3_next",
        limit_name="max_seq_len",
        factory=_tiny(
            _LANG + "qwen.qwen3_next", "Qwen3Next", hidden_size=32,
            num_layers=1, num_attention_heads=2, num_key_value_heads=1,
            max_seq_len=16, num_experts=2, num_experts_per_tok=1,
            moe_intermediate_size=32,
        ),
        feed=_mask_feed,
        message_regex=_SEQ_LEN,
    ),
    Subject(
        name="gemma3",
        module=_LANG + "gemma.gemma3",
        limit_name="max_seq_len",
        factory=_tiny(
            _LANG + "gemma.gemma3", "Gemma3", hidden_size=32, num_layers=1,
            num_attention_heads=2, num_key_value_heads=1, ffn_hidden_size=64,
            max_seq_len=16, sliding_window_size=8,
        ),
        feed=_mask_feed,
        message_regex=_SEQ_LEN,
    ),
    Subject(
        name="zamba2",
        module=_LANG + "zamba2.model",
        limit_name="max_seq_len",
        factory=_tiny(
            _LANG + "zamba2.model", "Zamba2Model", hidden_size=32,
            layer_mapping=["m", "g"], num_mem_blocks=1, num_heads=2,
            max_seq_len=16, d_state=8, headdim=8,
        ),
        feed=_array_feed,
        message_regex=_SEQ_LEN,
    ),
    # Stage 0 has an attention block (the encoder's `T1`): that is the only
    # layout shape with a position limit, see the no-limit arm below.
    Subject(
        name="hnet_stage0_attention",
        module=_LANG + "hnet.model",
        limit_name="max_seq_len",
        factory=_hnet(["T1", ["m1"], "m1"]),
        feed=_array_feed,
        message_regex=_SEQ_LEN,
    ),
    Subject(
        name="blt",
        module=_LANG + "blt.model",
        limit_name="max_sequence_length",
        factory=_tiny(
            _LANG + "blt.model", "ByteLatentTransformer", local_dim=32,
            global_dim=48, num_local_layers=1, num_global_layers=1,
            num_heads_local=2, num_heads_global=2, max_sequence_length=16,
            max_patches=4, dropout_rate=0.0,
        ),
        feed=_array_feed,
        message_regex=r"exceeds max_sequence_length={limit}(?![0-9])",
        single_length=True,
    ),
]

SUBJECT_IDS = [s.name for s in SUBJECTS]

# ---------------------------------------------------------------------


def _limit(subject: Subject, model: keras.Model) -> int:
    return int(getattr(model, subject.limit_name))


def _legal_lengths(subject: Subject, limit: int) -> tuple:
    """The lengths a healthy model is served, ending in the limit itself."""
    return (limit,) if subject.single_length else LEGAL_LENGTHS


def _finite(out: Any) -> bool:
    return all(
        np.isfinite(np.asarray(leaf)).all() for leaf in keras.tree.flatten(out)
    )


_JIT_CONFIGS = ("cpu_default", "gpu_default", "jit_compile_true")


def _resolve_jit(monkeypatch: pytest.MonkeyPatch, config: str) -> bool:
    """Pin how ``"auto"`` resolves; return the ``jit_compile`` the model must end with.

    ``jit_compile_true`` leaves the resolver alone: the caller sets it.
    """
    # Keras exposes no public switch for how ``jit_compile="auto"`` resolves,
    # so the private resolver is the only seam; reached through ``keras``.
    keras_trainer = keras.src.trainers.trainer

    if config == "jit_compile_true":
        return True
    expected = config == "gpu_default"
    monkeypatch.setattr(
        keras_trainer.Trainer,
        "_resolve_auto_jit_compile",
        (lambda self: keras_trainer.model_supports_jit(self))
        if expected
        else (lambda self: False),
    )
    return expected


# ---------------------------------------------------------------------


@pytest.mark.parametrize("subject", SUBJECTS, ids=SUBJECT_IDS)
def test_a_static_over_long_first_call_is_refused_before_any_variable_exists(subject):
    model = subject.factory()
    limit = _limit(subject, model)
    before = [w.path for w in model.weights]
    with pytest.raises(ValueError, match=subject.limit_name):
        model(subject.feed(model, limit + OVER_BY))
    after = [w.path for w in model.weights]
    if subject.variables_at_init:
        assert before and after == before, (before, after)
    else:
        assert before == [] and after == [], (before, after)


@pytest.mark.parametrize("subject", SUBJECTS, ids=SUBJECT_IDS)
def test_a_refused_first_predict_leaves_the_model_usable(subject):
    model = subject.factory()
    limit = _limit(subject, model)
    with pytest.raises(ValueError, match=subject.limit_name):
        model.predict(subject.feed(model, limit + OVER_BY), verbose=0)
    for seq_len in _legal_lengths(subject, limit):
        out = model.predict(subject.feed(model, seq_len), verbose=0)
        assert _finite(out)


#: One learned position table, one ``TextDecoder`` table, one RoPE decoder.
MID_FIT = ("bert", "gpt2", "qwen3")


@pytest.mark.parametrize("name", MID_FIT)
def test_a_refusal_caught_inside_fit_does_not_kill_the_fit(name):
    subject = next(s for s in SUBJECTS if s.name == name)
    tiny = TINY[name]
    model = tiny.cls()(**tiny.kwargs, include_top=True, task_config=CLF)
    limit = _limit(subject, model)
    refused_at = []

    class RefuseAnOverLongCallAtEpochEnd(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            # The model is built here (the first train step built it).
            with pytest.raises(ValueError, match=subject.limit_name):
                self.model(subject.feed(self.model, limit + OVER_BY))
            refused_at.append(epoch)

    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    history = model.fit(
        subject.feed(model, LEGAL_LENGTHS[0]),
        np.zeros((BATCH,), dtype="int32"),
        epochs=3,
        batch_size=BATCH,
        callbacks=[RefuseAnOverLongCallAtEpochEnd()],
        verbose=0,
    )
    assert refused_at == [0, 1, 2]
    assert len(history.history["loss"]) == 3
    assert np.isfinite(history.history["loss"]).all(), history.history
    for seq_len in LEGAL_LENGTHS:
        assert _finite(model.predict(subject.feed(model, seq_len), verbose=0))


@pytest.mark.parametrize("subject", SUBJECTS, ids=SUBJECT_IDS)
def test_the_declared_limit_itself_is_legal(subject):
    model = subject.factory()
    limit = _limit(subject, model)
    assert _finite(model.predict(subject.feed(model, limit), verbose=0))


@pytest.mark.parametrize("config", _JIT_CONFIGS)
@pytest.mark.parametrize("subject", SUBJECTS, ids=SUBJECT_IDS)
def test_a_relaxed_retrace_refuses_an_over_long_batch_and_stays_usable(
    monkeypatch, subject, config
):
    import tensorflow as tf

    expected_jit = _resolve_jit(monkeypatch, config)
    model = subject.factory()
    limit = _limit(subject, model)
    if config == "jit_compile_true":
        model.jit_compile = True
    for seq_len in _legal_lengths(subject, limit):
        assert _finite(model.predict(subject.feed(model, seq_len), verbose=0))
    assert model.jit_compile is expected_jit, (config, model.jit_compile)

    pattern = (
        rf"seq_len_exceeds_{subject.limit_name}_{limit}(?![0-9])"
        if expected_jit
        else subject.message_regex.format(limit=limit)
    )
    with pytest.raises(
        (ValueError, tf.errors.InvalidArgumentError), match=pattern
    ):
        model.predict(subject.feed(model, limit + OVER_BY), verbose=0)
    # And the model still serves a legal length afterwards.
    assert _finite(model.predict(subject.feed(model, limit), verbose=0))


@pytest.mark.parametrize("subject", SUBJECTS, ids=SUBJECT_IDS)
def test_the_model_reaches_the_shared_guard(monkeypatch, subject):
    """Both shared primitives are reached, however the model names them.

    The model calls ``check_sequence_length`` (which looks the primitives up in
    the guard module) and may also import a primitive by name for its
    ``build``; the counter wraps every binding that exists.
    """
    module = importlib.import_module(subject.module)
    assert any(
        hasattr(module, name)
        for name in (
            "check_sequence_length",
            "refuse_static_sequence_length",
            "gate_dynamic_sequence_length",
        )
    ), f"{subject.module} imports nothing from the shared guard"
    calls = {"refuse": 0, "gate": 0}

    def counted(key, real):
        def wrapper(*args, **kwargs):
            calls[key] += 1
            return real(*args, **kwargs)

        return wrapper

    for key, name in (
        ("refuse", "refuse_static_sequence_length"),
        ("gate", "gate_dynamic_sequence_length"),
    ):
        wrapper = counted(key, getattr(guard, name))
        monkeypatch.setattr(guard, name, wrapper)
        if hasattr(module, name):
            monkeypatch.setattr(module, name, wrapper)

    model = subject.factory()
    limit = _limit(subject, model)
    with pytest.raises(ValueError):
        model(subject.feed(model, limit + OVER_BY))
    assert calls["refuse"] >= 1, calls
    # Two lengths make Keras retrace `predict` with a relaxed (None) axis.
    for seq_len in _legal_lengths(subject, limit):
        model.predict(subject.feed(model, seq_len), verbose=0)
    if subject.single_length:
        with pytest.raises(Exception):
            model.predict(subject.feed(model, limit + OVER_BY), verbose=0)
    assert calls["gate"] >= 1, calls


# ---------------------------------------------------------------------


@dataclass(frozen=True)
class NoLimit:
    """A model with no position limit: ``lengths`` are all served, finite."""

    name: str
    factory: Callable[[], keras.Model]
    lengths: tuple


NO_LIMIT = [
    NoLimit("mamba", _tiny(
        _LANG + "mamba.mamba_v1", "Mamba", d_model=32, num_layers=1,
        d_state=8,
    ), (8, 32)),
    NoLimit("mamba2", _tiny(
        _LANG + "mamba.mamba_v2", "Mamba2", d_model=32, num_layers=1,
        d_state=16, headdim=8,
    ), (8, 32)),
    NoLimit("zamba2_without_attention", _tiny(
        _LANG + "zamba2.model", "Zamba2Model", hidden_size=32,
        layer_mapping=["m", "m"], num_mem_blocks=1, num_heads=2,
        max_seq_len=16, d_state=8, headdim=8,
    ), (8, 40)),
    NoLimit("hnet_attention_in_inner_stage_only",
            _hnet(["m1", ["T1"], "m1"]), (8, 40)),
    NoLimit("hnet_without_attention",
            _hnet(["m1", ["m1"], "m1"]), (8, 40)),
    # Two layers under the default global_attention_interval of 3: no global
    # layer, so no RoPE table and no limit (the guarded subject has three).
    NoLimit("modern_bert_without_global_layer", _tiny(
        _LANG + "modern_bert.model", "ModernBERT", hidden_size=32,
        num_layers=2, num_heads=2, intermediate_size=64,
        max_position_embeddings=16, local_attention_window_size=8,
    ), (8, 40)),
]


@pytest.mark.parametrize("jit_compile", (False, True), ids=("eager", "jit"))
@pytest.mark.parametrize("case", NO_LIMIT, ids=[c.name for c in NO_LIMIT])
def test_a_model_without_a_position_limit_serves_any_length(case, jit_compile):
    model = case.factory()
    model.jit_compile = jit_compile
    for seq_len in case.lengths:
        out = model.predict(_ids(seq_len), verbose=0)
        assert _finite(out)
        # Mamba returns a dict of tensors; some leaf is per input position.
        assert any(
            np.asarray(leaf).shape[:2] == (BATCH, seq_len)
            for leaf in keras.tree.flatten(out)
        ), seq_len
    assert model.jit_compile is jit_compile


# ---------------------------------------------------------------------


def _set_recorded_length(src: Path, dst: Path, old: int, new: int) -> None:
    """Copy the archive ``src`` to ``dst`` with its recorded length ``old`` -> ``new``.

    The recorded ``build_config.input_shape`` is a list for a model fed one
    id tensor (Zamba2, HNet, BLT) and a dict of lists for a dict-fed model;
    the last axis of every recorded shape is the sequence.
    """
    with zipfile.ZipFile(src) as source, zipfile.ZipFile(dst, "w") as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == "config.json":
                config = json.loads(data)
                recorded = config["build_config"]["input_shape"]
                shapes = (
                    recorded.values() if isinstance(recorded, dict) else [recorded]
                )
                for shape in shapes:
                    assert shape[-1] == old, recorded
                    shape[-1] = new
                data = json.dumps(config).encode()
            target.writestr(item, data)


#: Subjects with a variable shaped by the recorded length itself. FNet's
#: ``(S, S)`` DFT matrix is sized by the first length built, so an archive that
#: recorded ``limit + 1`` holds a ``(limit + 1)``-sized matrix; editing a legal
#: archive's config would fabricate one that no model ever wrote. Its archive is
#: built past the limit with the shared refusal bypassed (a pre-guard build).
_SIZED_BY_RECORDED_LENGTH = ("fnet",)


def _over_long_archive(subject, model, limit, tmp_path, monkeypatch) -> Path:
    """Build ``model`` and save it as an archive that recorded ``limit + OVER_BY``."""
    old_path = tmp_path / "old.keras"
    if subject.name in _SIZED_BY_RECORDED_LENGTH:
        with monkeypatch.context() as patch:
            patch.setattr(
                guard, "refuse_static_sequence_length", lambda *a, **k: None
            )
            model.build({
                key: (None, limit + OVER_BY)
                for key in subject.feed(model, limit)
            })
        model.save(old_path)
        return old_path
    model.predict(subject.feed(model, limit), verbose=0)
    legal_path = tmp_path / "legal.keras"
    model.save(legal_path)
    _set_recorded_length(legal_path, old_path, limit, limit + OVER_BY)
    return old_path


@pytest.mark.parametrize("subject", SUBJECTS, ids=SUBJECT_IDS)
def test_an_archive_with_an_over_long_recorded_shape_loads_every_weight(
    subject, tmp_path, caplog, monkeypatch
):
    model = subject.factory()
    limit = _limit(subject, model)
    old_path = _over_long_archive(subject, model, limit, tmp_path, monkeypatch)
    expected = {w.path: keras.ops.convert_to_numpy(w) for w in model.weights}

    with caplog.at_level(logging.WARNING):
        loaded = keras.saving.load_model(old_path)

    got = {w.path: keras.ops.convert_to_numpy(w) for w in loaded.weights}
    assert sorted(got) == sorted(expected), (len(got), len(expected))
    unequal = [p for p in expected if not np.array_equal(got[p], expected[p])]
    assert not unequal, f"{len(unequal)}/{len(expected)} variables not restored"
    pattern = subject.message_regex.format(limit=limit)
    warned = [
        r.getMessage() for r in caplog.records
        if r.levelno >= logging.WARNING and re.search(pattern, r.getMessage())
    ]
    assert warned, "an over-long recorded shape must log a warning"
    # The load-only escape hatch is closed again, on the model and every layer.
    still_open = [
        layer.name for layer in loaded._flatten_layers()
        if getattr(layer, "_loading_recorded_build_shape", False)
    ]
    assert not still_open, still_open
    with pytest.raises(ValueError, match=subject.limit_name):
        loaded(subject.feed(loaded, limit + OVER_BY))


def test_blt_warns_and_loads_an_archive_with_an_over_long_recorded_shape(
    tmp_path, caplog
):
    subject = next(s for s in SUBJECTS if s.name == "blt")
    model = subject.factory()
    limit = _limit(subject, model)
    legal = subject.feed(model, limit)
    expected = model.predict(legal, verbose=0)
    legal_path = tmp_path / "legal.keras"
    model.save(legal_path)

    # Rewrite the archive so its recorded build shape is over-long.
    old_path = tmp_path / "old.keras"
    _set_recorded_length(legal_path, old_path, limit, limit + OVER_BY)

    with caplog.at_level(logging.WARNING):
        loaded = keras.saving.load_model(old_path)
    warned = [
        r.getMessage() for r in caplog.records
        if r.levelno >= logging.WARNING
        and f"exceeds max_sequence_length={limit}" in r.getMessage()
    ]
    assert warned, "an over-long recorded shape must log a warning"
    np.testing.assert_array_equal(loaded.predict(legal, verbose=0), expected)
    # The model still refuses an over-long input at call time.
    with pytest.raises(ValueError, match="max_sequence_length"):
        loaded(subject.feed(loaded, limit + OVER_BY))
    # And the strict path is unchanged: a fresh model's own build refuses.
    with pytest.raises(ValueError, match="max_sequence_length"):
        subject.factory().build((None, limit + OVER_BY))
