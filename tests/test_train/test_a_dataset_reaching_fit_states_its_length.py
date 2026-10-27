"""Census: every ``tf.data`` source of unknown length under ``src/`` is classified.

The rule (D-265 of plan-2026-10-14T062616-81938e81)
---------------------------------------------------
Keras 3.8 ``fit``/``evaluate`` over a dataset of UNKNOWN cardinality fails in
two silent ways (MEASURED, D-258 and D-265):

* with no ``steps_per_epoch``/``validation_steps``, the epoch end is found by one
  more call whose data-free sample-count update still runs when the step is not
  XLA-compiled and the batch axis is static, so every logged loss and ``Mean``
  metric is ``n / (n + 1)`` of the mean over ``n`` batches;
* with ``steps_per_epoch`` equal to the dataset's length, ``fit`` keeps one
  iterator across epochs and every second epoch trains ZERO batches (loss 0.0).

So a dataset that reaches ``fit``/``evaluate`` states its length: a count
(``assert_cardinality``, or ``state_counted_length`` for a stream countable only
by a pass) or an infinite stream (``assert_cardinality(INFINITE)`` or a bare
``repeat()``) driven by ``steps_per_epoch``. Where the length is truly
unknowable the source is EXEMPT here, with the reason its consumers are safe.

What this file checks
---------------------
1. AST, over ``src/``: every ``from_generator`` call sits in a function listed
   in :data:`STATED` (the function applies a stating call, checked) or in
   :data:`EXEMPT` (a reason, and the consumers whose ``fit`` calls must pass
   ``steps_per_epoch``, checked), never both, and no entry is stale.
2. AST, over ``src/``: every other length-losing ``tf.data`` op (``filter``,
   ``flat_map``, ``interleave``, ``unbatch``, ``ignore_errors``,
   ``sample_from_datasets``, ``TFRecordDataset``) is classified in
   :data:`OTHER_LENGTH_LOSING_SITES`.
3. Runtime: the sources buildable offline at a tiny size are built and their
   cardinality is compared with the count one pass actually yields.
"""

import ast
import math
import pathlib
from typing import Dict, List, Tuple

import numpy as np
import pytest
import tensorflow as tf

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"

UNKNOWN = int(tf.data.UNKNOWN_CARDINALITY)
INFINITE = int(tf.data.INFINITE_CARDINALITY)

#: ``from_generator`` sites whose function states the length. Value: how, and
#: so which call the function must make (:data:`REQUIRED_CALL`).
STATED: Dict[str, str] = {
    # --- stated before this step (D-022 of 77d4a04e, D-258, D-262, nano_vlm) ---
    "src/train/bit_diffusion/synthetic_data.py::build_bridge_dataset": "count",
    "src/train/dit/synthetic_data.py::build_dit_dataset": "count",
    "src/train/nano_vlm/train_nano_vlm.py::sequence_as_dataset": "count",
    "src/train/sam/data.py::build_sam_dataset": "count",
    "src/train/sam2/data.py::build_sam2_video_dataset": "count",
    "src/train/sam3/data.py::build_sam3_dataset": "count",
    # --- infinite by repeat(), unchanged; the one consumer passes steps ---
    "src/dl_techniques/datasets/pusht_hdf5.py::synthetic_lewm_dataset": "repeat",
    "src/dl_techniques/datasets/pusht_hdf5.py::PushTHDF5Dataset.as_tf_dataset": "repeat",
    "src/dl_techniques/datasets/synthetic_drone_video.py::synthetic_drone_video_dataset": "repeat",
    # --- step 12.10: a computable count ---
    "src/dl_techniques/datasets/graphs/fraud.py::build_fraud_subgraph_dataset": "count",
    "src/dl_techniques/datasets/graphs/tudataset.py::build_tudataset_graph_dataset": "count",
    "src/dl_techniques/datasets/nlp.py::_hf_to_tf_dataset._build_single_shard": "count",
    "src/dl_techniques/datasets/time_series/pipeline.py::make_tf_dataset": "count",
    "src/dl_techniques/datasets/time_series/pipeline.py::make_tf_dataset_from_arrays": "count",
    "src/dl_techniques/datasets/vision/coco.py::create_dummy_coco_dataset": "count",
    "src/train/bfunet/common.py::create_self_iterate_dataset": "count",
    "src/train/common/megadepth.py::pair_labeled_unlabeled": "count",
    # --- step 12.10: a packed stream, counted by one pass when finite ---
    "src/dl_techniques/datasets/byte_lm.py::build_byte_clm_dataset": "counted",
    "src/train/common/nlp.py::preprocess_clm_packed_dataset": "counted",
    # --- step 12.10: a never-ending generator states INFINITE ---
    "src/dl_techniques/datasets/bdd100k_video.py::bdd100k_video_dataset": "infinite",
    "src/train/common/timeseries.py::WindowedTimeSeriesProcessor.prepare_datasets": "infinite",
    "src/train/deepsort/train_deepsort.py::create_pk_dataset": "infinite",
    "src/train/sd3_mmdit/train_sd3_mmdit.py::make_synthetic_dataset": "infinite",
    "src/train/superpoint/train_magicpoint.py::create_dataset": "infinite",
    "src/train/superpoint/train_superpoint.py::create_dataset": "infinite",
}

#: ``from_generator`` sites whose length is unknowable. Value:
#: ``(reason, consumers)``; every ``fit`` call in each consumer function must
#: pass ``steps_per_epoch`` (checked by :func:`test_an_exempt_sources_consumers_pass_steps`).
EXEMPT: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "src/dl_techniques/datasets/nlp.py::load_hf_text_dataset": (
        "streaming=True: a Hub stream filtered by min_length has no length; the "
        "one src caller (gpt2/finetune.py) passes streaming=False, whose path "
        "goes through _hf_to_tf_dataset and states its row count",
        (),
    ),
    "src/dl_techniques/datasets/sut.py::OptimizedSUTDataset.create_tf_dataset": (
        "flat_map over sampled patches (an annotation can yield fewer); the one "
        "consumer repeats the train split under steps_per_epoch and evaluates the "
        "val split whole with a DYNAMIC batch axis (drop_remainder=False), whose "
        "sample count is data-dependent, so its logged mean is true",
        ("src/train/yolo12/train_multitask.py::train_model",),
    ),
    "src/dl_techniques/datasets/universal_dataset_loader.py::UniversalDatasetLoader.to_tf_dataset": (
        "a streaming Hub source, filter_fn and skip_errors make the count unknowable; "
        "no src consumer; the docstring states the steps_per_epoch contract",
        (),
    ),
    "src/dl_techniques/datasets/universal_dataset_loader.py::UniversalDatasetLoader.to_tf_dataset_tuple": (
        "as to_tf_dataset; its example fits the repeated stream with steps_per_epoch",
        (),
    ),
    "src/train/bert/wikipedia/pretrain.py::create_tf_dataset": (
        "a streaming Wikipedia + BookCorpus interleave; fit passes steps_per_epoch",
        ("src/train/bert/wikipedia/pretrain.py::main",),
    ),
    "src/train/bert/wikipedia/pretrain_english.py::create_tf_dataset": (
        "a streaming Wikipedia interleave; fit passes steps_per_epoch",
        ("src/train/bert/wikipedia/pretrain_english.py::main",),
    ),
    "src/train/dasiamrpn/train_dasiamrpn.py::create_pair_dataset": (
        "train branch repeat() under steps_per_epoch; val branch a DYNAMIC batch "
        "axis with validation_steps <= its length (required for COCO)",
        ("src/train/dasiamrpn/train_dasiamrpn.py::train_dasiamrpn",),
    ),
    "src/train/mambalct/train_mambalct.py::create_clip_dataset": (
        "as dasiamrpn's create_pair_dataset",
        ("src/train/mambalct/train_mambalct.py::train_mambalct",),
    ),
    "src/train/siamfc/train_siamfc.py::create_pair_dataset": (
        "as dasiamrpn's create_pair_dataset",
        ("src/train/siamfc/train_siamfc.py::train_siamfc",),
    ),
}

#: Every other length-losing op under ``src/``, with why its consumer is safe.
OTHER_LENGTH_LOSING_SITES: Dict[str, str] = {
    "src/dl_techniques/analyzer/utils.py::DataSampler._sample_tf_dataset::unbatch":
        "analysis sampling (take n); never reaches fit",
    "src/dl_techniques/datasets/nlp.py::load_wikipedia_train_val::filter":
        "Hugging Face Dataset.filter (Arrow); the row count is known after it",
    "src/dl_techniques/datasets/nlp.py::load_hf_text_dataset::filter":
        "Hugging Face Dataset.filter (Arrow); the row count is known after it",
    "src/dl_techniques/datasets/nlp.py::_hf_to_tf_dataset::sample_from_datasets":
        "every shard drains; the mix states len(hf_dataset)",
    "src/dl_techniques/datasets/sut.py::OptimizedSUTDataset.create_tf_dataset::flat_map":
        "see EXEMPT for the same function",
    "src/dl_techniques/datasets/vision/coco.py::COCODatasetBuilder.create_datasets::filter":
        "train repeat() under steps_per_epoch; val padded_batch with a DYNAMIC "
        "batch axis and validation_steps=50 (train_coco.py)",
    "src/train/bfunet/common.py::create_dataset::filter":
        "after repeat(): an infinite stream; fit passes steps_per_epoch and validation_steps",
    "src/train/bfunet/common.py::create_dataset::flat_map":
        "after repeat(): an infinite stream; fit passes steps_per_epoch and validation_steps",
    "src/train/common/nlp.py::prepare_data_for_analyzer::unbatch":
        "analyzer input preparation (take n); never reaches fit",
    "src/train/common/tfrecord.py::read_tfrecord_dataset::interleave":
        "no src consumer; the docstring states the steps / assert_cardinality contract",
    "src/train/common/tfrecord.py::read_tfrecord_dataset::TFRecordDataset":
        "no src consumer; as interleave above",
    "src/train/doc_res/common.py::create_dataset::filter":
        "after repeat(): an infinite stream; fit passes steps_per_epoch and validation_steps",
    "src/train/doc_res/common.py::create_dataset::flat_map":
        "after repeat(): an infinite stream; fit passes steps_per_epoch and validation_steps",
    "src/train/levjepa/train_levjepa.py::_build_dataset::unbatch":
        "of an infinite clip stream; fit passes steps_per_epoch",
    "src/train/lightglue/data.py::make_pair_dataset::ignore_errors":
        "the train split only (repeat=True, steps_per_epoch); val uses ignore_errors=False",
}

LENGTH_LOSING_OPS = {
    "filter", "flat_map", "interleave", "unbatch", "ignore_errors",
    "sample_from_datasets", "TFRecordDataset",
}
#: The call each kind of statement needs in the site's function.
REQUIRED_CALL = {
    "count": "assert_cardinality",
    "infinite": "assert_cardinality(INFINITE_CARDINALITY)",
    "counted": "state_counted_length",
    "repeat": "repeat()",
}


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------
def _call_name(node: ast.Call):
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _functions_and_calls(tree: ast.AST):
    """Yield ``(qualname, function_node_or_None, call)`` for every call."""
    def visit(node, stack, function):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield from visit(child, stack + [child.name], child)
            elif isinstance(child, ast.ClassDef):
                yield from visit(child, stack + [child.name], function)
            else:
                if isinstance(child, ast.Call):
                    yield ".".join(stack) or "<module>", function, child
                yield from visit(child, stack, function)
    yield from visit(tree, [], None)


def _source_index() -> Dict[str, List[Tuple[str, ast.AST, ast.Call]]]:
    index = {}
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        index[rel] = list(_functions_and_calls(ast.parse(path.read_text())))
    return index


@pytest.fixture(scope="module")
def source_index():
    return _source_index()


def _sites(index, names) -> Dict[str, List[Tuple[ast.AST, ast.Call]]]:
    found: Dict[str, List[Tuple[ast.AST, ast.Call]]] = {}
    for rel, calls in index.items():
        for qualname, function, call in calls:
            if _call_name(call) in names:
                found.setdefault(f"{rel}::{qualname}", []).append((function, call))
    return found


def _function_node(index, key: str) -> ast.AST:
    rel, qualname = key.split("::")
    for name, function, _ in index[rel]:
        if name == qualname and function is not None:
            return function
    raise AssertionError(f"no function {key}")


def _states_its_length(function: ast.AST, kind: str) -> bool:
    """Does ``function`` make the call a ``kind`` statement needs?"""
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if kind == "count" and name == "assert_cardinality":
            return True
        if kind == "infinite" and name == "assert_cardinality" and any(
                isinstance(arg, ast.Attribute) and arg.attr == "INFINITE_CARDINALITY"
                for arg in node.args):
            return True
        if kind == "counted" and name == "state_counted_length":
            return True
        if kind == "repeat" and name == "repeat" and not node.args and not node.keywords:
            return True
    return False


# ---------------------------------------------------------------------------
# 1. from_generator sites
# ---------------------------------------------------------------------------
def test_every_from_generator_site_is_classified(source_index):
    discovered = set(_sites(source_index, {"from_generator"}))
    classified = set(STATED) | set(EXEMPT)
    assert sorted(discovered - classified) == [], (
        "from_generator sites with no classification: state the length "
        "(assert_cardinality / state_counted_length / INFINITE) and add the "
        "site to STATED, or add it to EXEMPT with the reason it is unknowable "
        "and the consumers that pass steps_per_epoch (D-265)"
    )
    assert sorted(classified - discovered) == [], "stale entries: no such site"


def test_no_site_is_both_stated_and_exempt():
    assert sorted(set(STATED) & set(EXEMPT)) == []


@pytest.mark.parametrize("key", sorted(STATED))
def test_a_stated_site_applies_a_stating_call(source_index, key):
    kind = STATED[key]
    states = _states_its_length(_function_node(source_index, key), kind)
    assert states, (
        f"{key} is listed as stating its length ({kind}) but makes no "
        f"{REQUIRED_CALL[kind]} call"
    )


def _consumer_fit_calls(index, consumer: str) -> List[ast.Call]:
    function = _function_node(index, consumer)
    return [node for node in ast.walk(function)
            if isinstance(node, ast.Call) and _call_name(node) == "fit"]


@pytest.mark.parametrize(
    "key,consumer",
    sorted((key, consumer) for key, (_, consumers) in EXEMPT.items()
           for consumer in consumers),
)
def test_an_exempt_sources_consumers_pass_steps(source_index, key, consumer):
    calls = _consumer_fit_calls(source_index, consumer)
    assert calls, f"{consumer} (consumer of {key}) has no fit call"
    for call in calls:
        keywords = {kw.arg: kw.value for kw in call.keywords}
        value = keywords.get("steps_per_epoch")
        assert value is not None and not (
            isinstance(value, ast.Constant) and value.value is None), (
            f"{consumer} fits {key}'s unknown-length stream without "
            f"steps_per_epoch (line {call.lineno})"
        )


# ---------------------------------------------------------------------------
# 2. the other length-losing ops
# ---------------------------------------------------------------------------
def test_every_other_length_losing_op_is_classified(source_index):
    discovered = {
        f"{site}::{_call_name(call)}"
        for site, entries in _sites(source_index, LENGTH_LOSING_OPS).items()
        for _, call in entries
    }
    assert sorted(discovered - set(OTHER_LENGTH_LOSING_SITES)) == [], (
        "a new length-losing tf.data op: classify it (D-265)"
    )
    assert sorted(set(OTHER_LENGTH_LOSING_SITES) - discovered) == [], "stale entries"


# ---------------------------------------------------------------------------
# 3. runtime: the stated cardinality is the count a pass yields
# ---------------------------------------------------------------------------
def _passes_to(dataset: tf.data.Dataset) -> int:
    return sum(1 for _ in dataset)


def _texts(count: int) -> tf.data.Dataset:
    words = ["alpha beta gamma delta", "epsilon zeta", "eta theta iota kappa lambda"]
    return tf.data.Dataset.from_tensor_slices(
        [" ".join(words[i % 3] for _ in range(i % 4 + 1)) for i in range(count)])


def test_the_byte_stream_states_its_counted_length():
    from dl_techniques.datasets.byte_lm import build_byte_clm_dataset

    finite = build_byte_clm_dataset(_texts(9), seq_len=8, batch_size=2, shuffle_buffer=1)
    assert int(finite.cardinality()) == _passes_to(finite) > 0
    infinite = build_byte_clm_dataset(_texts(9), seq_len=8, batch_size=2, repeat=True)
    assert int(infinite.cardinality()) == INFINITE


def test_the_packed_token_stream_states_its_counted_length():
    from train.common.nlp import preprocess_clm_packed_dataset

    finite = preprocess_clm_packed_dataset(
        _texts(12), encoding_name="gpt2", chunk_length=6, batch_size=2,
        eot_token_id=50256, shuffle_buffer=1)
    assert int(finite.cardinality()) == _passes_to(finite) > 0
    infinite = preprocess_clm_packed_dataset(
        _texts(12), encoding_name="gpt2", chunk_length=6, batch_size=2,
        eot_token_id=50256, repeat=True)
    assert int(infinite.cardinality()) == INFINITE


def test_the_fraud_subgraphs_state_their_batch_count():
    from dl_techniques.datasets.graphs import build_fraud_subgraph_dataset

    for split in ("train", "val"):
        dataset, meta = build_fraud_subgraph_dataset(
            split=split, batch_size=16, max_nodes=8, synthetic=True,
            n_synth_nodes=120, synth_num_features=4, seed=0)
        expected = math.ceil(meta["split_sizes"][split] / 16)
        assert int(dataset.cardinality()) == expected == _passes_to(dataset)


def test_the_hugging_face_text_source_states_its_row_count():
    import datasets
    from dl_techniques.datasets.nlp import _hf_to_tf_dataset

    table = datasets.Dataset.from_dict({"text": [f"row {i}" for i in range(11)]})
    for shards in (1, 3):
        dataset = _hf_to_tf_dataset(table, shuffle=True, seed=0, num_shards=shards)
        assert int(dataset.cardinality()) == 11 == _passes_to(dataset)


def test_the_windowed_time_series_states_its_window_count():
    import pandas as pd
    from dl_techniques.datasets.time_series.pipeline import make_tf_dataset

    frame = pd.DataFrame({
        "unique_id": ["a"] * 9 + ["b"] * 4 + ["c"] * 2,
        "ds": list(range(9)) + list(range(4)) + list(range(2)),
        "y": np.arange(15, dtype="float32"),
    })
    dataset = make_tf_dataset(frame, window_size=2, horizon=1, batch_size=2, shuffle=False)
    # windows: a 9-3+1 = 7, b 4-3+1 = 2, c none -> 9 windows -> 5 batches of 2
    assert int(dataset.cardinality()) == 5 == _passes_to(dataset)


def test_the_dummy_coco_set_states_its_sample_count():
    from dl_techniques.datasets.vision.coco import create_dummy_coco_dataset

    dataset = create_dummy_coco_dataset(num_samples=3, img_size=32, max_boxes=2)
    assert int(dataset.cardinality()) == 3 == _passes_to(dataset)


class _Batches:
    """A ``PyDataset``-shaped list of fixed batches (``len`` and indexing)."""

    def __init__(self, count: int, shape: Tuple[int, ...], labelled: bool):
        self._count, self._shape, self._labelled = count, shape, labelled

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int):
        rgb = np.full(self._shape + (3,), float(index), np.float32)
        if not self._labelled:
            return rgb
        return rgb, np.full(self._shape + (2,), float(index), np.float32)


def test_the_megadepth_pairs_state_the_labelled_count():
    from train.common.megadepth import pair_labeled_unlabeled

    paired = pair_labeled_unlabeled(
        _Batches(3, (2, 4, 4), True), _Batches(2, (2, 4, 4), False),
        patch_size=4, batch_size=2)
    assert int(paired.cardinality()) == 3 == _passes_to(paired)
