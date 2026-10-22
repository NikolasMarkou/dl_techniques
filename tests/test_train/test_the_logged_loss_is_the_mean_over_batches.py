"""The datasets step 12.10 found reaching ``fit`` at unknown length now state it (D-265).

Each class is one defect found by the census
(``tests/test_train/test_a_dataset_reaching_fit_states_its_length.py``), guarded
by the number it got wrong:

* ``TestTheMegaDepthPairsTrainEveryEpoch``: ``pair_labeled_unlabeled`` zipped two
  unknown-length generators and the trainer passed ``steps_per_epoch =
  len(train_ds)``: ``fit`` kept one iterator across epochs, so every second
  epoch trained ZERO batches and logged a loss of 0.0.
* ``TestTheSelfIteratePoolIsMeanedWhole``: bfunet's self-iterate pool is a finite
  generator with a static batch axis fitted with ``steps_per_epoch=None`` (D-005
  of plan_2026-06-20_88705c63): under ``jit_compile=False`` (its mixed-precision
  and symmetry-penalty modes) every logged loss was ``n / (n + 1)`` of the mean.
* ``TestThePackedTokenStreamIsMeanedWhole``: the packed causal-LM stream (every
  CLM trainer's validation split; gpt2 / wave_field / cliffordnet_nlp's train
  split too) had the same unknown length and static batch axis; its logged loss
  was ``n / (n + 1)`` of the mean whenever the step ran without XLA.

The ``n / (n + 1)`` mechanism needs a NON-XLA step (MEASURED, D-265): under
``jit_compile=True`` the metric updates live inside the compiled step, which
never runs when the iterator is exhausted. The two value tests therefore compile
with ``jit_compile=False``, a mode the affected trainers run in (bfunet sets it
for mixed precision and its symmetry penalty; ``"auto"`` resolves to it on a
CPU-only machine).
* ``TestTheClmPretrainersDriveAnEpochBySteps``: those three trainers fitted the
  stream with no ``steps_per_epoch``; the Wikipedia branch is now infinite and
  the epoch is ``steps_per_epoch``, the TFDS branch's steps are its counted length.
* ``TestTheGpt2FinetuneHoldsOutAValidationSplit``: the HF text source returned
  cardinality -2, so the "no val split" fallback skipped the split and failed on
  an unbound name.
* ``TestTheGraphTrainersUseTheTrueHorizon``: both Graph Energy Transformer
  trainers guessed the LR horizon (``synth_nodes // batch``, a per-dataset table)
  because the dataset's batch count was unknown; it is the stated count now.
"""

import math
import types
import warnings
from typing import List

import keras
import numpy as np
import pytest
import tensorflow as tf

INFINITE = int(tf.data.INFINITE_CARDINALITY)


class _BatchCounter(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.per_epoch: List[int] = []

    def on_epoch_begin(self, epoch, logs=None):
        self.per_epoch.append(0)

    def on_train_batch_end(self, batch, logs=None):
        self.per_epoch[-1] += 1


def _quiet(call, *args, **kwargs):
    """Run a fit/evaluate with Keras' epoch-end warning silenced, so the VALUE fails."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call(*args, **kwargs)


# ---------------------------------------------------------------------------
class _Batches:
    """A ``PyDataset``-shaped list of fixed batches (``len`` and indexing)."""

    def __init__(self, count, shape, labelled):
        self._count, self._shape, self._labelled = count, shape, labelled

    def __len__(self):
        return self._count

    def __getitem__(self, index):
        rgb = np.full(self._shape + (3,), float(index + 1), np.float32)
        if not self._labelled:
            return rgb
        return rgb, np.full(self._shape + (2,), float(index), np.float32)


class TestTheMegaDepthPairsTrainEveryEpoch:

    def test_steps_equal_to_the_length_train_every_epoch(self):
        from train.common.megadepth import pair_labeled_unlabeled

        keras.utils.set_random_seed(0)
        paired = pair_labeled_unlabeled(
            _Batches(3, (2, 4, 4), True), _Batches(2, (2, 4, 4), False),
            patch_size=4, batch_size=2)
        labelled = keras.Input((4, 4, 3))
        unlabelled = keras.Input((4, 4, 3))
        output = keras.layers.Conv2D(2, 1)(keras.layers.Add()([labelled, unlabelled]))
        model = keras.Model([labelled, unlabelled], output)
        model.compile(optimizer=keras.optimizers.SGD(0.0), loss="mse")
        counter = _BatchCounter()
        history = _quiet(model.fit, paired, epochs=4, steps_per_epoch=3,
                         verbose=0, callbacks=[counter])
        assert counter.per_epoch == [3, 3, 3, 3], (
            f"batches trained per epoch {counter.per_epoch}: an unstated length "
            f"under steps_per_epoch empties every second epoch"
        )
        losses = history.history["loss"]
        assert min(losses) > 0.0 and max(losses) == pytest.approx(min(losses), rel=1e-6)


# ---------------------------------------------------------------------------
def _hand_mean_loss(model, batches) -> float:
    """Mean over batches of the compiled loss, by hand (equal-size batches)."""
    values = [float(model.compute_loss(x=x, y=y, y_pred=model(x, training=False)))
              for x, y in batches]
    return float(np.mean(values))


class TestTheSelfIteratePoolIsMeanedWhole:

    @pytest.fixture
    def dataset(self):
        from train.bfunet.common import create_self_iterate_dataset

        config = types.SimpleNamespace(
            batch_size=2, steps_per_epoch=None, patch_size=4, channels=1,
            seed=3, patch_shuffle_buffer=1)
        rng = np.random.default_rng(0)
        clean = rng.random((6, 4, 4, 1)).astype("float32")
        noisy = (clean + 0.1).astype("float32")
        dataset, steps = create_self_iterate_dataset(clean, noisy, config)
        assert steps == 3
        return dataset

    def test_the_pool_states_its_batch_count(self, dataset):
        assert int(dataset.cardinality()) == 3

    def test_evaluate_and_fit_log_the_mean_over_the_pool(self, dataset):
        keras.utils.set_random_seed(0)
        model = keras.Sequential([keras.Input((4, 4, 1)), keras.layers.Conv2D(1, 1)])
        # jit_compile=False, as bfunet compiles under mixed precision or the
        # symmetry penalty: under XLA the extra call's count update never runs.
        model.compile(optimizer=keras.optimizers.SGD(0.0), loss="mse", jit_compile=False)
        expected = _hand_mean_loss(model, list(dataset))
        assert _quiet(model.evaluate, dataset, verbose=0) == pytest.approx(expected, rel=1e-5)
        fitted = _quiet(model.fit, dataset, epochs=2, verbose=0).history["loss"]
        assert fitted == pytest.approx([expected, expected], rel=1e-5)


# ---------------------------------------------------------------------------
def _texts(count: int) -> tf.data.Dataset:
    words = ["alpha beta gamma delta", "epsilon zeta", "eta theta iota kappa lambda"]
    return tf.data.Dataset.from_tensor_slices(
        [" ".join(words[i % 3] for _ in range(i % 4 + 1)) for i in range(count)])


class TestThePackedTokenStreamIsMeanedWhole:

    def test_evaluate_logs_the_mean_over_the_stream(self):
        from train.common.nlp import preprocess_clm_packed_dataset

        stream = preprocess_clm_packed_dataset(
            _texts(16), encoding_name="gpt2", chunk_length=6, batch_size=2,
            eot_token_id=50256, shuffle_buffer=1)
        batches = list(stream)
        assert len(batches) >= 3
        keras.utils.set_random_seed(0)
        model = keras.Sequential([
            keras.Input((5,), dtype="int32"),
            keras.layers.Embedding(50257, 4),
            keras.layers.Dense(50257),
        ])
        # jit_compile=False: the non-XLA step, where the extra call's count update
        # runs (a CPU-only machine's "auto", or any model that cannot jit).
        model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      jit_compile=False)
        expected = _hand_mean_loss(model, batches)
        assert _quiet(model.evaluate, stream, verbose=0) == pytest.approx(expected, rel=1e-5)
        assert int(stream.cardinality()) == len(batches)


# ---------------------------------------------------------------------------
def _tiny_wikipedia(*args, **kwargs):
    train = _texts(20)
    val = _texts(10)
    return train, val, 20, 10


class _Encoder:
    """The two attributes ``preprocess_clm_dataset`` reads off a tokenizer."""

    def __init__(self):
        import tiktoken
        self.tokenizer = tiktoken.get_encoding("gpt2")


class TestTheClmPretrainersDriveAnEpochBySteps:

    def test_the_wikipedia_branch_is_infinite_and_its_val_split_is_counted(self, monkeypatch):
        from train.common import clm_pretrain as cp

        monkeypatch.setattr(cp, "load_wikipedia_train_val", _tiny_wikipedia)
        config = cp.ClmPretrainConfig(max_seq_length=6, batch_size=2)
        train, val, articles = cp.load_hf_clm_datasets(config, _Encoder(), data_seed=0)
        assert articles == 20
        assert int(train.cardinality()) == INFINITE
        assert int(val.cardinality()) == sum(1 for _ in val) > 0

    def test_a_stated_train_length_is_the_epoch(self):
        from train.common import clm_pretrain as cp

        config = cp.ClmPretrainConfig(dataset_source="tfds", max_samples=100, batch_size=8)
        stated = tf.data.Dataset.range(5)
        assert cp.make_clm_steps_per_epoch(config, None, stated) == 5
        config.steps_per_epoch = 9
        assert cp.make_clm_steps_per_epoch(config, None, stated) == 9

    @pytest.mark.parametrize("module_name,train_name,kwargs", [
        ("train.gpt2.pretrain", "train_gpt2", {}),
        ("train.wave_field.pretrain", "train_wave_field_llm", {}),
        ("train.cliffordnet.train_cliffordnet_nlp", "train_cliffordnet_nlp",
         {"model_loader": "create_model"}),
    ])
    def test_fit_gets_the_epoch_as_steps(self, monkeypatch, tmp_path, module_name,
                                         train_name, kwargs):
        import importlib
        from ._trainer_run_harness import run_trainer

        module = importlib.import_module(module_name)
        results_dir = tmp_path / "run"
        results_dir.mkdir()
        config = module.TrainingConfig(save_dir=str(tmp_path / "save"))
        _, model = run_trainer(monkeypatch, module, getattr(module, train_name),
                               config, results_dir, **kwargs)
        assert model.fit_calls == 1
        assert model.fit_kwargs.get("steps_per_epoch") == 7, (
            f"{module_name} fits the packed stream with steps_per_epoch="
            f"{model.fit_kwargs.get('steps_per_epoch')!r}"
        )
        assert model.fit_args[0].repeated, "the train stream is not repeated"


# ---------------------------------------------------------------------------
class TestTheGpt2FinetuneHoldsOutAValidationSplit:

    def test_a_dataset_without_a_val_split_is_split(self, monkeypatch):
        import datasets as hf
        from dl_techniques.datasets import nlp as nlp_module
        from train.gpt2 import finetune as ft

        # Long enough to pass the config's min_text_length filter.
        table = hf.Dataset.from_dict({"text": [f"document number {i} " * 40 for i in range(20)]})

        def fake_load_dataset(path, name=None, split="train", **kwargs):
            if split != "train":
                raise ValueError(f"no split {split}")
            return table

        monkeypatch.setattr(nlp_module.datasets, "load_dataset", fake_load_dataset)
        config = ft.FinetuneConfig()
        config.val_fraction = 0.25
        train, val = ft._load_hf_train_val(config)
        assert sum(1 for _ in val) == 5
        assert sum(1 for _ in train) == 15


# ---------------------------------------------------------------------------
class _Captured(Exception):
    pass


class TestTheGraphTrainersUseTheTrueHorizon:

    @staticmethod
    def _capture(monkeypatch, module):
        seen = {}

        def build_optimizer(config, horizon_steps):
            seen["horizon"] = horizon_steps
            raise _Captured()

        monkeypatch.setattr(module, "build_optimizer", build_optimizer)
        monkeypatch.setattr(module, "setup_gpu", lambda *a, **k: None)
        return seen

    def test_the_anomaly_trainer(self, monkeypatch, tmp_path):
        from train.graph_energy_transformer import train_anomaly as module

        seen = self._capture(monkeypatch, module)
        config = module.TrainingConfig(
            dataset="synthetic", batch_size=16, max_nodes=8, synth_nodes=200,
            synth_features=4, embed_dim=8, num_heads=2, head_dim=4, hopfield_dim=8,
            mlp_hidden_dim=8, epochs=1, output_dir=str(tmp_path))
        train, meta = module.build_fraud_subgraph_dataset(
            name="synthetic", split="train", batch_size=16, max_nodes=8, synthetic=True,
            n_synth_nodes=200, synth_num_features=4, seed=config.seed)
        with pytest.raises(_Captured):
            module.train_anomaly(config)
        assert seen["horizon"] == math.ceil(meta["split_sizes"]["train"] / 16)

    def test_the_classification_trainer(self, monkeypatch, tmp_path):
        from dl_techniques.datasets.graphs.tudataset import DEFAULT_GRAPH_CACHE
        from train.graph_energy_transformer import train_classification as module

        import os
        if not os.path.isdir(os.path.join(DEFAULT_GRAPH_CACHE, "MUTAG")):
            pytest.skip("MUTAG is not cached on this machine")
        seen = self._capture(monkeypatch, module)
        config = module.TrainingConfig(dataset="mutag", batch_size=16, epochs=1,
                                       output_dir=str(tmp_path))
        _, meta = module.build_tudataset_graph_dataset(
            name="MUTAG", split="train", batch_size=16, k_pe=config.k_pe,
            max_nodes=config.max_nodes, add_self_loops=config.add_self_loops,
            seed=config.seed)
        with pytest.raises(_Captured):
            module.train_classification(config)
        assert seen["horizon"] == math.ceil(meta["split_sizes"]["train"] / 16)
