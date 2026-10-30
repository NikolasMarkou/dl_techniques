"""The two Phase B experiment workers and drivers, end to end on CPU.

Plan plan-2026-10-30T025213-f75ab4e9 steps 3 and 4. Exp1 compares four pooling heads on
vit_pico, exp2 compares MeanShift Attention against multi_head in a small ViT. The claims
pinned here are the ones a quiet mistake would break without any run failing:

* the four exp1 arms differ only in the pooling head, with equal parameter counts, and the
  mean and mode heads ignore the CLS position;
* every arm scores the very same corrupted images for the same seed and condition, the
  selection score reads the held-out slice of train and never the test split;
* all 17 exp2 configs build and run, every vMF config starts at ``bandwidth_init=0.3``, the
  parameter ratio is measured, and the learned bandwidths are read through the layer's own
  helper.

Workers run in this process on a tiny fake CIFAR (``tests.classification_fakes``); two tests
drive the real subprocess path with ``--cpu`` and the Keras cache. Everything is written
under ``tmp_path``; nothing here touches ``results/`` or a GPU.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.classification_fakes import use_tiny_raw
from train.common.classification_data import load_split_arrays
from train.mean_shift import common, corruption
from train.mean_shift import exp1_mode_pooling as exp1
from train.mean_shift import exp2_mean_shift_attention as exp2

REPO = Path(__file__).resolve().parents[3]
CIFAR_CACHE = Path.home() / ".keras" / "datasets" / "cifar-10-batches-py-target"
MAX_SAMPLES = 256
DEFAULT_MS = "blur_s3_per_head_gaussian"


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def expected_condition(x, name, seed):
    """The images a worker must score for ``name``, built from the two primitives.

    Deliberately not ``corruption.apply_condition``: a worker that bypassed or re-seeded
    that dispatcher must not make this reference follow it.
    """
    kind, parameter = corruption.CONDITIONS[name]
    if kind == "occlude":
        return corruption.occlude(x, parameter, seed, name)
    if kind == "noise":
        return corruption.add_noise(x, parameter, seed, name)
    return np.array(x, copy=True)


def run_worker(module, root, arm, config, *, seed=0, dataset="cifar10", epochs=1):
    cell_dir = Path(root) / f"{arm}__{config}__s{seed}"
    cell_dir.mkdir(parents=True)
    argv = ["--worker", "--cell-dir", str(cell_dir), "--arm", arm, "--config", config,
            "--seed", str(seed), "--dataset", dataset, "--epochs", str(epochs),
            "--max-samples", str(MAX_SAMPLES)]
    assert module.main(argv) == 0
    return json.loads((cell_dir / "result.json").read_text(encoding="utf-8"))


def dataset_images_sha(dataset):
    """The hash of the images of a batched ``tf.data`` dataset of ``(x, y)``, in iteration order."""
    return sha(np.concatenate([np.asarray(x) for x, _ in dataset.as_numpy_iterator()]))


class Recorder:
    """Wraps ``common.top1`` and ``Model.fit``.

    ``hashes`` keeps a hash of every array a worker scored; ``fit_validation`` keeps a hash of
    the images of the ``validation_data`` every ``fit`` call received, so a worker that
    monitors the wrong split during training is visible.
    """

    def __init__(self, monkeypatch):
        import keras
        self.hashes = []
        self.fit_validation = []
        real = common.top1
        real_fit = keras.Model.fit
        recorder = self

        def recording(model, x, y, batch_size=256):
            self.hashes.append(sha(x))
            return real(model, x, y, batch_size)

        def recording_fit(model, *args, **kwargs):
            recorder.fit_validation.append(dataset_images_sha(kwargs["validation_data"]))
            return real_fit(model, *args, **kwargs)

        monkeypatch.setattr(common, "top1", recording)
        monkeypatch.setattr(keras.Model, "fit", recording_fit)


@pytest.fixture(scope="module")
def fake_cifar():
    """A tiny offline CIFAR for the in-process workers (module scoped)."""
    patch = pytest.MonkeyPatch()
    use_tiny_raw(patch, n_train=400, n_test=300)
    yield
    patch.undo()


@pytest.fixture(scope="module")
def split_arrays(fake_cifar):
    return load_split_arrays("cifar10", seed=0, max_samples=MAX_SAMPLES)


# ---------------------------------------------------------------------------
# exp1: arms, heads, cells
# ---------------------------------------------------------------------------

ARM_NAMES = ("mean", "cls", "mode", "mode_s4")


def test_the_four_arms_are_the_planned_ones():
    assert tuple(exp1.ARMS) == ARM_NAMES


def test_the_four_arms_have_equal_parameter_counts():
    counts = {arm: exp1.build_model(arm, 10).count_params() for arm in ARM_NAMES}
    assert len(set(counts.values())) == 1, counts
    # vit_pico at patch 4 (sequence 65) plus Dense(10): 2,710,948 - Dense(100) + Dense(10)
    assert counts["mean"] == 2_693_578
    assert exp1.build_model("mean", 100).count_params() == 2_710_948


def test_the_backbone_sees_64_patches_not_4():
    model = exp1.build_model("mean", 10)
    backbone = next(layer for layer in model.layers if layer.name == "backbone")
    assert backbone.compute_output_shape((None, 32, 32, 3))[1] == 65


def test_each_arm_pools_with_its_planned_head():
    heads = {arm: exp1.pooling_layer(arm) for arm in ARM_NAMES}
    assert heads["mean"].strategy == ["mean"] and heads["mean"].exclude_positions == [0]
    assert heads["cls"].strategy == ["cls"]
    for arm, starts in (("mode", 1), ("mode_s4", 4)):
        mode = heads[arm]
        assert mode.num_steps == 5 and mode.bandwidth == "auto"
        assert mode.exclude_positions == [0] and mode.num_starts == starts


@pytest.mark.parametrize("arm", ["mean", "mode", "mode_s4"])
def test_the_mean_and_mode_heads_ignore_the_cls_position(arm):
    sequence = np.random.default_rng(0).normal(size=(3, 65, 16)).astype("float32")
    changed = sequence.copy()
    changed[:, 0, :] += 50.0
    head = exp1.pooling_layer(arm)
    assert np.array_equal(np.asarray(head(sequence)), np.asarray(head(changed)))


def test_the_cls_head_reads_the_cls_position():
    sequence = np.random.default_rng(1).normal(size=(3, 65, 16)).astype("float32")
    changed = sequence.copy()
    changed[:, 0, :] += 50.0
    head = exp1.pooling_layer("cls")
    assert np.array_equal(np.asarray(head(sequence))[:, :], sequence[:, 0, :])
    assert not np.array_equal(np.asarray(head(changed)), np.asarray(head(sequence)))


def test_the_exp1_cells_are_four_arms_by_three_seeds_seed_major():
    args = exp1.parse_driver_arguments(["--out-dir", "x", "--epochs", "7"])
    cells = exp1.build_cells(args)
    assert len(cells) == 12 and len({c.key for c in cells}) == 12
    assert [c.seed for c in cells] == [0] * 4 + [1] * 4 + [2] * 4
    assert [c.arm for c in cells[:4]] == list(ARM_NAMES)
    assert {c.experiment for c in cells} == {"exp1"} and {c.config for c in cells} == {"cifar10"}
    assert all("--epochs" in c.extra_args and "7" in c.extra_args for c in cells)


def test_the_exp1_driver_takes_a_dataset_arm_and_seed_subset():
    args = exp1.parse_driver_arguments(
        ["--out-dir", "x", "--epochs", "3", "--dataset", "cifar100", "--arms", "mode", "cls",
         "--seeds", "5", "--max-samples", "256"])
    cells = exp1.build_cells(args)
    assert [(c.arm, c.config, c.seed) for c in cells] == [
        ("mode", "cifar100", 5), ("cls", "cifar100", 5)]
    assert "--max-samples" in cells[0].extra_args


def test_an_unknown_exp1_arm_is_refused():
    with pytest.raises(SystemExit):
        exp1.parse_driver_arguments(["--out-dir", "x", "--epochs", "3", "--arms", "median"])


@pytest.fixture(scope="module")
def exp1_runs(fake_cifar, tmp_path_factory):
    """One in-process worker per arm, with the scored arrays recorded."""
    root = tmp_path_factory.mktemp("exp1_workers")
    patch = pytest.MonkeyPatch()
    recorders = {}
    results = {}
    for arm in ARM_NAMES:
        recorders[arm] = Recorder(patch)
        results[arm] = run_worker(exp1, root, arm, "cifar10")
        patch.undo()
    return results, recorders


@pytest.mark.parametrize("arm", ARM_NAMES)
def test_an_exp1_worker_writes_one_complete_result(arm, exp1_runs):
    result = exp1_runs[0][arm]
    metrics = result["metrics"]
    assert result["finite"] is True and result["epochs_run"] == 1
    assert result["seconds_per_epoch"] is not None and result["seconds_per_epoch"] > 0
    assert set(metrics["accuracy"]) == set(corruption.CONDITIONS)
    assert all(0.0 <= v <= 1.0 for v in metrics["accuracy"].values())
    assert 0.0 <= metrics["select_accuracy"] <= 1.0
    assert metrics["dtype_policy"] == "float32" and metrics["n_test"] == MAX_SAMPLES
    assert metrics["dataset"] == "cifar10" and metrics["arm"] == arm
    assert result["devices"] and all("GPU" not in d for d in result["devices"])


def test_the_exp1_arms_report_the_same_parameter_count(exp1_runs):
    counts = {arm: r["params"] for arm, r in exp1_runs[0].items()}
    assert len(set(counts.values())) == 1 and counts["mean"] == 2_693_578


def test_every_arm_scores_the_same_corrupted_images(exp1_runs, split_arrays):
    _, recorders = exp1_runs
    expected = {c: sha(expected_condition(split_arrays.x_test, c, 0))
                for c in corruption.CONDITIONS}
    selection = sha(split_arrays.x_select)
    for arm, recorder in recorders.items():
        assert sorted(recorder.hashes) == sorted([*expected.values(), selection]), arm


def test_the_scored_conditions_really_differ_from_the_clean_images(split_arrays):
    clean = sha(split_arrays.x_test)
    others = [sha(expected_condition(split_arrays.x_test, c, 0))
              for c in corruption.CONDITIONS if c != "clean"]
    assert clean not in others and len(set(others)) == len(others)


def test_the_exp1_selection_score_reads_the_selection_split_not_the_test_split(
        exp1_runs, split_arrays):
    _, recorders = exp1_runs
    for recorder in recorders.values():
        assert sha(split_arrays.x_select) in recorder.hashes
    assert len(split_arrays.x_select) != len(split_arrays.x_test)
    assert not np.array_equal(split_arrays.x_select[:5], split_arrays.x_test[:5])


def test_the_exp1_fit_validates_on_the_selection_split_not_the_test_split(
        exp1_runs, split_arrays):
    """The stored ``val_*`` history is the selection curve; a ``fit`` that monitored the test
    split would turn it into a test curve without any score changing."""
    _, recorders = exp1_runs
    selection, test = sha(split_arrays.x_select), sha(split_arrays.x_test)
    assert selection != test
    for arm, recorder in recorders.items():
        assert recorder.fit_validation == [selection], arm


# ---------------------------------------------------------------------------
# exp2: configs
# ---------------------------------------------------------------------------

def test_the_config_table_is_mha_plus_the_sixteen_grid_configs():
    ids = list(exp2.CONFIGS)
    assert ids[0] == "mha"
    assert len(ids) == 17 and len(set(ids)) == 17
    grid = [exp2.CONFIGS[i] for i in ids if i != "mha"]
    combos = {(c["attention_args"]["blurring"], c["attention_args"]["num_steps"],
               c["attention_args"]["bandwidth_mode"], c["attention_args"]["kernel"])
              for c in grid}
    assert combos == {(b, s, m, k) for b in (False, True) for s in (1, 3)
                      for m in ("per_head", "per_token") for k in ("gaussian", "vmf")}
    assert exp2.CONFIGS["mha"]["attention_type"] == "multi_head"
    assert all(c["attention_type"] == "mean_shift" for c in grid)
    assert DEFAULT_MS in exp2.CONFIGS


def test_every_vmf_config_starts_at_bandwidth_0_3_and_gaussian_ones_keep_the_default():
    for config_id, config in exp2.CONFIGS.items():
        args = config["attention_args"]
        if args.get("kernel") == "vmf":
            assert args["bandwidth_init"] == 0.3, config_id
        else:
            assert "bandwidth_init" not in args, config_id


@pytest.mark.parametrize("config_id", list(exp2.ALL_CONFIGS))
def test_every_config_builds_and_runs_a_forward_pass(config_id):
    model = exp2.build_model(config_id, 100)
    out = np.asarray(model(np.random.default_rng(0).random((2, 32, 32, 3), dtype="float32")))
    assert out.shape == (2, 100) and np.isfinite(out).all()
    layers = exp2.attention_layers(model)
    spec = exp2.ALL_CONFIGS[config_id]
    assert len(layers) == 6
    if spec["attention_type"] == "mean_shift":
        for layer in layers:
            for name, value in spec["attention_args"].items():
                assert getattr(layer, name) == value, (config_id, name)


# ---------------------------------------------------------------------------
# exp2: the output-projection initializer control (mha_oinit)
# ---------------------------------------------------------------------------

CONTROL = "mha_oinit"
#: Values compared per block: the first N of the larger kernel against the whole smaller one
#: (192x192 proj against the first 192*192 values of the 192x576 qkv).
PROJ_SIZE = 192 * 192


def test_the_control_is_outside_the_grid_and_the_baseline_is_exactly_as_run():
    assert CONTROL not in exp2.CONFIGS and len(exp2.CONFIGS) == 17
    assert list(exp2.CONTROL_CONFIGS) == [CONTROL]
    assert list(exp2.ALL_CONFIGS) == [*exp2.CONFIGS, CONTROL]
    # mha keeps the empty attention_args its stored rows were run with
    assert exp2.CONFIGS["mha"] == {"attention_type": "multi_head", "attention_args": {}}
    control = exp2.CONTROL_CONFIGS[CONTROL]
    assert control["attention_type"] == "multi_head"
    assert control["attention_args"] == {"output_kernel_initializer": "glorot_uniform"}


def kernel_pair(model, config_id):
    """Per block: (the first PROJ_SIZE values of the input kernel, the output kernel values)."""
    names = {"multi_head": ("qkv/kernel", "proj/kernel"),
             "mean_shift": ("projection/kernel", "output_projection/kernel")}
    first, second = names[exp2.ALL_CONFIGS[config_id]["attention_type"]]
    pairs = []
    for layer in exp2.attention_layers(model):
        kernels = {w.path.split("attention/", 1)[1].replace("cross_attention/", ""): np.asarray(w)
                   for w in layer.weights}
        pairs.append((kernels[first].ravel()[:PROJ_SIZE], kernels[second].ravel()))
    return pairs


def init_correlations(config_id):
    """The correlations the worker would start from: ``set_seeds`` then ``build_model``."""
    from train.common import set_seeds
    set_seeds(0)
    pairs = kernel_pair(exp2.build_model(config_id, 100), config_id)
    assert len(pairs) == 6 and all(a.size == b.size == PROJ_SIZE for a, b in pairs)
    return [float(np.corrcoef(a, b)[0, 1]) for a, b in pairs]


def test_mha_replays_one_draw_for_qkv_and_proj_the_documented_confound():
    """A property of the layer's shared initializer instance, pinned so it stays named, not a goal."""
    assert min(init_correlations("mha")) > 0.999


@pytest.mark.parametrize("config_id", [CONTROL, DEFAULT_MS, "noblur_s3_per_token_vmf"])
def test_the_control_and_the_mean_shift_configs_start_with_independent_projections(config_id):
    correlations = init_correlations(config_id)
    assert max(abs(c) for c in correlations) < 0.1, correlations


def test_the_control_has_the_parameter_count_of_mha():
    counts = {c: exp2.build_model(c, 100).count_params() for c in ("mha", CONTROL)}
    assert counts == {"mha": 2_710_564, CONTROL: 2_710_564}


def test_the_parameter_ratio_of_mean_shift_to_multi_head_is_measured(record_property):
    reference = exp2.build_model("mha", 100).count_params()
    shifted = exp2.build_model(DEFAULT_MS, 100).count_params()
    ratio = shifted / reference
    record_property("params_multi_head", reference)
    record_property("params_mean_shift", shifted)
    record_property("param_ratio", round(ratio, 4))
    assert (reference, shifted) == (2_710_564, 2_265_910)
    assert 0.835 < ratio < 0.837


# ---------------------------------------------------------------------------
# exp2: learned bandwidth extraction
# ---------------------------------------------------------------------------

def softplus(v):
    return np.log1p(np.exp(v))


def toy_layer(**kwargs):
    from dl_techniques.layers.attention.mean_shift_attention import MeanShiftAttention
    options = dict(dim=12, num_heads=3, num_steps=1)
    options.update(kwargs)
    layer = MeanShiftAttention(**options)
    layer.build((None, 5, 12))
    return layer


def test_per_head_bandwidths_are_softplus_of_the_raw_weight_plus_the_minimum():
    layer = toy_layer(bandwidth_mode="per_head")
    raw = np.array([-0.5, 0.3, 1.2], dtype="float32")
    layer.raw_bandwidth.assign(raw)
    tokens = np.random.default_rng(0).normal(size=(2, 5, 12)).astype("float32")
    (recorded,) = exp2.record_bandwidths([layer], lambda: layer(tokens))
    expected = layer.min_bandwidth + softplus(raw.astype("float64"))
    assert recorded.shape == (1, 3, 1, 1)
    assert np.allclose(recorded.reshape(3), expected, rtol=1e-6)
    summary = exp2.summarize_bandwidth(recorded, layer)
    assert summary["per_head_mean"] == pytest.approx(expected.tolist(), rel=1e-6)
    assert summary["min"] == pytest.approx(expected.min(), rel=1e-6)
    assert summary["max"] == pytest.approx(expected.max(), rel=1e-6)
    assert summary["inside_clip_range"] is True


def test_per_token_bandwidths_go_through_the_layers_own_helper():
    layer = toy_layer(bandwidth_mode="per_token", bandwidth_init=0.7)
    rng = np.random.default_rng(3)
    kernel = rng.normal(scale=0.2, size=(12, 3)).astype("float32")
    bias = rng.normal(scale=0.3, size=(3,)).astype("float32")
    layer.bandwidth_projection.kernel.assign(kernel)
    layer.bandwidth_projection.bias.assign(bias)
    tokens = rng.normal(size=(4, 5, 12)).astype("float32")
    (recorded,) = exp2.record_bandwidths([layer], lambda: layer(tokens))
    projected = np.asarray(layer.projection(tokens))
    raw = projected.astype("float64") @ kernel.astype("float64") + bias
    expected = (layer.min_bandwidth + softplus(raw)).transpose(0, 2, 1)[..., None]
    assert recorded.shape == (4, 3, 5, 1)
    assert np.allclose(recorded, expected, rtol=1e-4)
    summary = exp2.summarize_bandwidth(recorded, layer)
    assert summary["mean"] == pytest.approx(float(expected.mean()), rel=1e-4)
    assert summary["min"] == pytest.approx(float(expected.min()), rel=1e-4)
    assert summary["max"] == pytest.approx(float(expected.max()), rel=1e-4)


def test_a_bandwidth_on_the_clip_bound_is_flagged():
    layer = toy_layer(bandwidth_mode="per_head")
    layer.raw_bandwidth.assign(np.array([0.0, 0.0, 200.0], dtype="float32"))
    tokens = np.zeros((1, 5, 12), dtype="float32")
    (recorded,) = exp2.record_bandwidths([layer], lambda: layer(tokens))
    assert exp2.summarize_bandwidth(recorded, layer)["inside_clip_range"] is False


def test_recording_leaves_the_layer_unpatched():
    layer = toy_layer(bandwidth_mode="per_head")
    tokens = np.zeros((1, 5, 12), dtype="float32")
    exp2.record_bandwidths([layer], lambda: layer(tokens))
    assert "_bandwidth" not in layer.__dict__


def test_a_multi_head_model_has_no_bandwidth_to_extract():
    model = exp2.build_model("mha", 10)
    block = exp2.learned_bandwidths(model, np.zeros((2, 32, 32, 3), dtype="float32"))
    assert block["status"] == "not applicable: the model has no MeanShiftAttention layer"
    assert block["layers"] == []


# ---------------------------------------------------------------------------
# exp2: cells
# ---------------------------------------------------------------------------

def parse2(*argv):
    return exp2.parse_driver_arguments(["--out-dir", "x", "--epochs", "4", *argv])


def test_the_smoke_mode_is_mha_and_the_default_mean_shift_config_on_one_seed():
    args = parse2("--mode", "smoke")
    cells = exp2.build_cells(args)
    assert [(c.arm, c.config, c.seed) for c in cells] == [
        ("multi_head", "mha", 0), ("mean_shift", DEFAULT_MS, 0)]
    assert args.dataset == "cifar100"


def test_the_grid_mode_is_seventeen_cells_cheapest_first():
    cells = exp2.build_cells(parse2("--mode", "grid"))
    assert len(cells) == 17 and {c.seed for c in cells} == {0}
    assert {c.config for c in cells} == set(exp2.CONFIGS)
    steps = [exp2.CONFIGS[c.config]["attention_args"].get("num_steps") for c in cells
             if c.config != "mha"]
    assert steps == sorted(steps)
    first = exp2.CONFIGS[next(c.config for c in cells if c.config != "mha")]
    assert first["attention_args"]["blurring"] is False and first["attention_args"]["num_steps"] == 1


def test_the_main_mode_is_mha_plus_the_two_chosen_configs_by_three_seeds():
    chosen = [DEFAULT_MS, "noblur_s1_per_head_gaussian"]
    cells = exp2.build_cells(parse2("--mode", "main", "--configs", *chosen))
    assert len(cells) == 9 and len({c.key for c in cells}) == 9
    assert [c.seed for c in cells] == [0] * 3 + [1] * 3 + [2] * 3
    assert {c.config for c in cells} == {"mha", *chosen}
    arms = {c.config: c.arm for c in cells}
    assert arms["mha"] == "multi_head" and arms[DEFAULT_MS] == "mean_shift"


def test_the_control_is_reachable_through_main_mode_next_to_mha_and_not_through_the_grid():
    cells = exp2.build_cells(parse2("--mode", "main", "--configs", CONTROL))
    assert len(cells) == 6 and len({c.key for c in cells}) == 6
    assert [c.seed for c in cells] == [0] * 2 + [1] * 2 + [2] * 2
    assert [c.config for c in cells[:2]] == ["mha", CONTROL]
    assert {c.arm for c in cells} == {"multi_head"}
    assert CONTROL not in {c.config for c in exp2.build_cells(parse2("--mode", "grid"))}
    assert CONTROL not in {c.config for c in exp2.build_cells(parse2("--mode", "smoke"))}
    with pytest.raises(SystemExit):
        parse2("--mode", "main", "--configs", "mha")


def test_the_main_mode_needs_known_configs_and_at_least_one():
    with pytest.raises(SystemExit):
        exp2.build_cells(parse2("--mode", "main"))
    with pytest.raises(SystemExit):
        exp2.build_cells(parse2("--mode", "main", "--configs", "no_such_config"))


# ---------------------------------------------------------------------------
# exp2: workers
# ---------------------------------------------------------------------------

EXP2_RUNS = [("multi_head", "mha"), ("multi_head", CONTROL), ("mean_shift", DEFAULT_MS),
             ("mean_shift", "noblur_s1_per_token_vmf")]


@pytest.fixture(scope="module")
def exp2_runs(fake_cifar, tmp_path_factory):
    root = tmp_path_factory.mktemp("exp2_workers")
    patch = pytest.MonkeyPatch()
    recorders, results = {}, {}
    for arm, config in EXP2_RUNS:
        recorders[config] = Recorder(patch)
        results[config] = run_worker(exp2, root, arm, config, dataset="cifar100")
        patch.undo()
    return results, recorders


@pytest.mark.parametrize("arm, config", EXP2_RUNS)
def test_an_exp2_worker_writes_one_complete_result(arm, config, exp2_runs):
    result = exp2_runs[0][config]
    metrics = result["metrics"]
    assert result["finite"] is True and result["epochs_run"] == 1
    assert result["params"] > 0
    assert (result["params"] == 2_710_564) == (config in ("mha", CONTROL))
    assert 0.0 <= metrics["select_accuracy"] <= 1.0 and 0.0 <= metrics["test_accuracy"] <= 1.0
    assert metrics["throughput_images_per_s"] > 0 and metrics["n_test"] == MAX_SAMPLES
    assert metrics["dataset"] == "cifar100" and metrics["config"] == config
    assert metrics["attention_type"] == arm and metrics["dtype_policy"] == "float32"
    assert len(result["history"]["loss"]) == 1


def test_the_exp2_rows_carry_the_learned_bandwidths(exp2_runs):
    results = exp2_runs[0]
    assert results["mha"]["metrics"]["bandwidths"]["layers"] == []
    assert results[CONTROL]["metrics"]["bandwidths"]["layers"] == []
    per_head = results[DEFAULT_MS]["metrics"]["bandwidths"]
    assert per_head["status"] == "extracted" and per_head["mode"] == "per_head"
    assert len(per_head["layers"]) == 6
    assert all(len(layer["per_head_mean"]) == 3 for layer in per_head["layers"])
    per_token = results["noblur_s1_per_token_vmf"]["metrics"]["bandwidths"]
    assert per_token["status"] == "extracted" and per_token["mode"] == "per_token"
    assert per_token["probe_images"] == results["noblur_s1_per_token_vmf"]["metrics"]["n_select"]
    assert all(layer["min"] <= layer["mean"] <= layer["max"] for layer in per_token["layers"])


def test_the_control_row_names_its_arm_config_and_initializer(exp2_runs):
    result = exp2_runs[0][CONTROL]
    metrics = result["metrics"]
    assert metrics["attention_type"] == "multi_head" and metrics["config"] == CONTROL
    assert metrics["config_args"] == {"output_kernel_initializer": "glorot_uniform"}
    assert exp2_runs[0]["mha"]["metrics"]["config_args"] == {}
    json.dumps(result)


def test_a_vmf_cell_starts_its_bandwidth_at_0_3(exp2_runs):
    first = exp2_runs[0]["noblur_s1_per_token_vmf"]["metrics"]["config_args"]
    assert first["bandwidth_init"] == 0.3 and first["kernel"] == "vmf"
    assert "bandwidth_init" not in exp2_runs[0][DEFAULT_MS]["metrics"]["config_args"]


def test_the_exp2_selection_and_test_scores_read_their_own_splits(exp2_runs):
    arrays = load_split_arrays("cifar100", seed=0, max_samples=MAX_SAMPLES)
    for config, recorder in exp2_runs[1].items():
        assert sorted(recorder.hashes) == sorted([sha(arrays.x_select), sha(arrays.x_test)]), config
    assert not np.array_equal(arrays.x_select[:5], arrays.x_test[:5])


def test_the_exp2_fit_validates_on_the_selection_split_not_the_test_split(exp2_runs):
    arrays = load_split_arrays("cifar100", seed=0, max_samples=MAX_SAMPLES)
    selection, test = sha(arrays.x_select), sha(arrays.x_test)
    assert selection != test
    for config, recorder in exp2_runs[1].items():
        assert recorder.fit_validation == [selection], config


# ---------------------------------------------------------------------------
# the real subprocess path
# ---------------------------------------------------------------------------

needs_cache = pytest.mark.skipif(not CIFAR_CACHE.exists(), reason="no CIFAR-10 in the Keras cache")


def run_module(module, *argv, cwd=REPO, timeout=900):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg", TF_CPP_MIN_LOG_LEVEL="2")
    return subprocess.run([sys.executable, "-m", module, *argv], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=timeout)


@pytest.mark.parametrize("module", ["train.mean_shift.exp1_mode_pooling",
                                    "train.mean_shift.exp2_mean_shift_attention"])
def test_help_prints_usage_and_nothing_else_is_written(module, tmp_path):
    done = run_module(module, "--help", cwd=tmp_path)
    assert done.returncode == 0 and done.stdout.lstrip().startswith("usage:")
    assert not any(tmp_path.iterdir())


@needs_cache
def test_the_exp1_driver_runs_a_cell_in_a_subprocess_on_cpu(tmp_path):
    root = tmp_path / "sweep"
    done = run_module("train.mean_shift.exp1_mode_pooling", "--cpu", "--out-dir", str(root),
                      "--epochs", "1", "--max-samples", str(MAX_SAMPLES), "--arms", "mode_s4",
                      "--seeds", "0")
    assert done.returncode == 0, done.stderr[-2000:]
    rows = [json.loads(line) for line in (root / "rows.jsonl").read_text().splitlines()]
    assert [(r["arm"], r["status"], r["seed"]) for r in rows] == [("mode_s4", "ok", 0)]
    assert rows[0]["device"] == "CUDA_VISIBLE_DEVICES="
    accuracy = rows[0]["metrics"]["metrics"]["accuracy"]
    assert set(accuracy) == set(corruption.CONDITIONS)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["resolved"]["epochs"] == 1 and manifest["devices"] == ""


@needs_cache
def test_the_exp2_smoke_driver_runs_both_cells_in_a_subprocess_on_cpu(tmp_path):
    root = tmp_path / "sweep"
    done = run_module("train.mean_shift.exp2_mean_shift_attention", "--cpu", "--out-dir",
                      str(root), "--mode", "smoke", "--dataset", "cifar10", "--epochs", "1",
                      "--max-samples", str(MAX_SAMPLES))
    assert done.returncode == 0, done.stderr[-2000:]
    rows = [json.loads(line) for line in (root / "rows.jsonl").read_text().splitlines()]
    assert sorted((r["config"], r["status"]) for r in rows) == [
        (DEFAULT_MS, "ok"), ("mha", "ok")]
    by_config = {r["config"]: r["metrics"]["metrics"] for r in rows}
    assert by_config["mha"]["bandwidths"]["layers"] == []
    assert len(by_config[DEFAULT_MS]["bandwidths"]["layers"]) == 6
