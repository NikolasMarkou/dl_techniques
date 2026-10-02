"""Pure functions of `train.lightglue.eval_homography` and one tiny end-to-end `main()`."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import train.lightglue.eval_homography as ev

from .conftest import make_lightglue, write_images

REPO_ROOT = Path(__file__).resolve().parents[3]
SIZE = (640.0, 480.0)


def _translation(tx, ty=0.0):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1.0]])


def _mild_h(rng):
    return np.array([[1.0, 0.05, 6.0], [-0.03, 1.02, -4.0], [1e-5, 2e-5, 1.0]]) + 0.002 * rng.randn(3, 3) * np.eye(3)


# ----------------------------- corner error -----------------------------


def test_corner_error_of_the_exact_homography_is_zero():
    H = _mild_h(np.random.RandomState(0))
    assert ev.corner_error(H, H, SIZE) == pytest.approx(0.0, abs=1e-9)


def test_corner_error_of_a_three_pixel_translation_is_three():
    assert ev.corner_error(_translation(3.0), np.eye(3), SIZE) == pytest.approx(3.0, abs=1e-9)
    assert ev.corner_error(_translation(0.0, 3.0), np.eye(3), SIZE) == pytest.approx(3.0, abs=1e-9)


def test_corner_error_uses_the_estimate_not_its_inverse():
    # scale 2 about the origin on a 10x10 image: corner distances 0, 10, 10*sqrt(2), 10.
    expected = (0 + 10 + 10 * np.sqrt(2) + 10) / 4
    est = np.diag([2.0, 2.0, 1.0])
    assert ev.corner_error(est, np.eye(3), (10, 10)) == pytest.approx(expected)
    assert ev.corner_error(np.linalg.inv(est), np.eye(3), (10, 10)) != pytest.approx(expected)


@pytest.mark.parametrize("bad", [None, np.full((3, 3), np.nan), np.full((3, 3), np.inf), np.eye(2),
                                 np.zeros((3, 3))])
def test_an_invalid_homography_counts_as_the_max_error(bad):
    assert ev.corner_error(bad, np.eye(3), SIZE, max_error=77.0) == 77.0


# ----------------------------- AUC -----------------------------


def test_auc_extremes():
    assert ev.error_auc([0.0, 0.0], [1, 3]) == [1.0, 1.0]
    assert ev.error_auc([5.0, 9.0], [1, 3]) == [0.0, 0.0]
    assert ev.error_auc([], [1]) != ev.error_auc([], [1])  # nan


def test_auc_of_a_uniform_ladder_matches_the_hand_derived_area():
    # errors 1..10, recall 0.1..1.0, curve (0,0),(1,.1),...,(9,.9) then flat .9 to 10:
    # trapezoids sum (2k+1)/20 for k=0..8 = 4.05, plus 0.9, over 10.
    assert ev.error_auc(list(range(1, 11)), [10]) == [0.495]


def test_auc_prepends_the_origin_point():
    # one error at 2, threshold 4: curve (0,0),(2,1),(4,1): area 1 + 2 = 3 -> 0.75.
    assert ev.error_auc([2.0], [4]) == [0.75]


def test_auc_counts_inf_errors_as_never_recalled():
    assert ev.error_auc([0.0, float("inf")], [1]) == [0.5]


# ----------------------------- estimation -----------------------------


def _synthetic(rng, count=60, H=None):
    H = _mild_h(rng) if H is None else H
    kp0 = rng.uniform(20, 600, size=(count, 2))
    hom = np.c_[kp0, np.ones(count)] @ H.T
    return kp0, hom[:, :2] / hom[:, 2:], H


def test_perfect_matches_recover_the_homography():
    rng = np.random.RandomState(1)
    kp0, kp1, H = _synthetic(rng)
    matches = np.stack([np.arange(60), np.arange(60)], 1)
    est = ev.estimate_homography(kp0, kp1, matches, reproj_threshold=1.0)
    assert ev.corner_error(est, H, SIZE) < 0.5
    est_dlt = ev.estimate_homography(kp0, kp1, matches, method="dlt")
    assert ev.corner_error(est_dlt, H, SIZE) < 0.5


def test_fewer_than_four_matches_is_a_failure_not_a_skip():
    rng = np.random.RandomState(2)
    kp0, kp1, H = _synthetic(rng, count=10)
    three = np.stack([np.arange(3), np.arange(3)], 1)
    assert ev.estimate_homography(kp0, kp1, three) is None
    assert ev.estimate_homography(kp0, kp1, np.zeros((0, 2), int)) is None
    record = ev.score_matches(kp0, kp1, three, H, SIZE, max_error=500.0)
    assert record["error"] == 500.0 and record["failed"] == 1.0 and record["num_matches"] == 3.0


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="method"):
        ev.estimate_homography(np.zeros((5, 2)), np.zeros((5, 2)), np.zeros((5, 2), int), method="x")


def test_pipeline_level_perfect_matches_give_auc_one_and_wrong_h_gives_low_auc():
    rng = np.random.RandomState(3)
    good, bad = [], []
    for _ in range(8):
        kp0, kp1, H = _synthetic(rng, count=40)
        matches = np.stack([np.arange(40), np.arange(40)], 1)
        labels = np.arange(40)
        good.append(ev.score_matches(kp0, kp1, matches, H, SIZE, labels, reproj_threshold=1.0))
        # the matches follow a very different homography than the ground truth
        wrong = _translation(80.0, -60.0) @ H
        bad.append(ev.score_matches(kp0, kp1, matches, wrong, SIZE, labels, reproj_threshold=1.0))
    summary_good = ev.summarize_records(good)
    summary_bad = ev.summarize_records(bad)
    assert summary_good["auc"] == {"auc@1": 1.0, "auc@3": 1.0, "auc@5": 1.0, "auc@10": 1.0}
    assert summary_good["mean_precision"] == 1.0 and summary_good["mean_recall"] == 1.0
    assert summary_good["failures"] == 0
    assert summary_bad["auc"]["auc@10"] < 0.05
    assert summary_bad["median_error"] > 50.0


def test_summarize_reports_stop_layer_and_nan_precision_is_excluded():
    records = [
        {"error": 1.0, "failed": 0.0, "num_matches": 10.0, "precision": 1.0, "recall": 0.5, "stop": 3.0},
        {"error": 9.0, "failed": 1.0, "num_matches": 0.0, "precision": float("nan"),
         "recall": float("nan"), "stop": 5.0},
    ]
    summary = ev.summarize_records(records)
    assert summary["mean_stop_layer"] == 4.0 and summary["mean_precision"] == 1.0
    assert summary["failures"] == 1 and summary["mean_error"] == 5.0 and summary["median_error"] == 5.0


def test_match_quality_counts_against_the_labels():
    labels0 = np.array([2, -1, 0, -2])
    quality = ev.match_quality(np.array([[0, 2], [1, 3], [2, 0]]), labels0)
    assert quality["precision"] == pytest.approx(2 / 3) and quality["recall"] == 1.0
    assert quality["precision_strict"] == pytest.approx(2 / 3)
    empty = ev.match_quality(np.zeros((0, 2), int), labels0)
    assert np.isnan(empty["precision"]) and np.isnan(empty["precision_strict"])


def test_precision_excludes_ignored_partners_and_strict_precision_keeps_them():
    # image 0: kp0 -> 2 (true), kp1 dustbin, kp2 -> 0 (true), kp3 ignored (-2).
    labels0 = np.array([2, -1, 0, -2])
    # image 1: kp 1 is ignored (-2), everything else is unconstrained here.
    labels1 = np.array([2, -2, 0, -1])
    matches = np.array([[0, 2],   # correct
                        [1, 3],   # dustbin keypoint matched: a real false positive
                        [2, 0],   # correct
                        [3, 1]])  # label0 == -2: not a prediction
    quality = ev.match_quality(matches, labels0)
    assert quality["precision"] == pytest.approx(2 / 3)          # 4th pair leaves both sides
    assert quality["precision_strict"] == pytest.approx(2 / 4)   # 4th pair is a false positive
    # a pair into an ignored image-1 keypoint is excluded too, only when labels1 is given
    with_l1 = ev.match_quality(np.array([[0, 2], [2, 1]]), labels0, labels1)
    assert with_l1["precision"] == 1.0 and with_l1["precision_strict"] == pytest.approx(0.5)
    without_l1 = ev.match_quality(np.array([[0, 2], [2, 1]]), labels0)
    assert without_l1["precision"] == pytest.approx(0.5)
    # only ignored pairs: lenient precision undefined, strict precision 0
    only = ev.match_quality(np.array([[3, 1]]), labels0, labels1)
    assert np.isnan(only["precision"]) and only["precision_strict"] == 0.0
    assert quality["recall"] == 1.0


def test_precision_matches_the_training_metric():
    """The eval's `precision` equals `KeypointMatchMetric` precision on the same matches."""
    from dl_techniques.metrics.keypoint_matching import KeypointMatchMetric
    from dl_techniques.losses.lightglue_loss import pack_matches

    labels0 = np.array([1, -1, -2, 0, 4])
    labels1 = np.array([3, 0, -2, -1, 4])
    matches = np.array([[0, 1], [1, 3], [2, 2], [3, 0], [4, 4]])
    quality = ev.match_quality(matches, labels0, labels1)
    scores = np.full((1, 6, 6), np.log(1e-6), "float32")
    for i, j in matches:
        scores[0, i, j] = np.log(0.9)
    metric = KeypointMatchMetric(mode="precision", threshold=0.1)
    metric.update_state(pack_matches(labels0[None], labels1[None]), scores)
    # (0,1) ok, (1,3) wrong, (2,2) ignored, (3,0) ok, (4,4) ok
    assert float(metric.result()) == pytest.approx(3 / 4)
    assert quality["precision"] == pytest.approx(float(metric.result()))
    assert quality["precision_strict"] == pytest.approx(3 / 5)


def test_summary_carries_both_precisions():
    record = {"error": 1.0, "failed": 0.0, "num_matches": 4.0, "precision": 1.0,
              "precision_strict": 0.5, "recall": 1.0}
    summary = ev.summarize_records([record])
    assert summary["mean_precision"] == 1.0 and summary["mean_precision_strict"] == 0.5


def test_border_default_is_the_trainers_default():
    from train.lightglue.train_lightglue import LightGlueTrainConfig, parse_arguments as trainer_parse
    args = ev.parse_arguments(["--lightglue", "a", "--superpoint-checkpoint", "b"])
    trainer = trainer_parse(["--superpoint-checkpoint", "b"])
    assert args.border == ev.DEFAULT_BORDER == LightGlueTrainConfig.border
    assert args.border == trainer.border
    assert ev.parse_arguments(["--lightglue", "a", "--superpoint-checkpoint", "b",
                               "--border", "9"]).border == 9
    with pytest.raises(SystemExit):
        ev.parse_arguments(["--lightglue", "a", "--superpoint-checkpoint", "b", "--border", "-1"])


# ----------------------------- MNN baseline -----------------------------


def _descriptors():
    d0 = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    d1 = np.array([[0.0, 1.0], [1.0, 0.05]])
    return d0, d1


def test_mnn_on_hand_built_descriptors():
    d0, d1 = _descriptors()
    assert ev.mutual_nn_matches(d0, d1).tolist() == [[0, 1], [1, 0]]  # c->y is not mutual (y->a)


def test_mnn_respects_masks_and_the_ratio_test():
    d0, d1 = _descriptors()
    assert ev.mutual_nn_matches(d0, d1, mask0=[1, 0, 1]).tolist() == [[0, 1]]
    assert ev.mutual_nn_matches(d0, d1, ratio=0.9).tolist() == [[0, 1], [1, 0]]
    assert ev.mutual_nn_matches(d0, d1, ratio=0.01).tolist() == [[1, 0]]
    assert ev.mutual_nn_matches(d0, d1, mask1=[0, 0]).shape == (0, 2)


# ----------------------------- CLI -----------------------------


def test_help_exits_zero_with_usage():
    proc = subprocess.run(
        [sys.executable, "-m", "train.lightglue.eval_homography", "--help"],
        cwd=REPO_ROOT / "src", capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0 and "usage:" in proc.stdout


@pytest.mark.parametrize("flags", [["--num-pairs", "0"], ["--mnn-ratio", "2"],
                                   ["--depth-confidence", "0.0"], ["--filter-threshold", "1.5"]])
def test_bad_flags_exit_two_before_any_work(flags):
    with pytest.raises(SystemExit) as info:
        ev.parse_arguments(["--lightglue", "a", "--superpoint-checkpoint", "b", *flags])
    assert info.value.code == 2


@pytest.fixture(scope="module")
def run(tmp_path_factory, superpoint_path):
    root = tmp_path_factory.mktemp("eval_run")
    images = write_images(root, 4)
    lightglue_path = root / "lightglue.keras"
    matcher = make_lightglue()
    ones = np.ones((1, 4, 2), "float32")
    matcher({"keypoints0": ones, "keypoints1": ones, "descriptors0": np.ones((1, 4, 32), "float32"),
             "descriptors1": np.ones((1, 4, 32), "float32"), "image_size0": np.full((1, 2), 64, "float32"),
             "image_size1": np.full((1, 2), 64, "float32")})
    matcher.save(str(lightglue_path))
    out = root / "out"
    code = ev.main([
        "--lightglue", str(lightglue_path), "--superpoint-checkpoint", superpoint_path,
        "--images-dir", str(images), "--num-pairs", "3", "--max-keypoints", "32",
        "--seed", "5", "--output-dir", str(out), "--experiment-name", "tiny",
        "--depth-confidence", "-1", "--width-confidence", "-1"])
    return {"code": code, "dir": out / "tiny", "images": images, "out": out,
            "lightglue": lightglue_path}


def test_main_writes_the_summary_with_both_methods(run):
    assert run["code"] == 0
    assert {p.name for p in run["dir"].iterdir()} >= {"config.json", "run.log", "results_summary.json"}
    summary = json.loads((run["dir"] / "results_summary.json").read_text())
    assert summary["pairs"] == 3 and summary["image_size"] == [64, 64]
    assert set(summary["methods"]) == {"lightglue", "mnn"}
    for stats in summary["methods"].values():
        assert set(stats["auc"]) == {"auc@1", "auc@3", "auc@5", "auc@10"}
        assert stats["pairs"] == 3 and stats["mean_error"] is not None
    assert summary["methods"]["lightglue"]["mean_stop_layer"] == 2.0  # adaptive off: last layer
    assert summary["border"] == ev.DEFAULT_BORDER
    for stats in summary["methods"].values():
        assert "mean_precision_strict" in stats and "mean_precision" in stats
    assert summary["adaptive"]["depth_confidence"] == -1.0


def test_main_refuses_a_reused_name_and_a_missing_checkpoint(run, superpoint_path, tmp_path):
    with pytest.raises(FileExistsError, match="already holds a run"):
        ev.main(["--lightglue", str(run["lightglue"]), "--superpoint-checkpoint", superpoint_path,
                 "--images-dir", str(run["images"]), "--output-dir", str(run["out"]),
                 "--experiment-name", "tiny"])
    code = ev.main(["--lightglue", str(tmp_path / "missing.keras"),
                    "--superpoint-checkpoint", superpoint_path, "--images-dir", str(run["images"]),
                    "--output-dir", str(tmp_path), "--experiment-name", "nope"])
    assert code == 2 and not (tmp_path / "nope").exists()


def test_main_runs_with_the_checkpoints_adaptive_defaults(run, superpoint_path, tmp_path):
    code = ev.main(["--lightglue", str(run["lightglue"]), "--superpoint-checkpoint", superpoint_path,
                    "--images-dir", str(run["images"]), "--num-pairs", "2", "--max-keypoints", "32",
                    "--output-dir", str(tmp_path), "--experiment-name", "adaptive"])
    assert code == 0
    summary = json.loads((tmp_path / "adaptive" / "results_summary.json").read_text())
    assert 1.0 <= summary["methods"]["lightglue"]["mean_stop_layer"] <= 3.0
    assert summary["adaptive"]["depth_confidence"] == pytest.approx(0.95)


def test_border_flag_reaches_the_decode(run, superpoint_path, tmp_path):
    """A border wider than half the image leaves no keypoint, so nothing can match."""
    default = json.loads((run["dir"] / "results_summary.json").read_text())
    assert default["methods"]["mnn"]["mean_matches"] > 0  # control: the default border keeps keypoints
    code = ev.main(["--lightglue", str(run["lightglue"]), "--superpoint-checkpoint", superpoint_path,
                    "--images-dir", str(run["images"]), "--num-pairs", "3", "--max-keypoints", "32",
                    "--seed", "5", "--output-dir", str(tmp_path), "--experiment-name", "wide",
                    "--border", "40", "--depth-confidence", "-1", "--width-confidence", "-1"])
    assert code == 0
    wide = json.loads((tmp_path / "wide" / "results_summary.json").read_text())
    assert wide["border"] == 40
    assert json.loads((tmp_path / "wide" / "config.json").read_text())["border"] == 40
    assert wide["methods"]["mnn"]["mean_matches"] == 0.0
    assert wide["methods"]["lightglue"]["mean_matches"] == 0.0
    assert wide["methods"]["mnn"]["failures"] == 3


def test_border_flag_is_what_reaches_decode_superpoint(run, superpoint_path, tmp_path, monkeypatch):
    seen = []
    real = ev.decode_superpoint

    def spy(outputs, max_keypoints, threshold, nms_radius, border):
        seen.append(border)
        return real(outputs, max_keypoints, threshold, nms_radius, border)

    monkeypatch.setattr(ev, "decode_superpoint", spy)
    ev.main(["--lightglue", str(run["lightglue"]), "--superpoint-checkpoint", superpoint_path,
             "--images-dir", str(run["images"]), "--num-pairs", "1", "--max-keypoints", "32",
             "--output-dir", str(tmp_path), "--experiment-name", "spy", "--border", "7"])
    assert seen == [7]
