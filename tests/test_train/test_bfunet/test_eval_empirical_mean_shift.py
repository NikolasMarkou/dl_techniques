"""Guards for ``src/train/bfunet/eval_empirical_mean_shift.py``, the exact Tweedie oracle.

The oracle is one mean-shift step with ``h = sigma`` over a bank of clean patches. What can go
wrong silently, and what pins it here:

- the weight exponent or the bandwidth (``2 sigma^2`` against ``sigma^2``): a two-point bank in
  one dimension has a closed-form MMSE (about 0.1124 at sigma 0.5) that a deterministic
  quadrature reaches without a random draw;
- a dropped final chunk, or a result that depends on the chunk size;
- the effective number in bits or by ``1 / sum w^2`` instead of ``exp(H)`` in nats;
- a memorization ratio that a mean would turn into 1e8 on exact twins: the worked example, the
  self-retrieval and degenerate counts on a bank with an exact twin and a near twin;
- a nearest-bank search that float32 cannot resolve;
- the model arm: a synthetic stamped checkpoint is scored, an unstamped one and one with the
  wrong input shape are refused, and a refusal writes nothing.

Everything runs on CPU and writes only under ``tmp_path``; nothing here touches ``results/``.

Run:
    CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg .venv/bin/python -m pytest \\
        tests/test_train/test_bfunet/test_eval_empirical_mean_shift.py -q
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

os.environ.setdefault("MPLBACKEND", "Agg")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import tensorflow as tf  # noqa: E402

from tests.numerics import true_float32_matmul  # noqa: E402
from train.bfunet import eval_empirical_mean_shift as em  # noqa: E402
from train.bfunet import eval_psnr_vs_noise as ev  # noqa: E402


@pytest.fixture(autouse=True)
def exact_float32():
    """TF32 matmul off for the numeric pins (a no-op on CPU, restored afterwards)."""
    with true_float32_matmul():
        yield


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def _config(tmp_path: Path, **overrides) -> "em.OracleConfig":
    values = dict(
        train_dir=str(tmp_path / "train"), test_dir=str(tmp_path / "test"),
        output_dir=str(tmp_path / "out"), bank_size=24, chunk_size=4, num_queries=6,
        sigmas_255=(25.0, 50.0), dtype="float32", clip=False, model_path=None, seed=7,
    )
    values.update(overrides)
    return em.OracleConfig(**values)


def _write_pngs(directory: Path, count: int, size: int = 40, seed: int = 0) -> List[np.ndarray]:
    """Write ``count`` random grayscale PNGs; return their uint8 pixel arrays."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    images = []
    for index in range(count):
        pixels = rng.integers(1, 256, size=(size, size, 1), dtype=np.uint8)
        (directory / f"img_{index}.png").write_bytes(tf.io.encode_png(pixels).numpy())
        images.append(pixels)
    return images


def _save_checkpoint(directory: Path, input_shape, stamp) -> Path:
    """A one-layer functional denoiser saved as ``best_model.keras`` with a hand-written config."""
    directory.mkdir(parents=True)
    keras.utils.set_random_seed(0)
    inputs = keras.Input(input_shape)
    model = keras.Model(inputs, keras.layers.Conv2D(input_shape[-1], 3, padding="same")(inputs))
    path = directory / "best_model.keras"
    model.save(path)
    if stamp is not None:
        (directory / "config.json").write_text(json.dumps(stamp))
    return path


def _reject_constant(token: str):
    raise AssertionError(f"summary.json holds the non-strict JSON token {token}")


# ---------------------------------------------------------------------
# the oracle against a closed form
# ---------------------------------------------------------------------

SIGMA = 0.5
CLOSED_FORM_MMSE = 0.1124  # quadrature of the analytic posterior mean, measured in findings


def _quadrature(grid: np.ndarray, estimate: np.ndarray) -> float:
    """MMSE of a posterior-mean estimate for the bank {0, 1}: trapezoid over the noisy value."""
    def density(centre: float) -> np.ndarray:
        return np.exp(-0.5 * ((grid - centre) / SIGMA) ** 2) / (SIGMA * np.sqrt(2.0 * np.pi))

    integrand = 0.5 * density(0.0) * estimate ** 2 + 0.5 * density(1.0) * (1.0 - estimate) ** 2
    return float(np.trapezoid(integrand, grid) if hasattr(np, "trapezoid") else np.trapz(integrand, grid))


class TestTheOracleMatchesTheClosedForm:
    """Two bank points at 0 and 1 in one dimension: the posterior mean is a logistic function."""

    GRID = np.linspace(-6.0, 7.0, 26001)

    def _closed_form_estimate(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(2.0 * self.GRID - 1.0) / (2.0 * SIGMA ** 2)))

    def test_the_analytic_value_is_the_one_in_the_findings(self) -> None:
        mmse = _quadrature(self.GRID, self._closed_form_estimate())
        assert mmse == pytest.approx(CLOSED_FORM_MMSE, abs=1e-4)

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_the_oracle_reaches_the_closed_form_mmse(self, dtype) -> None:
        bank = np.array([[0.0], [1.0]])
        estimate, _ = em.empirical_tweedie(self.GRID[:, None], bank, SIGMA, chunk_size=4000, dtype=dtype)
        assert estimate.dtype == np.dtype(dtype)
        oracle = _quadrature(self.GRID, estimate[:, 0].astype(np.float64))
        closed = _quadrature(self.GRID, self._closed_form_estimate())
        assert abs(oracle - closed) < 1e-4
        assert abs(oracle - closed) / closed < 1e-3

    def test_the_estimate_is_the_logistic_function_pointwise(self) -> None:
        estimate, _ = em.empirical_tweedie(
            self.GRID[:, None], np.array([[0.0], [1.0]]), SIGMA, chunk_size=1000, dtype="float64")
        assert np.max(np.abs(estimate[:, 0] - self._closed_form_estimate())) < 1e-9


# ---------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------

class TestChunkSizeChangesNothing:
    """A chunk is a memory knob; the answer for a query cannot depend on its neighbours.

    Not bit-exact for a chunk smaller than the batch: the matmul kernel is chosen by the
    row count, so one row differs from the same row in a batch by one rounding step (measured
    1.1e-16 in float64, 9e-8 in float32). The tolerance below is a few such steps; a dropped
    or repeated chunk moves values by orders of magnitude more.
    """

    TOLERANCE = {"float32": 1e-5, "float64": 1e-12}

    def _problem(self):
        rng = np.random.RandomState(0)
        bank = rng.uniform(size=(40, 6))
        return bank, bank[:7] + 0.2 * rng.standard_normal((7, 6))

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 6])
    def test_a_smaller_chunk_gives_the_same_answer_for_every_query(self, chunk_size, dtype) -> None:
        bank, queries = self._problem()
        whole, whole_eff = em.empirical_tweedie(queries, bank, 0.3, chunk_size=7, dtype=dtype)
        chunked, chunked_eff = em.empirical_tweedie(queries, bank, 0.3, chunk_size=chunk_size, dtype=dtype)
        assert chunked.shape == (7, 6) and chunked_eff.shape == (7,)
        tolerance = self.TOLERANCE[dtype]
        np.testing.assert_allclose(chunked, whole, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(chunked_eff, whole_eff, rtol=tolerance, atol=tolerance)

    @pytest.mark.parametrize("chunk_size", [7, 8, 100])
    def test_a_chunk_that_holds_the_whole_batch_is_bit_identical(self, chunk_size) -> None:
        bank, queries = self._problem()
        whole, whole_eff = em.empirical_tweedie(queries, bank, 0.3, chunk_size=7)
        other, other_eff = em.empirical_tweedie(queries, bank, 0.3, chunk_size=chunk_size)
        np.testing.assert_array_equal(other, whole)
        np.testing.assert_array_equal(other_eff, whole_eff)

    def test_a_bad_chunk_size_or_no_queries_is_refused(self) -> None:
        bank = np.zeros((3, 2))
        with pytest.raises(ValueError, match="chunk_size"):
            em.empirical_tweedie(np.zeros((2, 2)), bank, 0.1, chunk_size=0)
        with pytest.raises(ValueError, match="no queries"):
            em.empirical_tweedie(np.zeros((0, 2)), bank, 0.1, chunk_size=1)


# ---------------------------------------------------------------------
# effective number, memorization, counts
# ---------------------------------------------------------------------

class TestEffectiveNumber:
    def test_the_pins(self) -> None:
        weights = np.array([[0.5, 0.5], [0.9, 0.1], [1.0, 0.0]])
        values = em.effective_number(weights)
        assert values[0] == pytest.approx(2.0, abs=1e-12)
        assert values[1] == pytest.approx(1.3841, abs=1e-4)
        assert values[2] == pytest.approx(1.0, abs=1e-12)

    def test_a_uniform_row_over_k_entries_is_k(self) -> None:
        assert em.effective_number(np.full((1, 8), 0.125))[0] == pytest.approx(8.0, abs=1e-12)

    def test_it_is_not_the_inverse_participation_ratio(self) -> None:
        assert em.effective_number(np.array([[0.9, 0.1]]))[0] != pytest.approx(1.0 / (0.81 + 0.01), abs=1e-3)


class TestMemorizationRatio:
    BANK = np.array([[0.0, 0.0], [2.0, 0.0]])
    TRUTH = np.array([1.0, 0.5])

    def test_the_worked_example(self) -> None:
        estimates = np.array([[0.0, 0.0], [0.5, 0.25], [1.0, 0.5]])
        scores = em.memorization_scores(np.tile(self.TRUTH, (3, 1)), estimates, self.BANK, 8, in_bank=False)
        np.testing.assert_allclose(scores.ratio, [0.0, 0.5, 1.0], atol=1e-12)
        assert not scores.degenerate.any() and not scores.self_retrieved.any()
        summary = em.summarize_memorization(scores)
        assert summary["median"] == pytest.approx(0.5)
        assert summary["frac_below_half"] == pytest.approx(1.0 / 3.0)
        assert summary["n_scored"] == 3 and summary["self_retrieval_rate"] is None

    def test_a_bank_with_an_exact_twin_and_a_near_twin_counts_instead_of_dividing(self) -> None:
        a = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        near = a + np.float32(1e-5)
        far_b = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
        far_d = np.array([0.5, 0.1, 0.9, 0.2], dtype=np.float32)
        bank = np.stack([a, far_b, a.copy(), near, far_d])  # index 2 is an exact twin of 0
        clean = np.stack([a, a, near, a, far_d])             # bank members 0, 2, 3, 0, 4
        estimate = np.stack([a, a, near, near, far_b])
        scores = em.memorization_scores(clean, estimate, bank, 8, in_bank=True)
        assert scores.self_retrieved.tolist() == [True, True, True, False, False]
        assert scores.degenerate.tolist() == [False, False, False, True, False]
        assert np.isnan(scores.ratio[:4]).all() and scores.ratio[4] == pytest.approx(0.0)
        summary = em.summarize_memorization(scores)
        assert summary["self_retrieval_rate"] == pytest.approx(3 / 5)
        assert summary["n_degenerate"] == 1 and summary["n_scored"] == 1 and summary["median"] == 0.0

    def test_the_same_queries_held_out_are_all_degenerate_not_self_retrieved(self) -> None:
        a = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        bank = np.stack([a, a + np.float32(0.5), a.copy()])
        scores = em.memorization_scores(np.stack([a, a]), np.stack([a, a]), bank, 8, in_bank=False)
        assert scores.degenerate.all() and not scores.self_retrieved.any()
        summary = em.summarize_memorization(scores)
        assert summary["n_scored"] == 0 and summary["median"] is None
        assert summary["frac_below_half"] is None and summary["n_degenerate"] == 2

    def test_the_nearest_bank_patch_is_searched_in_float64(self) -> None:
        # Far from the origin two patches 1e-3 apart have squared norms (4e4) whose float32
        # spacing (4e-3) swallows their 1e-6 distance: only a float64 search tells them apart.
        bank = np.stack([np.zeros(4), np.full(4, 1000.0), np.full(4, 300.0),
                         np.array([300.001, 300.0, 300.0, 300.0])])
        index, distance = em.nearest_bank_patch(bank[3:4], bank, chunk_size=2)
        assert index.tolist() == [3] and distance[0] == 0.0
        index, _ = em.nearest_bank_patch(bank[2:3], bank, chunk_size=2)
        assert index.tolist() == [2]


class TestThePsnrCap:
    def test_the_cap_constant_is_what_psnr_per_image_returns_for_a_perfect_copy(self) -> None:
        clean = np.random.RandomState(0).uniform(size=(2, 4, 4, 1)).astype(np.float32)
        assert float(ev.psnr_per_image(clean, clean)[0]) == em.PSNR_CAP_DB

    def test_the_count_at_the_cap(self) -> None:
        rng = np.random.RandomState(1)
        clean = rng.uniform(size=(6, 4, 4, 1)).astype(np.float32)
        estimate = clean.copy()
        estimate[3:] += 0.1 * rng.standard_normal(estimate[3:].shape).astype(np.float32)
        bank = rng.uniform(size=(10, 4, 4, 1)).astype(np.float32)
        metrics = em.arm_metrics(clean, estimate, bank, 8, in_bank=False)
        assert metrics["psnr_at_cap"] == 3
        assert metrics["psnr_median"] < em.PSNR_CAP_DB and metrics["psnr_mean"] < em.PSNR_CAP_DB


# ---------------------------------------------------------------------
# a whole query set
# ---------------------------------------------------------------------

class TestEvaluateQuerySet:
    def test_a_tiny_sigma_makes_the_oracle_copy_every_in_bank_query(self, tmp_path) -> None:
        bank = np.random.RandomState(3).uniform(size=(20, 4, 4, 1)).astype(np.float32)
        config = _config(tmp_path, sigmas_255=(2.55,))
        rows = em.evaluate_query_set(em.IN_BANK, bank[:8], bank, config, np.random.RandomState(0), None)
        assert len(rows) == 1
        row = rows[0]
        assert row["n"] == 8 and row["oracle_neff_median"] == pytest.approx(1.0, abs=1e-6)
        assert row["oracle_psnr_at_cap"] == 8
        assert row["oracle_mem_self_retrieval_rate"] == 1.0 and row["oracle_mem_n_scored"] == 0
        assert not any(key.startswith("model_") for key in row)

    def test_noise_is_unclipped_by_default_and_clipping_is_opt_in(self, tmp_path) -> None:
        clean = np.full((16, 16, 16, 1), 0.99, dtype=np.float32)
        bank = np.random.RandomState(3).uniform(size=(5, 16, 16, 1)).astype(np.float32)
        free = em.evaluate_query_set(
            em.HELD_OUT, clean, bank, _config(tmp_path, sigmas_255=(50.0,)), np.random.RandomState(0), None)[0]
        clipped = em.evaluate_query_set(
            em.HELD_OUT, clean, bank, _config(tmp_path, sigmas_255=(50.0,), clip=True),
            np.random.RandomState(0), None)[0]
        assert free["noisy_psnr_mean"] != clipped["noisy_psnr_mean"]
        # unclipped noise has the nominal PSNR 10 log10(1 / sigma^2), up to sampling error
        assert free["noisy_psnr_mean"] == pytest.approx(10 * np.log10(1.0 / (50.0 / 255.0) ** 2), abs=0.5)


# ---------------------------------------------------------------------
# the model arm
# ---------------------------------------------------------------------

class TestTheModelArm:
    def test_a_stamped_checkpoint_is_scored_on_the_same_noisy_patches(self, tmp_path) -> None:
        path = _save_checkpoint(tmp_path / "ckpt", (32, 32, 1), {"data_range": "[0,1]"})
        model = em.load_model_arm(str(path))
        rng = np.random.RandomState(5)
        bank = rng.uniform(size=(12, 32, 32, 1)).astype(np.float32)
        clean = bank[:5]
        config = _config(tmp_path, sigmas_255=(25.0,))
        row = em.evaluate_query_set(em.IN_BANK, clean, bank, config, np.random.RandomState(9), model)[0]
        noisy = ev.add_awgn(clean, 25.0 / 255.0, False, np.random.RandomState(9))
        denoised = model.predict(noisy, batch_size=5, verbose=0)
        expected = ev.psnr_per_image(denoised.astype(np.float64), clean.astype(np.float64))
        assert row["model_psnr_median"] == pytest.approx(float(np.median(expected)), abs=1e-3)
        assert row["model_psnr_mean"] == pytest.approx(float(np.mean(expected)), abs=1e-3)
        assert "model_mem_median" in row and "oracle_psnr_median" in row

    @pytest.mark.parametrize("stamp", [None, {"variant": "base"}, {"data_range": "[-0.5,+0.5]"}],
                             ids=["no config", "no stamp", "legacy stamp"])
    def test_an_unstamped_checkpoint_is_refused(self, tmp_path, stamp) -> None:
        path = _save_checkpoint(tmp_path / "ckpt", (32, 32, 1), stamp)
        with pytest.raises(ValueError, match=r"REFUSING.*data_range"):
            em.load_model_arm(str(path))

    @pytest.mark.parametrize("shape", [(16, 16, 1), (32, 32, 3), (256, 256, 3)],
                             ids=["small", "rgb", "trainer default"])
    def test_a_checkpoint_that_cannot_score_32x32x1_is_refused(self, tmp_path, shape) -> None:
        path = _save_checkpoint(tmp_path / "ckpt", shape, {"data_range": "[0,1]"})
        with pytest.raises(ValueError, match=r"REFUSING.*input shape"):
            em.load_model_arm(str(path))

    def test_a_missing_checkpoint_is_an_error(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            em.load_model_arm(str(tmp_path / "nope.keras"))


# ---------------------------------------------------------------------
# the bank builder
# ---------------------------------------------------------------------

class TestTheBankBuilder:
    def test_the_crops_are_the_images_pixels_scaled_to_unit_range(self, tmp_path) -> None:
        images = _write_pngs(tmp_path / "imgs", 4, size=32)
        paths = sorted(str(p) for p in (tmp_path / "imgs").glob("*.png"))
        patches, decoded = em.load_crop_patches(paths, 10, 32, 1, np.random.RandomState(0))
        assert patches.shape == (10, 32, 32, 1) and patches.dtype == np.float32 and decoded == 4
        assert patches.min() >= 0.0 and patches.max() <= 1.0
        expected = [image.astype(np.float32) / np.float32(255.0) for image in images]
        for patch in patches:
            assert any(np.array_equal(patch, image) for image in expected)

    def test_the_duplicate_share_is_reported(self, tmp_path) -> None:
        # four 32 px images allow one crop offset each: 10 crops are 4 distinct patches
        _write_pngs(tmp_path / "imgs", 4, size=32)
        paths = sorted(str(p) for p in (tmp_path / "imgs").glob("*.png"))
        patches, _ = em.load_crop_patches(paths, 10, 32, 1, np.random.RandomState(0))
        stats = em.duplicate_statistics(patches)
        assert stats["n"] == 10 and stats["n_unique"] == 4
        assert stats["duplicate_share"] == pytest.approx(0.6) and stats["share_with_twin"] == 1.0

    def test_distinct_patches_have_no_duplicates(self) -> None:
        patches = np.random.RandomState(0).uniform(size=(9, 4, 4, 1)).astype(np.float32)
        stats = em.duplicate_statistics(patches)
        assert stats["n_unique"] == 9 and stats["duplicate_share"] == 0.0 and stats["share_with_twin"] == 0.0

    def test_fewer_patches_than_images_decodes_only_that_many_images(self, tmp_path) -> None:
        _write_pngs(tmp_path / "imgs", 6, size=40)
        paths = sorted(str(p) for p in (tmp_path / "imgs").glob("*.png"))
        patches, decoded = em.load_crop_patches(paths, 3, 32, 1, np.random.RandomState(0))
        assert patches.shape[0] == 3 and decoded == 3

    def test_a_run_is_reproducible_from_its_generator_and_an_empty_list_is_refused(self, tmp_path) -> None:
        _write_pngs(tmp_path / "imgs", 3, size=40)
        paths = sorted(str(p) for p in (tmp_path / "imgs").glob("*.png"))
        first, _ = em.load_crop_patches(paths, 9, 32, 1, np.random.RandomState(4))
        second, _ = em.load_crop_patches(paths, 9, 32, 1, np.random.RandomState(4))
        np.testing.assert_array_equal(first, second)
        with pytest.raises(ValueError, match="no image paths"):
            em.load_crop_patches([], 3, 32, 1, np.random.RandomState(0))
        with pytest.raises(ValueError, match="count"):
            em.load_crop_patches(paths, 0, 32, 1, np.random.RandomState(0))

    def test_an_image_smaller_than_the_patch_is_upscaled_not_dropped(self, tmp_path) -> None:
        _write_pngs(tmp_path / "imgs", 1, size=16)
        paths = [str(p) for p in (tmp_path / "imgs").glob("*.png")]
        patches, _ = em.load_crop_patches(paths, 2, 32, 1, np.random.RandomState(0))
        assert patches.shape == (2, 32, 32, 1)


# ---------------------------------------------------------------------
# the run directory
# ---------------------------------------------------------------------

def _inputs(tmp_path: Path) -> None:
    _write_pngs(tmp_path / "train", 6, size=40, seed=1)
    _write_pngs(tmp_path / "test", 3, size=40, seed=2)


class TestTheRunDirectory:
    def test_an_oracle_only_run_writes_the_four_files_and_says_so(self, tmp_path) -> None:
        _inputs(tmp_path)
        run_dir = em.run(_config(tmp_path), run_name="first")
        assert run_dir == tmp_path / "out" / "first"
        assert sorted(p.name for p in run_dir.iterdir()) == sorted(em.RUN_ARTIFACTS)
        text = (run_dir / em.SUMMARY_NAME).read_text()
        summary = json.loads(text, parse_constant=_reject_constant)
        assert summary["oracle_only"] is True and summary["model"] is None
        assert "oracle only" in summary["statement"]
        assert summary["psnr_cap_db"] == 120.0 and summary["noise"]["clipped"] is False
        assert summary["bank"]["n"] == 24 and 0.0 <= summary["bank"]["duplicate_share"] < 1.0
        assert summary["held_out"]["n"] == 6 and summary["in_bank"]["n"] == 6
        rows = summary["rows"]
        assert [(r["query_set"], r["sigma_255"]) for r in rows] == [
            ("in_bank", 25.0), ("in_bank", 50.0), ("held_out", 25.0), ("held_out", 50.0)]
        with open(run_dir / em.CSV_NAME, newline="") as handle:
            table = list(csv.DictReader(handle))
        assert len(table) == 4 and not any(key.startswith("model_") for key in table[0])
        assert (run_dir / em.PLOT_NAME).stat().st_size > 0
        recorded = json.loads((run_dir / em.CONFIG_NAME).read_text())
        assert recorded["bank_size"] == 24 and recorded["clip"] is False and recorded["patch_size"] == 32

    def test_the_same_seed_gives_the_same_rows(self, tmp_path) -> None:
        _inputs(tmp_path)
        rows = [json.loads((em.run(_config(tmp_path), run_name=name) / em.SUMMARY_NAME).read_text())["rows"]
                for name in ("a", "b")]
        assert rows[0] == rows[1]

    def test_a_run_directory_that_holds_a_run_is_refused_and_left_byte_identical(self, tmp_path) -> None:
        _inputs(tmp_path)
        run_dir = em.run(_config(tmp_path), run_name="taken")
        before = {p.name: p.read_bytes() for p in run_dir.iterdir()}
        with pytest.raises(FileExistsError, match="already holds a run"):
            em.run(_config(tmp_path), run_name="taken")
        assert {p.name: p.read_bytes() for p in run_dir.iterdir()} == before

    def test_a_scored_run_carries_the_model_columns(self, tmp_path) -> None:
        _inputs(tmp_path)
        path = _save_checkpoint(tmp_path / "ckpt", (32, 32, 1), {"data_range": "[0,1]"})
        run_dir = em.run(_config(tmp_path, model_path=str(path), sigmas_255=(25.0,)), run_name="with_model")
        summary = json.loads((run_dir / em.SUMMARY_NAME).read_text())
        assert summary["oracle_only"] is False and summary["model"]["path"] == str(path)
        assert summary["model"]["input_shape"] == [None, 32, 32, 1]
        assert "oracle only" not in summary["statement"]
        assert all("model_psnr_median" in row for row in summary["rows"])
        with open(run_dir / em.CSV_NAME, newline="") as handle:
            assert "model_psnr_median" in next(csv.reader(handle))

    def test_a_refused_checkpoint_writes_nothing(self, tmp_path) -> None:
        _inputs(tmp_path)
        path = _save_checkpoint(tmp_path / "ckpt", (32, 32, 1), None)
        with pytest.raises(ValueError, match="REFUSING"):
            em.run(_config(tmp_path, model_path=str(path)), run_name="refused")
        assert not (tmp_path / "out").exists()

    def test_an_empty_image_directory_is_an_error_before_anything_is_written(self, tmp_path) -> None:
        (tmp_path / "train").mkdir()
        _write_pngs(tmp_path / "test", 2)
        with pytest.raises(ValueError, match="no images found in the train directory"):
            em.run(_config(tmp_path), run_name="empty")
        assert not (tmp_path / "out").exists()

    def test_the_summary_writer_is_strict_json(self, tmp_path) -> None:
        path = tmp_path / "s.json"
        written = em.write_strict_json(path, {
            "nan": float("nan"), "inf": float("inf"), "np": np.float32(1.5), "list": [float("nan"), 2],
            "nested": {"x": np.int64(3)}})
        assert written == {"nan": None, "inf": None, "np": 1.5, "list": [None, 2], "nested": {"x": 3}}
        assert json.loads(path.read_text(), parse_constant=_reject_constant) == written


# ---------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------

class TestTheCommandLine:
    def _stubs(self, monkeypatch) -> List[tuple]:
        calls: List[tuple] = []
        monkeypatch.setattr(em, "run", lambda config, run_name=None: calls.append(("run", config)))
        monkeypatch.setattr(em, "setup_gpu", lambda gpu_id=None: calls.append(("setup_gpu", gpu_id)))
        monkeypatch.setattr(em, "hide_gpus", lambda: calls.append(("hide_gpus", None)))
        return calls

    def test_help_prints_usage_and_does_nothing_else(self, monkeypatch, capsys) -> None:
        calls = self._stubs(monkeypatch)
        with pytest.raises(SystemExit) as stop:
            em.main(["--help"])
        assert stop.value.code == 0
        assert "usage:" in capsys.readouterr().out and calls == []

    def test_a_real_invocation_does_reach_the_stubs(self, monkeypatch) -> None:
        calls = self._stubs(monkeypatch)
        em.main([])
        assert [name for name, _ in calls] == ["hide_gpus", "run"]

    def test_gpu_is_selected_only_through_setup_gpu_and_the_default_is_cpu(self, monkeypatch) -> None:
        calls = self._stubs(monkeypatch)
        em.main(["--gpu", "0"])
        assert calls[0] == ("setup_gpu", 0) and ("hide_gpus", None) not in calls

    def test_every_flag_reaches_the_config(self, monkeypatch) -> None:
        calls = self._stubs(monkeypatch)
        em.main(["--bank-size", "111", "--chunk-size", "13", "--num-queries", "17",
                 "--sigmas-255", "5", "7.5", "--dtype", "float64", "--clip",
                 "--model-path", "m.keras", "--train-dir", "tr", "--test-dir", "te",
                 "--output-dir", "od", "--seed", "99"])
        config = next(value for name, value in calls if name == "run")
        assert config == em.OracleConfig(
            train_dir="tr", test_dir="te", output_dir="od", bank_size=111, chunk_size=13,
            num_queries=17, sigmas_255=(5.0, 7.5), dtype="float64", clip=True,
            model_path="m.keras", seed=99)

    def test_the_defaults(self) -> None:
        config = em.config_from_args(em.parse_arguments([]))
        assert config.bank_size == 16384 and config.clip is False and config.dtype == "float32"
        assert config.sigmas_255 == tuple(ev.DEFAULT_SIGMAS_255) and config.model_path is None
        assert config.output_dir == "results/mean_shift_denoiser"

    def test_help_in_a_subprocess_prints_usage_and_writes_nothing(self, tmp_path) -> None:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg")
        result = subprocess.run(
            [sys.executable, "-m", "train.bfunet.eval_empirical_mean_shift", "--help"],
            cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout
        assert list(tmp_path.iterdir()) == []
