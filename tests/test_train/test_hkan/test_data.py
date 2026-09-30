"""Tests of ``src/train/hkan/data.py``: the synthetic targets and the CSV loader.

What is pinned here, and the mutation each group is meant to catch:

- the input count, input range, sizes and noise level of every target are the
  paper's, as literals typed in this file (a changed table entry, including an
  input range, which no shape assertion can see);
- the scaling statistics are the TRAIN statistics (statistics taken from train and
  test together keep every test value inside ``[0, 1]``);
- EVERY target and input is min-max scaled, TF1 and TF2 included (``decisions.md``
  D-041 of plan-2026-09-30T082355-4d999dbc; an unscaled TF1 no longer equals the
  scaled formula);
- the TF2 noise is U(-0.2, 0.2) on the SCALED train targets, the scaling statistics
  are those of the noiseless train targets, and the test targets are the scaled
  noiseless formula exactly;
- every target formula, at points worked out by hand in the test body;
- the CSV loader's refusals and an exact round trip.

Nothing here touches a model, a GPU or the file system outside ``tmp_path``.
"""

import math

import numpy as np
import pytest

from train.hkan.data import TARGETS, load_csv_dataset, make_dataset, minmax_scale

#: name -> (inputs, train rows, test rows, input low, input high, train noise), from
#: the paper's Table VI and Section VI-A. Typed here on purpose: reading it off
#: ``TARGETS`` would compare the module with itself.
PAPER = {
    "tf1": (2, 5000, 10000, 0.0, 1.0, 0.0),
    "tf2": (2, 5000, 10000, 0.0, 1.0, 0.2),
    "tf3": (2, 5000, 10000, -500.0, 500.0, 0.0),
    "tf4": (10, 3750, 1250, -4.0, 4.0, 0.0),
    "tf5": (2, 5000, 10000, 0.0, 3.141592653589793, 0.0),
    "tf5_5": (5, 7500, 2500, 0.0, 3.141592653589793, 0.0),
}
NOISELESS = sorted(name for name, row in PAPER.items() if row[5] == 0.0)
KEYS = {"x_train", "y_train", "x_test", "y_test"}


def _expected(name: str, seed: int, num_train: int, num_test: int):
    """The data set rebuilt from the literals above, without ``make_dataset``.

    Raw inputs uniform on the literal range (train drawn first), the target
    function, min-max scaling with the train statistics. Returns the scaled
    arrays with the NOISELESS train target, and the generator, whose next draw
    is the TF2 noise.
    """
    inputs, _, _, low, high, _ = PAPER[name]
    rng = np.random.default_rng(seed)
    x_train = rng.uniform(low, high, (num_train, inputs))
    x_test = rng.uniform(low, high, (num_test, inputs))
    function = TARGETS[name].function  # pinned at hand-computed points below
    y_train, y_test = function(x_train), function(x_test)

    def scale(train, other):
        minimum = train.min(axis=0)
        span = train.max(axis=0) - minimum
        return (train - minimum) / span, (other - minimum) / span

    return (*scale(x_train, x_test), *scale(y_train, y_test), rng)


def _tf1_formula(x: np.ndarray) -> np.ndarray:
    return (2.0 * x[:, 0] - 1.0) * (2.0 * x[:, 1] - 1.0)


def _tf2_formula(x: np.ndarray) -> np.ndarray:
    return np.sum(np.sin(20.0 * np.exp(x)) * x ** 2, axis=1)


# ---------------------------------------------------------------------
# sizes, shapes, ranges
# ---------------------------------------------------------------------


def test_the_target_table_holds_exactly_the_six_paper_targets() -> None:
    assert set(TARGETS) == set(PAPER)


@pytest.mark.parametrize("name", sorted(PAPER))
def test_default_sizes_and_input_counts_are_the_papers(name) -> None:
    inputs, num_train, num_test = PAPER[name][:3]
    data = make_dataset(name, seed=0)
    assert set(data) == KEYS
    assert data["x_train"].shape == (num_train, inputs)
    assert data["y_train"].shape == (num_train,)
    assert data["x_test"].shape == (num_test, inputs)
    assert data["y_test"].shape == (num_test,)
    for key, array in data.items():
        assert array.dtype == np.float64, key
        assert np.all(np.isfinite(array)), key


@pytest.mark.parametrize("name", sorted(PAPER))
def test_explicit_sizes_override_the_defaults(name) -> None:
    data = make_dataset(name, seed=0, num_train=17, num_test=9)
    assert data["x_train"].shape == (17, PAPER[name][0])
    assert data["y_test"].shape == (9,)


@pytest.mark.parametrize("name", sorted(PAPER))
def test_every_table_entry_is_the_papers(name) -> None:
    """Input count, range, sizes and noise, against literals typed in this file.

    The ranges are compared as numbers AND by what they do: the raw inputs of a
    target whose ``high`` was changed from pi to 3 have the same shape and, once
    scaled, the same [0, 1] span, so only the values can show it.
    """
    inputs, num_train, num_test, low, high, noise = PAPER[name]
    spec = TARGETS[name]
    assert (spec.num_inputs, spec.num_train, spec.num_test) == (inputs, num_train, num_test)
    assert (spec.low, spec.high) == (low, high), f"{name}: input range {spec.low}, {spec.high}"
    assert spec.train_noise == noise
    data = make_dataset(name, seed=0, num_train=60, num_test=50)
    x_train, x_test, _, y_test, _ = _expected(name, 0, 60, 50)
    np.testing.assert_array_equal(data["x_train"], x_train)
    np.testing.assert_array_equal(data["x_test"], x_test)
    np.testing.assert_array_equal(
        data["y_test"], y_test,
        err_msg=f"{name}: test targets are not the target function on the paper's input range")


@pytest.mark.parametrize("name", sorted(PAPER))
def test_every_input_is_scaled_to_exactly_zero_to_one_on_train(name) -> None:
    data = make_dataset(name, seed=0)
    np.testing.assert_array_equal(data["x_train"].min(axis=0), 0.0)
    np.testing.assert_array_equal(data["x_train"].max(axis=0), 1.0)


@pytest.mark.parametrize("name", NOISELESS)
def test_every_noiseless_target_spans_exactly_zero_to_one_on_train(name) -> None:
    """TF1 included: unscaled, its train targets span about [-1, 1]."""
    data = make_dataset(name, seed=0)
    assert data["y_train"].min() == 0.0, f"{name}: train target minimum {data['y_train'].min()}"
    assert data["y_train"].max() == 1.0, f"{name}: train target maximum {data['y_train'].max()}"


# ---------------------------------------------------------------------
# scaling statistics are the TRAIN statistics
# ---------------------------------------------------------------------


def test_minmax_scale_uses_train_statistics_so_a_wider_test_leaves_the_unit_range() -> None:
    train = np.array([[2.0, 10.0], [4.0, 30.0], [3.0, 20.0]])
    test = np.array([[0.0, 50.0], [6.0, 10.0]])
    scaled_train, scaled_test = minmax_scale(train, test)
    # train: column 0 has min 2 and range 2, column 1 has min 10 and range 20
    np.testing.assert_array_equal(scaled_train, [[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    # test, same map: (0-2)/2, (50-10)/20, (6-2)/2, (10-10)/20
    np.testing.assert_array_equal(scaled_test, [[-1.0, 2.0], [2.0, 0.0]])


def test_minmax_scale_handles_one_dimensional_targets_and_several_others() -> None:
    train = np.array([1.0, 5.0, 3.0])
    a, b, c = minmax_scale(train, np.array([9.0]), np.array([-3.0, 1.0]))
    np.testing.assert_array_equal(a, [0.0, 1.0, 0.5])
    np.testing.assert_array_equal(b, [2.0])        # (9-1)/4
    np.testing.assert_array_equal(c, [-1.0, 0.0])  # (-3-1)/4, (1-1)/4


def test_minmax_scale_maps_a_constant_column_to_zero() -> None:
    train = np.array([[7.0, 1.0], [7.0, 3.0]])
    scaled_train, scaled_test = minmax_scale(train, np.array([[9.0, 2.0]]))
    np.testing.assert_array_equal(scaled_train, [[0.0, 0.0], [0.0, 1.0]])
    np.testing.assert_array_equal(scaled_test, [[2.0, 0.5]])  # (9-7)/1: range 0 becomes 1


@pytest.mark.parametrize("name", sorted(PAPER))
def test_a_scaled_target_with_a_wider_test_split_maps_test_outside_zero_to_one(name) -> None:
    """5 train rows against 4000 test rows: the test split is certainly wider."""
    data = make_dataset(name, seed=3, num_train=5, num_test=4000)
    np.testing.assert_array_equal(data["x_train"].min(axis=0), 0.0)
    np.testing.assert_array_equal(data["x_train"].max(axis=0), 1.0)
    if name in NOISELESS:
        assert data["y_train"].min() == 0.0 and data["y_train"].max() == 1.0
    assert data["x_test"].min() < 0.0 and data["x_test"].max() > 1.0, (
        "x_test lies inside [0, 1]: the input statistics were not taken from train alone"
    )
    assert data["y_test"].min() < 0.0 or data["y_test"].max() > 1.0, (
        "y_test lies inside [0, 1]: the target statistics were not taken from train alone"
    )


# ---------------------------------------------------------------------
# TF1 / TF2: scaled like every other target; TF2 noise after scaling, train only
# ---------------------------------------------------------------------


def test_tf1_targets_are_the_scaled_formula_exactly_on_both_splits() -> None:
    data = make_dataset("tf1", seed=0)
    _, _, y_train, y_test, _ = _expected("tf1", 0, 5000, 10000)
    np.testing.assert_array_equal(data["y_train"], y_train)
    np.testing.assert_array_equal(data["y_test"], y_test)
    # the formula itself, written out: unscaled it reaches below -0.9
    assert data["y_test"].min() > -0.01, (
        f"TF1 test targets reach {data['y_test'].min():.3f}: the target is not scaled"
    )


def test_tf2_test_targets_are_the_scaled_noiseless_formula_exactly() -> None:
    """Scaled with the statistics of the NOISELESS train targets.

    Statistics taken from the noisy train targets, noise on the test targets and
    a missing scaling each break the exact equality.
    """
    data = make_dataset("tf2", seed=0)
    _, _, _, y_test, _ = _expected("tf2", 0, 5000, 10000)
    np.testing.assert_array_equal(
        data["y_test"], y_test,
        err_msg="TF2 test targets differ from the noiseless formula scaled with the "
                "noiseless train statistics",
    )


def test_tf2_noise_is_uniform_on_the_scaled_train_targets() -> None:
    """U(-0.2, 0.2) relative to a unit range, not shrunk by the target's range.

    Noise added BEFORE the scaling would be divided by the range of the (noisy)
    unscaled target, about 3.7, and reach only about 0.055 here.
    """
    data = make_dataset("tf2", seed=0)
    _, _, scaled_noiseless, _, rng = _expected("tf2", 0, 5000, 10000)
    noise = data["y_train"] - scaled_noiseless
    assert np.max(np.abs(noise)) <= 0.2 + 1e-12
    assert np.max(np.abs(noise)) > 0.19, (
        f"the largest noise on the scaled train target is {np.max(np.abs(noise)):.4f}; "
        "5000 draws of U(-0.2, 0.2) reach beyond 0.19 (noise added before scaling?)"
    )
    # U(-0.2, 0.2) has standard deviation 0.4 / sqrt(12) = 0.11547
    assert 0.105 < float(np.std(noise)) < 0.125
    assert abs(float(np.mean(noise))) < 0.01
    assert len(np.unique(np.round(noise, 6))) > 4000, "the noise is (nearly) constant"
    # and it is the generator's next draw after the inputs, added to the scaled target
    np.testing.assert_array_equal(
        data["y_train"], scaled_noiseless + rng.uniform(-0.2, 0.2, 5000))
    # so a noisy train target can leave [0, 1], by less than 0.2
    assert data["y_train"].min() < 0.0 and data["y_train"].max() > 1.0
    assert data["y_train"].min() >= -0.2 and data["y_train"].max() <= 1.2


# ---------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(PAPER))
@pytest.mark.parametrize("seed", [0, 7])
def test_the_same_seed_gives_the_same_data(name, seed) -> None:
    first = make_dataset(name, seed=seed, num_train=50, num_test=40)
    second = make_dataset(name, seed=seed, num_train=50, num_test=40)
    for key in KEYS:
        np.testing.assert_array_equal(first[key], second[key], err_msg=key)


@pytest.mark.parametrize("name", sorted(PAPER))
def test_different_seeds_give_different_data_and_zero_is_a_seed(name) -> None:
    zero = make_dataset(name, seed=0, num_train=50, num_test=40)
    one = make_dataset(name, seed=1, num_train=50, num_test=40)
    unseeded = make_dataset(name, seed=None, num_train=50, num_test=40)
    for key in KEYS:
        assert not np.array_equal(zero[key], one[key]), key
        assert not np.array_equal(zero[key], unseeded[key]), (
            f"{key}: seed=0 behaved like seed=None"
        )


def test_train_and_test_are_different_draws() -> None:
    data = make_dataset("tf1", seed=0, num_train=40, num_test=40)
    assert not np.array_equal(data["x_train"], data["x_test"])


# ---------------------------------------------------------------------
# the formulas, at hand-computed points
# ---------------------------------------------------------------------


def test_tf1_at_hand_computed_points() -> None:
    f = TARGETS["tf1"].function
    # (2*0.25 - 1) * (2*1 - 1) = -0.5 * 1 ; (2*0.75 - 1) * (2*0.75 - 1) = 0.5 * 0.5
    np.testing.assert_allclose(
        f(np.array([[0.25, 1.0], [0.75, 0.75], [0.5, 0.123]])), [-0.5, 0.25, 0.0],
        rtol=0, atol=1e-15)


def test_tf2_at_hand_computed_points() -> None:
    f = TARGETS["tf2"].function
    # x = (0, 1):   sin(20*e^0)*0^2 + sin(20*e^1)*1^2 = sin(20 e)
    # x = (0.5, 0): sin(20*e^0.5)*0.25 + 0
    # x = (1, 1):   2 sin(20 e)
    expected = [
        math.sin(20.0 * math.e),
        math.sin(20.0 * math.exp(0.5)) * 0.25,
        2.0 * math.sin(20.0 * math.e),
    ]
    np.testing.assert_allclose(
        f(np.array([[0.0, 1.0], [0.5, 0.0], [1.0, 1.0]])), expected, rtol=1e-13, atol=0)


def test_tf3_at_hand_computed_points() -> None:
    f = TARGETS["tf3"].function
    # x = (100, -400): -(100*sin(sqrt(100)) + (-400)*sin(sqrt(400)))
    #                = -100 sin(10) + 400 sin(20)
    # x = (0, 25):     -(0 + 25 sin(5))
    expected = [
        -100.0 * math.sin(10.0) + 400.0 * math.sin(20.0),
        -25.0 * math.sin(5.0),
    ]
    np.testing.assert_allclose(
        f(np.array([[100.0, -400.0], [0.0, 25.0]])), expected, rtol=1e-13, atol=0)


def test_tf4_at_hand_computed_points() -> None:
    f = TARGETS["tf4"].function
    # x = (3, 4, 0, ..., 0): radius 5, 1 - cos(10 pi) + 0.5 = 1 - 1 + 0.5 = 0.5
    # x = (0.5, 0, ..., 0):  radius 0.5, 1 - cos(pi) + 0.05 = 1 + 1 + 0.05 = 2.05
    # x = 0:                 1 - cos(0) + 0 = 0
    points = np.zeros((3, 10))
    points[0, 0], points[0, 1] = 3.0, 4.0
    points[1, 6] = 0.5
    np.testing.assert_allclose(f(points), [0.5, 2.05, 0.0], rtol=0, atol=1e-12)


def test_tf5_at_hand_computed_points() -> None:
    f = TARGETS["tf5"].function
    half_pi = math.pi / 2.0
    # x = (pi/2, 0): i=1: sin(pi/2) * sin(1*(pi/2)^2/pi)^20 = sin(pi/4)^20 = 2^-10
    #                i=2: sin(0) * ... = 0                         -> -1/1024
    # x = (0, pi/2): i=2: sin(pi/2) * sin(2*(pi/2)^2/pi)^20 = sin(pi/2)^20 = 1 -> -1
    # x = (pi/2, pi/2): -(1/1024 + 1)
    np.testing.assert_allclose(
        f(np.array([[half_pi, 0.0], [0.0, half_pi], [half_pi, half_pi]])),
        [-1.0 / 1024.0, -1.0, -1025.0 / 1024.0], rtol=1e-12, atol=0)


def test_tf5_5_uses_the_input_index_up_to_five() -> None:
    f = TARGETS["tf5_5"].function
    assert TARGETS["tf5_5"].num_inputs == 5
    # only x_4 = pi/2 is non-zero: sin(pi/2) * sin(4*(pi/2)^2/pi)^20 = sin(pi)^20 = 0
    # only x_3 = pi/2:             sin(pi/2) * sin(3*pi/4)^20 = (1/sqrt 2)^20 = 2^-10
    points = np.zeros((2, 5))
    points[0, 3] = math.pi / 2.0
    points[1, 2] = math.pi / 2.0
    np.testing.assert_allclose(f(points), [0.0, -1.0 / 1024.0], rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------


def test_an_unknown_target_name_raises() -> None:
    with pytest.raises(ValueError, match="unknown dataset 'tf9'"):
        make_dataset("tf9", seed=0)


@pytest.mark.parametrize("sizes", [{"num_train": 1}, {"num_test": 1}, {"num_train": 0}])
def test_fewer_than_two_rows_raises(sizes) -> None:
    with pytest.raises(ValueError, match="must be >= 2"):
        make_dataset("tf1", seed=0, **sizes)


# ---------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------


def _write(path, table) -> str:
    np.savetxt(path, np.asarray(table, dtype=np.float64), delimiter=",", fmt="%.17g")
    return str(path)


@pytest.fixture
def csv_pair(tmp_path):
    """A train file and a WIDER test file, three inputs and a target."""
    rng = np.random.default_rng(11)
    train = rng.uniform(-2.0, 3.0, (12, 4))
    test = rng.uniform(-6.0, 9.0, (30, 4))
    return (_write(tmp_path / "train.csv", train), _write(tmp_path / "test.csv", test),
            train, test)


def test_csv_round_trip_without_scale_is_exact(csv_pair) -> None:
    train_csv, test_csv, train, test = csv_pair
    data = load_csv_dataset(train_csv, test_csv)
    assert set(data) == KEYS
    np.testing.assert_array_equal(data["x_train"], train[:, :-1])
    np.testing.assert_array_equal(data["y_train"], train[:, -1])
    np.testing.assert_array_equal(data["x_test"], test[:, :-1])
    np.testing.assert_array_equal(data["y_test"], test[:, -1])
    assert all(array.dtype == np.float64 for array in data.values())


def test_csv_round_trip_with_scale_uses_the_train_files_statistics(csv_pair) -> None:
    train_csv, test_csv, train, test = csv_pair
    data = load_csv_dataset(train_csv, test_csv, scale=True)
    low, span = train.min(axis=0), train.max(axis=0) - train.min(axis=0)
    np.testing.assert_allclose(data["x_train"], ((train - low) / span)[:, :-1], rtol=0, atol=1e-15)
    np.testing.assert_allclose(data["y_train"], ((train - low) / span)[:, -1], rtol=0, atol=1e-15)
    np.testing.assert_allclose(data["x_test"], ((test - low) / span)[:, :-1], rtol=0, atol=1e-14)
    np.testing.assert_allclose(data["y_test"], ((test - low) / span)[:, -1], rtol=0, atol=1e-14)
    assert data["x_train"].min() == 0.0 and data["x_train"].max() == 1.0
    assert data["x_test"].min() < 0.0 and data["x_test"].max() > 1.0
    assert data["y_test"].min() < 0.0 and data["y_test"].max() > 1.0


def test_csv_missing_file_raises(csv_pair, tmp_path) -> None:
    train_csv, test_csv, _, _ = csv_pair
    with pytest.raises(ValueError, match="CSV file not found"):
        load_csv_dataset(tmp_path / "absent.csv", test_csv)
    with pytest.raises(ValueError, match="CSV file not found"):
        load_csv_dataset(train_csv, tmp_path / "absent.csv")


def test_csv_with_one_column_raises(csv_pair, tmp_path) -> None:
    _, test_csv, _, _ = csv_pair
    one_column = _write(tmp_path / "one.csv", [[1.0], [2.0], [3.0]])
    with pytest.raises(ValueError, match="at least 2 rows and 2 columns"):
        load_csv_dataset(one_column, test_csv)


def test_csv_with_one_row_raises(csv_pair, tmp_path) -> None:
    train_csv, _, _, _ = csv_pair
    one_row = _write(tmp_path / "row.csv", [[1.0, 2.0, 3.0, 4.0]])
    with pytest.raises(ValueError, match="at least 2 rows and 2 columns"):
        load_csv_dataset(train_csv, one_row)


def test_csv_with_a_non_numeric_cell_raises(csv_pair, tmp_path) -> None:
    _, test_csv, _, _ = csv_pair
    bad = tmp_path / "bad.csv"
    bad.write_text("1.0,2.0,3.0,4.0\n1.0,abc,3.0,4.0\n")
    with pytest.raises(ValueError, match="is not a numeric comma-separated file"):
        load_csv_dataset(bad, test_csv)


def test_csv_with_a_header_row_raises(csv_pair, tmp_path) -> None:
    _, test_csv, _, _ = csv_pair
    bad = tmp_path / "header.csv"
    bad.write_text("a,b,c,y\n1.0,2.0,3.0,4.0\n1.0,2.5,3.0,4.0\n")
    with pytest.raises(ValueError, match="is not a numeric comma-separated file"):
        load_csv_dataset(bad, test_csv)


@pytest.mark.parametrize("cell", ["nan", "inf", "-inf"])
def test_csv_with_a_non_finite_value_raises(csv_pair, tmp_path, cell) -> None:
    train_csv, _, _, _ = csv_pair
    bad = tmp_path / "nonfinite.csv"
    bad.write_text(f"1.0,2.0,3.0,4.0\n1.0,{cell},3.0,4.0\n")
    with pytest.raises(ValueError, match="holds a non-finite value"):
        load_csv_dataset(train_csv, bad)


def test_csv_with_mismatched_column_counts_raises(csv_pair, tmp_path) -> None:
    train_csv, _, _, _ = csv_pair
    narrow = _write(tmp_path / "narrow.csv", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with pytest.raises(ValueError, match="train has 4 columns but test has 3"):
        load_csv_dataset(train_csv, narrow)
