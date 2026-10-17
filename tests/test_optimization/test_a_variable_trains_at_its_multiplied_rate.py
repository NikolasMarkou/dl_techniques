"""A variable trains at ``multiplier x`` the base rate, and only that variable.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 1.5 (D10,
``LRMultiplierAdamW``). Depth Anything (arXiv 2401.10891, section 4.1) trains
the randomly initialised decoder at 10x the pretrained encoder's rate; Keras
has no parameter groups, so the optimizer scales ``learning_rate`` inside the
per-variable ``update_step`` and the decoupled weight decay by the same factor
(PyTorch's AdamW over a group whose ``lr`` is ``k * lr``).

Every displacement here is compared between twin variables that hold the same
values and receive the same gradient, so the ratio is exactly ``k`` up to one
float32 rounding of ``k * lr``; an unscaled variable must move bit-identically
to a stock ``keras.optimizers.AdamW``.
"""

import numpy as np
import pytest
import keras

from dl_techniques.optimization import LRMultiplierAdamW, optimizer_builder

K = 10.0
LR = 1e-2


def _twins(n=2, shape=(3, 4), seed=0):
    values = np.random.default_rng(seed).normal(size=shape).astype("float32")
    return [keras.Variable(values, name=f"w{i}") for i in range(n)], values


def _ulps(start, n=4):
    """A displacement is read as ``after - start`` in float32: each of the
    assign and the subtraction rounds once at the scale of ``start``, so the
    bound is a few ulps of the start values, not a relative tolerance on a
    displacement 1e-4 times smaller than the value it is taken from."""
    return n * float(np.spacing(np.abs(start).astype(np.float32)).max())


def _one_step(optimizer, variables, grad):
    optimizer.build(variables)
    optimizer.apply_gradients([(grad, v) for v in variables])
    return [keras.ops.convert_to_numpy(v) for v in variables]


def test_a_scaled_variable_moves_k_times_further_in_one_step():
    (a, b), start = _twins()
    grad = np.random.default_rng(1).normal(size=start.shape).astype("float32")
    opt = LRMultiplierAdamW(lr_multipliers={b.path: K}, learning_rate=LR, weight_decay=0.0)
    after_a, after_b = _one_step(opt, [a, b], grad)
    moved_a, moved_b = after_a - start, after_b - start
    assert np.all(np.abs(moved_a) > 0)
    np.testing.assert_allclose(moved_b, K * moved_a, rtol=0, atol=K * _ulps(start))


def test_an_unscaled_variable_moves_exactly_as_stock_adamw():
    (a, b), start = _twins()
    (c,), _ = _twins(1)
    grad = np.random.default_rng(1).normal(size=start.shape).astype("float32")
    opt = LRMultiplierAdamW(lr_multipliers={b.path: K}, learning_rate=LR, weight_decay=0.01)
    after_a, _ = _one_step(opt, [a, b], grad)
    stock = keras.optimizers.AdamW(learning_rate=LR, weight_decay=0.01)
    (after_c,) = _one_step(stock, [c], grad)
    np.testing.assert_array_equal(after_a, after_c)


def test_the_weight_decay_is_scaled_by_the_same_multiplier():
    (a, b), start = _twins()
    zero = np.zeros(start.shape, dtype="float32")  # Adam's step is 0; only decay moves
    opt = LRMultiplierAdamW(lr_multipliers={b.path: K}, learning_rate=LR, weight_decay=0.1)
    after_a, after_b = _one_step(opt, [a, b], zero)
    np.testing.assert_allclose(after_a - start, -start * 0.1 * LR, rtol=0, atol=_ulps(start))
    np.testing.assert_allclose(after_b - start, K * (after_a - start), rtol=0, atol=K * _ulps(start))


def test_stock_fit_moves_the_scaled_branch_k_times_further():
    """Under ``compile``/``fit`` the TF backend hands ``update_step`` raw tf.Variables."""
    keras.utils.set_random_seed(0)
    x = keras.Input((4,))
    left = keras.layers.Dense(1, use_bias=False, name="left")
    right = keras.layers.Dense(1, use_bias=False, name="right")
    model = keras.Model(x, keras.layers.Add()([left(x), right(x)]))
    right.kernel.assign(left.kernel)
    start = keras.ops.convert_to_numpy(left.kernel)
    opt = LRMultiplierAdamW(lr_multipliers={right.kernel.path: K},
                            learning_rate=LR, weight_decay=0.0)
    model.compile(optimizer=opt, loss="mse", jit_compile=True)
    data = np.random.default_rng(2).normal(size=(8, 4)).astype("float32")
    model.fit(data, np.ones((8, 1), "float32"), batch_size=8, epochs=1, verbose=0)
    moved_left = keras.ops.convert_to_numpy(left.kernel) - start
    moved_right = keras.ops.convert_to_numpy(right.kernel) - start
    assert np.all(np.abs(moved_left) > 0)
    np.testing.assert_allclose(moved_right, K * moved_left, rtol=0, atol=K * _ulps(start))


def test_the_longest_path_prefix_wins_at_a_slash_boundary():
    opt = LRMultiplierAdamW(lr_multipliers={"m/dec": 10.0, "m/dec/out": 3.0, "m/enc/": 0.5})
    assert opt.multiplier_for("m/dec/conv/kernel") == 10.0
    assert opt.multiplier_for("m/dec/out/kernel") == 3.0
    assert opt.multiplier_for("m/dec") == 10.0
    assert opt.multiplier_for("m/decoder/conv/kernel") == 1.0  # a name prefix is not a path prefix
    assert opt.multiplier_for("m/enc/norm/gamma") == 0.5
    assert opt.multiplier_for("other/kernel") == 1.0


@pytest.mark.parametrize("table", [
    {"": 2.0}, {"/": 2.0}, {3: 2.0}, {"a": 0.0}, {"a": -1.0},
    {"a": float("nan")}, {"a": float("inf")}, {"a": True}, {"a": "10"},
])
def test_a_bad_table_is_refused(table):
    with pytest.raises(ValueError):
        LRMultiplierAdamW(lr_multipliers=table)


def test_the_optimizer_survives_get_config_and_from_config():
    # Free variables are named, not nested: the twins' paths are "w0" and "w1".
    table = {"w1": K, "m/enc/norm": 0.25}
    opt = LRMultiplierAdamW(lr_multipliers=table, learning_rate=LR, weight_decay=0.03,
                            global_clipnorm=1.0)
    config = opt.get_config()
    assert config["lr_multipliers"] == table
    clone = LRMultiplierAdamW.from_config(config)
    assert clone.lr_multipliers == table
    assert clone.get_config() == config
    revived = keras.optimizers.deserialize(keras.optimizers.serialize(opt))
    assert type(revived) is LRMultiplierAdamW
    assert revived.get_config() == config
    # The revived optimizer steps exactly as the original, multiplier included.
    (a, b), start = _twins()
    (c, d), _ = _twins()
    grad = np.random.default_rng(3).normal(size=start.shape).astype("float32")
    one = _one_step(opt, [a, b], grad)
    two = _one_step(revived, [c, d], grad)
    for u, w in zip(one, two):
        np.testing.assert_array_equal(u, w)
    assert not np.allclose(one[0], one[1])


def test_the_builder_returns_it_only_for_a_table():
    plain = optimizer_builder({"type": "adamw", "weight_decay": 1e-5}, 1e-3)
    assert type(plain) is keras.optimizers.AdamW
    scaled = optimizer_builder(
        {"type": "adamw", "weight_decay": 1e-5, "gradient_clipping_by_norm": 1.0,
         "lr_multipliers": {"m/dec": 10.0}}, 1e-3)
    assert type(scaled) is LRMultiplierAdamW
    assert scaled.lr_multipliers == {"m/dec": 10.0}
    assert scaled.global_clipnorm == 1.0
    assert scaled.weight_decay == 1e-5


@pytest.mark.parametrize("kind", ["adam", "sgd", "gefen"])
def test_the_builder_refuses_a_table_it_would_ignore(kind):
    with pytest.raises(ValueError, match="lr_multipliers"):
        optimizer_builder({"type": kind, "lr_multipliers": {"m/dec": 10.0}}, 1e-3)


def test_it_is_registered_under_its_package(registration_contract):
    registration_contract(LRMultiplierAdamW)
