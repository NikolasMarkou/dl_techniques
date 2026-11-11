"""``TubeMaskGenerator`` takes an optional ``seed``; without it nothing changes.

Plan plan-2026-11-11T131349-13b28b37 D-020: the JEPA visualization callbacks need ONE
fixed mask that does not advance the global Keras seed generator, so ``call`` accepts
``seed`` and forwards it to ``random.uniform`` (an int seed is stateless). The default
path (``seed=None``) draws from the global generator exactly as before.
"""

from __future__ import annotations

import keras
import numpy as np

from dl_techniques.models.world_models.video_jepa.masking import TubeMaskGenerator


def _state() -> np.ndarray:
    return np.array(keras.src.random.seed_generator.global_seed_generator().state)


def _reset(state: np.ndarray) -> None:
    keras.src.random.seed_generator.global_seed_generator().state.assign(state)


class TestSeededMask:

    def test_the_same_seed_gives_the_same_mask(self):
        gen = TubeMaskGenerator(mask_ratio=0.5, patches_per_side=4)
        a = np.asarray(gen(3, training=False, seed=7))
        b = np.asarray(gen(3, training=False, seed=7))
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_give_different_masks(self):
        gen = TubeMaskGenerator(mask_ratio=0.5, patches_per_side=4)
        a = np.asarray(gen(3, training=False, seed=7))
        b = np.asarray(gen(3, training=False, seed=8))
        assert not np.array_equal(a, b)

    # DECISION plan-2026-11-11T131349-13b28b37/D-020 guard: an int seed leaves the global generator alone
    def test_a_seeded_call_leaves_the_global_seed_generator_alone(self):
        gen = TubeMaskGenerator(mask_ratio=0.5, patches_per_side=4)
        before = _state()
        gen(3, training=False, seed=7)
        np.testing.assert_array_equal(_state(), before)

    def test_the_seeded_mask_keeps_its_exact_cardinality(self):
        gen = TubeMaskGenerator(mask_ratio=0.5, patches_per_side=4)
        mask = np.asarray(gen(5, training=False, seed=3))
        assert mask.shape == (5, 4, 4)
        np.testing.assert_array_equal(mask.sum(axis=(1, 2)), np.full(5, gen.num_masked))


class TestDefaultPathIsUnchanged:

    def test_the_unseeded_call_is_the_global_argsort_of_uniform_sampler(self):
        gen = TubeMaskGenerator(mask_ratio=0.5, patches_per_side=4)
        before = _state()
        mask = np.asarray(gen(3, training=False))
        after = _state()
        assert not np.array_equal(before, after), "the default path must draw from the global generator"

        _reset(before)
        noise = keras.random.uniform(shape=(3, 16))
        rank = keras.ops.argsort(keras.ops.argsort(noise, axis=-1), axis=-1)
        expected = np.asarray(keras.ops.cast(rank < gen.num_masked, "float32")).reshape(3, 4, 4)
        np.testing.assert_array_equal(mask, expected)

    def test_two_unseeded_calls_differ(self):
        gen = TubeMaskGenerator(mask_ratio=0.5, patches_per_side=4)
        assert not np.array_equal(np.asarray(gen(8)), np.asarray(gen(8)))

    def test_a_zero_ratio_is_all_zeros_with_or_without_a_seed(self):
        gen = TubeMaskGenerator(mask_ratio=0.0, patches_per_side=4)
        assert float(np.asarray(gen(2, seed=1)).sum()) == 0.0
        assert float(np.asarray(gen(2)).sum()) == 0.0
