"""The DFSA-STE trainer's periodic quick eval does not reseed the training stream.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 12.6(a) (D-248).
``train.nam.train_dfsa_ste._eval_quick`` drew its fixed test expressions after
``random.seed(42)`` on the global ``random`` module. The training batches come from
``train.nam.data_generator.generate_curriculum_batch``, whose expression operands and
operators are drawn from that same global module, so every eval (each
``--eval-interval`` steps) reset the training stream: each window between two evals
trained on the operand sequence of the one before.
"""

import random

import numpy as np

from train.nam import train_dfsa_ste


class _Tokenizer:
    def encode_batch(self, expressions):
        return np.zeros((len(expressions), 4), dtype=np.int32)


class _Model:
    """Answers every expression with 0.0, which is all ``_eval_quick`` reads."""

    def multi_reduce(self, ids, max_steps, training):
        import tensorflow as tf
        return [{"result": tf.zeros((int(ids.shape[0]), 1))}]


def test_the_quick_eval_does_not_touch_the_global_random_state():
    random.seed(1234)
    expected_next = [random.random() for _ in range(5)]

    random.seed(1234)
    train_dfsa_ste._eval_quick(_Model(), _Tokenizer())
    assert [random.random() for _ in range(5)] == expected_next


def test_the_quick_eval_still_draws_the_same_expressions_every_time():
    """Its own fixed seed is kept: two evals score the same expression set."""
    seen = []

    class _Recording(_Tokenizer):
        def encode_batch(self, expressions):
            seen.append(tuple(expressions))
            return super().encode_batch(expressions)

    train_dfsa_ste._eval_quick(_Model(), _Recording())
    first = list(seen)
    seen.clear()
    random.seed(99)
    train_dfsa_ste._eval_quick(_Model(), _Recording())
    assert seen == first and len(first) > 50
