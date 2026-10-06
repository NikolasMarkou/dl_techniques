"""What the ``isinstance(inputs, keras.KerasTensor)`` guard in ``call`` does.

The claim that was withdrawn
----------------------------
``SpatialSmoothness.call`` skips ``add_loss`` when its input is a
``keras.KerasTensor``. Its comment said: ``add_loss`` accepts a ``KerasTensor``
without complaint, stores it, that stored tensor never becomes a value, and so a
training loss would silently lack the spatial term while nothing raised.

**Not what happens. The comment has been corrected.** With the guard REMOVED, the
spatial term is absent from the loss on every path tried, exactly as it is with
the guard in place.

What is stable, and what is not
-------------------------------
Two of the measurements behind the original claim turned out to be
**path-dependent**, which is itself the useful finding:

* Whether a subclassed model hands its tap a ``KerasTensor`` or an eager tensor
  varied between a standalone script and this test run under pytest, with
  identical code. So "the guard is reachable" is NOT asserted here: it is not a
  stable property of the code.
* ``add_loss`` called inside a ``tf.function`` **does** store the concrete traced
  value -- and that is CORRECT. It is how the training loss gets its spatial term.
  The original comment's "the stored tensor never becomes a value" is false for
  the path that matters most.

The stable facts, all asserted below:

1. A symbolic pass adds no loss.
2. A symbolic pass still returns its input -- the guard is not an early ``return``.
3. The first EAGER pass adds exactly one loss, so the guard skips symbolic inputs
   and nothing else.
4. ``TopoLM``'s ``build()`` invokes no tap at all, so its lifecycle never depends
   on the guard either way.

Net: the guard is belt-and-braces. Deleting it would change no assertion in this
file, which is exactly why none of these tests pretend otherwise.
"""

import numpy as np
import tensorflow as tf

import keras
from keras import ops

from dl_techniques.layers.regularization.spatial_smoothness import SpatialSmoothness

NUM_UNITS = 784
RADIUS = 5


def _built_tap(alpha=2.5):
    """A tap built EAGERLY, with its losses cleared."""
    tap = SpatialSmoothness(alpha=alpha, radius=RADIUS, seed=0, name="tap")
    tap(np.zeros((1, 4, NUM_UNITS), dtype="float32"), training=False)
    tap.losses.clear()
    return tap


def _parent(tap):
    """A subclassed model whose ``call`` is the tap's only caller."""

    class _Parent(keras.Model):
        def __init__(self, child):
            super().__init__()
            self.child = child

        def call(self, inputs, training=None):
            return self.child(inputs, training=training)

    return _Parent(tap)


class TestTheStableBehaviour:
    def test_a_symbolic_pass_adds_no_loss(self):
        """The shipped behaviour on the tracing path.

        Read IMMEDIATELY after the call: Keras clears ``losses`` at the start of
        every call, so a count read later would be ``0`` for any implementation.
        """
        tap = _built_tap()
        _parent(tap)(keras.Input(shape=(4, NUM_UNITS)), training=True)
        assert tap.losses == []

    def test_a_symbolic_pass_returns_the_input_unchanged(self):
        """Skipping the loss must not skip the forward pass.

        A guard written as an early ``return`` satisfies the test above.
        """
        tap = _built_tap()
        output = _parent(tap)(keras.Input(shape=(4, NUM_UNITS)), training=True)
        assert ops.shape(output)[-1] == NUM_UNITS

    def test_the_first_eager_pass_adds_exactly_one_loss(self):
        """The other arm: the guard skips SYMBOLIC inputs, not eager ones.

        A guard with the condition inverted satisfies
        :meth:`test_a_symbolic_pass_adds_no_loss` by never adding a loss at all.
        """
        tap = _built_tap()
        parent = _parent(tap)
        parent(keras.Input(shape=(4, NUM_UNITS)), training=True)
        assert tap.losses == []

        parent(
            np.random.default_rng(0).normal(size=(2, 4, NUM_UNITS)).astype(
                "float32"
            ),
            training=True,
        )
        assert len(tap.losses) == 1
        assert np.isfinite(float(tap.losses[0]))

    def test_a_tf_function_step_adds_the_loss(self):
        """The path a real training step takes, and the one that matters.

        Inside ``tf.function`` the tap receives a concrete traced tensor, so the
        spatial term IS added and stored. The withdrawn comment claimed the
        opposite for this path; it is the comment that was wrong.
        """
        tap = _built_tap()

        @tf.function(
            input_signature=[tf.TensorSpec([None, 4, NUM_UNITS], tf.float32)]
        )
        def traced(x):
            return tap(x, training=True)

        traced(tf.zeros((2, 4, NUM_UNITS), dtype=tf.float32))
        assert len(tap.losses) == 1, (
            f"a tf.function step added {len(tap.losses)} losses; the spatial "
            f"term would be missing from the training objective"
        )


class TestTheModelDoesNotDependOnTheGuard:
    def test_build_invokes_no_tap_and_the_first_step_gets_one_loss_each(
        self, monkeypatch
    ):
        """Why the withdrawn claim could not bite ``TopoLM`` in any case.

        ``build()`` materialises the sub-layer tree without a forward pass, so
        there is no tracing step in which a tap could be handed a
        ``KerasTensor``. MEASURED: zero tap calls after ``build()``, then one
        eager call and one loss per tap on the first ``training=True`` step.
        """
        from dl_techniques.models.language.topolm import TopoLM

        calls = []
        original = SpatialSmoothness.call

        def counting(self, inputs, training=None):
            calls.append(
                "symbolic" if isinstance(inputs, keras.KerasTensor) else "eager"
            )
            return original(self, inputs, training=training)

        monkeypatch.setattr(SpatialSmoothness, "call", counting)

        model = TopoLM(
            vocab_size=200, embed_dim=256, depth=2, num_heads=4,
            max_seq_len=32, ffn_intermediate_size=512, radius=3,
            alpha=2.5, name="topolm",
        )
        model.build((None, 12))
        assert calls == [], (
            f"build() invoked {len(calls)} tap call(s) ({calls}); it is supposed "
            f"to materialise the tree without a forward pass"
        )

        tokens = np.random.default_rng(0).integers(
            0, 200, size=(2, 12)
        ).astype("int32")
        model(tokens, training=True)

        assert "symbolic" not in calls, calls
        assert calls.count("eager") == 4, calls
        assert len(model.losses) == 4