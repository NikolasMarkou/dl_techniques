"""
End-to-end integration gate for `MobileClipV2Model`.

Everything in the sibling modules is a unit-level pin. This module is the only
place where the WHOLE pipeline runs: build -> forward -> save/reload -> a real
`fit()` under `@tf.function` -> gradients -> the graph-vs-eager control.

Five things are exercised here that nothing else exercises:

1.  **Both stage topologies at the REAL 256px geometry** — a 4-stage variant
    (`mobileclip2_s0`, stem/4 then 64->64->32->16->8) and a 5-stage variant
    (`mobileclip2_s3`, ... ->8->4). The 5-stage tower is the one whose last
    attention stage runs a 7x7 RepCPE depthwise kernel over a 4x4 feature map.
2.  **A `fit()` step.** `fit()` traces `train_step` into a `@tf.function`; every
    other test in this package is eager. The repo has MEASURED a refactor that
    is bit-identical eager and 4.2e-04 apart under `tf.function`, so an
    eager-only claim does not transfer to the regime training actually uses.
3.  **The real `CLIPContrastiveLoss`**, unmodified. It expects a
    `{'logits_per_image', 'logits_per_text'}` dict (or a 2-tuple); our model's
    `call` returns a 5-key dict. The bridge lives HERE (`_StackedLogitsModel` +
    `_StackedClipLoss`) — neither the model's output contract nor the loss is
    changed to suit the test. Training uses STOCK `fit()`; there is no custom
    `train_step` anywhere in this file.
4.  **Gradients reaching every trainable weight of both towers** after that
    traced step, with every failing weight NAMED.
5.  **The within-version eager-vs-graph delta**, MEASURED on the same unchanged
    model and reported, not asserted to be exactly zero.

Reduction discipline (see the step-8 correction in the plan's findings: a
depth-1/width-1 "fast" config made a real causal-mask mechanism structurally
unobservable and pinned it at exactly 0.0):

  PRESERVED — the real 256px input and therefore the real per-stage spatial
  geometry including the 4x4 map of the 5-stage tower; the full stage topology
  (4 vs 5 stages, downsample flags, RepCPE placement); at least one `attention`
  token-mixer stage per tower (two for the 5-stage one); `text_layers >= 2`, so
  the text tower has real inter-layer composition and a causal mask has
  somewhere to act; both norm families (`batch_norm` for mci0,
  `layer_norm` for mci3).

  DROPPED — depth (1 block per stage instead of 2..24), width (32..512 instead
  of 64..1536), text width/vocab/context length. Any claim that depends on
  depth or width (parameter counts, capacity, published accuracy) is NOT
  testable here and is not asserted.
"""

import os
import tempfile

import keras
import numpy as np
import tensorflow as tf
from keras import ops

from dl_techniques.losses.clip_contrastive_loss import CLIPContrastiveLoss
from dl_techniques.models.mobile_clip.mobile_clip_v2 import MobileClipV2Model

# ---------------------------------------------------------------------
# Reduced-but-representative configurations.
# ---------------------------------------------------------------------

#: The REAL MobileCLIP2 image resolution. Not reduced — the whole point of the
#: 4-stage / 5-stage split is the spatial geometry it produces at 256px.
_IMG = 256

_BATCH = 4
_VOCAB = 96
_SEQ = 16
_EMBED = 32

#: Expected per-stage spatial sizes at 256px, stem included.
_FOUR_STAGE_GEOMETRY = (64, 64, 32, 16, 8)
_FIVE_STAGE_GEOMETRY = (64, 64, 32, 16, 8, 4)

_OUTPUT_KEYS = {
    'image_features',
    'text_features',
    'logits_per_image',
    'logits_per_text',
    'logit_scale',
}


def _tower_kwargs(num_stages: int) -> dict:
    """Depth/width reduction that keeps every structural feature alive.

    Only `layers` and `embed_dims` are overridden; `token_mixers`,
    `downsamples`, `pos_embs`, `se_downsamples`, `mlp_ratios`, `norm_layer` and
    `stem_use_scale_branch` all come from the real variant row, so the
    attention stages, the downsample ladder and the RepCPE placement are the
    reference ones.
    """
    return {
        'layers': (1,) * num_stages,
        'embed_dims': tuple(32 * 2 ** i for i in range(num_stages)),
    }


def _model(variant: str, num_stages: int, **overrides) -> MobileClipV2Model:
    config = dict(
        image_size=_IMG,
        vocab_size=_VOCAB,
        context_length=_SEQ,
        text_width=64,
        text_heads=4,
        # >= 2: at text_layers=1 a text-tower mechanism can be structurally
        # unobservable (MEASURED during step 8).
        text_layers=2,
        text_intermediate=128,
        embed_dim=_EMBED,
        image_encoder_kwargs=_tower_kwargs(num_stages),
    )
    config.update(overrides)
    return MobileClipV2Model.from_variant(variant, **config)


def _images(batch: int = _BATCH, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((batch, _IMG, _IMG, 3)).astype('float32')


def _tokens(batch: int = _BATCH, seed: int = 5) -> np.ndarray:
    """Token ids with the EOT (numeric maximum) at the LAST position.

    `MobileClipTextEncoder` pools by `argmax` of the ids; a maximum at position
    0 would make the pooled vector independent of everything after it.
    """
    rng = np.random.default_rng(seed)
    tokens = rng.integers(1, _VOCAB - 1, size=(batch, _SEQ)).astype('int32')
    tokens[:, -1] = _VOCAB - 1
    return tokens


def _inputs(batch: int = _BATCH) -> dict:
    return {'image': _images(batch), 'text': _tokens(batch)}


def _finite(x) -> bool:
    return bool(np.all(np.isfinite(ops.convert_to_numpy(x))))


# ---------------------------------------------------------------------
# The loss bridge.
#
# `CLIPContrastiveLoss` wants {'logits_per_image', 'logits_per_text'}; the model
# emits five keys, and Keras' `CompileLoss` broadcasts a single `Loss` object
# across every element of a structured output (so handing it the raw dict would
# call the loss once per key, on a single [N, N] tensor it cannot parse).
#
# The bridge is two thin pieces, both LOCAL TO THIS TEST:
#   * `_StackedLogitsModel` — a plain `keras.Model` whose `call` stacks the two
#     logit matrices into one (N, 2, N) tensor, i.e. ONE output. It has no
#     custom `train_step`; `fit()` below is stock.
#   * `_StackedClipLoss` — unstacks and delegates to the REAL
#     `CLIPContrastiveLoss.call`. `.call` (not `__call__`) is used so the
#     per-sample vector is returned and the outer `Loss` applies the reduction
#     exactly once.
# ---------------------------------------------------------------------


class _StackedLogitsModel(keras.Model):
    """Adapts the 5-key CLIP output down to the single tensor `fit()` wants."""

    def __init__(self, clip_model: MobileClipV2Model, **kwargs):
        # An EXPLICIT name is load-bearing: Keras derives the default name from
        # the class name, and the leading underscore of `_StackedLogitsModel`
        # produces `'__stacked_logits_model'`, which TensorFlow rejects as a
        # root scope name the moment `fit()` traces the step
        # ("is not a valid root scope name").
        kwargs.setdefault('name', 'stacked_logits_model')
        super().__init__(**kwargs)
        self.clip_model = clip_model

    def call(self, inputs, training=None):
        out = self.clip_model(inputs, training=training)
        return ops.stack(
            [out['logits_per_image'], out['logits_per_text']], axis=1
        )


class _StackedClipLoss(keras.losses.Loss):
    """Unstacks `(N, 2, N)` and delegates to the unmodified CLIP loss."""

    def __init__(self, name='stacked_clip_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.inner = CLIPContrastiveLoss()

    def call(self, y_true, y_pred):
        return self.inner.call(
            None,
            {
                'logits_per_image': y_pred[:, 0],
                'logits_per_text': y_pred[:, 1],
            },
        )


# ---------------------------------------------------------------------


class TestEndToEndForward:
    """Both stage topologies, at the real 256px geometry."""

    def test_four_stage_forward(self):
        """`mobileclip2_s0` (4 stages, batch_norm, 1 attention stage).

        PRESERVED: 256px input, the 64/64/32/16/8 ladder, the stage-3 attention
        token mixer, `text_layers=2`.
        DROPPED: depth (1 block/stage vs 2,6,10,2) and width (32..256 vs
        64..512).
        """
        model = _model('mobileclip2_s0', 4)
        out = model(_inputs(), training=False)

        assert set(out) == _OUTPUT_KEYS, sorted(out)
        assert ops.shape(out['image_features']) == (_BATCH, _EMBED)
        assert ops.shape(out['text_features']) == (_BATCH, _EMBED)
        assert ops.shape(out['logits_per_image']) == (_BATCH, _BATCH)
        assert ops.shape(out['logits_per_text']) == (_BATCH, _BATCH)
        assert ops.shape(out['logit_scale']) == ()
        for key, value in out.items():
            assert _finite(value), f"{key} contains non-finite values"

        # The topology this variant is here to cover.
        encoder = model.image_encoder
        assert len(encoder.layers_per_stage) == 4
        assert 'attention' in encoder.token_mixers, encoder.token_mixers
        assert encoder.norm_layer == 'batch_norm'

    def test_five_stage_forward(self):
        """`mobileclip2_s3` (5 stages, layer_norm, 2 attention stages).

        PRESERVED: 256px input, the 64/64/32/16/8/4 ladder — so the deepest
        stage really does run a 7x7 RepCPE kernel over a 4x4 map — both
        attention stages, and `layer_norm` as the attention pre-norm.
        DROPPED: depth (1 block/stage vs 2,12,24,4,2) and width (32..512 vs
        96..1536).
        """
        model = _model('mobileclip2_s3', 5)
        out = model(_inputs(), training=False)

        assert set(out) == _OUTPUT_KEYS, sorted(out)
        assert ops.shape(out['image_features']) == (_BATCH, _EMBED)
        assert ops.shape(out['text_features']) == (_BATCH, _EMBED)
        assert ops.shape(out['logits_per_image']) == (_BATCH, _BATCH)
        for key, value in out.items():
            assert _finite(value), f"{key} contains non-finite values"

        encoder = model.image_encoder
        assert len(encoder.layers_per_stage) == 5
        assert encoder.token_mixers.count('attention') == 2, encoder.token_mixers
        assert encoder.norm_layer == 'layer_norm'
        assert encoder.pos_embs[-1] == (7, 7), encoder.pos_embs

    def test_stage_geometry_at_256px(self):
        """The tabulated ladders, measured on the intermediate feature maps.

        This is the assertion the reduced width/depth CANNOT defeat: spatial
        geometry depends only on the stem, the downsample flags and the input
        size, all of which are the reference ones here.
        """
        for variant, num_stages, expected in (
                ('mobileclip2_s0', 4, _FOUR_STAGE_GEOMETRY),
                ('mobileclip2_s3', 5, _FIVE_STAGE_GEOMETRY),
        ):
            model = _model(variant, num_stages)
            model(_inputs(batch=1), training=False)
            encoder = model.image_encoder

            x = ops.convert_to_tensor(_images(batch=1))
            for block in encoder.stem:
                x = block(x, training=False)
            sizes = [int(ops.shape(x)[1])]
            for stage in encoder.stages:
                x = stage(x, training=False)
                sizes.append(int(ops.shape(x)[1]))

            assert tuple(sizes) == expected, (
                f"{variant}: per-stage spatial sizes {tuple(sizes)} != the "
                f"tabulated {expected}"
            )


class TestEndToEndSerialization:
    """Save -> reload -> compare BOTH towers' features BY VALUE."""

    def _roundtrip(self, variant: str, num_stages: int):
        model = _model(variant, num_stages)
        inputs = _inputs()
        before = model(inputs, training=False)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'model.keras')
            model.save(path)
            restored = keras.models.load_model(path)
            after = restored(inputs, training=False)

        for key in ('image_features', 'text_features'):
            np.testing.assert_allclose(
                ops.convert_to_numpy(before[key]),
                ops.convert_to_numpy(after[key]),
                atol=1e-6,
                rtol=0,
                err_msg=(
                    f"{variant}: {key} changed across a .keras round trip. "
                    f"Matching shapes/paths/param counts are NOT evidence "
                    f"here — restored-fresh-kernel failures pass all three."
                ),
            )
        np.testing.assert_allclose(
            ops.convert_to_numpy(before['logits_per_image']),
            ops.convert_to_numpy(after['logits_per_image']),
            atol=1e-6,
            rtol=0,
        )

    def test_four_stage_roundtrip(self):
        self._roundtrip('mobileclip2_s0', 4)

    def test_five_stage_roundtrip(self):
        self._roundtrip('mobileclip2_s3', 5)


class TestEndToEndTraining:
    """Stock `fit()` with the real `CLIPContrastiveLoss`, then gradients."""

    @staticmethod
    def _dataset(images, tokens, steps: int) -> tf.data.Dataset:
        """One fixed batch, repeated — the overfit-one-batch protocol.

        `y_true` is a dummy: contrastive ground truth is the batch diagonal,
        derived inside the loss. Keras still requires a target tensor.
        """
        dummy = np.zeros((_BATCH,), dtype='float32')
        ds = tf.data.Dataset.from_tensors(
            ({'image': images, 'text': tokens}, dummy)
        )
        return ds.repeat(steps)

    def test_fit_runs_and_stays_finite(self):
        """STOCK `fit()` on one fixed batch, 8 single-step epochs.

        `steps_per_epoch=1` makes `history['loss']` a per-STEP series (one
        epoch mean over one step) instead of a single running mean, so the
        train-regime trend is readable.

        ASSERTED: the traced step runs at all, and every per-step loss is
        finite. NOT ASSERTED: that the loss decreases. Two independent reasons
        it need not, both measured rather than assumed:
          * a single Adam step's direction is not a reliable signal at all;
          * the eval-mode (`training=False`) loss reported alongside is
            additionally confounded by BatchNorm — the image tower's moving
            statistics start at (0, 1) and are still far from the batch
            statistics after a handful of steps, so the eval-mode number can
            move UP while the train-mode number moves DOWN. Both are printed.
        """
        clip_model = _model('mobileclip2_s0', 4)
        wrapper = _StackedLogitsModel(clip_model)
        wrapper.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=_StackedClipLoss(),
        )

        images, tokens = _images(), _tokens()
        eval_loss_fn = _StackedClipLoss()
        dummy = np.zeros((_BATCH,), dtype='float32')

        def _eval_loss() -> float:
            return float(
                ops.convert_to_numpy(
                    eval_loss_fn(
                        dummy,
                        wrapper(
                            {'image': images, 'text': tokens}, training=False
                        ),
                    )
                )
            )

        before_eval = _eval_loss()
        steps = 8
        history = wrapper.fit(
            self._dataset(images, tokens, steps),
            epochs=steps,
            steps_per_epoch=1,
            verbose=0,
        )
        after_eval = _eval_loss()

        losses = [float(v) for v in history.history['loss']]
        print(
            f"\n[fit] train-mode per-step loss over {steps} stock fit() steps: "
            + ", ".join(f"{v:.6f}" for v in losses)
            + f"\n[fit] train-mode delta {losses[-1] - losses[0]:+.6f}; "
            f"eval-mode (training=False, BN moving stats) "
            f"{before_eval:.6f} -> {after_eval:.6f} "
            f"({after_eval - before_eval:+.6f})"
        )

        assert len(losses) == steps
        assert all(np.isfinite(v) for v in losses), losses
        assert np.isfinite(before_eval) and np.isfinite(after_eval)

    def test_gradients_reach_every_trainable_weight_after_a_fit_step(self):
        """Every trainable weight of BOTH towers, through the REAL loss.

        Catches a tower that is constructed and tracked but never called, and a
        `stop_gradient` slipping into either path. Failures are NAMED.
        """
        clip_model = _model('mobileclip2_s0', 4)
        wrapper = _StackedLogitsModel(clip_model)
        wrapper.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=_StackedClipLoss(),
        )
        images, tokens = _images(), _tokens()
        wrapper.fit(
            self._dataset(images, tokens, 1),
            epochs=1,
            steps_per_epoch=1,
            verbose=0,
        )

        loss_fn = _StackedClipLoss()
        dummy = ops.zeros((_BATCH,))
        with tf.GradientTape() as tape:
            stacked = wrapper(
                {'image': images, 'text': tokens}, training=True
            )
            loss = loss_fn(dummy, stacked)
        weights = wrapper.trainable_weights
        grads = tape.gradient(loss, weights)

        assert len(weights) > 0
        missing = [w.path for w, g in zip(weights, grads) if g is None]
        assert not missing, (
            f"{len(missing)} of {len(weights)} trainable weights got a None "
            f"gradient: {missing}"
        )
        zero = [
            w.path for w, g in zip(weights, grads)
            if float(np.max(np.abs(ops.convert_to_numpy(g)))) == 0.0
        ]
        assert not zero, (
            f"{len(zero)} of {len(weights)} trainable weights got an all-zero "
            f"gradient: {zero}"
        )

        paths = [w.path for w in weights]
        assert any('image_encoder' in p for p in paths), paths[:5]
        assert any('text_encoder' in p for p in paths), paths[:5]
        assert any(p.endswith('logit_scale') for p in paths), paths[:5]


class TestGraphVsEagerControl:
    """The within-version control for the regime `fit()` actually runs in."""

    def test_eager_vs_graph_delta_is_measured_not_assumed(self):
        """MEASURE the delta on the SAME unchanged model, and report it.

        `fit()` traces its step into a `@tf.function`; every other test in this
        package is eager. The repo has MEASURED a case that is exactly 0.0
        eager and 4.233e-04 under `tf.function`, so "eager-identical" is not
        transferable evidence. This test therefore reports the NUMBER and only
        bounds it loosely — it does NOT assert exact 0.0, which would be a
        claim about the platform rather than about this model.
        """
        model = _model('mobileclip2_s0', 4)
        inputs = _inputs()
        eager = model(inputs, training=False)

        traced = tf.function(lambda d: model(d, training=False))
        graph = traced({
            'image': tf.convert_to_tensor(inputs['image']),
            'text': tf.convert_to_tensor(inputs['text']),
        })

        deltas = {}
        relative = {}
        for key in sorted(_OUTPUT_KEYS):
            a = ops.convert_to_numpy(eager[key])
            b = ops.convert_to_numpy(graph[key])
            deltas[key] = float(np.max(np.abs(a - b)))
            scale = float(np.max(np.abs(a)))
            relative[key] = deltas[key] / scale if scale > 0.0 else deltas[key]
        worst_abs = max(deltas.values())
        worst_rel = max(relative.values())

        print(
            "\n[eager-vs-graph] within-version max|delta| per key: "
            + ", ".join(
                f"{k}={deltas[k]:.6e} (rel {relative[k]:.3e})"
                for k in sorted(deltas)
            )
            + f" -> worst abs {worst_abs:.6e}, worst rel {worst_rel:.3e}"
        )

        assert set(graph) == _OUTPUT_KEYS, sorted(graph)
        # MEASURED on this machine (GPU 1): features ~6e-05 absolute, and the
        # logits inherit that through `logit_scale` (~14.3 at init) and the
        # embed_dim-wide dot product, landing at ~2.4e-03 absolute — i.e. ~2e-04
        # RELATIVE on every key. Bounding the ABSOLUTE delta would therefore
        # make this test a hidden assertion about `logit_scale`'s magnitude, so
        # the bound is relative. It is NOT zero, and it is not claimed to be:
        # this repo has measured a fixed model disagreeing with ITSELF by 6e-05
        # relative across GPU processes.
        assert worst_rel < 1e-2, (
            f"eager-vs-graph relative delta {worst_rel:.3e} is far larger than "
            f"floating-point reassociation explains; per key abs={deltas}, "
            f"rel={relative}"
        )
