"""BiT/BiB bidirectional text<->image bridge training package.

Trains :class:`dl_techniques.models.vision_language.bit_diffusion.DiTXA` on the
denoising-score-matching objective of the bridge process, in both directions at
once, through **stock** ``compile()`` / ``fit()``.

* ``synthetic_data.py``     -- the pre-encoded-latent INPUT CONTRACT, a synthetic
  generator that satisfies it, an ``.npz`` reader/writer for the real thing, and
  the ``tf.data`` pipeline that turns records into training elements.
* ``train_bit_diffusion.py`` -- ``TrainingConfig``, the CLI, and the run.

Run with::

    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" \\
        python -m train.bit_diffusion.train_bit_diffusion --help
    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 \\
        python -m train.bit_diffusion.train_bit_diffusion --smoke

The ``sample_weight`` mechanism, and why there is NO custom ``train_step``
-------------------------------------------------------------------------
The bridge objective is upstream's ``dsm_loss``::

    loss = mean( (pred - target)**2 * w(t) )

``w(t)`` is per-sample and direction-specific (``C(0,t,t)`` reverse,
``C(t,1,1)/phi(t,1)`` forward), and ``t`` is drawn per sample by the data
pipeline. A custom ``train_step`` is the obvious way to get ``t`` into the loss
and it is FORBIDDEN here: every direction-dependent quantity is computed
pipeline-side instead and the weighting rides in as ``sample_weight``, the third
element of the ``tf.data`` tuple. ``src/train/sd3_mmdit/train_sd3_mmdit.py``
overrides ``train_step`` and is untested; its dict-batch SHAPE is copied here,
its mechanism deliberately is not.

Three measurements from this port's step-1 probe fix the mechanism, and none of
them may be re-guessed (``plans/.../probes/step1_readings.txt``, decisions D-006):

1. A ``tf.data`` dataset yielding ``(dict_inputs, y, w)`` IS accepted by stock
   ``fit()`` on a subclassed ``keras.Model``, with ``type(m).train_step is
   keras.Model.train_step``.
2. A ``(B,)`` ``sample_weight`` against a RANK-4 prediction **raises**
   ``InvalidArgumentError: Incompatible shapes [B,H,W] vs [B]``. Stock
   ``keras.losses.MeanSquaredError`` raises identically, so this is a general
   Keras property and not a defect of ``FlowMatchingVelocityLoss``.
3. The remedy, verified at ``rel = 0.0`` through ``Loss.__call__``, ``evaluate``
   AND ``fit``: emit the weight ALREADY BROADCAST to the loss's own reduction
   shape, ``(B, H, W)``. That is what this pipeline does, and
   ``tests/test_train/test_bit_diffusion/`` pins the RANK so the latent crash
   stays a guarded contract.

Consequence: **zero** new ``losses/`` classes. ``FlowMatchingVelocityLoss``
(mean over the channel axis) composed with a ``(B,H,W)`` weight and Keras'
``sum_over_batch_size`` reduction reproduces upstream's ``.mean()`` exactly.

``direction`` is data, not a Python flag
----------------------------------------
Upstream's ``reverse`` is a Python bool selecting a whole sub-branch. Here it is
a per-sample input tensor and ``keras.ops.where`` selects (D-005), so one graph
serves mixed-direction batches and forward-only / reverse-only training are
DATA settings (``--direction``), not model variants.
"""
