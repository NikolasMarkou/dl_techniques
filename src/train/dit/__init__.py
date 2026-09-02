r"""Class-conditional latent DiT training package.

Trains :class:`dl_techniques.models.vision_language.dit.DiT` on upstream's
``LossType.MSE`` + ``ModelVarType.LEARNED_RANGE`` objective through **stock**
``compile()`` / ``fit()``, with no custom ``train_step``.

* ``synthetic_data.py`` -- the pre-encoded-latent INPUT CONTRACT, a synthetic
  class-correlated generator that satisfies it, an ``.npz`` reader/writer for the
  real thing, and the ``tf.data`` pipeline that turns records into training
  elements.
* ``ema_callback.py``   -- ``WeightEMACallback``, upstream's decay-0.9999
  exponential moving average of the trainable weights.
* ``train_dit.py``      -- ``TrainingConfig``, the CLI, and the run.

Run with::

    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" \
        python -m train.dit.train_dit --help
    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" \
        python -m train.dit.train_dit --smoke

Why there is NO custom ``train_step``
-------------------------------------
A ``keras.losses.Loss`` sees only ``(y_true, y_pred, sample_weight)``, and the
DDPM objective needs the per-sample timestep ``t`` and the clean latent
``x_start`` as well. The obvious way to get them there is a ``train_step``
override, and that is FORBIDDEN in this repository. Everything the loss needs is
instead PACKED INTO ``y_true`` by the data pipeline --
``concat([noise, x_start, t_plane], axis=-1)``, layout ``[0:C]``/``[C:2C]``/
``[2C:2C+1]`` -- and ``DDPMHybridLoss`` re-derives ``x_t`` from those three.
``sample_weight`` is deliberately left unused: Keras MULTIPLIES the per-sample
loss by it, so it is not a free side channel. See decisions.md D-002.

No VAE, and what that means
---------------------------
Latent diffusion needs an autoencoder, and **none ships with this repository**.
``synthetic_data.py`` states the latent contract precisely so a producer can
satisfy it externally, and the built-in generator draws class-correlated
Gaussian fields so the training loop is exercisable end-to-end without one.
Samples therefore come out as latents; nothing here decodes them to pixels.
"""
