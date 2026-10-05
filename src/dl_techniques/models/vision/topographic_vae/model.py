"""The Topographic VAE: a variational autoencoder whose latent variables are
organized on a lattice, and which learns *approximately equivariant capsules*
from unlabelled sequences of transformed observations.

The problem it solves
---------------------
Two literatures meet here and are usually kept apart. **Topographic generative
models** drop the independence assumption of a VAE prior, arranging latent
variables on a lattice so neighbours share a variance and correlate in energy;
the ordering that emerges looks like V1's orientation maps. **Equivariant
networks** impose symmetry by construction — a group convolution groups features
into equivalence classes and permutes activations *within* a group when the input
is transformed, so pose is preserved rather than averaged away. The first is
unsupervised and learns its own organization; the second is supervised data or
hard-coded structure. This model asks whether the first can produce the second,
answering yes by adding one inductive bias: **temporal coherence with a shift**.

The mechanism
-------------
A Student's-t variable is a scale mixture: `T = Z * sqrt(nu / sum_i U_i^2)`
(Eq. 2). Correlate neighbouring `T`s by sharing their `U`s and you get a
topographic prior whose construction is *arithmetic*, not energy — which is what
makes it trainable by ordinary variational inference
(:class:`dl_techniques.layers.generative.topographic_product.TopographicProduct`
implements it; read that layer's docstring for the full derivation).

Now extend the neighbourhoods over **time**, and choose *which* correlation:

- the same latent location at neighbouring timesteps (`temporal_coherence=
  "stationary"`, Eq. 8, the "Bubbles" model) yields an **invariant** capsule —
  correlated energy, constant representation. This is the baseline the paper
  reports as scoring a good `equivariance_error` for the wrong reason.
- a location shifted by one step along the capsule (`temporal_coherence=
  "shifting"`, Eq. 9) yields an **equivariant** one. A transformation of the input
  shows up as a cyclic roll *inside* a capsule. Nothing supervises this; it is
  induced by the shifted correlation structure plus the reconstruction objective.

Because a roll is invertible, `t_0` alone determines the whole sequence: decode
`Roll_l(t_0)` and you get frame `l` without ever encoding it. That is
:meth:`TopographicVAE.traverse_capsules`, and it is what makes the learned
equivariance *checkable* rather than merely claimed.

The architecture
----------------
Per paper Section A.3, plain 3-layer ReLU MLPs — no convolutions. The point is to
test whether the inductive bias alone produces the structure, and a conv encoder
would supply its own translation equivariance and confound the result.

    x  (B, S, H, W, C)
              │
       flatten │ (B, S, H*W*C)
              ▼
    ┌──────────────────────────────────┐
    │  z-encoder  3-layer MLP + ReLU   │
    │  → z_mean, z_log_var   (B,S,C*D) │
    ├──────────────────────────────────┤
    │  u-encoder  3-layer MLP + ReLU   │
    │  → u_mean, u_log_var   (B,S,C*D) │
    └───────────────┬──────────────────┘
                    ▼  reparameterize (N(mu, exp(logvar)))
    ┌──────────────────────────────────────────────┐
    │  TopographicProduct:                          │
    │    energy = Σ_δ W R_δ u_{l+δ}²               │
    │    t = sqrt(2)(z-µ) / sqrt(nu·energy + eps)   │
    └───────────────┬──────────────────────────────┘
                    ▼  t  (B, S, C, D)
    ┌──────────────────────────────────┐
    │  decoder    3-layer MLP + ReLU   │
    └───────────────┬──────────────────┘
                    ▼
    x_hat  (B, S, H, W, C)

The decoder takes **`t` alone**. Equation 12 writes the likelihood as
`p(x | g_θ(t))` and Section 4.3 calls `T` "the first layer of the generative
decoder", so that is the literal reading. It is also the only choice under which
the topography is load-bearing: handing the decoder `concat([u, z])` alongside
`t` would let it read the topography off the side channel and bypass the
normalization entirely, and the whole model would then be unfalsifiable — the
capsule structure could vanish while the likelihood stayed flat.

Baselines are modes, not subclasses
-----------------------------------
Tables 1-2 of the paper compare against a plain VAE and against "BubbleVAE". Both
fall out of one constructor here:

- ``use_variance_variables=False`` drops `u` altogether. There is then no energy
  to build and `t = z - µ` directly — the layer's normalization is skipped rather
  than divided by a near-zero, which would silently turn `epsilon` into a
  temperature. This is the plain VAE baseline.
- ``temporal_coherence="stationary"`` is BubbleVAE.

References:
    - Keller & Welling, 2022. Topographic VAEs learn Equivariant Capsules.
      NeurIPS 2021. (https://arxiv.org/abs/2109.01394)
    - Welling, Osindero & Hinton, 2003. Learning Sparse Topographic
      Representations with Products of Student-t Distributions. NeurIPS.
    - Hyvarinen, Hurri & Varrynen, 2004. A Unifying Framework for Natural Image
      Statistics: Spatiotemporal Activity Bubbles. Neurocomputing 58-60.
      (https://doi.org/10.1016/j.neucom.2004.09.007)
    - Kingma & Welling, 2014. Auto-Encoding Variational Bayes. ICLR.
      (https://arxiv.org/abs/1312.6114)
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import keras
from keras import ops

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.layers.activations.common import resolve_activation
from dl_techniques.layers.activations.common import serialize_activation
from dl_techniques.layers.capsules import CapsuleRoll
from dl_techniques.layers.generative.sampling import create_sampling_layer
from dl_techniques.losses.topographic_vae_loss import TopographicVAELoss
from dl_techniques.layers.generative.topographic_product import (
    TEMPORAL_COHERENCE_TYPES,
    TOPOGRAPHY_TYPES,
    TopographicProduct,
    TemporalCoherenceType,
    TopographyType,
)

# ---------------------------------------------------------------------


#: Distinct seed offsets for a caller-pinned initializer, keyed by layer position.
#:
#: A fixed table rather than a running counter, for two reasons. The offsets must
#: depend only on WHERE a layer sits, so two runs of the same configuration
#: produce the same weights -- that is the whole content of reproducibility. And
#: they must be distinct across positions, so no two layers replay one draw.
#: ``hash()`` is unusable for this: ``str`` hashing is salted per process, which
#: would make a "reproducible" model differ between runs.
_INITIALIZER_SLOTS = tuple(range(1, 64))

#: The stacks whose layers draw from the table above, in a FIXED order. The order
#: is part of the seed scheme, not a display detail: `_fresh_initializer`
#: derives each layer's offset from this list's position, so reordering it
#: changes every seeded model's weights.
_stack_slots = ("z_encoder", "u_encoder", "decoder")


@register_dl_technique("dl_techniques.models.topographic_vae.model")
class TopographicVAE(keras.Model):
    """Topographic variational autoencoder over sequences of transformed images.

    Encodes each frame of an input sequence into a pair of Gaussian
    posteriors (`z` and `u`), constructs topographic Student-t variables `t` from
    them, and decodes `t` back to a reconstruction of the frame.

    ``call`` takes the sequence rank-natively as ``(batch, sequence, height,
    width, channels)`` and returns a dict of the same-ranked pieces plus the flat
    posterior parameters. The dict is the model's public interface:
    :class:`dl_techniques.losses.topographic_vae_loss.TopographicVAELoss` consumes
    it, and ``compile``/`fit`` work with it directly.

    Variants (``MODEL_VARIANTS``, used by :meth:`from_variant`):

    .. code-block:: text

        name      capsules  dim   enc           dec           input
        mnist     18        18    972,648,648  648,972,2352   28x28x3
        dsprites  15        15    674,450,450  450,674,4096   64x64x1

    The widths are transcribed from Section A.3, where they are stated directly
    per dataset (the 2352 = 28*28*3 and 4096 = 64*64*1 decoder outputs pin the
    input resolutions too).

    :param input_shape: Per-frame shape ``(height, width, channels)``.
    :type input_shape: Tuple[int, int, int]
    :param sequence_length: Number of frames per sequence ``S``. Defaults to 18.
    :type sequence_length: int
    :param num_capsules: Number of capsules ``C``. Disjoint capsules are
        statistically independent, which is the model's structural prior.
        Defaults to 18.
    :type num_capsules: int
    :param capsule_dim: Dimensions per capsule ``D``. Defaults to 18.
    :type capsule_dim: int
    :param coherence_window: Temporal-coherence half-width ``L``; ``2L`` is the
        time extent of the coherence and ``L = 0`` means single frames. The paper
        reports that ``L`` near ``S/3`` is where equivariance is strongest.
        Defaults to 0.
    :type coherence_window: int
    :param neighborhood_size: Window width ``K`` within one capsule. Defaults
        to 3.
    :type neighborhood_size: int
    :param temporal_coherence: One of :data:`TEMPORAL_COHERENCE_TYPES`.
        ``"shifting"`` is the Topographic VAE, ``"stationary"`` is BubbleVAE,
        ``"none"`` is a single frame. Defaults to ``"shifting"``.
    :type temporal_coherence: str
    :param topography: One of :data:`TOPOGRAPHY_TYPES`. Defaults to
        ``"capsule_1d"``.
    :type topography: str
    :param grid_shape: ``(height, width)`` of the ``torus_2d`` lattice. Ignored
        for ``capsule_1d``, and required by ``torus_2d``.
    :type grid_shape: Optional[Tuple[int, int]]
    :param use_variance_variables: Whether the model has the ``u`` variables at
        all. ``False`` gives the plain-VAE baseline, where ``t = z - µ`` with no
        topography. Defaults to ``True``.
    :type use_variance_variables: bool
    :param degrees_of_freedom: Student's-t degrees of freedom ``nu``. Defaults
        to 1.
    :type degrees_of_freedom: float
    :param encoder_hidden_dims: Hidden widths of each encoder MLP. Defaults to
        ``[972, 648]``, giving three weight layers with the output head.
    :type encoder_hidden_dims: Optional[List[int]]
    :param decoder_hidden_dims: Hidden widths of the decoder MLP. Defaults to
        ``[648, 972]``.
    :type decoder_hidden_dims: Optional[List[int]]
    :param prior_mean: The scalar ``mu`` of Eq. 6. Defaults to 30.0, the value
        Section A.2 initializes topographic models to.
    :type prior_mean: float
    :param activation: Activation inside both MLPs. Defaults to ``"relu"``.
    :type activation: str
    :param final_activation: Activation on the reconstruction. Defaults to
        ``"sigmoid"``, matching the ``[0, 1]`` Bernoulli likelihood the ELBO
        assumes.
    :type final_activation: str
    :param kernel_initializer: Initializer for every MLP kernel.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param use_bias: Whether the MLPs carry biases. Defaults to ``True``.
    :type use_bias: bool
    :param sampling_seed: Seed for the reparameterization samplers, or ``None``
        for the usual unsampled-training behaviour.

        A concrete value makes ``model(x, training=False)`` REPRODUCIBLE: the same
        inputs give the same reconstruction on every call. That matters for
        evaluation rather than for training — the ELBO wants an unbiased sample,
        and a fixed draw would bias it — and it is the only way to make a
        metric comparable across two runs.

        ``None`` (the default) leaves the samplers drawing from the global
        generator, and setting ``keras.utils.set_random_seed`` does NOT change
        that: MEASURED on this build, three ``set_random_seed(3)`` calls gave
        three different draws, while an explicit ``seed=3`` gave the same one
        three times. So reproducibility has to be asked for, not assumed.
    :type sampling_seed: Optional[int]
    :param name: Model name.
    :type name: Optional[str]
    :param kwargs: Forwarded to ``keras.Model``.

    :raises ValueError: On a non-positive ``sequence_length`` / ``num_capsules`` /
        ``capsule_dim``, a negative ``coherence_window``, an unknown
        ``temporal_coherence`` or ``topography``, a ``input_shape`` that is not
        3D or is too small to hold a digit, or a
        ``temporal_coherence="shifting"`` request under ``torus_2d`` (which the
        topography layer refuses; see its docstring).

    :Example:

    >>> model = TopographicVAE.from_variant("mnist")
    >>> model = TopographicVAE(
    ...     input_shape=(32, 32, 1),
    ...     sequence_length=10,
    ...     num_capsules=10,
    ...     capsule_dim=10,
    ...     coherence_window=3,
    ... )
    >>> model = TopographicVAE.from_variant("mnist", pretrained=True)
    Traceback (most recent call last):
        ...
    NotImplementedError: TopographicVAE ships no pretrained weights ...
    """

    # Section A.3 states the per-dataset widths directly; the input shapes are
    # pinned by the decoder output widths (2352 = 28*28*3, 4096 = 64*64*1).
    MODEL_VARIANTS = {
        "mnist": {
            "input_shape": (28, 28, 3),
            "sequence_length": 18,
            "num_capsules": 18,
            "capsule_dim": 18,
            "encoder_hidden_dims": [972, 648],
            "decoder_hidden_dims": [648, 972],
        },
        "dsprites": {
            "input_shape": (64, 64, 1),
            "sequence_length": 15,
            "num_capsules": 15,
            "capsule_dim": 15,
            "encoder_hidden_dims": [674, 450],
            "decoder_hidden_dims": [450, 674],
        },
    }

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (28, 28, 3),
        sequence_length: int = 18,
        num_capsules: int = 18,
        capsule_dim: int = 18,
        coherence_window: int = 0,
        neighborhood_size: int = 3,
        temporal_coherence: TemporalCoherenceType = "shifting",
        topography: TopographyType = "capsule_1d",
        grid_shape: Optional[Tuple[int, int]] = None,
        use_variance_variables: bool = True,
        degrees_of_freedom: float = 1.0,
        encoder_hidden_dims: Optional[List[int]] = None,
        decoder_hidden_dims: Optional[List[int]] = None,
        prior_mean: float = 30.0,
        activation: str = "relu",
        final_activation: str = "sigmoid",
        kernel_initializer: Union[
            str, keras.initializers.Initializer
        ] = "he_normal",
        use_bias: bool = True,
        sampling_seed: Optional[int] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # First statement, before any sub-layer is created: a Layer must be
        # initialized before anything is attached to it, or Keras raises
        # "you forgot to call super().__init__()".
        super().__init__(name=name, **kwargs)

        if len(input_shape) != 3:
            raise ValueError(
                f"input_shape must be 3D (height, width, channels), got "
                f"{input_shape}"
            )
        if sequence_length <= 0:
            raise ValueError(
                f"sequence_length must be positive, got {sequence_length}"
            )
        if num_capsules <= 0:
            raise ValueError(f"num_capsules must be positive, got {num_capsules}")
        if capsule_dim <= 0:
            raise ValueError(f"capsule_dim must be positive, got {capsule_dim}")
        if coherence_window < 0:
            raise ValueError(
                f"coherence_window must be non-negative, got {coherence_window}"
            )
        if temporal_coherence not in TEMPORAL_COHERENCE_TYPES:
            raise ValueError(
                f"temporal_coherence must be one of "
                f"{list(TEMPORAL_COHERENCE_TYPES)}, got {temporal_coherence!r}"
            )
        if topography not in TOPOGRAPHY_TYPES:
            raise ValueError(
                f"topography must be one of {list(TOPOGRAPHY_TYPES)}, got "
                f"{topography!r}"
            )
        # With no u variables there is no energy, so a temporal coherence window
        # has nothing to correlate. Refusing here rather than silently ignoring
        # L keeps a forgotten flag from looking like it took effect.
        if not use_variance_variables and coherence_window != 0:
            raise ValueError(
                f"coherence_window={coherence_window} requires "
                f"use_variance_variables=True: temporal coherence correlates the "
                f"energy built from u, and with use_variance_variables=False "
                f"there is no u to correlate. Use use_variance_variables=True, "
                f"or set coherence_window=0 for the plain-VAE baseline."
            )
        if not use_variance_variables and topography == "torus_2d":
            raise ValueError(
                "topography='torus_2d' requires use_variance_variables=True: "
                "without u there is no neighbourhood to build, so the torus "
                "would describe nothing"
            )

        height, width, channels = input_shape
        if height < 8 or width < 8:
            raise ValueError(
                f"input dimensions must be at least 8x8, got {height}x{width}"
            )

        self._input_shape = tuple(input_shape)
        self.sequence_length = int(sequence_length)
        self.num_capsules = int(num_capsules)
        self.capsule_dim = int(capsule_dim)
        self.coherence_window = int(coherence_window)
        self.neighborhood_size = int(neighborhood_size)
        self.temporal_coherence = temporal_coherence
        self.topography = topography
        self.grid_shape = (
            None if grid_shape is None else tuple(grid_shape)
        )
        self.use_variance_variables = bool(use_variance_variables)
        self.degrees_of_freedom = degrees_of_freedom
        self.encoder_hidden_dims = (
            [972, 648] if encoder_hidden_dims is None else list(encoder_hidden_dims)
        )
        self.decoder_hidden_dims = (
            [648, 972] if decoder_hidden_dims is None else list(decoder_hidden_dims)
        )
        self.prior_mean = prior_mean
        self.activation_name = activation
        self.final_activation_name = final_activation
        # A STRING, deliberately, not a resolved initializer object.
        #
        # A `keras.initializers` instance carries a seed generator, and a seeded
        # initializer returns the SAME values on every call. Passing one shared
        # instance to every `Dense` therefore made `u_encoder` a bit-identical
        # clone of `z_encoder` -- MEASURED, max delta 0.0 across both encoders'
        # kernels -- so the topographic prior's second Gaussian `u` was the first
        # one relabelled and `KL_u == KL_z` identically at initialization. The
        # string is resolved per layer by `_fresh_initializer`, which costs one
        # `keras.initializers.get` call per Dense and makes every sub-layer draw
        # independently. A caller-pinned `seed` is recorded separately and
        # OFFSET per layer there; see `_fresh_initializer`.
        self.kernel_initializer = kernel_initializer
        self._initializer_seed = self._resolve_initializer_seed(kernel_initializer)
        self.use_bias = use_bias
        self._sampling_seed = sampling_seed
        self.latent_dim = self.num_capsules * self.capsule_dim
        self.frame_elements = int(height) * int(width) * int(channels)

        self.activation = resolve_activation(activation)
        self.final_activation = resolve_activation(final_activation)

        # Sub-layers are CREATED here and materialized by build(). Names are
        # explicit throughout so two instances agree on every weight path --
        # Keras' auto_name counter is process-global, and an unnamed sub-layer
        # makes the build-parity comparison compare nothing.
        self.z_encoder = self._build_mlp(
            self.encoder_hidden_dims,
            2 * self.latent_dim,
            "z_encoder",
        )
        if self.use_variance_variables:
            self.u_encoder = self._build_mlp(
                self.encoder_hidden_dims,
                2 * self.latent_dim,
                "u_encoder",
            )
        else:
            self.u_encoder = None

        self.z_sampling = create_sampling_layer(
            "gaussian", name="z_sampling", seed=sampling_seed
        )
        if self.use_variance_variables:
            self.u_sampling = create_sampling_layer(
                "gaussian", name="u_sampling", seed=sampling_seed
            )
        else:
            self.u_sampling = None

        self.topographic_product = self._build_topographic_product()
        self.roll = CapsuleRoll(
            num_capsules=self.num_capsules,
            capsule_dim=self.capsule_dim,
            name="capsule_roll",
        )

        self.decoder = self._build_decoder()

        # No loss trackers here on purpose. `compute_loss` routes the whole ELBO
        # through TopographicVAELoss in one call, which keeps its OWN per-term
        # trackers -- a set mirrored on the model would read an exact 0.0 in every
        # epoch log while the real loss was 85.05, because nothing ever updates
        # them. Dead metrics that report 0.0 are worse than no metrics.
        logger.info(
            f"Created TopographicVAE: input={self._input_shape}, S={self.sequence_length}, "
            f"capsules={self.num_capsules}x{self.capsule_dim} "
            f"(latent_dim={self.latent_dim}), L={self.coherence_window}, "
            f"K={self.neighborhood_size}, coherence={self.temporal_coherence}, "
            f"topography={self.topography}, u={'on' if self.use_variance_variables else 'off'}"
        )

    @staticmethod
    def _resolve_initializer_seed(spec) -> Optional[int]:
        """The pinned seed inside an initializer spec, if the caller set one.

        :param spec: A string alias, an ``Initializer``, or a serialized dict.
        :return: The seed, or ``None`` when the spec carries none.
        :rtype: Optional[int]
        """
        if isinstance(spec, str):
            # A string alias cannot carry a seed, and the alias' own class default
            # is `seed=None`.
            return None
        try:
            config = keras.initializers.serialize(spec).get("config") or {}
        except (TypeError, ValueError):
            return None
        seed = config.get("seed")
        return None if seed is None else int(seed)

    def _fresh_initializer(self, stack: str, index: int):
        """A NEW ``Dense`` initializer instance for layer ``index``.

        Resolving per layer is not an optimisation detail. A Keras initializer
        carries a seed generator, and a seeded one REPLAYS identical values on
        every call, so a single shared instance initializes every kernel in the
        model to the same numbers. Two encoders of the same shape then come out
        bit-identical, which silently halves the model's posterior -- MEASURED
        before this fix, ``max|ΔW| == 0.0`` across ``z_encoder`` and
        ``u_encoder``, and therefore ``KL_u == KL_z`` identically at
        initialization.

        A caller who pinned ``seed=`` gets the draws *offset by the layer index*
        rather than replayed: reproducibility is a property of a run, not a
        request that every layer hold the same numbers, and replaying would
        satisfy the former only by breaking the model.

        The offset is derived from ``(stack, index)`` rather than a running
        counter, so the draws depend only on WHERE a layer sits -- two models
        with the same configuration get the same weights, which is what makes a
        seeded model reproducible at all.

        :param stack: The stack the layer belongs to, e.g. ``"z_encoder"``.
        :type stack: str
        :param index: The layer's position within its stack.
        :type index: int
        :return: A fresh initializer.
        :rtype: keras.initializers.Initializer
        """
        spec = self.kernel_initializer
        if self._initializer_seed is None or isinstance(spec, str):
            # No pinned seed: `keras.initializers.get` draws from the global
            # generator, so a fresh instance is already independent.
            return keras.initializers.get(spec)

        # A stable per-(stack, index) offset. Hash-free on purpose: `hash()` is
        # salted per process for `str`, which would make a "reproducible" model
        # differ between runs -- precisely the guarantee being claimed.
        slot = _stack_slots.index(stack) * len(_stack_slots) + int(index)
        offset = _INITIALIZER_SLOTS[slot % len(_INITIALIZER_SLOTS)]
        serialized = keras.initializers.serialize(spec)
        config = dict(serialized.get("config") or {})
        config["seed"] = int(self._initializer_seed) + offset
        return keras.initializers.deserialize({**serialized, "config": config})

    def _build_mlp(
        self, hidden_dims: Sequence[int], output_dim: int, prefix: str
    ) -> keras.layers.Layer:
        """Build a per-frame MLP that acts on the last axis of a rank-3 tensor.

        ``Dense`` on a ``(batch, sequence, features)`` tensor already treats the
        sequence axis as independent, so one stack serves every frame without a
        loop or a reshape.

        :param hidden_dims: Hidden widths.
        :type hidden_dims: Sequence[int]
        :param output_dim: Width of the final projection.
        :type output_dim: int
        :param prefix: Name prefix for every sub-layer.
        :type prefix: str
        :return: The assembled stack.
        :rtype: keras.layers.Layer
        """
        layers = []
        for index, width in enumerate(hidden_dims):
            layers.append(
                keras.layers.Dense(
                    width,
                    activation=self.activation,
                    use_bias=self.use_bias,
                    kernel_initializer=self._fresh_initializer(prefix, index),
                    name=f"{prefix}_dense_{index}",
                )
            )
        layers.append(
            keras.layers.Dense(
                output_dim,
                use_bias=self.use_bias,
                kernel_initializer=self._fresh_initializer(prefix, len(hidden_dims)),
                name=f"{prefix}_head",
            )
        )
        return keras.Sequential(layers, name=prefix)

    def _build_topographic_product(self) -> Optional[TopographicProduct]:
        """Build the topography layer, or ``None`` for the plain-VAE baseline.

        :return: The layer, or ``None`` when ``use_variance_variables`` is off.
        :rtype: Optional[TopographicProduct]
        """
        if not self.use_variance_variables:
            return None
        return TopographicProduct(
            num_capsules=self.num_capsules,
            capsule_dim=self.capsule_dim,
            coherence_window=self.coherence_window,
            neighborhood_size=self.neighborhood_size,
            temporal_coherence=self.temporal_coherence,
            topography=self.topography,
            grid_shape=self.grid_shape,
            degrees_of_freedom=self.degrees_of_freedom,
            prior_mean=self.prior_mean,
            name="topographic_product",
        )

    def _build_decoder(self) -> keras.layers.Layer:
        """Build the decoder MLP from ``t`` to a flattened reconstruction.

        :return: The assembled stack.
        :rtype: keras.layers.Layer
        """
        layers = []
        for index, width in enumerate(self.decoder_hidden_dims):
            layers.append(
                keras.layers.Dense(
                    width,
                    activation=self.activation,
                    use_bias=self.use_bias,
                    kernel_initializer=self._fresh_initializer("decoder", index),
                    name=f"decoder_dense_{index}",
                )
            )
        layers.append(
            keras.layers.Dense(
                self.frame_elements,
                activation=self.final_activation,
                use_bias=self.use_bias,
                kernel_initializer=self._fresh_initializer(
                    "decoder", len(self.decoder_hidden_dims)
                ),
                name="decoder_output",
            )
        )
        return keras.Sequential(layers, name="decoder")

    def build(self, input_shape) -> None:
        """Materialize the sub-layer tree exactly as ``call`` runs it.

        A subclassed ``keras.Model`` that does NOT override ``build`` walks its
        sub-layers on the first forward pass, and a ``load_model`` that only sets
        weights then restores into an unbuilt tree: the count matches and nothing
        raises. Building here — and building exactly the layers ``call`` uses —
        is what makes a checkpoint restore.

        :param input_shape: ``(batch, sequence, height, width, channels)``.
        :type input_shape: Any
        :raises ValueError: On a rank other than 5, or a last three axes that do
            not match the configured ``input_shape``.
        """
        shape = tuple(input_shape)
        if len(shape) != 5:
            raise ValueError(
                f"TopographicVAE expects rank-5 input "
                f"(batch, sequence, height, width, channels), got shape {shape}"
            )
        expected = (None, None) + self._input_shape
        if shape[2:] != expected[2:]:
            raise ValueError(
                f"input frames {shape[2:]} do not match the configured "
                f"input_shape {self._input_shape}"
            )

        # (batch, sequence, frame_elements) -- the rank the MLPs act on.
        flat_shape = (shape[0], shape[1], self.frame_elements)
        posterior_shape = (shape[0], shape[1], self.latent_dim)

        self.z_encoder.build(flat_shape)
        self.z_sampling.build((posterior_shape, posterior_shape))
        if self.use_variance_variables:
            self.u_encoder.build(flat_shape)
            self.u_sampling.build((posterior_shape, posterior_shape))
            self.topographic_product.build(
                (posterior_shape, posterior_shape)
            )

        # t is reshaped to (batch, sequence, C*D) before the decoder.
        self.decoder.build(posterior_shape)

        super().build(input_shape)

    def _encode(
        self, frames: keras.KerasTensor, prefix: str
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Encode frames into ``(mean, log_variance)`` for one latent family.

        :param frames: Flattened frames ``(batch, sequence, frame_elements)``.
        :type frames: keras.KerasTensor
        :param prefix: ``"z"`` or ``"u"``; selects the sub-layer names.
        :type prefix: str
        :return: ``(mean, log_variance)``, each ``(batch, sequence, latent_dim)``.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        outputs = getattr(self, f"{prefix}_encoder")(frames)
        latent_dim = self.latent_dim
        return (
            outputs[..., :latent_dim],
            outputs[..., latent_dim:],
        )

    def call(self, inputs, training=None):
        """Reconstruct a sequence and expose the posterior parameters.

        :param inputs: Sequence of frames ``(batch, sequence, height, width,
            channels)`` on ``[0, 1]``.
        :type inputs: Any
        :param training: Forwarded to the samplers.
        :type training: Optional[bool]
        :return: A dict with ``reconstruction`` (same shape as the input) and
            ``t``, ``z_mean``, ``z_log_var`` -- plus ``u_mean`` and
            ``u_log_var`` when ``use_variance_variables`` -- each shaped
            ``(batch, sequence, latent_dim)``. ``t`` is *flat*: the capsule
            split is a view, ``t.reshape(batch, sequence, num_capsules,
            capsule_dim)``, and :func:`dl_techniques.metrics.topographic.roll_capsules`
            reads it back.
        :rtype: Dict[str, keras.KerasTensor]
        """
        frames = ops.cast(inputs, self.compute_dtype)
        shape = ops.shape(frames)
        flat = ops.reshape(
            frames, ops.stack([shape[0], shape[1], self.frame_elements])
        )

        z_mean, z_log_var = self._encode(flat, "z")
        z = self.z_sampling([z_mean, z_log_var], training=training)

        outputs = {
            "reconstruction": None,
            "t": None,
            "z_mean": z_mean,
            "z_log_var": z_log_var,
        }

        if self.use_variance_variables:
            u_mean, u_log_var = self._encode(flat, "u")
            u = self.u_sampling([u_mean, u_log_var], training=training)
            t = self.topographic_product([z, u], training=training)
            outputs["u_mean"] = u_mean
            outputs["u_log_var"] = u_log_var
        else:
            # Plain-VAE baseline: there is no energy, so the topography layer's
            # 1/sqrt(nu * energy) is skipped rather than divided by a near-zero.
            # t = z - mu keeps the SAME downstream width and decoder, so the
            # baseline differs only by the topography and not by the architecture.
            t = z - ops.cast(self.prior_mean, z.dtype)

        t_flat = ops.reshape(
            t, ops.stack([shape[0], shape[1], self.latent_dim])
        )
        reconstruction = ops.reshape(
            self.decoder(t_flat),
            ops.stack(
                [
                    shape[0],
                    shape[1],
                    ops.cast(self._input_shape[0], "int32"),
                    ops.cast(self._input_shape[1], "int32"),
                    ops.cast(self._input_shape[2], "int32"),
                ]
            ),
        )

        outputs["t"] = t
        outputs["reconstruction"] = reconstruction
        return outputs

    def compute_loss(
        self, x=None, y=None, y_pred=None, sample_weight=None, training=True
    ) -> keras.KerasTensor:
        """Route the dict output to this model's ELBO loss.

        Keras' own loss dispatch cannot express this objective from a dict
        ``y_pred``: handing a single ``Loss`` a dict output makes
        ``CompileLoss._build_nested`` treat the ``Loss`` itself as a nested key
        and raise ``KeyError: The path: ('reconstruction',) ...``. Supplying a
        dict ``loss`` keyed by output name does not help either — each key then
        receives only its own leaf, so the two KL terms would be silently
        dropped, and the model would train as a plain autoencoder.

        Overriding ``compute_loss`` (rather than ``train_step``) keeps stock
        ``fit``: Keras' default ``train_step`` still calls
        ``optimizer.scale_loss`` inside the tape, which is the mixed-precision
        correctness requirement. Overriding ``train_step`` would opt out of it.

        :param x: The input sequence; used when ``y`` is ``None``.
        :type x: Any
        :param y: The labels, or ``None`` to use ``x`` (the VAE case).
        :type y: Any
        :param y_pred: The model's dict output.
        :type y_pred: Any
        :param sample_weight: Per-sample weights. Must be ``None``: the ELBO's
            three terms reduce over the sequence and the latent axes with fixed
            semantics, and broadcasting a per-sample weight into them is not
            supported.
        :type sample_weight: Any
        :param training: Whether this is a training step.
        :type training: bool
        :return: The scalar negative ELBO.
        :rtype: keras.KerasTensor
        :raises ValueError: If ``sample_weight`` is not ``None``.
        """
        if sample_weight is not None:
            raise ValueError(
                "TopographicVAE does not support sample_weight: the ELBO's "
                "reconstruction and both KL terms reduce over the sequence and "
                "the latent axes, and there is no single per-sample axis a "
                "weight could multiply without changing those semantics"
            )

        if y_pred is None:
            y_pred = self(x, training=training)

        compile_loss = self._compile_loss
        if compile_loss is None:
            raise ValueError(
                "TopographicVAE.compute_loss needs a compiled loss. Call "
                "model.compile(loss=TopographicVAELoss(...)) first, or use "
                "create_topographic_vae(), which does it for you."
            )

        user_loss = getattr(compile_loss, "_user_loss", None)
        if isinstance(user_loss, TopographicVAELoss):
            # The whole objective at once, with every posterior term present.
            return user_loss(x if y is None else y, y_pred)

        # Any other loss (a plain MSE on the reconstruction, say) applies only
        # to that leaf, and the KL terms are reported as unscaled diagnostics
        # rather than silently contributing nothing.
        reconstruction = y_pred["reconstruction"]
        target = x if y is None else y
        return super().compute_loss(
            x=target,
            y=target,
            y_pred=reconstruction,
            sample_weight=None,
            training=training,
        )

    def compute_output_shape(self, input_shape) -> Dict[str, Tuple]:
        """Output shapes, from the stored configuration alone.

        Works on an unbuilt model: the sequence and frame extents come from the
        input and every other extent comes from the constructor arguments, never
        from a weight's shape.

        :param input_shape: ``(batch, sequence, height, width, channels)``.
        :type input_shape: Any
        :return: The same keys :meth:`call` returns.
        :rtype: Dict[str, Tuple]
        """
        shape = tuple(input_shape)
        if len(shape) != 5:
            raise ValueError(
                f"TopographicVAE expects rank-5 input "
                f"(batch, sequence, height, width, channels), got shape {shape}"
            )
        batch, sequence = shape[0], shape[1]
        posterior = (batch, sequence, self.latent_dim)
        reconstruction = (batch, sequence) + self._input_shape

        outputs = {
            "reconstruction": reconstruction,
            "t": posterior,
            "z_mean": posterior,
            "z_log_var": posterior,
        }
        if self.use_variance_variables:
            outputs["u_mean"] = posterior
            outputs["u_log_var"] = posterior
        return outputs

    def encode(
        self, inputs, training=None
    ) -> Dict[str, keras.KerasTensor]:
        """Encode a sequence to posterior parameters, without decoding.

        The deterministic half of the model, and the surface the round-trip
        value comparison uses: ``call`` samples, so its output cannot be
        compared at ``atol=0.0``, while these parameters cannot.

        :param inputs: Sequence of frames, same rank as ``call``.
        :type inputs: Any
        :param training: Forwarded to the encoders.
        :type training: Optional[bool]
        :return: ``z_mean``, ``z_log_var`` and (when enabled) ``u_mean`` and
            ``u_log_var``, each ``(batch, sequence, latent_dim)``.
        :rtype: Dict[str, keras.KerasTensor]
        """
        frames = ops.cast(inputs, self.compute_dtype)
        shape = ops.shape(frames)
        flat = ops.reshape(
            frames, ops.stack([shape[0], shape[1], self.frame_elements])
        )
        z_mean, z_log_var = self._encode(flat, "z")
        outputs = {"z_mean": z_mean, "z_log_var": z_log_var}
        if self.use_variance_variables:
            u_mean, u_log_var = self._encode(flat, "u")
            outputs["u_mean"] = u_mean
            outputs["u_log_var"] = u_log_var
        return outputs

    def decode(
        self, topographic_variables, training=None
    ) -> keras.KerasTensor:
        """Decode topographic variables to a frame sequence.

        Takes ``t`` and nothing else, matching ``call``.

        :param topographic_variables: ``(batch, sequence, latent_dim)``,
            ``(batch, num_capsules, capsule_dim)``, the bare
            ``(num_capsules, capsule_dim)``, or the rank-4
            ``(batch, sequence, num_capsules, capsule_dim)``.
        :type topographic_variables: Any
        :param training: Forwarded to the decoder.
        :type training: Optional[bool]
        :return: ``(batch, sequence, height, width, channels)``, with
            ``sequence == 1`` for a rank-2 or rank-3 input.
        :rtype: keras.KerasTensor
        """
        variables = ops.cast(topographic_variables, self.compute_dtype)
        shape = ops.shape(variables)
        # The rank comes from the STATIC shape, so this branch is decided at
        # trace time; branching on ops.shape() would be a tensor-value test.
        #
        # Rank 2 is read as (batch, latent_dim) -- ONE frame per batch element --
        # not as the bare (C, D) capsule, which is genuinely ambiguous with a
        # batch of single-frame sequences. Callers holding a bare capsule use
        # traverse_capsules, which reshapes explicitly rather than guessing.
        if len(variables.shape) == 2:
            flat = ops.reshape(
                variables,
                ops.stack([shape[0], 1, self.latent_dim]),
            )
            shape = ops.stack([shape[0], ops.cast(1, "int32")])
        elif len(variables.shape) == 3:
            flat = ops.reshape(
                variables,
                ops.stack([shape[0], shape[1], self.latent_dim]),
            )
        else:
            flat = variables
        reconstruction = self.decoder(flat, training=training)
        return ops.reshape(
            reconstruction,
            ops.stack(
                [
                    shape[0],
                    shape[1],
                    ops.cast(self._input_shape[0], "int32"),
                    ops.cast(self._input_shape[1], "int32"),
                    ops.cast(self._input_shape[2], "int32"),
                ]
            ),
        )

    def traverse_capsules(
        self, topographic_variable, num_steps: Optional[int] = None
    ) -> keras.KerasTensor:
        """Decode a whole sequence from ONE encoded capsule activation.

        Rolls ``t`` by ``l`` steps within each capsule for ``l = 0 .. S-1`` and
        decodes each. This is the paper's Figure 1 and its capsule traversals,
        and it is the direct test of what the model claims: an equivariant
        representation makes the unseen frames of a transformation sequence
        recoverable from a single encoded frame.

        :param topographic_variable: A flat latent ``(batch, latent_dim)`` or
            ``(batch, sequence, latent_dim)`` — the ``"t"`` slot of this model's
            own output — whose **first timestep** is the activation to traverse.
            A single unbatched capsule ``(num_capsules, capsule_dim)`` is also
            accepted and is treated as a batch of one.
        :type topographic_variable: Any
        :param num_steps: Number of frames to produce. Defaults to
            ``sequence_length``.
        :type num_steps: Optional[int]
        :return: ``(batch, num_steps, height, width, channels)``.
        :rtype: keras.KerasTensor
        """
        steps = self.sequence_length if num_steps is None else int(num_steps)
        if steps <= 0:
            raise ValueError(f"num_steps must be positive, got {steps}")

        variable = ops.cast(topographic_variable, self.compute_dtype)
        rank = len(variable.shape)

        # Reduce to ONE timestep first, then to capsule form. The order matters:
        # a rank-3 (batch, sequence, latent_dim) must be indexed BEFORE the
        # rank-2 flat reshape, or the sequence axis is folded into the capsule
        # width and the roll shifts activations across capsule boundaries.
        if rank == 4:
            variable = variable[:, 0]
        elif rank == 3 and variable.shape[-1] == self.latent_dim:
            variable = variable[:, 0]
        elif rank not in (2, 3):
            raise ValueError(
                f"traverse_capsules accepts (batch, latent_dim), "
                f"(num_capsules, capsule_dim), (batch, C, D) or "
                f"(batch, sequence, latent_dim); got rank {rank} "
                f"(shape {tuple(variable.shape)})"
            )

        # Now (batch, C*D) or (batch, C, D) -> (batch, C, D).
        if len(variable.shape) == 2:
            if variable.shape[-1] == self.capsule_dim:
                variable = ops.expand_dims(variable, axis=0)
            else:
                variable = ops.reshape(
                    variable,
                    ops.stack(
                        [ops.shape(variable)[0], self.num_capsules, self.capsule_dim]
                    ),
                )

        # Roll the CAPSULE axis, not the flat latent width: a roll over the flat
        # width would shift activations across capsule boundaries, which is a
        # different operator with no interpretation.
        frames = []
        for index in range(steps):
            rolled = self.roll(variable, shift=index)
            flat = ops.reshape(
                rolled,
                ops.stack([ops.shape(rolled)[0], self.latent_dim]),
            )
            frames.append(ops.squeeze(self.decode(flat, training=False), axis=1))
        return ops.stack(frames, axis=1)

    def sample_prior(
        self, num_samples: int, seed: Optional[int] = None
    ) -> Dict[str, keras.KerasTensor]:
        """Draw ``(z, u)`` from the prior and construct ``t``.

        The topographic prior is not a simple distribution to sample from
        directly; it is *constructed* from independent standard normals exactly
        as Eq. 6 defines it. Sampling it this way is what Section B.3 validates:
        reconstructions of prior draws should look like the data.

        :param num_samples: How many latent points to draw.
        :type num_samples: int
        :param seed: Optional seed for reproducibility.
        :type seed: Optional[int]
        :return: ``z``, ``u`` (and ``t``) each ``(num_samples,
            num_capsules, capsule_dim)``.
        :rtype: Dict[str, keras.KerasTensor]
        """
        shape = (num_samples, self.num_capsules, self.capsule_dim)
        z = keras.random.normal(shape=shape, seed=seed, dtype=self.compute_dtype)
        if not self.use_variance_variables:
            return {"z": z, "t": z - ops.cast(self.prior_mean, z.dtype)}

        u = keras.random.normal(shape=shape, seed=seed, dtype=self.compute_dtype)
        # A single-frame window: prior draws have no time axis to correlate.
        product = TopographicProduct(
            num_capsules=self.num_capsules,
            capsule_dim=self.capsule_dim,
            coherence_window=0,
            neighborhood_size=self.neighborhood_size,
            temporal_coherence="none",
            topography=self.topography,
            grid_shape=self.grid_shape,
            degrees_of_freedom=self.degrees_of_freedom,
            prior_mean=self.prior_mean,
            name="prior_topographic_product",
        )
        z_flat = ops.reshape(z, (num_samples, 1, self.latent_dim))
        u_flat = ops.reshape(u, (num_samples, 1, self.latent_dim))
        t = ops.reshape(
            product([z_flat, u_flat], training=False),
            shape,
        )
        return {"z": z, "u": u, "t": t}

    def log_likelihood(
        self,
        inputs,
        num_samples: int = 10,
        seed: Optional[int] = None,
        training: bool = False,
    ) -> keras.KerasTensor:
        """Importance-weighted log-likelihood estimate, ``log p(x)`` in nats.

        The tight upper bound of the IWAE family (Eq. 20 of the paper's
        evaluation), which is what Tables 1 and 2 report. Uses ``num_samples``
        draws from the approximate posterior, so it is an *estimate*: it needs
        many samples to converge and a single draw reduces to the plain ELBO.

        Returns the **sum** over pixels and sequence of the log-probability
        per sample, averaged over the batch. That is the paper's unit, and it is
        a different number from :class:`~dl_techniques.losses.topographic_vae_loss.TopographicVAELoss`
        (whose reconstruction term is a mean over pixels) — see that loss's
        docstring.

        :param inputs: Sequence of frames.
        :type inputs: Any
        :param num_samples: Importance samples per input. Must be positive.
        :type num_samples: int
        :param seed: Optional seed, which makes the estimate deterministic.
        :type seed: Optional[int]
        :param training: Forwarded to the samplers.
        :type training: bool
        :return: Per-example log-likelihood, ``(batch,)``.
        :rtype: keras.KerasTensor
        :raises ValueError: On a non-positive ``num_samples``.
        """
        if num_samples <= 0:
            raise ValueError(
                f"num_samples must be positive, got {num_samples}"
            )

        frames = ops.cast(inputs, self.compute_dtype)
        shape = ops.shape(frames)
        flat = ops.reshape(
            frames, ops.stack([shape[0], shape[1], self.frame_elements])
        )

        z_mean, z_log_var = self._encode(flat, "z")
        if self.use_variance_variables:
            u_mean, u_log_var = self._encode(flat, "u")

        log_weights = []
        for draw in range(num_samples):
            noise_seed = None if seed is None else seed + 1000 * draw
            z = z_mean + ops.exp(
                0.5 * ops.clip(z_log_var, -20.0, 20.0)
            ) * keras.random.normal(
                ops.shape(z_mean), seed=noise_seed, dtype=z_mean.dtype
            )
            if self.use_variance_variables:
                u = u_mean + ops.exp(
                    0.5 * ops.clip(u_log_var, -20.0, 20.0)
                ) * keras.random.normal(
                    ops.shape(u_mean), seed=noise_seed, dtype=u_mean.dtype
                )
                t = self.topographic_product([z, u], training=training)
            else:
                t = z - ops.cast(self.prior_mean, z.dtype)

            reconstruction = self.decode(t, training=training)
            # Sum over the FRAME elements only, keeping (batch, sequence): the
            # sequence axis must survive because the KL terms below are also
            # per-step and the two are subtracted elementwise. Summing everything
            # into a (batch,) shape here would make the subtraction below fail on
            # a shape mismatch rather than silently mis-reduce.
            log_likelihood = ops.sum(
                self._bernoulli_log_likelihood(frames, reconstruction),
                axis=list(range(2, len(self._input_shape) + 2)),
            )

            log_posterior = -0.5 * ops.sum(
                ops.square(z - z_mean)
                + z_log_var
                + 2.0 * ops.log(2.0 * np.pi),
                axis=-1,
            )
            log_prior = -0.5 * ops.sum(
                ops.square(z) + 2.0 * ops.log(2.0 * np.pi), axis=-1
            )
            log_weight = log_posterior - log_prior
            if self.use_variance_variables:
                log_posterior_u = -0.5 * ops.sum(
                    ops.square(u - u_mean)
                    + u_log_var
                    + 2.0 * ops.log(2.0 * np.pi),
                    axis=-1,
                )
                log_prior_u = -0.5 * ops.sum(
                    ops.square(u) + 2.0 * ops.log(2.0 * np.pi), axis=-1
                )
                log_weight = log_weight + log_posterior_u - log_prior_u
            log_weights.append(log_likelihood - log_weight)

        # log_weights[s] is (batch, sequence). The log-mean-exp over the SAMPLES is the
        # IWAE bound; the sequence axis is then summed so the result is a
        # per-example total, which is the paper's "log p(x)" unit.
        # log-mean-exp over the SAMPLES: a numerically stable `log mean_k exp(w_k)`,
        # written with the max pulled out. NOT negated -- `w_k` is already
        # `log p(x|z_k) + log p(z_k) - log q(z_k|x)`, so the log-mean-exp of those
        # IS the IWAE upper bound. The outer minus that used to sit here reported
        # the log-mean-exp of the NEGATED weight, which is neither a bound nor the
        # paper's quantity: for a confident decoder it returned a large POSITIVE
        # number, and its ordering against a wrong reconstruction was inverted.
        stacked = ops.stack(log_weights, axis=0)
        peak = ops.max(stacked, axis=0)
        per_example = ops.log(
            ops.mean(ops.exp(stacked - peak), axis=0)
        ) + peak
        return ops.sum(per_example, axis=-1)

    def _bernoulli_log_likelihood(self, targets, predictions) -> keras.KerasTensor:
        """Elementwise Bernoulli log-likelihood of ``targets`` under ``predictions``.

        The **log**-likelihood ``t log p + (1-t) log(1-p)``, not the NLL: a
        confident, correct decoder returns values near zero and a wrong one
        returns large NEGATIVE numbers. That sign is what makes the IWAE bound in
        :meth:`log_likelihood` an upper bound on ``log p(x)``, and therefore
        comparable to the paper's Table 1; the opposite convention reports the
        NLL and every number reads with the wrong sign.

        :param targets: Values on ``[0, 1]``.
        :type targets: keras.KerasTensor
        :param predictions: Probabilities on ``[0, 1]``.
        :type predictions: keras.KerasTensor
        :return: Elementwise log-probabilities, same shape as the inputs.
        :rtype: keras.KerasTensor
        """
        targets = ops.cast(targets, "float32")
        predictions = ops.cast(predictions, "float32")
        clipped = ops.clip(predictions, 1e-7, 1.0 - 1e-7)
        return targets * ops.log(clipped) + (1.0 - targets) * ops.log1p(-clipped)

    @property
    def sampling_seed(self) -> Optional[int]:
        """The reparameterization samplers' seed. See the constructor.

        Read AND written through to the sampler layers, so setting it after
        construction takes effect immediately. That is what lets an evaluation
        pin the seed for its duration and release it again, while training keeps
        drawing freely -- the ELBO wants an unbiased sample, and a permanently
        fixed draw would bias it.

        :return: The seed, or ``None`` for the global generator.
        :rtype: Optional[int]
        """
        return self._sampling_seed

    @sampling_seed.setter
    def sampling_seed(self, value: Optional[int]) -> None:
        self._sampling_seed = value
        for sampler in (self.z_sampling, self.u_sampling):
            if sampler is not None:
                sampler.seed = value

    @property
    def metric_trackers(self) -> List[keras.metrics.Metric]:
        """The loss trackers this model actually reports.

        These live on the compiled :class:`~dl_techniques.losses.topographic_vae_loss.TopographicVAELoss`,
        not on the model, and this property reads them from there. It returns an
        empty list when the loss is not a ``TopographicVAELoss`` (an uncompiled
        model, or a plain MSE objective), because in that case the decomposition is
        not being computed and claiming it would be a zero.

        :return: The per-term trackers, or ``[]``.
        :rtype: List[keras.metrics.Metric]
        """
        compile_loss = self._compile_loss
        user_loss = getattr(compile_loss, "_user_loss", None)
        if not isinstance(user_loss, TopographicVAELoss):
            return []
        return [
            user_loss.reconstruction_tracker,
            user_loss.z_kl_tracker,
            user_loss.u_kl_tracker,
        ]

    @classmethod
    def from_variant(
        cls,
        variant: str,
        pretrained: bool = False,
        weights_path: Optional[str] = None,
        **kwargs: Any,
    ) -> "TopographicVAE":
        """Create a TopographicVAE from a predefined variant.

        :param variant: One of ``"mnist"``, ``"dsprites"``.
        :type variant: str
        :param pretrained: Must be ``False``. This package distributes no
            pretrained weights, so ``True`` raises rather than returning a
            randomly initialized model that would silently train from scratch.
        :type pretrained: bool
        :param weights_path: Local ``.keras`` checkpoint to load after
            construction. Mutually exclusive with ``pretrained=True``.
        :type weights_path: Optional[str]
        :param kwargs: Overrides passed to the constructor.
        :type kwargs: Any
        :return: A configured model.
        :rtype: TopographicVAE
        :raises ValueError: If ``variant`` is not recognized.
        :raises NotImplementedError: If ``pretrained=True``.
        :raises ValueError: If both ``pretrained=True`` and ``weights_path`` are
            given.

        :Example:

        >>> model = TopographicVAE.from_variant("mnist")
        >>> model = TopographicVAE.from_variant("dsprites", coherence_window=5)
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant {variant!r}. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )
        if pretrained and weights_path is not None:
            raise ValueError(
                "pass either pretrained=True or weights_path=<file>, not both: "
                "there are no released weights to fetch, so pretrained=True "
                "raises rather than fetching anything"
            )
        if pretrained:
            raise NotImplementedError(
                f"TopographicVAE ships no pretrained weights for variant "
                f"{variant!r} (nor for any other variant). To load a trained "
                f"model, pass weights_path='path/to/checkpoint.keras' to "
                f"from_variant, or call TopographicVAE.from_variant(variant). "
                f"then load_weights(...). Omit pretrained to get a "
                f"randomly-initialized model."
            )

        config = dict(cls.MODEL_VARIANTS[variant])
        config.update(kwargs)
        model = cls(**config)
        if weights_path is not None:
            model.load_weights(weights_path)
        return model

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "input_shape": self._input_shape,
            "sequence_length": self.sequence_length,
            "num_capsules": self.num_capsules,
            "capsule_dim": self.capsule_dim,
            "coherence_window": self.coherence_window,
            "neighborhood_size": self.neighborhood_size,
            "temporal_coherence": self.temporal_coherence,
            "topography": self.topography,
            "grid_shape": self.grid_shape,
            "use_variance_variables": self.use_variance_variables,
            "degrees_of_freedom": self.degrees_of_freedom,
            "encoder_hidden_dims": list(self.encoder_hidden_dims),
            "decoder_hidden_dims": list(self.decoder_hidden_dims),
            "prior_mean": self.prior_mean,
            "activation": self.activation_name,
            "final_activation": self.final_activation_name,
            "kernel_initializer": keras.initializers.serialize(
                self.kernel_initializer
            ),
            "use_bias": self.use_bias,
            "sampling_seed": self.sampling_seed,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TopographicVAE":
        """Create a model from a configuration dictionary.

        :param config: Configuration dictionary.
        :type config: Dict[str, Any]
        :return: A new model instance.
        :rtype: TopographicVAE
        """
        config = dict(config)
        if config.get("kernel_initializer"):
            config["kernel_initializer"] = keras.initializers.deserialize(
                config["kernel_initializer"]
            )
        # Keras' own base-class keys are FORWARDED, never popped: a from_config
        # that strips `name` and `trainable` reloads a frozen model unfrozen
        # with bit-identical outputs, which no value test would catch.
        return cls(**config)

# ---------------------------------------------------------------------


def create_topographic_vae(
    variant: str = "mnist",
    optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
    learning_rate: float = 1e-4,
    kl_loss_weight: float = 1.0,
    reconstruction_sum_reduction: bool = False,
    pretrained: bool = False,
    **kwargs: Any,
) -> TopographicVAE:
    """Create and compile a TopographicVAE with its ELBO loss attached.

    :param variant: One of ``"mnist"``, ``"dsprites"``.
    :type variant: str
    :param optimizer: Optimizer name or instance.
    :type optimizer: Union[str, keras.optimizers.Optimizer]
    :param learning_rate: Learning rate applied to a named optimizer.
    :type learning_rate: float
    :param kl_loss_weight: Weight on the two KL terms; see
        :class:`~dl_techniques.losses.topographic_vae_loss.TopographicVAELoss`.
    :type kl_loss_weight: float
    :param reconstruction_sum_reduction: Whether the ELBO's reconstruction is a
        sum over pixels (making ``kl_loss_weight`` the paper's ``beta``) or a
        mean. Defaults to ``False``.
    :type reconstruction_sum_reduction: bool
    :param pretrained: Must be ``False``; see :meth:`TopographicVAE.from_variant`.
    :type pretrained: bool
    :param kwargs: Forwarded to :meth:`TopographicVAE.from_variant`.
    :type kwargs: Any
    :return: A compiled model, ready for training.
    :rtype: TopographicVAE

    :Example:

    >>> model = create_topographic_vae("mnist")
    >>> model = create_topographic_vae(
    ...     "dsprites", coherence_window=5, neighborhood_size=1
    ... )
    """
    model = TopographicVAE.from_variant(
        variant=variant, pretrained=pretrained, **kwargs
    )

    if isinstance(optimizer, str):
        optimizer_instance = keras.optimizers.get(optimizer)
        if hasattr(optimizer_instance, "learning_rate"):
            optimizer_instance.learning_rate = learning_rate
    else:
        optimizer_instance = optimizer

    loss = TopographicVAELoss(
        kl_loss_weight=kl_loss_weight,
        reconstruction_sum_reduction=reconstruction_sum_reduction,
        name="topographic_elbo",
    )
    model.compile(optimizer=optimizer_instance, loss=loss)

    # The sibling `vae` factory logs count_params() here. This model cannot:
    # count_params() on an unbuilt layer RAISES, and this factory deliberately
    # does not run a dummy forward pass to build it (same reasoning as the
    # sibling's DECISION D-078). The parameter count is reported by `summary()`
    # and by the trainer's run log, after the first batch has built the model.
    logger.info(
        f"Created compiled TopographicVAE-{variant} "
        f"(latent_dim={model.latent_dim}, sequence_length={model.sequence_length}): "
        f"kl_loss_weight={kl_loss_weight}"
    )
    return model

# ---------------------------------------------------------------------