"""Class-label embedding with a classifier-free-guidance dropout token.

This module provides :class:`ClassLabelEmbedding`, a lookup table over
``num_classes`` labels plus **one extra row** that stands for "no label".
During training a label is replaced by that extra row with probability
``dropout_rate``, so the same network learns both the conditional and the
unconditional score. At sampling time the caller forces the extra row for the
unconditional branch of classifier-free guidance.

Ported from the ``LabelEmbedder`` that appears in DiT and every DiT descendant
(Peebles & Xie 2023, *Scalable Diffusion Models with Transformers*), whose CFG
recipe is Ho & Salimans 2022, *Classifier-Free Diffusion Guidance*.

Architecture:

.. code-block:: text

    labels (B,) int          force_drop_ids (B,) int, optional
      |                        |
      |   dropout_rate > 0 ?   |   present ?
      |        |               |      |
      |        v               |      v
      |   training=True  --> drop = uniform(B) < dropout_rate
      |                      OR  (when force_drop_ids given, ANY training value)
      |                          drop = (force_drop_ids == 1)
      |        |
      v        v
    labels = where(drop, num_classes, labels)
      |
      v
    Embedding(num_classes + (1 if dropout_rate > 0 else 0), hidden_size)
      |
      v
    (B, hidden_size)

    Table layout, dropout_rate > 0:

        row 0 .. num_classes-1   the real classes
        row num_classes          the CFG / "unconditional" row   <-- the +1

    Table layout, dropout_rate == 0:

        row 0 .. num_classes-1   the real classes
        (no extra row exists, and force_drop_ids is then an error)

Why this layer exists, given the factory's other 13 keys:
    ``create_embedding_layer()``'s registry was surveyed before this class was
    written. None of the existing keys is this:

    * ``bert_embeddings``, ``modern_bert_embeddings``, ``albert_factorized`` are
      **token/vocabulary** embeddings. They are the right shape (a lookup table)
      and the wrong contract: they add positional and token-type streams, they
      normalize, and none of them has a dropout token. Using one for three
      prompt-kind labels would carry a normalizer, a position table and a
      segment table that condition on nothing.
    * ``positional_learned`` is a table indexed by POSITION, added to a sequence
      rather than looked up by an arbitrary label.
    * ``scalar_sinusoidal`` and ``continuous_sincos`` embed CONTINUOUS values.
      A class label is categorical; a sinusoid of the integer 2 is not a class
      identity.
    * ``patch_*``, the four RoPE variants and ``positional_sine_2d`` are all
      positional.

    Nor does the near-miss exist elsewhere in the tree: ``ideogram4``'s CFG is an
    inference-time blend of two forward passes with no dropout token, and
    ``sd3_mmdit`` conditions on continuous pooled embeddings, so neither owns a
    reusable label table. The gap is real, and the mechanism (a categorical table
    with a learned null row) is standard across the whole conditional-diffusion
    family, which is why this lives in ``layers/embedding/`` with a factory key
    rather than inside one model package.

The ``training`` flag, and how it differs from the PyTorch original:
    Upstream's ``LabelEmbedder.forward(labels, train, force_drop_ids=None)``
    takes the flag as an explicit **argument**, which the caller fills from
    ``self.training`` -- PyTorch's persistent per-module flag, flipped by
    ``.train()`` / ``.eval()``. Keras 3 has no such persistent flag: ``training``
    arrives as a ``call()`` keyword that Keras itself supplies (``True`` inside
    ``fit()``, ``False`` inside ``predict()``/``evaluate()``). This layer
    therefore reads ``call(..., training=...)`` and stores nothing. Do **not**
    add a mutable ``self.training`` attribute to "match upstream": it would not
    be graph-safe and Keras would never set it. ``training=None`` -- the default,
    meaning "no ambient context said otherwise" -- is treated as False, which is
    the same convention ``keras.layers.Dropout`` uses.

Naming divergence from upstream:
    Upstream calls the drop probability ``dropout_prob``. This port spells it
    ``dropout_rate``, which is this repository's convention -- one name for one
    concept across 1400-odd parameters under ``layers/``, closed by
    ``tests/test_the_dropout_rate_naming_convention_holds.py``. A reader
    diffing this file against ``reference/dit.py`` should expect that one
    rename and no other. There is deliberately no ``dropout_prob`` alias: the
    layer is new and unreleased, so there is nothing to be compatible with, and
    an alias would be a second name for one concept plus dead code.

Serialization:
    ``get_config()`` returns every constructor argument, with the initializer
    serialized, under the same names ``__init__`` takes -- so the emitted key is
    ``dropout_rate``. The embedding table itself is an ordinary trainable weight
    of the inner :class:`keras.layers.Embedding`, so it round-trips normally.
"""

from typing import Any, Dict, Optional, Tuple, Union

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

__all__ = ["ClassLabelEmbedding"]


@register_dl_technique(
    package="dl_techniques.layers.embedding.class_label_embedding"
)
class ClassLabelEmbedding(keras.layers.Layer):
    """Embed integer class labels, with an optional CFG dropout row.

    The table holds ``num_classes + 1`` rows when ``dropout_rate > 0`` and
    exactly ``num_classes`` rows when it is ``0``. The extra row is index
    ``num_classes``; it is what a dropped label is replaced by, and what the
    unconditional branch of classifier-free guidance asks for explicitly.

    :param num_classes: Number of real classes. Valid labels are
        ``0 .. num_classes - 1``.
    :type num_classes: int
    :param hidden_size: Width of each embedding row.
    :type hidden_size: int
    :param dropout_rate: Probability of replacing a label with the CFG row, per
        sample, per call, when ``training`` is true. ``0.0`` disables the
        mechanism **and removes the extra row from the table**, so the table
        size is a visible function of this argument rather than a hidden one.
    :type dropout_rate: float
    :param embeddings_initializer: Initializer for the table. A **fresh copy** is
        made in :meth:`build`, so passing one instance to several layers cannot
        make them draw identically.
    :type embeddings_initializer: Union[str, keras.initializers.Initializer]
    :param seed: Seed for the layer's own :class:`keras.random.SeedGenerator`.
        The drop mask never touches the global RNG, so two layers with different
        seeds drop different samples in the same step.
    :type seed: Optional[int]
    :param kwargs: Standard ``keras.layers.Layer`` keyword arguments.

    :raises ValueError: If ``num_classes`` or ``hidden_size`` is not positive, or
        if ``dropout_rate`` is outside ``[0, 1]``.

    Input shape:
        Integer tensor of shape ``(B,)`` or ``(B, 1)``. A trailing singleton axis
        is squeezed.

    Output shape:
        ``(B, hidden_size)``.

    Example:
        >>> import keras
        >>> layer = ClassLabelEmbedding(num_classes=3, hidden_size=8,
        ...                             dropout_rate=0.1, seed=0)
        >>> labels = keras.ops.convert_to_tensor([0, 1, 2])
        >>> layer(labels, training=False).shape
        (3, 8)
        >>> # the unconditional branch of CFG: force every label to the CFG row
        >>> uncond = layer(labels,
        ...                force_drop_ids=keras.ops.ones((3,), dtype="int32"))
        >>> uncond.shape
        (3, 8)

    Note:
        ``force_drop_ids`` is honoured regardless of ``training``, matching
        upstream: the sampler needs the unconditional row at inference time, when
        ``training`` is false. It is an ERROR when ``dropout_rate == 0``, because
        the row it asks for does not exist; upstream would gather out of bounds
        instead, which is silent on some backends.

    Attributes:
        embedding_table: The inner :class:`keras.layers.Embedding`.
        table_size: ``num_classes + 1`` when ``dropout_rate > 0``, else
            ``num_classes``.
        seed_generator: The layer's :class:`keras.random.SeedGenerator`, created
            in :meth:`build`.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_size: int,
        dropout_rate: float = 0.0,
        embeddings_initializer: Union[
            str, keras.initializers.Initializer
        ] = "uniform",
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if not 0.0 <= float(dropout_rate) <= 1.0:
            raise ValueError(
                f"dropout_rate must be in [0, 1], got {dropout_rate}"
            )

        self.num_classes = int(num_classes)
        self.hidden_size = int(hidden_size)
        self.dropout_rate = float(dropout_rate)
        self.embeddings_initializer = keras.initializers.get(
            embeddings_initializer
        )
        self.seed = seed

        # The +1 is a function of dropout_rate, exactly as upstream's
        # `num_classes + use_cfg_embedding`. Sizing the table at `num_classes`
        # while still dropping to index `num_classes` is an out-of-bounds gather
        # that is silent on some backends, so the two facts are derived from one
        # expression here rather than written twice.
        self.use_cfg_embedding = self.dropout_rate > 0.0
        self.table_size = self.num_classes + int(self.use_cfg_embedding)

        # A fresh initializer object per layer: a shared Initializer INSTANCE
        # draws bit-identically forever, and no default surfaces it.
        table_initializer = keras.initializers.deserialize(
            keras.initializers.serialize(self.embeddings_initializer)
        )
        self.embedding_table = keras.layers.Embedding(
            input_dim=self.table_size,
            output_dim=self.hidden_size,
            embeddings_initializer=table_initializer,
            name="embedding_table",
        )

        self.seed_generator = None

    def build(self, input_shape: Any) -> None:
        """Build the inner table and create the seed generator.

        :param input_shape: ``(B,)`` or ``(B, 1)``.
        :type input_shape: Any
        """
        if self.built:
            return

        shape = tuple(input_shape)
        if len(shape) > 1 and shape[-1] == 1:
            shape = shape[:-1]
        self.embedding_table.build(shape)

        self.seed_generator = keras.random.SeedGenerator(self.seed)

        super().build(input_shape)

    def _token_drop(
        self,
        labels: keras.KerasTensor,
        force_drop_ids: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        """Replace dropped labels with the CFG row index.

        :param labels: Integer labels, shape ``(B,)``.
        :type labels: keras.KerasTensor
        :param force_drop_ids: Optional ``(B,)`` tensor; a value of ``1`` forces
            the drop for that sample. When given, the random draw is not made at
            all, so a forced call consumes no RNG state.
        :type force_drop_ids: Optional[keras.KerasTensor]
        :return: Labels with dropped entries set to ``num_classes``.
        :rtype: keras.KerasTensor
        """
        if force_drop_ids is None:
            drop_ids = (
                keras.random.uniform(
                    shape=keras.ops.shape(labels),
                    seed=self.seed_generator,
                )
                < self.dropout_rate
            )
        else:
            drop_ids = keras.ops.equal(
                keras.ops.cast(force_drop_ids, labels.dtype), 1
            )
        cfg_row = keras.ops.full_like(labels, self.num_classes)
        return keras.ops.where(drop_ids, cfg_row, labels)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        force_drop_ids: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        """Look the labels up, dropping some of them first.

        The drop runs when ``training`` is true and ``dropout_rate > 0``, **or**
        whenever ``force_drop_ids`` is given -- the second clause is independent
        of ``training`` on purpose, because classifier-free guidance needs the
        unconditional row at inference time.

        :param inputs: Integer labels, ``(B,)`` or ``(B, 1)``.
        :type inputs: keras.KerasTensor
        :param training: Keras training flag. ``None`` counts as false.
        :type training: Optional[bool]
        :param force_drop_ids: Optional ``(B,)`` tensor; ``1`` forces the drop.
        :type force_drop_ids: Optional[keras.KerasTensor]
        :return: ``(B, hidden_size)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If ``force_drop_ids`` is given while
            ``dropout_rate == 0``, i.e. while the CFG row does not exist.
        """
        labels = keras.ops.cast(inputs, "int32")
        if len(labels.shape) > 1 and labels.shape[-1] == 1:
            labels = keras.ops.squeeze(labels, axis=-1)

        if force_drop_ids is not None and not self.use_cfg_embedding:
            raise ValueError(
                f"{self.__class__.__name__} '{self.name}' was constructed with "
                f"dropout_rate={self.dropout_rate}, so its table has only "
                f"{self.table_size} rows and there is no CFG row at index "
                f"{self.num_classes} for force_drop_ids to select. Construct the "
                "layer with dropout_rate > 0 if you need classifier-free "
                "guidance."
            )

        # `training` is a Python bool by the time Keras calls this; branching on
        # it is the same thing `keras.layers.Dropout` does. Do NOT try to make
        # this a `keras.ops.where` over a traced flag -- the two branches differ
        # in RNG consumption, not just in value.
        if (training and self.use_cfg_embedding) or force_drop_ids is not None:
            labels = self._token_drop(labels, force_drop_ids)

        return self.embedding_table(labels)

    def compute_output_shape(
        self, input_shape: Any
    ) -> Tuple[Optional[int], ...]:
        """Return ``(B, hidden_size)``.

        :param input_shape: ``(B,)`` or ``(B, 1)``.
        :type input_shape: Any
        :return: Input shape with a trailing singleton dropped and
            ``hidden_size`` appended.
        :rtype: Tuple[Optional[int], ...]
        """
        shape = tuple(input_shape)
        if len(shape) > 1 and shape[-1] == 1:
            shape = shape[:-1]
        return shape + (self.hidden_size,)

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument.

        :return: A JSON-serializable configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "hidden_size": self.hidden_size,
                "dropout_rate": self.dropout_rate,
                "embeddings_initializer": keras.initializers.serialize(
                    self.embeddings_initializer
                ),
                "seed": self.seed,
            }
        )
        return config
