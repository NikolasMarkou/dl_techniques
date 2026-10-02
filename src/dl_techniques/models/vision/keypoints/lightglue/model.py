"""
LightGlue: local feature matching between two sets of keypoints.

Defines :class:`LightGlue`, a transformer that takes the keypoints and descriptors
of two images and predicts which keypoint of one image corresponds to which keypoint
of the other. It is detector-agnostic: any extractor that yields keypoint positions
and one descriptor per keypoint can feed it. Each of ``num_layers`` layers runs a
rotary self-attention on each image, then a bidirectional cross-attention between the
images, and ends with a :class:`MatchAssignment` head that reads the descriptors as a
soft assignment with a dustbin row and column. Positions enter only through a
learned-Fourier rotary encoding of the keypoints that every self-attention layer shares.

This module defines the STATIC, fully masked path, ``call()``. It always runs every
layer, prunes nothing and returns the per-layer log assignments and token confidences,
so it can be trained, jit-compiled and batched. Padded keypoint slots are described by
the optional ``mask0`` / ``mask1`` inputs and take no part in the result. The adaptive
inference path (early exit on depth confidence, point pruning on width confidence) is a
separate method; the constructor stores ``depth_confidence`` and ``width_confidence`` for
it and ``call()`` ignores them.

Architecture:
    ::

        keypoints0/1, image_size0/1                 descriptors0/1
              |                                           |
              | normalise per image                       | stop_gradient (the
              | (p - size/2) / (max(size)/2)              |  reference detaches them)
              v                                           v
        posenc: LearnedFourierRotaryEncoding        input_proj (Dense, or identity
              |   cos/sin table per image                  when input_dim == descriptor_dim)
              |                                           |
              |   +---------------- layer i = 0 .. L-1 --------------------+
              +-->| self block on image 0 and on image 1 (shared weights)  |
                  | cross block between the images (shared weights)        |
                  | MatchAssignment -> log_assignments[i]  (M+1, N+1)      |
                  | MatchTokenConfidence -> confidences[i]  (i < L-1)      |
                  +--------------------------------------------------------+
                                      |
              filter_matches(log_assignments[L-1]) -> matches0/1, matching_scores0/1

Output layout. Every per-layer tensor is BATCH-FIRST so that ``predict()`` can
concatenate batches along axis 0: ``log_assignments`` is ``(B, L, M+1, N+1)`` (the
dustbin row and column are the last index) and ``token_confidences0/1`` are
``(B, L-1, M)`` / ``(B, L-1, N)``. One stacked tensor per quantity, instead of a list,
makes ``compute_output_shape`` a plain shape dict and lets a per-layer loss read layer
``i`` as ``log_assignments[:, i]``.

There is no ``MODEL_VARIANTS``: the paper publishes a single size (9 layers, width 256,
4 heads) and the defaults equal it.

References:
    - Lindenberger, P., Sarlin, P.-E., & Pollefeys, M. (2023). "LightGlue: Local
      Feature Matching at Light Speed". ICCV 2023. (https://arxiv.org/abs/2306.13643)
    - Sarlin, P.-E., DeTone, D., Malisiewicz, T., & Rabinovich, A. (2020). "SuperGlue:
      Learning Feature Matching with Graph Neural Networks". CVPR 2020.
      (https://arxiv.org/abs/1911.11763)
"""

import keras
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.matching.learned_fourier_rotary import LearnedFourierRotaryEncoding
from dl_techniques.layers.matching.lightglue_blocks import LightGlueCrossBlock, LightGlueSelfBlock
from dl_techniques.layers.matching.match_assignment import MatchAssignment, filter_matches
from dl_techniques.layers.matching.token_confidence import MatchTokenConfidence
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


def normalize_keypoints(keypoints: Any, image_size: Any) -> Any:
    """Map pixel keypoints to roughly ``[-1, 1]`` per image, in float32.

    Interface contract. ``keypoints`` is ``(B, N, 2)`` pixel ``(x, y)``; ``image_size``
    is ``(B, 2)`` as ``(width, height)``. The result is ``(p - size / 2) / (max(size) / 2)``
    in float32 whatever the policy, so a pixel coordinate of several hundred is not
    rounded to float16 before it is scaled. Pure function, no variables.

    :param keypoints: ``(B, N, 2)`` pixel coordinates.
    :param image_size: ``(B, 2)`` ``(width, height)``.
    :return: ``(B, N, 2)`` float32.
    """
    kpts = keras.ops.cast(keypoints, "float32")
    size = keras.ops.cast(image_size, "float32")
    shift = size / 2.0                                              # (B, 2)
    scale = keras.ops.max(size, axis=-1, keepdims=True) / 2.0       # (B, 1)
    return (kpts - shift[:, None, :]) / scale[:, None, :]


def _detach_descriptors(descriptors: Any) -> Any:
    """Stop gradient into the input descriptors, as the reference ``descriptors.detach()``.

    Its own function so a test can replace it and show that the zero-gradient guard
    can fail.
    """
    return keras.ops.stop_gradient(descriptors)


@register_dl_technique("dl_techniques.models.lightglue.model")
class LightGlue(keras.Model):
    """LightGlue matcher, static masked training path.

    :param input_dim: Width of the incoming descriptors. A ``Dense`` projection to
        ``descriptor_dim`` is created when it differs, else the projection is the
        identity (reference behaviour). Default 256.
    :type input_dim: int
    :param descriptor_dim: Internal node width ``d``. Must be divisible by ``num_heads``
        with an even head dimension. Default 256.
    :type descriptor_dim: int
    :param num_layers: Number of transformer layers ``L`` (at least 1). Default 9.
    :type num_layers: int
    :param num_heads: Attention heads. Default 4.
    :type num_heads: int
    :param filter_threshold: Match threshold on ``exp(score)`` used by the final
        :func:`filter_matches`, in ``[0, 1]``. Default 0.1.
    :type filter_threshold: float
    :param depth_confidence: Early-exit confidence for the adaptive path, ``-1`` to
        disable, else in ``[0, 1]``. Stored; ``call()`` ignores it. Default 0.95.
    :type depth_confidence: float
    :param width_confidence: Point-pruning confidence for the adaptive path, ``-1`` to
        disable, else in ``[0, 1]``. Stored; ``call()`` ignores it. Default 0.99.
    :type width_confidence: float
    :param add_scale_ori: Append keypoint scale and orientation to the positional
        encoding input (feature width 4 instead of 2). Needs ``scales0/1`` and
        ``oris0/1`` in the inputs. Default False.
    :type add_scale_ori: bool
    :param gamma: Scale of the positional-encoding frequency initialiser (kernel std
        ``gamma ** -2``). Default 1.0.
    :type gamma: float
    :param kwargs: Extra arguments for ``keras.Model``. ``autocast`` is forced off: the
        keypoint coordinates and the positional-encoding kernel must stay float32 under
        a mixed policy, and every sublayer casts its own inputs.

    Input:
        A dict with ``keypoints0`` ``(B, M, 2)`` and ``keypoints1`` ``(B, N, 2)`` pixel
        ``(x, y)``; ``descriptors0`` ``(B, M, input_dim)`` and ``descriptors1``
        ``(B, N, input_dim)``; ``image_size0`` and ``image_size1`` ``(B, 2)`` as
        ``(width, height)``. Optional ``mask0`` ``(B, M)`` and ``mask1`` ``(B, N)``,
        1 = real keypoint (default all real). With ``add_scale_ori``: ``scales0/1`` and
        ``oris0/1``, each ``(B, M)`` / ``(B, N)``.

    Output:
        A dict with ``log_assignments`` ``(B, L, M+1, N+1)`` (float32 or wider; padded
        rows and columns are 0), ``token_confidences0`` ``(B, L-1, M)`` and
        ``token_confidences1`` ``(B, L-1, N)``, and from the last layer
        ``matches0`` ``(B, M)`` / ``matches1`` ``(B, N)`` int32 (partner index or -1) and
        ``matching_scores0`` / ``matching_scores1``.

    :raises ValueError: From ``__init__`` for a non-positive size, a ``descriptor_dim``
        that does not split into an even head dimension, or a threshold out of range.

    Example:

    .. code-block:: python

        model = create_lightglue(input_dim=256)
        out = model({
            "keypoints0": kp0, "keypoints1": kp1,            # (B, N, 2) pixels
            "descriptors0": d0, "descriptors1": d1,          # (B, N, 256)
            "image_size0": size0, "image_size1": size1,      # (B, 2) as (w, h)
        })
        matches0 = out["matches0"]                           # (B, N), -1 = unmatched
    """

    def __init__(
            self,
            input_dim: int = 256,
            descriptor_dim: int = 256,
            num_layers: int = 9,
            num_heads: int = 4,
            filter_threshold: float = 0.1,
            depth_confidence: float = 0.95,
            width_confidence: float = 0.99,
            add_scale_ori: bool = False,
            gamma: float = 1.0,
            **kwargs: Any
    ) -> None:
        # See the class docstring: keypoints and the posenc kernel must stay float32.
        kwargs["autocast"] = False
        super().__init__(**kwargs)

        if input_dim <= 0 or descriptor_dim <= 0 or num_layers <= 0 or num_heads <= 0:
            raise ValueError(
                "input_dim, descriptor_dim, num_layers and num_heads must be positive, got "
                f"{input_dim}, {descriptor_dim}, {num_layers}, {num_heads}"
            )
        if descriptor_dim % num_heads != 0 or (descriptor_dim // num_heads) % 2 != 0:
            raise ValueError(
                f"descriptor_dim ({descriptor_dim}) must split into num_heads ({num_heads}) "
                "heads of even width: the rotary encoding rotates adjacent channel pairs"
            )
        if not 0.0 <= filter_threshold <= 1.0:
            raise ValueError(f"filter_threshold must be in [0, 1], got {filter_threshold}")
        for name, value in (("depth_confidence", depth_confidence),
                            ("width_confidence", width_confidence)):
            if value != -1 and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be -1 (disabled) or in [0, 1], got {value}")
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")

        self.input_dim = input_dim
        self.descriptor_dim = descriptor_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.filter_threshold = filter_threshold
        self.depth_confidence = depth_confidence
        self.width_confidence = width_confidence
        self.add_scale_ori = add_scale_ori
        self.gamma = gamma
        self.head_dim = descriptor_dim // num_heads
        self.num_pos_features = 4 if add_scale_ori else 2

        policy = self.dtype_policy
        self.input_proj = (
            keras.layers.Dense(descriptor_dim, dtype=policy, name="input_proj")
            if input_dim != descriptor_dim else None
        )
        # autocast off: the keypoints and the kernel reach the layer in float32 under a
        # mixed policy, and the layer runs its trigonometry in a never-narrowing dtype.
        self.posenc = LearnedFourierRotaryEncoding(
            head_dim=self.head_dim,
            num_input_features=self.num_pos_features,
            gamma=gamma,
            autocast=False,
            dtype=policy,
            name="posenc",
        )
        self.self_blocks: List[LightGlueSelfBlock] = []
        self.cross_blocks: List[LightGlueCrossBlock] = []
        self.assignments: List[MatchAssignment] = []
        self.confidences: List[MatchTokenConfidence] = []
        for i in range(num_layers):
            self.self_blocks.append(LightGlueSelfBlock(
                dim=descriptor_dim, num_heads=num_heads, dtype=policy, name=f"self_attn_{i}"))
            self.cross_blocks.append(LightGlueCrossBlock(
                dim=descriptor_dim, num_heads=num_heads, dtype=policy, name=f"cross_attn_{i}"))
            self.assignments.append(MatchAssignment(
                dim=descriptor_dim, dtype=policy, name=f"log_assignment_{i}"))
            if i < num_layers - 1:
                self.confidences.append(MatchTokenConfidence(
                    dim=descriptor_dim, dtype=policy, name=f"token_confidence_{i}"))

        logger.info(
            f"Created LightGlue (input_dim={input_dim}, descriptor_dim={descriptor_dim}, "
            f"num_layers={num_layers}, num_heads={num_heads}, add_scale_ori={add_scale_ori})"
        )

    # -----------------------------------------------------------------
    # build
    # -----------------------------------------------------------------

    def build(self, input_shape: Optional[Dict[str, Tuple[Any, ...]]] = None) -> None:
        """Build every sublayer in forward order from the configuration.

        All shapes are fixed by the constructor arguments (the point counts and the batch
        stay symbolic), so building does not depend on which optional keys the first call
        carries. Building here, not lazily in ``call``, makes every weight exist before a
        ``.keras`` weight restore.

        :param input_shape: Dict of input shapes. When it carries ``descriptors0/1`` their
            last dimension is checked against ``input_dim``.
        :raises ValueError: If a descriptor width differs from ``input_dim``.
        """
        if self.built:
            return
        if isinstance(input_shape, dict):
            for key in ("descriptors0", "descriptors1"):
                shape = input_shape.get(key)
                if shape is not None and shape[-1] is not None and shape[-1] != self.input_dim:
                    raise ValueError(
                        f"{key} last dimension ({shape[-1]}) must equal input_dim ({self.input_dim})"
                    )
        node = (None, None, self.descriptor_dim)
        if self.input_proj is not None:
            self.input_proj.build((None, None, self.input_dim))
        self.posenc.build((None, None, self.num_pos_features))
        freqs = (2, None, 1, None, self.head_dim)
        for i in range(self.num_layers):
            self.self_blocks[i].build(node, freqs)
            self.cross_blocks[i].build(node, node)
            self.assignments[i].build(node, node)
            if i < self.num_layers - 1:
                self.confidences[i].build(node, node)
        super().build(input_shape)

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------

    def _positions(self, inputs: Dict[str, Any], side: str) -> Any:
        """Normalised keypoints, with scale and orientation appended when configured."""
        kpts = normalize_keypoints(inputs["keypoints" + side], inputs["image_size" + side])
        if self.add_scale_ori:
            scales = keras.ops.cast(inputs["scales" + side], "float32")[..., None]
            oris = keras.ops.cast(inputs["oris" + side], "float32")[..., None]
            kpts = keras.ops.concatenate([kpts, scales, oris], axis=-1)
        return kpts

    def _embed(self, descriptors: Any) -> Any:
        """Detach, cast to the compute dtype, project to ``descriptor_dim``."""
        desc = keras.ops.cast(_detach_descriptors(descriptors), self.compute_dtype)
        return desc if self.input_proj is None else self.input_proj(desc)

    def call(self, inputs: Dict[str, Any], training: Optional[bool] = None) -> Dict[str, Any]:
        """Run all layers with padding masks and return per-layer assignments.

        :param inputs: Input dict, see the class docstring.
        :param training: Unused; the model has no stochastic layer. Kept for the Keras
            call contract.
        :return: Output dict, see the class docstring.
        """
        mask0, mask1 = inputs.get("mask0"), inputs.get("mask1")
        if mask0 is not None or mask1 is not None:
            # One-sided masks are completed with all-real so every block sees both.
            if mask0 is None:
                mask0 = keras.ops.ones_like(inputs["keypoints0"][..., 0])
            if mask1 is None:
                mask1 = keras.ops.ones_like(inputs["keypoints1"][..., 0])

        freqs0 = self.posenc(self._positions(inputs, "0"))
        freqs1 = self.posenc(self._positions(inputs, "1"))
        desc0 = self._embed(inputs["descriptors0"])
        desc1 = self._embed(inputs["descriptors1"])

        log_assignments, conf0, conf1 = [], [], []
        for i in range(self.num_layers):
            desc0 = self.self_blocks[i](desc0, freqs0, mask0)
            desc1 = self.self_blocks[i](desc1, freqs1, mask1)
            desc0, desc1 = self.cross_blocks[i](desc0, desc1, mask0, mask1)
            scores, _ = self.assignments[i](desc0, desc1, mask0, mask1)
            log_assignments.append(scores)
            if i < self.num_layers - 1:
                c0, c1 = self.confidences[i](desc0, desc1)
                conf0.append(c0)
                conf1.append(c1)

        if conf0:
            token_confidences0 = keras.ops.stack(conf0, axis=1)
            token_confidences1 = keras.ops.stack(conf1, axis=1)
        else:
            # a single layer has no confidence head: empty (B, 0, M) / (B, 0, N)
            token_confidences0 = keras.ops.zeros_like(log_assignments[0][:, None, :-1, 0])[:, :0]
            token_confidences1 = keras.ops.zeros_like(log_assignments[0][:, None, 0, :-1])[:, :0]

        matches0, matches1, scores0, scores1 = filter_matches(
            log_assignments[-1], self.filter_threshold, mask0, mask1)
        return {
            "log_assignments": keras.ops.stack(log_assignments, axis=1),
            "token_confidences0": token_confidences0,
            "token_confidences1": token_confidences1,
            "matches0": matches0,
            "matches1": matches1,
            "matching_scores0": scores0,
            "matching_scores1": scores1,
        }

    def compute_output_shape(self, input_shape: Dict[str, Tuple[Any, ...]]) -> Dict[str, Tuple]:
        """Output shapes from the input shape dict.

        :param input_shape: Dict with at least ``keypoints0`` ``(B, M, 2)`` and
            ``keypoints1`` ``(B, N, 2)``.
        :return: Dict of shapes keyed like the output of :meth:`call`.
        """
        batch, m_pts = input_shape["keypoints0"][0], input_shape["keypoints0"][1]
        n_pts = input_shape["keypoints1"][1]
        m_plus = None if m_pts is None else m_pts + 1
        n_plus = None if n_pts is None else n_pts + 1
        return {
            "log_assignments": (batch, self.num_layers, m_plus, n_plus),
            "token_confidences0": (batch, self.num_layers - 1, m_pts),
            "token_confidences1": (batch, self.num_layers - 1, n_pts),
            "matches0": (batch, m_pts),
            "matches1": (batch, n_pts),
            "matching_scores0": (batch, m_pts),
            "matching_scores1": (batch, n_pts),
        }

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument.

        :return: Config dict.
        """
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "descriptor_dim": self.descriptor_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "filter_threshold": self.filter_threshold,
            "depth_confidence": self.depth_confidence,
            "width_confidence": self.width_confidence,
            "add_scale_ori": self.add_scale_ori,
            "gamma": self.gamma,
        })
        return config


# ---------------------------------------------------------------------


def create_lightglue(
        input_dim: int = 256,
        descriptor_dim: int = 256,
        num_layers: int = 9,
        num_heads: int = 4,
        **kwargs: Any
) -> LightGlue:
    """Build a LightGlue matcher.

    The defaults are the published size (9 layers, width 256, 4 heads). There are no
    named variants and no pretrained weights are available.

    :param input_dim: Width of the incoming descriptors (SuperPoint: 256). Default 256.
    :param descriptor_dim: Internal node width. Default 256.
    :param num_layers: Number of layers. Default 9.
    :param num_heads: Number of attention heads. Default 4.
    :param kwargs: Forwarded to :class:`LightGlue` (``filter_threshold``,
        ``depth_confidence``, ``width_confidence``, ``add_scale_ori``, ``gamma``, ...).
    :return: A :class:`LightGlue` instance.
    :raises ValueError: On an invalid size or threshold, from the constructor.
    """
    return LightGlue(
        input_dim=input_dim,
        descriptor_dim=descriptor_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        **kwargs,
    )

# ---------------------------------------------------------------------
