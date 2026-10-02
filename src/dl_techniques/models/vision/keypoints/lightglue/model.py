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
separate method, :meth:`LightGlue.match`; the constructor stores ``depth_confidence``,
``width_confidence`` and ``pruning_min_kpts`` for it and ``call()`` ignores them.

The adaptive path, ``match()``, is eager and for inference only. After each layer except
the last it can stop (the confident fraction of points exceeds ``depth_confidence``) and
prune (points that are matchable-unlikely and confidently resolved are dropped, so later
layers see fewer points). Shapes therefore depend on the data, which is why it is not a
graph function. The pure decision rules (:func:`confidence_threshold`,
:func:`get_pruning_mask`, :func:`check_if_stop`) are module functions so that a test can
compare them to the reference one by one.

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
``(B, L-1, M)`` / ``(B, L-1, N)``, with ``token_logits0/1`` (pre-sigmoid) of the same
shapes. One stacked tensor per quantity, instead of a list,
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

import math

import keras
import numpy as np
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


def normalize_keypoints(keypoints: Any, image_size: Any, dtype: Optional[str] = None) -> Any:
    """Map pixel keypoints to roughly ``[-1, 1]`` per image, in a never-narrowing dtype.

    Interface contract. ``keypoints`` is ``(B, N, 2)`` pixel ``(x, y)``; ``image_size``
    is ``(B, 2)`` as ``(width, height)``. The result is ``(p - size / 2) / (max(size) / 2)``
    in a work dtype that is the widest of the input dtypes and ``dtype`` (typically the
    model compute dtype), floored at float32: float32 for float16, bfloat16, float32
    and integer inputs, float64 when any of them is float64. So a pixel coordinate of
    several hundred is never rounded to float16 before it is scaled, and a float64 model
    is not narrowed to float32 (same rule as ``LearnedFourierRotaryEncoding``). Pure
    function, no variables.

    :param keypoints: ``(B, N, 2)`` pixel coordinates.
    :param image_size: ``(B, 2)`` ``(width, height)``.
    :param dtype: Optional compute dtype of the caller; only ``"float64"`` changes the
        result.
    :return: ``(B, N, 2)`` in the work dtype (float32 or float64).
    """
    names = [keras.backend.standardize_dtype(getattr(t, "dtype", "float32"))
             for t in (keypoints, image_size)]
    names.append(dtype or "float32")
    work = "float64" if "float64" in names else "float32"
    kpts = keras.ops.cast(keypoints, work)
    size = keras.ops.cast(image_size, work)
    shift = size / 2.0                                              # (B, 2)
    scale = keras.ops.max(size, axis=-1, keepdims=True) / 2.0       # (B, 1)
    return (kpts - shift[:, None, :]) / scale[:, None, :]


def confidence_threshold(layer_index: int, num_layers: int) -> float:
    """Confidence threshold of layer ``i``: ``clip(0.8 + 0.1 * exp(-4 i / L), 0, 1)``.

    Interface contract. Pure python, no variables. The threshold falls from 0.9 at layer 0
    toward 0.8 as depth grows.

    :param layer_index: Zero-based layer index ``i``.
    :param num_layers: Total layer count ``L``.
    :return: The threshold as a float.
    """
    return float(min(max(0.8 + 0.1 * math.exp(-4.0 * layer_index / num_layers), 0.0), 1.0))


def get_pruning_mask(
        confidences: Optional[Any],
        matchability: Any,
        layer_index: int,
        num_layers: int,
        width_confidence: float,
) -> Any:
    """Boolean keep mask for one image's points, as the reference ``get_pruning_mask``.

    Interface contract. A point is KEPT when ``matchability > 1 - width_confidence`` or,
    when ``confidences`` is given, its token confidence is ``<= confidence_threshold``.
    So a low-confidence point is never pruned. ``confidences`` is ``None`` when the depth
    confidence is disabled (the confidence head was not evaluated). Pure numpy.

    :param confidences: ``(N,)`` token confidences or ``None``.
    :param matchability: ``(N,)`` matchability in ``[0, 1]``.
    :param layer_index: Zero-based layer index.
    :param num_layers: Total layer count.
    :param width_confidence: Pruning confidence in ``(0, 1]``.
    :return: ``(N,)`` bool array, True = keep.
    """
    keep = np.asarray(matchability) > (1.0 - width_confidence)
    if confidences is not None:
        keep = np.logical_or(
            keep, np.asarray(confidences) <= confidence_threshold(layer_index, num_layers))
    return keep


def check_if_stop(
        confidences0: Any,
        confidences1: Any,
        layer_index: int,
        num_layers: int,
        num_points: int,
        depth_confidence: float,
) -> bool:
    """Early-exit rule, as the reference ``check_if_stop``.

    Interface contract. The confident ratio is
    ``1 - #(confidence < threshold) / num_points`` over both images; the model stops when
    it exceeds ``depth_confidence``. ``num_points`` is the ORIGINAL ``m + n`` real
    points, so a point pruned earlier counts as confident. Pure numpy.

    :param confidences0: ``(M',)`` confidences of the points still alive in image 0.
    :param confidences1: ``(N',)`` confidences of image 1.
    :param layer_index: Zero-based layer index.
    :param num_layers: Total layer count.
    :param num_points: Original real point count ``m + n``.
    :param depth_confidence: Stop confidence in ``(0, 1]``.
    :return: True to stop at this layer.
    """
    confidences = np.concatenate([np.asarray(confidences0), np.asarray(confidences1)], axis=-1)
    below = float(np.sum(confidences < confidence_threshold(layer_index, num_layers)))
    return bool(1.0 - below / num_points > depth_confidence)


def _scatter_back(
        matches0: Any, matches1: Any, scores0: Any, scores1: Any,
        ind0: Any, ind1: Any, size0: int, size1: int,
) -> Tuple[Any, Any, Any, Any]:
    """Map matches on the surviving points back to full-size, original-index arrays.

    Interface contract. ``matches0`` ``(M',)`` holds a position in the surviving set of
    image 1 (or -1); ``ind0`` / ``ind1`` list the original index of each survivor.
    Returns ``(m0 (size0,), m1 (size1,), s0, s1)`` where pruned points are -1 / 0 and
    partners are expressed as ORIGINAL indices. Pure numpy.
    """
    out0 = -np.ones(size0, dtype=np.int32)
    out1 = -np.ones(size1, dtype=np.int32)
    sc0 = np.zeros(size0, dtype=np.float32)
    sc1 = np.zeros(size1, dtype=np.float32)
    out0[ind0] = np.where(matches0 == -1, -1, ind1[np.clip(matches0, 0, None)])
    out1[ind1] = np.where(matches1 == -1, -1, ind0[np.clip(matches1, 0, None)])
    sc0[ind0] = scores0
    sc1[ind1] = scores1
    return out0, out1, sc0, sc1


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
        disable, else in ``[0, 1]``. Used by :meth:`match` only; ``call()`` ignores it.
        Default 0.95.
    :type depth_confidence: float
    :param width_confidence: Point-pruning confidence for the adaptive path, ``-1`` to
        disable, else in ``[0, 1]``. Used by :meth:`match` only; ``call()`` ignores it.
        Default 0.99.
    :type width_confidence: float
    :param pruning_min_kpts: :meth:`match` prunes an image only while it has MORE than this
        many points alive. ``-1`` (default) always prunes, which is the reference's CPU
        setting; the reference uses 1024 on a GPU, where pruning small sets costs more than
        it saves.
    :type pruning_min_kpts: int
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
        ``token_confidences1`` ``(B, L-1, N)`` (sigmoid outputs), their pre-sigmoid logits
        ``token_logits0`` / ``token_logits1`` of the same shapes (what a loss should read:
        ``confidences = sigmoid(logits)``), and from the last layer
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
            pruning_min_kpts: int = -1,
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
        if not isinstance(pruning_min_kpts, int) or isinstance(pruning_min_kpts, bool):
            raise ValueError(f"pruning_min_kpts must be an int, got {pruning_min_kpts!r}")

        self.input_dim = input_dim
        self.descriptor_dim = descriptor_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.filter_threshold = filter_threshold
        self.depth_confidence = depth_confidence
        self.width_confidence = width_confidence
        self.pruning_min_kpts = pruning_min_kpts
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
        kpts = normalize_keypoints(
            inputs["keypoints" + side], inputs["image_size" + side], self.compute_dtype)
        if self.add_scale_ori:
            scales = keras.ops.cast(inputs["scales" + side], kpts.dtype)[..., None]
            oris = keras.ops.cast(inputs["oris" + side], kpts.dtype)[..., None]
            kpts = keras.ops.concatenate([kpts, scales, oris], axis=-1)
        return kpts

    def _embed(self, descriptors: Any) -> Any:
        """Detach, cast to the compute dtype, project to ``descriptor_dim``."""
        desc = keras.ops.cast(_detach_descriptors(descriptors), self.compute_dtype)
        return desc if self.input_proj is None else self.input_proj(desc)

    def _run_layer(self, i: int, desc0: Any, desc1: Any, freqs0: Any, freqs1: Any,
                   mask0: Optional[Any] = None, mask1: Optional[Any] = None) -> Tuple[Any, Any]:
        """Layer ``i``: self block on each image (shared weights), then the cross block.

        Shared by :meth:`call` (full size, optional masks) and :meth:`match` (pruned size,
        no masks), so the two paths cannot drift apart.

        :return: ``(desc0, desc1)`` after the layer.
        """
        desc0 = self.self_blocks[i](desc0, freqs0, mask0)
        desc1 = self.self_blocks[i](desc1, freqs1, mask1)
        return self.cross_blocks[i](desc0, desc1, mask0, mask1)

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

        log_assignments, conf0, conf1, logit0, logit1 = [], [], [], [], []
        for i in range(self.num_layers):
            desc0, desc1 = self._run_layer(i, desc0, desc1, freqs0, freqs1, mask0, mask1)
            scores, _ = self.assignments[i](desc0, desc1, mask0, mask1)
            log_assignments.append(scores)
            if i < self.num_layers - 1:
                c0, c1, z0, z1 = self.confidences[i](desc0, desc1, return_logits=True)
                conf0.append(c0)
                conf1.append(c1)
                logit0.append(z0)
                logit1.append(z1)

        if conf0:
            token_confidences0 = keras.ops.stack(conf0, axis=1)
            token_confidences1 = keras.ops.stack(conf1, axis=1)
            token_logits0 = keras.ops.stack(logit0, axis=1)
            token_logits1 = keras.ops.stack(logit1, axis=1)
        else:
            # a single layer has no confidence head: empty (B, 0, M) / (B, 0, N)
            token_confidences0 = keras.ops.zeros_like(log_assignments[0][:, None, :-1, 0])[:, :0]
            token_confidences1 = keras.ops.zeros_like(log_assignments[0][:, None, 0, :-1])[:, :0]
            token_logits0, token_logits1 = token_confidences0, token_confidences1

        matches0, matches1, scores0, scores1 = filter_matches(
            log_assignments[-1], self.filter_threshold, mask0, mask1)
        return {
            "log_assignments": keras.ops.stack(log_assignments, axis=1),
            "token_confidences0": token_confidences0,
            "token_confidences1": token_confidences1,
            "token_logits0": token_logits0,
            "token_logits1": token_logits1,
            "matches0": matches0,
            "matches1": matches1,
            "matching_scores0": scores0,
            "matching_scores1": scores1,
        }

    # -----------------------------------------------------------------
    # adaptive inference
    # -----------------------------------------------------------------

    @staticmethod
    def _real_indices(mask: Optional[Any], count: int) -> Any:
        """Indices of the real points: all of them without a mask, else ``mask > 0``."""
        if mask is None:
            return np.arange(count)
        return np.nonzero(np.asarray(keras.ops.convert_to_numpy(mask))[0] > 0)[0]

    def _empty_result(self, size0: int, size1: int, stop: int, prune0: Any, prune1: Any) -> Dict[str, Any]:
        """The reference result when one image has no point left: all -1, scores 0."""
        return {
            "matches0": -np.ones((1, size0), dtype=np.int32),
            "matches1": -np.ones((1, size1), dtype=np.int32),
            "matching_scores0": np.zeros((1, size0), dtype=np.float32),
            "matching_scores1": np.zeros((1, size1), dtype=np.float32),
            "stop": stop,
            "matches": np.zeros((0, 2), dtype=np.int32),
            "scores": np.zeros((0,), dtype=np.float32),
            "prune0": prune0,
            "prune1": prune1,
        }

    # DECISION plan-2026-10-02T084508-dd2c07ac/D-008
    # call() is the static masked path and match() is a SEPARATE eager batch-1 method.
    # Do NOT fold early exit and pruning into call(): their shapes depend on the data,
    # which breaks fit, jit and masking. The two paths are kept in sync by tests, not by
    # shared code. Guards: tests/test_models/test_lightglue/test_match.py
    # TestEqualsStaticPath::test_disabled_knobs_equal_the_last_static_layer and
    # test_torch_reference.py (match() against the official stop layer and prune counters).
    # See decisions.md D-008.
    def match(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Adaptive inference: early exit on depth confidence, point pruning on width confidence.

        This is the reference ``_forward`` and the ONLY path that uses ``depth_confidence``,
        ``width_confidence`` and ``pruning_min_kpts``. It runs eagerly (numpy control flow
        between layers) and its shapes depend on the data, so it is not a ``tf.function`` /
        ``jit`` target; use :meth:`call` for training and batched graph inference. It
        handles ONE image pair per call (batch size 1), like the reference's pruning, which
        indexes with a single keep list.

        Per layer ``i`` (every layer but the last): run the self and cross blocks; if
        ``depth_confidence > 0`` evaluate the token confidences and stop when
        :func:`check_if_stop` holds; if ``width_confidence > 0`` and an image still has more
        than ``pruning_min_kpts`` points, keep only the points of :func:`get_pruning_mask`
        (matchability of that layer's assignment head) and index-select the descriptors and
        the rotary table to them. The last layer never stops or prunes. The final assignment
        uses the head of the layer where the loop ended, on the surviving points only, then
        :func:`filter_matches`, and the result is scattered back to the original indices
        (pruned points: -1 and score 0). Optional ``mask0`` / ``mask1`` select the real
        points first; padded slots come back as -1 / 0 as well.

        Reference quirk kept on purpose: when a side runs empty after pruning, the check sits
        at the top of the next iteration, so ``stop`` is one more than the layers that ran.
        Token confidences are only computed when ``depth_confidence > 0``; pruning with
        ``depth_confidence <= 0`` uses matchability alone. With both knobs disabled the
        result equals the last layer of :meth:`call`.

        :param inputs: Input dict as for :meth:`call`, with batch size 1.
        :return: Dict of numpy values (not tensors): ``matches0`` ``(1, M)`` / ``matches1``
            ``(1, N)`` int32 with ORIGINAL partner indices or -1; ``matching_scores0`` /
            ``matching_scores1`` float32, 0 where pruned or unmatched; ``stop`` int, the
            1-based index of the layer whose assignment was used (reference value, see the
            quirk above); ``matches`` ``(S, 2)`` int32 original index pairs and ``scores``
            ``(S,)`` float32, the compact form (a list of per-image arrays in the reference,
            here the single array because the batch is 1); ``prune0`` ``(1, M)`` /
            ``prune1`` ``(1, N)`` int32: with width pruning, 1 plus the number of pruning
            steps the point survived; without it, ``num_layers`` everywhere.
        :raises ValueError: If the batch size is not 1.
        """
        kp0, kp1 = inputs["keypoints0"], inputs["keypoints1"]
        if keras.ops.shape(kp0)[0] != 1 or keras.ops.shape(kp1)[0] != 1:
            raise ValueError("match() handles one image pair per call (batch size 1)")
        size0, size1 = int(keras.ops.shape(kp0)[1]), int(keras.ops.shape(kp1)[1])
        ind0 = self._real_indices(inputs.get("mask0"), size0)
        ind1 = self._real_indices(inputs.get("mask1"), size1)
        num_points = len(ind0) + len(ind1)
        num_layers = self.num_layers

        do_early_stop = self.depth_confidence > 0
        do_point_pruning = self.width_confidence > 0
        prune0 = np.ones((size0,), dtype=np.int32)
        prune1 = np.ones((size1,), dtype=np.int32)
        if len(ind0) == 0 or len(ind1) == 0:
            if not do_point_pruning:
                prune0, prune1 = prune0 * num_layers, prune1 * num_layers
            return self._empty_result(size0, size1, 1, prune0[None], prune1[None])

        pos0, pos1 = self._positions(inputs, "0"), self._positions(inputs, "1")
        pos0 = keras.ops.take(pos0, ind0, axis=1)
        pos1 = keras.ops.take(pos1, ind1, axis=1)
        freqs0, freqs1 = self.posenc(pos0), self.posenc(pos1)
        desc0 = self._embed(keras.ops.take(inputs["descriptors0"], ind0, axis=1))
        desc1 = self._embed(keras.ops.take(inputs["descriptors1"], ind1, axis=1))

        i = 0
        for i in range(num_layers):
            if desc0.shape[1] == 0 or desc1.shape[1] == 0:
                break
            desc0, desc1 = self._run_layer(i, desc0, desc1, freqs0, freqs1)
            if i == num_layers - 1:
                continue                      # the last layer never stops or prunes
            token0 = token1 = None
            if do_early_stop:
                token0, token1 = self.confidences[i](desc0, desc1)
                token0 = keras.ops.convert_to_numpy(token0)[0]
                token1 = keras.ops.convert_to_numpy(token1)[0]
                if check_if_stop(token0, token1, i, num_layers, num_points, self.depth_confidence):
                    break
            if do_point_pruning and desc0.shape[1] > self.pruning_min_kpts:
                score0 = keras.ops.convert_to_numpy(self.assignments[i].get_matchability(desc0))[0]
                keep0 = np.nonzero(get_pruning_mask(
                    token0, score0, i, num_layers, self.width_confidence))[0]
                ind0 = ind0[keep0]
                desc0 = keras.ops.take(desc0, keep0, axis=1)
                freqs0 = keras.ops.take(freqs0, keep0, axis=3)
                prune0[ind0] += 1
            if do_point_pruning and desc1.shape[1] > self.pruning_min_kpts:
                score1 = keras.ops.convert_to_numpy(self.assignments[i].get_matchability(desc1))[0]
                keep1 = np.nonzero(get_pruning_mask(
                    token1, score1, i, num_layers, self.width_confidence))[0]
                ind1 = ind1[keep1]
                desc1 = keras.ops.take(desc1, keep1, axis=1)
                freqs1 = keras.ops.take(freqs1, keep1, axis=3)
                prune1[ind1] += 1

        if not do_point_pruning:
            prune0, prune1 = prune0 * 0 + num_layers, prune1 * 0 + num_layers
        if desc0.shape[1] == 0 or desc1.shape[1] == 0:
            return self._empty_result(size0, size1, i + 1, prune0[None], prune1[None])

        scores, _ = self.assignments[i](desc0, desc1)
        m0, m1, ms0, ms1 = filter_matches(scores, self.filter_threshold)
        m0 = keras.ops.convert_to_numpy(m0)[0]
        m1 = keras.ops.convert_to_numpy(m1)[0]
        ms0 = keras.ops.convert_to_numpy(ms0)[0]
        ms1 = keras.ops.convert_to_numpy(ms1)[0]
        valid = np.nonzero(m0 > -1)[0]
        pairs = np.stack([ind0[valid], ind1[m0[valid]]], axis=-1).astype(np.int32)
        out0, out1, sc0, sc1 = _scatter_back(m0, m1, ms0, ms1, ind0, ind1, size0, size1)
        return {
            "matches0": out0[None],
            "matches1": out1[None],
            "matching_scores0": sc0[None],
            "matching_scores1": sc1[None],
            "stop": i + 1,
            "matches": pairs,
            "scores": ms0[valid].astype(np.float32),
            "prune0": prune0[None],
            "prune1": prune1[None],
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
            "token_logits0": (batch, self.num_layers - 1, m_pts),
            "token_logits1": (batch, self.num_layers - 1, n_pts),
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
            "pruning_min_kpts": self.pruning_min_kpts,
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
