"""GUI-free core wrapper around the frozen bias-free CliffordUNet denoiser.

This module holds the single load-bearing "prior" object for the inverse-problem
app: :class:`DenoiserPrior`. It loads the frozen denoiser :math:`D`, exposes the
Miyasawa/Tweedie score estimate :math:`f(y) = D(y) - y`, and provides the
domain-normalization helpers every caller needs. It imports NO GUI framework
(streamlit stays out of the core, INV-7 / H7).

Loading contract (F1 H2 / INV-2)
--------------------------------
``keras.models.load_model`` on the saved ``.keras`` file FAILS unless the
registrar module ``dl_techniques.models.bias_free_denoisers.bfcliffordunet`` has
been *imported* (executed) first, so that ``GaborFiltersInitializer`` and the
Clifford / Laplacian / LayerScale custom objects are present in the Keras
serialization registry. A bare ``import dl_techniques`` is NOT enough (its
``__init__`` is empty). This mirrors the canonical loader at
``src/train/bfunet/eval_per_pixel_uncertainty.py:109-177`` (``_load_denoiser``).

Resolution contract (F1 §2 / D-006)
-----------------------------------
The saved graph's ``Input`` is hard-locked to ``(256, 256, 3)`` (the training
patch size) and raises on any other spatial size. Two loader paths are provided:

* ``resolution="dynamic"`` (DEFAULT, D-006): rebuild the fully-convolutional
  architecture at ``input_shape=(None, None, 3)`` via the SAME factory
  (:func:`create_cliffordunet_denoiser`) and transfer weights with
  :func:`load_weights_from_checkpoint`. Runs any ``H, W`` divisible by ``8`` in a
  single pass. Factory kwargs are reconstructed from the checkpoint's sibling
  ``config.json`` mapped through ``CLIFFORDUNET_CONFIGS[variant] + overrides``
  (mirroring the trainer's ``build_model``) — never hand-copied resolved numbers.
* ``resolution="fixed256"`` (fallback): load the saved ``.keras`` graph directly
  (locked to ``(256, 256, 3)``); use :meth:`tile` / :meth:`untile` for larger
  inputs.

Domain contract (F1 §3 / D-002 / INV-1)
---------------------------------------
All pixels live in ``[-0.5, +0.5]`` with domain center ``c0 = 0.0``. Ingest via
``(x / 255) - 0.5``; denorm via ``+0.5`` back to ``[0, 1]`` for display/export.
The ``residual = score`` identity is only valid in this trained domain.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import keras
import numpy as np

from dl_techniques.utils.logger import logger
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint

# NumPy/array image input (host-side ingest helpers operate on concrete arrays).
ArrayLike = Union[np.ndarray, "keras.KerasTensor"]

# Default checkpoint artifact names inside a training results directory.
_DEFAULT_KERAS_NAME = "best_model.keras"
_CONFIG_JSON_NAME = "config.json"

# Numeric domain constants (INV-1 / D-002). Kept as attributes on instances too,
# but centralized here as the single source of truth for the module.
DOMAIN_CENTER = 0.0
DOMAIN_HALFWIDTH = 0.5


class DenoiserPrior:
    """Frozen bias-free denoiser wrapped as an implicit image prior.

    Wraps a loaded Keras denoiser :math:`D` and exposes the residual
    :math:`f(y) = D(y) - y` as the score estimate used by the inverse-problem
    solver, plus domain ingest/denorm helpers. Construct either from a checkpoint
    via :meth:`from_pretrained` or directly from an in-memory model via
    ``DenoiserPrior(model)`` (used by unit tests and callers that already hold a
    denoiser).

    Attributes:
        model: The wrapped Keras denoiser (single-output, bias-free, homogeneous).
        domain_center: Center of the pixel domain, ``0.0`` (INV-1 / D-002).
        domain_halfwidth: Half-width of the pixel domain, ``0.5`` (INV-1 / D-002).
    """

    def __init__(self, model: keras.Model) -> None:
        """Wrap an already-loaded denoiser model.

        Args:
            model: A built Keras denoiser mapping ``[B, H, W, 3]`` in
                ``[-0.5, +0.5]`` to a same-shaped estimate of the clean image.
        """
        self.model = model
        self.domain_center: float = DOMAIN_CENTER
        self.domain_halfwidth: float = DOMAIN_HALFWIDTH

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        *,
        resolution: str = "dynamic",
        input_shape: Tuple[Optional[int], Optional[int], int] = (None, None, 3),
    ) -> "DenoiserPrior":
        """Load the frozen denoiser from a training checkpoint.

        The registrar module ``bfcliffordunet`` is imported BEFORE any
        ``keras.models.load_model`` call so the custom objects (Gabor initializer,
        Clifford blocks, Laplacian pyramid, LayerScale) resolve from the registry
        (F1 H2 / INV-2). On any failure the error is logged and re-raised.

        Args:
            checkpoint_path: Path to the saved ``.keras`` file OR to the training
                results directory containing ``best_model.keras`` +
                ``config.json``.
            resolution: ``"dynamic"`` (DEFAULT, D-006) rebuilds at ``input_shape``
                and transfers weights for arbitrary ``H, W`` (÷8). ``"fixed256"``
                loads the saved ``(256, 256, 3)``-locked graph directly.
            input_shape: Rebuild input shape for the ``"dynamic"`` path. Defaults
                to ``(None, None, 3)`` (any spatial size divisible by 8).

        Returns:
            A :class:`DenoiserPrior` wrapping the loaded model.

        Raises:
            ValueError: If ``resolution`` is not ``"dynamic"`` or ``"fixed256"``.
            FileNotFoundError: If the resolved checkpoint / config paths are absent.
        """
        # Registrar-first import (INV-2). This MUST precede any load_model call —
        # bare `import dl_techniques` does not register the custom objects.
        import dl_techniques.models.bias_free_denoisers.bfcliffordunet  # noqa: F401

        if resolution not in ("dynamic", "fixed256"):
            raise ValueError(
                f"resolution must be 'dynamic' or 'fixed256', got {resolution!r}"
            )

        keras_path, config_path = cls._resolve_paths(checkpoint_path)
        try:
            if resolution == "dynamic":
                model = cls._build_dynamic(keras_path, config_path, input_shape)
            else:
                model = cls._load_fixed(keras_path)
        except Exception as exc:  # noqa: BLE001 — log + re-raise per F1 template
            logger.error(
                "failed to load denoiser (resolution=%s) from %s: %s",
                resolution, checkpoint_path, exc,
            )
            raise

        n_out = len(model.outputs) if isinstance(model.outputs, (list, tuple)) else 1
        logger.info(
            "loaded frozen denoiser '%s' (%s params, %d output(s), resolution=%s)",
            model.name, f"{model.count_params():,}", n_out, resolution,
        )
        return cls(model)

    @staticmethod
    def _resolve_paths(checkpoint_path: str) -> Tuple[Path, Path]:
        """Resolve the ``.keras`` file and sibling ``config.json`` paths.

        Accepts either a direct ``.keras`` file (config.json is its sibling) or a
        results directory (``best_model.keras`` + ``config.json`` inside it).

        Args:
            checkpoint_path: File or directory path.

        Returns:
            ``(keras_path, config_path)``.

        Raises:
            FileNotFoundError: If the ``.keras`` file cannot be found.
        """
        p = Path(checkpoint_path)
        if p.is_dir():
            keras_path = p / _DEFAULT_KERAS_NAME
        else:
            keras_path = p
        if not keras_path.is_file():
            raise FileNotFoundError(f"denoiser checkpoint not found: {keras_path}")
        config_path = keras_path.parent / _CONFIG_JSON_NAME
        return keras_path, config_path

    @staticmethod
    def _load_fixed(keras_path: Path) -> keras.Model:
        """Load the saved ``(256, 256, 3)``-locked graph directly (compile-free)."""
        model = keras.models.load_model(keras_path, compile=False)
        return model

    @classmethod
    def _build_dynamic(
        cls,
        keras_path: Path,
        config_path: Path,
        input_shape: Tuple[Optional[int], Optional[int], int],
    ) -> keras.Model:
        """Rebuild the architecture at ``input_shape`` and transfer weights.

        # DECISION plan_2026-07-06_d6b88914/D-006: this rebuild + weight-transfer
        # is the DEFAULT loader, NOT `keras.models.load_model` fed non-256 inputs.
        # The saved graph's Input is hard-locked to (256,256,3) (F1 §2) and RAISES
        # on any other spatial size, so we reconstruct the fully-convolutional
        # architecture at (None,None,3) via the SAME factory and copy weights by
        # name. Reconstruct factory kwargs by mapping config.json through
        # CLIFFORDUNET_CONFIGS[variant] + overrides (mirroring the trainer's
        # build_model); do NOT hand-copy resolved numbers (drift risk, F1 §Risks).
        # See decisions.md D-006.
        """
        from dl_techniques.models.bias_free_denoisers.bfcliffordunet import (
            create_cliffordunet_denoiser,
        )

        factory_kwargs = cls._factory_kwargs_from_config(config_path)
        model = create_cliffordunet_denoiser(input_shape=input_shape, **factory_kwargs)
        # Functional models are built at construction; weight_transfer requires it.
        report = load_weights_from_checkpoint(
            model, str(keras_path), skip_prefixes=(), strict=False,
        )
        if report.shape_mismatch:
            logger.warning(
                "dynamic rebuild had %d shape-mismatched layer(s): %s",
                len(report.shape_mismatch),
                [name for name, *_ in report.shape_mismatch][:10],
            )
        return model

    @staticmethod
    def _factory_kwargs_from_config(config_path: Path) -> Dict[str, Any]:
        """Map a checkpoint ``config.json`` to ``create_cliffordunet_denoiser`` kwargs.

        Faithfully mirrors ``train_cliffordunet_denoiser.build_model``: start from
        ``CLIFFORDUNET_CONFIGS[variant]`` (drop ``description``), apply the same
        field overrides, and resolve ``final_projection_groups == -1`` to the
        channel count. The per-level topology (``depth`` / ``initial_filters`` /
        ``blocks_per_level`` / ``shifts`` / ``drop_path_rate``) is delivered via
        the variant config + overrides — never hand-copied resolved numbers (D-006).

        Args:
            config_path: Path to the checkpoint's sibling ``config.json``.

        Returns:
            Keyword arguments for :func:`create_cliffordunet_denoiser` (excluding
            ``input_shape``, which the caller supplies).

        Raises:
            FileNotFoundError: If ``config.json`` is absent (required for rebuild).
        """
        from dl_techniques.models.bias_free_denoisers.bfcliffordunet import (
            CLIFFORDUNET_CONFIGS,
        )

        if not config_path.is_file():
            raise FileNotFoundError(
                f"config.json required for dynamic rebuild but not found: "
                f"{config_path}. Use resolution='fixed256' to load the saved graph."
            )
        raw = json.loads(config_path.read_text())

        variant = raw.get("variant", "base")
        if variant not in CLIFFORDUNET_CONFIGS:
            raise ValueError(
                f"config.json variant {variant!r} not in "
                f"{list(CLIFFORDUNET_CONFIGS)}"
            )
        # Variant topology base; overridable fields mirror build_model.
        cfg = CLIFFORDUNET_CONFIGS[variant].copy()
        cfg.pop("description", None)
        if raw.get("initial_filters") is not None:
            cfg["initial_filters"] = raw["initial_filters"]
        if raw.get("shifts") is not None:
            cfg["shifts"] = list(raw["shifts"])
        if raw.get("depth") is not None:
            cfg["depth"] = raw["depth"]
        if raw.get("blocks_per_level") is not None:
            cfg["blocks_per_level"] = raw["blocks_per_level"]

        channels = raw.get("channels", 3)
        # -1 means one group per output channel (groups == channels), per build_model.
        fpg_raw = raw.get("final_projection_groups", 1)
        final_projection_groups = channels if fpg_raw == -1 else fpg_raw

        # Non-topology factory kwargs (defaults match the factory / TrainingConfig
        # when absent from config.json).
        factory_kwargs: Dict[str, Any] = dict(
            filter_multiplier=raw.get("filter_multiplier", 2.0),
            cli_mode=raw.get("cli_mode", "full"),
            ctx_mode=raw.get("ctx_mode", "abs"),
            layer_scale_init=raw.get("layer_scale_init", 1e-5),
            use_gabor_stem=raw.get("use_gabor_stem", False),
            gabor_filters=raw.get("gabor_filters", 32),
            gabor_kernel_size=raw.get("gabor_kernel_size", 7),
            gabor_stem_projection=raw.get("gabor_stem_projection", True),
            use_laplacian_pyramid=raw.get("use_laplacian_pyramid", False),
            zero_pad_channels=raw.get("zero_pad_channels", False),
            final_projection_groups=final_projection_groups,
            downsample_pool_type=raw.get("downsample_pool_type", "max"),
            enable_deep_supervision=False,
            expose_bottleneck=False,
            final_activation="linear",  # HARD invariant: bias-free homogeneity.
            model_name=f"cliffordunet_denoiser_{variant}",
        )
        factory_kwargs.update(cfg)
        return factory_kwargs

    # ------------------------------------------------------------------
    # Core symbolic methods
    # ------------------------------------------------------------------

    def denoise(self, y: ArrayLike) -> "keras.KerasTensor":
        """Return :math:`D(y)`, the denoiser's clean-image estimate.

        Args:
            y: A ``[B, H, W, 3]`` tensor in ``[-0.5, +0.5]``.

        Returns:
            ``D(y)``, same shape as ``y``.
        """
        out = self.model(y, training=False)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out

    def residual(self, y: ArrayLike) -> "keras.KerasTensor":
        """Return the score estimate :math:`f(y) = D(y) - y`.

        This is the Miyasawa/Tweedie residual the inverse-problem solver treats as
        the (scaled) score of the implicit prior. Thin symbolic method — no host
        transfer, no branching.

        Args:
            y: A ``[B, H, W, 3]`` tensor in ``[-0.5, +0.5]``.

        Returns:
            ``D(y) - y``, same shape as ``y``.
        """
        return keras.ops.subtract(self.denoise(y), y)

    # ------------------------------------------------------------------
    # Domain helpers (INV-1 / D-002)
    # ------------------------------------------------------------------

    @staticmethod
    def ingest(image: ArrayLike) -> np.ndarray:
        """Normalize an input image to the model domain ``[-0.5, +0.5]`` (float32).

        Domain rule (D-002), applied by inspecting dtype + value range:

        * ``uint8`` inputs, or float inputs whose max exceeds ``1.5``, are treated
          as ``[0, 255]``: ``x / 255 - 0.5``.
        * float inputs within ``[0, 1]`` are treated as ``[0, 1]``: ``x - 0.5``.
        * anything else is assumed to be ALREADY in ``[-0.5, +0.5]`` and returned
          unchanged (as float32).

        Args:
            image: An array-like image (uint8 ``[0,255]``, float ``[0,1]``, or an
                already-normalized float ``[-0.5,0.5]``).

        Returns:
            A float32 ``numpy.ndarray`` in ``[-0.5, +0.5]``.
        """
        x = np.asarray(image)
        is_uint8 = x.dtype == np.uint8
        x = x.astype(np.float32)
        if is_uint8 or float(x.max(initial=0.0)) > 1.5:
            return x / 255.0 - 0.5
        lo, hi = float(x.min(initial=0.0)), float(x.max(initial=0.0))
        if lo >= -1e-6 and hi <= 1.0 + 1e-6:
            return x - 0.5
        return x  # already normalized to [-0.5, +0.5]

    @staticmethod
    def denorm(x: ArrayLike) -> np.ndarray:
        """Map model-domain ``[-0.5, +0.5]`` back to ``[0, 1]`` for display/export.

        Inverse of the ``[0,1]`` branch of :meth:`ingest` (``x + 0.5``).

        Args:
            x: An array-like in ``[-0.5, +0.5]``.

        Returns:
            A float32 ``numpy.ndarray`` in ``[0, 1]``.
        """
        return np.asarray(x, dtype=np.float32) + 0.5

    # ------------------------------------------------------------------
    # Fixed-256 tiling helpers (fallback path)
    # ------------------------------------------------------------------

    @staticmethod
    def tile(
        image: ArrayLike, tile_size: int = 256,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """Split ``[B, H, W, C]`` into non-overlapping ``tile_size`` blocks.

        Zero-pads ``H``/``W`` up to a multiple of ``tile_size`` when needed; the
        padding is recorded in ``meta`` so :meth:`untile` crops back exactly.

        Args:
            image: A ``[B, H, W, C]`` array.
            tile_size: Square tile edge length (default 256).

        Returns:
            ``(tiles, meta)`` where ``tiles`` is ``[B * nh * nw, tile_size,
            tile_size, C]`` and ``meta`` records ``batch, orig_h, orig_w, nh, nw,
            tile_size`` for reconstruction.
        """
        x = np.asarray(image)
        if x.ndim != 4:
            raise ValueError(f"tile expects a 4-D [B,H,W,C] array, got shape {x.shape}")
        b, h, w, c = x.shape
        nh = (h + tile_size - 1) // tile_size
        nw = (w + tile_size - 1) // tile_size
        pad_h, pad_w = nh * tile_size - h, nw * tile_size - w
        if pad_h or pad_w:
            x = np.pad(x, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode="constant")
        # [B, nh, T, nw, T, C] -> [B, nh, nw, T, T, C] -> [B*nh*nw, T, T, C]
        x = x.reshape(b, nh, tile_size, nw, tile_size, c)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        tiles = x.reshape(b * nh * nw, tile_size, tile_size, c)
        meta = {
            "batch": b, "orig_h": h, "orig_w": w,
            "nh": nh, "nw": nw, "tile_size": tile_size,
        }
        return tiles, meta

    @staticmethod
    def untile(tiles: ArrayLike, meta: Dict[str, int]) -> np.ndarray:
        """Reconstruct the original ``[B, H, W, C]`` image from :meth:`tile` output.

        Args:
            tiles: A ``[B * nh * nw, tile_size, tile_size, C]`` array.
            meta: The metadata dict returned by :meth:`tile`.

        Returns:
            The reconstructed ``[B, orig_h, orig_w, C]`` array (padding cropped).
        """
        x = np.asarray(tiles)
        b, nh, nw = meta["batch"], meta["nh"], meta["nw"]
        t = meta["tile_size"]
        c = x.shape[-1]
        # [B*nh*nw, T, T, C] -> [B, nh, nw, T, T, C] -> [B, nh, T, nw, T, C]
        x = x.reshape(b, nh, nw, t, t, c)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(b, nh * t, nw * t, c)
        return x[:, : meta["orig_h"], : meta["orig_w"], :]
