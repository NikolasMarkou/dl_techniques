"""
SelfIteratePoolCallback — epoch-boundary input regeneration over a bounded RAM
patch pool, the core mechanism that makes a bias-free additive-Gaussian denoiser
self-iterable (applying ``f`` 2-5 times in sequence improves rather than
over-smooths).

plan_2026-06-20_88705c63/D-001 + D-003 context: the model ``f`` is a blind,
single-pass direct-prediction denoiser; naive re-application over-smooths. To make
``f`` robust against its own output we train it toward the clean fixed point — so
that ``f(noisy)≈clean``, ``f(f(noisy))≈clean``, AND ``f(clean)≈clean``. This is
realized WITHOUT a custom ``train_step`` and WITHOUT graph unrolling: the model is
applied exactly once per forward. "Feeding the previous result back" happens
BETWEEN epochs — this callback runs ``model.predict`` over a bounded RAM pool at
epoch cadence and writes regenerated inputs back into a LIVE numpy buffer that the
tf.data source reads (mutated IN PLACE). The trainer's pool-backed dataset is built
via ``from_generator`` over the same live buffer (D-004), so ``model.fit``
re-reads the mutated pool on the next epoch.

Two pool arrays:
  * ``clean_pool``    — FIXED regression targets, never mutated.
  * ``current_input`` — the SAME mutable array the tf.data source indexes; this
    callback overwrites slots of it in place each regeneration.

Mix-not-replace (D-003): each regeneration mixes a ``mix_ratio`` fraction of slots
filled with the model's previous output fed back AS-IS (no re-injected noise) with
the remaining slots refilled with FRESH ``clean + N(0, sigma)`` additive noise. The
union keeps ``f(noisy)≈clean`` anchored (single-pass quality does not drift) while
adding the ``f(f(noisy))≈clean`` / ``f(clean)≈clean`` signal. Additive only —
multiplicative/composite break the Miyasawa residual=score identity (LESSONS).

Serialization: ``get_config`` returns only JSON scalars; the live arrays
(``clean_pool``, ``current_input``) and the ``get_sigma`` callable are NOT
serialized. A callback reconstructed via ``from_config`` is non-functional until
``clean_pool``, ``current_input`` and ``get_sigma`` are re-attached at construction
(these are live training state, re-supplied by the trainer — same contract as
``noise_sigma_curriculum.py``'s live Variable). History is kept in memory only.
"""

from typing import Callable, Dict, List, Optional

import keras
import numpy as np

from dl_techniques.utils.logger import logger


@keras.saving.register_keras_serializable()
class SelfIteratePoolCallback(keras.callbacks.Callback):
    """Regenerate denoiser training inputs at epoch boundaries over a RAM pool.

    On ``on_epoch_end`` (at ``regen_freq`` cadence) the callback runs
    ``model.predict`` over the entire ``current_input`` pool, then overwrites a
    ``mix_ratio`` fraction of slots with the model's output fed back AS-IS
    (regenerated) and refills the remaining slots with fresh additive-noise inputs
    drawn from the FIXED ``clean_pool``. The writes happen IN PLACE on
    ``current_input`` so the tf.data source (built via ``from_generator`` over the
    same buffer) serves the mutated inputs on the next epoch.

    The model is applied exactly once per forward — there is no custom
    ``train_step`` and no graph unrolling; single-pass memory is unchanged.

    Args:
        clean_pool: ``[N, H, W, C]`` float32 array in ``[-1, +1]``. The FIXED
            regression targets. Never mutated. Not serialized.
        current_input: ``[N, H, W, C]`` float32 array, SAME shape as ``clean_pool``.
            The mutable buffer the tf.data source reads; mutated IN PLACE by this
            callback. Not serialized.
        get_sigma: Zero-arg callable returning the current noise sigma (float) used
            to generate fresh-noise inputs. Pass a plain callable so a live
            ``keras.Variable`` can be wrapped as ``lambda: float(var)``. Not
            serialized.
        regen_freq: Regenerate every ``regen_freq`` epochs (cadence). ``1`` = every
            epoch.
        mix_ratio: Fraction in ``[0, 1]`` of pool slots filled with regenerated
            (fed-back) outputs each cadence. The rest get fresh additive noise.
            ``0.0`` degenerates to "fresh noisy only"; ``1.0`` to "regenerated only".
        predict_batch_size: Batch size for the epoch-end ``model.predict`` over the
            pool.
        seed: Seed for the internal ``np.random.default_rng`` (slot permutation +
            fresh-noise generation).

    Note:
        A deserialized instance (via ``from_config``) is NON-FUNCTIONAL until
        ``clean_pool``, ``current_input`` and ``get_sigma`` are re-attached — these
        are live training state re-supplied at construction, never serialized.

    Raises:
        ValueError: if ``clean_pool`` and ``current_input`` shapes differ, or if
            ``mix_ratio`` is outside ``[0, 1]``.
    """

    def __init__(
        self,
        clean_pool: Optional[np.ndarray] = None,
        current_input: Optional[np.ndarray] = None,
        get_sigma: Optional[Callable[[], float]] = None,
        regen_freq: int = 1,
        mix_ratio: float = 0.5,
        predict_batch_size: int = 32,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if not (0.0 <= mix_ratio <= 1.0):
            raise ValueError(f"mix_ratio must be in [0, 1]; got {mix_ratio}.")
        if regen_freq < 1:
            raise ValueError(f"regen_freq must be >= 1; got {regen_freq}.")
        if clean_pool is not None and current_input is not None:
            if clean_pool.shape != current_input.shape:
                raise ValueError(
                    "clean_pool and current_input must have identical shapes; got "
                    f"{clean_pool.shape} vs {current_input.shape}."
                )

        self.clean_pool = clean_pool
        self.current_input = current_input
        self.get_sigma = get_sigma
        self.regen_freq = int(regen_freq)
        self.mix_ratio = float(mix_ratio)
        self.predict_batch_size = int(predict_batch_size)
        self.seed = int(seed)

        self._rng = np.random.default_rng(seed)
        self.epochs_seen: int = 0
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "mean_residual": [],
            "sigma": [],
            "n_regen": [],
        }

    # ------------------------------------------------------------------
    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        if (epoch + 1) % self.regen_freq != 0:
            return
        # Fail-soft: a regeneration error must NOT kill training (mirror
        # DenoisingVisualizationCallback). On failure the pool keeps its previous
        # contents and that epoch's regeneration is simply skipped.
        try:
            if (
                self.clean_pool is None
                or self.current_input is None
                or self.get_sigma is None
            ):
                logger.warning(
                    "SelfIteratePoolCallback: live pool/sigma not attached; "
                    "regeneration is a no-op this run."
                )
                return

            denoised = self.model.predict(
                self.current_input,
                batch_size=self.predict_batch_size,
                verbose=0,
            )

            n = self.current_input.shape[0]
            m = int(round(self.mix_ratio * n))

            # DECISION plan_2026-06-20_88705c63/D-003: mix-not-replace +
            # feed-back-as-is + additive-only. Do NOT replace the whole pool with
            # regenerated outputs (single-pass quality would drift toward the
            # model's own over-smoothing). Do NOT re-inject noise onto the fed-back
            # outputs (the user contract is "apply f as-is 2-5 times"; the clean
            # image must be a fixed point). Fresh slots use ADDITIVE Gaussian only
            # (multiplicative/composite break the Miyasawa residual=score identity).
            perm = self._rng.permutation(n)
            regen_idx = perm[:m]
            fresh_idx = perm[m:]

            # Regenerated slots: feed the model output back AS-IS (no re-injected
            # noise), clipped to the valid domain.
            self.current_input[regen_idx] = np.clip(denoised[regen_idx], -1.0, 1.0)

            # Fresh slots: refill from the FIXED clean targets with fresh additive
            # Gaussian noise at the current sigma.
            sigma = float(self.get_sigma())
            noise = self._rng.normal(
                size=self.current_input[fresh_idx].shape
            ).astype(np.float32)
            self.current_input[fresh_idx] = np.clip(
                self.clean_pool[fresh_idx] + noise * sigma, -1.0, 1.0
            )

            mean_residual = float(
                np.mean(np.abs(self.current_input - denoised))
            )
            self.history["epoch"].append(int(epoch))
            self.history["mean_residual"].append(mean_residual)
            self.history["sigma"].append(sigma)
            self.history["n_regen"].append(int(m))
            self.epochs_seen += 1

            logger.info(
                f"SelfIteratePool: epoch {epoch} regenerated {m}/{n} slots "
                f"(fresh={n - m}), sigma={sigma:.4f}, "
                f"mean|current-denoised|={mean_residual:.5f}"
            )
        except Exception as exc:  # fail-soft: never kill training
            logger.warning(
                f"SelfIteratePoolCallback: regeneration failed at epoch {epoch} "
                f"({exc!r}); pool unchanged, skipping this regeneration."
            )

    # ------------------------------------------------------------------
    def get_config(self) -> dict:
        # Live training state (clean_pool, current_input, get_sigma) is
        # intentionally NOT serialized; the trainer re-supplies it at construction.
        return {
            "regen_freq": self.regen_freq,
            "mix_ratio": self.mix_ratio,
            "predict_batch_size": self.predict_batch_size,
            "seed": self.seed,
        }

    @classmethod
    def from_config(cls, config: dict) -> "SelfIteratePoolCallback":
        # keras.callbacks.Callback provides no default from_config; reconstruct with
        # no live arrays/sigma attached. The instance is non-functional until
        # clean_pool/current_input/get_sigma are re-attached at construction.
        return cls(**config)
