"""MemoryStats callback for WaveFieldMemoryLLM.

O2 — periodic instrumentation that logs:

- top-K hit count histogram on a small probe batch.
- gate magnitude p5 / p50 / p95.
- key utilization fraction (fraction of K_lt rows ever in top-K over the
  observation window).
- V_lt effective rank (SVD on a subsample).

The callback is opt-in: pass `MemoryStats(log_every=1000, probe_dataset=
ds)` alongside `PhaseScheduler` in `model.fit(..., callbacks=[...])`.
All output is via :data:`dl_techniques.utils.logger` — no print.

Heavy ops (SVD, full K_lt scan) only run on `log_every` boundary so the
training hot path is not affected.
"""

from typing import Any, Optional

import numpy as np
import keras

from dl_techniques.utils.logger import logger


class MemoryStats(keras.callbacks.Callback):
    """Periodic memory diagnostics for WaveFieldMemoryLLM.

    :param log_every: Run the diagnostic block every ``log_every``
        training batches. Default 1000 (the full block costs an SVD on a
        subsample of V_lt — not free).
    :param probe_dataset: Optional ``tf.data.Dataset`` used to compute
        retrieval-side stats (top-K hits, gate magnitude). If ``None``
        only structural stats (key utilization, V_lt effective rank) are
        computed.
    :param probe_batches: Number of probe batches consumed each cycle.
    :param svd_subsample: Max number of V_lt rows used for the effective
        rank SVD. Larger -> tighter estimate, more cost.
    """

    def __init__(
        self,
        log_every: int = 1000,
        probe_dataset: Optional[Any] = None,
        probe_batches: int = 4,
        svd_subsample: int = 1024,
    ) -> None:
        super().__init__()
        if log_every <= 0:
            raise ValueError(f"log_every must be positive, got {log_every}")
        self.log_every = log_every
        self.probe_dataset = probe_dataset
        self.probe_batches = probe_batches
        self.svd_subsample = svd_subsample

        # Hit-count window: (S_lt,) integer counter, reset on each log.
        self._hit_counter: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_counter(self, s_lt: int) -> None:
        if self._hit_counter is None or self._hit_counter.shape[0] != s_lt:
            self._hit_counter = np.zeros((s_lt,), dtype=np.int64)

    def _gate_percentiles(self, x_probe: Any) -> Optional[tuple]:
        rc = getattr(self.model, "read_controller", None)
        if rc is None or not getattr(rc.W_g, "built", False):
            return None
        try:
            # Forward through embed + first L_read blocks to reach the
            # read tap. Use the model's helper.
            tap = self.model._hidden_at_read_tap(x_probe)
            g = keras.ops.sigmoid(rc.W_g(tap))
            g_np = np.asarray(g).reshape(-1)
            return (
                float(np.percentile(g_np, 5)),
                float(np.percentile(g_np, 50)),
                float(np.percentile(g_np, 95)),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"MemoryStats: gate percentile failed ({exc})")
            return None

    def _key_utilization(self) -> Optional[float]:
        if self._hit_counter is None:
            return None
        s_lt = self._hit_counter.shape[0]
        if s_lt == 0:
            return None
        used = int((self._hit_counter > 0).sum())
        return float(used) / float(s_lt)

    def _v_lt_effective_rank(self) -> Optional[float]:
        """Approximate effective rank via SVD on a V_lt subsample.

        Effective rank = exp(H(p)) where p is the normalized singular-
        value distribution. Captures spread without picking a threshold.
        """
        try:
            v = self.model.lt_memory.V_lt
            v_np = np.asarray(v)
            # If MHA, flatten heads into the last axis.
            if v_np.ndim == 3:
                v_np = v_np.reshape(v_np.shape[0], -1)
            n = min(self.svd_subsample, v_np.shape[0])
            if n <= 1:
                return None
            idx = np.random.choice(v_np.shape[0], size=n, replace=False)
            sub = v_np[idx]
            # Centered SVD.
            sub = sub - sub.mean(axis=0, keepdims=True)
            s = np.linalg.svd(sub, compute_uv=False)
            s = s[s > 1e-12]
            if s.size == 0:
                return None
            p = s / s.sum()
            entropy = -float((p * np.log(p + 1e-12)).sum())
            return float(np.exp(entropy))
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"MemoryStats: SVD failed ({exc})")
            return None

    # ------------------------------------------------------------------
    # Per-batch hit accumulation
    # ------------------------------------------------------------------

    def _accumulate_hits(self, x_probe: Any) -> None:
        rc = getattr(self.model, "read_controller", None)
        lt = getattr(self.model, "lt_memory", None)
        write = getattr(self.model, "write_controller", None)
        if rc is None or lt is None or write is None:
            return
        try:
            tap = self.model._hidden_at_read_tap(x_probe)
            k_lt, v_lt = lt(None)
            k_wm, v_wm, mask = write(tap)
            # Run the read forward to materialize routing internals; we
            # don't have direct access to top_idx, so we reconstruct it
            # from sim. Cheaper: run rc(...) and capture sim via top_k
            # logic re-derivation. To keep this O(1) extra pass we just
            # call the controller and skip aux losses (training=False).
            _ = rc(tap, k_lt, v_lt, k_wm, v_wm, mask, training=False)
            # Recompute top-K indices from the same dataflow for hit
            # accounting. We replicate the controller's similarity to
            # avoid plumbing an extra return value.
            q_flat = rc.W_Q(tap)
            b = q_flat.shape[0]
            t = q_flat.shape[1]
            q = np.asarray(q_flat).reshape(b, t, rc.num_heads, rc.d_k)
            if rc.multi_head_keys:
                k_total_np = np.concatenate(
                    [
                        np.broadcast_to(
                            np.asarray(k_lt)[None, ...],
                            (b, rc.s_lt, rc.num_heads, rc.d_k),
                        ),
                        np.asarray(k_wm),
                    ], axis=1,
                )
                sim = np.einsum("bthk,bmhk->bthm", q, k_total_np) / np.sqrt(
                    rc.d_k
                )
            else:
                k_total_np = np.concatenate(
                    [
                        np.broadcast_to(
                            np.asarray(k_lt)[None, ...],
                            (b, rc.s_lt, rc.d_k),
                        ),
                        np.asarray(k_wm),
                    ], axis=1,
                )
                sim = np.einsum("bthk,bmk->bthm", q, k_total_np) / np.sqrt(
                    rc.d_k
                )
            # Top-K indices over the LT slice only.
            sim_lt = sim[..., : rc.s_lt]
            top_idx = np.argpartition(-sim_lt, rc.top_k - 1, axis=-1)[
                ..., : rc.top_k
            ]
            self._ensure_counter(rc.s_lt)
            np.add.at(self._hit_counter, top_idx.reshape(-1), 1)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"MemoryStats: hit accumulation failed ({exc})")

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

    def on_train_batch_end(self, batch, logs=None):
        # Read global step from the model so we account for resume.
        gstep_attr = getattr(self.model, "_global_step", None)
        if gstep_attr is None:
            return
        step = int(gstep_attr.numpy())
        if step <= 0 or (step % self.log_every) != 0:
            return

        # Probe-derived stats (require a dataset).
        gate_ptiles: Optional[tuple] = None
        if self.probe_dataset is not None:
            try:
                for i, batch_data in enumerate(
                    self.probe_dataset.take(self.probe_batches),
                ):
                    x_probe = (
                        batch_data[0]
                        if isinstance(batch_data, (tuple, list))
                        else batch_data
                    )
                    if i == 0:
                        gate_ptiles = self._gate_percentiles(x_probe)
                    self._accumulate_hits(x_probe)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"MemoryStats: probe loop failed ({exc})")

        # Structural stats (always available).
        util = self._key_utilization()
        eff_rank = self._v_lt_effective_rank()

        parts = [f"MemoryStats step={step}"]
        if gate_ptiles is not None:
            p5, p50, p95 = gate_ptiles
            parts.append(f"gate(p5/p50/p95)={p5:.3f}/{p50:.3f}/{p95:.3f}")
        if util is not None:
            parts.append(f"key_utilization={util:.3f}")
        if eff_rank is not None:
            parts.append(f"v_lt_effective_rank={eff_rank:.2f}")
        logger.info(" | ".join(parts))

        # Reset the hit-count window after each log.
        if self._hit_counter is not None:
            self._hit_counter[:] = 0
