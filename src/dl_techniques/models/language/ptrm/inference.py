"""PTRM Inference: stochastic rollouts with Q-head selection.

Implements the Probabilistic Tiny Recursive Model (PTRM) inference procedure
from Sghaier et al. (2026). PTRM runs K parallel stochastic rollouts by
injecting Gaussian noise into the reasoning latent ``z_L`` at each supervision
step, and selects the final answer by the Q head's halting logit ``q_halt``.

References:
    - Sghaier et al., 2026. Probabilistic Tiny Recursive Model.
      (https://arxiv.org/abs/2605.19943)
"""

import keras
from typing import Dict, Tuple, Optional, Any
import numpy as np


class PTRMInference:
    """PTRM inference engine: K parallel stochastic rollouts with Q-head selection.

    Implements the "PTRM Inference" procedure of Sghaier et al. (2026),
    arXiv:2605.19943, Figure 4 (left):

    PTRM Inference
    --------------
    Input: puzzle x, rollouts K, supervision steps D, noise scale σ
    for k=1,…,K in parallel do
        Initialize z₀⁽ᵏ⁾, y₀⁽ᵏ⁾
        for t=1,…,D do
            z_{t-1}⁽ᵏ⁾ += ε,  ε ∼ 𝒩(0, σ²I)
            z_t⁽ᵏ⁾, y_t⁽ᵏ⁾ ← rec(x, z_{t-1}⁽ᵏ⁾, y_{t-1}⁽ᵏ⁾)
        end for
        ŷ⁽ᵏ⁾ ← argmax f_O(y_D⁽ᵏ⁾)
        q̂⁽ᵏ⁾ ← f_Q(y_D⁽ᵏ⁾)
    end for
    return ŷ⁽ᵏ*⁾, k* = argmax_k q̂⁽ᵏ⁾

    The `rec()` function is one deep recursion step, one supervision step of
    the model. In the paper (Sec. 2) and the reference ``trm.py`` it is one
    shared 2-layer network ``f`` run as ``z <- f(x + y + z)`` ``n`` times, then
    ``y <- f(y + z)`` (``trm.py``: ``z_L <- L_level(z_L, z_H + x)``,
    ``z_H <- L_level(z_H, z_L)``, with ``H_cycles - 1`` extra no-grad cycles).
    This repository's TRM (``TRMInner.call``) computes, per supervision step,
    ``z_L <- L_level(z_L + x)`` then ``z_H <- H_level(z_H + z_L)``: two separate
    networks, one update each, no inner ``n``-step recursion, and ``z_L``
    never reads ``z_H``.

    Paper names to code names: ``z`` (reasoning latent) is ``z_L`` and ``y``
    (answer state, read by ``f_O`` = ``lm_head`` and ``f_Q`` = ``q_head``) is
    ``z_H``. The roles match the reference despite the different update: the
    heads read ``z_H`` only, and ``z_H`` is computed from the updated ``z_L``.
    Line 6 perturbs ``z_L`` only (:meth:`_inject_noise`), and line 12
    selects by ``q_halt`` (:meth:`_selection_q`).

    :param model: Trained PTRM or TRM instance.
    :param noise_scale: Standard deviation σ of Gaussian noise injected into
        ``z_L`` at each supervision step. Default 0.2 (matches paper's PPBench
        setting).
    :param use_q_continue: Default False: select by ``q_halt``, as the paper
        does. The paper's ``q̂ = f_Q(y_D)`` is the halting logit TRM trains
        against answer correctness (Sec. 2; Figure 4 left, lines 10 and 12).
        True selects by ``q_continue`` instead, an opt-in that only makes sense
        for a model trained with Q-learning halting (``no_act_continue=False``);
        under ``no_act_continue=True`` (the PTRM default) ``q_continue`` is never
        trained.
    """

    def __init__(
        self,
        model: keras.Model,
        noise_scale: float = 0.2,
        use_q_continue: bool = False,
    ) -> None:
        self.model = model
        self.noise_scale = noise_scale
        self.use_q_continue = use_q_continue

        # Verify model has required attributes
        if not hasattr(model, 'initial_carry'):
            raise ValueError("Model must have `initial_carry` method")
        if not hasattr(model, 'call'):
            raise ValueError("Model must have `call` method")

    def __call__(
        self,
        batch: Dict[str, keras.KerasTensor],
        num_rollouts: int = 100,
        supervision_steps: int = 48,
        training: bool = False,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """Run PTRM inference on a batch.

        :param batch: Input batch dictionary with key "inputs" of shape (B, seq_len).
        :param num_rollouts: Number of parallel stochastic rollouts K. Default 100.
        :param supervision_steps: Number of deep recursion steps D. Default 48.
        :param training: Whether to run in training mode. Default False.
        :return: Tuple of (best_logits, best_q_values, all_q_values) where:
            - best_logits: Logits from highest-Q rollout, shape (B, seq_len, vocab_size)
            - best_q_values: Q values of selected rollout per batch item, shape (B,)
            - all_q_values: Q values for all rollouts, shape (K, B)
        """
        # Python loop over rollouts, kept for clarity and debuggability. The
        # rollouts are independent, so tf.vectorized_map or tf.map_fn could run
        # them in parallel on an accelerator; at K=100 this loop is the main
        # inference bottleneck.

        batch_size = keras.ops.shape(batch["inputs"])[0]
        all_logits = []
        all_q_values = []

        for k in range(num_rollouts):
            outputs = self._rollout(batch, supervision_steps, training)

            # Get final logits and Q value for this rollout
            all_logits.append(outputs["logits"])  # (B, seq_len, vocab_size)
            all_q_values.append(self._selection_q(outputs))  # (B,)

        # Stack rollouts: (K, B, seq_len, vocab_size) and (K, B)
        all_logits = keras.ops.stack(all_logits, axis=0)
        all_q_values = keras.ops.stack(all_q_values, axis=0)  # (K, B)

        # Select best rollout per batch item via Q-head
        # argmax over K dimension for each batch item
        best_rollout_indices = keras.ops.argmax(all_q_values, axis=0)  # (B,)

        # Gather best logits and Q values per batch item
        # Using advanced indexing / take_along_axis equivalent
        best_logits = self._gather_best_rollouts(all_logits, best_rollout_indices)
        best_q_values = self._gather_best_q(all_q_values, best_rollout_indices)

        return best_logits, best_q_values, all_q_values

    def _rollout(
        self,
        batch: Dict[str, keras.KerasTensor],
        supervision_steps: int,
        training: bool,
    ) -> Dict[str, keras.KerasTensor]:
        """Run one noisy rollout of exactly ``supervision_steps`` deep recursions.

        The paper's procedure has no halting: its inner loop runs ``t = 1..D``
        with no exit (Sghaier et al. 2026, arXiv:2605.19943, Figure 4 left,
        lines 5-8), Sec. 5 "Models and inference" has each rollout run ``D``
        supervision steps, and Sec. 2 states that at inference "the Q head is
        not used, and the model performs N_sup supervision steps". The reference
        TRM agrees (SamsungSAILMontreal/TinyRecursiveModels,
        ``models/recursive_reasoning/trm.py``: ``halted = is_last_step``, Q-based
        halting only under ``self.training``). So every rollout runs all ``D``
        steps from the init states the reference resets a fresh carry to,
        ``H_init``/``L_init``, with noise added before each step. The model's own
        ACT logic would get in the way twice. ``initial_carry`` marks every item
        halted, so the first call resets it to ``H_init``/``L_init`` and discards
        noise added to the zero carry. And at inference the model halts an item
        whose ``q_halt > 0``, so the next call resets it again, discarding that
        step's noise and the chain so far. An item whose init state halts then
        returns the same Q value in every rollout (measured: spread exactly 0.0).
        The rollout therefore starts from the init states with ``halted`` False,
        and clears ``halted`` after every step.

        :param batch: Input batch with key ``"inputs"``.
        :param supervision_steps: Number of deep recursion steps ``D``.
        :param training: Training-mode flag passed to the model.
        :return: The model outputs of the last step.
        """
        carry = self.model.initial_carry(batch)
        not_halted = keras.ops.zeros_like(carry["halted"])
        zeros = carry["inner_carry"]["z_H"]
        carry = dict(
            carry,
            inner_carry={
                name: keras.ops.broadcast_to(
                    keras.ops.cast(init, zeros.dtype), keras.ops.shape(zeros)
                )
                for name, init in (
                    ("z_H", self.model.inner.H_init),
                    ("z_L", self.model.inner.L_init),
                )
            },
            halted=not_halted,
            current_data=dict(batch),
        )
        outputs = None
        # DECISION plan-2026-10-13T035756-5dd4d94d/D-010: WHAT NOT TO DO: do not
        # start from `initial_carry` as is, and do not keep the `halted` the model
        # returns. Either one hands the reset back to the ACT logic, which discards
        # the injected noise (see the docstring above).
        for _ in range(supervision_steps):
            carry = self._inject_noise(carry)
            carry, outputs = self.model(carry, batch, training=training)
            carry = dict(carry, halted=not_halted)
        return outputs

    def _selection_q(
        self, outputs: Dict[str, keras.KerasTensor]
    ) -> keras.KerasTensor:
        """The Q value a rollout is selected by: ``q_halt`` unless opted out.

        :param outputs: Model outputs of a rollout's last step.
        :return: Q values, shape (B,).
        """
        # DECISION plan-2026-10-13T035756-5dd4d94d/D-018: WHAT NOT TO DO: do not
        # make q_continue the default again. The paper selects by q_hat =
        # f_Q(y_D), the halting logit (arXiv:2605.19943 Fig. 4 left, lines 10
        # and 12; Sec. 2), which is q_halt; under no_act_continue=True (the PTRM
        # default) q_continue is never trained, so the old default selected by
        # an untrained logit. See decisions.md D-018.
        if self.use_q_continue:
            return outputs["q_continue_logits"]
        return outputs["q_halt_logits"]

    def _inject_noise(self, carry: Dict[str, Any]) -> Dict[str, Any]:
        """Inject Gaussian noise into the reasoning latent ``z_L`` only.

        The paper's line 6 is ``z_{t-1} += ε, ε ∼ 𝒩(0, σ²I)`` (Sghaier et al.
        2026, arXiv:2605.19943, Figure 4 left; Sec. 4: "we add Gaussian noise
        (scaled by σ) to the latent state input"). ``z`` is the reasoning
        latent, ``z_L`` in TRM's code; the answer state ``y`` is ``z_H``, which
        ``lm_head`` and ``q_head`` read, and it is left as is. Both live in
        ``carry["inner_carry"]`` with shape (B, seq_len + puzzle_emb_len,
        hidden_size).

        What the noisy ``z_L`` feeds differs from the paper. The paper's next
        step is ``z <- f(x + y + z)`` (Sec. 2; reference ``trm.py``:
        ``z_L <- L_level(z_L, z_H + x)``, one network shared with the ``y``
        update). This repository's ``TRMInner.call`` computes
        ``z_L <- L_level(z_L + x)`` and then ``z_H <- H_level(z_H + z_L)``, with
        separate ``L_level`` and ``H_level`` networks and one update of each per
        supervision step. Here the noise therefore reaches ``z_H`` only through
        the updated ``z_L``, as in the paper, but ``z_L``'s own update never
        sees ``z_H``.

        :param carry: Current carry state.
        :return: Updated carry: ``z_L`` plus noise, ``z_H`` the same tensor.
        """
        inner_carry = carry["inner_carry"]
        z_L = inner_carry["z_L"]

        # DECISION plan-2026-10-13T035756-5dd4d94d/D-018: WHAT NOT TO DO: do not
        # add noise to z_H as well. The paper perturbs z (line 6), which is z_L
        # by role: in the REFERENCE trm.py z_L <- L_level(z_L, z_H + x) and
        # z_H <- L_level(z_H, z_L), and both heads read z_H. (This repo's
        # TRMInner differs: z_L <- L_level(z_L + x), z_H <- H_level(z_H + z_L);
        # the roles are the same.) Noise on z_H perturbs the answer state y
        # directly, which the procedure does not do. See decisions.md D-018.
        noise_L = keras.random.normal(
            keras.ops.shape(z_L),
            mean=0.0,
            stddev=self.noise_scale,
            dtype=z_L.dtype,
        )

        new_carry = dict(carry)
        new_carry["inner_carry"] = {
            "z_H": inner_carry["z_H"],
            "z_L": z_L + noise_L,
        }
        return new_carry

    def _gather_best_rollouts(
        self,
        all_logits: keras.KerasTensor,
        best_indices: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Gather the best rollout logits per batch item.

        :param all_logits: Tensor of shape (K, B, seq_len, vocab_size).
        :param best_indices: Tensor of shape (B,) with indices in [0, K-1].
        :return: Tensor of shape (B, seq_len, vocab_size).
        """
        # Use take_along_axis equivalent via keras.ops
        # We need to gather along axis 0 (K dimension) for each batch item
        K, B, seq_len, vocab_size = keras.ops.shape(all_logits)

        # Create batch indices for gathering
        batch_indices = keras.ops.arange(B, dtype=best_indices.dtype)
        # We need to gather: for each batch item b, take all_logits[best_indices[b], b, :, :]
        # This is equivalent to: all_logits[best_indices, batch_indices, :, :]

        # Keras doesn't have direct advanced indexing, so we reshape and use take
        # Flatten first two dims: (K*B, seq_len, vocab_size)
        flat_logits = keras.ops.reshape(all_logits, (K * B, seq_len, vocab_size))
        # Compute flat indices: best_indices[b] * B + b
        flat_indices = best_indices * B + batch_indices
        # Gather
        best = keras.ops.take(flat_logits, flat_indices, axis=0)  # (B, seq_len, vocab_size)
        return best

    def _gather_best_q(
        self,
        all_q: keras.KerasTensor,
        best_indices: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Gather the best Q values per batch item.

        :param all_q: Tensor of shape (K, B).
        :param best_indices: Tensor of shape (B,) with indices in [0, K-1].
        :return: Tensor of shape (B,).
        """
        K, B = keras.ops.shape(all_q)
        batch_indices = keras.ops.arange(B, dtype=best_indices.dtype)
        flat_q = keras.ops.reshape(all_q, (K * B,))
        flat_indices = best_indices * B + batch_indices
        best = keras.ops.take(flat_q, flat_indices, axis=0)
        return best

    def compute_pass_at_k(
        self,
        batch: Dict[str, keras.KerasTensor],
        labels: keras.KerasTensor,
        num_rollouts: int = 100,
        supervision_steps: int = 48,
    ) -> Dict[str, float]:
        """Compute pass@1, pass@k and best-Q@k over one set of K rollouts.

        Useful for evaluation: pass@k is the oracle upper bound (how many items
        have a correct solution among the K rollouts), and best-Q@k is what the
        Q-head selection of :meth:`__call__` actually achieves on the same
        rollouts. The gap between the two measures the Q head as a verifier.

        :param batch: Input batch.
        :param labels: Ground truth labels, shape (B, seq_len).
        :param num_rollouts: Number of rollouts K.
        :param supervision_steps: Number of supervision steps D.
        :return: Dictionary with three exact-match rates in [0, 1]:
            ``pass_at_1`` (rollout 0 is correct), ``pass_at_k`` (any rollout is
            correct) and ``best_q_at_k`` (the rollout with the highest selection
            Q, as in :meth:`__call__`, is correct).
        """
        all_predictions = []
        all_q_values = []

        for k in range(num_rollouts):
            outputs = self._rollout(batch, supervision_steps, training=False)
            # (B, seq_len)
            all_predictions.append(keras.ops.argmax(outputs["logits"], axis=-1))
            all_q_values.append(self._selection_q(outputs))  # (B,)

        all_predictions = keras.ops.stack(all_predictions, axis=0)  # (K, B, seq_len)
        all_q_values = keras.ops.stack(all_q_values, axis=0)  # (K, B)

        # Compute exact match accuracy per rollout
        # Expand labels to (K, B, seq_len)
        labels_expanded = keras.ops.expand_dims(labels, axis=0)
        labels_expanded = keras.ops.tile(labels_expanded, (num_rollouts, 1, 1))

        # Exact match: all tokens correct per sequence
        exact_match = keras.ops.all(
            keras.ops.equal(all_predictions, labels_expanded), axis=-1
        )  # (K, B)

        # pass@1: first rollout correct
        pass_at_1 = keras.ops.mean(keras.ops.cast(exact_match[0], "float32"))

        # pass@k: any rollout correct per batch item
        any_correct = keras.ops.any(exact_match, axis=0)  # (B,)
        pass_at_k = keras.ops.mean(keras.ops.cast(any_correct, "float32"))

        # best-Q@k: the rollout with the highest selection Q is correct
        best_rollout_indices = keras.ops.argmax(all_q_values, axis=0)  # (B,)
        best_correct = self._gather_best_q(
            keras.ops.cast(exact_match, "float32"), best_rollout_indices
        )  # (B,)
        best_q_at_k = keras.ops.mean(best_correct)

        return {
            "pass_at_1": float(pass_at_1),
            "pass_at_k": float(pass_at_k),
            "best_q_at_k": float(best_q_at_k),
        }


def run_ptrm_inference(
    model: keras.Model,
    batch: Dict[str, keras.KerasTensor],
    num_rollouts: int = 100,
    supervision_steps: int = 48,
    noise_scale: float = 0.2,
    use_q_continue: bool = False,
    training: bool = False,
) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    """Convenience function for PTRM inference.

    :param model: Trained PTRM or TRM instance.
    :param batch: Input batch.
    :param num_rollouts: Number of rollouts K.
    :param supervision_steps: Number of supervision steps D.
    :param noise_scale: Gaussian noise scale σ (into ``z_L``).
    :param use_q_continue: Default False selects by ``q_halt``, as the paper
        does; True selects by ``q_continue`` (see :class:`PTRMInference`).
    :param training: Training mode flag.
    :return: Tuple of (best_logits, best_q_values).
    """
    inference = PTRMInference(model, noise_scale=noise_scale, use_q_continue=use_q_continue)
    best_logits, best_q, _ = inference(batch, num_rollouts, supervision_steps, training)
    return best_logits, best_q