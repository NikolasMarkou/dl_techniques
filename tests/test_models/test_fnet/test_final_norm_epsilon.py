"""RED proof for N-04 -- ``FNet``'s own final norm ignored ``layer_norm_eps``.

``FNet`` validates, stores and serialises ``layer_norm_eps`` (default
``DEFAULT_LAYER_NORM_EPSILON`` = 1e-12) and forwards it to ``BertEmbeddings``.
The stack-final normalization built for the ``'pre'`` layout --
``create_normalization_layer(normalization_type, name="final_norm")`` -- was
constructed with no ``epsilon`` at all, so it ran at the norms factory's own
default of 1e-6: a 1e6x mismatch between the model's last normalization and
every other normalization in the same model, silently, for any caller who set
the knob and for the default configuration alike.

``epsilon`` is a VALUE knob with no weights of its own, so this is asserted by
running the SUBTREE -- ``model.final_norm`` -- rather than the whole model.
A whole-model output diff would be unsound: ``layer_norm_eps`` already reaches
the embeddings and the encoder blocks at HEAD, so two arms differ there whether
or not the final norm honours it.
"""

import numpy as np
import pytest

from dl_techniques.models.fnet.model import FNet

from ..knob_sensitivity_oracle import as_array, build_seeded

FNET_CONFIG = dict(
    vocab_size=64,
    hidden_size=16,
    num_layers=1,
    intermediate_size=32,
    max_position_embeddings=16,
    hidden_dropout_prob=0.0,
    normalization_position="pre",
)

IDS = np.array([[3, 5, 7, 9, 11, 13, 2, 1]], dtype="int32")

#: A hidden state with a deliberately TINY variance, so the epsilon term is a
#: large fraction of ``var + eps`` and the two arms separate by a wide margin.
#: With unit-variance input the 1e-1 arm would still differ, but only in the
#: third decimal, and the test would be measuring float noise instead.
H = (np.random.default_rng(3).random((1, 8, 16)).astype("float32") - 0.5) * 1e-2


def _build(eps: float) -> FNet:
    model = build_seeded(lambda: FNet(layer_norm_eps=eps, **FNET_CONFIG))
    model(IDS, training=False)
    return model


class TestFNetFinalNormEpsilon:
    """The model's own trailing norm must run at the epsilon it was configured with."""

    def test_final_norm_exists_for_the_pre_norm_layout(self):
        """Guard for the two probes below: a ``None`` final norm would make both
        of them vacuous, and the 'pre' layout is the only one that builds it."""
        assert _build(1e-12).final_norm is not None

    def test_layer_norm_eps_reaches_the_final_norm(self):
        low = _build(1e-12)
        high = _build(1e-1)
        out_low = as_array(low.final_norm(H))
        out_high = as_array(high.final_norm(H))
        delta = float(np.max(np.abs(out_low - out_high)))
        assert delta > 0.0, (
            "layer_norm_eps does not reach FNet.final_norm: eps=1e-12 and "
            f"eps=1e-1 give BIT-IDENTICAL outputs (max|delta| = {delta:.6e}) for "
            "the same input and the same (ones/zeros) norm weights. The trailing "
            "norm is running at the norms-factory default."
        )

    def test_the_final_norm_matches_the_stack_it_closes(self):
        """The actual contract, not just "the knob does something".

        The failure mode this catches is a fix that forwards *an* epsilon --
        e.g. a hard-coded 1e-5 -- and so passes the probe above while leaving
        the model's last normalization disagreeing with every other one in it.
        """
        model = _build(1e-9)
        assert float(model.final_norm.epsilon) == pytest.approx(1e-9), (
            f"final_norm runs at epsilon={model.final_norm.epsilon} while the "
            "rest of the model runs at layer_norm_eps=1e-9"
        )
