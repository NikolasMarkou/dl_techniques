"""The trainer's evaluation path reads the tapped tensor of either placement.

plan-2026-10-23T190557-972a9ac5 step 11.1 (D-031 d). ``tap_placement='residual'`` moves
every tap onto the post-add tensor; the analysis reads activations through the
``taps=`` capture layers, which must see the SAME tensor as the penalty tap. These
tests drive the real :func:`extract_tap_activations` and :func:`evaluate_topography`
with the real gpt2 preprocessor on a tiny model and check that

* the responses are finite and have the shape the contrast expects, in both placements,
* with identical weights the two placements give DIFFERENT pooled responses (the
  residual stream is added), and the residual response of the first attention tap
  equals the branch response plus the pooled block input, computed independently,
* the contrast chain (t-map, joint BH-FDR, clusters) runs on a ``'residual'`` model.

CPU only (``CUDA_VISIBLE_DEVICES=``).
"""

import numpy as np
import pytest

import keras

from dl_techniques.models.language.topolm import TopoLM
from train.topolm import SMOKE_STIMULI, evaluate_topography, extract_tap_activations
from train.topolm import common as C

WINDOW = 128
PROMPTS = ["the cat sat on the mat", "a dog ran in the park", "birds fly south",
           "she wrote a long letter"]


def _preprocessor():
    return C.create_tokenizer("gpt2", 512, 50257, 50258, 50259, 50260)


def _model(placement: str, **kwargs) -> TopoLM:
    keras.utils.set_random_seed(0)
    return TopoLM(
        vocab_size=50261, embed_dim=64, depth=2, num_heads=4, max_seq_len=WINDOW,
        radius=1, tap_placement=placement, name="topolm", **kwargs,
    )


def _pair():
    """Two models, the same weights, differing only in the placement."""
    branch, residual = _model("branch"), _model("residual")
    ids = np.zeros((1, 8), "int32")
    branch(ids, training=False)
    residual(ids, training=False)
    residual.set_weights(branch.get_weights())
    return branch, residual


class TestTheEvaluationPathReadsEitherPlacement:
    @pytest.mark.parametrize("placement", ["branch", "residual"])
    def test_extract_tap_activations_is_finite_with_one_row_per_prompt(self, placement):
        model = _model(placement)
        responses = extract_tap_activations(model, PROMPTS, _preprocessor())
        assert len(responses) == len(model.tap_layers) == 4
        for path, matrix in responses.items():
            assert matrix.shape == (len(PROMPTS), 64), path
            assert np.isfinite(matrix).all(), path

    def test_the_two_placements_pool_different_responses_from_the_same_weights(self):
        branch, residual = _pair()
        a = extract_tap_activations(branch, PROMPTS, _preprocessor())
        b = extract_tap_activations(residual, PROMPTS, _preprocessor())
        assert list(a) == list(b)
        for path in a:
            assert np.abs(a[path] - b[path]).max() > 1e-3, path

    def test_the_residual_response_of_the_first_tap_is_branch_plus_the_stream(self):
        """Independent of the capture mechanism: pooled(x + attn) = pooled(x) + pooled(attn).

        The first tap of a ``'residual'`` model is ``embeddings + attention branch``,
        so over the same sentence tokens its pooled response equals the branch model's
        first tap plus the pooled block input. The block input is read from the
        model's own embedding layers over the ids the preprocessor produced.
        """
        branch, residual = _pair()
        preprocessor = _preprocessor()
        a = extract_tap_activations(branch, PROMPTS, preprocessor)
        b = extract_tap_activations(residual, PROMPTS, preprocessor)

        tokens, text, _ = C._encode_stimuli(preprocessor, PROMPTS, WINDOW,
                                            context=C.STIMULUS_CONTEXT)
        embedded = keras.ops.convert_to_numpy(
            branch.word_embeddings(tokens)
            + branch.positional_embeddings(np.arange(tokens.shape[1]))
        ).astype("float64")
        weights = text.astype("float64")
        stream = np.einsum("blu,bl->bu", embedded, weights) / weights.sum(1)[:, None]

        first = branch.tap_layers[0].path
        np.testing.assert_allclose(b[first], a[first] + stream, rtol=0, atol=2e-5)

    def test_the_contrast_chain_runs_on_a_residual_model(self):
        model = _model("residual")
        stimuli = {"a": SMOKE_STIMULI["a"], "b": SMOKE_STIMULI["b"]}
        report, _ = evaluate_topography(
            model, stimuli, ("a", "b"), _preprocessor(),
            readout_fwhm=None, min_cluster_size=2,
        )
        assert report["stimuli"]["contrast"] == ["a", "b"]
        per_tap = report["arms"]["raw"]["per_tap"]
        assert list(per_tap) == [tap.path for tap in model.tap_layers]
        assert len(per_tap) == report["num_taps"] == 4
        assert np.isfinite(report["arms"]["raw"]["mean_morans_i"])
        for path, entry in per_tap.items():
            assert entry["num_significant_units"] >= 0, path
            assert np.isfinite(entry["morans_i"]["standard"]), path
            assert entry["morans_i"]["num_units"] == 64, path
