"""``DiTXA``: construction, the variant table, and a ``.keras`` round trip on VALUES.

The per-feature guards in this directory each pin one mechanism (the packing,
the SDE closed forms, the 12-way modulation, the mask, the CFG algebra, the
sampler). This file pins the model as an *artifact*: it constructs, it reports
the shape it promises, it round-trips through ``.keras`` without losing weights,
and the named variants are the ones the package advertises.

**Why the round trip is asserted on VALUES and on COUNTS, never on "it loaded".**
``keras.models.load_model`` returning an object is not evidence. This repo has a
measured case of a model reloading **1 of 65 weights** with the load reported as
successful and no exception anywhere: a mismatch between the tree ``build()``
materializes and the tree ``call()`` runs silently drops the rest, and every
shape assertion downstream still passes because the shapes are right -- only the
numbers are wrong. So each round-trip arm here asserts three separate things:

1. the reloaded model holds the same NUMBER of weight tensors and the same total
   parameter count (a dropped sub-tree changes these, a wrong value does not);
2. the outputs agree at ``atol=1e-6, rtol=0`` with ``training=False`` passed
   EXPLICITLY (the default ``training=None`` resolves differently inside and
   outside a ``fit()`` scope, and the label embedder carries a dropout row);
3. the compared tensor is not the zero tensor.

Point 3 is not paranoia. A freshly built ``DiTXA`` emits the **exact** zero
tensor -- every block's adaLN ``Dense`` is zero in kernel and bias and the final
projection is zero too -- so ``assert_allclose(after, before)`` on a fresh model
compares zeros with zeros and passes for a model that loaded nothing at all.
Every value arm here therefore runs through ``_ditxa_helpers.activate`` first.
"""

import gc

import keras
import numpy as np
import pytest

from dl_techniques.models.vision_language.bit_diffusion.model import (
    DiTXA,
    create_ditxa,
)
from dl_techniques.models.vision_language.bit_diffusion.token_decoder import (
    SharedTokenDecoder,
)

from ._ditxa_helpers import activate, batch, np_

# The tiny variant's measured surface. Pinned as integers so a sub-layer that
# stops being built (or starts being built twice) reddens here rather than
# surfacing as a wrong number somewhere downstream.
TINY_WEIGHT_TENSORS = 54
TINY_PARAMS = 287_952

#: Every shipped variant, with the weight-tensor and parameter counts measured
#: on 2026-09-02. The S/B/L/XL rows sanity-check against the DiT paper: DiT-L/2
#: is 458M and XL/2 675M, and scaling by the 12-way cross-attention block
#: predicts 712M and 1.050B -- within 0.3% of the two rows below.
VARIANT_SURFACE = {
    "tiny": (54, 287_952),
    "S": (214, 50_574_992),
    "B": (214, 201_419_792),
    "L": (406, 710_317_328),
    "XL": (470, 1_047_538_064),
}


def build_tiny(**kwargs):
    """A built, ACTIVATED tiny model plus a matching input dict."""
    keras.utils.set_random_seed(0)
    model = DiTXA.from_variant("tiny", **kwargs)
    inputs = batch(model, batch_size=2)
    model(inputs)
    activate(model, seed=5)
    return model, inputs


def shape_spec(model, batch_size=1):
    """The ``build()``/``compute_output_shape`` shape dict for ``model``."""
    n, c = model.input_size, model.in_channels
    return {
        "x_t": (batch_size, n, n, c),
        "t": (batch_size,),
        "y": (batch_size,),
        "x_cond": (batch_size, n, n, c),
        "direction": (batch_size,),
    }


class TestConstructionAndShape:
    """The model builds, and it reports the shape it actually produces."""

    def test_the_tiny_model_builds_with_the_measured_surface(self):
        model, inputs = build_tiny()
        assert len(model.weights) == TINY_WEIGHT_TENSORS
        assert model.count_params() == TINY_PARAMS
        assert tuple(model(inputs, training=False).shape) == (
            2,
            model.input_size,
            model.input_size,
            model.out_channels,
        )

    def test_compute_output_shape_needs_no_build(self):
        """The unbuilt path: a shape query must not materialize weights."""
        model = DiTXA.from_variant("tiny")
        assert not model.built
        assert model.compute_output_shape(shape_spec(model, batch_size=7)) == (
            7,
            8,
            8,
            4,
        )
        assert not model.built, "compute_output_shape built the model"

    def test_compute_output_shape_agrees_with_the_forward_pass(self):
        """Anti-vacuity for the arm above: the promise must match reality."""
        model, inputs = build_tiny()
        promised = model.compute_output_shape(shape_spec(model, batch_size=2))
        assert tuple(model(inputs, training=False).shape) == promised

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"input_size": 9, "patch_size": 2}, "divisible by patch_size"),
            ({"hidden_size": 65, "num_heads": 4}, "divisible by num_heads"),
            ({"depth": 0}, "must be positive"),
            ({"drop_path_rate": 1.0}, r"must be in \[0, 1\)"),
        ],
    )
    def test_it_rejects_an_impossible_geometry(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            DiTXA(**kwargs)

    @pytest.mark.parametrize("missing", ["x_t", "t", "y", "x_cond", "direction"])
    def test_a_missing_required_input_raises(self, missing):
        """Named, not a ``KeyError`` from three frames down."""
        model, inputs = build_tiny()
        partial = {k: v for k, v in inputs.items() if k != missing}
        with pytest.raises(ValueError, match=missing):
            model(partial)


class TestTheVariantTable:
    """The advertised names, and only those, with their measured sizes."""

    def test_the_keys_are_the_five_the_package_documents(self):
        assert list(DiTXA.MODEL_VARIANTS) == ["tiny", "S", "B", "L", "XL"]

    def test_every_variant_is_internally_consistent(self):
        """Config-level arithmetic, checked before anything is allocated."""
        for name, spec in DiTXA.MODEL_VARIANTS.items():
            assert spec["input_size"] % spec["patch_size"] == 0, name
            assert spec["hidden_size"] % spec["num_heads"] == 0, name
            assert spec["patch_size"] == 2, f"{name}: D-003 ships patch size 2 only"
            assert spec["description"], name

    @pytest.mark.parametrize("variant", list(VARIANT_SURFACE))
    def test_every_variant_actually_constructs(self, variant):
        """All five, built -- not just the one the other guards use.

        ``XL`` allocates ~4.2 GB of float32 weights, so each variant is dropped
        and collected before the next parametrization runs. The counts are
        pinned rather than merely non-zero: a variant whose entry stopped
        reaching the constructor would still build *something*.
        """
        model = DiTXA.from_variant(variant)
        model.build(shape_spec(model))
        tensors, params = VARIANT_SURFACE[variant]
        try:
            assert len(model.weights) == tensors, variant
            assert model.count_params() == params, variant
        finally:
            del model
            gc.collect()

    def test_an_unknown_variant_raises_and_lists_the_real_ones(self):
        with pytest.raises(ValueError, match="Unknown variant 'nope'") as excinfo:
            DiTXA.from_variant("nope")
        for known in DiTXA.MODEL_VARIANTS:
            assert known in str(excinfo.value)

    def test_pretrained_true_raises_before_allocating_anything(self):
        """A caller who asked for trained weights never gets a random model."""
        with pytest.raises(NotImplementedError, match="No pretrained DiTXA weights"):
            DiTXA.from_variant("tiny", pretrained=True)

    def test_the_factory_is_a_thin_delegate(self):
        """Same configuration, modulo the auto-generated instance ``name``.

        ``name`` is excluded on purpose and not by convenience: Keras derives it
        from a process-global per-class counter, so two identically configured
        models are ``di_txa`` and ``di_txa_1``. Every other key is compared.
        """
        viafactory = create_ditxa("tiny").get_config()
        direct = DiTXA.from_variant("tiny").get_config()
        assert set(viafactory) == set(direct)
        differing = {
            k for k in direct if k != "name" and viafactory[k] != direct[k]
        }
        assert not differing, differing
        assert len(direct) > 15, "the config shrank; this comparison went thin"

    def test_a_kwarg_override_beats_the_table(self):
        model = DiTXA.from_variant("tiny", hidden_size=32, num_heads=4)
        assert model.hidden_size == 32
        assert DiTXA.MODEL_VARIANTS["tiny"]["hidden_size"] == 64, (
            "from_variant mutated the class-level table"
        )


class TestTheConfigRoundTrip:
    """``get_config`` carries every constructor argument."""

    def test_get_config_and_from_config_agree(self):
        model = DiTXA.from_variant("tiny", forward_cond_scale=2.5, label_seed=9)
        config = model.get_config()
        rebuilt = DiTXA.from_config(config)
        assert rebuilt.get_config() == config

    def test_the_config_names_every_constructor_argument(self):
        """A knob absent from ``get_config`` is silently reset on every load."""
        import inspect

        signature = inspect.signature(DiTXA.__init__)
        expected = {
            name
            for name, p in signature.parameters.items()
            if name not in ("self", "kwargs")
            and p.kind is not inspect.Parameter.VAR_KEYWORD
        }
        config = DiTXA.from_variant("tiny").get_config()
        missing = sorted(expected - set(config))
        assert not missing, (
            f"these constructor knobs never reach get_config(), so a saved "
            f"model silently reverts them to their defaults on load: {missing}"
        )


class TestTheKerasRoundTripOnValues:
    """Weight count, parameter count, and outputs at ``atol=1e-6, rtol=0``."""

    def test_the_ditxa_round_trip_preserves_counts_and_values(self, tmp_path):
        model, inputs = build_tiny(label_seed=13)
        before = np_(model(inputs, training=False))
        assert np.any(before != 0.0), (
            "the reference output is the exact zero tensor -- a fresh DiTXA "
            "emits exactly that, so this comparison would pass for a model "
            "that loaded no weights at all. activate() did not run."
        )

        path = tmp_path / "ditxa_tiny.keras"
        model.save(path)
        loaded = keras.models.load_model(path)
        after = np_(loaded(inputs, training=False))

        assert len(loaded.weights) == len(model.weights) == TINY_WEIGHT_TENSORS
        assert loaded.count_params() == model.count_params() == TINY_PARAMS
        np.testing.assert_allclose(after, before, atol=1e-6, rtol=0)

    def test_the_round_trip_preserves_every_weight_TENSOR_not_just_the_output(
        self, tmp_path
    ):
        """Per weight, by path. An aggregate output match can hide one tensor.

        The recorded 1-of-65 failure was found this way and not by an output
        comparison: a partially-loaded model whose dropped sub-tree happens to
        sit behind a zero gate still reproduces its output exactly.
        """
        model, _ = build_tiny(label_seed=13)
        path = tmp_path / "ditxa_tiny_weights.keras"
        model.save(path)
        loaded = keras.models.load_model(path)

        before = {w.path: np_(w) for w in model.weights}
        after = {w.path: np_(w) for w in loaded.weights}
        assert sorted(before) == sorted(after)
        mismatched = [
            p
            for p in before
            if not np.allclose(before[p], after[p], atol=0, rtol=0)
        ]
        assert not mismatched, (
            f"{len(mismatched)} of {len(before)} weight tensors changed value "
            f"across the round trip: {mismatched[:8]}"
        )
        # Anti-vacuity: the comparison must be over real numbers, not zeros.
        nonzero = sum(1 for v in before.values() if np.any(v != 0.0))
        assert nonzero >= 40, (
            f"only {nonzero} of {len(before)} saved tensors are non-zero; the "
            "per-tensor comparison is mostly comparing zeros"
        )

    def test_the_shared_token_decoder_round_trip_preserves_counts_and_values(
        self, tmp_path
    ):
        """The package's second ``keras.Model``, on the same three claims.

        ``test_the_token_decoder.py`` already pins the logits at
        ``atol=1e-6, rtol=0``; what it does not assert is the weight and
        parameter COUNT, which is the half of the claim a partial load breaks.
        """
        keras.utils.set_random_seed(0)
        decoder = SharedTokenDecoder(
            vocab_size=37, hidden_dim=24, token_seq_len=8, token_emb_dim=8
        )
        x = keras.ops.convert_to_tensor(
            np.random.default_rng(4).normal(size=(2, 64)).astype("float32")
        )
        before = np_(decoder(x, training=False))
        assert np.any(before != 0.0)

        path = tmp_path / "shared_token_decoder_counts.keras"
        decoder.save(path)
        loaded = keras.models.load_model(path)
        after = np_(loaded(x, training=False))

        assert len(loaded.weights) == len(decoder.weights) == 6
        assert loaded.count_params() == decoder.count_params()
        np.testing.assert_allclose(after, before, atol=1e-6, rtol=0)

    def test_the_value_arm_would_notice_a_perturbed_reload(self, tmp_path):
        """Anti-vacuity: ``atol=1e-6`` must be tight enough to convict.

        Without this the arm above is a claim about ``assert_allclose``'s
        default behaviour, not about the round trip. One weight is nudged by a
        small amount after loading; the comparison must fail.
        """
        model, inputs = build_tiny(label_seed=13)
        before = np_(model(inputs, training=False))
        path = tmp_path / "ditxa_tiny_perturbed.keras"
        model.save(path)
        loaded = keras.models.load_model(path)

        target = next(
            w
            for w in loaded.weights
            if w.trainable and np.any(np_(w) != 0.0)
        )
        target.assign(np_(target) + 0.05)
        after = np_(loaded(inputs, training=False))

        delta = float(np.max(np.abs(after - before)))
        assert delta > 1e-6, (
            f"perturbing '{target.path}' by 0.05 moved the output by only "
            f"{delta:.3e}, which is under the round-trip arm's own atol -- "
            "that arm cannot see a wrong weight"
        )
