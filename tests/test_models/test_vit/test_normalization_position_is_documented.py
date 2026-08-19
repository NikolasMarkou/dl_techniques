"""F-28: ``ViT``'s post-LN default is a DOCUMENTED divergence, not the paper.

``vit/model.py`` defaults ``normalization_position='post'`` while its module
docstring cited Dosovitskiy et al. 2020 and claimed "the defaults reproduce the
published configuration". Published ViT is pre-LN. The sibling
``vit_hmlp/model.py`` defaults to ``'pre'``, so the repository's two ViTs
disagree on the same knob.

The plan's Assumption A2 rules that the DOCSTRING is corrected and the default
is NOT flipped: ``normalization_position`` selects between two different
functions (``TransformerLayer.call``'s two branches) and every ``vit``
checkpoint and training script in the repository was fitted under ``'post'``.
So the assertions here are documentation assertions plus a behaviour PIN on the
default, and ``test_model.py:46``'s ``== "post"`` stays green.

These tests exist so that the divergence cannot become undocumented again: if
someone flips the default, the pins here go red and force the docstring to move
with it.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import inspect

import dl_techniques.models.vit.model as vit_model
import dl_techniques.models.vit_hmlp.model as vit_hmlp_model


def _default(fn, name):
    return inspect.signature(fn).parameters[name].default


class TestTheDefaultIsPinned:
    """A2: the default is deliberately NOT flipped."""

    def test_the_class_default_is_post(self):
        assert _default(vit_model.ViT.__init__, "normalization_position") == "post"

    def test_the_factory_default_matches_the_class_default(self):
        assert (_default(vit_model.create_vit, "normalization_position")
                == _default(vit_model.ViT.__init__, "normalization_position")), (
            "the factory and the class must not drift apart -- a user reading "
            "one and calling the other would silently get the other function"
        )

    def test_the_sibling_vit_hmlp_still_defaults_to_pre(self):
        """The disagreement is real and is what the docstring must disclose."""
        assert _default(vit_hmlp_model.ViTHMLP.__init__,
                        "normalization_position") == "pre"


class TestTheDocstringNoLongerClaimsThePublishedConfiguration:
    """The false claim is the defect F-28 actually reports."""

    def test_the_false_blanket_claim_is_gone(self):
        doc = vit_model.__doc__
        assert "the defaults reproduce the\npublished configuration" not in doc
        assert "defaults reproduce the published configuration" not in doc

    def test_the_module_docstring_names_the_divergence(self):
        doc = vit_model.__doc__
        assert "`normalization_position` defaults to `\"post\"`" in doc
        assert "published ViT is pre-LN" in doc

    def test_it_tells_the_reader_what_to_pass_instead(self):
        assert "normalization_position='pre'" in vit_model.__doc__

    def test_it_names_the_sibling_that_disagrees(self):
        assert "vit_hmlp" in vit_model.__doc__

    def test_the_parameter_entry_repeats_the_warning_where_it_is_read(self):
        """The module docstring is not where a user looks up one argument."""
        doc = vit_model.ViT.__doc__
        assert "NOT the published ViT configuration" in doc

    def test_the_factory_docstring_carries_it_too(self):
        assert "NOT the published ViT configuration" in vit_model.create_vit.__doc__


class TestTheDivergenceIsRealAndNotCosmetic:
    """The two settings build different functions, which is why A2 matters."""

    def test_the_two_positions_produce_different_outputs(self):
        import numpy as np
        import keras

        common = dict(input_shape=(32, 32, 3), num_classes=4, scale="tiny",
                      patch_size=16, include_top=True)
        keras.utils.set_random_seed(0)
        post = vit_model.ViT(**common, normalization_position="post")
        keras.utils.set_random_seed(0)
        pre = vit_model.ViT(**common, normalization_position="pre")
        x = np.random.RandomState(0).randn(2, 32, 32, 3).astype("float32")
        a = keras.ops.convert_to_numpy(post(x, training=False))
        b = keras.ops.convert_to_numpy(pre(x, training=False))
        assert float(np.max(np.abs(a - b))) > 1e-4, (
            "if these agree the knob is inert and the whole finding is moot"
        )
