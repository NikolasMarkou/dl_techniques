"""Training pipelines for language models.

A **family directory**, not a namespace. ``src/train/`` is otherwise flat --
every other trainer sits directly under it -- and this level exists because the
user asked for the ColBERT trainers at ``src/train/language/colbert/``
specifically. It is deliberate, not an unfinished migration: do not "fix" it by
flattening ``language/colbert/`` to ``colbert/``, and do not "finish" it by
moving the other 47 trainers underneath family directories.

Like the ``src/dl_techniques/models/`` family directories it mirrors, this
module holds a docstring and nothing else. There is no re-export here; always
import from the leaf package::

    from train.language.colbert.common import TrainingConfig  # correct
    from train.language import colbert                        # works, but the
                                                              # leaf import is
                                                              # the convention

Leaf packages:

- ``colbert/`` -- ColBERT v1 (pairwise softmax) and v2 (KL distillation)
  late-interaction retrieval training.
"""
