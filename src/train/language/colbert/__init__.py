"""ColBERT v1 / v2 late-interaction retrieval training.

Two entry points, one shared scaffold:

- ``train_colbert_v1.py`` -- pairwise/listwise softmax cross-entropy
  (:class:`~dl_techniques.losses.ColBERTPairwiseSoftmaxLoss`).
- ``train_colbert_v2.py`` -- cross-encoder KL distillation
  (:class:`~dl_techniques.losses.ColBERTDistillationLoss`).
- ``common.py`` -- the shared ``TrainingConfig``, the synthetic-triples
  ``tf.data`` pipeline and ``build_model``.

.. warning::

   **Everything these scripts produce is a wiring result, never a
   retrieval-quality claim.** No pretrained weights exist for ColBERT or for the
   BERT backbone in this repository, and there is no MS MARCO or any other IR
   dataset here either -- the data is synthetic. See either trainer's module
   docstring for the full statement.

Run them as modules, never as files (the package layout matters for imports)::

    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 .venv/bin/python \\
        -m train.language.colbert.train_colbert_v1 --smoke
"""
