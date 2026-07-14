"""Energy Transformer training package.

Two Pattern-1 trainers sharing a small ``common.py`` (dataset builder, optimizer block,
energy-trace callback):

* ``train_masked_completion.py`` — self-supervised masked-image completion (the paper's §3 model).
* ``train_classification.py``    — supervised classifier, warm-startable from an MIM checkpoint.

Run with ``python -m train.energy_transformer.train_masked_completion --help``.
"""
