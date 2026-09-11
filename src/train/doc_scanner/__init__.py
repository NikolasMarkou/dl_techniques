"""Training pipelines for the DocScanner port (localization + rectification).

Two entry points, because the paper trains the two modules INDEPENDENTLY
(arXiv:2110.14968v2 §4.3): ``train_doc_scanner_segmenter.py`` and
``train_doc_scanner_rectifier.py``. Everything they share -- the config, the
flags, the data gates, the ``tf.data`` pipeline, the losses, the optimizers and
``train()`` -- lives in :mod:`train.doc_scanner.common`.
"""
