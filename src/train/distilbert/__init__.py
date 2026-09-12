"""Pattern-3 (MLM) trainer for the DistilBERT model package.

Single-file shape, cloned from ``train.bert.pretrain`` (the bert/fnet/
tree_transformer precedent for this pattern): ``pretrain.py`` holds the
config, the ``MaskedLanguageModel``-wrapped model builder, the training
loop and the CLI entry point together, with no separate
``train_distilbert.py`` — the MLM siblings this package follows never split
that way either. See ``decisions.md`` D-003 (non-harmonization) for why a
shared scaffold across bert/fnet/tree_transformer/distilbert is deliberately
not attempted.
"""
