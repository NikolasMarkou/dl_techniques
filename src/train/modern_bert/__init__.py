"""Pattern-3 (MLM) trainer for the ModernBERT model package.

Single-file shape, cloned from ``train.bert.pretrain`` via
``train.distilbert.pretrain`` (the bert/fnet/tree_transformer/distilbert
precedent for this pattern): ``pretrain.py`` holds the config, the
``MaskedLanguageModel``-wrapped model builder, the training loop and the CLI
entry point together, with no separate ``train_modern_bert.py`` -- the MLM
siblings this package follows never split that way either. Unlike
DistilBERT, ModernBERT does NOT share ``BertEmbeddings`` with bert/fnet --
it has its own RoPE-based ``ModernBertEmbeddings`` -- but this does not
change the trainer shape: ``ModernBERT`` still satisfies
``MaskedLanguageModel``'s encoder contract (``hidden_size`` attribute,
``call()`` returning a dict with ``last_hidden_state``) unmodified. See
``decisions.md`` D-003 (non-harmonization) for why a shared scaffold across
bert/fnet/tree_transformer/distilbert/modern_bert is deliberately not
attempted.
"""
