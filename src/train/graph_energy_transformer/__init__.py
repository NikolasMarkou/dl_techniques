"""Graph Energy Transformer training package.

Thin Pattern-1 trainers for the graph-domain ``EnergyTransformer`` heads, sharing a small
``common.py`` (the optimizer block, re-exported from the image trainer — it is
dataset/model-agnostic):

* ``train_anomaly.py``        — variant B, supervised node-anomaly detection on the CARE-GNN
  fraud benchmarks (Amazon / YelpChi) with a synthetic network-free fallback. Class imbalance
  is handled through stock ``fit`` (``class_weight`` / ``sample_weight``), NO custom
  ``train_step``.
* ``train_classification.py`` — variant C-lite, whole-graph classification on TUDataset
  (MUTAG / PROTEINS / ...). CLS token + Laplacian-PE + ``S=4`` stacked blocks + saddle-escape
  noise; label-smoothed ``CategoricalCrossentropy`` fed one-hot targets through stock ``fit``,
  NO custom ``train_step``.

Both trainers expose the same importable, testable ``parse_arguments`` / ``config_from_args``
pair (fail-closed CLI-to-config wiring, guarded by
``tests/test_train/test_graph_energy_transformer/test_cli_wiring.py``).

Run with ``python -m train.graph_energy_transformer.train_anomaly --help`` or
``python -m train.graph_energy_transformer.train_classification --help``.
"""
