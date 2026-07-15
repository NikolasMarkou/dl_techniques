"""Graph Energy Transformer training package.

Thin Pattern-1 trainers for the graph-domain ``EnergyTransformer`` heads, sharing a small
``common.py`` (the optimizer block, re-exported from the image trainer — it is
dataset/model-agnostic):

* ``train_anomaly.py``        — variant B, supervised node-anomaly detection on the CARE-GNN
  fraud benchmarks (Amazon / YelpChi) with a synthetic network-free fallback. Class imbalance
  is handled through stock ``fit`` (``class_weight`` / ``sample_weight``), NO custom
  ``train_step``.
* ``train_classification.py`` — variant C-lite, graph classification on TUDataset (added in a
  later plan step).

Run with ``python -m train.graph_energy_transformer.train_anomaly --help``.
"""
