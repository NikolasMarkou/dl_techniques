"""DINO self-supervised pretraining pipeline.

Package marker. Import the trainer explicitly::

    from train.dino.train_dino import TrainingConfig, train_dino

The runnable entry point is ``python -m train.dino.train_dino`` (never
``train.py`` -- a module by that name shadows the ``train`` package and breaks
``from train.common import ...``).
"""
