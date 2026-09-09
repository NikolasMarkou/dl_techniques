"""Training and data-staging entry points for the H-Net byte-level language model port.

Modules:
    * :mod:`~train.hnet.prepare_hnet_data` — the corpus manifest, the
      already-staged Wikipedia slot, and the GATED FineWeb-Edu ``sample-10BT``
      slot under ``/media/arxwn/data0_4tb/datasets/fineweb_edu/``. It downloads
      nothing unless ``--download`` is passed explicitly.
"""
