"""BEiT training package — three stages, three trainers, one small ``common.py``.

BEiT (arXiv:2106.08254) pre-trains a ViT encoder by predicting the DISCRETE visual-token
id of each masked patch. That target has to come from somewhere, so this package is a
three-stage pipeline rather than the usual pretrain/finetune pair:

* ``train_tokenizer.py``      — stage 0: train the discrete visual tokenizer that produces
  the MIM targets. **Deviation X-1**: it is a VQ-VAE (``VQVAERotationTrick``), not BEiT
  v1's Gumbel-softmax DALL-E dVAE, so comparison to published BEiT numbers is invalid by
  construction. See ``models/beit/README.md`` section 15.
* ``train_mim.py``            — stage 1: self-supervised masked image modeling against the
  FROZEN stage-0 tokenizer's code ids.
* ``train_classification.py`` — stage 2: supervised classifier, warm-started from a
  stage-1 checkpoint via ``load_weights_from_checkpoint(..., skip_prefixes=("decoder_",
  "head_"))`` with the transfer ASSERTED, never merely logged.

``common.py`` holds only what all three genuinely share: the raw-image ``tf.data``
pipeline, the optimizer block, and the frozen-tokenizer loader. There is deliberately NO
shared ``TrainingConfig`` and NO shared ``train()`` orchestrator — each trainer owns its
own dataclass, ``parse_arguments()`` and ``config_from_args()`` so that a CLI flag which
never reaches the config is a LOCAL, greppable, testable defect rather than an inherited
one.

Run with, e.g.::

    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 python -m train.beit.train_mim --help
"""
