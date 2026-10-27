"""World models — learned latent predictors of how observations evolve.

Three leaf packages, all JEPA-style (prediction happens in embedding space, not pixel space):

- `lewm/` — LeWM, an action-conditioned latent-energy world model with an autoregressive
  predictor and a rollout method
- `video_jepa/` — Video JEPA (joint embedding predictive): future-frame embeddings from
  pixels, EMA target encoder, streaming rolling buffer
- `levjepa/` — LeVJEPA joint-embedding video pretraining ViT encoder

This module is deliberately free of re-exports, and so are the other family packages.
Import from the leaf package, e.g. `dl_techniques.models.world_models.lewm.model`.
"""
