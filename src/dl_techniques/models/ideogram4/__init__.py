"""Ideogram4 text-to-image flow-matching DiT package (Keras 3 port).

A faithful, self-contained Keras 3 reimplementation of the Ideogram4 neural
core: the flow-matching DiT transformer, the Flux2 KL-VAE, the logit-normal +
Euler scheduler/sampler, the velocity loss, an inference pipeline, and training
code. Conditioning is abstracted as a precomputed ``llm_features`` call input
(no Qwen3-VL in Keras). See ``plans/plan_2026-06-12_59a18a10/`` for the design
rationale and the "what doesn't fit / skipped / changed" report.

Current components (this iteration):

- :class:`Ideogram4Config` / :class:`AutoEncoderParams` / ``PRESETS`` /
  :func:`get_ideogram4_config` (:mod:`.config`).
- Sequence/conditioning constants (:mod:`.constants`).
- :data:`LATENT_SHIFT` / :data:`LATENT_SCALE` / :func:`get_latent_norm`
  (:mod:`.latent_norm`).

Per ``models/CLAUDE.md`` this ``__init__.py`` exports nothing — import from
submodules directly (e.g.
``from dl_techniques.models.ideogram4.config import get_ideogram4_config``).
"""
