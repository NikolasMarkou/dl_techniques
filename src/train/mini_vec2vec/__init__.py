"""Bespoke (non-``fit()``) trainer for the mini-vec2vec unsupervised
embedding-space alignment algorithm.

Single-file shape: unlike the zamba2/hnet ``common.py`` + thin entry-point
split used by the CLM trainers in this same plan, ``train_mini_vec2vec.py``
holds the config, the synthetic-data builder and the alignment run together.
The algorithm has exactly one build step (``create_mini_vec2vec_aligner``)
and one entry point (``MiniVec2VecAligner.align``) with no dataset pipeline,
optimizer or callback machinery to factor out -- a second file would hold
imports and nothing else. See ``train_mini_vec2vec.py``'s own module
docstring for the falsification check against
``dl_techniques.models.language.mini_vec2vec.example_alignment``.
"""
