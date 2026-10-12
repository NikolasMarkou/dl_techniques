"""Characterisation of the shipped `crystal_frame` conflict metric.

The packing search minimises the SIGNED max inner product, so antipodal
pairs (cos -> -1, the same line in the opposite sense) are never penalised.
This test pins today's measured state at `dimensions=2, width=64, seed=0` --
an exact antipodal pair and no exact duplicate -- so a future change of the
metric to `max|.|` becomes a deliberate, recorded decision rather than
silent drift. Documentation-only: the metric itself is unchanged.
"""

import numpy as np

from dl_techniques.layers.structured_linear.crystal_frames import crystal_frame


def test_the_shipped_frame_holds_an_antipodal_pair_but_no_exact_duplicate():
    frame = crystal_frame(2, 64, seed=0)
    gram = frame.T @ frame
    off = gram[np.triu_indices(64, 1)]
    # No exact +1 duplicate ...
    assert float(off.max()) < 1.0
    # ... but an (almost-)exact antipodal pair on the same line.
    assert float(off.min()) <= -0.999
    # Distinct columns, as documented: the degeneracy is in lines, not columns.
    assert len(np.unique(np.round(frame.T, decimals=6), axis=0)) == 64
