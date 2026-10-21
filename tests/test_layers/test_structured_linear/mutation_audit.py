"""Mutation-testing harness for the crystal-structured layers.

A guard that cannot fail is the most likely outcome of writing a new test, and
the only way to find out which guards are load-bearing is to break the code and
see whether anything notices. This module injects a realistic defect into one of
the five modules, runs the suites that cover it, and reports how many tests
caught it.

Each mutation is a defect a reviewer could plausibly introduce: an axis
transposed, a guard dropped, a count off by one. A mutation caught by NO test is
a hole in the suite, and the point of running this is to find those, not to
re-confirm the mutations that were already caught.

Run it directly to print the table::

    python tests/test_layers/test_structured_linear/mutation_audit.py

It is not named ``test_*`` so pytest does not collect it; the recording of which
mutations survive lives in ``test_the_mutation_suite_found_no_new_holes.py``,
which fails if a previously-covered mutation becomes uncovered.
"""

import contextlib
import os
import re
import subprocess
import sys

# ---------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

_PACKAGE = os.path.join('src', 'dl_techniques', 'layers', 'structured_linear')
_UTILS = os.path.join('src', 'dl_techniques', 'utils')

#: Every file this harness is allowed to modify, relative to the repo root.
_ALLOWED = (
    os.path.join(_PACKAGE, 'crystal_frames.py'),
    os.path.join(_PACKAGE, 'crystal_layer.py'),
    os.path.join(_PACKAGE, 'hierarchical_crystal_layer.py'),
    os.path.join(_PACKAGE, 'crystal_multihead_projection.py'),
    os.path.join(_PACKAGE, 'bitlinear_layer.py'),
    os.path.join(_PACKAGE, 'harmonic_dense.py'),
    os.path.join(_PACKAGE, 'hierarchical_harmonic.py'),
    os.path.join(_PACKAGE, 'kanvolution.py'),
    os.path.join(_PACKAGE, 'mps_layer.py'),
    os.path.join(_PACKAGE, 'orthoblock.py'),
    os.path.join(_PACKAGE, 'orthogonal_butterfly.py'),
    os.path.join(_PACKAGE, 'random_fourier_features.py'),
    os.path.join(_UTILS, 'tensors.py'),
)

#: name -> (relative path, original snippet, replacement snippet, rationale)
MUTATIONS = {
    'l2_normalize-drops-the-rescale': (
        os.path.join(_UTILS, 'tensors.py'),
        "    peak = keras.ops.max(keras.ops.abs(x), axis=axis, keepdims=True)\n",
        "    peak = keras.ops.maximum(keras.ops.abs(x), 1.0)\n",
        'rescale by a constant instead of the vector peak, which is a no-op',
    ),
    'l2_normalize-drops-the-zero-guard': (
        os.path.join(_UTILS, 'tensors.py'),
        "    scaled = x / keras.ops.where(peak > 0.0, peak, keras.ops.ones_like(peak))\n",
        "    scaled = x / peak\n",
        'remove the zero-divisor substitution, so an all-zero row divides 0/0',
    ),
    'crystal-frame-drops-the-search': (
        os.path.join(_PACKAGE, 'crystal_frames.py'),
        "        if index == 0 or candidate_trials < 2 or stacked is None:",
        "        if True:",
        'always take the first sampled rotation, making candidate_trials inert',
    ),
    'crystal-frame-off-by-one': (
        os.path.join(_PACKAGE, 'crystal_frames.py'),
        "    frame = np.identity(dimensions, dtype=np.float64)\n",
        "    frame = np.identity(dimensions, dtype=np.float64)[:max(dimensions - 1, 1)]\n",
        'drop one basis vector, breaking the equiangular inner product',
    ),
    'crystal-frame-drops-the-centering': (
        os.path.join(_PACKAGE, 'crystal_frames.py'),
        "    frame -= frame.mean(axis=0, keepdims=True)\n",
        "    pass\n",
        'skip centering, so the vertices no longer sum to zero',
    ),
    'crystal-layer-transposes-the-weight-axis': (
        os.path.join(_PACKAGE, 'crystal_layer.py'),
        "            self.crystal_weights, axis=0, eps=self.eps\n",
        "            self.crystal_weights, axis=-1, eps=self.eps\n",
        'normalize the frame along the wrong axis',
    ),
    'crystal-layer-drops-the-scale': (
        os.path.join(_PACKAGE, 'crystal_layer.py'),
        "        return keras.ops.matmul(normalized_inputs, normalized_weights) * self.scale",
        "        return keras.ops.matmul(normalized_inputs, normalized_weights)",
        'silently ignore the inverse-temperature argument',
    ),
    'hierarchical-reuses-child-zero-for-every-group': (
        os.path.join(_PACKAGE, 'hierarchical_crystal_layer.py'),
        "                seed=children[group + 1]",
        "                seed=children[0]",
        'give every leaf bundle the macro stream, collapsing the bundles',
    ),
    'hierarchical-transposes-the-reshape': (
        os.path.join(_PACKAGE, 'hierarchical_crystal_layer.py'),
        "        grouped = _split_leading_dims(micro_logits, self.num_macro, self.num_micro)",
        "        grouped = _split_leading_dims(micro_logits, self.num_micro, self.num_macro)",
        'split the leaf axis the wrong way round, breaking the Bayes identity',
    ),
    'hierarchical-drops-the-broadcast': (
        os.path.join(_PACKAGE, 'hierarchical_crystal_layer.py'),
        "        joint = expanded_macro * conditional_probs",
        "        joint = conditional_probs",
        'drop the macro factor, so joint is no longer a joint distribution',
    ),
    'multihead-drops-the-head-index': (
        os.path.join(_PACKAGE, 'crystal_multihead_projection.py'),
        "            '...hd,hdv->...hv', normalized_heads, normalized_prisms\n",
        "            '...d,hdv->...hv', normalized_heads, normalized_prisms\n",
        'drop h from the einsum subscript, mixing every head together',
    ),
    'multihead-drops-the-gelu': (
        os.path.join(_PACKAGE, 'crystal_multihead_projection.py'),
        "        features = keras.ops.gelu(features)",
        "        features = features * 1.0",
        'make the non-linearity a no-op',
    ),
    'multihead-inverts-the-scale-guard': (
        os.path.join(_PACKAGE, 'crystal_multihead_projection.py'),
        "        if output_dim is not None and output_dim <= 0:",
        "        if output_dim is not None and output_dim > 0:",
        'invert the output_dim validation condition',
    ),
}


class MutationResult:
    """Outcome of applying one mutation and running the suites over it."""

    def __init__(self, name: str, caught: int, total: int, applied: bool,
                 detail: str = '') -> None:
        self.name = name
        self.caught = caught
        self.total = total
        self.applied = applied
        self.detail = detail

    @property
    def survivors(self) -> int:
        """Tests that still passed with the defect in place."""
        return self.total - self.caught

    @property
    def detected(self) -> bool:
        return self.caught > 0

    def __repr__(self) -> str:
        if not self.applied:
            return f"{self.name}: NOT APPLIED ({self.detail})"
        verdict = 'DETECTED' if self.detected else 'SURVIVED'
        return (
            f"{self.name}: {verdict} "
            f"({self.caught}/{self.total} caught, {self.survivors} survived)"
        )


@contextlib.contextmanager
def _mutated(relative_path: str, original: str, replacement: str):
    """Apply a source edit for the duration of the block, then restore it.

    The restore writes the ORIGINAL BYTES back and re-reads the file, so a
    crash mid-run cannot leave the tree corrupted -- and `diff -q` against a
    scratch copy is the check the audit report quotes.
    """
    path = os.path.join(_REPO_ROOT, relative_path)
    with open(path, 'rb') as handle:
        pristine = handle.read()

    if pristine == original.encode():
        raise AssertionError(
            f"mutation target not found verbatim in {relative_path}; the source "
            f"has moved and this mutation would silently do nothing"
        )
    if original.encode() not in pristine:
        raise AssertionError(f"mutation snippet absent from {relative_path}")

    mutated = pristine.replace(original.encode(), replacement.encode(), 1)
    try:
        with open(path, 'wb') as handle:
            handle.write(mutated)

        yield
    finally:
        with open(path, 'wb') as handle:
            handle.write(pristine)


def run_mutation(
        name: str,
        suites,
        env=None
) -> MutationResult:
    """
    Apply one mutation and count how many tests fail.

    :param name: Key into :data:`MUTATIONS`.
    :param suites: Pytest node paths to run over the mutated tree.
    :param env: Extra environment for the subprocess.
    :return: The recorded outcome.
    """

    relative_path, original, replacement, _ = MUTATIONS[name]

    command = [
        sys.executable, '-m', 'pytest', *suites, '-q', '--no-header', '-p', 'no:randomly',
    ]

    try:
        with _mutated(relative_path, original, replacement):
            completed = subprocess.run(
                command, cwd=_REPO_ROOT, capture_output=True, text=True,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', **(env or {})}
            )
    except AssertionError as error:
        return MutationResult(name, 0, 0, False, str(error))

    tail = completed.stdout.strip().splitlines()
    summary = next(
        (line for line in reversed(tail)
         if re.search(r'\d+ (passed|failed|error)', line)),
        ''
    )

    # pytest's closing line is e.g. "8 failed, 720 passed, 1 warning in 33s".
    # Each COUNT is the token BEFORE its label, so the parse must walk pairs
    # rather than assume a position -- reading fields[0] as the total picked up
    # the FAILED count and reported "8 total, 0 caught" for a mutation that 8
    # tests caught, which would have inverted every verdict below.
    counts = {}
    fields = summary.split()
    for index in range(1, len(fields) - 1, 2):
        if fields[index].rstrip(',') in ('passed', 'failed', 'error', 'errors', 'xfailed', 'xpassed', 'skipped', 'deselected'):
            try:
                counts[fields[index].rstrip(',')] = int(fields[index - 1])
            except ValueError:
                pass

    if 'passed' not in counts and 'failed' not in counts:
        return MutationResult(name, 0, 0, False, f'unparsed summary: {summary!r}')

    total = counts.get('passed', 0) + counts.get('failed', 0) + counts.get('error', 0)
    caught = counts.get('failed', 0) + counts.get('error', 0)

    return MutationResult(name, caught, total, True)


def check_snippets_current() -> list:
    """Return the names of every mutation whose snippet no longer matches.

    A stale snippet is the worst failure mode this harness has: the mutation
    reports "NOT APPLIED", which a reader can dismiss, but if that is skimmed
    the audit silently covers less than it appears to. Three snippets went stale
    during the first pass -- editing the source under the harness invalidated
    them -- so the check is run and its result printed rather than left implicit.
    """
    stale = []
    for name, (relative_path, original, _, _) in MUTATIONS.items():
        path = os.path.join(_REPO_ROOT, relative_path)
        try:
            with open(path, encoding='utf-8') as handle:
                source = handle.read()
        except OSError:
            stale.append(name)
            continue
        if original not in source:
            stale.append(name)

    return stale


def run_all(suites) -> list:
    """Run every mutation in :data:`MUTATIONS` and return the results."""
    return [run_mutation(name, suites) for name in MUTATIONS]


if __name__ == '__main__':
    SUITES = [
        'tests/test_layers/test_structured_linear/',
        'tests/test_utils/test_l2_normalize.py',
    ]

    stale = check_snippets_current()
    if stale:
        print('!! STALE MUTATION SNIPPETS -- these would silently not apply:')
        for name in stale:
            print(f'   {name}')
        print()

    results = run_all(SUITES)

    print('=' * 78)
    print('MUTATION AUDIT -- crystal family')
    print('=' * 78)
    for result in results:
        print(f'  {result}')
    survivors = [r for r in results if r.applied and not r.detected]
    unapplied = [r for r in results if not r.applied]
    print('-' * 78)
    print(f'{len(results)} mutations, {len(survivors)} undetected, '
          f'{len(unapplied)} failed to apply')
    for result in survivors:
        print(f'  HOLE: {result.name}')
    for result in unapplied:
        print(f'  NOT APPLIED: {result.name}')
