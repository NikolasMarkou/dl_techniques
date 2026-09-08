"""The mirrored RED proof for the shared instrument ``tests/complex_dead_knob_ast.py``.

`src/dl_techniques/CLAUDE.md` § Testing: "A **shared instrument** carries **no
``test_`` prefix**, so pytest does not collect it; each has a mirrored
``test_<name>.py`` RED proof". `complex_dead_knob_ast.py` shipped without one at
`91713b820`, and the gap was not cosmetic. MEASURED at `0dc3bc1ed`: with the
package `__path__` pointed at an empty directory,
`self_attribute_site_counts("epsilon")` returned `{}` and
`expected_dead_knob_sites()` returned `{}` — so BOTH collected dead-knob guards
evaluated `{} == {}` and passed. An enumeration that saw nothing certified that
nothing was wrong.

This module pins the two properties the instrument's callers actually depend on,
and neither is asserted anywhere else:

1. **It fails CLOSED.** A truncated or empty package enumeration raises
   `BlindPackageScanError` out of every entry point, rather than returning an
   empty map that compares equal to an empty expectation.
2. **It has POWER.** Handed a module that really does read `self.<knob>`, the AST
   scan reports the read — so the `0 in every leaf` half of the invariant is a
   measurement, not a default.

Both are exercised WITHOUT touching any tracked file: (1) by pointing the
package's `__path__` at a `tmp_path`, (2) by parsing a synthetic module written
into `tmp_path`.
"""

import sys
import textwrap

import pytest

from tests import complex_dead_knob_ast as instrument


# ---------------------------------------------------------------------
# 1. FAIL CLOSED — an empty enumeration must raise, never certify
# ---------------------------------------------------------------------

@pytest.fixture
def blinded_package(tmp_path, monkeypatch):
    """Point the package's ``__path__`` at an empty directory.

    This is the exact failure mode the instrument exists to prevent: a scan that
    reaches no module at all. Nothing on disk is modified — `monkeypatch`
    restores `__path__` at teardown.
    """
    monkeypatch.setattr(
        instrument._complex_package, "__path__", [str(tmp_path)], raising=True
    )
    return tmp_path


def test_the_guard_fails_closed_when_the_package_enumeration_is_empty(blinded_package):
    """`complex_package_modules()` must RAISE on an empty scan, not return `{}`."""
    with pytest.raises(instrument.BlindPackageScanError) as excinfo:
        instrument.complex_package_modules()

    message = str(excinfo.value)
    assert "enumerated only []" in message, (
        "the fail-closed error must name what it DID see, so a partial scan is "
        f"diagnosable from the failure alone; got: {message}"
    )
    assert "base" in message, (
        "the fail-closed error must name the missing modules; got: " + message
    )


@pytest.mark.parametrize("knob", instrument.DEAD_KNOBS)
def test_the_guard_fails_closed_in_every_entry_point_the_callers_use(
    knob, blinded_package
):
    """The two functions the collected guards compare must BOTH raise.

    This is the assertion that would have caught the vacuity directly: before the
    fail-closed change, `self_attribute_site_counts(knob) == expected_dead_knob_sites()`
    was `{} == {}` under exactly this fixture — True, for both knobs.
    """
    with pytest.raises(instrument.BlindPackageScanError):
        instrument.self_attribute_site_counts(knob)

    with pytest.raises(instrument.BlindPackageScanError):
        instrument.expected_dead_knob_sites()


def test_the_guard_fails_closed_on_a_PARTIAL_enumeration_too(tmp_path, monkeypatch):
    """A scan that finds SOME modules but not all is equally blind.

    An empty scan is the loud case. The quiet one is a scan that reaches `base.py`
    — where both knobs legitimately live, so the `2` half of the invariant still
    reads correctly — and misses the six leaves, where the `0` half then reads
    vacuously. That is the precise shape MEASURED at `d148888a7`.
    """
    (tmp_path / "base.py").write_text("class ComplexLayer:\n    pass\n")
    monkeypatch.setattr(
        instrument._complex_package, "__path__", [str(tmp_path)], raising=True
    )

    with pytest.raises(instrument.BlindPackageScanError) as excinfo:
        instrument.complex_package_modules()

    message = str(excinfo.value)
    assert "complex_conv2d" in message and "complex_dense" in message, (
        "a partial scan must name the leaves it missed; got: " + message
    )


def test_the_guard_is_not_vacuously_fail_closed_on_the_real_package():
    """ANTI-VACUITY for the three tests above: the REAL package must NOT raise.

    A `complex_package_modules()` that raised unconditionally would satisfy every
    fail-closed assertion above while destroying the instrument. This cell is the
    only thing separating "fails closed" from "always fails".
    """
    modules = instrument.complex_package_modules()

    assert instrument.KNOWN_MODULE_FLOOR <= set(modules), (
        f"the real package enumerated {sorted(modules)}, below the known floor"
    )
    assert instrument.expected_dead_knob_sites()[instrument.BASE_MODULE_NAME] == (
        instrument.EXPECTED_BASE_SITES
    )


# ---------------------------------------------------------------------
# 2. POWER — the scan must SEE a read that is really there
# ---------------------------------------------------------------------

_INJECTED_MODULE = textwrap.dedent(
    '''
    """A synthetic leaf that DOES read the knob — the instrument must see it."""


    class Injected:
        def __init__(self):
            self.kernel_initializer = None      # site 1 (Store)

        def call(self, x):
            # A comment that merely NAMES self.kernel_initializer must not count.
            return self.kernel_initializer(x)   # site 2 (Load)

        def unrelated(self, other):
            return other.kernel_initializer     # NOT self. -- must not count
    '''
)


@pytest.fixture
def injected_module(tmp_path, monkeypatch):
    """Import a synthetic module carrying two genuine ``self.<knob>`` reads."""
    name = "_injected_dead_knob_probe"
    (tmp_path / f"{name}.py").write_text(_INJECTED_MODULE)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(name, None)
    try:
        yield __import__(name)
    finally:
        sys.modules.pop(name, None)


def test_the_probe_detects_an_injected_read_and_ignores_its_lookalikes(injected_module):
    """`self_attribute_sites` counts exactly the two real `self.<knob>` nodes.

    The count is `2`, not `4`: the mention inside the comment is invisible to an
    AST walk (a `source.count("self.kernel_initializer")` would read `3` — the
    reason this instrument parses rather than greps), and `other.kernel_initializer`
    is an attribute of something that is not `self`.
    """
    sites = instrument.self_attribute_sites(injected_module, "kernel_initializer")

    assert len(sites) == 2, (
        "the AST scan read "
        f"{len(sites)} `self.kernel_initializer` sites in a module that has "
        "exactly 2 (one Store, one Load) plus one comment mention and one "
        "`other.` lookalike — the instrument is miscounting, so both dead-knob "
        "guards' numbers are unreliable"
    )
    assert sorted(type(site.ctx).__name__ for site in sites) == ["Load", "Store"], (
        "the two sites should be one Store (the assignment) and one Load (the "
        "read); the instrument is not distinguishing them"
    )


def test_the_probe_reports_zero_for_a_knob_that_is_genuinely_absent(injected_module):
    """ANTI-VACUITY for the cell above: `0` must still be reachable.

    A scan hard-wired to return `2` would pass the power test. `epsilon` does not
    appear in the synthetic module at all, so this must read `0` — which is also
    the exact value the leaf half of the real invariant asserts.
    """
    assert instrument.self_attribute_sites(injected_module, "epsilon") == []


def test_the_probe_liveness_arm_passes_on_the_real_package():
    """`assert_scan_reaches_the_leaves()` is green against the shipped package.

    Its RED behaviour is covered by the fail-closed cells above (it calls
    `self_attribute_site_counts`, which raises under a blinded `__path__`); this
    cell pins that it is not RED at HEAD for some unrelated reason.
    """
    instrument.assert_scan_reaches_the_leaves()
