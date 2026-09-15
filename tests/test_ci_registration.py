"""Every test file must be reachable by CI.

CI runs unit tests two ways, and neither covers everything on its own:

* ``unit-tests`` is a hand-maintained matrix invoking ``python tests/<file>``.
  It is the only thing that runs the *script-style* files -- the ones whose
  body is a ``main()`` rather than ``test_`` functions, which pytest collects
  nothing from.
* ``all-unit-tests`` runs ``pytest tests/`` and picks up every pytest-style
  file automatically, including ones nobody remembered to list.

The failure this guards against is a file that falls between them: added,
committed, and run by neither. Before the catch-all existed that was the
default and nothing reported it -- a PR adding tests went green either way, and
eight files holding 137 tests had accumulated unrun (issue #173). The catch-all
closes that for pytest-style files; this module closes it for the rest, and
fails in the PR that opens the gap rather than silently months later.
"""

import ast
import os
import shlex

import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(REPO, "tests")
WORKFLOW = os.path.join(REPO, ".github", "workflows", "main.yml")

# What `pytest tests/` picks up, by pytest's default naming conventions.
PYTEST_FUNC_PREFIX = "test"
PYTEST_CLASS_PREFIX = "Test"


def _workflow():
    with open(WORKFLOW) as f:
        return yaml.safe_load(f)


def _test_files():
    names = sorted(
        n for n in os.listdir(TESTS) if n.startswith("test_") and n.endswith(".py")
    )
    assert names, "no test files found; the test is looking in the wrong place"
    return names


def _matrix_entries():
    """Files named in the hand-maintained `unit-tests` matrix."""
    job = _workflow()["jobs"]["unit-tests"]
    return set(job["strategy"]["matrix"]["test"])


def _collects_under_pytest(name):
    """Whether `pytest tests/<name>` would collect at least one test.

    Static, by pytest's default conventions (top-level ``test_*`` functions and
    ``Test*`` classes), so this test stays hermetic -- it does not import the
    module or shell out to a nested pytest run.
    """
    with open(os.path.join(TESTS, name)) as f:
        tree = ast.parse(f.read(), filename=name)
    for node in tree.body:
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and node.name.startswith(PYTEST_FUNC_PREFIX):
            return True
        if isinstance(node, ast.ClassDef) and node.name.startswith(PYTEST_CLASS_PREFIX):
            return True
    return False


def test_the_catch_all_job_exists_and_runs_the_whole_suite():
    """The other assertions here lean on `pytest tests/` actually running.

    Without this, deleting the catch-all job would leave every pytest-style
    file uncovered while this module still reported everything was fine.
    """
    jobs = _workflow()["jobs"]
    assert "all-unit-tests" in jobs, (
        "the catch-all job is gone; unlisted test files no longer run anywhere "
        "(see issue #173)"
    )
    # Token-wise, not substring: `pytest tests/test_bbstat.py` contains the
    # string "pytest tests/" while covering one file out of twenty.
    runs = [step.get("run", "") for step in jobs["all-unit-tests"].get("steps", [])]
    whole_suite = any(
        "pytest" in run and {"tests", "tests/"} & set(shlex.split(run)) for run in runs
    )
    assert whole_suite, (
        "the catch-all job no longer runs pytest over the whole tests/ "
        "directory, so it does not cover unlisted files; its steps run: "
        f"{[r for r in runs if r]!r}"
    )


@pytest.mark.parametrize("name", _test_files())
def test_every_test_file_is_run_by_something(name):
    """Either pytest collects it, or the matrix names it explicitly."""
    if _collects_under_pytest(name):
        return
    assert name in _matrix_entries(), (
        f"tests/{name} defines no pytest-collectable tests, so `pytest tests/` "
        "skips it, and it is not in the `unit-tests` matrix either -- it would "
        "run nowhere in CI. Add it to the matrix in .github/workflows/main.yml, "
        "or give it top-level test_* functions."
    )


def test_the_matrix_names_only_files_that_exist():
    """A renamed or deleted file leaves an entry that fails the job outright;
    catching it here says which file, rather than a bare `No such file`."""
    missing = sorted(_matrix_entries() - set(_test_files()))
    assert not missing, (
        f"the `unit-tests` matrix names files that do not exist: {missing}. "
        "Rename or drop the entries in .github/workflows/main.yml."
    )


def test_cross_test_imports_are_not_bare():
    """A bare `from test_x import ...` breaks the catch-all at collection time.

    It resolves only when tests/ is itself on sys.path, which `pytest tests/`
    does not arrange, and the resulting ImportError aborts collection for the
    *whole run*, not just that file -- which is how one stray import kept the
    catch-all from being turned on at all (issue #173).

    Both the forms already used in this suite are fine: relative
    (`from .test_x import ...`) and package-qualified
    (`from tests.test_x import ...`). Only the bare absolute form is flagged.
    """
    offenders = []
    for name in _test_files():
        with open(os.path.join(TESTS, name)) as f:
            tree = ast.parse(f.read(), filename=name)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                # level > 0 is a relative import (`from .test_x import ...`),
                # which resolves through the tests package and is fine.
                if node.level == 0 and (node.module or "").startswith("test_"):
                    offenders.append(f"tests/{name}:{node.lineno}: {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("test_"):
                        offenders.append(f"tests/{name}:{node.lineno}: {alias.name}")
    assert not offenders, (
        "bare cross-test imports break `pytest tests/` at collection time; "
        "use `tests.<module>` or a relative `.<module>`:\n  " + "\n  ".join(offenders)
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
