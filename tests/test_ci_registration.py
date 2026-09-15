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
import re
import shlex

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(REPO, "tests")
WORKFLOW = os.path.join(REPO, ".github", "workflows", "main.yml")

# What `pytest tests/` picks up, by pytest's default naming conventions.
PYTEST_FUNC_PREFIX = "test"
PYTEST_CLASS_PREFIX = "Test"


def _job_block(name):
    """The lines of the ``name:`` job, delimited by indentation.

    Read as text rather than with PyYAML: the CI runner has no yaml module, and
    an ImportError here is not a skipped test but a collection error that
    aborts the whole `pytest tests/` run -- the exact failure this file exists
    to prevent, which is how it announced itself the first time this job ran.
    The parse is cross-checked against a real YAML parser in
    test_the_text_parse_agrees_with_pyyaml, wherever one is installed.

    Returns None when there is no such job.
    """
    with open(WORKFLOW) as f:
        lines = f.read().splitlines()
    out, indent = None, None
    for line in lines:
        if out is None:
            if re.match(rf"^(\s*){re.escape(name)}:\s*$", line):
                indent = len(line) - len(line.rstrip("\n").lstrip())
                out = []
            continue
        if line.strip() and (len(line) - len(line.lstrip())) <= indent:
            break
        out.append(line)
    return out


def _test_files():
    names = sorted(
        n for n in os.listdir(TESTS) if n.startswith("test_") and n.endswith(".py")
    )
    assert names, "no test files found; the test is looking in the wrong place"
    return names


def _matrix_entries():
    """Files named in the hand-maintained `unit-tests` matrix."""
    block = _job_block("unit-tests")
    assert block is not None, "no `unit-tests` job in the workflow"
    entries = {
        m.group(1)
        for m in (re.match(r"^\s*-\s*(test_\S+\.py)\s*$", ln) for ln in block)
        if m
    }
    # A parse that silently returned nothing would make every check below pass
    # vacuously; under- and over-reading are both caught by the tests, but an
    # empty read is worth refusing outright.
    assert entries, "parsed no entries from the `unit-tests` matrix"
    return entries


def _pytest_commands(job):
    """Shell words of each command in ``job`` that invokes pytest.

    Scans every line of the block rather than only inline ``run:`` values, so
    a command inside a ``run: |`` block scalar counts too.
    """
    block = _job_block(job)
    if block is None:
        return None
    cmds = []
    for line in block:
        text = line.split("#", 1)[0].strip()
        text = re.sub(r"^-?\s*run:\s*", "", text)
        if "pytest" in text:
            try:
                cmds.append(shlex.split(text))
            except ValueError:
                continue
    return cmds


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
    cmds = _pytest_commands("all-unit-tests")
    assert cmds is not None, (
        "the catch-all job is gone; unlisted test files no longer run anywhere "
        "(see issue #173)"
    )
    # Token-wise, not substring: `pytest tests/test_bbstat.py` contains the
    # string "pytest tests/" while covering one file out of twenty.
    whole_suite = any({"tests", "tests/"} & set(words) for words in cmds)
    assert whole_suite, (
        "the catch-all job no longer runs pytest over the whole tests/ "
        f"directory, so it does not cover unlisted files; it runs: {cmds!r}"
    )


def test_the_text_parse_agrees_with_pyyaml():
    """Wherever PyYAML is installed, hold the hand-rolled parse to it.

    The CI runner has no yaml module, so the parser above cannot use one; this
    keeps it honest anywhere that does, rather than letting a text parse drift
    from what the workflow actually means.
    """
    yaml = pytest.importorskip("yaml", reason="no PyYAML here; parser runs unchecked")
    with open(WORKFLOW) as f:
        jobs = yaml.safe_load(f)["jobs"]

    assert _matrix_entries() == set(
        jobs["unit-tests"]["strategy"]["matrix"]["test"]
    ), "the text parse of the unit-tests matrix disagrees with PyYAML"

    assert ("all-unit-tests" in jobs) == (
        _pytest_commands("all-unit-tests") is not None
    ), "the text parse disagrees with PyYAML on whether the catch-all job exists"

    steps = " ".join(s.get("run", "") for s in jobs["all-unit-tests"].get("steps", []))
    assert ("pytest" in steps) == bool(
        _pytest_commands("all-unit-tests")
    ), "the text parse disagrees with PyYAML on whether the catch-all runs pytest"


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
