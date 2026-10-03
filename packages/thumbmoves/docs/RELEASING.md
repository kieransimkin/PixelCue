# Releasing ThumbMoves

ThumbMoves is independently packaged inside the PixelCue repository. Its CI is path-scoped so unrelated PixelCue changes do not start the package workflow, while changes to PixelCue's adapter and dependency contract do.

## Release contract

1. Update the version in both `pyproject.toml` and `src/thumbmoves/__init__.py`.
2. Run the standalone tests, PixelCue integration-contract tests, distribution validation and wheel smoke test.
3. Commit the complete release source.
4. Create and push an annotated tag named `thumbmoves-vX.Y.Z` at that exact commit.
5. The release workflow checks out the tag, repeats CI on the tagged commit, builds one universal wheel and one source distribution, validates their embedded identity and contents, and writes `SHA256SUMS.txt`.
6. A manual run defaults to the TestPyPI environment and publishes only to TestPyPI. This is the rehearsal route.
7. A pushed version tag, or a manual run explicitly set to `production`, publishes the exact tested artifacts to PyPI and creates the GitHub release. Both results are reported independently; a successful upload to one destination is not proof that the other succeeded.
8. The workflow creates the GitHub release as a draft, uploads the wheel, source distribution and checksums, then publishes it. Draft-first publication is compatible with GitHub immutable releases.
9. Download the GitHub assets independently and compare their hashes with `SHA256SUMS.txt`; also inspect the PyPI project/version before treating the release as fully verified.

## One-time package-repository setup

Create GitHub environments named `testpypi` and `pypi`. Require approval for production through the `pypi` environment. In each package index, register a pending Trusted Publisher with:

- owner: `kieransimkin`;
- repository: `PixelCue`;
- workflow: `thumbmoves-release.yml`;
- environment: `testpypi` on TestPyPI and `pypi` on PyPI;
- project name: `thumbmoves`.

No long-lived PyPI token is required. The publishing jobs receive only `id-token: write`, and the distributions are built in the separate CI job before either publishing job runs. GitHub Packages does not provide a Python/PyPI registry, so the supported package repositories here are PyPI and TestPyPI; GitHub Releases remains the independently downloadable asset channel.

The public JSON endpoints for `thumbmoves` returned HTTP 404 on both PyPI and TestPyPI on 3 October 2026, so no existing project was visible at verification time. This is availability evidence, not a reservation: register the pending publishers before pushing the first production tag.

## Tag examples

- Stable: `thumbmoves-v0.1.0`
- Prerelease: `thumbmoves-v0.2.0rc1`

Plain `v0.1.0` tags are reserved for repository-level PixelCue releases and are rejected by the ThumbMoves release validator.

## Potential problems

### Standalone tests cannot import ThumbMoves from the monorepo root

- **Symptom (3 October 2026):** pytest collection failed with `ModuleNotFoundError: No module named 'thumbmoves'` for the package tests when they were invoked from the PixelCue environment.
- **Cause:** ThumbMoves uses a `src` layout and had not been installed into that interpreter. The current pytest guidance for a `src` layout requires installing the project or deliberately adding its source directory to the import path.
- **Corrective action:** install the package as an editable dependency with `python -m pip install -e "./packages/thumbmoves[dev]"` before source-tree tests. The CI workflow performs this explicitly. Distribution smoke testing instead installs the built wheel into a fresh environment, preventing the checkout from masking packaging omissions.
- **Verification:** run the package and PixelCue integration-contract suites after the editable install, then run the wheel smoke test outside the checkout.
- **Limit:** setting `PYTHONPATH` can make source tests import, but it does not verify that the built wheel contains the intended files, so it is not a substitute for the release smoke test.

### Nested workflows are ignored by GitHub

- **Symptom:** a workflow stored under `packages/thumbmoves/.github/workflows` does not appear in the PixelCue repository's Actions tab.
- **Cause:** GitHub only loads workflows from `.github/workflows` at the repository root.
- **Corrective action:** keep the active `thumbmoves-ci.yml` and `thumbmoves-release.yml` files in PixelCue's root workflow directory while ThumbMoves remains embedded.
- **Verification:** both workflows appear by name in the repository Actions view after the files reach the default branch.
- **Limit:** if ThumbMoves moves to its own repository, move the workflows to that repository root and reconsider whether the `thumbmoves-` tag prefix is still needed.

### The uv-managed PixelCue environment has no pip module

- **Symptom (3 October 2026):** `Z:\My Songs\Tools\PixelCue\.venv\Scripts\python.exe -m pip` failed with `No module named pip`.
- **Cause:** uv-managed environments do not need to contain pip; uv provides its own pip-compatible interface.
- **Corrective action:** use `uv pip install --python "Z:\My Songs\Tools\PixelCue\.venv\Scripts\python.exe" ...` when intentionally installing into that existing local environment.
- **Verification:** uv built and installed the local ThumbMoves 0.1.0 editable package into the selected interpreter.
- **Limit:** GitHub's `actions/setup-python` runners do include pip, so the CI workflow uses the standard `python -m pip` commands recommended by Python packaging guidance.

### Restricted local uv cache or network prevents the editable build

- **Symptom (3 October 2026):** uv first reported `Failed to initialize cache` with `Access is denied` for the default user cache, then could not fetch the isolated build requirement `wheel` through the restricted network.
- **Cause:** the local execution sandbox could not write the default uv cache or reach PyPI. This did not indicate a ThumbMoves dependency or metadata error.
- **Corrective action:** point `UV_CACHE_DIR` to the writable PixelCue cache and, when authorised, run the narrowly scoped install with network access so uv can resolve `setuptools>=68` and `wheel`.
- **Verification:** the editable build completed and installed `thumbmoves==0.1.0` from the local source tree.
- **Limit:** do not disable TLS checks or substitute unverified indexes. GitHub-hosted CI should use the normal public PyPI route.

### Monorepo test modules have the same filename

- **Symptom (3 October 2026):** collecting the standalone and PixelCue contract suites together failed with pytest's `import file mismatch` because both suites contained `test_thumbmoves_identity.py`.
- **Cause:** pytest's default prepend import mode imported both un-packaged test files under the same top-level module name.
- **Corrective action:** run the standalone package suite and PixelCue integration-contract suite as separate pytest invocations. The PixelCue step is conditional so the independently releasable package commit does not require unrelated in-progress PixelCue changes.
- **Verification:** both invocations pass independently; together they cover all 24 current tests without an import-path collision.
- **Limit:** if both test trees are deliberately collected in one pytest process, use pytest's `--import-mode=importlib` or give the files unique module names.

### Restricted Windows test or build temporary directories become inaccessible

- **Symptom (3 October 2026):** pytest and an isolated local sdist build raised `PermissionError` / `Access is denied` while creating or cleaning generated temporary directories, even though non-temporary source files under the same workspace remained writable.
- **Cause supported by current evidence:** the restricted local host applied unusable permissions to tool-created temporary directories. The same test suite passed when run outside that filesystem restriction with a fresh, explicitly named test-only base directory.
- **Corrective action:** for local verification, use `pytest --basetemp` with a new dedicated path directly beneath an existing writable parent, and run the build in an execution context that can create and clean build-backend temporary directories. GitHub-hosted runners do not use this local restricted-host filesystem.
- **Verification:** all 24 ThumbMoves and PixelCue contract tests passed with the fresh explicit base directory.
- **Limit:** pytest clears the `--basetemp` target before a run. Only point it at a newly created, dedicated test directory; never at a workspace or user-data directory.

### Setuptools warns about legacy licence metadata

- **Symptom (3 October 2026):** the package build warned that `project.license` as a TOML table and the MIT licence classifier were deprecated.
- **Cause:** setuptools 77 and later implement PEP 639 licence expressions and deprecate the previous table form and licence classifiers.
- **Corrective action:** require `setuptools>=77.0.3`, use the SPDX expression `license = "MIT"`, declare `license-files = ["LICENSE"]`, and remove the deprecated licence classifier.
- **Verification:** build validation checks both distribution metadata and the included licence file; `twine check --strict` remains part of CI.
- **Limit:** this is build metadata only and does not change ThumbMoves' MIT licence terms.

### The Windows launcher cannot find the uv-managed Python 3.13 runtime

- **Symptom (3 October 2026):** `py -3.13 -m venv ...` reported `No suitable Python runtime found`, although `py -0p` listed `-V:Astral/CPython3.13.14` and the PixelCue environment was running Python 3.13.14.
- **Cause:** the short `-3.13` launcher form selects PythonCore releases, not the separately registered Astral/uv runtime. Python's Windows documentation says that runtimes from other distributors may require the company-qualified `-V:Company/Tag` form.
- **Corrective action:** inspect `py -0p`, then select the exact registered runtime with `py -V:Astral/CPython3.13.14 -m venv <dedicated-check-path>` for the release install check. Do not install another interpreter merely to satisfy the short launcher form.
- **Verification:** the qualified runtime created a fresh environment, installed `thumbmoves==0.1.0` from public PyPI, and imported version 0.1.0 successfully.
- **Limit:** the distributor and tag are local runtime identifiers and may differ on another machine. Re-read the installed-runtime list rather than copying this identifier blindly. Source: [Python on Windows](https://docs.python.org/3/using/windows.html#basic-use), accessed 3 October 2026.

### `python -m thumbmoves` cannot run the installed command

- **Symptom (3 October 2026):** the public package installed and imported successfully, but `python -m thumbmoves --help` failed with `No module named thumbmoves.__main__; 'thumbmoves' is a package and cannot be directly executed`.
- **Cause:** `python -m <package>` executes that package's `__main__.py`; ThumbMoves intentionally exposes its CLI through the `thumbmoves` console-script entry point instead.
- **Corrective action:** smoke-test the installed `Scripts\thumbmoves.exe --help` on Windows, or the corresponding `thumbmoves --help` command after activating the environment.
- **Verification:** the console script displayed the documented path, output, size and format options and exited successfully after a clean install from PyPI.
- **Limit:** add a minimal `thumbmoves/__main__.py` only if supporting both invocation forms becomes an explicit compatibility requirement. Source: [Python `__main__` documentation](https://docs.python.org/3/library/__main__.html#main-py-in-python-packages), accessed 3 October 2026.
