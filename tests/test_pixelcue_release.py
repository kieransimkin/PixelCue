from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pixelcue_release as release


def test_pixelcue_source_identity_and_publication_policy():
    # Source/package identity also applies to development wheels. Publication
    # remains restricted to stable tags; do not loosen validate_tag for CI.
    current = release.version()
    if ".dev" in current:
        with pytest.raises(ValueError, match="release tag"):
            release.validate_tag("v" + current, check_git=False)
    else:
        assert release.validate_tag("v" + current, check_git=False) == current


def _source_fixture(root, version, module_version=None):
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "pixelcue"\nversion = "{version}"\n', encoding="utf-8"
    )
    module = root / "src" / "pixelcue" / "__init__.py"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text(
        f'__version__ = "{version if module_version is None else module_version}"\n',
        encoding="utf-8",
    )
    return root


@pytest.mark.parametrize("version", ["0.9.1", "1.0.0", "0.9.1.dev1", "1.0.0.dev12"])
def test_source_identity_accepts_stable_and_development_versions(tmp_path, version):
    root = _source_fixture(tmp_path / "source", version)
    assert release.version(root) == version


@pytest.mark.parametrize("version", ["0.9.1", "1.0.0"])
def test_stable_publication_requires_an_exact_source_match(tmp_path, version):
    root = _source_fixture(tmp_path / "source", version)
    assert release.validate_tag("v" + version, root, check_git=False) == version
    with pytest.raises(ValueError, match="does not match"):
        release.validate_tag("v99.0.0", root, check_git=False)


@pytest.mark.parametrize("version", ["0.9.1.dev1", "1.0.0.dev12"])
def test_development_source_cannot_publish_as_development_or_final_tag(tmp_path, version):
    root = _source_fixture(tmp_path / "source", version)
    with pytest.raises(ValueError, match="release tag"):
        release.validate_tag("v" + version, root, check_git=False)
    # Removing the suffix must not turn a development build into a final release.
    with pytest.raises(ValueError, match="does not match"):
        release.validate_tag("v" + version.split(".dev", 1)[0], root, check_git=False)


@pytest.mark.parametrize("version", ["0.9.1", "0.9.1.dev1"])
def test_source_version_disagreement_is_rejected(tmp_path, version):
    root = _source_fixture(tmp_path / "source", version, module_version="99.0.0")
    with pytest.raises(ValueError, match="disagree"):
        release.version(root)


@pytest.mark.parametrize("tag", ["v0.9.1.dev1", "v0.9.1rc1", "v0.9.1.post1"])
def test_nonfinal_tags_remain_forbidden_on_stable_source(tmp_path, tag):
    root = _source_fixture(tmp_path / "source", "0.9.1")
    with pytest.raises(ValueError, match="release tag"):
        release.validate_tag(tag, root, check_git=False)


@pytest.mark.parametrize("tag", ["thumbmoves-v0.8.5", "v0.8", "0.8.5"])
def test_rejects_ambiguous_monorepo_tags(tag):
    with pytest.raises(ValueError, match="release tag"):
        release.validate_tag(tag, check_git=False)


def test_rejects_version_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        release.validate_tag("v99.0.0", check_git=False)


def test_requires_both_installable_distributions(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        release.distributions(tmp_path)
