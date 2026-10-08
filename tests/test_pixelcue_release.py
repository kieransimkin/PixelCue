from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pixelcue_release as release


def test_pixelcue_versions_and_release_tag_agree():
    current = release.version()
    assert release.validate_tag("v" + current, check_git=False) == current


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
