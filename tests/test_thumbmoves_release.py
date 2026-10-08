from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SPEC = spec_from_file_location(
    "thumbmoves_release", ROOT / "scripts" / "thumbmoves_release.py"
)
assert SPEC is not None and SPEC.loader is not None
release = module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def test_thumbmoves_source_versions_agree():
    assert release.source_version() == "0.1.1"


def test_release_tag_is_namespaced_for_pixelcue_monorepo():
    metadata = release.release_metadata(
        "thumbmoves-v0.1.1", "kieransimkin/PixelCue", check_git=False
    )
    assert metadata == {
        "tag": "thumbmoves-v0.1.1",
        "version": "0.1.1",
        "prerelease": "false",
    }


@pytest.mark.parametrize("tag", ["v0.1.0", "thumbmoves-0.1.0", "thumbmoves-v0.1"])
def test_release_rejects_ambiguous_or_incomplete_tags(tag):
    with pytest.raises(ValueError, match="Release tag"):
        release.release_metadata(tag, "kieransimkin/PixelCue", check_git=False)


def test_release_rejects_version_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        release.release_metadata(
            "thumbmoves-v0.2.0", "kieransimkin/PixelCue", check_git=False
        )
