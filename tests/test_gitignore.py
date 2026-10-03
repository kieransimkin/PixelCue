from pathlib import Path


def test_all_tsv_files_are_gitignored():
    gitignore = (
        Path(__file__).parents[1] / ".gitignore"
    ).read_text(encoding="utf-8").splitlines()

    assert "*.tsv" in {line.strip() for line in gitignore}
