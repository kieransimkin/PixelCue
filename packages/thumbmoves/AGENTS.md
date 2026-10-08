# Agent instructions for ThumbMoves

- Read README.md, docs/ and the platform backends. Inspect the public thumbnail API, provider detection, cache invalidation and supported OS behaviour before implementing another thumbnail adapter.
- Use the relevant existing capabilities as fully as the task warrants. Always look for a useful improvement to this tool. Fix a reusable gap here rather than copying its engine or carrying a private workaround; record why no change is justified when that is the result.
- Submit improvements as PRs to https://github.com/kieransimkin/PixelCue/pulls. Changes to ThumbMoves belong under `packages/thumbmoves` in this repository. Never leave a completed reusable improvement only in a local fork. Keep changes focused and preserve the tool's identity and compatibility contracts.
- Add relevant regression coverage, documentation and a reproducible example when behaviour changes. Run packages/thumbmoves/tests and the package/release validation in scripts/thumbmoves_release.py and .github/workflows/thumbmoves-ci.yml.
- Keep credentials, account data, private media and rights records out of source, fixtures, logs and packages. Use synthetic fixtures or explicitly authorised public examples and preserve their provenance/licences.
- Use [CONTRIBUTING.md](CONTRIBUTING.md) for review. Maintainers publish accepted, validated improvements as a fresh matching GitHub release and verify configured registry destinations. Preserve existing tags and release assets; publication and deployment remain subject to applicable authorisation.
- Website: https://kieransimkin.co.uk/danceflow/
