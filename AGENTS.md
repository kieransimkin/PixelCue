# Agent instructions for PixelCue

- Read README.md, docs/ and the service/model adapters. Inspect batch analysis, metadata, model loading, local HTTP integration and the shared ThumbMoves package before adding a parallel implementation.
- Use the relevant existing capabilities as fully as the task warrants. Always look for a useful improvement to this tool. Fix a reusable gap here rather than copying its engine or carrying a private workaround; record why no change is justified when that is the result.
- Submit improvements as PRs to https://github.com/kieransimkin/PixelCue/pulls. Never leave a completed reusable improvement only in a local fork. Keep changes focused and preserve the tool's identity and compatibility contracts.
- Add relevant regression coverage, documentation and a reproducible example when behaviour changes. Run the relevant tests in .github/workflows/release.yml and validate source/package identity with scripts/pixelcue_release.py. Use fixtures for model-dependent tests; do not download models or process private files without authorisation.
- Keep credentials, account data, private media and rights records out of source, fixtures, logs and packages. Use synthetic fixtures or explicitly authorised public examples and preserve their provenance/licences.
- Use [CONTRIBUTING.md](CONTRIBUTING.md) for review. Maintainers publish accepted, validated improvements as a fresh matching GitHub release and verify configured registry destinations. Preserve existing tags and release assets; publication and deployment remain subject to applicable authorisation.
- Website: https://kieransimkin.co.uk/danceflow/
