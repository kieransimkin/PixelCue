# Agent instructions for PixelCue

- Store downloaded model weights and model-specific caches in the owning project's `models/`, excluded by `models/` in `.gitignore` and distributions. Use `PIXELCUE_MODELS_DIR` for a composed project's model root. Do not silently populate a global cache or relocate existing weights without verifying references and provenance.

- Read README.md, docs/ and the service/model adapters. Inspect batch analysis, metadata, model loading, local HTTP integration and the shared ThumbMoves package before adding a parallel implementation.
- Use the relevant existing capabilities as fully as the task warrants. Always look for a useful improvement to this tool. Fix a reusable gap here rather than copying its engine or carrying a private workaround; record why no change is justified when that is the result.
- Submit improvements as PRs to https://github.com/kieransimkin/PixelCue/pulls. Never leave a completed reusable improvement only in a local fork. Keep changes focused and preserve the tool's identity and compatibility contracts.
- Add relevant regression coverage, documentation and a reproducible example when behaviour changes. Run the relevant tests in .github/workflows/release.yml and validate source/package identity with scripts/pixelcue_release.py. Use fixtures for model-dependent tests; do not download models or process private files without authorisation.
- Keep credentials, account data, private media and rights records out of source, fixtures, logs and packages. Use synthetic fixtures or explicitly authorised public examples and preserve their provenance/licences.
- Use [CONTRIBUTING.md](CONTRIBUTING.md) for review. Maintainers publish accepted, validated improvements as a fresh matching GitHub release and verify configured registry destinations. Preserve existing tags and release assets; publication and deployment remain subject to applicable authorisation.
- Website: https://kieransimkin.co.uk/danceflow/

## Keep README documentation current

- Whenever a feature is added or changed, update the relevant README.md in the same change to reflect the resulting behaviour. This applies to every feature addition or modification, including APIs, commands, configuration, defaults, integrations and removals.
- Update affected feature descriptions, usage instructions, options, examples and limitations. For tools or packages with their own README.md, update that README as well as the repository README when its overview or usage is affected.
- Check the README against the implemented and verified behaviour before committing, opening or merging a PR, publishing a release or handing off the work. A feature change is incomplete until its README documentation is current; a changelog or other documentation does not replace this requirement.
