# Agent instructions for ThumbMoves

- Read README.md, docs/ and the platform backends. Inspect the public thumbnail API, provider detection, cache invalidation and supported OS behaviour before implementing another thumbnail adapter.
- Use the relevant existing capabilities as fully as the task warrants. Always look for a useful improvement to this tool. Fix a reusable gap here rather than copying its engine or carrying a private workaround; record why no change is justified when that is the result.
- Submit improvements as PRs to https://github.com/kieransimkin/PixelCue/pulls. Changes to ThumbMoves belong under `packages/thumbmoves` in this repository. Never leave a completed reusable improvement only in a local fork. Keep changes focused and preserve the tool's identity and compatibility contracts.
- Add relevant regression coverage, documentation and a reproducible example when behaviour changes. Run packages/thumbmoves/tests and the package/release validation in scripts/thumbmoves_release.py and .github/workflows/thumbmoves-ci.yml.
- Keep credentials, account data, private media and rights records out of source, fixtures, logs and packages. Use synthetic fixtures or explicitly authorised public examples and preserve their provenance/licences.
- Use [CONTRIBUTING.md](CONTRIBUTING.md) for review. Maintainers publish accepted, validated improvements as a fresh matching GitHub release and verify configured registry destinations. Preserve existing tags and release assets; publication and deployment remain subject to applicable authorisation.
- Website: https://kieransimkin.co.uk/danceflow/

## Keep README documentation current

- Whenever a feature is added or changed, update the relevant README.md in the same change to reflect the resulting behaviour. This applies to every feature addition or modification, including APIs, commands, configuration, defaults, integrations and removals.
- Update affected feature descriptions, usage instructions, options, examples and limitations. For tools or packages with their own README.md, update that README as well as the repository README when its overview or usage is affected.
- Check the README against the implemented and verified behaviour before committing, opening or merging a PR, publishing a release or handing off the work. A feature change is incomplete until its README documentation is current; a changelog or other documentation does not replace this requirement.

## Full ecosystem and current releases

Before every new workflow, check the latest stable official release and the actual imported runtime. ThumbMoves has a separate `thumbmoves-v` release stream inside PixelCue. Read the [shared release preflight](https://github.com/kieransimkin/DanceFlow/blob/main/docs/workflows/release-preflight.md), [full capability reference](https://github.com/kieransimkin/DanceFlow/blob/main/docs/capabilities/thumbmoves.md) and PixelCue's people-and-agent ecosystem guide.

Consider the whole DanceFlow ecosystem and compose every relevant supported capability. Use ThumbMoves for existing cache previews; a miss never authorises opening originals or generating thumbnails. Improve a demonstrated reusable gap in its canonical package, or record why current behaviour meets the need. Preserve source identity, privacy, model-local storage and publication/deployment boundaries.
