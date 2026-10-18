# Contributing to ThumbMoves

Agents and people are welcome to improve ThumbMoves. Review its current capabilities and use the supported features before choosing an approach. Look for useful improvements wherever possible: a missing feature, a reproducible defect, simpler composition, clearer evidence, better performance or documentation. Avoid changes that do not serve a concrete need.

1. Read README.md, docs/ and the platform backends. Inspect the public thumbnail API, provider detection, cache invalidation and supported OS behaviour before implementing another thumbnail adapter.
2. Start a focused branch from the current default branch. Describe the problem, intended behaviour and compatibility constraints. Implement the smallest useful shared improvement; add relevant tests, documentation and a reproducible example when behaviour changes.
3. Run packages/thumbmoves/tests and the package/release validation in scripts/thumbmoves_release.py and .github/workflows/thumbmoves-ci.yml. Report the commands, results and any genuine verification limits. Keep private inputs, credentials and generated caches out of the contribution.
4. Open a pull request at https://github.com/kieransimkin/PixelCue/pulls. Changes to ThumbMoves belong under `packages/thumbmoves` in this repository. Include the problem, resulting behaviour, validation and any risks. Submit improvements upstream rather than leaving them only in a local fork; do not push directly to the default branch as a substitute for review.

Maintainers review accepted changes, reserve a fresh version and publish a matching GitHub release from the validated source. They verify installable assets, checksums and each configured registry separately. Existing tags and release assets remain immutable. A submitted PR is a contribution awaiting review, not a published release or permission for unrelated deployment.

Website: [https://kieransimkin.co.uk/danceflow/](https://kieransimkin.co.uk/danceflow/).

## Full ecosystem and current releases

Before every new workflow, check the latest stable official release and the actual imported runtime. ThumbMoves has a separate `thumbmoves-v` release stream inside PixelCue. Read the [shared release preflight](https://github.com/kieransimkin/DanceFlow/blob/main/docs/workflows/release-preflight.md), [full capability reference](https://github.com/kieransimkin/DanceFlow/blob/main/docs/capabilities/thumbmoves.md) and PixelCue's people-and-agent ecosystem guide.

Consider the whole DanceFlow ecosystem and compose every relevant supported capability. Use ThumbMoves for existing cache previews; a miss never authorises opening originals or generating thumbnails. Improve a demonstrated reusable gap in its canonical package, or record why current behaviour meets the need. Preserve source identity, privacy, model-local storage and publication/deployment boundaries.
