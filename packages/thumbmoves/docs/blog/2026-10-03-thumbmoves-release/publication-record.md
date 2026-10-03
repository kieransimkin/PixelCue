# ThumbMoves 0.1.0 release — project-blog record

- Date: 3 October 2026 (Europe/London)
- Blogworthiness: **pass** — ThumbMoves has moved from an internal PixelCue boundary to a reusable public Python package, with a verified PyPI release, matching GitHub assets and a documented trusted-publishing route. That is a concrete open-source milestone with a useful technical story about cache-only thumbnail access and small-library release engineering.
- Duplicate check: the public blog and the related 3 October PixelCue/KeywordMoves article were inspected. That article covers PixelCue's visual-keyword service and explicitly predates a finished public release; it does not cover ThumbMoves, OS thumbnail caches or this package publication.
- Status: ThumbMoves 0.1.0 is publicly released and independently installable.

## Verified release evidence

- Source tag: `thumbmoves-v0.1.0`
- Tagged commit: `9562e03170011e7080b54407d39c9da21877a5b4`
- PyPI: <https://pypi.org/project/thumbmoves/0.1.0/>
- GitHub release: <https://github.com/kieransimkin/PixelCue/releases/tag/thumbmoves-v0.1.0>
- Release workflow: <https://github.com/kieransimkin/PixelCue/actions/runs/37106672265>
- Post-release `main` CI: <https://github.com/kieransimkin/PixelCue/actions/runs/37107511187>
- PyPI identifies the release as Trusted Publishing from `kieransimkin/PixelCue`, workflow `thumbmoves-release.yml`, commit `9562e03` and run `37106672265`.
- The PyPI wheel and source archive hashes match the corresponding GitHub Release assets and `SHA256SUMS.txt`.
- A clean Python 3.13 virtual environment installed `thumbmoves==0.1.0` from public PyPI; `import thumbmoves` reported version `0.1.0`, and the installed `thumbmoves --help` command completed successfully.
- TestPyPI was deliberately deferred at Kieran's request; it is not presented as configured or verified.

## Media

- Editable source: `thumbmoves-release-pipeline.svg`
- Publishable file: `thumbmoves-release-pipeline.png`
- Dimensions: 1600 × 900 pixels
- PNG format: 24-bit RGB
- SVG SHA-256: `e855e99cad220c44b964cc94d589f8d7614769d114fec524ac230cbd7c246031`
- PNG SHA-256: `5707be4686b60cd9e83f9538864a4ec271b4625f65b2f4df192e7e7522ea83f5`
- Creation method: project-authored SVG rendered locally to PNG from verified release facts; no third-party artwork or logos were used.
- Rights/provenance: created for this project from Kieran's own package and public release data; suitable for public use.
- Redactions: none required; the graphic contains no credentials, private paths, unpublished work or personal data.
- Alt text: `Diagram showing ThumbMoves 0.1.0 moving from its Git tag through cross-platform GitHub Actions checks to PyPI trusted publishing and GitHub release assets.`
- Caption: `ThumbMoves 0.1.0 passed its cross-platform checks before the same verified wheel and source archive went to PyPI and GitHub Releases.`

## Proposed WordPress metadata

- Title: `Releasing ThumbMoves: a small Python library for thumbnail caches`
- Slug: `releasing-thumbmoves-python-thumbnail-cache`
- Excerpt: `ThumbMoves is a small Python library for reading thumbnails already held in Windows and Linux desktop caches. Version 0.1.0 is now on PyPI, with cross-platform checks, trusted publishing and matching GitHub release assets.`
- Category: `Uncategorised` (matching the existing PixelCue and StemLab technical posts)
- Tags: `ThumbMoves`, `PixelCue`, `Python`, `open source`, `music technology`

## Article draft

PixelCue needs small preview images, but opening and decoding every original file just to draw a thumbnail is wasteful. Desktop operating systems often already have the right image in their thumbnail cache. ThumbMoves is the small library I have separated out to ask for that cached image without quietly doing the expensive fallback itself.

Version 0.1.0 is now available on PyPI. It has one deliberately conservative contract: return a cached thumbnail when the operating system already has one, or return a miss. It does not open the source media file and it does not trigger thumbnail generation.

## One boundary instead of three copies

On Windows, ThumbMoves uses the Shell thumbnail interface with cache-only and thumbnail-only flags. On Linux and other Freedesktop-style desktops, it looks up the standard thumbnail-cache path derived from the file URI. macOS does not expose an equivalent public, strict cache-only lookup, so the current backend returns a miss rather than pretending it can keep the same promise.

That last detail is important. A cross-platform API is only useful if the shared name does not hide materially different behaviour. PixelCue can try ThumbMoves first and decide what to do after a miss, while another application can make a different choice.

Separating the code also gives it a proper reusable boundary. PixelCue no longer needs to own a one-off copy of the cache logic, and other Python tools can install the same small package.

## A release route that can be repeated

The package now has cross-platform GitHub Actions checks, wheel and source-archive validation, and a tag-driven release workflow. Production publication uses PyPI's trusted-publishing route, so the workflow exchanges its GitHub identity for a short-lived publishing token rather than storing a permanent PyPI password in the repository.

The release job also creates a matching GitHub Release with the wheel, source archive and a `SHA256SUMS.txt` file. I checked the public PyPI files against those GitHub assets, then installed version 0.1.0 into a clean Python 3.13 environment and ran the installed command-line help.

TestPyPI is not part of this first completed route. I deliberately left that optional path aside for now rather than holding up the production package.

## Where it is now

ThumbMoves 0.1.0 supports Python 3.10 and later and is released under the MIT licence. The package is available from [PyPI](https://pypi.org/project/thumbmoves/0.1.0/), with the matching files and checksums on the [GitHub release](https://github.com/kieransimkin/PixelCue/releases/tag/thumbmoves-v0.1.0).

It is a small piece of PixelCue, but that is rather the point: one narrow job, an honest platform contract and a release process that can be repeated without turning the library back into a private implementation detail.

## Potential problems

### Headless Edge render appeared to fail even though the PNG was created

- Symptom: invoking Edge directly with `--headless=new` produced the requested PNG, but PowerShell's `$LASTEXITCODE` was unset, so a wrapper incorrectly reported `Edge failed rendering`.
- Environment: Windows PowerShell, Microsoft Edge executable launched directly, 3 October 2026.
- Research: current Chrome Headless documentation uses the unified `--headless` flag; Microsoft documents that `Start-Process` is asynchronous by default and that `-Wait -PassThru` exposes a process object and exit code. Sources accessed 3 October 2026: <https://developer.chrome.com/docs/automation-and-testing/headless#command-line-flags> and <https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.management/start-process?view=powershell-7.5>.
- Corrective procedure: launch the browser with `Start-Process -WindowStyle Hidden -Wait -PassThru`, use the current `--headless` flag, check the returned process `ExitCode`, then independently verify that the output exists and has the expected dimensions and hash.
- Verification: the retained PNG exists, is 1600 × 900 24-bit RGB, has SHA-256 `5707be4686b60cd9e83f9538864a4ec271b4625f65b2f4df192e7e7522ea83f5`, and was visually inspected successfully.
- Reuse limit: this procedure applies to command-line rendering with GUI browser executables on Windows; it does not prove that every browser-rendered page or SVG is visually correct, so image inspection remains required.

### WordPress editor tab lost browser-session ownership

- Symptom: after the user-confirmation turn, the retained WordPress editor handle failed with `Tab 13 is not part of browser session 01a1002e-6c65-78a0-80a1-ea3958806bf1`.
- Environment: Codex in-app browser on Windows, WordPress 7.1.2 block editor, 3 October 2026.
- Research: an OpenAI Codex issue records the same `Tab ... is not part of browser session` symptom after in-app browser tab lifecycle teardown. Source accessed 3 October 2026: <https://github.com/openai/codex/issues/33133>. The current browser-tool guidance also says to list the selected browser's tabs and acquire or create a fresh tab when a handle is stale, without reselecting the browser.
- Corrective procedure: list the selected browser's live tabs, open a fresh tab in that same browser session, and navigate directly to the already allocated WordPress post identity rather than creating a second post.
- Verification: post `1260` reopened as the existing `Auto Draft`; the title and body were then entered once, saved and later published under that same identity. Media attachment `1261` remained persisted throughout.
- Reuse limit: only reuse a server-side post or media identity that has already been visibly verified. Never guess an ID or create a duplicate to work around a stale browser handle.

### Visual-editor iframe inspection timed out

- Symptom: a direct Playwright inspection of `iframe[name="editor-canvas"]` ended with `Playwright selector deadline exceeded`, and the editor canvas screenshot appeared grey even though the saved blocks were present.
- Environment: WordPress 7.1.2 block editor in the Codex in-app browser, 3 October 2026.
- Research: Gutenberg uses an iframe-backed editor canvas and its own Playwright runs show locator timeouts around editor-canvas and save operations. Sources accessed 3 October 2026: <https://github.com/WordPress/gutenberg/issues/20797> and <https://github.com/WordPress/gutenberg/actions/runs/20130756938>.
- Corrective procedure: save the draft first, verify WordPress reports `Saved`, then open the authenticated preview route in a fresh tab and inspect the rendered article there instead of treating the editor iframe timeout as a content failure.
- Verification: the saved preview visibly contained the intended title, complete 448-word body, image, alt text, caption, three headings, two release links, category and five tags before publication.
- Reuse limit: an authenticated preview proves the draft rendering for that account; it is not a signed-out public verification and must remain labelled separately.

## Publication verification

- WordPress post ID: `1260`
- WordPress media ID: `1261`
- Public URL: <https://kieransimkin.co.uk/2026/10/03/releasing-thumbmoves-python-thumbnail-cache/>
- Publication time: 3 October 2026 at 09:07 Europe/London, as persisted by WordPress.
- Media upload time: 3 October 2026 at 09:03 Europe/London, as persisted by WordPress.
- WordPress persistence evidence: the editor status changed to `Published`, the editor remained on `post.php?post=1260&action=edit`, the success notice read `Post published.`, and the editor exposed the intended public `View Post` URL.
- Fresh public-route desktop verification: **pass in the authenticated browser profile**. At a 1,265-pixel document client width, scroll width equalled client width; the title, all three article headings, both release links, image, caption, category and all five tags were present. The image completed loading, had its intended alt text and rendered at 650 pixels wide.
- Fresh public-route tablet verification: **pass in the authenticated browser profile**. At a requested 768 by 1,024 viewport, the effective document width was 753 pixels, scroll width equalled client width, all three headings remained present and the responsive image completed loading at 650 pixels wide.
- Fresh public-route mobile verification: **pass in the authenticated browser profile**. At a requested 390 by 844 viewport, the effective document width was 375 pixels, scroll width equalled client width, the title and all three headings remained present, and the responsive image completed loading at 327 pixels wide with its alt text intact. The viewport override was reset after testing.
- Blog-index verification: **pass**. A fresh `/blog/` route showed the article first, with the intended title, excerpt, date and public URL.
- Media and metadata: **pass**. Attachment `1261` is `thumbmoves-release-pipeline.png` at <https://kieransimkin.co.uk/wp-content/uploads/2026/10/thumbmoves-release-pipeline.png>; WordPress reported PNG, 399 KB and 1600 by 900 pixels, and persisted the intended alt text and caption. The post persisted category `Uncategorised` and tags `ThumbMoves`, `PixelCue`, `Python`, `open source` and `music technology`.
- Signed-out verification: **unavailable, not passed**. The browser service exposed only the authenticated in-app profile and MCP Apps; no separate anonymous browser was available. The public page must not be described as independently signed-out verified from this run.
- Browser interaction note: this WordPress editor published on the first top-bar `Publish` click; the complete preview and field readback were therefore finished before the user-confirmed committing action.
