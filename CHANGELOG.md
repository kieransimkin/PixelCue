# Changelog

## 0.8.5 - 2026-10-08

- Publish the reviewed local visual-keyword service with HTTP and request-scoped Socket.IO progress.
- Add persistent analysis reuse, worker diagnostics, model selection and media queue reliability improvements.
- Use the independently released ThumbMoves thumbnail-cache library.
- Keep model downloads opt-in at use time; no media or model checkpoints are included in release packages.
- Add model-free CI, tag/version and package-content validation, and a GitHub release workflow with checksums.
- Retire stale nested source and generated metadata; make thumbnail integration tests target ThumbMoves and read UTF-8 consistently.
