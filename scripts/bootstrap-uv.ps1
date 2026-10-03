$ErrorActionPreference = "Stop"
uv python install 3.13
uv lock
uv sync --extra faces
Write-Host "PixelCue environment ready. Run: uv run pixelcue <path>"
