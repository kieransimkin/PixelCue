from __future__ import annotations

import argparse
import asyncio
import os
import uuid
from dataclasses import dataclass
from typing import Any, Sequence

import socketio
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import __version__
from .model import DEFAULT_TAGGER_PROFILE, TAGGER_PROFILES
from .service import KeywordExtractionService, UnsupportedMediaError


class KeywordRequest(BaseModel):
    path: str = Field(min_length=1, description="Path to one file on the PixelCue server.")
    model: str = Field(default=DEFAULT_TAGGER_PROFILE)
    request_id: str | None = Field(default=None, min_length=1, max_length=128)


@dataclass(frozen=True)
class ServerBundle:
    app: Any
    http: FastAPI
    socket: socketio.AsyncServer


def _configured_origins() -> tuple[str, ...]:
    raw = os.environ.get("PIXELCUE_CORS_ORIGINS", "")
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def create_server(
    service: KeywordExtractionService | None = None,
    *,
    cors_origins: Sequence[str] | None = None,
) -> ServerBundle:
    origins = tuple(cors_origins) if cors_origins is not None else _configured_origins()
    sio = socketio.AsyncServer(
        async_mode="asgi",
        cors_allowed_origins=list(origins) if origins else None,
    )
    http = FastAPI(title="PixelCue", version=__version__)
    keyword_service = service or KeywordExtractionService()

    if origins:
        http.add_middleware(
            CORSMiddleware,
            allow_origins=list(origins),
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    @sio.event
    async def connect(sid: str, environ: dict[str, Any], auth: Any = None) -> None:
        del environ, auth
        await sio.emit("connected", {"sid": sid}, to=sid)

    @sio.event
    async def subscribe(sid: str, data: Any) -> dict[str, str]:
        request_id = ""
        if isinstance(data, dict):
            request_id = str(data.get("request_id") or "").strip()
        if not request_id or len(request_id) > 128:
            return {"status": "error", "message": "request_id is required."}
        await sio.enter_room(sid, request_id)
        return {"status": "subscribed", "request_id": request_id}

    @http.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "pixelcue",
            "version": __version__,
            "models": sorted(TAGGER_PROFILES),
        }

    @http.post("/v1/keywords", response_model=list[str])
    async def keywords(request: KeywordRequest) -> list[str]:
        request_id = request.request_id or str(uuid.uuid4())
        loop = asyncio.get_running_loop()

        def progress(payload: dict[str, Any]) -> None:
            event = {"request_id": request_id, **payload}
            asyncio.run_coroutine_threadsafe(
                sio.emit("keyword_progress", event, room=request_id),
                loop,
            )

        await sio.emit(
            "keyword_progress",
            {"request_id": request_id, "stage": "started", "message": "Request accepted."},
            room=request_id,
        )
        try:
            result = await asyncio.to_thread(
                keyword_service.extract_keywords,
                request.path,
                model=request.model,
                progress=progress,
            )
        except FileNotFoundError as exc:
            status = 404
            detail = str(exc)
        except IsADirectoryError as exc:
            status = 400
            detail = str(exc)
        except UnsupportedMediaError as exc:
            status = 415
            detail = str(exc)
        except ValueError as exc:
            status = 422
            detail = str(exc)
        except Exception as exc:
            status = 500
            detail = f"PixelCue extraction failed: {type(exc).__name__}: {exc}"
        else:
            await sio.emit(
                "keyword_progress",
                {
                    "request_id": request_id,
                    "stage": "complete",
                    "message": "Keyword extraction complete.",
                    "completed": len(result),
                    "total": len(result),
                },
                room=request_id,
            )
            return result

        await sio.emit(
            "keyword_progress",
            {"request_id": request_id, "stage": "failed", "message": detail},
            room=request_id,
        )
        raise HTTPException(status_code=status, detail=detail)

    return ServerBundle(app=socketio.ASGIApp(sio, http), http=http, socket=sio)


bundle = create_server()
app = bundle.app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the PixelCue REST and Socket.IO server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
