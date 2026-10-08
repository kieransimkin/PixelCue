from fastapi.testclient import TestClient

from pixelcue.server import create_server


class FakeService:
    def extract_keywords(self, path, *, model, progress):
        progress({"stage": "analysis", "message": "fake progress"})
        return ["one", "two words"]


def test_health_lists_available_models():
    bundle = create_server(FakeService())
    response = TestClient(bundle.http).get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "pixelcue"
    assert "smolvlm-256m" in payload["models"]


def test_keyword_route_returns_only_a_json_list(monkeypatch):
    bundle = create_server(FakeService())
    events = []

    async def fake_emit(name, payload, **options):
        events.append((name, payload, options))

    monkeypatch.setattr(bundle.socket, "emit", fake_emit)
    response = TestClient(bundle.http).post(
        "/v1/keywords",
        json={
            "path": "C:/media/example.png",
            "model": "smolvlm-256m",
            "request_id": "test-request",
        },
    )
    assert response.status_code == 200
    assert response.json() == ["one", "two words"]
    stages = [event[1]["stage"] for event in events]
    assert stages[0] == "started"
    assert "complete" in stages
    assert all(event[0] == "keyword_progress" for event in events)


def test_socketio_subscribe_uses_request_id_as_room(monkeypatch):
    bundle = create_server(FakeService())
    calls = []

    async def fake_enter_room(sid, room):
        calls.append((sid, room))

    monkeypatch.setattr(bundle.socket, "enter_room", fake_enter_room)
    handler = bundle.socket.handlers["/"]["subscribe"]

    import asyncio

    result = asyncio.run(handler("client-1", {"request_id": "job-1"}))
    assert result == {"status": "subscribed", "request_id": "job-1"}
    assert calls == [("client-1", "job-1")]
