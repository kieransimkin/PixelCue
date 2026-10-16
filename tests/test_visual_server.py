from fastapi.testclient import TestClient
from PIL import Image

from pixelcue.server import create_server


class Keywords:
    def extract_keywords(self, path, *, model, **kwargs):
        return ["mock tag"]


def test_real_analysis_route(tmp_path):
    path = tmp_path/"fixture.png"
    Image.new("RGB", (32, 32), "red").save(path)
    client = TestClient(create_server(Keywords()).http)
    assert client.get("/health").json()["visual_analysis"]["route"] == "/v1/analysis"
    response = client.post("/v1/analysis", json={"path": str(path)})
    assert response.status_code == 200
    assert response.json()["schema"] == "pixelcue.visual-analysis/1"
    assert response.json()["measurements"]["colour"]["palette"][0]["hex"] == "#ff0000"
    assert client.post("/v1/analysis", json={"path": str(path), "max_edge": 9999}).status_code == 422
    assert client.post("/v1/analysis", json={"path": str(path), "models": ["unknown"]}).status_code == 422
    assert client.post("/v1/analysis", json={"path": str(tmp_path/"missing")}).status_code == 404
