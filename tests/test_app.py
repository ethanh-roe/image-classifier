import io

from fastapi.testclient import TestClient
from PIL import Image

from app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_three_labels():
    buffer = io.BytesIO()
    Image.new("RGB", (224, 224), color="red").save(buffer, format="PNG")
    response = client.post("/predict", files={"file": ("red.png", buffer.getvalue(), "image/png")})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 3


def test_predict_rejects_non_image():
    response = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400