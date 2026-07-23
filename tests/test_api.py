from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "MedInc": 8.3252,
    "HouseAge": 41,
    "AveRooms": 6.984,
    "AveBedrms": 1.023,
    "Population": 322,
    "AveOccup": 2.555,
    "Latitude": 37.88,
    "Longitude": -122.23,
}


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "running"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "running"
    assert "features" in body


def test_predict_valid_request():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert "predicted_price" in body
    assert "predicted_price_short" in body
    assert "confidence_range" in body


def test_predict_rejects_out_of_range_input():
    payload = {**VALID_PAYLOAD, "Latitude": 999}  # outside the valid 32-42 range
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_invalid_type():
    payload = {**VALID_PAYLOAD, "MedInc": "not-a-number"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_file_rejects_non_csv():
    files = {"file": ("not_a_csv.txt", b"hello world", "text/plain")}
    response = client.post("/predict-file", files=files)
    assert response.status_code == 400
