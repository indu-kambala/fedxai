from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["phase"] == "0"
    assert "not a diagnosis" in body["disclaimer"].lower()


def test_analyze_is_not_implemented() -> None:
    response = client.post("/api/analyze", json={"hospital_id": "hospital_a"})
    assert response.status_code == 501


def test_experiments_do_not_invent_metrics() -> None:
    response = client.get("/api/experiments")
    assert response.status_code == 200
    body = response.json()
    assert body["unrun_display"] == "Not evaluated"
    for item in body["matrix"]:
        assert item["status"] == "not_evaluated"


def test_privacy_never_centralizes_raw_images() -> None:
    response = client.get("/api/privacy")
    assert response.status_code == 200
    body = response.json()
    assert body["raw_patient_images_centralized"] is False
    assert body["perfect_privacy_claimed"] is False


def test_federated_status_forbids_raw_images_on_wire() -> None:
    response = client.get("/api/federated/status")
    assert response.status_code == 200
    assert response.json()["raw_images_on_wire"] is False
