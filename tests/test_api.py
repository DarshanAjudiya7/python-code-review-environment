from fastapi.testclient import TestClient

from server.app import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ok", "healthy"}
    assert payload["environment"] == "python_code_review_env"


def test_reset_returns_expected_observation():
    response = client.post("/reset", json={"task_id": "syntax-fix-easy"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["observation"]["task_id"] == "syntax-fix-easy"
    assert "current_code" in payload["observation"]


def test_tasks_endpoint_lists_three_tasks():
    response = client.get("/tasks")

    assert response.status_code == 200
    assert len(response.json()) == 3
