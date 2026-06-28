import json
import os
import uuid

import pytest


@pytest.mark.django_db
def test_upload_files(client):
    """POST /api/upload creates a task and stores uploaded photos."""
    test_image = os.path.join(
        os.path.dirname(__file__), "..", "..", "uploads",
        "05f335be-e6f3-4340-b71e-d3f84ec1878b", "167111-01.png"
    )
    if not os.path.exists(test_image):
        pytest.skip("No test image available")

    with open(test_image, "rb") as f:
        response = client.post(
            "/api/upload",
            {
                "brand": "蔚来",
                "model": "ES8",
                "year": "2024",
                "files": f,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["uploaded_count"] == 1
    assert data["vehicle_info"]["brand"] == "蔚来"


@pytest.mark.django_db
def test_assess_stream_missing_task(client):
    """GET /api/assess/<task_id> for unknown task returns error SSE event."""
    response = client.get("/api/assess/non-existent-task")
    assert response.status_code == 200
    assert response["Content-Type"] == "text/event-stream"
    body = b"".join(response.streaming_content).decode("utf-8")
    assert 'event: error\ndata: {"message": "任务ID格式无效"}' in body


@pytest.mark.django_db
def test_assess_stream_unknown_uuid(client):
    """GET /api/assess/<task_id> for valid but unknown UUID returns error."""
    response = client.get(f"/api/assess/{uuid.uuid4()}")
    assert response.status_code == 200
    assert response["Content-Type"] == "text/event-stream"
    body = b"".join(response.streaming_content).decode("utf-8")
    assert 'event: error\ndata: {"message": "任务不存在"}' in body
