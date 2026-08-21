from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from medical_evaluation.app import create_app
from medical_evaluation.settings import Settings


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    videos_dir = tmp_path / "presets"
    videos_dir.mkdir()
    for name in ("橡皮障完整.mp4", "橡皮障失败.mp4", "橡皮障夹子飞了.mp4"):
        (videos_dir / name).write_bytes(b"video")
    rubric_path = Path(__file__).parents[1] / "config" / "rubric.yaml"
    return Settings(
        project_root=tmp_path,
        data_dir="data",
        videos_dir=videos_dir,
        rubric_path=rubric_path,
    )


def test_home_lists_three_presets(test_settings: Settings) -> None:
    with TestClient(create_app(test_settings)) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "橡皮障完整.mp4" in response.text
    assert "橡皮障失败.mp4" in response.text
    assert "橡皮障夹子飞了.mp4" in response.text


def test_create_preset_job_returns_pollable_id(test_settings: Settings) -> None:
    with TestClient(create_app(test_settings)) as client:
        response = client.post("/api/jobs", json={"preset_id": "success"})
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        payload = _wait_for_terminal_status(client, job_id)

    assert payload["status"] == "complete"
    assert payload["progress"] == 1.0
    report_path = test_settings.data_dir / "jobs" / job_id / "report.json"
    assert report_path.is_file()
    assert len(__import__("json").loads(report_path.read_text(encoding="utf-8"))["checkpoints"]) == 11


def test_upload_uses_generated_name_instead_of_client_filename(
    test_settings: Settings,
) -> None:
    with TestClient(create_app(test_settings)) as client:
        response = client.post(
            "/api/jobs",
            files={"video": ("../../unsafe.mp4", b"video", "video/mp4")},
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        payload = client.get(f"/api/jobs/{job_id}").json()

    stored = Path(payload["video_path"])
    assert stored.parent == test_settings.data_dir / "uploads"
    assert stored.name != "unsafe.mp4"
    assert stored.suffix == ".mp4"
    assert stored.read_bytes() == b"video"


def test_job_request_requires_exactly_one_source(test_settings: Settings) -> None:
    with TestClient(create_app(test_settings)) as client:
        assert client.post("/api/jobs", json={}).status_code == 422
        response = client.post(
            "/api/jobs",
            data={"preset_id": "success"},
            files={"video": ("sample.mp4", b"video", "video/mp4")},
        )

    assert response.status_code == 422


def _wait_for_terminal_status(client: TestClient, job_id: str) -> dict[str, object]:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        payload = client.get(f"/api/jobs/{job_id}").json()
        if payload["status"] in {"complete", "failed"}:
            return payload
        time.sleep(0.01)
    raise AssertionError("job did not reach a terminal status")
