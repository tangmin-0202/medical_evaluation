from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from medical_evaluation.app import create_app
from medical_evaluation.settings import Settings


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    videos_dir = tmp_path / "presets"
    videos_dir.mkdir()
    for name in ("橡皮障完整.mp4", "橡皮障失败.mp4", "橡皮障夹子飞了.mp4"):
        (videos_dir / name).write_bytes(b"video")
    rubric_path = Path(__file__).parents[1] / "config" / "rubric.yaml"
    settings = Settings(
        project_root=tmp_path,
        data_dir="data",
        videos_dir=videos_dir,
        rubric_path=rubric_path,
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def valid_steps_payload() -> dict[str, object]:
    ranges = [
        (0, 14),
        (14, 23),
        (23, 40),
        (41, 47),
        (48, 62),
        (63, 72),
        (104, 126),
        (134, 174),
        (175, 195),
        (200, 211),
        (217, 258),
    ]
    return {
        "actor": "tangmin",
        "steps": [
            {
                "checkpoint_id": f"cp_{number:02d}",
                "time_range": {"start_sec": start, "end_sec": end},
                "label": "needs_review",
                "reason": "等待人工确认",
            }
            for number, (start, end) in enumerate(ranges, start=1)
        ],
    }


def test_annotation_update_requires_all_11_ordered_steps(
    client: TestClient,
    valid_steps_payload: dict[str, object],
) -> None:
    short = {"actor": "tangmin", "steps": valid_steps_payload["steps"][:10]}

    assert client.put("/api/videos/success/segments", json=short).status_code == 422


def test_annotation_update_round_trips(
    client: TestClient,
    valid_steps_payload: dict[str, object],
) -> None:
    response = client.put("/api/videos/success/segments", json=valid_steps_payload)

    assert response.status_code == 200
    loaded = client.get("/api/videos/success/segments").json()
    assert loaded["steps"] == valid_steps_payload["steps"]
    assert loaded["audit_history"][-1]["actor"] == "tangmin"


def test_initial_segments_use_corrected_reference_times(client: TestClient) -> None:
    response = client.get("/api/videos/failure/segments")

    assert response.status_code == 200
    steps = response.json()["steps"]
    assert len(steps) == 11
    assert steps[3]["time_range"] == {"start_sec": 41.0, "end_sec": 47.0}
    assert steps[7]["time_range"] == {"start_sec": 134.0, "end_sec": 174.0}
    assert steps[8]["time_range"] == {"start_sec": 175.0, "end_sec": 195.0}


def test_known_video_content_is_available_to_annotation_page(client: TestClient) -> None:
    response = client.get("/api/videos/success/content")

    assert response.status_code == 200
    assert response.content == b"video"


def test_reference_video_exposes_one_time_cp01_reference_guidance(
    client: TestClient,
) -> None:
    response = client.get("/annotate/success")

    assert response.status_code == 200
    guide_payload = response.text.split(
        '<script id="annotation-guides" type="application/json">',
        maxsplit=1,
    )[1].split("</script>", maxsplit=1)[0]
    guides = json.loads(guide_payload)
    assert guides == {
        "cp_01": {
            "objects": ["rubber_dam", "cp01_reference"],
            "hint": "基准视频仅标一次：框选完整橡皮布，并点36牙固定正确位置；学员实际标记由系统自动识别。",
        },
        "cp_11": {
            "objects": ["rubber_dam", "nose_region"],
            "hint": "末尾清晰帧：在绿色橡皮布内分散打3–5个正点，并紧框鼻部。",
        },
    }


def test_learner_video_cp01_guidance_does_not_request_reference_point(
    client: TestClient,
) -> None:
    response = client.get("/annotate/failure")

    guide_payload = response.text.split(
        '<script id="annotation-guides" type="application/json">',
        maxsplit=1,
    )[1].split("</script>", maxsplit=1)[0]
    guides = json.loads(guide_payload)
    assert guides["cp_01"] == {
        "objects": ["rubber_dam"],
        "hint": "框选完整橡皮布；系统观察完整CP01阶段，自动识别学员最终打孔点并与固定标准点比较。",
    }


def test_prompt_update_round_trips_normalized_point(client: TestClient) -> None:
    payload = {
        "actor": "tangmin",
        "prompts": [
            {
                "kind": "point",
                "video_id": "success",
                "frame_time_sec": 5,
                "object_id": "rubber_dam",
                "x": 0.25,
                "y": 0.75,
                "positive": True,
            }
        ],
    }

    assert client.put("/api/videos/success/prompts", json=payload).status_code == 200
    assert client.get("/api/videos/success/prompts").json()["prompts"] == payload["prompts"]


def test_unknown_video_and_blank_actor_are_rejected(
    client: TestClient,
    valid_steps_payload: dict[str, object],
) -> None:
    assert client.get("/api/videos/not-known/segments").status_code == 404
    invalid = {**valid_steps_payload, "actor": ""}
    assert client.put("/api/videos/success/segments", json=invalid).status_code == 422
