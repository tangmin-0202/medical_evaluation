from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from medical_evaluation.app import create_app
from medical_evaluation.domain import CheckpointStatus
from medical_evaluation.reporting import CheckpointResult, EvaluationReport, RunAudit
from medical_evaluation.settings import Settings
from medical_evaluation.storage import atomic_write_json
from medical_evaluation.vlm.schemas import VlmReview


@pytest.fixture
def report_client(tmp_path: Path) -> tuple[TestClient, Settings, EvaluationReport]:
    videos_dir = tmp_path / "presets"
    videos_dir.mkdir()
    source_video = videos_dir / "橡皮障完整.mp4"
    source_video.write_bytes(b"video")
    for name in ("橡皮障失败.mp4", "橡皮障夹子飞了.mp4"):
        (videos_dir / name).write_bytes(b"video")
    settings = Settings(
        project_root=tmp_path,
        data_dir="data",
        videos_dir=videos_dir,
        rubric_path=Path(__file__).parents[1] / "config" / "rubric.yaml",
    )
    checkpoints = [
        CheckpointResult(
            checkpoint_id=f"cp_{number:02d}",
            status=(
                CheckpointStatus.NEEDS_REVIEW if number == 7 else CheckpointStatus.CORRECT
            ),
            confidence=0.9,
            reason_code="test_result",
            reason="自动判定结果",
            suggestion="继续练习",
        )
        for number in range(1, 12)
    ]
    report = EvaluationReport(
        job_id="report-job",
        video_id="success",
        checkpoints=checkpoints,
        audit=RunAudit(
            rubric_version="test-v1",
            model_versions={"pipeline": "fake"},
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            runtime_sec=1.2,
        ),
        overall_feedback="测试总评",
    )
    atomic_write_json(
        settings.data_dir / "jobs" / report.job_id / "report.json",
        report.model_dump(mode="json"),
    )
    app = create_app(settings)
    with TestClient(app) as client:
        yield client, settings, report


def test_report_page_contains_11_checkpoint_rows(report_client) -> None:
    client, _, report = report_client

    response = client.get(f"/reports/{report.job_id}")

    assert response.status_code == 200
    assert response.text.count('data-checkpoint-id="cp_') == 11
    assert "测试总评" in response.text


def test_partial_report_shows_provisional_score_and_ai_commentary(report_client) -> None:
    client, settings, report = report_client
    checkpoints = []
    for item in report.checkpoints:
        included = item.checkpoint_id in {"cp_09", "cp_11"}
        update = {"included_in_provisional_score": included}
        if item.checkpoint_id == "cp_09":
            update["ai_commentary"] = VlmReview(
                evidence_supported=True,
                semantic_status="supports",
                reason_zh="证据支持支架居中。",
                suggestion_zh="保持支架位置。",
                cited_evidence_indices=[],
            )
        if item.checkpoint_id == "cp_11":
            update["status"] = CheckpointStatus.INCORRECT
        checkpoints.append(item.model_copy(update=update))
    partial = report.model_copy(
        update={"checkpoints": checkpoints, "overall_feedback": "阶段性总评"}
    )
    atomic_write_json(
        settings.data_dir / "jobs" / report.job_id / "report.json",
        partial.model_dump(mode="json"),
    )

    response = client.get(f"/reports/{report.job_id}")

    assert response.status_code == 200
    assert "阶段性得分" in response.text
    assert "已评估 2/11 项" in response.text
    assert "非最终成绩" in response.text
    assert "50.0" in response.text
    assert "AI 点评" in response.text
    assert "证据支持支架居中。" in response.text
    assert "尚未接入自动评估" in response.text
    assert ">/ 100<" not in response.text


def test_resolving_review_recomputes_final_score(report_client) -> None:
    client, _, report = report_client

    response = client.put(
        f"/api/reports/{report.job_id}/cp_07/review",
        json={"actor": "tangmin", "status": "incorrect", "reason": "人工确认接触不足"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["final_score"] is not None
    cp07 = payload["checkpoints"][6]
    assert cp07["status"] == "incorrect"
    assert cp07["review_history"][-1]["automatic_status"] == "needs_review"


def test_single_checkpoint_rerun_preserves_prior_audit(report_client) -> None:
    client, settings, report = report_client

    response = client.post(f"/api/jobs/{report.job_id}/checkpoints/cp_07/rerun")

    assert response.status_code == 202
    assert response.json()["checkpoint_id"] == "cp_07"
    assert (settings.data_dir / "jobs" / report.job_id / "rerun-requests.json").is_file()


def test_delete_job_removes_only_derived_artifacts(report_client) -> None:
    client, settings, report = report_client
    source_video = settings.videos_dir / "橡皮障完整.mp4"
    run_dir = settings.data_dir / "runs" / report.job_id
    run_dir.mkdir(parents=True)
    (run_dir / "overlay.jpg").write_bytes(b"derived")

    response = client.delete(f"/api/jobs/{report.job_id}/artifacts")

    assert response.status_code == 204
    assert not run_dir.exists()
    assert source_video.exists()
    assert (settings.data_dir / "jobs" / report.job_id / "report.json").exists()
