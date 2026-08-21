from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from medical_evaluation.domain import CheckpointStatus
from medical_evaluation.jobs import JobManager, JobRecord, ProgressCallback
from medical_evaluation.reporting import CheckpointResult, EvaluationReport, RunAudit
from medical_evaluation.settings import Settings
from medical_evaluation.storage import atomic_write_json
from medical_evaluation.web.routes import create_router


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or Settings()

    async def fake_pipeline(job: JobRecord, update: ProgressCallback) -> None:
        started_at = datetime.now(UTC)
        results: list[CheckpointResult] = []
        for number in range(1, 12):
            checkpoint_id = f"cp_{number:02d}"
            await update(number / 12, checkpoint_id)
            results.append(
                CheckpointResult(
                    checkpoint_id=checkpoint_id,
                    status=CheckpointStatus.NEEDS_REVIEW,
                    confidence=0,
                    reason_code="demo_pipeline_pending",
                    reason="视觉模型尚未运行，请人工复核。",
                    suggestion="在标注页确认时间段和提示点后再运行正式分析。",
                )
            )
        report = EvaluationReport(
            job_id=job.id,
            video_id=job.video_id,
            checkpoints=results,
            audit=RunAudit(
                rubric_version="rubber-dam-v1",
                model_versions={"pipeline": "fake-demo"},
                started_at=started_at,
                completed_at=datetime.now(UTC),
                degradations=["GPU pipeline is not connected yet"],
            ),
            overall_feedback="当前为网页流程演示，11 个考核点均等待视觉模型或人工复核。",
        )
        atomic_write_json(
            resolved.data_dir / "jobs" / job.id / "report.json",
            report.model_dump(mode="json"),
        )

    manager = JobManager(resolved.data_dir / "jobs", fake_pipeline)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await manager.start()
        try:
            yield
        finally:
            await manager.stop()

    app = FastAPI(title="橡皮障隔离术视频考核 Demo", lifespan=lifespan)
    app.state.settings = resolved
    app.state.job_manager = manager
    web_root = Path(__file__).parent / "web"
    app.mount("/static", StaticFiles(directory=web_root / "static"), name="static")
    app.include_router(create_router(resolved, manager, web_root / "templates"))
    return app
