from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field

from medical_evaluation.storage import atomic_write_json


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class JobRecord(BaseModel):
    id: str
    video_id: str
    video_path: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = Field(default=0, ge=0, le=1)
    current_checkpoint: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


ProgressCallback = Callable[[float, str | None], Awaitable[None]]
Pipeline = Callable[[JobRecord, ProgressCallback], Awaitable[None]]


class JobManager:
    def __init__(self, jobs_dir: Path, pipeline: Pipeline) -> None:
        self.jobs_dir = jobs_dir
        self.pipeline = pipeline
        self.jobs: dict[str, JobRecord] = {}
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._load_jobs()
        self._worker = asyncio.create_task(self._run_worker())
        for record in self.jobs.values():
            if record.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                record.status = JobStatus.QUEUED
                self._save(record)
                await self.queue.put(record.id)

    async def stop(self) -> None:
        if self._worker is None:
            return
        self._worker.cancel()
        try:
            await self._worker
        except asyncio.CancelledError:
            pass
        self._worker = None

    async def create(self, video_id: str, video_path: Path) -> JobRecord:
        record = JobRecord(id=uuid4().hex, video_id=video_id, video_path=str(video_path))
        self.jobs[record.id] = record
        self._save(record)
        await self.queue.put(record.id)
        return record

    def get(self, job_id: str) -> JobRecord | None:
        return self.jobs.get(job_id)

    async def _run_worker(self) -> None:
        while True:
            job_id = await self.queue.get()
            record = self.jobs[job_id]
            try:
                await self._update(record, status=JobStatus.RUNNING, error=None)

                async def update_progress(
                    progress: float,
                    checkpoint: str | None = None,
                    _record: JobRecord = record,
                ) -> None:
                    await self._update(
                        _record,
                        progress=progress,
                        current_checkpoint=checkpoint,
                    )

                await self.pipeline(record, update_progress)
                await self._update(
                    record,
                    status=JobStatus.COMPLETE,
                    progress=1,
                    current_checkpoint=None,
                )
            except asyncio.CancelledError:
                await self._update(record, status=JobStatus.QUEUED)
                raise
            except Exception as exc:  # noqa: BLE001  # pragma: no cover
                await self._update(record, status=JobStatus.FAILED, error=str(exc))
            finally:
                self.queue.task_done()

    async def _update(self, record: JobRecord, **changes: object) -> None:
        changes["updated_at"] = datetime.now(UTC)
        updated = record.model_copy(update=changes)
        self.jobs[record.id] = updated
        record.__dict__.update(updated.__dict__)
        self._save(record)

    def _load_jobs(self) -> None:
        for path in self.jobs_dir.glob("*/job.json"):
            try:
                record = JobRecord.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            self.jobs[record.id] = record

    def _save(self, record: JobRecord) -> None:
        atomic_write_json(
            self.jobs_dir / record.id / "job.json",
            record.model_dump(mode="json"),
        )
