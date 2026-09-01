from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.datastructures import UploadFile

from medical_evaluation.annotations import (
    AnnotationStore,
    PromptUpdateRequest,
    SegmentAnnotation,
    SegmentUpdateRequest,
    VideoAnnotations,
)
from medical_evaluation.domain import CheckpointStatus
from medical_evaluation.jobs import JobManager
from medical_evaluation.reporting import EvaluationReport, ReviewAuditEntry
from medical_evaluation.rubric import Rubric, load_rubric
from medical_evaluation.settings import Settings
from medical_evaluation.storage import atomic_write_json, safe_child
from medical_evaluation.video import SUPPORTED_VIDEO_EXTENSIONS

PRESETS = {
    "success": "橡皮障完整.mp4",
    "failure": "橡皮障失败.mp4",
    "clamp_failure": "橡皮障夹子飞了.mp4",
}

ANNOTATION_GUIDES = {
    "cp_01": {
        "objects": ["rubber_dam"],
        "hint": "框选完整橡皮布；系统观察完整CP01阶段，自动识别学员最终打孔点并与固定标准点比较。",
    },
    "cp_11": {
        "objects": ["rubber_dam", "nose_region"],
        "hint": "末尾清晰帧：在绿色橡皮布内分散打3–5个正点，并紧框鼻部。",
    },
}

CP01_REFERENCE_GUIDE = {
    "objects": ["rubber_dam", "cp01_reference"],
    "hint": "基准视频仅标一次：框选完整橡皮布，并点36牙固定正确位置；学员实际标记由系统自动识别。",
}


class ReviewResolutionRequest(BaseModel):
    actor: str = Field(min_length=1)
    status: Literal["correct", "incorrect", "incomplete"]
    reason: str = Field(min_length=1)


def create_router(settings: Settings, manager: JobManager, template_dir: Path) -> APIRouter:
    router = APIRouter()
    templates = Jinja2Templates(directory=template_dir)
    annotation_store = AnnotationStore(settings.data_dir / "annotations")
    rubric = load_rubric(settings.rubric_path)

    @router.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"presets": PRESETS},
        )

    @router.post("/api/jobs", status_code=status.HTTP_202_ACCEPTED)
    async def create_job(request: Request) -> dict[str, str]:
        content_type = request.headers.get("content-type", "")
        if content_type.startswith("application/json"):
            try:
                payload = await request.json()
            except ValueError as exc:
                raise HTTPException(status_code=422, detail="invalid JSON") from exc
            if not isinstance(payload, dict) or set(payload) != {"preset_id"}:
                raise HTTPException(status_code=422, detail="provide exactly one preset_id")
            preset_id = payload.get("preset_id")
            if preset_id not in PRESETS:
                raise HTTPException(status_code=422, detail="unknown preset_id")
            video_path = settings.videos_dir / PRESETS[preset_id]
            if not video_path.is_file():
                raise HTTPException(status_code=404, detail="preset video is missing")
            record = await manager.create(preset_id, video_path)
            return {"job_id": record.id}

        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            upload = form.get("video")
            if not isinstance(upload, UploadFile) or set(form) != {"video"}:
                raise HTTPException(status_code=422, detail="provide exactly one video upload")
            stored_path = await _store_upload(upload, settings)
            record = await manager.create(f"upload-{stored_path.stem}", stored_path)
            return {"job_id": record.id}

        raise HTTPException(status_code=422, detail="use JSON or multipart form data")

    @router.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, object]:
        record = manager.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="job not found")
        return record.model_dump(mode="json")

    @router.get("/jobs/{job_id}", response_class=HTMLResponse)
    async def job_page(request: Request, job_id: str) -> HTMLResponse:
        if manager.get(job_id) is None:
            raise HTTPException(status_code=404, detail="job not found")
        return templates.TemplateResponse(
            request=request,
            name="job.html",
            context={"job_id": job_id},
        )

    @router.get("/api/videos/{video_id}/segments")
    async def get_segments(video_id: str) -> dict[str, object]:
        _require_known_video(video_id, manager)
        return _load_or_default(annotation_store, rubric, video_id).model_dump(mode="json")

    @router.get("/api/videos/{video_id}/content", response_class=FileResponse)
    async def get_video_content(video_id: str) -> FileResponse:
        path = _known_video_path(video_id, settings, manager)
        return FileResponse(path)

    @router.put("/api/videos/{video_id}/segments")
    async def put_segments(
        video_id: str,
        update: SegmentUpdateRequest,
    ) -> dict[str, object]:
        _require_known_video(video_id, manager)
        saved = annotation_store.save_segments(video_id, update.steps, actor=update.actor)
        return saved.model_dump(mode="json")

    @router.get("/api/videos/{video_id}/prompts")
    async def get_prompts(video_id: str) -> dict[str, object]:
        _require_known_video(video_id, manager)
        return _load_or_default(annotation_store, rubric, video_id).model_dump(mode="json")

    @router.put("/api/videos/{video_id}/prompts")
    async def put_prompts(
        video_id: str,
        update: PromptUpdateRequest,
    ) -> dict[str, object]:
        _require_known_video(video_id, manager)
        if any(prompt.video_id != video_id for prompt in update.prompts):
            raise HTTPException(status_code=422, detail="prompt video_id does not match URL")
        saved = annotation_store.save_prompts(
            video_id,
            update.prompts,
            actor=update.actor,
            default_steps=_default_steps(rubric),
        )
        return saved.model_dump(mode="json")

    @router.get("/annotate/{video_id}", response_class=HTMLResponse)
    async def annotate_page(request: Request, video_id: str) -> HTMLResponse:
        _require_known_video(video_id, manager)
        annotation_guides = dict(ANNOTATION_GUIDES)
        if video_id == "success":
            annotation_guides["cp_01"] = CP01_REFERENCE_GUIDE
        return templates.TemplateResponse(
            request=request,
            name="annotate.html",
            context={
                "video_id": video_id,
                "rubric": rubric,
                "annotation_guides": annotation_guides,
            },
        )

    @router.get("/api/reports/{job_id}")
    async def get_report(job_id: str) -> dict[str, object]:
        return _load_report(settings, job_id).model_dump(mode="json")

    @router.get("/reports/{job_id}", response_class=HTMLResponse)
    async def report_page(request: Request, job_id: str) -> HTMLResponse:
        report = _load_report(settings, job_id)
        checkpoint_names = {checkpoint.id: checkpoint.name for checkpoint in rubric.checkpoints}
        return templates.TemplateResponse(
            request=request,
            name="report.html",
            context={"report": report, "checkpoint_names": checkpoint_names},
        )

    @router.put("/api/reports/{job_id}/{checkpoint_id}/review")
    async def resolve_review(
        job_id: str,
        checkpoint_id: str,
        resolution: ReviewResolutionRequest,
    ) -> dict[str, object]:
        report = _load_report(settings, job_id)
        index = next(
            (
                item_index
                for item_index, checkpoint in enumerate(report.checkpoints)
                if checkpoint.checkpoint_id == checkpoint_id
            ),
            None,
        )
        if index is None:
            raise HTTPException(status_code=404, detail="checkpoint not found")
        current = report.checkpoints[index]
        automatic_status = (
            current.review_history[0].automatic_status
            if current.review_history
            else current.status
        )
        resolved_status = CheckpointStatus(resolution.status)
        audit = ReviewAuditEntry(
            timestamp=datetime.now(UTC),
            actor=resolution.actor,
            automatic_status=automatic_status,
            prior_status=current.status,
            resolved_status=resolved_status,
            reason=resolution.reason,
        )
        report.checkpoints[index] = current.model_copy(
            update={
                "status": resolved_status,
                "confidence": 1.0,
                "reason_code": "human_review_resolved",
                "reason": resolution.reason,
                "review_history": [*current.review_history, audit],
            }
        )
        _save_report(settings, report)
        return report.model_dump(mode="json")

    @router.post(
        "/api/jobs/{job_id}/checkpoints/{checkpoint_id}/rerun",
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def rerun_checkpoint(job_id: str, checkpoint_id: str) -> dict[str, str]:
        report = _load_report(settings, job_id)
        if checkpoint_id not in {item.checkpoint_id for item in report.checkpoints}:
            raise HTTPException(status_code=404, detail="checkpoint not found")
        path = safe_child(settings.data_dir / "jobs", f"{job_id}/rerun-requests.json")
        payload = json.loads(path.read_text("utf-8")) if path.exists() else {"requests": []}
        payload["requests"].append(
            {
                "checkpoint_id": checkpoint_id,
                "requested_at": datetime.now(UTC).isoformat(),
                "prior_report_digest": _report_digest(report),
            }
        )
        atomic_write_json(path, payload)
        return {"job_id": job_id, "checkpoint_id": checkpoint_id, "status": "queued"}

    @router.delete(
        "/api/jobs/{job_id}/artifacts",
        status_code=status.HTTP_204_NO_CONTENT,
        response_class=Response,
    )
    async def delete_artifacts(job_id: str) -> Response:
        _load_report(settings, job_id)
        try:
            run_dir = safe_child((settings.data_dir / "runs").resolve(), job_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        if run_dir.exists():
            shutil.rmtree(run_dir)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return router


async def _store_upload(upload: UploadFile, settings: Settings) -> Path:
    extension = Path(upload.filename or "").suffix.lower()
    if extension not in SUPPORTED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=422, detail="unsupported video extension")
    upload_dir = settings.data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / f"{uuid4().hex}{extension}"
    maximum = settings.max_upload_mb * 1024 * 1024
    written = 0
    try:
        with destination.open("wb") as handle:
            while chunk := await upload.read(1024 * 1024):
                written += len(chunk)
                if written > maximum:
                    raise HTTPException(status_code=413, detail="uploaded video is too large")
                handle.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    return destination


def _require_known_video(video_id: str, manager: JobManager) -> None:
    uploaded_ids = {record.video_id for record in manager.jobs.values()}
    if video_id not in PRESETS and video_id not in uploaded_ids:
        raise HTTPException(status_code=404, detail="video not found")


def _known_video_path(video_id: str, settings: Settings, manager: JobManager) -> Path:
    _require_known_video(video_id, manager)
    if video_id in PRESETS:
        path = settings.videos_dir / PRESETS[video_id]
    else:
        path = Path(next(record.video_path for record in manager.jobs.values() if record.video_id == video_id))
    if not path.is_file():
        raise HTTPException(status_code=404, detail="video file is missing")
    return path


def _default_steps(rubric: Rubric) -> list[SegmentAnnotation]:
    return [
        SegmentAnnotation(
            checkpoint_id=checkpoint.id,
            time_range=checkpoint.reference_time,
            label=CheckpointStatus.NEEDS_REVIEW,
            reason="等待人工确认",
        )
        for checkpoint in rubric.checkpoints
    ]


def _load_or_default(store: AnnotationStore, rubric: Rubric, video_id: str) -> VideoAnnotations:
    try:
        return store.load_segments(video_id)
    except FileNotFoundError:
        return VideoAnnotations(video_id=video_id, steps=_default_steps(rubric))


def _load_report(settings: Settings, job_id: str) -> EvaluationReport:
    try:
        path = safe_child(settings.data_dir / "jobs", f"{job_id}/report.json")
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="report not found") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="report not found")
    try:
        return EvaluationReport.model_validate_json(path.read_text("utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="stored report is invalid") from exc


def _save_report(settings: Settings, report: EvaluationReport) -> None:
    path = safe_child(settings.data_dir / "jobs", f"{report.job_id}/report.json")
    atomic_write_json(path, report.model_dump(mode="json"))


def _report_digest(report: EvaluationReport) -> str:
    payload = json.dumps(
        report.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
