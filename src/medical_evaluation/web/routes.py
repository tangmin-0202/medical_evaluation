from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.datastructures import UploadFile

from medical_evaluation.jobs import JobManager
from medical_evaluation.settings import Settings
from medical_evaluation.video import SUPPORTED_VIDEO_EXTENSIONS

PRESETS = {
    "success": "橡皮障完整.mp4",
    "failure": "橡皮障失败.mp4",
    "clamp_failure": "橡皮障夹子飞了.mp4",
}


def create_router(settings: Settings, manager: JobManager, template_dir: Path) -> APIRouter:
    router = APIRouter()
    templates = Jinja2Templates(directory=template_dir)

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
