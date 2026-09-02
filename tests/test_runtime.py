from pathlib import Path

import pytest

from medical_evaluation.annotations import SegmentAnnotation, VideoAnnotations
from medical_evaluation.app import create_app
from medical_evaluation.domain import CheckpointStatus, TimeRange
from medical_evaluation.extractors.cp09_cp11 import Cp09Cp11FeatureExtractor
from medical_evaluation.jobs import JobRecord
from medical_evaluation.runtime import build_analysis_pipeline
from medical_evaluation.settings import Settings


class FakeBackend:
    model_version = "fake-sam2"


def make_settings(tmp_path: Path, *, pipeline_mode: str = "real") -> Settings:
    checkpoint = tmp_path / "models" / "sam2.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"weights")
    rubric_path = Path(__file__).parents[1] / "config" / "rubric.yaml"
    data_dir = tmp_path / "data"
    annotation_dir = data_dir / "annotations"
    annotation_dir.mkdir(parents=True)
    annotations = VideoAnnotations(
        video_id="success",
        steps=[
            SegmentAnnotation(
                checkpoint_id="cp_09",
                time_range=TimeRange(start_sec=175, end_sec=195),
                label=CheckpointStatus.CORRECT,
                reason="test",
            )
        ],
    )
    (annotation_dir / "success.json").write_text(
        annotations.model_dump_json(),
        encoding="utf-8",
    )
    return Settings(
        project_root=tmp_path,
        data_dir=data_dir,
        videos_dir=tmp_path / "videos",
        rubric_path=rubric_path,
        pipeline_mode=pipeline_mode,
        sam2_checkpoint_path=checkpoint,
        sam2_model_config="configs/sam2.1/sam2.1_hiera_l.yaml",
        vlm_base_url="http://qwen.local/v1",
        vlm_model="Qwen-Test",
    )


def test_runtime_builds_job_scoped_cp09_cp11_extractor(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    backend_calls = []

    def segmenter_factory(model_config, checkpoint_path, *, device):
        backend_calls.append((model_config, checkpoint_path, device))
        return FakeBackend()

    pipeline = build_analysis_pipeline(settings, segmenter_factory=segmenter_factory)
    job = JobRecord(id="job-1", video_id="success", video_path=str(tmp_path / "video.mp4"))
    extractor = pipeline.extractor_factory(job)

    assert isinstance(extractor, Cp09Cp11FeatureExtractor)
    assert extractor.cp09.evidence_root == settings.data_dir / "jobs" / "job-1"
    assert extractor.cp11.evidence_root == settings.data_dir / "jobs" / "job-1"
    assert pipeline.commentary_provider.model == "Qwen-Test"
    assert backend_calls == [
        (
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            settings.sam2_checkpoint_path,
            settings.sam_device,
        )
    ]


def test_runtime_rejects_missing_sam2_checkpoint(tmp_path: Path) -> None:
    settings = make_settings(tmp_path).model_copy(
        update={"sam2_checkpoint_path": tmp_path / "missing.pt"}
    )

    with pytest.raises(ValueError, match="SAM2 checkpoint"):
        build_analysis_pipeline(settings, segmenter_factory=lambda *args, **kwargs: FakeBackend())


def test_create_app_real_mode_uses_runtime_factory(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    sentinel = object()
    calls = []

    monkeypatch.setattr(
        "medical_evaluation.runtime.build_analysis_pipeline",
        lambda received: calls.append(received) or sentinel,
    )

    app = create_app(settings)

    assert calls == [settings]
    assert app.state.job_manager is not None
