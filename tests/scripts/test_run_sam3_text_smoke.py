import importlib.util
import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from medical_evaluation.segmentation.base import FrameMasks

SCRIPT = Path(__file__).parents[2] / "scripts" / "run_sam3_text_smoke.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_sam3_text_smoke", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeBackend:
    init_kwargs: ClassVar[dict[str, object]] = {}

    def __init__(self, *_args, **kwargs) -> None:
        type(self).init_kwargs = kwargs

    def track(self, *_args, **_kwargs):
        for index in (10, 20):
            mask = np.zeros((8, 10), dtype=bool)
            mask[2:6, 3:7] = True
            yield FrameMasks(
                frame_index=index,
                frame_time_sec=index / 10,
                masks={"rubber_dam_frame": mask},
            )


def _args(tmp_path: Path) -> list[str]:
    return [
        "--video", str(tmp_path / "video.mp4"),
        "--start-sec", "1", "--end-sec", "3", "--sample-fps", "1",
        "--object-id", "rubber_dam_frame",
        "--text", "white U-shaped dental frame",
        "--checkpoint", str(tmp_path / "sam3.pt"),
        "--bpe-path", str(tmp_path / "bpe.gz"),
        "--device", "cuda:0",
        "--output-prob-threshold", "0.2",
        "--grounding-batch-size", "4",
        "--output-dir", str(tmp_path / "output"),
    ]


def test_smoke_writes_masks_and_summary(tmp_path: Path, monkeypatch) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "Sam3Backend", FakeBackend)

    assert module.main(_args(tmp_path)) == 0

    output = tmp_path / "output"
    assert (output / "00000010.png").is_file()
    assert (output / "00000020.png").is_file()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["text"] == "white U-shaped dental frame"
    assert summary["frame_indices"] == [10, 20]
    assert summary["nonempty_area_ratios"] == [0.2, 0.2]
    assert summary["output_prob_threshold"] == 0.2
    assert summary["grounding_batch_size"] == 4
    assert FakeBackend.init_kwargs["output_prob_threshold"] == 0.2
    assert FakeBackend.init_kwargs["grounding_batch_size"] == 4
    assert summary["elapsed_seconds"] >= 0


def test_smoke_refuses_existing_output_directory(tmp_path: Path, monkeypatch) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "Sam3Backend", FakeBackend)
    (tmp_path / "output").mkdir()

    with pytest.raises(FileExistsError):
        module.main(_args(tmp_path))
