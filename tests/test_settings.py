from pathlib import Path

import pytest
from pydantic import ValidationError


def test_settings_resolve_relative_paths_under_project_root(tmp_path: Path) -> None:
    from medical_evaluation.settings import Settings

    settings = Settings(
        project_root=tmp_path,
        data_dir=Path("runtime-data"),
        model_config_path=Path("config/models.yaml"),
        sam2_checkpoint_path=Path("models/sam2.pt"),
    )

    assert settings.project_root == tmp_path.resolve()
    assert settings.data_dir == (tmp_path / "runtime-data").resolve()
    assert settings.model_config_path == (tmp_path / "config/models.yaml").resolve()
    assert settings.sam2_checkpoint_path == (tmp_path / "models/sam2.pt").resolve()


def test_settings_resolve_sam3_paths(tmp_path: Path) -> None:
    from medical_evaluation.settings import Settings

    settings = Settings(
        project_root=tmp_path,
        sam_backend="sam3",
        sam3_checkpoint_path=Path("models/sam3.pt"),
        sam3_bpe_path=Path("external/sam3/assets/bpe.txt.gz"),
    )

    assert settings.sam3_checkpoint_path == (tmp_path / "models/sam3.pt").resolve()
    assert settings.sam3_bpe_path == (
        tmp_path / "external/sam3/assets/bpe.txt.gz"
    ).resolve()
    assert settings.sam3_grounding_batch_size == 4


def test_settings_reject_nonpositive_sam3_grounding_batch_size(tmp_path: Path) -> None:
    from medical_evaluation.settings import Settings

    with pytest.raises(ValidationError):
        Settings(project_root=tmp_path, sam3_grounding_batch_size=0)


def test_settings_reject_unknown_sam_backend(tmp_path: Path) -> None:
    from medical_evaluation.settings import Settings

    with pytest.raises(ValidationError):
        Settings(project_root=tmp_path, sam_backend="unknown")
