from pathlib import Path


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
