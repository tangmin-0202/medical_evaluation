from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MED_EVAL_",
        env_file=".env",
        extra="ignore",
    )

    project_root: Path = Field(default_factory=Path.cwd)
    data_dir: Path = Path("data")
    videos_dir: Path = Path("橡皮障视频/橡皮障视频")
    rubric_path: Path = Path("config/rubric.yaml")
    model_config_path: Path = Path("config/models.yaml")
    max_upload_mb: int = Field(default=2048, gt=0)
    pipeline_mode: Literal["fake", "real"] = "fake"
    sam_backend: str = "sam3"
    sam_device: str = "cuda:0"
    sam2_checkpoint_path: Path = Path(
        "external/sam2/checkpoints/sam2.1_hiera_large.pt"
    )
    sam2_model_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sample_fps: float = Field(default=2.0, gt=0)
    analysis_width: int = Field(default=1280, gt=0)
    vlm_base_url: str = "http://127.0.0.1:8001/v1"
    vlm_model: str = "Qwen/Qwen3-VL-4B-Instruct"

    @model_validator(mode="after")
    def resolve_paths(self) -> "Settings":
        self.project_root = self.project_root.resolve()
        if not self.data_dir.is_absolute():
            self.data_dir = (self.project_root / self.data_dir).resolve()
        if not self.videos_dir.is_absolute():
            self.videos_dir = (self.project_root / self.videos_dir).resolve()
        if not self.rubric_path.is_absolute():
            self.rubric_path = (self.project_root / self.rubric_path).resolve()
        if not self.model_config_path.is_absolute():
            self.model_config_path = (self.project_root / self.model_config_path).resolve()
        if not self.sam2_checkpoint_path.is_absolute():
            self.sam2_checkpoint_path = (
                self.project_root / self.sam2_checkpoint_path
            ).resolve()
        return self
