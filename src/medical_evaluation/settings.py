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
    sam_backend: Literal["sam2", "sam3"] = "sam3"
    sam_device: str = "cuda:0"
    sam2_checkpoint_path: Path = Path(
        "external/sam2/checkpoints/sam2.1_hiera_large.pt"
    )
    sam2_model_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam3_checkpoint_path: Path = Path("models/SAM3.1/sam3.1_multiplex.pt")
    sam3_bpe_path: Path = Path(
        "external/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    )
    mannequin_template_path: Path = Path(
        "config/mannequin_template.registration-gate.v1.json"
    )
    sam3_output_prob_threshold: float = Field(default=0.2, ge=0, le=1)
    sam3_grounding_batch_size: int = Field(default=4, gt=0)
    sam3_source_revision: str = "660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7"
    sam3_checkpoint_sha256: str = (
        "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
    )
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
        if not self.sam3_checkpoint_path.is_absolute():
            self.sam3_checkpoint_path = (
                self.project_root / self.sam3_checkpoint_path
            ).resolve()
        if not self.sam3_bpe_path.is_absolute():
            self.sam3_bpe_path = (self.project_root / self.sam3_bpe_path).resolve()
        if not self.mannequin_template_path.is_absolute():
            self.mannequin_template_path = (
                self.project_root / self.mannequin_template_path
            ).resolve()
        return self
