from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    """CLIP model configuration."""

    name: str = Field(default="ViT-H-14", description="CLIP model name")
    type: str = Field(default="open_clip", description="Model type")
    pretrained: str = Field(default="laion2b_s32b_b79k", description="Pretrained weights")


class ProcessingConfig(BaseModel):
    """Video processing configuration."""

    scene_threshold: float = Field(default=27.0, ge=0.0, description="Scene detection threshold")
    skip_intro_seconds: int = Field(default=120, ge=0, description="Skip intro duration")
    skip_outro_seconds: int = Field(default=300, ge=0, description="Skip outro duration")
    use_motion_filter: bool = Field(default=False, description="Enable motion filtering")
    min_motion_threshold: float = Field(default=3.0, ge=0.0, description="Minimum motion threshold")
    frame_extraction_fps: float = Field(default=1.0, gt=0.0, description="Frame extraction rate")


class PathsConfig(BaseModel):
    """Directory paths configuration."""

    films_dir: Path = Field(default=Path("./films"), description="Input films directory")
    output_dir: Path = Field(default=Path("./output"), description="Output directory")
    cache_dir: Path = Field(default=Path("./cache"), description="Cache directory")

    @field_validator("films_dir", "output_dir", "cache_dir", mode="before")
    @classmethod
    def convert_to_path(cls, v):
        return Path(v) if not isinstance(v, Path) else v


class AppConfig(BaseModel):
    """Main application configuration."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    films_dir: Path = Field(default=Path("./films"), description="Films directory")
    output_dir: Path = Field(default=Path("./output"), description="Output directory")
    cache_dir: Path = Field(default=Path("./cache"), description="Cache directory")

    device: str = Field(default="cuda", description="Compute device")
    batch_size: int = Field(default=32, gt=0, description="Batch size for processing")
    fragment_length: int = Field(default=5, gt=0, description="Fragment length in seconds")
    similarity_threshold: float = Field(default=0.25, ge=0.0, le=1.0, description="Similarity threshold")
    aggregation_method: str = Field(default="max", description="Score aggregation method: max or mean")
    
    query: str | None = Field(default=None, description="Search query")
    queries: list[str] = Field(default_factory=list, description="List of queries for batch search")

    @field_validator("films_dir", "output_dir", "cache_dir", mode="before")
    @classmethod
    def convert_to_path(cls, v):
        return Path(v) if not isinstance(v, Path) else v

