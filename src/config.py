from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env file."""

    LOG_LEVEL: str = Field(default="INFO")
    WHISPER_MODEL: str = Field(default="turbo")
    DEVICE_PREFERENCE: str = Field(default="mps")
    WHISPER_CONFIDENCE_THRESHOLD: float = Field(default=0.6)
    HF_TOKEN: str = Field(default="")
    MAX_FILE_SIZE_BYTES: int = Field(default=104857600)  # 100MB
    MAX_AUDIO_DURATION_SECONDS: int = Field(default=3600)  # 1 hour

    # Directory Paths
    RAW_AUDIO_DIR: Path = Field(default=Path("audio/raw"))
    NORMALIZED_AUDIO_DIR: Path = Field(default=Path("audio/normalized"))
    OUTPUT_DIR: Path = Field(default=Path("output"))
    LOGS_DIR: Path = Field(default=Path("logs"))

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
