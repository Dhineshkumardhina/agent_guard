"""AgentGuard Application & Research Configuration.

Provides strongly typed settings using pydantic-settings, supporting
loading from .env files, system environment variables, or explicit overrides.
"""

from pathlib import Path
from typing import List, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Core settings for AgentGuard research platform and backend services."""

    # Project Information
    PROJECT_NAME: str = "AgentGuard"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = (
        "Temporal Graph-Based Detection and Prediction of Cascading Failures in Multi-Agent AI Systems"
    )
    
    # Environment
    ENV: str = Field(default="development", description="Environment mode: development, testing, production")
    DEBUG: bool = Field(default=True, description="Debug mode flag")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    RANDOM_SEED: int = Field(default=42, description="Global random seed for scientific reproducibility")

    # API / Server
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    SECRET_KEY: str = "agentguard-research-secret-key-change-in-production"
    ALLOWED_ORIGINS: Union[List[str], str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]

    # Database
    DATABASE_URL: str = Field(
        default="sqlite:///./agentguard.db",
        description="SQLAlchemy database connection string. SQLite for local research, PostgreSQL for production.",
    )

    # Directories
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    DATASET_DIR: Path = BASE_DIR / "datasets"
    RESULTS_DIR: Path = BASE_DIR / "results"
    ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
    CONFIGS_DIR: Path = BASE_DIR / "configs"

    @field_validator("ALLOWED_ORIGINS", mode="after")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                import json
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, list):
            return v
        return []

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
