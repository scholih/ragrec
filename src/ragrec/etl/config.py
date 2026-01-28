"""ETL configuration and database connection."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class ETLConfig(BaseSettings):
    """Configuration for ETL operations."""

    model_config = SettingsConfigDict(env_file=".env")

    database_url: str = "postgresql://ragrec:changeme@localhost:5432/ragrec"
    batch_size: int = 1000
    log_level: str = "INFO"
