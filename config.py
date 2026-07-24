from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    MODELS_DIR: Path
    MLP_MODEL: str
    FCD_MODEL: str
    PORT: int


settings = Settings()
