from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime configuration for the API gateway layer."""

    project_name: str = Field(default="DTAQ MLOps API")
    version: str = Field(default="0.1.0")
    data_acquisition_url: str = Field(default="http://data-acquisition:8000")
    data_preparation_url: str = Field(default="http://data-preparation:8000")
    model_training_url: str = Field(default="http://model-training:8000")
    model_evaluation_url: str = Field(default="http://model-evaluation:8000")
    model_serving_url: str = Field(default="http://model-serving:8000")
    pdf_to_csv_url: str = Field(default="http://pdf-to-csv:8000")

    class Config:
        env_prefix = "DTAQ_"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings to avoid reparsing environment variables."""
    return Settings()
