from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_DICT_LOCAL_PATH = Path(
    "quotaclimat/data_ingestion/advertising/s03_classification/reference_data/dictionnaire_marques_secteurs.xlsx"
)
DEFAULT_VERSIONS_DIR = Path(
    "quotaclimat/data_ingestion/advertising/s03_classification/reference_data/dictionary_versions"
)


class BrandDictionarySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DICT_", extra="ignore")

    source: Literal["local", "google_sheets"] = "local"
    local_path: Path = DEFAULT_DICT_LOCAL_PATH
    versions_dir: Path = DEFAULT_VERSIONS_DIR
    spreadsheet_id: str | None = None
    google_credentials: Path | None = None

    @model_validator(mode="after")
    def _validate_google_sheets_config(self) -> "BrandDictionarySettings":
        if self.source != "google_sheets":
            return self
        missing = [
            name
            for name, value in (
                ("DICT_SPREADSHEET_ID", self.spreadsheet_id),
                ("DICT_GOOGLE_CREDENTIALS", self.google_credentials),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"{', '.join(missing)} required when DICT_SOURCE=google_sheets"
            )
        return self


class ASRSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ASR_", extra="ignore")
    model_name: str = Field(...)
    api_url: str = Field(...)
    api_token: str = Field(...)


class VLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VLM_", extra="ignore")
    model_name: str = Field(...)
    api_url: str = Field(...)
    api_token: str = Field(...)
