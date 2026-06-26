"""Load brand-dictionary sheets from a local Excel file or Google Sheets."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

from quotaclimat.data_ingestion.advertising.s03_classification.settings import (
    BrandDictionarySettings,
)

SHEETS_SCOPE = "https://www.googleapis.com/auth/spreadsheets"

TIER_SHEET_NAMES = (
    "marques_type_1",
    "marques_type_2",
    "marques_type_3",
)


@lru_cache(maxsize=1)
def _gspread_client(credentials_path: str) -> gspread.Client:
    creds = Credentials.from_service_account_file(credentials_path, scopes=[SHEETS_SCOPE])
    return gspread.authorize(creds)


def get_client(credentials_path: str | Path) -> gspread.Client:
    return _gspread_client(str(credentials_path))


def _read_google_sheet(spreadsheet_id: str, sheet_name: str, credentials_path: Path) -> pd.DataFrame:
    worksheet = get_client(credentials_path).open_by_key(spreadsheet_id).worksheet(
        sheet_name
    )
    records = worksheet.get_all_records()
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def fetch_dictionary_sheets(settings: BrandDictionarySettings) -> dict[str, pd.DataFrame]:
    """fetch all dictionary tiers from google sheets"""
    if settings.source != "google_sheets":
        raise ValueError("fetch_dictionary_sheets requires DICT_SOURCE=google_sheets")
    return {
        sheet_name: _read_google_sheet(
            settings.spreadsheet_id,  # type: ignore[arg-type]
            sheet_name,
            settings.google_credentials,  # type: ignore[arg-type]
        )
        for sheet_name in TIER_SHEET_NAMES
    }


def read_dictionary_xlsx(path: Path) -> dict[str, pd.DataFrame]:
    return {
        sheet_name: pd.read_excel(path, sheet_name=sheet_name)
        for sheet_name in TIER_SHEET_NAMES
    }


def read_dictionary_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    """return one dictionary tier from a local excel file"""
    if sheet_name not in TIER_SHEET_NAMES:
        raise ValueError(f"Unknown dictionary sheet: {sheet_name!r}")
    return pd.read_excel(path, sheet_name=sheet_name)